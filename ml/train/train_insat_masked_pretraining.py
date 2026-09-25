"""
VAYU-NET — INSAT-3D SELF-SUPERVISED PRETRAINING RUNNER
======================================================
Pretrains lightweight spatial encoders using Masked Spatial Autoencoding on the
unlabeled 2014–2024 INSAT-3D archive (757 train frames, 671 val frames).

Exports:
  - data/interim/ml/checkpoints/insat_pretrained_encoder.pt (3-channel encoder)
  - data/interim/ml/checkpoints/insat_pretrained_encoder_2ch.pt (2-channel encoder)
  - data/interim/ml/insat_pretraining_results.json
  - Visual diagnostics in docs/figures/insat_pretraining/
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ml.models.insat_masked_autoencoder import InsatMaskedAutoencoder


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InsatPretrainDataset(Dataset):
    def __init__(self, cache_data: Dict[str, Any], split: str, in_channels: int = 3):
        indices = [i for i, s in enumerate(cache_data["splits"]) if s == split]
        tensor_key = "tensors_3ch" if in_channels == 3 else "tensors_2ch"
        self.tensors = cache_data[tensor_key][indices]
        self.timestamps = [cache_data["timestamps"][i] for i in indices]
        self.years = [cache_data["years"][i] for i in indices]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]


def train_insat_mae(in_channels: int = 3,
                    cache_data: Dict[str, Any] = None,
                    epochs: int = 25,
                    batch_size: int = 32,
                    lr: float = 1e-3,
                    mask_ratio: float = 0.35,
                    device: torch.device = torch.device("cpu"),
                    ckpt_out: str = "data/interim/ml/checkpoints/insat_pretrained_encoder.pt") -> Dict[str, Any]:
    set_seed(42)
    ch_label = "3-Channel (TIR1+TIR2+WV)" if in_channels == 3 else "2-Channel (TIR1+TIR2)"
    print(f"\n=======================================================")
    print(f"PRETRAINING INSAT MASKED AUTOENCODER: {ch_label}")
    print(f"=======================================================")

    train_ds = InsatPretrainDataset(cache_data, "PRETRAIN_TRAIN", in_channels=in_channels)
    val_ds = InsatPretrainDataset(cache_data, "PRETRAIN_VAL", in_channels=in_channels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset: PRETRAIN_TRAIN = {len(train_ds)} frames (2014-2020), PRETRAIN_VAL = {len(val_ds)} frames (2021-2024)")

    model = InsatMaskedAutoencoder(in_channels=in_channels, embed_dim=64, mask_ratio=mask_ratio).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Parameter Budget: Total={total_params:,} (Encoder={enc_params:,}, Decoder={dec_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = -1
    history = []

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            x = batch.to(device)
            out = model(x)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse_total = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                out = model(x)
                val_loss += out["loss"].item()
                val_mse_total += out["loss_total"].item()
                n_val += 1

        avg_val_loss = val_loss / max(1, n_val)
        avg_val_mse = val_mse_total / max(1, n_val)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (MSE: {avg_val_mse:.4f}) | LR: {curr_lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_mse": round(avg_val_mse, 4),
            "lr": curr_lr
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = {
                "encoder_state_dict": model.get_encoder_state_dict(),
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_mse": avg_val_mse
            }

    print(f"Pretraining completed in {time.time() - t0_train:.1f}s. Best Epoch {best_epoch} with Val Loss {best_val_loss:.4f}")

    # Save checkpoint
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, ckpt_out)
    print(f"Saved pretrained encoder checkpoint to {ckpt_out}")

    return {
        "in_channels": in_channels,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_weights": best_weights,
        "history": history,
        "model": model
    }


def generate_representation_diagnostics(model_3ch: InsatMaskedAutoencoder,
                                        cache_data: Dict[str, Any],
                                        output_dir: str = "docs/figures/insat_pretraining",
                                        device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    print("\nGenerating representation diagnostics and visual proof...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_3ch.eval()

    # 1. Collect latent representations across all 1,428 frames
    all_tensors = cache_data["tensors_3ch"]
    latents = []
    with torch.no_grad():
        for i in range(0, len(all_tensors), 32):
            batch = all_tensors[i:i+32].to(device)
            z = model_3ch.encoder(batch)
            latents.append(z.cpu().numpy())

    z_all = np.concatenate(latents, axis=0)  # [1428, 64]
    feature_variance = float(np.mean(np.var(z_all, axis=0)))
    active_dims = int(np.sum(np.var(z_all, axis=0) > 1e-4))

    # Singular Value Spectrum
    u, s, vh = np.linalg.svd(z_all - np.mean(z_all, axis=0), full_matrices=False)
    variance_explained_top10 = float(np.sum(s[:10]**2) / np.sum(s**2))

    # 2. PCA Embedding Visualization
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_all)
    years = np.array(cache_data["years"])

    plt.figure(figsize=(9, 7), dpi=150)
    sc = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=years, cmap="viridis", alpha=0.75, s=20)
    plt.colorbar(sc, label="Observation Year")
    plt.title("INSAT-3D Self-Supervised Latent Space PCA (N = 1,428 Frames)", fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.grid(True, linestyle=":", alpha=0.5)
    pca_file = Path(output_dir) / "insat_latent_pca.png"
    plt.savefig(pca_file, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA embedding plot to {pca_file}")

    # 3. Visual Reconstruction Example (Original vs Masked vs Reconstructed)
    sample_batch = all_tensors[:4].to(device)
    with torch.no_grad():
        out = model_3ch(sample_batch)
    orig = sample_batch.cpu().numpy()
    masked = out["x_masked"].cpu().numpy()
    rec = out["reconstruction"].cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=150)
    channels_labels = ["TIR1 (10.8µm)", "TIR2 (12.0µm)", "WV (6.8µm)"]

    for b in range(4):
        # Top: Original TIR1
        axes[0, b].imshow(orig[b, 0], cmap="gray_r")
        axes[0, b].set_title(f"Sample {b+1}: Ground Truth TIR1", fontsize=10)
        axes[0, b].axis("off")

        # Mid: Masked Input TIR1
        axes[1, b].imshow(masked[b, 0], cmap="gray_r")
        axes[1, b].set_title(f"Sample {b+1}: Masked Input (35%)", fontsize=10)
        axes[1, b].axis("off")

        # Bot: Reconstructed TIR1
        axes[2, b].imshow(rec[b, 0], cmap="gray_r")
        axes[2, b].set_title(f"Sample {b+1}: MAE Reconstruction", fontsize=10)
        axes[2, b].axis("off")

    plt.suptitle("INSAT-3D Self-Supervised Masked Autoencoder Reconstructions", fontsize=14, fontweight="bold")
    rec_file = Path(output_dir) / "insat_reconstruction_example.png"
    plt.savefig(rec_file, bbox_inches="tight")
    plt.close()
    print(f"Saved visual reconstruction example to {rec_file}")

    return {
        "total_frames_embedded": len(z_all),
        "embedding_dimension": 64,
        "active_latent_dimensions": active_dims,
        "mean_feature_variance": feature_variance,
        "pca_variance_top2": float(np.sum(pca.explained_variance_ratio_)),
        "svd_variance_explained_top10": variance_explained_top10
    }


def main():
    device = torch.device("cpu")
    cache_path = Path("data/interim/ml/cache/insat_pretraining_cache.pt")
    assert cache_path.exists(), f"Cache missing at {cache_path}"
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    # 1. Train 3-channel MAE
    res_3ch = train_insat_mae(
        in_channels=3,
        cache_data=cache_data,
        epochs=25,
        batch_size=32,
        lr=1e-3,
        mask_ratio=0.35,
        device=device,
        ckpt_out="data/interim/ml/checkpoints/insat_pretrained_encoder.pt"
    )

    # 2. Train 2-channel MAE
    res_2ch = train_insat_mae(
        in_channels=2,
        cache_data=cache_data,
        epochs=25,
        batch_size=32,
        lr=1e-3,
        mask_ratio=0.35,
        device=device,
        ckpt_out="data/interim/ml/checkpoints/insat_pretrained_encoder_2ch.pt"
    )

    # 3. Generate Diagnostics & Visuals
    diag = generate_representation_diagnostics(res_3ch["model"], cache_data, device=device)

    # Plot Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for idx, (res, title) in enumerate([(res_3ch, "3-Channel (TIR1+TIR2+WV)"), (res_2ch, "2-Channel (TIR1+TIR2)")]):
        ax = axes[idx]
        epochs = [h["epoch"] for h in res["history"]]
        ax.plot(epochs, [h["train_loss"] for h in res["history"]], label="Train Loss", lw=2, color="#1f77b4")
        ax.plot(epochs, [h["val_loss"] for h in res["history"]], label="Val Loss", lw=2, linestyle="--", color="#ff7f0e")
        ax.set_title(f"INSAT MAE: {title}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    curve_file = Path("docs/figures/insat_pretraining/insat_pretraining_loss_curves.png")
    plt.savefig(curve_file, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves to {curve_file}")

    # 4. Save JSON results
    out_json = {
        "metadata": {
            "title": "VAYU-NET INSAT-3D Self-Supervised Masked Autoencoder Pretraining",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_observations": 1428,
            "train_observations": 757,
            "val_observations": 671,
            "mask_ratio": 0.35,
            "patch_size": 8,
            "epochs": 25,
            "seed": 42
        },
        "experiments": {
            "insat_3ch_pretraining": {
                "in_channels": 3,
                "best_epoch": res_3ch["best_epoch"],
                "best_val_loss": round(res_3ch["best_val_loss"], 4),
                "history": res_3ch["history"]
            },
            "insat_2ch_pretraining": {
                "in_channels": 2,
                "best_epoch": res_2ch["best_epoch"],
                "best_val_loss": round(res_2ch["best_val_loss"], 4),
                "history": res_2ch["history"]
            }
        },
        "representation_diagnostics": diag
    }

    json_path = Path("data/interim/ml/insat_pretraining_results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved pretraining results JSON to {json_path}")


if __name__ == "__main__":
    main()
