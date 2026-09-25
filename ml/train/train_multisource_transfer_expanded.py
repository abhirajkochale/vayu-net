"""
VAYU-NET — MULTI-SOURCE SATELLITE TRANSFER LEARNING ON EXPANDED TRAIN DATASET
=============================================================================
Retrains and evaluates the multisource transfer-learning suite using the
expanded 207-sample TRAIN population (25 storms), keeping VAL (252 samples, 14 storms)
and TEST (298 samples, 24 storms) strictly locked and identical.

Experiments:
  - EXP1: GridSat-only transfer baseline (frozen GridSat temporal encoder + trainable heads)
  - EXP2: GridSat + INSAT TIR1 + TIR2 (lightweight INSAT branch, 2 channels)
  - EXP3: GridSat + INSAT TIR1 + TIR2 + WV (lightweight INSAT branch, 3 channels)
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score

from ml.models.multisource_transfer_fusion import (
    MultisourceTransferFusionModel,
    haversine_km,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)
from ml.train.train_multisource_ablation import compute_baselines_on_test


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultisourceTransferDataset(Dataset):
    def __init__(self, cache_data: Dict[str, Any], split: str, insat_channels: Optional[List[int]] = None):
        self.split = split
        self.samples = [s for s in cache_data["samples"] if s["split"] == split]
        self.insat_channels = insat_channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        item = dict(s)
        if self.insat_channels is not None:
            # Slice channels along C dimension: [6, C, 72, 116]
            item["insat_seq"] = s["insat_seq"][:, self.insat_channels, :, :]
        return item


def compute_metrics(predictions: Dict[str, np.ndarray],
                    ground_truth: Dict[str, np.ndarray]) -> Dict[str, Any]:
    pred_centers = predictions["center_deg"]
    true_centers = ground_truth["center_deg"]
    dpes_center = [haversine_km(pc[0], pc[1], tc[0], tc[1]) for pc, tc in zip(pred_centers, true_centers)]
    lat_mae = float(np.mean(np.abs(pred_centers[:, 0] - true_centers[:, 0])))
    lon_mae = float(np.mean(np.abs(pred_centers[:, 1] - true_centers[:, 1])))

    pred_cats = predictions["category_pred"]
    true_cats = ground_truth["category_idx"]
    acc = float(accuracy_score(true_cats, pred_cats))
    macro_f1 = float(f1_score(true_cats, pred_cats, average="macro", zero_division=0))
    recalls = recall_score(true_cats, pred_cats, average=None, labels=list(range(7)), zero_division=0)
    per_class_recall = {IDX_TO_CATEGORY[i]: float(recalls[i]) for i in range(7)}

    pred_winds = predictions["wind_kt"].squeeze()
    true_winds = ground_truth["wind_kt"].squeeze()
    wind_diff = pred_winds - true_winds
    abs_wind_diff = np.abs(wind_diff)
    wind_mae = float(np.mean(abs_wind_diff))
    wind_rmse = float(np.sqrt(np.mean(wind_diff ** 2)))
    wind_med_ae = float(np.median(abs_wind_diff))
    wind_p90_ae = float(np.percentile(abs_wind_diff, 90))
    wind_bias = float(np.mean(wind_diff))
    wind_r = float(np.corrcoef(pred_winds, true_winds)[0, 1]) if (np.std(pred_winds) > 1e-6 and np.std(true_winds) > 1e-6) else 0.0

    dpes_12, dpes_24, dpes_48 = [], [], []
    m12 = ground_truth["mask_12"]
    m24 = ground_truth["mask_24"]
    m48 = ground_truth["mask_48"]

    for i in range(len(true_centers)):
        if m12[i] > 0.5:
            dpes_12.append(haversine_km(predictions["track_12h_deg"][i, 0], predictions["track_12h_deg"][i, 1],
                                        ground_truth["t12_deg"][i, 0], ground_truth["t12_deg"][i, 1]))
        if m24[i] > 0.5:
            dpes_24.append(haversine_km(predictions["track_24h_deg"][i, 0], predictions["track_24h_deg"][i, 1],
                                        ground_truth["t24_deg"][i, 0], ground_truth["t24_deg"][i, 1]))
        if m48[i] > 0.5:
            dpes_48.append(haversine_km(predictions["track_48h_deg"][i, 0], predictions["track_48h_deg"][i, 1],
                                        ground_truth["t48_deg"][i, 0], ground_truth["t48_deg"][i, 1]))

    all_dpes = dpes_12 + dpes_24 + dpes_48
    return {
        "identification": {
            "center_mean_dpe_km": float(np.mean(dpes_center)),
            "center_median_dpe_km": float(np.median(dpes_center)),
            "center_p90_dpe_km": float(np.percentile(dpes_center, 90)),
            "center_lat_mae_deg": lat_mae,
            "center_lon_mae_deg": lon_mae
        },
        "classification": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_recall": per_class_recall
        },
        "wind_regression": {
            "wind_mae_kt": wind_mae,
            "wind_rmse_kt": wind_rmse,
            "wind_median_ae_kt": wind_med_ae,
            "wind_p90_ae_kt": wind_p90_ae,
            "wind_bias_kt": wind_bias,
            "wind_pearson_r": wind_r
        },
        "track_prediction": {
            "track_12h_mean_dpe_km": float(np.mean(dpes_12)) if dpes_12 else 0.0,
            "track_12h_median_dpe_km": float(np.median(dpes_12)) if dpes_12 else 0.0,
            "track_12h_p90_dpe_km": float(np.percentile(dpes_12, 90)) if dpes_12 else 0.0,
            "track_24h_mean_dpe_km": float(np.mean(dpes_24)) if dpes_24 else 0.0,
            "track_24h_median_dpe_km": float(np.median(dpes_24)) if dpes_24 else 0.0,
            "track_24h_p90_dpe_km": float(np.percentile(dpes_24, 90)) if dpes_24 else 0.0,
            "track_48h_mean_dpe_km": float(np.mean(dpes_48)) if dpes_48 else 0.0,
            "track_48h_median_dpe_km": float(np.median(dpes_48)) if dpes_48 else 0.0,
            "track_48h_p90_dpe_km": float(np.percentile(dpes_48, 90)) if dpes_48 else 0.0,
            "track_aggregate_mean_dpe_km": float(np.mean(all_dpes)) if all_dpes else 0.0
        }
    }


def evaluate_transfer_loader(model: MultisourceTransferFusionModel,
                             loader: DataLoader,
                             device: torch.device) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    preds_all = {
        "center_deg": [],
        "category_pred": [],
        "wind_kt": [],
        "track_12h_deg": [],
        "track_24h_deg": [],
        "track_48h_deg": []
    }
    gt_all = {
        "center_deg": [],
        "category_idx": [],
        "wind_kt": [],
        "t12_deg": [],
        "mask_12": [],
        "t24_deg": [],
        "mask_24": [],
        "t48_deg": [],
        "mask_48": []
    }

    with torch.no_grad():
        for batch in loader:
            g_feat = batch["gridsat_feat_seq"].to(device)
            i_seq = batch["insat_seq"].to(device) if model.insat_branch is not None else None

            out = model(gridsat_feat_seq=g_feat, insat_seq=i_seq)

            c_norm = batch["center_norm"].to(device)
            cat_idx = batch["category_idx"].to(device)
            w_norm = batch["wind_norm"].to(device)
            t12_norm = batch["t12_norm"].to(device)
            m12 = batch["mask_12"].to(device)
            t24_norm = batch["t24_norm"].to(device)
            m24 = batch["mask_24"].to(device)
            t48_norm = batch["t48_norm"].to(device)
            m48 = batch["mask_48"].to(device)

            loss_c = F.smooth_l1_loss(out["center_norm"], c_norm)
            loss_cls = F.cross_entropy(out["class_logits"], cat_idx)
            loss_w = F.smooth_l1_loss(out["wind_norm"], w_norm)

            loss_12 = (F.smooth_l1_loss(out["track_12h_norm"], t12_norm, reduction="none") * m12.unsqueeze(-1)).sum() / (m12.sum() * 2.0 + 1e-6)
            loss_24 = (F.smooth_l1_loss(out["track_24h_norm"], t24_norm, reduction="none") * m24.unsqueeze(-1)).sum() / (m24.sum() * 2.0 + 1e-6)
            loss_48 = (F.smooth_l1_loss(out["track_48h_norm"], t48_norm, reduction="none") * m48.unsqueeze(-1)).sum() / (m48.sum() * 2.0 + 1e-6)
            loss_track = (loss_12 + loss_24 + loss_48) / 3.0

            loss = loss_c + loss_cls + loss_w + loss_track
            total_loss += loss.item()
            num_batches += 1

            preds_all["center_deg"].append(out["center_deg"].cpu().numpy())
            preds_all["category_pred"].append(torch.argmax(out["class_logits"], dim=-1).cpu().numpy())
            preds_all["wind_kt"].append(out["wind_kt"].cpu().numpy())
            preds_all["track_12h_deg"].append(out["track_12h_deg"].cpu().numpy())
            preds_all["track_24h_deg"].append(out["track_24h_deg"].cpu().numpy())
            preds_all["track_48h_deg"].append(out["track_48h_deg"].cpu().numpy())

            gt_all["center_deg"].append(batch["center_deg"].numpy())
            gt_all["category_idx"].append(batch["category_idx"].numpy())
            gt_all["wind_kt"].append(batch["wind_kt"].numpy())
            gt_all["t12_deg"].append(batch["t12_deg"].numpy())
            gt_all["mask_12"].append(batch["mask_12"].numpy())
            gt_all["t24_deg"].append(batch["t24_deg"].numpy())
            gt_all["mask_24"].append(batch["mask_24"].numpy())
            gt_all["t48_deg"].append(batch["t48_deg"].numpy())
            gt_all["mask_48"].append(batch["mask_48"].numpy())

    avg_loss = total_loss / max(1, num_batches)
    preds_concat = {k: np.concatenate(v, axis=0) for k, v in preds_all.items()}
    gt_concat = {k: np.concatenate(v, axis=0) for k, v in gt_all.items()}
    metrics = compute_metrics(preds_concat, gt_concat)
    return avg_loss, metrics


def train_transfer_experiment(exp_name: str,
                              mode: str,
                              cache_data: Dict[str, Any],
                              device: torch.device,
                              insat_channels: Optional[List[int]] = None,
                              in_channels_insat: int = 3,
                              epochs: int = 20,
                              batch_size: int = 16,
                              lr: float = 1e-3,
                              patience: int = 5,
                              checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    set_seed(42)
    print(f"\n=======================================================")
    print(f"TRAINING EXPERIMENT: {exp_name.upper()} (mode='{mode}', ch={in_channels_insat})")
    print(f"=======================================================")

    train_ds = MultisourceTransferDataset(cache_data, "TRAIN", insat_channels=insat_channels)
    val_ds = MultisourceTransferDataset(cache_data, "VALIDATION", insat_channels=insat_channels)
    test_ds = MultisourceTransferDataset(cache_data, "TEST", insat_channels=insat_channels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset partition: TRAIN={len(train_ds)}, VALIDATION={len(val_ds)}, TEST={len(test_ds)}")

    model = MultisourceTransferFusionModel(
        mode=mode,
        in_channels_insat=in_channels_insat
    ).to(device)

    param_counts = model.get_parameter_counts()
    print(f"Parameter Budget: Total={param_counts['total_parameters']:,} | "
          f"Trainable={param_counts['trainable_parameters']:,} | Frozen={param_counts['frozen_parameters']:,}")

    # Optimizer: only optimize parameters with requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_weights = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            g_feat = batch["gridsat_feat_seq"].to(device)
            i_seq = batch["insat_seq"].to(device) if model.insat_branch is not None else None

            out = model(gridsat_feat_seq=g_feat, insat_seq=i_seq)

            c_norm = batch["center_norm"].to(device)
            cat_idx = batch["category_idx"].to(device)
            w_norm = batch["wind_norm"].to(device)
            t12_norm = batch["t12_norm"].to(device)
            m12 = batch["mask_12"].to(device)
            t24_norm = batch["t24_norm"].to(device)
            m24 = batch["mask_24"].to(device)
            t48_norm = batch["t48_norm"].to(device)
            m48 = batch["mask_48"].to(device)

            loss_c = F.smooth_l1_loss(out["center_norm"], c_norm)
            loss_cls = F.cross_entropy(out["class_logits"], cat_idx)
            loss_w = F.smooth_l1_loss(out["wind_norm"], w_norm)

            loss_12 = (F.smooth_l1_loss(out["track_12h_norm"], t12_norm, reduction="none") * m12.unsqueeze(-1)).sum() / (m12.sum() * 2.0 + 1e-6)
            loss_24 = (F.smooth_l1_loss(out["track_24h_norm"], t24_norm, reduction="none") * m24.unsqueeze(-1)).sum() / (m24.sum() * 2.0 + 1e-6)
            loss_48 = (F.smooth_l1_loss(out["track_48h_norm"], t48_norm, reduction="none") * m48.unsqueeze(-1)).sum() / (m48.sum() * 2.0 + 1e-6)
            loss_track = (loss_12 + loss_24 + loss_48) / 3.0

            loss = loss_c + loss_cls + loss_w + loss_track
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train += 1

        avg_train_loss = train_loss / max(1, n_train)
        val_loss, val_m = evaluate_transfer_loader(model, val_loader, device)
        scheduler.step(val_loss)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Center DPE: {val_m['identification']['center_mean_dpe_km']:.1f}km | "
              f"Val Wind MAE: {val_m['wind_regression']['wind_mae_kt']:.1f}kt | "
              f"Val Track DPE: {val_m['track_prediction']['track_aggregate_mean_dpe_km']:.1f}km | LR: {curr_lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_center_dpe_km": round(val_m["identification"]["center_mean_dpe_km"], 2),
            "val_class_macro_f1": round(val_m["classification"]["macro_f1"], 4),
            "val_wind_mae_kt": round(val_m["wind_regression"]["wind_mae_kt"], 2),
            "val_track_dpe_km": round(val_m["track_prediction"]["track_aggregate_mean_dpe_km"], 2),
            "lr": curr_lr
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_m
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  >>> Best validation checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    print(f"\nEvaluating Best Checkpoint (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}) on TEST set...")
    model.load_state_dict(best_weights)
    test_loss, test_m = evaluate_transfer_loader(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f} | Center DPE: {test_m['identification']['center_mean_dpe_km']:.1f}km | "
          f"Class Macro-F1: {test_m['classification']['macro_f1']:.4f} | "
          f"Wind MAE: {test_m['wind_regression']['wind_mae_kt']:.1f}kt | "
          f"Track Agg DPE: {test_m['track_prediction']['track_aggregate_mean_dpe_km']:.1f}km")

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "exp_name": exp_name,
            "mode": mode,
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "model_state_dict": best_weights,
            "val_metrics": best_val_metrics,
            "test_metrics": test_m,
            "test_loss": test_loss,
            "history": history
        }, checkpoint_path)
        print(f"Saved best model checkpoint to {checkpoint_path}")

    return {
        "exp_name": exp_name,
        "mode": mode,
        "in_channels_insat": in_channels_insat,
        "best_epoch": best_epoch,
        "parameter_counts": param_counts,
        "val_loss": best_val_loss,
        "val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_m,
        "history": history
    }


def plot_expanded_transfer_curves(results: Dict[str, Any], output_dir: str = "docs/figures/multisource_transfer_expanded"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    colors = {
        "EXP-1": "#1f77b4",
        "EXP-2": "#2ca02c",
        "EXP-3": "#d62728"
    }

    # 1. Total Val Loss
    ax = axes[0, 0]
    for exp_key in ["EXP-1", "EXP-2", "EXP-3"]:
        if exp_key in results:
            hist = results[exp_key]["history"]
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["train_loss"] for h in hist], linestyle="--", alpha=0.5, color=colors[exp_key])
            ax.plot(epochs, [h["val_loss"] for h in hist], linestyle="-", lw=2, color=colors[exp_key], label=f"{exp_key} (Val)")
    ax.set_title("Total Loss (Train vs. Val) - Expanded TRAIN", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 2. Val Center DPE
    ax = axes[0, 1]
    for exp_key in ["EXP-1", "EXP-2", "EXP-3"]:
        if exp_key in results:
            hist = results[exp_key]["history"]
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["val_center_dpe_km"] for h in hist], marker="o", color=colors[exp_key], label=exp_key)
    ax.set_title("Validation Center Mean DPE (km)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPE (km)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 3. Val Wind MAE
    ax = axes[1, 0]
    for exp_key in ["EXP-1", "EXP-2", "EXP-3"]:
        if exp_key in results:
            hist = results[exp_key]["history"]
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["val_wind_mae_kt"] for h in hist], marker="s", color=colors[exp_key], label=exp_key)
    ax.set_title("Validation Wind MAE (kt)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (kt)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 4. Val Track Aggregate DPE
    ax = axes[1, 1]
    for exp_key in ["EXP-1", "EXP-2", "EXP-3"]:
        if exp_key in results:
            hist = results[exp_key]["history"]
            epochs = [h["epoch"] for h in hist]
            ax.plot(epochs, [h["val_track_dpe_km"] for h in hist], marker="^", color=colors[exp_key], label=exp_key)
    ax.set_title("Validation Track Aggregate Mean DPE (km)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPE (km)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    out_file = Path(output_dir) / "transfer_training_curves_expanded.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved transfer training curves to {out_file}")


def run_all():
    print("=========================================================================")
    print("VAYU-NET — MULTI-SOURCE TRANSFER LEARNING EXPERIMENT (EXPANDED TRAIN)")
    print("=========================================================================")

    # Load unified expanded cache
    cache_path = Path("data/interim/ml/cache/multisource_transfer_cache_expanded.pt")
    assert cache_path.exists(), f"Expanded cache missing at {cache_path}!"
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"Loaded cache with {len(cache_data['samples'])} samples.")

    device = torch.device("cpu")
    results = {}

    # 1. EXP1 — GridSat-only transfer baseline
    res_exp1 = train_transfer_experiment(
        exp_name="EXP-1",
        mode="exp1_gridsat_frozen",
        cache_data=cache_data,
        device=device,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_exp1_expanded.pt"
    )
    results["EXP-1"] = res_exp1

    # 2. EXP2 — GridSat + INSAT TIR1 + TIR2 (2 channels)
    res_exp2 = train_transfer_experiment(
        exp_name="EXP-2",
        mode="exp2_fusion_frozen",
        cache_data=cache_data,
        device=device,
        insat_channels=[0, 1],
        in_channels_insat=2,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_exp2_expanded.pt"
    )
    results["EXP-2"] = res_exp2

    # 3. EXP3 — GridSat + INSAT TIR1 + TIR2 + WV (3 channels)
    res_exp3 = train_transfer_experiment(
        exp_name="EXP-3",
        mode="exp2_fusion_frozen",
        cache_data=cache_data,
        device=device,
        insat_channels=[0, 1, 2],
        in_channels_insat=3,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_exp3_expanded.pt"
    )
    results["EXP-3"] = res_exp3

    # Load previous experiment results for direct side-by-side comparison
    prev_results_path = Path("data/interim/ml/multisource_transfer_results.json")
    prev_results = {}
    if prev_results_path.exists():
        with open(prev_results_path, "r") as f:
            prev_results = json.load(f)

    # Compile structured comparison table
    comp_rows = []
    for exp_k, label in [("EXP-1", "EXP1: GridSat Frozen"),
                         ("EXP-2", "EXP2: GridSat + INSAT TIR1/TIR2"),
                         ("EXP-3", "EXP3: GridSat + INSAT TIR1/TIR2/WV")]:
        tm = results[exp_k]["test_metrics"]
        vm = results[exp_k]["val_metrics"]
        
        row = {
            "Experiment": label,
            "Train_Samples": 207,
            "Val_Loss": round(results[exp_k]["val_loss"], 4),
            "Test_Loss": round(results[exp_k]["test_loss"], 4),
            "Center_Mean_DPE_km": round(tm["identification"]["center_mean_dpe_km"], 2),
            "Center_Median_DPE_km": round(tm["identification"]["center_median_dpe_km"], 2),
            "Center_P90_DPE_km": round(tm["identification"]["center_p90_dpe_km"], 2),
            "Center_Lat_MAE_deg": round(tm["identification"]["center_lat_mae_deg"], 3),
            "Center_Lon_MAE_deg": round(tm["identification"]["center_lon_mae_deg"], 3),
            "Category_Acc": round(tm["classification"]["accuracy"], 4),
            "Category_Macro_F1": round(tm["classification"]["macro_f1"], 4),
            "Wind_MAE_kt": round(tm["wind_regression"]["wind_mae_kt"], 2),
            "Wind_RMSE_kt": round(tm["wind_regression"]["wind_rmse_kt"], 2),
            "Wind_Median_AE_kt": round(tm["wind_regression"]["wind_median_ae_kt"], 2),
            "Wind_P90_AE_kt": round(tm["wind_regression"]["wind_p90_ae_kt"], 2),
            "Wind_Bias_kt": round(tm["wind_regression"]["wind_bias_kt"], 2),
            "Wind_Pearson_r": round(tm["wind_regression"]["wind_pearson_r"], 4),
            "Track_12h_DPE_km": round(tm["track_prediction"]["track_12h_mean_dpe_km"], 2),
            "Track_24h_DPE_km": round(tm["track_prediction"]["track_24h_mean_dpe_km"], 2),
            "Track_48h_DPE_km": round(tm["track_prediction"]["track_48h_mean_dpe_km"], 2),
            "Track_Aggregate_DPE_km": round(tm["track_prediction"]["track_aggregate_mean_dpe_km"], 2)
        }
        comp_rows.append(row)

    df_comp = pd.DataFrame(comp_rows)
    csv_out = "data/interim/ml/multisource_transfer_expanded_comparison.csv"
    df_comp.to_csv(csv_out, index=False)
    print(f"\nSaved comparison CSV to {csv_out}")

    # Save comprehensive results JSON
    results_out = {
        "metadata": {
            "title": "VAYU-NET Multi-Source Satellite Transfer Learning Experiment (Expanded TRAIN)",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_samples": 207,
            "train_storms": 25,
            "val_samples": 252,
            "val_storms": 14,
            "test_samples": 298,
            "test_storms": 24,
            "seed": 42,
            "training_hyperparameters": {
                "optimizer": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 16,
                "max_epochs": 20,
                "patience": 5,
                "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)"
            }
        },
        "experiments": results,
        "comparison_table": comp_rows,
        "previous_175_sample_results": {
            k: prev_results.get(k, {}) for k in ["EXP-1", "EXP-2", "Channel_B_TIR1_TIR2"]
        }
    }

    json_out = "data/interim/ml/multisource_transfer_expanded_results.json"
    with open(json_out, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"Saved results JSON to {json_out}")

    # Plot curves
    plot_expanded_transfer_curves(results)
    print("\nAll experiments complete.")

if __name__ == "__main__":
    run_all()
