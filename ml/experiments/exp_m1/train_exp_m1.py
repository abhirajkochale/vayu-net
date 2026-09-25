"""
VAYU-NET: EXP-M1 Multi-Task Model Training Pipeline.
Trains M1A (GridSat-only), M1B (IMERG-only), and M1C (Multimodal Fusion)
under strictly identical optimization, data partitions, loss definitions, and early stopping.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.experiments.exp_m1.model import (
    ExpM1Model, haversine_km,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY
)
from ml.experiments.exp_m1.losses import MaskedMultiTaskLoss, encode_coordinates, normalize_wind

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
CHECKPOINT_DIR = REPO_ROOT / "ml/experiments/exp_m1/checkpoints"
RESULTS_DIR = REPO_ROOT / "ml/experiments/exp_m1/results"
LOGS_DIR = REPO_ROOT / "ml/experiments/exp_m1/logs"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedMultiTaskDataset(Dataset):
    """Serves preprocessed tensors from the EXP-M1 memory cache."""
    def __init__(self, samples: List[Dict[str, Any]], split: str):
        self.split = split
        self.samples = [s for s in samples if s["split"] == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def evaluate_split(model: ExpM1Model,
                   loader: DataLoader,
                   criterion: MaskedMultiTaskLoss,
                   device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Evaluates validation loss and task-specific component metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    loss_acc = {
        "loss_center": 0.0, "loss_wind": 0.0, "loss_class": 0.0,
        "loss_12": 0.0, "loss_24": 0.0, "loss_48": 0.0, "loss_track": 0.0
    }

    all_dpes_center = []
    all_dpes_12 = []
    all_dpes_24 = []
    all_dpes_48 = []
    all_wind_abs_diff = []
    all_cat_preds = []
    all_cat_trues = []

    with torch.no_grad():
        for batch in loader:
            g_seq = batch["gridsat"].to(device) if model.mode in ["m1a_gridsat", "m1c_fusion"] else None
            i_seq = batch["imerg"].to(device) if model.mode in ["m1b_imerg", "m1c_fusion"] else None

            preds = model(gridsat_seq=g_seq, imerg_seq=i_seq)
            loss, loss_dict = criterion(preds, batch)

            total_loss += loss.item()
            for k in loss_acc:
                loss_acc[k] += loss_dict.get(k, 0.0)
            num_batches += 1

            # Center DPE
            pred_centers = preds["center_deg"].cpu().numpy()
            true_centers = batch["center_t0"].numpy()
            for pc, tc in zip(pred_centers, true_centers):
                all_dpes_center.append(haversine_km(pc[0], pc[1], tc[0], tc[1]))

            # Wind MAE
            pred_winds = preds["wind_kt"].squeeze(-1).cpu().numpy()
            true_winds = batch["wind_t0"].numpy()
            w_masks = batch["wind_t0_mask"].numpy()
            for pw, tw, wm in zip(pred_winds, true_winds, w_masks):
                if wm > 0.5:
                    all_wind_abs_diff.append(abs(pw - tw))

            # Classification
            pred_cats = torch.argmax(preds["class_logits"], dim=-1).cpu().numpy()
            true_cats = batch["category_t0"].numpy()
            cat_masks = batch["category_t0_mask"].numpy()
            for pc, tc, cm in zip(pred_cats, true_cats, cat_masks):
                if cm > 0.5 and tc >= 0:
                    all_cat_preds.append(pc)
                    all_cat_trues.append(tc)

            # Track DPEs
            p12 = preds["track_12h_deg"].cpu().numpy()
            t12 = batch["center_12h"].numpy()
            m12 = batch["wind_12h_mask"].numpy()
            for p, t, m in zip(p12, t12, m12):
                if m > 0.5:
                    all_dpes_12.append(haversine_km(p[0], p[1], t[0], t[1]))

            p24 = preds["track_24h_deg"].cpu().numpy()
            t24 = batch["center_24h"].numpy()
            m24 = batch["wind_24h_mask"].numpy()
            for p, t, m in zip(p24, t24, m24):
                if m > 0.5:
                    all_dpes_24.append(haversine_km(p[0], p[1], t[0], t[1]))

            p48 = preds["track_48h_deg"].cpu().numpy()
            t48 = batch["center_48h"].numpy()
            m48 = batch["wind_48h_mask"].numpy()
            for p, t, m in zip(p48, t48, m48):
                if m > 0.5:
                    all_dpes_48.append(haversine_km(p[0], p[1], t[0], t[1]))

    avg_loss = total_loss / max(1, num_batches)
    avg_loss_dict = {k: v / max(1, num_batches) for k, v in loss_acc.items()}
    avg_loss_dict["loss_total"] = avg_loss

    # Compute validation scientific metrics
    metrics = {
        "val_total_loss": avg_loss,
        "val_center_dpe_km": float(np.mean(all_dpes_center)) if all_dpes_center else 0.0,
        "val_center_median_dpe_km": float(np.median(all_dpes_center)) if all_dpes_center else 0.0,
        "val_wind_mae_kt": float(np.mean(all_wind_abs_diff)) if all_wind_abs_diff else 0.0,
        "val_cat_accuracy": float(accuracy_score(all_cat_trues, all_cat_preds)) if all_cat_trues else 0.0,
        "val_cat_macro_f1": float(f1_score(all_cat_trues, all_cat_preds, average="macro", zero_division=0)) if all_cat_trues else 0.0,
        "val_track_12h_dpe_km": float(np.mean(all_dpes_12)) if all_dpes_12 else 0.0,
        "val_track_24h_dpe_km": float(np.mean(all_dpes_24)) if all_dpes_24 else 0.0,
        "val_track_48h_dpe_km": float(np.mean(all_dpes_48)) if all_dpes_48 else 0.0,
    }
    metrics.update(avg_loss_dict)
    return avg_loss, metrics


def train_single_experiment(mode: str,
                            cache_payload: Dict[str, Any],
                            device: torch.device,
                            epochs: int = 20,
                            batch_size: int = 16,
                            lr: float = 1e-3,
                            weight_decay: float = 1e-4,
                            patience: int = 5,
                            checkpoint_name: str = "model.pt") -> Dict[str, Any]:
    """Trains a single model under EXP-M1 governance."""
    set_seed(42)
    print(f"\n=======================================================")
    print(f"TRAINING EXP-M1: {mode.upper()}")
    print(f"=======================================================")

    samples = cache_payload["samples"]
    train_ds = CachedMultiTaskDataset(samples, "TRAIN")
    val_ds = CachedMultiTaskDataset(samples, "VALIDATION")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset: TRAIN={len(train_ds)}, VALIDATION={len(val_ds)}")

    model = ExpM1Model(mode=mode).to(device)
    params = model.get_parameter_counts()
    print(f"Parameters: Total={params['total_parameters']:,} | Trainable={params['trainable_parameters']:,}")

    criterion = MaskedMultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    best_metrics = None
    epochs_no_improve = 0

    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            g_seq = batch["gridsat"].to(device) if mode in ["m1a_gridsat", "m1c_fusion"] else None
            i_seq = batch["imerg"].to(device) if mode in ["m1b_imerg", "m1c_fusion"] else None

            preds = model(gridsat_seq=g_seq, imerg_seq=i_seq)
            loss, _ = criterion(preds, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)
        val_loss, val_metrics = evaluate_split(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics
        }
        history.append(epoch_record)

        print(f"  Epoch {epoch:2d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Center DPE: {val_metrics['val_center_dpe_km']:.1f} km | Wind MAE: {val_metrics['val_wind_mae_kt']:.2f} kt | "
              f"Cat Acc: {val_metrics['val_cat_accuracy']*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    training_time = time.time() - t_start
    print(f"\nTraining Complete ({training_time:.1f}s). Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Save best checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    checkpoint_payload = {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "model_state_dict": best_state_dict,
        "parameter_counts": params,
        "training_time_seconds": training_time,
        "epochs_trained": len(history),
        "seed": 42
    }
    torch.save(checkpoint_payload, ckpt_path)
    print(f"Saved best checkpoint to: {ckpt_path}")

    # Save history
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = RESULTS_DIR / f"{Path(checkpoint_name).stem}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to: {hist_path}")

    return {
        "mode": mode,
        "checkpoint_path": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "training_time_seconds": training_time,
        "parameters": params,
        "history": history
    }


def main():
    if not CACHE_PATH.exists():
        print(f"Error: Tensor cache not found at {CACHE_PATH}. Run build_cache.py first.")
        sys.exit(1)

    print(f"Loading tensor cache from {CACHE_PATH}...")
    cache_payload = torch.load(CACHE_PATH)
    print(f"Cache loaded with {cache_payload['metadata']['num_samples']} total samples.")

    device = torch.device("cpu")
    print(f"Training device: {device}")

    runs = [
        ("m1a_gridsat", "m1a_gridsat_only.pt"),
        ("m1b_imerg",   "m1b_imerg_only.pt"),
        ("m1c_fusion",  "m1c_gridsat_imerg.pt")
    ]

    all_results = {}
    for mode, ckpt_name in runs:
        res = train_single_experiment(
            mode=mode,
            cache_payload=cache_payload,
            device=device,
            epochs=20,
            batch_size=16,
            lr=1e-3,
            patience=5,
            checkpoint_name=ckpt_name
        )
        all_results[mode] = res

    # Summary table
    print("\n" + "=" * 65)
    print("VAYU-NET: EXP-M1 TRAINING SUMMARY (VALIDATION SELECTION)")
    print("=" * 65)
    print(f"{'Configuration':<20} {'Parameters':<12} {'Best Epoch':<12} {'Val Loss':<12} {'Center DPE':<14} {'Wind MAE':<12}")
    print("-" * 82)
    for mode, res in all_results.items():
        params = res["parameters"]["trainable_parameters"]
        b_ep = res["best_epoch"]
        v_loss = res["best_val_loss"]
        c_dpe = res["best_metrics"]["val_center_dpe_km"]
        w_mae = res["best_metrics"]["val_wind_mae_kt"]
        print(f"{mode:<20} {params:<12,d} {b_ep:<12d} {v_loss:<12.4f} {c_dpe:<14.1f} {w_mae:<12.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()