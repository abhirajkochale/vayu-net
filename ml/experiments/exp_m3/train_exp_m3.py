"""
VAYU-NET: EXP-M3 Multi-Task Model Training Pipeline.
Trains M3A (GridSat-only), M3B (IMERG-only), and M3C (Granular Spatially Adaptive Fusion)
under strictly locked EXP-M1 training protocol: batch_size=16, seed=42, AdamW, ReduceLROnPlateau.
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

from ml.experiments.exp_m3.model import (
    ExpM3Model, haversine_km,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY
)
from ml.experiments.exp_m3.losses import MaskedMultiTaskLoss, encode_coordinates, normalize_wind

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
CHECKPOINT_DIR = REPO_ROOT / "ml/experiments/exp_m3/checkpoints"
RESULTS_DIR = REPO_ROOT / "ml/experiments/exp_m3/results"
LOGS_DIR = REPO_ROOT / "ml/experiments/exp_m3/logs"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedMultiTaskDataset(Dataset):
    """Serves preprocessed tensors from the memory cache."""
    def __init__(self, samples: List[Dict[str, Any]], split: str):
        self.split = split
        self.samples = [s for s in samples if s["split"] == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def evaluate_split(model: ExpM3Model,
                   loader: DataLoader,
                   criterion: MaskedMultiTaskLoss,
                   device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Evaluates validation loss, task-specific component metrics, and spatial gates."""
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
    all_spatial_gates = []

    with torch.no_grad():
        for batch in loader:
            g_seq = batch["gridsat"].to(device) if model.mode in ["m3a_gridsat", "m3c_spatial"] else None
            i_seq = batch["imerg"].to(device) if model.mode in ["m3b_imerg", "m3c_spatial"] else None

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

            # Track DPEs (+12, +24, +48)
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

            # Wind speed MAE
            pred_wind = preds["wind_kt"].cpu().numpy().flatten()
            true_wind = batch["wind_t0"].numpy().flatten()
            w_mask = batch["wind_t0_mask"].numpy().flatten()
            for pw, tw, m in zip(pred_wind, true_wind, w_mask):
                if m > 0.5:
                    all_wind_abs_diff.append(abs(pw - tw))

            # Category Accuracy and Macro F1
            pred_logits = preds["class_logits"].cpu().numpy()
            true_cats = batch["category_t0"].numpy()
            cat_mask = batch["category_t0_mask"].numpy()
            for pl, tc, m in zip(pred_logits, true_cats, cat_mask):
                if m > 0.5:
                    all_cat_preds.append(np.argmax(pl))
                    all_cat_trues.append(tc)

            # Record spatial gate metrics for M3C
            if "spatial_gate_alpha" in preds:
                gates = preds["spatial_gate_alpha"].cpu().numpy()  # [B, 6, 64, 9, 15]
                all_spatial_gates.append(gates)

    mean_loss = total_loss / max(1, num_batches)
    avg_losses = {k: v / max(1, num_batches) for k, v in loss_acc.items()}

    center_dpe = float(np.mean(all_dpes_center)) if all_dpes_center else 0.0
    center_median_dpe = float(np.median(all_dpes_center)) if all_dpes_center else 0.0
    wind_mae = float(np.mean(all_wind_abs_diff)) if all_wind_abs_diff else 0.0
    t12_dpe = float(np.mean(all_dpes_12)) if all_dpes_12 else 0.0
    t24_dpe = float(np.mean(all_dpes_24)) if all_dpes_24 else 0.0
    t48_dpe = float(np.mean(all_dpes_48)) if all_dpes_48 else 0.0

    if all_cat_trues and len(all_cat_trues) > 0:
        cat_acc = float(accuracy_score(all_cat_trues, all_cat_preds))
        cat_f1 = float(f1_score(all_cat_trues, all_cat_preds, average="macro", zero_division=0))
    else:
        cat_acc, cat_f1 = 0.0, 0.0

    metrics = {
        "val_total_loss": mean_loss,
        "val_center_dpe_km": center_dpe,
        "val_center_median_dpe_km": center_median_dpe,
        "val_wind_mae_kt": wind_mae,
        "val_cat_accuracy": cat_acc,
        "val_cat_macro_f1": cat_f1,
        "val_track_12h_dpe_km": t12_dpe,
        "val_track_24h_dpe_km": t24_dpe,
        "val_track_48h_dpe_km": t48_dpe,
        **avg_losses
    }

    if all_spatial_gates:
        cat_gates = np.concatenate(all_spatial_gates, axis=0) # [N, 6, 64, 9, 15]
        metrics["val_gate_gridsat_mean"] = float(np.mean(cat_gates))
        metrics["val_gate_gridsat_std"] = float(np.std(cat_gates))
        metrics["val_gate_imerg_mean"] = float(1.0 - np.mean(cat_gates))
        metrics["val_gate_spatial_variance"] = float(np.mean(np.var(cat_gates, axis=(-2, -1))))

    return mean_loss, metrics


def train_single_model(mode: str,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       device: torch.device,
                       max_epochs: int = 20,
                       lr: float = 1e-3,
                       weight_decay: float = 1e-4,
                       early_stopping_patience: int = 5) -> Dict[str, Any]:
    """Trains a single model under strict early stopping and validation selection."""
    print(f"\n{'='*55}")
    print(f"TRAINING EXP-M3: {mode.upper()} (batch_size=16)")
    print(f"{'='*55}")
    print(f"Dataset: TRAIN={len(train_loader.dataset)}, VALIDATION={len(val_loader.dataset)}")

    set_seed(42)
    model = ExpM3Model(mode=mode).to(device)
    params = model.get_parameter_counts()
    print(f"Parameters: Total={params['total_parameters']:,} | Trainable={params['trainable_parameters']:,}")

    criterion = MaskedMultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    best_metrics = {}
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            g_seq = batch["gridsat"].to(device) if mode in ["m3a_gridsat", "m3c_spatial"] else None
            i_seq = batch["imerg"].to(device) if mode in ["m3b_imerg", "m3c_spatial"] else None

            preds = model(gridsat_seq=g_seq, imerg_seq=i_seq)
            loss, _ = criterion(preds, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = train_loss / max(1, num_train_batches)
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

        gate_str = ""
        if "val_gate_gridsat_mean" in val_metrics:
            gate_str = f" | Spatial Gate(GS/IM): {val_metrics['val_gate_gridsat_mean']:.3f}/{val_metrics['val_gate_imerg_mean']:.3f} (spat_var={val_metrics['val_gate_spatial_variance']:.5f})"

        print(f"  Epoch {epoch:2d}/{max_epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Center DPE: {val_metrics['val_center_dpe_km']:.1f} km | Wind MAE: {val_metrics['val_wind_mae_kt']:.2f} kt | "
              f"Cat Acc: {val_metrics['val_cat_accuracy']*100:.1f}%{gate_str}")

        # Validation-only model selection
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping triggered at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining Complete ({elapsed:.1f}s). Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Checkpoint saving
    ckpt_name = {
        "m3a_gridsat": "m3a_gridsat_only.pt",
        "m3b_imerg": "m3b_imerg_only.pt",
        "m3c_spatial": "m3c_spatial_adaptive_fusion.pt"
    }[mode]
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    torch.save({
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": best_state_dict,
        "parameter_counts": params,
        "training_time_seconds": elapsed,
        "best_validation_metrics": best_metrics,
        "batch_size": 16,
        "seed": 42
    }, ckpt_path)
    print(f"Saved best checkpoint to: {ckpt_path}")

    # History saving
    hist_name = {
        "m3a_gridsat": "m3a_gridsat_only_history.json",
        "m3b_imerg": "m3b_imerg_only_history.json",
        "m3c_spatial": "m3c_spatial_adaptive_fusion_history.json"
    }[mode]
    hist_path = RESULTS_DIR / hist_name
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to: {hist_path}")

    return {
        "mode": mode,
        "checkpoint_path": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time": elapsed,
        "parameter_counts": params,
        "best_metrics": best_metrics,
        "history": history
    }


def main():
    print("=" * 65)
    print("VAYU-NET: EXP-M3 MULTIMODAL ABLATION TRAINING (LOCKED BATCH_SIZE=16)")
    print("=" * 65)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tensor cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    all_samples = cache["samples"]
    print(f"Cache loaded with {len(all_samples)} total samples.")

    train_ds = CachedMultiTaskDataset(all_samples, split="TRAIN")
    val_ds = CachedMultiTaskDataset(all_samples, split="VALIDATION")

    set_seed(42)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=False)

    device = torch.device("cpu")
    print(f"Training device: {device}")

    results = {}
    for mode in ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]:
        res = train_single_model(
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=20,
            lr=1e-3,
            weight_decay=1e-4,
            early_stopping_patience=5
        )
        results[mode] = res

    print("\n" + "=" * 65)
    print("VAYU-NET: EXP-M3 TRAINING SUMMARY (VALIDATION SELECTION)")
    print("=" * 65)
    print(f"{'Configuration':<20} {'Parameters':<12} {'Best Epoch':<12} {'Val Loss':<12} {'Center DPE':<14} {'Wind MAE':<12}")
    print("-" * 82)
    for mode, r in results.items():
        p_count = f"{r['parameter_counts']['trainable_parameters']:,}"
        bm = r["best_metrics"]
        print(f"{mode:<20} {p_count:<12} {r['best_epoch']:<12d} {r['best_val_loss']:<12.4f} "
              f"{bm['val_center_dpe_km']:<14.1f} {bm['val_wind_mae_kt']:<12.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
