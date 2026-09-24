"""
VAYU-NET PHASE 6 — TRAINING PIPELINE
====================================
Controlled Experiments:
  - EXP-1: Single-Frame t0 Satellite Baseline (mode='single_frame')
  - EXP-2: Six-Frame Satellite Temporal Model (mode='temporal_sat')
  - EXP-3: Six-Frame Satellite + ERA5 Multimodal Model (mode='multimodal_era5')

Discipline & Constraints:
  - TRAIN (696 samples / 81 storms) for parameter optimization
  - VALIDATION (252 samples / 14 storms) for model selection, early stopping, and temperature scaling
  - TEST (371 samples / 31 storms) evaluated strictly ONCE after checkpoint selection and freezing
  - Zero leakage, zero modifications to locked data or historical artifacts
"""

import os
import sys
import json
import math
import random
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.phase6_intensity_wind import (
    Phase6IntensityWindModel,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    DEFAULT_TRAIN_WIND_MEAN_KT,
    DEFAULT_TRAIN_WIND_STD_KT,
    DEFAULT_TRAIN_PRES_MEAN_HPA,
    DEFAULT_TRAIN_PRES_STD_HPA
)

CACHE_PATH = "data/interim/ml/cache/phase6_intensity_features.pt"
CHECKPOINT_DIR = "data/interim/ml/checkpoints"
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_phase6_intensity_wind.pt")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Phase6CachedDataset(Dataset):
    """Dataset serving cached multimodal features and intensity/wind targets."""
    def __init__(self, samples, split="TRAIN"):
        self.split = split.upper()
        self.samples = [s for s in samples if s["split"] == self.split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_phase6(batch):
    """Custom collate function for batch assembly."""
    sample_ids = [b["sample_id"] for b in batch]
    storm_ids = [b["storm_id"] for b in batch]
    t0s = [b["t0"] for b in batch]

    sat_seq = torch.stack([b["sat_seq"] for b in batch], dim=0) # [B, 6, 132]
    sat_t0 = torch.stack([b["sat_t0"] for b in batch], dim=0)   # [B, 132]
    env_seq = torch.stack([b["env_seq"] for b in batch], dim=0) # [B, 6, 8, 41, 66]

    category_t0 = torch.stack([b["category_t0"] for b in batch], dim=0) # [B]
    category_mask = torch.stack([b["category_t0_mask"] for b in batch], dim=0) # [B]

    wind_t0 = torch.stack([b["wind_t0"] for b in batch], dim=0) # [B]
    wind_norm = torch.stack([b["wind_t0_norm"] for b in batch], dim=0) # [B]
    wind_mask = torch.stack([b["wind_t0_mask"] for b in batch], dim=0) # [B]

    pressure_t0 = torch.stack([b["pressure_t0"] for b in batch], dim=0) # [B]
    pressure_norm = torch.stack([b["pressure_t0_norm"] for b in batch], dim=0) # [B]
    pressure_mask = torch.stack([b["pressure_t0_mask"] for b in batch], dim=0) # [B]

    lats = torch.tensor([b["imd_lat_t0"] for b in batch], dtype=torch.float32)
    lons = torch.tensor([b["imd_lon_t0"] for b in batch], dtype=torch.float32)

    return {
        "sample_id": sample_ids,
        "storm_id": storm_ids,
        "t0": t0s,
        "sat_seq": sat_seq,
        "sat_t0": sat_t0,
        "env_seq": env_seq,
        "category_t0": category_t0,
        "category_mask": category_mask,
        "wind_t0": wind_t0,
        "wind_norm": wind_norm,
        "wind_mask": wind_mask,
        "pressure_t0": pressure_t0,
        "pressure_norm": pressure_norm,
        "pressure_mask": pressure_mask,
        "lat_t0": lats,
        "lon_t0": lons
    }


def compute_train_class_weights(train_samples):
    """
    Computes smoothed class weights based ONLY on TRAIN split distribution.
    Uses square-root smoothed inverse frequency clipped to [0.25, 5.0] and normalized to mean 1.0.
    """
    counts = np.zeros(7, dtype=np.float32)
    for s in train_samples:
        c = s["category_t0"].item()
        if c >= 0:
            counts[c] += 1.0

    total = counts.sum()
    smoothed = np.clip(np.sqrt(total / (7.0 * counts)), 0.25, 5.0)
    weights = smoothed / smoothed.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model, dataloader, device, ce_loss_fn, smooth_l1_fn, loss_weights):
    """Evaluates multi-task model performance across a split."""
    model.eval()

    total_loss_sum = 0.0
    cat_loss_sum = 0.0
    wind_loss_sum = 0.0
    total_samples = 0

    all_pred_cats, all_true_cats = [], []
    all_pred_probs = []
    all_pred_winds_kt, all_true_winds_kt = [], []
    all_pred_pres_hpa, all_true_pres_hpa = [], []

    train_wind_mean = model.train_wind_mean.item()
    train_wind_std = model.train_wind_std.item()

    with torch.no_grad():
        for batch in dataloader:
            B = batch["sat_seq"].size(0)
            total_samples += B

            sat_seq = batch["sat_seq"].to(device)
            sat_t0 = batch["sat_t0"].to(device)
            env_seq = batch["env_seq"].to(device)

            target_cat = batch["category_t0"].to(device)
            target_wind_norm = batch["wind_norm"].to(device)
            wind_mask = batch["wind_mask"].to(device)

            out = model(sat_seq=sat_seq, sat_t0=sat_t0, env_seq=env_seq)

            # Category loss
            loss_c = ce_loss_fn(out["category_logits"], target_cat)

            # Wind loss
            wind_diff = smooth_l1_fn(out["norm_wind"], target_wind_norm)
            valid_w = torch.clamp(wind_mask.sum(), min=1.0)
            loss_w = (wind_diff * wind_mask).sum() / valid_w

            total_loss = loss_weights["category"] * loss_c + loss_weights["wind"] * loss_w

            total_loss_sum += total_loss.item() * B
            cat_loss_sum += loss_c.item() * B
            wind_loss_sum += loss_w.item() * B

            # Predictions
            p_cat = out["category_pred"].cpu().numpy()
            p_prob = out["category_probs"].cpu().numpy()
            t_cat = target_cat.cpu().numpy()
            for pc, pp, tc in zip(p_cat, p_prob, t_cat):
                if tc >= 0:
                    all_pred_cats.append(int(pc))
                    all_pred_probs.append(pp)
                    all_true_cats.append(int(tc))

            p_wind = out["pred_wind_kt"].cpu().numpy()
            t_wind = batch["wind_t0"].numpy()
            w_m = wind_mask.cpu().numpy()
            for pw, tw, m in zip(p_wind, t_wind, w_m):
                if m > 0.5:
                    all_pred_winds_kt.append(float(pw))
                    all_true_winds_kt.append(float(tw))

    # Metrics computation
    avg_total_loss = total_loss_sum / total_samples
    avg_cat_loss = cat_loss_sum / total_samples
    avg_wind_loss = wind_loss_sum / total_samples

    # Classification metrics
    if len(all_true_cats) > 0:
        acc = float(accuracy_score(all_true_cats, all_pred_cats))
        macro_f1 = float(f1_score(all_true_cats, all_pred_cats, average="macro", zero_division=0))
        macro_prec = float(precision_score(all_true_cats, all_pred_cats, average="macro", zero_division=0))
        macro_rec = float(recall_score(all_true_cats, all_pred_cats, average="macro", zero_division=0))
        per_class_f1 = f1_score(all_true_cats, all_pred_cats, average=None, labels=list(range(7)), zero_division=0).tolist()
        cm = confusion_matrix(all_true_cats, all_pred_cats, labels=list(range(7))).tolist()
    else:
        acc, macro_f1, macro_prec, macro_rec, per_class_f1, cm = 0.0, 0.0, 0.0, 0.0, [0.0]*7, []

    # Wind regression metrics
    if len(all_true_winds_kt) > 0:
        pred_w_arr = np.array(all_pred_winds_kt)
        true_w_arr = np.array(all_true_winds_kt)
        diff = pred_w_arr - true_w_arr
        abs_diff = np.abs(diff)

        wind_mae = float(np.mean(abs_diff))
        wind_rmse = float(np.sqrt(np.mean(diff**2)))
        wind_median_ae = float(np.median(abs_diff))
        wind_p80_ae = float(np.percentile(abs_diff, 80))
        wind_p90_ae = float(np.percentile(abs_diff, 90))
        wind_bias = float(np.mean(diff))
        corr_matrix = np.corrcoef(pred_w_arr, true_w_arr)
        wind_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        wind_mae, wind_rmse, wind_median_ae, wind_p80_ae, wind_p90_ae, wind_bias, wind_corr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Composite validation error index: (1 - Macro F1) + (Wind MAE / Train Wind Std)
    norm_wind_mae = wind_mae / train_wind_std
    composite_error = (1.0 - macro_f1) + norm_wind_mae

    return {
        "loss_total": avg_total_loss,
        "loss_category": avg_cat_loss,
        "loss_wind": avg_wind_loss,
        "composite_error": composite_error,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "wind_mae": wind_mae,
        "wind_rmse": wind_rmse,
        "wind_median_ae": wind_median_ae,
        "wind_p80_ae": wind_p80_ae,
        "wind_p90_ae": wind_p90_ae,
        "wind_bias": wind_bias,
        "wind_corr": wind_corr,
        "y_true_cat": all_true_cats,
        "y_pred_cat": all_pred_cats,
        "y_pred_probs": all_pred_probs,
        "y_true_wind": all_true_winds_kt,
        "y_pred_wind": all_pred_winds_kt
    }


def fit_temperature_scaling(model, val_loader, device):
    """
    Fits temperature scaling parameter T on VALIDATION split using Adam optimizer.
    Minimizes CrossEntropyLoss on validation logits with frozen model trunk.
    """
    print("Fitting Temperature Scaling on VALIDATION split...")
    model.eval()
    
    # Collect all validation logits and targets
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            sat_seq = batch["sat_seq"].to(device)
            sat_t0 = batch["sat_t0"].to(device)
            env_seq = batch["env_seq"].to(device)
            cat_target = batch["category_t0"].to(device)

            out = model(sat_seq=sat_seq, sat_t0=sat_t0, env_seq=env_seq)
            mask = cat_target >= 0
            if mask.sum() > 0:
                all_logits.append(out["category_logits"][mask])
                all_targets.append(cat_target[mask])

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # Temperature parameter
    temperature = nn.Parameter(torch.ones(1, device=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([temperature], lr=0.01)

    initial_loss = criterion(logits_tensor, targets_tensor).item()

    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(logits_tensor / torch.clamp(temperature, min=0.01), targets_tensor)
        loss.backward()
        optimizer.step()

    optimal_T = float(torch.clamp(temperature, min=0.01).item())
    final_loss = criterion(logits_tensor / optimal_T, targets_tensor).item()

    model.set_temperature(optimal_T)
    print(f"  -> Temperature fitted: T = {optimal_T:.4f} (Validation NLL: {initial_loss:.4f} -> {final_loss:.4f})")
    return optimal_T


def train_experiment(exp_name, mode, train_loader, val_loader, device, class_weights, loss_weights, max_epochs=25, patience=6, lr=1e-3):
    """Trains a single model variant with early stopping governed by composite validation error."""
    print("=" * 80)
    print(f"STARTING EXPERIMENT: {exp_name} (mode={mode})")
    print("=" * 80)

    model = Phase6IntensityWindModel(
        mode=mode,
        latent_dim=128,
        gru_num_layers=2,
        dropout=0.1
    ).to(device)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=-1)
    smooth_l1_fn = nn.SmoothL1Loss(reduction="none", beta=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_error = float("inf")
    best_epoch = -1
    best_metrics = None
    best_state_dict = None
    no_improve = 0

    history = []

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_cat_loss_sum = 0.0
        train_wind_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            B = batch["sat_seq"].size(0)
            train_samples += B

            sat_seq = batch["sat_seq"].to(device)
            sat_t0 = batch["sat_t0"].to(device)
            env_seq = batch["env_seq"].to(device)

            target_cat = batch["category_t0"].to(device)
            target_wind_norm = batch["wind_norm"].to(device)
            wind_mask = batch["wind_mask"].to(device)

            optimizer.zero_grad()
            out = model(sat_seq=sat_seq, sat_t0=sat_t0, env_seq=env_seq)

            loss_c = ce_loss_fn(out["category_logits"], target_cat)
            wind_diff = smooth_l1_fn(out["norm_wind"], target_wind_norm)
            valid_w = torch.clamp(wind_mask.sum(), min=1.0)
            loss_w = (wind_diff * wind_mask).sum() / valid_w

            total_loss = loss_weights["category"] * loss_c + loss_weights["wind"] * loss_w
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += total_loss.item() * B
            train_cat_loss_sum += loss_c.item() * B
            train_wind_loss_sum += loss_w.item() * B

        epoch_train_loss = train_loss_sum / train_samples
        epoch_cat_loss = train_cat_loss_sum / train_samples
        epoch_wind_loss = train_wind_loss_sum / train_samples

        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, device, ce_loss_fn, smooth_l1_fn, loss_weights)
        scheduler.step(val_metrics["composite_error"])

        curr_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_cat_loss": epoch_cat_loss,
            "train_wind_loss": epoch_wind_loss,
            "val_loss": val_metrics["loss_total"],
            "val_composite_error": val_metrics["composite_error"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_wind_mae": val_metrics["wind_mae"],
            "val_wind_rmse": val_metrics["wind_rmse"],
            "learning_rate": curr_lr
        })

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss_total']:.4f} | "
              f"Val CompErr: {val_metrics['composite_error']:.4f} | "
              f"Val MacroF1: {val_metrics['macro_f1']:.4f} (Acc: {val_metrics['accuracy']*100:.1f}%) | "
              f"Val Wind MAE: {val_metrics['wind_mae']:.2f} kt | LR: {curr_lr:.1e}")

        # Checkpoint selection strictly governed by validation composite error
        if val_metrics["composite_error"] < best_val_error:
            best_val_error = val_metrics["composite_error"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> Early stopping triggered after {patience} epochs without validation improvement.")
                break

    elapsed = time.time() - start_time
    print(f"Experiment {exp_name} completed in {elapsed:.1f}s. Best epoch: {best_epoch} with Val CompErr: {best_val_error:.4f}")

    # Load best weights
    model.load_state_dict(best_state_dict)

    # Fit temperature scaling calibration on VALIDATION
    optimal_T = fit_temperature_scaling(model, val_loader, device)

    # Calculate empirical uncertainty bounds on VALIDATION residuals
    val_final_eval = evaluate_model(model, val_loader, device, ce_loss_fn, smooth_l1_fn, loss_weights)
    model.set_uncertainty_bounds(
        median_ae=val_final_eval["wind_median_ae"],
        p80_ae=val_final_eval["wind_p80_ae"],
        p90_ae=val_final_eval["wind_p90_ae"]
    )

    return {
        "exp_name": exp_name,
        "mode": mode,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_error": best_val_error,
        "best_metrics": best_metrics,
        "val_final_metrics": val_final_eval,
        "temperature": optimal_T,
        "uncertainty_bounds": {
            "median_ae": val_final_eval["wind_median_ae"],
            "p80_ae": val_final_eval["wind_p80_ae"],
            "p90_ae": val_final_eval["wind_p90_ae"]
        },
        "history": history
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load cached dataset
    print(f"Loading cached features from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    samples = cache["samples"]

    train_dataset = Phase6CachedDataset(samples, split="TRAIN")
    val_dataset = Phase6CachedDataset(samples, split="VALIDATION")
    test_dataset = Phase6CachedDataset(samples, split="TEST")

    print(f"Dataset splits: TRAIN={len(train_dataset)}, VAL={len(val_dataset)}, TEST={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_phase6)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_phase6)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_phase6)

    # 2. Compute TRAIN-only class weights
    class_weights = compute_train_class_weights(train_dataset.samples)
    print("TRAIN Class Weights:")
    for c, w in zip(["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"], class_weights):
        print(f"  {c:5s}: {w.item():.4f}")

    loss_weights = {
        "category": 1.0,
        "wind": 1.0
    }

    # 3. Controlled Experiments
    experiments = [
        {"name": "EXP-1_SingleFrame_t0", "mode": "single_frame", "epochs": 20, "patience": 5, "lr": 1e-3},
        {"name": "EXP-2_Temporal_Sat_GRU", "mode": "temporal_sat", "epochs": 25, "patience": 6, "lr": 1e-3},
        {"name": "EXP-3_Multimodal_ERA5", "mode": "multimodal_era5", "epochs": 25, "patience": 6, "lr": 5e-4}
    ]

    results = {}
    for exp in experiments:
        res = train_experiment(
            exp_name=exp["name"],
            mode=exp["mode"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=class_weights,
            loss_weights=loss_weights,
            max_epochs=exp["epochs"],
            patience=exp["patience"],
            lr=exp["lr"]
        )
        results[exp["name"]] = res

    # 4. Checkpoint Selection Governed Strictly by Validation Composite Error
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY & MODEL SELECTION")
    print("=" * 80)

    summary_rows = []
    best_overall_exp = None
    lowest_val_error = float("inf")

    for name, res in results.items():
        vm = res["val_final_metrics"]
        err = vm["composite_error"]
        summary_rows.append({
            "Experiment": name,
            "Mode": res["mode"],
            "Best Epoch": res["best_epoch"],
            "Val Composite Error": f"{err:.4f}",
            "Val Macro F1": f"{vm['macro_f1']:.4f}",
            "Val Accuracy": f"{vm['accuracy']*100:.2f}%",
            "Val Wind MAE": f"{vm['wind_mae']:.2f} kt",
            "Val Wind RMSE": f"{vm['wind_rmse']:.2f} kt",
            "Calibration T": f"{res['temperature']:.4f}"
        })
        if err < lowest_val_error:
            lowest_val_error = err
            best_overall_exp = name

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n>>> Selected Winning Model: {best_overall_exp} (Lowest Val CompErr = {lowest_val_error:.4f}) <<<")

    # 5. Freeze Selected Checkpoint
    selected_result = results[best_overall_exp]
    selected_model = selected_result["model"]

    checkpoint_payload = {
        "model_state_dict": selected_model.state_dict(),
        "experiment_name": best_overall_exp,
        "mode": selected_result["mode"],
        "best_epoch": selected_result["best_epoch"],
        "val_metrics": {
            "composite_error": selected_result["val_final_metrics"]["composite_error"],
            "macro_f1": selected_result["val_final_metrics"]["macro_f1"],
            "accuracy": selected_result["val_final_metrics"]["accuracy"],
            "wind_mae": selected_result["val_final_metrics"]["wind_mae"],
            "wind_rmse": selected_result["val_final_metrics"]["wind_rmse"],
            "per_class_f1": selected_result["val_final_metrics"]["per_class_f1"]
        },
        "temperature": selected_result["temperature"],
        "uncertainty_bounds": selected_result["uncertainty_bounds"],
        "train_class_weights": class_weights.tolist(),
        "loss_weights": loss_weights,
        "config": {
            "latent_dim": 128,
            "gru_num_layers": 2,
            "dropout": 0.1,
            "train_wind_mean": DEFAULT_TRAIN_WIND_MEAN_KT,
            "train_wind_std": DEFAULT_TRAIN_WIND_STD_KT
        }
    }

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(checkpoint_payload, BEST_CHECKPOINT_PATH)
    print(f"[Frozen Checkpoint] Successfully saved to {BEST_CHECKPOINT_PATH}")

    # Also save training history for later visualization
    history_path = "data/interim/ml/phase6_training_history.json"
    clean_history = {}
    for k, v in results.items():
        clean_history[k] = {
            "history": v["history"],
            "best_epoch": v["best_epoch"],
            "temperature": v["temperature"],
            "val_metrics": {
                "composite_error": v["val_final_metrics"]["composite_error"],
                "macro_f1": v["val_final_metrics"]["macro_f1"],
                "accuracy": v["val_final_metrics"]["accuracy"],
                "wind_mae": v["val_final_metrics"]["wind_mae"],
                "wind_rmse": v["val_final_metrics"]["wind_rmse"]
            }
        }
    with open(history_path, "w") as f:
        json.dump(clean_history, f, indent=2)
    print(f"[Training History] Saved to {history_path}")

    print("\nTraining and validation-based selection complete. Checkpoint is frozen.")
    print("Ready for single-pass TEST evaluation via scripts/production/evaluate_phase6_intensity_wind.py.")


if __name__ == "__main__":
    main()
