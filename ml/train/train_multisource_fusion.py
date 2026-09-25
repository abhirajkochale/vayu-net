"""
VAYU-NET — MULTI-SOURCE SATELLITE MODEL TRAINING PIPELINE
==========================================================
Trains and validates Model A (GridSat only), Model B (INSAT only),
and Model C (GridSat + INSAT Fusion) on the locked 725-sample paired dataset.

Tasks Evaluated:
  A. Identification: Current center latitude/longitude (DPE km, MAE deg)
  B. Classification: 7-class IMD intensity category (Accuracy, Macro-F1, Recall)
  C. Wind Regression: Maximum sustained wind in knots (MAE, RMSE, MedAE, P90, Bias, r)
  D. Track Prediction: +12h, +24h, +48h displacements (DPE km per horizon and aggregate)

Governance:
  - Fixed seed: 42
  - Deterministic splits: 175 TRAIN, 252 VALIDATION, 298 TEST
  - Strict early stopping on VALIDATION loss
  - Model selection based strictly on validation performance
  - Test set evaluated exactly ONCE upon selection
"""

import os
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

from ml.models.multisource_fusion import (
    MultisourceFusionModel,
    haversine_km,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultisourceCachedDataset(Dataset):
    """Dataset serving preprocessed paired sequences and multi-task targets."""
    def __init__(self, cache_data: Dict[str, Any], split: str):
        self.split = split
        self.samples = [s for s in cache_data["samples"] if s["split"] == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def compute_metrics(predictions: Dict[str, np.ndarray],
                    ground_truth: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Computes all standard scientific metrics for Identification, Classification,
    Wind Regression, and Track Prediction.
    """
    # -------------------------------------------------------------
    # Task A: Identification (Current Center Lat/Lon)
    # -------------------------------------------------------------
    pred_centers = predictions["center_deg"]  # [N, 2]
    true_centers = ground_truth["center_deg"]  # [N, 2]
    dpes_center = [
        haversine_km(pc[0], pc[1], tc[0], tc[1])
        for pc, tc in zip(pred_centers, true_centers)
    ]
    lat_mae = float(np.mean(np.abs(pred_centers[:, 0] - true_centers[:, 0])))
    lon_mae = float(np.mean(np.abs(pred_centers[:, 1] - true_centers[:, 1])))

    # -------------------------------------------------------------
    # Task B: Classification (7-class IMD Intensity)
    # -------------------------------------------------------------
    pred_cats = predictions["category_pred"]  # [N]
    true_cats = ground_truth["category_idx"]  # [N]
    acc = float(accuracy_score(true_cats, pred_cats))
    macro_f1 = float(f1_score(true_cats, pred_cats, average="macro", zero_division=0))
    recalls = recall_score(true_cats, pred_cats, average=None, labels=list(range(7)), zero_division=0)
    per_class_recall = {IDX_TO_CATEGORY[i]: float(recalls[i]) for i in range(7)}

    # -------------------------------------------------------------
    # Task C: Wind Speed Regression (kt)
    # -------------------------------------------------------------
    pred_winds = predictions["wind_kt"].squeeze()  # [N]
    true_winds = ground_truth["wind_kt"].squeeze()  # [N]
    wind_diff = pred_winds - true_winds
    abs_wind_diff = np.abs(wind_diff)
    wind_mae = float(np.mean(abs_wind_diff))
    wind_rmse = float(np.sqrt(np.mean(wind_diff ** 2)))
    wind_med_ae = float(np.median(abs_wind_diff))
    wind_p90_ae = float(np.percentile(abs_wind_diff, 90))
    wind_bias = float(np.mean(wind_diff))
    # Pearson r
    if np.std(pred_winds) > 1e-6 and np.std(true_winds) > 1e-6:
        wind_r = float(np.corrcoef(pred_winds, true_winds)[0, 1])
    else:
        wind_r = 0.0

    # -------------------------------------------------------------
    # Task D: Track Prediction (+12h, +24h, +48h)
    # -------------------------------------------------------------
    dpes_12 = []
    dpes_24 = []
    dpes_48 = []

    m12 = ground_truth["mask_12"]
    m24 = ground_truth["mask_24"]
    m48 = ground_truth["mask_48"]

    for i in range(len(true_centers)):
        if m12[i] > 0.5:
            d = haversine_km(predictions["track_12h_deg"][i, 0], predictions["track_12h_deg"][i, 1],
                             ground_truth["t12_deg"][i, 0], ground_truth["t12_deg"][i, 1])
            dpes_12.append(d)
        if m24[i] > 0.5:
            d = haversine_km(predictions["track_24h_deg"][i, 0], predictions["track_24h_deg"][i, 1],
                             ground_truth["t24_deg"][i, 0], ground_truth["t24_deg"][i, 1])
            dpes_24.append(d)
        if m48[i] > 0.5:
            d = haversine_km(predictions["track_48h_deg"][i, 0], predictions["track_48h_deg"][i, 1],
                             ground_truth["t48_deg"][i, 0], ground_truth["t48_deg"][i, 1])
            dpes_48.append(d)

    all_horizon_dpes = dpes_12 + dpes_24 + dpes_48

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
            "track_aggregate_mean_dpe_km": float(np.mean(all_horizon_dpes)) if all_horizon_dpes else 0.0
        }
    }


def evaluate_loader(model: MultisourceFusionModel,
                    loader: DataLoader,
                    device: torch.device) -> Tuple[float, Dict[str, Any]]:
    """Evaluates loss and full metrics over an entire split DataLoader."""
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
            g_seq = batch["gridsat_seq"].to(device) if model.mode in ["gridsat", "fusion"] else None
            i_seq = batch["insat_seq"].to(device) if model.mode in ["insat", "fusion"] else None

            out = model(gridsat_seq=g_seq, insat_seq=i_seq)

            # Compute losses
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

            # Accumulate predictions
            preds_all["center_deg"].append(out["center_deg"].cpu().numpy())
            preds_all["category_pred"].append(torch.argmax(out["class_logits"], dim=-1).cpu().numpy())
            preds_all["wind_kt"].append(out["wind_kt"].cpu().numpy())
            preds_all["track_12h_deg"].append(out["track_12h_deg"].cpu().numpy())
            preds_all["track_24h_deg"].append(out["track_24h_deg"].cpu().numpy())
            preds_all["track_48h_deg"].append(out["track_48h_deg"].cpu().numpy())

            # Accumulate ground truth
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


def train_multisource_experiment(mode: str,
                                 cache_data: Dict[str, Any],
                                 device: torch.device,
                                 epochs: int = 20,
                                 batch_size: int = 16,
                                 lr: float = 1e-3,
                                 patience: int = 5,
                                 checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Trains and validates a single model experiment (gridsat, insat, or fusion).
    """
    set_seed(42)
    print(f"\n=======================================================")
    print(f"STARTING EXPERIMENT: MODEL {mode.upper()} (mode='{mode}')")
    print(f"=======================================================")

    train_ds = MultisourceCachedDataset(cache_data, "TRAIN")
    val_ds = MultisourceCachedDataset(cache_data, "VALIDATION")
    test_ds = MultisourceCachedDataset(cache_data, "TEST")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Splits loaded: TRAIN={len(train_ds)}, VALIDATION={len(val_ds)}, TEST={len(test_ds)}")

    model = MultisourceFusionModel(mode=mode).to(device)
    param_counts = model.get_parameter_counts()
    print(f"Architecture Parameters: Total={param_counts['total_parameters']:,} | "
          f"Trainable={param_counts['trainable_parameters']:,} | Frozen={param_counts['frozen_parameters']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
            g_seq = batch["gridsat_seq"].to(device) if mode in ["gridsat", "fusion"] else None
            i_seq = batch["insat_seq"].to(device) if mode in ["insat", "fusion"] else None

            out = model(gridsat_seq=g_seq, insat_seq=i_seq)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train += 1

        avg_train_loss = train_loss / max(1, n_train)

        # Validation evaluation
        val_loss, val_m = evaluate_loader(model, val_loader, device)
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

    print(f"\nLoading best checkpoint from epoch {best_epoch} (Val Loss: {best_val_loss:.4f}) for single-shot Test evaluation...")
    model.load_state_dict(best_weights)

    # Final Single-Shot Evaluation on Test Set
    test_loss, test_m = evaluate_loader(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Center DPE: {test_m['identification']['center_mean_dpe_km']:.1f}km | "
          f"Class Macro-F1: {test_m['classification']['macro_f1']:.4f} | "
          f"Wind MAE: {test_m['wind_regression']['wind_mae_kt']:.1f}kt | "
          f"Track Agg DPE: {test_m['track_prediction']['track_aggregate_mean_dpe_km']:.1f}km")

    # Save checkpoint
    if checkpoint_path:
        ckpt_p = Path(checkpoint_path)
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_weights,
            "mode": mode,
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_metrics": best_val_metrics,
            "test_loss": test_loss,
            "test_metrics": test_m,
            "parameter_counts": param_counts,
            "history": history
        }, ckpt_p)
        print(f"Checkpoint saved to {ckpt_p}")

    return {
        "mode": mode,
        "best_epoch": best_epoch,
        "parameter_counts": param_counts,
        "val_loss": best_val_loss,
        "val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_m,
        "history": history
    }
