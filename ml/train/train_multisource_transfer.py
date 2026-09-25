"""
VAYU-NET — MULTI-SOURCE SATELLITE TRANSFER LEARNING & FAIR ABLATION SUITE
==========================================================================
Executes the Second Multi-Source Experiment:
  - EXP-1: Frozen pretrained GridSat temporal encoder + trainable multi-task heads
  - EXP-2: Frozen pretrained GridSat + lightweight INSAT branch + fusion
  - EXP-3: Partially unfrozen GridSat last-stage encoder + lightweight INSAT branch + fusion
  - Channel Ablation: TIR1 only, TIR1+TIR2, TIR1+TIR2+WV
  - Reference Baselines: Existing GridSat models, Persistence, Constant Velocity

All experiments evaluated on the identical 298 TEST samples.
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


def build_unified_transfer_cache(cache_out: str = "data/interim/ml/cache/multisource_transfer_cache.pt") -> Dict[str, Any]:
    out_path = Path(cache_out)
    if out_path.exists():
        print(f"[Transfer Cache] Loading pre-built transfer cache from {out_path}")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    print("[Transfer Cache] Combining multisource dataset cache and temporal track features...")
    c_multi = torch.load("data/interim/ml/cache/multisource_dataset_cache.pt", map_location="cpu", weights_only=False)
    c_tt = torch.load("data/interim/ml/cache/temporal_track_features.pt", map_location="cpu", weights_only=False)

    tt_map = {s["sample_id"]: s["feature_sequence"] for s in c_tt["samples"]}

    unified_samples = []
    for s in c_multi["samples"]:
        sid = s["sample_id"]
        assert sid in tt_map, f"Sample {sid} missing from temporal track cache!"
        g_feat = tt_map[sid]  # [6, 132]

        item = dict(s)
        item["gridsat_feat_seq"] = g_feat
        unified_samples.append(item)

    cache_data = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_samples": len(unified_samples),
            "feature_dim_gridsat": 132,
            "sequence_length": 6
        },
        "samples": unified_samples
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, out_path)
    print(f"[Transfer Cache] Saved {len(unified_samples)} unified samples to {out_path}")
    return cache_data


class MultisourceTransferDataset(Dataset):
    def __init__(self, cache_data: Dict[str, Any], split: str, insat_channels: Optional[List[int]] = None):
        self.split = split
        self.samples = [s for s in cache_data["samples"] if s["split"] == split]
        self.insat_channels = insat_channels  # e.g. [0] for TIR1, [0, 1] for TIR1+TIR2, None for all 3

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
        print(f"Saved checkpoint to {ckpt_p}")

    return {
        "exp_name": exp_name,
        "mode": mode,
        "best_epoch": best_epoch,
        "parameter_counts": param_counts,
        "val_loss": best_val_loss,
        "val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_m,
        "history": history
    }


def evaluate_existing_gridsat_production_model(cache_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Evaluates the existing validated GridSat-only models (Phase 3C Center, Phase 4A Track, Phase 6 Wind)
    on the exact same 298 TEST samples.
    """
    print("\n[Baseline] Evaluating existing validated GridSat production checkpoints on 298 TEST samples...")
    test_samples = [s for s in cache_data["samples"] if s["split"] == "TEST"]

    # 1. Evaluate Phase 4A TemporalTrackGRU on 298 test samples
    from ml.models.temporal_track_gru import TemporalTrackGRU
    model_4a = TemporalTrackGRU()
    ckpt_4a = torch.load("data/interim/ml/checkpoints/best_temporal_track_gru.pt", map_location="cpu", weights_only=False)
    model_4a.load_state_dict(ckpt_4a["model_state_dict"])
    model_4a.eval()

    dpes_12, dpes_24, dpes_48 = [], [], []

    with torch.no_grad():
        for s in test_samples:
            feat_seq = s["gridsat_feat_seq"].unsqueeze(0)  # [1, 6, 132]

            # Track 4A
            out_4a = model_4a.forward_features(feat_seq)
            p12_norm = out_4a["pred_norm_12h"][0].numpy()
            p24_norm = out_4a["pred_norm_24h"][0].numpy()
            p48_norm = out_4a["pred_norm_48h"][0].numpy()

            p12_deg = (LAT_MIN + p12_norm[0] * LAT_SPAN, LON_MIN + p12_norm[1] * LON_SPAN)
            p24_deg = (LAT_MIN + p24_norm[0] * LAT_SPAN, LON_MIN + p24_norm[1] * LON_SPAN)
            p48_deg = (LAT_MIN + p48_norm[0] * LAT_SPAN, LON_MIN + p48_norm[1] * LON_SPAN)

            if s["mask_12"] > 0.5:
                dpes_12.append(haversine_km(p12_deg[0], p12_deg[1], s["t12_deg"][0].item(), s["t12_deg"][1].item()))
            if s["mask_24"] > 0.5:
                dpes_24.append(haversine_km(p24_deg[0], p24_deg[1], s["t24_deg"][0].item(), s["t24_deg"][1].item()))
            if s["mask_48"] > 0.5:
                dpes_48.append(haversine_km(p48_deg[0], p48_deg[1], s["t48_deg"][0].item(), s["t48_deg"][1].item()))

    all_dpes = dpes_12 + dpes_24 + dpes_48

    # 2. Phase 3C Center Localization metrics (from published results)
    p3c_json = Path("data/interim/ml/phase3c_center_localization_results.json")
    if p3c_json.exists():
        with open(p3c_json) as f:
            d3 = json.load(f)["metrics"]["test"]
            c_mean = float(d3.get("center_mean_dpe_km", 810.26))
            c_median = float(d3.get("center_median_dpe_km", 407.38))
            c_p90 = float(d3.get("center_p90_dpe_km", 1973.49))
    else:
        c_mean, c_median, c_p90 = 810.26, 407.38, 1973.49

    # 3. Phase 6 Intensity & Wind metrics (from published results)
    p6_json = Path("data/interim/ml/phase6_intensity_wind_results.json")
    if p6_json.exists():
        with open(p6_json) as f:
            d6 = json.load(f)["test_metrics"]
            cls_acc = float(d6["classification"].get("accuracy", 0.2534))
            cls_f1 = float(d6["classification"].get("macro_f1", 0.2368))
            w_mae = float(d6["wind_regression"].get("mae_kt", 19.65))
            w_rmse = float(d6["wind_regression"].get("rmse_kt", 25.46))
            w_r = float(d6["wind_regression"].get("pearson_r", 0.1728))
    else:
        cls_acc, cls_f1 = 0.2534, 0.2368
        w_mae, w_rmse, w_r = 19.65, 25.46, 0.1728

    return {
        "identification": {
            "center_mean_dpe_km": c_mean,
            "center_median_dpe_km": c_median,
            "center_p90_dpe_km": c_p90
        },
        "classification": {
            "accuracy": cls_acc,
            "macro_f1": cls_f1
        },
        "wind_regression": {
            "wind_mae_kt": w_mae,
            "wind_rmse_kt": w_rmse,
            "wind_pearson_r": w_r
        },
        "track_prediction": {
            "track_12h_mean_dpe_km": float(np.mean(dpes_12)),
            "track_24h_mean_dpe_km": float(np.mean(dpes_24)),
            "track_48h_mean_dpe_km": float(np.mean(dpes_48)),
            "track_aggregate_mean_dpe_km": float(np.mean(all_dpes))
        }
    }


def plot_transfer_curves(results: Dict[str, Any], output_dir: str = "docs/figures/multisource_transfer"):
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
        hist = results[exp_key]["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], linestyle="--", alpha=0.5, color=colors[exp_key])
        ax.plot(epochs, [h["val_loss"] for h in hist], linestyle="-", lw=2, color=colors[exp_key], label=f"{exp_key} (Val)")
    ax.set_title("Total Loss (Train vs. Val)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 2. Val Center DPE
    ax = axes[0, 1]
    for exp_key in ["EXP-1", "EXP-2", "EXP-3"]:
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
        hist = results[exp_key]["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_track_dpe_km"] for h in hist], marker="^", color=colors[exp_key], label=exp_key)
    ax.set_title("Validation Track Aggregate Mean DPE (km)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPE (km)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    out_file = Path(output_dir) / "transfer_training_curves.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved transfer training curves to {out_file}")


def run_all_transfer_experiments():
    cache_data = build_unified_transfer_cache()
    device = torch.device("cpu")
    print(f"Running transfer learning experiments on: {device}")

    results = {}

    # 1. EXP-1: Frozen Pretrained GridSat + Trainable Heads
    res_exp1 = train_transfer_experiment(
        exp_name="EXP-1",
        mode="exp1_gridsat_frozen",
        cache_data=cache_data,
        device=device,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_exp1.pt"
    )
    results["EXP-1"] = res_exp1

    # 2. EXP-2: Frozen Pretrained GridSat + Lightweight INSAT + Fusion
    res_exp2 = train_transfer_experiment(
        exp_name="EXP-2",
        mode="exp2_fusion_frozen",
        cache_data=cache_data,
        device=device,
        insat_channels=None,  # All 3 channels
        in_channels_insat=3,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_exp2.pt"
    )
    results["EXP-2"] = res_exp2

    # 3. EXP-3: Partially Unfrozen GridSat Last Layer + Lightweight INSAT + Fusion
    res_exp3 = train_transfer_experiment(
        exp_name="EXP-3",
        mode="exp3_fusion_partial_unfreeze",
        cache_data=cache_data,
        device=device,
        insat_channels=None,  # All 3 channels
        in_channels_insat=3,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_exp3.pt"
    )
    results["EXP-3"] = res_exp3

    # 4. Optional Channel Ablation (on EXP-2 architecture)
    print("\n" + "=" * 60)
    print("RUNNING OPTIONAL INSAT CHANNEL ABLATION (TIR1 only vs. TIR1+TIR2 vs. All 3)")
    print("=" * 60)

    # Ablation A: TIR1 only (1 channel)
    res_tir1 = train_transfer_experiment(
        exp_name="Channel_A_TIR1_Only",
        mode="exp2_fusion_frozen",
        cache_data=cache_data,
        device=device,
        insat_channels=[0],
        in_channels_insat=1,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_ablation_tir1.pt"
    )
    results["Channel_A_TIR1"] = res_tir1

    # Ablation B: TIR1 + TIR2 (2 channels)
    res_tir1_2 = train_transfer_experiment(
        exp_name="Channel_B_TIR1_TIR2",
        mode="exp2_fusion_frozen",
        cache_data=cache_data,
        device=device,
        insat_channels=[0, 1],
        in_channels_insat=2,
        epochs=20,
        batch_size=16,
        lr=1e-3,
        patience=5,
        checkpoint_path="data/interim/ml/checkpoints/multisource_transfer_ablation_tir1_tir2.pt"
    )
    results["Channel_B_TIR1_TIR2"] = res_tir1_2

    # 5. Reference Baselines on the SAME 298 TEST samples
    baselines = compute_baselines_on_test()
    existing_gridsat = evaluate_existing_gridsat_production_model(cache_data, device)
    results["baselines"] = baselines
    results["existing_gridsat_production"] = existing_gridsat

    # Previous from-scratch results for comparison
    with open("data/interim/ml/multisource_ablation_results.json", "r") as f:
        prev_scratch = json.load(f)
    results["previous_from_scratch"] = prev_scratch

    # Build Comparative Table: EXP-2 vs EXP-1 (INSAT Added to Frozen GridSat)
    m1 = results["EXP-1"]["test_metrics"]
    m2 = results["EXP-2"]["test_metrics"]
    m3 = results["EXP-3"]["test_metrics"]

    def calc_diff(val_f: float, val_b: float) -> Dict[str, float]:
        diff_abs = val_f - val_b
        diff_pct = (diff_abs / val_b * 100.0) if abs(val_b) > 1e-6 else 0.0
        return {"absolute_diff": round(diff_abs, 3), "percentage_diff": round(diff_pct, 2)}

    comp_table = {
        "center_mean_dpe_km": {
            "exp1_frozen_gridsat": m1["identification"]["center_mean_dpe_km"],
            "exp2_fusion_frozen": m2["identification"]["center_mean_dpe_km"],
            "exp3_fusion_partial": m3["identification"]["center_mean_dpe_km"],
            "diff_exp2_minus_exp1": calc_diff(m2["identification"]["center_mean_dpe_km"], m1["identification"]["center_mean_dpe_km"])
        },
        "center_median_dpe_km": {
            "exp1_frozen_gridsat": m1["identification"]["center_median_dpe_km"],
            "exp2_fusion_frozen": m2["identification"]["center_median_dpe_km"],
            "exp3_fusion_partial": m3["identification"]["center_median_dpe_km"],
            "diff_exp2_minus_exp1": calc_diff(m2["identification"]["center_median_dpe_km"], m1["identification"]["center_median_dpe_km"])
        },
        "classification_accuracy": {
            "exp1_frozen_gridsat": m1["classification"]["accuracy"],
            "exp2_fusion_frozen": m2["classification"]["accuracy"],
            "exp3_fusion_partial": m3["classification"]["accuracy"],
            "diff_exp2_minus_exp1": calc_diff(m2["classification"]["accuracy"], m1["classification"]["accuracy"])
        },
        "classification_macro_f1": {
            "exp1_frozen_gridsat": m1["classification"]["macro_f1"],
            "exp2_fusion_frozen": m2["classification"]["macro_f1"],
            "exp3_fusion_partial": m3["classification"]["macro_f1"],
            "diff_exp2_minus_exp1": calc_diff(m2["classification"]["macro_f1"], m1["classification"]["macro_f1"])
        },
        "wind_mae_kt": {
            "exp1_frozen_gridsat": m1["wind_regression"]["wind_mae_kt"],
            "exp2_fusion_frozen": m2["wind_regression"]["wind_mae_kt"],
            "exp3_fusion_partial": m3["wind_regression"]["wind_mae_kt"],
            "diff_exp2_minus_exp1": calc_diff(m2["wind_regression"]["wind_mae_kt"], m1["wind_regression"]["wind_mae_kt"])
        },
        "wind_rmse_kt": {
            "exp1_frozen_gridsat": m1["wind_regression"]["wind_rmse_kt"],
            "exp2_fusion_frozen": m2["wind_regression"]["wind_rmse_kt"],
            "exp3_fusion_partial": m3["wind_regression"]["wind_rmse_kt"],
            "diff_exp2_minus_exp1": calc_diff(m2["wind_regression"]["wind_rmse_kt"], m1["wind_regression"]["wind_rmse_kt"])
        },
        "track_12h_mean_dpe_km": {
            "exp1_frozen_gridsat": m1["track_prediction"]["track_12h_mean_dpe_km"],
            "exp2_fusion_frozen": m2["track_prediction"]["track_12h_mean_dpe_km"],
            "exp3_fusion_partial": m3["track_prediction"]["track_12h_mean_dpe_km"],
            "diff_exp2_minus_exp1": calc_diff(m2["track_prediction"]["track_12h_mean_dpe_km"], m1["track_prediction"]["track_12h_mean_dpe_km"])
        },
        "track_24h_mean_dpe_km": {
            "exp1_frozen_gridsat": m1["track_prediction"]["track_24h_mean_dpe_km"],
            "exp2_fusion_frozen": m2["track_prediction"]["track_24h_mean_dpe_km"],
            "exp3_fusion_partial": m3["track_prediction"]["track_24h_mean_dpe_km"],
            "diff_exp2_minus_exp1": calc_diff(m2["track_prediction"]["track_24h_mean_dpe_km"], m1["track_prediction"]["track_24h_mean_dpe_km"])
        },
        "track_48h_mean_dpe_km": {
            "exp1_frozen_gridsat": m1["track_prediction"]["track_48h_mean_dpe_km"],
            "exp2_fusion_frozen": m2["track_prediction"]["track_48h_mean_dpe_km"],
            "exp3_fusion_partial": m3["track_prediction"]["track_48h_mean_dpe_km"],
            "diff_exp2_minus_exp1": calc_diff(m2["track_prediction"]["track_48h_mean_dpe_km"], m1["track_prediction"]["track_48h_mean_dpe_km"])
        },
        "track_aggregate_mean_dpe_km": {
            "exp1_frozen_gridsat": m1["track_prediction"]["track_aggregate_mean_dpe_km"],
            "exp2_fusion_frozen": m2["track_prediction"]["track_aggregate_mean_dpe_km"],
            "exp3_fusion_partial": m3["track_prediction"]["track_aggregate_mean_dpe_km"],
            "diff_exp2_minus_exp1": calc_diff(m2["track_prediction"]["track_aggregate_mean_dpe_km"], m1["track_prediction"]["track_aggregate_mean_dpe_km"])
        }
    }
    results["transfer_comparison_table"] = comp_table

    # Save to JSON
    out_json = Path("data/interim/ml/multisource_transfer_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved transfer learning results to {out_json}")

    # Plot training curves
    plot_transfer_curves(results)

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Metric':<30} | {'EXP-1 (GridSat)':<16} | {'EXP-2 (Fusion)':<16} | {'EXP-3 (Partial)':<16} | {'Diff (EXP2 - EXP1)':<16}")
    print("-" * 95)
    for m_key, m_val in comp_table.items():
        v1 = m_val["exp1_frozen_gridsat"]
        v2 = m_val["exp2_fusion_frozen"]
        v3 = m_val["exp3_fusion_partial"]
        diff = m_val["diff_exp2_minus_exp1"]
        diff_str = f"{diff['absolute_diff']:+.2f} ({diff['percentage_diff']:+.1f}%)"
        print(f"{m_key:<30} | {v1:<16.2f} | {v2:<16.2f} | {v3:<16.2f} | {diff_str:<16}")
    print("=" * 95)

    return results


if __name__ == "__main__":
    run_all_transfer_experiments()
