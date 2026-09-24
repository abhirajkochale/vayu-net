"""
VAYU-NET PHASE 3C — DEDICATED CYCLONE CENTER LOCALIZATION TRAINING
==================================================================
Model: DedicatedCenterLocalizationResNet (1-channel ResNet-18, weights=None)
Objective: Isolated Center Localization (10 * MSE_Heatmap + 1 * SmoothL1_Coord)
Checkpoint Selection: Minimum VALIDATION Mean Center DPE (km)
Early Stopping: Patience = 3 on VALIDATION Mean Center DPE

Experimental Discipline:
  - TRAIN (696 samples / 81 storms) for model optimization
  - VALIDATION (252 samples / 14 storms) for model selection & early stopping
  - TEST (371 samples / 31 storms) evaluated strictly ONCE using best checkpoint
  - Checkpoint: data/interim/ml/checkpoints/best_center_localization_cnn.pt
  - Results JSON: data/interim/ml/phase3c_center_localization_results.json
"""

import os
import sys
import json
import math
import random
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.data.vayu_dataset import SingleFrameVayuDataset
from ml.models.center_localization_cnn import (
    DedicatedCenterLocalizationResNet,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    HEATMAP_H, HEATMAP_W
)

# ---------------------------------------------------------
# Configuration & Hyperparameters
# ---------------------------------------------------------
CONFIG = {
    "experiment_name": "Phase3C_DedicatedCenterLocalization",
    "random_seed": 42,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 10,
    "patience": 3,
    "loss_weights": {
        "heatmap": 10.0,
        "coord": 1.0
    },
    "gaussian_sigma": 1.5,
    "sample_index_path": "data/manifests/vayu_net_sample_index.csv",
    "norm_stats_path": "data/interim/ml/train_normalization_stats.json",
    "checkpoint_dir": "data/interim/ml/checkpoints",
    "checkpoint_path": "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
    "results_json_path": "data/interim/ml/phase3c_center_localization_results.json",
    "figures_dir": "docs/figures/center_localization",
    "num_workers": 0
}

EARTH_RADIUS_KM = 6371.0


def set_seed(seed):
    """Sets random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between two geographic coordinates in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def evaluate_split(model, dataloader, device, weights, sigma=1.5):
    """Evaluates dedicated center localization performance across a dataset split."""
    model.eval()
    
    total_loss_sum = 0.0
    hm_loss_sum = 0.0
    coord_loss_sum = 0.0
    num_samples = 0
    
    pred_lats, pred_lons = [], []
    peak_lats, peak_lons = [], []
    true_lats, true_lons = [], []
    dpes = []
    sample_ids = []
    storm_ids = []
    timestamps = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["satellite_image"].to(device)
            target_norm_center = batch["norm_center"].to(device)
            
            b_size = images.size(0)
            num_samples += b_size
            
            outputs = model(images)
            heatmap_prob = outputs["heatmap_prob"]
            norm_center = outputs["norm_center"]
            peak_center = outputs["peak_center"]
            
            # Target heatmap
            target_hm = DedicatedCenterLocalizationResNet.generate_gaussian_target(
                target_norm_center, height=HEATMAP_H, width=HEATMAP_W, sigma=sigma, device=device
            )
            
            # Losses
            loss_hm = nn.functional.mse_loss(heatmap_prob, target_hm)
            loss_coord = nn.functional.smooth_l1_loss(norm_center, target_norm_center)
            loss_tot = weights["heatmap"] * loss_hm + weights["coord"] * loss_coord
            
            total_loss_sum += loss_tot.item() * b_size
            hm_loss_sum += loss_hm.item() * b_size
            coord_loss_sum += loss_coord.item() * b_size
            
            # Geographic coordinate denormalization
            pred_deg = DedicatedCenterLocalizationResNet.denormalize_center(norm_center.cpu())
            peak_deg = DedicatedCenterLocalizationResNet.denormalize_center(peak_center.cpu())
            
            p_lat = pred_deg[:, 0].numpy()
            p_lon = pred_deg[:, 1].numpy()
            pk_lat = peak_deg[:, 0].numpy()
            pk_lon = peak_deg[:, 1].numpy()
            
            t_lat = batch["center_deg"][:, 0].numpy()
            t_lon = batch["center_deg"][:, 1].numpy()
            
            s_ids = batch["sample_id"]
            st_ids = batch["storm_id"]
            t0s = batch["t0"]
            
            for pl, po, pkl, pko, tl, to, sid, stid, t0_val in zip(p_lat, p_lon, pk_lat, pk_lon, t_lat, t_lon, s_ids, st_ids, t0s):
                pred_lats.append(float(pl))
                pred_lons.append(float(po))
                peak_lats.append(float(pkl))
                peak_lons.append(float(pko))
                true_lats.append(float(tl))
                true_lons.append(float(to))
                dpes.append(haversine_km(tl, to, pl, po))
                sample_ids.append(sid)
                storm_ids.append(stid)
                timestamps.append(t0_val)
                
    avg_tot_loss = total_loss_sum / num_samples
    avg_hm_loss = hm_loss_sum / num_samples
    avg_coord_loss = coord_loss_sum / num_samples
    
    # Geographic metrics
    lat_mae = float(np.mean(np.abs(np.array(pred_lats) - np.array(true_lats))))
    lon_mae = float(np.mean(np.abs(np.array(pred_lons) - np.array(true_lons))))
    mean_dpe = float(np.mean(dpes))
    median_dpe = float(np.median(dpes))
    p90_dpe = float(np.percentile(dpes, 90))
    
    pred_lat_std = float(np.std(pred_lats))
    pred_lon_std = float(np.std(pred_lons))
    true_lat_std = float(np.std(true_lats))
    true_lon_std = float(np.std(true_lons))
    var_ratio_lat = pred_lat_std / true_lat_std if true_lat_std > 0 else 0.0
    var_ratio_lon = pred_lon_std / true_lon_std if true_lon_std > 0 else 0.0
    
    metrics = {
        "loss_total": avg_tot_loss,
        "loss_heatmap": avg_hm_loss,
        "loss_coord": avg_coord_loss,
        "center_lat_mae_deg": lat_mae,
        "center_lon_mae_deg": lon_mae,
        "center_mean_dpe_km": mean_dpe,
        "center_median_dpe_km": median_dpe,
        "center_p90_dpe_km": p90_dpe,
        "pred_lat_std_deg": pred_lat_std,
        "pred_lon_std_deg": pred_lon_std,
        "true_lat_std_deg": true_lat_std,
        "true_lon_std_deg": true_lon_std,
        "var_ratio_lat": var_ratio_lat,
        "var_ratio_lon": var_ratio_lon,
        "raw_predictions": {
            "sample_ids": sample_ids,
            "storm_ids": storm_ids,
            "timestamps": timestamps,
            "pred_lats": pred_lats,
            "pred_lons": pred_lons,
            "peak_lats": peak_lats,
            "peak_lons": peak_lons,
            "true_lats": true_lats,
            "true_lons": true_lons,
            "dpes_km": dpes
        }
    }
    return metrics


def plot_diagnostics(history, best_epoch, val_final, test_final, save_dir, model=None, test_loader=None, device=None):
    """Generates all 7 requested visual diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)
    test_metrics = test_final
    epochs = [h["epoch"] for h in history]
    
    # 1. train_val_center_loss.png
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h["train_loss"] for h in history], 'o-', label='Train Total Center Loss', color='#1f77b4', lw=2)
    plt.plot(epochs, [h["val_loss"] for h in history], 's--', label='Val Total Center Loss', color='#ff7f0e', lw=2)
    plt.plot(epochs, [h["val_loss_hm"] * 10.0 for h in history], 'd:', label='Val Heatmap MSE (x10)', color='#2ca02c', lw=1.5)
    plt.plot(epochs, [h["val_loss_coord"] for h in history], '^:', label='Val Coord L1', color='#d62728', lw=1.5)
    plt.axvline(best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Best Checkpoint (Epoch {best_epoch})')
    plt.title("Phase 3C — Training and Validation Center Loss", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_val_center_loss.png"), dpi=250)
    plt.close()
    print("Saved: train_val_center_loss.png")
    
    # 2. val_dpe_curve.png
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h["val_mean_dpe_km"] for h in history], 'o-', label='Val Mean DPE (km)', color='#d62728', lw=2.5)
    plt.plot(epochs, [h["val_median_dpe_km"] for h in history], 's--', label='Val Median DPE (km)', color='#2ca02c', lw=2)
    plt.plot(epochs, [h["val_p90_dpe_km"] for h in history], '^:', label='Val 90th %ile DPE (km)', color='#ff7f0e', lw=1.5)
    plt.axhline(1186.0, color='gray', linestyle='--', label='Constant Centroid Baseline (1,186 km)')
    plt.axvline(best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Selected Epoch {best_epoch}')
    plt.title("Phase 3C — Validation Direct Position Error (DPE) Progression", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("DPE (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_dpe_curve.png"), dpi=250)
    plt.close()
    print("Saved: val_dpe_curve.png")
    
    # 3. predicted_vs_actual_centers.png (VAL & TEST)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, split_name, m_data, p_color in [(axes[0], "VALIDATION (N=252)", val_final, '#2ca02c'), (axes[1], "TEST (N=371)", test_final, '#d62728')]:
        tlats = m_data["raw_predictions"]["true_lats"]
        tlons = m_data["raw_predictions"]["true_lons"]
        plats = m_data["raw_predictions"]["pred_lats"]
        plons = m_data["raw_predictions"]["pred_lons"]
        
        ax.scatter(tlons, tlats, color='#1f77b4', alpha=0.6, s=40, label='Ground Truth (IMD)')
        ax.scatter(plons, plats, color=p_color, alpha=0.6, marker='^', s=40, label='Phase 3C Pred Center')
        for tl, to, pl, po in zip(tlats[::6], tlons[::6], plats[::6], plons[::6]):
            ax.plot([to, po], [tl, pl], color='gray', linestyle=':', alpha=0.4, lw=0.8)
        ax.set_xlim(40.0, 105.0); ax.set_ylim(-5.0, 35.0)
        ax.set_xlabel("Longitude (°E)", fontsize=11); ax.set_ylabel("Latitude (°N)", fontsize=11)
        ax.set_title(f"{split_name}\nMean DPE: {m_data['center_mean_dpe_km']:.1f} km, Median: {m_data['center_median_dpe_km']:.1f} km", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_vs_actual_centers.png"), dpi=250)
    plt.close()
    print("Saved: predicted_vs_actual_centers.png")
    
    # 4. center_error_distribution.png (TEST Set)
    test_dpes = test_metrics["raw_predictions"]["dpes_km"]
    plt.figure(figsize=(9, 6))
    plt.hist(test_dpes, bins=35, color='#31688e', edgecolor='black', alpha=0.75)
    plt.axvline(test_metrics["center_mean_dpe_km"], color='#d62728', linestyle='--', lw=2.5, label=f'Mean: {test_metrics["center_mean_dpe_km"]:.1f} km')
    plt.axvline(test_metrics["center_median_dpe_km"], color='#2ca02c', linestyle='-', lw=2.5, label=f'Median: {test_metrics["center_median_dpe_km"]:.1f} km')
    plt.axvline(test_metrics["center_p90_dpe_km"], color='#ff7f0e', linestyle=':', lw=2.0, label=f'90th %ile: {test_metrics["center_p90_dpe_km"]:.1f} km')
    plt.title("Direct Position Error Distribution — Final TEST Split (N=371)", fontsize=13, fontweight='bold')
    plt.xlabel("Direct Position Error (km)", fontsize=11); plt.ylabel("Number of Samples", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "center_error_distribution.png"), dpi=250)
    plt.close()
    print("Saved: center_error_distribution.png")
    
    # 5. prediction_variance_analysis.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(test_metrics["raw_predictions"]["true_lats"], bins=25, alpha=0.5, color='#1f77b4', label=f'Actual Lat (std={test_metrics["true_lat_std_deg"]:.2f}°)', density=True)
    axes[0].hist(test_metrics["raw_predictions"]["pred_lats"], bins=25, alpha=0.5, color='#d62728', label=f'Pred Lat (std={test_metrics["pred_lat_std_deg"]:.2f}°)', density=True)
    axes[0].set_title(f"Latitude Distribution (Var Ratio: {test_metrics['var_ratio_lat']:.2f})", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Latitude (°N)"); axes[0].grid(True, linestyle=":", alpha=0.5); axes[0].legend()
    
    axes[1].hist(test_metrics["raw_predictions"]["true_lons"], bins=25, alpha=0.5, color='#1f77b4', label=f'Actual Lon (std={test_metrics["true_lon_std_deg"]:.2f}°)', density=True)
    axes[1].hist(test_metrics["raw_predictions"]["pred_lons"], bins=25, alpha=0.5, color='#d62728', label=f'Pred Lon (std={test_metrics["pred_lon_std_deg"]:.2f}°)', density=True)
    axes[1].set_title(f"Longitude Distribution (Var Ratio: {test_metrics['var_ratio_lon']:.2f})", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Longitude (°E)"); axes[1].grid(True, linestyle=":", alpha=0.5); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_variance_analysis.png"), dpi=250)
    plt.close()
    print("Saved: prediction_variance_analysis.png")
    
    # 6. heatmap_examples.png
    if model is not None and test_loader is not None and device is not None:
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(test_loader))
            imgs = sample_batch["satellite_image"].to(device)
            target_norm_c = sample_batch["norm_center"].to(device)
            outs = model(imgs)
            
            pred_hm = outs["heatmap_prob"].cpu().numpy()
            target_hm = DedicatedCenterLocalizationResNet.generate_gaussian_target(
                target_norm_c, height=HEATMAP_H, width=HEATMAP_W, sigma=1.5, device=device
            ).cpu().numpy()
            
            raw_imgs = imgs.cpu().numpy()
            pred_centers = DedicatedCenterLocalizationResNet.denormalize_center(outs["norm_center"].cpu()).numpy()
            true_centers = sample_batch["center_deg"].numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            extent = [LON_MIN, LON_MIN + LON_SPAN, LAT_MIN, LAT_MIN + LAT_SPAN]
            
            for b_idx in range(min(2, imgs.size(0))):
                sid = sample_batch["storm_id"][b_idx]
                t0_str = sample_batch["t0"][b_idx]
                dpe_val = haversine_km(true_centers[b_idx, 0], true_centers[b_idx, 1], pred_centers[b_idx, 0], pred_centers[b_idx, 1])
                
                axes[b_idx, 0].imshow(raw_imgs[b_idx, 0], cmap='Greys', origin='lower', extent=extent)
                axes[b_idx, 0].plot(true_centers[b_idx, 1], true_centers[b_idx, 0], 'o', color='yellow', markersize=9, label='Ground Truth')
                axes[b_idx, 0].plot(pred_centers[b_idx, 1], pred_centers[b_idx, 0], '^', color='red', markersize=9, label='Phase 3C Pred')
                axes[b_idx, 0].set_title(f"{sid} ({t0_str})\nDPE: {dpe_val:.1f} km", fontsize=10, fontweight='bold')
                axes[b_idx, 0].set_xlabel("Lon (°E)"); axes[b_idx, 0].set_ylabel("Lat (°N)")
                axes[b_idx, 0].legend(loc='lower left', fontsize=8); axes[b_idx, 0].grid(True, linestyle=":", alpha=0.5)
                
                axes[b_idx, 1].imshow(target_hm[b_idx, 0], cmap='turbo', origin='lower', extent=extent)
                axes[b_idx, 1].set_title("Ground-Truth Target Heatmap (Gaussian)", fontsize=10, fontweight='bold')
                axes[b_idx, 1].set_xlabel("Lon (°E)"); axes[b_idx, 1].set_ylabel("Lat (°N)"); axes[b_idx, 1].grid(True, linestyle=":", alpha=0.5)
                
                axes[b_idx, 2].imshow(pred_hm[b_idx, 0], cmap='turbo', origin='lower', extent=extent)
                axes[b_idx, 2].plot(pred_centers[b_idx, 1], pred_centers[b_idx, 0], 'x', color='white', markersize=11, markeredgewidth=2, label='Soft-Argmax')
                axes[b_idx, 2].set_title("Predicted Probability Heatmap", fontsize=10, fontweight='bold')
                axes[b_idx, 2].set_xlabel("Lon (°E)"); axes[b_idx, 2].set_ylabel("Lat (°N)"); axes[b_idx, 2].grid(True, linestyle=":", alpha=0.5)
                axes[b_idx, 2].legend(loc='lower left', fontsize=8)
                
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "heatmap_examples.png"), dpi=250)
            plt.close()
            print("Saved: heatmap_examples.png")
            
    # 7. worst_case_examples.png (Top Worst Cases in VALIDATION)
    val_preds = val_final["raw_predictions"]
    worst_indices = np.argsort(val_preds["dpes_km"])[::-1][:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    extent = [LON_MIN, LON_MIN + LON_SPAN, LAT_MIN, LAT_MIN + LAT_SPAN]
    for plot_idx, w_idx in enumerate(worst_indices):
        r_ax = axes[plot_idx // 3, plot_idx % 3]
        t_lat = val_preds["true_lats"][w_idx]
        t_lon = val_preds["true_lons"][w_idx]
        p_lat = val_preds["pred_lats"][w_idx]
        p_lon = val_preds["pred_lons"][w_idx]
        dpe = val_preds["dpes_km"][w_idx]
        stid = val_preds["storm_ids"][w_idx]
        t0 = val_preds["timestamps"][w_idx]
        
        r_ax.plot(t_lon, t_lat, 'o', color='blue', markersize=10, label=f'True: ({t_lat:.1f}°, {t_lon:.1f}°)')
        r_ax.plot(p_lon, p_lat, '^', color='red', markersize=10, label=f'Pred: ({p_lat:.1f}°, {p_lon:.1f}°)')
        r_ax.plot([t_lon, p_lon], [t_lat, p_lat], 'k:', lw=1.5)
        r_ax.set_xlim(40.0, 105.0); r_ax.set_ylim(-5.0, 35.0)
        r_ax.set_title(f"Worst Case #{plot_idx+1}: {stid}\nTime: {t0} | DPE: {dpe:.1f} km", fontsize=10, fontweight='bold')
        r_ax.set_xlabel("Lon (°E)"); r_ax.set_ylabel("Lat (°N)")
        r_ax.grid(True, linestyle=":", alpha=0.5); r_ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "worst_case_examples.png"), dpi=250)
    plt.close()
    print("Saved: worst_case_examples.png")


def main():
    print("=" * 80)
    print("VAYU-NET PHASE 3C — DEDICATED CYCLONE CENTER LOCALIZATION TRAINING")
    print("=" * 80)
    
    set_seed(CONFIG["random_seed"])
    device = torch.device("cpu")
    print(f"Device:                 {device}")
    print(f"Random Seed:            {CONFIG['random_seed']}")
    print(f"Batch Size:             {CONFIG['batch_size']}")
    print(f"Max Epochs:             {CONFIG['max_epochs']}")
    print(f"Early Stopping Metric:  VALIDATION MEAN CENTER DPE (Patience = {CONFIG['patience']})")
    
    # Datasets and Loaders
    train_dataset = SingleFrameVayuDataset(
        sample_index_csv=CONFIG["sample_index_path"],
        split="TRAIN",
        norm_stats_path=CONFIG["norm_stats_path"],
        normalize=True
    )
    val_dataset = SingleFrameVayuDataset(
        sample_index_csv=CONFIG["sample_index_path"],
        split="VALIDATION",
        norm_stats_path=CONFIG["norm_stats_path"],
        normalize=True
    )
    test_dataset = SingleFrameVayuDataset(
        sample_index_csv=CONFIG["sample_index_path"],
        split="TEST",
        norm_stats_path=CONFIG["norm_stats_path"],
        normalize=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    
    print(f"\nDataset Splits:")
    print(f"  TRAIN:      {len(train_dataset)} samples / 81 storms")
    print(f"  VALIDATION: {len(val_dataset)} samples / 14 storms")
    print(f"  TEST:       {len(test_dataset)} samples / 31 storms (Strictly held out until final checkpoint)")
    
    # Model Instantiation
    model = DedicatedCenterLocalizationResNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: DedicatedCenterLocalizationResNet")
    print(f"Total Parameters: {total_params:,}")
    
    # Optimizer & Scheduler (Governed by Validation Mean DPE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)
    
    history = []
    best_val_mean_dpe = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    
    eval_only = "--eval_only" in sys.argv
    if eval_only:
        print("\n[EVALUATION ONLY MODE] Skipping training loop, loading best checkpoint...")
        checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
        best_epoch = checkpoint["epoch"]
        best_val_mean_dpe = checkpoint["val_mean_dpe_km"]
        history = [
            {"epoch": 1, "train_loss": 1.5056, "train_loss_hm": 0.1491, "train_loss_coord": 0.0146, "val_loss": 1.2063, "val_loss_hm": 0.1194, "val_loss_coord": 0.0120, "val_mean_dpe_km": 1259.0, "val_median_dpe_km": 1304.8, "val_p90_dpe_km": 1956.4, "lr": 0.0001},
            {"epoch": 2, "train_loss": 0.9668, "train_loss_hm": 0.0956, "train_loss_coord": 0.0109, "val_loss": 0.9972, "val_loss_hm": 0.0990, "val_loss_coord": 0.0074, "val_mean_dpe_km": 703.0, "val_median_dpe_km": 262.5, "val_p90_dpe_km": 1922.5, "lr": 0.0001},
            {"epoch": 3, "train_loss": 0.8481, "train_loss_hm": 0.0838, "train_loss_coord": 0.0099, "val_loss": 0.7963, "val_loss_hm": 0.0786, "val_loss_coord": 0.0103, "val_mean_dpe_km": 1086.1, "val_median_dpe_km": 1122.5, "val_p90_dpe_km": 1960.6, "lr": 0.0001},
            {"epoch": 4, "train_loss": 0.7467, "train_loss_hm": 0.0735, "train_loss_coord": 0.0113, "val_loss": 0.7876, "val_loss_hm": 0.0780, "val_loss_coord": 0.0077, "val_mean_dpe_km": 736.4, "val_median_dpe_km": 286.1, "val_p90_dpe_km": 1938.6, "lr": 0.0001},
            {"epoch": 5, "train_loss": 0.6868, "train_loss_hm": 0.0675, "train_loss_coord": 0.0113, "val_loss": 0.6899, "val_loss_hm": 0.0682, "val_loss_coord": 0.0084, "val_mean_dpe_km": 819.6, "val_median_dpe_km": 471.4, "val_p90_dpe_km": 1971.0, "lr": 0.0001}
        ]
    else:
        print("\n" + "=" * 80)
        print("BEGINNING TRAINING (Criterion: Minimum Validation Mean DPE)")
        print("=" * 80)
        
        for epoch in range(1, CONFIG["max_epochs"] + 1):
            model.train()
            train_loss_total = 0.0
            train_loss_hm = 0.0
            train_loss_coord = 0.0
            train_samples = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                images = batch["satellite_image"].to(device)
                target_norm_center = batch["norm_center"].to(device)
                b_size = images.size(0)
                train_samples += b_size
                
                outputs = model(images)
                heatmap_prob = outputs["heatmap_prob"]
                norm_center = outputs["norm_center"]
                
                target_hm = DedicatedCenterLocalizationResNet.generate_gaussian_target(
                    target_norm_center, height=HEATMAP_H, width=HEATMAP_W, sigma=CONFIG["gaussian_sigma"], device=device
                )
                
                loss_hm = nn.functional.mse_loss(heatmap_prob, target_hm)
                loss_coord = nn.functional.smooth_l1_loss(norm_center, target_norm_center)
                
                loss = CONFIG["loss_weights"]["heatmap"] * loss_hm + CONFIG["loss_weights"]["coord"] * loss_coord
                
                loss.backward()
                optimizer.step()
                
                train_loss_total += loss.item() * b_size
                train_loss_hm += loss_hm.item() * b_size
                train_loss_coord += loss_coord.item() * b_size
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"  Epoch {epoch:2d}/{CONFIG['max_epochs']:2d} | Batch {batch_idx+1:2d}/{len(train_loader):2d} | "
                          f"Center Loss: {train_loss_total / train_samples:.4f} "
                          f"(HM MSE: {train_loss_hm / train_samples:.4f}, Coord L1: {train_loss_coord / train_samples:.4f})")
                          
            avg_train_loss = train_loss_total / train_samples
            avg_train_hm = train_loss_hm / train_samples
            avg_train_coord = train_loss_coord / train_samples
            
            # Evaluate on VALIDATION ONLY
            val_metrics = evaluate_split(model, val_loader, device, CONFIG["loss_weights"], sigma=CONFIG["gaussian_sigma"])
            val_mean_dpe = val_metrics["center_mean_dpe_km"]
            val_med_dpe = val_metrics["center_median_dpe_km"]
            val_p90_dpe = val_metrics["center_p90_dpe_km"]
            
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_mean_dpe)
            
            print("-" * 80)
            print(f"Epoch {epoch:2d} Summary:")
            print(f"  TRAIN Center Loss: {avg_train_loss:.4f} (HM: {avg_train_hm:.4f}, Coord: {avg_train_coord:.4f})")
            print(f"  VAL   Center Loss: {val_metrics['loss_total']:.4f} (HM: {val_metrics['loss_heatmap']:.4f}, Coord: {val_metrics['loss_coord']:.4f})")
            print(f"  VAL DPE Metrics:   Mean: {val_mean_dpe:.1f} km | Median: {val_med_dpe:.1f} km | P90: {val_p90_dpe:.1f} km | LR: {current_lr:.6f}")
            print("-" * 80)
            
            record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_loss_hm": avg_train_hm,
                "train_loss_coord": avg_train_coord,
                "val_loss": val_metrics["loss_total"],
                "val_loss_hm": val_metrics["loss_heatmap"],
                "val_loss_coord": val_metrics["loss_coord"],
                "val_mean_dpe_km": val_mean_dpe,
                "val_median_dpe_km": val_med_dpe,
                "val_p90_dpe_km": val_p90_dpe,
                "val_lat_mae_deg": val_metrics["center_lat_mae_deg"],
                "val_lon_mae_deg": val_metrics["center_lon_mae_deg"],
                "lr": current_lr
            }
            history.append(record)
            
            # Primary Model Selection Criterion: VALIDATION MEAN DPE
            if val_mean_dpe < best_val_mean_dpe:
                prev_best = best_val_mean_dpe
                best_val_mean_dpe = val_mean_dpe
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mean_dpe_km": val_mean_dpe,
                    "val_median_dpe_km": val_med_dpe,
                    "config": CONFIG
                }, CONFIG["checkpoint_path"])
                print(f"  >>> NEW BEST CHECKPOINT saved! Val Mean DPE improved from {prev_best:.1f} km to {val_mean_dpe:.1f} km at epoch {epoch}")
            else:
                epochs_no_improve += 1
                print(f"  >>> Val Mean DPE did not improve ({val_mean_dpe:.1f} km >= {best_val_mean_dpe:.1f} km). Early stop counter: {epochs_no_improve}/{CONFIG['patience']}")
                if epochs_no_improve >= CONFIG["patience"]:
                    print(f"\n[EARLY STOPPING TRIGGERED] Validation Mean DPE did not improve for {CONFIG['patience']} consecutive epochs.")
                    break
                
    # -------------------------------------------------------------
    # FREEZE CHECKPOINT & EVALUATE TEST STRICTLY ONCE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. LOADING BEST CHECKPOINT (Epoch {best_epoch}, Val Mean DPE: {best_val_mean_dpe:.1f} km)")
    print("=" * 80)
    
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Evaluate final splits
    train_final = evaluate_split(model, train_loader, device, CONFIG["loss_weights"], sigma=CONFIG["gaussian_sigma"])
    val_final = evaluate_split(model, val_loader, device, CONFIG["loss_weights"], sigma=CONFIG["gaussian_sigma"])
    
    print("\nEvaluating TEST split strictly ONCE using frozen best checkpoint...")
    test_final = evaluate_split(model, test_loader, device, CONFIG["loss_weights"], sigma=CONFIG["gaussian_sigma"])
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION METRICS SUMMARY (PHASE 3C)")
    print("=" * 80)
    print(f"{'Split':<12} | {'Center DPE (Mean / Med / P90)':<34} | {'Lat / Lon MAE':<20} | {'Variance Ratio (Lat / Lon)'}")
    print("-" * 90)
    for name, m in [("TRAIN", train_final), ("VALIDATION", val_final), ("TEST", test_final)]:
        print(f"{name:<12} | {m['center_mean_dpe_km']:6.1f} / {m['center_median_dpe_km']:6.1f} / {m['center_p90_dpe_km']:6.1f} km | "
              f"{m['center_lat_mae_deg']:4.2f}° / {m['center_lon_mae_deg']:4.2f}°    | "
              f"{m['var_ratio_lat']:4.2f} / {m['var_ratio_lon']:4.2f}")
    print("-" * 90)
    
    # Identify Top 20 Worst-Case Validation Samples
    val_raw = val_final["raw_predictions"]
    worst_idx = np.argsort(val_raw["dpes_km"])[::-1][:20]
    worst_cases = []
    print("\nTop 20 Worst-Case Validation Samples:")
    print(f"{'Rank':<5} | {'Storm ID':<22} | {'Timestamp':<16} | {'IMD Center':<16} | {'Pred Center':<16} | {'DPE (km)':<10}")
    print("-" * 95)
    for rank, idx in enumerate(worst_idx, 1):
        item = {
            "rank": rank,
            "sample_id": val_raw["sample_ids"][idx],
            "storm_id": val_raw["storm_ids"][idx],
            "timestamp": val_raw["timestamps"][idx],
            "true_center": [val_raw["true_lats"][idx], val_raw["true_lons"][idx]],
            "pred_center": [val_raw["pred_lats"][idx], val_raw["pred_lons"][idx]],
            "dpe_km": float(val_raw["dpes_km"][idx])
        }
        worst_cases.append(item)
        print(f"{rank:<5} | {item['storm_id']:<22} | {item['timestamp']:<16} | "
              f"({item['true_center'][0]:5.2f}°, {item['true_center'][1]:5.2f}°) | "
              f"({item['pred_center'][0]:5.2f}°, {item['pred_center'][1]:5.2f}°) | "
              f"{item['dpe_km']:6.1f} km")
              
    # Generate Visual Diagnostics
    plot_diagnostics(history, best_epoch, val_final, test_final, CONFIG["figures_dir"], model=model, test_loader=test_loader, device=device)
    
    # Save Results JSON
    results = {
        "metadata": {
            "experiment_name": CONFIG["experiment_name"],
            "model_architecture": "DedicatedCenterLocalizationResNet (1-channel ResNet-18, weights=None)",
            "parameter_count": total_params,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "device": str(device),
            "random_seed": CONFIG["random_seed"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "weight_decay": CONFIG["weight_decay"],
            "epochs_trained": len(history),
            "selected_epoch": best_epoch,
            "patience": CONFIG["patience"],
            "model_selection_criterion": "Minimum VALIDATION Mean Center DPE (km)",
            "loss_weights": CONFIG["loss_weights"],
            "heatmap_resolution": [HEATMAP_H, HEATMAP_W],
            "gaussian_sigma_cells": CONFIG["gaussian_sigma"],
            "loss_formulation": {
                "heatmap": "MSELoss(pred_heatmap_prob, target_gaussian_heatmap)",
                "coord": "SmoothL1Loss(soft_argmax_center, target_norm_center)",
                "total": "10.0 * heatmap + 1.0 * coord"
            }
        },
        "training_history": history,
        "metrics": {
            "train": {
                "center_lat_mae_deg": train_final["center_lat_mae_deg"],
                "center_lon_mae_deg": train_final["center_lon_mae_deg"],
                "center_mean_dpe_km": train_final["center_mean_dpe_km"],
                "center_median_dpe_km": train_final["center_median_dpe_km"],
                "center_p90_dpe_km": train_final["center_p90_dpe_km"],
                "pred_lat_std_deg": train_final["pred_lat_std_deg"],
                "pred_lon_std_deg": train_final["pred_lon_std_deg"],
                "true_lat_std_deg": train_final["true_lat_std_deg"],
                "true_lon_std_deg": train_final["true_lon_std_deg"],
                "var_ratio_lat": train_final["var_ratio_lat"],
                "var_ratio_lon": train_final["var_ratio_lon"]
            },
            "validation": {
                "center_lat_mae_deg": val_final["center_lat_mae_deg"],
                "center_lon_mae_deg": val_final["center_lon_mae_deg"],
                "center_mean_dpe_km": val_final["center_mean_dpe_km"],
                "center_median_dpe_km": val_final["center_median_dpe_km"],
                "center_p90_dpe_km": val_final["center_p90_dpe_km"],
                "pred_lat_std_deg": val_final["pred_lat_std_deg"],
                "pred_lon_std_deg": val_final["pred_lon_std_deg"],
                "true_lat_std_deg": val_final["true_lat_std_deg"],
                "true_lon_std_deg": val_final["true_lon_std_deg"],
                "var_ratio_lat": val_final["var_ratio_lat"],
                "var_ratio_lon": val_final["var_ratio_lon"]
            },
            "test": {
                "center_lat_mae_deg": test_final["center_lat_mae_deg"],
                "center_lon_mae_deg": test_final["center_lon_mae_deg"],
                "center_mean_dpe_km": test_final["center_mean_dpe_km"],
                "center_median_dpe_km": test_final["center_median_dpe_km"],
                "center_p90_dpe_km": test_final["center_p90_dpe_km"],
                "pred_lat_std_deg": test_final["pred_lat_std_deg"],
                "pred_lon_std_deg": test_final["pred_lon_std_deg"],
                "true_lat_std_deg": test_final["true_lat_std_deg"],
                "true_lon_std_deg": test_final["true_lon_std_deg"],
                "var_ratio_lat": test_final["var_ratio_lat"],
                "var_ratio_lon": test_final["var_ratio_lon"]
            }
        },
        "top_20_worst_validation_cases": worst_cases,
        "baseline_comparison": {
            "constant_train_centroid": {
                "mean_dpe_km": 1186.0,
                "median_dpe_km": 1082.6
            },
            "phase3_single_frame_cnn": {
                "mean_dpe_km": 1215.1,
                "median_dpe_km": 1204.9
            },
            "phase3b_spatial_aware_cnn": {
                "mean_dpe_km": 1268.6,
                "median_dpe_km": 1364.0
            },
            "phase3c_dedicated_center_cnn": {
                "mean_dpe_km": test_final["center_mean_dpe_km"],
                "median_dpe_km": test_final["center_median_dpe_km"]
            },
            "dpe_reduction_vs_centroid_km": 1186.0 - test_final["center_mean_dpe_km"],
            "dpe_reduction_vs_phase3_km": 1215.1 - test_final["center_mean_dpe_km"],
            "dpe_reduction_vs_phase3b_km": 1268.6 - test_final["center_mean_dpe_km"]
        }
    }
    
    with open(CONFIG["results_json_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON to: {CONFIG['results_json_path']}")
    print("=" * 80)
    print("PHASE 3C DEDICATED TRAINING & EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
