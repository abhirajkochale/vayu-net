"""
VAYU-NET Phase 4A — Temporal Cyclone Track Prediction Training Script

Trains TemporalTrackGRU model on 6-frame satellite sequences (t-15h to t0)
to predict future cyclone track positions at +12h, +24h, and +48h.

Primary Model Selection Metric:
  Lowest VALIDATION Mean Track DPE (km) = mean(DPE_12h, DPE_24h, DPE_48h)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import json
import math
import random
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ml.models.temporal_track_gru import TemporalTrackGRU
from ml.models.center_localization_cnn import (
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)

def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return 6371.0 * c

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    "experiment_name": "Phase4A_TemporalTrackGRU",
    "random_seed": 42,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 15,
    "patience": 4,
    "spatial_emb_dim": 128,
    "gru_hidden_dim": 128,
    "gru_num_layers": 2,
    "dropout": 0.1,
    "use_center_features": True,
    "sample_index_path": "data/manifests/vayu_net_sample_index.csv",
    "norm_stats_path": "data/interim/ml/train_normalization_stats.json",
    "spatial_checkpoint_path": "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
    "checkpoint_dir": "data/interim/ml/checkpoints",
    "checkpoint_path": "data/interim/ml/checkpoints/best_temporal_track_gru.pt",
    "results_json_path": "data/interim/ml/phase4a_temporal_track_results.json",
    "figures_dir": "docs/figures/temporal_track",
    "cache_dir": "data/interim/ml/cache",
    "cache_file": "data/interim/ml/cache/temporal_track_features.pt"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Dataset & Feature Extraction / Caching
# -----------------------------------------------------------------------------
FRAME_COLS = [
    "frame_t_minus_15h",
    "frame_t_minus_12h",
    "frame_t_minus_9h",
    "frame_t_minus_6h",
    "frame_t_minus_3h",
    "frame_t0"
]

def extract_and_cache_features(config, device):
    """
    Extracts frame-level spatial embeddings and center estimates from the frozen
    Phase 3C spatial encoder for all unique frames in the dataset.
    Caches the pre-assembled 6-step temporal sequences.
    """
    cache_path = config["cache_file"]
    if os.path.exists(cache_path):
        print(f"[Cache] Found pre-computed temporal features at: {cache_path}")
        return torch.load(cache_path, map_location="cpu")
        
    print(f"\n[Feature Extraction] Building temporal feature cache...")
    os.makedirs(config["cache_dir"], exist_ok=True)
    
    with open(config["norm_stats_path"], "r") as f:
        stats = json.load(f)
    mean_k = float(stats["mean_kelvin"])
    std_k = float(stats["std_kelvin"])
    
    # Load model in eval mode
    model = TemporalTrackGRU(
        spatial_checkpoint_path=config["spatial_checkpoint_path"],
        freeze_spatial_encoder=True,
        spatial_emb_dim=config["spatial_emb_dim"]
    ).to(device)
    model.eval()
    
    df = pd.read_csv(config["sample_index_path"])
    all_unique_frames = sorted(list(set(df[FRAME_COLS].values.flatten())))
    print(f"Total dataset samples: {len(df)}")
    print(f"Total unique NPZ frames to extract: {len(all_unique_frames)}")
    
    frame_dict = {}
    batch_size = 16
    t0_start = time.time()
    
    for b_start in range(0, len(all_unique_frames), batch_size):
        b_frames = all_unique_frames[b_start:b_start + batch_size]
        batch_arrays = []
        for fpath in b_frames:
            with np.load(fpath) as npz:
                arr = npz["irwin_cdr"].astype(np.float32)
            inv = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
            if np.any(inv):
                arr[inv] = mean_k
            arr = (arr - mean_k) / std_k
            batch_arrays.append(arr)
            
        t_batch = torch.from_numpy(np.stack(batch_arrays, axis=0)).unsqueeze(1).to(device)
        with torch.no_grad():
            embs, centers = model.extract_single_frame_feature(t_batch)
            embs_cpu = embs.cpu()
            centers_cpu = centers.cpu()
            
        for i, fpath in enumerate(b_frames):
            frame_dict[fpath] = (embs_cpu[i], centers_cpu[i])
            
        if (b_start + batch_size) % 200 < batch_size or (b_start + batch_size) >= len(all_unique_frames):
            done = min(b_start + batch_size, len(all_unique_frames))
            elapsed = time.time() - t0_start
            rate = done / elapsed
            rem = (len(all_unique_frames) - done) / rate if rate > 0 else 0
            print(f"  Extracted {done:4d}/{len(all_unique_frames)} frames ({rate:.1f} frames/s, ~{rem/60:.1f}m remaining)")
            
    print(f"[Feature Extraction] Complete in {time.time() - t0_start:.1f}s. Assembling 6-step temporal sequences...")
    
    # Assemble sequences for each sample
    sample_records = []
    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        storm_id = row["storm_id"]
        split = row["split"]
        t0_str = row["t0"]
        
        # 6-frame sequence
        step_features = []
        prev_center = None
        for col in FRAME_COLS:
            fpath = row[col]
            emb_t, center_t = frame_dict[fpath]
            if prev_center is None:
                delta_t = torch.zeros(2, dtype=torch.float32)
            else:
                delta_t = center_t - prev_center
            prev_center = center_t
            
            feat_t = torch.cat([emb_t, center_t, delta_t], dim=-1) # [132]
            step_features.append(feat_t)
            
        seq_tensor = torch.stack(step_features, dim=0) # [6, 132]
        
        # True future coordinates
        lat0, lon0 = float(row["imd_lat_t0"]), float(row["imd_lon_t0"])
        lat12, lon12 = float(row["imd_lat_12h"]), float(row["imd_lon_12h"])
        lat24, lon24 = float(row["imd_lat_24h"]), float(row["imd_lon_24h"])
        lat48, lon48 = float(row["imd_lat_48h"]), float(row["imd_lon_48h"])
        
        # Normalized coordinates in [0, 1]
        c0_norm = torch.tensor([(lat0 - LAT_MIN) / LAT_SPAN, (lon0 - LON_MIN) / LON_SPAN], dtype=torch.float32)
        c12_norm = torch.tensor([(lat12 - LAT_MIN) / LAT_SPAN, (lon12 - LON_MIN) / LON_SPAN], dtype=torch.float32)
        c24_norm = torch.tensor([(lat24 - LAT_MIN) / LAT_SPAN, (lon24 - LON_MIN) / LON_SPAN], dtype=torch.float32)
        c48_norm = torch.tensor([(lat48 - LAT_MIN) / LAT_SPAN, (lon48 - LON_MIN) / LON_SPAN], dtype=torch.float32)
        
        sample_records.append({
            "sample_id": sample_id,
            "storm_id": storm_id,
            "split": split,
            "t0": t0_str,
            "feature_sequence": seq_tensor,
            "c0_norm": c0_norm,
            "c12_norm": c12_norm,
            "c24_norm": c24_norm,
            "c48_norm": c48_norm,
            "c0_deg": torch.tensor([lat0, lon0], dtype=torch.float32),
            "c12_deg": torch.tensor([lat12, lon12], dtype=torch.float32),
            "c24_deg": torch.tensor([lat24, lon24], dtype=torch.float32),
            "c48_deg": torch.tensor([lat48, lon48], dtype=torch.float32)
        })
        
    cache_data = {
        "metadata": {
            "num_samples": len(sample_records),
            "feature_dim": sample_records[0]["feature_sequence"].shape[-1],
            "sequence_length": 6
        },
        "samples": sample_records
    }
    torch.save(cache_data, cache_path)
    print(f"[Cache] Saved temporal features to {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.2f} MB)")
    return cache_data


class CachedTemporalDataset(Dataset):
    """PyTorch Dataset loading pre-assembled temporal feature sequences."""
    def __init__(self, sample_records, split="TRAIN"):
        super().__init__()
        self.split = split.upper()
        self.records = [r for r in sample_records if r["split"] == self.split]
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        return self.records[idx]


# -----------------------------------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------------------------------
def evaluate_split(model, dataloader, device):
    """
    Evaluates multi-horizon track predictions across a dataset split.
    Computes DPE (km) at +12h, +24h, +48h and Mean Track DPE.
    """
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    
    total_loss = 0.0
    total_samples = 0
    
    dpes_12 = []
    dpes_24 = []
    dpes_48 = []
    
    pred_12_list, true_12_list = [], []
    pred_24_list, true_24_list = [], []
    pred_48_list, true_48_list = [], []
    sample_ids, storm_ids, t0s = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            feats = batch["feature_sequence"].to(device)
            c12_true = batch["c12_norm"].to(device)
            c24_true = batch["c24_norm"].to(device)
            c48_true = batch["c48_norm"].to(device)
            b_size = feats.size(0)
            total_samples += b_size
            
            out = model.forward_features(feats)
            p12 = out["pred_norm_12h"]
            p24 = out["pred_norm_24h"]
            p48 = out["pred_norm_48h"]
            
            l12 = loss_fn(p12, c12_true)
            l24 = loss_fn(p24, c24_true)
            l48 = loss_fn(p48, c48_true)
            total_loss += (l12 + l24 + l48).item() * b_size
            
            # Physical coordinates
            phys12 = TemporalTrackGRU.denormalize_coords(p12.cpu()).numpy()
            phys24 = TemporalTrackGRU.denormalize_coords(p24.cpu()).numpy()
            phys48 = TemporalTrackGRU.denormalize_coords(p48.cpu()).numpy()
            
            true12 = batch["c12_deg"].numpy()
            true24 = batch["c24_deg"].numpy()
            true48 = batch["c48_deg"].numpy()
            
            for i in range(b_size):
                d12 = haversine_km(true12[i, 0], true12[i, 1], phys12[i, 0], phys12[i, 1])
                d24 = haversine_km(true24[i, 0], true24[i, 1], phys24[i, 0], phys24[i, 1])
                d48 = haversine_km(true48[i, 0], true48[i, 1], phys48[i, 0], phys48[i, 1])
                dpes_12.append(d12)
                dpes_24.append(d24)
                dpes_48.append(d48)
                
                pred_12_list.append(phys12[i])
                pred_24_list.append(phys24[i])
                pred_48_list.append(phys48[i])
                true_12_list.append(true12[i])
                true_24_list.append(true24[i])
                true_48_list.append(true48[i])
                
                sample_ids.append(batch["sample_id"][i])
                storm_ids.append(batch["storm_id"][i])
                t0s.append(batch["t0"][i])
                
    dpes_12 = np.array(dpes_12)
    dpes_24 = np.array(dpes_24)
    dpes_48 = np.array(dpes_48)
    track_dpes = (dpes_12 + dpes_24 + dpes_48) / 3.0
    
    pred_12_arr = np.array(pred_12_list)
    true_12_arr = np.array(true_12_list)
    pred_24_arr = np.array(pred_24_list)
    true_24_arr = np.array(true_24_list)
    pred_48_arr = np.array(pred_48_list)
    true_48_arr = np.array(true_48_list)
    
    def calc_stats(arr):
        return {
            "mean_dpe_km": float(np.mean(arr)),
            "median_dpe_km": float(np.median(arr)),
            "p90_dpe_km": float(np.percentile(arr, 90)),
            "min_dpe_km": float(np.min(arr)),
            "max_dpe_km": float(np.max(arr))
        }
        
    metrics = {
        "loss_total": total_loss / total_samples,
        "mean_track_dpe_km": float(np.mean(track_dpes)),
        "median_track_dpe_km": float(np.median(track_dpes)),
        "h12": calc_stats(dpes_12),
        "h24": calc_stats(dpes_24),
        "h48": calc_stats(dpes_48),
        "lat_mae_deg": {
            "12h": float(np.mean(np.abs(pred_12_arr[:, 0] - true_12_arr[:, 0]))),
            "24h": float(np.mean(np.abs(pred_24_arr[:, 0] - true_24_arr[:, 0]))),
            "48h": float(np.mean(np.abs(pred_48_arr[:, 0] - true_48_arr[:, 0])))
        },
        "lon_mae_deg": {
            "12h": float(np.mean(np.abs(pred_12_arr[:, 1] - true_12_arr[:, 1]))),
            "24h": float(np.mean(np.abs(pred_24_arr[:, 1] - true_24_arr[:, 1]))),
            "48h": float(np.mean(np.abs(pred_48_arr[:, 1] - true_48_arr[:, 1])))
        },
        "raw": {
            "sample_ids": sample_ids,
            "storm_ids": storm_ids,
            "t0s": t0s,
            "dpes_12": dpes_12.tolist(),
            "dpes_24": dpes_24.tolist(),
            "dpes_48": dpes_48.tolist(),
            "track_dpes": track_dpes.tolist(),
            "pred_12": pred_12_arr.tolist(),
            "true_12": true_12_arr.tolist(),
            "pred_24": pred_24_arr.tolist(),
            "true_24": true_24_arr.tolist(),
            "pred_48": pred_48_arr.tolist(),
            "true_48": true_48_arr.tolist()
        }
    }
    return metrics


# -----------------------------------------------------------------------------
# Diagnostics & Plotting
# -----------------------------------------------------------------------------
def generate_all_diagnostics(history, best_epoch, val_final, test_final, save_dir):
    """Generates all 7 required Phase 4A diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    
    # 1. train_loss.png
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], 'o-', color='#1f77b4', lw=2, label='Train Multi-Horizon Loss')
    plt.plot(epochs, [h["val_loss"] for h in history], 's--', color='#ff7f0e', lw=2, label='Val Multi-Horizon Loss')
    plt.axvline(best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Best Checkpoint (Epoch {best_epoch})')
    plt.title("Phase 4A — Training & Validation Track Loss", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("Smooth L1 Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_loss.png"), dpi=250)
    plt.close()
    print("Saved: train_loss.png")
    
    # 2. validation_track_dpe.png
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [h["val_mean_track_dpe_km"] for h in history], 'o-', color='#d62728', lw=2.5, label='Val Mean Track DPE')
    plt.plot(epochs, [h["val_median_track_dpe_km"] for h in history], 's--', color='#2ca02c', lw=2, label='Val Median Track DPE')
    plt.axhline(189.06, color='gray', linestyle='--', label='Constant-Velocity Baseline Mean (189.1 km)')
    plt.axhline(308.95, color='orange', linestyle=':', label='Stationary Persistence Mean (309.0 km)')
    plt.axvline(best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Selected Epoch {best_epoch}')
    plt.title("Phase 4A — Validation Mean Track DPE Progression", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("Mean Track DPE (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "validation_track_dpe.png"), dpi=250)
    plt.close()
    print("Saved: validation_track_dpe.png")
    
    # 3. horizon_dpe_comparison.png
    horizons = ["+12h", "+24h", "+48h"]
    stat_means = [132.67, 263.37, 530.81]
    cv_means = [71.28, 150.74, 345.15]
    gru_means = [test_final["h12"]["mean_dpe_km"], test_final["h24"]["mean_dpe_km"], test_final["h48"]["mean_dpe_km"]]
    gru_meds = [test_final["h12"]["median_dpe_km"], test_final["h24"]["median_dpe_km"], test_final["h48"]["median_dpe_km"]]
    
    x = np.arange(len(horizons))
    width = 0.22
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5*width, stat_means, width, label='Stationary Persistence', color='#aec7e8', edgecolor='black')
    plt.bar(x - 0.5*width, cv_means, width, label='Constant Velocity', color='#ffbb78', edgecolor='black')
    plt.bar(x + 0.5*width, gru_means, width, label='TemporalTrackGRU (Mean)', color='#2ca02c', edgecolor='black')
    plt.bar(x + 1.5*width, gru_meds, width, label='TemporalTrackGRU (Median)', color='#98df8a', edgecolor='black')
    plt.xticks(x, horizons, fontsize=12)
    plt.ylabel("Direct Position Error (km)", fontsize=12)
    plt.title("Forecast Horizon Direct Position Error Comparison (TEST Split, N=371)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.5, axis='y'); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "horizon_dpe_comparison.png"), dpi=250)
    plt.close()
    print("Saved: horizon_dpe_comparison.png")
    
    # 4. test_dpe_distributions.png
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, h_key, h_title in [(axes[0], "dpes_12", "+12h Horizon"), (axes[1], "dpes_24", "+24h Horizon"), (axes[2], "dpes_48", "+48h Horizon")]:
        dpes = test_final["raw"][h_key]
        ax.hist(dpes, bins=30, color='#31688e', edgecolor='black', alpha=0.75)
        m_val = np.mean(dpes); med_val = np.median(dpes); p90_val = np.percentile(dpes, 90)
        ax.axvline(m_val, color='#d62728', linestyle='--', lw=2, label=f'Mean: {m_val:.1f} km')
        ax.axvline(med_val, color='#2ca02c', linestyle='-', lw=2, label=f'Median: {med_val:.1f} km')
        ax.axvline(p90_val, color='#ff7f0e', linestyle=':', lw=2, label=f'P90: {p90_val:.1f} km')
        ax.set_title(f"TEST DPE Distribution ({h_title})", fontsize=12, fontweight='bold')
        ax.set_xlabel("DPE (km)", fontsize=10); ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_dpe_distributions.png"), dpi=250)
    plt.close()
    print("Saved: test_dpe_distributions.png")
    
    # 5. predicted_vs_actual_tracks.png
    plt.figure(figsize=(12, 8))
    t_raw = test_final["raw"]
    for i in range(0, len(t_raw["sample_ids"]), 12): # Subsample for visual clarity
        t12 = t_raw["true_12"][i]; p12 = t_raw["pred_12"][i]
        t24 = t_raw["true_24"][i]; p24 = t_raw["pred_24"][i]
        t48 = t_raw["true_48"][i]; p48 = t_raw["pred_48"][i]
        
        plt.plot([t12[1], t24[1], t48[1]], [t12[0], t24[0], t48[0]], 'b-o', markersize=4, alpha=0.4)
        plt.plot([p12[1], p24[1], p48[1]], [p12[0], p24[0], p48[0]], 'r--^', markersize=4, alpha=0.4)
    plt.plot([], [], 'b-o', label='Ground Truth Future Tracks (IMD)')
    plt.plot([], [], 'r--^', label='TemporalTrackGRU Predicted Tracks')
    plt.xlim(40.0, 105.0); plt.ylim(-5.0, 35.0)
    plt.xlabel("Longitude (°E)", fontsize=11); plt.ylabel("Latitude (°N)", fontsize=11)
    plt.title("Predicted vs Actual Cyclone Future Tracks (+12h, +24h, +48h) — TEST Split", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_vs_actual_tracks.png"), dpi=250)
    plt.close()
    print("Saved: predicted_vs_actual_tracks.png")
    
    # 6. representative_tracks.png (Low & Median Error Examples)
    t_dpes = np.array(t_raw["track_dpes"])
    sorted_indices = np.argsort(t_dpes)
    low_idx = sorted_indices[:3] # Best 3
    med_idx = sorted_indices[len(sorted_indices)//2 - 1 : len(sorted_indices)//2 + 2] # Median 3
    
    rep_indices = list(low_idx) + list(med_idx)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for plot_i, idx in enumerate(rep_indices):
        ax = axes[plot_i // 3, plot_i % 3]
        sid = t_raw["storm_ids"][idx]
        t0 = t_raw["t0s"][idx]
        t_dpe = t_dpes[idx]
        
        t_lats = [t_raw["true_12"][idx][0], t_raw["true_24"][idx][0], t_raw["true_48"][idx][0]]
        t_lons = [t_raw["true_12"][idx][1], t_raw["true_24"][idx][1], t_raw["true_48"][idx][1]]
        
        p_lats = [t_raw["pred_12"][idx][0], t_raw["pred_24"][idx][0], t_raw["pred_48"][idx][0]]
        p_lons = [t_raw["pred_12"][idx][1], t_raw["pred_24"][idx][1], t_raw["pred_48"][idx][1]]
        
        ax.plot(t_lons, t_lats, 'b-o', lw=2, markersize=7, label='Ground Truth Track')
        ax.plot(p_lons, p_lats, 'r--^', lw=2, markersize=7, label='Predicted Track')
        for h_label, tx, ty, px, py in zip(["+12h", "+24h", "+48h"], t_lons, t_lats, p_lons, p_lats):
            ax.plot([tx, px], [ty, py], 'k:', alpha=0.5)
            ax.text(tx, ty+0.4, h_label, color='blue', fontsize=8)
            ax.text(px, py-0.6, h_label, color='red', fontsize=8)
            
        group_type = "Low-Error Example" if plot_i < 3 else "Median-Error Example"
        ax.set_title(f"{group_type}: {sid}\nt0: {t0} | Track DPE: {t_dpe:.1f} km", fontsize=10, fontweight='bold')
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "representative_tracks.png"), dpi=250)
    plt.close()
    print("Saved: representative_tracks.png")
    
    # 7. worst_case_tracks.png (Top Worst Track Errors)
    worst_indices = sorted_indices[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for plot_i, idx in enumerate(worst_indices):
        ax = axes[plot_i // 3, plot_i % 3]
        sid = t_raw["storm_ids"][idx]
        t0 = t_raw["t0s"][idx]
        t_dpe = t_dpes[idx]
        
        t_lats = [t_raw["true_12"][idx][0], t_raw["true_24"][idx][0], t_raw["true_48"][idx][0]]
        t_lons = [t_raw["true_12"][idx][1], t_raw["true_24"][idx][1], t_raw["true_48"][idx][1]]
        
        p_lats = [t_raw["pred_12"][idx][0], t_raw["pred_24"][idx][0], t_raw["pred_48"][idx][0]]
        p_lons = [t_raw["pred_12"][idx][1], t_raw["pred_24"][idx][1], t_raw["pred_48"][idx][1]]
        
        ax.plot(t_lons, t_lats, 'b-o', lw=2, markersize=7, label='Ground Truth Track')
        ax.plot(p_lons, p_lats, 'r--^', lw=2, markersize=7, label='Predicted Track')
        for h_label, tx, ty, px, py in zip(["+12h", "+24h", "+48h"], t_lons, t_lats, p_lons, p_lats):
            ax.plot([tx, px], [ty, py], 'k:', alpha=0.5)
            ax.text(tx, ty+0.4, h_label, color='blue', fontsize=8)
            ax.text(px, py-0.6, h_label, color='red', fontsize=8)
            
        ax.set_title(f"Worst Case #{plot_i+1}: {sid}\nt0: {t0} | Track DPE: {t_dpe:.1f} km", fontsize=10, fontweight='bold')
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "worst_case_tracks.png"), dpi=250)
    plt.close()
    print("Saved: worst_case_tracks.png")


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("VAYU-NET PHASE 4A — TEMPORAL CYCLONE TRACK PREDICTION TRAINING")
    print("=" * 80)
    
    set_seed(CONFIG["random_seed"])
    device = torch.device("cpu")
    print(f"Device:                 {device}")
    print(f"Random Seed:            {CONFIG['random_seed']}")
    print(f"Batch Size:             {CONFIG['batch_size']}")
    print(f"Max Epochs:             {CONFIG['max_epochs']}")
    print(f"Early Stopping Metric:  VALIDATION MEAN TRACK DPE (Patience = {CONFIG['patience']})")
    
    # 1. Feature Extraction & Caching
    cached_data = extract_and_cache_features(CONFIG, device)
    sample_records = cached_data["samples"]
    
    train_dataset = CachedTemporalDataset(sample_records, split="TRAIN")
    val_dataset = CachedTemporalDataset(sample_records, split="VALIDATION")
    test_dataset = CachedTemporalDataset(sample_records, split="TEST")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    
    print(f"\nDataset Splits:")
    print(f"  TRAIN:      {len(train_dataset)} samples / 81 storms")
    print(f"  VALIDATION: {len(val_dataset)} samples / 14 storms")
    print(f"  TEST:       {len(test_dataset)} samples / 31 storms (Strictly held out until final checkpoint)")
    
    # 2. Model Instantiation
    model = TemporalTrackGRU(
        spatial_checkpoint_path=CONFIG["spatial_checkpoint_path"],
        freeze_spatial_encoder=True,
        use_center_features=CONFIG["use_center_features"],
        spatial_emb_dim=CONFIG["spatial_emb_dim"],
        gru_hidden_dim=CONFIG["gru_hidden_dim"],
        gru_num_layers=CONFIG["gru_num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel: TemporalTrackGRU")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {frozen_params:,} (Phase 3C spatial encoder)")
    
    # 3. Optimizer & Scheduler (Governed by Validation Mean Track DPE)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss()
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)
    
    history = []
    best_val_track_dpe = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    t_train_start = time.time()
    
    print("\n" + "=" * 80)
    print("BEGINNING TRAINING (Criterion: Minimum Validation Mean Track DPE)")
    print("=" * 80)
    
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            feats = batch["feature_sequence"].to(device)
            c12 = batch["c12_norm"].to(device)
            c24 = batch["c24_norm"].to(device)
            c48 = batch["c48_norm"].to(device)
            b_size = feats.size(0)
            train_samples += b_size
            
            out = model.forward_features(feats)
            l12 = loss_fn(out["pred_norm_12h"], c12)
            l24 = loss_fn(out["pred_norm_24h"], c24)
            l48 = loss_fn(out["pred_norm_48h"], c48)
            
            loss = l12 + l24 + l48
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * b_size
            
        avg_train_loss = train_loss_total / train_samples
        
        # Evaluate on VALIDATION ONLY
        val_metrics = evaluate_split(model, val_loader, device)
        val_mean_track_dpe = val_metrics["mean_track_dpe_km"]
        val_med_track_dpe = val_metrics["median_track_dpe_km"]
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_mean_track_dpe)
        
        print(f"Epoch {epoch:2d}/{CONFIG['max_epochs']:2d} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss_total']:.4f} | LR: {current_lr:.6f}")
        print(f"  VAL Mean Track DPE:   {val_mean_track_dpe:.1f} km (Median: {val_med_track_dpe:.1f} km)")
        print(f"  VAL Horizon Breakdown: +12h: {val_metrics['h12']['mean_dpe_km']:.1f} km | "
              f"+24h: {val_metrics['h24']['mean_dpe_km']:.1f} km | +48h: {val_metrics['h48']['mean_dpe_km']:.1f} km")
              
        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss_total"],
            "val_mean_track_dpe_km": val_mean_track_dpe,
            "val_median_track_dpe_km": val_med_track_dpe,
            "val_12h_mean_dpe_km": val_metrics["h12"]["mean_dpe_km"],
            "val_24h_mean_dpe_km": val_metrics["h24"]["mean_dpe_km"],
            "val_48h_mean_dpe_km": val_metrics["h48"]["mean_dpe_km"],
            "lr": current_lr
        }
        history.append(record)
        
        # Primary Model Selection Criterion: VALIDATION MEAN TRACK DPE
        if val_mean_track_dpe < best_val_track_dpe:
            prev_best = best_val_track_dpe
            best_val_track_dpe = val_mean_track_dpe
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_track_dpe_km": val_mean_track_dpe,
                "val_metrics": val_metrics,
                "config": CONFIG
            }, CONFIG["checkpoint_path"])
            print(f"  >>> NEW BEST CHECKPOINT saved! Val Mean Track DPE: {prev_best:.1f} km -> {val_mean_track_dpe:.1f} km")
        else:
            epochs_no_improve += 1
            print(f"  >>> Val Mean Track DPE did not improve ({val_mean_track_dpe:.1f} km >= {best_val_track_dpe:.1f} km). Counter: {epochs_no_improve}/{CONFIG['patience']}")
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation Mean Track DPE did not improve for {CONFIG['patience']} consecutive epochs.")
                break
        print("-" * 80)
        
    training_duration = time.time() - t_train_start
    print(f"\nTraining completed in {training_duration:.1f}s.")
    
    # -------------------------------------------------------------
    # FREEZE CHECKPOINT & EVALUATE TEST STRICTLY ONCE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"LOADING BEST CHECKPOINT (Epoch {best_epoch}, Val Mean Track DPE: {best_val_track_dpe:.1f} km)")
    print("=" * 80)
    
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    train_final = evaluate_split(model, train_loader, device)
    val_final = evaluate_split(model, val_loader, device)
    
    print("\nEvaluating TEST split strictly ONCE using frozen best checkpoint...")
    test_final = evaluate_split(model, test_loader, device)
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION METRICS SUMMARY (PHASE 4A TEMPORAL TRACK)")
    print("=" * 80)
    print(f"{'Split':<12} | {'Mean Track DPE':<16} | {'+12h (Mean/Med/P90)':<24} | {'+24h (Mean/Med/P90)':<24} | {'+48h (Mean/Med/P90)'}")
    print("-" * 105)
    for name, m in [("TRAIN", train_final), ("VALIDATION", val_final), ("TEST", test_final)]:
        print(f"{name:<12} | {m['mean_track_dpe_km']:6.1f} km        | "
              f"{m['h12']['mean_dpe_km']:5.1f} / {m['h12']['median_dpe_km']:5.1f} / {m['h12']['p90_dpe_km']:5.1f} km | "
              f"{m['h24']['mean_dpe_km']:5.1f} / {m['h24']['median_dpe_km']:5.1f} / {m['h24']['p90_dpe_km']:5.1f} km | "
              f"{m['h48']['mean_dpe_km']:5.1f} / {m['h48']['median_dpe_km']:5.1f} / {m['h48']['p90_dpe_km']:5.1f} km")
    print("-" * 105)
    
    # Baseline comparison (TEST split)
    base_stationary = {"12h": 132.67, "24h": 263.37, "48h": 530.81, "mean_track": 308.95}
    base_cv = {"12h": 71.28, "24h": 150.74, "48h": 345.15, "mean_track": 189.06}
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH LOCKED BASELINES (TEST SPLIT, N=371)")
    print("=" * 80)
    print(f"{'Model / Baseline':<28} | {'Mean Track DPE':<16} | {'+12h DPE':<12} | {'+24h DPE':<12} | {'+48h DPE'}")
    print("-" * 85)
    print(f"{'Stationary Persistence':<28} | {base_stationary['mean_track']:6.1f} km        | {base_stationary['12h']:6.1f} km    | {base_stationary['24h']:6.1f} km    | {base_stationary['48h']:6.1f} km")
    print(f"{'Constant Velocity':<28} | {base_cv['mean_track']:6.1f} km        | {base_cv['12h']:6.1f} km    | {base_cv['24h']:6.1f} km    | {base_cv['48h']:6.1f} km")
    print(f"{'TemporalTrackGRU (Phase 4A)':<28} | {test_final['mean_track_dpe_km']:6.1f} km        | "
          f"{test_final['h12']['mean_dpe_km']:6.1f} km    | {test_final['h24']['mean_dpe_km']:6.1f} km    | {test_final['h48']['mean_dpe_km']:6.1f} km")
    print("-" * 85)
    
    # Generate Visual Diagnostics
    generate_all_diagnostics(history, best_epoch, val_final, test_final, CONFIG["figures_dir"])
    
    # Save Results JSON
    results = {
        "metadata": {
            "experiment_name": CONFIG["experiment_name"],
            "model_architecture": "TemporalTrackGRU (Shared Phase 3C Spatial Encoder + 2-layer GRU)",
            "parameter_count": {
                "total": total_params,
                "trainable": trainable_params,
                "frozen": frozen_params
            },
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
            "model_selection_criterion": "Minimum VALIDATION Mean Track DPE (km)",
            "training_duration_seconds": training_duration,
            "transferred_layers": "All convolutional layers from Phase 3C DedicatedCenterLocalizationResNet (frozen)",
            "sequence_length": 6,
            "temporal_cadence_hours": 3
        },
        "training_history": history,
        "metrics": {
            "train": {
                "mean_track_dpe_km": train_final["mean_track_dpe_km"],
                "median_track_dpe_km": train_final["median_track_dpe_km"],
                "h12": train_final["h12"],
                "h24": train_final["h24"],
                "h48": train_final["h48"],
                "lat_mae_deg": train_final["lat_mae_deg"],
                "lon_mae_deg": train_final["lon_mae_deg"]
            },
            "validation": {
                "mean_track_dpe_km": val_final["mean_track_dpe_km"],
                "median_track_dpe_km": val_final["median_track_dpe_km"],
                "h12": val_final["h12"],
                "h24": val_final["h24"],
                "h48": val_final["h48"],
                "lat_mae_deg": val_final["lat_mae_deg"],
                "lon_mae_deg": val_final["lon_mae_deg"]
            },
            "test": {
                "mean_track_dpe_km": test_final["mean_track_dpe_km"],
                "median_track_dpe_km": test_final["median_track_dpe_km"],
                "h12": test_final["h12"],
                "h24": test_final["h24"],
                "h48": test_final["h48"],
                "lat_mae_deg": test_final["lat_mae_deg"],
                "lon_mae_deg": test_final["lon_mae_deg"]
            }
        },
        "baseline_comparison_test": {
            "stationary_persistence": base_stationary,
            "constant_velocity": base_cv,
            "temporal_track_gru": {
                "12h_mean_dpe_km": test_final["h12"]["mean_dpe_km"],
                "24h_mean_dpe_km": test_final["h24"]["mean_dpe_km"],
                "48h_mean_dpe_km": test_final["h48"]["mean_dpe_km"],
                "mean_track_dpe_km": test_final["mean_track_dpe_km"]
            },
            "improvement_vs_stationary_km": {
                "12h": base_stationary["12h"] - test_final["h12"]["mean_dpe_km"],
                "24h": base_stationary["24h"] - test_final["h24"]["mean_dpe_km"],
                "48h": base_stationary["48h"] - test_final["h48"]["mean_dpe_km"],
                "mean_track": base_stationary["mean_track"] - test_final["mean_track_dpe_km"]
            },
            "improvement_vs_constant_velocity_km": {
                "12h": base_cv["12h"] - test_final["h12"]["mean_dpe_km"],
                "24h": base_cv["24h"] - test_final["h24"]["mean_dpe_km"],
                "48h": base_cv["48h"] - test_final["h48"]["mean_dpe_km"],
                "mean_track": base_cv["mean_track"] - test_final["mean_track_dpe_km"]
            }
        }
    }
    
    with open(CONFIG["results_json_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON to: {CONFIG['results_json_path']}")
    print("=" * 80)
    print("PHASE 4A TEMPORAL TRACK TRAINING & EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
