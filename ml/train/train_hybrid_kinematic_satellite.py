"""
VAYU-NET Phase 4B — Hybrid Kinematic + Satellite Residual Track Prediction Training

Trains two distinct variants:
  Variant A — Observed-Center Hybrid (Kinematic base from exact IMD centers at t-3h and t0)
  Variant B — Satellite-Derived Hybrid (Kinematic base from Phase 3C estimated centers at t-3h and t0)

Primary Model Selection Metric:
  Lowest VALIDATION Mean Track DPE (km) = mean(DPE_12h, DPE_24h, DPE_48h)
  computed on the reconstructed hybrid forecast: P_hat = P_kinematic + R_hat
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

from ml.models.hybrid_kinematic_satellite import (
    HybridKinematicSatelliteGRU,
    RESIDUAL_SCALE_KM,
    R_EARTH_KM, DEG_TO_RAD, RAD_TO_DEG,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)

CONFIG = {
    "experiment_name": "Phase4B_HybridKinematicSatellite",
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
    "spatial_checkpoint_path": "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
    "phase4a_cache_path": "data/interim/ml/cache/temporal_track_features.pt",
    "hybrid_cache_path": "data/interim/ml/cache/phase4b_hybrid_features.pt",
    "sample_index_path": "data/manifests/vayu_net_sample_index.csv",
    "imd_v2_path": "data/processed/imd_best_track_v2.csv",
    "checkpoint_dir": "data/interim/ml/checkpoints",
    "ckpt_variant_a": "data/interim/ml/checkpoints/best_phase4b_variant_a.pt",
    "ckpt_variant_b": "data/interim/ml/checkpoints/best_phase4b_variant_b.pt",
    "results_json_path": "data/interim/ml/phase4b_hybrid_results.json",
    "figures_dir": "docs/figures/phase4b_hybrid"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return 6371.0 * c


# -----------------------------------------------------------------------------
# Cache Assembly for Hybrid Features
# -----------------------------------------------------------------------------
def build_hybrid_cache(config):
    """
    Builds and caches hybrid kinematic features and residual targets for
    both Variant A (observed IMD centers) and Variant B (satellite-derived centers).
    """
    cache_path = config["hybrid_cache_path"]
    if os.path.exists(cache_path):
        print(f"[Cache] Found pre-computed hybrid features at: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)
        
    print(f"\n[Cache Assembly] Building hybrid features cache: {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    assert os.path.exists(config["phase4a_cache_path"]), f"Phase 4A cache missing: {config['phase4a_cache_path']}"
    raw_cache = torch.load(config["phase4a_cache_path"], map_location="cpu", weights_only=False)
    sample_records = raw_cache["samples"]
    
    df_samples = pd.read_csv(config["sample_index_path"])
    df_v2 = pd.read_csv(config["imd_v2_path"])
    
    storm_obs_sorted = {}
    for sid, group in df_v2.groupby("storm_id"):
        gdf = group.copy()
        gdf["dt"] = pd.to_datetime(gdf["timestamp_utc"])
        gdf = gdf.sort_values("dt").reset_index(drop=True)
        storm_obs_sorted[sid] = gdf
        
    sample_map = {r["sample_id"]: r for _, r in df_samples.iterrows()}
    
    hybrid_samples = []
    for s in sample_records:
        sid_sample = s["sample_id"]
        row = sample_map[sid_sample]
        sid = row["storm_id"]
        t0_dt = pd.to_datetime(row["t0"])
        
        # 1. True targets
        true_12 = s["c12_deg"].numpy()
        true_24 = s["c24_deg"].numpy()
        true_48 = s["c48_deg"].numpy()
        
        # 2. Variant A: Observed IMD Kinematic Base
        lat0, lon0 = float(row["imd_lat_t0"]), float(row["imd_lon_t0"])
        s_obs = storm_obs_sorted[sid]
        priors = s_obs[s_obs["dt"] < t0_dt]
        if len(priors) > 0:
            latest_prior = priors.iloc[-1]
            delta_h = (t0_dt - latest_prior["dt"]).total_seconds() / 3600.0
            v_lat_a = (lat0 - latest_prior["latitude"]) / delta_h
            d_lon = lon0 - latest_prior["longitude"]
            if d_lon > 180.0: d_lon -= 360.0
            elif d_lon < -180.0: d_lon += 360.0
            v_lon_a = d_lon / delta_h
        else:
            v_lat_a, v_lon_a = 0.0, 0.0
            
        kin_a_12 = np.array([lat0 + v_lat_a * 12.0, lon0 + v_lon_a * 12.0], dtype=np.float32)
        kin_a_24 = np.array([lat0 + v_lat_a * 24.0, lon0 + v_lon_a * 24.0], dtype=np.float32)
        kin_a_48 = np.array([lat0 + v_lat_a * 48.0, lon0 + v_lon_a * 48.0], dtype=np.float32)
        
        # Kinematic context vector A: [v_north, v_east, speed, norm_lat0, norm_lon0]
        v_north_a = v_lat_a * DEG_TO_RAD * R_EARTH_KM
        v_east_a = v_lon_a * math.cos(lat0 * DEG_TO_RAD) * DEG_TO_RAD * R_EARTH_KM
        speed_a = math.sqrt(v_north_a**2 + v_east_a**2)
        kin_ctx_a = torch.tensor([
            v_north_a / 50.0,
            v_east_a / 50.0,
            speed_a / 50.0,
            (lat0 - LAT_MIN) / LAT_SPAN,
            (lon0 - LON_MIN) / LON_SPAN
        ], dtype=torch.float32)
        
        # True residuals in km (Variant A)
        res_a_12 = HybridKinematicSatelliteGRU.latlon_to_residual_km(torch.from_numpy(kin_a_12), torch.from_numpy(true_12))
        res_a_24 = HybridKinematicSatelliteGRU.latlon_to_residual_km(torch.from_numpy(kin_a_24), torch.from_numpy(true_24))
        res_a_48 = HybridKinematicSatelliteGRU.latlon_to_residual_km(torch.from_numpy(kin_a_48), torch.from_numpy(true_48))
        
        # 3. Variant B: Satellite-Derived Kinematic Base
        seq = s["feature_sequence"]
        c_prev_b = seq[4, 128:130] # t-3h Phase 3C estimated normalized center
        c_curr_b = seq[5, 128:130] # t0 Phase 3C estimated normalized center
        
        lat_prev_b = LAT_MIN + c_prev_b[0].item() * LAT_SPAN
        lon_prev_b = LON_MIN + c_prev_b[1].item() * LON_SPAN
        lat_curr_b = LAT_MIN + c_curr_b[0].item() * LAT_SPAN
        lon_curr_b = LON_MIN + c_curr_b[1].item() * LON_SPAN
        
        v_lat_b = (lat_curr_b - lat_prev_b) / 3.0
        v_lon_b = (lon_curr_b - lon_prev_b) / 3.0
        
        kin_b_12 = np.array([lat_curr_b + v_lat_b * 12.0, lon_curr_b + v_lon_b * 12.0], dtype=np.float32)
        kin_b_24 = np.array([lat_curr_b + v_lat_b * 24.0, lon_curr_b + v_lon_b * 24.0], dtype=np.float32)
        kin_b_48 = np.array([lat_curr_b + v_lat_b * 48.0, lon_curr_b + v_lon_b * 48.0], dtype=np.float32)
        
        v_north_b = v_lat_b * DEG_TO_RAD * R_EARTH_KM
        v_east_b = v_lon_b * math.cos(lat_curr_b * DEG_TO_RAD) * DEG_TO_RAD * R_EARTH_KM
        speed_b = math.sqrt(v_north_b**2 + v_east_b**2)
        kin_ctx_b = torch.tensor([
            v_north_b / 50.0,
            v_east_b / 50.0,
            speed_b / 50.0,
            c_curr_b[0].item(),
            c_curr_b[1].item()
        ], dtype=torch.float32)
        
        # True residuals in km (Variant B)
        res_b_12 = HybridKinematicSatelliteGRU.latlon_to_residual_km(torch.from_numpy(kin_b_12), torch.from_numpy(true_12))
        res_b_24 = HybridKinematicSatelliteGRU.latlon_to_residual_km(torch.from_numpy(kin_b_24), torch.from_numpy(true_24))
        res_b_48 = HybridKinematicSatelliteGRU.latlon_to_residual_km(torch.from_numpy(kin_b_48), torch.from_numpy(true_48))
        
        hybrid_samples.append({
            "sample_id": sid_sample,
            "storm_id": sid,
            "split": s["split"],
            "t0": s["t0"],
            "feature_sequence": seq,
            "true_12": torch.from_numpy(true_12),
            "true_24": torch.from_numpy(true_24),
            "true_48": torch.from_numpy(true_48),
            # Variant A
            "kin_a_12": torch.from_numpy(kin_a_12),
            "kin_a_24": torch.from_numpy(kin_a_24),
            "kin_a_48": torch.from_numpy(kin_a_48),
            "kin_ctx_a": kin_ctx_a,
            "res_a_12": res_a_12,
            "res_a_24": res_a_24,
            "res_a_48": res_a_48,
            # Variant B
            "kin_b_12": torch.from_numpy(kin_b_12),
            "kin_b_24": torch.from_numpy(kin_b_24),
            "kin_b_48": torch.from_numpy(kin_b_48),
            "kin_ctx_b": kin_ctx_b,
            "res_b_12": res_b_12,
            "res_b_24": res_b_24,
            "res_b_48": res_b_48
        })
        
    cache_payload = {
        "metadata": {
            "num_samples": len(hybrid_samples),
            "feature_dim": hybrid_samples[0]["feature_sequence"].shape[-1],
            "sequence_length": 6
        },
        "samples": hybrid_samples
    }
    torch.save(cache_payload, cache_path)
    print(f"[Cache Assembly] Saved hybrid features to {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.2f} MB)")
    return cache_payload


class HybridDataset(Dataset):
    """Dataset serving either Variant A or Variant B data."""
    def __init__(self, samples, split="TRAIN", variant="A"):
        super().__init__()
        self.split = split.upper()
        self.variant = variant.upper()
        self.records = [s for s in samples if s["split"] == self.split]
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        r = self.records[idx]
        v_key = "a" if self.variant == "A" else "b"
        return {
            "sample_id": r["sample_id"],
            "storm_id": r["storm_id"],
            "split": r["split"],
            "t0": r["t0"],
            "feature_sequence": r["feature_sequence"],
            "true_12": r["true_12"],
            "true_24": r["true_24"],
            "true_48": r["true_48"],
            "kin_12": r[f"kin_{v_key}_12"],
            "kin_24": r[f"kin_{v_key}_24"],
            "kin_48": r[f"kin_{v_key}_48"],
            "kin_ctx": r[f"kin_ctx_{v_key}"],
            "res_12": r[f"res_{v_key}_12"],
            "res_24": r[f"res_{v_key}_24"],
            "res_48": r[f"res_{v_key}_48"]
        }


# -----------------------------------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------------------------------
def evaluate_hybrid_split(model, dataloader, device):
    """
    Evaluates hybrid residual forecasting performance across a dataset split.
    Reconstructs physical coordinates: P_hat = P_kinematic + R_hat
    Computes DPE (km) against true IMD future coordinates.
    """
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    
    total_loss = 0.0
    total_samples = 0
    
    dpes_12, dpes_24, dpes_48 = [], [], []
    kin_dpes_12, kin_dpes_24, kin_dpes_48 = [], [], []
    res_mags_12, res_mags_24, res_mags_48 = [], [], []
    res_bias_n_12, res_bias_e_12 = [], []
    res_bias_n_24, res_bias_e_24 = [], []
    res_bias_n_48, res_bias_e_48 = [], []
    
    sample_ids, storm_ids, t0s = [], [], []
    pred_12_all, true_12_all, kin_12_all = [], [], []
    pred_24_all, true_24_all, kin_24_all = [], [], []
    pred_48_all, true_48_all, kin_48_all = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            feats = batch["feature_sequence"].to(device)
            kin_ctx = batch["kin_ctx"].to(device)
            kin12 = batch["kin_12"].to(device)
            kin24 = batch["kin_24"].to(device)
            kin48 = batch["kin_48"].to(device)
            
            res12_true = batch["res_12"].to(device)
            res24_true = batch["res_24"].to(device)
            res48_true = batch["res_48"].to(device)
            
            true12 = batch["true_12"].numpy()
            true24 = batch["true_24"].numpy()
            true48 = batch["true_48"].numpy()
            
            b_size = feats.size(0)
            total_samples += b_size
            
            out = model.forward_features(feats, kin_ctx)
            
            # Loss computed on scaled normalized residuals (residual_km / 100.0)
            l12 = loss_fn(out["r12_norm"], res12_true / RESIDUAL_SCALE_KM)
            l24 = loss_fn(out["r24_norm"], res24_true / RESIDUAL_SCALE_KM)
            l48 = loss_fn(out["r48_norm"], res48_true / RESIDUAL_SCALE_KM)
            total_loss += (l12 + l24 + l48).item() * b_size
            
            # Reconstruct physical hybrid predictions
            pred12 = HybridKinematicSatelliteGRU.residual_km_to_latlon(kin12, out["r12_km"]).cpu().numpy()
            pred24 = HybridKinematicSatelliteGRU.residual_km_to_latlon(kin24, out["r24_km"]).cpu().numpy()
            pred48 = HybridKinematicSatelliteGRU.residual_km_to_latlon(kin48, out["r48_km"]).cpu().numpy()
            
            kin12_np = kin12.cpu().numpy()
            kin24_np = kin24.cpu().numpy()
            kin48_np = kin48.cpu().numpy()
            
            r12_np = out["r12_km"].cpu().numpy()
            r24_np = out["r24_km"].cpu().numpy()
            r48_np = out["r48_km"].cpu().numpy()
            
            for i in range(b_size):
                # Hybrid DPE
                d12 = haversine_km(pred12[i, 0], pred12[i, 1], true12[i, 0], true12[i, 1])
                d24 = haversine_km(pred24[i, 0], pred24[i, 1], true24[i, 0], true24[i, 1])
                d48 = haversine_km(pred48[i, 0], pred48[i, 1], true48[i, 0], true48[i, 1])
                dpes_12.append(d12); dpes_24.append(d24); dpes_48.append(d48)
                
                # Kinematic Base DPE
                kd12 = haversine_km(kin12_np[i, 0], kin12_np[i, 1], true12[i, 0], true12[i, 1])
                kd24 = haversine_km(kin24_np[i, 0], kin24_np[i, 1], true24[i, 0], true24[i, 1])
                kd48 = haversine_km(kin48_np[i, 0], kin48_np[i, 1], true48[i, 0], true48[i, 1])
                kin_dpes_12.append(kd12); kin_dpes_24.append(kd24); kin_dpes_48.append(kd48)
                
                # Residual magnitudes & biases
                res_mags_12.append(math.sqrt(r12_np[i, 0]**2 + r12_np[i, 1]**2))
                res_mags_24.append(math.sqrt(r24_np[i, 0]**2 + r24_np[i, 1]**2))
                res_mags_48.append(math.sqrt(r48_np[i, 0]**2 + r48_np[i, 1]**2))
                
                res_bias_n_12.append(r12_np[i, 0]); res_bias_e_12.append(r12_np[i, 1])
                res_bias_n_24.append(r24_np[i, 0]); res_bias_e_24.append(r24_np[i, 1])
                res_bias_n_48.append(r48_np[i, 0]); res_bias_e_48.append(r48_np[i, 1])
                
                sample_ids.append(batch["sample_id"][i])
                storm_ids.append(batch["storm_id"][i])
                t0s.append(batch["t0"][i])
                
                pred_12_all.append(pred12[i]); true_12_all.append(true12[i]); kin_12_all.append(kin12_np[i])
                pred_24_all.append(pred24[i]); true_24_all.append(true24[i]); kin_24_all.append(kin24_np[i])
                pred_48_all.append(pred48[i]); true_48_all.append(true48[i]); kin_48_all.append(kin48_np[i])
                
    d12_arr, d24_arr, d48_arr = np.array(dpes_12), np.array(dpes_24), np.array(dpes_48)
    kd12_arr, kd24_arr, kd48_arr = np.array(kin_dpes_12), np.array(kin_dpes_24), np.array(kin_dpes_48)
    track_dpes = (d12_arr + d24_arr + d48_arr) / 3.0
    kin_track_dpes = (kd12_arr + kd24_arr + kd48_arr) / 3.0
    
    def stats(arr):
        return {
            "mean_dpe_km": float(np.mean(arr)),
            "median_dpe_km": float(np.median(arr)),
            "p90_dpe_km": float(np.percentile(arr, 90)),
            "min_dpe_km": float(np.min(arr)),
            "max_dpe_km": float(np.max(arr))
        }
        
    return {
        "loss_total": total_loss / total_samples,
        "mean_track_dpe_km": float(np.mean(track_dpes)),
        "median_track_dpe_km": float(np.median(track_dpes)),
        "kin_mean_track_dpe_km": float(np.mean(kin_track_dpes)),
        "kin_median_track_dpe_km": float(np.median(kin_track_dpes)),
        "improvement_over_kin_mean_km": float(np.mean(kin_track_dpes) - np.mean(track_dpes)),
        "h12": stats(d12_arr),
        "h24": stats(d24_arr),
        "h48": stats(d48_arr),
        "kin_h12": stats(kd12_arr),
        "kin_h24": stats(kd24_arr),
        "kin_h48": stats(kd48_arr),
        "residuals": {
            "12h": {"mean_mag_km": float(np.mean(res_mags_12)), "median_mag_km": float(np.median(res_mags_12)), "bias_north_km": float(np.mean(res_bias_n_12)), "bias_east_km": float(np.mean(res_bias_e_12))},
            "24h": {"mean_mag_km": float(np.mean(res_mags_24)), "median_mag_km": float(np.median(res_mags_24)), "bias_north_km": float(np.mean(res_bias_n_24)), "bias_east_km": float(np.mean(res_bias_e_24))},
            "48h": {"mean_mag_km": float(np.mean(res_mags_48)), "median_mag_km": float(np.median(res_mags_48)), "bias_north_km": float(np.mean(res_bias_n_48)), "bias_east_km": float(np.mean(res_bias_e_48))}
        },
        "raw": {
            "sample_ids": sample_ids,
            "storm_ids": storm_ids,
            "t0s": t0s,
            "dpes_12": d12_arr.tolist(), "dpes_24": d24_arr.tolist(), "dpes_48": d48_arr.tolist(), "track_dpes": track_dpes.tolist(),
            "kin_dpes_12": kd12_arr.tolist(), "kin_dpes_24": kd24_arr.tolist(), "kin_dpes_48": kd48_arr.tolist(), "kin_track_dpes": kin_track_dpes.tolist(),
            "pred_12": np.array(pred_12_all).tolist(), "true_12": np.array(true_12_all).tolist(), "kin_12": np.array(kin_12_all).tolist(),
            "pred_24": np.array(pred_24_all).tolist(), "true_24": np.array(true_24_all).tolist(), "kin_24": np.array(kin_24_all).tolist(),
            "pred_48": np.array(pred_48_all).tolist(), "true_48": np.array(true_48_all).tolist(), "kin_48": np.array(kin_48_all).tolist(),
            "res_mags_12": res_mags_12, "res_mags_24": res_mags_24, "res_mags_48": res_mags_48,
            "res_n_12": res_bias_n_12, "res_e_12": res_bias_e_12,
            "res_n_24": res_bias_n_24, "res_e_24": res_bias_e_24,
            "res_n_48": res_bias_n_48, "res_e_48": res_bias_e_48
        }
    }


# -----------------------------------------------------------------------------
# Training Routine for a Single Variant
# -----------------------------------------------------------------------------
def train_variant(variant_name, sample_records, config, device):
    """Trains HybridKinematicSatelliteGRU on Variant A or B with early stopping."""
    print("\n" + "=" * 80)
    print(f"STARTING TRAINING: VARIANT {variant_name.upper()}")
    print("=" * 80)
    
    ckpt_path = config["ckpt_variant_a"] if variant_name == "A" else config["ckpt_variant_b"]
    
    train_ds = HybridDataset(sample_records, split="TRAIN", variant=variant_name)
    val_ds = HybridDataset(sample_records, split="VALIDATION", variant=variant_name)
    test_ds = HybridDataset(sample_records, split="TEST", variant=variant_name)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    set_seed(config["random_seed"])
    model = HybridKinematicSatelliteGRU(
        spatial_checkpoint_path=config["spatial_checkpoint_path"],
        freeze_spatial_encoder=True,
        spatial_emb_dim=config["spatial_emb_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        gru_num_layers=config["gru_num_layers"],
        dropout=config["dropout"]
    ).to(device)
    
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss()
    
    history = []
    best_val_track_dpe = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    t_start = time.time()
    
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            feats = batch["feature_sequence"].to(device)
            kin_ctx = batch["kin_ctx"].to(device)
            res12_true = batch["res_12"].to(device)
            res24_true = batch["res_24"].to(device)
            res48_true = batch["res_48"].to(device)
            b_size = feats.size(0)
            train_samples += b_size
            
            out = model.forward_features(feats, kin_ctx)
            l12 = loss_fn(out["r12_norm"], res12_true / RESIDUAL_SCALE_KM)
            l24 = loss_fn(out["r24_norm"], res24_true / RESIDUAL_SCALE_KM)
            l48 = loss_fn(out["r48_norm"], res48_true / RESIDUAL_SCALE_KM)
            loss = l12 + l24 + l48
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * b_size
            
        avg_train_loss = train_loss_total / train_samples
        
        # Validation evaluation on final reconstructed hybrid track DPE
        val_metrics = evaluate_hybrid_split(model, val_loader, device)
        val_mean_track_dpe = val_metrics["mean_track_dpe_km"]
        val_med_track_dpe = val_metrics["median_track_dpe_km"]
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_mean_track_dpe)
        
        print(f"Epoch {epoch:2d}/{config['max_epochs']:2d} Summary (Variant {variant_name}):")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss_total']:.4f} | LR: {current_lr:.6f}")
        print(f"  VAL Mean Track DPE:   {val_mean_track_dpe:.1f} km (Kinematic Base: {val_metrics['kin_mean_track_dpe_km']:.1f} km | Diff: {val_metrics['improvement_over_kin_mean_km']:+.1f} km)")
        print(f"  VAL Horizon Breakdown: +12h: {val_metrics['h12']['mean_dpe_km']:.1f} km | +24h: {val_metrics['h24']['mean_dpe_km']:.1f} km | +48h: {val_metrics['h48']['mean_dpe_km']:.1f} km")
        
        rec = {
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
        history.append(rec)
        
        if val_mean_track_dpe < best_val_track_dpe:
            prev_best = best_val_track_dpe
            best_val_track_dpe = val_mean_track_dpe
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "variant": variant_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mean_track_dpe_km": val_mean_track_dpe,
                "val_metrics": val_metrics,
                "config": config
            }, ckpt_path)
            print(f"  >>> NEW BEST CHECKPOINT saved! Val Mean Track DPE: {prev_best:.1f} km -> {val_mean_track_dpe:.1f} km")
        else:
            epochs_no_improve += 1
            print(f"  >>> Val Mean Track DPE did not improve. Counter: {epochs_no_improve}/{config['patience']}")
            if epochs_no_improve >= config["patience"]:
                print(f"\n[EARLY STOPPING] Validation Mean Track DPE did not improve for {config['patience']} consecutive epochs.")
                break
        print("-" * 80)
        
    duration = time.time() - t_start
    print(f"Training Variant {variant_name} complete in {duration:.1f}s.")
    
    # Freeze best checkpoint and evaluate TEST strictly once
    print(f"\nLoading best checkpoint for Variant {variant_name} (Epoch {best_epoch}, Val Mean Track DPE: {best_val_track_dpe:.1f} km)...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    train_final = evaluate_hybrid_split(model, train_loader, device)
    val_final = evaluate_hybrid_split(model, val_loader, device)
    
    print(f"\nEvaluating TEST split strictly ONCE for Variant {variant_name}...")
    test_final = evaluate_hybrid_split(model, test_loader, device)
    
    return {
        "best_epoch": best_epoch,
        "best_val_dpe": best_val_track_dpe,
        "history": history,
        "duration": duration,
        "train_final": train_final,
        "val_final": val_final,
        "test_final": test_final
    }


# -----------------------------------------------------------------------------
# Diagnostics & Plotting
# -----------------------------------------------------------------------------
def generate_all_diagnostics(res_a, res_b, save_dir):
    """Generates all 10 requested diagnostic figures for Phase 4B."""
    os.makedirs(save_dir, exist_ok=True)
    
    t_final_a = res_a["test_final"]
    t_final_b = res_b["test_final"]
    
    # 1. baseline_comparison.png
    models = ["Stationary Persistence", "Constant Velocity", "Phase 4A Satellite GRU", "Phase 4B Hybrid A", "Phase 4B Hybrid B"]
    means = [308.95, 189.06, 848.50, t_final_a["mean_track_dpe_km"], t_final_b["mean_track_dpe_km"]]
    meds = [266.16, 124.09, 661.90, t_final_a["median_track_dpe_km"], t_final_b["median_track_dpe_km"]]
    
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, means, width, label='Mean Track DPE (km)', color='#3182bd', edgecolor='black')
    plt.bar(x + width/2, meds, width, label='Median Track DPE (km)', color='#9ecae1', edgecolor='black')
    plt.xticks(x, models, rotation=15, ha='right', fontsize=10)
    plt.ylabel("Direct Position Error (km)", fontsize=11)
    plt.title("Phase 4B — Comprehensive Model & Baseline Comparison (TEST Split, N=371)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.5, axis='y'); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_comparison.png"), dpi=250)
    plt.close()
    print("Saved: baseline_comparison.png")
    
    # 2. validation_track_dpe.png
    plt.figure(figsize=(9, 5))
    plt.plot([h["epoch"] for h in res_a["history"]], [h["val_mean_track_dpe_km"] for h in res_a["history"]], 'o-', label='Variant A (Observed Center)', color='#2ca02c', lw=2)
    plt.plot([h["epoch"] for h in res_b["history"]], [h["val_mean_track_dpe_km"] for h in res_b["history"]], 's--', label='Variant B (Satellite-Derived)', color='#d62728', lw=2)
    plt.axhline(189.06, color='gray', linestyle='--', label='Exact-Center Constant Velocity (189.1 km)')
    plt.title("Phase 4B — Validation Mean Track DPE Progression Across Epochs", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("Mean Track DPE (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5); plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "validation_track_dpe.png"), dpi=250)
    plt.close()
    print("Saved: validation_track_dpe.png")
    
    # 3. horizon_dpe_comparison.png
    horizons = ["+12h", "+24h", "+48h"]
    cv_means = [71.28, 150.74, 345.15]
    hyb_a_means = [t_final_a["h12"]["mean_dpe_km"], t_final_a["h24"]["mean_dpe_km"], t_final_a["h48"]["mean_dpe_km"]]
    hyb_b_means = [t_final_b["h12"]["mean_dpe_km"], t_final_b["h24"]["mean_dpe_km"], t_final_b["h48"]["mean_dpe_km"]]
    
    x = np.arange(len(horizons))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, cv_means, width, label='Exact Constant Velocity Base', color='#ffbb78', edgecolor='black')
    plt.bar(x, hyb_a_means, width, label='Variant A Hybrid (Observed Base)', color='#2ca02c', edgecolor='black')
    plt.bar(x + width, hyb_b_means, width, label='Variant B Hybrid (Sat Base)', color='#d62728', edgecolor='black')
    plt.xticks(x, horizons, fontsize=11)
    plt.ylabel("Mean DPE (km)", fontsize=11)
    plt.title("Forecast Horizon Direct Position Error Comparison (TEST Split)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.5, axis='y'); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "horizon_dpe_comparison.png"), dpi=250)
    plt.close()
    print("Saved: horizon_dpe_comparison.png")
    
    # 4. residual_distributions.png
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, h, title in [(axes[0], "12", "+12h Residual Magnitude"), (axes[1], "24", "+24h Residual Magnitude"), (axes[2], "48", "+48h Residual Magnitude")]:
        mags_a = t_final_a["raw"][f"res_mags_{h}"]
        mags_b = t_final_b["raw"][f"res_mags_{h}"]
        ax.hist(mags_a, bins=25, alpha=0.6, label='Variant A', color='#2ca02c', density=True)
        ax.hist(mags_b, bins=25, alpha=0.6, label='Variant B', color='#d62728', density=True)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Residual Magnitude (km)"); ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_distributions.png"), dpi=250)
    plt.close()
    print("Saved: residual_distributions.png")
    
    # 5. predicted_vs_actual_residuals.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    true_res_n_24 = np.array(t_final_a["raw"]["true_24"])[:, 0] - np.array(t_final_a["raw"]["kin_24"])[:, 0]
    true_res_n_24_km = true_res_n_24 * DEG_TO_RAD * R_EARTH_KM
    pred_res_n_24_km = np.array(t_final_a["raw"]["res_n_24"])
    
    axes[0].scatter(true_res_n_24_km, pred_res_n_24_km, alpha=0.5, color='#2ca02c', s=20)
    axes[0].plot([-600, 600], [-600, 600], 'k--', lw=1.5, label='Identity (Ideal)')
    axes[0].set_title("Variant A: Northward Residual (+24h)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("True Residual (km)"); axes[0].set_ylabel("Predicted Residual (km)")
    axes[0].set_xlim(-600, 600); axes[0].set_ylim(-600, 600); axes[0].grid(True, linestyle=":", alpha=0.5); axes[0].legend()
    
    true_res_e_24 = np.array(t_final_a["raw"]["true_24"])[:, 1] - np.array(t_final_a["raw"]["kin_24"])[:, 1]
    true_res_e_24_km = true_res_e_24 * np.cos(np.array(t_final_a["raw"]["kin_24"])[:, 0] * DEG_TO_RAD) * DEG_TO_RAD * R_EARTH_KM
    pred_res_e_24_km = np.array(t_final_a["raw"]["res_e_24"])
    
    axes[1].scatter(true_res_e_24_km, pred_res_e_24_km, alpha=0.5, color='#1f77b4', s=20)
    axes[1].plot([-600, 600], [-600, 600], 'k--', lw=1.5, label='Identity (Ideal)')
    axes[1].set_title("Variant A: Eastward Residual (+24h)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("True Residual (km)"); axes[1].set_ylabel("Predicted Residual (km)")
    axes[1].set_xlim(-600, 600); axes[1].set_ylim(-600, 600); axes[1].grid(True, linestyle=":", alpha=0.5); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_vs_actual_residuals.png"), dpi=250)
    plt.close()
    print("Saved: predicted_vs_actual_residuals.png")
    
    # 6. hybrid_vs_kinematic_tracks.png
    plt.figure(figsize=(12, 8))
    raw_a = t_final_a["raw"]
    for i in range(0, len(raw_a["sample_ids"]), 15):
        t_lats = [raw_a["true_12"][i][0], raw_a["true_24"][i][0], raw_a["true_48"][i][0]]
        t_lons = [raw_a["true_12"][i][1], raw_a["true_24"][i][1], raw_a["true_48"][i][1]]
        k_lats = [raw_a["kin_12"][i][0], raw_a["kin_24"][i][0], raw_a["kin_48"][i][0]]
        k_lons = [raw_a["kin_12"][i][1], raw_a["kin_24"][i][1], raw_a["kin_48"][i][1]]
        p_lats = [raw_a["pred_12"][i][0], raw_a["pred_24"][i][0], raw_a["pred_48"][i][0]]
        p_lons = [raw_a["pred_12"][i][1], raw_a["pred_24"][i][1], raw_a["pred_48"][i][1]]
        
        plt.plot(t_lons, t_lats, 'b-o', markersize=4, alpha=0.4)
        plt.plot(k_lons, k_lats, 'k--', markersize=3, alpha=0.4)
        plt.plot(p_lons, p_lats, 'r-^', markersize=4, alpha=0.4)
    plt.plot([], [], 'b-o', label='Ground Truth Track (IMD)')
    plt.plot([], [], 'k--', label='Kinematic Constant Velocity Base')
    plt.plot([], [], 'r-^', label='Variant A Hybrid Forecast (Kinematic + Residual)')
    plt.xlim(40.0, 105.0); plt.ylim(-5.0, 35.0)
    plt.xlabel("Longitude (°E)", fontsize=11); plt.ylabel("Latitude (°N)", fontsize=11)
    plt.title("Kinematic Base vs Hybrid Prediction vs Ground Truth (TEST Split)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.5); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hybrid_vs_kinematic_tracks.png"), dpi=250)
    plt.close()
    print("Saved: hybrid_vs_kinematic_tracks.png")
    
    # Helper for case study plots
    def plot_cases(raw_data, variant_label, out_name):
        dpes = np.array(raw_data["track_dpes"])
        kin_dpes = np.array(raw_data["kin_track_dpes"])
        diffs = kin_dpes - dpes # Positive means hybrid improved over kinematic
        
        # Pick top 3 successful corrections and top 3 degraded cases
        best_corr_idx = np.argsort(diffs)[::-1][:3]
        worst_corr_idx = np.argsort(diffs)[:3]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for plot_i, idx in enumerate(list(best_corr_idx) + list(worst_corr_idx)):
            ax = axes[plot_i // 3, plot_i % 3]
            sid = raw_data["storm_ids"][idx]; t0 = raw_data["t0s"][idx]
            d = dpes[idx]; kd = kin_dpes[idx]; gain = diffs[idx]
            
            t_lats = [raw_data["true_12"][idx][0], raw_data["true_24"][idx][0], raw_data["true_48"][idx][0]]
            t_lons = [raw_data["true_12"][idx][1], raw_data["true_24"][idx][1], raw_data["true_48"][idx][1]]
            k_lats = [raw_data["kin_12"][idx][0], raw_data["kin_24"][idx][0], raw_data["kin_48"][idx][0]]
            k_lons = [raw_data["kin_12"][idx][1], raw_data["kin_24"][idx][1], raw_data["kin_48"][idx][1]]
            p_lats = [raw_data["pred_12"][idx][0], raw_data["pred_24"][idx][0], raw_data["pred_48"][idx][0]]
            p_lons = [raw_data["pred_12"][idx][1], raw_data["pred_24"][idx][1], raw_data["pred_48"][idx][1]]
            
            ax.plot(t_lons, t_lats, 'b-o', lw=2, markersize=6, label='True IMD')
            ax.plot(k_lons, k_lats, 'k--', lw=1.5, markersize=5, label=f'Kin Base ({kd:.1f} km)')
            ax.plot(p_lons, p_lats, 'r-^', lw=2, markersize=6, label=f'Hybrid ({d:.1f} km)')
            
            status = f"Improvement: +{gain:.1f} km" if gain > 0 else f"Degradation: {gain:.1f} km"
            ax.set_title(f"{variant_label} Case #{plot_i+1}: {sid}\n{status}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
            ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, out_name), dpi=250)
        plt.close()
        print(f"Saved: {out_name}")
        
    # 7. representative_variant_a.png
    plot_cases(raw_a, "Variant A", "representative_variant_a.png")
    
    # 8. representative_variant_b.png
    plot_cases(t_final_b["raw"], "Variant B", "representative_variant_b.png")
    
    # 9. worst_case_hybrid_tracks.png
    worst_idx = np.argsort(raw_a["track_dpes"])[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for plot_i, idx in enumerate(worst_idx):
        ax = axes[plot_i // 3, plot_i % 3]
        sid = raw_a["storm_ids"][idx]; t0 = raw_a["t0s"][idx]; d = raw_a["track_dpes"][idx]
        t_lats = [raw_a["true_12"][idx][0], raw_a["true_24"][idx][0], raw_a["true_48"][idx][0]]
        t_lons = [raw_a["true_12"][idx][1], raw_a["true_24"][idx][1], raw_a["true_48"][idx][1]]
        p_lats = [raw_a["pred_12"][idx][0], raw_a["pred_24"][idx][0], raw_a["pred_48"][idx][0]]
        p_lons = [raw_a["pred_12"][idx][1], raw_a["pred_24"][idx][1], raw_a["pred_48"][idx][1]]
        ax.plot(t_lons, t_lats, 'b-o', lw=2, label='True Track')
        ax.plot(p_lons, p_lats, 'r--^', lw=2, label='Hybrid A Track')
        ax.set_title(f"Worst Case #{plot_i+1}: {sid}\nt0: {t0} | DPE: {d:.1f} km", fontsize=10, fontweight='bold')
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "worst_case_hybrid_tracks.png"), dpi=250)
    plt.close()
    print("Saved: worst_case_hybrid_tracks.png")
    
    # 10. arabian_sea_vs_bob_comparison.png
    # Arabian Sea: lon <= 77.5E, Bay of Bengal: lon > 77.5E
    as_mask = np.array([r[1] <= 77.5 for r in raw_a["kin_12"]])
    bob_mask = ~as_mask
    
    as_dpe_a = np.mean(np.array(raw_a["track_dpes"])[as_mask])
    bob_dpe_a = np.mean(np.array(raw_a["track_dpes"])[bob_mask])
    as_dpe_b = np.mean(np.array(t_final_b["raw"]["track_dpes"])[as_mask])
    bob_dpe_b = np.mean(np.array(t_final_b["raw"]["track_dpes"])[bob_mask])
    
    as_kin_a = np.mean(np.array(raw_a["kin_track_dpes"])[as_mask])
    bob_kin_a = np.mean(np.array(raw_a["kin_track_dpes"])[bob_mask])
    
    plt.figure(figsize=(10, 6))
    basins = ["Arabian Sea (N={})".format(np.sum(as_mask)), "Bay of Bengal (N={})".format(np.sum(bob_mask))]
    x = np.arange(len(basins))
    width = 0.25
    plt.bar(x - width, [as_kin_a, bob_kin_a], width, label='Exact Kinematic Base', color='#ffbb78', edgecolor='black')
    plt.bar(x, [as_dpe_a, bob_dpe_a], width, label='Hybrid Variant A', color='#2ca02c', edgecolor='black')
    plt.bar(x + width, [as_dpe_b, bob_dpe_b], width, label='Hybrid Variant B', color='#d62728', edgecolor='black')
    plt.xticks(x, basins, fontsize=11)
    plt.ylabel("Mean Track DPE (km)", fontsize=11)
    plt.title("Geographic Error Stratification: Arabian Sea vs Bay of Bengal", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.5, axis='y'); plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "arabian_sea_vs_bob_comparison.png"), dpi=250)
    plt.close()
    print("Saved: arabian_sea_vs_bob_comparison.png")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("VAYU-NET PHASE 4B — HYBRID KINEMATIC + SATELLITE RESIDUAL EXPERIMENT")
    print("=" * 80)
    
    device = torch.device("cpu")
    print(f"Device:                 {device}")
    print(f"Random Seed:            {CONFIG['random_seed']}")
    print(f"Batch Size:             {CONFIG['batch_size']}")
    print(f"Max Epochs:             {CONFIG['max_epochs']}")
    print(f"Early Stopping Metric:  VALIDATION MEAN TRACK DPE (Patience = {CONFIG['patience']})")
    
    # 1. Build or Load Hybrid Features Cache
    cache_data = build_hybrid_cache(CONFIG)
    sample_records = cache_data["samples"]
    
    # 2. Train Variant A (Observed-Center Hybrid)
    res_a = train_variant("A", sample_records, CONFIG, device)
    
    # 3. Train Variant B (Satellite-Derived Hybrid)
    res_b = train_variant("B", sample_records, CONFIG, device)
    
    # 4. Summary & Comparison Tables
    t_a = res_a["test_final"]
    t_b = res_b["test_final"]
    
    print("\n" + "=" * 90)
    print("PHASE 4B FINAL TEST SPLIT RESULTS SUMMARY (N=371)")
    print("=" * 90)
    print(f"{'Model / Variant':<32} | {'Mean Track DPE':<16} | {'+12h (Mean/Med/P90)':<22} | {'+24h (Mean/Med/P90)':<22} | {'+48h (Mean/Med/P90)'}")
    print("-" * 115)
    print(f"{'Stationary Persistence':<32} | 309.0 km         | 132.7 / 130.1 / 211.7 km | 263.4 / 266.2 / 398.4 km | 530.8 / 534.8 / 785.1 km")
    print(f"{'Constant Velocity (Exact)':<32} | 189.1 km         |  71.3 /  58.2 / 133.9 km | 150.7 / 124.1 / 285.9 km | 345.1 / 295.0 / 625.6 km")
    print(f"{'Satellite-Derived CV (Noisy)':<32} | 3447.0 km        | 2271.8 / 1217.1 km       | 3512.1 / 1668.1 km       | 4557.0 / 2385.9 km")
    print(f"{'Phase 4A Satellite-Only GRU':<32} | 848.5 km         | 823.0 / 628.0 / 1695.5 km | 832.5 / 658.8 / 1668.6 km | 890.0 / 732.8 / 1730.2 km")
    print("-" * 115)
    print(f"{'Phase 4B Variant A (Obs Base)':<32} | {t_a['mean_track_dpe_km']:6.1f} km         | "
          f"{t_a['h12']['mean_dpe_km']:5.1f} / {t_a['h12']['median_dpe_km']:5.1f} / {t_a['h12']['p90_dpe_km']:5.1f} km | "
          f"{t_a['h24']['mean_dpe_km']:5.1f} / {t_a['h24']['median_dpe_km']:5.1f} / {t_a['h24']['p90_dpe_km']:5.1f} km | "
          f"{t_a['h48']['mean_dpe_km']:5.1f} / {t_a['h48']['median_dpe_km']:5.1f} / {t_a['h48']['p90_dpe_km']:5.1f} km")
    print(f"{'Phase 4B Variant B (Sat Base)':<32} | {t_b['mean_track_dpe_km']:6.1f} km         | "
          f"{t_b['h12']['mean_dpe_km']:5.1f} / {t_b['h12']['median_dpe_km']:5.1f} / {t_b['h12']['p90_dpe_km']:5.1f} km | "
          f"{t_b['h24']['mean_dpe_km']:5.1f} / {t_b['h24']['median_dpe_km']:5.1f} / {t_b['h24']['p90_dpe_km']:5.1f} km | "
          f"{t_b['h48']['mean_dpe_km']:5.1f} / {t_b['h48']['median_dpe_km']:5.1f} / {t_b['h48']['p90_dpe_km']:5.1f} km")
    print("=" * 90)
    
    # 5. Generate Visual Diagnostics
    generate_all_diagnostics(res_a, res_b, CONFIG["figures_dir"])
    
    # 6. Save Comprehensive JSON Results
    results = {
        "metadata": {
            "experiment_name": CONFIG["experiment_name"],
            "model_architecture": "HybridKinematicSatelliteGRU (Phase 3C Spatial Encoder + Motion Context + Multi-Horizon Residual Heads)",
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "device": str(device),
            "random_seed": CONFIG["random_seed"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "weight_decay": CONFIG["weight_decay"],
            "max_epochs": CONFIG["max_epochs"],
            "patience": CONFIG["patience"],
            "model_selection_criterion": "Minimum VALIDATION Mean Track DPE (km)",
            "transferred_layers": "All convolutional layers from Phase 3C DedicatedCenterLocalizationResNet (frozen)",
            "coordinate_system": "Local tangent-plane geodesic displacement (north_km, east_km)",
            "residual_scale_km": RESIDUAL_SCALE_KM
        },
        "variant_a_observed_center": {
            "selected_epoch": res_a["best_epoch"],
            "best_val_track_dpe_km": res_a["best_val_dpe"],
            "training_duration_seconds": res_a["duration"],
            "metrics": {
                "train": res_a["train_final"],
                "validation": res_a["val_final"],
                "test": res_a["test_final"]
            },
            "comparison_with_exact_kinematic_base_test": {
                "kinematic_base_mean_track_dpe_km": res_a["test_final"]["kin_mean_track_dpe_km"],
                "hybrid_mean_track_dpe_km": res_a["test_final"]["mean_track_dpe_km"],
                "improvement_km": res_a["test_final"]["improvement_over_kin_mean_km"],
                "improved": bool(res_a["test_final"]["improvement_over_kin_mean_km"] > 0)
            }
        },
        "variant_b_satellite_derived": {
            "selected_epoch": res_b["best_epoch"],
            "best_val_track_dpe_km": res_b["best_val_dpe"],
            "training_duration_seconds": res_b["duration"],
            "metrics": {
                "train": res_b["train_final"],
                "validation": res_b["val_final"],
                "test": res_b["test_final"]
            },
            "comparison_with_satellite_kinematic_base_test": {
                "satellite_kinematic_base_mean_track_dpe_km": res_b["test_final"]["kin_mean_track_dpe_km"],
                "hybrid_mean_track_dpe_km": res_b["test_final"]["mean_track_dpe_km"],
                "improvement_km": res_b["test_final"]["improvement_over_kin_mean_km"],
                "improved": bool(res_b["test_final"]["improvement_over_kin_mean_km"] > 0)
            }
        },
        "baseline_reproduction_test": {
            "stationary_persistence": {"12h": 132.67, "24h": 263.37, "48h": 530.81, "mean_track": 308.95},
            "constant_velocity_exact": {"12h": 71.28, "24h": 150.74, "48h": 345.15, "mean_track": 189.06},
            "phase4a_satellite_only_gru": {"12h": 823.0, "24h": 832.5, "48h": 890.0, "mean_track": 848.5}
        }
    }
    
    def to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    with open(CONFIG["results_json_path"], "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)
    print(f"\nSaved Phase 4B results JSON to: {CONFIG['results_json_path']}")
    print("=" * 80)
    print("PHASE 4B HYBRID TRAINING & EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
