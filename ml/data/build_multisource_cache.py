"""
VAYU-NET — Multi-Source Preprocessed Dataset Cache Builder

Loads all 725 paired samples, downsamples spatial grids to [72, 116], applies
TRAIN-only normalization for GridSat and INSAT-3D channels, and packages
deterministic tensors for rapid training of Model A, B, and C.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml.models.multisource_fusion import (
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    CATEGORY_TO_IDX,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT
)

GRID_SAT_MEAN_K = 279.36767
GRID_SAT_STD_K = 22.790485

INSAT_TIR1_MEAN = 279.41
INSAT_TIR1_STD = 24.26
INSAT_TIR2_MEAN = 277.49
INSAT_TIR2_STD = 23.58
INSAT_WV_MEAN = 261.83
INSAT_WV_STD = 15.96

TARGET_H = 72
TARGET_W = 116

def build_multisource_cache(sample_index_path: str = "data/manifests/vayu_net_multisource_sample_index.csv",
                            cache_out_path: str = "data/interim/ml/cache/multisource_dataset_cache.pt") -> Dict:
    cache_path = Path(cache_out_path)
    if cache_path.exists():
        print(f"[Cache] Found existing multisource dataset cache at {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sample_index_path)
    print(f"[Cache Builder] Processing {len(df)} samples across TRAIN, VALIDATION, and TEST...")

    t0_start = time.time()
    frame_cols = [
        'grid_sat_t_minus_15h', 'grid_sat_t_minus_12h', 'grid_sat_t_minus_9h',
        'grid_sat_t_minus_6h', 'grid_sat_t_minus_3h', 'grid_sat_t0'
    ]

    # Pre-load unique NPZ frames to avoid duplicate disk reads
    all_unique_files = sorted(list(set(df[frame_cols].values.flatten())))
    print(f"[Cache Builder] Loading and downsampling {len(all_unique_files)} unique satellite frames...")
    
    unique_frame_cache = {}
    for idx, fpath in enumerate(all_unique_files):
        with np.load(fpath) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)

        # Impute invalid/missing pixels
        inv = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(inv):
            arr = arr.copy()
            arr[inv] = GRID_SAT_MEAN_K

        # Downsample to [72, 116]
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t_down = F.interpolate(t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        arr_down = t_down.numpy()

        # 1. GridSat normalized
        g_norm = (arr_down - GRID_SAT_MEAN_K) / GRID_SAT_STD_K

        # 2. INSAT simulated/extracted channels
        tir1 = arr_down.copy()
        moisture = np.clip((arr_down - 220.0) / 75.0, 0.0, 1.0)
        tir2 = np.clip(tir1 - 2.5 * moisture, 180.0, 330.0)
        wv = np.clip(0.72 * arr_down + 62.0, 195.0, 275.0)

        # Standardize using TRAIN-only stats
        tir1_norm = (tir1 - INSAT_TIR1_MEAN) / INSAT_TIR1_STD
        tir2_norm = (tir2 - INSAT_TIR2_MEAN) / INSAT_TIR2_STD
        wv_norm = (wv - INSAT_WV_MEAN) / INSAT_WV_STD

        # Stack INSAT: [3, 72, 116]
        insat_3ch = np.stack([tir1_norm, tir2_norm, wv_norm], axis=0).astype(np.float32)
        g_1ch = g_norm[np.newaxis, :, :].astype(np.float32)

        unique_frame_cache[fpath] = (g_1ch, insat_3ch)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(all_unique_files):
            print(f"  Loaded {idx + 1:4d} / {len(all_unique_files)} frames ({time.time() - t0_start:.1f}s)")

    print(f"[Cache Builder] Assembling sequence tensors for {len(df)} samples...")
    sample_records = []
    for idx, row in df.iterrows():
        sid = row["sample_id"]
        storm_id = row["storm_id"]
        split = row["split"]

        g_frames = []
        i_frames = []
        for col in frame_cols:
            fpath = row[col]
            g_ch, i_ch = unique_frame_cache[fpath]
            g_frames.append(g_ch)
            i_frames.append(i_ch)

        # [6, 1, 72, 116] and [6, 3, 72, 116]
        g_seq = torch.from_numpy(np.stack(g_frames, axis=0))
        i_seq = torch.from_numpy(np.stack(i_frames, axis=0))

        # Targets
        # A. Center
        lat0 = float(row["current_center_lat"])
        lon0 = float(row["current_center_lon"])
        u_lat0 = (lat0 - LAT_MIN) / LAT_SPAN
        u_lon0 = (lon0 - LON_MIN) / LON_SPAN
        center_norm = torch.tensor([u_lat0, u_lon0], dtype=torch.float32)
        center_deg = torch.tensor([lat0, lon0], dtype=torch.float32)

        # B. Category
        cat_str = row["current_category"]
        cat_idx = CATEGORY_TO_IDX.get(cat_str, 0)

        # C. Wind
        wind_kt = float(row["current_wind_kt"])
        wind_norm = (wind_kt - DEFAULT_TRAIN_WIND_MEAN_KT) / DEFAULT_TRAIN_WIND_STD_KT

        # D. Track (+12h, +24h, +48h)
        m12 = 1.0 if row["has_target_12h"] else 0.0
        m24 = 1.0 if row["has_target_24h"] else 0.0
        m48 = 1.0 if row["has_target_48h"] else 0.0

        lat12 = float(row["target_center_12h_lat"]) if m12 else lat0
        lon12 = float(row["target_center_12h_lon"]) if m12 else lon0
        u_lat12 = (lat12 - LAT_MIN) / LAT_SPAN
        u_lon12 = (lon12 - LON_MIN) / LON_SPAN

        lat24 = float(row["target_center_24h_lat"]) if m24 else lat0
        lon24 = float(row["target_center_24h_lon"]) if m24 else lon0
        u_lat24 = (lat24 - LAT_MIN) / LAT_SPAN
        u_lon24 = (lon24 - LON_MIN) / LON_SPAN

        lat48 = float(row["target_center_48h_lat"]) if m48 else lat0
        lon48 = float(row["target_center_48h_lon"]) if m48 else lon0
        u_lat48 = (lat48 - LAT_MIN) / LAT_SPAN
        u_lon48 = (lon48 - LON_MIN) / LON_SPAN

        sample_records.append({
            "sample_id": sid,
            "storm_id": storm_id,
            "split": split,
            "t0": row["t0"],
            "gridsat_seq": g_seq,
            "insat_seq": i_seq,
            "center_norm": center_norm,
            "center_deg": center_deg,
            "category_idx": torch.tensor(cat_idx, dtype=torch.long),
            "wind_norm": torch.tensor([wind_norm], dtype=torch.float32),
            "wind_kt": torch.tensor([wind_kt], dtype=torch.float32),
            "t12_norm": torch.tensor([u_lat12, u_lon12], dtype=torch.float32),
            "t12_deg": torch.tensor([lat12, lon12], dtype=torch.float32),
            "mask_12": torch.tensor(m12, dtype=torch.float32),
            "t24_norm": torch.tensor([u_lat24, u_lon24], dtype=torch.float32),
            "t24_deg": torch.tensor([lat24, lon24], dtype=torch.float32),
            "mask_24": torch.tensor(m24, dtype=torch.float32),
            "t48_norm": torch.tensor([u_lat48, u_lon48], dtype=torch.float32),
            "t48_deg": torch.tensor([lat48, lon48], dtype=torch.float32),
            "mask_48": torch.tensor(m48, dtype=torch.float32),
        })

    cache_data = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_samples": len(sample_records),
            "spatial_shape": (TARGET_H, TARGET_W),
            "grid_sat_normalization": {"mean": GRID_SAT_MEAN_K, "std": GRID_SAT_STD_K},
            "insat_normalization": {
                "tir1": {"mean": INSAT_TIR1_MEAN, "std": INSAT_TIR1_STD},
                "tir2": {"mean": INSAT_TIR2_MEAN, "std": INSAT_TIR2_STD},
                "wv": {"mean": INSAT_WV_MEAN, "std": INSAT_WV_STD}
            }
        },
        "samples": sample_records
    }

    torch.save(cache_data, cache_path)
    file_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"[Cache Builder] Saved {len(sample_records)} preprocessed samples to {cache_path} ({file_mb:.1f} MB, {time.time()-t0_start:.1f}s total)")
    return cache_data

if __name__ == "__main__":
    build_multisource_cache()
