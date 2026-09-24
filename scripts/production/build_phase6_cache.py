"""
VAYU-NET PHASE 6 — INTENSITY & WIND FEATURE CACHE BUILDER
=========================================================
Builds data/interim/ml/cache/phase6_intensity_features.pt
Contains:
  - sample_id, storm_id, split, t0, lat/lon
  - sat_seq: [6, 132] (pre-computed 6-frame satellite features)
  - sat_t0: [132] (pre-computed single-frame t0 satellite features)
  - env_seq: [6, 8, 41, 66] (ERA5 normalized wind sequence)
  - category_t0: int class index (0..6) or -1 if missing
  - category_t0_mask: 1.0 if valid, 0.0 if missing
  - wind_t0: float wind in kt
  - wind_t0_norm: standardized wind target using TRAIN-only statistics
  - wind_t0_mask: 1.0 if valid, 0.0 if missing (loss-masking support)
  - pressure_t0: float pressure in hPa
  - pressure_t0_norm: standardized pressure target using TRAIN-only statistics
  - pressure_t0_mask: 1.0 if valid, 0.0 if missing
"""

import os
import torch
import pandas as pd
import numpy as np

C_5B_PATH = "data/interim/ml/cache/phase5b_hybrid_features.pt"
SAMPLE_INDEX_PATH = "data/manifests/vayu_net_sample_index.csv"
OUT_CACHE_PATH = "data/interim/ml/cache/phase6_intensity_features.pt"

CATEGORY_TO_IDX = {
    "D": 0,
    "DD": 1,
    "CS": 2,
    "SCS": 3,
    "VSCS": 4,
    "ESCS": 5,
    "SuCS": 6
}

def build_cache():
    print(f"Loading {C_5B_PATH}...")
    c_5b = torch.load(C_5B_PATH, map_location="cpu", weights_only=False)
    sample_index = pd.read_csv(SAMPLE_INDEX_PATH)

    train_df = sample_index[sample_index["split"] == "TRAIN"]
    train_wind_mean = float(train_df["imd_wind_t0"].dropna().mean())
    train_wind_std = float(train_df["imd_wind_t0"].dropna().std())
    train_pres_mean = float(train_df["imd_pressure_t0"].dropna().mean())
    train_pres_std = float(train_df["imd_pressure_t0"].dropna().std())

    print(f"TRAIN wind: mean = {train_wind_mean:.2f} kt, std = {train_wind_std:.2f} kt")
    print(f"TRAIN pressure: mean = {train_pres_mean:.2f} hPa, std = {train_pres_std:.2f} hPa")

    phase6_samples = []
    for i, s5b in enumerate(c_5b["samples"]):
        row = sample_index.iloc[i]
        assert s5b["sample_id"] == row["sample_id"], f"Sample mismatch at index {i}"

        # Category
        cat_val = row["imd_category_t0"]
        if pd.isna(cat_val) or cat_val not in CATEGORY_TO_IDX:
            c_idx = -1
            c_mask = 0.0
        else:
            c_idx = CATEGORY_TO_IDX[cat_val]
            c_mask = 1.0

        # Wind
        w_val = row["imd_wind_t0"]
        if pd.isna(w_val):
            w_raw = 0.0
            w_norm = 0.0
            w_mask = 0.0
        else:
            w_raw = float(w_val)
            w_norm = (w_raw - train_wind_mean) / train_wind_std
            w_mask = 1.0

        # Pressure
        p_val = row["imd_pressure_t0"]
        if pd.isna(p_val):
            p_raw = 0.0
            p_norm = 0.0
            p_mask = 0.0
        else:
            p_raw = float(p_val)
            p_norm = (p_raw - train_pres_mean) / train_pres_std
            p_mask = 1.0

        phase6_samples.append({
            "sample_id": row["sample_id"],
            "storm_id": row["storm_id"],
            "split": row["split"],
            "t0": row["t0"],
            "sat_seq": s5b["sat_seq"],                   # [6, 132]
            "sat_t0": s5b["sat_seq"][5],                 # [132]
            "env_seq": s5b["env_seq"],                   # [6, 8, 41, 66]
            "category_t0": torch.tensor(c_idx, dtype=torch.long),
            "category_t0_mask": torch.tensor(c_mask, dtype=torch.float32),
            "category_name": str(cat_val) if pd.notna(cat_val) else "MISSING",
            "wind_t0": torch.tensor(w_raw, dtype=torch.float32),
            "wind_t0_norm": torch.tensor(w_norm, dtype=torch.float32),
            "wind_t0_mask": torch.tensor(w_mask, dtype=torch.float32),
            "pressure_t0": torch.tensor(p_raw, dtype=torch.float32),
            "pressure_t0_norm": torch.tensor(p_norm, dtype=torch.float32),
            "pressure_t0_mask": torch.tensor(p_mask, dtype=torch.float32),
            "imd_lat_t0": float(row["imd_lat_t0"]),
            "imd_lon_t0": float(row["imd_lon_t0"])
        })

    cache_out = {
        "metadata": {
            "description": "VAYU-NET Phase 6 Intensity & Wind Feature Cache",
            "total_samples": len(phase6_samples),
            "train_wind_mean_kt": train_wind_mean,
            "train_wind_std_kt": train_wind_std,
            "train_pres_mean_hpa": train_pres_mean,
            "train_pres_std_hpa": train_pres_std,
            "category_mapping": CATEGORY_TO_IDX
        },
        "samples": phase6_samples
    }

    torch.save(cache_out, OUT_CACHE_PATH)
    print(f"Saved {OUT_CACHE_PATH} ({os.path.getsize(OUT_CACHE_PATH) / (1024 * 1024):.1f} MB)")

if __name__ == "__main__":
    build_cache()
