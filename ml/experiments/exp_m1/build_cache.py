"""
VAYU-NET: EXP-M1 High-Performance Tensor Cache Builder.
Uses direct HDF5 hyperslab spatial slicing for NIO basin (lat [-5, 35], lon [40, 105]),
pre-standardizing all 1,319 sequences to [6, 72, 116] tensors.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.process_imerg import TARGET_MODEL_H, TARGET_MODEL_W

CACHE_OUT = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
MANIFEST_PATH = REPO_ROOT / "data/manifests/gridsat_imerg_multimodal_dataset_manifest.csv"
SAMPLE_INDEX_PATH = REPO_ROOT / "data/manifests/vayu_net_sample_index.csv"
GRIDSAT_NORM_PATH = REPO_ROOT / "data/interim/ml/train_normalization_stats.json"
IMERG_NORM_PATH = REPO_ROOT / "data/interim/ml/imerg_train_normalization_stats.json"
IMERG_RAW_DIR = REPO_ROOT / "data/raw/imerg"

# Fixed NIO HDF5 slice indices
# lat in [-5.0, 35.0] -> [850, 1250] (400 points)
# lon in [40.0, 105.0] -> [2200, 2850] (650 points)
LAT_SLICE = slice(850, 1250)
LON_SLICE = slice(2200, 2850)

CATEGORY_TO_IDX = {
    "D": 0, "DD": 1, "CS": 2, "SCS": 3, "VSCS": 4, "ESCS": 5, "SuCS": 6
}


def load_norm_stats():
    with open(GRIDSAT_NORM_PATH, "r") as f:
        g_stats = json.load(f)
    g_mean = float(g_stats["mean_kelvin"])
    g_std = float(g_stats["std_kelvin"])

    with open(IMERG_NORM_PATH, "r") as f:
        i_stats = json.load(f)
    i_mean = float(i_stats["raw_stats"]["mean"])
    i_std = float(i_stats["raw_stats"]["std"])
    return g_mean, g_std, i_mean, i_std


def process_gridsat_frame(fpath: str, g_mean: float, g_std: float) -> np.ndarray:
    with np.load(fpath) as npz:
        arr = npz["irwin_cdr"].astype(np.float32)
    invalid = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
    if np.any(invalid):
        arr = arr.copy()
        arr[invalid] = g_mean
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t_down = F.interpolate(t, size=(TARGET_MODEL_H, TARGET_MODEL_W), mode="bilinear", align_corners=False).squeeze()
    return ((t_down.numpy() - g_mean) / g_std).astype(np.float32)


def process_imerg_granule(filename: str, i_mean: float, i_std: float) -> np.ndarray:
    h5_path = IMERG_RAW_DIR / filename
    with h5py.File(h5_path, "r") as h5:
        # Direct spatial slice [1, lon, lat]
        raw = h5["Grid"]["precipitation"][0, LON_SLICE, LAT_SLICE]
    
    # Transpose [lon, lat] -> [lat, lon]
    sub = np.transpose(raw, (1, 0)).astype(np.float32)
    sub = np.where(sub >= 0.0, sub, np.nan)

    # Downsample with NaN preservation
    nan_mask = np.isnan(sub)
    filled = np.where(nan_mask, 0.0, sub)
    t_in = torch.from_numpy(filled).unsqueeze(0).unsqueeze(0).float()
    t_out = F.interpolate(t_in, size=(TARGET_MODEL_H, TARGET_MODEL_W), mode="bilinear", align_corners=False).squeeze()
    arr = t_out.numpy()

    m_in = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m_out = F.interpolate(m_in, size=(TARGET_MODEL_H, TARGET_MODEL_W), mode="bilinear", align_corners=False).squeeze()
    resampled_nan = m_out.numpy() > 0.5
    arr[resampled_nan] = i_mean  # Impute NaNs with train mean

    # Standardize
    arr = (arr - i_mean) / i_std
    return arr.astype(np.float32)


def main():
    print("=" * 65)
    print("VAYU-NET: BUILDING HIGH-PERFORMANCE EXP-M1 TENSOR CACHE")
    print("=" * 65)
    t0 = time.time()

    g_mean, g_std, i_mean, i_std = load_norm_stats()
    print(f"Loaded norm stats: GridSat mean={g_mean:.2f}, std={g_std:.2f} | IMERG mean={i_mean:.4f}, std={i_std:.4f}")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    idx_df = pd.read_csv(SAMPLE_INDEX_PATH)

    target_cols = [
        "sample_id",
        "imd_lat_t0", "imd_lon_t0", "imd_wind_t0", "imd_pressure_t0", "imd_category_t0",
        "imd_lat_12h", "imd_lon_12h", "wind_12h", "pressure_12h", "category_12h",
        "imd_lat_24h", "imd_lon_24h", "wind_24h", "pressure_24h", "category_24h",
        "imd_lat_48h", "imd_lon_48h", "wind_48h", "pressure_48h", "category_48h"
    ]
    df = pd.merge(manifest_df, idx_df[target_cols], on="sample_id", how="inner")
    print(f"Total sequences: {len(df)}")

    # 1. Collect unique files
    unique_gridsat = sorted(list(set(
        p for ref in df["gridsat_sequence_reference"] for p in ref.split("|")
    )))
    unique_imerg = sorted(list(set(
        g for ref in df["imerg_sequence_reference"] for g in ref.split("|")
    )))
    print(f"Unique files: GridSat={len(unique_gridsat)}, IMERG={len(unique_imerg)}")

    # 2. Parallel pre-process unique GridSat frames
    print("Pre-processing unique GridSat frames...")
    t_gs = time.time()
    gridsat_cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda p: (p, process_gridsat_frame(p, g_mean, g_std)), unique_gridsat))
    gridsat_cache = dict(results)
    print(f"  Processed {len(gridsat_cache)} GridSat frames in {time.time() - t_gs:.1f}s")

    # 3. Parallel pre-process unique IMERG granules
    print("Pre-processing unique IMERG granules (direct HDF5 slices)...")
    t_im = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda g: (g, process_imerg_granule(g, i_mean, i_std)), unique_imerg))
    imerg_cache = dict(results)
    print(f"  Processed {len(imerg_cache)} IMERG granules in {time.time() - t_im:.1f}s")

    # 4. Assemble samples
    print("Assembling 1,319 sequence samples...")
    samples = []
    for _, row in df.iterrows():
        # Gridsat [6, 72, 116]
        gs_paths = row["gridsat_sequence_reference"].split("|")
        gs_frames = [gridsat_cache[p] for p in gs_paths]
        g_tensor = torch.from_numpy(np.stack(gs_frames, axis=0)).float()

        # IMERG [6, 72, 116]
        im_granules = row["imerg_sequence_reference"].split("|")
        im_frames = [imerg_cache[g] for g in im_granules]
        i_tensor = torch.from_numpy(np.stack(im_frames, axis=0)).float()

        # Targets
        def parse_target(lat_val, lon_val, wind_val, pres_val, cat_val):
            c_lat = float(lat_val)
            c_lon = float(lon_val)
            center_t = torch.tensor([c_lat, c_lon], dtype=torch.float32)
            w_val = float(wind_val) if not pd.isna(wind_val) else 0.0
            w_mask = 1.0 if not pd.isna(wind_val) else 0.0
            p_val = float(pres_val) if not pd.isna(pres_val) else 0.0
            p_mask = 1.0 if not pd.isna(pres_val) else 0.0
            c_idx = CATEGORY_TO_IDX[cat_val] if (not pd.isna(cat_val) and cat_val in CATEGORY_TO_IDX) else -1
            c_mask = 1.0 if (not pd.isna(cat_val) and cat_val in CATEGORY_TO_IDX) else 0.0
            return (center_t,
                    torch.tensor(w_val, dtype=torch.float32), torch.tensor(w_mask, dtype=torch.float32),
                    torch.tensor(p_val, dtype=torch.float32), torch.tensor(p_mask, dtype=torch.float32),
                    torch.tensor(c_idx, dtype=torch.long), torch.tensor(c_mask, dtype=torch.float32))

        c0, w0, wm0, p0, pm0, cat0, catm0 = parse_target(
            row["imd_lat_t0"], row["imd_lon_t0"], row["imd_wind_t0"], row["imd_pressure_t0"], row["imd_category_t0"]
        )
        c12, w12, wm12, p12, pm12, cat12, catm12 = parse_target(
            row["imd_lat_12h"], row["imd_lon_12h"], row["wind_12h"], row["pressure_12h"], row["category_12h"]
        )
        c24, w24, wm24, p24, pm24, cat24, catm24 = parse_target(
            row["imd_lat_24h"], row["imd_lon_24h"], row["wind_24h"], row["pressure_24h"], row["category_24h"]
        )
        c48, w48, wm48, p48, pm48, cat48, catm48 = parse_target(
            row["imd_lat_48h"], row["imd_lon_48h"], row["wind_48h"], row["pressure_48h"], row["category_48h"]
        )

        samples.append({
            "sample_id": row["sample_id"],
            "storm_id": row["storm_id"],
            "storm_name": row["storm_name"],
            "split": row["split"],
            "t0_utc": row["t0_utc"],
            "gridsat": g_tensor,  # [6, 72, 116]
            "imerg": i_tensor,    # [6, 72, 116]

            "center_t0": c0,
            "wind_t0": w0, "wind_t0_mask": wm0,
            "pressure_t0": p0, "pressure_t0_mask": pm0,
            "category_t0": cat0, "category_t0_mask": catm0,

            "center_12h": c12,
            "wind_12h": w12, "wind_12h_mask": wm12,
            "pressure_12h": p12, "pressure_12h_mask": pm12,
            "category_12h": cat12, "category_12h_mask": catm12,

            "center_24h": c24,
            "wind_24h": w24, "wind_24h_mask": wm24,
            "pressure_24h": p24, "pressure_24h_mask": pm24,
            "category_24h": cat24, "category_24h_mask": catm24,

            "center_48h": c48,
            "wind_48h": w48, "wind_48h_mask": wm48,
            "pressure_48h": p48, "pressure_48h_mask": pm48,
            "category_48h": cat48, "category_48h_mask": catm48,
        })

    payload = {
        "metadata": {
            "experiment": "EXP-M1",
            "num_samples": len(samples),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "splits": {
                "TRAIN": sum(1 for s in samples if s["split"] == "TRAIN"),
                "VALIDATION": sum(1 for s in samples if s["split"] == "VALIDATION"),
                "TEST": sum(1 for s in samples if s["split"] == "TEST")
            }
        },
        "samples": samples
    }

    CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE_OUT)
    size_mb = CACHE_OUT.stat().st_size / (1024 ** 2)
    print(f"Successfully saved tensor cache to {CACHE_OUT} ({size_mb:.2f} MB)")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()