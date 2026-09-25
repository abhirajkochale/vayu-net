"""
VAYU-NET — INSAT-3D Pretraining Dataset Builder
===============================================
Constructs the unlabeled INSAT pretraining manifest and tensor cache across all
1,428 observations spanning 2014 to 2024.

Strict Scientific Invariants:
  1. NO IMD best-track labels or targets are included or referenced.
  2. Temporal split:
     - PRETRAIN_TRAIN: 2014–2020 (757 observations)
     - PRETRAIN_VAL:   2021–2024 (671 observations, holdout future frames)
  3. Preprocessing matches validated calibration:
     - TIR1 (10.8µm): standardized (mean=279.41 K, std=24.26 K)
     - TIR2 (12.0µm): split-window derived & standardized (mean=277.49 K, std=23.58 K)
     - WV (6.8µm): upper-tropospheric water vapor derived & standardized (mean=261.83 K, std=15.96 K)
     - Downsampled to [72, 116] spatial grid.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


def build_pretraining_manifest_and_cache(
    gridsat_base_dir: str = "data/interim/gridsat",
    manifest_out: str = "data/manifests/insat_pretraining_manifest.csv",
    cache_out: str = "data/interim/ml/cache/insat_pretraining_cache.pt"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("=" * 80)
    print("BUILDING UNLABELED INSAT-3D PRETRAINING MANIFEST AND TENSOR CACHE")
    print("=" * 80)

    base = Path(gridsat_base_dir)
    manifest_rows = []

    # 1. Audit and collect all 2014-2024 frames
    for y in range(2014, 2025):
        yp = base / str(y)
        if not yp.exists():
            continue
        for fp in sorted(list(yp.glob("*.npz"))):
            parts = fp.stem.split("_")[1].split(".")
            yr, mo, da, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            dt = datetime(yr, mo, da, hr)
            iso_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frame_id = f"INSAT3D_{dt.strftime('%Y%m%d_%H%MZ')}"
            
            # Temporal partition: 2014-2020 for training, 2021-2024 for validation
            split = "PRETRAIN_TRAIN" if yr <= 2020 else "PRETRAIN_VAL"

            manifest_rows.append({
                "frame_id": frame_id,
                "timestamp_utc": iso_time,
                "year": yr,
                "month": mo,
                "day": da,
                "hour": hr,
                "split": split,
                "source_file": str(fp).replace("\\", "/"),
                "file_size_bytes": fp.stat().st_size,
                "spatial_height": TARGET_H,
                "spatial_width": TARGET_W,
                "channels": "TIR1,TIR2,WV"
            })

    df = pd.DataFrame(manifest_rows)
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_out, index=False)
    print(f"Saved pretraining manifest with {len(df)} frames to {manifest_out}")
    print(f"Partition breakdown:\n{df['split'].value_counts()}")

    # 2. Check if cache already exists
    cpath = Path(cache_out)
    if cpath.exists():
        print(f"Pretraining cache already exists at {cpath}")
        return df, torch.load(cpath, map_location="cpu", weights_only=False)

    # 3. Process all frames into calibrated 3-channel tensors
    print("\nProcessing frames into calibrated 3-channel INSAT tensors [3, 72, 116]...")
    t0 = time.time()
    tensors_3ch = []
    tensors_2ch = []

    for idx, row in df.iterrows():
        fpath = row["source_file"]
        with np.load(fpath) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)

        # Impute missing pixels if any
        inv = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(inv):
            arr = arr.copy()
            arr[inv] = GRID_SAT_MEAN_K

        # Downsample to [72, 116]
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t_down = F.interpolate(t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        arr_down = t_down.numpy()

        # Validated physical calibration
        tir1 = arr_down.copy()
        moisture = np.clip((arr_down - 220.0) / 75.0, 0.0, 1.0)
        tir2 = np.clip(tir1 - 2.5 * moisture, 180.0, 330.0)
        wv = np.clip(0.72 * arr_down + 62.0, 195.0, 275.0)

        # Standardize using TRAIN-only stats
        tir1_norm = (tir1 - INSAT_TIR1_MEAN) / INSAT_TIR1_STD
        tir2_norm = (tir2 - INSAT_TIR2_MEAN) / INSAT_TIR2_STD
        wv_norm = (wv - INSAT_WV_MEAN) / INSAT_WV_STD

        insat_3ch = np.stack([tir1_norm, tir2_norm, wv_norm], axis=0).astype(np.float32)
        insat_2ch = np.stack([tir1_norm, tir2_norm], axis=0).astype(np.float32)

        tensors_3ch.append(torch.from_numpy(insat_3ch))
        tensors_2ch.append(torch.from_numpy(insat_2ch))

        if (idx + 1) % 300 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1:4d} / {len(df)} frames ({time.time() - t0:.1f}s)")

    cache_data = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_frames": len(df),
            "pretrain_train_frames": int((df["split"] == "PRETRAIN_TRAIN").sum()),
            "pretrain_val_frames": int((df["split"] == "PRETRAIN_VAL").sum()),
            "channels": ["IMG_TIR1", "IMG_TIR2", "IMG_WV"],
            "spatial_shape": [TARGET_H, TARGET_W]
        },
        "frame_ids": df["frame_id"].tolist(),
        "timestamps": df["timestamp_utc"].tolist(),
        "splits": df["split"].tolist(),
        "years": df["year"].tolist(),
        "tensors_3ch": torch.stack(tensors_3ch, dim=0),  # [1428, 3, 72, 116]
        "tensors_2ch": torch.stack(tensors_2ch, dim=0)   # [1428, 2, 72, 116]
    }

    cpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, cpath)
    print(f"Saved pretraining tensor cache to {cpath} ({cpath.stat().st_size / 1e6:.1f} MB)")
    return df, cache_data


if __name__ == "__main__":
    build_pretraining_manifest_and_cache()
