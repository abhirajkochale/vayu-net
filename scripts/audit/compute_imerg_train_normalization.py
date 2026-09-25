"""
VAYU-NET: Compute NASA GPM IMERG Training Normalization Statistics.
Strictly calculates statistics over TRAIN storms only (zero validation/test leakage).
"""

import os
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from ml.data.process_imerg import IMERGProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ComputeIMERGStats")

def main():
    t0 = time.time()
    seq_path = Path("data/manifests/gridsat_imerg_sequence_pairing_manifest.csv")
    seq_df = pd.read_csv(seq_path)
    
    train_df = seq_df[seq_df["split"] == "TRAIN"].copy()
    logger.info(f"Loaded {len(train_df)} TRAIN sequences across {train_df['storm_id'].nunique()} unique storms.")
    
    imerg_cols = [
        "imerg_t_minus_15h_granule", "imerg_t_minus_12h_granule", "imerg_t_minus_9h_granule",
        "imerg_t_minus_6h_granule", "imerg_t_minus_3h_granule", "imerg_t0_granule"
    ]
    train_granules = sorted(list(set(train_df[imerg_cols].values.flatten())))
    logger.info(f"Found {len(train_granules)} unique training IMERG granules.")
    
    processor = IMERGProcessor(target_resolution="model")
    raw_dir = Path("data/raw/imerg")
    
    total_pixels = 0
    nan_pixels = 0
    sum_val = 0.0
    sum_sq_val = 0.0
    sum_log1p = 0.0
    sum_sq_log1p = 0.0
    min_val = float("inf")
    max_val = float("-inf")
    sample_vals = []
    
    for idx, g in enumerate(train_granules):
        res = processor.process_granule(raw_dir / g)
        arr = res["precipitation"]  # Shape (72, 116)
        
        nans = np.isnan(arr)
        n_nans = int(nans.sum())
        nan_pixels += n_nans
        total_pixels += arr.size
        
        valid = arr[~nans]
        if len(valid) > 0:
            v_min = float(valid.min())
            v_max = float(valid.max())
            if v_min < min_val: min_val = v_min
            if v_max > max_val: max_val = v_max
            
            sum_val += float(np.sum(valid))
            sum_sq_val += float(np.sum(valid ** 2))
            
            l1p = np.log1p(valid)
            sum_log1p += float(np.sum(l1p))
            sum_sq_log1p += float(np.sum(l1p ** 2))
            
            step = max(1, len(valid) // 100)
            sample_vals.extend(valid[::step].tolist())
            
        if (idx + 1) % 250 == 0 or (idx + 1) == len(train_granules):
            logger.info(f"Processed {idx + 1:4d} / {len(train_granules)} granules ({time.time() - t0:.1f}s)")
            
    valid_pixels = total_pixels - nan_pixels
    mean_val = sum_val / valid_pixels
    var_val = (sum_sq_val / valid_pixels) - (mean_val ** 2)
    std_val = float(np.sqrt(max(0.0, var_val)))
    
    mean_log1p = sum_log1p / valid_pixels
    var_log1p = (sum_sq_log1p / valid_pixels) - (mean_log1p ** 2)
    std_log1p = float(np.sqrt(max(0.0, var_log1p)))
    
    sample_arr = np.array(sample_vals, dtype=np.float32)
    percentiles = {
        "p10": float(np.percentile(sample_arr, 10)),
        "p25": float(np.percentile(sample_arr, 25)),
        "p50": float(np.percentile(sample_arr, 50)),
        "p75": float(np.percentile(sample_arr, 75)),
        "p90": float(np.percentile(sample_arr, 90)),
        "p95": float(np.percentile(sample_arr, 95)),
        "p99": float(np.percentile(sample_arr, 99)),
        "p99_9": float(np.percentile(sample_arr, 99.9))
    }
    
    stats = {
        "dataset": "NASA GPM IMERG Final Run V07B",
        "split": "TRAIN_ONLY",
        "num_train_sequences": len(train_df),
        "num_unique_train_granules": len(train_granules),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nan_pixels": nan_pixels,
        "nan_fraction": float(nan_pixels / total_pixels),
        "spatial_grid": [72, 116],
        "domain_bounds": {
            "lat_min": -5.0, "lat_max": 35.0,
            "lon_min": 40.0, "lon_max": 105.0
        },
        "raw_units": "mm/hr",
        "raw_stats": {
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(min_val),
            "max": float(max_val),
            "percentiles": percentiles
        },
        "log1p_stats": {
            "mean": float(mean_log1p),
            "std": float(std_log1p),
            "min": float(np.log1p(min_val)),
            "max": float(np.log1p(max_val))
        },
        "baseline_transform": "standardize_raw",
        "transform_rationale": "Baseline uses z-score standardization of physical mm/hr (x - mean) / std. NaNs (fill values) are imputed with train mean. log1p statistics are also preserved for non-linear precipitation scaling ablations.",
        "elapsed_seconds": round(time.time() - t0, 2)
    }
    
    out_path = Path("data/interim/ml/imerg_train_normalization_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    logger.info(f"Saved IMERG train normalization statistics to {out_path}")
    print("\n--- IMERG TRAIN NORMALIZATION STATISTICS ---")
    print(f"Mean (mm/hr):       {mean_val:.4f}")
    print(f"Std (mm/hr):        {std_val:.4f}")
    print(f"Min / Max:          {min_val:.4f} / {max_val:.4f}")
    print(f"NaN fraction:       {stats['nan_fraction']:.6f}")
    print(f"log1p Mean / Std:   {mean_log1p:.4f} / {std_log1p:.4f}")
    print(f"Elapsed:            {stats['elapsed_seconds']}s")

if __name__ == "__main__":
    main()