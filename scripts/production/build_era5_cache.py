"""
VAYU-NET PHASE 5A — ERA5 ENVIRONMENTAL FEATURE CACHE BUILDER
=============================================================
Precomputes and packages normalized 6-frame ERA5 environmental tensors
for all VAYU-NET candidate samples into:
  data/interim/ml/cache/era5_environment_features.pt

Includes explicit provenance references to raw source ERA5 NPZ files.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import torch
import numpy as np
import pandas as pd
from ml.data.era5_loader import (
    RAW_ERA5_DIR,
    SAMPLE_INDEX_PATH,
    NORM_STATS_PATH,
    FRAME_COLS,
    parse_era5_filename_from_gridsat,
    load_normalization_stats
)

CACHE_OUT_PATH = "data/interim/ml/cache/era5_environment_features.pt"

def build_cache(
    sample_index_path=SAMPLE_INDEX_PATH,
    era5_dir=RAW_ERA5_DIR,
    stats_path=NORM_STATS_PATH,
    out_path=CACHE_OUT_PATH
):
    print("=" * 80)
    print("BUILDING ERA5 ENVIRONMENTAL FEATURE CACHE")
    print("=" * 80)
    
    df = pd.read_csv(sample_index_path)
    print(f"Total candidate samples to process: {len(df)}")
    
    norm_stats = load_normalization_stats(stats_path)
    if not norm_stats:
        raise RuntimeError(f"Missing normalization stats at {stats_path}")
        
    means = torch.tensor(norm_stats['means_list'], dtype=torch.float32).view(1, 8, 1, 1)
    stds = torch.tensor(norm_stats['stds_list'], dtype=torch.float32).view(1, 8, 1, 1)
    
    samples_by_split = {'TRAIN': [], 'VALIDATION': [], 'TEST': []}
    available_era5_count = 0
    missing_era5_count = 0

    for idx, row in df.iterrows():
        sid = row['sample_id']
        storm = row['storm_id']
        split = row['split']
        t0 = row['t0']
        
        frames = []
        source_files = []
        
        for col in FRAME_COLS:
            fname = parse_era5_filename_from_gridsat(str(row[col]))
            fpath = os.path.join(era5_dir, fname)
            source_files.append(fpath)
            if os.path.exists(fpath):
                arr = np.load(fpath)['env']
                available_era5_count += 1
            else:
                arr = np.zeros((8, 41, 66), dtype=np.float32)
                missing_era5_count += 1
            frames.append(arr)
            
        seq_tensor = torch.from_numpy(np.stack(frames, axis=0)) # [6, 8, 41, 66]
        seq_norm = (seq_tensor - means) / stds
        
        t12 = [row['imd_lat_12h'], row['imd_lon_12h']]
        t24 = [row['imd_lat_24h'], row['imd_lon_24h']]
        t48 = [row['imd_lat_48h'], row['imd_lon_48h']]
        
        m12 = 1.0 if not (np.isnan(t12[0]) or np.isnan(t12[1])) else 0.0
        m24 = 1.0 if not (np.isnan(t24[0]) or np.isnan(t24[1])) else 0.0
        m48 = 1.0 if not (np.isnan(t48[0]) or np.isnan(t48[1])) else 0.0

        item = {
            'sample_id': sid,
            'storm_id': storm,
            'split': split,
            't0': t0,
            'env_seq': seq_norm.half(), # float16 to preserve RAM and disk
            'source_era5_files': source_files,
            'target_12': torch.tensor([t12[0] if m12 else 0.0, t12[1] if m12 else 0.0], dtype=torch.float32),
            'target_24': torch.tensor([t24[0] if m24 else 0.0, t24[1] if m24 else 0.0], dtype=torch.float32),
            'target_48': torch.tensor([t48[0] if m48 else 0.0, t48[1] if m48 else 0.0], dtype=torch.float32),
            'mask_12': torch.tensor(m12, dtype=torch.float32),
            'mask_24': torch.tensor(m24, dtype=torch.float32),
            'mask_48': torch.tensor(m48, dtype=torch.float32),
        }
        samples_by_split[split].append(item)

    metadata = {
        'total_samples': len(df),
        'train_samples': len(samples_by_split['TRAIN']),
        'val_samples': len(samples_by_split['VALIDATION']),
        'test_samples': len(samples_by_split['TEST']),
        'available_frames': available_era5_count,
        'missing_frames': missing_era5_count,
        'normalization_stats_source': stats_path,
        'spatial_resolution': '41x66 (~1.0 deg)',
        'variables': 'u, v at 850, 700, 500, 300 hPa (8 channels)'
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'metadata': metadata,
        'samples': samples_by_split
    }
    torch.save(payload, out_path)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved feature cache to {out_path} ({file_size_mb:.2f} MB)")
    print(f"  TRAIN: {len(samples_by_split['TRAIN'])}")
    print(f"  VAL:   {len(samples_by_split['VALIDATION'])}")
    print(f"  TEST:  {len(samples_by_split['TEST'])}")
    print("=" * 80)

if __name__ == '__main__':
    build_cache()
