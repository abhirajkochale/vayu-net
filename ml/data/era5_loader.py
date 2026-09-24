"""
VAYU-NET PHASE 5A — ERA5 DATA LOADER & NORMALIZATION PIPELINE
=============================================================
Manages environmental atmospheric wind tensors for VAYU-NET samples:
  - Sequence: 6 frames [t-15h, t-12h, t-9h, t-6h, t-3h, t0]
  - Channels: 8 [U(850, 700, 500, 300), V(850, 700, 500, 300)]
  - Spatial: [41, 66]
  - Normalization: Per-channel (mean, std) calculated strictly from TRAIN samples
  - Stored in: data/interim/ml/era5_train_normalization_stats.json
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

RAW_ERA5_DIR = "data/raw/era5"
SAMPLE_INDEX_PATH = "data/manifests/vayu_net_sample_index.csv"
NORM_STATS_PATH = "data/interim/ml/era5_train_normalization_stats.json"

FRAME_COLS = [
    'frame_t_minus_15h',
    'frame_t_minus_12h',
    'frame_t_minus_9h',
    'frame_t_minus_6h',
    'frame_t_minus_3h',
    'frame_t0'
]

GRID_PATTERN = re.compile(r'gridsat_(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.npz')

def parse_era5_filename_from_gridsat(gridsat_path):
    m = GRID_PATTERN.search(gridsat_path)
    if not m:
        raise ValueError(f"Cannot parse timestamp from gridsat path: {gridsat_path}")
    y, mo, d, h = m.groups()
    return f"era5_{y}.{mo}.{d}.{h}.npz"

def compute_and_save_train_normalization_stats(
    sample_index_path=SAMPLE_INDEX_PATH,
    era5_dir=RAW_ERA5_DIR,
    stats_out_path=NORM_STATS_PATH
):
    df = pd.read_csv(sample_index_path)
    train_df = df[df['split'] == 'TRAIN']
    print(f"[compute_train_norm] Found {len(train_df)} TRAIN candidate samples.")
    
    # Collect all unique TRAIN files
    train_files = set()
    for _, row in train_df.iterrows():
        for col in FRAME_COLS:
            f = parse_era5_filename_from_gridsat(str(row[col]))
            p = os.path.join(era5_dir, f)
            if os.path.exists(p):
                train_files.add(p)
                
    print(f"[compute_train_norm] Found {len(train_files)} available TRAIN ERA5 files.")
    if len(train_files) == 0:
        raise RuntimeError("No TRAIN ERA5 files found in data/raw/era5 to compute normalization stats!")

    # Channels: 8
    # Accumulate sums and squared sums for mean/std
    sums = np.zeros(8, dtype=np.float64)
    sq_sums = np.zeros(8, dtype=np.float64)
    min_vals = np.full(8, np.inf, dtype=np.float64)
    max_vals = np.full(8, -np.inf, dtype=np.float64)
    pixel_count_per_channel = 0

    for p in sorted(train_files):
        data = np.load(p)['env'] # [8, 41, 66]
        for c in range(8):
            c_data = data[c].astype(np.float64)
            sums[c] += np.sum(c_data)
            sq_sums[c] += np.sum(c_data ** 2)
            min_vals[c] = min(min_vals[c], np.min(c_data))
            max_vals[c] = max(max_vals[c], np.max(c_data))
        pixel_count_per_channel += data.shape[1] * data.shape[2]

    means = sums / pixel_count_per_channel
    variances = (sq_sums / pixel_count_per_channel) - (means ** 2)
    stds = np.sqrt(np.maximum(variances, 1e-6))

    channel_names = [
        "u_850hPa", "u_700hPa", "u_500hPa", "u_300hPa",
        "v_850hPa", "v_700hPa", "v_500hPa", "v_300hPa"
    ]

    stats = {
        "split_source": "TRAIN",
        "num_files_analyzed": len(train_files),
        "total_pixels_per_channel": int(pixel_count_per_channel),
        "means": {channel_names[c]: float(means[c]) for c in range(8)},
        "stds": {channel_names[c]: float(stds[c]) for c in range(8)},
        "mins": {channel_names[c]: float(min_vals[c]) for c in range(8)},
        "maxs": {channel_names[c]: float(max_vals[c]) for c in range(8)},
        "means_list": [float(m) for m in means],
        "stds_list": [float(s) for s in stds]
    }

    os.makedirs(os.path.dirname(stats_out_path), exist_ok=True)
    with open(stats_out_path, 'w') as fp:
        json.dump(stats, fp, indent=2)
    print(f"[compute_train_norm] Successfully wrote normalization stats to {stats_out_path}")
    return stats

def load_normalization_stats(stats_path=NORM_STATS_PATH):
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, 'r') as fp:
        return json.load(fp)


class ERA5EnvironmentDataset(Dataset):
    """
    Dataset returning 6-frame ERA5 environmental sequences and future track targets.
    """
    def __init__(self, sample_index_path=SAMPLE_INDEX_PATH, era5_dir=RAW_ERA5_DIR, split='TRAIN', norm_stats_path=NORM_STATS_PATH):
        df = pd.read_csv(sample_index_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.era5_dir = era5_dir
        self.split = split
        
        # Load norm stats
        stats = load_normalization_stats(norm_stats_path)
        if stats:
            self.means = torch.tensor(stats['means_list'], dtype=torch.float32).view(1, 8, 1, 1)
            self.stds = torch.tensor(stats['stds_list'], dtype=torch.float32).view(1, 8, 1, 1)
        else:
            self.means = torch.zeros(1, 8, 1, 1)
            self.stds = torch.ones(1, 8, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']
        storm_id = row['storm_id']
        t0 = row['t0']
        
        # Load 6 ERA5 frames
        frames = []
        for col in FRAME_COLS:
            fname = parse_era5_filename_from_gridsat(str(row[col]))
            fpath = os.path.join(self.era5_dir, fname)
            if os.path.exists(fpath):
                arr = np.load(fpath)['env'] # [8, 41, 66]
            else:
                # Fallback zero array if specific file pending
                arr = np.zeros((8, 41, 66), dtype=np.float32)
            frames.append(arr)
            
        seq_tensor = torch.from_numpy(np.stack(frames, axis=0)) # [6, 8, 41, 66]
        # Normalize
        seq_norm = (seq_tensor - self.means) / self.stds

        # Targets (+12h, +24h, +48h)
        t12 = [row['imd_lat_12h'], row['imd_lon_12h']]
        t24 = [row['imd_lat_24h'], row['imd_lon_24h']]
        t48 = [row['imd_lat_48h'], row['imd_lon_48h']]
        
        m12 = 1.0 if not (np.isnan(t12[0]) or np.isnan(t12[1])) else 0.0
        m24 = 1.0 if not (np.isnan(t24[0]) or np.isnan(t24[1])) else 0.0
        m48 = 1.0 if not (np.isnan(t48[0]) or np.isnan(t48[1])) else 0.0
        
        return {
            'sample_id': sample_id,
            'storm_id': storm_id,
            't0': t0,
            'env_seq': seq_norm, # [6, 8, 41, 66]
            'target_12': torch.tensor([t12[0] if m12 else 0.0, t12[1] if m12 else 0.0], dtype=torch.float32),
            'target_24': torch.tensor([t24[0] if m24 else 0.0, t24[1] if m24 else 0.0], dtype=torch.float32),
            'target_48': torch.tensor([t48[0] if m48 else 0.0, t48[1] if m48 else 0.0], dtype=torch.float32),
            'mask_12': torch.tensor(m12, dtype=torch.float32),
            'mask_24': torch.tensor(m24, dtype=torch.float32),
            'mask_48': torch.tensor(m48, dtype=torch.float32),
        }
