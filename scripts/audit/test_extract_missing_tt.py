import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml.models.temporal_track_gru import TemporalTrackGRU
from ml.models.multisource_fusion import (
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    CATEGORY_TO_IDX,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT
)

device = torch.device("cpu")
print("Loading model and checking missing samples...")

df_multi = pd.read_csv("data/manifests/vayu_net_multisource_sample_index.csv")
print(f"Total samples in multisource index: {len(df_multi)} ({df_multi['split'].value_counts().to_dict()})")

# Load existing temporal track cache
c_tt = torch.load("data/interim/ml/cache/temporal_track_features.pt", map_location="cpu", weights_only=False)
tt_map = {s["sample_id"]: s["feature_sequence"] for s in c_tt["samples"]}

missing_ids = [sid for sid in df_multi["sample_id"] if sid not in tt_map]
print(f"Number of samples missing from temporal track cache: {len(missing_ids)}")

if missing_ids:
    print("Extracting features for missing samples using best_center_localization_cnn...")
    with open("data/interim/ml/train_normalization_stats.json", "r") as f:
        stats = json.load(f)
    mean_k = float(stats["mean_kelvin"])
    std_k = float(stats["std_kelvin"])
    
    model = TemporalTrackGRU(
        spatial_checkpoint_path="data/interim/ml/checkpoints/best_center_localization_cnn.pt",
        freeze_spatial_encoder=True,
        spatial_emb_dim=128
    ).to(device)
    model.eval()
    
    # Check if temporal track weights exist to load spatial_proj if present
    tt_ckpt_path = "data/interim/ml/checkpoints/best_temporal_track_gru.pt"
    if os.path.exists(tt_ckpt_path):
        tt_ckpt = torch.load(tt_ckpt_path, map_location="cpu", weights_only=False)
        tt_state = tt_ckpt["model_state_dict"] if "model_state_dict" in tt_ckpt else tt_ckpt
        model.load_state_dict(tt_state, strict=False)
        print("Loaded spatial_proj weights from best_temporal_track_gru.pt")
        
    df_missing = df_multi[df_multi["sample_id"].isin(missing_ids)].copy()
    frame_cols = [
        'grid_sat_t_minus_15h', 'grid_sat_t_minus_12h', 'grid_sat_t_minus_9h',
        'grid_sat_t_minus_6h', 'grid_sat_t_minus_3h', 'grid_sat_t0'
    ]
    unique_missing_frames = sorted(list(set(df_missing[frame_cols].values.flatten())))
    print(f"Unique frames to extract for missing samples: {len(unique_missing_frames)}")
    
    frame_dict = {}
    for fpath in unique_missing_frames:
        with np.load(fpath) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)
        inv = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(inv):
            arr[inv] = mean_k
        arr = (arr - mean_k) / std_k
        t_frame = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            emb, center = model.extract_single_frame_feature(t_frame)
            frame_dict[fpath] = (emb.squeeze(0).cpu(), center.squeeze(0).cpu())
            
    print("Assembling 6-step sequences for missing samples...")
    for idx, row in df_missing.iterrows():
        sid = row["sample_id"]
        step_features = []
        prev_center = None
        for col in frame_cols:
            fpath = row[col]
            emb_t, center_t = frame_dict[fpath]
            if prev_center is None:
                delta_t = torch.zeros(2, dtype=torch.float32)
            else:
                delta_t = center_t - prev_center
            prev_center = center_t
            feat_t = torch.cat([emb_t, center_t, delta_t], dim=-1)
            step_features.append(feat_t)
        seq_tensor = torch.stack(step_features, dim=0)
        tt_map[sid] = seq_tensor
        
    print(f"All {len(df_multi)} samples now have gridsat_feat_seq in memory!")
