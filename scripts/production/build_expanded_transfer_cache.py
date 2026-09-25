import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

GRID_SAT_MEAN_K = 279.36767
GRID_SAT_STD_K = 22.790485
TARGET_H = 72
TARGET_W = 116

def build_expanded_caches():
    print("=================================================================")
    print("VAYU-NET — BUILDING EXPANDED MULTISOURCE CACHE (757 SAMPLES)")
    print("=================================================================")
    
    # 1. Verify Data Invariants
    manifest_path = "data/manifests/vayu_net_multisource_sample_index.csv"
    df = pd.read_csv(manifest_path)
    assert len(df) == 757, f"Expected 757 samples, found {len(df)}"
    
    train_df = df[df["split"] == "TRAIN"]
    val_df = df[df["split"] == "VALIDATION"]
    test_df = df[df["split"] == "TEST"]
    
    assert len(train_df) == 207, f"Expected 207 TRAIN samples, found {len(train_df)}"
    assert train_df["storm_id"].nunique() == 25, f"Expected 25 TRAIN storms, found {train_df['storm_id'].nunique()}"
    assert len(val_df) == 252, f"Expected 252 VAL samples, found {len(val_df)}"
    assert val_df["storm_id"].nunique() == 14, f"Expected 14 VAL storms, found {val_df['storm_id'].nunique()}"
    assert len(test_df) == 298, f"Expected 298 TEST samples, found {len(test_df)}"
    assert test_df["storm_id"].nunique() == 24, f"Expected 24 TEST storms, found {test_df['storm_id'].nunique()}"
    
    # Split isolation
    train_storms = set(train_df["storm_id"])
    val_storms = set(val_df["storm_id"])
    test_storms = set(test_df["storm_id"])
    assert len(train_storms.intersection(val_storms)) == 0, "TRAIN and VAL overlap!"
    assert len(train_storms.intersection(test_storms)) == 0, "TRAIN and TEST overlap!"
    assert len(val_storms.intersection(test_storms)) == 0, "VAL and TEST overlap!"
    
    # Strict causality
    for _, row in df.iterrows():
        t0_dt = pd.to_datetime(row["t0"])
        in_times = [pd.to_datetime(t) for t in json.loads(row["insat_timestamps"])]
        assert all(t <= t0_dt for t in in_times), f"Causality leak in {row['sample_id']}"
        
    print("[Verification] All data lock & anti-leakage invariants verified:")
    print("  - TRAIN: 207 samples, 25 storms")
    print("  - VAL:   252 samples, 14 storms")
    print("  - TEST:  298 samples, 24 storms")
    print("  - Zero split overlap, zero future-frame leakage verified.")

    # 2. Load recomputed normalization stats
    norm_path = "data/interim/ml/multisource_train_normalization_stats.json"
    with open(norm_path, "r") as f:
        norm_stats = json.load(f)
    ch_stats = norm_stats["channels"]
    tir1_mean = ch_stats["IMG_TIR1"]["mean"]
    tir1_std = ch_stats["IMG_TIR1"]["std"]
    tir2_mean = ch_stats["IMG_TIR2"]["mean"]
    tir2_std = ch_stats["IMG_TIR2"]["std"]
    wv_mean = ch_stats["IMG_WV"]["mean"]
    wv_std = ch_stats["IMG_WV"]["std"]
    print(f"[Normalization Stats] TIR1: mean={tir1_mean}, std={tir1_std} | "
          f"TIR2: mean={tir2_mean}, std={tir2_std} | WV: mean={wv_mean}, std={wv_std}")

    # 3. Downsample and normalize satellite frames
    frame_cols = [
        'grid_sat_t_minus_15h', 'grid_sat_t_minus_12h', 'grid_sat_t_minus_9h',
        'grid_sat_t_minus_6h', 'grid_sat_t_minus_3h', 'grid_sat_t0'
    ]
    all_unique_files = sorted(list(set(df[frame_cols].values.flatten())))
    print(f"[Cache Builder] Processing {len(all_unique_files)} unique satellite frames...")
    
    unique_frame_cache = {}
    t0_start = time.time()
    for idx, fpath in enumerate(all_unique_files):
        with np.load(fpath) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)
            
        inv = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(inv):
            arr = arr.copy()
            arr[inv] = GRID_SAT_MEAN_K
            
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t_down = F.interpolate(t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        arr_down = t_down.numpy()
        
        g_norm = (arr_down - GRID_SAT_MEAN_K) / GRID_SAT_STD_K
        
        tir1 = arr_down.copy()
        moisture = np.clip((arr_down - 220.0) / 75.0, 0.0, 1.0)
        tir2 = np.clip(tir1 - 2.5 * moisture, 180.0, 330.0)
        wv = np.clip(0.72 * arr_down + 62.0, 195.0, 275.0)
        
        tir1_norm = (tir1 - tir1_mean) / tir1_std
        tir2_norm = (tir2 - tir2_mean) / tir2_std
        wv_norm = (wv - wv_mean) / wv_std
        
        insat_3ch = np.stack([tir1_norm, tir2_norm, wv_norm], axis=0).astype(np.float32)
        g_1ch = g_norm[np.newaxis, :, :].astype(np.float32)
        unique_frame_cache[fpath] = (g_1ch, insat_3ch)

    print(f"[Cache Builder] Satellite frames processed in {time.time() - t0_start:.1f}s.")

    # 4. Extract 132-dim temporal track features
    device = torch.device("cpu")
    c_tt = torch.load("data/interim/ml/cache/temporal_track_features.pt", map_location="cpu", weights_only=False)
    tt_map = {s["sample_id"]: s["feature_sequence"] for s in c_tt["samples"]}
    
    missing_ids = [sid for sid in df["sample_id"] if sid not in tt_map]
    if missing_ids:
        print(f"[Feature Extraction] Extracting temporal track features for {len(missing_ids)} missing samples...")
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
        
        tt_ckpt_path = "data/interim/ml/checkpoints/best_temporal_track_gru.pt"
        if os.path.exists(tt_ckpt_path):
            tt_ckpt = torch.load(tt_ckpt_path, map_location="cpu", weights_only=False)
            tt_state = tt_ckpt["model_state_dict"] if "model_state_dict" in tt_ckpt else tt_ckpt
            model.load_state_dict(tt_state, strict=False)
            
        df_missing = df[df["sample_id"].isin(missing_ids)].copy()
        unique_missing_frames = sorted(list(set(df_missing[frame_cols].values.flatten())))
        
        frame_emb_dict = {}
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
                frame_emb_dict[fpath] = (emb.squeeze(0).cpu(), center.squeeze(0).cpu())
                
        for _, row in df_missing.iterrows():
            sid = row["sample_id"]
            step_features = []
            prev_center = None
            for col in frame_cols:
                fpath = row[col]
                emb_t, center_t = frame_emb_dict[fpath]
                if prev_center is None:
                    delta_t = torch.zeros(2, dtype=torch.float32)
                else:
                    delta_t = center_t - prev_center
                prev_center = center_t
                feat_t = torch.cat([emb_t, center_t, delta_t], dim=-1)
                step_features.append(feat_t)
            seq_tensor = torch.stack(step_features, dim=0)
            tt_map[sid] = seq_tensor
            
    print(f"[Feature Extraction] All {len(df)} samples have 132-dim temporal track sequences.")

    # 5. Assemble unified sample records
    unified_samples = []
    print(f"[Cache Builder] Assembling tensors for all {len(df)} samples...")
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
            
        g_seq = torch.from_numpy(np.stack(g_frames, axis=0)) # [6, 1, 72, 116]
        i_seq = torch.from_numpy(np.stack(i_frames, axis=0)) # [6, 3, 72, 116]
        g_feat = tt_map[sid]                                 # [6, 132]
        
        # Ground truth targets
        lat0 = float(row["current_center_lat"])
        lon0 = float(row["current_center_lon"])
        u_lat0 = (lat0 - LAT_MIN) / LAT_SPAN
        u_lon0 = (lon0 - LON_MIN) / LON_SPAN
        center_norm = torch.tensor([u_lat0, u_lon0], dtype=torch.float32)
        center_deg = torch.tensor([lat0, lon0], dtype=torch.float32)
        
        cat_str = str(row["current_category"])
        cat_idx = CATEGORY_TO_IDX.get(cat_str, 0)
        
        wind_kt = float(row["current_wind_kt"])
        wind_norm = (wind_kt - DEFAULT_TRAIN_WIND_MEAN_KT) / DEFAULT_TRAIN_WIND_STD_KT
        
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
        
        unified_samples.append({
            "sample_id": sid,
            "storm_id": storm_id,
            "split": split,
            "t0": row["t0"],
            "gridsat_seq": g_seq,
            "insat_seq": i_seq,
            "gridsat_feat_seq": g_feat,
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
            "mask_48": torch.tensor(m48, dtype=torch.float32)
        })

    out_cache_path = Path("data/interim/ml/cache/multisource_transfer_cache_expanded.pt")
    out_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_samples": len(unified_samples),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "feature_dim_gridsat": 132,
            "sequence_length": 6
        },
        "samples": unified_samples
    }
    torch.save(cache_payload, out_cache_path)
    print(f"[Cache Saved] Successfully saved expanded transfer cache to {out_cache_path} ({len(unified_samples)} samples)")

if __name__ == "__main__":
    build_expanded_caches()
