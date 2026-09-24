"""
Rebuild Sample Index and Candidate Manifest
Regenerates data/manifests/vayu_net_sample_index.csv and
data/manifests/vayu_net_candidate_t0_manifest.csv using corrected IMD Best Track V2.

Guarantees:
- Preserves exact 1,319 candidate t0 timestamps and storm split assignments.
- Re-maps ground truth labels (lat, lon, wind, pressure, category) directly from corrected IMD V2.
- Retains legitimate source NaNs.
- Validates 100% path resolution for all six GridSat frames.
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta

IMD_V2_CSV = "data/processed/imd_best_track_v2.csv"
CAND_MANIFEST_CSV = "data/manifests/vayu_net_candidate_t0_manifest.csv"
SAMPLE_INDEX_CSV = "data/manifests/vayu_net_sample_index.csv"
INTERIM_GRIDSAT_DIR = "data/interim/gridsat"

def rebuild():
    # 1. Load corrected IMD V2
    print("Loading corrected IMD Best Track V2...")
    df_v2 = pd.read_csv(IMD_V2_CSV)
    
    # Create lookup dictionary (storm_id, timestamp_utc) -> row
    obs_map = {}
    for _, row in df_v2.iterrows():
        key = (row["storm_id"], row["timestamp_utc"])
        obs_map[key] = row

    # 2. Load existing candidate manifest
    cand_df = pd.read_csv(CAND_MANIFEST_CSV)
    print(f"Loaded candidate manifest with {len(cand_df)} candidates.")

    updated_cand_rows = []
    sample_records = []

    for idx, row in cand_df.iterrows():
        storm_id = row["storm_id"]
        t0_str = row["t0"]
        t0 = pd.to_datetime(t0_str)
        t_12_str = row["target_12h_timestamp"]
        t_24_str = row["target_24h_timestamp"]
        t_48_str = row["target_48h_timestamp"]

        # Ground truth observations from corrected IMD V2
        obs_t0 = obs_map.get((storm_id, t0_str))
        obs_12 = obs_map.get((storm_id, t_12_str))
        obs_24 = obs_map.get((storm_id, t_24_str))
        obs_48 = obs_map.get((storm_id, t_48_str))

        assert obs_t0 is not None, f"Missing t0 obs: {storm_id} at {t0_str}"
        assert obs_12 is not None, f"Missing +12h obs: {storm_id} at {t_12_str}"
        assert obs_24 is not None, f"Missing +24h obs: {storm_id} at {t_24_str}"
        assert obs_48 is not None, f"Missing +48h obs: {storm_id} at {t_48_str}"

        # Update candidate manifest row
        cand_rec = dict(row)
        cand_rec["lat_t0"] = obs_t0["latitude"]
        cand_rec["lon_t0"] = obs_t0["longitude"]
        cand_rec["wind_t0"] = obs_t0["maximum_sustained_wind_kt"]
        cand_rec["pressure_t0"] = obs_t0["central_pressure_hpa"]
        cand_rec["category_t0"] = obs_t0["category"]

        cand_rec["lat_12h"] = obs_12["latitude"]
        cand_rec["lon_12h"] = obs_12["longitude"]
        cand_rec["wind_12h"] = obs_12["maximum_sustained_wind_kt"]
        cand_rec["pressure_12h"] = obs_12["central_pressure_hpa"]
        cand_rec["category_12h"] = obs_12["category"]

        cand_rec["lat_24h"] = obs_24["latitude"]
        cand_rec["lon_24h"] = obs_24["longitude"]
        cand_rec["wind_24h"] = obs_24["maximum_sustained_wind_kt"]
        cand_rec["pressure_24h"] = obs_24["central_pressure_hpa"]
        cand_rec["category_24h"] = obs_24["category"]

        cand_rec["lat_48h"] = obs_48["latitude"]
        cand_rec["lon_48h"] = obs_48["longitude"]
        cand_rec["wind_48h"] = obs_48["maximum_sustained_wind_kt"]
        cand_rec["pressure_48h"] = obs_48["central_pressure_hpa"]
        cand_rec["category_48h"] = obs_48["category"]
        updated_cand_rows.append(cand_rec)

        # Build sample index record
        sample_id = f"{storm_id}_{t0.strftime('%Y%m%d_%H%MZ')}"
        frames = {}
        for h in [15, 12, 9, 6, 3, 0]:
            ts = t0 - timedelta(hours=h)
            tag = f"frame_t_minus_{h}h" if h > 0 else "frame_t0"
            fname = f"gridsat_{ts.year:04d}.{ts.month:02d}.{ts.day:02d}.{ts.hour:02d}.npz"
            rel_path = f"data/interim/gridsat/{ts.year}/{fname}"
            frames[tag] = rel_path

        sample_rec = {
            "sample_id": sample_id,
            "storm_id": storm_id,
            "split": row["split"],
            "t0": t0_str,
            "frame_t_minus_15h": frames["frame_t_minus_15h"],
            "frame_t_minus_12h": frames["frame_t_minus_12h"],
            "frame_t_minus_9h": frames["frame_t_minus_9h"],
            "frame_t_minus_6h": frames["frame_t_minus_6h"],
            "frame_t_minus_3h": frames["frame_t_minus_3h"],
            "frame_t0": frames["frame_t0"],
            "imd_lat_t0": obs_t0["latitude"],
            "imd_lon_t0": obs_t0["longitude"],
            "imd_wind_t0": obs_t0["maximum_sustained_wind_kt"],
            "imd_pressure_t0": obs_t0["central_pressure_hpa"],
            "imd_category_t0": obs_t0["category"],
            "imd_lat_12h": obs_12["latitude"],
            "imd_lon_12h": obs_12["longitude"],
            "wind_12h": obs_12["maximum_sustained_wind_kt"],
            "pressure_12h": obs_12["central_pressure_hpa"],
            "category_12h": obs_12["category"],
            "imd_lat_24h": obs_24["latitude"],
            "imd_lon_24h": obs_24["longitude"],
            "wind_24h": obs_24["maximum_sustained_wind_kt"],
            "pressure_24h": obs_24["central_pressure_hpa"],
            "category_24h": obs_24["category"],
            "imd_lat_48h": obs_48["latitude"],
            "imd_lon_48h": obs_48["longitude"],
            "wind_48h": obs_48["maximum_sustained_wind_kt"],
            "pressure_48h": obs_48["central_pressure_hpa"],
            "category_48h": obs_48["category"],
        }
        sample_records.append(sample_rec)

    new_cand_df = pd.DataFrame(updated_cand_rows)
    new_sample_df = pd.DataFrame(sample_records)

    # 3. Validation before saving
    assert len(new_cand_df) == len(cand_df) == 1319
    assert len(new_sample_df) == 1319

    # Verify frame resolution
    print("Verifying 100% resolution of GridSat frames...")
    missing_frames = []
    frame_cols = ["frame_t_minus_15h", "frame_t_minus_12h", "frame_t_minus_9h",
                  "frame_t_minus_6h", "frame_t_minus_3h", "frame_t0"]
    for idx, row in new_sample_df.iterrows():
        for col in frame_cols:
            p = row[col]
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                missing_frames.append((row["sample_id"], col, p))

    if missing_frames:
        raise RuntimeError(f"Found {len(missing_frames)} missing GridSat frames! E.g. {missing_frames[:3]}")
    print(f"All {len(new_sample_df) * 6} frame references resolve successfully on disk.")

    # 4. Save
    new_cand_df.to_csv(CAND_MANIFEST_CSV, index=False)
    new_sample_df.to_csv(SAMPLE_INDEX_CSV, index=False)
    print(f"Successfully saved {CAND_MANIFEST_CSV} and {SAMPLE_INDEX_CSV} ({len(new_sample_df)} samples).")

if __name__ == "__main__":
    rebuild()
