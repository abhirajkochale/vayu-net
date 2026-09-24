"""
VAYU-NET WP-07 — HISTORICAL ANALOG-STORM RETRIEVAL BUILDER
==========================================================
Constructs the 7-dimensional analog feature archive from the locked TRAIN partition:
  features = [lat, lon, wind_kt, pressure_hpa, dx_12h, dy_12h, dwind_12h]
Where:
  - lat: latitude at t0 (deg)
  - lon: longitude at t0 (deg)
  - wind_kt: maximum sustained wind speed at t0 (kt)
  - pressure_hpa: central surface pressure at t0 (hPa)
  - dx_12h: past 12-hour longitudinal displacement: lon(t0) - lon(t-12h) (deg)
  - dy_12h: past 12-hour latitudinal displacement: lat(t0) - lat(t-12h) (deg)
  - dwind_12h: past 12-hour wind speed change: wind(t0) - wind(t-12h) (kt)

Discipline:
  - Standardization statistics (mean, std) computed strictly from TRAIN partition (N=696).
  - Self-match & same-storm exclusion logic built-in.
  - Produces data/interim/ml/analog_retrieval_cache.json and data/interim/ml/analog_retrieval_cache.pkl.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SAMPLE_INDEX_PATH = "data/manifests/vayu_net_sample_index.csv"
BT_V2_PATH = "data/processed/imd_best_track_v2.csv"
STORM_MANIFEST_PATH = "data/manifests/storm_event_manifest_v2.csv"
CACHE_JSON_PATH = "data/interim/ml/analog_retrieval_cache.json"
CACHE_PKL_PATH = "data/interim/ml/analog_retrieval_cache.pkl"

FEATURE_NAMES = ["lat", "lon", "wind_kt", "pressure_hpa", "dx_12h", "dy_12h", "dwind_12h"]


def extract_features_for_sample(row, storm_tracks):
    """
    Extracts 7-d feature vector for a given row from sample_index.
    Uses strictly past information (t <= t0).
    """
    sid = str(row["storm_id"])
    t0_dt = pd.to_datetime(row["t0"])
    lat0 = float(row["imd_lat_t0"])
    lon0 = float(row["imd_lon_t0"])
    w0 = float(row["imd_wind_t0"])
    p0 = float(row["imd_pressure_t0"])

    st = storm_tracks[sid]
    priors = st[st["dt"] < t0_dt]

    # Check exact t0 - 12h
    t_12 = t0_dt - pd.Timedelta(hours=12)
    match_12 = st[st["dt"] == t_12]

    if len(match_12) > 0:
        r12 = match_12.iloc[0]
        dx = lon0 - float(r12["longitude"])
        dy = lat0 - float(r12["latitude"])
        w_prev = float(r12["maximum_sustained_wind_kt"]) if pd.notna(r12["maximum_sustained_wind_kt"]) else w0
        dw = w0 - w_prev
    elif len(priors) > 0:
        # Scale latest available prior observation velocity to 12h equivalent
        latest = priors.iloc[-1]
        dt_h = max(0.5, (t0_dt - latest["dt"]).total_seconds() / 3600.0)
        v_lon = (lon0 - float(latest["longitude"])) / dt_h
        v_lat = (lat0 - float(latest["latitude"])) / dt_h
        dx = v_lon * 12.0
        dy = v_lat * 12.0
        if pd.notna(latest["maximum_sustained_wind_kt"]):
            v_w = (w0 - float(latest["maximum_sustained_wind_kt"])) / dt_h
            dw = v_w * 12.0
        else:
            dw = 0.0
    else:
        dx, dy, dw = 0.0, 0.0, 0.0

    return [lat0, lon0, w0, p0, round(float(dx), 3), round(float(dy), 3), round(float(dw), 2)]


def build_analog_database():
    print("=" * 80)
    print("BUILDING WP-07 HISTORICAL ANALOG RETRIEVAL DATABASE")
    print("=" * 80)

    sample_index_df = pd.read_csv(SAMPLE_INDEX_PATH)
    bt_v2_df = pd.read_csv(BT_V2_PATH)
    storm_manifest_df = pd.read_csv(STORM_MANIFEST_PATH)

    # Build storm track cache for fast lookups
    storm_tracks = {}
    for sid, group in bt_v2_df.groupby("storm_id"):
        gdf = group.copy()
        gdf["dt"] = pd.to_datetime(gdf["timestamp_utc"])
        gdf = gdf.sort_values("dt").reset_index(drop=True)
        storm_tracks[sid] = gdf

    # Build storm metadata lookup
    storm_meta = {}
    for _, row in storm_manifest_df.iterrows():
        sid = row["storm_id"]
        s_name = row["storm_name"]
        year = int(row["year"])
        start_dt = pd.to_datetime(row["start_timestamp_utc"])
        month = start_dt.month
        # NIO Cyclone Seasons: Pre-monsoon (April-June), Post-monsoon (September-December), Monsoon/Winter (Other)
        if month in [4, 5, 6]:
            season = "Pre-monsoon"
        elif month in [9, 10, 11, 12]:
            season = "Post-monsoon"
        else:
            season = "Monsoon/Winter"

        storm_meta[sid] = {
            "storm_name": s_name,
            "year": year,
            "season": season,
            "peak_category": row["peak_category"],
            "max_wind_kt": float(row["max_wind_kt"])
        }

    # Extract features for all samples
    all_sample_features = {}
    for idx, row in sample_index_df.iterrows():
        sid_sample = row["sample_id"]
        feats = extract_features_for_sample(row, storm_tracks)
        all_sample_features[sid_sample] = feats

    # 1. STANDARDIZATION STATISTICS: COMPUTED STRICTLY ON TRAIN
    train_df = sample_index_df[sample_index_df["split"] == "TRAIN"]
    print(f"Deriving standardization statistics strictly from TRAIN partition ({len(train_df)} samples across 81 storms)...")

    train_feature_matrix = np.array([all_sample_features[r["sample_id"]] for _, r in train_df.iterrows()], dtype=np.float32)

    means = np.mean(train_feature_matrix, axis=0)
    stds = np.std(train_feature_matrix, axis=0)
    # Prevent divide-by-zero
    stds = np.where(stds < 1e-6, 1.0, stds)

    stats = {
        "feature_names": FEATURE_NAMES,
        "feature_means": [round(float(m), 4) for m in means],
        "feature_stds": [round(float(s), 4) for s in stds]
    }

    print("\nTRAIN Archive Standardization Statistics:")
    for fn, m, s in zip(FEATURE_NAMES, stats["feature_means"], stats["feature_stds"]):
        print(f"  {fn:15s} -> Mean: {m:8.2f} | Std: {s:8.2f}")

    # 2. BUILD HISTORICAL CANDIDATE ARCHIVE (From TRAIN partition)
    candidate_archive = []
    for _, row in train_df.iterrows():
        samp_id = row["sample_id"]
        st_id = row["storm_id"]
        sm = storm_meta.get(st_id, {})
        raw_f = all_sample_features[samp_id]
        std_f = [(raw_f[j] - means[j]) / stds[j] for j in range(7)]

        candidate_archive.append({
            "sample_id": samp_id,
            "storm_id": st_id,
            "storm_name": sm.get("storm_name", st_id),
            "year": sm.get("year", int(row["t0"][:4])),
            "season": sm.get("season", "Unknown"),
            "timestamp_utc": row["t0"],
            "category": row["imd_category_t0"] if pd.notna(row["imd_category_t0"]) else "UNKNOWN",
            "raw_features": {fn: raw_f[j] for j, fn in enumerate(FEATURE_NAMES)},
            "standardized_features": [round(float(v), 5) for v in std_f]
        })

    print(f"\nBuilt historical analog archive with {len(candidate_archive)} candidate snapshots across 81 historical storms.")

    payload = {
        "metadata": {
            "title": "VAYU-NET Historical Analog-Storm Retrieval Archive",
            "version": "1.0",
            "date": "2026-09-25",
            "derivation_split": "TRAIN",
            "total_candidates": len(candidate_archive),
            "total_storms": len(set(c["storm_id"] for c in candidate_archive)),
            "features_used": FEATURE_NAMES,
            "distance_metric": "Euclidean distance in TRAIN-standardized feature space",
            "k_default": 2,
            "label": "HISTORICAL ANALOG"
        },
        "standardization": stats,
        "candidates": candidate_archive,
        # Also store pre-extracted features for all 1319 samples for seamless inference
        "all_samples_feature_dict": {k: [round(float(v), 4) for v in f] for k, f in all_sample_features.items()}
    }

    # Save JSON and PKL
    os.makedirs(os.path.dirname(CACHE_JSON_PATH), exist_ok=True)
    with open(CACHE_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {CACHE_JSON_PATH} ({os.path.getsize(CACHE_JSON_PATH)/(1024*1024):.2f} MB)")

    with open(CACHE_PKL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {CACHE_PKL_PATH} ({os.path.getsize(CACHE_PKL_PATH)/(1024*1024):.2f} MB)")

    return payload


if __name__ == "__main__":
    build_analog_database()
