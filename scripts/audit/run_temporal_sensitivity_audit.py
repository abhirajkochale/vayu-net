"""
VAYU-NET — Temporal Matching Tolerance Sensitivity Audit

Calculates detailed frame-level and sample-level retention, storm counts,
and rejection statistics under:
  A. delta_t <= 15 minutes
  B. delta_t <= 30 minutes
  C. delta_t <= 60 minutes
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def run_sensitivity_audit():
    sample_index_path = Path("data/manifests/vayu_net_multisource_sample_index.csv")
    align_path = Path("data/manifests/multisource_timestamp_alignment.csv")
    
    assert sample_index_path.exists(), f"Sample index not found at {sample_index_path}"
    assert align_path.exists(), f"Alignment manifest not found at {align_path}"
    
    df_samples = pd.read_csv(sample_index_path)
    df_align = pd.read_csv(align_path)
    
    total_samples = len(df_samples)
    total_frames = len(df_align)
    
    thresholds = [15, 30, 60]
    results = {
        "metadata": {
            "total_paired_samples_in_dataset": total_samples,
            "total_frames_audited": total_frames,
            "lookback_window_hours": 15,
            "frames_per_sequence": 6,
            "thresholds_minutes": thresholds,
            "sample_index": str(sample_index_path),
            "alignment_manifest": str(align_path),
        },
        "frame_distribution": {
            "exact_matches_0min": int((df_align["delta_minutes"] == 0.0).sum()),
            "exact_matches_pct": float(round((df_align["delta_minutes"] == 0.0).mean() * 100.0, 2)),
            "offset_30min": int((df_align["delta_minutes"] == 30.0).sum()),
            "offset_30min_pct": float(round((df_align["delta_minutes"] == 30.0).mean() * 100.0, 2)),
            "offset_60min": int((df_align["delta_minutes"] == 60.0).sum()),
            "offset_60min_pct": float(round((df_align["delta_minutes"] == 60.0).mean() * 100.0, 2)),
        },
        "threshold_evaluations": {}
    }
    
    for thresh in thresholds:
        key = f"delta_t_leq_{thresh}min"
        
        # Frame-level retention
        valid_frames = df_align[df_align["delta_minutes"] <= thresh]
        rejected_frames_count = int(total_frames - len(valid_frames))
        rejected_frames_pct = float(round(rejected_frames_count / total_frames * 100.0, 2))
        
        # Sample-level retention (all 6 frames must be within tolerance)
        valid_samples = df_samples[df_samples["max_offset_minutes"] <= thresh]
        rejected_samples_count = int(total_samples - len(valid_samples))
        rejected_samples_pct = float(round(rejected_samples_count / total_samples * 100.0, 2))
        
        # Split breakdowns
        train_s = valid_samples[valid_samples["split"] == "TRAIN"]
        val_s = valid_samples[valid_samples["split"] == "VALIDATION"]
        test_s = valid_samples[valid_samples["split"] == "TEST"]
        
        train_storms = int(train_s["storm_id"].nunique())
        val_storms = int(val_s["storm_id"].nunique())
        test_storms = int(test_s["storm_id"].nunique())
        
        # Dropped storms relative to original paired dataset
        all_train_storms = set(df_samples[df_samples["split"] == "TRAIN"]["storm_id"])
        all_val_storms = set(df_samples[df_samples["split"] == "VALIDATION"]["storm_id"])
        all_test_storms = set(df_samples[df_samples["split"] == "TEST"]["storm_id"])
        
        dropped_train_storms = sorted(list(all_train_storms - set(train_s["storm_id"])))
        dropped_val_storms = sorted(list(all_val_storms - set(val_s["storm_id"])))
        dropped_test_storms = sorted(list(all_test_storms - set(test_s["storm_id"])))
        
        results["threshold_evaluations"][key] = {
            "threshold_minutes": thresh,
            "total_paired_samples": int(len(valid_samples)),
            "complete_six_frame_paired_samples": int(len(valid_samples)),
            "sample_retention_pct": float(round(len(valid_samples) / total_samples * 100.0, 2)),
            "rejected_samples": rejected_samples_count,
            "rejected_samples_pct": rejected_samples_pct,
            "total_observation_frames": int(len(valid_frames)),
            "rejected_frames": rejected_frames_count,
            "rejected_frames_pct": rejected_frames_pct,
            "splits": {
                "TRAIN": {
                    "samples": int(len(train_s)),
                    "original_paired_samples": 175,
                    "sample_retention_pct": float(round(len(train_s) / 175 * 100.0, 2)),
                    "unique_storms": train_storms,
                    "original_unique_storms": len(all_train_storms),
                    "dropped_storms": dropped_train_storms
                },
                "VALIDATION": {
                    "samples": int(len(val_s)),
                    "original_paired_samples": 252,
                    "sample_retention_pct": float(round(len(val_s) / 252 * 100.0, 2)),
                    "unique_storms": val_storms,
                    "original_unique_storms": len(all_val_storms),
                    "dropped_storms": dropped_val_storms
                },
                "TEST": {
                    "samples": int(len(test_s)),
                    "original_paired_samples": 298,
                    "sample_retention_pct": float(round(len(test_s) / 298 * 100.0, 2)),
                    "unique_storms": test_storms,
                    "original_unique_storms": len(all_test_storms),
                    "dropped_storms": dropped_test_storms
                }
            }
        }
        
    out_json = Path("data/interim/ml/multisource_temporal_tolerance_sensitivity.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sensitivity analysis to {out_json}")
    return results

if __name__ == "__main__":
    run_sensitivity_audit()
