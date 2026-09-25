"""
VAYU-NET — MULTISOURCE REPRODUCIBILITY & STATISTICAL VALIDATION SUITE
=====================================================================
Performs:
  1. Detailed comparison of Previous EXP1 vs New Control Baseline across 24 dimensions
  2. Per-storm metric computation across all 24 TEST storms for all 7 model configurations
  3. Storm-level bootstrap uncertainty analysis (1,000 resamples, fixed seed=42)
  4. Per-storm improved vs. degraded counts (Arch C vs. Verified Baseline)
  5. Exports:
     - multisource_storm_level_metrics.csv
     - multisource_bootstrap_results.json
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.multisource_transfer_fusion import MultisourceTransferFusionModel
from ml.models.multisource_decoupled import MultisourceDecoupledModel, haversine_km
from ml.train.train_multisource_transfer_expanded import MultisourceTransferDataset
from ml.train.train_multisource_decoupled import MultisourceDecoupledDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_model_inference_on_test(model: nn.Module,
                                loader: DataLoader,
                                device: torch.device) -> List[Dict[str, Any]]:
    """Runs inference and collects sample-level predictions and ground truth."""
    model.eval()
    sample_records = []

    with torch.no_grad():
        for batch in loader:
            g_feat = batch["gridsat_feat_seq"].to(device)
            i_seq = batch["insat_seq"].to(device) if "insat_seq" in batch else None

            # Handle both model interfaces
            if isinstance(model, MultisourceTransferFusionModel):
                out = model(gridsat_feat_seq=g_feat, insat_seq=i_seq)
            else:
                out = model(gridsat_feat_seq=g_feat, insat_seq=i_seq)

            B = g_feat.shape[0]
            pred_centers = out["center_deg"].cpu().numpy()
            pred_winds = out["wind_kt"].cpu().numpy().squeeze(-1)
            pred_t12 = out["track_12h_deg"].cpu().numpy()
            pred_t24 = out["track_24h_deg"].cpu().numpy()
            pred_t48 = out["track_48h_deg"].cpu().numpy()

            gt_centers = batch["center_deg"].numpy()
            gt_winds = batch["wind_kt"].numpy().squeeze(-1)
            gt_t12 = batch["t12_deg"].numpy()
            m12 = batch["mask_12"].numpy()
            gt_t24 = batch["t24_deg"].numpy()
            m24 = batch["mask_24"].numpy()
            gt_t48 = batch["t48_deg"].numpy()
            m48 = batch["mask_48"].numpy()

            storm_ids = batch["storm_id"]
            sample_ids = batch["sample_id"]

            for i in range(B):
                # Center DPE
                dpe_c = haversine_km(pred_centers[i, 0], pred_centers[i, 1],
                                     gt_centers[i, 0], gt_centers[i, 1])

                # Wind diff
                w_err = float(pred_winds[i] - gt_winds[i])
                abs_w_err = abs(w_err)

                # Track DPEs
                dpe_12 = haversine_km(pred_t12[i, 0], pred_t12[i, 1],
                                      gt_t12[i, 0], gt_t12[i, 1]) if m12[i] > 0.5 else None
                dpe_24 = haversine_km(pred_t24[i, 0], pred_t24[i, 1],
                                      gt_t24[i, 0], gt_t24[i, 1]) if m24[i] > 0.5 else None
                dpe_48 = haversine_km(pred_t48[i, 0], pred_t48[i, 1],
                                      gt_t48[i, 0], gt_t48[i, 1]) if m48[i] > 0.5 else None

                track_dpes = [d for d in [dpe_12, dpe_24, dpe_48] if d is not None]
                agg_track = float(np.mean(track_dpes)) if track_dpes else None

                sample_records.append({
                    "sample_id": sample_ids[i],
                    "storm_id": storm_ids[i],
                    "center_dpe_km": dpe_c,
                    "wind_err_kt": w_err,
                    "abs_wind_err_kt": abs_w_err,
                    "track_12h_dpe_km": dpe_12,
                    "track_24h_dpe_km": dpe_24,
                    "track_48h_dpe_km": dpe_48,
                    "track_agg_dpe_km": agg_track
                })

    return sample_records


def compute_storm_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Groups sample records by storm_id and calculates per-storm aggregates."""
    df = pd.DataFrame(records)
    storms = df["storm_id"].unique()
    storm_metrics = {}

    for s in storms:
        sdf = df[df["storm_id"] == s]
        c_dpes = sdf["center_dpe_km"].values
        w_errs = sdf["wind_err_kt"].values
        abs_w = sdf["abs_wind_err_kt"].values

        t12 = sdf["track_12h_dpe_km"].dropna().values
        t24 = sdf["track_24h_dpe_km"].dropna().values
        t48 = sdf["track_48h_dpe_km"].dropna().values
        tagg = sdf["track_agg_dpe_km"].dropna().values

        storm_metrics[s] = {
            "n_samples": len(sdf),
            "center_mean_dpe_km": float(np.mean(c_dpes)),
            "center_median_dpe_km": float(np.median(c_dpes)),
            "wind_mae_kt": float(np.mean(abs_w)),
            "wind_bias_kt": float(np.mean(w_errs)),
            "track_12h_dpe_km": float(np.mean(t12)) if len(t12) > 0 else np.nan,
            "track_24h_dpe_km": float(np.mean(t24)) if len(t24) > 0 else np.nan,
            "track_48h_dpe_km": float(np.mean(t48)) if len(t48) > 0 else np.nan,
            "track_agg_dpe_km": float(np.mean(tagg)) if len(tagg) > 0 else np.nan
        }

    return storm_metrics


def summarize_across_storms(storm_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Computes Mean, Median, Std, Min, Max across all 24 storms."""
    keys = ["center_mean_dpe_km", "center_median_dpe_km", "wind_mae_kt", "wind_bias_kt",
            "track_12h_dpe_km", "track_24h_dpe_km", "track_48h_dpe_km", "track_agg_dpe_km"]
    summary = {}

    for k in keys:
        vals = [s[k] for s in storm_dict.values() if not np.isnan(s[k])]
        if vals:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            }
        else:
            summary[k] = {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

    return summary


def run_storm_bootstrap(records: List[Dict[str, Any]],
                        n_bootstrap: int = 1000,
                        seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Performs storm-level block bootstrap.
    Resamples storms with replacement to maintain intra-storm correlation.
    """
    np.random.seed(seed)
    df = pd.DataFrame(records)
    storms = np.array(df["storm_id"].unique())
    n_storms = len(storms)

    # Pre-group samples by storm
    storm_groups = {s: df[df["storm_id"] == s] for s in storms}

    # Point estimates on full dataset
    pe_center = float(df["center_dpe_km"].mean())
    pe_wind = float(df["abs_wind_err_kt"].mean())
    pe_track = float(df["track_agg_dpe_km"].dropna().mean())

    bs_center = []
    bs_wind = []
    bs_track = []

    for _ in range(n_bootstrap):
        sampled_storms = np.random.choice(storms, size=n_storms, replace=True)
        sampled_dfs = [storm_groups[s] for s in sampled_storms]
        b_df = pd.concat(sampled_dfs, ignore_index=True)

        bs_center.append(float(b_df["center_dpe_km"].mean()))
        bs_wind.append(float(b_df["abs_wind_err_kt"].mean()))
        t_vals = b_df["track_agg_dpe_km"].dropna()
        if len(t_vals) > 0:
            bs_track.append(float(t_vals.mean()))

    return {
        "center_mean_dpe_km": {
            "point_estimate": pe_center,
            "ci_95_lower": float(np.percentile(bs_center, 2.5)),
            "ci_95_upper": float(np.percentile(bs_center, 97.5)),
            "bootstrap_std": float(np.std(bs_center))
        },
        "wind_mae_kt": {
            "point_estimate": pe_wind,
            "ci_95_lower": float(np.percentile(bs_wind, 2.5)),
            "ci_95_upper": float(np.percentile(bs_wind, 97.5)),
            "bootstrap_std": float(np.std(bs_wind))
        },
        "track_agg_dpe_km": {
            "point_estimate": pe_track,
            "ci_95_lower": float(np.percentile(bs_track, 2.5)),
            "ci_95_upper": float(np.percentile(bs_track, 97.5)),
            "bootstrap_std": float(np.std(bs_track))
        }
    }


def main():
    print("=" * 80)
    print("VAYU-NET MULTISOURCE REPRODUCIBILITY & STATISTICAL AUDIT RUNNER")
    print("=" * 80)

    device = torch.device("cpu")
    cache_path = PROJECT_ROOT / "data/interim/ml/cache/multisource_transfer_cache_expanded.pt"
    assert cache_path.exists()
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    test_ds = MultisourceDecoupledDataset(cache_data, "TEST")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    print(f"Loaded {len(test_ds)} TEST samples across 24 test storms.")

    # 1. Models to evaluate
    model_configs = [
        ("Previous EXP1 (Verified)", "transfer", "data/interim/ml/checkpoints/multisource_transfer_exp1_expanded.pt", "exp1_gridsat_frozen"),
        ("New Control Baseline", "decoupled", "data/interim/ml/checkpoints/multisource_decoupled_baseline.pt", "baseline_gridsat"),
        ("Arch A: Decoupled Track", "decoupled", "data/interim/ml/checkpoints/multisource_decoupled_track.pt", "arch_a_decoupled_track"),
        ("Arch B: Decoupled Intensity", "decoupled", "data/interim/ml/checkpoints/multisource_decoupled_intensity.pt", "arch_b_decoupled_intensity"),
        ("Arch Center: Decoupled Center", "decoupled", "data/interim/ml/checkpoints/multisource_decoupled_center.pt", "decoupled_center"),
        ("Ablation I2: Intensity (TIR1+2)", "decoupled", "data/interim/ml/checkpoints/multisource_decoupled_intensity_tir.pt", "ablation_i2_decoupled_intensity_tir"),
        ("Arch C: Decoupled Full", "decoupled", "data/interim/ml/checkpoints/multisource_decoupled_full.pt", "arch_c_decoupled_full")
    ]

    all_records = {}
    storm_summaries = {}
    across_storm_summaries = {}
    bootstrap_results = {}

    for name, mtype, cpath, mode in model_configs:
        print(f"\nEvaluating: {name} from {cpath}...")
        ckpt = torch.load(PROJECT_ROOT / cpath, map_location="cpu", weights_only=False)

        if mtype == "transfer":
            model = MultisourceTransferFusionModel(mode=mode).to(device)
        else:
            model = MultisourceDecoupledModel(mode=mode).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        records = run_model_inference_on_test(model, test_loader, device)
        all_records[name] = records

        s_summary = compute_storm_summary(records)
        storm_summaries[name] = s_summary
        across_storm_summaries[name] = summarize_across_storms(s_summary)

        # Bootstrap
        bs = run_storm_bootstrap(records, n_bootstrap=1000, seed=42)
        bootstrap_results[name] = bs
        print(f"  Point Estimates: Center={bs['center_mean_dpe_km']['point_estimate']:.1f} km, "
              f"Wind={bs['wind_mae_kt']['point_estimate']:.2f} kt, "
              f"Track={bs['track_agg_dpe_km']['point_estimate']:.1f} km")
        print(f"  Bootstrap 95% CIs:")
        print(f"    Center: [{bs['center_mean_dpe_km']['ci_95_lower']:.1f}, {bs['center_mean_dpe_km']['ci_95_upper']:.1f}] km")
        print(f"    Wind:   [{bs['wind_mae_kt']['ci_95_lower']:.2f}, {bs['wind_mae_kt']['ci_95_upper']:.2f}] kt")
        print(f"    Track:  [{bs['track_agg_dpe_km']['ci_95_lower']:.1f}, {bs['track_agg_dpe_km']['ci_95_upper']:.1f}] km")

    # 2. Export per-storm table to CSV
    # Structure: Storm_ID, n_samples, then for each model: Center_Mean, Track_Agg, Wind_MAE
    storms = list(storm_summaries["Previous EXP1 (Verified)"].keys())
    storm_rows = []

    for s in storms:
        row = {"storm_id": s, "n_samples": storm_summaries["Previous EXP1 (Verified)"][s]["n_samples"]}
        for name in storm_summaries:
            prefix = name.split(":")[0].replace(" ", "_")
            m = storm_summaries[name][s]
            row[f"{prefix}_center_mean_dpe"] = round(m["center_mean_dpe_km"], 2)
            row[f"{prefix}_center_median_dpe"] = round(m["center_median_dpe_km"], 2)
            row[f"{prefix}_track_12h_dpe"] = round(m["track_12h_dpe_km"], 2) if not np.isnan(m["track_12h_dpe_km"]) else None
            row[f"{prefix}_track_24h_dpe"] = round(m["track_24h_dpe_km"], 2) if not np.isnan(m["track_24h_dpe_km"]) else None
            row[f"{prefix}_track_48h_dpe"] = round(m["track_48h_dpe_km"], 2) if not np.isnan(m["track_48h_dpe_km"]) else None
            row[f"{prefix}_track_agg_dpe"] = round(m["track_agg_dpe_km"], 2) if not np.isnan(m["track_agg_dpe_km"]) else None
            row[f"{prefix}_wind_mae"] = round(m["wind_mae_kt"], 2)
            row[f"{prefix}_wind_bias"] = round(m["wind_bias_kt"], 2)
        storm_rows.append(row)

    df_storm = pd.DataFrame(storm_rows)
    csv_path = PROJECT_ROOT / "data/interim/ml/multisource_storm_level_metrics.csv"
    df_storm.to_csv(csv_path, index=False)
    print(f"\nSaved storm-level metrics CSV to {csv_path}")

    # 3. Export bootstrap JSON
    json_path = PROJECT_ROOT / "data/interim/ml/multisource_bootstrap_results.json"
    bootstrap_out = {
        "metadata": {
            "title": "VAYU-NET Multi-Source Storm-Level Bootstrap Uncertainty Analysis",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_bootstrap": 1000,
            "seed": 42,
            "test_storms": 24,
            "test_samples": 298
        },
        "bootstrap_uncertainty": bootstrap_results,
        "across_storm_aggregates": across_storm_summaries
    }

    # 4. Count improved vs degraded storms: Arch C vs Previous EXP1 (Verified)
    arch_c_storms = storm_summaries["Arch C: Decoupled Full"]
    exp1_storms = storm_summaries["Previous EXP1 (Verified)"]
    new_ctrl_storms = storm_summaries["New Control Baseline"]

    comparison_counts = {
        "Arch_C_vs_Previous_EXP1": {
            "center_improved": sum(1 for s in storms if arch_c_storms[s]["center_mean_dpe_km"] < exp1_storms[s]["center_mean_dpe_km"]),
            "center_degraded": sum(1 for s in storms if arch_c_storms[s]["center_mean_dpe_km"] > exp1_storms[s]["center_mean_dpe_km"]),
            "center_tied": sum(1 for s in storms if arch_c_storms[s]["center_mean_dpe_km"] == exp1_storms[s]["center_mean_dpe_km"]),
            "track_improved": sum(1 for s in storms if arch_c_storms[s]["track_agg_dpe_km"] < exp1_storms[s]["track_agg_dpe_km"]),
            "track_degraded": sum(1 for s in storms if arch_c_storms[s]["track_agg_dpe_km"] > exp1_storms[s]["track_agg_dpe_km"]),
            "track_tied": sum(1 for s in storms if arch_c_storms[s]["track_agg_dpe_km"] == exp1_storms[s]["track_agg_dpe_km"]),
            "wind_improved": sum(1 for s in storms if arch_c_storms[s]["wind_mae_kt"] < exp1_storms[s]["wind_mae_kt"]),
            "wind_degraded": sum(1 for s in storms if arch_c_storms[s]["wind_mae_kt"] > exp1_storms[s]["wind_mae_kt"]),
            "wind_tied": sum(1 for s in storms if arch_c_storms[s]["wind_mae_kt"] == exp1_storms[s]["wind_mae_kt"]),
        },
        "Arch_C_vs_New_Control": {
            "center_improved": sum(1 for s in storms if arch_c_storms[s]["center_mean_dpe_km"] < new_ctrl_storms[s]["center_mean_dpe_km"]),
            "center_degraded": sum(1 for s in storms if arch_c_storms[s]["center_mean_dpe_km"] > new_ctrl_storms[s]["center_mean_dpe_km"]),
            "center_tied": sum(1 for s in storms if arch_c_storms[s]["center_mean_dpe_km"] == new_ctrl_storms[s]["center_mean_dpe_km"]),
            "track_improved": sum(1 for s in storms if arch_c_storms[s]["track_agg_dpe_km"] < new_ctrl_storms[s]["track_agg_dpe_km"]),
            "track_degraded": sum(1 for s in storms if arch_c_storms[s]["track_agg_dpe_km"] > new_ctrl_storms[s]["track_agg_dpe_km"]),
            "track_tied": sum(1 for s in storms if arch_c_storms[s]["track_agg_dpe_km"] == new_ctrl_storms[s]["track_agg_dpe_km"]),
            "wind_improved": sum(1 for s in storms if arch_c_storms[s]["wind_mae_kt"] < new_ctrl_storms[s]["wind_mae_kt"]),
            "wind_degraded": sum(1 for s in storms if arch_c_storms[s]["wind_mae_kt"] > new_ctrl_storms[s]["wind_mae_kt"]),
            "wind_tied": sum(1 for s in storms if arch_c_storms[s]["wind_mae_kt"] == new_ctrl_storms[s]["wind_mae_kt"]),
        }
    }
    bootstrap_out["improved_vs_degraded_storm_counts"] = comparison_counts

    with open(json_path, "w") as f:
        json.dump(bootstrap_out, f, indent=2)
    print(f"Saved bootstrap results JSON to {json_path}")

    print("\n--- Improved vs. Degraded Storm Counts (out of 24 storms) ---")
    print("Arch C vs. Previous EXP1 (Verified):")
    print(f"  Center: Improved={comparison_counts['Arch_C_vs_Previous_EXP1']['center_improved']}, "
          f"Degraded={comparison_counts['Arch_C_vs_Previous_EXP1']['center_degraded']}")
    print(f"  Track:  Improved={comparison_counts['Arch_C_vs_Previous_EXP1']['track_improved']}, "
          f"Degraded={comparison_counts['Arch_C_vs_Previous_EXP1']['track_degraded']}")
    print(f"  Wind:   Improved={comparison_counts['Arch_C_vs_Previous_EXP1']['wind_improved']}, "
          f"Degraded={comparison_counts['Arch_C_vs_Previous_EXP1']['wind_degraded']}")

    print("\nArch C vs. New Control Baseline:")
    print(f"  Center: Improved={comparison_counts['Arch_C_vs_New_Control']['center_improved']}, "
          f"Degraded={comparison_counts['Arch_C_vs_New_Control']['center_degraded']}")
    print(f"  Track:  Improved={comparison_counts['Arch_C_vs_New_Control']['track_improved']}, "
          f"Degraded={comparison_counts['Arch_C_vs_New_Control']['track_degraded']}")
    print(f"  Wind:   Improved={comparison_counts['Arch_C_vs_New_Control']['wind_improved']}, "
          f"Degraded={comparison_counts['Arch_C_vs_New_Control']['wind_degraded']}")


if __name__ == "__main__":
    main()
