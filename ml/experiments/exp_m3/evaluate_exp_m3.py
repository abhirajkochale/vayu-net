"""
VAYU-NET: EXP-M3 Test Set Evaluation, Paired Storm Statistics, Gate Analysis, and Plots.
Evaluates M3A, M3B, and M3C on the locked 371-sequence TEST set across all 31 storms.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.experiments.exp_m3.model import (
    ExpM3Model, haversine_km,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY
)

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
CHECKPOINT_DIR = REPO_ROOT / "ml/experiments/exp_m3/checkpoints"
RESULTS_DIR = REPO_ROOT / "ml/experiments/exp_m3/results"
PLOTS_DIR = RESULTS_DIR / "plots"


def run_evaluation():
    print("=" * 65)
    print("VAYU-NET: EXP-M3 FINAL TEST EVALUATION (371 SEQUENCES / 31 STORMS)")
    print("=" * 65)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE_PATH, map_location="cpu")
    test_samples = [s for s in cache["samples"] if s["split"] == "TEST"]
    assert len(test_samples) == 371, f"Expected 371 test samples, got {len(test_samples)}"

    test_storms = sorted(list(set(s["storm_id"] for s in test_samples)))
    assert len(test_storms) == 31, f"Expected 31 test storms, got {len(test_storms)}"
    print(f"Loaded TEST partition: {len(test_samples)} sequences across {len(test_storms)} unique storms.\n")

    models_info = [
        ("m3a_gridsat", "m3a_gridsat_only.pt"),
        ("m3b_imerg", "m3b_imerg_only.pt"),
        ("m3c_spatial", "m3c_spatial_adaptive_fusion.pt")
    ]

    all_results = {}
    per_model_sample_records = {}
    sample_spatial_gates = {} # For storm heatmap plotting

    for mode, ckpt_name in models_info:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print(f"--- Evaluating {mode.upper()} ({ckpt_name}) ---")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = ExpM3Model(mode=mode)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        sample_records = []
        latencies = []

        with torch.no_grad():
            for s in test_samples:
                g = s["gridsat"].unsqueeze(0) if mode in ["m3a_gridsat", "m3c_spatial"] else None
                i = s["imerg"].unsqueeze(0) if mode in ["m3b_imerg", "m3c_spatial"] else None

                t_start = time.perf_counter()
                preds = model(gridsat_seq=g, imerg_seq=i)
                latencies.append((time.perf_counter() - t_start) * 1000.0)

                pred_c = preds["center_deg"].squeeze(0).numpy()
                true_c = s["center_t0"].numpy()
                c_dpe = haversine_km(pred_c[0], pred_c[1], true_c[0], true_c[1])

                # Track DPEs
                t12_dpe, t24_dpe, t48_dpe = None, None, None
                if s["wind_12h_mask"].item() > 0.5:
                    p = preds["track_12h_deg"].squeeze(0).numpy()
                    t = s["center_12h"].numpy()
                    t12_dpe = haversine_km(p[0], p[1], t[0], t[1])
                if s["wind_24h_mask"].item() > 0.5:
                    p = preds["track_24h_deg"].squeeze(0).numpy()
                    t = s["center_24h"].numpy()
                    t24_dpe = haversine_km(p[0], p[1], t[0], t[1])
                if s["wind_48h_mask"].item() > 0.5:
                    p = preds["track_48h_deg"].squeeze(0).numpy()
                    t = s["center_48h"].numpy()
                    t48_dpe = haversine_km(p[0], p[1], t[0], t[1])

                # Wind
                pred_w = preds["wind_kt"].item()
                true_w = s["wind_t0"].item()
                w_err = abs(pred_w - true_w)

                # Intensity Category
                pred_cat = int(np.argmax(preds["class_logits"].squeeze(0).numpy()))
                true_cat = int(s["category_t0"].item())

                # Spatial Gate metrics for M3C
                gate_mean, gate_median, gate_p10, gate_p90, gate_var = None, None, None, None, None
                if "spatial_gate_alpha" in preds:
                    # shape: [1, 6, 64, 9, 15]
                    alpha_arr = preds["spatial_gate_alpha"].squeeze(0).numpy() # [6, 64, 9, 15]
                    # collapse across channels and frames for sequence summary
                    gate_mean = float(np.mean(alpha_arr))
                    gate_median = float(np.median(alpha_arr))
                    gate_p10 = float(np.percentile(alpha_arr, 10))
                    gate_p90 = float(np.percentile(alpha_arr, 90))
                    gate_var = float(np.mean(np.var(alpha_arr, axis=(-2, -1))))
                    sample_spatial_gates[s["sample_id"]] = {
                        "storm_id": s["storm_id"],
                        "t0_gate_map": np.mean(alpha_arr[-1], axis=0) # [9, 15] average over channels at t0
                    }

                sample_records.append({
                    "sample_id": s["sample_id"],
                    "storm_id": s["storm_id"],
                    "storm_name": s.get("storm_name", s["storm_id"]),
                    "t0_utc": s["t0_utc"],
                    "center_dpe_km": c_dpe,
                    "pred_center_lat": float(pred_c[0]),
                    "pred_center_lon": float(pred_c[1]),
                    "true_center_lat": float(true_c[0]),
                    "true_center_lon": float(true_c[1]),
                    "center_lat_ae": abs(float(pred_c[0]) - float(true_c[0])),
                    "center_lon_ae": abs(float(pred_c[1]) - float(true_c[1])),
                    "track_12h_dpe_km": t12_dpe,
                    "track_24h_dpe_km": t24_dpe,
                    "track_48h_dpe_km": t48_dpe,
                    "pred_wind_kt": pred_w,
                    "true_wind_kt": true_w,
                    "wind_ae_kt": w_err,
                    "wind_err_kt": pred_w - true_w,
                    "pred_cat": pred_cat,
                    "true_cat": true_cat,
                    "cat_correct": int(pred_cat == true_cat),
                    "gate_gridsat_mean": gate_mean,
                    "gate_imerg_mean": (1.0 - gate_mean) if gate_mean is not None else None,
                    "gate_median": gate_median,
                    "gate_p10": gate_p10,
                    "gate_p90": gate_p90,
                    "gate_spatial_variance": gate_var
                })

        per_model_sample_records[mode] = sample_records
        df_m = pd.DataFrame(sample_records)

        # Aggregate Metrics
        center_mean = float(df_m["center_dpe_km"].mean())
        center_median = float(df_m["center_dpe_km"].median())
        center_p90 = float(df_m["center_dpe_km"].quantile(0.90))
        lat_mae = float(df_m["center_lat_ae"].mean())
        lon_mae = float(df_m["center_lon_ae"].mean())

        wind_mae = float(df_m["wind_ae_kt"].mean())
        wind_rmse = float(np.sqrt((df_m["wind_err_kt"] ** 2).mean()))
        wind_median_ae = float(df_m["wind_ae_kt"].median())
        wind_p90_ae = float(df_m["wind_ae_kt"].quantile(0.90))
        wind_bias = float(df_m["wind_err_kt"].mean())
        r_val, _ = pearsonr(df_m["true_wind_kt"], df_m["pred_wind_kt"])
        wind_r = float(r_val)

        cat_acc = float(accuracy_score(df_m["true_cat"], df_m["pred_cat"]))
        cat_f1 = float(f1_score(df_m["true_cat"], df_m["pred_cat"], average="macro", zero_division=0))
        conf_mat = confusion_matrix(df_m["true_cat"], df_m["pred_cat"], labels=list(range(7))).tolist()

        t12_valid = df_m["track_12h_dpe_km"].dropna()
        t24_valid = df_m["track_24h_dpe_km"].dropna()
        t48_valid = df_m["track_48h_dpe_km"].dropna()

        t12_mean = float(t12_valid.mean()) if len(t12_valid) > 0 else 0.0
        t24_mean = float(t24_valid.mean()) if len(t24_valid) > 0 else 0.0
        t48_mean = float(t48_valid.mean()) if len(t48_valid) > 0 else 0.0
        track_agg = float(np.mean([t12_mean, t24_mean, t48_mean]))

        # Storm-level Aggregation
        storm_group = df_m.groupby("storm_id")
        storm_metrics = {}
        for s_id, s_df in storm_group:
            s_t12 = s_df["track_12h_dpe_km"].dropna()
            s_t24 = s_df["track_24h_dpe_km"].dropna()
            s_t48 = s_df["track_48h_dpe_km"].dropna()
            storm_metrics[s_id] = {
                "storm_name": str(s_df["storm_name"].iloc[0]),
                "num_sequences": int(len(s_df)),
                "center_mean_dpe_km": float(s_df["center_dpe_km"].mean()),
                "wind_mae_kt": float(s_df["wind_ae_kt"].mean()),
                "cat_accuracy": float(accuracy_score(s_df["true_cat"], s_df["pred_cat"])),
                "track_12h_mean_dpe_km": float(s_t12.mean()) if len(s_t12) > 0 else None,
                "track_24h_mean_dpe_km": float(s_t24.mean()) if len(s_t24) > 0 else None,
                "track_48h_mean_dpe_km": float(s_t48.mean()) if len(s_t48) > 0 else None,
            }
            valid_tracks = [x for x in [storm_metrics[s_id]["track_12h_mean_dpe_km"],
                                       storm_metrics[s_id]["track_24h_mean_dpe_km"],
                                       storm_metrics[s_id]["track_48h_mean_dpe_km"]] if x is not None]
            storm_metrics[s_id]["track_aggregate_dpe_km"] = float(np.mean(valid_tracks)) if valid_tracks else None

        # 1,000-Resample Storm-Level Bootstrap
        np.random.seed(42)
        n_boot = 1000
        storm_ids_arr = np.array(test_storms)
        boot_center, boot_wind, boot_acc, boot_f1 = [], [], [], []
        boot_t12, boot_t24, boot_t48, boot_track_agg = [], [], [], []

        for _ in range(n_boot):
            sampled_storms = np.random.choice(storm_ids_arr, size=len(storm_ids_arr), replace=True)
            sampled_df = df_m[df_m["storm_id"].isin(sampled_storms)]
            boot_center.append(sampled_df["center_dpe_km"].mean())
            boot_wind.append(sampled_df["wind_ae_kt"].mean())
            boot_acc.append(accuracy_score(sampled_df["true_cat"], sampled_df["pred_cat"]))
            boot_f1.append(f1_score(sampled_df["true_cat"], sampled_df["pred_cat"], average="macro", zero_division=0))

            b_t12 = sampled_df["track_12h_dpe_km"].dropna().mean()
            b_t24 = sampled_df["track_24h_dpe_km"].dropna().mean()
            b_t48 = sampled_df["track_48h_dpe_km"].dropna().mean()
            boot_t12.append(b_t12)
            boot_t24.append(b_t24)
            boot_t48.append(b_t48)
            boot_track_agg.append(np.mean([b_t12, b_t24, b_t48]))

        ci_center = [float(np.percentile(boot_center, 2.5)), float(np.percentile(boot_center, 97.5))]
        ci_wind = [float(np.percentile(boot_wind, 2.5)), float(np.percentile(boot_wind, 97.5))]
        ci_acc = [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]
        ci_f1 = [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))]
        ci_t12 = [float(np.percentile(boot_t12, 2.5)), float(np.percentile(boot_t12, 97.5))]
        ci_t24 = [float(np.percentile(boot_t24, 2.5)), float(np.percentile(boot_t24, 97.5))]
        ci_t48 = [float(np.percentile(boot_t48, 2.5)), float(np.percentile(boot_t48, 97.5))]
        ci_track_agg = [float(np.percentile(boot_track_agg, 2.5)), float(np.percentile(boot_track_agg, 97.5))]

        # Save Gate CSV for M3C
        gate_summary = None
        if mode == "m3c_spatial":
            gate_csv_path = RESULTS_DIR / "modality_gate_statistics.csv"
            df_gates = df_m[["sample_id", "storm_id", "t0_utc", "gate_gridsat_mean", "gate_imerg_mean",
                             "gate_median", "gate_p10", "gate_p90", "gate_spatial_variance"]].copy()
            df_gates.columns = ["sample_id", "storm_id", "t0_utc", "gridsat_spatial_mean", "imerg_spatial_mean",
                                "gate_median", "gate_p10", "gate_p90", "spatial_variance"]
            df_gates.to_csv(gate_csv_path, index=False)
            print(f"  Saved spatial modality gates CSV to: {gate_csv_path}")

            gate_summary = {
                "gridsat_weight_mean": float(df_gates["gridsat_spatial_mean"].mean()),
                "gridsat_weight_std": float(df_gates["gridsat_spatial_mean"].std()),
                "gridsat_weight_median": float(df_gates["gate_median"].median()),
                "gridsat_weight_p10": float(df_gates["gate_p10"].mean()),
                "gridsat_weight_p90": float(df_gates["gate_p90"].mean()),
                "spatial_variance_mean": float(df_gates["spatial_variance"].mean()),
                "imerg_weight_mean": float(df_gates["imerg_spatial_mean"].mean()),
                "imerg_weight_std": float(df_gates["imerg_spatial_mean"].std())
            }

        all_results[mode] = {
            "mode": mode,
            "checkpoint_name": ckpt_name,
            "best_validation_epoch": checkpoint["best_epoch"],
            "best_validation_loss": checkpoint["best_val_loss"],
            "training_time_seconds": checkpoint.get("training_time_seconds", 0.0),
            "parameter_counts": model.get_parameter_counts(),
            "inference_latency_ms_per_seq": float(np.mean(latencies)),
            "test_sample_count": len(df_m),
            "test_storm_count": len(storm_metrics),
            "test_aggregate_metrics": {
                "center": {
                    "mean_dpe_km": center_mean,
                    "median_dpe_km": center_median,
                    "p90_dpe_km": center_p90,
                    "latitude_mae_deg": lat_mae,
                    "longitude_mae_deg": lon_mae,
                    "ci_95": ci_center
                },
                "track": {
                    "track_12h_dpe_km": t12_mean,
                    "track_24h_dpe_km": t24_mean,
                    "track_48h_dpe_km": t48_mean,
                    "aggregate_dpe_km": track_agg,
                    "track_12h_ci_95": ci_t12,
                    "track_24h_ci_95": ci_t24,
                    "track_48h_ci_95": ci_t48,
                    "aggregate_ci_95": ci_track_agg
                },
                "intensity": {
                    "accuracy": cat_acc,
                    "macro_f1": cat_f1,
                    "confusion_matrix": conf_mat,
                    "accuracy_ci_95": ci_acc,
                    "macro_f1_ci_95": ci_f1
                },
                "wind": {
                    "mae_kt": wind_mae,
                    "rmse_kt": wind_rmse,
                    "median_ae_kt": wind_median_ae,
                    "p90_ae_kt": wind_p90_ae,
                    "bias_kt": wind_bias,
                    "pearson_r": wind_r,
                    "ci_95": ci_wind
                }
            },
            "gate_summary": gate_summary,
            "storm_level_metrics": storm_metrics
        }

        print(f"  Center Mean DPE:   {center_mean:.1f} km (95% CI: [{ci_center[0]:.1f}, {ci_center[1]:.1f}])")
        print(f"  Wind MAE:          {wind_mae:.2f} kt (95% CI: [{ci_wind[0]:.2f}, {ci_wind[1]:.2f}])")
        print(f"  Intensity Acc:     {cat_acc*100:.1f}% | Macro-F1: {cat_f1:.4f}")
        print(f"  Track Aggregate:   {track_agg:.1f} km (95% CI: [{ci_track_agg[0]:.1f}, {ci_track_agg[1]:.1f}])")
        print(f"  Latency:           {np.mean(latencies):.2f} ms / seq | Parameters: {model.get_parameter_counts()['trainable_parameters']:,}\n")

    # =========================================================================
    # Paired Storm-Level Bootstrap Differences: M3C - M3A
    # =========================================================================
    print("=" * 65)
    print("COMPUTING PAIRED STORM-LEVEL BOOTSTRAP DIFFERENCES: (M3C - M3A)")
    print("=" * 65)
    df_a = pd.DataFrame(per_model_sample_records["m3a_gridsat"])
    df_c = pd.DataFrame(per_model_sample_records["m3c_spatial"])

    paired_storm_diffs = []
    for s_id in test_storms:
        sub_a = df_a[df_a["storm_id"] == s_id]
        sub_c = df_c[df_c["storm_id"] == s_id]

        c_dpe_diff = sub_c["center_dpe_km"].mean() - sub_a["center_dpe_km"].mean()
        w_mae_diff = sub_c["wind_ae_kt"].mean() - sub_a["wind_ae_kt"].mean()
        acc_diff = accuracy_score(sub_c["true_cat"], sub_c["pred_cat"]) - accuracy_score(sub_a["true_cat"], sub_a["pred_cat"])

        t_agg_c = np.mean([sub_c["track_12h_dpe_km"].dropna().mean(),
                           sub_c["track_24h_dpe_km"].dropna().mean(),
                           sub_c["track_48h_dpe_km"].dropna().mean()])
        t_agg_a = np.mean([sub_a["track_12h_dpe_km"].dropna().mean(),
                           sub_a["track_24h_dpe_km"].dropna().mean(),
                           sub_a["track_48h_dpe_km"].dropna().mean()])
        t_dpe_diff = t_agg_c - t_agg_a

        paired_storm_diffs.append({
            "storm_id": s_id,
            "center_dpe_diff_km": c_dpe_diff,
            "wind_mae_diff_kt": w_mae_diff,
            "cat_acc_diff": acc_diff,
            "track_agg_diff_km": t_dpe_diff
        })

    df_paired = pd.DataFrame(paired_storm_diffs)

    np.random.seed(42)
    boot_paired_c, boot_paired_w, boot_paired_t, boot_paired_acc = [], [], [], []

    for _ in range(1000):
        sampled_rows = df_paired.sample(n=len(df_paired), replace=True)
        boot_paired_c.append(sampled_rows["center_dpe_diff_km"].mean())
        boot_paired_w.append(sampled_rows["wind_mae_diff_kt"].mean())
        boot_paired_t.append(sampled_rows["track_agg_diff_km"].mean())
        boot_paired_acc.append(sampled_rows["cat_acc_diff"].mean())

    paired_summary = {
        "center_dpe_diff_km": {
            "mean": float(np.mean(boot_paired_c)),
            "ci_95": [float(np.percentile(boot_paired_c, 2.5)), float(np.percentile(boot_paired_c, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_c, 2.5) <= 0 <= np.percentile(boot_paired_c, 97.5)),
            "favors": "M3A" if np.mean(boot_paired_c) > 0 else "M3C"
        },
        "wind_mae_diff_kt": {
            "mean": float(np.mean(boot_paired_w)),
            "ci_95": [float(np.percentile(boot_paired_w, 2.5)), float(np.percentile(boot_paired_w, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_w, 2.5) <= 0 <= np.percentile(boot_paired_w, 97.5)),
            "favors": "M3A" if np.mean(boot_paired_w) > 0 else "M3C"
        },
        "track_agg_diff_km": {
            "mean": float(np.mean(boot_paired_t)),
            "ci_95": [float(np.percentile(boot_paired_t, 2.5)), float(np.percentile(boot_paired_t, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_t, 2.5) <= 0 <= np.percentile(boot_paired_t, 97.5)),
            "favors": "M3A" if np.mean(boot_paired_t) > 0 else "M3C"
        },
        "cat_acc_diff": {
            "mean": float(np.mean(boot_paired_acc)),
            "ci_95": [float(np.percentile(boot_paired_acc, 2.5)), float(np.percentile(boot_paired_acc, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_acc, 2.5) <= 0 <= np.percentile(boot_paired_acc, 97.5)),
            "favors": "M3C" if np.mean(boot_paired_acc) > 0 else "M3A"
        }
    }

    all_results["paired_differences_m3c_minus_m3a"] = paired_summary

    print(f"  Center DPE Paired Diff: {paired_summary['center_dpe_diff_km']['mean']:+.1f} km "
          f"(95% CI: [{paired_summary['center_dpe_diff_km']['ci_95'][0]:+.1f}, {paired_summary['center_dpe_diff_km']['ci_95'][1]:+.1f}], "
          f"Zero in CI: {paired_summary['center_dpe_diff_km']['zero_in_ci']})")
    print(f"  Wind MAE Paired Diff:   {paired_summary['wind_mae_diff_kt']['mean']:+.2f} kt "
          f"(95% CI: [{paired_summary['wind_mae_diff_kt']['ci_95'][0]:+.2f}, {paired_summary['wind_mae_diff_kt']['ci_95'][1]:+.2f}], "
          f"Zero in CI: {paired_summary['wind_mae_diff_kt']['zero_in_ci']})")
    print(f"  Track Agg Paired Diff:  {paired_summary['track_agg_diff_km']['mean']:+.1f} km "
          f"(95% CI: [{paired_summary['track_agg_diff_km']['ci_95'][0]:+.1f}, {paired_summary['track_agg_diff_km']['ci_95'][1]:+.1f}], "
          f"Zero in CI: {paired_summary['track_agg_diff_km']['zero_in_ci']})")

    # =========================================================================
    # Failure Case Analysis
    # =========================================================================
    print("\n" + "=" * 65)
    print("IDENTIFYING FAILURE CASES (WORST STORMS IN M3C)")
    print("=" * 65)
    storm_metrics_m3c = all_results["m3c_spatial"]["storm_level_metrics"]
    sorted_by_center = sorted(storm_metrics_m3c.items(), key=lambda x: x[1]["center_mean_dpe_km"], reverse=True)
    sorted_by_track = sorted([x for x in storm_metrics_m3c.items() if x[1]["track_aggregate_dpe_km"] is not None],
                             key=lambda x: x[1]["track_aggregate_dpe_km"], reverse=True)

    worst_storms_center = [x[0] for x in sorted_by_center[:3]]
    worst_storms_track = [x[0] for x in sorted_by_track[:3]]
    worst_storm_ids = list(set(worst_storms_center + worst_storms_track))

    failure_cases = {}
    for sid in worst_storm_ids:
        sub_samples = [s for s in per_model_sample_records["m3c_spatial"] if s["storm_id"] == sid]
        failure_cases[sid] = {
            "storm_name": storm_metrics_m3c[sid]["storm_name"],
            "num_sequences": len(sub_samples),
            "center_mean_dpe_km": storm_metrics_m3c[sid]["center_mean_dpe_km"],
            "wind_mae_kt": storm_metrics_m3c[sid]["wind_mae_kt"],
            "track_aggregate_dpe_km": storm_metrics_m3c[sid]["track_aggregate_dpe_km"],
            "samples": [{
                "sample_id": s["sample_id"],
                "t0_utc": s["t0_utc"],
                "center_dpe_km": s["center_dpe_km"],
                "pred_center": [s["pred_center_lat"], s["pred_center_lon"]],
                "true_center": [s["true_center_lat"], s["true_center_lon"]],
                "pred_wind_kt": s["pred_wind_kt"],
                "true_wind_kt": s["true_wind_kt"],
                "pred_cat": s["pred_cat"],
                "true_cat": s["true_cat"],
                "track_12h_dpe_km": s["track_12h_dpe_km"],
                "track_24h_dpe_km": s["track_24h_dpe_km"],
                "track_48h_dpe_km": s["track_48h_dpe_km"]
            } for s in sub_samples]
        }
        print(f"  Worst Storm: {sid} ({storm_metrics_m3c[sid]['storm_name']}) | Center DPE: {storm_metrics_m3c[sid]['center_mean_dpe_km']:.1f} km | Track: {storm_metrics_m3c[sid]['track_aggregate_dpe_km']:.1f} km")

    all_results["failure_cases"] = failure_cases
    with open(RESULTS_DIR / "failure_cases.json", "w") as f:
        json.dump(failure_cases, f, indent=2)
    print(f"Saved failure cases to: {RESULTS_DIR / 'failure_cases.json'}")

    # Save results JSON
    results_path = RESULTS_DIR / "exp_m3_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved comprehensive test results to: {results_path}")

    # =========================================================================
    # Visualizations
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\nGenerating EXP-M3 research plots...")

        # Plot 1: Center DPE Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        modes_lbl = ["M3A: GridSat-only", "M3B: IMERG-only", "M3C: Spatial Fusion"]
        c_means = [all_results[m]["test_aggregate_metrics"]["center"]["mean_dpe_km"] for m in ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]]
        c_cis = [all_results[m]["test_aggregate_metrics"]["center"]["ci_95"] for m in ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]]
        yerr = [[c_means[i] - c_cis[i][0] for i in range(3)], [c_cis[i][1] - c_means[i] for i in range(3)]]
        ax.bar(modes_lbl, c_means, yerr=yerr, capsize=6, color=["#1b9e77", "#d95f02", "#7570b3"], alpha=0.85)
        ax.set_ylabel("Center Mean DPE (km)")
        ax.set_title("VAYU-NET EXP-M3: Cyclone Center Mean DPE (with 95% Storm Bootstrap CI)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m3_center_dpe_comparison.png", dpi=150)
        plt.close()

        # Plot 2: Wind MAE Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        w_means = [all_results[m]["test_aggregate_metrics"]["wind"]["mae_kt"] for m in ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]]
        w_cis = [all_results[m]["test_aggregate_metrics"]["wind"]["ci_95"] for m in ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]]
        yerr_w = [[w_means[i] - w_cis[i][0] for i in range(3)], [w_cis[i][1] - w_means[i] for i in range(3)]]
        ax.bar(modes_lbl, w_means, yerr=yerr_w, capsize=6, color=["#1b9e77", "#d95f02", "#7570b3"], alpha=0.85)
        ax.set_ylabel("Wind Speed MAE (knots)")
        ax.set_title("VAYU-NET EXP-M3: Maximum Sustained Wind MAE (with 95% Storm Bootstrap CI)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m3_wind_mae_comparison.png", dpi=150)
        plt.close()

        # Plot 3: Track DPE by Horizon
        fig, ax = plt.subplots(figsize=(9, 5))
        horizons = ["+12h", "+24h", "+48h", "Aggregate"]
        x = np.arange(len(horizons))
        w_bar = 0.25
        for idx, (m, lbl, col) in enumerate(zip(["m3a_gridsat", "m3b_imerg", "m3c_spatial"],
                                                ["M3A (GridSat)", "M3B (IMERG)", "M3C (Spatial Fusion)"],
                                                ["#1b9e77", "#d95f02", "#7570b3"])):
            vals = [
                all_results[m]["test_aggregate_metrics"]["track"]["track_12h_dpe_km"],
                all_results[m]["test_aggregate_metrics"]["track"]["track_24h_dpe_km"],
                all_results[m]["test_aggregate_metrics"]["track"]["track_48h_dpe_km"],
                all_results[m]["test_aggregate_metrics"]["track"]["aggregate_dpe_km"]
            ]
            ax.bar(x + idx * w_bar, vals, width=w_bar, label=lbl, color=col, alpha=0.85)
        ax.set_xticks(x + w_bar)
        ax.set_xticklabels(horizons)
        ax.set_ylabel("Track DPE (km)")
        ax.set_title("VAYU-NET EXP-M3: Multi-Horizon Track Forecast DPE (batch_size=16)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m3_track_dpe_by_horizon.png", dpi=150)
        plt.close()

        # Plot 4: Intensity Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (m, title) in enumerate(zip(["m3a_gridsat", "m3b_imerg", "m3c_spatial"],
                                             ["M3A (GridSat)", "M3B (IMERG)", "M3C (Spatial Fusion)"])):
            cm = np.array(all_results[m]["test_aggregate_metrics"]["intensity"]["confusion_matrix"])
            im = axes[idx].imshow(cm, cmap="Blues", interpolation="nearest")
            axes[idx].set_title(title)
            axes[idx].set_xlabel("Predicted Category")
            axes[idx].set_ylabel("True Category")
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    axes[idx].text(c, r, str(cm[r, c]), ha="center", va="center", color="black" if cm[r, c] < cm.max()/2 else "white", fontsize=7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m3_intensity_confusion_matrices.png", dpi=150)
        plt.close()

        # Plot 5: Paired Storm-Level Differences (M3C - M3A)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.scatter(range(len(df_paired)), df_paired["center_dpe_diff_km"], color="#7570b3", s=50, label="Center DPE Diff (km)")
        ax.scatter(range(len(df_paired)), df_paired["wind_mae_diff_kt"], color="#e7298a", s=50, label="Wind MAE Diff (kt)")
        ax.set_xticks(range(len(df_paired)))
        ax.set_xticklabels(df_paired["storm_id"].apply(lambda x: x.split("_")[-1]), rotation=90, fontsize=8)
        ax.set_ylabel("Difference: M3C - M3A")
        ax.set_title("VAYU-NET EXP-M3: Paired Per-Storm Metric Differences (M3C - M3A)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m3c_minus_m3a_paired_storm_differences.png", dpi=150)
        plt.close()

        # Plot 6 & 7: Spatial GridSat and IMERG Gate Maps for a Representative Storm (e.g. BIPARJOY or TAUKTAE)
        target_storm = "NIO_2023_BIPARJOY" if "NIO_2023_BIPARJOY" in test_storms else test_storms[0]
        sub_samples_g = [v for k, v in sample_spatial_gates.items() if v["storm_id"] == target_storm]
        if sub_samples_g:
            # Average spatial gate over sequences of this storm
            storm_avg_gate = np.mean([s["t0_gate_map"] for s in sub_samples_g], axis=0) # [9, 15]

            # 6. GridSat Contribution Map
            fig, ax = plt.subplots(figsize=(6, 4))
            c = ax.imshow(storm_avg_gate, cmap="YlGnBu", vmin=0.0, vmax=1.0, origin="lower")
            plt.colorbar(c, ax=ax, label="GridSat Contribution A_t")
            ax.set_title(f"M3C Spatial GridSat Contribution ({target_storm})")
            ax.set_xlabel("GridSat/IMERG X (Feature Map)")
            ax.set_ylabel("GridSat/IMERG Y (Feature Map)")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "m3c_spatial_gridsat_gate_map.png", dpi=150)
            plt.close()

            # 7. IMERG Contribution Map (1 - A_t)
            fig, ax = plt.subplots(figsize=(6, 4))
            c = ax.imshow(1.0 - storm_avg_gate, cmap="YlOrRd", vmin=0.0, vmax=1.0, origin="lower")
            plt.colorbar(c, ax=ax, label="IMERG Contribution (1 - A_t)")
            ax.set_title(f"M3C Spatial IMERG Contribution ({target_storm})")
            ax.set_xlabel("GridSat/IMERG X (Feature Map)")
            ax.set_ylabel("GridSat/IMERG Y (Feature Map)")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "m3c_spatial_imerg_gate_map.png", dpi=150)
            plt.close()

        # Plot 8: Gate-Value Distribution
        if all_results["m3c_spatial"]["gate_summary"] is not None:
            df_g = pd.read_csv(RESULTS_DIR / "modality_gate_statistics.csv")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_g["gridsat_spatial_mean"], bins=20, color="#1b9e77", alpha=0.7, label="GridSat Spatial Weight")
            ax.hist(df_g["imerg_spatial_mean"], bins=20, color="#d95f02", alpha=0.7, label="IMERG Spatial Weight")
            ax.set_xlabel("Mean Spatial Weight in [0, 1]")
            ax.set_ylabel("Sequence Count")
            ax.set_title("VAYU-NET EXP-M3: Learned Spatial Modality Gate Distribution on Test Set")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "m3c_gate_value_distribution.png", dpi=150)
            plt.close()

        print(f"  All research plots saved under: {PLOTS_DIR}")

    except Exception as pe:
        print(f"Plotting warning: {pe}")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
