"""
VAYU-NET: EXP-M2 Test Set Evaluation, Paired Storm Statistics, and Gate Analysis.
Evaluates M2A, M2B, and M2C on the locked 371-sequence TEST set across all 31 storms.
Computes:
  - Aggregate metrics across Center, Wind, Intensity, and Track tasks
  - Storm-level metrics for all 31 test storms
  - 1,000-resample storm-level bootstrap 95% CIs
  - Paired storm-level difference bootstrap distributions (M2C - M2A)
  - Modality gate statistics and per-sample gate CSV
  - Generates publication-ready research plots
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

from ml.experiments.exp_m2.model import (
    ExpM2Model, haversine_km,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY
)

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
CHECKPOINT_DIR = REPO_ROOT / "ml/experiments/exp_m2/checkpoints"
RESULTS_DIR = REPO_ROOT / "ml/experiments/exp_m2/results"
PLOTS_DIR = RESULTS_DIR / "plots"


def run_evaluation():
    print("=" * 65)
    print("VAYU-NET: EXP-M2 FINAL TEST EVALUATION (371 SEQUENCES / 31 STORMS)")
    print("=" * 65)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE_PATH, map_location="cpu")
    test_samples = [s for s in cache["samples"] if s["split"] == "TEST"]
    assert len(test_samples) == 371, f"Expected 371 test samples, got {len(test_samples)}"

    test_storms = sorted(list(set(s["storm_id"] for s in test_samples)))
    assert len(test_storms) == 31, f"Expected 31 test storms, got {len(test_storms)}"
    print(f"Loaded TEST partition: {len(test_samples)} sequences across {len(test_storms)} unique storms.\n")

    models_info = [
        ("m2a_gridsat", "m2a_gridsat_only.pt"),
        ("m2b_imerg", "m2b_imerg_only.pt"),
        ("m2c_adaptive", "m2c_adaptive_fusion.pt")
    ]

    all_results = {}
    per_model_sample_records = {}

    for mode, ckpt_name in models_info:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print(f"--- Evaluating {mode.upper()} ({ckpt_name}) ---")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = ExpM2Model(mode=mode)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        sample_records = []
        latencies = []

        with torch.no_grad():
            for s in test_samples:
                g = s["gridsat"].unsqueeze(0) if mode in ["m2a_gridsat", "m2c_adaptive"] else None
                i = s["imerg"].unsqueeze(0) if mode in ["m2b_imerg", "m2c_adaptive"] else None

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

                # Gate values (M2C only)
                gate_alpha_t0 = None
                gate_alpha_seq_mean = None
                if "gate_alpha" in preds:
                    alphas = preds["gate_alpha"].squeeze().numpy()  # [6]
                    gate_alpha_t0 = float(alphas[-1])
                    gate_alpha_seq_mean = float(np.mean(alphas))

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
                    "gate_gridsat_t0": gate_alpha_t0,
                    "gate_imerg_t0": (1.0 - gate_alpha_t0) if gate_alpha_t0 is not None else None,
                    "gate_gridsat_mean": gate_alpha_seq_mean,
                    "gate_imerg_mean": (1.0 - gate_alpha_seq_mean) if gate_alpha_seq_mean is not None else None,
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

        # Save Gate CSV for M2C
        gate_summary = None
        if mode == "m2c_adaptive":
            gate_csv_path = RESULTS_DIR / "modality_gates_test.csv"
            df_gates = df_m[["sample_id", "storm_id", "t0_utc", "gate_gridsat_t0", "gate_imerg_t0",
                             "gate_gridsat_mean", "gate_imerg_mean"]].copy()
            df_gates.columns = ["sample_id", "storm_id", "t0_utc", "gridsat_weight", "imerg_weight",
                                "gridsat_seq_mean", "imerg_seq_mean"]
            df_gates.to_csv(gate_csv_path, index=False)
            print(f"  Saved modality gates CSV to: {gate_csv_path}")

            gate_summary = {
                "gridsat_weight_mean": float(df_gates["gridsat_weight"].mean()),
                "gridsat_weight_std": float(df_gates["gridsat_weight"].std()),
                "gridsat_weight_median": float(df_gates["gridsat_weight"].median()),
                "gridsat_weight_p10": float(df_gates["gridsat_weight"].quantile(0.10)),
                "gridsat_weight_p90": float(df_gates["gridsat_weight"].quantile(0.90)),
                "imerg_weight_mean": float(df_gates["imerg_weight"].mean()),
                "imerg_weight_std": float(df_gates["imerg_weight"].std()),
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
    # Paired Storm-Level Bootstrap Differences: M2C - M2A
    # =========================================================================
    print("=" * 65)
    print("COMPUTING PAIRED STORM-LEVEL BOOTSTRAP DIFFERENCES: (M2C - M2A)")
    print("=" * 65)
    df_a = pd.DataFrame(per_model_sample_records["m2a_gridsat"])
    df_c = pd.DataFrame(per_model_sample_records["m2c_adaptive"])

    # Compute paired differences per storm
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

    # 1,000 resamples of paired storm differences
    np.random.seed(42)
    boot_paired_c = []
    boot_paired_w = []
    boot_paired_t = []
    boot_paired_acc = []

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
            "favors": "M2A" if np.mean(boot_paired_c) > 0 else "M2C"
        },
        "wind_mae_diff_kt": {
            "mean": float(np.mean(boot_paired_w)),
            "ci_95": [float(np.percentile(boot_paired_w, 2.5)), float(np.percentile(boot_paired_w, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_w, 2.5) <= 0 <= np.percentile(boot_paired_w, 97.5)),
            "favors": "M2A" if np.mean(boot_paired_w) > 0 else "M2C"
        },
        "track_agg_diff_km": {
            "mean": float(np.mean(boot_paired_t)),
            "ci_95": [float(np.percentile(boot_paired_t, 2.5)), float(np.percentile(boot_paired_t, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_t, 2.5) <= 0 <= np.percentile(boot_paired_t, 97.5)),
            "favors": "M2A" if np.mean(boot_paired_t) > 0 else "M2C"
        },
        "cat_acc_diff": {
            "mean": float(np.mean(boot_paired_acc)),
            "ci_95": [float(np.percentile(boot_paired_acc, 2.5)), float(np.percentile(boot_paired_acc, 97.5))],
            "zero_in_ci": bool(np.percentile(boot_paired_acc, 2.5) <= 0 <= np.percentile(boot_paired_acc, 97.5)),
            "favors": "M2C" if np.mean(boot_paired_acc) > 0 else "M2A"
        }
    }

    all_results["paired_differences_m2c_minus_m2a"] = paired_summary

    print(f"  Center DPE Paired Diff: {paired_summary['center_dpe_diff_km']['mean']:+.1f} km "
          f"(95% CI: [{paired_summary['center_dpe_diff_km']['ci_95'][0]:+.1f}, {paired_summary['center_dpe_diff_km']['ci_95'][1]:+.1f}], "
          f"Zero in CI: {paired_summary['center_dpe_diff_km']['zero_in_ci']})")
    print(f"  Wind MAE Paired Diff:   {paired_summary['wind_mae_diff_kt']['mean']:+.2f} kt "
          f"(95% CI: [{paired_summary['wind_mae_diff_kt']['ci_95'][0]:+.2f}, {paired_summary['wind_mae_diff_kt']['ci_95'][1]:+.2f}], "
          f"Zero in CI: {paired_summary['wind_mae_diff_kt']['zero_in_ci']})")
    print(f"  Track Agg Paired Diff:  {paired_summary['track_agg_diff_km']['mean']:+.1f} km "
          f"(95% CI: [{paired_summary['track_agg_diff_km']['ci_95'][0]:+.1f}, {paired_summary['track_agg_diff_km']['ci_95'][1]:+.1f}], "
          f"Zero in CI: {paired_summary['track_agg_diff_km']['zero_in_ci']})")

    # Save results JSON
    results_path = RESULTS_DIR / "exp_m2_results.json"
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

        print("\nGenerating EXP-M2 research plots...")

        # Plot 1: Center DPE Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        modes_lbl = ["M2A: GridSat-only", "M2B: IMERG-only", "M2C: Adaptive Fusion"]
        c_means = [all_results[m]["test_aggregate_metrics"]["center"]["mean_dpe_km"] for m in ["m2a_gridsat", "m2b_imerg", "m2c_adaptive"]]
        c_cis = [all_results[m]["test_aggregate_metrics"]["center"]["ci_95"] for m in ["m2a_gridsat", "m2b_imerg", "m2c_adaptive"]]
        yerr = [[c_means[i] - c_cis[i][0] for i in range(3)], [c_cis[i][1] - c_means[i] for i in range(3)]]
        bars = ax.bar(modes_lbl, c_means, yerr=yerr, capsize=6, color=["#2b5c8f", "#d95f02", "#7570b3"], alpha=0.85)
        ax.set_ylabel("Center Mean DPE (km)")
        ax.set_title("VAYU-NET EXP-M2: Cyclone Center Mean DPE (with 95% Storm Bootstrap CI)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m2_center_dpe_comparison.png", dpi=150)
        plt.close()

        # Plot 2: Wind MAE Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        w_means = [all_results[m]["test_aggregate_metrics"]["wind"]["mae_kt"] for m in ["m2a_gridsat", "m2b_imerg", "m2c_adaptive"]]
        w_cis = [all_results[m]["test_aggregate_metrics"]["wind"]["ci_95"] for m in ["m2a_gridsat", "m2b_imerg", "m2c_adaptive"]]
        yerr_w = [[w_means[i] - w_cis[i][0] for i in range(3)], [w_cis[i][1] - w_means[i] for i in range(3)]]
        ax.bar(modes_lbl, w_means, yerr=yerr_w, capsize=6, color=["#2b5c8f", "#d95f02", "#7570b3"], alpha=0.85)
        ax.set_ylabel("Wind Speed MAE (knots)")
        ax.set_title("VAYU-NET EXP-M2: Maximum Sustained Wind MAE (with 95% Storm Bootstrap CI)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m2_wind_mae_comparison.png", dpi=150)
        plt.close()

        # Plot 3: Track DPE by Horizon
        fig, ax = plt.subplots(figsize=(9, 5))
        horizons = ["+12h", "+24h", "+48h", "Aggregate"]
        x = np.arange(len(horizons))
        w_bar = 0.25
        for idx, (m, lbl, col) in enumerate(zip(["m2a_gridsat", "m2b_imerg", "m2c_adaptive"],
                                                ["M2A (GridSat)", "M2B (IMERG)", "M2C (Adaptive)"],
                                                ["#2b5c8f", "#d95f02", "#7570b3"])):
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
        ax.set_title("VAYU-NET EXP-M2: Multi-Horizon Track Forecast DPE")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m2_track_dpe_by_horizon.png", dpi=150)
        plt.close()

        # Plot 4: Modality Gate Distribution (M2C)
        if all_results["m2c_adaptive"]["gate_summary"] is not None:
            df_g = pd.read_csv(RESULTS_DIR / "modality_gates_test.csv")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df_g["gridsat_weight"], bins=20, color="#2b5c8f", alpha=0.7, label="GridSat Weight (alpha)")
            ax.hist(df_g["imerg_weight"], bins=20, color="#d95f02", alpha=0.7, label="IMERG Weight (1 - alpha)")
            ax.set_xlabel("Modality Weight in [0, 1]")
            ax.set_ylabel("Sequence Count")
            ax.set_title("VAYU-NET EXP-M2: Learned Modality Gate Distribution on Test Set")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "m2c_modality_gate_distribution.png", dpi=150)
            plt.close()

        # Plot 5: Paired Storm-Level Differences (M2C - M2A)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.scatter(range(len(df_paired)), df_paired["center_dpe_diff_km"], color="#7570b3", s=50, label="Center DPE Diff (km)")
        ax.scatter(range(len(df_paired)), df_paired["wind_mae_diff_kt"], color="#e7298a", s=50, label="Wind MAE Diff (kt)")
        ax.set_xticks(range(len(df_paired)))
        ax.set_xticklabels(df_paired["storm_id"].apply(lambda x: x.split("_")[-1]), rotation=90, fontsize=8)
        ax.set_ylabel("Difference: M2C - M2A")
        ax.set_title("VAYU-NET EXP-M2: Paired Per-Storm Metric Differences (M2C - M2A)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "m2c_minus_m2a_paired_storm_differences.png", dpi=150)
        plt.close()

        print(f"  All research plots saved under: {PLOTS_DIR}")

    except Exception as pe:
        print(f"Plotting warning: {pe}")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
