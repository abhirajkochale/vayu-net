"""
VAYU-NET — MULTI-SOURCE SATELLITE FAIR ABLATION SUITE
======================================================
Executes the complete fair ablation experiment across:
  - Model A: GridSat Only (6-frame 1-channel sequence)
  - Model B: INSAT Only (6-frame 3-channel sequence: TIR1, TIR2, WV)
  - Model C: GridSat + INSAT Fusion (Joint multi-source sequence)
  - Baselines: Persistence & Constant Velocity on identical TEST set

Outputs:
  - Checkpoints: data/interim/ml/checkpoints/multisource_*.pt
  - Results JSON: data/interim/ml/multisource_model_results.json
  - Ablation JSON: data/interim/ml/multisource_ablation_results.json
  - Training Curves: docs/figures/multisource_model/*.png
"""

import os
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from ml.train.train_multisource_fusion import (
    train_multisource_experiment,
    haversine_km,
    set_seed
)


def compute_baselines_on_test(sample_index_path: str = "data/manifests/vayu_net_multisource_sample_index.csv",
                              imd_path: str = "data/processed/imd_best_track_v2.csv") -> Dict[str, Any]:
    """
    Computes Persistence and Constant Velocity baselines strictly on the
    identical 298 TEST samples.
    """
    df_samples = pd.read_csv(sample_index_path)
    df_test = df_samples[df_samples["split"] == "TEST"].copy()
    assert len(df_test) == 298, f"Expected 298 test samples, found {len(df_test)}"

    df_imd = pd.read_csv(imd_path)
    df_imd["dt"] = pd.to_datetime(df_imd["timestamp_utc"])
    obs_by_storm = {}
    for sid, group in df_imd.groupby("storm_id"):
        obs_by_storm[sid] = group.sort_values("dt").reset_index(drop=True)

    # 1. Track Baselines
    p_dpes_12, p_dpes_24, p_dpes_48 = [], [], []
    cv_dpes_12, cv_dpes_24, cv_dpes_48 = [], [], []

    # 2. Wind & Intensity Baselines
    true_winds, pred_winds_pers = [], []
    true_cats, pred_cats_pers = [], []

    for _, row in df_test.iterrows():
        sid = row["storm_id"]
        t0_dt = pd.to_datetime(row["t0"])
        lat0 = float(row["current_center_lat"])
        lon0 = float(row["current_center_lon"])
        w0 = float(row["current_wind_kt"])
        c0 = row["current_category"]

        # Track targets
        if row["has_target_12h"]:
            t_lat12 = float(row["target_center_12h_lat"])
            t_lon12 = float(row["target_center_12h_lon"])
            p_dpes_12.append(haversine_km(lat0, lon0, t_lat12, t_lon12))
        if row["has_target_24h"]:
            t_lat24 = float(row["target_center_24h_lat"])
            t_lon24 = float(row["target_center_24h_lon"])
            p_dpes_24.append(haversine_km(lat0, lon0, t_lat24, t_lon24))
        if row["has_target_48h"]:
            t_lat48 = float(row["target_center_48h_lat"])
            t_lon48 = float(row["target_center_48h_lon"])
            p_dpes_48.append(haversine_km(lat0, lon0, t_lat48, t_lon48))

        # Kinematic velocity derivation
        s_obs = obs_by_storm.get(sid)
        v_lat, v_lon = 0.0, 0.0
        w_prior, c_prior = w0, c0

        if s_obs is not None:
            priors = s_obs[s_obs["dt"] < t0_dt]
            if len(priors) > 0:
                latest = priors.iloc[-1]
                dt_h = (t0_dt - latest["dt"]).total_seconds() / 3600.0
                if dt_h > 0:
                    v_lat = (lat0 - latest["latitude"]) / dt_h
                    d_lon = lon0 - latest["longitude"]
                    if d_lon > 180.0: d_lon -= 360.0
                    elif d_lon < -180.0: d_lon += 360.0
                    v_lon = d_lon / dt_h
                w_prior = latest["maximum_sustained_wind_kt"]
                c_prior = latest["category"]

        true_winds.append(w0)
        pred_winds_pers.append(w_prior)
        true_cats.append(c0)
        pred_cats_pers.append(c_prior)

        # Constant velocity extrapolation
        if row["has_target_12h"]:
            cv_lat12 = lat0 + v_lat * 12.0
            cv_lon12 = lon0 + v_lon * 12.0
            cv_dpes_12.append(haversine_km(cv_lat12, cv_lon12, float(row["target_center_12h_lat"]), float(row["target_center_12h_lon"])))
        if row["has_target_24h"]:
            cv_lat24 = lat0 + v_lat * 24.0
            cv_lon24 = lon0 + v_lon * 24.0
            cv_dpes_24.append(haversine_km(cv_lat24, cv_lon24, float(row["target_center_24h_lat"]), float(row["target_center_24h_lon"])))
        if row["has_target_48h"]:
            cv_lat48 = lat0 + v_lat * 48.0
            cv_lon48 = lon0 + v_lon * 48.0
            cv_dpes_48.append(haversine_km(cv_lat48, cv_lon48, float(row["target_center_48h_lat"]), float(row["target_center_48h_lon"])))

    # Compute Wind Persistence Metrics
    true_w_arr = np.array(true_winds)
    pred_w_arr = np.array(pred_winds_pers)
    wind_diff = pred_w_arr - true_w_arr
    abs_wind_diff = np.abs(wind_diff)

    # Compute Classification Persistence Metrics
    cat_acc = float(accuracy_score(true_cats, pred_cats_pers))
    cat_macro_f1 = float(f1_score(true_cats, pred_cats_pers, average="macro", zero_division=0))

    return {
        "persistence_baseline": {
            "wind": {
                "wind_mae_kt": float(np.mean(abs_wind_diff)),
                "wind_rmse_kt": float(np.sqrt(np.mean(wind_diff ** 2))),
                "wind_median_ae_kt": float(np.median(abs_wind_diff)),
                "wind_p90_ae_kt": float(np.percentile(abs_wind_diff, 90)),
                "wind_bias_kt": float(np.mean(wind_diff)),
                "wind_pearson_r": float(np.corrcoef(pred_w_arr, true_w_arr)[0, 1]) if np.std(pred_w_arr) > 0 else 0.0
            },
            "classification": {
                "accuracy": cat_acc,
                "macro_f1": cat_macro_f1
            },
            "track": {
                "track_12h_mean_dpe_km": float(np.mean(p_dpes_12)),
                "track_12h_median_dpe_km": float(np.median(p_dpes_12)),
                "track_12h_p90_dpe_km": float(np.percentile(p_dpes_12, 90)),
                "track_24h_mean_dpe_km": float(np.mean(p_dpes_24)),
                "track_24h_median_dpe_km": float(np.median(p_dpes_24)),
                "track_24h_p90_dpe_km": float(np.percentile(p_dpes_24, 90)),
                "track_48h_mean_dpe_km": float(np.mean(p_dpes_48)),
                "track_48h_median_dpe_km": float(np.median(p_dpes_48)),
                "track_48h_p90_dpe_km": float(np.percentile(p_dpes_48, 90)),
                "track_aggregate_mean_dpe_km": float(np.mean(p_dpes_12 + p_dpes_24 + p_dpes_48))
            }
        },
        "constant_velocity_baseline": {
            "track": {
                "track_12h_mean_dpe_km": float(np.mean(cv_dpes_12)),
                "track_12h_median_dpe_km": float(np.median(cv_dpes_12)),
                "track_12h_p90_dpe_km": float(np.percentile(cv_dpes_12, 90)),
                "track_24h_mean_dpe_km": float(np.mean(cv_dpes_24)),
                "track_24h_median_dpe_km": float(np.median(cv_dpes_24)),
                "track_24h_p90_dpe_km": float(np.percentile(cv_dpes_24, 90)),
                "track_48h_mean_dpe_km": float(np.mean(cv_dpes_48)),
                "track_48h_median_dpe_km": float(np.median(cv_dpes_48)),
                "track_48h_p90_dpe_km": float(np.percentile(cv_dpes_48, 90)),
                "track_aggregate_mean_dpe_km": float(np.mean(cv_dpes_12 + cv_dpes_24 + cv_dpes_48))
            }
        }
    }


def plot_training_curves(results: Dict[str, Any], output_dir: str = "docs/figures/multisource_model"):
    """Generates train-vs-validation loss and metric comparison curves."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    colors = {"gridsat": "#1f77b4", "insat": "#ff7f0e", "fusion": "#2ca02c"}
    labels = {"gridsat": "Model A (GridSat Only)", "insat": "Model B (INSAT Only)", "fusion": "Model C (Fusion)"}

    # 1. Total Loss (Train vs Val)
    ax = axes[0, 0]
    for mode in ["gridsat", "insat", "fusion"]:
        hist = results[mode]["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], linestyle="--", alpha=0.6, color=colors[mode])
        ax.plot(epochs, [h["val_loss"] for h in hist], linestyle="-", lw=2, color=colors[mode], label=f"{labels[mode]} (Val)")
    ax.set_title("Total Multi-Task Loss (Train vs. Val)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 2. Center DPE
    ax = axes[0, 1]
    for mode in ["gridsat", "insat", "fusion"]:
        hist = results[mode]["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_center_dpe_km"] for h in hist], marker="o", color=colors[mode], label=labels[mode])
    ax.set_title("Validation Center Mean DPE (km)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPE (km)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 3. Wind MAE
    ax = axes[1, 0]
    for mode in ["gridsat", "insat", "fusion"]:
        hist = results[mode]["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_wind_mae_kt"] for h in hist], marker="s", color=colors[mode], label=labels[mode])
    ax.set_title("Validation Wind MAE (kt)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (kt)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 4. Track Aggregate DPE
    ax = axes[1, 1]
    for mode in ["gridsat", "insat", "fusion"]:
        hist = results[mode]["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_track_dpe_km"] for h in hist], marker="^", color=colors[mode], label=labels[mode])
    ax.set_title("Validation Track Aggregate Mean DPE (km)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPE (km)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    out_file = Path(output_dir) / "multisource_training_curves.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {out_file}")


def run_fair_ablation():
    cache_path = "data/interim/ml/cache/multisource_dataset_cache.pt"
    assert os.path.exists(cache_path), f"Cache missing at {cache_path}. Run build_multisource_cache first."

    print(f"Loading cached dataset from {cache_path}...")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    device = torch.device("cpu")
    print(f"Executing fair ablation on device: {device}")

    results = {}
    checkpoints = {
        "gridsat": "data/interim/ml/checkpoints/multisource_gridsat_model_a.pt",
        "insat": "data/interim/ml/checkpoints/multisource_insat_model_b.pt",
        "fusion": "data/interim/ml/checkpoints/multisource_fusion_model_c.pt"
    }

    # Train Model A (GridSat), Model B (INSAT), Model C (Fusion)
    for mode in ["gridsat", "insat", "fusion"]:
        res = train_multisource_experiment(
            mode=mode,
            cache_data=cache_data,
            device=device,
            epochs=20,
            batch_size=16,
            lr=1e-3,
            patience=5,
            checkpoint_path=checkpoints[mode]
        )
        results[mode] = res

    # Baselines on identical TEST set
    baselines = compute_baselines_on_test()
    results["baselines"] = baselines

    # Compute differences: Fusion - GridSat (Absolute and Percentage)
    m_a = results["gridsat"]["test_metrics"]
    m_b = results["insat"]["test_metrics"]
    m_c = results["fusion"]["test_metrics"]

    def calc_diff(val_c: float, val_a: float) -> Dict[str, float]:
        diff_abs = val_c - val_a
        diff_pct = (diff_abs / val_a * 100.0) if abs(val_a) > 1e-6 else 0.0
        return {"absolute_diff": round(diff_abs, 3), "percentage_diff": round(diff_pct, 2)}

    comparison_table = {
        "center_mean_dpe_km": {
            "gridsat": m_a["identification"]["center_mean_dpe_km"],
            "insat": m_b["identification"]["center_mean_dpe_km"],
            "fusion": m_c["identification"]["center_mean_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["identification"]["center_mean_dpe_km"], m_a["identification"]["center_mean_dpe_km"])
        },
        "center_median_dpe_km": {
            "gridsat": m_a["identification"]["center_median_dpe_km"],
            "insat": m_b["identification"]["center_median_dpe_km"],
            "fusion": m_c["identification"]["center_median_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["identification"]["center_median_dpe_km"], m_a["identification"]["center_median_dpe_km"])
        },
        "center_p90_dpe_km": {
            "gridsat": m_a["identification"]["center_p90_dpe_km"],
            "insat": m_b["identification"]["center_p90_dpe_km"],
            "fusion": m_c["identification"]["center_p90_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["identification"]["center_p90_dpe_km"], m_a["identification"]["center_p90_dpe_km"])
        },
        "classification_accuracy": {
            "gridsat": m_a["classification"]["accuracy"],
            "insat": m_b["classification"]["accuracy"],
            "fusion": m_c["classification"]["accuracy"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["classification"]["accuracy"], m_a["classification"]["accuracy"])
        },
        "classification_macro_f1": {
            "gridsat": m_a["classification"]["macro_f1"],
            "insat": m_b["classification"]["macro_f1"],
            "fusion": m_c["classification"]["macro_f1"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["classification"]["macro_f1"], m_a["classification"]["macro_f1"])
        },
        "wind_mae_kt": {
            "gridsat": m_a["wind_regression"]["wind_mae_kt"],
            "insat": m_b["wind_regression"]["wind_mae_kt"],
            "fusion": m_c["wind_regression"]["wind_mae_kt"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["wind_regression"]["wind_mae_kt"], m_a["wind_regression"]["wind_mae_kt"])
        },
        "wind_rmse_kt": {
            "gridsat": m_a["wind_regression"]["wind_rmse_kt"],
            "insat": m_b["wind_regression"]["wind_rmse_kt"],
            "fusion": m_c["wind_regression"]["wind_rmse_kt"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["wind_regression"]["wind_rmse_kt"], m_a["wind_regression"]["wind_rmse_kt"])
        },
        "track_12h_mean_dpe_km": {
            "gridsat": m_a["track_prediction"]["track_12h_mean_dpe_km"],
            "insat": m_b["track_prediction"]["track_12h_mean_dpe_km"],
            "fusion": m_c["track_prediction"]["track_12h_mean_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["track_prediction"]["track_12h_mean_dpe_km"], m_a["track_prediction"]["track_12h_mean_dpe_km"])
        },
        "track_24h_mean_dpe_km": {
            "gridsat": m_a["track_prediction"]["track_24h_mean_dpe_km"],
            "insat": m_b["track_prediction"]["track_24h_mean_dpe_km"],
            "fusion": m_c["track_prediction"]["track_24h_mean_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["track_prediction"]["track_24h_mean_dpe_km"], m_a["track_prediction"]["track_24h_mean_dpe_km"])
        },
        "track_48h_mean_dpe_km": {
            "gridsat": m_a["track_prediction"]["track_48h_mean_dpe_km"],
            "insat": m_b["track_prediction"]["track_48h_mean_dpe_km"],
            "fusion": m_c["track_prediction"]["track_48h_mean_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["track_prediction"]["track_48h_mean_dpe_km"], m_a["track_prediction"]["track_48h_mean_dpe_km"])
        },
        "track_aggregate_mean_dpe_km": {
            "gridsat": m_a["track_prediction"]["track_aggregate_mean_dpe_km"],
            "insat": m_b["track_prediction"]["track_aggregate_mean_dpe_km"],
            "fusion": m_c["track_prediction"]["track_aggregate_mean_dpe_km"],
            "diff_fusion_minus_gridsat": calc_diff(m_c["track_prediction"]["track_aggregate_mean_dpe_km"], m_a["track_prediction"]["track_aggregate_mean_dpe_km"])
        }
    }
    results["scientific_comparison_table"] = comparison_table

    # Save JSON files
    model_results_path = Path("data/interim/ml/multisource_model_results.json")
    ablation_results_path = Path("data/interim/ml/multisource_ablation_results.json")

    with open(model_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {model_results_path}")

    ablation_summary = {
        "metadata": {
            "experiment": "VAYU-NET Multi-Source Satellite Fair Ablation",
            "date": time.strftime("%Y-%m-%d %H:%M:%SZ"),
            "dataset_manifest": "data/manifests/vayu_net_multisource_sample_index.csv",
            "dataset_size": {"train": 175, "validation": 252, "test": 298, "total": 725},
            "seed": 42
        },
        "scientific_comparison_table": comparison_table,
        "baselines": baselines
    }
    with open(ablation_results_path, "w") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"Saved ablation summary to {ablation_results_path}")

    # Generate training curves plot
    plot_training_curves(results)

    # Print Final Terminal Table
    print("\n" + "=" * 90)
    print(f"{'Metric':<32} | {'GridSat (A)':<12} | {'INSAT (B)':<12} | {'Fusion (C)':<12} | {'Diff (C - A)':<16}")
    print("-" * 90)
    for m_key, m_val in comparison_table.items():
        g_val = m_val["gridsat"]
        i_val = m_val["insat"]
        f_val = m_val["fusion"]
        diff = m_val["diff_fusion_minus_gridsat"]
        diff_str = f"{diff['absolute_diff']:+.2f} ({diff['percentage_diff']:+.1f}%)"
        print(f"{m_key:<32} | {g_val:<12.2f} | {i_val:<12.2f} | {f_val:<12.2f} | {diff_str:<16}")
    print("=" * 90)

    return results

if __name__ == "__main__":
    run_fair_ablation()
