"""
VAYU-NET — Baseline Forecasting System
Implements deterministic non-ML baselines:
1. Stationary Position Persistence
2. Constant-Velocity Track Extrapolation
3. Intensity (Wind, Pressure, Category) Persistence

Evaluates on VALIDATION (252 samples) and TEST (371 samples).
Saves metrics to data/interim/ml/baseline_results.json and creates diagnostic plots.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SAMPLE_INDEX_CSV = "data/manifests/vayu_net_sample_index.csv"
IMD_V2_CSV = "data/processed/imd_best_track_v2.csv"
RESULTS_JSON = "data/interim/ml/baseline_results.json"
FIGURES_DIR = "docs/figures/baselines"

EARTH_RADIUS_KM = 6371.0

CATEGORY_MAP = {"D": 0, "DD": 1, "CS": 2, "SCS": 3, "VSCS": 4, "ESCS": 5, "SuCS": 6}
INV_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

def haversine(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between two points in km"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c

def run_baselines():
    print("=" * 80)
    print("VAYU-NET — RUNNING DETERMINISTIC PERSISTENCE BASELINES")
    print("=" * 80)

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df_samples = pd.read_csv(SAMPLE_INDEX_CSV)
    df_v2 = pd.read_csv(IMD_V2_CSV)

    # Index IMD observations by (storm_id, timestamp_utc)
    v2_map = {(r["storm_id"], r["timestamp_utc"]): r for _, r in df_v2.iterrows()}
    
    # Also index all prior observations for each storm sorted by time
    storm_obs_sorted = {}
    for sid, group in df_v2.groupby("storm_id"):
        gdf = group.copy()
        gdf["dt"] = pd.to_datetime(gdf["timestamp_utc"])
        gdf = gdf.sort_values("dt").reset_index(drop=True)
        storm_obs_sorted[sid] = gdf

    splits = ["VALIDATION", "TEST"]
    all_results = {}

    for split in splits:
        split_df = df_samples[df_samples["split"] == split].copy().reset_index(drop=True)
        print(f"\n--- Evaluating Split: {split} ({len(split_df)} samples, {split_df['storm_id'].nunique()} storms) ---")

        records = []
        for idx, row in split_df.iterrows():
            sid = row["storm_id"]
            t0_dt = pd.to_datetime(row["t0"])
            t0_str = row["t0"]

            # Current state (t0)
            lat0 = row["imd_lat_t0"]
            lon0 = row["imd_lon_t0"]
            wind0 = row["imd_wind_t0"]
            pres0 = row["imd_pressure_t0"]
            cat0 = row["imd_category_t0"]

            # --- Constant-Velocity Estimation ---
            # Look for latest prior observation in the same storm
            s_obs = storm_obs_sorted[sid]
            priors = s_obs[s_obs["dt"] < t0_dt]
            
            if len(priors) > 0:
                latest_prior = priors.iloc[-1]
                t_prev_dt = latest_prior["dt"]
                delta_h = (t0_dt - t_prev_dt).total_seconds() / 3600.0
                
                lat_prev = latest_prior["latitude"]
                lon_prev = latest_prior["longitude"]
                
                # Handle longitude boundary (NIO is strictly 40 to 105, but handle general wraparound)
                d_lon = lon0 - lon_prev
                if d_lon > 180.0:
                    d_lon -= 360.0
                elif d_lon < -180.0:
                    d_lon += 360.0
                    
                v_lat = (lat0 - lat_prev) / delta_h
                v_lon = d_lon / delta_h
                has_strict_t3 = (delta_h == 3.0)
            else:
                # First observation of storm
                v_lat = 0.0
                v_lon = 0.0
                has_strict_t3 = False

            rec = {
                "sample_id": row["sample_id"],
                "storm_id": sid,
                "split": split,
                "t0": t0_str,
                "has_strict_t3": has_strict_t3,
                "lat0": lat0,
                "lon0": lon0,
                "wind0": wind0,
                "pres0": pres0,
                "cat0": cat0
            }

            for h in [12, 24, 48]:
                # True targets
                true_lat = row[f"imd_lat_{h}h"]
                true_lon = row[f"imd_lon_{h}h"]
                true_w = row[f"wind_{h}h"]
                true_p = row[f"pressure_{h}h"]
                true_cat = row[f"category_{h}h"]

                # 1. Stationary Position Prediction
                stat_pred_lat = lat0
                stat_pred_lon = lon0
                stat_dpe = haversine(stat_pred_lat, stat_pred_lon, true_lat, true_lon)
                stat_lat_err = abs(stat_pred_lat - true_lat)
                stat_lon_err = abs(stat_pred_lon - true_lon)

                # 2. Constant-Velocity Prediction
                vel_pred_lat = lat0 + v_lat * h
                vel_pred_lon = lon0 + v_lon * h
                vel_dpe = haversine(vel_pred_lat, vel_pred_lon, true_lat, true_lon)
                vel_lat_err = abs(vel_pred_lat - true_lat)
                vel_lon_err = abs(vel_pred_lon - true_lon)

                # 3. Intensity Predictions
                # Wind
                valid_wind = pd.notna(true_w)
                wind_err = abs(wind0 - true_w) if valid_wind else np.nan
                wind_sq_err = ((wind0 - true_w)**2) if valid_wind else np.nan

                # Pressure
                valid_pres = pd.notna(true_p)
                pres_err = abs(pres0 - true_p) if valid_pres else np.nan
                pres_sq_err = ((pres0 - true_p)**2) if valid_pres else np.nan

                # Category
                valid_cat = pd.notna(true_cat) and true_cat in CATEGORY_MAP and pd.notna(cat0) and cat0 in CATEGORY_MAP
                cat_match = (cat0 == true_cat) if valid_cat else np.nan

                rec[f"stat_dpe_{h}h"] = stat_dpe
                rec[f"stat_lat_err_{h}h"] = stat_lat_err
                rec[f"stat_lon_err_{h}h"] = stat_lon_err

                rec[f"vel_dpe_{h}h"] = vel_dpe
                rec[f"vel_lat_err_{h}h"] = vel_lat_err
                rec[f"vel_lon_err_{h}h"] = vel_lon_err

                rec[f"true_lat_{h}h"] = true_lat
                rec[f"true_lon_{h}h"] = true_lon
                rec[f"stat_pred_lat_{h}h"] = stat_pred_lat
                rec[f"stat_pred_lon_{h}h"] = stat_pred_lon
                rec[f"vel_pred_lat_{h}h"] = vel_pred_lat
                rec[f"vel_pred_lon_{h}h"] = vel_pred_lon

                rec[f"wind_err_{h}h"] = wind_err
                rec[f"wind_sq_err_{h}h"] = wind_sq_err
                rec[f"valid_wind_{h}h"] = valid_wind

                rec[f"pres_err_{h}h"] = pres_err
                rec[f"pres_sq_err_{h}h"] = pres_sq_err
                rec[f"valid_pres_{h}h"] = valid_pres

                rec[f"cat_match_{h}h"] = cat_match
                rec[f"valid_cat_{h}h"] = valid_cat
                rec[f"pred_cat_{h}h"] = cat0
                rec[f"true_cat_{h}h"] = true_cat

            records.append(rec)

        res_df = pd.DataFrame(records)

        # Compute aggregate metrics
        split_metrics = {"total_samples": len(res_df), "total_storms": res_df["storm_id"].nunique()}

        # Track Metrics (Stationary vs Velocity)
        track_metrics = {}
        for h in [12, 24, 48]:
            stat_dpe = res_df[f"stat_dpe_{h}h"].values
            vel_dpe = res_df[f"vel_dpe_{h}h"].values
            
            # Strict t-3h subset
            strict_vel_dpe = res_df[res_df["has_strict_t3"]][f"vel_dpe_{h}h"].values

            track_metrics[f"{h}h"] = {
                "stationary": {
                    "mean_dpe_km": float(np.mean(stat_dpe)),
                    "median_dpe_km": float(np.median(stat_dpe)),
                    "rmse_dpe_km": float(np.sqrt(np.mean(stat_dpe**2))),
                    "p90_dpe_km": float(np.percentile(stat_dpe, 90)),
                    "max_dpe_km": float(np.max(stat_dpe)),
                    "lat_mae_deg": float(np.mean(res_df[f"stat_lat_err_{h}h"])),
                    "lon_mae_deg": float(np.mean(res_df[f"stat_lon_err_{h}h"])),
                    "evaluated_samples": len(stat_dpe)
                },
                "constant_velocity_all": {
                    "mean_dpe_km": float(np.mean(vel_dpe)),
                    "median_dpe_km": float(np.median(vel_dpe)),
                    "rmse_dpe_km": float(np.sqrt(np.mean(vel_dpe**2))),
                    "p90_dpe_km": float(np.percentile(vel_dpe, 90)),
                    "max_dpe_km": float(np.max(vel_dpe)),
                    "lat_mae_deg": float(np.mean(res_df[f"vel_lat_err_{h}h"])),
                    "lon_mae_deg": float(np.mean(res_df[f"vel_lon_err_{h}h"])),
                    "evaluated_samples": len(vel_dpe)
                },
                "constant_velocity_strict_t3": {
                    "mean_dpe_km": float(np.mean(strict_vel_dpe)),
                    "median_dpe_km": float(np.median(strict_vel_dpe)),
                    "rmse_dpe_km": float(np.sqrt(np.mean(strict_vel_dpe**2))),
                    "p90_dpe_km": float(np.percentile(strict_vel_dpe, 90)),
                    "max_dpe_km": float(np.max(strict_vel_dpe)),
                    "evaluated_samples": len(strict_vel_dpe)
                }
            }
        split_metrics["track"] = track_metrics

        # Intensity Metrics (Wind, Pressure, Category)
        intensity_metrics = {}
        for h in [12, 24, 48]:
            # Wind
            valid_w_mask = res_df[f"valid_wind_{h}h"]
            w_errs = res_df.loc[valid_w_mask, f"wind_err_{h}h"].values
            w_sq_errs = res_df.loc[valid_w_mask, f"wind_sq_err_{h}h"].values

            # Pressure
            valid_p_mask = res_df[f"valid_pres_{h}h"]
            p_errs = res_df.loc[valid_p_mask, f"pres_err_{h}h"].values
            p_sq_errs = res_df.loc[valid_p_mask, f"pres_sq_err_{h}h"].values

            # Category
            valid_c_mask = res_df[f"valid_cat_{h}h"]
            c_df = res_df[valid_c_mask]
            
            y_true_cat = [CATEGORY_MAP[c] for c in c_df[f"true_cat_{h}h"]]
            y_pred_cat = [CATEGORY_MAP[c] for c in c_df[f"pred_cat_{h}h"]]

            cat_acc = accuracy_score(y_true_cat, y_pred_cat) if len(y_true_cat) > 0 else 0.0
            cat_f1 = f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0) if len(y_true_cat) > 0 else 0.0
            cm = confusion_matrix(y_true_cat, y_pred_cat, labels=list(range(7))).tolist()

            intensity_metrics[f"{h}h"] = {
                "wind": {
                    "mae_kt": float(np.mean(w_errs)) if len(w_errs) > 0 else None,
                    "rmse_kt": float(np.sqrt(np.mean(w_sq_errs))) if len(w_sq_errs) > 0 else None,
                    "valid_samples": int(np.sum(valid_w_mask)),
                    "masked_samples": int(np.sum(~valid_w_mask))
                },
                "pressure": {
                    "mae_hpa": float(np.mean(p_errs)) if len(p_errs) > 0 else None,
                    "rmse_hpa": float(np.sqrt(np.mean(p_sq_errs))) if len(p_sq_errs) > 0 else None,
                    "valid_samples": int(np.sum(valid_p_mask)),
                    "masked_samples": int(np.sum(~valid_p_mask))
                },
                "category": {
                    "accuracy": float(cat_acc),
                    "macro_f1": float(cat_f1),
                    "valid_samples": int(np.sum(valid_c_mask)),
                    "masked_samples": int(np.sum(~valid_c_mask)),
                    "confusion_matrix": cm
                }
            }
        split_metrics["intensity"] = intensity_metrics

        # Per-storm analysis
        storm_stats = []
        for sid, sgroup in res_df.groupby("storm_id"):
            s_rec = {
                "storm_id": sid,
                "samples": len(sgroup),
                "stat_dpe_12h_mean": float(sgroup["stat_dpe_12h"].mean()),
                "stat_dpe_24h_mean": float(sgroup["stat_dpe_24h"].mean()),
                "stat_dpe_48h_mean": float(sgroup["stat_dpe_48h"].mean()),
                "vel_dpe_12h_mean": float(sgroup["vel_dpe_12h"].mean()),
                "vel_dpe_24h_mean": float(sgroup["vel_dpe_24h"].mean()),
                "vel_dpe_48h_mean": float(sgroup["vel_dpe_48h"].mean()),
                "wind_mae_12h": float(sgroup[sgroup["valid_wind_12h"]]["wind_err_12h"].mean()) if sgroup["valid_wind_12h"].any() else None,
                "wind_mae_24h": float(sgroup[sgroup["valid_wind_24h"]]["wind_err_24h"].mean()) if sgroup["valid_wind_24h"].any() else None,
                "wind_mae_48h": float(sgroup[sgroup["valid_wind_48h"]]["wind_err_48h"].mean()) if sgroup["valid_wind_48h"].any() else None,
                "cat_acc_12h": float(sgroup[sgroup["valid_cat_12h"]]["cat_match_12h"].mean()) if sgroup["valid_cat_12h"].any() else None,
                "cat_acc_24h": float(sgroup[sgroup["valid_cat_24h"]]["cat_match_24h"].mean()) if sgroup["valid_cat_24h"].any() else None,
                "cat_acc_48h": float(sgroup[sgroup["valid_cat_48h"]]["cat_match_48h"].mean()) if sgroup["valid_cat_48h"].any() else None,
            }
            storm_stats.append(s_rec)
        split_metrics["per_storm"] = storm_stats

        all_results[split] = split_metrics

    # Save to JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved baseline evaluation results to {RESULTS_JSON}")

    # =========================================================================
    # DIAGNOSTIC PLOTS (TEST SPLIT ONLY)
    # =========================================================================
    print("\n--- Generating Diagnostic Plots for TEST Split ---")
    test_metrics = all_results["TEST"]
    test_df_eval = pd.DataFrame(records)[pd.DataFrame(records)["split"] == "TEST"]

    # 1. Mean DPE vs Forecast Horizon
    horizons = [12, 24, 48]
    stat_means = [test_metrics["track"][f"{h}h"]["stationary"]["mean_dpe_km"] for h in horizons]
    vel_means = [test_metrics["track"][f"{h}h"]["constant_velocity_all"]["mean_dpe_km"] for h in horizons]
    vel_strict_means = [test_metrics["track"][f"{h}h"]["constant_velocity_strict_t3"]["mean_dpe_km"] for h in horizons]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(horizons, stat_means, marker="o", linewidth=2.5, color="#1f77b4", label="Stationary Persistence")
    plt.plot(horizons, vel_means, marker="s", linewidth=2.5, color="#ff7f0e", label="Constant Velocity (All Samples)")
    plt.plot(horizons, vel_strict_means, marker="^", linewidth=2.0, linestyle="--", color="#2ca02c", label="Constant Velocity (Strict t-3h subset)")
    
    for h, sm, vm in zip(horizons, stat_means, vel_means):
        plt.annotate(f"{sm:.1f} km", (h, sm), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        plt.annotate(f"{vm:.1f} km", (h, vm), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9)

    plt.title("Direct Position Error (DPE) vs Forecast Horizon (TEST Split, 371 Samples)", fontsize=12, fontweight="bold")
    plt.xlabel("Forecast Horizon (Hours)", fontsize=11)
    plt.ylabel("Mean DPE (km)", fontsize=11)
    plt.xticks(horizons)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plot1_path = os.path.join(FIGURES_DIR, "dpe_vs_horizon.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. Distribution of DPE at 12h, 24h, 48h
    plt.figure(figsize=(10, 6), dpi=300)
    data_to_plot = [
        test_df_eval["stat_dpe_12h"].values,
        test_df_eval["vel_dpe_12h"].values,
        test_df_eval["stat_dpe_24h"].values,
        test_df_eval["vel_dpe_24h"].values,
        test_df_eval["stat_dpe_48h"].values,
        test_df_eval["vel_dpe_48h"].values,
    ]
    labels = ["+12h Stat", "+12h Vel", "+24h Stat", "+24h Vel", "+48h Stat", "+48h Vel"]
    colors = ["#aec7e8", "#ffbb78", "#1f77b4", "#ff7f0e", "#1b4f72", "#d35400"]

    box = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True,
                      meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "yellow"})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    plt.title("Distribution of Direct Position Error (DPE) across Forecast Horizons (TEST Split)", fontsize=12, fontweight="bold")
    plt.xlabel("Baseline & Forecast Horizon", fontsize=11)
    plt.ylabel("DPE (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    plot2_path = os.path.join(FIGURES_DIR, "dpe_distributions.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. Example Historical Cyclone Track (Cyclone BIPARJOY 2023)
    biparjoy_sample = test_df_eval[test_df_eval["storm_id"] == "NIO_2023_BIPARJOY"].iloc[len(test_df_eval[test_df_eval["storm_id"] == "NIO_2023_BIPARJOY"]) // 3]
    
    true_lats = [biparjoy_sample["lat0"], biparjoy_sample["true_lat_12h"], biparjoy_sample["true_lat_24h"], biparjoy_sample["true_lat_48h"]]
    true_lons = [biparjoy_sample["lon0"], biparjoy_sample["true_lon_12h"], biparjoy_sample["true_lon_24h"], biparjoy_sample["true_lon_48h"]]
    
    stat_lats = [biparjoy_sample["lat0"], biparjoy_sample["stat_pred_lat_12h"], biparjoy_sample["stat_pred_lat_24h"], biparjoy_sample["stat_pred_lat_48h"]]
    stat_lons = [biparjoy_sample["lon0"], biparjoy_sample["stat_pred_lon_12h"], biparjoy_sample["stat_pred_lon_24h"], biparjoy_sample["stat_pred_lon_48h"]]
    
    vel_lats = [biparjoy_sample["lat0"], biparjoy_sample["vel_pred_lat_12h"], biparjoy_sample["vel_pred_lat_24h"], biparjoy_sample["vel_pred_lat_48h"]]
    vel_lons = [biparjoy_sample["lon0"], biparjoy_sample["vel_pred_lon_12h"], biparjoy_sample["vel_pred_lon_24h"], biparjoy_sample["vel_pred_lon_48h"]]

    plt.figure(figsize=(9, 7), dpi=300)
    plt.plot(true_lons, true_lats, marker="o", markersize=8, color="black", linewidth=2.5, label="Observed Ground Truth")
    plt.plot(stat_lons, stat_lats, marker="s", markersize=8, color="#1f77b4", linestyle=":", linewidth=2, label="Stationary Persistence")
    plt.plot(vel_lons, vel_lats, marker="^", markersize=8, color="#ff7f0e", linestyle="--", linewidth=2, label="Constant-Velocity Persistence")

    for i, h_tag in enumerate(["t0", "+12h", "+24h", "+48h"]):
        plt.annotate(f"{h_tag} ({true_lats[i]:.1f}N, {true_lons[i]:.1f}E)", (true_lons[i], true_lats[i]),
                     textcoords="offset points", xytext=(8, -4), fontsize=9, fontweight="bold")

    plt.title(f"Track Forecast Benchmark: Cyclone BIPARJOY (t0 = {biparjoy_sample['t0']})", fontsize=12, fontweight="bold")
    plt.xlabel("Longitude (°E)", fontsize=11)
    plt.ylabel("Latitude (°N)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plot3_path = os.path.join(FIGURES_DIR, "example_track_comparison.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"Saved: {plot3_path}")

    # =========================================================================
    # PRINT SUMMARY REQUIRED BY SPECIFICATION
    # =========================================================================
    tm = test_metrics["track"]
    im = test_metrics["intensity"]

    print("\n" + "=" * 80)
    print("BASELINE PHASE COMPLETE")
    print("=" * 80)
    print(f"TEST sample count: {test_metrics['total_samples']}")
    print(f"TEST storm count: {test_metrics['total_storms']}")
    print(f"+12h DPE: Stationary = {tm['12h']['stationary']['mean_dpe_km']:.2f} km | Constant-Velocity = {tm['12h']['constant_velocity_all']['mean_dpe_km']:.2f} km")
    print(f"+24h DPE: Stationary = {tm['24h']['stationary']['mean_dpe_km']:.2f} km | Constant-Velocity = {tm['24h']['constant_velocity_all']['mean_dpe_km']:.2f} km")
    print(f"+48h DPE: Stationary = {tm['48h']['stationary']['mean_dpe_km']:.2f} km | Constant-Velocity = {tm['48h']['constant_velocity_all']['mean_dpe_km']:.2f} km")
    print(f"Wind MAE: +12h = {im['12h']['wind']['mae_kt']:.2f} kt | +24h = {im['24h']['wind']['mae_kt']:.2f} kt | +48h = {im['48h']['wind']['mae_kt']:.2f} kt")
    print(f"Category Accuracy / Macro F1:")
    print(f"   +12h: Acc = {im['12h']['category']['accuracy']*100:.2f}%, F1 = {im['12h']['category']['macro_f1']:.4f}")
    print(f"   +24h: Acc = {im['24h']['category']['accuracy']*100:.2f}%, F1 = {im['24h']['category']['macro_f1']:.4f}")
    print(f"   +48h: Acc = {im['48h']['category']['accuracy']*100:.2f}%, F1 = {im['48h']['category']['macro_f1']:.4f}")
    print(f"Pressure MAE: +12h = {im['12h']['pressure']['mae_hpa']:.2f} hPa | +24h = {im['24h']['pressure']['mae_hpa']:.2f} hPa | +48h = {im['48h']['pressure']['mae_hpa']:.2f} hPa")
    print("=" * 80)

if __name__ == "__main__":
    run_baselines()
