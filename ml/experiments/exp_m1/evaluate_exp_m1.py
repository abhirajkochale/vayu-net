"""
VAYU-NET: EXP-M1 Multi-Task Model Evaluation & Storm-Level Bootstrap Analysis.
Evaluates M1A, M1B, and M1C on the locked 371-sequence, 31-storm TEST partition.
Computes:
  - Overall test metrics (Center, Track, Wind, Intensity)
  - Parameter counts and inference latency
  - Granular per-storm error profiles across all 31 test storms
  - 1,000-sample storm-level bootstrap 95% confidence intervals
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.experiments.exp_m1.model import (
    ExpM1Model, haversine_km,
    CATEGORY_TO_IDX, IDX_TO_CATEGORY
)
from ml.experiments.exp_m1.train_exp_m1 import CachedMultiTaskDataset

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
CHECKPOINT_DIR = REPO_ROOT / "ml/experiments/exp_m1/checkpoints"
RESULTS_DIR = REPO_ROOT / "ml/experiments/exp_m1/results"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_test_predictions(model: ExpM1Model,
                              loader: DataLoader,
                              device: torch.device) -> Tuple[pd.DataFrame, float]:
    """Generates predictions and records inference timing."""
    model.eval()
    records = []
    t_start = time.time()
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            B = len(batch["sample_id"])
            n_samples += B

            g_seq = batch["gridsat"].to(device) if model.mode in ["m1a_gridsat", "m1c_fusion"] else None
            i_seq = batch["imerg"].to(device) if model.mode in ["m1b_imerg", "m1c_fusion"] else None

            preds = model(gridsat_seq=g_seq, imerg_seq=i_seq)

            pc = preds["center_deg"].cpu().numpy()
            tc = batch["center_t0"].numpy()

            pw = preds["wind_kt"].squeeze(-1).cpu().numpy()
            tw = batch["wind_t0"].numpy()
            wm = batch["wind_t0_mask"].numpy()

            pcat = torch.argmax(preds["class_logits"], dim=-1).cpu().numpy()
            tcat = batch["category_t0"].numpy()
            cm = batch["category_t0_mask"].numpy()

            p12 = preds["track_12h_deg"].cpu().numpy()
            t12 = batch["center_12h"].numpy()
            m12 = batch["wind_12h_mask"].numpy()

            p24 = preds["track_24h_deg"].cpu().numpy()
            t24 = batch["center_24h"].numpy()
            m24 = batch["wind_24h_mask"].numpy()

            p48 = preds["track_48h_deg"].cpu().numpy()
            t48 = batch["center_48h"].numpy()
            m48 = batch["wind_48h_mask"].numpy()

            for i in range(B):
                # Center DPE & component errors
                c_dpe = haversine_km(pc[i, 0], pc[i, 1], tc[i, 0], tc[i, 1])
                lat_mae = abs(float(pc[i, 0] - tc[i, 0]))
                lon_mae = abs(float(pc[i, 1] - tc[i, 1]))

                # Wind
                w_mae = abs(float(pw[i] - tw[i])) if wm[i] > 0.5 else np.nan
                w_diff = float(pw[i] - tw[i]) if wm[i] > 0.5 else np.nan

                # Cat
                cat_correct = (pcat[i] == tcat[i]) if (cm[i] > 0.5 and tcat[i] >= 0) else np.nan

                # Track DPEs
                dpe_12 = haversine_km(p12[i, 0], p12[i, 1], t12[i, 0], t12[i, 1]) if m12[i] > 0.5 else np.nan
                dpe_24 = haversine_km(p24[i, 0], p24[i, 1], t24[i, 0], t24[i, 1]) if m24[i] > 0.5 else np.nan
                dpe_48 = haversine_km(p48[i, 0], p48[i, 1], t48[i, 0], t48[i, 1]) if m48[i] > 0.5 else np.nan

                records.append({
                    "sample_id": batch["sample_id"][i],
                    "storm_id": batch["storm_id"][i],
                    "storm_name": batch["storm_name"][i],
                    "t0_utc": batch["t0_utc"][i],
                    "pred_center_lat": float(pc[i, 0]),
                    "pred_center_lon": float(pc[i, 1]),
                    "true_center_lat": float(tc[i, 0]),
                    "true_center_lon": float(tc[i, 1]),
                    "center_dpe_km": c_dpe,
                    "center_lat_mae_deg": lat_mae,
                    "center_lon_mae_deg": lon_mae,
                    "pred_wind_kt": float(pw[i]),
                    "true_wind_kt": float(tw[i]) if wm[i] > 0.5 else np.nan,
                    "wind_abs_error_kt": w_mae,
                    "wind_error_kt": w_diff,
                    "pred_category_idx": int(pcat[i]),
                    "true_category_idx": int(tcat[i]) if cm[i] > 0.5 else -1,
                    "cat_correct": cat_correct,
                    "track_12h_dpe_km": dpe_12,
                    "track_24h_dpe_km": dpe_24,
                    "track_48h_dpe_km": dpe_48,
                })

    latency_ms_per_seq = ((time.time() - t_start) / max(1, n_samples)) * 1000.0
    return pd.DataFrame(records), latency_ms_per_seq


def compute_aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive test metrics from evaluation dataframe."""
    # Center
    c_dpes = df["center_dpe_km"].dropna().values
    center_metrics = {
        "mean_dpe_km": float(np.mean(c_dpes)),
        "median_dpe_km": float(np.median(c_dpes)),
        "p90_dpe_km": float(np.percentile(c_dpes, 90)),
        "latitude_mae_deg": float(df["center_lat_mae_deg"].mean()),
        "longitude_mae_deg": float(df["center_lon_mae_deg"].mean())
    }

    # Track
    d12 = df["track_12h_dpe_km"].dropna().values
    d24 = df["track_24h_dpe_km"].dropna().values
    d48 = df["track_48h_dpe_km"].dropna().values
    all_dpes = np.concatenate([d12, d24, d48]) if len(d12) + len(d24) + len(d48) > 0 else np.array([])
    track_metrics = {
        "track_12h_dpe_km": float(np.mean(d12)) if len(d12) > 0 else 0.0,
        "track_24h_dpe_km": float(np.mean(d24)) if len(d24) > 0 else 0.0,
        "track_48h_dpe_km": float(np.mean(d48)) if len(d48) > 0 else 0.0,
        "aggregate_dpe_km": float(np.mean(all_dpes)) if len(all_dpes) > 0 else 0.0,
        "valid_counts": {"12h": len(d12), "24h": len(d24), "48h": len(d48)}
    }

    # Intensity Classification
    valid_cat_df = df[df["true_category_idx"] >= 0]
    trues = valid_cat_df["true_category_idx"].values
    preds = valid_cat_df["pred_category_idx"].values
    acc = float(accuracy_score(trues, preds)) if len(trues) > 0 else 0.0
    f1 = float(f1_score(trues, preds, average="macro", zero_division=0)) if len(trues) > 0 else 0.0
    cm = confusion_matrix(trues, preds, labels=list(range(7))).tolist() if len(trues) > 0 else []
    intensity_metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "num_evaluated_samples": len(trues),
        "confusion_matrix": cm
    }

    # Wind
    w_valid = df.dropna(subset=["wind_abs_error_kt"])
    p_wind = w_valid["pred_wind_kt"].values
    t_wind = w_valid["true_wind_kt"].values
    w_err = w_valid["wind_error_kt"].values
    w_abs = w_valid["wind_abs_error_kt"].values

    wind_mae = float(np.mean(w_abs)) if len(w_abs) > 0 else 0.0
    wind_rmse = float(np.sqrt(np.mean(w_err ** 2))) if len(w_err) > 0 else 0.0
    wind_med = float(np.median(w_abs)) if len(w_abs) > 0 else 0.0
    wind_p90 = float(np.percentile(w_abs, 90)) if len(w_abs) > 0 else 0.0
    wind_bias = float(np.mean(w_err)) if len(w_err) > 0 else 0.0
    corr = float(np.corrcoef(p_wind, t_wind)[0, 1]) if (len(p_wind) > 1 and np.std(p_wind) > 1e-6 and np.std(t_wind) > 1e-6) else 0.0

    wind_metrics = {
        "mae_kt": wind_mae,
        "rmse_kt": wind_rmse,
        "median_ae_kt": wind_med,
        "p90_ae_kt": wind_p90,
        "bias_kt": wind_bias,
        "pearson_r": corr,
        "num_evaluated_samples": len(w_valid)
    }

    return {
        "center": center_metrics,
        "track": track_metrics,
        "intensity": intensity_metrics,
        "wind": wind_metrics
    }


def compute_storm_level_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculates granular evaluation metrics broken down by individual storm."""
    storm_metrics = {}
    for storm_id, s_df in df.groupby("storm_id"):
        storm_name = s_df["storm_name"].iloc[0]
        n_seq = len(s_df)
        
        c_mean = float(s_df["center_dpe_km"].mean())
        w_mae = float(s_df["wind_abs_error_kt"].dropna().mean()) if s_df["wind_abs_error_kt"].notna().any() else None
        
        valid_cat = s_df[s_df["true_category_idx"] >= 0]
        cat_acc = float(accuracy_score(valid_cat["true_category_idx"], valid_cat["pred_category_idx"])) if len(valid_cat) > 0 else None
        
        d12 = s_df["track_12h_dpe_km"].dropna().values
        d24 = s_df["track_24h_dpe_km"].dropna().values
        d48 = s_df["track_48h_dpe_km"].dropna().values
        
        storm_metrics[storm_id] = {
            "storm_name": storm_name,
            "num_sequences": n_seq,
            "center_mean_dpe_km": c_mean,
            "wind_mae_kt": w_mae,
            "cat_accuracy": cat_acc,
            "track_12h_mean_dpe_km": float(np.mean(d12)) if len(d12) > 0 else None,
            "track_24h_mean_dpe_km": float(np.mean(d24)) if len(d24) > 0 else None,
            "track_48h_mean_dpe_km": float(np.mean(d48)) if len(d48) > 0 else None,
        }
    return storm_metrics


def compute_storm_bootstrap_cis(df: pd.DataFrame,
                                n_boot: int = 1000,
                                seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% bootstrap confidence intervals by resampling storms with replacement.
    Preserves intra-storm correlation by treating the storm as the sampling unit.
    """
    rng = np.random.RandomState(seed)
    storm_ids = sorted(list(df["storm_id"].unique()))
    n_storms = len(storm_ids)
    
    # Pre-group storm dataframes
    storm_dfs = {s: s_df for s, s_df in df.groupby("storm_id")}
    
    boot_c_dpe = []
    boot_w_mae = []
    boot_acc = []
    boot_f1 = []
    boot_t12 = []
    boot_t24 = []
    boot_t48 = []
    boot_agg = []
    
    for _ in range(n_boot):
        sample_storms = rng.choice(storm_ids, size=n_storms, replace=True)
        boot_df = pd.concat([storm_dfs[s] for s in sample_storms], ignore_index=True)
        
        # Center DPE
        boot_c_dpe.append(np.mean(boot_df["center_dpe_km"]))
        
        # Wind MAE
        w_vals = boot_df["wind_abs_error_kt"].dropna().values
        if len(w_vals) > 0:
            boot_w_mae.append(np.mean(w_vals))
            
        # Cat Acc & F1
        valid_cat = boot_df[boot_df["true_category_idx"] >= 0]
        if len(valid_cat) > 0:
            t = valid_cat["true_category_idx"].values
            p = valid_cat["pred_category_idx"].values
            boot_acc.append(accuracy_score(t, p))
            boot_f1.append(f1_score(t, p, average="macro", zero_division=0))
            
        # Track DPEs
        d12 = boot_df["track_12h_dpe_km"].dropna().values
        d24 = boot_df["track_24h_dpe_km"].dropna().values
        d48 = boot_df["track_48h_dpe_km"].dropna().values
        if len(d12) > 0: boot_t12.append(np.mean(d12))
        if len(d24) > 0: boot_t24.append(np.mean(d24))
        if len(d48) > 0: boot_t48.append(np.mean(d48))
        all_d = np.concatenate([d12, d24, d48]) if len(d12) + len(d24) + len(d48) > 0 else np.array([])
        if len(all_d) > 0: boot_agg.append(np.mean(all_d))
        
    def get_ci(arr):
        if not arr: return {"ci_lower": 0.0, "ci_upper": 0.0}
        return {
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5))
        }
        
    return {
        "center_mean_dpe_km": get_ci(boot_c_dpe),
        "wind_mae_kt": get_ci(boot_w_mae),
        "cat_accuracy": get_ci(boot_acc),
        "cat_macro_f1": get_ci(boot_f1),
        "track_12h_dpe_km": get_ci(boot_t12),
        "track_24h_dpe_km": get_ci(boot_t24),
        "track_48h_dpe_km": get_ci(boot_t48),
        "track_aggregate_dpe_km": get_ci(boot_agg),
    }


def main():
    print("=" * 65)
    print("VAYU-NET: EXP-M1 FINAL TEST EVALUATION (371 SEQUENCES / 31 STORMS)")
    print("=" * 65)

    if not CACHE_PATH.exists():
        print(f"Error: Cache not found at {CACHE_PATH}")
        sys.exit(1)

    cache_payload = torch.load(CACHE_PATH)
    test_ds = CachedMultiTaskDataset(cache_payload["samples"], "TEST")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    print(f"Loaded TEST partition: {len(test_ds)} sequences across 31 unique storms.")

    device = torch.device("cpu")
    experiments = [
        ("m1a_gridsat", "m1a_gridsat_only.pt"),
        ("m1b_imerg",   "m1b_imerg_only.pt"),
        ("m1c_fusion",  "m1c_gridsat_imerg.pt")
    ]

    all_eval_results = {}

    for mode, ckpt_name in experiments:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"\n--- Evaluating {mode.upper()} ({ckpt_name}) ---")
        checkpoint = torch.load(ckpt_path)

        model = ExpM1Model(mode=mode).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        params = model.get_parameter_counts()

        # Run test predictions & latency
        pred_df, latency_ms = evaluate_test_predictions(model, test_loader, device)

        # Aggregate metrics
        agg_metrics = compute_aggregate_metrics(pred_df)

        # Storm-level breakdown
        storm_metrics = compute_storm_level_metrics(pred_df)

        # 1000-sample storm bootstrap CIs
        boot_cis = compute_storm_bootstrap_cis(pred_df, n_boot=1000, seed=42)

        # Attach CIs to point estimates
        for metric_name, ci_dict in boot_cis.items():
            if metric_name in ["center_mean_dpe_km"]:
                agg_metrics["center"]["ci_95"] = [ci_dict["ci_lower"], ci_dict["ci_upper"]]
            elif metric_name in ["wind_mae_kt"]:
                agg_metrics["wind"]["ci_95"] = [ci_dict["ci_lower"], ci_dict["ci_upper"]]
            elif metric_name in ["cat_accuracy", "cat_macro_f1"]:
                agg_metrics["intensity"][f"{metric_name}_ci_95"] = [ci_dict["ci_lower"], ci_dict["ci_upper"]]
            elif metric_name in ["track_12h_dpe_km", "track_24h_dpe_km", "track_48h_dpe_km", "track_aggregate_dpe_km"]:
                agg_metrics["track"][f"{metric_name}_ci_95"] = [ci_dict["ci_lower"], ci_dict["ci_upper"]]

        all_eval_results[mode] = {
            "mode": mode,
            "checkpoint_name": ckpt_name,
            "best_validation_epoch": checkpoint["best_epoch"],
            "best_validation_loss": checkpoint["best_val_loss"],
            "parameter_counts": params,
            "inference_latency_ms_per_seq": latency_ms,
            "test_aggregate_metrics": agg_metrics,
            "storm_bootstrap_95ci": boot_cis,
            "storm_level_metrics": storm_metrics,
            "test_sample_count": len(pred_df),
            "test_storm_count": len(storm_metrics)
        }

        # Print summary
        c = agg_metrics["center"]
        w = agg_metrics["wind"]
        i = agg_metrics["intensity"]
        t = agg_metrics["track"]
        print(f"  Center Mean DPE:   {c['mean_dpe_km']:.1f} km (95% CI: [{boot_cis['center_mean_dpe_km']['ci_lower']:.1f}, {boot_cis['center_mean_dpe_km']['ci_upper']:.1f}])")
        print(f"  Center Median DPE: {c['median_dpe_km']:.1f} km | P90 DPE: {c['p90_dpe_km']:.1f} km")
        print(f"  Wind MAE:          {w['mae_kt']:.2f} kt (95% CI: [{boot_cis['wind_mae_kt']['ci_lower']:.2f}, {boot_cis['wind_mae_kt']['ci_upper']:.2f}])")
        print(f"  Intensity Acc:     {i['accuracy']*100:.1f}% | Macro-F1: {i['macro_f1']:.4f}")
        print(f"  Track Aggregate:   {t['aggregate_dpe_km']:.1f} km (95% CI: [{boot_cis['track_aggregate_dpe_km']['ci_lower']:.1f}, {boot_cis['track_aggregate_dpe_km']['ci_upper']:.1f}])")
        print(f"  Track +12h DPE:    {t['track_12h_dpe_km']:.1f} km | +24h DPE: {t['track_24h_dpe_km']:.1f} km | +48h DPE: {t['track_48h_dpe_km']:.1f} km")
        print(f"  Latency:           {latency_ms:.2f} ms / sequence | Parameters: {params['trainable_parameters']:,}")

    # Save comprehensive machine-readable results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_results_path = RESULTS_DIR / "exp_m1_results.json"
    with open(out_results_path, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"\nSaved comprehensive test results to: {out_results_path}")

    # Print Comparative Table
    print("\n" + "=" * 95)
    print(f"{'Configuration':<18} {'Params':<9} {'Center DPE':<18} {'Wind MAE':<15} {'Cat Acc':<10} {'Track Agg DPE':<18}")
    print("-" * 95)
    for mode, data in all_eval_results.items():
        p = data["parameter_counts"]["trainable_parameters"]
        c_pt = data["test_aggregate_metrics"]["center"]["mean_dpe_km"]
        c_ci = data["storm_bootstrap_95ci"]["center_mean_dpe_km"]
        w_pt = data["test_aggregate_metrics"]["wind"]["mae_kt"]
        w_ci = data["storm_bootstrap_95ci"]["wind_mae_kt"]
        acc = data["test_aggregate_metrics"]["intensity"]["accuracy"] * 100
        t_pt = data["test_aggregate_metrics"]["track"]["aggregate_dpe_km"]
        t_ci = data["storm_bootstrap_95ci"]["track_aggregate_dpe_km"]
        
        c_str = f"{c_pt:.1f} [{c_ci['ci_lower']:.1f}-{c_ci['ci_upper']:.1f}]"
        w_str = f"{w_pt:.2f} [{w_ci['ci_lower']:.2f}-{w_ci['ci_upper']:.2f}]"
        t_str = f"{t_pt:.1f} [{t_ci['ci_lower']:.1f}-{t_ci['ci_upper']:.1f}]"
        print(f"{mode:<18} {p:<9,d} {c_str:<18} {w_str:<15} {acc:<10.1f}% {t_str:<18}")
    print("=" * 95)


if __name__ == "__main__":
    main()