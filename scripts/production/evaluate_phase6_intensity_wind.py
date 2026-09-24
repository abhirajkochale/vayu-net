"""
VAYU-NET PHASE 6 — PRODUCTION EVALUATION & DIAGNOSTICS PIPELINE
===============================================================
Evaluates the frozen Phase 6 intensity & wind multi-task checkpoint strictly ONCE against TEST.
Generates:
  1. Full TEST metrics for classification & wind regression
  2. Baselines comparison (majority class, mean wind, persistence)
  3. Subgroup / error analysis (intensity tiers, ocean basins, top outliers)
  4. Probability calibration analysis & Expected Calibration Error (ECE)
  5. Empirical wind uncertainty coverage analysis
  6. 10 production figures in docs/figures/phase6_intensity/
  7. Final results JSON in data/interim/ml/phase6_intensity_wind_results.json
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.phase6_intensity_wind import (
    Phase6IntensityWindModel,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    DEFAULT_TRAIN_WIND_MEAN_KT,
    DEFAULT_TRAIN_WIND_STD_KT
)
from ml.train.train_phase6_intensity_wind import Phase6CachedDataset, collate_phase6

CHECKPOINT_PATH = "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt"
CACHE_PATH = "data/interim/ml/cache/phase6_intensity_features.pt"
RESULTS_JSON_PATH = "data/interim/ml/phase6_intensity_wind_results.json"
HISTORY_PATH = "data/interim/ml/phase6_training_history.json"
FIGURES_DIR = "docs/figures/phase6_intensity"
SAMPLE_INDEX_PATH = "data/manifests/vayu_net_sample_index.csv"
BT_V2_PATH = "data/processed/imd_best_track_v2.csv"

CATEGORIES = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]


def compute_ece(probs, targets, n_bins=10):
    """Computes Expected Calibration Error (ECE) and bin statistics."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_data = []
    n_samples = len(targets)

    for i in range(n_bins):
        mask = (bin_indices == i)
        count = np.sum(mask)
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            weight = count / n_samples
            ece += weight * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin_idx": i,
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i+1]),
                "count": int(count),
                "accuracy": bin_acc,
                "confidence": bin_conf
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i+1]),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0
            })

    return float(ece), bin_data


def evaluate_baselines(test_samples, train_samples, sample_index_df, bt_v2_df):
    """Computes deterministic baselines: Majority Class, Mean Wind, and Persistence."""
    # 1. Majority Category in TRAIN
    train_cats = [s["category_t0"].item() for s in train_samples if s["category_t0"].item() >= 0]
    maj_cat_idx = int(pd.Series(train_cats).mode()[0])
    maj_cat_name = IDX_TO_CATEGORY[maj_cat_idx]

    # TEST true categories
    test_true_cats = [s["category_t0"].item() for s in test_samples if s["category_t0"].item() >= 0]
    maj_pred_cats = [maj_cat_idx] * len(test_true_cats)

    maj_acc = float(accuracy_score(test_true_cats, maj_pred_cats))
    maj_macro_f1 = float(f1_score(test_true_cats, maj_pred_cats, average="macro", zero_division=0))
    maj_cm = confusion_matrix(test_true_cats, maj_pred_cats, labels=list(range(7))).tolist()

    # 2. Mean Wind in TRAIN
    train_winds = [s["wind_t0"].item() for s in train_samples if s["wind_t0_mask"].item() > 0.5]
    train_mean_wind = float(np.mean(train_winds))

    # TEST true winds
    test_true_winds = np.array([s["wind_t0"].item() for s in test_samples if s["wind_t0_mask"].item() > 0.5])
    mean_pred_winds = np.full_like(test_true_winds, train_mean_wind)

    wind_errs = np.abs(test_true_winds - mean_pred_winds)
    mean_wind_mae = float(np.mean(wind_errs))
    mean_wind_rmse = float(np.sqrt(np.mean((test_true_winds - mean_pred_winds)**2)))
    mean_wind_median_ae = float(np.median(wind_errs))
    mean_wind_p90_ae = float(np.percentile(wind_errs, 90))
    mean_wind_bias = float(np.mean(mean_pred_winds - test_true_winds))

    # 3. Persistence Baseline (where valid past observation exists)
    bt_lookup = {}
    for _, row in bt_v2_df.iterrows():
        bt_lookup[(row["storm_id"], row["timestamp_utc"])] = (row["category"], row["maximum_sustained_wind_kt"])

    test_df = sample_index_df[sample_index_df["split"] == "TEST"].copy()
    test_df["t0_dt"] = pd.to_datetime(test_df["t0"])

    pers_6h_true_w, pers_6h_pred_w = [], []
    pers_6h_true_c, pers_6h_pred_c = [], []

    for _, row in test_df.iterrows():
        s_id = row["storm_id"]
        t0_dt = row["t0_dt"]
        t_minus_6h = (t0_dt - pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        prev_cat, prev_wind = bt_lookup.get((s_id, t_minus_6h), (None, None))
        if pd.notna(prev_wind) and pd.notna(row["imd_wind_t0"]):
            pers_6h_true_w.append(float(row["imd_wind_t0"]))
            pers_6h_pred_w.append(float(prev_wind))

        if pd.notna(prev_cat) and pd.notna(row["imd_category_t0"]) and prev_cat in CATEGORY_TO_IDX and row["imd_category_t0"] in CATEGORY_TO_IDX:
            pers_6h_true_c.append(CATEGORY_TO_IDX[row["imd_category_t0"]])
            pers_6h_pred_c.append(CATEGORY_TO_IDX[prev_cat])

    pers_wind_mae = float(np.mean(np.abs(np.array(pers_6h_true_w) - np.array(pers_6h_pred_w)))) if len(pers_6h_true_w) > 0 else 0.0
    pers_wind_rmse = float(np.sqrt(np.mean((np.array(pers_6h_true_w) - np.array(pers_6h_pred_w))**2))) if len(pers_6h_true_w) > 0 else 0.0
    pers_cat_acc = float(accuracy_score(pers_6h_true_c, pers_6h_pred_c)) if len(pers_6h_true_c) > 0 else 0.0
    pers_cat_macro_f1 = float(f1_score(pers_6h_true_c, pers_6h_pred_c, average="macro", zero_division=0)) if len(pers_6h_true_c) > 0 else 0.0

    return {
        "majority_category": {
            "predicted_category": maj_cat_name,
            "accuracy": maj_acc,
            "macro_f1": maj_macro_f1,
            "confusion_matrix": maj_cm
        },
        "mean_wind": {
            "predicted_wind_kt": train_mean_wind,
            "mae_kt": mean_wind_mae,
            "rmse_kt": mean_wind_rmse,
            "median_ae_kt": mean_wind_median_ae,
            "p90_ae_kt": mean_wind_p90_ae,
            "bias_kt": mean_wind_bias
        },
        "persistence_6h": {
            "eligible_samples_count": len(pers_6h_true_w),
            "wind_mae_kt": pers_wind_mae,
            "wind_rmse_kt": pers_wind_rmse,
            "category_accuracy": pers_cat_acc,
            "category_macro_f1": pers_cat_macro_f1
        }
    }


def generate_all_figures(history_data, test_eval, baselines, sub_df):
    """Generates the 10 comprehensive diagnostic figures required for Phase 6."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Figure 1: Training & Validation Loss & Metrics Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for exp_name, exp_data in history_data.items():
        hist = exp_data["history"]
        eps = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        val_comp = [h["val_composite_error"] for h in hist]
        val_f1 = [h["val_macro_f1"] for h in hist]
        val_wmae = [h["val_wind_mae"] for h in hist]

        axes[0].plot(eps, val_comp, marker="o", label=f"{exp_name} (Val)")
        axes[1].plot(eps, val_f1, marker="s", label=f"{exp_name} (Val)")
        axes[2].plot(eps, val_wmae, marker="^", label=f"{exp_name} (Val)")

    axes[0].set_title("Validation Composite Error Index (Lower = Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Composite Error")
    axes[0].legend()

    axes[1].set_title("Validation Category Macro F1 (Higher = Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()

    axes[2].set_title("Validation Wind MAE [kt] (Lower = Better)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Wind MAE (kt)")
    axes[2].legend()

    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "loss_curves.png")
    fig.savefig(f1_path, dpi=200)
    plt.close(fig)

    # 2. Figure 2: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = np.array(test_eval["confusion_matrix"])
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, fontsize=10, fontweight="bold")
    ax.set_yticklabels(CATEGORIES, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Intensity Category", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual IMD Category", fontsize=12, fontweight="bold")
    ax.set_title("VAYU-NET Phase 6 — TEST Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig.savefig(f2_path, dpi=200)
    plt.close(fig)

    # 3. Figure 3: Per-Class F1 Score Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CATEGORIES))
    model_f1 = test_eval["per_class_f1"]
    bars = ax.bar(x, model_f1, color="#2b5c8f", width=0.55, edgecolor="black", label="Selected Model")

    for bar, val in zip(bars, model_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Class F1 Scores on TEST (Macro F1 = {test_eval['macro_f1']:.3f}, Acc = {test_eval['accuracy']*100:.1f}%)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "per_class_f1.png")
    fig.savefig(f3_path, dpi=200)
    plt.close(fig)

    # 4. Figure 4: Actual vs Predicted Wind Scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    true_w = np.array(test_eval["y_true_wind"])
    pred_w = np.array(test_eval["y_pred_wind"])

    ax.scatter(true_w, pred_w, alpha=0.6, color="#1f77b4", edgecolor="k", s=35, label="Test Predictions")
    min_v = 15.0
    max_v = max(true_w.max(), pred_w.max()) + 10.0
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Identity (1:1)")
    ax.fill_between([min_v, max_v], [min_v - 10, max_v - 10], [min_v + 10, max_v + 10], color="gray", alpha=0.15, label="±10 kt Margin")

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Actual IMD Sustained Wind (kt)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Sustained Wind (kt)", fontsize=12, fontweight="bold")
    ax.set_title(f"Actual vs Predicted Wind on TEST\nMAE: {test_eval['wind_mae']:.2f} kt | RMSE: {test_eval['wind_rmse']:.2f} kt | Corr: {test_eval['wind_corr']:.3f}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "wind_scatter.png")
    fig.savefig(f4_path, dpi=200)
    plt.close(fig)

    # 5. Figure 5: Wind Residual Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = pred_w - true_w
    n, bins_h, _ = ax.hist(residuals, bins=25, color="#4a90e2", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Bias Line")
    ax.axvline(test_eval["wind_bias"], color="green", linestyle="-", linewidth=2, label=f"Mean Bias ({test_eval['wind_bias']:+.2f} kt)")

    ax.set_xlabel("Residual: Predicted - Actual Wind (kt)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(f"Wind Residual Error Distribution (N={len(residuals)})\nMean: {test_eval['wind_bias']:+.2f} kt | Median AE: {test_eval['wind_median_ae']:.2f} kt | P90 AE: {test_eval['wind_p90_ae']:.2f} kt", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    f5_path = os.path.join(FIGURES_DIR, "wind_residuals.png")
    fig.savefig(f5_path, dpi=200)
    plt.close(fig)

    # 6. Figure 6: Wind Error vs Actual Intensity Category
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_indices = np.array(test_eval["y_true_cat"])
    cat_maes = []
    cat_labels = []
    for c_idx, c_name in enumerate(CATEGORIES):
        mask = (cat_indices == c_idx)
        if mask.sum() > 0:
            c_mae = float(np.mean(np.abs(residuals[mask])))
            cat_maes.append(c_mae)
            cat_labels.append(f"{c_name}\n(N={mask.sum()})")
        else:
            cat_maes.append(0.0)
            cat_labels.append(f"{c_name}\n(N=0)")

    bars = ax.bar(range(len(CATEGORIES)), cat_maes, color="#e67e22", edgecolor="black", width=0.55)
    for bar, val in zip(bars, cat_maes):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f} kt", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(cat_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Wind MAE (kt)", fontsize=12, fontweight="bold")
    ax.set_title("Wind Speed Regression Error across Actual IMD Intensity Stages", fontsize=13, fontweight="bold")
    plt.tight_layout()
    f6_path = os.path.join(FIGURES_DIR, "wind_error_by_intensity.png")
    fig.savefig(f6_path, dpi=200)
    plt.close(fig)

    # 7. Figure 7: Wind MAE by Basin (Arabian Sea vs Bay of Bengal)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lons = sub_df["lon"].values
    as_mask = (lons < 77.5)
    bob_mask = (lons >= 77.5)

    as_mae = float(np.mean(np.abs(residuals[as_mask])))
    bob_mae = float(np.mean(np.abs(residuals[bob_mask])))

    as_true_cats = cat_indices[as_mask]
    as_pred_cats = np.array(test_eval["y_pred_cat"])[as_mask]
    bob_true_cats = cat_indices[bob_mask]
    bob_pred_cats = np.array(test_eval["y_pred_cat"])[bob_mask]

    as_f1 = float(f1_score(as_true_cats, as_pred_cats, average="macro", zero_division=0))
    bob_f1 = float(f1_score(bob_true_cats, bob_pred_cats, average="macro", zero_division=0))

    # Bar 1: Wind MAE
    bars1 = axes[0].bar(["Arabian Sea\n(N=" + str(as_mask.sum()) + ")", "Bay of Bengal\n(N=" + str(bob_mask.sum()) + ")"],
                        [as_mae, bob_mae], color=["#16a085", "#2980b9"], edgecolor="black", width=0.45)
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f"{b.get_height():.2f} kt", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Wind MAE (kt)", fontsize=11, fontweight="bold")
    axes[0].set_title("Wind MAE by Ocean Basin", fontsize=12, fontweight="bold")

    # Bar 2: Category Macro F1
    bars2 = axes[1].bar(["Arabian Sea", "Bay of Bengal"], [as_f1, bob_f1], color=["#16a085", "#2980b9"], edgecolor="black", width=0.45)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Macro F1", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Intensity Macro F1 by Ocean Basin", fontsize=12, fontweight="bold")

    plt.tight_layout()
    f7_path = os.path.join(FIGURES_DIR, "wind_mae_by_basin.png")
    fig.savefig(f7_path, dpi=200)
    plt.close(fig)

    # 8. Figure 8: Calibration Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    bins = [b["confidence"] for b in test_eval["calibration"]["bins"] if b["count"] > 0]
    accs = [b["accuracy"] for b in test_eval["calibration"]["bins"] if b["count"] > 0]

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bins, accs, marker="o", linewidth=2, color="#8e44ad", label=f"Calibrated (T={test_eval['temperature']:.2f}, ECE={test_eval['calibration']['ece']:.3f})")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Probability Calibration Diagram (TEST)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    f8_path = os.path.join(FIGURES_DIR, "calibration_reliability.png")
    fig.savefig(f8_path, dpi=200)
    plt.close(fig)

    # 9. Figure 9: Sample Prediction Diagnostic Case
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Select a high-performing or representative mature storm sample
    idx_sample = int(np.argmin(np.abs(residuals))) # lowest residual sample
    s_row = sub_df.iloc[idx_sample]

    probs_sample = test_eval["y_pred_probs"][idx_sample]
    bars_p = ax.bar(CATEGORIES, probs_sample, color="#34495e", edgecolor="black", width=0.55)
    true_cat_idx = test_eval["y_true_cat"][idx_sample]
    bars_p[true_cat_idx].set_color("#27ae60") # highlight ground truth

    ax.set_ylabel("Calibrated Probability", fontsize=11, fontweight="bold")
    ax.set_title(f"Sample Case: {s_row['sample_id']}\nTrue: {CATEGORIES[true_cat_idx]} ({s_row['true_wind']:.0f} kt) | Predicted: {CATEGORIES[test_eval['y_pred_cat'][idx_sample]]} ({test_eval['y_pred_wind'][idx_sample]:.1f} kt ± {test_eval['uncertainty_bounds']['p80_ae']:.1f} kt)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    f9_path = os.path.join(FIGURES_DIR, "sample_prediction.png")
    fig.savefig(f9_path, dpi=200)
    plt.close(fig)

    # 10. Figure 10: Error Cases & Outlier Visualizations
    fig, ax = plt.subplots(figsize=(10, 5))
    top_err_indices = np.argsort(np.abs(residuals))[-10:][::-1]
    top_errs = residuals[top_err_indices]
    top_names = [f"{sub_df.iloc[i]['sample_id'][:15]}... ({sub_df.iloc[i]['true_cat']})" for i in top_err_indices]

    y_pos = np.arange(len(top_names))
    colors = ["#c0392b" if e > 0 else "#2980b9" for e in top_errs]
    ax.barh(y_pos, top_errs, color=colors, edgecolor="black", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Prediction Error (Pred - True Wind in kt)", fontsize=11, fontweight="bold")
    ax.set_title("Top 10 Largest Wind Regression Errors on TEST", fontsize=13, fontweight="bold")
    plt.tight_layout()
    f10_path = os.path.join(FIGURES_DIR, "error_cases.png")
    fig.savefig(f10_path, dpi=200)
    plt.close(fig)

    print(f"[Visualizations] All 10 diagnostic figures generated in {FIGURES_DIR}/")


def main():
    print("=" * 80)
    print("VAYU-NET PHASE 6 — PRODUCTION TEST EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load frozen checkpoint
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    print(f"Loading frozen checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    exp_name = ckpt["experiment_name"]
    mode = ckpt["mode"]
    best_epoch = ckpt["best_epoch"]
    temperature = ckpt["temperature"]
    val_metrics = ckpt["val_metrics"]
    unc_bounds = ckpt["uncertainty_bounds"]

    print(f"Loaded Winning Experiment: {exp_name} (Mode: {mode}, Best Epoch: {best_epoch})")
    print(f"Fitted Temperature: T = {temperature:.4f}")
    print(f"Validation Residual Error Bands: Median={unc_bounds['median_ae']:.2f} kt, P80={unc_bounds['p80_ae']:.2f} kt, P90={unc_bounds['p90_ae']:.2f} kt")

    # 2. Build model and load weights
    model = Phase6IntensityWindModel(
        mode=mode,
        latent_dim=ckpt["config"]["latent_dim"],
        gru_num_layers=ckpt["config"]["gru_num_layers"],
        dropout=ckpt["config"]["dropout"],
        train_wind_mean=ckpt["config"]["train_wind_mean"],
        train_wind_std=ckpt["config"]["train_wind_std"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load dataset
    print(f"Loading dataset cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    samples = cache["samples"]

    train_samples = [s for s in samples if s["split"] == "TRAIN"]
    test_samples = [s for s in samples if s["split"] == "TEST"]
    print(f"Test samples count: {len(test_samples)} across 31 storms")

    test_loader = DataLoader(
        Phase6CachedDataset(samples, split="TEST"),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_phase6
    )

    # 4. Strictly SINGLE-PASS TEST evaluation
    print("Executing single-pass evaluation on TEST split...")
    all_pred_cats, all_true_cats = [], []
    all_uncal_probs, all_cal_probs = [], []
    all_pred_winds_kt, all_true_winds_kt = [], []
    sample_records = []

    with torch.no_grad():
        for batch in test_loader:
            sat_seq = batch["sat_seq"].to(device)
            sat_t0 = batch["sat_t0"].to(device)
            env_seq = batch["env_seq"].to(device)
            cat_targets = batch["category_t0"].to(device)

            out = model(sat_seq=sat_seq, sat_t0=sat_t0, env_seq=env_seq)

            # Uncalibrated vs Calibrated probabilities
            uncal_p = F.softmax(out["category_logits"], dim=-1).cpu().numpy()
            cal_p = out["category_probs"].cpu().numpy()
            p_cat = out["category_pred"].cpu().numpy()
            t_cat = cat_targets.cpu().numpy()

            p_wind = out["pred_wind_kt"].cpu().numpy()
            t_wind = batch["wind_t0"].numpy()

            for i in range(len(t_cat)):
                tc = int(t_cat[i])
                pc = int(p_cat[i])
                tw = float(t_wind[i])
                pw = float(p_wind[i])

                all_pred_cats.append(pc)
                all_true_cats.append(tc)
                all_uncal_probs.append(uncal_p[i])
                all_cal_probs.append(cal_p[i])
                all_pred_winds_kt.append(pw)
                all_true_winds_kt.append(tw)

                sample_records.append({
                    "sample_id": batch["sample_id"][i],
                    "storm_id": batch["storm_id"][i],
                    "t0": batch["t0"][i],
                    "lat": float(batch["lat_t0"][i]),
                    "lon": float(batch["lon_t0"][i]),
                    "true_cat": CATEGORIES[tc],
                    "pred_cat": CATEGORIES[pc],
                    "true_wind": tw,
                    "pred_wind": pw,
                    "wind_error": pw - tw,
                    "abs_wind_error": abs(pw - tw)
                })

    sub_df = pd.DataFrame(sample_records)

    # 5. TEST Metrics Computation
    # Classification
    test_acc = float(accuracy_score(all_true_cats, all_pred_cats))
    test_macro_f1 = float(f1_score(all_true_cats, all_pred_cats, average="macro", zero_division=0))
    test_macro_prec = float(precision_score(all_true_cats, all_pred_cats, average="macro", zero_division=0))
    test_macro_rec = float(recall_score(all_true_cats, all_pred_cats, average="macro", zero_division=0))
    test_per_class_f1 = f1_score(all_true_cats, all_pred_cats, average=None, labels=list(range(7)), zero_division=0).tolist()
    test_cm = confusion_matrix(all_true_cats, all_pred_cats, labels=list(range(7))).tolist()

    # Probability calibration (ECE)
    uncal_ece, uncal_bins = compute_ece(np.array(all_uncal_probs), np.array(all_true_cats))
    cal_ece, cal_bins = compute_ece(np.array(all_cal_probs), np.array(all_true_cats))

    # Wind Regression
    diffs = np.array(all_pred_winds_kt) - np.array(all_true_winds_kt)
    abs_diffs = np.abs(diffs)

    test_wind_mae = float(np.mean(abs_diffs))
    test_wind_rmse = float(np.sqrt(np.mean(diffs**2)))
    test_wind_median_ae = float(np.median(abs_diffs))
    test_wind_p90_ae = float(np.percentile(abs_diffs, 90))
    test_wind_bias = float(np.mean(diffs))
    corr_matrix = np.corrcoef(all_pred_winds_kt, all_true_winds_kt)
    test_wind_corr = float(corr_matrix[0, 1])

    # Empirical Uncertainty Coverage
    coverage_p80 = float(np.mean(abs_diffs <= unc_bounds["p80_ae"])) * 100.0
    coverage_p90 = float(np.mean(abs_diffs <= unc_bounds["p90_ae"])) * 100.0

    # 6. Baselines Comparison
    sample_index_df = pd.read_csv(SAMPLE_INDEX_PATH)
    bt_v2_df = pd.read_csv(BT_V2_PATH)
    baselines = evaluate_baselines(test_samples, train_samples, sample_index_df, bt_v2_df)

    # 7. Subgroup Analysis
    # Weak (D, DD: < 34 kt)
    weak_mask = (sub_df["true_wind"] < 34.0)
    weak_mae = float(sub_df[weak_mask]["abs_wind_error"].mean())
    weak_f1 = float(f1_score(sub_df[weak_mask]["true_cat"], sub_df[weak_mask]["pred_cat"], average="macro", zero_division=0))

    # Moderate / Strong (CS, SCS: 34 to 63 kt)
    mod_mask = (sub_df["true_wind"] >= 34.0) & (sub_df["true_wind"] < 64.0)
    mod_mae = float(sub_df[mod_mask]["abs_wind_error"].mean())
    mod_f1 = float(f1_score(sub_df[mod_mask]["true_cat"], sub_df[mod_mask]["pred_cat"], average="macro", zero_division=0))

    # Mature / Intense (VSCS, ESCS, SuCS: >= 64 kt)
    intense_mask = (sub_df["true_wind"] >= 64.0)
    intense_mae = float(sub_df[intense_mask]["abs_wind_error"].mean())
    intense_f1 = float(f1_score(sub_df[intense_mask]["true_cat"], sub_df[intense_mask]["pred_cat"], average="macro", zero_division=0))

    # Basins: Arabian Sea (< 77.5 E) vs Bay of Bengal (>= 77.5 E)
    as_mask = (sub_df["lon"] < 77.5)
    bob_mask = (sub_df["lon"] >= 77.5)
    as_mae = float(sub_df[as_mask]["abs_wind_error"].mean())
    bob_mae = float(sub_df[bob_mask]["abs_wind_error"].mean())

    test_eval = {
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "macro_precision": test_macro_prec,
        "macro_recall": test_macro_rec,
        "per_class_f1": test_per_class_f1,
        "confusion_matrix": test_cm,
        "calibration": {
            "uncalibrated_ece": uncal_ece,
            "calibrated_ece": cal_ece,
            "ece": cal_ece,
            "temperature": temperature,
            "bins": cal_bins
        },
        "temperature": temperature,
        "uncertainty_bounds": unc_bounds,
        "wind_mae": test_wind_mae,
        "wind_rmse": test_wind_rmse,
        "wind_median_ae": test_wind_median_ae,
        "wind_p90_ae": test_wind_p90_ae,
        "wind_bias": test_wind_bias,
        "wind_corr": test_wind_corr,
        "empirical_uncertainty_coverage": {
            "p80_target_coverage_pct": 80.0,
            "p80_empirical_coverage_pct": coverage_p80,
            "p90_target_coverage_pct": 90.0,
            "p90_empirical_coverage_pct": coverage_p90
        },
        "y_true_cat": all_true_cats,
        "y_pred_cat": all_pred_cats,
        "y_pred_probs": all_cal_probs,
        "y_true_wind": all_true_winds_kt,
        "y_pred_wind": all_pred_winds_kt
    }

    # 8. Generate Visualizations
    with open(HISTORY_PATH, "r") as f:
        history_data = json.load(f)
    generate_all_figures(history_data, test_eval, baselines, sub_df)

    # 9. Assemble Complete Deliverable Payload
    results_payload = {
        "metadata": {
            "phase": "Phase 6",
            "title": "VAYU-NET Multi-Task Cyclone Intensity Classification & Wind Speed Regression",
            "problem_statement": "SIH 2026 PS 26070",
            "timestamp_evaluation": pd.Timestamp.now().isoformat(),
            "selected_model": exp_name,
            "mode": mode,
            "best_epoch": best_epoch,
            "total_test_samples": len(test_samples),
            "test_storms_count": 31,
            "frozen_checkpoint_path": CHECKPOINT_PATH
        },
        "validation_metrics": val_metrics,
        "test_metrics": {
            "classification": {
                "accuracy": test_acc,
                "macro_f1": test_macro_f1,
                "macro_precision": test_macro_prec,
                "macro_recall": test_macro_rec,
                "per_class_f1": dict(zip(CATEGORIES, test_per_class_f1)),
                "confusion_matrix": test_cm,
                "uncalibrated_ece": uncal_ece,
                "calibrated_ece": cal_ece,
                "temperature": temperature
            },
            "wind_regression": {
                "mae_kt": test_wind_mae,
                "rmse_kt": test_wind_rmse,
                "median_ae_kt": test_wind_median_ae,
                "p90_ae_kt": test_wind_p90_ae,
                "bias_kt": test_wind_bias,
                "pearson_r": test_wind_corr,
                "uncertainty_p80_ae_kt": unc_bounds["p80_ae"],
                "uncertainty_p90_ae_kt": unc_bounds["p90_ae"],
                "coverage_p80_pct": coverage_p80,
                "coverage_p90_pct": coverage_p90
            }
        },
        "baselines": baselines,
        "subgroup_analysis": {
            "weak_systems_under_34kt": {
                "sample_count": int(weak_mask.sum()),
                "wind_mae_kt": weak_mae,
                "category_macro_f1": weak_f1
            },
            "moderate_systems_34_to_63kt": {
                "sample_count": int(mod_mask.sum()),
                "wind_mae_kt": mod_mae,
                "category_macro_f1": mod_f1
            },
            "intense_systems_64kt_plus": {
                "sample_count": int(intense_mask.sum()),
                "wind_mae_kt": intense_mae,
                "category_macro_f1": intense_f1
            },
            "arabian_sea": {
                "sample_count": int(as_mask.sum()),
                "wind_mae_kt": as_mae
            },
            "bay_of_bengal": {
                "sample_count": int(bob_mask.sum()),
                "wind_mae_kt": bob_mae
            }
        },
        "top_5_largest_wind_errors": sub_df.sort_values("abs_wind_error", ascending=False).head(5)[[
            "sample_id", "storm_id", "t0", "true_cat", "pred_cat", "true_wind", "pred_wind", "abs_wind_error"
        ]].to_dict(orient="records")
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"[Results JSON] Successfully saved to {RESULTS_JSON_PATH}")

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL PHASE 6 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Selected Model: {exp_name} (Epoch {best_epoch})")
    print(f"Temperature Calibration: T = {temperature:.4f} (ECE: {uncal_ece:.3f} -> {cal_ece:.3f})")
    print("\n--- CLASSIFICATION ---")
    print(f"Accuracy:         {test_acc*100:.2f}% (Majority Baseline: {baselines['majority_category']['accuracy']*100:.2f}%)")
    print(f"Macro Precision:  {test_macro_prec:.4f}")
    print(f"Macro Recall:     {test_macro_rec:.4f}")
    print(f"Macro F1:         {test_macro_f1:.4f} (Majority Baseline: {baselines['majority_category']['macro_f1']:.4f})")
    print("\n--- PER-CLASS F1 ---")
    for c, f in zip(CATEGORIES, test_per_class_f1):
        print(f"  {c:5s}: {f:.4f}")

    print("\n--- MAXIMUM SUSTAINED WIND (kt) ---")
    print(f"MAE:              {test_wind_mae:.2f} kt (Mean Baseline: {baselines['mean_wind']['mae_kt']:.2f} kt, -{baselines['mean_wind']['mae_kt'] - test_wind_mae:.2f} kt)")
    print(f"RMSE:             {test_wind_rmse:.2f} kt (Mean Baseline: {baselines['mean_wind']['rmse_kt']:.2f} kt)")
    print(f"Median AE:        {test_wind_median_ae:.2f} kt")
    print(f"P90 AE:           {test_wind_p90_ae:.2f} kt")
    print(f"Bias:             {test_wind_bias:+.2f} kt")
    print(f"Pearson r:        {test_wind_corr:.4f}")
    print(f"Empirical P80 Coverage: {coverage_p80:.1f}% (target ~80%)")
    print(f"Empirical P90 Coverage: {coverage_p90:.1f}% (target ~90%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
