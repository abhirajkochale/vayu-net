"""
VAYU-NET WP-06 — EMPIRICAL UNCERTAINTY PARAMETER BUILDER
=========================================================
Extracts exact validation residuals from Phase 5B Variant A (Observed-Center Hybrid Residual Model)
and calculates empirical uncertainty radii for horizons +12h, +24h, +48h.
Guarantees:
  - Strict validation derivation (derivation_split = 'VALIDATION', N=252)
  - Zero test leakage (TEST used only for verification coverage reporting)
  - Produces data/interim/ml/uncertainty_parameters.json
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.phase5b_hybrid_residual import Phase5BHybridResidualModel
from ml.train.train_phase5b_hybrid_residual import (
    Phase5BDataset,
    evaluate_variant,
    CACHE_PATH,
    CKPT_PATH_A
)

OUT_PATH = "data/interim/ml/uncertainty_parameters.json"


def build_uncertainty_parameters():
    print("=" * 80)
    print("BUILDING WP-06 EMPIRICAL UNCERTAINTY PARAMETERS FROM VALIDATION SPLIT")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Loading feature cache from {CACHE_PATH}...")
    cache_data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

    print(f"Loading Phase 5B Variant A checkpoint from {CKPT_PATH_A}...")
    ckpt = torch.load(CKPT_PATH_A, map_location="cpu", weights_only=False)
    model = Phase5BHybridResidualModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. VALIDATION EVALUATION (Strict derivation source)
    print("Evaluating Phase 5B Variant A on VALIDATION split (N=252)...")
    val_ds = Phase5BDataset(cache_data, "VALIDATION", "A")
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    val_metrics = evaluate_variant(model, val_loader, device)

    raw_val = val_metrics["raw"]

    horizons_data = {}
    for h in ["12", "24", "48"]:
        dpes = np.array(raw_val[f"dpe_{h}"])
        h_key = f"{h}h"
        horizons_data[h_key] = {
            "residual_count": int(len(dpes)),
            "mean_dpe_km": round(float(np.mean(dpes)), 2),
            "std_dpe_km": round(float(np.std(dpes)), 2),
            "median_dpe_km": round(float(np.median(dpes)), 2),
            "p50_km": round(float(np.percentile(dpes, 50)), 2),
            "p80_km": round(float(np.percentile(dpes, 80)), 2),
            "p90_km": round(float(np.percentile(dpes, 90)), 2),
            "p95_km": round(float(np.percentile(dpes, 95)), 2)
        }
        print(f"  +{h}h (N={len(dpes)}): Mean={horizons_data[h_key]['mean_dpe_km']} km, Median={horizons_data[h_key]['median_dpe_km']} km, P80={horizons_data[h_key]['p80_km']} km, P90={horizons_data[h_key]['p90_km']} km")

    # 2. TEST EVALUATION (Coverage verification only)
    print("\nVerifying empirical coverage against held-out TEST split (N=371)...")
    test_ds = Phase5BDataset(cache_data, "TEST", "A")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    test_metrics = evaluate_variant(model, test_loader, device)
    raw_test = test_metrics["raw"]

    test_coverage = {}
    for h in ["12", "24", "48"]:
        test_dpes = np.array(raw_test[f"dpe_{h}"])
        h_key = f"{h}h"
        p80_radius = horizons_data[h_key]["p80_km"]
        p90_radius = horizons_data[h_key]["p90_km"]
        cov_80 = float(np.mean(test_dpes <= p80_radius) * 100.0)
        cov_90 = float(np.mean(test_dpes <= p90_radius) * 100.0)
        test_coverage[h_key] = {
            "test_sample_count": len(test_dpes),
            "p80_empirical_radius_km": p80_radius,
            "test_coverage_p80_pct": round(cov_80, 2),
            "p90_empirical_radius_km": p90_radius,
            "test_coverage_p90_pct": round(cov_90, 2)
        }
        print(f"  +{h}h Coverage on TEST: P80 ({p80_radius} km) -> {cov_80:.1f}%, P90 ({p90_radius} km) -> {cov_90:.1f}%")

    payload = {
        "metadata": {
            "title": "VAYU-NET Empirical Track Forecast Uncertainty Parameters",
            "version": "1.0",
            "date": "2026-09-25",
            "derivation_split": "VALIDATION",
            "derivation_model": "Phase 5B Variant A Hybrid Residual",
            "derivation_samples_count": 252,
            "derivation_storms_count": 14,
            "label": "empirical",
            "methodology": "Empirical Great-Circle Distance Prediction Error (DPE) percentiles derived strictly from held-out validation residuals."
        },
        "derivation_split": "VALIDATION",
        "horizons": horizons_data,
        "test_held_out_verification_coverage": test_coverage
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSuccessfully saved uncertainty parameters to {OUT_PATH}")
    return payload


if __name__ == "__main__":
    build_uncertainty_parameters()
