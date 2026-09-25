"""
VAYU-NET — MULTI-SOURCE SATELLITE MODEL EVALUATION & INFERENCE BENCHMARK
=========================================================================
Module: Production Verification & Modality Comparison Evaluator
Evaluates trained multi-source checkpoints:
  - Model A: GridSat Only (data/interim/ml/checkpoints/multisource_gridsat_model_a.pt)
  - Model B: INSAT Only (data/interim/ml/checkpoints/multisource_insat_model_b.pt)
  - Model C: GridSat + INSAT Fusion (data/interim/ml/checkpoints/multisource_fusion_model_c.pt)
against Persistence and Constant Velocity baselines on the locked TEST set.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from ml.models.multisource_fusion import MultisourceFusionModel
from ml.train.train_multisource_fusion import MultisourceCachedDataset, evaluate_loader
from ml.train.train_multisource_ablation import compute_baselines_on_test


def evaluate_checkpoint(ckpt_path: str, cache_data: Dict[str, Any], split: str = "TEST", device: str = "cpu") -> Dict[str, Any]:
    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    mode = ckpt["mode"]
    model = MultisourceFusionModel(mode=mode).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = MultisourceCachedDataset(cache_data, split=split)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    loss, metrics = evaluate_loader(model, loader, dev)
    return {
        "mode": mode,
        "checkpoint": str(ckpt_path),
        "split": split,
        "loss": loss,
        "best_epoch": ckpt.get("best_epoch", -1),
        "parameter_counts": ckpt.get("parameter_counts", model.get_parameter_counts()),
        "metrics": metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Source Models on Test Split")
    parser.add_argument("--cache", type=str, default="data/interim/ml/cache/multisource_dataset_cache.pt")
    parser.add_argument("--split", type=str, default="TEST")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    assert os.path.exists(args.cache), f"Dataset cache missing at {args.cache}"
    cache_data = torch.load(args.cache, map_location="cpu", weights_only=False)

    ckpts = {
        "GridSat Only (Model A)": "data/interim/ml/checkpoints/multisource_gridsat_model_a.pt",
        "INSAT Only (Model B)": "data/interim/ml/checkpoints/multisource_insat_model_b.pt",
        "Fusion (Model C)": "data/interim/ml/checkpoints/multisource_fusion_model_c.pt"
    }

    print("=" * 80)
    print(f"VAYU-NET MULTI-SOURCE EVALUATION ON {args.split} SPLIT")
    print("=" * 80)

    results = {}
    for name, cpath in ckpts.items():
        if os.path.exists(cpath):
            eval_res = evaluate_checkpoint(cpath, cache_data, split=args.split, device=args.device)
            results[name] = eval_res
            m = eval_res["metrics"]
            print(f"\n--- {name} ---")
            print(f"  Params: Total={eval_res['parameter_counts']['total_parameters']:,} | Trainable={eval_res['parameter_counts']['trainable_parameters']:,}")
            print(f"  Loss: {eval_res['loss']:.4f} (Selected Epoch: {eval_res['best_epoch']})")
            print(f"  Center Localization: Mean DPE = {m['identification']['center_mean_dpe_km']:.1f} km, Median = {m['identification']['center_median_dpe_km']:.1f} km")
            print(f"  Intensity Classification: Acc = {m['classification']['accuracy']:.4f}, Macro-F1 = {m['classification']['macro_f1']:.4f}")
            print(f"  Wind Regression: MAE = {m['wind_regression']['wind_mae_kt']:.2f} kt, RMSE = {m['wind_regression']['wind_rmse_kt']:.2f} kt, r = {m['wind_regression']['wind_pearson_r']:.3f}")
            print(f"  Track Prediction: +12h = {m['track_prediction']['track_12h_mean_dpe_km']:.1f} km, +24h = {m['track_prediction']['track_24h_mean_dpe_km']:.1f} km, +48h = {m['track_prediction']['track_48h_mean_dpe_km']:.1f} km")
            print(f"  Track Aggregate: Mean DPE = {m['track_prediction']['track_aggregate_mean_dpe_km']:.1f} km")
        else:
            print(f"Checkpoint not found at {cpath}")

    # Baselines
    if args.split == "TEST":
        print("\n" + "=" * 80)
        print("REFERENCE BASELINES (Evaluated on Identical 298 TEST samples)")
        print("=" * 80)
        base = compute_baselines_on_test()
        p_track = base["persistence_baseline"]["track"]
        p_wind = base["persistence_baseline"]["wind"]
        cv_track = base["constant_velocity_baseline"]["track"]
        print(f"Persistence Baseline:")
        print(f"  Wind MAE: {p_wind['wind_mae_kt']:.2f} kt | Track Agg DPE: {p_track['track_aggregate_mean_dpe_km']:.1f} km (+12h: {p_track['track_12h_mean_dpe_km']:.1f} km, +24h: {p_track['track_24h_mean_dpe_km']:.1f} km, +48h: {p_track['track_48h_mean_dpe_km']:.1f} km)")
        print(f"Constant Velocity Baseline:")
        print(f"  Track Agg DPE: {cv_track['track_aggregate_mean_dpe_km']:.1f} km (+12h: {cv_track['track_12h_mean_dpe_km']:.1f} km, +24h: {cv_track['track_24h_mean_dpe_km']:.1f} km, +48h: {cv_track['track_48h_mean_dpe_km']:.1f} km)")


if __name__ == "__main__":
    main()
