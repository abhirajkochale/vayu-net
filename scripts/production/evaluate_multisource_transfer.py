"""
VAYU-NET — MULTI-SOURCE SATELLITE TRANSFER LEARNING EVALUATOR
==============================================================
Production verification script to evaluate trained transfer learning checkpoints:
  - EXP-1: Frozen GridSat + Trainable Heads
  - EXP-2: Frozen GridSat + Lightweight INSAT + Fusion
  - EXP-3: Partially Unfrozen GridSat + Lightweight INSAT + Fusion
  - Channel Ablation Checkpoints (TIR1 only, TIR1+TIR2, All 3)
against reference baselines on the locked 298 TEST samples.
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

from ml.models.multisource_transfer_fusion import MultisourceTransferFusionModel
from ml.train.train_multisource_transfer import (
    MultisourceTransferDataset,
    evaluate_transfer_loader,
    build_unified_transfer_cache
)
from ml.train.train_multisource_ablation import compute_baselines_on_test


def evaluate_transfer_checkpoint(ckpt_path: str,
                                 cache_data: Dict[str, Any],
                                 split: str = "TEST",
                                 device: str = "cpu",
                                 in_channels_insat: int = 3,
                                 insat_channels=None) -> Dict[str, Any]:
    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    mode = ckpt["mode"]
    model = MultisourceTransferFusionModel(
        mode=mode,
        in_channels_insat=in_channels_insat
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = MultisourceTransferDataset(cache_data, split=split, insat_channels=insat_channels)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    loss, metrics = evaluate_transfer_loader(model, loader, dev)
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
    parser = argparse.ArgumentParser(description="Evaluate Multi-Source Transfer Models")
    parser.add_argument("--cache", type=str, default="data/interim/ml/cache/multisource_transfer_cache.pt")
    parser.add_argument("--split", type=str, default="TEST")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cache_data = build_unified_transfer_cache(args.cache)

    ckpts = {
        "EXP-1 (Frozen GridSat)": ("data/interim/ml/checkpoints/multisource_transfer_exp1.pt", 3, None),
        "EXP-2 (Fusion Frozen)": ("data/interim/ml/checkpoints/multisource_transfer_exp2.pt", 3, None),
        "EXP-3 (Fusion Partial)": ("data/interim/ml/checkpoints/multisource_transfer_exp3.pt", 3, None),
        "Channel Ablation: TIR1": ("data/interim/ml/checkpoints/multisource_transfer_ablation_tir1.pt", 1, [0]),
        "Channel Ablation: TIR1+TIR2": ("data/interim/ml/checkpoints/multisource_transfer_ablation_tir1_tir2.pt", 2, [0, 1]),
    }

    print("=" * 80)
    print(f"VAYU-NET MULTI-SOURCE TRANSFER LEARNING EVALUATION ON {args.split} SPLIT")
    print("=" * 80)

    for name, (cpath, in_ch, ch_idx) in ckpts.items():
        if os.path.exists(cpath):
            res = evaluate_transfer_checkpoint(cpath, cache_data, split=args.split, device=args.device,
                                               in_channels_insat=in_ch, insat_channels=ch_idx)
            m = res["metrics"]
            pc = res["parameter_counts"]
            print(f"\n--- {name} ---")
            print(f"  Params: Total={pc['total_parameters']:,} | Trainable={pc['trainable_parameters']:,} | Frozen={pc['frozen_parameters']:,}")
            print(f"  Loss: {res['loss']:.4f} (Selected Epoch: {res['best_epoch']})")
            print(f"  Center: Mean DPE = {m['identification']['center_mean_dpe_km']:.1f} km, Median = {m['identification']['center_median_dpe_km']:.1f} km")
            print(f"  Intensity: Accuracy = {m['classification']['accuracy']:.4f}, Macro-F1 = {m['classification']['macro_f1']:.4f}")
            print(f"  Wind: MAE = {m['wind_regression']['wind_mae_kt']:.2f} kt, RMSE = {m['wind_regression']['wind_rmse_kt']:.2f} kt, r = {m['wind_regression']['wind_pearson_r']:.3f}")
            print(f"  Track: +12h = {m['track_prediction']['track_12h_mean_dpe_km']:.1f} km, +24h = {m['track_prediction']['track_24h_mean_dpe_km']:.1f} km, +48h = {m['track_prediction']['track_48h_mean_dpe_km']:.1f} km")
            print(f"  Track Aggregate Mean DPE: {m['track_prediction']['track_aggregate_mean_dpe_km']:.1f} km")
        else:
            print(f"Checkpoint not found at {cpath}")


if __name__ == "__main__":
    main()
