"""
VAYU-NET — DOWNSTREAM EVALUATION: RANDOM VS PRETRAINED INSAT ENCODER
=====================================================================
Executes the downstream transfer learning test on the locked multi-source dataset:
  - 207 TRAIN samples (25 storms)
  - 252 VAL samples (14 storms)
  - 298 TEST samples (24 storms)

Evaluates:
  1. CONTROL: Decoupled Full Model with RANDOMLY INITIALIZED INSAT branches
  2. PRETRAINED: Decoupled Full Model with SELF-SUPERVISED PRETRAINED INSAT branches
  3. ABLATION A: Decoupled Track with PRETRAINED Dual-IR (TIR1+TIR2)
  4. ABLATION B: Decoupled Intensity with PRETRAINED 3-Channel (TIR1+TIR2+WV)

Saves results to:
  data/interim/ml/insat_downstream_pretrained_results.json
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.multisource_decoupled import MultisourceDecoupledModel
from ml.train.train_multisource_decoupled import (
    MultisourceDecoupledDataset,
    evaluate_decoupled_loader,
    set_seed
)


def train_downstream_model(exp_name: str,
                           mode: str,
                           cache_data: Dict[str, Any],
                           device: torch.device,
                           pretrained_2ch_path: Optional[str] = None,
                           pretrained_3ch_path: Optional[str] = None,
                           epochs: int = 20,
                           batch_size: int = 16,
                           lr: float = 1e-3,
                           patience: int = 5,
                           ckpt_out: Optional[str] = None) -> Dict[str, Any]:
    set_seed(42)
    print(f"\n=======================================================")
    print(f"DOWNSTREAM TRAINING: {exp_name.upper()}")
    print(f"=======================================================")

    train_ds = MultisourceDecoupledDataset(cache_data, "TRAIN")
    val_ds = MultisourceDecoupledDataset(cache_data, "VALIDATION")
    test_ds = MultisourceDecoupledDataset(cache_data, "TEST")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultisourceDecoupledModel(mode=mode).to(device)

    # Inject pretrained weights if provided
    if pretrained_2ch_path and model.insat_ir_branch is not None:
        p2 = torch.load(pretrained_2ch_path, map_location="cpu", weights_only=False)
        sd2 = p2["encoder_state_dict"] if "encoder_state_dict" in p2 else p2
        model.insat_ir_branch.spatial.load_state_dict(sd2)
        print(f"  [Init] Successfully loaded 2-channel pretrained spatial weights from {pretrained_2ch_path}")

    if pretrained_3ch_path and model.insat_wv_branch is not None:
        p3 = torch.load(pretrained_3ch_path, map_location="cpu", weights_only=False)
        sd3 = p3["encoder_state_dict"] if "encoder_state_dict" in p3 else p3
        model.insat_wv_branch.spatial.load_state_dict(sd3)
        print(f"  [Init] Successfully loaded 3-channel pretrained spatial weights from {pretrained_3ch_path}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = None
    best_weights = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            g_feat = batch["gridsat_feat_seq"].to(device)
            i_seq = batch["insat_seq"].to(device)

            out = model(gridsat_feat_seq=g_feat, insat_seq=i_seq)

            c_norm = batch["center_norm"].to(device)
            cat_idx = batch["category_idx"].to(device)
            w_norm = batch["wind_norm"].to(device)
            t12_norm = batch["t12_norm"].to(device)
            m12 = batch["mask_12"].to(device)
            t24_norm = batch["t24_norm"].to(device)
            m24 = batch["mask_24"].to(device)
            t48_norm = batch["t48_norm"].to(device)
            m48 = batch["mask_48"].to(device)

            loss_c = nn.functional.smooth_l1_loss(out["center_norm"], c_norm)
            loss_cls = nn.functional.cross_entropy(out["class_logits"], cat_idx)
            loss_w = nn.functional.smooth_l1_loss(out["wind_norm"], w_norm)

            loss_12 = (nn.functional.smooth_l1_loss(out["track_12h_norm"], t12_norm, reduction="none") * m12.unsqueeze(-1)).sum() / (m12.sum() * 2.0 + 1e-6)
            loss_24 = (nn.functional.smooth_l1_loss(out["track_24h_norm"], t24_norm, reduction="none") * m24.unsqueeze(-1)).sum() / (m24.sum() * 2.0 + 1e-6)
            loss_48 = (nn.functional.smooth_l1_loss(out["track_48h_norm"], t48_norm, reduction="none") * m48.unsqueeze(-1)).sum() / (m48.sum() * 2.0 + 1e-6)
            loss_track = (loss_12 + loss_24 + loss_48) / 3.0

            loss = loss_c + loss_cls + loss_w + loss_track
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train += 1

        avg_train_loss = train_loss / max(1, n_train)
        val_loss, val_m = evaluate_decoupled_loader(model, val_loader, device)
        scheduler.step(val_loss)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Center: {val_m['identification']['center_mean_dpe_km']:.1f}km | "
              f"Val Wind: {val_m['wind_regression']['wind_mae_kt']:.1f}kt | "
              f"Val Track: {val_m['track_prediction']['track_aggregate_mean_dpe_km']:.1f}km")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_center_dpe_km": round(val_m["identification"]["center_mean_dpe_km"], 2),
            "val_wind_mae_kt": round(val_m["wind_regression"]["wind_mae_kt"], 2),
            "val_track_dpe_km": round(val_m["track_prediction"]["track_aggregate_mean_dpe_km"], 2),
            "lr": curr_lr
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_m
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  >>> Best validation checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nEvaluating Best Checkpoint (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}) on TEST set...")
    model.load_state_dict(best_weights)
    test_loss, test_m = evaluate_decoupled_loader(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f} | Center: {test_m['identification']['center_mean_dpe_km']:.1f}km | "
          f"Wind: {test_m['wind_regression']['wind_mae_kt']:.1f}kt | "
          f"Track: {test_m['track_prediction']['track_aggregate_mean_dpe_km']:.1f}km")

    if ckpt_out:
        Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "exp_name": exp_name,
            "mode": mode,
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "model_state_dict": best_weights,
            "test_metrics": test_m,
            "test_loss": test_loss,
            "history": history
        }, ckpt_out)
        print(f"Saved model checkpoint to {ckpt_out}")

    return {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_m,
        "history": history
    }


def main():
    device = torch.device("cpu")
    cache_path = Path("data/interim/ml/cache/multisource_transfer_cache_expanded.pt")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    ckpt_2ch = "data/interim/ml/checkpoints/insat_pretrained_encoder_2ch.pt"
    ckpt_3ch = "data/interim/ml/checkpoints/insat_pretrained_encoder.pt"

    results = {}

    # 1. CONTROL: Random Initialization (Decoupled Full Model)
    res_random = train_downstream_model(
        exp_name="Random INSAT (Control)",
        mode="arch_c_decoupled_full",
        cache_data=cache_data,
        device=device,
        pretrained_2ch_path=None,
        pretrained_3ch_path=None,
        ckpt_out="data/interim/ml/checkpoints/multisource_downstream_control_random.pt"
    )
    results["Random INSAT (Control)"] = res_random

    # 2. PRETRAINED: Pretrained Initialization (Decoupled Full Model)
    res_pretrained = train_downstream_model(
        exp_name="Pretrained INSAT",
        mode="arch_c_decoupled_full",
        cache_data=cache_data,
        device=device,
        pretrained_2ch_path=ckpt_2ch,
        pretrained_3ch_path=ckpt_3ch,
        ckpt_out="data/interim/ml/checkpoints/multisource_downstream_pretrained.pt"
    )
    results["Pretrained INSAT"] = res_pretrained

    # 3. ABLATION A: Decoupled Track with Pretrained 2-channel IR
    res_ablation_track = train_downstream_model(
        exp_name="Pretrained Track (TIR1+2)",
        mode="arch_a_decoupled_track",
        cache_data=cache_data,
        device=device,
        pretrained_2ch_path=ckpt_2ch,
        pretrained_3ch_path=None,
        ckpt_out="data/interim/ml/checkpoints/multisource_downstream_pretrained_track.pt"
    )
    results["Pretrained Track (TIR1+2)"] = res_ablation_track

    # 4. ABLATION B: Decoupled Intensity with Pretrained 3-channel
    res_ablation_int = train_downstream_model(
        exp_name="Pretrained Intensity (TIR1+2+WV)",
        mode="arch_b_decoupled_intensity",
        cache_data=cache_data,
        device=device,
        pretrained_2ch_path=None,
        pretrained_3ch_path=ckpt_3ch,
        ckpt_out="data/interim/ml/checkpoints/multisource_downstream_pretrained_intensity.pt"
    )
    results["Pretrained Intensity (TIR1+2+WV)"] = res_ablation_int

    # Compile Comparison Table
    m_rand = res_random["test_metrics"]
    m_pre = res_pretrained["test_metrics"]

    comparison_table = [
        {"Metric": "Center Mean DPE (km)", "Random_INSAT": round(m_rand["identification"]["center_mean_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["identification"]["center_mean_dpe_km"], 2), "Delta": round(m_pre["identification"]["center_mean_dpe_km"] - m_rand["identification"]["center_mean_dpe_km"], 2)},
        {"Metric": "Center Median DPE (km)", "Random_INSAT": round(m_rand["identification"]["center_median_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["identification"]["center_median_dpe_km"], 2), "Delta": round(m_pre["identification"]["center_median_dpe_km"] - m_rand["identification"]["center_median_dpe_km"], 2)},
        {"Metric": "Center P90 DPE (km)", "Random_INSAT": round(m_rand["identification"]["center_p90_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["identification"]["center_p90_dpe_km"], 2), "Delta": round(m_pre["identification"]["center_p90_dpe_km"] - m_rand["identification"]["center_p90_dpe_km"], 2)},
        {"Metric": "Track +12h DPE (km)", "Random_INSAT": round(m_rand["track_prediction"]["track_12h_mean_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["track_prediction"]["track_12h_mean_dpe_km"], 2), "Delta": round(m_pre["track_prediction"]["track_12h_mean_dpe_km"] - m_rand["track_prediction"]["track_12h_mean_dpe_km"], 2)},
        {"Metric": "Track +24h DPE (km)", "Random_INSAT": round(m_rand["track_prediction"]["track_24h_mean_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["track_prediction"]["track_24h_mean_dpe_km"], 2), "Delta": round(m_pre["track_prediction"]["track_24h_mean_dpe_km"] - m_rand["track_prediction"]["track_24h_mean_dpe_km"], 2)},
        {"Metric": "Track +48h DPE (km)", "Random_INSAT": round(m_rand["track_prediction"]["track_48h_mean_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["track_prediction"]["track_48h_mean_dpe_km"], 2), "Delta": round(m_pre["track_prediction"]["track_48h_mean_dpe_km"] - m_rand["track_prediction"]["track_48h_mean_dpe_km"], 2)},
        {"Metric": "Track Aggregate DPE (km)", "Random_INSAT": round(m_rand["track_prediction"]["track_aggregate_mean_dpe_km"], 2), "Pretrained_INSAT": round(m_pre["track_prediction"]["track_aggregate_mean_dpe_km"], 2), "Delta": round(m_pre["track_prediction"]["track_aggregate_mean_dpe_km"] - m_rand["track_prediction"]["track_aggregate_mean_dpe_km"], 2)},
        {"Metric": "Wind MAE (kt)", "Random_INSAT": round(m_rand["wind_regression"]["wind_mae_kt"], 2), "Pretrained_INSAT": round(m_pre["wind_regression"]["wind_mae_kt"], 2), "Delta": round(m_pre["wind_regression"]["wind_mae_kt"] - m_rand["wind_regression"]["wind_mae_kt"], 2)},
        {"Metric": "Wind RMSE (kt)", "Random_INSAT": round(m_rand["wind_regression"]["wind_rmse_kt"], 2), "Pretrained_INSAT": round(m_pre["wind_regression"]["wind_rmse_kt"], 2), "Delta": round(m_pre["wind_regression"]["wind_rmse_kt"] - m_rand["wind_regression"]["wind_rmse_kt"], 2)},
        {"Metric": "Wind P90 AE (kt)", "Random_INSAT": round(m_rand["wind_regression"]["wind_p90_ae_kt"], 2), "Pretrained_INSAT": round(m_pre["wind_regression"]["wind_p90_ae_kt"], 2), "Delta": round(m_pre["wind_regression"]["wind_p90_ae_kt"] - m_rand["wind_regression"]["wind_p90_ae_kt"], 2)},
        {"Metric": "Wind Bias (kt)", "Random_INSAT": round(m_rand["wind_regression"]["wind_bias_kt"], 2), "Pretrained_INSAT": round(m_pre["wind_regression"]["wind_bias_kt"], 2), "Delta": round(m_pre["wind_regression"]["wind_bias_kt"] - m_rand["wind_regression"]["wind_bias_kt"], 2)},
        {"Metric": "Wind Pearson r", "Random_INSAT": round(m_rand["wind_regression"]["wind_pearson_r"], 4), "Pretrained_INSAT": round(m_pre["wind_regression"]["wind_pearson_r"], 4), "Delta": round(m_pre["wind_regression"]["wind_pearson_r"] - m_rand["wind_regression"]["wind_pearson_r"], 4)},
        {"Metric": "Classification Accuracy", "Random_INSAT": round(m_rand["classification"]["accuracy"], 4), "Pretrained_INSAT": round(m_pre["classification"]["accuracy"], 4), "Delta": round(m_pre["classification"]["accuracy"] - m_rand["classification"]["accuracy"], 4)},
        {"Metric": "Classification Macro-F1", "Random_INSAT": round(m_rand["classification"]["macro_f1"], 4), "Pretrained_INSAT": round(m_pre["classification"]["macro_f1"], 4), "Delta": round(m_pre["classification"]["macro_f1"] - m_rand["classification"]["macro_f1"], 4)}
    ]

    out_data = {
        "metadata": {
            "title": "VAYU-NET Downstream Evaluation: Random vs Pretrained INSAT",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_samples": 207,
            "val_samples": 252,
            "test_samples": 298
        },
        "comparison_table": comparison_table,
        "experiments": results
    }

    out_file = Path("data/interim/ml/insat_downstream_pretrained_results.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved downstream comparison JSON to {out_file}")


if __name__ == "__main__":
    main()
