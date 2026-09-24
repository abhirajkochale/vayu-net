"""
VAYU-NET — Single-Frame CNN Baseline Training & Evaluation Pipeline
====================================================================
Architecture: 1-channel ResNet-18 (from-scratch, weights=None)
Input: strictly frame_t0 GridSat IR observation (shape: [B, 1, 572, 929])
Prediction heads:
  1. Current track center (normalized lat/lon in [0, 1]^2)
  2. Current maximum sustained wind (normalized continuous regression in kt)
  3. Current cyclonic category (7-class classification)

Training Discipline:
  - TRAIN split (696 samples / 81 storms) for model optimization
  - VALIDATION split (252 samples / 14 storms) for model selection & early stopping
  - TEST split (371 samples / 31 storms) evaluated strictly ONCE after training
"""

import os
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import math
import random
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from ml.data.vayu_dataset import SingleFrameVayuDataset, CATEGORY_TO_IDX, IDX_TO_CATEGORY
from ml.models.single_frame_cnn import SingleFrameResNet, LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN

# ---------------------------------------------------------
# Configuration & Hyperparameters
# ---------------------------------------------------------
CONFIG = {
    "random_seed": 42,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 5,
    "patience": 2,
    "loss_weights": {
        "center": 1.0,
        "wind": 1.0,
        "category": 1.0
    },
    "sample_index_path": "data/manifests/vayu_net_sample_index.csv",
    "norm_stats_path": "data/interim/ml/train_normalization_stats.json",
    "checkpoint_dir": "data/interim/ml/checkpoints",
    "checkpoint_path": "data/interim/ml/checkpoints/best_single_frame_cnn.pt",
    "results_json_path": "data/interim/ml/single_frame_cnn_results.json",
    "figures_dir": "docs/figures/single_frame_cnn",
    "num_workers": 0
}

EARTH_RADIUS_KM = 6371.0


def set_seed(seed):
    """Sets random seeds for complete reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two geographic coordinates in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def compute_multi_task_loss(preds, batch, smooth_l1_fn, ce_fn, weights):
    """
    Computes balanced multi-task loss:
      - Center: Smooth L1 on normalized coordinates [0, 1]^2
      - Wind: Masked Smooth L1 on normalized wind targets (TRAIN standardized)
      - Category: CrossEntropyLoss on 7-class logits (ignoring index -1)
    """
    # 1. Center loss
    pred_center = preds["norm_center"]          # [B, 2]
    target_center = batch["norm_center"]        # [B, 2]
    loss_center = smooth_l1_fn(pred_center, target_center)
    
    # 2. Wind loss (scaled)
    pred_wind = preds["norm_wind"]              # [B]
    target_wind = batch["norm_wind"]            # [B]
    wind_mask = batch["wind_mask"]              # [B]
    
    wind_diff = smooth_l1_fn(pred_wind, target_wind)  # element-wise or reduced
    if smooth_l1_fn.reduction == "none":
        valid_count = torch.clamp(wind_mask.sum(), min=1.0)
        loss_wind = (wind_diff * wind_mask).sum() / valid_count
    else:
        loss_wind = wind_diff
        
    # 3. Category loss
    pred_cat = preds["category_logits"]         # [B, 7]
    target_cat = batch["category"]              # [B]
    loss_cat = ce_fn(pred_cat, target_cat)
    
    total_loss = (weights["center"] * loss_center +
                  weights["wind"] * loss_wind +
                  weights["category"] * loss_cat)
                  
    return total_loss, loss_center, loss_wind, loss_cat


def evaluate_split(model, dataloader, device, smooth_l1_none, ce_fn, weights, norm_stats):
    """Evaluates model performance and metrics across a dataset split."""
    model.eval()
    
    total_loss_sum = 0.0
    center_loss_sum = 0.0
    wind_loss_sum = 0.0
    cat_loss_sum = 0.0
    num_samples = 0
    
    pred_lats, pred_lons = [], []
    true_lats, true_lons = [], []
    dpes = []
    
    pred_winds_kt, true_winds_kt = [], []
    
    pred_cats, true_cats = [], []
    
    train_wind_mean = norm_stats["train_wind_mean_kt"]
    train_wind_std = norm_stats["train_wind_std_kt"]
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["satellite_image"].to(device)
            target_norm_center = batch["norm_center"].to(device)
            target_norm_wind = batch["norm_wind"].to(device)
            wind_mask = batch["wind_mask"].to(device)
            target_cat = batch["category"].to(device)
            
            b_size = images.size(0)
            num_samples += b_size
            
            preds = model(images)
            
            # Loss computation
            loss_c = nn.functional.smooth_l1_loss(preds["norm_center"], target_norm_center)
            
            w_diff = smooth_l1_none(preds["norm_wind"], target_norm_wind)
            valid_w = torch.clamp(wind_mask.sum(), min=1.0)
            loss_w = (w_diff * wind_mask).sum() / valid_w
            
            loss_k = ce_fn(preds["category_logits"], target_cat)
            
            loss_tot = (weights["center"] * loss_c +
                        weights["wind"] * loss_w +
                        weights["category"] * loss_k)
                        
            total_loss_sum += loss_tot.item() * b_size
            center_loss_sum += loss_c.item() * b_size
            wind_loss_sum += loss_w.item() * b_size
            cat_loss_sum += loss_k.item() * b_size
            
            # Geographic Denormalization
            pred_deg = SingleFrameResNet.denormalize_center(preds["norm_center"].cpu())
            p_lat = pred_deg[:, 0].numpy()
            p_lon = pred_deg[:, 1].numpy()
            
            t_lat = batch["center_deg"][:, 0].numpy()
            t_lon = batch["center_deg"][:, 1].numpy()
            
            for pl, po, tl, to in zip(p_lat, p_lon, t_lat, t_lon):
                pred_lats.append(float(pl))
                pred_lons.append(float(po))
                true_lats.append(float(tl))
                true_lons.append(float(to))
                dpes.append(haversine_km(tl, to, pl, po))
                
            # Wind Denormalization
            p_w_kt = (preds["norm_wind"].cpu().numpy() * train_wind_std) + train_wind_mean
            t_w_kt = batch["wind_kt"].numpy()
            w_m = batch["wind_mask"].numpy()
            
            for pw, tw, m in zip(p_w_kt, t_w_kt, w_m):
                if m > 0.5:
                    pred_winds_kt.append(float(pw))
                    true_winds_kt.append(float(tw))
                    
            # Category prediction
            p_cat = torch.argmax(preds["category_logits"].cpu(), dim=-1).numpy()
            t_cat = batch["category"].numpy()
            for pc, tc in zip(p_cat, t_cat):
                if tc >= 0:
                    pred_cats.append(int(pc))
                    true_cats.append(int(tc))
                    
    # Aggregations
    avg_tot_loss = total_loss_sum / num_samples
    avg_c_loss = center_loss_sum / num_samples
    avg_w_loss = wind_loss_sum / num_samples
    avg_k_loss = cat_loss_sum / num_samples
    
    # Geographic metrics
    lat_mae = float(np.mean(np.abs(np.array(pred_lats) - np.array(true_lats))))
    lon_mae = float(np.mean(np.abs(np.array(pred_lons) - np.array(true_lons))))
    mean_dpe = float(np.mean(dpes))
    median_dpe = float(np.median(dpes))
    
    # Wind metrics
    if len(pred_winds_kt) > 0:
        wind_diffs = np.array(pred_winds_kt) - np.array(true_winds_kt)
        wind_mae = float(np.mean(np.abs(wind_diffs)))
        wind_rmse = float(np.sqrt(np.mean(wind_diffs**2)))
    else:
        wind_mae, wind_rmse = 0.0, 0.0
        
    # Category metrics
    if len(true_cats) > 0:
        cat_acc = float(accuracy_score(true_cats, pred_cats))
        cat_f1 = float(f1_score(true_cats, pred_cats, average="macro", zero_division=0))
        cm = confusion_matrix(true_cats, pred_cats, labels=list(range(7))).tolist()
    else:
        cat_acc, cat_f1, cm = 0.0, 0.0, []
        
    metrics = {
        "loss_total": avg_tot_loss,
        "loss_center": avg_c_loss,
        "loss_wind": avg_w_loss,
        "loss_category": avg_k_loss,
        "center_lat_mae_deg": lat_mae,
        "center_lon_mae_deg": lon_mae,
        "center_mean_dpe_km": mean_dpe,
        "center_median_dpe_km": median_dpe,
        "wind_mae_kt": wind_mae,
        "wind_rmse_kt": wind_rmse,
        "category_accuracy": cat_acc,
        "category_macro_f1": cat_f1,
        "confusion_matrix": cm,
        "raw_predictions": {
            "pred_lats": pred_lats,
            "pred_lons": pred_lons,
            "true_lats": true_lats,
            "true_lons": true_lons,
            "dpes_km": dpes,
            "pred_winds_kt": pred_winds_kt,
            "true_winds_kt": true_winds_kt,
            "pred_cats": pred_cats,
            "true_cats": true_cats
        }
    }
    return metrics


def plot_diagnostics(history, test_metrics, save_dir):
    """Generates all 4 requested diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Train / Val Loss History
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = [h["epoch"] for h in history]
    
    # Total loss
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], 'o-', label='Train Total Loss', color='#1f77b4', lw=2)
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], 's--', label='Val Total Loss', color='#ff7f0e', lw=2)
    axes[0, 0].set_title("Total Multi-Task Loss", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()
    
    # Center loss
    axes[0, 1].plot(epochs, [h["train_loss_center"] for h in history], 'o-', label='Train Center Loss', color='#2ca02c', lw=2)
    axes[0, 1].plot(epochs, [h["val_loss_center"] for h in history], 's--', label='Val Center Loss', color='#d62728', lw=2)
    axes[0, 1].set_title("Center Position Loss (Smooth L1)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()
    
    # Wind loss
    axes[1, 0].plot(epochs, [h["train_loss_wind"] for h in history], 'o-', label='Train Wind Loss', color='#9467bd', lw=2)
    axes[1, 0].plot(epochs, [h["val_loss_wind"] for h in history], 's--', label='Val Wind Loss', color='#8c564b', lw=2)
    axes[1, 0].set_title("Wind Regression Loss (Normalized Smooth L1)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()
    
    # Category loss
    axes[1, 1].plot(epochs, [h["train_loss_category"] for h in history], 'o-', label='Train Category Loss', color='#e377c2', lw=2)
    axes[1, 1].plot(epochs, [h["val_loss_category"] for h in history], 's--', label='Val Category Loss', color='#7f7f7f', lw=2)
    axes[1, 1].set_title("Category Classification Loss (Cross Entropy)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()
    
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "train_val_loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved: {loss_path}")
    
    # 2. Center Error Distribution (TEST set)
    dpes = test_metrics["raw_predictions"]["dpes_km"]
    mean_dpe = test_metrics["center_mean_dpe_km"]
    median_dpe = test_metrics["center_median_dpe_km"]
    
    plt.figure(figsize=(9, 6))
    n, bins, patches = plt.hist(dpes, bins=35, color='#3b528b', edgecolor='black', alpha=0.75, density=False)
    plt.axvline(mean_dpe, color='#d62728', linestyle='--', linewidth=2.5, label=f'Mean DPE: {mean_dpe:.1f} km')
    plt.axvline(median_dpe, color='#2ca02c', linestyle='-', linewidth=2.5, label=f'Median DPE: {median_dpe:.1f} km')
    plt.title("Direct Position Error (DPE) Distribution — TEST Split (N=371)", fontsize=14, fontweight='bold')
    plt.xlabel("DPE (km)", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    hist_path = os.path.join(save_dir, "center_error_distribution.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_path}")
    
    # 3. Center Geographic Scatter (TEST set)
    true_lats = test_metrics["raw_predictions"]["true_lats"]
    true_lons = test_metrics["raw_predictions"]["true_lons"]
    pred_lats = test_metrics["raw_predictions"]["pred_lats"]
    pred_lons = test_metrics["raw_predictions"]["pred_lons"]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(true_lons, true_lats, color='#1f77b4', alpha=0.65, edgecolors='none', s=45, label='Ground Truth Center (IMD)')
    plt.scatter(pred_lons, pred_lats, color='#ff7f0e', alpha=0.65, edgecolors='none', s=45, marker='^', label='Single-Frame CNN Prediction')
    
    # Connect a subset of samples with error vectors for visual clarity
    for tl, to, pl, po in zip(true_lats[::5], true_lons[::5], pred_lats[::5], pred_lons[::5]):
        plt.plot([to, po], [tl, pl], color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
    plt.xlim(40.0, 105.0)
    plt.ylim(-5.0, 35.0)
    plt.xlabel("Longitude (°E)", fontsize=12)
    plt.ylabel("Latitude (°N)", fontsize=12)
    plt.title("Cyclone Center Coordinates: Ground Truth vs Prediction (TEST)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, "center_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")
    
    # 4. Category Confusion Matrix (TEST set)
    cm = np.array(test_metrics["confusion_matrix"])
    class_labels = [IDX_TO_CATEGORY[i] for i in range(7)]
    
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("IMD Category Confusion Matrix — TEST Split", fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # Annotate numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, 'd'),
                     ha="center", va="center", color=color, fontweight='bold')
                     
    plt.ylabel('True IMD Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "category_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")


def main():
    print("=" * 80)
    print("VAYU-NET PHASE 3 — SINGLE-FRAME CNN BASELINE (ResNet-18 From-Scratch)")
    print("=" * 80)
    
    # 1. Reproducibility & Environment
    set_seed(CONFIG["random_seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python Version:       {platform.python_version()}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"Torchvision Version:  {torchvision.__version__}")
    print(f"Execution Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU with ' + str(os.cpu_count()) + ' cores'})")
    print(f"Random Seed:          {CONFIG['random_seed']}")
    
    # 2. Report TRAIN Category Distribution
    df_sample = pd.read_csv(CONFIG["sample_index_path"])
    df_train = df_sample[df_sample["split"] == "TRAIN"]
    df_val = df_sample[df_sample["split"] == "VALIDATION"]
    df_test = df_sample[df_sample["split"] == "TEST"]
    
    print("\n" + "-" * 60)
    print("DATASET SPLIT COUNTS & TRAIN CATEGORY DISTRIBUTION")
    print("-" * 60)
    print(f"TRAIN:      {len(df_train)} samples across {df_train['storm_id'].nunique()} storms")
    print(f"VALIDATION: {len(df_val)} samples across {df_val['storm_id'].nunique()} storms")
    print(f"TEST:       {len(df_test)} samples across {df_test['storm_id'].nunique()} storms")
    
    train_cat_counts = df_train["imd_category_t0"].value_counts(dropna=False).to_dict()
    print("\nTRAIN Category Distribution (Class Imbalance Baseline):")
    for cat, cnt in sorted(train_cat_counts.items(), key=lambda x: str(x[0])):
        pct = (cnt / len(df_train)) * 100.0
        print(f"  Category {str(cat):<8}: {cnt:>4} samples ({pct:5.2f}%)")
        
    # 3. Model Instantiation & Parameter Count
    model = SingleFrameResNet(num_classes=7).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SingleFrameResNet (1-Channel ResNet-18, weights=None)")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 4. Datasets and Loaders
    train_dataset = SingleFrameVayuDataset(
        sample_index_csv=CONFIG["sample_index_path"],
        split="TRAIN",
        norm_stats_path=CONFIG["norm_stats_path"],
        normalize=True
    )
    val_dataset = SingleFrameVayuDataset(
        sample_index_csv=CONFIG["sample_index_path"],
        split="VALIDATION",
        norm_stats_path=CONFIG["norm_stats_path"],
        normalize=True
    )
    test_dataset = SingleFrameVayuDataset(
        sample_index_csv=CONFIG["sample_index_path"],
        split="TEST",
        norm_stats_path=CONFIG["norm_stats_path"],
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=False
    )
    
    with open(CONFIG["norm_stats_path"], "r") as f:
        norm_stats = json.load(f)
        
    print(f"\nNormalization Statistics (TRAIN-only):")
    print(f"  IR Mean: {norm_stats['mean_kelvin']:.4f} K, Std: {norm_stats['std_kelvin']:.4f} K")
    print(f"  Wind Mean: {norm_stats['train_wind_mean_kt']:.4f} kt, Std: {norm_stats['train_wind_std_kt']:.4f} kt")
    
    # 5. Sanity Check on 1 real TRAIN batch
    print("\n" + "-" * 60)
    print("SANITY CHECK — 1 REAL TRAIN BATCH")
    print("-" * 60)
    model.train()
    sample_batch = next(iter(train_loader))
    s_img = sample_batch["satellite_image"].to(device)
    
    s_out = model(s_img)
    s_loss_c = nn.functional.smooth_l1_loss(s_out["norm_center"], sample_batch["norm_center"].to(device))
    s_diff_w = nn.functional.smooth_l1_loss(s_out["norm_wind"], sample_batch["norm_wind"].to(device), reduction="none")
    s_mask_w = sample_batch["wind_mask"].to(device)
    s_loss_w = (s_diff_w * s_mask_w).sum() / torch.clamp(s_mask_w.sum(), min=1.0)
    s_loss_k = nn.functional.cross_entropy(s_out["category_logits"], sample_batch["category"].to(device), ignore_index=-1)
    s_loss_tot = s_loss_c + s_loss_w + s_loss_k
    
    s_loss_tot.backward()
    
    # Verify finite losses and gradients
    assert torch.isfinite(s_loss_tot), "Sanity check failed: Total loss is non-finite!"
    assert torch.isfinite(s_loss_c), "Sanity check failed: Center loss is non-finite!"
    assert torch.isfinite(s_loss_w), "Sanity check failed: Wind loss is non-finite!"
    assert torch.isfinite(s_loss_k), "Sanity check failed: Category loss is non-finite!"
    
    nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                nan_grad = True
                print(f"  NaN/Inf gradient in parameter: {name}")
    assert not nan_grad, "Sanity check failed: Non-finite gradients detected!"
    
    c_min = s_out["norm_center"].min().item()
    c_max = s_out["norm_center"].max().item()
    print(f"  Sanity check passed!")
    print(f"  Center output range: [{c_min:.4f}, {c_max:.4f}] (Expected [0, 1])")
    print(f"  Wind output shape:   {s_out['norm_wind'].shape}")
    print(f"  Cat output shape:    {s_out['category_logits'].shape}")
    print(f"  Losses: Center={s_loss_c.item():.4f}, Wind={s_loss_w.item():.4f}, Cat={s_loss_k.item():.4f}, Total={s_loss_tot.item():.4f}")
    
    # Reset model parameters after sanity check test to preserve seed state
    set_seed(CONFIG["random_seed"])
    model = SingleFrameResNet(num_classes=7).to(device)
    
    # 6. Optimizer, Scheduler, Loss functions
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    smooth_l1_none = nn.SmoothL1Loss(reduction="none")
    ce_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)
    
    # 7. Training Loop
    print("\n" + "=" * 80)
    print("BEGINNING TRAINING (Max Epochs: 5, Early Stopping Patience: 2)")
    print("=" * 80)
    
    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss_total = 0.0
        train_loss_c = 0.0
        train_loss_w = 0.0
        train_loss_k = 0.0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = batch["satellite_image"].to(device)
            target_norm_center = batch["norm_center"].to(device)
            target_norm_wind = batch["norm_wind"].to(device)
            wind_mask = batch["wind_mask"].to(device)
            target_cat = batch["category"].to(device)
            
            b_size = images.size(0)
            train_samples += b_size
            
            preds = model(images)
            
            loss_c = nn.functional.smooth_l1_loss(preds["norm_center"], target_norm_center)
            
            w_diff = smooth_l1_none(preds["norm_wind"], target_norm_wind)
            valid_w = torch.clamp(wind_mask.sum(), min=1.0)
            loss_w = (w_diff * wind_mask).sum() / valid_w
            
            loss_k = ce_fn(preds["category_logits"], target_cat)
            
            loss = (CONFIG["loss_weights"]["center"] * loss_c +
                    CONFIG["loss_weights"]["wind"] * loss_w +
                    CONFIG["loss_weights"]["category"] * loss_k)
                    
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item() * b_size
            train_loss_c += loss_c.item() * b_size
            train_loss_w += loss_w.item() * b_size
            train_loss_k += loss_k.item() * b_size
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch:2d}/{CONFIG['epochs']:2d} | Batch {batch_idx+1:2d}/{len(train_loader):2d} | "
                      f"Running Train Loss: {train_loss_total / train_samples:.4f} "
                      f"(C: {train_loss_c / train_samples:.4f}, W: {train_loss_w / train_samples:.4f}, Cat: {train_loss_k / train_samples:.4f})")
                      
        avg_train_loss = train_loss_total / train_samples
        avg_train_c = train_loss_c / train_samples
        avg_train_w = train_loss_w / train_samples
        avg_train_k = train_loss_k / train_samples
        
        # Validation Evaluation
        val_metrics = evaluate_split(
            model, val_loader, device, smooth_l1_none, ce_fn, CONFIG["loss_weights"], norm_stats
        )
        avg_val_loss = val_metrics["loss_total"]
        avg_val_c = val_metrics["loss_center"]
        avg_val_w = val_metrics["loss_wind"]
        avg_val_k = val_metrics["loss_category"]
        
        # Scheduler Step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        
        print("-" * 80)
        print(f"Epoch {epoch:2d} Summary:")
        print(f"  TRAIN Loss: {avg_train_loss:.4f} (Center: {avg_train_c:.4f}, Wind: {avg_train_w:.4f}, Cat: {avg_train_k:.4f})")
        print(f"  VAL   Loss: {avg_val_loss:.4f} (Center: {avg_val_c:.4f}, Wind: {avg_val_w:.4f}, Cat: {avg_val_k:.4f})")
        print(f"  VAL Metrics: DPE Mean: {val_metrics['center_mean_dpe_km']:.1f} km, Wind MAE: {val_metrics['wind_mae_kt']:.2f} kt, Cat Acc: {val_metrics['category_accuracy']*100:.2f}%, LR: {current_lr:.6f}")
        print("-" * 80)
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_loss_center": avg_train_c,
            "train_loss_wind": avg_train_w,
            "train_loss_category": avg_train_k,
            "val_loss": avg_val_loss,
            "val_loss_center": avg_val_c,
            "val_loss_wind": avg_val_w,
            "val_loss_category": avg_val_k,
            "val_mean_dpe_km": val_metrics["center_mean_dpe_km"],
            "val_wind_mae_kt": val_metrics["wind_mae_kt"],
            "val_category_acc": val_metrics["category_accuracy"],
            "lr": current_lr
        }
        history.append(epoch_record)
        
        # Model Selection (VALIDATION loss only)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": CONFIG
            }, CONFIG["checkpoint_path"])
            print(f"  >>> Best model saved at epoch {epoch} with Val Loss: {avg_val_loss:.4f} to {CONFIG['checkpoint_path']}")
        else:
            epochs_no_improve += 1
            print(f"  >>> Val loss did not improve ({avg_val_loss:.4f} >= {best_val_loss:.4f}). Early stopping counter: {epochs_no_improve}/{CONFIG['patience']}")
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation loss did not improve for {CONFIG['patience']} consecutive epochs.")
                break
                
    # 8. Load Best Checkpoint for Final Unbiased Evaluation
    print("\n" + "=" * 80)
    print(f"EVALUATING BEST CHECKPOINT (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
    print("=" * 80)
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate TRAIN (for diagnostics)
    train_final = evaluate_split(
        model, train_loader, device, smooth_l1_none, ce_fn, CONFIG["loss_weights"], norm_stats
    )
    # Evaluate VALIDATION (for diagnostics)
    val_final = evaluate_split(
        model, val_loader, device, smooth_l1_none, ce_fn, CONFIG["loss_weights"], norm_stats
    )
    # Evaluate TEST (STRICTLY ONCE, final unbiased result)
    test_final = evaluate_split(
        model, test_loader, device, smooth_l1_none, ce_fn, CONFIG["loss_weights"], norm_stats
    )
    
    print("\nFINAL EVALUATION SUMMARY:")
    print("-" * 80)
    print(f"{'Split':<12} | {'Center DPE (Mean / Med)':<25} | {'Wind MAE / RMSE':<20} | {'Cat Acc / Macro F1':<20}")
    print("-" * 80)
    print(f"{'TRAIN':<12} | {train_final['center_mean_dpe_km']:6.1f} km / {train_final['center_median_dpe_km']:6.1f} km | "
          f"{train_final['wind_mae_kt']:5.2f} kt / {train_final['wind_rmse_kt']:5.2f} kt | "
          f"{train_final['category_accuracy']*100:5.2f}% / {train_final['category_macro_f1']:.4f}")
          
    print(f"{'VALIDATION':<12} | {val_final['center_mean_dpe_km']:6.1f} km / {val_final['center_median_dpe_km']:6.1f} km | "
          f"{val_final['wind_mae_kt']:5.2f} kt / {val_final['wind_rmse_kt']:5.2f} kt | "
          f"{val_final['category_accuracy']*100:5.2f}% / {val_final['category_macro_f1']:.4f}")
          
    print(f"{'TEST':<12} | {test_final['center_mean_dpe_km']:6.1f} km / {test_final['center_median_dpe_km']:6.1f} km | "
          f"{test_final['wind_mae_kt']:5.2f} kt / {test_final['wind_rmse_kt']:5.2f} kt | "
          f"{test_final['category_accuracy']*100:5.2f}% / {test_final['category_macro_f1']:.4f}")
    print("-" * 80)
    
    # 9. Plot Diagnostics
    plot_diagnostics(history, test_final, CONFIG["figures_dir"])
    
    # 10. Save JSON Results
    results = {
        "metadata": {
            "model_architecture": "SingleFrameResNet (1-channel ResNet-18, weights=None)",
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(device),
            "random_seed": CONFIG["random_seed"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "weight_decay": CONFIG["weight_decay"],
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "patience": CONFIG["patience"],
            "loss_weights": CONFIG["loss_weights"],
            "loss_formulation": {
                "center": "SmoothL1Loss(pred_norm_center, true_norm_center)",
                "wind": "MaskedSmoothL1Loss(pred_norm_wind, true_norm_wind) / sum(mask)",
                "category": "CrossEntropyLoss(pred_cat_logits, true_cat, ignore_index=-1)",
                "total": "1.0 * center + 1.0 * wind + 1.0 * category"
            },
            "wind_normalization_stats": {
                "mean_kt": norm_stats["train_wind_mean_kt"],
                "std_kt": norm_stats["train_wind_std_kt"]
            },
            "train_category_distribution": train_cat_counts,
            "evaluation_rule": "TEST set evaluated exactly once using best validation checkpoint"
        },
        "training_history": history,
        "metrics": {
            "train": {
                "center_lat_mae_deg": train_final["center_lat_mae_deg"],
                "center_lon_mae_deg": train_final["center_lon_mae_deg"],
                "center_mean_dpe_km": train_final["center_mean_dpe_km"],
                "center_median_dpe_km": train_final["center_median_dpe_km"],
                "wind_mae_kt": train_final["wind_mae_kt"],
                "wind_rmse_kt": train_final["wind_rmse_kt"],
                "category_accuracy": train_final["category_accuracy"],
                "category_macro_f1": train_final["category_macro_f1"],
                "confusion_matrix": train_final["confusion_matrix"]
            },
            "validation": {
                "center_lat_mae_deg": val_final["center_lat_mae_deg"],
                "center_lon_mae_deg": val_final["center_lon_mae_deg"],
                "center_mean_dpe_km": val_final["center_mean_dpe_km"],
                "center_median_dpe_km": val_final["center_median_dpe_km"],
                "wind_mae_kt": val_final["wind_mae_kt"],
                "wind_rmse_kt": val_final["wind_rmse_kt"],
                "category_accuracy": val_final["category_accuracy"],
                "category_macro_f1": val_final["category_macro_f1"],
                "confusion_matrix": val_final["confusion_matrix"]
            },
            "test": {
                "center_lat_mae_deg": test_final["center_lat_mae_deg"],
                "center_lon_mae_deg": test_final["center_lon_mae_deg"],
                "center_mean_dpe_km": test_final["center_mean_dpe_km"],
                "center_median_dpe_km": test_final["center_median_dpe_km"],
                "wind_mae_kt": test_final["wind_mae_kt"],
                "wind_rmse_kt": test_final["wind_rmse_kt"],
                "category_accuracy": test_final["category_accuracy"],
                "category_macro_f1": test_final["category_macro_f1"],
                "confusion_matrix": test_final["confusion_matrix"]
            }
        }
    }
    
    with open(CONFIG["results_json_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON to: {CONFIG['results_json_path']}")
    print("=" * 80)
    print("PHASE 3 SINGLE-FRAME CNN BASELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
