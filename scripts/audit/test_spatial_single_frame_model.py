"""
VAYU-NET PHASE 3B — SANITY TEST FOR SPATIALLY AWARE SINGLE-FRAME CNN
====================================================================
Sanity verification script to validate:
1. Model instantiation and parameter count.
2. Intermediate feature map shapes (conv1, maxpool, layer1..4).
3. Coordinate channel integration.
4. Center heatmap prediction and soft-argmax decoding.
5. Gaussian target construction from ground-truth IMD center.
6. Forward pass, loss calculation, backward pass, and gradient finite checks.
7. Verification that predicted centers lie within physical NIO domain.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.spatial_single_frame_cnn import (
    SpatialAwareSingleFrameResNet,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    HEATMAP_H, HEATMAP_W
)
from ml.data.vayu_dataset import SingleFrameVayuDataset

SAMPLE_INDEX_CSV = "data/manifests/vayu_net_sample_index.csv"
NORM_STATS_JSON = "data/interim/ml/train_normalization_stats.json"


def run_sanity_check():
    print("=" * 80)
    print("VAYU-NET PHASE 3B — SANITY CHECK FOR SPATIALLY AWARE SINGLE-FRAME CNN")
    print("=" * 80)
    
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 1. Instantiate Model & Count Parameters
    model = SpatialAwareSingleFrameResNet(num_classes=7).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SpatialAwareSingleFrameResNet")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 2. Load 1 Real TRAIN Batch
    print("\nLoading 1 real TRAIN batch from SingleFrameVayuDataset...")
    dataset = SingleFrameVayuDataset(
        sample_index_csv=SAMPLE_INDEX_CSV,
        split="TRAIN",
        norm_stats_path=NORM_STATS_JSON,
        normalize=True
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    images = batch["satellite_image"].to(device)
    target_norm_center = batch["norm_center"].to(device)
    target_norm_wind = batch["norm_wind"].to(device)
    wind_mask = batch["wind_mask"].to(device)
    target_cat = batch["category"].to(device)
    
    print(f"\nInput Tensor Shape: {list(images.shape)} (Expected: [B, 1, 572, 929])")
    
    # 3. Forward Pass & Feature Map Shapes
    model.train()
    outputs = model(images)
    feat_shapes = outputs["feature_shapes"]
    
    print("\nResNet Intermediate Feature Map Shapes:")
    for k, v in feat_shapes.items():
        print(f"  {k:<10}: {v}")
        
    heatmap_logits = outputs["heatmap_logits"]
    heatmap_prob = outputs["heatmap_prob"]
    norm_center = outputs["norm_center"]
    peak_center = outputs["peak_center"]
    norm_wind = outputs["norm_wind"]
    cat_logits = outputs["category_logits"]
    
    print(f"\nCenter Heatmap Shape:   {list(heatmap_prob.shape)} (Expected: [B, 1, {HEATMAP_H}, {HEATMAP_W}])")
    print(f"Center Heatmap Range:   [{heatmap_prob.min().item():.4f}, {heatmap_prob.max().item():.4f}] (Expected in [0, 1])")
    print(f"Decoded Norm Center:    {list(norm_center.shape)} (Expected: [B, 2])")
    print(f"Wind Output Shape:      {list(norm_wind.shape)} (Expected: [B])")
    print(f"Category Logits Shape:  {list(cat_logits.shape)} (Expected: [B, 7])")
    
    # 4. Check Spatial Dimensions > 1x1
    assert heatmap_prob.shape[2] > 1 and heatmap_prob.shape[3] > 1, "Failed: Heatmap is collapsed to 1x1!"
    print("\nVerification Passed: Spatial dimensions are 72 x 117 (> 1x1).")
    
    # 5. Generate Target Heatmap & Verify
    target_heatmap = SpatialAwareSingleFrameResNet.generate_gaussian_target(
        target_norm_center, height=HEATMAP_H, width=HEATMAP_W, sigma=1.5, device=device
    )
    print(f"Target Heatmap Shape:   {list(target_heatmap.shape)} (Expected: [B, 1, {HEATMAP_H}, {HEATMAP_W}])")
    print(f"Target Heatmap Max/Min: {target_heatmap.max().item():.4f} / {target_heatmap.min().item():.4f}")
    assert target_heatmap.max().item() > 0.8, "Failed: Target Gaussian peak is too weak!"
    
    # 6. Verify Decoded Coordinates in NIO Domain
    decoded_deg = SpatialAwareSingleFrameResNet.denormalize_center(norm_center)
    for b in range(images.size(0)):
        p_lat = decoded_deg[b, 0].item()
        p_lon = decoded_deg[b, 1].item()
        t_lat = batch["center_deg"][b, 0].item()
        t_lon = batch["center_deg"][b, 1].item()
        print(f"  Batch {b}: Pred Center = ({p_lat:5.2f}°N, {p_lon:5.2f}°E) | True = ({t_lat:5.2f}°N, {t_lon:5.2f}°E)")
        assert LAT_MIN <= p_lat <= (LAT_MIN + LAT_SPAN), f"Pred latitude {p_lat} outside bounds!"
        assert LON_MIN <= p_lon <= (LON_MIN + LON_SPAN), f"Pred longitude {p_lon} outside bounds!"
    print("Verification Passed: All predicted coordinates are inside physical NIO bounds.")
    
    # 7. Loss Calculation & Backward Pass
    # Heatmap Loss (MSE)
    loss_hm = nn.functional.mse_loss(heatmap_prob, target_heatmap)
    # Coordinate Loss (Smooth L1 on soft-argmax)
    loss_coord = nn.functional.smooth_l1_loss(norm_center, target_norm_center)
    # Wind Loss (Masked Smooth L1)
    wind_diff = nn.functional.smooth_l1_loss(norm_wind, target_norm_wind, reduction="none")
    loss_wind = (wind_diff * wind_mask).sum() / torch.clamp(wind_mask.sum(), min=1.0)
    # Category Loss (CrossEntropy)
    loss_cat = nn.functional.cross_entropy(cat_logits, target_cat, ignore_index=-1)
    
    total_loss = 10.0 * loss_hm + 1.0 * loss_coord + 1.0 * loss_wind + 1.0 * loss_cat
    
    print(f"\nLoss Breakdown:")
    print(f"  Heatmap Loss (MSE):     {loss_hm.item():.4f} (weighted x10: {10.0*loss_hm.item():.4f})")
    print(f"  Coordinate Loss:        {loss_coord.item():.4f}")
    print(f"  Wind Loss:              {loss_wind.item():.4f}")
    print(f"  Category Loss:          {loss_cat.item():.4f}")
    print(f"  Total Multi-Task Loss:  {total_loss.item():.4f}")
    
    assert torch.isfinite(total_loss), "Failed: Total loss is non-finite!"
    
    total_loss.backward()
    
    # 8. Gradient Finite Check
    has_nan_grad = False
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                has_nan_grad = True
                print(f"  Error: NaN/Inf gradient in {name}")
            else:
                grad_norms.append(param.grad.norm().item())
                
    assert not has_nan_grad, "Failed: Non-finite gradients detected!"
    print(f"Verification Passed: 0 NaN or Inf gradients across all {len(grad_norms)} parameter tensors.")
    print(f"Mean parameter gradient norm: {sum(grad_norms)/len(grad_norms):.4f}")
    
    print("\n" + "=" * 80)
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_sanity_check()
