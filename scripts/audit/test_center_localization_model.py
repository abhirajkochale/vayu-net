"""
VAYU-NET PHASE 3C — SANITY TEST FOR DEDICATED CENTER LOCALIZATION MODEL
=======================================================================
Pre-training forensic sanity verification script:
1. Coordinate ordering & bounds verification.
2. Tensor dimensions (input [B,1,572,929], layer2 [B,128,72,117], heatmap [B,1,72,117], decoded [B,2]).
3. Reversible geographic normalization.
4. Target Gaussian heatmap generation and peak verification.
5. Finite loss check on isolated center objective (10*MSE + 1*SmoothL1).
6. Backward pass and 0 NaN/Inf gradient verification.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.center_localization_cnn import (
    DedicatedCenterLocalizationResNet,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    HEATMAP_H, HEATMAP_W
)
from ml.data.vayu_dataset import SingleFrameVayuDataset

SAMPLE_INDEX_CSV = "data/manifests/vayu_net_sample_index.csv"
NORM_STATS_JSON = "data/interim/ml/train_normalization_stats.json"


def run_sanity_check():
    print("=" * 80)
    print("VAYU-NET PHASE 3C — DEDICATED CENTER LOCALIZATION SANITY CHECK")
    print("=" * 80)
    
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 1. Model Instantiation & Parameter Count
    model = DedicatedCenterLocalizationResNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: DedicatedCenterLocalizationResNet")
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
    
    print(f"Input Shape:          {list(images.shape)} (Expected: [B, 1, 572, 929])")
    assert images.shape[1] == 1 and images.shape[2] == 572 and images.shape[3] == 929, "Unexpected input dimensions!"
    assert torch.isfinite(images).all(), "Non-finite values in input images!"
    
    # 3. Forward Pass & Feature Map Shapes
    model.train()
    outputs = model(images)
    feat_shapes = outputs["feature_shapes"]
    
    print("\nFeature Map Dimensions:")
    for k, v in feat_shapes.items():
        print(f"  {k:<10}: {v}")
    assert feat_shapes["layer2"] == [4, 128, 72, 117], "layer2 shape mismatch!"
    
    heatmap_logits = outputs["heatmap_logits"]
    heatmap_prob = outputs["heatmap_prob"]
    norm_center = outputs["norm_center"]
    peak_center = outputs["peak_center"]
    
    print(f"\nHeatmap Shape:        {list(heatmap_prob.shape)} (Expected: [B, 1, {HEATMAP_H}, {HEATMAP_W}])")
    print(f"Decoded Center Shape: {list(norm_center.shape)} (Expected: [B, 2])")
    print(f"Heatmap Range:        [{heatmap_prob.min().item():.4f}, {heatmap_prob.max().item():.4f}]")
    
    assert heatmap_prob.shape == torch.Size([4, 1, 72, 117]), "Heatmap shape mismatch!"
    assert norm_center.shape == torch.Size([4, 2]), "Decoded center shape mismatch!"
    
    # 4. Target Gaussian Heatmap Construction
    target_heatmap = DedicatedCenterLocalizationResNet.generate_gaussian_target(
        target_norm_center, height=HEATMAP_H, width=HEATMAP_W, sigma=1.5, device=device
    )
    print(f"Target Heatmap Shape: {list(target_heatmap.shape)} (Expected: [B, 1, {HEATMAP_H}, {HEATMAP_W}])")
    print(f"Target Heatmap Peak:  {target_heatmap.max().item():.4f}")
    assert target_heatmap.max().item() > 0.8, "Target Gaussian peak too weak!"
    
    # 5. Coordinate Ordering & Domain Bounds
    decoded_deg = DedicatedCenterLocalizationResNet.denormalize_center(norm_center)
    for b in range(images.size(0)):
        p_lat = decoded_deg[b, 0].item()
        p_lon = decoded_deg[b, 1].item()
        t_lat = batch["center_deg"][b, 0].item()
        t_lon = batch["center_deg"][b, 1].item()
        print(f"  Batch {b}: Pred Center = ({p_lat:5.2f}°N, {p_lon:5.2f}°E) | True = ({t_lat:5.2f}°N, {t_lon:5.2f}°E)")
        assert LAT_MIN <= p_lat <= (LAT_MIN + LAT_SPAN), f"Pred latitude {p_lat} outside NIO domain!"
        assert LON_MIN <= p_lon <= (LON_MIN + LON_SPAN), f"Pred longitude {p_lon} outside NIO domain!"
    print("Verification Passed: All decoded coordinates lie within physical NIO domain.")
    
    # 6. Reversible Normalization Check
    rec_norm = DedicatedCenterLocalizationResNet.normalize_center(decoded_deg)
    diff = torch.max(torch.abs(rec_norm - norm_center)).item()
    print(f"Reversible Normalization Max Error: {diff:.2e}°")
    assert diff < 1e-6, "Normalization not reversible!"
    
    # 7. Dedicated Loss & Backward Pass
    loss_hm = nn.functional.mse_loss(heatmap_prob, target_heatmap)
    loss_coord = nn.functional.smooth_l1_loss(norm_center, target_norm_center)
    loss_total = 10.0 * loss_hm + 1.0 * loss_coord
    
    print(f"\nDedicated Center Loss Breakdown:")
    print(f"  Heatmap Loss (MSE):     {loss_hm.item():.4f} (x10 = {10.0 * loss_hm.item():.4f})")
    print(f"  Coordinate Loss (L1):   {loss_coord.item():.4f}")
    print(f"  Total Loss:             {loss_total.item():.4f}")
    
    assert torch.isfinite(loss_total), "Non-finite loss!"
    loss_total.backward()
    
    # 8. Gradient Finite Check
    has_nan_grad = False
    param_count_checked = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_count_checked += 1
            if not torch.isfinite(param.grad).all():
                has_nan_grad = True
                print(f"  Error: NaN/Inf gradient in {name}")
                
    assert not has_nan_grad, "Non-finite gradients detected!"
    print(f"Verification Passed: 0 NaN or Inf gradients across all {param_count_checked} active parameter tensors.")
    
    print("\n" + "=" * 80)
    print("ALL PHASE 3C SANITY CHECKS PASSED SUCCESSFULLY")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_sanity_check()
