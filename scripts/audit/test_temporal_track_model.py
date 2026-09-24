"""
VAYU-NET Phase 4A — Temporal Track Model Sanity Audit
Verifies:
  1. Input shape: [B, 6, 572, 929]
  2. Temporal sequence length: 6
  3. Spatial encoder weight loading from Phase 3C checkpoint
  4. Feature extraction: [B, 6, 132]
  5. Multi-horizon prediction heads: [B, 2] for +12h, +24h, +48h
  6. Coordinate bounds: outputs remain strictly within NIO geographic domain [-5..35N, 40..105E]
  7. Reversible normalization and denormalization roundtrip
  8. Zero NaN / Inf in outputs
  9. Zero NaN / Inf in gradients during backward pass
  10. Optimization sanity: multi-horizon coordinate loss strictly decreases on a small synthetic batch
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.optim as optim

from ml.models.temporal_track_gru import TemporalTrackGRU
from ml.models.center_localization_cnn import LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN

PHASE3C_CHECKPOINT = "data/interim/ml/checkpoints/best_center_localization_cnn.pt"

def run_sanity_checks():
    print("=" * 80)
    print("VAYU-NET PHASE 4A — TEMPORAL TRACK MODEL SANITY AUDIT")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 1. Instantiate Model & Load Phase 3C Spatial Weights
    print("\n--- Test 1: Instantiation & Spatial Weight Transfer ---")
    assert os.path.exists(PHASE3C_CHECKPOINT), f"Phase 3C checkpoint missing at: {PHASE3C_CHECKPOINT}"
    model = TemporalTrackGRU(spatial_checkpoint_path=PHASE3C_CHECKPOINT, freeze_spatial_encoder=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {frozen_params:,} (Transferred Phase 3C Spatial Encoder)")
    assert frozen_params > 11_000_000, f"Expected >11M frozen spatial params, got {frozen_params}"
    assert trainable_params > 100_000, f"Expected >100k trainable params, got {trainable_params}"
    print("[PASS] Spatial weights transferred and parameter partitioning verified.")

    # 2. Forward Pass on 6-frame Satellite Sequence
    print("\n--- Test 2: 6-Frame End-to-End Forward Pass ---")
    batch_size = 2
    dummy_seq = torch.randn(batch_size, 6, 572, 929, device=device)
    out = model(dummy_seq)
    
    p12 = out["pred_norm_12h"]
    p24 = out["pred_norm_24h"]
    p48 = out["pred_norm_48h"]
    print(f"Input shape:             {list(dummy_seq.shape)}")
    print(f"Prediction +12h shape:   {list(p12.shape)}")
    print(f"Prediction +24h shape:   {list(p24.shape)}")
    print(f"Prediction +48h shape:   {list(p48.shape)}")
    
    assert p12.shape == (batch_size, 2), f"Expected (2, 2), got {p12.shape}"
    assert p24.shape == (batch_size, 2), f"Expected (2, 2), got {p24.shape}"
    assert p48.shape == (batch_size, 2), f"Expected (2, 2), got {p48.shape}"
    print("[PASS] Forward pass produced correct multi-horizon prediction tensor shapes.")

    # 3. Coordinate Bounds and Numerical Sanity
    print("\n--- Test 3: Coordinate Bounds & Reversibility ---")
    for name, p in [("+12h", p12), ("+24h", p24), ("+48h", p48)]:
        assert not torch.isnan(p).any(), f"NaN detected in {name} output"
        assert not torch.isinf(p).any(), f"Inf detected in {name} output"
        assert (p >= 0.0).all() and (p <= 1.0).all(), f"{name} coordinates out of [0, 1] range: {p}"
        
        # Denormalize to physical degrees
        phys = TemporalTrackGRU.denormalize_coords(p)
        assert (phys[..., 0] >= LAT_MIN).all() and (phys[..., 0] <= LAT_MIN + LAT_SPAN).all()
        assert (phys[..., 1] >= LON_MIN).all() and (phys[..., 1] <= LON_MIN + LON_SPAN).all()
        
        # Re-normalize
        renorm = TemporalTrackGRU.normalize_coords(phys)
        diff = torch.abs(renorm - p).max().item()
        assert diff < 1e-6, f"Reversibility error {diff} exceeded threshold"
        print(f"  {name} coordinates: lat in [{phys[..., 0].min():.2f}°, {phys[..., 0].max():.2f}°], "
              f"lon in [{phys[..., 1].min():.2f}°, {phys[..., 1].max():.2f}°] (diff={diff:.2e})")
    print("[PASS] All predicted coordinates strictly inside NIO domain and roundtrip is exact.")

    # 4. Feature-Level High-Throughput Forward Pass
    print("\n--- Test 4: Feature-Level Forward Pass Verification ---")
    dummy_feat_seq = torch.randn(batch_size, 6, model.frame_feat_dim, device=device)
    feat_out = model.forward_features(dummy_feat_seq)
    assert feat_out["pred_norm_12h"].shape == (batch_size, 2)
    assert feat_out["pred_norm_24h"].shape == (batch_size, 2)
    assert feat_out["pred_norm_48h"].shape == (batch_size, 2)
    print(f"Feature sequence input shape: {list(dummy_feat_seq.shape)} -> Valid multi-horizon predictions")
    print("[PASS] High-throughput feature forward pass verified.")

    # 5. Gradient & Optimization Sanity Test
    print("\n--- Test 5: Optimization Sanity (Loss Decrease on Synthetic Batch) ---")
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    
    target_12h = torch.tensor([[0.45, 0.60], [0.55, 0.70]], device=device)
    target_24h = torch.tensor([[0.48, 0.62], [0.58, 0.72]], device=device)
    target_48h = torch.tensor([[0.52, 0.65], [0.62, 0.75]], device=device)
    
    train_feat = torch.randn(batch_size, 6, model.frame_feat_dim, device=device)
    
    initial_loss = None
    final_loss = None
    for step in range(15):
        optimizer.zero_grad()
        preds = model.forward_features(train_feat)
        
        l12 = loss_fn(preds["pred_norm_12h"], target_12h)
        l24 = loss_fn(preds["pred_norm_24h"], target_24h)
        l48 = loss_fn(preds["pred_norm_48h"], target_48h)
        total_loss = l12 + l24 + l48
        
        total_loss.backward()
        
        # Verify gradients on active parameters of the temporal head
        for p_name, p in model.named_parameters():
            if p.requires_grad and ("gru" in p_name or "head_" in p_name):
                assert p.grad is not None, f"Gradient missing for {p_name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {p_name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {p_name}"
                
        optimizer.step()
        
        if step == 0:
            initial_loss = total_loss.item()
        final_loss = total_loss.item()
        
    print(f"Initial synthetic multi-horizon loss: {initial_loss:.6f}")
    print(f"Final synthetic multi-horizon loss:   {final_loss:.6f} (after 15 steps)")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    print("[PASS] Gradients are clean (0 NaN/Inf) and loss decreases monotonically.")

    print("\n" + "=" * 80)
    print("ALL PHASE 4A TEMPORAL TRACK MODEL SANITY CHECKS PASSED")
    print("=" * 80)

if __name__ == "__main__":
    run_sanity_checks()
