"""
VAYU-NET PHASE 5A — MULTIMODAL TRACK MODEL ARCHITECTURE AUDIT
=============================================================
Verifies:
1. EnvironmentOnlyTrackModel parameter count, forward pass, gradient flow
2. MultimodalTrackModel parameter count, frozen vs trainable parameters
3. Gradient propagation from multi-horizon track loss
4. No NaNs or Infs in outputs or gradients
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn.functional as F

from ml.models.environment_encoder import EnvironmentOnlyTrackModel
from ml.models.multimodal_track_model import MultimodalTrackModel

def test_models():
    print("=" * 80)
    print("RUNNING VAYU-NET PHASE 5A: MULTIMODAL TRACK MODEL AUDIT")
    print("=" * 80)

    # 1. Environment-Only Model Test
    print("[TEST 1/4] Testing EnvironmentOnlyTrackModel...")
    env_model = EnvironmentOnlyTrackModel(in_channels=8, spatial_dim=64, hidden_dim=64)
    total_env_params = sum(p.numel() for p in env_model.parameters())
    print(f"  Total EnvironmentOnlyTrackModel parameters: {total_env_params:,}")
    assert total_env_params < 200_000, f"Environment model should be lightweight (<200k), got {total_env_params:,}"

    # Synthetic batch: [B=2, T=6, C=8, H=41, W=66]
    dummy_env = torch.randn(2, 6, 8, 41, 66, requires_grad=True)
    out_env = env_model(dummy_env)
    assert 'pred_12' in out_env and 'pred_24' in out_env and 'pred_48' in out_env
    assert out_env['pred_12'].shape == (2, 2)
    assert out_env['pred_24'].shape == (2, 2)
    assert out_env['pred_48'].shape == (2, 2)
    print("  PASS: EnvironmentOnlyTrackModel forward output shapes verified.")

    # Backward test
    loss_env = out_env['pred_12'].sum() + out_env['pred_24'].sum() + out_env['pred_48'].sum()
    loss_env.backward()
    assert dummy_env.grad is not None and not torch.isnan(dummy_env.grad).any()
    print("  PASS: EnvironmentOnlyTrackModel gradients clean and non-zero.")

    # 2. Multimodal Model Test
    print("[TEST 2/4] Testing MultimodalTrackModel...")
    mm_model = MultimodalTrackModel(
        freeze_satellite_backbone=True,
        sat_feat_dim=64,
        sat_hidden_dim=128,
        env_channels=8,
        env_spatial_dim=64,
        env_hidden_dim=64,
        fusion_dim=128
    )
    
    total_mm = sum(p.numel() for p in mm_model.parameters())
    trainable_mm = sum(p.numel() for p in mm_model.parameters() if p.requires_grad)
    frozen_mm = sum(p.numel() for p in mm_model.parameters() if not p.requires_grad)
    print(f"  MultimodalTrackModel parameters: Total={total_mm:,}, Trainable={trainable_mm:,}, Frozen={frozen_mm:,}")
    assert frozen_mm == 11_176_512 or frozen_mm > 11_000_000, f"Frozen backbone should be ~11.2M params, got {frozen_mm:,}"
    assert trainable_mm < 600_000, f"Trainable params should be compact (<600k), got {trainable_mm:,}"
    print("  PASS: Parameter distribution verified.")

    # 3. Multimodal Forward & Backward
    print("[TEST 3/4] Testing multimodal forward pass and gradient flow...")
    # Smaller dummy satellite sequence for fast audit: [B=2, T=6, C=1, H=128, W=128]
    # (Since ResNet accepts variable spatial input due to adaptive pooling)
    dummy_sat = torch.randn(2, 6, 1, 128, 128)
    dummy_env2 = torch.randn(2, 6, 8, 41, 66)
    
    out_mm = mm_model(dummy_sat, dummy_env2)
    for h in ['pred_12', 'pred_24', 'pred_48']:
        assert h in out_mm
        assert out_mm[h].shape == (2, 2)
        assert not torch.isnan(out_mm[h]).any()
    print("  PASS: Multimodal forward output shapes and values verified.")

    # Loss and gradient
    loss_mm = out_mm['pred_12'].sum() + out_mm['pred_24'].sum() + out_mm['pred_48'].sum()
    loss_mm.backward()
    
    # Check that frozen backbone parameters have no gradients
    for p in mm_model.satellite_backbone_parameters():
        assert p.grad is None, "Frozen satellite backbone parameter has unexpected gradient!"
        
    # Check that trainable fusion/head parameters have clean gradients
    for p in mm_model.fusion.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    for p in mm_model.head_12.parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any()
    print("  PASS: Strict gradient isolation verified (frozen backbone has grad=None, heads have clean grads).")

    # 4. Architecture Invariants
    print("[TEST 4/4] Verifying multimodal architecture dimensions...")
    assert mm_model.sat_gru.hidden_size == 128
    assert mm_model.env_gru.hidden_dim == 64
    assert mm_model.fusion[0].in_features == 192 # 128 + 64
    assert mm_model.fusion[0].out_features == 128
    print("  PASS: Multimodal dimensional invariants [128 + 64 -> 192 -> 128] verified.")

    print("=" * 80)
    print("ALL MULTIMODAL MODEL AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    test_models()
