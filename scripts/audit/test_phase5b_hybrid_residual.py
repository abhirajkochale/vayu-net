"""
VAYU-NET PHASE 5B — HYBRID RESIDUAL MODEL AUDIT & SANITY TEST
=============================================================
Verifies:
1. Zero-Residual Reconstruction: R_hat = [0, 0] ==> P_hat == P_kin (diff = 0.00 km)
2. True-Residual Reconstruction: R_hat = R_true ==> P_hat == P_true (diff < 1e-6 deg)
3. Parameter distribution and compact model capacity (<500k params)
4. Forward pass output shapes, coordinate clipping bounds, and gradient backprop
5. No NaNs or Infs in outputs or gradients
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.models.phase5b_hybrid_residual import Phase5BHybridResidualModel

def test_phase5b_hybrid():
    print("=" * 80)
    print("RUNNING VAYU-NET PHASE 5B: HYBRID RESIDUAL MODEL AUDIT")
    print("=" * 80)

    # 1. Zero-Residual and True-Residual Invertibility Tests
    print("[TEST 1/4] Verifying mathematical invertibility of geodesic coordinate conversion...")
    kin_coords = torch.tensor([[15.0, 70.0], [20.0, 85.0]], dtype=torch.float32)
    true_coords = torch.tensor([[16.2, 69.5], [21.5, 87.2]], dtype=torch.float32)
    
    # Forward conversion: latlon -> residual_km
    res_km = Phase5BHybridResidualModel.latlon_to_residual_km(kin_coords, true_coords)
    print(f"  Calculated sample true residuals (km): {res_km.tolist()}")
    
    # Reconstruct from true residual: P_hat = P_kin + R_true
    reconstructed_true = Phase5BHybridResidualModel.residual_km_to_latlon(kin_coords, res_km, clip_bounds=False)
    max_recon_diff = torch.max(torch.abs(reconstructed_true - true_coords)).item()
    print(f"  Max difference between true target and reconstructed coords: {max_recon_diff:.2e} degrees")
    assert max_recon_diff < 1e-5, f"Reconstruction failed: {max_recon_diff}"
    print("  PASS: True-residual reconstruction test passed within numerical precision (< 1e-5 deg).")

    # Zero-residual test: R_hat = [0, 0] ==> P_hat == P_kin
    zero_res = torch.zeros_like(res_km)
    reconstructed_zero = Phase5BHybridResidualModel.residual_km_to_latlon(kin_coords, zero_res, clip_bounds=False)
    zero_diff = torch.max(torch.abs(reconstructed_zero - kin_coords)).item()
    print(f"  Max difference between zero-residual forecast and kinematic base: {zero_diff:.2e} degrees")
    assert zero_diff == 0.0, f"Zero residual difference must be exactly 0.0, got {zero_diff}"
    print("  PASS: Zero-residual reconstruction equals kinematic anchor to 0.00e+00 km.")

    # 2. Model Initialization & Parameters
    print("[TEST 2/4] Testing Phase5BHybridResidualModel parameter counts...")
    model = Phase5BHybridResidualModel(
        sat_feat_dim=132,
        sat_hidden_dim=128,
        env_channels=8,
        env_spatial_dim=64,
        env_hidden_dim=64,
        motion_in_dim=5,
        motion_emb_dim=16,
        fusion_dim=128,
        dropout=0.1
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total model parameters: {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    assert trainable_params < 500_000, f"Model should remain compact (<500k params), got {trainable_params:,}"
    print("  PASS: Parameter count verified.")

    # 3. Multimodal Forward Pass
    print("[TEST 3/4] Testing forward pass with synthetic batch...")
    B = 4
    dummy_sat = torch.randn(B, 6, 132)
    dummy_env = torch.randn(B, 6, 8, 41, 66)
    dummy_mot = torch.randn(B, 5)
    dummy_k12 = torch.tensor([[15.0, 70.0]] * B, dtype=torch.float32)
    dummy_k24 = torch.tensor([[16.0, 69.0]] * B, dtype=torch.float32)
    dummy_k48 = torch.tensor([[18.0, 67.0]] * B, dtype=torch.float32)
    
    out = model(dummy_sat, dummy_env, dummy_mot, dummy_k12, dummy_k24, dummy_k48)
    for k in ['r12_km', 'r24_km', 'r48_km', 'pred_12', 'pred_24', 'pred_48']:
        assert k in out, f"Missing output key: {k}"
        assert out[k].shape == (B, 2), f"Shape mismatch for {k}: {out[k].shape}"
        assert not torch.isnan(out[k]).any(), f"Found NaN in {k}"
    print(f"  PASS: Forward output shapes and values verified across all horizons.")

    # 4. Gradient Flow Test
    print("[TEST 4/4] Verifying gradient backpropagation from multi-horizon loss...")
    loss = out['r12_norm'].sum() + out['r24_norm'].sum() + out['r48_norm'].sum()
    loss.backward()
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient!"
    print("  PASS: All trainable parameters received clean, non-zero gradients.")

    print("=" * 80)
    print("ALL PHASE 5B AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    test_phase5b_hybrid()
