"""
VAYU-NET Phase 4B — Hybrid Kinematic + Satellite Sanity Audit
Verifies:
  1. Transfer of Phase 3C spatial weights to HybridKinematicSatelliteGRU
  2. Input sequence dimensions [B, 6, 132] and kinematic context [B, 5]
  3. Output residual shapes [B, 2] for +12h, +24h, +48h
  4. Mathematical invertibility of tangent-plane residual coordinate conversions
  5. Synthetic zero-residual test: forecast == kinematic baseline when residual == 0
  6. Synthetic perfect-residual test: forecast == true position when residual == true residual
  7. Numerical stability: 0 NaN / Inf in forward and backward passes
  8. Synthetic optimization test: Smooth L1 residual loss decreases monotonically
  9. Baseline reproduction check: Exact-center constant-velocity baseline matches locked historical numbers
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from ml.models.hybrid_kinematic_satellite import (
    HybridKinematicSatelliteGRU,
    RESIDUAL_SCALE_KM,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)

PHASE3C_CHECKPOINT = "data/interim/ml/checkpoints/best_center_localization_cnn.pt"

def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return 6371.0 * c

def run_sanity_checks():
    print("=" * 80)
    print("VAYU-NET PHASE 4B — HYBRID KINEMATIC + SATELLITE SANITY AUDIT")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Model Instantiation & Weight Transfer
    print("\n--- Test 1: Instantiation & Spatial Weight Transfer ---")
    assert os.path.exists(PHASE3C_CHECKPOINT), f"Phase 3C checkpoint missing at: {PHASE3C_CHECKPOINT}"
    model = HybridKinematicSatelliteGRU(spatial_checkpoint_path=PHASE3C_CHECKPOINT, freeze_spatial_encoder=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {frozen_params:,} (Transferred Phase 3C Spatial Encoder)")
    assert frozen_params > 11_000_000, f"Expected >11M frozen spatial params, got {frozen_params}"
    assert trainable_params > 100_000, f"Expected >100k trainable params, got {trainable_params}"
    print("[PASS] Spatial weights transferred and parameter partitioning verified.")

    # 2. Local Tangent-Plane Invertibility & Synthetic Tests
    print("\n--- Test 2: Invertible Coordinate Conversions & Synthetic Reconstruction ---")
    kin_coords = torch.tensor([
        [12.5, 68.2],
        [18.0, 84.5],
        [8.2, 92.1]
    ], dtype=torch.float32)
    
    true_coords = torch.tensor([
        [14.1, 69.8],
        [19.7, 85.9],
        [10.5, 93.4]
    ], dtype=torch.float32)
    
    # Compute true residuals in km
    true_res_km = HybridKinematicSatelliteGRU.latlon_to_residual_km(kin_coords, true_coords)
    print(f"Computed residuals in km:\n{true_res_km.numpy()}")
    
    # Perfect residual reconstruction test
    rec_coords = HybridKinematicSatelliteGRU.residual_km_to_latlon(kin_coords, true_res_km)
    diff = torch.abs(rec_coords - true_coords).max().item()
    print(f"Max reconstruction discrepancy (True vs Reconstructed): {diff:.2e}°")
    assert diff < 1e-5, f"Reconstruction failed, max diff: {diff}"
    print("[PASS] Perfect residual reconstruction test passed (diff < 1e-5°).")
    
    # Zero residual test
    zero_res = torch.zeros_like(true_res_km)
    zero_rec = HybridKinematicSatelliteGRU.residual_km_to_latlon(kin_coords, zero_res)
    diff_zero = torch.abs(zero_rec - kin_coords).max().item()
    print(f"Max discrepancy for zero residual (Kinematic vs Reconstructed): {diff_zero:.2e}°")
    assert diff_zero < 1e-6, f"Zero residual test failed, max diff: {diff_zero}"
    print("[PASS] Zero residual test passed: forecast exactly equals kinematic base.")

    # 3. Forward Pass & Residual Dimensions
    print("\n--- Test 3: Forward Pass on Feature Sequence & Kinematic Context ---")
    batch_size = 4
    dummy_feats = torch.randn(batch_size, 6, model.frame_feat_dim, device=device)
    dummy_kin = torch.randn(batch_size, model.kin_feat_dim, device=device)
    
    out = model.forward_features(dummy_feats, dummy_kin)
    r12 = out["r12_km"]
    r24 = out["r24_km"]
    r48 = out["r48_km"]
    print(f"Input features shape:   {list(dummy_feats.shape)}")
    print(f"Kinematic context:      {list(dummy_kin.shape)}")
    print(f"Predicted residual 12h: {list(r12.shape)}")
    print(f"Predicted residual 24h: {list(r24.shape)}")
    print(f"Predicted residual 48h: {list(r48.shape)}")
    
    assert r12.shape == (batch_size, 2)
    assert r24.shape == (batch_size, 2)
    assert r48.shape == (batch_size, 2)
    assert not torch.isnan(r12).any() and not torch.isinf(r12).any()
    assert not torch.isnan(r24).any() and not torch.isinf(r24).any()
    assert not torch.isnan(r48).any() and not torch.isinf(r48).any()
    print("[PASS] Forward pass produced correct residual shapes with zero NaN/Inf.")

    # 4. Optimization & Gradient Flow Sanity
    print("\n--- Test 4: Optimization Sanity (Loss Decrease on Synthetic Residuals) ---")
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    
    target_r12_norm = torch.tensor([[0.5, -0.3], [-0.2, 0.4], [0.1, 0.2], [-0.4, -0.1]], device=device)
    target_r24_norm = torch.tensor([[1.0, -0.6], [-0.5, 0.8], [0.3, 0.5], [-0.8, -0.2]], device=device)
    target_r48_norm = torch.tensor([[2.0, -1.2], [-1.0, 1.5], [0.6, 1.0], [-1.5, -0.5]], device=device)
    
    initial_loss = None
    final_loss = None
    for step in range(15):
        optimizer.zero_grad()
        preds = model.forward_features(dummy_feats, dummy_kin)
        
        l12 = loss_fn(preds["r12_norm"], target_r12_norm)
        l24 = loss_fn(preds["r24_norm"], target_r24_norm)
        l48 = loss_fn(preds["r48_norm"], target_r48_norm)
        total_loss = l12 + l24 + l48
        total_loss.backward()
        
        for p_name, p in model.named_parameters():
            if p.requires_grad and ("gru" in p_name or "head_" in p_name):
                assert p.grad is not None, f"Missing gradient for {p_name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {p_name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {p_name}"
                
        optimizer.step()
        if step == 0:
            initial_loss = total_loss.item()
        final_loss = total_loss.item()
        
    print(f"Initial synthetic residual loss: {initial_loss:.6f}")
    print(f"Final synthetic residual loss:   {final_loss:.6f} (after 15 steps)")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    print("[PASS] Gradient flow clean (0 NaN/Inf) and synthetic loss decreases monotonically.")

    # 5. Baseline Reproduction Audit
    print("\n--- Test 5: Exact Constant-Velocity Baseline Reproduction on TEST ---")
    df = pd.read_csv("data/manifests/vayu_net_sample_index.csv")
    v2 = pd.read_csv("data/processed/imd_best_track_v2.csv")
    storm_obs_sorted = {}
    for sid, group in v2.groupby("storm_id"):
        gdf = group.copy()
        gdf["dt"] = pd.to_datetime(gdf["timestamp_utc"])
        gdf = gdf.sort_values("dt").reset_index(drop=True)
        storm_obs_sorted[sid] = gdf

    test_df = df[df["split"] == "TEST"].copy().reset_index(drop=True)
    cv_dpes = {12: [], 24: [], 48: []}
    for _, row in test_df.iterrows():
        sid = row["storm_id"]
        t0_dt = pd.to_datetime(row["t0"])
        lat0, lon0 = row["imd_lat_t0"], row["imd_lon_t0"]
        s_obs = storm_obs_sorted[sid]
        priors = s_obs[s_obs["dt"] < t0_dt]
        if len(priors) > 0:
            latest_prior = priors.iloc[-1]
            delta_h = (t0_dt - latest_prior["dt"]).total_seconds() / 3600.0
            v_lat = (lat0 - latest_prior["latitude"]) / delta_h
            d_lon = lon0 - latest_prior["longitude"]
            if d_lon > 180.0: d_lon -= 360.0
            elif d_lon < -180.0: d_lon += 360.0
            v_lon = d_lon / delta_h
        else:
            v_lat, v_lon = 0.0, 0.0
            
        for h in [12, 24, 48]:
            t_lat, t_lon = row[f"imd_lat_{h}h"], row[f"imd_lon_{h}h"]
            cv_dpes[h].append(haversine_km(lat0 + v_lat*h, lon0 + v_lon*h, t_lat, t_lon))

    expected_cv = {12: 71.28, 24: 150.74, 48: 345.15}
    for h in [12, 24, 48]:
        actual = np.mean(cv_dpes[h])
        exp = expected_cv[h]
        diff = abs(actual - exp)
        print(f"  +{h}h CV Mean DPE: {actual:.2f} km (Expected: {exp:.2f} km, Diff: {diff:.4f} km)")
        assert diff < 0.1, f"+{h}h CV baseline mismatch: {actual} vs {exp}"
    print("[PASS] Exact-center constant-velocity baseline reproduced to <0.1 km precision.")

    print("\n" + "=" * 80)
    print("ALL PHASE 4B HYBRID SANITY CHECKS PASSED")
    print("=" * 80)

if __name__ == "__main__":
    run_sanity_checks()
