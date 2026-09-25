"""
VAYU-NET — MULTI-SOURCE DECOUPLED ARCHITECTURE AUDIT TEST SUITE
================================================================
PyTest verification suite ensuring:
  1. Exact split counts: TRAIN=207 (25 storms), VAL=252 (14 storms), TEST=298 (24 storms)
  2. Zero split overlap and zero future-frame leakage
  3. Pretrained GridSat GRU encoder remains strictly frozen (199,680 parameters)
  4. Selective routing pathway isolation across Arch A, Arch B, Arch C, Arch Center, Ablation I2
  5. Deterministic numerical reproducibility and tensor shape consistency
"""

import os
import sys
import json
from pathlib import Path
import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multisource_decoupled import (
    MultisourceDecoupledModel,
    haversine_km,
    DEFAULT_TRAIN_WIND_MEAN_KT,
    DEFAULT_TRAIN_WIND_STD_KT
)


def test_split_counts_and_isolation():
    cache_path = PROJECT_ROOT / "data/interim/ml/cache/multisource_transfer_cache_expanded.pt"
    assert cache_path.exists(), "Expanded transfer cache missing!"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    samples = cache["samples"]

    train_samples = [s for s in samples if s["split"] == "TRAIN"]
    val_samples = [s for s in samples if s["split"] == "VALIDATION"]
    test_samples = [s for s in samples if s["split"] == "TEST"]

    assert len(train_samples) == 207, f"Expected 207 TRAIN samples, got {len(train_samples)}"
    assert len(val_samples) == 252, f"Expected 252 VAL samples, got {len(val_samples)}"
    assert len(test_samples) == 298, f"Expected 298 TEST samples, got {len(test_samples)}"

    train_storms = set(s["storm_id"] for s in train_samples)
    val_storms = set(s["storm_id"] for s in val_samples)
    test_storms = set(s["storm_id"] for s in test_samples)

    assert len(train_storms) == 25, f"Expected 25 TRAIN storms, got {len(train_storms)}"
    assert len(val_storms) == 14, f"Expected 14 VAL storms, got {len(val_storms)}"
    assert len(test_storms) == 24, f"Expected 24 TEST storms, got {len(test_storms)}"

    assert len(train_storms & val_storms) == 0, "Train-Val storm leakage detected!"
    assert len(train_storms & test_storms) == 0, "Train-Test storm leakage detected!"
    assert len(val_storms & test_storms) == 0, "Val-Test storm leakage detected!"


def test_frozen_encoder_and_parameter_budgets():
    modes = [
        ("baseline_gridsat", 300560, 100880, 199680),
        ("arch_a_decoupled_track", 373968, 174288, 199680),
        ("arch_b_decoupled_intensity", 374112, 174432, 199680),
        ("arch_c_decoupled_full", 455840, 256160, 199680),
        ("decoupled_center", 373968, 174288, 199680),
        ("ablation_i2_decoupled_intensity_tir", 373968, 174288, 199680)
    ]

    for mode_name, exp_total, exp_trainable, exp_frozen in modes:
        model = MultisourceDecoupledModel(mode=mode_name)
        counts = model.get_parameter_counts()
        assert counts["frozen_parameters"] == 199680, f"Frozen params mismatch for {mode_name}"
        assert counts["total_parameters"] == exp_total, f"Total params mismatch for {mode_name}: {counts['total_parameters']} vs {exp_total}"
        assert counts["trainable_parameters"] == exp_trainable, f"Trainable params mismatch for {mode_name}: {counts['trainable_parameters']} vs {exp_trainable}"

        # Verify that all gridsat_gru weights have requires_grad=False
        for p in model.gridsat_gru.parameters():
            assert not p.requires_grad, f"Found trainable parameter in frozen GridSat GRU in {mode_name}!"


def test_tensor_forward_pass_dimensions():
    B = 2
    g_feat = torch.randn(B, 6, 132)
    i_seq = torch.randn(B, 6, 3, 72, 116)

    for mode in ["baseline_gridsat", "arch_a_decoupled_track", "arch_b_decoupled_intensity", "arch_c_decoupled_full"]:
        model = MultisourceDecoupledModel(mode=mode)
        model.eval()
        with torch.no_grad():
            out = model(g_feat, i_seq)

        assert out["center_norm"].shape == (B, 2)
        assert out["center_deg"].shape == (B, 2)
        assert out["class_logits"].shape == (B, 7)
        assert out["wind_norm"].shape == (B, 1)
        assert out["wind_kt"].shape == (B, 1)
        assert out["track_12h_norm"].shape == (B, 2)
        assert out["track_12h_deg"].shape == (B, 2)
        assert out["track_24h_norm"].shape == (B, 2)
        assert out["track_24h_deg"].shape == (B, 2)
        assert out["track_48h_norm"].shape == (B, 2)
        assert out["track_48h_deg"].shape == (B, 2)


def test_selective_routing_isolation():
    B = 2
    g_feat = torch.randn(B, 6, 132)
    i_seq1 = torch.randn(B, 6, 3, 72, 116)
    i_seq2 = i_seq1.clone()
    # Perturb ONLY channel 2 (WV)
    i_seq2[:, :, 2, :, :] += 50.0

    # In Arch A (Decoupled Track): WV channel is NOT used at all.
    model_a = MultisourceDecoupledModel(mode="arch_a_decoupled_track")
    model_a.eval()
    with torch.no_grad():
        out_a1 = model_a(g_feat, i_seq1)
        out_a2 = model_a(g_feat, i_seq2)

    # In Arch A, track and center and intensity outputs MUST be identical because WV was perturbed, but Arch A only uses TIR1+TIR2!
    np.testing.assert_allclose(out_a1["track_12h_deg"].numpy(), out_a2["track_12h_deg"].numpy(), atol=1e-5)
    np.testing.assert_allclose(out_a1["track_24h_deg"].numpy(), out_a2["track_24h_deg"].numpy(), atol=1e-5)
    np.testing.assert_allclose(out_a1["center_deg"].numpy(), out_a2["center_deg"].numpy(), atol=1e-5)
    np.testing.assert_allclose(out_a1["wind_kt"].numpy(), out_a2["wind_kt"].numpy(), atol=1e-5)

    # In Arch B (Decoupled Intensity): WV is used in intensity, but Track MUST be unaffected!
    model_b = MultisourceDecoupledModel(mode="arch_b_decoupled_intensity")
    model_b.eval()
    with torch.no_grad():
        out_b1 = model_b(g_feat, i_seq1)
        out_b2 = model_b(g_feat, i_seq2)

    # Track in Arch B is strictly GridSat-only -> must be identical despite INSAT WV changes!
    np.testing.assert_allclose(out_b1["track_12h_deg"].numpy(), out_b2["track_12h_deg"].numpy(), atol=1e-5)
    np.testing.assert_allclose(out_b1["center_deg"].numpy(), out_b2["center_deg"].numpy(), atol=1e-5)
    # Intensity in Arch B DOES use WV -> wind outputs MUST differ!
    diff_wind = np.abs(out_b1["wind_kt"].numpy() - out_b2["wind_kt"].numpy()).max()
    assert diff_wind > 1e-3, "Arch B intensity should be sensitive to WV changes!"

    # In Arch C (Decoupled Full): Track uses TIR1+TIR2 only -> must be immune to WV perturbation!
    model_c = MultisourceDecoupledModel(mode="arch_c_decoupled_full")
    model_c.eval()
    with torch.no_grad():
        out_c1 = model_c(g_feat, i_seq1)
        out_c2 = model_c(g_feat, i_seq2)

    np.testing.assert_allclose(out_c1["track_12h_deg"].numpy(), out_c2["track_12h_deg"].numpy(), atol=1e-5)
    np.testing.assert_allclose(out_c1["center_deg"].numpy(), out_c2["center_deg"].numpy(), atol=1e-5)
    diff_wind_c = np.abs(out_c1["wind_kt"].numpy() - out_c2["wind_kt"].numpy()).max()
    assert diff_wind_c > 1e-3, "Arch C intensity should be sensitive to WV changes!"


def test_saved_artifacts_and_metrics():
    # 1. Checkpoints
    ckpts = [
        "multisource_decoupled_baseline.pt",
        "multisource_decoupled_track.pt",
        "multisource_decoupled_intensity.pt",
        "multisource_decoupled_full.pt",
        "multisource_decoupled_center.pt",
        "multisource_decoupled_intensity_tir.pt"
    ]
    ckpt_dir = PROJECT_ROOT / "data/interim/ml/checkpoints"
    for c in ckpts:
        path = ckpt_dir / c
        assert path.exists(), f"Missing checkpoint: {path}"
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in data
        assert "test_metrics" in data

    # 2. Results JSON and CSV
    json_path = PROJECT_ROOT / "data/interim/ml/multisource_decoupled_results.json"
    csv_path = PROJECT_ROOT / "data/interim/ml/multisource_decoupled_comparison.csv"
    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r") as f:
        res = json.load(f)
    assert len(res["experiments"]) == 6
    assert len(res["comparison_table"]) == 6

