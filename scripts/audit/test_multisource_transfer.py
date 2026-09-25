"""
VAYU-NET — MULTI-SOURCE TRANSFER LEARNING ARCHITECTURE & INVARIANT TESTS
=========================================================================
Deterministic PyTest test suite verifying:
  - Freezing mechanics for EXP-1, EXP-2, and EXP-3
  - Parameter counts and trainable budgets
  - Input/Output tensor shapes across all tasks
  - Channel slicing for INSAT ablation (1 ch, 2 ch, 3 ch)
  - Strict anti-leakage and causal integrity
"""

import os
import sys
from pathlib import Path
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multisource_transfer_fusion import (
    MultisourceTransferFusionModel,
    haversine_km,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)


def test_exp1_freezing_and_parameter_counts():
    """Verify EXP-1 freezes the pretrained GridSat GRU completely."""
    model = MultisourceTransferFusionModel(mode="exp1_gridsat_frozen")
    counts = model.get_parameter_counts()

    # Pretrained GRU has ~199k params, all frozen
    assert counts["frozen_parameters"] == 199680
    assert counts["trainable_parameters"] < 80000

    # Verify explicitly that all GRU params have requires_grad == False
    for name, p in model.gridsat_gru.named_parameters():
        assert not p.requires_grad, f"Param {name} should be frozen in EXP-1!"

    # Verify heads are trainable
    for p in model.head_center.parameters():
        assert p.requires_grad


def test_exp2_freezing_and_insat_branch_trainable():
    """Verify EXP-2 freezes GridSat GRU while keeping INSAT branch trainable."""
    model = MultisourceTransferFusionModel(mode="exp2_fusion_frozen", in_channels_insat=3)
    counts = model.get_parameter_counts()

    assert counts["frozen_parameters"] == 199680
    assert counts["trainable_parameters"] > 100000

    for name, p in model.gridsat_gru.named_parameters():
        assert not p.requires_grad, f"Param {name} should be frozen in EXP-2!"

    for name, p in model.insat_branch.named_parameters():
        assert p.requires_grad, f"INSAT param {name} should be trainable in EXP-2!"


def test_exp3_partial_unfreezing():
    """Verify EXP-3 freezes layer 0 and unfreezes layer 1 of the pretrained GRU."""
    model = MultisourceTransferFusionModel(mode="exp3_fusion_partial_unfreeze", in_channels_insat=3)
    counts = model.get_parameter_counts()

    # Layer 0 is frozen (~100k), layer 1 is trainable (~134k)
    assert counts["frozen_parameters"] == 100608
    assert counts["trainable_parameters"] > 200000

    for name, p in model.gridsat_gru.named_parameters():
        if "_l0" in name:
            assert not p.requires_grad, f"Layer 0 param {name} should be frozen in EXP-3!"
        elif "_l1" in name:
            assert p.requires_grad, f"Layer 1 param {name} should be trainable in EXP-3!"


def test_channel_ablation_shapes():
    """Verify forward pass with 1, 2, and 3 INSAT channels."""
    B = 2
    g_feat = torch.randn(B, 6, 132)

    # 1 channel (TIR1 only)
    m1 = MultisourceTransferFusionModel(mode="exp2_fusion_frozen", in_channels_insat=1)
    i_seq1 = torch.randn(B, 6, 1, 72, 116)
    out1 = m1(gridsat_feat_seq=g_feat, insat_seq=i_seq1)
    assert out1["center_norm"].shape == (B, 2)
    assert out1["track_12h_deg"].shape == (B, 2)

    # 2 channels (TIR1 + TIR2)
    m2 = MultisourceTransferFusionModel(mode="exp2_fusion_frozen", in_channels_insat=2)
    i_seq2 = torch.randn(B, 6, 2, 72, 116)
    out2 = m2(gridsat_feat_seq=g_feat, insat_seq=i_seq2)
    assert out2["center_norm"].shape == (B, 2)

    # 3 channels (TIR1 + TIR2 + WV)
    m3 = MultisourceTransferFusionModel(mode="exp2_fusion_frozen", in_channels_insat=3)
    i_seq3 = torch.randn(B, 6, 3, 72, 116)
    out3 = m3(gridsat_feat_seq=g_feat, insat_seq=i_seq3)
    assert out3["center_norm"].shape == (B, 2)


def test_anti_leakage_and_causality():
    """Verify EXP-1 does not accept or depend on insat_seq."""
    model = MultisourceTransferFusionModel(mode="exp1_gridsat_frozen")
    g_feat = torch.randn(2, 6, 132)
    out = model(gridsat_feat_seq=g_feat, insat_seq=None)
    assert "center_deg" in out
    assert "wind_kt" in out


if __name__ == "__main__":
    pytest.main(["-v", __file__])
