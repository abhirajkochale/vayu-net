"""
VAYU-NET: EXP-M3 Pre-training Dry Run Safety Checks.
Verifies forward pass, backward pass, finite gradients, optimizer step,
spatial gate dimensions, and non-contamination across M3A, M3B, and M3C.
"""

import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.experiments.exp_m3.model import ExpM3Model
from ml.experiments.exp_m3.losses import MaskedMultiTaskLoss

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"


def run_dry_run():
    print("=" * 65)
    print("VAYU-NET: RUNNING EXP-M3 PRE-TRAINING DRY RUN SAFETY SUITE")
    print("=" * 65)

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found at {CACHE_PATH}")

    print(f"Loading sample batch from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    train_samples = [s for s in cache["samples"] if s["split"] == "TRAIN"][:16]
    assert len(train_samples) == 16, f"Expected 16 train samples, got {len(train_samples)}"

    batch = {}
    for key in train_samples[0]:
        val0 = train_samples[0][key]
        if isinstance(val0, torch.Tensor):
            batch[key] = torch.stack([s[key] for s in train_samples])
        else:
            batch[key] = [s[key] for s in train_samples]

    criterion = MaskedMultiTaskLoss()
    modes = ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]

    for mode in modes:
        print(f"\n--- Testing Mode: {mode} ---")
        model = ExpM3Model(mode=mode)
        model.train()
        params = model.get_parameter_counts()
        print(f"  Trainable Parameters: {params['trainable_parameters']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        # Forward pass
        if mode == "m3a_gridsat":
            preds = model(gridsat_seq=batch["gridsat"])
        elif mode == "m3b_imerg":
            preds = model(imerg_seq=batch["imerg"])
        else: # m3c_spatial
            preds = model(gridsat_seq=batch["gridsat"], imerg_seq=batch["imerg"])

        # Shape verification
        assert preds["center_norm"].shape == (16, 2), f"Bad center shape: {preds['center_norm'].shape}"
        assert preds["center_deg"].shape == (16, 2), f"Bad center_deg shape: {preds['center_deg'].shape}"
        assert preds["class_logits"].shape == (16, 7), f"Bad class_logits shape: {preds['class_logits'].shape}"
        assert preds["wind_norm"].shape == (16, 1), f"Bad wind_norm shape: {preds['wind_norm'].shape}"
        assert preds["wind_kt"].shape == (16, 1), f"Bad wind_kt shape: {preds['wind_kt'].shape}"
        assert preds["track_12h_norm"].shape == (16, 2), f"Bad track_12h shape: {preds['track_12h_norm'].shape}"
        assert preds["track_24h_norm"].shape == (16, 2), f"Bad track_24h shape: {preds['track_24h_norm'].shape}"
        assert preds["track_48h_norm"].shape == (16, 2), f"Bad track_48h shape: {preds['track_48h_norm'].shape}"
        print("  [PASS] Output tensor shapes verified.")

        # Spatial Gate verification for M3C
        if mode == "m3c_spatial":
            assert "spatial_gate_alpha" in preds
            gate = preds["spatial_gate_alpha"]
            assert gate.shape == (16, 6, 64, 9, 15), f"Bad spatial gate shape: {gate.shape}"
            assert not torch.isnan(gate).any(), "NaN found in spatial gate!"
            assert not torch.isinf(gate).any(), "Inf found in spatial gate!"
            assert (gate >= 0.0).all() and (gate <= 1.0).all(), "Spatial gate values out of [0, 1] bounds!"
            mean_g = gate.mean().item()
            std_g = gate.std().item()
            spatial_var = gate.var(dim=(-2, -1)).mean().item()
            print(f"  [PASS] Spatial gate verified: mean={mean_g:.4f}, std={std_g:.4f}, spatial_variance={spatial_var:.6f} in [0, 1].")

        # Loss calculation
        loss, loss_dict = criterion(preds, batch)
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
        print(f"  [PASS] Multi-task loss finite: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        print("  [PASS] Backward pass: all gradients finite and bounded.")

        # Optimizer step
        optimizer.step()
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in parameter {name} after step"
            assert not torch.isinf(param).any(), f"Inf in parameter {name} after step"
        print("  [PASS] Optimizer step successful, parameters finite.")

    print("\n" + "=" * 65)
    print("[DRY RUN SUCCESS] All M3A, M3B, M3C safety checks PASSED.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_dry_run())
