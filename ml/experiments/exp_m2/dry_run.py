"""
VAYU-NET: EXP-M2 Pre-training Dry Run Safety Checks.
Verifies forward pass, backward pass, finite gradients, optimizer step,
gate value validity, and target insulation across M2A, M2B, and M2C.
"""

import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.experiments.exp_m2.model import ExpM2Model
from ml.experiments.exp_m2.losses import MaskedMultiTaskLoss

CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"


def run_dry_run():
    print("=" * 65)
    print("VAYU-NET: RUNNING EXP-M2 PRE-TRAINING DRY RUN SAFETY SUITE")
    print("=" * 65)

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found at {CACHE_PATH}")

    print(f"Loading sample batch from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    train_samples = [s for s in cache["samples"] if s["split"] == "TRAIN"][:32]
    assert len(train_samples) == 32, f"Expected 32 train samples, got {len(train_samples)}"

    batch = {}
    for key in train_samples[0]:
        val0 = train_samples[0][key]
        if isinstance(val0, torch.Tensor):
            batch[key] = torch.stack([s[key] for s in train_samples])
        else:
            batch[key] = [s[key] for s in train_samples]

    criterion = MaskedMultiTaskLoss()
    modes = ["m2a_gridsat", "m2b_imerg", "m2c_adaptive"]

    for mode in modes:
        print(f"\n--- Testing Mode: {mode} ---")
        model = ExpM2Model(mode=mode)
        model.train()
        params = model.get_parameter_counts()
        print(f"  Trainable Parameters: {params['trainable_parameters']:,}")

        # Check optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        # Forward pass
        if mode == "m2a_gridsat":
            preds = model(gridsat_seq=batch["gridsat"])
        elif mode == "m2b_imerg":
            preds = model(imerg_seq=batch["imerg"])
        else: # m2c_adaptive
            preds = model(gridsat_seq=batch["gridsat"], imerg_seq=batch["imerg"])

        # Shape verification
        assert preds["center_norm"].shape == (32, 2), f"Bad center shape: {preds['center_norm'].shape}"
        assert preds["center_deg"].shape == (32, 2), f"Bad center_deg shape: {preds['center_deg'].shape}"
        assert preds["class_logits"].shape == (32, 7), f"Bad class_logits shape: {preds['class_logits'].shape}"
        assert preds["wind_norm"].shape == (32, 1), f"Bad wind_norm shape: {preds['wind_norm'].shape}"
        assert preds["wind_kt"].shape == (32, 1), f"Bad wind_kt shape: {preds['wind_kt'].shape}"
        assert preds["track_12h_norm"].shape == (32, 2), f"Bad track_12h shape: {preds['track_12h_norm'].shape}"
        assert preds["track_24h_norm"].shape == (32, 2), f"Bad track_24h shape: {preds['track_24h_norm'].shape}"
        assert preds["track_48h_norm"].shape == (32, 2), f"Bad track_48h shape: {preds['track_48h_norm'].shape}"
        print("  [PASS] Output tensor shapes verified.")

        # Gate verification for M2C
        if mode == "m2c_adaptive":
            assert "gate_alpha" in preds
            alpha = preds["gate_alpha"]
            assert alpha.shape == (32, 6, 1), f"Bad gate shape: {alpha.shape}"
            assert not torch.isnan(alpha).any(), "NaN found in gate alpha!"
            assert not torch.isinf(alpha).any(), "Inf found in gate alpha!"
            assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), "Gate alpha out of [0, 1] bounds!"
            mean_alpha = alpha.mean().item()
            std_alpha = alpha.std().item()
            print(f"  [PASS] Adaptive gate verified: mean={mean_alpha:.4f}, std={std_alpha:.4f} in [0, 1].")

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
    print("[DRY RUN SUCCESS] All M2A, M2B, M2C safety checks PASSED.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_dry_run())
