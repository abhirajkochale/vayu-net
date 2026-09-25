"""
VAYU-NET: EXP-M1 Dry-Run Verification.
Tests M1A (GridSat-only), M1B (IMERG-only), and M1C (Multimodal) across:
1. Instantiation
2. Single-batch loading
3. Forward pass & output shapes
4. Loss computation
5. Backward pass & finite gradients
6. Optimizer step
7. Parameter counting
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.multimodal_dataset import MultimodalVayuDataset
from ml.experiments.exp_m1.model import ExpM1Model
from ml.experiments.exp_m1.losses import MaskedMultiTaskLoss


def run_dry_run():
    print("=" * 65)
    print("VAYU-NET: EXP-M1 MULTIMODAL ARCHITECTURE DRY RUN")
    print("=" * 65)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load one small batch (batch_size=4 for speed)
    print("\nLoading single batch of 4 TRAIN sequences...")
    ds = MultimodalVayuDataset(split="TRAIN", normalize=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"Loaded batch with keys: {list(batch.keys())}")
    print(f"GridSat tensor shape: {batch['gridsat'].shape}")
    print(f"IMERG tensor shape:   {batch['imerg'].shape}")

    configs = [
        ("M1A (GridSat-only)", "m1a_gridsat", True, False),
        ("M1B (IMERG-only)",   "m1b_imerg",   False, True),
        ("M1C (GridSat+IMERG)","m1c_fusion",  True, True)
    ]

    all_passed = True
    results = {}

    for name, mode, use_g, use_i in configs:
        print(f"\n--- Testing {name} ---")
        try:
            # 1. Instantiate model
            model = ExpM1Model(mode=mode).to(device)
            params = model.get_parameter_counts()
            print(f"  [1] Instantiated: {params['total_parameters']:,} total parameters ({params['trainable_parameters']:,} trainable)")

            # 2. Forward pass
            g_in = batch["gridsat"].to(device) if use_g else None
            i_in = batch["imerg"].to(device) if use_i else None
            preds = model(gridsat_seq=g_in, imerg_seq=i_in)

            print(f"  [2] Forward pass successful. Output tensor shapes:")
            print(f"      center_norm:     {tuple(preds['center_norm'].shape)}")
            print(f"      center_deg:      {tuple(preds['center_deg'].shape)}")
            print(f"      class_logits:    {tuple(preds['class_logits'].shape)}")
            print(f"      wind_norm:       {tuple(preds['wind_norm'].shape)}")
            print(f"      wind_kt:         {tuple(preds['wind_kt'].shape)}")
            print(f"      track_12h_deg:   {tuple(preds['track_12h_deg'].shape)}")
            print(f"      track_24h_deg:   {tuple(preds['track_24h_deg'].shape)}")
            print(f"      track_48h_deg:   {tuple(preds['track_48h_deg'].shape)}")

            # Check no NaNs/Infs in predictions
            for k, v in preds.items():
                if isinstance(v, torch.Tensor):
                    assert not torch.isnan(v).any(), f"NaN in prediction {k}"
                    assert not torch.isinf(v).any(), f"Inf in prediction {k}"
            print("  [3] Verified: Zero NaNs/Infs in output tensors.")

            # 3. Loss calculation
            criterion = MaskedMultiTaskLoss()
            loss, loss_dict = criterion(preds, batch)
            print(f"  [4] Loss computed: total={loss.item():.4f}, center={loss_dict['loss_center']:.4f}, wind={loss_dict['loss_wind']:.4f}, class={loss_dict['loss_class']:.4f}, track={loss_dict['loss_track']:.4f}")
            assert not torch.isnan(loss), "Loss is NaN"
            assert not torch.isinf(loss), "Loss is Inf"

            # 4. Backward pass & finite gradients
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            loss.backward()

            grad_norms = []
            for p_name, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"Missing gradient for {p_name}"
                    assert not torch.isnan(p.grad).any(), f"NaN gradient in {p_name}"
                    assert not torch.isinf(p.grad).any(), f"Inf gradient in {p_name}"
                    grad_norms.append(p.grad.norm().item())

            mean_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            print(f"  [5] Backward pass: All gradients finite (mean norm={mean_grad:.4f}, max norm={max_grad:.4f})")

            # 5. Optimizer step
            optimizer.step()
            print("  [6] Optimizer step executed cleanly.")

            results[mode] = {
                "status": "PASS",
                "parameters": params["trainable_parameters"],
                "total_loss": float(loss.item())
            }
            print(f"  >>> {name}: ALL CHECKS PASSED.")

        except Exception as e:
            print(f"  >>> [FAIL] {name}: {e}")
            all_passed = False
            results[mode] = {"status": "FAIL", "error": str(e)}

    print("\n" + "=" * 65)
    if all_passed:
        print("[DRY RUN SUCCESS] All 3 models (M1A, M1B, M1C) passed forward, backward, and optimizer checks.")
    else:
        print("[DRY RUN FAILURE] One or more dry runs failed.")
    print("=" * 65)
    return all_passed

if __name__ == "__main__":
    ok = run_dry_run()
    sys.exit(0 if ok else 1)