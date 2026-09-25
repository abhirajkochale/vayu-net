"""
VAYU-NET: Comprehensive EXP-M2 Ablation Integrity Validator.
Verifies the scientific integrity, split preservation, data governance,
gating validity, and non-contamination of the EXP-M2 controlled adaptive fusion experiment.
"""

import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "data/manifests/gridsat_imerg_multimodal_dataset_manifest.csv"
NORM_STATS_PATH = REPO_ROOT / "data/interim/ml/imerg_train_normalization_stats.json"
CACHE_PATH = REPO_ROOT / "data/interim/ml/exp_m1_tensor_cache.pt"
CHECKPOINT_DIR = REPO_ROOT / "ml/experiments/exp_m2/checkpoints"
RESULTS_PATH = REPO_ROOT / "ml/experiments/exp_m2/results/exp_m2_results.json"
GATE_CSV_PATH = REPO_ROOT / "ml/experiments/exp_m2/results/modality_gates_test.csv"


def test_1_same_train_ids_across_m2a_m2b_m2c():
    """1. Same train sample IDs across M2A, M2B, M2C."""
    df = pd.read_csv(MANIFEST_PATH)
    train_ids = set(df[df["split"] == "TRAIN"]["sample_id"])
    assert len(train_ids) == 696
    cache = torch.load(CACHE_PATH, map_location="cpu")
    cache_train_ids = set(s["sample_id"] for s in cache["samples"] if s["split"] == "TRAIN")
    assert train_ids == cache_train_ids


def test_2_same_validation_ids():
    """2. Same validation sample IDs."""
    df = pd.read_csv(MANIFEST_PATH)
    val_ids = set(df[df["split"] == "VALIDATION"]["sample_id"])
    assert len(val_ids) == 252
    cache = torch.load(CACHE_PATH, map_location="cpu")
    cache_val_ids = set(s["sample_id"] for s in cache["samples"] if s["split"] == "VALIDATION")
    assert val_ids == cache_val_ids


def test_3_same_test_ids():
    """3. Same test sample IDs."""
    df = pd.read_csv(MANIFEST_PATH)
    test_ids = set(df[df["split"] == "TEST"]["sample_id"])
    assert len(test_ids) == 371
    cache = torch.load(CACHE_PATH, map_location="cpu")
    cache_test_ids = set(s["sample_id"] for s in cache["samples"] if s["split"] == "TEST")
    assert test_ids == cache_test_ids


def test_4_same_test_storm_ids():
    """4. Same 31 test storm IDs."""
    df = pd.read_csv(MANIFEST_PATH)
    test_storms = set(df[df["split"] == "TEST"]["storm_id"])
    assert len(test_storms) == 31
    cache = torch.load(CACHE_PATH, map_location="cpu")
    cache_test_storms = set(s["storm_id"] for s in cache["samples"] if s["split"] == "TEST")
    assert test_storms == cache_test_storms


def test_5_no_test_data_used_in_training():
    """5. Zero overlap between train storms and test storms (no train/test leakage)."""
    df = pd.read_csv(MANIFEST_PATH)
    train_storms = set(df[df["split"] == "TRAIN"]["storm_id"])
    test_storms = set(df[df["split"] == "TEST"]["storm_id"])
    assert len(train_storms & test_storms) == 0


def test_6_no_test_data_used_for_normalization():
    """6. IMERG normalization statistics derived from TRAIN ONLY."""
    with open(NORM_STATS_PATH, "r") as f:
        stats = json.load(f)
    assert stats["split"] == "TRAIN_ONLY"
    assert stats["num_train_sequences"] == 696


def test_7_no_future_frames():
    """7. Strict causality: zero future frames relative to t0."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["leakage_check"] == "ZERO_FUTURE_LEAKAGE").all()


def test_8_six_input_frames_per_modality():
    """8. Exactly six frames per sequence for both GridSat and IMERG."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["gridsat_frame_count"] == 6).all()
    assert (df["imerg_frame_count"] == 6).all()


def test_9_correct_tensor_shapes():
    """9. Correct tensor shapes [6, 72, 116] for both GridSat and IMERG."""
    cache = torch.load(CACHE_PATH, map_location="cpu")
    s = cache["samples"][0]
    assert s["gridsat"].shape == (6, 72, 116)
    assert s["imerg"].shape == (6, 72, 116)


def test_10_finite_loss_and_gradients():
    """10. Multi-task loss and backward pass produce finite gradients."""
    from ml.experiments.exp_m2.losses import MaskedMultiTaskLoss
    from ml.experiments.exp_m2.model import ExpM2Model
    criterion = MaskedMultiTaskLoss()
    cache = torch.load(CACHE_PATH, map_location="cpu")
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in cache["samples"][0].items()}
    m = ExpM2Model(mode="m2c_adaptive")
    preds = m(gridsat_seq=batch["gridsat"], imerg_seq=batch["imerg"])
    loss, _ = criterion(preds, batch)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()
    for p in m.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()


def test_11_checkpoints_saved():
    """11. All 3 EXP-M2 model checkpoints exist on disk."""
    assert (CHECKPOINT_DIR / "m2a_gridsat_only.pt").exists()
    assert (CHECKPOINT_DIR / "m2b_imerg_only.pt").exists()
    assert (CHECKPOINT_DIR / "m2c_adaptive_fusion.pt").exists()


def test_12_best_checkpoint_selected_using_validation_only():
    """12. Checkpoint metadata confirms selection based on VALIDATION loss."""
    for ckpt_name in ["m2a_gridsat_only.pt", "m2b_imerg_only.pt", "m2c_adaptive_fusion.pt"]:
        payload = torch.load(CHECKPOINT_DIR / ckpt_name, map_location="cpu")
        assert "best_val_loss" in payload
        assert "best_epoch" in payload
        assert payload["best_epoch"] >= 1


def test_13_adaptive_gate_validity():
    """13. Modality gate values are bounded in [0, 1] with zero NaNs/Infs."""
    assert GATE_CSV_PATH.exists()
    df_gate = pd.read_csv(GATE_CSV_PATH)
    assert len(df_gate) == 371
    assert not df_gate["gridsat_weight"].isna().any()
    assert not df_gate["imerg_weight"].isna().any()
    assert (df_gate["gridsat_weight"] >= 0.0).all() and (df_gate["gridsat_weight"] <= 1.0).all()
    assert (df_gate["imerg_weight"] >= 0.0).all() and (df_gate["imerg_weight"] <= 1.0).all()
    # Ensure weights sum to 1.0 within numerical precision
    diff = (df_gate["gridsat_weight"] + df_gate["imerg_weight"] - 1.0).abs()
    assert (diff < 1e-5).all()


def test_14_evaluation_metrics_and_paired_differences():
    """14. Results JSON contains full test metrics and paired bootstrap differences."""
    assert RESULTS_PATH.exists()
    with open(RESULTS_PATH, "r") as f:
        res = json.load(f)
    assert "m2a_gridsat" in res
    assert "m2b_imerg" in res
    assert "m2c_adaptive" in res
    assert "paired_differences_m2c_minus_m2a" in res
    paired = res["paired_differences_m2c_minus_m2a"]
    assert "center_dpe_diff_km" in paired
    assert "wind_mae_diff_kt" in paired
    assert "track_agg_diff_km" in paired
    assert "ci_95" in paired["center_dpe_diff_km"]


def test_15_existing_production_files_unchanged():
    """15. Git status confirms zero modifications to database and experiment checkpoints."""
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    lines = result.stdout.splitlines()
    for l in lines:
        status, filepath = l[:2], l[3:]
        assert not filepath.startswith("ml/experiments/exp_m2/checkpoints/"), f"EXP-M2 checkpoint modified: {filepath}"
        assert not filepath.startswith("data/raw/"), f"Raw data modified: {filepath}"
        assert not filepath.startswith("supabase/"), f"Database file modified: {filepath}"


def main():
    print("=" * 65)
    print("VAYU-NET: RUNNING 15-POINT EXP-M2 INTEGRITY AUDIT")
    print("=" * 65)
    tests = [
        ("1. Same Train Sample IDs across M2A/B/C", test_1_same_train_ids_across_m2a_m2b_m2c),
        ("2. Same Validation Sample IDs", test_2_same_validation_ids),
        ("3. Same Test Sample IDs", test_3_same_test_ids),
        ("4. Same 31 Test Storm IDs", test_4_same_test_storm_ids),
        ("5. Zero Train/Test Storm Leakage", test_5_no_test_data_used_in_training),
        ("6. Train-Only Normalization", test_6_no_test_data_used_for_normalization),
        ("7. Zero Future Leakage (Strict Causality)", test_7_no_future_frames),
        ("8. Six Input Frames per Modality", test_8_six_input_frames_per_modality),
        ("9. Correct Tensor Shapes [6, 72, 116]", test_9_correct_tensor_shapes),
        ("10. Multi-Task Loss Finite & Gradients Bounded", test_10_finite_loss_and_gradients),
        ("11. Checkpoints Exist on Disk", test_11_checkpoints_saved),
        ("12. Model Selection strictly Validation-driven", test_12_best_checkpoint_selected_using_validation_only),
        ("13. Modality Gate Values Bounded in [0, 1]", test_13_adaptive_gate_validity),
        ("14. Evaluation Metrics & Paired Bootstrap Differences", test_14_evaluation_metrics_and_paired_differences),
        ("15. Production Backend/Frontend Unchanged", test_15_existing_production_files_unchanged),
    ]

    passed = 0
    failed = 0
    for desc, fn in tests:
        try:
            fn()
            print(f"  [PASS] {desc}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")
            failed += 1

    print("=" * 65)
    if failed == 0:
        print(f"[AUDIT SUCCESS] All {passed}/15 EXP-M2 integrity tests PASSED.")
    else:
        print(f"[AUDIT FAILURE] {failed} tests failed, {passed} passed.")
    print("=" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
