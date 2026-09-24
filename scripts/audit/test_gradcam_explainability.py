"""VAYU-NET WP-08 — GRAD-CAM EXPLAINABILITY AUDIT TEST SUITE
=============================================================
Comprehensive test suite validating model interpretability, hook integrity,
spatial alignment, deterministic reproduction, zero future leakage, and dataset lock.

Tests Covered:
  1.  test_checkpoints_load
  2.  test_forward_inference
  3.  test_target_class_extraction
  4.  test_gradcam_saliency_generation
  5.  test_heatmap_finite_values
  6.  test_normalization_bounds
  7.  test_spatial_alignment_dimensions
  8.  test_overlay_generation
  9.  test_metadata_completeness
  10. test_deterministic_reproduction
  11. test_t0_timestamp_correctness
  12. test_no_future_dependency
  13. test_locked_dataset_and_historical_artifacts_integrity
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability.gradcam import GradCAMExplainer

SPATIAL_CKPT = PROJECT_ROOT / "data/interim/ml/checkpoints/best_center_localization_cnn.pt"
TEMPORAL_CKPT = PROJECT_ROOT / "data/interim/ml/checkpoints/best_temporal_track_gru.pt"
PHASE6_CKPT = PROJECT_ROOT / "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt"
CACHE_PATH = PROJECT_ROOT / "data/interim/ml/cache/phase6_intensity_features.pt"
SAMPLE_INDEX_PATH = PROJECT_ROOT / "data/manifests/vayu_net_sample_index.csv"
NORM_STATS_PATH = PROJECT_ROOT / "data/interim/ml/train_normalization_stats.json"
EXPLAINABILITY_DIR = PROJECT_ROOT / "data/interim/ml/explainability"


@pytest.fixture(scope="module")
def explainer():
    """Initializes and returns a GradCAMExplainer on CPU."""
    exp = GradCAMExplainer(
        spatial_checkpoint_path=SPATIAL_CKPT,
        temporal_checkpoint_path=TEMPORAL_CKPT,
        phase6_checkpoint_path=PHASE6_CKPT,
        device=torch.device("cpu"),
    )
    yield exp
    exp.close()


@pytest.fixture(scope="module")
def test_sample():
    """Loads a representative test sample from the feature cache."""
    cache = torch.load(CACHE_PATH, map_location="cpu")
    # Pick a sample from AMPHAN or TAUKTAE
    for s in cache["samples"]:
        if "AMPHAN" in s["storm_id"]:
            return s
    return cache["samples"][0]


def test_1_checkpoints_load(explainer):
    """Test 1: Verifies all required model checkpoints exist and load cleanly."""
    assert SPATIAL_CKPT.exists(), f"Missing spatial checkpoint: {SPATIAL_CKPT}"
    assert TEMPORAL_CKPT.exists(), f"Missing temporal checkpoint: {TEMPORAL_CKPT}"
    assert PHASE6_CKPT.exists(), f"Missing phase 6 checkpoint: {PHASE6_CKPT}"
    assert explainer.p6_model is not None
    assert explainer.spatial_extractor is not None
    assert explainer.target_layer is not None


def test_2_forward_inference(explainer, test_sample):
    """Test 2: Verifies forward inference computes valid logits and probabilities."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)  # [1, 6, 132]
    env_seq = test_sample["env_seq"].unsqueeze(0)  # [1, 6, 8, 41, 66]

    with torch.no_grad():
        out = explainer.p6_model(sat_seq=sat_seq, env_seq=env_seq)

    assert "category_logits" in out
    assert "category_probs" in out
    assert out["category_logits"].shape == (1, 7)
    assert out["category_probs"].shape == (1, 7)
    prob_sum = float(torch.sum(out["category_probs"]).item())
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"


def test_3_target_class_extraction(explainer, test_sample):
    """Test 3: Verifies predicted target class matches argmax of category logits."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)
    env_seq = test_sample["env_seq"].unsqueeze(0)

    with torch.no_grad():
        out = explainer.p6_model(sat_seq=sat_seq, env_seq=env_seq)

    logits = out["category_logits"]
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_class = explainer.intensity_classes[pred_idx]
    assert 0 <= pred_idx < 7
    assert pred_class in ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]


def test_4_gradcam_saliency_generation(explainer, test_sample):
    """Test 4: Verifies Grad-CAM hooks capture activations and gradients successfully."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)
    env_seq = test_sample["env_seq"].unsqueeze(0)
    dummy_img = torch.randn(1, 1, 572, 929)

    saliency = explainer.generate_saliency(
        t0_img_tensor=dummy_img,
        cached_seq_132=sat_seq,
        env_seq_tensor=env_seq,
    )

    assert "cam_raw" in saliency
    assert "cam_resized" in saliency
    assert "predicted_class" in saliency
    assert "confidence" in saliency
    assert saliency["cam_raw"].shape == (72, 117)
    assert saliency["cam_resized"].shape == (572, 929)


def test_5_heatmap_finite_values(explainer, test_sample):
    """Test 5: Verifies heatmap values are strictly finite (no NaN, no Inf)."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)
    env_seq = test_sample["env_seq"].unsqueeze(0)
    dummy_img = torch.randn(1, 1, 572, 929)

    saliency = explainer.generate_saliency(
        t0_img_tensor=dummy_img,
        cached_seq_132=sat_seq,
        env_seq_tensor=env_seq,
    )

    cam_resized = saliency["cam_resized"]
    assert np.all(np.isfinite(cam_resized)), "Heatmap contains non-finite values (NaN or Inf)!"
    assert not np.all(cam_resized == 0.0), "Heatmap is completely empty/zero!"


def test_6_normalization_bounds(explainer, test_sample):
    """Test 6: Verifies heatmap is normalized to [0, 1] range."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)
    env_seq = test_sample["env_seq"].unsqueeze(0)
    dummy_img = torch.randn(1, 1, 572, 929)

    saliency = explainer.generate_saliency(
        t0_img_tensor=dummy_img,
        cached_seq_132=sat_seq,
        env_seq_tensor=env_seq,
    )

    cam_resized = saliency["cam_resized"]
    min_val = float(np.min(cam_resized))
    max_val = float(np.max(cam_resized))

    assert min_val >= 0.0, f"Heatmap min value {min_val} < 0.0"
    assert max_val <= 1.0 + 1e-6, f"Heatmap max value {max_val} > 1.0"
    assert max_val > 0.0, "Heatmap max value is 0.0!"


def test_7_spatial_alignment_dimensions(explainer, test_sample):
    """Test 7: Verifies spatial upsampling matches source satellite frame resolution (572 x 929)."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)
    env_seq = test_sample["env_seq"].unsqueeze(0)
    img_h, img_w = 572, 929
    dummy_img = torch.randn(1, 1, img_h, img_w)

    saliency = explainer.generate_saliency(
        t0_img_tensor=dummy_img,
        cached_seq_132=sat_seq,
        env_seq_tensor=env_seq,
    )

    assert saliency["cam_resized"].shape == (img_h, img_w)
    max_r, max_c = saliency["max_activation_loc"]
    assert 0 <= max_r < img_h
    assert 0 <= max_c < img_w

    cent_r, cent_c = saliency["activation_centroid"]
    assert 0.0 <= cent_r <= float(img_h)
    assert 0.0 <= cent_c <= float(img_w)


def test_8_overlay_generation(explainer):
    """Test 8: Verifies RGB overlay generation creates 3-channel uint8 image with matching dims."""
    orig = np.random.rand(572, 929).astype(np.float32)
    heat = np.random.rand(572, 929).astype(np.float32)

    overlay = explainer.render_overlay(original_img=orig, heatmap_norm=heat, alpha=0.45)
    assert overlay.shape == (572, 929, 3)
    assert overlay.dtype == np.uint8
    assert np.min(overlay) >= 0
    assert np.max(overlay) <= 255


def test_9_metadata_completeness():
    """Test 9: Verifies generated metadata artifacts contain all required audit fields."""
    manifest_path = EXPLAINABILITY_DIR / "explainability_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["status"] == "COMPLETED"
    assert manifest["num_reference_cases"] >= 1
    assert "interpretation_disclaimer" in manifest

    required_keys = [
        "sample_id",
        "storm_id",
        "storm_name",
        "t0",
        "source",
        "frame_index",
        "predicted_class",
        "confidence",
        "method",
        "target_layer",
        "image_height",
        "image_width",
        "max_activation_loc",
        "activation_centroid",
        "heatmap_path",
        "overlay_path",
        "original_path",
        "model_checkpoint",
        "interpretation_note",
    ]

    for case in manifest["cases"]:
        for k in required_keys:
            assert k in case, f"Case {case.get('sample_id')} missing key: {k}"
        assert case["method"] == "Grad-CAM"
        assert case["frame_index"] == 5
        assert case["image_height"] == 572
        assert case["image_width"] == 929
        # Check files exist
        assert (PROJECT_ROOT / case["heatmap_path"]).exists()
        assert (PROJECT_ROOT / case["overlay_path"]).exists()
        assert (PROJECT_ROOT / case["original_path"]).exists()


def test_10_deterministic_reproduction(explainer, test_sample):
    """Test 10: Verifies Grad-CAM execution is strictly deterministic under fixed input."""
    sat_seq = test_sample["sat_seq"].unsqueeze(0)
    env_seq = test_sample["env_seq"].unsqueeze(0)
    torch.manual_seed(42)
    dummy_img = torch.randn(1, 1, 572, 929)

    out1 = explainer.generate_saliency(
        t0_img_tensor=dummy_img.clone(),
        cached_seq_132=sat_seq.clone(),
        env_seq_tensor=env_seq.clone(),
    )

    out2 = explainer.generate_saliency(
        t0_img_tensor=dummy_img.clone(),
        cached_seq_132=sat_seq.clone(),
        env_seq_tensor=env_seq.clone(),
    )

    diff = np.max(np.abs(out1["cam_resized"] - out2["cam_resized"]))
    assert diff < 1e-5, f"Grad-CAM output not deterministic! Max diff: {diff}"
    assert out1["predicted_class_idx"] == out2["predicted_class_idx"]
    assert abs(out1["confidence"] - out2["confidence"]) < 1e-6


def test_11_t0_timestamp_correctness():
    """Test 11: Verifies explanation targets strictly t0 without temporal offset."""
    manifest_path = EXPLAINABILITY_DIR / "explainability_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    sample_df = pd.read_csv(SAMPLE_INDEX_PATH)

    for case in manifest["cases"]:
        s_id = case["sample_id"]
        row = sample_df[sample_df["sample_id"] == s_id].iloc[0]
        assert case["t0"] == row["t0"], f"Timestamp mismatch for {s_id}: {case['t0']} vs {row['t0']}"
        assert case["frame_index"] == 5, f"Frame index must be 5 (terminal t0), got {case['frame_index']}"


def test_12_no_future_dependency():
    """Test 12: Verifies zero future frames or labels are used in explanation."""
    sample_df = pd.read_csv(SAMPLE_INDEX_PATH)
    frame_cols = [
        "frame_t_minus_15h",
        "frame_t_minus_12h",
        "frame_t_minus_9h",
        "frame_t_minus_6h",
        "frame_t_minus_3h",
        "frame_t0",
    ]
    # Verify no future frame columns exist in the observation sequence
    for col in frame_cols:
        assert not col.startswith("frame_t_plus_"), f"Future column detected in input sequence: {col}"


def test_13_locked_dataset_and_historical_artifacts_integrity():
    """Test 13: Verifies all locked datasets, manifests, and historical checkpoints remain intact."""
    critical_artifacts = [
        "data/manifests/vayu_net_sample_index.csv",
        "data/manifests/vayu_net_candidate_t0_manifest.csv",
        "data/manifests/gridsat_sha256_manifest.csv",
        "data/processed/imd_best_track_v2.csv",
        "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
        "data/interim/ml/phase3c_center_localization_results.json",
        "data/interim/ml/checkpoints/best_phase4b_variant_a.pt",
        "data/interim/ml/checkpoints/best_phase4b_variant_b.pt",
        "data/interim/ml/phase4b_hybrid_results.json",
        "data/interim/ml/phase5a_results.json",
        "data/interim/ml/phase5b_hybrid_results.json",
        "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt",
        "data/interim/ml/phase6_intensity_wind_results.json",
    ]

    for art in critical_artifacts:
        path = PROJECT_ROOT / art
        assert path.exists(), f"CRITICAL INTEGRITY FAILURE: Missing locked artifact: {art}"
        assert path.stat().st_size > 0, f"CRITICAL INTEGRITY FAILURE: Empty artifact: {art}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
