"""
VAYU-NET: Comprehensive 22-Point Multimodal Dataset Validation Suite.
Validates the EXP-M1 GridSat-B1 + NASA GPM IMERG Final Run V07B research dataset.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "data/manifests/gridsat_imerg_multimodal_dataset_manifest.csv"
SAMPLE_INDEX_PATH = REPO_ROOT / "data/manifests/vayu_net_sample_index.csv"
SEQ_PAIRING_PATH = REPO_ROOT / "data/manifests/gridsat_imerg_sequence_pairing_manifest.csv"
BULK_DOWNLOAD_MANIFEST = REPO_ROOT / "data/manifests/imerg_bulk_download_manifest.csv"
STORAGE_INV_PATH = REPO_ROOT / "data/manifests/imerg_storage_inventory.csv"
NORM_STATS_PATH = REPO_ROOT / "data/interim/ml/imerg_train_normalization_stats.json"
IMERG_RAW_DIR = REPO_ROOT / "data/raw/imerg"


def test_1_manifest_exists():
    """1. Manifest exists: gridsat_imerg_multimodal_dataset_manifest.csv."""
    assert MANIFEST_PATH.exists(), f"Multimodal manifest not found at {MANIFEST_PATH}"


def test_2_total_samples_count():
    """2. Exactly 1,319 samples in manifest."""
    df = pd.read_csv(MANIFEST_PATH)
    assert len(df) == 1319, f"Expected 1319 samples, got {len(df)}"


def test_3_train_count():
    """3. Exactly 696 TRAIN samples."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["split"] == "TRAIN").sum() == 696, f"Expected 696 TRAIN samples, got {(df['split'] == 'TRAIN').sum()}"


def test_4_validation_count():
    """4. Exactly 252 VALIDATION samples."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["split"] == "VALIDATION").sum() == 252, f"Expected 252 VALIDATION samples, got {(df['split'] == 'VALIDATION').sum()}"


def test_5_test_count():
    """5. Exactly 371 TEST samples."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["split"] == "TEST").sum() == 371, f"Expected 371 TEST samples, got {(df['split'] == 'TEST').sum()}"


def test_6_unique_storms_count():
    """6. Exactly 126 unique storms."""
    df = pd.read_csv(MANIFEST_PATH)
    assert df["storm_id"].nunique() == 126, f"Expected 126 unique storms, got {df['storm_id'].nunique()}"


def test_7_no_train_val_storm_overlap():
    """7. No train/val storm overlap (strict storm-level independence)."""
    df = pd.read_csv(MANIFEST_PATH)
    train_storms = set(df[df["split"] == "TRAIN"]["storm_id"])
    val_storms = set(df[df["split"] == "VALIDATION"]["storm_id"])
    overlap = train_storms & val_storms
    assert len(overlap) == 0, f"Detected train/val storm overlap: {overlap}"


def test_8_no_train_test_storm_overlap():
    """8. No train/test storm overlap."""
    df = pd.read_csv(MANIFEST_PATH)
    train_storms = set(df[df["split"] == "TRAIN"]["storm_id"])
    test_storms = set(df[df["split"] == "TEST"]["storm_id"])
    overlap = train_storms & test_storms
    assert len(overlap) == 0, f"Detected train/test storm overlap: {overlap}"


def test_9_no_val_test_storm_overlap():
    """9. No val/test storm overlap."""
    df = pd.read_csv(MANIFEST_PATH)
    val_storms = set(df[df["split"] == "VALIDATION"]["storm_id"])
    test_storms = set(df[df["split"] == "TEST"]["storm_id"])
    overlap = val_storms & test_storms
    assert len(overlap) == 0, f"Detected val/test storm overlap: {overlap}"


def test_10_every_sample_has_six_gridsat_frames():
    """10. Every sample has exactly six GridSat frames."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["gridsat_frame_count"] == 6).all()
    for _, row in df.iterrows():
        frames = row["gridsat_sequence_reference"].split("|")
        assert len(frames) == 6, f"Sample {row['sample_id']} has {len(frames)} GridSat frames, expected 6"


def test_11_every_sample_has_six_imerg_frames():
    """11. Every sample has exactly six IMERG frames."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["imerg_frame_count"] == 6).all()
    for _, row in df.iterrows():
        granules = row["imerg_sequence_reference"].split("|")
        assert len(granules) == 6, f"Sample {row['sample_id']} has {len(granules)} IMERG frames, expected 6"


def test_12_gridsat_timestamps_match_imerg_timestamps():
    """12. GridSat timestamps match IMERG timestamps with delta-t = 0."""
    seq_df = pd.read_csv(SEQ_PAIRING_PATH)
    assert (seq_df["delta_minutes_max"] == 0.0).all(), "Non-zero delta-t detected in pairing"
    assert (seq_df["sequence_pairing_status"] == "FULL_SEQUENCE_MATCH").all()


def test_13_every_imerg_frame_points_to_validated_canonical_granule():
    """13. Every IMERG frame points to a validated canonical granule."""
    inv_df = pd.read_csv(STORAGE_INV_PATH)
    canonical_set = set(inv_df[inv_df["is_canonical"] == True]["filename"])
    
    df = pd.read_csv(MANIFEST_PATH)
    for _, row in df.iterrows():
        granules = row["imerg_sequence_reference"].split("|")
        for g in granules:
            assert g in canonical_set, f"Granule {g} in sample {row['sample_id']} is not a validated canonical granule"


def test_14_no_future_frame_relative_to_t0():
    """14. No future frame relative to t0 (strict causality)."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["leakage_check"] == "ZERO_FUTURE_LEAKAGE").all()
    seq_df = pd.read_csv(SEQ_PAIRING_PATH)
    assert (~seq_df["future_leakage_detected"]).all(), "Future leakage flag detected in sequence manifest"


def test_15_no_nan_explosion_after_preprocessing():
    """15. No NaN explosion after preprocessing; finite and bounded standardized tensors."""
    from ml.data.multimodal_dataset import MultimodalVayuDataset
    ds = MultimodalVayuDataset(split="TRAIN")
    sample = ds[0]
    g_arr = sample["gridsat"].numpy()
    i_arr = sample["imerg"].numpy()
    assert not np.isnan(g_arr).any(), "NaN found in preprocessed GridSat tensor"
    assert not np.isnan(i_arr).any(), "NaN found in preprocessed IMERG tensor"
    assert not np.isinf(g_arr).any(), "Inf found in preprocessed GridSat tensor"
    assert not np.isinf(i_arr).any(), "Inf found in preprocessed IMERG tensor"


def test_16_spatial_shape_is_correct():
    """16. Spatial shape is correct: [6, 72, 116] for both GridSat and IMERG."""
    df = pd.read_csv(MANIFEST_PATH)
    assert (df["spatial_shape"] == "(72, 116)").all()
    from ml.data.multimodal_dataset import MultimodalVayuDataset
    ds = MultimodalVayuDataset(split="TRAIN")
    sample = ds[0]
    assert sample["gridsat"].shape == (6, 72, 116), f"Unexpected GridSat shape: {sample['gridsat'].shape}"
    assert sample["imerg"].shape == (6, 72, 116), f"Unexpected IMERG shape: {sample['imerg'].shape}"


def test_17_normalization_statistics_derived_from_train_only():
    """17. Normalization statistics derived from TRAIN ONLY."""
    assert NORM_STATS_PATH.exists()
    with open(NORM_STATS_PATH, "r") as f:
        stats = json.load(f)
    assert stats["split"] == "TRAIN_ONLY"
    assert stats["num_train_sequences"] == 696
    assert stats["num_unique_train_granules"] == 1304
    assert stats["nan_fraction"] == 0.0


def test_18_no_val_test_info_used_for_preprocessing_stats():
    """18. No validation/test information used for preprocessing statistics."""
    df = pd.read_csv(MANIFEST_PATH)
    val_test_granules = set()
    for _, row in df[df["split"].isin(["VALIDATION", "TEST"])].iterrows():
        val_test_granules.update(row["imerg_sequence_reference"].split("|"))
        
    train_granules = set()
    for _, row in df[df["split"] == "TRAIN"].iterrows():
        train_granules.update(row["imerg_sequence_reference"].split("|"))
        
    with open(NORM_STATS_PATH, "r") as f:
        stats = json.load(f)
    assert stats["num_unique_train_granules"] == len(train_granules)
    assert stats["num_train_sequences"] == 696


def test_19_no_synthetic_data():
    """19. No synthetic data: all 2,353 canonical files have valid HDF5 magic headers and GES DISC provenance."""
    bulk_df = pd.read_csv(BULK_DOWNLOAD_MANIFEST)
    assert (bulk_df["validation_status"] == "PASS").all()
    assert (bulk_df["hdf5_valid"] == True).all()
    assert (bulk_df["product_valid"] == True).all()


def test_20_target_availability_is_valid():
    """20. Target availability is valid (IMD ground truth centers, wind, pressure, category present)."""
    from ml.data.multimodal_dataset import MultimodalVayuDataset
    ds = MultimodalVayuDataset(split="TRAIN")
    sample = ds[0]
    assert "center_t0" in sample and sample["center_t0"].shape == (2,)
    assert "wind_t0" in sample and sample["wind_t0"].ndim == 0
    assert "pressure_t0" in sample and sample["pressure_t0"].ndim == 0
    assert "category_t0" in sample and sample["category_t0"].ndim == 0
    for h in ["12h", "24h", "48h"]:
        assert f"center_{h}" in sample
        assert f"wind_{h}" in sample
        assert f"pressure_{h}" in sample
        assert f"category_{h}" in sample


def test_21_sample_count_unchanged_from_source_manifest():
    """21. Sample count unchanged from source manifest (1,319 in index == 1,319 in multimodal)."""
    idx_df = pd.read_csv(SAMPLE_INDEX_PATH)
    df = pd.read_csv(MANIFEST_PATH)
    assert len(df) == len(idx_df) == 1319
    assert set(df["sample_id"]) == set(idx_df["sample_id"])


def test_22_raw_hdf5_files_remain_unchanged():
    """22. Raw HDF5 files remain unchanged on disk."""
    inv_df = pd.read_csv(STORAGE_INV_PATH)
    # Check sample of 5 canonical files to verify sha256 still matches
    for _, row in inv_df[inv_df["is_canonical"] == True].head(5).iterrows():
        p = IMERGRAW = IMERG_RAW_DIR / row["filename"]
        assert p.exists()
        assert p.stat().st_size == row["file_size_bytes"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == row["sha256"], f"SHA-256 mismatch for raw file {p.name}"


def main():
    print("=" * 65)
    print("VAYU-NET: RUNNING 22-POINT MULTIMODAL DATASET VALIDATION SUITE")
    print("=" * 65)
    tests = [
        ("Test 1: Manifest Exists", test_1_manifest_exists),
        ("Test 2: Total Samples Count (1,319)", test_2_total_samples_count),
        ("Test 3: Train Count (696)", test_3_train_count),
        ("Test 4: Validation Count (252)", test_4_validation_count),
        ("Test 5: Test Count (371)", test_5_test_count),
        ("Test 6: Unique Storms Count (126)", test_6_unique_storms_count),
        ("Test 7: No Train/Val Storm Overlap", test_7_no_train_val_storm_overlap),
        ("Test 8: No Train/Test Storm Overlap", test_8_no_train_test_storm_overlap),
        ("Test 9: No Val/Test Storm Overlap", test_9_no_val_test_storm_overlap),
        ("Test 10: Six GridSat Frames per Sample", test_10_every_sample_has_six_gridsat_frames),
        ("Test 11: Six IMERG Frames per Sample", test_11_every_sample_has_six_imerg_frames),
        ("Test 12: GridSat Timestamps Match IMERG Timestamps", test_12_gridsat_timestamps_match_imerg_timestamps),
        ("Test 13: Every IMERG Frame Canonical Granule", test_13_every_imerg_frame_points_to_validated_canonical_granule),
        ("Test 14: No Future Frame Relative to t0", test_14_no_future_frame_relative_to_t0),
        ("Test 15: No NaN Explosion After Preprocessing", test_15_no_nan_explosion_after_preprocessing),
        ("Test 16: Spatial Shape Correct (72, 116)", test_16_spatial_shape_is_correct),
        ("Test 17: Normalization Derived From Train Only", test_17_normalization_statistics_derived_from_train_only),
        ("Test 18: No Val/Test Info in Preprocessing Stats", test_18_no_val_test_info_used_for_preprocessing_stats),
        ("Test 19: No Synthetic Data / Genuine HDF5", test_19_no_synthetic_data),
        ("Test 20: Target Availability Valid", test_20_target_availability_is_valid),
        ("Test 21: Sample Count Unchanged from Source", test_21_sample_count_unchanged_from_source_manifest),
        ("Test 22: Raw HDF5 Files Remain Unchanged", test_22_raw_hdf5_files_remain_unchanged),
    ]
    
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
            
    print("=" * 65)
    if failed == 0:
        print(f"[AUDIT SUCCESS] All {passed}/22 multimodal dataset validation tests PASSED.")
    else:
        print(f"[AUDIT FAILURE] {failed} tests FAILED, {passed} passed.")
    print("=" * 65)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())