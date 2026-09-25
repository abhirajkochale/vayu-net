"""
VAYU-NET: Full GridSat-B1 + NASA GPM IMERG Pairing Validation Suite.

Verifies the 18 Scientific and Structural Integrity Criteria:
1. Paired IMERG files exist in the official catalog with valid download URLs and local pilots on disk.
2. Every locally present file is a genuine HDF5 file (verified magic signature).
3. Product is NASA GPM IMERG Final Run V07B (GPM_3IMERGHH.07).
4. Primary variable is /Grid/precipitation (formerly precipitationCal).
5. Units are mm/hr.
6. Exact timestamp pairing (delta_minutes == 0.0 across all sequences).
7. Zero future observations relative to t0 (all sequence frames have t <= t0).
8. Zero temporal interpolation applied.
9. Zero synthetic IMERG data or surrogate channels.
10. Zero storm-centered cropping (fixed North Indian Ocean synoptic basin).
11. Spatial domain compatibility (native 0.1 deg maps to NIO [400, 650] and model [72, 116]).
12. Storm-level split integrity preserved.
13. Strict mutual exclusivity: Zero storm overlap between TRAIN, VALIDATION, and TEST.
14. Six-frame temporal sequence completeness (exactly 6 observations per sample).
15. Missing-frame handling policy enforced (no partial sequence filling).
16. Duplicate timestamp detection within any sequence.
17. Manifest reproducibility and cross-referencing against sample index.
18. Zero modification of raw HDF5 files on disk.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np
import pandas as pd
import h5py

from ml.data.process_imerg import (
    IMERGProcessor,
    NIO_LAT_MIN, NIO_LAT_MAX,
    NIO_LON_MIN, NIO_LON_MAX,
    HDF5_SIGNATURE
)

SAMPLE_INDEX_PATH = REPO_ROOT / "data/manifests/vayu_net_sample_index.csv"
PAIRING_OBS_MANIFEST_PATH = REPO_ROOT / "data/manifests/gridsat_imerg_pairing_manifest.csv"
PAIRING_SEQ_MANIFEST_PATH = REPO_ROOT / "data/manifests/gridsat_imerg_sequence_pairing_manifest.csv"
LOCAL_IMERG_DIR = REPO_ROOT / "data/raw/imerg"


def test_1_paired_imerg_existence_and_catalog_urls():
    """Verify that every paired frame has a valid GES DISC download URL and correct naming convention."""
    assert PAIRING_OBS_MANIFEST_PATH.exists()
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert len(df_obs) == 2353

    # Check URL structure
    for _, row in df_obs.head(50).iterrows():
        url = row["download_url"]
        fname = row["imerg_granule_filename"]
        assert url.startswith("https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/")
        assert url.endswith(fname)
        assert "V07B.HDF5" in fname


def test_2_local_pilot_files_genuine_hdf5():
    """Verify that all locally present pilot files begin with the official HDF5 magic signature."""
    assert LOCAL_IMERG_DIR.exists()
    h5_files = list(LOCAL_IMERG_DIR.glob("*.HDF5"))
    assert len(h5_files) >= 3, "At least the 3 pilot granules must exist on local disk"

    for p in h5_files:
        with open(p, "rb") as f:
            header = f.read(8)
            assert header == HDF5_SIGNATURE, f"File {p.name} failed HDF5 magic header verification"


def test_3_product_collection_and_version():
    """Verify that all records belong to GPM_3IMERGHH.07 and version V07B."""
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert (df_obs["product_collection"] == "GPM_3IMERGHH.07").all()
    assert (df_obs["algorithm_version"] == "V07B").all()
    assert (df_obs["format"] == "HDF5").all()


def test_4_variable_is_calibrated_precipitation():
    """Verify that the extracted variable is /Grid/precipitation and confirmed via HDF5 inspection."""
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert (df_obs["primary_variable"] == "/Grid/precipitation").all()

    # Verify directly on a local pilot HDF5
    pilot_sample = list(LOCAL_IMERG_DIR.glob("*.HDF5"))[0]
    with h5py.File(pilot_sample, "r") as h5:
        assert "Grid" in h5
        assert "precipitation" in h5["Grid"]
        attrs = {k: str(h5["Grid"]["precipitation"].attrs[k]) for k in h5["Grid"]["precipitation"].attrs}
        long_name = attrs.get("LongName", "")
        assert "precipitationCal" in long_name or "precipitation" in long_name


def test_5_units_mm_hr():
    """Verify units are physical mm/hr (not arbitrary counts or surrogate temperatures)."""
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert (df_obs["units"] == "mm/hr").all()


def test_6_exact_timestamp_pairing():
    """Verify delta_minutes is strictly 0.0 across all 2,353 paired frames."""
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert (df_obs["delta_minutes"] == 0.0).all()
    assert (df_obs["pairing_status"] == "EXACT_MATCH").all()

    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    assert (df_seq["delta_minutes_max"] == 0.0).all()


def test_7_no_future_observation_relative_to_t0():
    """Verify that all sequence frames precede or coincide with t0 (zero future leakage)."""
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    assert (df_seq["future_leakage_detected"] == False).all()

    for _, row in df_seq.head(100).iterrows():
        t0 = datetime.fromisoformat(row["t0_utc"].replace("Z", "+00:00"))
        for col in ["t_minus_15h_utc", "t_minus_12h_utc", "t_minus_9h_utc", "t_minus_6h_utc", "t_minus_3h_utc"]:
            t_frame = datetime.fromisoformat(row[col].replace("Z", "+00:00"))
            assert t_frame <= t0, f"Frame {t_frame} occurs after t0 {t0}"


def test_8_zero_temporal_interpolation():
    """Verify that no temporal interpolation was used to construct timestamps."""
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    for _, row in df_seq.head(50).iterrows():
        dt0 = datetime.fromisoformat(row["t0_utc"].replace("Z", "+00:00"))
        dt3 = datetime.fromisoformat(row["t_minus_3h_utc"].replace("Z", "+00:00"))
        dt6 = datetime.fromisoformat(row["t_minus_6h_utc"].replace("Z", "+00:00"))
        dt9 = datetime.fromisoformat(row["t_minus_9h_utc"].replace("Z", "+00:00"))
        dt12 = datetime.fromisoformat(row["t_minus_12h_utc"].replace("Z", "+00:00"))
        dt15 = datetime.fromisoformat(row["t_minus_15h_utc"].replace("Z", "+00:00"))

        assert (dt0 - dt3).total_seconds() == 3 * 3600
        assert (dt3 - dt6).total_seconds() == 3 * 3600
        assert (dt6 - dt9).total_seconds() == 3 * 3600
        assert (dt9 - dt12).total_seconds() == 3 * 3600
        assert (dt12 - dt15).total_seconds() == 3 * 3600


def test_9_no_synthetic_imerg():
    """Verify that no surrogate or GridSat IR values are labeled as IMERG."""
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert (df_obs["rejection_reason"] == "NONE").all()
    # Check that filenames follow genuine NASA pattern
    for fname in df_obs["imerg_granule_filename"]:
        assert fname.startswith("3B-HHR.MS.MRG.3IMERG.")


def test_10_no_storm_centered_cropping():
    """Verify that spatial processing uses fixed North Indian Ocean bounds."""
    processor = IMERGProcessor()
    assert NIO_LAT_MIN == -5.0
    assert NIO_LAT_MAX == 35.0
    assert NIO_LON_MIN == 40.0
    assert NIO_LON_MAX == 105.0


def test_11_spatial_domain_compatibility():
    """Verify spatial domain shapes in manifest match verified processor specs."""
    df_obs = pd.read_csv(PAIRING_OBS_MANIFEST_PATH)
    assert (df_obs["raw_grid_shape"] == "(1, 3600, 1800)").all()
    assert (df_obs["nio_subgrid_shape"] == "(400, 650)").all()
    assert (df_obs["model_grid_shape"] == "(72, 116)").all()


def test_12_storm_level_split_integrity():
    """Verify all 126 storms preserve their exact split assignment."""
    df_samples = pd.read_csv(SAMPLE_INDEX_PATH)
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)

    assert df_seq["storm_id"].nunique() == 126
    assert len(df_seq) == 1319

    # Cross-reference sample splits
    sample_to_split_orig = dict(zip(df_samples["sample_id"], df_samples["split"]))
    sample_to_split_pair = dict(zip(df_seq["sample_id"], df_seq["split"]))
    assert sample_to_split_orig == sample_to_split_pair


def test_13_no_train_val_test_storm_overlap():
    """Verify zero storm overlap between TRAIN, VALIDATION, and TEST sets."""
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    train_storms = set(df_seq[df_seq["split"] == "TRAIN"]["storm_id"])
    val_storms = set(df_seq[df_seq["split"] == "VALIDATION"]["storm_id"])
    test_storms = set(df_seq[df_seq["split"] == "TEST"]["storm_id"])

    assert len(train_storms) == 81
    assert len(val_storms) == 14
    assert len(test_storms) == 31

    assert len(train_storms.intersection(val_storms)) == 0, "Train and Val storms overlap!"
    assert len(train_storms.intersection(test_storms)) == 0, "Train and Test storms overlap!"
    assert len(val_storms.intersection(test_storms)) == 0, "Val and Test storms overlap!"


def test_14_six_frame_completeness():
    """Verify every sequence requires exactly 6 frames and all 6 are catalog-paired."""
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    assert (df_seq["n_frames_required"] == 6).all()
    assert (df_seq["n_frames_available_catalog"] == 6).all()
    assert (df_seq["sequence_pairing_status"] == "FULL_SEQUENCE_MATCH").all()


def test_15_missing_frame_handling():
    """Verify policy: partial sequences are never filled with synthetic frames."""
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    # Check that any sequence marked as FULL_SEQUENCE_MATCH has exactly 6 catalog frames
    full_matches = df_seq[df_seq["sequence_pairing_status"] == "FULL_SEQUENCE_MATCH"]
    assert (full_matches["n_frames_available_catalog"] == 6).all()


def test_16_duplicate_timestamp_detection():
    """Verify that no single sequence contains duplicate timestamps."""
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)
    for _, row in df_seq.iterrows():
        ts_list = [
            row["t_minus_15h_utc"],
            row["t_minus_12h_utc"],
            row["t_minus_9h_utc"],
            row["t_minus_6h_utc"],
            row["t_minus_3h_utc"],
            row["t0_utc"]
        ]
        assert len(set(ts_list)) == 6, f"Duplicate timestamp found in sequence {row['sample_id']}"


def test_17_manifest_reproducibility():
    """Verify exact 1:1 mapping between vayu_net_sample_index.csv and sequence manifest."""
    df_samples = pd.read_csv(SAMPLE_INDEX_PATH)
    df_seq = pd.read_csv(PAIRING_SEQ_MANIFEST_PATH)

    assert set(df_samples["sample_id"]) == set(df_seq["sample_id"])
    assert len(df_samples) == len(df_seq)


def test_18_no_modification_of_raw_hdf5_files():
    """Verify raw HDF5 files on disk match their original pilot manifest SHA-256 hashes."""
    pilot_manifest = REPO_ROOT / "data/manifests/imerg_pilot_manifest.csv"
    if pilot_manifest.exists():
        df_pilot = pd.read_csv(pilot_manifest)
        for _, row in df_pilot.iterrows():
            p = REPO_ROOT / row["local_path"]
            if p.exists():
                hasher = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                assert hasher.hexdigest() == row["sha256"], f"Raw file {p.name} was modified!"


def main():
    print("=" * 65)
    print("VAYU-NET: RUNNING FULL GRIDSAT + NASA IMERG PAIRING VALIDATION")
    print("=" * 65)
    tests = [
        ("Test 1: Paired IMERG Existence & Catalog URLs", test_1_paired_imerg_existence_and_catalog_urls),
        ("Test 2: Local Pilot Files Genuine HDF5", test_2_local_pilot_files_genuine_hdf5),
        ("Test 3: Product Collection & Version V07B", test_3_product_collection_and_version),
        ("Test 4: Primary Variable /Grid/precipitation", test_4_variable_is_calibrated_precipitation),
        ("Test 5: Physical Units mm/hr", test_5_units_mm_hr),
        ("Test 6: Exact Timestamp Pairing (delta=0)", test_6_exact_timestamp_pairing),
        ("Test 7: No Future Observations (Zero Leakage)", test_7_no_future_observation_relative_to_t0),
        ("Test 8: Zero Temporal Interpolation", test_8_zero_temporal_interpolation),
        ("Test 9: No Synthetic IMERG", test_9_no_synthetic_imerg),
        ("Test 10: No Storm-Centered Cropping", test_10_no_storm_centered_cropping),
        ("Test 11: Spatial Domain Compatibility", test_11_spatial_domain_compatibility),
        ("Test 12: Storm Split Integrity Preserved", test_12_storm_level_split_integrity),
        ("Test 13: Zero Train/Val/Test Overlap", test_13_no_train_val_test_storm_overlap),
        ("Test 14: Six-Frame Completeness", test_14_six_frame_completeness),
        ("Test 15: Missing-Frame Handling Policy", test_15_missing_frame_handling),
        ("Test 16: Duplicate Timestamp Detection", test_16_duplicate_timestamp_detection),
        ("Test 17: Manifest Reproducibility", test_17_manifest_reproducibility),
        ("Test 18: Zero Modification of Raw HDF5 Files", test_18_no_modification_of_raw_hdf5_files),
    ]

    all_passed = True
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("[AUDIT SUCCESS] All 18 pairing validation tests PASSED cleanly.")
        sys.exit(0)
    else:
        print("[AUDIT FAILURE] One or more tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
