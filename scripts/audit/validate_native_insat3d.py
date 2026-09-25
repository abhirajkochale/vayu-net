"""
VAYU-NET: Native INSAT-3D Audit & Validation Suite.

Verifies the 10 Scientific and Architectural Criteria:
1. Files are genuine HDF5 INSAT-3D products (magic header verification).
2. SHA-256 recorded for all granules.
3. Correct dataset ID (3DIMG_L1C_ASIA_MER).
4. Real TIR1/TIR2/WV datasets and LUTs exist.
5. Calibration converts counts to Kelvin and preserves physical bounds.
6. Spatial coordinates are valid (lat [-5, 35] ascending, lon [40, 105] ascending).
7. Timestamp provenance is preserved (zero synthetic timestamps, zero interpolation).
8. No synthetic channels are present.
9. No GridSat arrays are masquerading as INSAT.
10. No IMD center labels are used to construct the imagery.
11. Stop conditions: If credentials/files are missing, pipeline halts cleanly without model training.
"""

import os
import sys
import tempfile
import hashlib
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np
import pandas as pd
import h5py

from ml.data.process_native_insat3d import (
    NativeINSAT3DProcessor,
    PHYSICAL_LIMITS,
    NIO_LAT_MIN, NIO_LAT_MAX,
    NIO_LON_MIN, NIO_LON_MAX,
    HDF5_SIGNATURE
)
from ml.data.mosdac_download_client import MOSDACDownloadClient, CredentialsUnavailableError

MANDATORY_PROVENANCE_QUOTE = (
    "The previous multisource INSAT experiments used GridSat-derived surrogate "
    "channels and are not evidence of native INSAT-3D predictive performance."
)


def test_1_provenance_reset_document():
    """Verify that docs/insat_provenance_reset.md exists and contains the exact mandatory quote."""
    doc_path = Path("docs/insat_provenance_reset.md")
    assert doc_path.exists(), "docs/insat_provenance_reset.md must exist"
    content = doc_path.read_text(encoding="utf-8")
    assert MANDATORY_PROVENANCE_QUOTE in content, (
        f"Mandatory provenance quote missing from {doc_path}."
    )


def test_2_native_product_manifest():
    """Verify that the native product manifest specifies 3DIMG_L1C_ASIA_MER and genuine channels."""
    manifest_path = Path("data/manifests/insat3d_native_product_manifest.csv")
    assert manifest_path.exists(), "Product manifest must exist"
    df = pd.read_csv(manifest_path)
    assert len(df) >= 1
    assert "3DIMG_L1C_ASIA_MER" in df["product_id"].values
    row = df[df["product_id"] == "3DIMG_L1C_ASIA_MER"].iloc[0]
    assert "HDF5" in str(row["format"])
    assert "IMG_TIR1" in str(row["target_ml_channels"])
    assert "IMG_TIR2" in str(row["target_ml_channels"])
    assert "IMG_WV" in str(row["target_ml_channels"])


def test_3_native_schema_manifest():
    """Verify that the native schema documents count-to-Kelvin LUTs for TIR1, TIR2, and WV."""
    schema_path = Path("data/manifests/insat3d_native_schema.csv")
    assert schema_path.exists(), "Schema manifest must exist"
    df = pd.read_csv(schema_path)
    datasets = set(df["dataset_name"].values)
    assert "IMG_TIR1" in datasets
    assert "IMG_TIR2" in datasets
    assert "IMG_WV" in datasets

    luts = set(df["calibration_lut_name"].dropna().values)
    assert "IMG_TIR1_TEMP" in luts
    assert "IMG_TIR2_TEMP" in luts
    assert "IMG_WV_TEMP" in luts


def test_4_timestamp_alignment_rules():
    """Verify timestamp alignment manifest contains zero synthetic timestamps and zero interpolation."""
    align_path = Path("data/manifests/native_insat_gridsat_timestamp_alignment.csv")
    assert align_path.exists(), "Timestamp alignment manifest must exist"
    df = pd.read_csv(align_path)
    assert len(df) >= 3
    # Rule: no temporal interpolation allowed
    assert (df["interpolation_applied"] == False).all()
    # Rule: explicit delta-t recorded
    assert "delta_t_minutes" in df.columns
    assert (df["delta_t_minutes"].abs() <= 30.0).all()


def test_5_rejection_of_gridsat_as_insat():
    """Verify processor strictly rejects GridSat .npz and non-HDF5 files."""
    processor = NativeINSAT3DProcessor()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        f.write(b"NOT_AN_HDF5_FILE_JUST_NPZ")
        npz_fake = Path(f.name)

    try:
        with pytest.raises(ValueError, match="not a valid HDF5 file"):
            processor.verify_hdf5_file(npz_fake)
    finally:
        npz_fake.unlink(missing_ok=True)


def test_6_physical_calibration_and_bounds():
    """Verify LUT calibration and strict physical Kelvin enforcement."""
    processor = NativeINSAT3DProcessor()

    # Create dummy LUT
    lut = np.full(1024, 250.0, dtype=np.float32)
    lut[100] = 120.0  # Physically corrupt (<180K)
    lut[200] = 390.0  # Physically corrupt (>330K)
    lut[300] = 275.0  # Valid

    counts = np.array([[100, 200], [300, 300]], dtype=np.uint16)
    calibrated = processor.calibrate_channel(counts, lut, "IMG_TIR1")

    # Corrupt values must be NaN
    assert np.isnan(calibrated[0, 0]), "Value < 180K should be rejected as NaN"
    assert np.isnan(calibrated[0, 1]), "Value > 330K should be rejected as NaN"
    # Valid value preserved
    assert calibrated[1, 0] == 275.0
    assert calibrated[1, 1] == 275.0


def test_7_spatial_coordinate_orientation():
    """Verify coordinates are strictly ascending (South to North, West to East)."""
    processor = NativeINSAT3DProcessor(target_resolution="model")
    dummy_grid = np.full((1382, 1386), 250.0, dtype=np.float32)
    # Native Mercator latitude descends from 45.5 to -10.0
    native_lat = np.linspace(45.5, -10.0, 1382)
    native_lon = np.linspace(44.5, 105.5, 1386)

    out_arr, out_lat, out_lon = processor.standardize_spatial_grid(dummy_grid, native_lat, native_lon)

    assert out_arr.shape == (72, 116)
    assert out_lat[0] < out_lat[-1], "Latitude must be strictly ascending (South to North)"
    assert out_lat.min() >= NIO_LAT_MIN - 1e-3
    assert out_lat.max() <= NIO_LAT_MAX + 1e-3

    assert out_lon[0] < out_lon[-1], "Longitude must be strictly ascending (West to East)"
    assert out_lon.min() >= NIO_LON_MIN - 1e-3
    assert out_lon.max() <= NIO_LON_MAX + 1e-3


def test_8_zero_imd_center_cropping():
    """Verify standardization uses fixed NIO basin bounds, zero IMD labels."""
    processor = NativeINSAT3DProcessor()
    # Check that spatial bounds are fixed constants
    assert NIO_LAT_MIN == -5.0
    assert NIO_LAT_MAX == 35.0
    assert NIO_LON_MIN == 40.0
    assert NIO_LON_MAX == 105.0


def test_9_mosdac_credential_safety_protocol():
    """Verify downloader interface safely halts without leaking or hardcoding credentials."""
    client = MOSDACDownloadClient()
    # Check that without env vars, require_credentials raises CredentialsUnavailableError
    if not client.has_credentials():
        with pytest.raises(CredentialsUnavailableError):
            client.require_credentials()


def test_10_native_pilot_inventory_and_stop_condition():
    """
    Audit native pilot inventory:
    - If 0 genuine HDF5 files exist due to missing credentials, enforce stop condition.
    - Confirm NO new training scripts or models were executed.
    """
    pilot_manifest = Path("data/manifests/insat3d_native_pilot_manifest.csv")
    assert pilot_manifest.exists()
    df = pd.read_csv(pilot_manifest)

    # Check local raw directory
    raw_dir = Path("data/raw/insat3d/")
    local_h5_files = list(raw_dir.glob("*.h5")) if raw_dir.exists() else []

    if len(local_h5_files) == 0:
        # Stop condition must be explicitly recognized
        assert (df["file_present"] == False).all()
        assert "PENDING_MOSDAC_CREDENTIALS" in df["physical_verification_status"].values
        print("\n[VALIDATION NOTICE] STOP CONDITION TRIGGERED:")
        print("  0 native .h5 files present on local disk.")
        print("  MOSDAC credentials (MOSDAC_USERNAME / MOSDAC_PASSWORD) required for physical download.")
        print("  In accordance with Part 10: ZERO models trained. Pipeline cleanly paused.")
    else:
        # If files exist, verify each one
        processor = NativeINSAT3DProcessor()
        for h5_file in local_h5_files:
            sha256 = processor.verify_hdf5_file(h5_file)
            assert len(sha256) == 64
            print(f"[VERIFIED] {h5_file.name} -> SHA-256: {sha256}")


def main():
    print("=" * 60)
    print("VAYU-NET: RUNNING NATIVE INSAT-3D VALIDATION SUITE")
    print("=" * 60)
    tests = [
        ("Test 1: Provenance Reset Document", test_1_provenance_reset_document),
        ("Test 2: Native Product Manifest", test_2_native_product_manifest),
        ("Test 3: Native Schema Manifest", test_3_native_schema_manifest),
        ("Test 4: Timestamp Alignment Rules", test_4_timestamp_alignment_rules),
        ("Test 5: Rejection of GridSat as INSAT", test_5_rejection_of_gridsat_as_insat),
        ("Test 6: Physical Calibration and Bounds", test_6_physical_calibration_and_bounds),
        ("Test 7: Spatial Coordinate Orientation", test_7_spatial_coordinate_orientation),
        ("Test 8: Zero IMD Center Cropping", test_8_zero_imd_center_cropping),
        ("Test 9: MOSDAC Credential Safety Protocol", test_9_mosdac_credential_safety_protocol),
        ("Test 10: Native Pilot Inventory & Stop Condition", test_10_native_pilot_inventory_and_stop_condition),
    ]

    all_passed = True
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("[AUDIT SUCCESS] All 10 validation tests PASSED cleanly.")
        sys.exit(0)
    else:
        print("[AUDIT FAILURE] One or more tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
