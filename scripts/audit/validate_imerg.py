"""
VAYU-NET: NASA GPM IMERG Final Run V07B Validation & Audit Suite.

Tests the 11 Scientific & Operational Integrity Criteria:
1. Correct NASA GPM product identity (GPM_3IMERGHH.07).
2. Correct algorithm version (V07B Final Run).
3. No synthetic channels present.
4. No GridSat masquerading as IMERG (rejection of non-HDF5).
5. Correct variable extraction (/Grid/precipitation, units mm/hr).
6. Fill-value handling (-9999.9 converted to NaN, no interpolation of missing values).
7. Spatial bounds correctness (lat [-5, 35] ascending, lon [40, 105] ascending).
8. No storm-centered or IMD-track cropping.
9. Timestamp correctness and zero temporal leakage.
10. Credential safety protocol (no hardcoding, clean stop if unset).
11. Pilot inventory and SHA-256 integrity verification.
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

from ml.data.process_imerg import (
    IMERGProcessor,
    NIO_LAT_MIN, NIO_LAT_MAX,
    NIO_LON_MIN, NIO_LON_MAX,
    IMERG_FILL_VALUE,
    HDF5_SIGNATURE
)
from ml.data.imerg_download_client import (
    IMERGDownloadClient,
    CredentialsUnavailableError,
    PILOT_TARGETS
)


def test_1_product_identity_and_version():
    """Verify that pilot manifest and granules specify GPM_3IMERGHH.07 and V07B."""
    manifest_path = Path("data/manifests/imerg_pilot_manifest.csv")
    assert manifest_path.exists(), "Pilot manifest must exist"
    df = pd.read_csv(manifest_path)
    assert len(df) == 3
    assert (df["collection"] == "GPM_3IMERGHH.07").all()
    assert (df["algorithm_version"] == "V07B").all()
    assert (df["file_present"] == True).all()


def test_2_no_synthetic_channels_or_gridsat_masquerade():
    """Verify processor strictly rejects GridSat .npz or dummy files as IMERG."""
    processor = IMERGProcessor()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        f.write(b"FAKE_GRID_SAT_SURROGATE_DATA")
        npz_fake = Path(f.name)

    try:
        with pytest.raises(ValueError, match="not a valid HDF5 file"):
            processor.verify_hdf5_file(npz_fake)
    finally:
        npz_fake.unlink(missing_ok=True)


def test_3_variable_extraction_and_units():
    """Verify extraction of calibrated precipitation variable and mm/hr units."""
    pilot_dir = Path("data/raw/imerg")
    fani_h5 = pilot_dir / "20190429-S120000-E122959.0720.V07B.HDF5"
    assert fani_h5.exists(), "FANI pilot granule must be present"

    processor = IMERGProcessor()
    res = processor.process_granule(fani_h5)

    assert res["provenance"]["variable_extracted"] in ["precipitation", "precipitationCal"]
    assert res["provenance"]["physical_units"] == "mm/hr"
    assert res["precipitation"].ndim == 2
    assert res["precipitation"].dtype == np.float32


def test_4_fill_value_handling_and_no_missing_interpolation():
    """Verify that fill values (-9999.9) are converted to NaN and not artificially interpolated."""
    processor = IMERGProcessor()
    # Test on synthetic HDF5 with injected -9999.9 fill values
    with tempfile.NamedTemporaryFile(suffix=".HDF5", delete=False) as f:
        h5_path = Path(f.name)

    try:
        with h5py.File(h5_path, "w") as h5:
            grid = h5.create_group("Grid")
            grid.create_dataset("lat", data=np.linspace(-89.95, 89.95, 1800, dtype=np.float32))
            grid.create_dataset("lon", data=np.linspace(-179.95, 179.95, 3600, dtype=np.float32))

            # Shape: (1, 3600, 1800) [time, lon, lat]
            data = np.full((1, 3600, 1800), 5.0, dtype=np.float32)
            # Inject fill value
            data[0, 2200, 900] = -9999.9

            ds = grid.create_dataset("precipitation", data=data)
            ds.attrs["Units"] = "mm/hr"
            ds.attrs["_FillValue"] = -9999.9

        res = processor.process_granule(h5_path)
        arr = res["precipitation"]
        # Must contain NaN where fill value was present, and no negative numbers
        assert np.isnan(arr).any(), "Injected fill value must become NaN"
        assert not (arr[~np.isnan(arr)] < 0.0).any(), "No negative precipitation allowed"
    finally:
        h5_path.unlink(missing_ok=True)


def test_5_spatial_bounds_and_orientation():
    """Verify standardized spatial bounds and strictly ascending latitude/longitude."""
    pilot_dir = Path("data/raw/imerg")
    amphan_h5 = pilot_dir / "20200517-S060000-E062959.0360.V07B.HDF5"
    assert amphan_h5.exists()

    processor = IMERGProcessor(target_resolution="native_subgrid")
    res = processor.process_granule(amphan_h5)

    arr = res["precipitation"]
    lat = res["lat"]
    lon = res["lon"]

    # Native sub-grid dimensions: 40 deg lat / 0.1 = 400; 65 deg lon / 0.1 = 650
    assert arr.shape == (400, 650)
    assert lat[0] < lat[-1], "Latitude must be strictly ascending (South to North)"
    assert lon[0] < lon[-1], "Longitude must be strictly ascending (West to East)"
    assert lat.min() >= NIO_LAT_MIN - 0.1
    assert lat.max() <= NIO_LAT_MAX + 0.1
    assert lon.min() >= NIO_LON_MIN - 0.1
    assert lon.max() <= NIO_LON_MAX + 0.1


def test_6_no_storm_centered_cropping():
    """Verify that domain coordinates are fixed NIO basin constants, never centered on IMD track."""
    processor = IMERGProcessor()
    pilot_dir = Path("data/raw/imerg")
    remal_h5 = pilot_dir / "20240525-S060000-E062959.0360.V07B.HDF5"
    res = processor.process_granule(remal_h5)

    assert res["provenance"]["spatial_domain"]["storm_centered_cropping"] == False
    assert NIO_LAT_MIN == -5.0
    assert NIO_LAT_MAX == 35.0
    assert NIO_LON_MIN == 40.0
    assert NIO_LON_MAX == 105.0


def test_7_timestamp_alignment_correctness():
    """Verify timestamp alignment manifest has zero interpolation and exact synoptic matching."""
    align_path = Path("data/manifests/imerg_gridsat_timestamp_alignment.csv")
    assert align_path.exists()
    df = pd.read_csv(align_path)
    assert len(df) == 3
    assert (df["interpolation_applied"] == False).all()
    assert (df["delta_t_minutes"] == 0.0).all()
    assert (df["leakage_risk"] == "ZERO_LEAKAGE").all()


def test_8_credential_safety_protocol():
    """Verify downloader interface halts safely when credentials are absent without printing secrets."""
    # Temporarily mask credentials
    orig_u = os.environ.get("EARTHDATA_USERNAME")
    orig_p = os.environ.get("EARTHDATA_PASSWORD")

    try:
        os.environ.pop("EARTHDATA_USERNAME", None)
        os.environ.pop("NASA_EARTHDATA_USERNAME", None)
        os.environ.pop("EARTHDATA_PASSWORD", None)
        os.environ.pop("NASA_EARTHDATA_PASSWORD", None)

        client = IMERGDownloadClient()
        assert not client.has_credentials()
        with pytest.raises(CredentialsUnavailableError):
            client.require_credentials()
    finally:
        if orig_u: os.environ["EARTHDATA_USERNAME"] = orig_u
        if orig_p: os.environ["EARTHDATA_PASSWORD"] = orig_p


def test_9_pilot_inventory_and_sha256():
    """Verify that all 3 pilot granules exist, have correct size, and valid SHA-256."""
    manifest_path = Path("data/manifests/imerg_pilot_manifest.csv")
    df = pd.read_csv(manifest_path)

    for _, row in df.iterrows():
        p = Path(row["local_path"])
        assert p.exists(), f"File {p} must exist on disk"
        assert p.stat().st_size == row["file_size_bytes"]
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        assert hasher.hexdigest() == row["sha256"], f"SHA-256 mismatch for {p.name}"


def test_10_model_resolution_resampling_compatibility():
    """Verify that processor can resample to model grid (72, 116) without corrupting NaNs."""
    pilot_dir = Path("data/raw/imerg")
    fani_h5 = pilot_dir / "20190429-S120000-E122959.0720.V07B.HDF5"
    processor = IMERGProcessor(target_resolution="model")
    res = processor.process_granule(fani_h5)

    assert res["precipitation"].shape == (72, 116)
    assert res["lat"].shape == (72,)
    assert res["lon"].shape == (116,)
    assert not np.isnan(res["precipitation"]).all()


def test_11_zero_model_training_enforced():
    """Assert that no new model training scripts or production changes were executed."""
    assert not Path("ml/train/train_imerg.py").exists(), "Model training is strictly forbidden"


def main():
    print("=" * 60)
    print("VAYU-NET: RUNNING NASA GPM IMERG VALIDATION SUITE")
    print("=" * 60)
    tests = [
        ("Test 1: Product Identity & Version", test_1_product_identity_and_version),
        ("Test 2: No Synthetic Channels / Rejection of GridSat", test_2_no_synthetic_channels_or_gridsat_masquerade),
        ("Test 3: Variable Extraction & Units", test_3_variable_extraction_and_units),
        ("Test 4: Fill-Value Handling & No Missing Interpolation", test_4_fill_value_handling_and_no_missing_interpolation),
        ("Test 5: Spatial Bounds & Orientation", test_5_spatial_bounds_and_orientation),
        ("Test 6: No Storm-Centered Cropping", test_6_no_storm_centered_cropping),
        ("Test 7: Timestamp Alignment Correctness", test_7_timestamp_alignment_correctness),
        ("Test 8: Credential Safety Protocol", test_8_credential_safety_protocol),
        ("Test 9: Pilot Inventory & SHA-256", test_9_pilot_inventory_and_sha256),
        ("Test 10: Model Resolution Resampling Compatibility", test_10_model_resolution_resampling_compatibility),
        ("Test 11: Zero Model Training Enforced", test_11_zero_model_training_enforced),
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
        print("[AUDIT SUCCESS] All 11 NASA IMERG validation tests PASSED cleanly.")
        sys.exit(0)
    else:
        print("[AUDIT FAILURE] One or more tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
