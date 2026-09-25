"""
VAYU-NET: Post-Bulk-Download IMERG Validation Script.
Validates all 2,353 required granules against the bulk manifest.
"""

import os, sys, hashlib, re
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BULK_MANIFEST    = REPO_ROOT / "data/manifests/imerg_bulk_download_manifest.csv"
PAIRING_MANIFEST = REPO_ROOT / "data/manifests/gridsat_imerg_pairing_manifest.csv"
LOCAL_IMERG_DIR  = REPO_ROOT / "data/raw/imerg"
HDF5_SIG = b"\x89HDF\r\n\x1a\n"
EXPECTED_GRANULES = 2353


def test_1_manifest_exists():
    assert BULK_MANIFEST.exists()


def test_2_frame_rows():
    df = pd.read_csv(BULK_MANIFEST)
    assert len(df) == EXPECTED_GRANULES


def test_3_validation_pass():
    df = pd.read_csv(BULK_MANIFEST)
    assert (df["validation_status"] == "PASS").sum() == len(df)


def test_4_local_presence():
    df = pd.read_csv(BULK_MANIFEST)
    assert int(df["local_presence"].sum()) == len(df)


def test_5_hdf5_valid():
    df = pd.read_csv(BULK_MANIFEST)
    assert int(df["hdf5_valid"].sum()) == len(df)


def test_6_product_v07b():
    df = pd.read_csv(BULK_MANIFEST)
    assert int(df["product_valid"].sum()) == len(df)


def test_7_variable_precipitation():
    df = pd.read_csv(BULK_MANIFEST)
    assert int(df["variable_valid"].sum()) == len(df)


def test_8_spatial_dims():
    df = pd.read_csv(BULK_MANIFEST)
    assert int(df["spatial_valid"].sum()) == len(df)


def test_9_timestamp_parseable():
    df = pd.read_csv(BULK_MANIFEST)
    assert int(df["timestamp_valid"].sum()) == len(df)


def test_10_sha256_recorded():
    df = pd.read_csv(BULK_MANIFEST)
    assert int((df["sha256"].notna() & (df["sha256"].str.len() == 64)).sum()) == len(df)


def test_11_hdf5_files_on_disk():
    files = list(LOCAL_IMERG_DIR.glob("*.HDF5"))
    assert len(files) >= EXPECTED_GRANULES


def test_12_cross_check_pairing():
    df = pd.read_csv(BULK_MANIFEST)
    df_pair = pd.read_csv(PAIRING_MANIFEST)
    required = set(df_pair["imerg_granule_filename"].unique())
    have = set(df["imerg_granule_filename"].unique())
    assert len(required - have) == 0


def main():
    print("=" * 65)
    print("VAYU-NET: POST-DOWNLOAD IMERG VALIDATION")
    print("=" * 65)
    tests = [
        ("1. Bulk manifest exists", test_1_manifest_exists),
        ("2. Frame rows = 2353", test_2_frame_rows),
        ("3. Validation PASS: 2353/2353", test_3_validation_pass),
        ("4. Local presence: 2353/2353", test_4_local_presence),
        ("5. HDF5 valid: 2353/2353", test_5_hdf5_valid),
        ("6. Product V07B: 2353/2353", test_6_product_v07b),
        ("7. Variable /Grid/precipitation", test_7_variable_precipitation),
        ("8. Spatial dims 1800x3600", test_8_spatial_dims),
        ("9. Timestamp parseable", test_9_timestamp_parseable),
        ("10. SHA-256 recorded", test_10_sha256_recorded),
        ("11. HDF5 files on disk >= 2353", test_11_hdf5_files_on_disk),
        ("12. Cross-check vs pairing manifest", test_12_cross_check_pairing),
    ]
    all_ok = True
    for desc, fn in tests:
        try:
            fn()
            print(f"  [PASS] {desc}")
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")
            all_ok = False
            
    print("=" * 65)
    if all_ok:
        print("[AUDIT SUCCESS] All post-download validation checks PASSED.")
    else:
        print("[AUDIT FAILURE] One or more checks failed.")
    print("=" * 65)
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)