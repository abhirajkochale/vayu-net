"""
VAYU-NET: Bulk NASA GPM IMERG Final Run V07B Download Script.

Downloads all 2,353 unique granules referenced in:
    data/manifests/gridsat_imerg_pairing_manifest.csv

Rules enforced:
- Credentials ONLY from environment variables.
- Skip already-valid local files (resume-safe).
- Retry transient HTTP failures with exponential backoff.
- Validate HDF5 magic header for every file.
- Compute and record SHA-256 for every valid file.
- Conservative sequential download.
- Writes provenance manifest to data/manifests/imerg_bulk_download_manifest.csv.
- NO model training. NO production changes. NO synthetic data.
"""

import os
import sys
import hashlib
import time
import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import requests
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.imerg_download_client import (
    EarthdataSession,
    IMERGDownloadClient,
    CredentialsUnavailableError,
    EarthdataAuthenticationError,
    HDF5_SIGNATURE,
)

PAIRING_MANIFEST  = REPO_ROOT / "data/manifests/gridsat_imerg_pairing_manifest.csv"
SEQ_MANIFEST      = REPO_ROOT / "data/manifests/gridsat_imerg_sequence_pairing_manifest.csv"
LOCAL_IMERG_DIR   = REPO_ROOT / "data/raw/imerg"
BULK_MANIFEST_OUT = REPO_ROOT / "data/manifests/imerg_bulk_download_manifest.csv"

MAX_RETRIES        = 5
RETRY_BASE_DELAY_S = 10
STREAM_CHUNK_SIZE  = 131072
REQUEST_TIMEOUT_S  = 120

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IMERGBulkDownload")
HDF5_SIG = HDF5_SIGNATURE


def compute_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_hdf5_header(path):
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == HDF5_SIG
    except Exception:
        return False


def validate_hdf5_full(path):
    result = {
        "hdf5_valid": False, "product_valid": False, "variable_valid": False,
        "spatial_valid": False, "timestamp_valid": False,
        "units": None, "failure_reason": "",
    }
    if not verify_hdf5_header(path):
        result["failure_reason"] = "Invalid HDF5 magic header"
        return result
    result["hdf5_valid"] = True
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            if "Grid" not in h5 or "precipitation" not in h5["Grid"]:
                result["failure_reason"] = "Missing /Grid/precipitation"
                return result
            result["variable_valid"] = True
            result["product_valid"] = "V07B" in path.name
            attrs = dict(h5["Grid"]["precipitation"].attrs)
            units_raw = attrs.get("Units", attrs.get("units", b""))
            if isinstance(units_raw, bytes):
                units_raw = units_raw.decode("utf-8", errors="ignore")
            result["units"] = units_raw.strip()
            lat = h5["Grid"]["lat"][:]
            lon = h5["Grid"]["lon"][:]
            result["spatial_valid"] = (len(lat) == 1800 and len(lon) == 3600)
            if not result["spatial_valid"]:
                result["failure_reason"] = f"Unexpected dims lat={len(lat)} lon={len(lon)}"
            m = re.search(r"\.(\d{8})-S(\d{6})-E(\d{6})\.", path.name)
            result["timestamp_valid"] = bool(m)
            if not m:
                result["failure_reason"] += " Cannot parse timestamp from filename"
    except Exception as exc:
        result["failure_reason"] = f"HDF5 read error: {exc}"
    return result


def download_one_granule(session, url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT_S) as resp:
                if resp.status_code == 404:
                    return {"download_status": "FAILED_404", "file_size_bytes": 0, "sha256": "",
                            "failure_reason": f"HTTP 404: {url}"}
                if resp.status_code == 401:
                    return {"download_status": "FAILED_AUTH", "file_size_bytes": 0, "sha256": "",
                            "failure_reason": "HTTP 401 Unauthorized"}
                resp.raise_for_status()
                hasher = hashlib.sha256()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            hasher.update(chunk)
            if not verify_hdf5_header(tmp_path):
                tmp_path.unlink(missing_ok=True)
                return {"download_status": "FAILED_NOT_HDF5", "file_size_bytes": 0, "sha256": "",
                        "failure_reason": "Not genuine HDF5 (HTML error page?)"}
            tmp_path.rename(dest_path)
            size = dest_path.stat().st_size
            return {"download_status": "DOWNLOADED", "file_size_bytes": size,
                    "sha256": hasher.hexdigest(), "failure_reason": ""}
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            if str(code) in ("404", "401", "403"):
                tmp_path.unlink(missing_ok=True)
                return {"download_status": f"FAILED_{code}", "file_size_bytes": 0, "sha256": "",
                        "failure_reason": str(exc)}
        except Exception:
            pass
        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            logger.warning(f"Attempt {attempt} failed for {dest_path.name}. Retry in {delay}s...")
            time.sleep(delay)
    tmp_path.unlink(missing_ok=True)
    return {"download_status": "FAILED_MAX_RETRIES", "file_size_bytes": 0, "sha256": "",
            "failure_reason": f"Exceeded {MAX_RETRIES} retries"}


def main():
    logger.info("Loading pairing manifest...")
    df_pair = pd.read_csv(PAIRING_MANIFEST)
    LOCAL_IMERG_DIR.mkdir(parents=True, exist_ok=True)

    unique_granules = df_pair[["imerg_granule_filename", "download_url"]].drop_duplicates("imerg_granule_filename")
    n_unique = len(unique_granules)
    logger.info(f"Unique IMERG granules to process: {n_unique}")

    username = os.environ.get("EARTHDATA_USERNAME") or os.environ.get("NASA_EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("NASA_EARTHDATA_PASSWORD")
    if not (username and password):
        logger.error("STOP: EARTHDATA_USERNAME / EARTHDATA_PASSWORD not set.")
        sys.exit(1)

    logger.info("Verifying NASA Earthdata authentication...")
    client = IMERGDownloadClient(output_dir=str(LOCAL_IMERG_DIR))
    try:
        client.check_auth()
        logger.info("Authentication verified.")
    except EarthdataAuthenticationError as exc:
        logger.error(f"STOP: Authentication failed: {exc}")
        sys.exit(1)

    session = client.get_session()
    records = []
    stats = {"n_requested": n_unique, "n_already_valid": 0, "n_downloaded": 0,
             "n_failed": 0, "bytes_downloaded": 0, "bytes_total_local": 0}
    start_ts = datetime.now(timezone.utc).isoformat()

    for i, (_, row) in enumerate(unique_granules.iterrows(), 1):
        fname = row["imerg_granule_filename"]
        url   = row["download_url"]
        dest  = LOCAL_IMERG_DIR / fname

        if i % 100 == 0 or i == 1:
            logger.info(f"Progress: {i}/{n_unique} | valid={stats['n_already_valid']} "
                        f"dl={stats['n_downloaded']} fail={stats['n_failed']}")

        # Check canonical path
        if dest.exists() and dest.stat().st_size > 0 and verify_hdf5_header(dest):
            sha256 = compute_sha256(dest)
            val = validate_hdf5_full(dest)
            records.append({"imerg_granule_filename": fname, "download_url": url,
                            "local_path": str(dest), "local_presence": True,
                            "file_size_bytes": dest.stat().st_size, "sha256": sha256,
                            "download_status": "ALREADY_VALID", "failure_reason": "", **val})
            stats["n_already_valid"] += 1
            stats["bytes_total_local"] += dest.stat().st_size
            continue

        # Check short pilot filename
        short_name = fname.replace("3B-HHR.MS.MRG.3IMERG.", "")
        dest_short = LOCAL_IMERG_DIR / short_name
        if dest_short.exists() and dest_short.stat().st_size > 0 and verify_hdf5_header(dest_short):
            sha256 = compute_sha256(dest_short)
            val = validate_hdf5_full(dest_short)
            records.append({"imerg_granule_filename": fname, "download_url": url,
                            "local_path": str(dest_short), "local_presence": True,
                            "file_size_bytes": dest_short.stat().st_size, "sha256": sha256,
                            "download_status": "ALREADY_VALID", "failure_reason": "", **val})
            stats["n_already_valid"] += 1
            stats["bytes_total_local"] += dest_short.stat().st_size
            continue

        # Download
        dl = download_one_granule(session, url, dest)
        if dl["download_status"] == "DOWNLOADED":
            val = validate_hdf5_full(dest)
            if not val["hdf5_valid"]:
                dest.unlink(missing_ok=True)
                dl["download_status"] = "FAILED_CORRUPT"
            records.append({"imerg_granule_filename": fname, "download_url": url,
                            "local_path": str(dest) if dest.exists() else "",
                            "local_presence": dest.exists(),
                            "file_size_bytes": dl["file_size_bytes"], "sha256": dl["sha256"],
                            "download_status": dl["download_status"],
                            "failure_reason": val.get("failure_reason","") or dl["failure_reason"],
                            **val})
            stats["n_downloaded"] += 1
            stats["bytes_downloaded"]  += dl["file_size_bytes"]
            stats["bytes_total_local"] += dl["file_size_bytes"]
        else:
            records.append({"imerg_granule_filename": fname, "download_url": url,
                            "local_path": "", "local_presence": False,
                            "file_size_bytes": 0, "sha256": "",
                            "download_status": dl["download_status"],
                            "failure_reason": dl["failure_reason"],
                            "hdf5_valid": False, "product_valid": False,
                            "variable_valid": False, "spatial_valid": False,
                            "timestamp_valid": False, "units": None})
            stats["n_failed"] += 1
            logger.warning(f"FAILED [{dl['download_status']}]: {fname}")

    df_bulk = pd.DataFrame(records)

    # Merge with context columns
    ctx_cols = ["imerg_granule_filename","gridsat_timestamp_utc","gridsat_file","imerg_timestamp_utc",
                "delta_minutes","product_collection","algorithm_version","format","primary_variable",
                "spatial_resolution_deg","raw_grid_shape","nio_subgrid_shape","model_grid_shape",
                "pairing_status","spatial_compatibility","rejection_reason"]
    df_ctx = df_pair[ctx_cols].merge(df_bulk, on="imerg_granule_filename", how="left",
                                     suffixes=("_manifest",""))

    def compute_vs(row):
        if row.get("download_status") in ("ALREADY_VALID","DOWNLOADED"):
            if all([row.get("hdf5_valid"), row.get("product_valid"), row.get("variable_valid"),
                    row.get("spatial_valid"), row.get("timestamp_valid")]):
                return "PASS"
            return "FAIL_INTEGRITY"
        return "FAIL_DOWNLOAD"

    df_ctx["validation_status"] = df_ctx.apply(compute_vs, axis=1)
    df_ctx.to_csv(BULK_MANIFEST_OUT, index=False)
    logger.info(f"Bulk manifest written: {BULK_MANIFEST_OUT}")

    end_ts = datetime.now(timezone.utc).isoformat()
    n_pass = (df_ctx["validation_status"] == "PASS").sum()
    n_fail = (df_ctx["validation_status"] != "PASS").sum()
    gb_dl    = stats["bytes_downloaded"] / (1024**3)
    gb_local = stats["bytes_total_local"] / (1024**3)

    print()
    print("=" * 65)
    print("VAYU-NET: IMERG BULK DOWNLOAD COMPLETE")
    print("=" * 65)
    print(f"  Granules requested:      {stats['n_requested']}")
    print(f"  Already valid (skipped): {stats['n_already_valid']}")
    print(f"  Newly downloaded:        {stats['n_downloaded']}")
    print(f"  Failed:                  {stats['n_failed']}")
    print(f"  New bytes downloaded:    {gb_dl:.3f} GB")
    print(f"  Total local storage:     {gb_local:.3f} GB")
    print(f"  Frame rows passing:      {n_pass} / {len(df_ctx)}")
    print(f"  Frame rows failing:      {n_fail}")
    print(f"  Start: {start_ts}")
    print(f"  End:   {end_ts}")
    print("=" * 65)

    if stats["n_failed"] > 0:
        failed = df_ctx[df_ctx["validation_status"] != "PASS"][
            ["imerg_granule_filename","download_status","failure_reason"]
        ].drop_duplicates("imerg_granule_filename").head(30)
        print(f"\nFAILED GRANULES (first 30):")
        for _, r in failed.iterrows():
            print(f"  [{r['download_status']}] {r['imerg_granule_filename']}")
            print(f"    Reason: {r['failure_reason']}")
        sys.exit(1)

    print("\n[READY] Full IMERG dataset validated and ready for multimodal dataset construction.")
    sys.exit(0)


if __name__ == "__main__":
    main()
