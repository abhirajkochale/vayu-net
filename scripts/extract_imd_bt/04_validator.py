"""
04_validator.py — VAYU-NET IMD Best Track Pipeline
Runs all 14 critical integrity tests + quality checks on the normalized dataset.
Produces validation verdict: PASS / PASS WITH WARNINGS / BLOCKED.
"""

import hashlib
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib
cfg = importlib.import_module("00_config")

logger = logging.getLogger("validator")

# ── Reference storms ─────────────────────────────────────────────────────────
REFERENCE_STORMS = {
    "NIO_2019_FANI":      {"year": 2019, "name": "FANI",      "split": "VALIDATION"},
    "NIO_2020_AMPHAN":    {"year": 2020, "name": "AMPHAN",    "split": "VALIDATION"},
    "NIO_2021_TAUKTAE":   {"year": 2021, "name": "TAUKTAE",   "split": "TEST"},
    "NIO_2023_BIPARJOY":  {"year": 2023, "name": "BIPARJOY",  "split": "TEST"},
    "NIO_2024_REMAL":     {"year": 2024, "name": "REMAL",     "split": "TEST"},
}

# ── SHA-256 verification ───────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_pdfs_unchanged(year_file_map: dict) -> list:
    """
    Returns list of (filename, expected_sha256, actual_sha256, match: bool).
    """
    results = []
    for year, info in sorted(year_file_map.items()):
        filepath = cfg.RAW_IMD_DIR / info["filename"]
        if not filepath.exists():
            results.append((info["filename"], info["sha256"], "FILE_MISSING", False))
            continue
        actual = _sha256_file(filepath)
        match = actual.lower() == info["sha256"].lower()
        results.append((info["filename"], info["sha256"], actual, match))
    return results


# ── Validation functions ───────────────────────────────────────────────────────

def run_validation(df: pd.DataFrame, year_file_map: dict, manual_review: list) -> dict:
    """
    Run all 14 integrity tests and quality checks.
    Returns a validation_report dict.
    """
    report = {
        "tests": {},
        "quality_checks": {},
        "reference_storms": {},
        "sha256_checks": [],
        "verdict": None,
        "critical_failures": [],
        "warnings": [],
    }

    # ── TEST 1: Every year 1998–2024 represented ───────────────────────────
    years_present = set(df["year"].dropna().astype(int).unique())
    missing_years = [y for y in cfg.CORE_YEARS if y not in years_present]
    test1_pass = len(missing_years) == 0
    report["tests"]["TEST1_all_years_represented"] = {
        "pass": test1_pass,
        "missing_years": missing_years,
        "years_found": sorted(years_present),
    }
    if not test1_pass:
        report["critical_failures"].append(f"TEST1: Missing years: {missing_years}")

    # ── TEST 2: No canonical report skipped ───────────────────────────────
    expected_files = set(info["filename"] for info in year_file_map.values())
    files_in_data = set(df["source_file"].dropna().unique())
    missing_files = expected_files - files_in_data
    test2_pass = len(missing_files) == 0
    report["tests"]["TEST2_no_report_skipped"] = {
        "pass": test2_pass,
        "missing_source_files": sorted(missing_files),
    }
    if not test2_pass:
        report["warnings"].append(f"TEST2: Reports with no extracted data: {missing_files}")

    # ── TEST 3: 1997 outside core dataset ─────────────────────────────────
    rows_1997 = len(df[df["year"] == 1997])
    test3_pass = rows_1997 == 0
    report["tests"]["TEST3_1997_excluded"] = {"pass": test3_pass, "rows_1997": rows_1997}
    if not test3_pass:
        report["critical_failures"].append(f"TEST3: 1997 has {rows_1997} rows in core dataset")

    # ── TEST 4: 2025 not in TRAIN/VAL/TEST ────────────────────────────────
    df_2025_in_ml = df[(df["year"] == 2025) & (df["split"].isin(["TRAIN", "VALIDATION", "TEST"]))]
    test4_pass = len(df_2025_in_ml) == 0
    report["tests"]["TEST4_2025_excluded"] = {"pass": test4_pass, "rows_2025_in_ml": len(df_2025_in_ml)}
    if not test4_pass:
        report["critical_failures"].append("TEST4: 2025 records found in ML splits")

    # ── TEST 5: No storm in multiple splits ───────────────────────────────
    storm_splits = df.groupby("storm_id")["split"].nunique()
    multi_split_storms = storm_splits[storm_splits > 1].index.tolist()
    test5_pass = len(multi_split_storms) == 0
    report["tests"]["TEST5_no_storm_in_multiple_splits"] = {
        "pass": test5_pass,
        "multi_split_storms": multi_split_storms,
    }
    if not test5_pass:
        report["critical_failures"].append(f"TEST5: Storms in multiple splits: {multi_split_storms}")

    # ── TEST 6: No duplicate identical observation rows ────────────────────
    dup_cols = ["storm_id", "timestamp_utc", "latitude", "longitude",
                "maximum_sustained_wind_kt", "central_pressure_hpa"]
    available_dup_cols = [c for c in dup_cols if c in df.columns]
    dup_mask = df.duplicated(subset=available_dup_cols, keep=False)
    n_dups = dup_mask.sum()
    test6_pass = n_dups == 0
    report["tests"]["TEST6_no_duplicate_rows"] = {"pass": test6_pass, "duplicate_count": int(n_dups)}
    if not test6_pass:
        report["warnings"].append(f"TEST6: {n_dups} duplicate observation rows found")

    # ── TEST 7: Every normalized row has provenance ────────────────────────
    provenance_cols = ["source_file", "source_page", "source_report"]
    missing_prov = {}
    for col in provenance_cols:
        n_missing = df[col].isna().sum() if col in df.columns else len(df)
        if n_missing > 0:
            missing_prov[col] = int(n_missing)
    test7_pass = len(missing_prov) == 0
    report["tests"]["TEST7_all_rows_have_provenance"] = {
        "pass": test7_pass,
        "missing_provenance": missing_prov,
    }
    if not test7_pass:
        report["warnings"].append(f"TEST7: Provenance gaps: {missing_prov}")

    # ── TEST 8: All 5 reference storms present ────────────────────────────
    ref_results = {}
    for sid, info in REFERENCE_STORMS.items():
        storm_rows = df[df["storm_id"] == sid]
        if len(storm_rows) == 0:
            # Try fuzzy match on storm_name
            name_rows = df[df["storm_name"] == info["name"]]
            if len(name_rows) > 0 and name_rows["year"].iloc[0] == info["year"]:
                actual_sid = name_rows["storm_id"].iloc[0]
                ref_results[sid] = {
                    "found": True, "fuzzy_match": actual_sid,
                    "n_obs": len(name_rows),
                    "issues": [f"storm_id mismatch: found {actual_sid}"],
                }
            else:
                ref_results[sid] = {"found": False, "n_obs": 0, "issues": ["NOT FOUND"]}
        else:
            issues = []
            n_obs = len(storm_rows)
            if n_obs < 2:
                issues.append(f"Only {n_obs} observation(s)")
            if storm_rows["latitude"].isna().all():
                issues.append("All latitudes missing")
            if storm_rows["longitude"].isna().all():
                issues.append("All longitudes missing")
            if storm_rows["maximum_sustained_wind_kt"].isna().all():
                issues.append("All wind values missing")
            # Check timestamp ordering
            ts_vals = storm_rows["timestamp_utc"].dropna().sort_values()
            if len(ts_vals) > 1 and list(ts_vals) != sorted(ts_vals):
                issues.append("Timestamps not in chronological order")
            # Check for duplicates within storm
            storm_dups = storm_rows.duplicated(subset=["timestamp_utc"], keep=False).sum()
            if storm_dups > 0:
                issues.append(f"{storm_dups} duplicate timestamps within storm")
            ref_results[sid] = {
                "found": True, "n_obs": n_obs,
                "year": int(storm_rows["year"].iloc[0]),
                "split": storm_rows["split"].iloc[0],
                "source_file": storm_rows["source_file"].iloc[0],
                "lat_ok": not storm_rows["latitude"].isna().all(),
                "lon_ok": not storm_rows["longitude"].isna().all(),
                "wind_ok": not storm_rows["maximum_sustained_wind_kt"].isna().all(),
                "pres_ok": not storm_rows["central_pressure_hpa"].isna().all(),
                "cat_ok": not storm_rows["category"].isna().all(),
                "issues": issues,
            }
    all_ref_found = all(v["found"] for v in ref_results.values())
    test8_pass = all_ref_found
    report["tests"]["TEST8_reference_storms_present"] = {"pass": test8_pass}
    report["reference_storms"] = ref_results
    if not test8_pass:
        missing_ref = [sid for sid, v in ref_results.items() if not v["found"]]
        report["critical_failures"].append(f"TEST8: Reference storms not found: {missing_ref}")

    # ── TEST 9: CSV and Parquet read back (deferred to 05_outputs) ────────
    report["tests"]["TEST9_csv_parquet_readback"] = {"pass": None, "note": "Verified in 05_outputs"}

    # ── TEST 10: Column names and dtypes stable ────────────────────────────
    expected_cols = set(cfg.CANONICAL_FIELDS)
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    test10_pass = len(missing_cols) == 0
    report["tests"]["TEST10_schema_stable"] = {
        "pass": test10_pass, "missing_cols": sorted(missing_cols),
        "extra_cols": sorted(extra_cols),
    }
    if not test10_pass:
        report["critical_failures"].append(f"TEST10: Missing columns: {missing_cols}")

    # ── TEST 11: No silent OCR corrections ────────────────────────────────
    # Validated structurally — the normalizer never mutates raw_* fields.
    # Check that raw_* values are preserved wherever normalized differ.
    silent_correction_count = 0
    for _, row in df.iterrows():
        raw_lat = row.get("raw_latitude")
        norm_lat = row.get("latitude")
        raw_lon = row.get("raw_longitude")
        norm_lon = row.get("longitude")
        # If normalized value differs significantly from raw AND no review flag → suspicious
        if raw_lat and norm_lat is not None and not pd.isna(norm_lat):
            try:
                raw_v = float(str(raw_lat).strip().rstrip("NnSs").strip())
                if abs(raw_v - abs(norm_lat)) > 0.1 and not row.get("manual_review_required", True):
                    silent_correction_count += 1
            except Exception:
                pass
    test11_pass = silent_correction_count == 0
    report["tests"]["TEST11_no_silent_ocr_corrections"] = {
        "pass": test11_pass, "suspected_silent_corrections": silent_correction_count,
    }
    if not test11_pass:
        report["warnings"].append(f"TEST11: {silent_correction_count} potential silent corrections")

    # ── TEST 12: No fabricated values ─────────────────────────────────────
    # Fabricated = normalized field has value but raw field is empty/None
    fabricated = 0
    for col_pair in [("latitude", "raw_latitude"), ("longitude", "raw_longitude"),
                     ("maximum_sustained_wind_kt", "raw_wind"),
                     ("central_pressure_hpa", "raw_pressure")]:
        norm_col, raw_col = col_pair
        if norm_col in df.columns and raw_col in df.columns:
            mask = df[norm_col].notna() & (df[raw_col].isna() | (df[raw_col] == ""))
            fabricated += mask.sum()
    test12_pass = fabricated == 0
    report["tests"]["TEST12_no_fabricated_values"] = {
        "pass": test12_pass, "fabricated_count": int(fabricated),
    }
    if not test12_pass:
        report["critical_failures"].append(f"TEST12: {fabricated} potential fabricated values")

    # ── TEST 13: No GridSat files downloaded ──────────────────────────────
    gridsat_raw = cfg.REPO_ROOT / "data" / "raw" / "gridsat"
    gridsat_interim = cfg.REPO_ROOT / "data" / "interim" / "gridsat_tensors"
    gridsat_downloaded = (
        (gridsat_raw.exists() and any(gridsat_raw.rglob("*.nc"))) or
        (gridsat_interim.exists() and any(gridsat_interim.rglob("*")))
    )
    test13_pass = not gridsat_downloaded
    report["tests"]["TEST13_no_gridsat_downloaded"] = {"pass": test13_pass}
    if not test13_pass:
        report["critical_failures"].append("TEST13: GridSat data found — pipeline boundary violated")

    # ── TEST 14: No source IMD PDF modified ───────────────────────────────
    sha_results = verify_source_pdfs_unchanged(year_file_map)
    report["sha256_checks"] = [
        {"filename": r[0], "expected": r[1][:12] + "…", "actual": r[2][:12] + "…" if len(r[2]) > 12 else r[2], "match": r[3]}
        for r in sha_results
    ]
    all_sha_match = all(r[3] for r in sha_results)
    test14_pass = all_sha_match
    report["tests"]["TEST14_source_pdfs_unchanged"] = {
        "pass": test14_pass,
        "mismatches": [(r[0], r[2]) for r in sha_results if not r[3]],
    }
    if not test14_pass:
        report["critical_failures"].append("TEST14: Source PDF SHA-256 mismatch — files may have been modified")

    # ── Quality checks ─────────────────────────────────────────────────────
    n_total = len(df)
    n_storms = df["storm_id"].nunique()

    # Missing values
    mv = {}
    for col in ["timestamp_utc", "latitude", "longitude",
                "maximum_sustained_wind_kt", "central_pressure_hpa", "category"]:
        if col in df.columns:
            mv[col] = int(df[col].isna().sum())

    # Invalid coordinates
    n_invalid_lat = int((df["latitude"].abs() > 90).sum()) if "latitude" in df.columns else 0
    n_invalid_lon = int((df["longitude"].abs() > 180).sum()) if "longitude" in df.columns else 0

    # Suspicious wind (outside physical range, already handled in normalizer but count NaN from it)
    n_missing_wind = int(df["maximum_sustained_wind_kt"].isna().sum())
    n_missing_pres = int(df["central_pressure_hpa"].isna().sum())
    n_missing_cat = int(df["category"].isna().sum())

    # Duplicate timestamps within storms
    dup_ts = df.groupby("storm_id")["timestamp_utc"].apply(
        lambda x: x.duplicated().sum()
    ).sum()

    # Manual review counts by type
    mr_df = pd.DataFrame(manual_review)
    mr_by_type = mr_df["problem_type"].value_counts().to_dict() if len(mr_df) else {}

    # Split counts
    split_counts = df.groupby("split").agg(
        storms=("storm_id", "nunique"),
        observations=("storm_id", "count")
    ).to_dict()

    # Year coverage
    obs_by_year = df.groupby("year").size().to_dict()

    # Timestamp anomalies: non-chronological within storm
    ts_anomaly_storms = []
    for sid, grp in df.groupby("storm_id"):
        ts_vals = grp["timestamp_utc"].dropna().sort_values(ascending=True)
        if list(ts_vals) != list(grp["timestamp_utc"].dropna()):
            ts_anomaly_storms.append(sid)

    report["quality_checks"] = {
        "total_rows": n_total,
        "total_storms": n_storms,
        "obs_by_year": {int(k): int(v) for k, v in obs_by_year.items()},
        "split_counts": {
            split: {"storms": split_counts.get("storms", {}).get(split, 0),
                    "observations": split_counts.get("observations", {}).get(split, 0)}
            for split in ["TRAIN", "VALIDATION", "TEST", "BLIND"]
        },
        "missing_values": mv,
        "invalid_coordinate_rows": n_invalid_lat + n_invalid_lon,
        "missing_wind_rows": n_missing_wind,
        "missing_pressure_rows": n_missing_pres,
        "missing_category_rows": n_missing_cat,
        "duplicate_timestamp_count": int(dup_ts),
        "duplicate_observation_count": int(n_dups),
        "manual_review_total": len(manual_review),
        "manual_review_by_type": mr_by_type,
        "timestamp_anomaly_storms": len(ts_anomaly_storms),
        "ocr_rows": int((df["extraction_method"] == "OCR").sum()) if "extraction_method" in df.columns else 0,
        "digital_rows": int((df["extraction_method"] == "DIGITAL_TABLE").sum()) if "extraction_method" in df.columns else 0,
    }

    # ── Verdict ────────────────────────────────────────────────────────────
    if report["critical_failures"]:
        report["verdict"] = "BLOCKED"
    elif report["warnings"] or report["quality_checks"]["manual_review_total"] > 0:
        report["verdict"] = "PASS WITH WARNINGS"
    else:
        report["verdict"] = "PASS"

    logger.info(f"Validation verdict: {report['verdict']}")
    for f in report["critical_failures"]:
        logger.error(f"CRITICAL: {f}")
    for w in report["warnings"]:
        logger.warning(f"WARNING: {w}")

    return report
