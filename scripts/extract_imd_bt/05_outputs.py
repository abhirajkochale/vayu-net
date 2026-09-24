"""
05_outputs.py — VAYU-NET IMD Best Track Pipeline
Writes all final output files and generates documentation reports.
"""

import json
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

logger = logging.getLogger("outputs")


# ── Storm event manifest ───────────────────────────────────────────────────────

def build_storm_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Build storm_event_manifest.csv from normalized observations."""
    records = []

    for sid, grp in df.groupby("storm_id"):
        grp_sorted = grp.sort_values("timestamp_utc", na_position="last")
        ts_valid = grp_sorted["timestamp_utc"].dropna()

        start_time = ts_valid.iloc[0] if len(ts_valid) > 0 else None
        end_time = ts_valid.iloc[-1] if len(ts_valid) > 0 else None

        pages = grp_sorted["source_page"].dropna()
        first_pg = int(pages.iloc[0]) if len(pages) > 0 else None
        last_pg = int(pages.iloc[-1]) if len(pages) > 0 else None

        mr_count = int(grp["manual_review_required"].sum())
        quality_status = "REVIEW_REQUIRED" if mr_count > 0 else "OK"

        records.append({
            "storm_id": sid,
            "storm_name": grp["storm_name"].dropna().iloc[0] if grp["storm_name"].notna().any() else None,
            "year": int(grp["year"].iloc[0]) if grp["year"].notna().any() else None,
            "start_time_utc": start_time,
            "end_time_utc": end_time,
            "num_observations": len(grp),
            "split": grp["split"].iloc[0],
            "source_file": grp["source_file"].iloc[0],
            "source_report": grp["source_report"].iloc[0],
            "first_source_page": first_pg,
            "last_source_page": last_pg,
            "quality_status": quality_status,
            "manual_review_count": mr_count,
        })

    manifest_df = pd.DataFrame(records).sort_values(["year", "start_time_utc"]).reset_index(drop=True)
    return manifest_df


# ── Manual review CSV ─────────────────────────────────────────────────────────

def write_manual_review_csv(manual_review_entries: list) -> int:
    """Write imd_manual_review.csv. Returns row count."""
    if not manual_review_entries:
        df = pd.DataFrame(columns=cfg.MANUAL_REVIEW_FIELDS)
    else:
        df = pd.DataFrame(manual_review_entries)
        # Ensure all required columns exist
        for col in cfg.MANUAL_REVIEW_FIELDS:
            if col not in df.columns:
                df[col] = None
        df = df[cfg.MANUAL_REVIEW_FIELDS]

    df.to_csv(cfg.MANUAL_REVIEW_CSV, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(df)} manual-review rows to {cfg.MANUAL_REVIEW_CSV}")
    return len(df)


# ── Final CSV + Parquet output ────────────────────────────────────────────────

def write_final_datasets(df: pd.DataFrame) -> dict:
    """Write imd_best_track.csv and .parquet. Returns read-back verification dict."""
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Write CSV
    df.to_csv(cfg.BEST_TRACK_CSV, index=False, encoding="utf-8")
    logger.info(f"Wrote CSV: {cfg.BEST_TRACK_CSV} ({len(df)} rows)")

    # Write Parquet
    df.to_parquet(cfg.BEST_TRACK_PARQUET, index=False, engine="pyarrow")
    logger.info(f"Wrote Parquet: {cfg.BEST_TRACK_PARQUET} ({len(df)} rows)")

    # Read-back verification
    df_csv = pd.read_csv(cfg.BEST_TRACK_CSV, encoding="utf-8")
    df_pq = pd.read_parquet(cfg.BEST_TRACK_PARQUET, engine="pyarrow")

    readback = {
        "csv_rows": len(df_csv),
        "parquet_rows": len(df_pq),
        "expected_rows": len(df),
        "csv_match": len(df_csv) == len(df),
        "parquet_match": len(df_pq) == len(df),
        "csv_cols": list(df_csv.columns),
        "parquet_cols": list(df_pq.columns),
    }
    if not readback["csv_match"]:
        logger.error(f"CSV read-back MISMATCH: wrote {len(df)}, read {len(df_csv)}")
    if not readback["parquet_match"]:
        logger.error(f"Parquet read-back MISMATCH: wrote {len(df)}, read {len(df_pq)}")

    return readback


# ── Extraction report ─────────────────────────────────────────────────────────

def write_extraction_report(
    df: pd.DataFrame,
    digital_results: dict,
    scanned_results: dict,
    validation_report: dict,
    manual_review: list,
    year_file_map: dict,
    readback: dict,
    run_ts: str,
) -> None:
    """Write docs/imd_best_track_extraction_report.md"""
    cfg.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    qc = validation_report.get("quality_checks", {})
    ref = validation_report.get("reference_storms", {})

    # Per-year storm and obs counts
    obs_by_year = qc.get("obs_by_year", {})
    storms_by_year = df.groupby("year")["storm_id"].nunique().to_dict()

    # Digital table count
    total_digital_tables = sum(
        len(r.get("tables_found", [])) for r in digital_results.values()
    )
    total_scanned_pages = sum(
        len(r.get("pages_processed", [])) for r in scanned_results.values()
    )

    lines = [
        "# IMD Best Track Extraction Report",
        f"\n**Generated:** {run_ts}",
        f"**Pipeline version:** VAYU-NET v1.0 / SIH 26070",
        "",
        "---",
        "",
        "## 1. Source Inventory",
        "",
        f"- Inventory CSV: `{cfg.INVENTORY_CSV.relative_to(cfg.REPO_ROOT)}`",
        f"- Total canonical reports: {len(year_file_map)}",
        f"- Digital reports (2005–2024): {len([y for y in year_file_map if y in cfg.DIGITAL_YEARS])}",
        f"- Scanned reports (1998–2004): {len([y for y in year_file_map if y in cfg.SCANNED_YEARS])}",
        "",
        "## 2. Reports Processed",
        "",
        "| Year | File | Type | Pages |",
        "|:-----|:-----|:-----|:------|",
    ]
    for year in sorted(year_file_map.keys()):
        info = year_file_map[year]
        lines.append(f"| {year} | `{info['filename']}` | {info['source_type']} | {info['page_count']} |")

    lines += [
        "",
        "## 3. Extraction Method by Year",
        "",
        "| Year | Method | BT Tables/Pages | Observations | Narrative Rows |",
        "|:-----|:-------|:----------------|:-------------|:---------------|",
    ]
    for year in sorted(cfg.CORE_YEARS):
        if year in digital_results:
            r = digital_results[year]
            method = "DIGITAL_TABLE (pdfplumber)"
            tables = len(r.get("tables_found", []))
            obs = r.get("total_observation_rows", 0)
            narr = r.get("total_narrative_rows", 0)
            lines.append(f"| {year} | {method} | {tables} | {obs} | {narr} |")
        elif year in scanned_results:
            r = scanned_results[year]
            method = "OCR (PyMuPDF + EasyOCR)"
            pages = len(r.get("pages_processed", []))
            obs = r.get("total_observation_rows", 0)
            narr = r.get("total_narrative_rows", 0)
            lines.append(f"| {year} | {method} | {pages} | {obs} | {narr} |")
        else:
            lines.append(f"| {year} | NOT PROCESSED | — | — | — |")

    lines += [
        "",
        "## 4. Storms and Observations per Year",
        "",
        "| Year | Storms | Observations | Split |",
        "|:-----|:-------|:-------------|:------|",
    ]
    for year in sorted(cfg.CORE_YEARS):
        n_obs = obs_by_year.get(year, 0)
        n_storms = storms_by_year.get(year, 0)
        split = cfg.get_split(year)
        lines.append(f"| {year} | {n_storms} | {n_obs} | {split} |")

    lines += [
        "",
        "## 5. Extraction Totals",
        "",
        f"- Total BT tables found (digital): {total_digital_tables}",
        f"- Total OCR pages processed: {total_scanned_pages}",
        f"- Total observation rows extracted: {qc.get('total_rows', 0)}",
        f"- Digital extraction rows: {qc.get('digital_rows', 0)}",
        f"- OCR extraction rows: {qc.get('ocr_rows', 0)}",
        f"- Manual-review rows: {qc.get('manual_review_total', 0)}",
        f"- Narrative rows (excluded from final dataset): {sum(1 for m in manual_review if m.get('problem_type') == 'NON_OBSERVATION_NARRATIVE_ROW')}",
        "",
        "## 6. Files Generated",
        "",
        f"- `{cfg.BEST_TRACK_CSV.relative_to(cfg.REPO_ROOT)}` — {readback['csv_rows']} rows",
        f"- `{cfg.BEST_TRACK_PARQUET.relative_to(cfg.REPO_ROOT)}` — {readback['parquet_rows']} rows",
        f"- `{cfg.STORM_MANIFEST_CSV.relative_to(cfg.REPO_ROOT)}`",
        f"- `{cfg.MANUAL_REVIEW_CSV.relative_to(cfg.REPO_ROOT)}`",
        f"- `{cfg.EXTRACTION_REPORT_MD.relative_to(cfg.REPO_ROOT)}`",
        f"- `{cfg.QUALITY_REPORT_MD.relative_to(cfg.REPO_ROOT)}`",
        "",
        "## 7. Reference Storm Verification",
        "",
        "| Storm ID | Found | Observations | Year | Split | Lat | Lon | Wind | Pressure | Category | Issues |",
        "|:---------|:------|:-------------|:-----|:------|:----|:----|:-----|:---------|:---------|:-------|",
    ]
    for sid, v in ref.items():
        found = "✓" if v.get("found") else "✗ NOT FOUND"
        n_obs = v.get("n_obs", 0)
        year_v = v.get("year", "—")
        split_v = v.get("split", "—")
        lat_ok = "✓" if v.get("lat_ok") else "✗"
        lon_ok = "✓" if v.get("lon_ok") else "✗"
        wind_ok = "✓" if v.get("wind_ok") else "✗"
        pres_ok = "✓" if v.get("pres_ok") else "✗"
        cat_ok = "✓" if v.get("cat_ok") else "✗"
        issues = "; ".join(v.get("issues", [])) or "None"
        lines.append(f"| `{sid}` | {found} | {n_obs} | {year_v} | {split_v} | {lat_ok} | {lon_ok} | {wind_ok} | {pres_ok} | {cat_ok} | {issues} |")

    lines += [
        "",
        "## 8. Read-Back Verification (TEST 9)",
        "",
        f"- CSV rows expected: {readback['expected_rows']}",
        f"- CSV rows read back: {readback['csv_rows']} — {'✓ MATCH' if readback['csv_match'] else '✗ MISMATCH'}",
        f"- Parquet rows read back: {readback['parquet_rows']} — {'✓ MATCH' if readback['parquet_match'] else '✗ MISMATCH'}",
        "",
        "## 9. Extraction Limitations and Unresolved Ambiguities",
        "",
        "- Scanned reports (1998–2004): OCR quality depends on scan resolution; all flagged values require human confirmation.",
        "- Storm names for scanned years may be incomplete where table headings were not machine-readable.",
        "- Narrative rows embedded in BT tables are excluded from the final dataset and recorded in imd_manual_review.csv.",
        "- Uncertain-identity tables are recorded for manual review without being included in or excluded from the final dataset silently.",
        "",
        f"**Validation Verdict: {validation_report.get('verdict', 'UNKNOWN')}**",
    ]

    cfg.EXTRACTION_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote extraction report: {cfg.EXTRACTION_REPORT_MD}")


# ── Quality report ────────────────────────────────────────────────────────────

def write_quality_report(
    df: pd.DataFrame,
    validation_report: dict,
    manual_review: list,
    run_ts: str,
) -> None:
    """Write docs/imd_best_track_quality_report.md"""
    qc = validation_report.get("quality_checks", {})
    tests = validation_report.get("tests", {})
    verdict = validation_report.get("verdict", "UNKNOWN")
    ref = validation_report.get("reference_storms", {})

    split_counts = qc.get("split_counts", {})
    mr_by_type = qc.get("manual_review_by_type", {})

    lines = [
        "# IMD Best Track Quality Report",
        f"\n**Generated:** {run_ts}",
        f"**Validation Verdict: {verdict}**",
        "",
        "---",
        "",
        "## 1. Dataset Counts",
        "",
        f"| Metric | Value |",
        "|:-------|:------|",
        f"| Total observation rows | {qc.get('total_rows', 0)} |",
        f"| Total unique storms | {qc.get('total_storms', 0)} |",
        f"| Years covered | {min(qc.get('obs_by_year', {1998: 0}).keys(), default=0)}–{max(qc.get('obs_by_year', {2024: 0}).keys(), default=0)} |",
        f"| Digital extraction rows | {qc.get('digital_rows', 0)} |",
        f"| OCR extraction rows | {qc.get('ocr_rows', 0)} |",
        f"| Manual-review rows | {qc.get('manual_review_total', 0)} |",
        "",
        "## 2. Split Counts",
        "",
        "| Split | Storms | Observations |",
        "|:------|:-------|:-------------|",
    ]
    for split in ["TRAIN", "VALIDATION", "TEST", "BLIND"]:
        sc = split_counts.get(split, {})
        lines.append(f"| {split} | {sc.get('storms', 0)} | {sc.get('observations', 0)} |")

    lines += [
        "",
        "## 3. Missing Values",
        "",
        "| Field | Missing Count |",
        "|:------|:--------------|",
    ]
    for col, n in qc.get("missing_values", {}).items():
        lines.append(f"| `{col}` | {n} |")

    lines += [
        "",
        "## 4. Quality Metrics",
        "",
        f"| Check | Count |",
        "|:------|:------|",
        f"| Duplicate observation rows | {qc.get('duplicate_observation_count', 0)} |",
        f"| Invalid coordinate rows | {qc.get('invalid_coordinate_rows', 0)} |",
        f"| Missing wind rows | {qc.get('missing_wind_rows', 0)} |",
        f"| Missing pressure rows | {qc.get('missing_pressure_rows', 0)} |",
        f"| Missing category rows | {qc.get('missing_category_rows', 0)} |",
        f"| Timestamp anomaly storms | {qc.get('timestamp_anomaly_storms', 0)} |",
        f"| Duplicate timestamps within storms | {qc.get('duplicate_timestamp_count', 0)} |",
        "",
        "## 5. Manual Review Summary",
        "",
        "| Problem Type | Count |",
        "|:-------------|:------|",
    ]
    for ptype, count in sorted(mr_by_type.items()):
        lines.append(f"| {ptype} | {count} |")

    lines += [
        "",
        "## 6. Integrity Test Results",
        "",
        "| Test | Pass/Fail | Notes |",
        "|:-----|:----------|:------|",
    ]
    for test_name, result in tests.items():
        status = "✓ PASS" if result.get("pass") else ("✗ FAIL" if result.get("pass") is False else "PENDING")
        note = ""
        if test_name == "TEST1_all_years_represented" and result.get("missing_years"):
            note = f"Missing: {result['missing_years']}"
        elif test_name == "TEST5_no_storm_in_multiple_splits" and result.get("multi_split_storms"):
            note = f"Storms: {result['multi_split_storms']}"
        elif test_name == "TEST6_no_duplicate_rows":
            note = f"Duplicates: {result.get('duplicate_count', 0)}"
        elif test_name == "TEST8_reference_storms_present":
            missing_ref = [sid for sid, v in ref.items() if not v.get("found")]
            note = f"Missing: {missing_ref}" if missing_ref else "All 5 found"
        lines.append(f"| {test_name} | {status} | {note} |")

    lines += [
        "",
        "## 7. Reference Storm Details",
        "",
        "| Storm ID | Found | Obs | Issues |",
        "|:---------|:------|:----|:-------|",
    ]
    for sid, v in ref.items():
        found = "✓" if v.get("found") else "✗ NOT FOUND"
        n_obs = v.get("n_obs", 0)
        issues = "; ".join(v.get("issues", [])) or "None"
        lines.append(f"| `{sid}` | {found} | {n_obs} | {issues} |")

    lines += [
        "",
        "## 8. Critical Failures",
        "",
    ]
    if validation_report.get("critical_failures"):
        for cf in validation_report["critical_failures"]:
            lines.append(f"- ❌ {cf}")
    else:
        lines.append("- None")

    lines += [
        "",
        "## 9. Warnings",
        "",
    ]
    if validation_report.get("warnings"):
        for w in validation_report["warnings"]:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("- None")

    lines += [
        "",
        f"---",
        f"\n## Final Status: **{verdict}**",
        "",
    ]
    if verdict == "PASS":
        lines.append("> All 14 integrity tests passed with no critical issues or warnings.")
    elif verdict == "PASS WITH WARNINGS":
        lines.append("> Pipeline completed. Non-critical issues require attention but do not block the dataset.")
        lines.append("> See manual review CSV for items requiring human confirmation.")
    else:
        lines.append("> ❌ BLOCKED: Critical failures prevent the dataset from being used.")

    cfg.QUALITY_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote quality report: {cfg.QUALITY_REPORT_MD}")


def run_outputs(df: pd.DataFrame, manifest_df: pd.DataFrame,
                manual_review: list, digital_results: dict, scanned_results: dict,
                validation_report: dict, year_file_map: dict) -> dict:
    """Write all final output files. Returns summary dict."""
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Write storm manifest
    manifest_df.to_csv(cfg.STORM_MANIFEST_CSV, index=False, encoding="utf-8")
    logger.info(f"Wrote storm manifest: {cfg.STORM_MANIFEST_CSV} ({len(manifest_df)} storms)")

    # Write manual review CSV
    n_mr = write_manual_review_csv(manual_review)

    # Write final datasets
    readback = write_final_datasets(df)

    # Update TEST 9
    validation_report["tests"]["TEST9_csv_parquet_readback"] = {
        "pass": readback["csv_match"] and readback["parquet_match"],
        "csv_rows": readback["csv_rows"],
        "parquet_rows": readback["parquet_rows"],
    }

    # Write reports
    write_extraction_report(
        df, digital_results, scanned_results, validation_report,
        manual_review, year_file_map, readback, run_ts
    )
    write_quality_report(df, validation_report, manual_review, run_ts)

    return {
        "readback": readback,
        "n_manual_review": n_mr,
        "n_storms": len(manifest_df),
        "run_ts": run_ts,
    }
