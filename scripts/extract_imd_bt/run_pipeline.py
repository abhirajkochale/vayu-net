"""
run_pipeline.py — VAYU-NET IMD Best Track Master Pipeline Runner
Executes all steps 00–05 in sequence with full logging.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parents[2] / "data" / "interim" / "imd" / "pipeline.log",
            mode="w", encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("pipeline")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import importlib
cfg = importlib.import_module("00_config")

from importlib import import_module
digital_extractor = import_module("01_digital_extractor")
scanned_extractor = import_module("02_scanned_extractor")
normalizer = import_module("03_normalizer")
validator = import_module("04_validator")
outputs = import_module("05_outputs")


def main():
    start = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("VAYU-NET IMD Best Track Extraction Pipeline — START")
    logger.info(f"Start time: {start.isoformat()}")
    logger.info("=" * 70)

    # ── Step 0: Load inventory ─────────────────────────────────────────────
    logger.info("STEP 0: Loading source inventory…")
    year_file_map = cfg.load_year_file_map()
    logger.info(f"  {len(year_file_map)} canonical reports loaded from inventory CSV")
    for year in sorted(year_file_map.keys()):
        logger.info(f"  {year}: {year_file_map[year]['filename']}")

    # ── Step 1: Digital extraction (2005–2024) ────────────────────────────
    logger.info("\nSTEP 1: Digital PDF extraction (2005–2024)…")
    digital_results = digital_extractor.run_digital_extraction(year_file_map)
    total_digital_obs = sum(r.get("total_observation_rows", 0) for r in digital_results.values())
    total_digital_tables = sum(len(r.get("tables_found", [])) for r in digital_results.values())
    logger.info(f"  Digital: {total_digital_tables} BT tables, {total_digital_obs} observation rows")

    # ── Step 2: Scanned extraction (1998–2004) ────────────────────────────
    logger.info("\nSTEP 2: Scanned PDF OCR extraction (1998–2004)…")
    scanned_results = scanned_extractor.run_scanned_extraction(year_file_map)
    total_scanned_obs = sum(r.get("total_observation_rows", 0) for r in scanned_results.values())
    logger.info(f"  Scanned: {total_scanned_obs} observation rows")

    # ── Step 3: Normalization ─────────────────────────────────────────────
    logger.info("\nSTEP 3: Normalization…")
    df, all_manual_review, all_narrative = normalizer.run_normalization(
        digital_results, scanned_results
    )
    logger.info(f"  Normalized: {len(df)} records, {df['storm_id'].nunique()} storms")

    # ── Step 4: Validation ────────────────────────────────────────────────
    logger.info("\nSTEP 4: Validation…")
    validation_report = validator.run_validation(df, year_file_map, all_manual_review)
    logger.info(f"  Verdict: {validation_report['verdict']}")
    for cf in validation_report.get("critical_failures", []):
        logger.error(f"  CRITICAL FAILURE: {cf}")
    for w in validation_report.get("warnings", []):
        logger.warning(f"  WARNING: {w}")

    # ── Step 5: Build storm manifest ──────────────────────────────────────
    logger.info("\nSTEP 5: Building storm event manifest…")
    manifest_df = outputs.build_storm_manifest(df)
    logger.info(f"  Storm manifest: {len(manifest_df)} storms")

    # ── Step 6: Write all outputs ─────────────────────────────────────────
    logger.info("\nSTEP 6: Writing output files…")
    output_summary = outputs.run_outputs(
        df, manifest_df, all_manual_review,
        digital_results, scanned_results,
        validation_report, year_file_map
    )

    # ── Final summary ─────────────────────────────────────────────────────
    end = datetime.now(timezone.utc)
    elapsed = (end - start).total_seconds()
    qc = validation_report.get("quality_checks", {})
    ref = validation_report.get("reference_storms", {})
    split_counts = qc.get("split_counts", {})

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info("")
    logger.info(f"TOTAL STORMS:        {qc.get('total_storms', 0)}")
    logger.info(f"TOTAL OBSERVATIONS:  {qc.get('total_rows', 0)}")
    logger.info("")
    logger.info(f"TRAIN STORMS:        {split_counts.get('TRAIN', {}).get('storms', 0)}")
    logger.info(f"TRAIN OBSERVATIONS:  {split_counts.get('TRAIN', {}).get('observations', 0)}")
    logger.info(f"VALIDATION STORMS:   {split_counts.get('VALIDATION', {}).get('storms', 0)}")
    logger.info(f"VALIDATION OBS:      {split_counts.get('VALIDATION', {}).get('observations', 0)}")
    logger.info(f"TEST STORMS:         {split_counts.get('TEST', {}).get('storms', 0)}")
    logger.info(f"TEST OBSERVATIONS:   {split_counts.get('TEST', {}).get('observations', 0)}")
    logger.info("")
    logger.info(f"DIGITAL ROWS:        {qc.get('digital_rows', 0)}")
    logger.info(f"OCR ROWS:            {qc.get('ocr_rows', 0)}")
    logger.info(f"MANUAL REVIEW ROWS:  {qc.get('manual_review_total', 0)}")
    logger.info(f"DUPLICATE ROWS:      {qc.get('duplicate_observation_count', 0)}")
    logger.info(f"MISSING WIND:        {qc.get('missing_wind_rows', 0)}")
    logger.info(f"MISSING PRESSURE:    {qc.get('missing_pressure_rows', 0)}")
    logger.info(f"MISSING CATEGORY:    {qc.get('missing_category_rows', 0)}")
    logger.info(f"TIMESTAMP ANOMALIES: {qc.get('timestamp_anomaly_storms', 0)}")
    logger.info("")
    for sid, v in ref.items():
        status = "FOUND" if v.get("found") else "NOT FOUND"
        logger.info(f"{sid}: {status} ({v.get('n_obs', 0)} obs, issues: {v.get('issues', [])})")
    logger.info("")
    logger.info(f"VALIDATION VERDICT:  {validation_report['verdict']}")
    logger.info("=" * 70)

    # Write final summary JSON for programmatic use
    summary_path = cfg.INTERIM_IMD_DIR / "pipeline_summary.json"
    summary = {
        "run_ts": output_summary["run_ts"],
        "elapsed_seconds": elapsed,
        "total_storms": qc.get("total_storms", 0),
        "total_observations": qc.get("total_rows", 0),
        "split_counts": split_counts,
        "obs_by_year": qc.get("obs_by_year", {}),
        "digital_rows": qc.get("digital_rows", 0),
        "ocr_rows": qc.get("ocr_rows", 0),
        "manual_review_rows": qc.get("manual_review_total", 0),
        "duplicate_rows": qc.get("duplicate_observation_count", 0),
        "missing_wind": qc.get("missing_wind_rows", 0),
        "missing_pressure": qc.get("missing_pressure_rows", 0),
        "missing_category": qc.get("missing_category_rows", 0),
        "timestamp_anomalies": qc.get("timestamp_anomaly_storms", 0),
        "reference_storms": {sid: {"found": v.get("found"), "n_obs": v.get("n_obs"), "issues": v.get("issues", [])} for sid, v in ref.items()},
        "validation_verdict": validation_report["verdict"],
        "critical_failures": validation_report.get("critical_failures", []),
        "warnings": validation_report.get("warnings", []),
        "readback_csv": output_summary["readback"]["csv_match"],
        "readback_parquet": output_summary["readback"]["parquet_match"],
        "tests": {k: v.get("pass") for k, v in validation_report.get("tests", {}).items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Pipeline summary JSON: {summary_path}")

    return summary


if __name__ == "__main__":
    main()
