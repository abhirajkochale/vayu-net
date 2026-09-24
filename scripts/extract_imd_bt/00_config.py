"""
00_config.py — VAYU-NET IMD Best Track Pipeline Configuration
All constants, schema, year→file mapping, split assignments, and header normalization.
No hardcoded filenames; inventory CSV is the single source of truth for year→file mapping.
"""

import csv
import os
from pathlib import Path

# ── Repository paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_IMD_DIR = REPO_ROOT / "data" / "raw" / "imd"
INTERIM_IMD_DIR = REPO_ROOT / "data" / "interim" / "imd"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
DOCS_DIR = REPO_ROOT / "docs"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "extract_imd_bt"

# ── Source inventory ──────────────────────────────────────────────────────────
INVENTORY_CSV = MANIFESTS_DIR / "imd_source_inventory.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
BEST_TRACK_CSV = PROCESSED_DIR / "imd_best_track.csv"
BEST_TRACK_PARQUET = PROCESSED_DIR / "imd_best_track.parquet"
STORM_MANIFEST_CSV = MANIFESTS_DIR / "storm_event_manifest.csv"
MANUAL_REVIEW_CSV = MANIFESTS_DIR / "imd_manual_review.csv"
EXTRACTION_REPORT_MD = DOCS_DIR / "imd_best_track_extraction_report.md"
QUALITY_REPORT_MD = DOCS_DIR / "imd_best_track_quality_report.md"

# ── ML split assignments (storm-level, chronological) ─────────────────────────
def get_split(year: int) -> str:
    """Return the frozen ML split for a given report year."""
    if 1998 <= year <= 2018:
        return "TRAIN"
    elif year in (2019, 2020):
        return "VALIDATION"
    elif 2021 <= year <= 2024:
        return "TEST"
    elif year == 2025:
        return "BLIND"
    elif year == 1997:
        return "HISTORICAL_SUPPLEMENT"
    else:
        return "UNKNOWN"

# ── Core ML years ─────────────────────────────────────────────────────────────
CORE_YEARS = list(range(1998, 2025))      # 1998–2024 inclusive
DIGITAL_YEARS = list(range(2005, 2025))   # 2005–2024
SCANNED_YEARS = list(range(1998, 2005))   # 1998–2004

# ── Load year→file mapping from inventory CSV ─────────────────────────────────
def load_year_file_map() -> dict:
    """
    Returns {year: {'filename': str, 'sha256': str, 'report_title': str,
                    'source_type': str, 'page_count': int}}
    Uses actual_report_year from the inventory (content-verified, not filename).
    Only loads CORE_BEST_TRACK_SOURCE records.
    """
    mapping = {}
    with open(INVENTORY_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            role = row.get("recommended_role", "").strip()
            if role not in ("CORE_BEST_TRACK_SOURCE",):
                continue
            try:
                year = int(row["actual_report_year"])
            except (ValueError, KeyError):
                continue
            mapping[year] = {
                "filename": row["filename"].strip(),
                "sha256": row["sha256"].strip(),
                "report_title": row["report_title"].strip(),
                "source_type": row["source_type"].strip(),
                "page_count": int(row.get("page_count", 0) or 0),
                "best_track_page_ranges": row.get("best_track_page_ranges", "").strip(),
            }
    return mapping

# ── Canonical schema field names ──────────────────────────────────────────────
CANONICAL_FIELDS = [
    # Normalized observation fields
    "storm_id",
    "storm_name",
    "source_storm_name",
    "year",
    "timestamp_utc",
    "latitude",
    "longitude",
    "maximum_sustained_wind_kt",
    "central_pressure_hpa",
    "category",
    "split",
    # Provenance fields
    "source_file",
    "source_report",
    "source_page",
    "source_table",
    "source_row",
    "extraction_method",
    # Quality fields
    "quality_flag",
    "manual_review_required",
    "manual_review_reason",
    # Raw source fields
    "raw_date",
    "raw_time",
    "raw_latitude",
    "raw_longitude",
    "raw_wind",
    "raw_pressure",
    "raw_category",
    "raw_ci_no",
    "raw_pressure_drop",
]

MANUAL_REVIEW_FIELDS = [
    "year",
    "storm_id",
    "source_file",
    "source_page",
    "source_table",
    "source_row",
    "raw_text",
    "problem_type",
    "problem_description",
    "recommended_action",
    "resolved",
]

# ── Column-header normalization map ───────────────────────────────────────────
# Maps cleaned header strings → canonical raw field names
# Keys are lowercased, whitespace-collapsed header strings.
HEADER_TO_FIELD = {
    # Date
    "date": "raw_date",
    # Time / UTC
    "time": "raw_time",
    "time(utc)": "raw_time",
    "time\n(utc)": "raw_time",
    "time\n( utc)": "raw_time",
    "time( utc)": "raw_time",
    "time (utc)": "raw_time",
    "utc": "raw_time",
    "time\nutc": "raw_time",
    # Latitude variants
    "lat.": "raw_latitude",
    "lat": "raw_latitude",
    "lat.(°n)": "raw_latitude",
    "lat. (°n)": "raw_latitude",
    "lat.(on)": "raw_latitude",
    "centre lat.0n/ long. 0 e": "raw_latitude",   # split col 0
    "centre lat.0\nn/ long. 0 e": "raw_latitude",
    "centre\nlat.0 n/\nlong. 0 e": "raw_latitude",
    "centre\nlat.0n/\nlong. 0 e": "raw_latitude",
    "lat.0n": "raw_latitude",
    "latitude": "raw_latitude",
    "centre lat.0n/long. 0 e": "raw_latitude",
    "centre lat.0 n/ long. 0 e": "raw_latitude",
    # Longitude variants
    "long.": "raw_longitude",
    "long": "raw_longitude",
    "lon": "raw_longitude",
    "long.(°e)": "raw_longitude",
    "long. (°e)": "raw_longitude",
    "longitude": "raw_longitude",
    "long.0e": "raw_longitude",
    # Wind variants
    "msw": "raw_wind",
    "msw(kt)": "raw_wind",
    "msw (kt)": "raw_wind",
    "estimated maximum sustained surface wind (kt)": "raw_wind",
    "estimated\nmaximum\nsustained\nsurface\nwind (kt)": "raw_wind",
    "estimatedmaximum sustained surface wind (kt)": "raw_wind",
    "estimated maximum sustainedsurface wind (kt)": "raw_wind",
    "estimated maximumsustained surface wind (kt)": "raw_wind",
    "estimatedmaximumsustained surface wind (kt)": "raw_wind",
    "estimated maximum wind speed (kt)": "raw_wind",
    "estimated maximum sustained wind speed (kt)": "raw_wind",
    "maximum sustained wind speed (kt)": "raw_wind",
    "maximum sustained wind (kt)": "raw_wind",
    "estimated\nmaximum\nsustained\nsurface\nwind(kt)": "raw_wind",
    # Pressure variants
    "ecp(hpa)": "raw_pressure",
    "ecp (hpa)": "raw_pressure",
    "ecp": "raw_pressure",
    "estimated central pressure (hpa)": "raw_pressure",
    "estimated\ncentral\npressure\n(hpa)": "raw_pressure",
    "estimatedcentral pressure (hpa)": "raw_pressure",
    "estimated centralpressure (hpa)": "raw_pressure",
    "estimated central pressure": "raw_pressure",
    "central pressure (hpa)": "raw_pressure",
    "cp": "raw_pressure",
    "c.p.": "raw_pressure",
    "cp (hpa)": "raw_pressure",
    "estimatedd central pressure (hpa)": "raw_pressure",
    "estimatedd\ncentral\npressure\n(hpa)": "raw_pressure",
    "estimated d central pressure (hpa)": "raw_pressure",
    # Grade / Category
    "grade": "raw_category",
    "category": "raw_category",
    "imd grade": "raw_category",
    # C.I. No / T.No (ancillary)
    "c.i. no": "raw_ci_no",
    "c.i. no.": "raw_ci_no",
    "c.i.no.": "raw_ci_no",
    "c.i.\nno.": "raw_ci_no",
    "c.i.\nno": "raw_ci_no",
    "ci no": "raw_ci_no",
    "ci no.": "raw_ci_no",
    "t.no.": "raw_ci_no",
    "t. no.": "raw_ci_no",
    "t.no": "raw_ci_no",
    # Pressure drop (ancillary)
    "∆p": "raw_pressure_drop",
    "δp": "raw_pressure_drop",
    "dp": "raw_pressure_drop",
    "estimated pressure drop at the centre (hpa)": "raw_pressure_drop",
    "pressure drop": "raw_pressure_drop",
    "pressure\ndrop": "raw_pressure_drop",
    "estimatedpressure drop at the centre (hpa)": "raw_pressure_drop",
    "estimated pressuredrop at the centre (hpa)": "raw_pressure_drop",
    "estimated\npressure\ndrop at the\ncentre\n(hpa)": "raw_pressure_drop",
    "estimated\npressure\ndrop at\nthe centre\n(hpa)": "raw_pressure_drop",
}


# Required field groups — all must be present for a table to qualify as BT observation
REQUIRED_BT_FIELD_GROUPS = {
    "date":     {"raw_date"},
    "time":     {"raw_time"},
    "latitude": {"raw_latitude"},
    "longitude": {"raw_longitude"},
    "wind":     {"raw_wind"},
    "pressure": {"raw_pressure"},
    "grade":    {"raw_category"},
}

# ── Best Track heading keywords ────────────────────────────────────────────────
BT_HEADING_KEYWORDS = [
    "best track",
    "best-track",
    "besttrack",
]

# ── Non-Best-Track rejection signatures ───────────────────────────────────────
NON_BT_HEADING_KEYWORDS = [
    "forecast track",
    "forecast position",
    "forecast verification",
    "realised wind",
    "realized wind",
    "warning bulletin",
    "advisory",
    "model guidance",
    "nwp",
    "predicted",
    "realised rainfall",
    "realised",
]

SUMMARY_TABLE_HEADER_SIGNATURES = [
    "date, time & lat",
    "date, time (utc) place of landfall",
    "date, time\n(utc) place of landfall",
    "date, time &\nlat",
    "date, time\n& lat",
]

# ── Narrative row detection keywords ──────────────────────────────────────────
NARRATIVE_KEYWORDS = [
    "crossed", "weakened", "landfall", "dissipated", "entered", "emerged",
    "moved", "close to", "between", "made landfall", "crossed coast",
    "weakening into", "intensified into", "further weakened",
    "concentrated into", "lay as",
]

# ── OCR ambiguity character pairs ─────────────────────────────────────────────
OCR_AMBIGUOUS_PAIRS = [
    ("O", "0"), ("I", "1"), ("l", "1"), ("S", "5"), ("B", "8"), ("G", "6"),
]
