"""
03_normalizer.py — VAYU-NET IMD Best Track Pipeline
Converts raw extraction JSONL into normalized observation records.

Rules:
- Raw values are NEVER silently changed.
- Suspicious values → normalized field = NaN, manual_review_required = True.
- Hemisphere read from source notation only; never inferred from magnitude.
- Timestamp = native IMD observation time; no resampling.
- Storm ID = NIO_{YEAR}_{NAME} for named, NIO_{YEAR}_UNNAMED_{id} for unnamed.
"""

import json
import re
import sys
import logging
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib
cfg = importlib.import_module("00_config")

logger = logging.getLogger("normalizer")

# ── Timestamp parsing ─────────────────────────────────────────────────────────

DATE_FORMATS = [
    "%d/%m/%Y", "%d/%m/%y",
    "%d.%m.%Y", "%d.%m.%y",
    "%d-%m-%Y", "%d-%m-%y",
    "%d/%m/%Y.", "%d.%m.%Y.",
]


def _parse_date(raw_date: str) -> tuple:
    """
    Returns (date_obj_or_None, parse_error_str_or_None)
    """
    if not raw_date:
        return None, "Missing date"
    s = raw_date.strip().rstrip(".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date(), None
        except ValueError:
            pass
    return None, f"Unrecognized date format: '{raw_date}'"


def _parse_time(raw_time: str) -> tuple:
    """
    Parse UTC time string like '0000', '0300', '1200'.
    Returns (hour: int, minute: int, error_str_or_None)
    """
    if not raw_time:
        return None, None, "Missing time"
    s = str(raw_time).strip()
    # 4-digit HHMM
    if re.match(r"^\d{4}$", s):
        h, m = int(s[:2]), int(s[2:])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m, None
        return None, None, f"Invalid time value: '{s}'"
    # 3-digit HMM
    if re.match(r"^\d{3}$", s):
        h, m = int(s[0]), int(s[1:])
        if 0 <= h <= 9 and 0 <= m <= 59:
            return h, m, None
    # 2-digit HH (e.g. '03', '06', '12' -> 03:00, 06:00)
    if re.match(r"^\d{1,2}$", s):
        h = int(s)
        if 0 <= h <= 23:
            return h, 0, None
    # HH:MM
    m2 = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m2:
        h, m = int(m2.group(1)), int(m2.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m, None
    return None, None, f"Unrecognized time format: '{raw_time}'"


def _build_timestamp(raw_date: str, raw_time: str) -> tuple:
    """
    Returns (timestamp_utc: datetime or None, error: str or None)
    """
    date_obj, date_err = _parse_date(raw_date)
    if date_err:
        return None, date_err
    hour, minute, time_err = _parse_time(raw_time)
    if time_err:
        return None, time_err
    try:
        ts = datetime(date_obj.year, date_obj.month, date_obj.day,
                      hour, minute, 0, tzinfo=timezone.utc)
        return ts, None
    except Exception as e:
        return None, str(e)

# ── Coordinate parsing ────────────────────────────────────────────────────────

def _parse_coordinate(raw_val: str, hemisphere_hint: str = None) -> tuple:
    """
    Parse a coordinate value. Hemisphere is read from source notation.
    Returns (value: float or None, error: str or None)
    raw_val examples: '14.5', '14.5N', '82.7E', '14.5 N'
    hemisphere_hint: column name context ('raw_latitude' or 'raw_longitude')
    """
    if not raw_val:
        return None, "Missing coordinate"
    s = str(raw_val).strip()

    # Extract hemisphere letter if present
    hemi_match = re.search(r"([NSEWnsew])\s*$", s)
    hemi = None
    num_str = s
    if hemi_match:
        hemi = hemi_match.group(1).upper()
        num_str = s[:hemi_match.start()].strip()

    try:
        val = float(num_str.replace(",", "."))
    except ValueError:
        return None, f"Non-numeric coordinate: '{raw_val}'"

    # Apply hemisphere sign — ONLY from source notation, never from magnitude
    if hemi == "S":
        val = -abs(val)
    elif hemi == "W":
        val = -abs(val)
    elif hemi in ("N", "E"):
        val = abs(val)
    # If no hemisphere letter in value, leave sign as parsed
    # (for scanned reports the sign must come from the column header context)

    return val, None

# ── Wind / Pressure parsing ───────────────────────────────────────────────────

def _parse_numeric(raw_val: str, field_name: str) -> tuple:
    """
    Parse a numeric field. Returns (value: float or None, error: str or None).
    Never corrects; flags suspicious values.
    """
    if raw_val is None or raw_val == "" or raw_val == "-":
        return None, None  # explicitly missing, not an error
    s = str(raw_val).strip()
    if s in ("-", "–", "—", "NA", "N/A", "na", "n/a", "NR", "nr"):
        return None, None
    try:
        return float(s.replace(",", ".")), None
    except ValueError:
        return None, f"Non-numeric {field_name}: '{raw_val}'"

# ── Storm ID construction ─────────────────────────────────────────────────────

def _normalize_name(raw_name: str) -> str:
    """Normalize storm name for storm_id: uppercase, spaces→underscores, ASCII."""
    if not raw_name:
        return None
    # Remove diacritics
    name = unicodedata.normalize("NFD", raw_name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Uppercase, strip quotes and special chars
    name = re.sub(r"['\"\u2018\u2019\u201c\u201d\u0060\u00b4]", "", name)
    name = name.upper().strip()
    # Replace spaces and dashes with underscore
    name = re.sub(r"[\s\-]+", "_", name)
    # Keep only alphanumeric and underscore
    name = re.sub(r"[^A-Z0-9_]", "", name)
    return name or None


def _make_storm_id(year: int, storm_name_raw: str, table_id: str) -> tuple:
    """
    Returns (storm_id: str, storm_name: str)
    Named: NIO_{YEAR}_{NORMALIZED_NAME}
    Unnamed: NIO_{YEAR}_UNNAMED_{table_derived_id}
    """
    norm_name = _normalize_name(storm_name_raw) if storm_name_raw else None

    # Filter out generic words that would produce false names
    GENERIC = {"OVER", "THE", "BAY", "OF", "BENGAL", "ARABIAN", "SEA",
               "NORTH", "INDIAN", "OCEAN", "RSMC", "BEST", "TRACK",
               "POSITIONS", "AND", "PARAMETERS", "DURING", "TABLE"}
    if norm_name and all(part in GENERIC for part in norm_name.split("_")):
        norm_name = None

    if norm_name:
        storm_id = f"NIO_{year}_{norm_name}"
        return storm_id, norm_name
    else:
        # Build deterministic unnamed ID from table_id
        if table_id:
            # "Table 2.3.1" → "2_3_1"
            tbl_nums = re.findall(r"\d+", table_id)
            unnamed_id = "_".join(tbl_nums) if tbl_nums else "UNK"
        else:
            unnamed_id = "UNK"
        storm_id = f"NIO_{year}_UNNAMED_{unnamed_id}"
        return storm_id, None

# ── Per-record normalization ──────────────────────────────────────────────────

def _normalize_record(raw_rec: dict, year: int, source_file: str, source_report: str,
                       source_page: int, source_table: str, storm_name_raw: str,
                       table_id: str, extraction_method: str) -> dict:
    """
    Convert one raw extracted record to the canonical schema.
    Returns a dict with all CANONICAL_FIELDS.
    """
    col_data = raw_rec.get("col_data", {})
    source_row = raw_rec.get("source_row", None)

    raw_date = col_data.get("raw_date") or raw_rec.get("raw_date")
    raw_time = col_data.get("raw_time")
    raw_lat = col_data.get("raw_latitude")
    raw_lon = col_data.get("raw_longitude")
    raw_wind = col_data.get("raw_wind")
    raw_pressure = col_data.get("raw_pressure")
    raw_category = col_data.get("raw_category")
    raw_ci_no = col_data.get("raw_ci_no")
    raw_pdrop = col_data.get("raw_pressure_drop")

    # Initialize review tracking
    review_reasons = []
    if raw_rec.get("manual_review_required"):
        review_reasons.append(raw_rec.get("manual_review_reason", ""))

    # ── Timestamp ───────────────────────────────────────────────────────────
    ts_utc, ts_err = _build_timestamp(raw_date, raw_time)
    if ts_err:
        ts_utc = None
        review_reasons.append(f"TIMESTAMP: {ts_err}")

    # ── Coordinates ─────────────────────────────────────────────────────────
    lat, lat_err = _parse_coordinate(raw_lat, "raw_latitude")
    if lat_err:
        lat = None
        review_reasons.append(f"LATITUDE: {lat_err}")
    elif lat is not None:
        if not (-90 <= lat <= 90):
            review_reasons.append(f"LATITUDE_OOR: normalized={lat} outside [-90,+90]")
            lat = None

    lon, lon_err = _parse_coordinate(raw_lon, "raw_longitude")
    if lon_err:
        lon = None
        review_reasons.append(f"LONGITUDE: {lon_err}")
    elif lon is not None:
        if not (-180 <= lon <= 180):
            review_reasons.append(f"LONGITUDE_OOR: normalized={lon} outside [-180,+180]")
            lon = None

    # ── Wind ────────────────────────────────────────────────────────────────
    wind_kt, wind_err = _parse_numeric(raw_wind, "wind_kt")
    if wind_err:
        wind_kt = None
        review_reasons.append(f"WIND: {wind_err}")
    elif wind_kt is not None and (wind_kt < 10 or wind_kt > 250):
        review_reasons.append(f"WIND_PLAUSIBILITY: value={wind_kt} kt is outside [10,250]")
        wind_kt = None

    # ── Pressure ────────────────────────────────────────────────────────────
    pres_hpa, pres_err = _parse_numeric(raw_pressure, "pressure_hpa")
    if pres_err:
        pres_hpa = None
        review_reasons.append(f"PRESSURE: {pres_err}")
    elif pres_hpa is not None and (pres_hpa < 850 or pres_hpa > 1030):
        review_reasons.append(f"PRESSURE_PLAUSIBILITY: value={pres_hpa} hPa outside [850,1030]")
        pres_hpa = None

    # ── Category ────────────────────────────────────────────────────────────
    category = raw_category.strip() if raw_category else None

    # ── Storm ID ────────────────────────────────────────────────────────────
    storm_id, storm_name_norm = _make_storm_id(year, storm_name_raw, table_id)

    # ── Quality flag ────────────────────────────────────────────────────────
    manual_review = len(review_reasons) > 0
    if ts_utc is None:
        quality_flag = "MISSING_TIMESTAMP"
    elif lat is None or lon is None:
        quality_flag = "MISSING_COORDINATES"
    elif manual_review:
        quality_flag = "REVIEW_REQUIRED"
    else:
        quality_flag = "OK"

    record = {
        "storm_id": storm_id,
        "storm_name": storm_name_norm or storm_name_raw,
        "source_storm_name": storm_name_raw,
        "year": year,
        "timestamp_utc": ts_utc.isoformat() if ts_utc else None,
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "maximum_sustained_wind_kt": float(wind_kt) if wind_kt is not None else None,
        "central_pressure_hpa": float(pres_hpa) if pres_hpa is not None else None,
        "category": category,
        "split": cfg.get_split(year),
        "source_file": source_file,
        "source_report": source_report,
        "source_page": source_page,
        "source_table": source_table,
        "source_row": source_row,
        "extraction_method": extraction_method,
        "quality_flag": quality_flag,
        "manual_review_required": manual_review,
        "manual_review_reason": "; ".join(r for r in review_reasons if r),
        "raw_date": raw_date,
        "raw_time": raw_time,
        "raw_latitude": raw_lat,
        "raw_longitude": raw_lon,
        "raw_wind": raw_wind,
        "raw_pressure": raw_pressure,
        "raw_category": raw_category,
        "raw_ci_no": raw_ci_no,
        "raw_pressure_drop": raw_pdrop,
    }
    return record


def _build_manual_review_entry(rec: dict, problem_type: str, description: str) -> dict:
    return {
        "year": rec.get("year"),
        "storm_id": rec.get("storm_id"),
        "source_file": rec.get("source_file"),
        "source_page": rec.get("source_page"),
        "source_table": rec.get("source_table"),
        "source_row": rec.get("source_row"),
        "raw_text": (
            f"date={rec.get('raw_date')} time={rec.get('raw_time')} "
            f"lat={rec.get('raw_latitude')} lon={rec.get('raw_longitude')} "
            f"wind={rec.get('raw_wind')} pres={rec.get('raw_pressure')} "
            f"cat={rec.get('raw_category')}"
        ),
        "problem_type": problem_type,
        "problem_description": description,
        "recommended_action": "HUMAN_REVIEW",
        "resolved": False,
    }


# ── Digital results normalization ─────────────────────────────────────────────

def normalize_digital_results(digital_results: dict) -> tuple:
    """
    Returns (records: list[dict], manual_review_entries: list[dict], narrative_entries: list[dict])
    """
    all_records = []
    manual_review_entries = []
    narrative_entries = []

    for year in sorted(digital_results.keys()):
        result = digital_results[year]
        source_file = result.get("source_file", "")
        source_report = result.get("source_report", "")

        # Handle uncertain tables
        for ut in result.get("uncertain_tables", []):
            manual_review_entries.append({
                "year": year,
                "storm_id": None,
                "source_file": source_file,
                "source_page": ut.get("page"),
                "source_table": None,
                "source_row": None,
                "raw_text": f"header={ut.get('header')}",
                "problem_type": "UNCERTAIN_TABLE_IDENTITY",
                "problem_description": ut.get("manual_review_reason", ""),
                "recommended_action": "HUMAN_REVIEW",
                "resolved": False,
            })

        # Process confirmed BT tables
        for tbl in result.get("tables_found", []):
            storm_name_raw = tbl.get("storm_name_raw")
            table_id = tbl.get("table_id")
            source_page = tbl.get("page")
            ext_method = "DIGITAL_TABLE"

            # Narrative rows → manual review
            for narr in tbl.get("narrative_rows", []):
                narrative_entries.append({
                    "year": year,
                    "storm_id": None,
                    "source_file": source_file,
                    "source_page": source_page,
                    "source_table": table_id,
                    "source_row": narr.get("source_row"),
                    "raw_text": narr.get("raw_text", ""),
                    "problem_type": "NON_OBSERVATION_NARRATIVE_ROW",
                    "problem_description": narr.get("reason", ""),
                    "recommended_action": "EXCLUDE_UNLESS_CONFIRMED",
                    "resolved": False,
                })

            # Normalize observation rows
            for raw_rec in tbl.get("observation_rows", []):
                rec = _normalize_record(
                    raw_rec, year, source_file, source_report,
                    source_page, table_id, storm_name_raw, table_id, ext_method
                )
                all_records.append(rec)
                if rec["manual_review_required"]:
                    manual_review_entries.append(
                        _build_manual_review_entry(
                            rec,
                            problem_type=rec.get("quality_flag", "REVIEW"),
                            description=rec.get("manual_review_reason", ""),
                        )
                    )

    return all_records, manual_review_entries, narrative_entries


# ── OCR results normalization ─────────────────────────────────────────────────

def normalize_scanned_results(scanned_results: dict) -> tuple:
    """
    Returns (records, manual_review_entries, narrative_entries)
    """
    all_records = []
    manual_review_entries = []
    narrative_entries = []

    for year in sorted(scanned_results.keys()):
        result = scanned_results[year]
        source_file = result.get("source_file", "")
        source_report = result.get("source_report", "")

        for page_result in result.get("pages_processed", []):
            source_page = page_result.get("page")
            ext_method = page_result.get("extraction_method", "OCR")
            table_id = f"OCR_PAGE_{source_page}"

            # OCR_PENDING pages
            if page_result.get("status") == "OCR_PENDING":
                manual_review_entries.append({
                    "year": year,
                    "storm_id": None,
                    "source_file": source_file,
                    "source_page": source_page,
                    "source_table": table_id,
                    "source_row": None,
                    "raw_text": "",
                    "problem_type": "OCR_PENDING",
                    "problem_description": page_result.get("manual_review_reason", ""),
                    "recommended_action": "MANUAL_OCR_REQUIRED",
                    "resolved": False,
                })
                continue

            # Narrative rows
            for narr in page_result.get("narrative_rows", []):
                narrative_entries.append({
                    "year": year,
                    "storm_id": None,
                    "source_file": source_file,
                    "source_page": source_page,
                    "source_table": table_id,
                    "source_row": narr.get("source_row"),
                    "raw_text": narr.get("raw_text", ""),
                    "problem_type": "NON_OBSERVATION_NARRATIVE_ROW",
                    "problem_description": narr.get("reason", ""),
                    "recommended_action": "EXCLUDE_UNLESS_CONFIRMED",
                    "resolved": False,
                })

            # Observation rows (OCR)
            for raw_rec in page_result.get("observation_rows", []):
                rec = _normalize_record(
                    raw_rec, year, source_file, source_report,
                    source_page, table_id,
                    storm_name_raw=None,   # scanned: resolved by sequence
                    table_id=table_id,
                    extraction_method=ext_method,
                )
                all_records.append(rec)
                if rec["manual_review_required"]:
                    manual_review_entries.append(
                        _build_manual_review_entry(
                            rec,
                            problem_type=rec.get("quality_flag", "OCR_REVIEW"),
                            description=rec.get("manual_review_reason", ""),
                        )
                    )

    return all_records, manual_review_entries, narrative_entries


# ── Storm grouping and deduplication ─────────────────────────────────────────

def _assign_storm_ids_within_year(records: list) -> list:
    """
    For scanned years, records may have no storm name.
    Group consecutive records by proximity (same storm if timestamps within 20 days)
    and assign sequential storm IDs within year.
    """
    # Group by year
    by_year = {}
    for r in records:
        by_year.setdefault(r["year"], []).append(r)

    result = []
    for year, year_recs in sorted(by_year.items()):
        # Sort by timestamp
        with_ts = [(r, r["timestamp_utc"]) for r in year_recs]
        with_ts.sort(key=lambda x: (x[1] or ""))

        # Group into storms: new storm if gap > 5 days or clear name change
        storm_groups = []
        current_group = []
        prev_ts = None

        for rec, ts_str in with_ts:
            if ts_str and prev_ts:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    prev = datetime.fromisoformat(prev_ts)
                    gap_days = (ts - prev).total_seconds() / 86400
                except Exception:
                    gap_days = 0

                # New storm if same storm_id (already named) or big gap
                if rec.get("storm_id") and current_group and current_group[0].get("storm_id") != rec.get("storm_id"):
                    storm_groups.append(current_group)
                    current_group = [rec]
                elif gap_days > 8:  # >8 days gap = new storm event
                    storm_groups.append(current_group)
                    current_group = [rec]
                else:
                    current_group.append(rec)
            else:
                if not current_group or (
                    rec.get("storm_id") and current_group and
                    current_group[0].get("storm_id") != rec.get("storm_id")
                ):
                    if current_group:
                        storm_groups.append(current_group)
                    current_group = [rec]
                else:
                    current_group.append(rec)
            prev_ts = ts_str

        if current_group:
            storm_groups.append(current_group)

        # Assign stable storm IDs for unnamed groups
        event_counter = 1
        for group in storm_groups:
            # Use the most common storm_id in the group (if any named)
            named_ids = [r["storm_id"] for r in group if r["storm_id"] and "UNNAMED" not in r["storm_id"]]
            if named_ids:
                canonical_id = max(set(named_ids), key=named_ids.count)
            else:
                canonical_id = f"NIO_{year}_UNNAMED_{event_counter:02d}"
                event_counter += 1
            for rec in group:
                if not rec["storm_id"] or "UNK" in rec["storm_id"]:
                    rec["storm_id"] = canonical_id
                    if not rec["storm_name"]:
                        rec["storm_name"] = None
            result.extend(group)

    return result


def run_normalization(digital_results: dict, scanned_results: dict) -> tuple:
    """
    Normalize all extracted records.
    Returns (df_observations, manual_review_entries, narrative_entries)
    """
    digital_recs, digital_mr, digital_narr = normalize_digital_results(digital_results)
    scanned_recs, scanned_mr, scanned_narr = normalize_scanned_results(scanned_results)

    all_records = digital_recs + scanned_recs
    all_manual_review = digital_mr + scanned_mr
    all_narrative = digital_narr + scanned_narr

    # Merge narrative into manual review
    all_manual_review.extend(all_narrative)

    # Storm ID grouping for scanned years
    all_records = _assign_storm_ids_within_year(all_records)

    # Build DataFrame
    df = pd.DataFrame(all_records, columns=cfg.CANONICAL_FIELDS)

    # Type enforcement
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["source_page"] = pd.to_numeric(df["source_page"], errors="coerce").astype("Int64")
    df["source_row"] = pd.to_numeric(df["source_row"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["maximum_sustained_wind_kt"] = pd.to_numeric(df["maximum_sustained_wind_kt"], errors="coerce")
    df["central_pressure_hpa"] = pd.to_numeric(df["central_pressure_hpa"], errors="coerce")
    df["manual_review_required"] = df["manual_review_required"].fillna(False).astype(bool)

    # Sort: by storm_id, then timestamp
    df = df.sort_values(["storm_id", "timestamp_utc"], na_position="last").reset_index(drop=True)

    # Deduplicate exact duplicate observation rows
    dup_cols = ["storm_id", "timestamp_utc", "latitude", "longitude",
                "maximum_sustained_wind_kt", "central_pressure_hpa"]
    avail_cols = [c for c in dup_cols if c in df.columns]
    dup_mask = df.duplicated(subset=avail_cols, keep="first")
    if dup_mask.sum() > 0:
        logger.info(f"Deduplicating {dup_mask.sum()} duplicate observation rows...")
        for _, row in df[dup_mask].iterrows():
            all_manual_review.append({
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "storm_id": row.get("storm_id"),
                "source_file": row.get("source_file"),
                "source_page": int(row["source_page"]) if pd.notna(row.get("source_page")) else None,
                "source_table": row.get("source_table"),
                "source_row": int(row["source_row"]) if pd.notna(row.get("source_row")) else None,
                "raw_text": f"Duplicate observation dropped for {row.get('storm_id')} at {row.get('timestamp_utc')}",
                "problem_type": "DUPLICATE_ROW",
                "problem_description": "Exact duplicate observation row found and excluded from final dataset",
                "recommended_action": "EXCLUDE_DUPLICATE",
                "resolved": True,
            })
        df = df[~dup_mask].reset_index(drop=True)

    logger.info(
        f"Normalization complete: {len(df)} records, "
        f"{df['storm_id'].nunique()} unique storm_ids, "
        f"{df['manual_review_required'].sum()} manual-review rows"
    )

    return df, all_manual_review, all_narrative
