"""
01_digital_extractor.py — VAYU-NET IMD Best Track Pipeline
Digital PDF extraction for years 2005–2024 using pdfplumber.

Rules:
- Only extracts Best Track observation tables (4-condition gate).
- Narrative/non-observation rows are separated and flagged.
- Raw values are preserved exactly; no normalization here.
- Writes per-year JSONL to data/interim/imd/{year}_digital_raw.jsonl
"""

import json
import re
import sys
import logging
from pathlib import Path

import pdfplumber

# Allow running as script or import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib
cfg = importlib.import_module("00_config")

logger = logging.getLogger("digital_extractor")

# ── Heading detection ─────────────────────────────────────────────────────────

def _clean_text(t: str) -> str:
    """Collapse whitespace and normalize for comparison."""
    if not t:
        return ""
    return re.sub(r"\s+", " ", t.replace("\n", " ").replace("\r", " ")).strip().lower()


def _is_bt_heading(text: str) -> bool:
    """True if text explicitly identifies a Best Track observation table."""
    ct = _clean_text(text)
    return any(kw in ct for kw in cfg.BT_HEADING_KEYWORDS)


def _is_non_bt_heading(text: str) -> bool:
    ct = _clean_text(text)
    return any(kw in ct for kw in cfg.NON_BT_HEADING_KEYWORDS)


def _is_summary_table_header(header_cells: list) -> bool:
    """Detect storm-summary (genesis/landfall) tables by header signature."""
    combined = _clean_text(" ".join(str(c or "") for c in header_cells))
    return any(sig in combined for sig in cfg.SUMMARY_TABLE_HEADER_SIGNATURES)

# ── Header normalization ───────────────────────────────────────────────────────

def _normalize_header_cell(cell: str) -> str:
    """Map a raw header cell string to a canonical field name or UNKNOWN."""
    if cell is None:
        return None
    # Collapse all whitespace sequences to single space, then lowercase
    import re as _re
    cleaned = _re.sub(r'\s+', ' ', cell.replace('\r', ' ')).strip().lower()
    # Also try newline-preserved version for multi-line headers
    cleaned_nl = cell.strip().lower()
    # Also try with all whitespace and hyphens removed (e.g. "Grad\ne" -> "grade")
    cleaned_nospace = _re.sub(r'[\s\-]+', '', cell).lower()
    result = cfg.HEADER_TO_FIELD.get(cleaned)
    if result is None:
        result = cfg.HEADER_TO_FIELD.get(cleaned_nl)
    if result is None:
        result = cfg.HEADER_TO_FIELD.get(cleaned_nospace)

    if result is None:
        # Robust keyword fallbacks for broken/noisy PDF headers:
        if "drop" in cleaned_nospace or "dp" in cleaned_nospace or "∆p" in cleaned_nospace or "δp" in cleaned_nospace:
            return "raw_pressure_drop"
        if ("pressure" in cleaned_nospace or "hpa" in cleaned_nospace or cleaned_nospace in ("cp", "c.p.")) and "drop" not in cleaned_nospace:
            return "raw_pressure"
        if "wind" in cleaned_nospace or "msw" in cleaned_nospace:
            return "raw_wind"
        if "grade" in cleaned_nospace or "category" in cleaned_nospace:
            return "raw_category"
        if "c.i." in cleaned or "ci" in cleaned_nospace or "t.no" in cleaned_nospace:
            return "raw_ci_no"
        if cleaned_nospace.startswith("date"):
            return "raw_date"
        if "time" in cleaned_nospace or cleaned_nospace.startswith("utc"):
            return "raw_time"
        if "lat" in cleaned_nospace and ("long" in cleaned_nospace or "lon" in cleaned_nospace):
            return "raw_latitude"
        elif "lat" in cleaned_nospace:
            return "raw_latitude"
        elif "long" in cleaned_nospace or "lon" in cleaned_nospace:
            return "raw_longitude"

    return result


def _map_header(raw_header: list) -> tuple:
    """
    Returns (col_map, missing_groups, unknown_cols)
    col_map: {col_index: canonical_field_name or None}
    missing_groups: list of required group names not found
    unknown_cols: list of (index, raw_text) that didn't map
    """
    col_map = {}
    found_fields = set()
    unknown_cols = []
    lat_col_idx = None

    for i, cell in enumerate(raw_header):
        mapped = _normalize_header_cell(str(cell) if cell is not None else "")
        col_map[i] = mapped
        if mapped:
            found_fields.add(mapped)
            if mapped == "raw_latitude":
                lat_col_idx = i
        else:
            if cell is not None and str(cell).strip():
                unknown_cols.append((i, str(cell)))

    # Detect merged lat/lon header: if raw_latitude is found but raw_longitude is not,
    # check if the header cell text mentions both lat AND lon (merged column).
    # If so, assign the NEXT column (lat_col_idx + 1) as raw_longitude.
    if lat_col_idx is not None and "raw_longitude" not in found_fields:
        lat_cell_text = str(raw_header[lat_col_idx] or "").lower()
        if "long" in lat_cell_text or "lon" in lat_cell_text:
            # Merged lat/lon header — next column is longitude
            lon_col_idx = lat_col_idx + 1
            col_map[lon_col_idx] = "raw_longitude"
            found_fields.add("raw_longitude")

    missing_groups = []
    for group_name, group_fields in cfg.REQUIRED_BT_FIELD_GROUPS.items():
        if not group_fields.intersection(found_fields):
            missing_groups.append(group_name)

    return col_map, missing_groups, unknown_cols


# ── Narrative row detection ────────────────────────────────────────────────────

def _is_narrative_row(row: list, col_map: dict) -> tuple:
    """
    Returns (is_narrative: bool, reason: str)
    A row is narrative if it has no parseable lat/lon/time AND contains prose text.
    """
    lat_col = next((i for i, f in col_map.items() if f == "raw_latitude"), None)
    lon_col = next((i for i, f in col_map.items() if f == "raw_longitude"), None)
    time_col = next((i for i, f in col_map.items() if f == "raw_time"), None)

    def _get(idx):
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    lat_val = _get(lat_col)
    lon_val = _get(lon_col)
    time_val = _get(time_col)

    # Check if lat/lon/time are all missing or non-numeric
    def _is_numeric(v):
        if v is None:
            return False
        try:
            float(str(v).strip().replace(",", "."))
            return True
        except ValueError:
            return False

    has_numeric_lat = _is_numeric(lat_val)
    has_numeric_lon = _is_numeric(lon_val)
    has_time = time_val is not None and re.match(r"^\d{3,4}$", str(time_val).strip())

    if has_numeric_lat and has_numeric_lon and has_time:
        return False, ""

    # Check for narrative prose in any cell
    full_text = " ".join(str(c) for c in row if c is not None).strip()
    if not full_text:
        return False, ""  # empty row — skip silently

    lc = full_text.lower()
    matched_keywords = [kw for kw in cfg.NARRATIVE_KEYWORDS if kw in lc]

    if matched_keywords:
        return True, f"Narrative keywords found: {matched_keywords}; no parseable lat/lon/time"

    # Long text with no numeric lat/lon
    if not has_numeric_lat and not has_numeric_lon and len(full_text) >= 10:
        # Check if the non-numeric content is long prose
        word_count = len(full_text.split())
        if word_count >= 5:
            return True, f"No parseable lat/lon/time; prose detected ({word_count} words)"

    return False, ""

# ── Date carry-forward ────────────────────────────────────────────────────────

def _get_date_val(row, col_map):
    date_col = next((i for i, f in col_map.items() if f == "raw_date"), None)
    if date_col is None or date_col >= len(row):
        return None
    v = row[date_col]
    if v is None:
        return None
    s = str(v).strip()
    # Validate date-like pattern (supports dots, slashes, and hyphens)
    if re.match(r"^\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}", s):
        return s
    return None

# ── Storm name extraction ─────────────────────────────────────────────────────

def _extract_storm_name_from_text(text: str) -> str:
    """
    Extract storm name from a Best Track table heading text.
    Handles multi-line headings and various quote styles.
    """
    if not text:
        return None

    # Focus search window around "best track" or "table"
    search_text = text
    idx = text.lower().find("best track")
    if idx != -1:
        start_pos = max(0, idx - 100)
        end_pos = min(len(text), idx + 400)
        search_text = text[start_pos:end_pos]

    QUOTE_CHARS = r"['\u2018\u2019\u201c\u201d\u201a\u201e\u201f`\ufffd\"]"

    NAME_STOPWORDS = {
        "THE", "NIO", "IMD", "RSMC", "BOB", "ARB", "OVER", "DURING",
        "BAY", "BENGAL", "ARABIAN", "SEA", "NORTH", "SOUTH", "EAST", "WEST",
        "SOUTHWEST", "SOUTHEAST", "NORTHEAST", "NORTHWEST", "CENTRAL",
        "ANDAMAN", "COASTAL", "COAST", "GUJARAT", "ODISHA", "TAMIL", "NADU",
        "ANDHRA", "PRADESH", "MYANMAR", "BANGLADESH",
        "INDIA", "OCEAN", "INDIAN", "LAND", "SYSTEM", "DISTURBANCE",
        "DEPRESSION", "CYCLONE", "STORM", "MSW", "IST", "UTC", "CATEGORY",
        "TABLE", "TRACK", "BEST", "POSITIONS", "PARAMETERS", "INTENSITY",
        "SPEED", "KNOTS", "FIGURE", "GRADE", "PRESSURE", "WIND",
    }

    # Pattern 1: quoted name in the search window (e.g. 'TAUKTAE', "Tauktae", \ufffdYAAS\ufffd)
    m = re.search(rf"{QUOTE_CHARS}\s*([A-Za-z\-]+)\s*{QUOTE_CHARS}", search_text)
    if m:
        name = m.group(1).strip().upper()
        if len(name) >= 2 and name not in NAME_STOPWORDS:
            return name

    # Pattern 2: "Storm/Cyclone [NAME] over"
    m = re.search(
        rf"(?:Storm|Cyclone)[,\s]+{QUOTE_CHARS}*\s*([A-Za-z\-]+)\s*{QUOTE_CHARS}*\s+over",
        search_text,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip().upper()
        if len(name) >= 2 and name not in NAME_STOPWORDS:
            return name

    # Pattern 3: "of [TYPE] NAME"
    m = re.search(
        r'of\s+(?:(?:Super|Extremely\s+Severe|Very\s+Severe|Severe|Deep|)\s*'
        r'(?:Cyclonic\s+Storm|Depression|Cyclone|CS|VSCS|SCS|ESCS|SSCS|DD|D)\s*'
        rf"{QUOTE_CHARS}?\s*)"
        r"([A-Za-z][A-Za-z\-]+)",
        search_text,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip().strip("''‘’\"").upper()
        if len(name) >= 2 and name not in NAME_STOPWORDS:
            return name

    # Pattern 4: parenthetical abbreviation, e.g. "(ESCS) Tauktae" or "(VSCS) HIKAA"
    m = re.search(r'\([A-Z]+\)\s+([A-Z][a-zA-Z]+)', search_text)
    if m:
        name = m.group(1).strip().upper()
        if len(name) >= 2 and name not in NAME_STOPWORDS:
            return name

    return None


def _extract_table_id_from_text(text: str) -> str:
    """Extract 'Table X.Y.Z' identifier from nearby text."""
    m = re.search(r"Table\s+(\d+(?:\.\d+)+)", text, re.IGNORECASE)
    if m:
        return f"Table {m.group(1)}"
    return None

# ── Page context extraction ────────────────────────────────────────────────────

def _get_page_context(pdf, page_idx: int) -> str:
    """Get the text context from the current page and the previous page."""
    texts = []
    for idx in [max(0, page_idx - 1), page_idx]:
        try:
            texts.append(pdf.pages[idx].extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts)

# ── Main per-page table extraction ────────────────────────────────────────────

def _extract_bt_tables_from_page(pdf, page_idx: int, year: int, filename: str,
                                  prev_page_text: str, last_bt_table: dict = None) -> list:
    """
    Attempt to extract Best Track observation tables from a single page.
    Supports continuation tables spanning from the previous page.
    Returns list of table extraction result dicts.
    """
    results = []
    page = pdf.pages[page_idx]
    page_num = page_idx + 1  # 1-based

    # Get page text for heading detection
    page_text = page.extract_text() or ""
    combined_context = (prev_page_text or "") + "\n" + page_text

    # Check for BT heading in page context (current + previous page)
    # Also treat as confirmed if immediately following an active BT table
    has_bt_heading = _is_bt_heading(combined_context) or (
        last_bt_table is not None and (page_num == last_bt_table.get("page", 0) + 1)
    )
    # Non-BT check applies only if the page has NO bt heading at all
    has_non_bt_heading = _is_non_bt_heading(page_text) and not _is_bt_heading(page_text)

    # Only hard-reject if no BT heading found anywhere in combined context
    if has_non_bt_heading and not has_bt_heading:
        return results

    # Extract pdfplumber tables
    try:
        tables = page.extract_tables()
    except Exception as e:
        logger.warning(f"Year {year} page {page_num}: pdfplumber table extraction error: {e}")
        return results

    for tbl_idx, table in enumerate(tables):
        if not table or len(table) < 2:
            continue

        header = table[0]
        data_rows = table[1:]

        # ── Check for continuation of previous page's Best Track table ────────
        is_continuation = False
        if last_bt_table and (page_num == last_bt_table.get("page", 0) + 1):
            last_col_map = last_bt_table.get("col_map", {})
            lat_col = next((i for i, f in last_col_map.items() if f == "raw_latitude"), None)
            lon_col = next((i for i, f in last_col_map.items() if f == "raw_longitude"), None)
            time_col = next((i for i, f in last_col_map.items() if f == "raw_time"), None)

            r0 = table[0]
            if time_col is not None and time_col < len(r0):
                time_val = str(r0[time_col] or "").strip()
                if re.match(r"^\d{3,4}$", time_val):
                    try:
                        lat_v = float(str(r0[lat_col]).rstrip("NnSs").strip()) if lat_col < len(r0) else None
                        lon_v = float(str(r0[lon_col]).rstrip("EeWw").strip()) if lon_col < len(r0) else None
                        if lat_v is not None and lon_v is not None:
                            is_continuation = True
                    except (ValueError, TypeError):
                        pass

        if is_continuation:
            header = last_bt_table.get("header_raw", [])
            col_map = dict(last_bt_table["col_map"])
            data_rows = table  # row 0 is data
            storm_name_raw = last_bt_table.get("storm_name_raw")
            table_id = last_bt_table.get("table_id")
            detection_method = "CONTINUATION_TABLE"
            missing_required = []
            unknown_cols = []
            has_valid_row = True
            fields_present = set(col_map.values())
        else:
            # ── Condition 4: Reject known non-BT table types ────────────────────
            if _is_summary_table_header(header):
                logger.debug(f"Year {year} page {page_num} table {tbl_idx}: rejected (summary table header)")
                continue

            # ── Condition 1: Heading check ──────────────────────────────────────
            if not has_bt_heading:
                # Content-fallback: must still pass conditions 2 and 3
                detection_method = "CONTENT_FALLBACK"
            else:
                detection_method = "HEADING_CONFIRMED"

            # ── Condition 2: Required field presence ────────────────────────────
            col_map, missing_groups, unknown_cols = _map_header(header)

            # Handle merged lat/lon column: if header count < data column count
            # and we're missing lat or lon, attempt positional split
            if ("latitude" in missing_groups or "longitude" in missing_groups):
                col_map, missing_groups, unknown_cols = _attempt_latlon_split_repair(
                    header, data_rows, col_map, missing_groups, unknown_cols
                )

            fields_present = set(v for v in col_map.values() if v)
            missing_required = [g for g, gf in cfg.REQUIRED_BT_FIELD_GROUPS.items()
                                if not gf.intersection(fields_present)]

            # ── Condition 3: Valid data row present ─────────────────────────────
            has_valid_row = False
            for row in data_rows:
                # Check for parseable lat/lon/time
                lat_col = next((i for i, f in col_map.items() if f == "raw_latitude"), None)
                lon_col = next((i for i, f in col_map.items() if f == "raw_longitude"), None)
                time_col = next((i for i, f in col_map.items() if f == "raw_time"), None)
                if lat_col is not None and lon_col is not None and time_col is not None:
                    try:
                        lat_v = row[lat_col] if lat_col < len(row) else None
                        lon_v = row[lon_col] if lon_col < len(row) else None
                        time_v = row[time_col] if time_col < len(row) else None
                        if (lat_v is not None and lon_v is not None and time_v is not None):
                            # Handle slash-format lat/lon data cell (e.g. "5.5/87.0")
                            lat_str = str(lat_v).strip()
                            lon_str = str(lon_v).strip()
                            if "/" in lat_str:
                                parts = lat_str.split("/", 1)
                                lat_str = parts[0].strip()
                                lon_str = parts[1].strip()
                            float(lat_str.rstrip("NnSs").strip())
                            float(lon_str.rstrip("EeWw").strip())
                            if re.match(r"^\d{3,4}$", str(time_v).strip()):
                                has_valid_row = True
                                break
                    except (ValueError, TypeError):
                        pass

            # ── Gate decision ───────────────────────────────────────────────────
            if missing_required or not has_valid_row:
                # If the table matched 0 BT columns and has no valid data rows, it is a
                # non-BT structure (figure border, image frame, etc.) — discard silently.
                if len(fields_present) == 0 and not has_valid_row:
                    continue

                # Record candidate tables that fail conditions 2/3 as uncertain for manual review.
                uncertain_result = {
                    "status": "UNCERTAIN_TABLE_IDENTITY",
                    "page": page_num,
                    "table_index": tbl_idx,
                    "header": header,
                    "missing_required_groups": missing_required,
                    "has_valid_row": has_valid_row,
                    "detection_method": detection_method,
                    "manual_review_required": True,
                    "manual_review_reason": (
                        f"UNCERTAIN_TABLE_IDENTITY: heading={'confirmed' if detection_method == 'HEADING_CONFIRMED' else 'content_fallback'}; "
                        f"missing_groups={missing_required}; has_valid_row={has_valid_row}"
                    ),
                }
                results.append(uncertain_result)
                continue  # Don't extract without passing all conditions

        # ── Extract storm name and table ID from context ────────────────────
        if not is_continuation:
            storm_name_raw = _extract_storm_name_from_text(combined_context)
            table_id = _extract_table_id_from_text(combined_context)

        # ── Extract rows ────────────────────────────────────────────────────
        observation_rows = []
        narrative_rows = []
        carried_date = last_bt_table.get("last_carried_date") if is_continuation else None

        for row_idx, row in enumerate(data_rows, start=1):
            # Skip fully empty rows
            if all(c is None or str(c).strip() == "" for c in row):
                continue

            # Handle slash-separated lat/lon in a single data cell (e.g. "5.5/87.0")
            # If lat col exists and its value contains '/', split it.
            lat_col_pos = next((i for i, f in col_map.items() if f == "raw_latitude"), None)
            lon_col_pos = next((i for i, f in col_map.items() if f == "raw_longitude"), None)
            if lat_col_pos is not None and lat_col_pos < len(row):
                lat_val = str(row[lat_col_pos] or "").strip()
                if "/" in lat_val and lon_col_pos is not None:
                    parts = lat_val.split("/", 1)
                    # Rebuild row with split values (non-destructive on original)
                    row = list(row)
                    row[lat_col_pos] = parts[0].strip()
                    row[lon_col_pos] = parts[1].strip()

            # Date carry-forward
            date_val = _get_date_val(row, col_map)
            if date_val:
                carried_date = date_val
                date_carried = False
            elif carried_date:
                date_carried = True
            else:
                date_carried = False

            # Narrative check
            is_narr, narr_reason = _is_narrative_row(row, col_map)
            if is_narr:
                narrative_rows.append({
                    "source_row": row_idx,
                    "raw_text": " | ".join(str(c) for c in row if c is not None),
                    "reason": narr_reason,
                })
                continue

            # Build raw record
            raw_record = {
                "source_row": row_idx,
                "date_carried_forward": date_carried,
                "raw_date": carried_date if date_carried else date_val,
                "col_data": {},
            }

            for col_idx, field in col_map.items():
                if col_idx < len(row):
                    cell_val = row[col_idx]
                    raw_record["col_data"][field or f"UNKNOWN_COL_{col_idx}"] = (
                        str(cell_val).strip() if cell_val is not None else None
                    )

            # Override raw_date with carried value
            if "raw_date" in raw_record["col_data"] and raw_record["date_carried_forward"]:
                raw_record["col_data"]["raw_date"] = carried_date

            observation_rows.append(raw_record)

        # Build table result
        table_result = {
            "status": "OK",
            "page": page_num,
            "table_index": tbl_idx,
            "table_id": table_id,
            "storm_name_raw": storm_name_raw,
            "detection_method": detection_method,
            "header_raw": [str(h) if h is not None else None for h in header],
            "col_map": {str(k): v for k, v in col_map.items()},
            "missing_required_groups": missing_required,
            "unknown_cols": unknown_cols,
            "observation_rows": observation_rows,
            "narrative_rows": narrative_rows,
            "last_carried_date": carried_date,
        }
        results.append(table_result)

    return results


def _attempt_latlon_split_repair(header, data_rows, col_map, missing_groups, unknown_cols):
    """
    If "Centre lat.0 N/ long. 0 E" header appears once but data has two columns,
    attempt to detect and fix by splitting into lat and lon by column position.
    """
    # Find the merged lat/lon header index
    merged_idx = None
    for i, cell in enumerate(header):
        if cell and "centre lat" in str(cell).lower():
            merged_idx = i
            break

    if merged_idx is None:
        return col_map, missing_groups, unknown_cols

    # Check if data rows have more columns than header
    sample_data_len = max((len(r) for r in data_rows[:5] if r), default=0)
    if sample_data_len > len(header):
        # The merged header covers two data columns: merged_idx → lat, merged_idx+1 → lon
        new_col_map = {}
        shift = 0
        for i, cell in enumerate(header):
            if i == merged_idx:
                new_col_map[i] = "raw_latitude"
                new_col_map[i + 1] = "raw_longitude"
                shift = 1
            else:
                new_col_map[i + (1 if i > merged_idx else 0)] = col_map.get(i)

        new_missing = [g for g, gf in cfg.REQUIRED_BT_FIELD_GROUPS.items()
                       if not gf.intersection(set(v for v in new_col_map.values() if v))]
        new_unknown = [(i, t) for i, t in unknown_cols if i != merged_idx and i != merged_idx + 1]
        return new_col_map, new_missing, new_unknown

    return col_map, missing_groups, unknown_cols

# ── Per-year extraction driver ────────────────────────────────────────────────

def extract_year(year: int, file_info: dict) -> dict:
    """
    Extract all Best Track tables from a single digital annual report.
    Returns a result dict with all tables, narrative rows, and stats.
    """
    filepath = cfg.RAW_IMD_DIR / file_info["filename"]
    result = {
        "year": year,
        "source_file": file_info["filename"],
        "source_report": file_info["report_title"],
        "tables_found": [],
        "uncertain_tables": [],
        "total_observation_rows": 0,
        "total_narrative_rows": 0,
        "errors": [],
    }

    if not filepath.exists():
        result["errors"].append(f"File not found: {filepath}")
        logger.error(f"Year {year}: file not found: {filepath}")
        return result

    logger.info(f"Year {year}: opening {file_info['filename']}")

    try:
        with pdfplumber.open(str(filepath)) as pdf:
            n_pages = len(pdf.pages)
            prev_page_text = ""

            last_bt_table = None
            for page_idx in range(n_pages):
                page_tables = _extract_bt_tables_from_page(
                    pdf, page_idx, year, file_info["filename"], prev_page_text, last_bt_table
                )
                for tbl in page_tables:
                    if tbl["status"] == "UNCERTAIN_TABLE_IDENTITY":
                        result["uncertain_tables"].append(tbl)
                    elif tbl["status"] == "OK":
                        result["tables_found"].append(tbl)
                        result["total_observation_rows"] += len(tbl["observation_rows"])
                        result["total_narrative_rows"] += len(tbl["narrative_rows"])
                        last_bt_table = {
                            "page": tbl["page"],
                            "col_map": {int(k): v for k, v in tbl["col_map"].items()},
                            "header_raw": tbl.get("header_raw", []),
                            "storm_name_raw": tbl.get("storm_name_raw"),
                            "table_id": tbl.get("table_id"),
                            "last_carried_date": tbl.get("last_carried_date"),
                        }

                # Update prev page text
                try:
                    prev_page_text = pdf.pages[page_idx].extract_text() or ""
                except Exception:
                    prev_page_text = ""

    except Exception as e:
        result["errors"].append(f"PDF open/parse error: {e}")
        logger.error(f"Year {year}: {e}")

    logger.info(
        f"Year {year}: {len(result['tables_found'])} BT tables, "
        f"{result['total_observation_rows']} obs rows, "
        f"{result['total_narrative_rows']} narrative rows, "
        f"{len(result['uncertain_tables'])} uncertain tables"
    )
    return result


def run_digital_extraction(year_file_map: dict) -> dict:
    """
    Run digital extraction for all DIGITAL_YEARS present in year_file_map.
    Writes per-year JSONL to data/interim/imd/.
    Returns {year: result_dict}.
    """
    cfg.INTERIM_IMD_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for year in sorted(cfg.DIGITAL_YEARS):
        if year not in year_file_map:
            logger.warning(f"Year {year}: not found in inventory — skipping")
            continue
        file_info = year_file_map[year]
        result = extract_year(year, file_info)
        all_results[year] = result

        # Write per-year JSONL
        out_path = cfg.INTERIM_IMD_DIR / f"{year}_digital_raw.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, default=str)
            fh.write("\n")
        logger.info(f"Year {year}: wrote {out_path}")

    return all_results
