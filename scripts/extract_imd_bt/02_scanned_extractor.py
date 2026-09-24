"""
02_scanned_extractor.py — VAYU-NET IMD Best Track Pipeline
Scanned PDF extraction for years 1998–2004 using PyMuPDF rendering + EasyOCR.

Rules:
- Renders each candidate page at 300 DPI.
- Runs EasyOCR on rendered image.
- Reconstructs table rows by bounding-box column alignment.
- Flags all suspicious OCR values for manual review.
- Never silently corrects OCR output.
- Writes per-year JSONL to data/interim/imd/{year}_ocr_raw.jsonl
"""

import json
import re
import sys
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib
cfg = importlib.import_module("00_config")

logger = logging.getLogger("scanned_extractor")

# Try importing EasyOCR
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
    _reader = None  # lazy init
except ImportError:
    _EASYOCR_AVAILABLE = False
    _reader = None

DPI = 300
SCALE = DPI / 72.0  # PyMuPDF uses 72 DPI base


def _get_ocr_reader():
    global _reader
    if _reader is None:
        logger.info("Initializing EasyOCR reader (English)…")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# ── OCR suspicious value detection ────────────────────────────────────────────

def _check_ocr_suspicious(raw_val: str, expected_type: str) -> tuple:
    """
    Returns (is_suspicious: bool, reason: str)
    expected_type: 'float', 'int', 'time', 'date', 'category'
    """
    if raw_val is None:
        return False, ""
    s = raw_val.strip()

    # Check for known OCR confusion characters
    for char_a, char_b in cfg.OCR_AMBIGUOUS_PAIRS:
        if char_a in s:
            return True, f"Possible OCR confusion: '{char_a}' may be '{char_b}' in value '{s}'"

    if expected_type in ("float", "int"):
        try:
            v = float(s.replace(",", "."))
        except ValueError:
            return True, f"Non-numeric value where {expected_type} expected: '{s}'"

        # Plausibility checks (flag only, never correct)
        if expected_type == "float":
            # Could be latitude (0–35) or longitude (40–110)
            # or wind (15–200) or pressure (850–1020)
            pass  # plausibility assessed in normalizer with full column context

    if expected_type == "time":
        if not re.match(r"^\d{3,4}$", s):
            return True, f"Unexpected time format: '{s}'"

    return False, ""


# ── Column grouping from bounding boxes ────────────────────────────────────────

def _group_into_columns(ocr_results: list, n_cols: int) -> list:
    """
    Given EasyOCR results [(bbox, text, conf), ...], group by x-centroid into n_cols buckets.
    Returns list of dicts: {'row': int, 'col': int, 'text': str, 'conf': float, 'y': float}
    """
    if not ocr_results:
        return []

    # Compute x-centroids
    items = []
    for bbox, text, conf in ocr_results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x_center = sum(xs) / len(xs)
        y_center = sum(ys) / len(ys)
        items.append({"x": x_center, "y": y_center, "text": text, "conf": conf})

    if not items:
        return []

    # Sort by y (rows) then x (columns)
    items.sort(key=lambda d: (d["y"], d["x"]))

    # Cluster x-centroids into columns using simple quantile bucketing
    xs_sorted = sorted(d["x"] for d in items)
    x_min, x_max = xs_sorted[0], xs_sorted[-1]

    if n_cols <= 1 or x_max == x_min:
        for d in items:
            d["col"] = 0
    else:
        bucket_width = (x_max - x_min) / n_cols
        for d in items:
            col_idx = min(int((d["x"] - x_min) / bucket_width), n_cols - 1)
            d["col"] = col_idx

    # Group into rows by y proximity (within ~20px at 300dpi)
    ROW_GAP = 25  # pixels
    rows = []
    current_row = [items[0]]
    for item in items[1:]:
        if abs(item["y"] - current_row[-1]["y"]) < ROW_GAP:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    # Flatten with row index
    result = []
    for row_idx, row_items in enumerate(rows):
        for item in row_items:
            result.append({
                "row": row_idx,
                "col": item["col"],
                "text": item["text"],
                "conf": item["conf"],
                "y": item["y"],
            })
    return result


def _reconstruct_table(grouped: list, n_cols: int) -> list:
    """Convert grouped items into list-of-lists table."""
    if not grouped:
        return []
    max_row = max(d["row"] for d in grouped)
    table = []
    for r in range(max_row + 1):
        row_cells = [""] * n_cols
        for item in grouped:
            if item["row"] == r:
                col = min(item["col"], n_cols - 1)
                if row_cells[col]:
                    row_cells[col] += " " + item["text"]
                else:
                    row_cells[col] = item["text"]
        table.append(row_cells)
    return table


# ── Scanned page extraction ───────────────────────────────────────────────────

# Expected column structure for scanned 1998–2004 reports
# Typical: Date | UTC | Lat | Lon | ECP | MSW | Grade
SCANNED_EXPECTED_COLS = 7
SCANNED_COL_NAMES = [
    "raw_date", "raw_time", "raw_latitude", "raw_longitude",
    "raw_pressure", "raw_wind", "raw_category"
]

# Heading keywords to identify BT pages in scanned reports
SCANNED_BT_HEADING_PATTERNS = [
    r"best\s+track",
    r"table\s+2\.\d+\.?\d*",
    r"date\s+utc\s+lat",
]


def _page_likely_bt(page_text: str, page_image_path: str = None) -> bool:
    """Quick check if a scanned page likely contains a BT table."""
    lc = (page_text or "").lower()
    for pat in SCANNED_BT_HEADING_PATTERNS:
        if re.search(pat, lc):
            return True
    return False


def _extract_scanned_page(doc, page_idx: int, year: int, filename: str,
                           ocr_image_dir: Path) -> dict:
    """
    Render and OCR a single scanned PDF page. Returns extraction dict.
    """
    page_num = page_idx + 1
    page = doc[page_idx]

    # Try text layer first (some scanned PDFs have partial text)
    text_layer = page.get_text().strip()
    if len(text_layer) > 50:
        logger.debug(f"Year {year} page {page_num}: has text layer ({len(text_layer)} chars)")

    if not _EASYOCR_AVAILABLE:
        return {
            "page": page_num,
            "status": "OCR_PENDING",
            "extraction_method": "OCR_PENDING",
            "manual_review_required": True,
            "manual_review_reason": "EasyOCR not available — page requires manual OCR extraction",
            "observation_rows": [],
            "narrative_rows": [],
        }

    # Render at 300 DPI
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat)

    # Save image for provenance (only retained if manual_review rows exist)
    img_path = ocr_image_dir / f"{year}_page{page_num:03d}.png"
    pix.save(str(img_path))

    # OCR
    reader = _get_ocr_reader()
    try:
        ocr_results = reader.readtext(str(img_path))
    except Exception as e:
        return {
            "page": page_num,
            "status": "OCR_ERROR",
            "extraction_method": "OCR",
            "error": str(e),
            "manual_review_required": True,
            "manual_review_reason": f"OCR error: {e}",
            "observation_rows": [],
            "narrative_rows": [],
            "image_path": str(img_path),
        }

    if not ocr_results:
        # Clean up image if empty
        img_path.unlink(missing_ok=True)
        return {
            "page": page_num,
            "status": "OCR_EMPTY",
            "extraction_method": "OCR",
            "manual_review_required": False,
            "observation_rows": [],
            "narrative_rows": [],
        }

    # Group into columns
    grouped = _group_into_columns(ocr_results, SCANNED_EXPECTED_COLS)
    raw_table = _reconstruct_table(grouped, SCANNED_EXPECTED_COLS)

    if not raw_table:
        img_path.unlink(missing_ok=True)
        return {
            "page": page_num, "status": "OCR_NO_TABLE",
            "extraction_method": "OCR", "observation_rows": [], "narrative_rows": [],
        }

    # Map columns using expected scanned column names
    col_map = {i: SCANNED_COL_NAMES[i] for i in range(min(len(SCANNED_COL_NAMES), SCANNED_EXPECTED_COLS))}

    # Process rows
    observation_rows = []
    narrative_rows = []
    carried_date = None
    has_manual_review = False

    # Skip first row if it looks like a header
    start_row = 0
    if raw_table and re.search(r"date|utc|lat|long|grade", str(raw_table[0]).lower()):
        start_row = 1

    for row_idx, row in enumerate(raw_table[start_row:], start=1):
        if all(not c.strip() for c in row):
            continue

        # Date carry-forward
        date_val = row[0].strip() if row else ""
        if re.match(r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}", date_val):
            carried_date = date_val
            date_carried = False
        elif carried_date:
            date_carried = True
        else:
            date_carried = False

        # Narrative check
        full_text = " ".join(c.strip() for c in row if c.strip())
        lc = full_text.lower()
        is_narr = any(kw in lc for kw in cfg.NARRATIVE_KEYWORDS)
        time_val = row[1].strip() if len(row) > 1 else ""
        lat_val = row[2].strip() if len(row) > 2 else ""
        lon_val = row[3].strip() if len(row) > 3 else ""

        if is_narr or not re.match(r"^\d{3,4}$", time_val):
            if len(full_text) >= 5:
                narrative_rows.append({
                    "source_row": row_idx,
                    "raw_text": full_text,
                    "reason": "Narrative or non-observation row (OCR)",
                })
            continue

        # OCR suspicious value checks
        review_reasons = []
        raw_record = {
            "source_row": row_idx,
            "date_carried_forward": date_carried,
            "col_data": {},
        }

        expected_types = ["date", "time", "float", "float", "float", "float", "category"]
        for col_i, (field, val, exp_type) in enumerate(
            zip(SCANNED_COL_NAMES, row[:len(SCANNED_COL_NAMES)], expected_types)
        ):
            val_str = val.strip() if val else None
            raw_record["col_data"][field] = val_str

            # Override raw_date if carried
            if field == "raw_date" and date_carried:
                raw_record["col_data"]["raw_date"] = carried_date

            susp, reason = _check_ocr_suspicious(val_str, exp_type)
            if susp:
                review_reasons.append(f"Col {col_i} ({field}): {reason}")

        if review_reasons:
            raw_record["manual_review_required"] = True
            raw_record["manual_review_reason"] = "; ".join(review_reasons)
            has_manual_review = True
        else:
            raw_record["manual_review_required"] = False
            raw_record["manual_review_reason"] = ""

        raw_record["extraction_method"] = "OCR"
        observation_rows.append(raw_record)

    # Clean up image only if no manual-review rows
    if not has_manual_review and img_path.exists():
        img_path.unlink(missing_ok=True)

    return {
        "page": page_num,
        "status": "OK",
        "extraction_method": "OCR",
        "observation_rows": observation_rows,
        "narrative_rows": narrative_rows,
        "has_manual_review": has_manual_review,
        "image_path": str(img_path) if has_manual_review else None,
    }


def _find_bt_pages_scanned(doc, year: int, file_info: dict = None) -> list:
    """
    Identify candidate Best Track pages in a scanned PDF.
    Prioritizes catalogued best_track_page_ranges from inventory CSV.
    """
    if file_info and file_info.get("best_track_page_ranges"):
        ranges_str = str(file_info["best_track_page_ranges"]).strip()
        if ranges_str and ranges_str != "nan":
            pages = []
            for part in ranges_str.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        start_p, end_p = part.split("-", 1)
                        for p in range(int(start_p), int(end_p) + 1):
                            if 1 <= p <= len(doc):
                                pages.append(p - 1)  # convert to 0-based
                    except ValueError:
                        pass
                else:
                    try:
                        p = int(part)
                        if 1 <= p <= len(doc):
                            pages.append(p - 1)  # convert to 0-based
                    except ValueError:
                        pass
            if pages:
                return sorted(list(set(pages)))

    bt_pages = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text().strip()
        # If text layer shows BT pattern, mark as candidate
        if _page_likely_bt(text):
            bt_pages.append(page_idx)
        elif not text:
            # Fully scanned — check by position (all non-cover pages are candidates)
            if page_idx >= 5:  # skip cover/intro pages
                bt_pages.append(page_idx)

    return bt_pages


def _extract_storm_name_scanned(year: int, table_idx: int) -> str:
    """Scanned reports: storm name is harder to extract; use index-based placeholder."""
    return None  # Will be resolved via normalizer context


def extract_year_scanned(year: int, file_info: dict) -> dict:
    """
    Extract all Best Track tables from a single scanned annual report.
    Returns a result dict.
    """
    jsonl_path = cfg.INTERIM_IMD_DIR / f"{year}_ocr_raw.jsonl"
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                cached = json.loads(f.readline())
            if cached.get("total_observation_rows", 0) > 0:
                logger.info(f"Year {year} (scanned): using cached extraction ({cached['total_observation_rows']} obs rows)")
                return cached
        except Exception:
            pass

    filepath = cfg.RAW_IMD_DIR / file_info["filename"]
    ocr_image_dir = cfg.INTERIM_IMD_DIR / f"{year}_ocr_images"
    ocr_image_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "year": year,
        "source_file": file_info["filename"],
        "source_report": file_info["report_title"],
        "pages_processed": [],
        "total_observation_rows": 0,
        "total_narrative_rows": 0,
        "total_ocr_pending": 0,
        "errors": [],
    }

    if not filepath.exists():
        result["errors"].append(f"File not found: {filepath}")
        return result

    logger.info(f"Year {year} (scanned): opening {file_info['filename']}")

    try:
        doc = fitz.open(str(filepath))
        bt_pages = _find_bt_pages_scanned(doc, year, file_info)
        logger.info(f"Year {year}: {len(bt_pages)} candidate BT pages found")

        for page_idx in bt_pages:
            page_result = _extract_scanned_page(doc, page_idx, year, file_info["filename"], ocr_image_dir)
            result["pages_processed"].append(page_result)
            if page_result.get("status") == "OCR_PENDING":
                result["total_ocr_pending"] += 1
            else:
                result["total_observation_rows"] += len(page_result.get("observation_rows", []))
                result["total_narrative_rows"] += len(page_result.get("narrative_rows", []))

        doc.close()

    except Exception as e:
        result["errors"].append(f"PDF open error: {e}")
        logger.error(f"Year {year}: {e}")

    # Clean up empty OCR image dir
    if ocr_image_dir.exists() and not any(ocr_image_dir.iterdir()):
        ocr_image_dir.rmdir()

    logger.info(
        f"Year {year}: {result['total_observation_rows']} obs rows, "
        f"{result['total_narrative_rows']} narrative rows, "
        f"{result['total_ocr_pending']} pending"
    )
    return result


def run_scanned_extraction(year_file_map: dict) -> dict:
    """
    Run scanned extraction for all SCANNED_YEARS present in year_file_map.
    Returns {year: result_dict}.
    """
    cfg.INTERIM_IMD_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for year in sorted(cfg.SCANNED_YEARS):
        if year not in year_file_map:
            logger.warning(f"Year {year}: not found in inventory — skipping")
            continue
        file_info = year_file_map[year]
        result = extract_year_scanned(year, file_info)
        all_results[year] = result

        out_path = cfg.INTERIM_IMD_DIR / f"{year}_ocr_raw.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, default=str)
            fh.write("\n")
        logger.info(f"Year {year}: wrote {out_path}")

    return all_results
