# VAYU-NET Sample Index Label Semantics Forensic QA Report

**Authoritative Technical Documentation**  
**Dataset:** Master Sample Index (`data/manifests/vayu_net_sample_index.csv`)  
**Source of Truth:** IMD Best Track Dataset V2 (`data/processed/imd_best_track_v2.csv` / `.parquet`)  
**Audit Scope:** 1,319 Candidate Samples across all analysis horizons ($t_0, +12\text{h}, +24\text{h}, +48\text{h}$)  
**Date of Audit:** 2026-09-24  

---

## 1. Executive Summary

Following the completion of the GridSat-B1 27-year production acquisition (2,353 files), a comprehensive forensic audit was conducted on the label semantics of `data/manifests/vayu_net_sample_index.csv`.

The audit specifically targeted anomalous non-schema values discovered in the cyclonic category fields (numeric strings such as `6, 7, 8, 9, 10, 15, 18, 20, 22, 24, 28, 32, 36, 40, 42, 45, 12, 5, 3`) and missing values in the maximum sustained wind fields.

### Key Forensic Findings:
1. **Zero Sample Index Construction Artifacts:** The sample-index generator (`scripts/production/build_production_dataset.py`) performed a **100% faithful 1-to-1 extraction** from `imd_best_track_v2.csv`. Not a single value was mutated, fabricated, or dropped during sample indexing.
2. **Root Cause of Numeric Categories (Mekunu 2018):** All numeric category values across all four horizons ($t_0, +12\text{h}, +24\text{h}, +48\text{h}$) originate exclusively from a single storm: **`NIO_2018_MEKUNU`** (`27_60dae9_rsmc-2018.pdf`, Table 2.3.1, page 70). The numeric values represent the **Estimated Pressure Drop at the Centre ($\Delta P$, hPa)** column, which was shifted into the Grade column during PDF table extraction due to an artifactual empty column detected by `pdfplumber` on continuation page 70.
3. **Root Cause of Missing Winds:**
   - 21 missing winds at $t_0$, 24 at $+12\text{h}$, 24 at $+24\text{h}$, and 16 at $+48\text{h}$ are directly caused by the exact same Mekunu 2018 page 70 column shift, where the wind column was displaced into pressure drop, leaving wind as `NaN`.
   - The remaining 13 missing winds at $+48\text{h}$ were individually traced:
     - **10 are Legitimate Source-Missing Fields:** Post-landfall inland observations where IMD officially recorded hyphens (`-`) or blanks during storm dissipation (Mala 2006, Sidr 2007, Keila 2011, and Unnamed 2002).
     - **3 are Extraction/Parsing Artifacts:** Post-landfall column alignment shift in Phyan 2009 (2 rows) and an internal whitespace in Ward 2009 raw wind string (`'2 5'`).
4. **Casing Discrepancy:** Uppercase `'SUCS'` (Super Cyclonic Storm) was detected in 2 sample rows for Super Cyclone GONU (`NIO_2007_UNNAMED_2_3_1`), requiring normalization to canonical title-case `'SuCS'`.
5. **Basin Coordinates Audit:** Two candidate samples in 2002 (`NIO_2002_UNNAMED_25` and `NIO_2002_UNNAMED_29`) contain out-of-basin coordinates resulting from legacy OCR text extraction on scanned pages.

---

## 2. Task 1 — Category Fields Schema Compliance Audit

The locked IMD classification schema strictly permits seven discrete operational category grades:
$$\mathcal{C} = \{\text{D, DD, CS, SCS, VSCS, ESCS, SuCS}\}$$

### Field-by-Field Category Distribution

| Category Field | Total Rows | Valid Schema Count | Non-Schema Count | Null Count | Observed Invalid Values |
|:---|:---:|:---:|:---:|:---:|:---|
| `imd_category_t0` | 1,319 | 1,290 | **23** | 6 | `SUCS` (2), `6` (1), `7` (2), `8` (1), `9` (1), `10` (1), `15` (1), `18` (1), `20` (1), `22` (1), `24` (4), `28` (1), `32` (5), `36` (1) |
| `category_12h` | 1,319 | 1,286 | **26** | 7 | `SUCS` (2), `6` (1), `7` (2), `8` (1), `10` (1), `15` (1), `18` (1), `20` (1), `22` (1), `24` (4), `28` (1), `32` (6), `36` (1), `40` (2), `45` (1) |
| `category_24h` | 1,319 | 1,287 | **27** | 5 | `SUCS` (2), `7` (1), `8` (1), `10` (1), `15` (1), `18` (1), `22` (1), `24` (4), `28` (2), `32` (6), `36` (1), `40` (2), `42` (1), `45` (3) |
| `category_48h` | 1,319 | 1,282 | **27** | 10 | `SUCS` (2), `3` (1), `5` (1), `7` (1), `8` (1), `10` (1), `12` (1), `18` (1), `22` (1), `24` (2), `28` (2), `32` (5), `36` (1), `40` (3), `42` (1), `45` (3) |

### Non-Schema Counts by Storm & Year

| Storm Identifier | Year | Affected Horizons | Total Invalid Labels | Category Values Encountered | Root Cause Classification |
|:---|:---:|:---:|:---:|:---|:---|
| `NIO_2018_MEKUNU` | 2018 | $t_0, +12\text{h}, +24\text{h}, +48\text{h}$ | **95** | `3, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 24, 28, 32, 36, 40, 42, 45` | `EXTRACTION_COLUMN_SHIFT` (PDF Table continuation) |
| `NIO_2007_UNNAMED_2_3_1` | 2007 | $t_0, +12\text{h}, +24\text{h}, +48\text{h}$ | **8** | `SUCS` | `CASING_INCONSISTENCY` (Uppercase vs Title-case) |
| **All Other 124 Storms** | 1998–2024 | All | **0** | None (All 100% compliant with schema) | **PASS** |

---

## 3. Tasks 2 & 3 — Forensic Investigation of Cyclone MEKUNU (2018)

### Source Provenance
- **Report Document:** `data/raw/imd/27_60dae9_rsmc-2018.pdf`
- **Table Heading:** Table 2.3.1: *Best track positions and other parameters of the Extremely Severe Cyclonic Storm, 'Mekunu' over the Arabian Sea during 21 May–27 May, 2018*
- **Report Pages:** Physical Page 69 (start of table) and Physical Page 70 (table continuation).

### Mechanical Mechanism of the Column Shift
On Physical Page 69, the table begins with the official 9-column RSMC header:
```text
[Date, Time(UTC), Centre lat./long., C.I. NO., Estimated Central Pressure (hPa), Estimated Maximum Sustained Surface Wind (kt), Estimated Pressure drop at the Centre (hPa), Grade]
```

On Physical Page 70, the table continues without a repeating header. When processed by `pdfplumber.extract_tables()`, an artifactual vertical whitespace was detected in the digital layout, splitting the table into **10 columns** instead of 9:

#### Rows 1 to 28 on Page 70 (22 May 0600 UTC to 25 May 1800 UTC):
```python
# Raw cell array returned by pdfplumber:
['', '0600', '9.5', '57.0', '2.0', '1000', None, '30', '6', 'DD']
```
When mapped against the inherited 9-column map:
- Column 5 (`'1000'`) $\to$ `central_pressure_hpa` = `1000.0`
- Column 6 (`None`) $\to$ `maximum_sustained_wind_kt` = **`NaN`** (WIND CORRUPTED)
- Column 7 (`'30'`) $\to$ `raw_pressure_drop` = `30` (ACTUAL WIND SHIFTED HERE)
- Column 8 (`'6'`) $\to$ `category` = **`'6'`** (PRESSURE DROP SHIFTED TO CATEGORY)
- Column 9 (`'DD'`) $\to$ **TRUNCATED / DROPPED** (ACTUAL GRADE DISCARDED)

#### Rows 30 to 38 on Page 70 (25 May 2100 UTC to 27 May 0000 UTC):
Following the narrative row ("Crossed south Oman coast..."), the empty column shifted between wind and pressure drop:
```python
# Raw cell array returned by pdfplumber:
[None, '2100', '17.1', '53.6', '-', '964', '90', None, '40', 'ESCS']
```
When mapped against the 9-column map:
- Column 5 (`'964'`) $\to$ `central_pressure_hpa` = `964.0`
- Column 6 (`'90'`) $\to$ `maximum_sustained_wind_kt` = `90.0`
- Column 7 (`None`) $\to$ `raw_pressure_drop` = `None`
- Column 8 (`'40'`) $\to$ `category` = **`'40'`** (PRESSURE DROP SHIFTED TO CATEGORY)
- Column 9 (`'ESCS'`) $\to$ **TRUNCATED / DROPPED** (ACTUAL GRADE DISCARDED)

### Ground-Truth Reconciliation for Mekunu
The numbers `6, 7, 8, 9, 10, 15, 18, 20, 22, 24, 28, 32, 36, 40, 42, 45, 12, 5, 3` are unambiguously the central pressure drop values ($\Delta P$) in hPa. The actual grades printed in the official report are standard IMD categories (`DD`, `CS`, `SCS`, `VSCS`, `ESCS`, `D`).

---

## 4. Task 4 — Comprehensive Audit of Missing Wind Values

Across all 1,319 samples and four horizons, missing wind values occur in 98 sample-horizon positions:

| Horizon | Total Missing | Storm Breakdown | Classification | Detailed Cause |
|:---|:---:|:---|:---:|:---|
| $t_0$ | **21** | `NIO_2018_MEKUNU` (21) | **B** (Extraction) | Page 70 column shift; actual wind is in raw col 7 |
| $+12\text{h}$ | **24** | `NIO_2018_MEKUNU` (24) | **B** (Extraction) | Page 70 column shift; actual wind is in raw col 7 |
| $+24\text{h}$ | **24** | `NIO_2018_MEKUNU` (24) | **B** (Extraction) | Page 70 column shift; actual wind is in raw col 7 |
| $+48\text{h}$ | **29** | `NIO_2018_MEKUNU` (16)<br>`NIO_2006_UNNAMED_2_2_1` (4)<br>`NIO_2007_UNNAMED_2_12_1` (4)<br>`NIO_2009_PHYAN` (2)<br>`NIO_2009_WARD` (1)<br>`NIO_2011_UNNAMED_2_9_1` (1)<br>`NIO_2002_UNNAMED_25` (1) | **B** (19)<br>**A** (10) | Mekunu (16, B)<br>Mala post-landfall (4, A)<br>Sidr post-landfall (4, A)<br>Phyan post-landfall col swap (2, B)<br>Ward `'2 5'` whitespace (1, B)<br>Keila WML narrative row (1, A)<br>2002 inland decay row (1, A) |

### Forensic Breakdown of Non-Mekunu Missing Winds:
1. **`NIO_2006_UNNAMED_2_2_1` (Cyclone MALA, 2006) — 4 rows:**
   - Source: `27_b1dd60_rsmc-2006 .pdf`, page 23.
   - Observations on 29 April 2006 at 0900, 1200, 1500, 1800 UTC occurred after landfall on the Myanmar coast.
   - The official IMD report explicitly prints `-` for pressure drop and leaves wind blank.
   - **Classification: A (Legitimately missing in source)**.
2. **`NIO_2007_UNNAMED_2_12_1` (Super Cyclone SIDR, 2007) — 4 rows:**
   - Source: `27_82fddf_rsmc-2007.pdf`, page 80.
   - Observations on 15–16 Nov 2007 (1800, 2100, 0000, 0300 UTC) occurred after landfall in Bangladesh.
   - The official IMD report prints `-` across all numeric fields during rapid dissipation.
   - **Classification: A (Legitimately missing in source)**.
3. **`NIO_2009_PHYAN` (Cyclone PHYAN, 2009) — 2 rows:**
   - Source: `27_4e34f3_rsmc-2009.pdf`, page 78.
   - Observations on 11 Nov 2009 at 1200 and 1800 UTC.
   - The PDF table swapped columns post-landfall: Wind was reported as 30 kt and 20 kt, but was placed in column 6, which the extractor failed to map.
   - **Classification: B (Extraction issue)**.
4. **`NIO_2009_WARD` (Cyclone WARD, 2009) — 1 row:**
   - Source: `27_4e34f3_rsmc-2009.pdf`, page 83.
   - Observation on 15 Dec 2009 at 0000 UTC.
   - The wind value in the report is 25 kt. Optical character extraction introduced an internal space (`'2 5'`), causing string-to-float parsing to fail.
   - **Classification: B (Extraction issue)**.
5. **`NIO_2011_UNNAMED_2_9_1` (Cyclone KEILA, 2011) — 1 row:**
   - Source: `27_2f6165_rsmc-2011.pdf`, page 107.
   - Observation on 01 Dec 2011 at 0600 UTC.
   - The row contains narrative text ("Weakened into a well marked low pressure area...") instead of numeric observations.
   - **Classification: A (Legitimately missing in source)**.
6. **`NIO_2002_UNNAMED_25` (2002) — 1 row:**
   - Source: `27_54cee6_35_6641c5_2002.pdf`, page 25.
   - Observation on 12 Nov 2002 at 1200 UTC.
   - Storm crossed Bengal coast near Sagar Island; post-landfall inland observations contain no wind reports.
   - **Classification: A (Legitimately missing in source)**.

---

## 5. Task 6 — Physical Range & Basin Coordinate Verification

All numerical label fields were checked against physical domain constraints:

| Parameter | Domain Constraint | Observed Range | Anomalies | Notes |
|:---|:---:|:---:|:---:|:---|
| **Latitude** | $[-5.0^\circ, +35.0^\circ\text{N}]$ | $[3.0^\circ, 87.0^\circ]$ | **2 rows out-of-basin** | In 2002, OCR swapped lat/lon for two observations |
| **Longitude** | $[40.0^\circ, 105.0^\circ\text{E}]$ | $[1.5^\circ, 99.7^\circ]$ | **5 rows out-of-basin** | In 2002, OCR swapped lat/lon for two observations |
| **Pressure** | $[880, 1020]\text{ hPa}$ | $[920, 1010]\text{ hPa}$ | **0 anomalies** | Valid physical cyclone pressure range |
| **Wind** | $[15, 170]\text{ kt}$ | $[15, 130]\text{ kt}$ | **0 anomalies** | Valid physical cyclone wind range |

### Detailed Coordinate Anomalies (Scanned 2002 Reports):
- `sample_id = NIO_2002_UNNAMED_25_20021110_1200Z`:
  - `imd_lat_t0 = 82.5`, `imd_lon_t0 = 2.0` (Row 9, page 25 of 2002 report). Longitude $82.5^\circ\text{E}$ was read into the latitude column, and C.I. $2.0$ was read into the longitude column.
  - `imd_lon_12h = 13.5` (Row 11, page 25 of 2002 report).
- `sample_id = NIO_2002_UNNAMED_29_20021123_1200Z`:
  - `imd_lat_t0 = 87.0`, `imd_lon_t0 = 1.5` (Row 9, page 29 of 2002 report). Longitude $87.0^\circ\text{E}$ was read into latitude.
- In `imd_best_track_v2.csv`, these rows are already flagged with `manual_review_required = True, manual_review_reason = 'OCR_OBSERVATION'`.

---

## 6. Generated QA Artifacts

1. **`data/manifests/vayu_net_label_qa_issues.csv`**  
   Contains 208 itemized anomaly records with schema:
   `sample_id, storm_id, timestamp, field, sample_index_value, imd_v2_value, raw_source_value, source_file, source_page, source_table, source_row, classification, reason, recommended_action`.
2. **`data/interim/vayu_net_sample_index_label_qa.json`**  
   Machine-readable telemetry summarizing invalid category counts, missing wind counts, traceability classifications, and root cause mechanics.
3. **`docs/vayu_net_sample_index_label_qa.md`**  
   This authoritative technical documentation.

---

## 7. Recommended Remediation Strategy (Pending User Approval)

To achieve 100% clean, schema-compliant labels without synthetic data fabrication:
1. **Targeted PDF Re-Extraction of Page 70 in RSMC 2018:**
   Apply a specialized 10-column handler for Table 2.3.1 page 70 that correctly maps:
   - Col 7 $\to$ `maximum_sustained_wind_kt`
   - Col 9 $\to$ `category` (`DD, CS, SCS, VSCS, ESCS, D`)
   - Col 8 $\to$ `pressure_drop_hpa`
2. **Targeted Fix for Phyan & Ward (RSMC 2009):**
   - Clean whitespace in `raw_wind` for Ward 2009 (`'2 5'` $\to$ `25.0`).
   - Re-extract post-landfall rows on page 78 for Phyan 2009 to recover wind.
3. **Casing Normalization for Gonu (RSMC 2007):**
   - Normalize `'SUCS'` to canonical schema `'SuCS'`.
4. **Coordinate Review for 2002 OCR Observations:**
   - Correct the swapped lat/lon columns for the 2 affected 2002 storms from the original scanned images, or exclude the 2 unphysical candidate windows.

---

---

## 8. Final QA Status & Post-Correction Resolution

### Post-Correction Remediation Summary
All 198 extraction-related issues identified during the forensic QA have been remediated via targeted source-supported corrections documented in `docs/imd_v2_targeted_correction_report.md`:
- **Cyclone Mekunu 2018 (Page 70)**: All 37 continuation rows reconstructed directly from primary source table; wind, CI, pressure drop, and categorical grades restored.
- **Cyclone Gonu 2007**: 2 rows normalized from `SUCS` $\to$ `SuCS`.
- **2002 Scanned Coordinates**: Unambiguous visual verification of high-resolution crops from physical pages 25 and 29 restored exact basin coordinates (`12.0 / 82.5`, `13.5 / 82.5`, `19.0 / 86.5`, `12.0 / 87.0`, `15.5 / 88.0`).
- **Cyclone Phyan 2009 & Ward 2009**: Post-landfall column shift and whitespace split repaired from source tables.
- **Legitimate Source NaNs**: 10 post-landfall dissipation observations explicitly retained as legitimate source `NaN` values.

```
LABEL QA POST-CORRECTION STATUS
===============================
Total samples: 1,319 (1,319 before / 1,319 after)
IMD Best Track V2 rows: 3,960 (3,960 before / 3,960 after)

Invalid t0 categories: 0
Invalid +12 categories: 0
Invalid +24 categories: 0
Invalid +48 categories: 0

Missing t0 wind: 0
Missing +12 wind: 0
Missing +24 wind: 0
Missing +48 wind: 10 (100% legitimate source-missing dissipation records)

Discrepancies between Sample Index and IMD V2: 0
GridSat frame path failures: 0 / 7,914
Source PDF modifications: 0 / 29
GridSat file modifications: 0 / 2,353

Overall Quality Status:
PASS
```

