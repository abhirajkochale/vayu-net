# IMD Best Track Quality Report

**Generated:** 2026-09-23T06:20:41Z
**Validation Verdict: PASS WITH WARNINGS**

---

## 1. Dataset Counts

| Metric | Value |
|:-------|:------|
| Total observation rows | 4181 |
| Total unique storms | 223 |
| Years covered | 1998–2024 |
| Digital extraction rows | 3852 |
| OCR extraction rows | 329 |
| Manual-review rows | 2255 |

## 2. Split Counts

| Split | Storms | Observations |
|:------|:-------|:-------------|
| TRAIN | 163 | 2749 |
| VALIDATION | 19 | 516 |
| TEST | 41 | 916 |
| BLIND | 0 | 0 |

## 3. Missing Values

| Field | Missing Count |
|:------|:--------------|
| `timestamp_utc` | 225 |
| `latitude` | 83 |
| `longitude` | 86 |
| `maximum_sustained_wind_kt` | 196 |
| `central_pressure_hpa` | 194 |
| `category` | 160 |

## 4. Quality Metrics

| Check | Count |
|:------|:------|
| Duplicate observation rows | 0 |
| Invalid coordinate rows | 0 |
| Missing wind rows | 196 |
| Missing pressure rows | 194 |
| Missing category rows | 160 |
| Timestamp anomaly storms | 0 |
| Duplicate timestamps within storms | 222 |

## 5. Manual Review Summary

| Problem Type | Count |
|:-------------|:------|
| DUPLICATE_ROW | 47 |
| MISSING_COORDINATES | 88 |
| MISSING_TIMESTAMP | 270 |
| NON_OBSERVATION_NARRATIVE_ROW | 756 |
| REVIEW_REQUIRED | 242 |
| UNCERTAIN_TABLE_IDENTITY | 852 |

## 6. Integrity Test Results

| Test | Pass/Fail | Notes |
|:-----|:----------|:------|
| TEST1_all_years_represented | ✓ PASS |  |
| TEST2_no_report_skipped | ✓ PASS |  |
| TEST3_1997_excluded | ✓ PASS |  |
| TEST4_2025_excluded | ✓ PASS |  |
| TEST5_no_storm_in_multiple_splits | ✓ PASS |  |
| TEST6_no_duplicate_rows | ✓ PASS | Duplicates: 0 |
| TEST7_all_rows_have_provenance | ✓ PASS |  |
| TEST8_reference_storms_present | ✓ PASS | All 5 found |
| TEST9_csv_parquet_readback | ✓ PASS |  |
| TEST10_schema_stable | ✓ PASS |  |
| TEST11_no_silent_ocr_corrections | ✓ PASS |  |
| TEST12_no_fabricated_values | ✓ PASS |  |
| TEST13_no_gridsat_downloaded | ✓ PASS |  |
| TEST14_source_pdfs_unchanged | ✓ PASS |  |

## 7. Reference Storm Details

| Storm ID | Found | Obs | Issues |
|:---------|:------|:----|:-------|
| `NIO_2019_FANI` | ✓ | 64 | None |
| `NIO_2020_AMPHAN` | ✓ | 36 | None |
| `NIO_2021_TAUKTAE` | ✓ | 40 | None |
| `NIO_2023_BIPARJOY` | ✓ | 97 | None |
| `NIO_2024_REMAL` | ✓ | 30 | None |

## 8. Critical Failures

- None

## 9. Warnings

- None

---

## Final Status: **PASS WITH WARNINGS**

> Pipeline completed. Non-critical issues require attention but do not block the dataset.
> See manual review CSV for items requiring human confirmation.