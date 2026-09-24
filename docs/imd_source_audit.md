# VAYU-NET IMD Source Audit

## 1. Audit Summary

- **Total PDFs Audited**: 29 files located in `data/raw/imd/`.
- **Unique Report Years Found**: 29 consecutive years (1997 through 2025 inclusive).
- **Missing Years (1998–2024 Core Period)**: None (0 missing years). Every single year in the core ML window is accounted for.
- **Duplicate / Alternative Files**: 0 true redundant duplicates. Two suspected filename duplicate pairs (for 2020 and 2024) were forensically audited and resolved as distinct consecutive years (the 2019 report was labeled with publication identifier `rsmc2020`, and the 2023 report was labeled with upload year `Report 2024`).
- **Valid Core Best Track Reports (1998–2024)**: 27 reports total.
  - **Digital Text-Ready Reports (2005–2024)**: 20 reports (`CORE_BEST_TRACK_SOURCE`).
  - **Scanned Image-Based Reports (1998–2004)**: 7 reports (`USABLE_BUT_REQUIRES_NORMALIZATION`).
- **Partial / Cover-Only Reports**: 0 reports. (Even `27_c8dbd0_0_COVER_PAGE_RSMC_Report_2025.pdf` is an exhaustive 397-page complete annual report).
- **Auxiliary / Supplementary Reports**: 2 reports.
  - `1997`: Historical supplementary material (`HISTORICAL_SUPPLEMENT`, pre-core satellite ML period).
  - `2025`: Optional blind future-year evaluation set (`AUXILIARY_REPORT` / `FUTURE_EVALUATION_SET`, frozen from core ML training/tuning/validation).

## 2. Year-by-Year Audit

Below is the complete audit record for every PDF currently present in `data/raw/imd/`:

| Year | File | Report | Best Track | Required Fields | Complete? | Status | Notes |
|:---:|:---|:---|:---:|:---:|:---:|:---|:---|
| 1997 | `27_db9543_35_7625bf_1997.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 1997 | NO | PARTIAL | YES | `HISTORICAL_SUPPLEMENT` | Scanned report. Pre-dates core satellite ML period (1998-2024). Contains narrative descriptions, track maps, and radar fixes, but lacks standardized numerical Best Track observation tables. |
| 1998 | `27_01a386_35_872370_1998.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 1998 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Contains standardized Chapter 2 Best Track tables (Tables 2.1.1, 2.2.1, 2.3.1, etc.) with Date, UTC, Lat, Long, ECP, MSW, and Grade. Requires OCR/table extraction normalization. |
| 1999 | `27_510109_35_66a331_1999.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 1999 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Includes the 1999 Odisha Super Cyclone. Contains standardized Best Track tables (Tables 2.1.1, etc.) with all required fields. Requires OCR/table extraction normalization. |
| 2000 | `27_5af63e_35_0c0afa_2000.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2000 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Contains Chapter 2 Best Track tables (Tables 2.1.1, 2.2.1, etc.) with Date, UTC, Lat, Long, ECP, MSW, and Grade. Requires OCR/table extraction normalization. |
| 2001 | `27_fa9a04_35_168f8a_2001.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2001 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Contains Chapter 2 Best Track tables (Tables 2.1.1, 2.2.1, 2.3.1, etc.) with Date, UTC, Lat, Long, ECP, MSW, and Grade. Requires OCR/table extraction normalization. |
| 2002 | `27_54cee6_35_6641c5_2002.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2002 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Contains Chapter 2 Best Track tables (Tables 2.4.1, 2.5.1, 2.6.1, etc.) with Date, UTC, Lat, Long, ECP, MSW, and Grade. Requires OCR/table extraction normalization. |
| 2003 | `27_becfa7_35_7946c7_2003.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2003 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Contains Chapter 2 Best Track tables (Tables 2.1.1, 2.6.1, etc.) with Date, UTC, Lat, Long, ECP, MSW, and Grade. Requires OCR/table extraction normalization. |
| 2004 | `27_be26e0_35_2bd72b_2004.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2004 | YES | YES | YES | `USABLE_BUT_REQUIRES_NORMALIZATION` | Scanned report. Contains Chapter 2 Best Track tables (Tables 2.1, etc.) with Date, UTC, Lat, Long, ECP, MSW, and Grade. Requires OCR/table extraction normalization. |
| 2005 | `27_684961_rsmc-2005 .pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2005 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2005 cyclonic disturbances. |
| 2006 | `27_b1dd60_rsmc-2006 .pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2006 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2006 cyclonic disturbances. |
| 2007 | `27_82fddf_rsmc-2007.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2007 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2007 cyclonic disturbances (including Gonu and Sidr). |
| 2008 | `27_56679c_rsmc-2008.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2008 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2008 cyclonic disturbances (including Nargis). |
| 2009 | `27_4e34f3_rsmc-2009.pdf` | CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2009: A REPORT | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2009 cyclonic disturbances (including Aila). |
| 2010 | `27_7aaf4d_rsmc-2010.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2010 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2010 cyclonic disturbances (including Giri and Laila). |
| 2011 | `27_2f6165_rsmc-2011.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2011 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2011 cyclonic disturbances (including Thane). |
| 2012 | `27_2138f3_rsmc-2012.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2012 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2012 cyclonic disturbances (including Nilam). |
| 2013 | `27_14ab8f_rsmc-2013.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2013 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2013 cyclonic disturbances (including Phailin). |
| 2014 | `27_4e1280_rsmc-2014.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2014 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2014 cyclonic disturbances (including Hudhud). |
| 2015 | `27_9bbd0d_RSMC-2015.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2015 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2015 cyclonic disturbances (including Chapala and Megh). |
| 2016 | `27_ad292c_rsmc-2016.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2016 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2016 cyclonic disturbances (including Vardah). |
| 2017 | `27_bbaf11_rsmc-2017.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2017 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2017 cyclonic disturbances (including Ockhi). |
| 2018 | `27_60dae9_rsmc-2018.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2018 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2018 cyclonic disturbances (including Titli and Luban). |
| 2019 | `27_fddc6c_rsmc2020.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2019 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Filename rsmc2020.pdf is an external misnomer reflecting publication identifier No. MOES/IMD/RSMC-Tropical Cyclone Report No/01 (2020)/10. Content covers 2019 cyclone season including FANI. Sole source for 2019. |
| 2020 | `27_26e77b_rsmc-2020 with damage.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2020 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Publication identifier No. MOES/IMD/RSMC-Tropical Cyclone Report/01 (2021)/11. Covers 2020 cyclone season including AMPHAN. Includes damage assessment sections. Sole source for 2020. |
| 2021 | `27_81bdf0_RSMC Report 2021.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2021 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2021 cyclonic disturbances (including TAUKTAE and Yaas). |
| 2022 | `27_501da8_RSMC full report 2022 13 Jan.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2022 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Digital text report. Contains comprehensive Best Track tables for all 2022 cyclonic disturbances (including Asani, Sitrang, Mandous). |
| 2023 | `27_7d3be4_Final_upload RSMC Report 2024 (1).pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2023 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Filename Final_upload RSMC Report 2024 (1).pdf is an external misnomer reflecting publication/upload date (Jan 2024). Content covers 2023 cyclone season including BIPARJOY and Mocha. Sole source for 2023. |
| 2024 | `27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2024 | YES | YES | YES | `CORE_BEST_TRACK_SOURCE` | Publication identifier ESSO/MoES/IMD/RSMC-Tropical Cyclone Report/01(2025)/15. Covers 2024 cyclone season including REMAL, Asna, and Dana. Sole source for 2024. |
| 2025 | `27_c8dbd0_0_COVER_PAGE_RSMC_Report_2025.pdf` | REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2025 | YES | YES | YES | `AUXILIARY_REPORT` | Filename indicates COVER_PAGE but document is a complete 397-page annual report with Best Track tables. Excluded from training/validation per frozen project rules; retained as optional future blind evaluation set. |

## 3. Duplicate Analysis

The repository contained two suspicious filename pairs that appeared to be duplicates or alternative uploads. Forensic content analysis of both pairs resolved these apparent duplicates:

### Group 1: 2020 Filename Pair
- **Files Analyzed**:
  1. `27_fddc6c_rsmc2020.pdf` (80.7 MB, 429 pages, SHA-256: `65cefc36...`)
  2. `27_26e77b_rsmc-2020 with damage.pdf` (84.0 MB, 366 pages, SHA-256: `084e311f...`)
- **Similarities**: Both files have `rsmc2020` or `rsmc-2020` in their filenames and represent official IMD/RSMC annual cyclone publications.
- **Differences**:
  - `27_fddc6c_rsmc2020.pdf`: The official title is *"REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2019"*. Document Control Number: `No. MOES/IMD/RSMC-Tropical Cyclone Report No/01 (2020)/10`. Cover imagery is *Extremely Severe Cyclonic Storm FANI* (May 2019). It documents the 2019 cyclone season (Pabuk, Fani, Vayu, Hikaa, Kyarr, Maha, Bulbul, Pawan).
  - `27_26e77b_rsmc-2020 with damage.pdf`: The official title is *"REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2020"*. Document Control Number: `No. MOES/IMD/RSMC-Tropical Cyclone Report/01 (2021)/11`. Cover imagery is *Super Cyclonic Storm AMPHAN* (May 2020). It documents the 2020 cyclone season (Amphan, Nisarga, Gati, Nivar, Burevi) along with damage assessments.
- **Canonical Recommendation**:
  - Retain `27_fddc6c_rsmc2020.pdf` as the **Canonical Source for Year 2019**.
  - Retain `27_26e77b_rsmc-2020 with damage.pdf` as the **Canonical Source for Year 2020**.
- **Reason**: They are NOT duplicates. They cover two entirely distinct, consecutive cyclone seasons (2019 vs 2020). The filename `rsmc2020.pdf` was a superficial misnomer resulting from the IMD document publication identifier `(2020)` issued in early 2020 for the 2019 calendar season. Without `27_fddc6c_rsmc2020.pdf`, year 2019 would be completely missing.

### Group 2: 2024 Filename Pair
- **Files Analyzed**:
  1. `27_7d3be4_Final_upload RSMC Report 2024 (1).pdf` (86.8 MB, 438 pages, SHA-256: `f9ee2df7...`)
  2. `27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf` (82.9 MB, 416 pages, SHA-256: `fa213f56...`)
- **Similarities**: Both files have `RSMC Report 2024` in their filenames.
- **Differences**:
  - `27_7d3be4_Final_upload RSMC Report 2024 (1).pdf`: The official title is *"REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2023"*. Published in January 2024. Cover imagery is *Extremely Severe Cyclonic Storm BIPARJOY* (June 2023). It documents the 2023 cyclone season (Mocha, Biparjoy, Tej, Hamoon, Midhili, Michaung).
  - `27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf`: The official title is *"REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2024"*. Document Control Number: `ESSO/MoES/IMD/RSMC-Tropical Cyclone Report/01(2025)/15`. Cover imagery is *Severe Cyclonic Storm REMAL* (May 2024). It documents the 2024 cyclone season (Remal, Asna, Dana).
- **Canonical Recommendation**:
  - Retain `27_7d3be4_Final_upload RSMC Report 2024 (1).pdf` as the **Canonical Source for Year 2023**.
  - Retain `27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf` as the **Canonical Source for Year 2024**.
- **Reason**: They are NOT duplicates. They cover two entirely distinct, consecutive cyclone seasons (2023 vs 2024). The filename `Final_upload RSMC Report 2024 (1).pdf` was named based on its upload date in January 2024, but its contents cover the 2023 season. Without this file, year 2023 would be completely missing.

## 4. Core Dataset Recommendation

The audit **CONFIRMS** that the repository provides complete, uninterrupted source material for the locked **1998–2024 core ML period**:
- **Coverage**: All 27 consecutive years (1998 through 2024) are present.
- **Best Track Tables**: Detailed numerical Best Track tables exist for every year from 1998 to 2024, providing Date, UTC time, Latitude, Longitude, Central Pressure, Maximum Sustained Wind (MSW), and Category/Grade.
- **Special Demo Storms**: All five required demo/reference storms are present in complete reports:
  - **FANI (2019)**: Verified in `27_fddc6c_rsmc2020.pdf`
  - **AMPHAN (2020)**: Verified in `27_26e77b_rsmc-2020 with damage.pdf`
  - **TAUKTAE (2021)**: Verified in `27_81bdf0_RSMC Report 2021.pdf`
  - **BIPARJOY (2023)**: Verified in `27_7d3be4_Final_upload RSMC Report 2024 (1).pdf`
  - **REMAL (2024)**: Verified in `27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf`
- **Dataset Partition Feasibility**:
  - **TRAIN (1998–2018)**: 21 years available (1998–2004 scanned requiring OCR/normalization; 2005–2018 digital text).
  - **VALIDATION (2019–2020)**: 2 years available (both digital text; Fani and Amphan).
  - **TEST (2021–2024)**: 4 years available (all digital text; Tauktae, Mandous, Biparjoy, Remal).

## 5. 1997 and 2025 Handling

### 1997 Report (`27_db9543_35_7625bf_1997.pdf`)
- **Report Title**: *REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 1997* (95 pages).
- **Nature of Content**: Scanned image PDF. Contains meteorological narratives, cyclone histories, weather realized, damage summaries, satellite images, and radar fix diagrams.
- **Best Track Finding**: Unlike reports from 1998 onward, the 1997 publication does not format Best Track observations into explicit, tabular time series (no standalone 3-hourly/6-hourly Lat/Long/ECP/MSW/Grade tables).
- **Recommended Role**: `HISTORICAL_SUPPLEMENT`. Retained as historical supplementary material; excluded from core 1998–2024 satellite ML training and evaluation.

### 2025 Report (`27_c8dbd0_0_COVER_PAGE_RSMC_Report_2025.pdf`)
- **Report Title**: *REPORT ON CYCLONIC DISTURBANCES OVER NORTH INDIAN OCEAN DURING 2025* (397 pages).
- **Nature of Content**: Despite its filename containing `0_COVER_PAGE`, the file is a complete 397-page digital annual RSMC report with full Best Track tables (e.g., Table 2.3.1 on page 62), satellite imageries, and track forecasts.
- **Project Context Constraint**: The frozen project context mandates that 2025 must NOT be used in model training, tuning, or validation.
- **Recommended Role**: `FUTURE_EVALUATION_SET` / `AUXILIARY_REPORT`. Retained strictly as an optional future blind-test evaluation benchmark.

## 6. Missing Information and Manual Verification Notes

1. **No Years Missing**: No cyclone season years between 1998 and 2024 are missing.
2. **Scanned PDF Parsing Strategy (1998–2004)**: Years 1998–2004 contain complete Best Track tables but exist exclusively as scanned bitmap pages (0 text pages). High-accuracy tabular OCR with schema normalization will be required during extraction.
3. **Table Layout Evolutions**: Digital reports from 2005 to 2024 show minor changes in table formatting (e.g., column names evolving from `C.P. (hPa)` to `ECP (hPa)`, variations in time column notation, and addition of `C.I. No.` and `ΔP`). Normalization rules must handle these schema evolutions.

## 7. Next Step

The source audit is complete, verified, and locked. The recommended next data-engineering action is:

> Develop an automated Best Track extraction and normalization pipeline that:
> 1. Extracts tables from digital RSMC reports (2005–2024) using structured PDF table parsers.
> 2. Extracts and verifies tables from scanned RSMC reports (1998–2004) using high-fidelity OCR.
> 3. Normalizes all records into the canonical IMD Best Track schema: `storm_id`, `storm_name`, `year`, `timestamp_utc`, `latitude`, `longitude`, `maximum_sustained_wind_kt`, `central_pressure_hpa`, `category`, `source_file`, `source_page`.
> 4. Validates continuity across the 1998–2024 timeline before ingestion into the VAYU-NET ML training datasets.