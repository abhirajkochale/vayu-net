# VAYU-NET IMD Source Audit — Executive Summary

This document provides a high-level executive summary of the comprehensive source audit conducted on the 29 IMD Tropical Cyclone annual reports in `data/raw/imd/` for Project VAYU-NET (SIH Problem Statement 26070).

## Key Findings at a Glance

| Metric | Result | Impact on VAYU-NET |
|:---|:---:|:---|
| **Total PDFs Audited** | 29 | Complete catalog across entire archive. |
| **Archive Year Span** | 1997–2025 (29 years) | Uninterrupted annual sequence. |
| **Core ML Period (1998–2024)** | 27 of 27 years present (100%) | Complete source material for ML training/val/test splits. |
| **Missing Years in Core Period** | 0 | No data collection gap in source PDFs. |
| **Apparent Duplicate Filenames** | 2 pairs (2020 & 2024) | Resolved by content audit: NOT duplicates; each represents distinct years (2019 vs 2020, 2023 vs 2024). |
| **Special Demo Storms** | 5 of 5 verified | FANI (2019), AMPHAN (2020), TAUKTAE (2021), BIPARJOY (2023), REMAL (2024) all present in primary sources. |
| **Digital Searchable Reports** | 21 reports (2005–2025) | Text-based, structured tables ready for direct programmatic parsing. |
| **Scanned Image Reports** | 8 reports (1997–2004) | Bitmap pages; 1998–2004 contain Best Track tables requiring OCR normalization. |
| **2025 Report Status** | Complete 397-page report | Retained exclusively as blind future-year evaluation set (frozen from training). |
| **1997 Report Status** | Complete 95-page report | Narrative & track maps only; retained as historical supplement. |

## Chronological Data Split Feasibility

- **TRAIN (1998–2018)**: 21 continuous years. All years contain Best Track tables. 1998–2004 will require OCR table extraction; 2005–2018 are digital text.
- **VALIDATION (2019–2020)**: 2 continuous years. Both digital text. Contains FANI (2019) and AMPHAN (2020).
- **TEST (2021–2024)**: 4 continuous years. All digital text. Contains TAUKTAE (2021), MANDOUS (2022), BIPARJOY (2023), and REMAL (2024).

## Apparent Duplicate Resolution

The two suspected duplicate groups were identified as external naming misnomers:

1. **`rsmc2020.pdf` vs `rsmc-2020 with damage.pdf`**:
   - `rsmc2020.pdf` is actually the **2019 Annual Report** (Report No. 01 (2020)/10). Covers Fani, Bulbul, etc.
   - `rsmc-2020 with damage.pdf` is the **2020 Annual Report** (Report No. 01 (2021)/11). Covers Amphan, Nisarga, etc.
   - *Conclusion*: Both are canonical and necessary; neither should be deleted or merged.
2. **`Final_upload RSMC Report 2024 (1).pdf` vs `RSMC-Report2024-for upload.pdf`**:
   - `Final_upload RSMC Report 2024 (1).pdf` is actually the **2023 Annual Report** (published Jan 2024). Covers Biparjoy, Mocha, etc.
   - `RSMC-Report2024-for upload.pdf` is the **2024 Annual Report** (Report No. 01 (2025)/15). Covers Remal, Asna, Dana.
   - *Conclusion*: Both are canonical and necessary; neither should be deleted or merged.

## Manifest and Provenance

- Full machine-readable inventory: [`data/manifests/imd_source_inventory.csv`](../data/manifests/imd_source_inventory.csv)
- Detailed audit report: [`docs/imd_source_audit.md`](imd_source_audit.md)
- Source PDFs remain 100% untouched and preserved in `data/raw/imd/`.