# VAYU-NET Final Dataset Immutability Audit Report

**Audit Timestamp:** `2026-09-23T19:53:22.476061+00:00`  
**Final Verdict:** **`DATASET LOCK READY`**  

> [!IMPORTANT]
> **Read-Only Final Verification**
> This audit evaluated cryptographic SHA-256 hashes, file modification times, and structural invariants without modifying any dataset files.

## 1. Summary of Audit Findings

| Verification Dimension | Target | Result | Status |
| :--- | :--- | :--- | :---: |
| **Original IMD PDFs (SHA-256)** | 29 files bit-identical to baseline | 29 / 29 bit-identical | **PASS** |
| **GridSat Archive Files** | 2,353 .npz + 2,353 .nc present | 2,353 .npz + 2,353 .nc verified | **PASS** |
| **GridSat Hash Baseline Availability** | Pre-correction hash manifest | **HASH BASELINE UNAVAILABLE** (Established now in manifest) | **NOTE** |
| **IMD Best Track V2 Rows** | Exactly 3,960 | 3960 | **PASS** |
| **Candidate ML Samples** | Exactly 1,319 | 1319 | **PASS** |
| **Sample Index ↔ IMD V2 Parity** | 0 discrepancies | 0 discrepancies | **PASS** |
| **Invalid Category Labels** | 0 non-schema values | 0 non-schema values | **PASS** |
| **Out-of-Basin Coordinates** | 0 sample coordinate errors | 0 errors (all within NIO basin) | **PASS** |
| **GridSat Frame References** | 7,914 / 7,914 resolved | 7,914 / 7,914 resolved | **PASS** |
| **Split Separation (Storm Leakage)**| 0 intersections | 0 intersections (Train 81, Val 14, Test 31) | **PASS** |

---

## 2. IMD Source PDF SHA-256 Cryptographic Audit

- **Baseline Source:** `data/manifests/imd_source_inventory.csv`
- **Result:** All 29 PDFs in `data/raw/imd/` match the pre-correction baseline SHA-256 hashes and byte lengths with 100% precision. Zero modifications were detected.

| Filename | Size (Bytes) | SHA-256 Hash | Status |
| :--- | :---: | :--- | :---: |
| `27_db9543_35_7625bf_1997.pdf` | 6,627,556 | `7279d8b54fa55109...70035cce` | PASS |
| `27_01a386_35_872370_1998.pdf` | 3,637,965 | `4a2aa494433ff4be...25ee8770` | PASS |
| `27_510109_35_66a331_1999.pdf` | 4,736,044 | `91b0dcec2d68dacd...27190735` | PASS |
| `27_5af63e_35_0c0afa_2000.pdf` | 3,947,945 | `d875d3bd1e9d3b54...4fb1220d` | PASS |
| `27_fa9a04_35_168f8a_2001.pdf` | 3,572,635 | `5e56d4b97ae73c11...fce4754c` | PASS |
| `27_54cee6_35_6641c5_2002.pdf` | 2,629,013 | `a0e6c794548a49d8...213df733` | PASS |
| `27_becfa7_35_7946c7_2003.pdf` | 2,874,949 | `ca5d3a373a1abdb9...4e26e5bb` | PASS |
| `27_be26e0_35_2bd72b_2004.pdf` | 4,016,781 | `8e114c0443afa985...0e62aae4` | PASS |
| `27_684961_rsmc-2005 .pdf` | 2,262,382 | `37a073fec1d53d77...341a4c99` | PASS |
| `27_b1dd60_rsmc-2006 .pdf` | 3,238,066 | `3c07a50dc5d97f67...eef05e5c` | PASS |
| `27_82fddf_rsmc-2007.pdf` | 4,094,091 | `ee339620c765f8db...75e534ca` | PASS |
| `27_56679c_rsmc-2008.pdf` | 5,116,647 | `a5c98f617b232bd1...0fcaf214` | PASS |
| `27_4e34f3_rsmc-2009.pdf` | 4,219,277 | `ef3160860dc8aff9...e9549644` | PASS |
| `27_7aaf4d_rsmc-2010.pdf` | 6,514,321 | `7a626a365d743b6b...b0323c72` | PASS |
| `27_2f6165_rsmc-2011.pdf` | 15,544,946 | `26510ef934e02547...0e24570e` | PASS |
| `27_2138f3_rsmc-2012.pdf` | 11,036,196 | `c33813590c5831d8...35731898` | PASS |
| `27_14ab8f_rsmc-2013.pdf` | 36,277,331 | `406b0ffad9e72967...bb14e3f5` | PASS |
| `27_4e1280_rsmc-2014.pdf` | 28,292,334 | `50db2e7fc85700c8...73e6645f` | PASS |
| `27_9bbd0d_RSMC-2015.pdf` | 16,086,198 | `fe6325b81136e066...ba89c0b6` | PASS |
| `27_ad292c_rsmc-2016.pdf` | 59,713,613 | `1c862162f981a8cf...eafdcf33` | PASS |
| `27_bbaf11_rsmc-2017.pdf` | 20,822,211 | `bc8014cb715733de...57864910` | PASS |
| `27_60dae9_rsmc-2018.pdf` | 80,308,759 | `2966e0f7899b0790...c92f155b` | PASS |
| `27_fddc6c_rsmc2020.pdf` | 80,716,260 | `21a6b7af66ee163b...f326ba16` | PASS |
| `27_26e77b_rsmc-2020 with damage.pdf` | 84,011,933 | `9f968b2ad68131ac...eb6e3138` | PASS |
| `27_81bdf0_RSMC Report 2021.pdf` | 59,943,504 | `2849fceb6442e949...1c26e12b` | PASS |
| `27_501da8_RSMC full report 2022 13 Jan.pdf` | 81,695,068 | `172e6491010c7ea4...0d751e71` | PASS |
| `27_7d3be4_Final_upload RSMC Report 2024 (1).pdf` | 86,828,972 | `3a0d2c2b041d1767...8f4d38fc` | PASS |
| `27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf` | 82,880,963 | `a87291681b238647...b7b9fc6e` | PASS |
| `27_c8dbd0_0_COVER_PAGE_RSMC_Report_2025.pdf` | 98,097,630 | `0dbd28e2ea006276...25568bab` | PASS |

---

## 3. GridSat Production Archive Cryptographic Audit

> [!NOTE]
> **Hash Baseline Availability Statement**
> **HASH BASELINE UNAVAILABLE**: No pre-correction cryptographic SHA-256 hash manifest was created for interim .npz and .nc files in data/interim/gridsat/ (gridsat_required_file_manifest.csv tracked raw source URLs, timestamps, and download status only).
> To permanently enforce cryptographic immutability moving forward, a complete SHA-256 manifest for all 2,353 `.npz` and 2,353 `.nc` interim files has been generated and saved to `data/manifests/gridsat_sha256_manifest.csv`.

- **Files Verified on Disk:** 2353 `.npz` files and 2353 `.nc` files across 1998–2024.
- **Pre-Correction MTime Verification:** File modification timestamps confirm all files were generated during the production acquisition phase and have remained completely untouched.
- **Structural Parity:** Matches 100% of the 2,353 required entries in `data/manifests/gridsat_required_file_manifest.csv`.

---

## 4. Final Lock Invariants Certification

- **IMD Best Track V2**: 3,960 rows (CSV and Parquet bit-verified).
- **Master Sample Index**: 1,319 samples with 0 discrepancies against IMD V2.
- **Classification Schema**: 100% compliant with `{D, DD, CS, SCS, VSCS, ESCS, SuCS}`.
- **Downstream Regression/Classification Labels**: Valid coordinates and retained legitimate dissipation NaNs.
- **Satellite Frame Integrity**: 7,914 / 7,914 6-frame sequences intact and validated.

### Final Verdict: **DATASET LOCK READY**

---

## 5. Data Governance & Cryptographic Provenance Baseline

> [!CAUTION]
> **Authoritative Immutability & Modification Rule**
> **Do not modify the locked dataset directly. Any future transformation must create a new version/artifact.**

### Provenance Baseline Declarations:
1. **IMD Source Documents (`data/raw/imd/`)**:
   - Verified against the pre-existing SHA-256 cryptographic baseline in `data/manifests/imd_source_inventory.csv`.
   - All 29 original IMD annual report PDFs are certified bit-identical to their pre-correction state.
2. **GridSat Production Archive (`data/interim/gridsat/`)**:
   - **No pre-correction SHA-256 baseline existed for GridSat**: Prior to the targeted IMD V2 correction phase, only `gridsat_required_file_manifest.csv` was tracked (recording URLs, timestamps, and download completion status).
   - **Historical byte-for-byte non-modification cannot be cryptographically proven retrospectively**: While filesystem modification timestamps (`mtime`) and structural checks confirm all 2,353 `.npz` and 2,353 `.nc` files pre-date the correction phase, cryptographic non-modification prior to lock cannot be proven retrospectively in the absence of an earlier hash manifest.
   - **Authoritative Baseline Established**: The current GridSat SHA-256 manifest (`data/manifests/gridsat_sha256_manifest.csv`) is now formally established as the **authoritative, permanent cryptographic baseline** for all future integrity audits.