# VAYU-NET: NASA GPM IMERG Bulk Download Audit

**Product:** NASA GPM IMERG Final Run V07B (GPM_3IMERGHH.07)  
**Status:** COMPLETED & FULLY VALIDATED  
**Date:** 2026-09-25  
**Auditor:** Antigravity research pipeline  
**Constraint:** Read-only audit. No models trained. No production changes. No commit.  

---

## 1. Objective

Download and locally validate all 2,353 unique NASA GPM IMERG Final Run V07B granules
required by the VAYU-NET GridSat-B1 dataset, as catalogued in:

    data/manifests/gridsat_imerg_pairing_manifest.csv

Each granule provides a 30-minute precipitation snapshot (0.1° × 0.1°) that
pairs exactly with a GridSat-B1 synoptic observation at $\Delta t = 0$ minutes.

---

## 2. Manifest Inventory

| Manifest | Path | Granules / Sequences | Status |
|---|---|---|---|
| Frame-level pairing | `data/manifests/gridsat_imerg_pairing_manifest.csv` | 2,353 unique granules (7,914 frame rows) | VERIFIED |
| Sequence-level pairing | `data/manifests/gridsat_imerg_sequence_pairing_manifest.csv` | 1,319 sequences | VERIFIED |
| Bulk download provenance | `data/manifests/imerg_bulk_download_manifest.csv` | 2,353 granules | 100% PASS |

---

## 3. Product Identity & Scientific Metadata

| Property | Value |
|---|---|
| Product | NASA GPM IMERG Final Run |
| Collection | `GPM_3IMERGHH.07` |
| Algorithm Version | `V07B` |
| Temporal Resolution | 30 minutes |
| Spatial Resolution | 0.1° × 0.1° |
| Primary Variable | `/Grid/precipitation` |
| Physical Units | `mm/hr` |
| Fill Value | `-9999.9` (converted to NaN, zero artificial interpolation) |
| Grid Dimensions | `(1, 3600, 1800)` [time, lon, lat] |
| NIO Subgrid Dimensions | `(400, 650)` [lat: -5° to +35°, lon: 40° to 105°] |
| Model Grid Resampling | `(72, 116)` compatible |
| Data Format | HDF5 |
| Source Archive | NASA GES DISC (`data.gesdisc.earthdata.nasa.gov`) |

---

## 4. Download Execution Summary

| Metric | Verified Value |
|---|---|
| Granules Requested | **2,353** |
| Granules Staged Locally | **2,353** (100.0%) |
| Newly Downloaded | **2,350** |
| Pilot Granules Pre-existing | **3** |
| Granules Failed | **0** |
| Download Protocol | Authenticated GES DISC HTTPS via Earthdata URS session |
| Download Rate | ~12–18 granules / minute |
| Total Download Time | ~3 hours 14 minutes |
| Zero Future Leakage | **CONFIRMED** ($\Delta t = 0$ min synoptic alignment) |
| No Storm-Centered Cropping | **CONFIRMED** (Fixed NIO synoptic basin coordinates) |

---

## 5. Local Storage Footprint

| Metric | Value |
|---|---|
| Granule Count on Disk | **2,353** canonical HDF5 files |
| Total Storage Footprint | **17.325 GB** |
| Mean File Size | **7.36 MB** / granule |
| Min File Size | **6.11 MB** (well exceeds 5 MB threshold) |
| Max File Size | **8.63 MB** |
| Storage Directory | `data/raw/imerg/` |
| Filename Pattern | `3B-HHR.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.NNNN.V07B.HDF5` |

---

## 6. HDF5 Integrity Validation

Every single file in `data/raw/imerg/` was subjected to strict structural and scientific checks:

| Check | Requirement | Result |
|---|---|---|
| HDF5 Magic Header | 8-byte signature `0x89 H D F \r \n \x1a \n` | **2,353 / 2,353 PASS** |
| Dataset Readability | Readable without corruption via `h5py` | **2,353 / 2,353 PASS** |
| Primary Variable | `/Grid/precipitation` dataset present | **2,353 / 2,353 PASS** |
| Spatial Dimensions | `lon: 3600, lat: 1800` (global 0.1° grid) | **2,353 / 2,353 PASS** |
| Timestamp Embedded | ISO timestamp matches synoptic target | **2,353 / 2,353 PASS** |
| Version Check | `V07B` present in metadata and filename | **2,353 / 2,353 PASS** |

---

## 7. SHA-256 Provenance & Audit Trail

A complete SHA-256 cryptographic hash was generated for every downloaded granule and indexed into:
`data/manifests/imerg_bulk_download_manifest.csv`

Key schema fields recorded:
- `imerg_granule_filename`: Canonical NASA granule filename
- `gridsat_timestamp_utc`: Synoptic timestamp ($\Delta t = 0$)
- `download_url`: Exact GES DISC URL from which the file was fetched
- `local_path`: Local disk path
- `file_size_bytes`: Byte-level file size
- `sha256`: 64-character SHA-256 checksum
- `hdf5_valid`: `True`
- `product_valid`: `True`
- `variable_valid`: `True`
- `spatial_valid`: `True`
- `timestamp_valid`: `True`
- `download_status`: `DOWNLOADED` / `ALREADY_VALID`
- `validation_status`: `PASS` (2,353/2,353)

---

## 8. Sequence-Level Reconciliation

Cross-referencing `data/manifests/gridsat_imerg_sequence_pairing_manifest.csv`:

| Metric | Target | Verified |
|---|---|---|
| Total Sequences | 1,319 | 1,319 (100%) |
| Frames Required per Sequence | 6 | 6 |
| Sequences with Full 6-Frame IMERG Coverage | 1,319 | **1,319 (100.0%)** |
| Storm Coverage | 126 storms | **126 / 126 storms** |
| Train Sequences Covered | 696 | **696 / 696 (100%)** |
| Val Sequences Covered | 252 | **252 / 252 (100%)** |
| Test Sequences Covered | 371 | **371 / 371 (100%)** |

---

## 9. Comprehensive Validation Test Results

All four audit and integrity validation suites pass cleanly:

| Test Suite Script | Tests | Result | Status |
|---|---|---|---|
| `scripts/audit/validate_imerg_bulk_download.py` | 12 | **12 / 12 PASS** | ✅ CLEAN PASS |
| `scripts/audit/validate_gridsat_imerg_pairing.py` | 18 | **18 / 18 PASS** | ✅ CLEAN PASS |
| `scripts/audit/validate_imerg.py` | 11 | **11 / 11 PASS** | ✅ CLEAN PASS |
| `scripts/audit/validate_native_insat3d.py` | 10 | **10 / 10 PASS** | ✅ CLEAN PASS |

### Breakdown of 12-Point Post-Download Verification:
1. `[PASS]` 1. Bulk manifest exists: `imerg_bulk_download_manifest.csv`
2. `[PASS]` 2. Frame rows = 2,353 (expected 2,353)
3. `[PASS]` 3. Validation PASS: 2,353 / 2,353
4. `[PASS]` 4. Local presence: 2,353 / 2,353
5. `[PASS]` 5. HDF5 valid: 2,353 / 2,353
6. `[PASS]` 6. Product V07B: 2,353 / 2,353
7. `[PASS]` 7. Variable `/Grid/precipitation`: 2,353 / 2,353
8. `[PASS]` 8. Spatial dims `1800 × 3600`: 2,353 / 2,353
9. `[PASS]` 9. Timestamp parseable: 2,353 / 2,353
10. `[PASS]` 10. SHA-256 recorded: 2,353 / 2,353
11. `[PASS]` 11. HDF5 files on disk: $\ge 2,353$
12. `[PASS]` 12. Cross-check vs pairing manifest: 0 missing

---

## 10. Constraints & Safety Adherence

- **No models trained or fine-tuned.**
- **Production weights and checkpoints unchanged.**
- **Zero synthetic precipitation data created.**
- **No GridSat data masqueraded as IMERG.**
- **Zero temporal interpolation applied.**
- **Zero storm-centered cropping (fixed North Indian Ocean basin coordinates preserved).**
- **Zero code committed or pushed to remote.**
- **NASA Earthdata credentials retrieved exclusively from environment variables.**
- **All raw HDF5 files preserved in pristine uncompressed format.**

---

**Conclusion:** The NASA GPM IMERG Final Run V07B dataset is 100% downloaded, locally staged, scientifically verified, and ready for multimodal tensor pipeline construction.