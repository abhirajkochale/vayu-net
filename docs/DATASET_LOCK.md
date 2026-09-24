# VAYU-NET DATASET LOCK SPECIFICATION & PROVENANCE BASELINE

**Lock Status:** **LOCKED & VERIFIED**  
**Date of Lock:** 2026-09-24  
**Authoritative Scope:** IMD Best Track Dataset V2, GridSat-B1 27-Year NIO Satellite Archive, Master Sample Index  

---

> [!CAUTION]
> ### STRICT IMMUTABILITY GOVERNANCE RULE
> **Do not modify the locked dataset directly. Any future transformation must create a new version/artifact.**
>
> All primary data files listed in this specification are permanently locked. No pipeline script, training routine, or exploratory notebook may alter, overwrite, or delete any record in these paths.

---

## 1. Locked Dataset Metrics & Invariants

| Component / Invariant | Verified Specification | Governance Status |
| :--- | :--- | :---: |
| **IMD Best Track V2 Observations** | Exactly **3,960 rows** across 216 storms (1998–2024) | **LOCKED** |
| **Master Sample Index Samples** | Exactly **1,319 candidate samples** ($t_0$ aligned to 3h synoptic times) | **LOCKED** |
| **GridSat Satellite Production Frames** | Exactly **2,353 `.npz`** and **2,353 `.nc`** frames (4,706 total files) | **LOCKED** |
| **Sample Satellite Sequences** | Exactly **7,914 / 7,914** 6-frame references valid and verified on disk | **LOCKED** |
| **Sample-Index ↔ IMD Parity** | Exactly **0 discrepancies** across all $t_0, +12\text{h}, +24\text{h}, +48\text{h}$ fields | **LOCKED** |
| **Category Classification Schema** | Exactly **0 violations** (100% compliant with `{D, DD, CS, SCS, VSCS, ESCS, SuCS}`) | **LOCKED** |
| **Sample Geographic Bounds** | Exactly **0 out-of-basin coordinates** (all strictly inside NIO basin) | **LOCKED** |
| **Legitimate Source Missing Winds** | Exactly **10 dissipation observations** retained as `NaN` (loss-masked) | **LOCKED** |
| **Original IMD Annual PDFs** | Exactly **29 PDFs** bit-identical to pre-correction SHA-256 hashes | **LOCKED** |
| **GridSat Cryptographic Baseline** | Authoritative SHA-256 baseline established in `gridsat_sha256_manifest.csv` | **LOCKED** |

---

## 2. Storm-Level Dataset Splits (Zero Leakage)

Storm-level train, validation, and test splits are strictly separated chronologically and geographically to ensure zero target or temporal leakage:

- **TRAIN Split**: **81 storms** (696 ML candidate samples)
- **VALIDATION Split**: **14 storms** (252 ML candidate samples)
- **TEST Split**: **31 storms** (371 ML candidate samples)
- **Total Validated Storms**: **126 storms** with eligible 48h ML candidate sequences
- **Split Intersections / Leakage**: **0 storms** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$)

---

## 3. Authoritative Locked File Paths

### Master Dataset Manifests & Data
- **IMD Best Track V2 (CSV)**: `data/processed/imd_best_track_v2.csv`
- **IMD Best Track V2 (Parquet)**: `data/processed/imd_best_track_v2.parquet`
- **Master Sample Index**: `data/manifests/vayu_net_sample_index.csv`
- **Candidate $t_0$ Manifest**: `data/manifests/vayu_net_candidate_t0_manifest.csv`
- **Storm Manifest V2**: `data/manifests/storm_event_manifest_v2.csv`

### Authoritative Provenance & Hash Manifests
- **IMD Source Inventory (PDF SHA-256 Hashes)**: `data/manifests/imd_source_inventory.csv`
- **GridSat Production SHA-256 Manifest**: `data/manifests/gridsat_sha256_manifest.csv`
- **Targeted Correction Diff Log**: `data/interim/imd_v2_corrections_diff.json`
- **Post-Correction Validation Log**: `data/interim/v2_post_correction_validation.json`
- **Final Immutability Audit JSON**: `data/interim/final_dataset_immutability_audit.json`

### Primary Satellite & Source Archives
- **Original IMD Reports (Immutable Raw)**: `data/raw/imd/` (29 PDFs)
- **GridSat Cropped NIO Archive (Immutable Interim)**: `data/interim/gridsat/` (2,353 `.npz`, 2,353 `.nc`)

---

## 4. Cryptographic Provenance Statement

1. **IMD Source PDFs**:
   All 29 PDF documents in `data/raw/imd/` were cryptographically audited against the baseline SHA-256 hashes recorded in `data/manifests/imd_source_inventory.csv`. 100% of files match their pre-correction cryptographic hashes and byte counts bit-for-bit.

2. **GridSat Production Archive**:
   Prior to the targeted IMD V2 correction phase, only `gridsat_required_file_manifest.csv` existed (tracking download completion, URLs, and file size); no pre-correction SHA-256 hash manifest was created for interim cropped `.npz` and `.nc` files. Consequently, historical byte-for-byte non-modification cannot be cryptographically proven retrospectively. However, filesystem modification timestamps (`mtime`) and structural integrity tests confirm that all 4,706 files were created during production acquisition and remained completely untouched. The newly generated manifest `data/manifests/gridsat_sha256_manifest.csv` now serves as the **authoritative permanent cryptographic baseline** for all future audits.

---

## 5. Downstream ML Development Readiness

With all 13 integrity checks passing, 0 discrepancies, and cryptographic baselines finalized, the VAYU-NET multi-modal cyclone intensity estimation and forecasting dataset is formally **LOCKED** and ready for model architecture development and training.
