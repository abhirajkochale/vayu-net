# VAYU-NET: INSAT-3D Provenance Reset & Quarantine Protocol

**Document Version:** 1.0.0  
**Effective Date:** September 25, 2026  
**Status:** OFFICIAL REPOSITORY AUDIT & QUARANTINE DIRECTIVE  

---

## 1. Executive Summary & Mandatory Provenance Statement

> [!CAUTION]
> ### MANDATORY SCIENTIFIC PROVENANCE DECLARATION
> **"The previous multisource INSAT experiments used GridSat-derived surrogate channels and are not evidence of native INSAT-3D predictive performance."**

Following a deep data-lineage audit of the VAYU-NET research pipeline, it was established that all local arrays designated as "INSAT-3D" (including `data/interim/insat3d_pilot/`, `data/manifests/insat_pretraining_manifest.csv`, and associated multisource tensors) were mathematically synthesized or copied from NOAA GridSat-B1 `irwin_cdr` infrared channels rather than ingested from real ISRO/MOSDAC INSAT-3D Level-1C HDF5 archives.

As a result:
1. **Zero native INSAT-3D observations** were used in previous multi-source, transfer-learning, or decoupled experiments.
2. **Historical numerical metrics** (e.g., reported MAE deltas of -0.87 kt or track improvements) reflect multi-channel processing of GridSat-derived perturbations, NOT true physical cross-sensor fusion between geostationary INSAT-3D and GridSat-B1.
3. Historical artifacts **must NOT be deleted**, ensuring full scientific auditability. They are formally classified and quarantined below.
4. **No model training** using surrogate channels is permitted.

---

## 2. Root Cause Analysis

### 2.1 The Pilot Construction Bug
In early development phases, `scratch/build_pilot_artifacts.py` was used to mock the multisource pipeline while waiting for MOSDAC data pipelines. That script loaded GridSat `irwin_cdr` NetCDF files from `data/interim/gridsat/`, clipped or scaled them, and exported them as `.npz` files into `data/interim/insat3d_pilot/` with keys `['TIR1', 'TIR2', 'WV']`:
- `TIR1` was copied directly from GridSat `irwin_cdr`.
- `TIR2` was created via synthetic perturbation: `irwin_cdr + offset`.
- `WV` was created via synthetic non-linear transformation of `irwin_cdr`.

### 2.2 Propagation into Pretraining and Transfer Experiments
Subsequent scripts (`build_multisource_cache.py`, `build_insat_pretraining_dataset.py`, `train_multisource_transfer.py`, `train_multisource_decoupled.py`) ingested these arrays or indexed the raw `data/interim/gridsat/` directory under the filename assumption that they were INSAT files. Consequently:
- `data/manifests/insat_pretraining_manifest.csv` indexed 1,428 NOAA GridSat-B1 files under the column label `satellite: INSAT-3D`.
- Checkpoints trained on these inputs learned representations of GridSat IR, not INSAT-3D Imager radiometry.

---

## 3. Artifact Provenance Classification Table

In accordance with strict scientific integrity standards, artifacts across the repository are formally classified into two mutually exclusive tiers:

| Tier | Classification Definition | Policy |
| :--- | :--- | :--- |
| **VALID** | Genuine GridSat-only experiments, official IMD Best Track data, production models trained exclusively on real GridSat-B1. | **Retained in active production / research baseline.** |
| **SURROGATE / INVALID FOR INSAT CLAIM** | All datasets, manifests, caches, models, and evaluation reports that utilized synthetic or GridSat-derived INSAT surrogate channels. | **QUARANTINED. Retained for scientific reproducibility; strictly forbidden from being cited as INSAT-3D evidence.** |

### 3.1 Granular Artifact Inventory

| Category | Artifact Path | Original Claim | True Provenance | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Production Model** | `models/vayu_net_production.pt` (and core weights) | GridSat-B1 Single-Source | NOAA GridSat-B1 (genuine) | **VALID** |
| **IMD Ground Truth** | `data/interim/imd/`, `data/manifests/storm_catalog.csv` | IMD Best Track 1982–2024 | RSMC New Delhi Official Bulletins | **VALID** |
| **GridSat Archive** | `data/interim/gridsat/**/*.nc` | NOAA NCEI GridSat-B1 CDR | NOAA CDR Archive (genuine) | **VALID** |
| **Interim Data** | `data/interim/insat3d_pilot/*.npz` | Native INSAT-3D Pilot (Fani/Amphan/Remal) | GridSat `irwin_cdr` + synthetic offset | **SURROGATE (INVALID)** |
| **Manifest** | `data/manifests/insat3d_pilot_samples.csv` | INSAT-3D Pilot Alignment | GridSat timestamps & synthetic pilot | **SURROGATE (INVALID)** |
| **Manifest** | `data/manifests/insat_pretraining_manifest.csv` | INSAT-3D Pretraining Archive (1,428 files) | NOAA GridSat-B1 `.npz` files | **SURROGATE (INVALID)** |
| **Manifest** | `data/manifests/vayu_net_multisource_sample_index.csv` | Multisource 757 paired samples | GridSat + synthetic pilot | **SURROGATE (INVALID)** |
| **Manifest** | `data/manifests/multisource_coverage_manifest.csv` | Multisource Storm Coverage | Synthetic pilot metadata | **SURROGATE (INVALID)** |
| **ML Cache** | `data/interim/ml/cache/insat_pretraining_cache.pt` | INSAT-3D Pretraining Tensor Cache | GridSat `irwin_cdr` patches | **SURROGATE (INVALID)** |
| **ML Cache** | `data/interim/ml/cache/multisource_dataset_cache.pt` | Multisource Supervised Cache | GridSat + synthetic INSAT | **SURROGATE (INVALID)** |
| **ML Cache** | `data/interim/ml/cache/multisource_transfer_cache*.pt` | Expanded Transfer Learning Cache | GridSat + synthetic INSAT | **SURROGATE (INVALID)** |
| **Checkpoint** | `data/interim/ml/checkpoints/insat_pretrained_encoder.pt` | INSAT-3D Masked Autoencoder | GridSat IR Autoencoder | **SURROGATE (INVALID)** |
| **Checkpoint** | `data/interim/ml/checkpoints/multisource_fusion_*.pt` | Early/Late Multisource Fusion | GridSat + synthetic INSAT | **SURROGATE (INVALID)** |
| **Checkpoint** | `data/interim/ml/checkpoints/multisource_transfer_*.pt` | Transfer Learning Experiments 1–3 | GridSat + synthetic INSAT | **SURROGATE (INVALID)** |
| **Checkpoint** | `data/interim/ml/checkpoints/multisource_decoupled_*.pt` | Decoupled Multi-Source Models | GridSat + synthetic INSAT | **SURROGATE (INVALID)** |
| **Results** | `data/interim/ml/multisource_*_results.json` | Empirical Multi-Source Evaluation | Synthetic channel evaluation | **SURROGATE (INVALID)** |
| **Documentation** | `docs/multisource_transfer_learning.md` | INSAT Transfer Learning Study | Based on surrogate data | **SURROGATE (INVALID)** |
| **Documentation** | `docs/multisource_decoupled_report.md` | Decoupled Routing Study | Based on surrogate data | **SURROGATE (INVALID)** |
| **Documentation** | `docs/insat_multisource_feasibility_audit.md` | Feasibility Assessment | Conflated catalog with local files | **SURROGATE (INVALID)** |

---

## 4. Quarantine Rules and Governance

1. **Non-Destructive Preservation:**
   Historical files remain on disk so that peer-reviewers, collaborators, and internal auditors can replicate the audit findings and verify that no results were silently doctored.
2. **Strict Prohibition on Production Promotion:**
   Under no circumstances may any model checkpoint from `data/interim/ml/checkpoints/multisource_*` or `insat_pretrained_*` be deployed to `apps/backend/` or promoted to production.
3. **No Retroactive Rewriting of Historical Numbers:**
   Historical reports (`docs/multisource_*.md`) must remain intact as historical records of surrogate experimentation, but each is superseded by this document.
4. **Zero Model Training Mandate:**
   No further neural network training claiming multi-source INSAT fusion may take place until genuine Level-1C HDF5 data are acquired, independently authenticated, calibrated, and validated.

---

## 5. Path Forward: The Native INSAT-3D Pipeline

To establish genuine multi-sensor capabilities, VAYU-NET transitions entirely to the **Native INSAT-3D Acquisition Pipeline**:
- **Source Authority:** ISRO Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC).
- **Target Product:** INSAT-3D Imager Level-1C Asian Sector (`3DIMG_L1C_ASIA_MER`).
- **Data Format:** Genuine Hierarchical Data Format 5 (`.h5`).
- **Channels:** Pure native radiometry: `IMG_TIR1` (10.8 µm), `IMG_TIR2` (12.0 µm), and `IMG_WV` (6.8 µm).
- **Calibration:** Look-Up Table (LUT) count-to-Kelvin transformation extracted directly from native HDF5 dataset metadata.
- **Independent Validation:** Comprehensive programmatic checks (`scripts/audit/validate_native_insat3d.py`) enforcing authentic HDF5 file headers, valid coordinates, and zero synthetic derivation.
