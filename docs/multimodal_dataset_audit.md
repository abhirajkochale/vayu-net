# VAYU-NET: EXP-M1 Multimodal Dataset Construction & Quality Audit

**Experiment:** EXP-M1 — GridSat + IMERG Multimodal Baseline  
**Date:** 2026-09-25  
**Auditor:** Antigravity research pipeline  
**Status:** COMPLETED & CERTIFIED READY FOR TRAINING  
**Constraint Enforcement:** Dataset construction only. Zero models trained. Zero fine-tuning. Production checkpoints untouched. Zero git commits or pushes.  

---

## 1. Executive Summary & Inventory

| Section | Audit Dimension | Verified Result |
|---|---|---|
| **A** | **Total Source GridSat Sequences** | **1,319 sequences** |
| **B** | **Total Multimodal Sequences** | **1,319 sequences (100.0% retention)** |
| **C** | **Partition Counts** | **TRAIN: 696 \| VALIDATION: 252 \| TEST: 371** |
| **D** | **Unique Storms** | **126 storms (Zero cross-partition overlap)** |
| **E** | **IMERG Frame Coverage** | **6 frames / sequence = 7,914 frame observations (2,353 unique granules)** |
| **F** | **GridSat Frame Coverage** | **6 frames / sequence = 7,914 frame observations (100.0% present)** |
| **G** | **Timestamp Alignment** | **$\Delta t = 0$ minutes exact synoptic match across all 7,914 frames** |
| **H** | **Spatial Representation** | **Standardized to `[6, 72, 116]` for both GridSat and IMERG** |
| **I** | **IMERG Normalization** | **Standardized using TRAIN-ONLY statistics ($\mu = 0.1806$ mm/hr, $\sigma = 1.0841$ mm/hr)** |
| **J** | **NaN Statistics** | **0.00% missing pixels across the entire North Indian Ocean domain** |
| **K** | **Samples Rejected** | **0 samples rejected (0.0%)** |
| **L** | **Reasons for Rejection** | **None** |
| **M** | **Storage Inventory & Duplication** | **2,353 canonical primary files + 3 byte-identical pilot duplicates catalogued** |
| **N** | **Leakage Checks** | **Zero future observation leakage relative to $t_0$; strict temporal causality** |
| **O** | **Final Dataset Readiness** | **CERTIFIED READY FOR EXPERIMENT EXP-M1** |

---

## 2. Multimodal Sample & Tensor Contract

Each sample is an isolated dictionary loaded via `ml.data.multimodal_dataset.MultimodalVayuDataset`:

```python
{
    "gridsat":        torch.Tensor,  # Shape [6, 72, 116], dtype float32 (standardized Kelvin)
    "imerg":          torch.Tensor,  # Shape [6, 72, 116], dtype float32 (standardized mm/hr)
    
    # Ground Truth Multi-Horizon Targets
    "center_t0":      torch.Tensor,  # [lat, lon]
    "wind_t0":        torch.Tensor,  # scalar kt
    "wind_t0_mask":   torch.Tensor,  # binary mask
    "pressure_t0":    torch.Tensor,  # scalar hPa
    "pressure_t0_mask": torch.Tensor,
    "category_t0":    torch.Tensor,  # 7-class index (0: D to 6: SuCS)
    "category_t0_mask": torch.Tensor,
    
    "center_12h":     torch.Tensor,  # Forecast +12h
    "wind_12h":       torch.Tensor,
    "pressure_12h":   torch.Tensor,
    "category_12h":   torch.Tensor,
    
    "center_24h":     torch.Tensor,  # Forecast +24h
    "wind_24h":       torch.Tensor,
    "pressure_24h":   torch.Tensor,
    "category_24h":   torch.Tensor,
    
    "center_48h":     torch.Tensor,  # Forecast +48h
    "wind_48h":       torch.Tensor,
    "pressure_48h":   torch.Tensor,
    "category_48h":   torch.Tensor,
    
    # Metadata
    "sample_id":      str,
    "storm_id":       str,
    "storm_name":     str,
    "split":          str,           # "TRAIN" | "VALIDATION" | "TEST"
    "t0_utc":         str,
    "gridsat_paths":  list[str],
    "imerg_granules": list[str]
}
```

---

## 3. Strict Preprocessing & Normalization Protocol

### GridSat Modality:
- Source: Calibrated `irwin_cdr` infrared brightness temperature (Kelvin).
- Imputation: Pixels with temperature $< 100\text{ K}$, $> 380\text{ K}$, or NaN imputed with training mean.
- Spatial Grid: Downsampled to `(72, 116)` via bilinear interpolation.
- Standardization: $z = (T - \mu_{\text{train}}) / \sigma_{\text{train}}$ ($\mu = 268.3267\text{ K}$, $\sigma = 26.6896\text{ K}$).

### NASA GPM IMERG Final Run V07B Modality:
- Source: `/Grid/precipitation` from genuine HDF5 granules (`GPM_3IMERGHH.07`).
- Domain: Subgrid to North Indian Ocean basin $[-5.0^\circ, 35.0^\circ]\text{N}$, $[40.0^\circ, 105.0^\circ]\text{E}$ (zero storm-centered cropping).
- Fill Values: Product fill value $-9999.9$ strictly mapped to NaN without synthetic interpolation.
- Spatial Grid: Downsampled to `(72, 116)` via bilinear interpolation with NaN preservation.
- Training-Only Statistics Computed (`data/interim/ml/imerg_train_normalization_stats.json`):
  - Total Training Pixels: 10,891,008 (across 1,304 unique granules)
  - NaN Pixels: 0 (0.00% missing in NIO domain)
  - Training Mean ($\mu$): **0.1806 mm/hr**
  - Training Standard Deviation ($\sigma$): **1.0841 mm/hr**
  - Min / Max: **0.0000 / 116.5812 mm/hr**
  - Percentiles: $p_{50} = 0.0000$, $p_{90} = 0.1677$, $p_{95} = 0.7420$, $p_{99} = 4.0248$, $p_{99.9} = 13.4254$ mm/hr
  - Baseline Transformation: $z = (x - \mu) / \sigma$.
  - Rationale: Standardizes physical precipitation rate in mm/hr without non-linear compression distortion, providing a clean linear baseline for cross-modal attention. `log1p` statistics ($\mu = 0.0819, \sigma = 0.2991$) are also recorded for future non-linear ablations.

---

## 4. Storage & File Inventory Audit

A dedicated storage inventory catalog was generated at `data/manifests/imerg_storage_inventory.csv`:
- **Physical HDF5 files on disk:** 2,356 files (17.325 GB)
- **Canonical primary granules:** 2,353 files (100% genuine NASA GES DISC products)
- **Pilot aliases:** 3 byte-identical duplicate files (`20190429...`, `20200517...`, `20240525...`)
  - Cryptographic verification: 100% SHA-256 match with their canonical counterparts.
  - Retained safely for backward-compatibility with pilot validation tests.
  - Zero impact on dataset construction: `MultimodalVayuDataset` references canonical filenames exclusively.

---

## 5. Comprehensive Verification Suites

All test suites executed via pytest pass cleanly:

| Test Suite | Module | Test Count | Result |
|---|---|---|---|
| **EXP-M1 Multimodal Dataset** | `scripts/audit/validate_multimodal_dataset.py` | 22 | **22 / 22 PASS** |
| **IMERG Bulk Download** | `scripts/audit/validate_imerg_bulk_download.py` | 12 | **12 / 12 PASS** |
| **GridSat × IMERG Pairing** | `scripts/audit/validate_gridsat_imerg_pairing.py` | 18 | **18 / 18 PASS** |
| **IMERG Scientific Standards** | `scripts/audit/validate_imerg.py` | 11 | **11 / 11 PASS** |
| **Native INSAT-3D Pipeline** | `scripts/audit/validate_native_insat3d.py` | 10 | **10 / 10 PASS** |
| **Total Test Coverage** | **All Modules** | **73** | **73 / 73 PASS (100%)** |

---

## 6. End-to-End Partition Load Test Results

Executed via `scripts/audit/test_multimodal_load.py`:

```
=================================================================
VAYU-NET: EXP-M1 MULTIMODAL DATASET LOAD TEST
=================================================================

>>> LOADING TRAIN SAMPLE...
Dataset split count: 696 sequences
  sample_id:                  NIO_1998_UNNAMED_31_19981007_0300Z
  storm_id:                   NIO_1998_UNNAMED_31
  split:                      TRAIN
  t0_utc:                     1998-10-07T03:00:00Z
  GridSat shape:              (6, 72, 116)
  IMERG shape:                (6, 72, 116)
  GridSat stats (norm):       min=-3.890, max=1.850, mean=0.045, std=0.891
  IMERG stats (norm):         min=-0.167, max=35.545, mean=-0.039, std=0.752, NaNs=0
  Target t0:                  center=[14.0, 69.0], wind=15.0 kt (mask=1.0), pres=1004.0 hPa, cat=-1
  Status:                     [PASS] Sequence successfully loaded and verified.

>>> LOADING VALIDATION SAMPLE...
Dataset split count: 252 sequences
  sample_id:                  NIO_2019_FANI_20190426_0600Z
  storm_id:                   NIO_2019_FANI
  split:                      VALIDATION
  t0_utc:                     2019-04-26T06:00:00Z
  GridSat shape:              (6, 72, 116)
  IMERG shape:                (6, 72, 116)
  GridSat stats (norm):       min=-4.174, max=1.923, mean=0.182, std=0.956
  IMERG stats (norm):         min=-0.167, max=25.033, mean=-0.037, std=0.789, NaNs=0
  Target t0:                  center=[3.0, 89.4], wind=25.0 kt (mask=1.0), pres=998.0 hPa, cat=0
  Status:                     [PASS] Sequence successfully loaded and verified.

>>> LOADING TEST SAMPLE...
Dataset split count: 371 sequences
  sample_id:                  NIO_2021_GULAB_20210924_1200Z
  storm_id:                   NIO_2021_GULAB
  split:                      TEST
  t0_utc:                     2021-09-24T12:00:00Z
  GridSat shape:              (6, 72, 116)
  IMERG shape:                (6, 72, 116)
  GridSat stats (norm):       min=-3.893, max=2.323, mean=0.020, std=1.060
  IMERG stats (norm):         min=-0.167, max=27.190, mean=0.022, std=0.863, NaNs=0
  Target t0:                  center=[18.3, 91.2], wind=25.0 kt (mask=1.0), pres=1000.0 hPa, cat=0
  Status:                     [PASS] Sequence successfully loaded and verified.
=================================================================
[LOAD TEST SUCCESS] All 3 partitions successfully loaded and validated.
=================================================================
```

---

## 7. Compliance with Research Constraints

1. **Zero Model Training:** No neural network weights trained or updated.
2. **Zero Fine-Tuning:** Existing checkpoints untouched.
3. **Zero Production Impact:** `apps/backend/`, `apps/frontend/`, and inference engines untouched.
4. **Data Leakage:** Strict storm-level partition independence maintained.
5. **No Synthetic Fallback:** All data is physical observational data from NASA GES DISC and NOAA GridSat-B1 archives.
6. **No Git Action:** Zero commits or pushes executed.