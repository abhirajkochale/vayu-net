# VAYU-NET — INSAT-3D Pretraining Data Source Verification Audit

**Module:** Multi-Source Satellite Data Governance & Sensor Attribution Audit  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — CRITICAL MISMATCH DETECTED  

---

## **PRETRAINING DATA SOURCE: INVALID / NOT INSAT-3D**

> [!CAUTION]
> **CRITICAL DATA SOURCE FINDING:**
> The 1,428 observations referenced in `data/manifests/insat_pretraining_manifest.csv` and packaged into `data/interim/ml/cache/insat_pretraining_cache.pt` **ARE NOT GENUINE ISRO INSAT-3D OBSERVATIONS**.
> They are **NOAA GridSat-B1 Climate Data Record (`irwin_cdr`)** observations cropped to the North Indian Ocean basin.
> 
> In accordance with strict scientific integrity rules:
> 1. The running pretraining task was **immediately aborted and killed**.
> 2. The checkpoint `data/interim/ml/checkpoints/insat_pretrained_encoder.pt` is **marked INVALID** and will **NOT** be used.
> 3. **Downstream transfer evaluation has been halted.**
> 4. Production systems and inference pipelines remain **100% untouched**.

---

## 1. Pretraining Manifest Inspection

**Target Manifest:** [`data/manifests/insat_pretraining_manifest.csv`](file:///c:/GitHub/vayu-net/data/manifests/insat_pretraining_manifest.csv)

### Manifest Structure:
- **Total Rows:** **1,428**
- **Columns:** `['frame_id', 'timestamp_utc', 'year', 'month', 'day', 'hour', 'split', 'source_file', 'file_size_bytes', 'spatial_height', 'spatial_width', 'channels']`
- **Claimed Channels:** `TIR1,TIR2,WV`

### First 10 Referenced File Paths:
```text
[1]  data/interim/gridsat/2014/gridsat_2014.06.09.21.npz
[2]  data/interim/gridsat/2014/gridsat_2014.06.10.00.npz
[3]  data/interim/gridsat/2014/gridsat_2014.06.10.03.npz
[4]  data/interim/gridsat/2014/gridsat_2014.06.10.06.npz
[5]  data/interim/gridsat/2014/gridsat_2014.06.10.09.npz
[6]  data/interim/gridsat/2014/gridsat_2014.06.10.12.npz
[7]  data/interim/gridsat/2014/gridsat_2014.06.10.15.npz
[8]  data/interim/gridsat/2014/gridsat_2014.06.10.18.npz
[9]  data/interim/gridsat/2014/gridsat_2014.06.10.21.npz
[10] data/interim/gridsat/2014/gridsat_2014.06.11.00.npz
```
*Observation:* Every single path in the manifest points directly to the `data/interim/gridsat/` directory.

### Year Distribution:
```text
2014:   60 observations
2015:  117 observations
2016:   81 observations
2017:   36 observations
2018:   85 observations
2019:  277 observations
2020:  101 observations
2021:  123 observations
2022:  144 observations
2023:  216 observations
2024:  188 observations
Total: 1,428 observations
```

---

## 2. Direct Binary File Inspection & Sensor Attribution

Representative files from the pretraining manifest were directly inspected at the binary level using `numpy.load`:

### Representative File: `data/interim/gridsat/2014/gridsat_2014.06.09.21.npz`
- **NPZ Internal Keys:** `['irwin_cdr', 'lat', 'lon', 'timestamp_utc']`
- **Primary Variable:** `irwin_cdr` (Shape: `[572, 929]`, dtype: `float32`)
- **Spatial Arrays:** `lat` (572,), `lon` (929,)
- **Variable Metadata:**
  - Standard Name: `toa_brightness_temperature`
  - Long Name: `NOAA FCDR of Brightness Temperature near 11 microns (Nadir-most observations) cropped to NIO`
  - Source Agency: **NOAA National Centers for Environmental Information (NCEI)**
  - Source Product: **GridSat-B1 Climate Data Record (CDR)**

### Channel Synthesis in Builder Script:
Inspection of [`ml/data/build_insat_pretraining_dataset.py`](file:///c:/GitHub/vayu-net/ml/data/build_insat_pretraining_dataset.py) and [`ml/data/build_multisource_cache.py`](file:///c:/GitHub/vayu-net/ml/data/build_multisource_cache.py) reveals how the channels were constructed:
```python
# 1. Base input read from GridSat NPZ
arr = npz["irwin_cdr"].astype(np.float32)

# 2. Synthetic channel derivations:
tir1 = arr_down.copy()
moisture = np.clip((arr_down - 220.0) / 75.0, 0.0, 1.0)
tir2 = np.clip(tir1 - 2.5 * moisture, 180.0, 330.0)
wv = np.clip(0.72 * arr_down + 62.0, 195.0, 275.0)
```
The pretraining script read NOAA GridSat-B1 $11\,\mu\text{m}$ images and applied synthetic transfer functions to simulate `TIR2` and `WV`.
**The underlying pixels are not genuine sensor measurements from the ISRO INSAT-3D IMAGER.**

---

## 3. Sensor Disambiguation Taxonomy

To establish absolute clarity across the codebase:

| Category | Sensor / Product | File Formats | Physical Origin | True Nature |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA GridSat-B1** | Geostationary Imager CDR | `.nc`, `.npz` (`irwin_cdr`) | NOAA NCEI S3 Archive | **Genuine NOAA Climate Data Record** (Baseline) |
| **INSAT-3D (Native)** | IMAGER Level-1C Asian Mercator | `.h5` (`3DIMG_L1C_ASIA_MER`) | ISRO / SAC (MOSDAC) | **Genuine ISRO Geostationary Satellite Imagery** |
| **INSAT-3DR (Native)**| IMAGER Level-1C Asian Mercator | `.h5` (`3RIMG_L1C_ASIA_MER`) | ISRO / SAC (MOSDAC) | **Genuine ISRO Geostationary Satellite Imagery** |
| **Pilot Paired NPZ** | Multi-source Pilot Package | `.npz` (`gridsat_irwin_11um`, `insat3d_tir1_10_8um`, etc.) | Generated in `scratch/build_pilot_artifacts.py` | **GridSat-derived paired pilot surrogate tensors** |
| **Pretraining Cache** | `insat_pretraining_manifest.csv` | `.npz` (`data/interim/gridsat/*`) | Generated in `ml/data/build_insat_pretraining_dataset.py` | **NOAA GridSat-B1 observations misattributed to INSAT** |

---

## 4. Local Archive Inventory: Genuine INSAT-3D Observations

A complete filesystem audit was performed across the entire repository for native HDF5 (`.h5`, `.hdf5`) files and genuine MOSDAC Level-1C products:

| Data Store | Number of Observations | Sensor Provenance | Notes |
| :--- | :---: | :--- | :--- |
| **Native MOSDAC HDF5 Files (`.h5`)** | **0** | ISRO INSAT-3D | None exist locally on disk. MOSDAC downloads require authenticated API bearer tokens (`MOSDAC_USERNAME`, `MOSDAC_PASSWORD`). |
| **Pilot Paired NPZ Files** | **3** | Derived from GridSat | Located in `data/interim/insat3d_pilot/` (FANI, AMPHAN, REMAL). Tensors are mathematically derived from GridSat. |
| **Pretraining Manifest Files** | **1,428** | NOAA GridSat-B1 | Located in `data/interim/gridsat/` (2014–2024). These are genuine GridSat files, not INSAT. |

**Final Local Availability Count:**
$$\mathbf{0} \text{ Genuine Native ISRO INSAT-3D Satellite Observations available locally.}$$

---

## 5. Audit Actions & Policy Enforcement

1. **Pretraining Aborted:**  
   Background task `f93d33fd-84a5-43f2-b3bd-f3a0c28c07a6/task-5552` running `train_insat_masked_pretraining.py` was **immediately killed and terminated**.
2. **Checkpoint Quarantined:**  
   `data/interim/ml/checkpoints/insat_pretrained_encoder.pt` is **flagged as INVALID**. It represents self-supervised autoencoding of GridSat-derived channels, not genuine INSAT-3D. It must **NOT** be used for downstream transfer evaluation.
3. **Downstream Test Blocked:**  
   Execution of `ml/train/train_insat_downstream_pretrained.py` has been **halted and cancelled**.
4. **Research Baseline Preserved:**  
   Previous valid transfer-learning checkpoints (`multisource_decoupled_full.pt`, `multisource_decoupled_baseline.pt`, `multisource_transfer_exp1_expanded.pt`) remain intact and are properly documented as using the calibrated GridSat-to-INSAT transfer benchmark.
5. **Production Isolation Maintained:**  
   Zero modifications to `apps/backend/`, `apps/frontend/`, production checkpoints, or database infrastructure. Zero Git commits or pushes.
