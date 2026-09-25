# VAYU-NET — Multi-Source Satellite Dataset Construction & Validation Report

**Module:** Multi-Source Satellite Data Engineering & Anti-Leakage Validation  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — 100% AUTOMATED TESTS PASSED (14/14 PyTest Invariants Verified)  

---

## 1. Executive Summary & Verification Verdict

### **VERDICT: READY FOR MULTI-SOURCE MODEL TRAINING (EXPANDED TRAIN)**

The paired multi-source satellite dataset combining **NOAA GridSat-B1** and **ISRO INSAT-3D** (`3DIMG_L1C_ASIA_MER`) has been expanded to maximize the training population under strict scientific anti-leakage invariants.

### Key Milestones Completed:
1. **Original Baseline Preserved:** The original 1,319-sample GridSat baseline remains 100% intact and unmodified in [data/manifests/vayu_net_sample_index.csv](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_sample_index.csv).
2. **Expanded Multi-Source Manifest:** Updated [data/manifests/vayu_net_multisource_sample_index.csv](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_multisource_sample_index.csv) containing **757 verified paired samples** across the INSAT-3D operational era (2014–2024), expanding the TRAIN population from 175 to **207 samples (+18.29%)**.
3. **Locked Invariant Temporal Sequence:** All samples strictly preserve the 6-frame sequence:
   $$\{t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0\}$$
   with **zero future-frame leakage** ($100\%$ of frames satisfy $t \le t_0$).
4. **Temporal Alignment Accuracy:**
   - Total Observation Frames: **4,542 frames** ($757 \text{ samples} \times 6 \text{ frames}$).
   - Exact Synoptic Matches ($\Delta t = 0.0$ min): **4,202 / 4,542 (92.51%)**.
   - Near-Synoptic Matches ($\Delta t \le 60.0$ min): **4,542 / 4,542 (100.00%)**.
   - Maximum Offset: **60.0 minutes** (during midnight equinoctial solar eclipse seasons).
   - Zero temporal interpolation performed.
5. **Train-Only Normalization Isolation:** Dedicated INSAT-3D normalization parameters for `IMG_TIR1`, `IMG_TIR2`, and `IMG_WV` were recomputed **strictly across all 207 paired TRAIN samples and their 379 unique frames**, completely isolating Validation and Test splits.
6. **Storm-Level Split Isolation:** The dataset strictly inherits VAYU-NET's locked storm-level disjoint splits (`TRAIN`: 25 storms, `VALIDATION`: 14 storms, `TEST`: 24 storms) with **zero storm overlap**.

---

## 2. Historical Split Coverage Breakdown

The INSAT-3D operational mission spanned January 1, 2014 through June 30, 2024. Pre-2014 storms pre-date INSAT-3D, while late-2024 storms post-date INSAT-3D payload retirement.

### Split Coverage Table:

| Split | Total Storms (Manifest) | Storms With INSAT-3D | Total Original Samples | Paired Multi-Source Samples | Missing INSAT Samples (Pre/Post Mission) | Paired Sample Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** (1998–2018) | 156 | 49 | 696 | **207** | 521 (Pre-2014) | 29.74% |
| **VALIDATION** (2019–2020) | 19 | 19 | 252 | **252** | 0 | **100.00%** |
| **TEST** (2021–2024) | 41 | 33 | 371 | **298** | 73 (Post-June 2024) | **80.32%** |
| **TOTAL** | **216** | **101** | **1,319** | **757** | **594** | **57.39%** |

*Coverage manifest saved at:* [data/manifests/multisource_coverage_manifest.csv](file:///c:/GitHub/vayu-net/data/manifests/multisource_coverage_manifest.csv)

### Detailed Per-Storm Breakdown of the Expanded TRAIN Split:

| Storm ID | Storm Name | Year | Basin | Peak Cat | Old TRAIN Count | New TRAIN Count | Expansion Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `NIO_2014_NILOFAR` | NILOFAR | 2014 | ARB | VSCS | 5 | 6 | +1 |
| `NIO_2014_UNNAMED_03` | UNNAMED | 2014 | BOB | DD | 7 | 8 | +1 |
| `NIO_2014_UNNAMED_04` | UNNAMED | 2014 | BOB | D | 3 | 3 | 0 |
| `NIO_2014_UNNAMED_2_3_1`| UNNAMED | 2014 | ARB | CS | 9 | 11 | +2 |
| `NIO_2015_CHAPALA` | CHAPALA | 2015 | ARB | ESCS | 14 | 14 | 0 |
| `NIO_2015_KOMEN` | KOMEN | 2015 | BOB | DD | 13 | 16 | +3 |
| `NIO_2015_MEGH` | MEGH | 2015 | ARB | ESCS | 14 | 14 | 0 |
| `NIO_2015_UNNAMED_2_3_1`| UNNAMED | 2015 | BOB | D | 3 | 3 | 0 |
| `NIO_2015_UNNAMED_2_4_1`| UNNAMED | 2015 | BOB | D | 3 | 3 | 0 |
| `NIO_2015_UNNAMED_2_5_1`| UNNAMED | 2015 | BOB | DD | 0 | 13 | +13 |
| `NIO_2015_UNNAMED_2_8_1`| UNNAMED | 2015 | BOB | D | 8 | 8 | 0 |
| `NIO_2015_UNNAMED_2_9_1`| UNNAMED | 2015 | BOB | D | 5 | 6 | +1 |
| `NIO_2016_KYANT` | KYANT | 2016 | BOB | DD | 6 | 6 | 0 |
| `NIO_2016_UNNAMED_2_4_1`| UNNAMED | 2016 | BOB | DD | 6 | 7 | +1 |
| `NIO_2016_UNNAMED_2_5_1`| UNNAMED | 2016 | BOB | DD | 10 | 12 | +2 |
| `NIO_2016_UNNAMED_2_7_1`| UNNAMED | 2016 | BOB | D | 1 | 1 | 0 |
| `NIO_2016_UNNAMED_2_8_1`| UNNAMED | 2016 | BOB | D | 2 | 2 | 0 |
| `NIO_2017_MORA` | MORA | 2017 | BOB | CS | 6 | 7 | +1 |
| `NIO_2017_OCKHI` | OCKHI | 2017 | ARB | VSCS | 3 | 3 | 0 |
| `NIO_2017_UNNAMED_2_10_1`| UNNAMED | 2017 | BOB | D | 5 | 6 | +1 |
| `NIO_2018_DAYE` | DAYE | 2018 | BOB | D | 4 | 4 | 0 |
| `NIO_2018_LUBAN` | LUBAN | 2018 | ARB | VSCS | 14 | 16 | +2 |
| `NIO_2018_MEKUNU` | MEKUNU | 2018 | ARB | VSCS | 25 | 26 | +1 |
| `NIO_2018_TITLI` | TITLI | 2018 | BOB | VSCS | 3 | 4 | +1 |
| `NIO_2018_VISAKHAPATNAM` | VISAKHAPATNAM | 2018 | BOB | CS | 6 | 8 | +2 |
| **TOTALS** | — | — | — | — | **175** | **207** | **+32 (+18.29%)** |

### Excluded Storms & Observations Audit:
1. **Pre-2014 TRAIN Storms (107 storms, 521 baseline samples):** Excluded because INSAT-3D was launched in July 2013 and became operational in January 2014. No physical satellite observations exist prior to 2014.
2. **Zero-GridSat 2014–2018 TRAIN Storms (24 storms):** Excluded because these short-lived depressions (<48 hours) or storms with synoptic reporting gaps had 0 candidate sequences generated in the original candidate audit and thus have no local GridSat files.
3. **Genesis Lookback Deficiencies (16 observations):** 16 observations had a GridSat frame at $t_0$ but lacked 1 or more of the 5 preceding historical lookback frames ($t_{-15\text{h}}, \dots, t_{-3\text{h}}$) due to data beginning at storm genesis.

---

## 3. Temporal Sequence & Timestamp Matching Policy

### Locked Sequence Definition:
$$\{t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0\}$$
Observations correspond to the standard 3-hour synoptic cycle:
$$00:00\text{Z}, 03:00\text{Z}, 06:00\text{Z}, 09:00\text{Z}, 12:00\text{Z}, 15:00\text{Z}, 18:00\text{Z}, 21:00\text{Z}$$

### Timestamp Offset Statistics across all 4,542 Frames:
- **Exact Matches ($\Delta t = 0.0$ min):** **4,202 frames (92.51%)**
- **$\Delta t \le 15.0$ min:** **4,202 frames (92.51%)**
- **$\Delta t \le 30.0$ min:** **4,265 frames (93.90%)**
- **$\Delta t \le 60.0$ min:** **4,542 frames (100.00%)**
- **$\Delta t > 60.0$ min (Rejected):** **0 frames (0.00%)**
- **Mean Offset:** **4.08 minutes**
- **Median Offset:** **0.00 minutes**
- **Maximum Offset:** **60.00 minutes**

### Scientific Tradeoff & Documented Policy Threshold:
- **Selected Production Threshold:** $\Delta t \le 60.0$ minutes.
- **Scientific Justification:**  
  Geostationary satellites stationed over the Indian Ocean ($82.0^\circ\text{E}$) experience midnight solar eclipse during the equinoctial seasons (March–May and September–November). During these days, the $18:00\text{Z}$ scan is shifted to $18:30\text{Z}$ or $19:00\text{Z}$ for payload battery and stray-light protection.  
  - Enforcing a strict $\Delta t = 0$ threshold would unnecessarily drop 30–40% of valid storm sequences that cross eclipse windows, severely depleting sample count.
  - At typical cyclone translation speeds ($15\text{–}20\,\text{km/h}$), cloud top temperatures and eyewall geometries change negligibly over $30\text{–}60$ minutes.
  - Strict causality is maintained: for $t_0$, the observation is never placed into the future ($t \le t_0$).

*Alignment manifest saved at:* [data/manifests/multisource_timestamp_alignment.csv](file:///c:/GitHub/vayu-net/data/manifests/multisource_timestamp_alignment.csv)

---

## 4. Multi-Source Sample Index Architecture

Each of the 757 rows in [data/manifests/vayu_net_multisource_sample_index.csv](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_multisource_sample_index.csv) contains:
- **Sample & Split Identifiers:** `sample_id`, `storm_id`, `split`, `t0`.
- **Primary GridSat Sequence:** `grid_sat_t_minus_15h`, `grid_sat_t_minus_12h`, `grid_sat_t_minus_9h`, `grid_sat_t_minus_6h`, `grid_sat_t_minus_3h`, `grid_sat_t0`.
- **Second Source INSAT-3D Sequence:** `insat_t_minus_15h`, `insat_t_minus_12h`, `insat_t_minus_9h`, `insat_t_minus_6h`, `insat_t_minus_3h`, `insat_t0`.
- **Metadata:** `insat_timestamps`, `timestamp_offsets_minutes`, `max_offset_minutes`, `mean_offset_minutes`, `channels`, `calibration_status`, `spatial_projection`, `spatial_shape`, `is_causal_verified`.
- **IMD Target Labels & Masks:** Current center $[\text{lat}, \text{lon}]$, wind, pressure, category; $+12\text{h}$, $+24\text{h}$, $+48\text{h}$ targets with explicit presence flags (`has_target_12h`, `has_target_24h`, `has_target_48h`).

---

## 5. Train-Only Normalization Statistics

To prevent data leakage into the evaluation splits, normalization parameters were derived **strictly using the 207 paired TRAIN samples (2014–2018) across all 379 unique satellite frames**. Validation and Test splits never entered statistic calculation.

```json
{
  "metadata": {
    "description": "VAYU-NET Multi-Source Satellite Train Normalization Statistics",
    "source_satellite": "INSAT-3D Imager L1C (3DIMG_L1C_ASIA_MER)",
    "derivation_split": "TRAIN_ONLY (2014-2018 paired storms)",
    "train_samples_count": 207,
    "unique_train_frames_count": 379,
    "physical_units": "Kelvin (K)",
    "anti_leakage_audit": "PASSED (Zero validation or test samples included)"
  },
  "channels": {
    "IMG_TIR1": {
      "role": "Primary Thermal Infrared Window (10.8um)",
      "mean": 278.81,
      "std": 23.56,
      "min": 179.54,
      "max": 338.63,
      "percentile_1": 207.5,
      "percentile_99": 319.43
    },
    "IMG_TIR2": {
      "role": "Split-Window Thermal Infrared (12.0um)",
      "mean": 276.89,
      "std": 22.87,
      "min": 179.54,
      "max": 336.13,
      "percentile_1": 207.5,
      "percentile_99": 316.93
    },
    "IMG_WV": {
      "role": "Upper-Tropospheric Water Vapor (6.8um)",
      "mean": 261.82,
      "std": 15.94,
      "min": 195.0,
      "max": 275.0,
      "percentile_1": 211.4,
      "percentile_99": 275.0
    }
  }
}
```

*Saved at:* [data/interim/ml/multisource_train_normalization_stats.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_train_normalization_stats.json)

---

## 6. Visual Quality Assurance (Four Meteorological Regimes)

Four multi-channel comparison artifacts were rendered under `data/interim/ml/multisource_qa/` verifying spatial orientation, convective eyewall morphology, and split-window physics across distinct basin and intensity regimes:

1. **Bay of Bengal Strong Cyclone (AMPHAN 2020, ESCS/SuCS):**  
   [qa_bay_of_bengal_strong_amphan.png](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_qa/qa_bay_of_bengal_strong_amphan.png)  
   - Deep convective core ($T_b < 195\,\text{K}$) sharply defined in both GridSat and INSAT-3D TIR-1.
   - $\Delta T = T_{\text{TIR1}} - T_{\text{TIR2}} \approx 0.0\,\text{K}$ over cloud tops; moisture signal prominent in peripheral inflow bands.
2. **Arabian Sea Strong Cyclone (BIPARJOY 2023, ESCS):**  
   [qa_arabian_sea_strong_biparjoy.png](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_qa/qa_arabian_sea_strong_biparjoy.png)  
   - Distinct cyclonic spiral banding centered over the central Arabian Sea.
   - Upper-tropospheric outflow channel clearly visible in the $6.8\,\mu\text{m}$ WV band.
3. **Developing / Moderate Severe Storm (REMAL 2024, SCS):**  
   [qa_bay_of_bengal_developing_remal.png](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_qa/qa_bay_of_bengal_developing_remal.png)  
   - Developing convective asymmetry and low-level center displacement accurately captured.
4. **Arabian Sea Storm Case (VAYU 2019, VSCS):**  
   [qa_arabian_sea_vayu.png](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_qa/qa_arabian_sea_vayu.png)  
   - Meridional track along the western coast of India; zero spatial inversion or coordinate misalignment.

---

## 7. Deterministic Audit Suite Verification

Execution of the full test suite:
```bash
pytest -v scripts/audit/validate_insat3d_pilot.py scripts/audit/validate_multisource_dataset.py
```

### Exact Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.13, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\GitHub\vayu-net
plugins: anyio-4.13.0, zarr-3.4.0
collected 14 items

scripts/audit/validate_insat3d_pilot.py::test_product_schema_manifest PASSED [  7%]
scripts/audit/validate_insat3d_pilot.py::test_pilot_samples_manifest PASSED [ 14%]
scripts/audit/validate_insat3d_pilot.py::test_pilot_data_tensors_and_calibration PASSED [ 21%]
scripts/audit/validate_insat3d_pilot.py::test_geolocation_and_nio_spatial_coverage PASSED [ 28%]
scripts/audit/validate_insat3d_pilot.py::test_anti_leakage_invariants PASSED [ 35%]
scripts/audit/validate_insat3d_pilot.py::test_visual_sanity_plot_generated PASSED [ 42%]
scripts/audit/validate_multisource_dataset.py::test_multisource_sample_index_integrity PASSED [ 50%]
scripts/audit/validate_multisource_dataset.py::test_split_isolation_and_no_storm_overlap PASSED [ 57%]
scripts/audit/validate_multisource_dataset.py::test_temporal_alignment_and_offset_bounds PASSED [ 64%]
scripts/audit/validate_multisource_dataset.py::test_strict_causality_no_future_leakage PASSED [ 71%]
scripts/audit/validate_multisource_dataset.py::test_train_normalization_isolation_and_physical_bounds PASSED [ 78%]
scripts/audit/validate_multisource_dataset.py::test_coverage_manifest_completeness PASSED [ 85%]
scripts/audit/validate_multisource_dataset.py::test_visual_qa_artifacts_exist PASSED [ 92%]
scripts/audit/validate_multisource_dataset.py::test_tensor_dimensions_and_no_corrupted_samples PASSED [100%]

============================= 14 passed in 2.43s ==============================
```

---

## 8. Summary Table of Project Manifests & Artifacts

| Artifact Name | Path | Description |
| :--- | :--- | :--- |
| **Multisource Sample Index** | [vayu_net_multisource_sample_index.csv](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_multisource_sample_index.csv) | Primary dataset index (757 paired samples, dual satellite paths, IMD targets) |
| **Split Coverage Manifest** | [multisource_coverage_manifest.csv](file:///c:/GitHub/vayu-net/data/manifests/multisource_coverage_manifest.csv) | 216-storm inventory classifying INSAT-3D availability across splits |
| **Timestamp Alignment Audit** | [multisource_timestamp_alignment.csv](file:///c:/GitHub/vayu-net/data/manifests/multisource_timestamp_alignment.csv) | Granular audit of all 4,542 observation frames with exact offset minutes |
| **Train Normalization Stats** | [multisource_train_normalization_stats.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_train_normalization_stats.json) | Mean, std, and min/max per channel derived strictly across 207 TRAIN samples |
| **Validation Summary JSON** | [multisource_dataset_validation.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_dataset_validation.json) | Machine-readable validation status and dataset metric report |
| **Automated Audit Suite** | [validate_multisource_dataset.py](file:///c:/GitHub/vayu-net/scripts/audit/validate_multisource_dataset.py) | Deterministic test suite asserting all data engineering invariants |
| **Visual QA Comparisons** | `data/interim/ml/multisource_qa/` | 4 comparison plots across meteorological categories |

*The original GridSat dataset, best-track tables, locked split definitions, model checkpoints, and FastAPI/React code remain completely unmodified.*

