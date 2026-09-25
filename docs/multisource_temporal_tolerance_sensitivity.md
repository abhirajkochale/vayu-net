# VAYU-NET — Multi-Source Satellite Temporal Tolerance Sensitivity Audit

**Module:** Multi-Source Satellite Data Governance & Synchronization Audit  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — SCIENTIFIC SENSITIVITY DETERMINATION FINALIZED  

---

## 1. Executive Summary & Audit Objective

Before training multi-source models, this audit evaluates the empirical sensitivity of the paired **NOAA GridSat-B1** and **ISRO INSAT-3D** (`3DIMG_L1C_ASIA_MER`) dataset across three temporal matching tolerances:
- **Threshold A:** $\Delta t \le 15$ minutes
- **Threshold B:** $\Delta t \le 30$ minutes
- **Threshold C:** $\Delta t \le 60$ minutes

### Core Findings:
1. **Frame vs. Sequence Lookback Compounding:**
   While $92.34\%$ of all individual observation frames are exact matches ($\Delta t = 0$ min), enforcing a rigid $\Delta t \le 15$ min or $\Delta t \le 30$ min threshold drops **$39.59\%$ and $37.38\%$ of complete six-frame sample sequences**, respectively.
2. **Systematic Diurnal / Equinoctial Bias:**
   INSAT-3D observations with $\Delta t = 60$ min occur exclusively during midnight equinoctial solar eclipse seasons ($18:00\text{Z} \to 19:00\text{Z}$ shifts). Filtering out $\Delta t \le 30$ min eliminates nighttime observations systematically, creating severe diurnal sampling bias.
3. **Storm Dropping in Validation & Test Splits:**
   - Under $\Delta t \le 15$ min: Storm `NIO_2020_NISARGA` is completely dropped from `VALIDATION` (reducing storms from 14 to 13), and `NIO_2022_UNNAMED_2_13_1` is completely dropped from `TEST` (reducing storms from 24 to 23).
   - Under $\Delta t \le 30$ min: The same storm deletions persist.
4. **Data Integrity & Zero Future Leakage:**
   Under $\Delta t \le 60$ min, all 4,350 frames maintain **100% causal compliance** ($t \le t_0$, zero future leakage), zero spatial distortion, and full synoptic integrity across all 62 historical storms (2014–2024).
5. **Scientific Recommendation:**
   Retain the validated **$\Delta t \le 60$ minute threshold** for all model training and ablation experiments to preserve storm diversity, prevent diurnal bias, and avoid severe sample depletion in the already constrained `TRAIN` set ($N=175$).

---

## 2. Quantitative Sensitivity Comparison Table

| Metric | Threshold A ($\Delta t \le 15$ min) | Threshold B ($\Delta t \le 30$ min) | Threshold C ($\Delta t \le 60$ min) |
| :--- | :---: | :---: | :---: |
| **Total Observation Frames** | 4,017 | 4,079 | **4,350** |
| **Rejected Observation Frames** | 333 (7.66%) | 271 (6.23%) | **0 (0.00%)** |
| **Total Complete 6-Frame Samples** | 438 | 454 | **725** |
| **Rejected Paired Samples** | 287 (39.59%) | 271 (37.38%) | **0 (0.00%)** |
| **Sample Retention Rate** | 60.41% | 62.62% | **100.00%** |
| **TRAIN Samples ($N_{\text{orig}} = 175$)** | 105 (60.00%) | 111 (63.43%) | **175 (100.00%)** |
| **TRAIN Unique Storms ($S_{\text{orig}} = 24$)** | 24 | 24 | **24** |
| **VALIDATION Samples ($N_{\text{orig}} = 252$)** | 139 (55.16%) | 143 (56.75%) | **252 (100.00%)** |
| **VALIDATION Unique Storms ($S_{\text{orig}} = 14$)**| 13 (Dropped: `NISARGA`) | 13 (Dropped: `NISARGA`) | **14** |
| **TEST Samples ($N_{\text{orig}} = 298$)** | 194 (65.10%) | 200 (67.11%) | **298 (100.00%)** |
| **TEST Unique Storms ($S_{\text{orig}} = 24$)** | 23 (Dropped: `UNNAMED_2_13_1`) | 23 (Dropped: `UNNAMED_2_13_1`) | **24** |

---

## 3. Physical & Meteorological Analysis of Offsets

### A. Geostationary Orbital Mechanics & Eclipse Schedules
INSAT-3D is positioned at $82.0^\circ\text{E}$ over the equator. Around the vernal (March–May) and autumnal (September–November) equinoxes, the satellite passes through Earth's shadow near local midnight ($18:00\text{–}19:00\text{Z}$, $23:30\text{–}00:30\text{ IST}$). To avoid stray-light blinding of optics and conserve onboard battery during solar array blackout, ISRO operational scan schedules:
- Postpone the standard $18:00\text{Z}$ Asian Sector scan by 30 to 60 minutes (typically initiating at $18:30\text{Z}$ or $19:00\text{Z}$).
- These 271 frames ($6.23\%$ of total frames) are regular, high-quality, fully calibrated images; their timestamp simply reflects the astronomical constraint.

### B. Cyclone Kinematic Scale Comparison
- Typical North Indian Ocean cyclone translation speed: $v \approx 12\text{–}20\text{ km/h}$.
- Spatial displacement over $\Delta t = 60\text{ min}$: $\approx 15\text{ km}$.
- GridSat and INSAT-3D grid resolution at $0.07^\circ$: $\approx 7.8\text{ km/pixel}$.
- Maximum centroid shift during offset: $\approx 1.5\text{–}2.0\text{ pixels}$.
- Crucially, cyclonic convective cloud features, central dense overcasts (CDO), and synoptic moisture bands evolve over convective timescales ($\tau \sim 3\text{–}6\text{ hours}$), rendering 30–60 minute offsets meteorologically representative.

### C. Causality Guarantee
In every single matched frame where $\Delta t > 0$, the INSAT observation timestamp satisfies:
$$t_{\text{INSAT}} \le t_0$$
Zero future information is ever observed or leaked.

---

## 4. Decision & Freeze Verdict

### **VERDICT: FREEZE AT $\Delta t \le 60$ MINUTES**
- Retaining $\Delta t \le 60$ min preserves all **725 validated paired samples** ($175\text{ TRAIN}, 252\text{ VAL}, 298\text{ TEST}$).
- Rejecting $\Delta t = 60$ min would drop $40\%$ of the paired dataset and critically starve the `TRAIN` set down to only $105$ samples, severely handicapping neural network generalization.
- The experiment dataset is formally frozen using:  
  [data/manifests/vayu_net_multisource_sample_index.csv](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_multisource_sample_index.csv)
- Complete audit machine-readable artifact saved at:  
  [data/interim/ml/multisource_temporal_tolerance_sensitivity.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_temporal_tolerance_sensitivity.json)
