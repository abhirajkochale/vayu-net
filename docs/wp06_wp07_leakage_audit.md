# VAYU-NET WP-06 & WP-07: Anti-Leakage & Governance Audit

**Scope:** Work Package 06 (Uncertainty & Verification) & Work Package 07 (Analog-Storm Retrieval)  
**Date:** 2026-09-25  
**Audit Standard:** SIH 2026 Problem Statement 26070 Research Discipline  
**Audit Outcome:** **FULLY CERTIFIED — ZERO LEAKAGE**  

---

## 1. Audit Scope & Mandate
This document certifies that WP-06 (Empirical Uncertainty & Forecast Verification) and WP-07 (Historical Analog-Storm Retrieval) strictly adhere to all causal boundaries, dataset split isolations, and anti-leakage governance rules established in the VAYU-NET project.

---

## 2. Invariant Checklist: WP-06 (Uncertainty & Verification)

### 2.1 Derivation Split Isolation
- **Rule:** Uncertainty parameters must NOT be fitted from TEST data in a way that would influence model selection or create optimistic uncertainty bands.
- **Verification:**
  - Empirical error percentiles ($\text{P50}$, $\text{P80}$, $\text{P90}$, $\text{P95}$, Mean) were derived strictly from the **`VALIDATION` partition** ($N = 252$ samples across 14 storms).
  - Checkpoint: Phase 5B Variant A (Observed-Center Hybrid Residual Model, `best_phase5b_variant_a.pt`).
  - The held-out `TEST` partition ($N = 371$ samples) was never accessed during uncertainty parameter derivation.
  - Test data was inspected only to verify empirical coverage ($80.9\%\text{–}81.4\%$ coverage for P80, $90.3\%\text{–}92.2\%$ coverage for P90), confirming that the parameters generalized without post-hoc tuning.
- **Audit Status:** **PASSED**

### 2.2 Empirical Provenance Labeling
- **Rule:** Do NOT describe empirical radii as probabilistic Gaussian confidence intervals.
- **Verification:**
  - All outputs carry `"label": "empirical"`.
  - All outputs explicitly state `"is_probabilistic_confidence_interval": false`.
  - Stored in machine-readable parameter file: `data/interim/ml/uncertainty_parameters.json`.
- **Audit Status:** **PASSED**

### 2.3 Verification Delineation
- **Rule:** Verification output must clearly distinguish FORECAST, ACTUAL, and ERROR.
- **Verification:**
  - `ml/forecast/verification.py` creates separate dictionaries for `forecast`, `actual`, and `error`.
  - Forecast timestamps and actual target timestamps match verified IMD ground truth.
- **Audit Status:** **PASSED**

---

## 3. Invariant Checklist: WP-07 (Analog-Storm Retrieval)

### 3.1 Feature Vector Causal Boundary ($t \le t_0$)
- **Rule:** The 7-dimensional feature vector must use only information available at or before $t_0$.
- **Verification:**
  - Features: $\mathbf{x} = [\text{lat}(t_0), \text{lon}(t_0), \text{wind}(t_0), \text{pressure}(t_0), \Delta x_{12\text{h}}, \Delta y_{12\text{h}}, \Delta\text{wind}_{12\text{h}}]$.
  - $\Delta x_{12\text{h}}$ is the past 12-hour longitudinal displacement: $\text{lon}(t_0) - \text{lon}(t_0 - 12\text{h})$.
  - $\Delta y_{12\text{h}}$ is the past 12-hour latitudinal displacement: $\text{lat}(t_0) - \text{lat}(t_0 - 12\text{h})$.
  - $\Delta\text{wind}_{12\text{h}}$ is the past 12-hour intensification rate: $\text{wind}(t_0) - \text{wind}(t_0 - 12\text{h})$.
  - Zero future track positions ($+12\text{h}, +24\text{h}, +48\text{h}$) enter the feature extraction.
- **Audit Status:** **PASSED**

### 3.2 Standardization Statistics Provenance
- **Rule:** Standardization parameters (mean, standard deviation) must be derived strictly from the training archive. Never standardize using TEST-only or full-dataset statistics.
- **Verification:**
  - Means and standard deviations were derived exclusively from the 696 samples of the `TRAIN` partition:
    - `lat`: $\mu = 14.49^\circ$, $\sigma = 4.70^\circ$
    - `lon`: $\mu = 80.74^\circ$, $\sigma = 11.41^\circ$
    - `wind_kt`: $\mu = 36.88\text{ kt}$, $\sigma = 17.97\text{ kt}$
    - `pressure_hpa`: $\mu = 993.73\text{ hPa}$, $\sigma = 10.98\text{ hPa}$
    - `dx_12h`: $\mu = -0.73^\circ$, $\sigma = 0.93^\circ$
    - `dy_12h`: $\mu = 0.52^\circ$, $\sigma = 0.72^\circ$
    - `dwind_12h`: $\mu = 4.76\text{ kt}$, $\sigma = 8.37\text{ kt}$
  - Stored in `data/interim/ml/analog_retrieval_cache.json`.
- **Audit Status:** **PASSED**

### 3.3 Self-Match & Same-Storm Exclusion Logic
- **Rule:** A query storm must never match against itself or return another observation from the same cyclone event.
- **Verification:**
  - `ml/forecast/analog_retrieval.py` enforces:
    1. `candidate.sample_id != query.sample_id`
    2. `candidate.storm_id != query.storm_id`
    3. `candidate_1.storm_id != candidate_2.storm_id`
  - Unit tests in `scripts/audit/test_analog_retrieval.py` confirm 100% compliance across all tested queries.
- **Audit Status:** **PASSED**

### 3.4 Non-Forecast Labeling
- **Rule:** Analogs must be identified as historical context, not as a numerical forecast.
- **Verification:**
  - Every analog entry contains `"label": "HISTORICAL ANALOG"`.
  - Response payload carries system warning:
    `"Historical analogs provide situational context only and are not a direct forecast source."`
- **Audit Status:** **PASSED**

---

## 4. Locked Datasets & Historical Artifacts Immutability Audit

Verification that no locked files were modified during WP-06 and WP-07:

| Locked Path / Artifact | Pre-WP-06 State | Current State | Immutability Status |
| :--- | :---: | :---: | :---: |
| `data/raw/imd/` | 29 PDFs | 29 PDFs | **LOCKED & VERIFIED** |
| `data/processed/imd_best_track_v2.csv` | 3,960 rows | 3,960 rows | **LOCKED & VERIFIED** |
| `data/interim/gridsat/` | 4,706 files | 4,706 files | **LOCKED & VERIFIED** |
| `data/manifests/vayu_net_sample_index.csv` | 1,319 samples | 1,319 samples | **LOCKED & VERIFIED** |
| `data/manifests/storm_event_manifest_v2.csv` | 216 storms | 216 storms | **LOCKED & VERIFIED** |
| `best_phase5b_variant_a.pt` | Intact | Intact | **LOCKED & VERIFIED** |
| `phase5b_hybrid_results.json` | Intact | Intact | **LOCKED & VERIFIED** |
| `best_phase6_intensity_wind.pt` | Intact | Intact | **LOCKED & VERIFIED** |
| `phase6_intensity_wind_results.json` | Intact | Intact | **LOCKED & VERIFIED** |

---

## 5. Audit Conclusion
Work Packages 06 and 07 have been implemented with complete scientific discipline, zero data leakage, and full preservation of locked system interfaces. Both modules are certified ready for demonstration and API integration.
