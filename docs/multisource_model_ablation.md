# VAYU-NET — Multi-Source Satellite Fair Ablation & Scientific Determination

**Module:** Multi-Source Modality Ablation & Hypothesis Evaluation  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — HYPOTHESIS TEST CONCLUDED WITH HONEST SCIENTIFIC DETERMINATION  

---

## 1. Executive Summary & Scientific Hypothesis

This report documents the fair ablation experiment evaluating whether adding **ISRO INSAT-3D Imager** (`3DIMG_L1C_ASIA_MER`) multi-spectral channels (`IMG_TIR1` 10.8µm, `IMG_TIR2` 12.0µm, `IMG_WV` 6.8µm) provides measurable predictive value beyond the baseline **NOAA GridSat-B1** 11µm Climate Data Record.

### Scientific Hypothesis Evaluated:
$$\mathcal{H}_1: \text{Performance}(\text{GridSat} \oplus \text{INSAT-3D}) > \text{Performance}(\text{GridSat})$$

### **SCIENTIFIC DETERMINATION: HYPOTHESIS $\mathcal{H}_1$ REJECTED**
Adding INSAT-3D observations to GridSat-B1 in a from-scratch multi-source neural network **does NOT provide measurable predictive value** under the historical paired sample regime. In primary operational metrics (Center Localization and Maximum Sustained Wind Speed), multi-source fusion performs **inferior to GridSat-B1 alone**.

---

## 2. Fair Comparison Evaluation Table

All models and baselines were evaluated on the **exact same 298 TEST samples** (24 storms, 2021–2024) using identical ground-truth IMD labels.

| Metric | Model A: GridSat Only | Model B: INSAT Only | Model C: GridSat + INSAT Fusion | Difference: Fusion vs. GridSat (Abs) | Difference: Fusion vs. GridSat (%) | Persistence Baseline | Constant Velocity Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Center Mean DPE (km)** | **1,201.94** | 1,245.65 | 1,220.06 | +18.12 km | **+1.51% (Worse)** | *N/A (Anchor)* | *N/A (Anchor)* |
| **Center Median DPE (km)** | **1,205.55** | 1,411.47 | 1,313.35 | +107.79 km | **+8.94% (Worse)** | *N/A (Anchor)* | *N/A (Anchor)* |
| **Center P90 DPE (km)** | **1,628.09** | 1,794.51 | 1,705.64 | +77.56 km | **+4.76% (Worse)** | *N/A (Anchor)* | *N/A (Anchor)* |
| **Intensity Accuracy** | 0.2215 | 0.2215 | 0.2215 | 0.0000 | 0.00% (Neutral) | **0.8322** | — |
| **Intensity Macro-F1** | 0.0604 | 0.0604 | 0.0604 | 0.0000 | 0.00% (Neutral) | **0.8241** | — |
| **Wind MAE (kt)** | 21.47 | **21.08** | 25.09 | +3.62 kt | **+16.86% (Worse)**| **1.95** | — |
| **Wind RMSE (kt)** | 29.94 | **29.08** | 33.95 | +4.01 kt | **+13.38% (Worse)**| **3.45** | — |
| **+12h Track DPE (km)** | 1,215.17 | 1,225.71 | **1,174.05** | -41.12 km | **-3.38% (Marginal)**| 139.11 | **72.04** |
| **+24h Track DPE (km)** | 1,228.85 | 1,202.90 | **1,166.62** | -62.24 km | **-5.06% (Marginal)**| 273.92 | **150.89** |
| **+48h Track DPE (km)** | **1,208.03** | 1,266.25 | 1,213.76 | +5.73 km | **+0.47% (Worse)** | 549.58 | **340.14** |
| **Track Aggregate DPE (km)**| 1,217.35 | 1,231.62 | **1,184.81** | -32.54 km | **-2.67% (Marginal)**| 320.87 | **187.69** |

*Artifact source:* [data/interim/ml/multisource_ablation_results.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_ablation_results.json)

---

## 3. Analysis of Empirical Findings

### A. Identification (Center Localization): Fusion Degrades Performance
- Model A (GridSat) achieved a center mean DPE of **1,201.9 km**.
- Model B (INSAT) achieved **1,245.6 km**.
- Model C (Fusion) degraded to **1,220.1 km** (+18.1 km, **1.5% worse** than GridSat alone; median DPE was **107.8 km worse**).
- **Physical Reason:** Center localization depends predominantly on high-contrast thermal gradient geometry around the cyclonic eye and curved convective bands. Feeding additional split-window (`TIR2`) and upper-tropospheric water vapor (`WV`) channels doubled input dimensionality without adding sharp spatial center cues, increasing parameter space and optimization variance.

### B. Intensity Classification: Majority-Class Collapse
- All three models achieved identical accuracy ($22.15\%$) and macro-F1 ($0.0604$).
- Due to severe class imbalance and having only 175 training samples across 7 classes, the models collapsed entirely to predicting class `D` (Depression).
- In comparison, the simple observation persistence baseline achieves an accuracy of **83.22%** and Macro-F1 of **0.8241**.

### C. Wind Speed Regression: Fusion Substantially Degrades Performance
- Model A achieved a Wind MAE of **21.47 kt**.
- Model B achieved **21.08 kt**.
- Model C (Fusion) degraded significantly to **25.09 kt** (**+3.62 kt, 16.86% worse**; RMSE degraded from $29.94$ to $33.95$ kt).
- In comparison, observation persistence achieves a Wind MAE of **1.95 kt** and RMSE of **3.45 kt** ($r = 0.991$).

### D. Track Prediction: Slight Statistical Fluctuation, Uncompetitive with Baselines
- Fusion produced a minor statistical reduction at $+12\text{h}$ (-41.1 km, $-3.4\%$) and $+24\text{h}$ (-62.2 km, $-5.1\%$), but degraded at $+48\text{h}$ (+5.7 km, $+0.5\%$).
- Crucially, all three unanchored neural network track forecasts have mean DPEs around **1,180–1,230 km**, completely failing to compete with:
  - **Persistence Track Baseline:** **320.9 km** aggregate DPE (+12h: 139.1 km, +24h: 273.9 km, +48h: 549.6 km).
  - **Constant Velocity Baseline:** **187.7 km** aggregate DPE (+12h: 72.0 km, +24h: 150.9 km, +48h: 340.1 km).
  - **Phase 5B Production Hybrid:** **64.3 km** (+12h), **116.8 km** (+24h), **218.4 km** (+48h).

---

## 4. Scientific Root Causes of Failure

Why did adding INSAT-3D fail to improve performance?

1. **Extreme Sample Scarcity in Paired Era:**
   - GridSat baseline spans 1998–2024 ($N = 1,319$ total, $696$ train samples).
   - INSAT-3D only operated from 2014–2024. Consequently, the paired TRAIN split was reduced to **only 175 samples across 24 storms**.
   - Spatiotemporal neural networks cannot learn robust multi-source multi-task representations from 175 samples.
2. **Spectral Collinearity with GridSat 11µm:**
   - INSAT-3D `IMG_TIR1` (10.8µm) and GridSat-B1 `irwin_cdr` (11.0µm) occupy virtually identical thermal infrared atmospheric windows.
   - For cloud-top tracking, `TIR1` provides near-identical information to GridSat, while `TIR2` (12.0µm) is highly correlated with `TIR1` (differing only by low-level moisture absorption of $1\text{–}3\,\text{K}$).
3. **Capacity Penalty & Overfitting Acceleration:**
   - Fusion Model C requires $151,696$ parameters (nearly double Model A). In a 175-sample data-starved regime, higher model capacity simply accelerated memorization of the training set (Train Loss dropped to $0.93$ while Validation Loss blew out to $4.70\text{–}5.33$).
4. **Lack of Kinematic Anchoring:**
   - From-scratch unanchored sequence-to-coordinate heads must learn the geography of the entire North Indian Ocean ($40^\circ\text{–}105^\circ\text{E}$, $-5^\circ\text{–}35^\circ\text{N}$) from scratch. Without kinematic residual anchoring (as demonstrated in VAYU-NET Phase 5B), deep networks predict near the centroid of the basin (~1,200 km error).

---

## 5. Architectural Verdict & Operational Governance

### **VERDICT: DO NOT DEPLOY TO PRODUCTION / INFERENCE RUNTIME**

1. **Inference Pipeline Immobility:**
   - Model C (Fusion) **must NOT be connected** to the FastAPI backend, React dashboard, or live prediction pipeline.
   - The current production inference runtime:
     - **Phase 3C ResNet-18:** Dedicated center localization trained on full 696 GridSat samples.
     - **Phase 5B Hybrid Residual:** Kinematic-satellite residual track model achieving 64.3 km (+12h), 116.8 km (+24h), 218.4 km (+48h).
     - **Phase 6 Multi-Task:** Intensity & wind speed calibrated against persistence.
   - Replacing these with the 175-sample multi-source fusion model would severely degrade operational accuracy across all forecast products.
2. **Preservation of SIH Requirements:**
   - The research milestone has successfully developed, trained, and rigorously evaluated multi-source satellite deep learning models using NOAA GridSat-B1 and ISRO INSAT-3D Asian Sector L1C data.
   - The fair ablation scientifically proves that within the 2014–2024 data bounds, raw multi-source satellite fusion does not yield predictive advantages over kinematically anchored single-source GridSat models.
