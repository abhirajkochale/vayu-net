# VAYU-NET — EXP-M1 Multimodal Controlled Ablation Experiment Report
**Research Artifact — Not for Operational Warning Deployment**  
**Date:** September 25, 2026 (Audited & Corrected)  
**Status:** COMPLETE & AUDITED (15/15 Integrity Tests Passed, 73/73 Pre-existing Tests Passed)

---

## 1. Executive Summary & Core Empirical Findings

The **EXP-M1** experiment is the first controlled multimodal machine learning ablation for the VAYU-NET project. It benchmarks three identical multi-task architectures under strictly controlled, identical training protocols, differing *solely* in the satellite input modalities available at inference:

- **EXP-M1A:** GridSat-B1 Geostationary Infrared only ($[B, 6, 72, 116]$)
- **EXP-M1B:** NASA GPM IMERG Final Run V07B Precipitation only ($[B, 6, 72, 116]$)
- **EXP-M1C:** GridSat-B1 + GPM IMERG Late-Feature Fusion ($[B, 6, 72, 116] \times 2$)

### Primary Empirical Findings:
1. **Measured Point Estimates Favor GridSat-only (M1A):**
   - **Center Mean DPE:** $896.3 \text{ km}$ ($95\%\text{ CI: } [717.5, 1116.6]$) vs. $1119.5 \text{ km}$ ($[837.2, 1335.5]$) for IMERG and $1147.1 \text{ km}$ ($[905.5, 1337.0]$) for Fusion.
   - **Wind MAE:** $13.66 \text{ kt}$ ($95\%\text{ CI: } [8.76, 18.01]$) with Pearson $r = 0.610$, vs. $18.03 \text{ kt}$ ($[9.69, 26.58]$, $r = -0.040$) for IMERG and $17.63 \text{ kt}$ ($[8.98, 26.57]$, $r = 0.095$) for Fusion.
   - **Intensity Classification Accuracy:** $33.7\%$ (Macro-$F_1: 0.2336$) vs. $22.1\%$ ($F_1: 0.0606$) for IMERG and $22.9\%$ ($F_1: 0.0621$) for Fusion.
   - **Track Aggregate DPE:** $961.0 \text{ km}$ ($95\%\text{ CI: } [773.5, 1186.6]$) vs. $1098.5 \text{ km}$ ($[826.6, 1329.4]$) for IMERG and $1155.3 \text{ km}$ ($[979.1, 1334.5]$) for Fusion.
2. **Naive Late-Feature Fusion (M1C) Did Not Improve Over GridSat-Only (M1A):**
   - Under the tested architecture (separate 3-layer CNN encoders $\to$ 128-to-64 linear projection $\to$ unidirectional GRU), the multimodal model achieved higher validation loss ($3.4048$ vs $3.1669$) and higher test error across center, wind, classification, and track tasks.
3. **Statistical Uncertainty & Overlapping Confidence Intervals:**
   - Crucially, the 95% storm-level bootstrap confidence intervals for M1A ($[717.5, 1116.6]\text{ km}$), M1B ($[837.2, 1335.5]\text{ km}$), and M1C ($[905.5, 1337.0]\text{ km}$) overlap substantially. Therefore, while point estimates favor M1A, overlapping CIs do not establish statistically significant separation between M1A and M1C under the current test sample size ($N=31$ storms).
4. **Strict Ablation Integrity:**
   - All 3 models were trained on the exact same 696 sequences (81 storms), validated on the exact same 252 sequences (14 storms), and evaluated on the exact same 371 sequences (31 storms), using identical seeds, optimizers, learning rate schedules, and loss weightings. Zero future leakage ($\Delta t = 0$) and zero train-test storm contamination was preserved.

---

## 2. Dataset & Governance

- **Manifest:** `data/manifests/gridsat_imerg_multimodal_dataset_manifest.csv`
- **Total Sequences:** 1,319 six-frame multimodal sequences ($7,914$ GridSat frames $+ 7,914$ IMERG frames).
- **Unique Cyclonic Disturbances:** 126 storms (1998–2024) across the North Indian Ocean (Arabian Sea and Bay of Bengal).
- **Strict Storm-Level Partitions:**
  - **TRAIN:** 696 sequences across 81 storms (zero overlap with validation or test storms).
  - **VALIDATION:** 252 sequences across 14 storms.
  - **TEST:** 371 sequences across 31 storms.
- **Normalization (TRAIN-ONLY):**
  - GridSat IR brightness temperature: centered around $280.0 \text{ K}$, scaled by $30.0 \text{ K}$.
  - IMERG precipitation rate: $\mu_{\text{train}} = 0.180569 \text{ mm/hr}$, $\sigma_{\text{train}} = 1.084084 \text{ mm/hr}$ (`imerg_train_normalization_stats.json`). Zero validation or test granules were used in statistic computation.
- **Cache Engine:** High-throughput serialized tensor cache (`exp_m1_tensor_cache.pt`, 514.57 MB) guaranteeing zero I/O jitter during training and exact sample alignment.

---

## 3. Model Architecture & Modality Encoders

```
               [GridSat-B1: Bx6x72x116]          [IMERG: Bx6x72x116]
                         │                                │
           ┌─────────────┴─────────────┐    ┌─────────────┴─────────────┐
           │   Spatial CNN (GridSat)   │    │    Spatial CNN (IMERG)    │
           │  Conv2d(1->16, k=3, s=2)  │    │  Conv2d(1->16, k=3, s=2)  │
           │  Conv2d(16->32, k=3, s=2) │    │  Conv2d(16->32, k=3, s=2) │
           │  Conv2d(32->64, k=3, s=2) │    │  Conv2d(32->64, k=3, s=2) │
           │   AdaptiveAvgPool2d(2,2)  │    │   AdaptiveAvgPool2d(2,2)  │
           │   Linear(256 -> 64)       │    │   Linear(256 -> 64)       │
           └─────────────┬─────────────┘    └─────────────┬─────────────┘
                         │ 64-dim                         │ 64-dim
                         └───────────────┬────────────────┘
                                         ▼
                               [Concatenation: 128-dim]
                              (64-dim in unimodal M1A/B)
                                         │
                                         ▼
                               [Spatial Fusion: 64-dim]
                              (Linear(128->64) + LN + GELU)
                                         │
                                         ▼
                         [Temporal GRU: 64-dim (Unidirectional)]
                               (nn.GRU(64, 64, num_layers=1))
                                         │
                                         ▼
                            [Shared Projection: 64-dim]
                                         │
                 ┌───────────────┬───────┴───────┬───────────────┐
                 ▼               ▼               ▼               ▼
          Center Head       Wind Head      Category Head    Track Heads (+12/24/48)
          Linear(64->32)  Linear(64->32)  Linear(64->32)   Linear(64->32)
          Linear(32->2)   Linear(32->1)   Linear(32->7)    Linear(32->2) each
          [u_lat, u_lon]  [norm_wind]     [7 logits]       [u_lat, u_lon] each
```

### Parameter Counts:
- **M1A (GridSat-only):** 82,240 trainable parameters (82,240 total, 0 frozen)
- **M1B (IMERG-only):** 82,240 trainable parameters (82,240 total, 0 frozen)
- **M1C (Fusion):** 130,608 trainable parameters (130,608 total, 0 frozen)
  - Parameter difference in M1C ($+48,368$) stems strictly from the second spatial CNN ($+40,096$ weights/biases) and the spatial fusion layer Linear(128, 64) ($+8,272$ weights/biases/LayerNorm). Both unimodal and multimodal models feed identical 64-dimensional feature sequences to the unidirectional temporal GRU.

---

## 4. Target Formulation & Loss Formulation

### 4.1 Target Coordinate Formulation (Critical Clarification)
- **Synoptic Domain:** North Indian Ocean bounding box:
  - $\text{lat} \in [-5.0^\circ, 35.0^\circ \text{N}]$ ($\text{span} = 40.0^\circ$)
  - $\text{lon} \in [40.0^\circ, 105.0^\circ \text{E}]$ ($\text{span} = 65.0^\circ$)
- **Center Target ($t_0$):** Absolute normalized coordinate $[u_{\text{lat}}, u_{\text{lon}}] \in [0, 1]$ mapped via Sigmoid activation:
  $$u_{\text{lat}} = \frac{\text{lat} - (-5.0)}{40.0}, \quad u_{\text{lon}} = \frac{\text{lon} - 40.0}{65.0}$$
- **Forecast Track Targets ($+12\text{h}, +24\text{h}, +48\text{h}$):** **Absolute normalized geographic coordinates** $[u_{\text{lat}}, u_{\text{lon}}] \in [0, 1]$ over the same NIO synoptic domain, decoded directly to degrees.
- **IMPORTANT BENCHMARK CAVEAT:** Unlike previous VAYU-NET models that predicted relative storm displacements $\Delta = (\text{lat}_{t+\tau} - \text{lat}_{t_0}, \text{lon}_{t+\tau} - \text{lon}_{t_0})$, EXP-M1 employs **direct absolute coordinate regression**. Consequently, track error magnitudes reported here reflect direct coordinate prediction across a large geographic domain ($40^\circ \times 65^\circ$) and cannot be directly compared against displacement-based models without explicitly accounting for this architectural difference.

### 4.2 Multi-Task Loss Objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{center}} \mathcal{L}_{\text{center}} + \lambda_{\text{wind}} \mathcal{L}_{\text{wind}} + \lambda_{\text{cat}} \mathcal{L}_{\text{cat}} + \sum_{\tau \in \{12, 24, 48\}} \lambda_{\tau} \mathcal{L}_{\tau}$$
Where:
- $\lambda_{\text{center}} = 1.0$, $\lambda_{\text{wind}} = 1.0$, $\lambda_{\text{cat}} = 1.0$, $\lambda_{12} = 0.5$, $\lambda_{24} = 0.5$, $\lambda_{48} = 0.5$.
- Target masks apply strictly so missing forecast points ($+12\text{h}, +24\text{h}, +48\text{h}$) are ignored and zero-loss masked.

---

## 5. Controlled Experimental Results

### 5.1 Training Convergence & Validation Selection
| Model | Trainable Params | Training Time | Epochs Run | Best Epoch | Best Val Loss | Val Center DPE | Val Wind MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1A (GridSat)** | 82,240 | 86.7 s | 8 | **3** | **3.1669** | **1051.8 km** | **25.20 kt** |
| **M1B (IMERG)** | 82,240 | 58.4 s | 6 | **1** | 3.4343 | 1241.0 km | 28.06 kt |
| **M1C (Fusion)** | 130,608 | 103.9 s | 6 | **1** | 3.4048 | 1228.7 km | 28.69 kt |

*Note: All checkpoints evaluated on the test set were selected solely based on minimum validation loss during training.*

---

### 5.2 Compact EXP-M1 Comparison Summary (Test Set: 371 Sequences / 31 Storms)
Bootstrap confidence intervals are computed at the **storm level** using 1,000 resamples. Absolute and relative differences are shown relative to the unimodal GridSat baseline (EXP-M1A).

| Metric | Target | EXP-M1A (GridSat) | EXP-M1B (IMERG) | Diff vs M1A (Abs / Rel) | EXP-M1C (Fusion) | Diff vs M1A (Abs / Rel) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Center Mean DPE** | km $\downarrow$ | 896.3 km ($[717.5, 1116.6]$) | 1119.5 km ($[837.2, 1335.5]$) | +223.3 km (+24.9%) | 1147.1 km ($[905.5, 1337.0]$) | +250.8 km (+28.0%) |
| **Center Median DPE**| km $\downarrow$ | 726.8 km | 1046.8 km | +320.0 km (+44.0%) | 1141.1 km | +414.3 km (+57.0%) |
| **Center P90 DPE** | km $\downarrow$ | 1751.4 km | 1854.6 km | +103.2 km (+5.9%) | 1702.2 km | -49.2 km (-2.8%) |
| **Wind MAE** | kt $\downarrow$ | 13.66 kt ($[8.76, 18.01]$) | 18.03 kt ($[9.69, 26.58]$) | +4.37 kt (+32.0%) | 17.63 kt ($[8.98, 26.57]$) | +3.97 kt (+29.1%) |
| **Wind RMSE** | kt $\downarrow$ | 18.45 kt | 25.23 kt | +6.78 kt (+36.7%) | 25.36 kt | +6.91 kt (+37.5%) |
| **Wind Pearson $r$**| $[-1, 1] \uparrow$| +0.610 | -0.040 | -0.650 | +0.095 | -0.515 |
| **Category Accuracy**| % $\uparrow$ | 33.7% ($[26.2, 40.6]$) | 22.1% ($[13.8, 35.8]$) | -11.6% (-34.4%) | 22.9% ($[14.3, 37.2]$) | -10.8% (-32.0%) |
| **Category Macro $F_1$**| $[0, 1] \uparrow$| 0.2336 ($[0.157, 0.256]$)| 0.0606 ($[0.040, 0.091]$)| -0.1730 (-74.1%) | 0.0621 ($[0.042, 0.092]$)| -0.1715 (-73.4%) |
| **Track +12h DPE** | km $\downarrow$ | 929.3 km ($[740.6, 1155.5]$) | 1098.3 km ($[773.0, 1356.6]$) | +169.0 km (+18.2%) | 1134.2 km ($[921.8, 1313.9]$) | +204.9 km (+22.1%) |
| **Track +24h DPE** | km $\downarrow$ | 949.8 km ($[767.8, 1156.3]$) | 1102.6 km ($[831.4, 1330.6]$) | +152.8 km (+16.1%) | 1131.2 km ($[965.5, 1296.4]$) | +181.4 km (+19.1%) |
| **Track +48h DPE** | km $\downarrow$ | 1004.0 km ($[800.2, 1251.3]$)| 1094.7 km ($[855.5, 1324.4]$) | +90.7 km (+9.0%) | 1200.5 km ($[1029.5, 1418.2]$)| +196.5 km (+19.6%) |
| **Track Aggregate DPE**| km $\downarrow$| 961.0 km ($[773.5, 1186.6]$) | 1098.5 km ($[826.6, 1329.4]$) | +137.5 km (+14.3%) | 1155.3 km ($[979.1, 1334.5]$) | +194.3 km (+20.2%) |
| **Trainable Params** | count | 82,240 | 82,240 | 0 (0.0%) | 130,608 | +48,368 (+58.8%) |
| **Training Time** | sec | 86.7 s | 58.4 s | -28.3 s (-32.6%) | 103.9 s | +17.2 s (+19.8%) |

> [!IMPORTANT]
> **Statistical Significance Clarification:**
> While point estimates consistently favor M1A across center localization, wind speed, intensity classification, and track forecasting, the **95% storm-level bootstrap confidence intervals overlap substantially** between all three configurations. Overlapping confidence intervals indicate that we cannot claim statistically significant performance divergence at the $p < 0.05$ level. The measured data demonstrates only that naive late-feature fusion did not produce an observable performance improvement over the single-modality GridSat model in this experimental setup.

---

### 5.3 Storm-Level Granular Breakdown (All 31 Test Storms)
Complete evaluation across all 31 storms in the test partition:

| Storm ID | Storm Name | N | M1A Ctr (km) | M1B Ctr (km) | M1C Ctr (km) | M1A Wnd (kt) | M1B Wnd (kt) | M1C Wnd (kt) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `NIO_2021_GULAB` | GULAB | 9 | **530.2** | 793.7 | 1168.5 | 6.8 | **4.9** | 5.6 |
| `NIO_2021_JAWAD` | JAWAD | 7 | **467.4** | 492.8 | 690.5 | **5.4** | 5.9 | 5.6 |
| `NIO_2021_SHAHEEN` | SHAHEEN | 15 | **1858.1** | 1876.6 | 2001.9 | **10.8** | 15.3 | 16.0 |
| `NIO_2021_TAUKTAE` | TAUKTAE | 23 | 709.8 | **654.2** | 696.9 | **21.4** | 26.1 | 26.6 |
| `NIO_2021_UNNAMED_2_5_1` | UNNAMED | 2 | 393.5 | **104.2** | 379.5 | 1.8 | 2.2 | **1.1** |
| `NIO_2021_YAAS` | YAAS | 15 | 934.3 | **488.4** | 875.2 | **15.1** | 18.0 | 17.7 |
| `NIO_2022_ASANI` | ASANI | 19 | 1066.5 | **923.9** | 955.7 | **9.9** | 16.4 | 16.2 |
| `NIO_2022_MANDOUS` | MANDOUS | 10 | **373.1** | 1051.0 | 813.7 | **5.9** | 6.8 | 5.7 |
| `NIO_2022_NO` | UNNAMED | 8 | 484.8 | **445.7** | 630.1 | 2.7 | **1.7** | 2.3 |
| `NIO_2022_SITRANG` | SITRANG | 4 | **670.9** | 959.0 | 1254.5 | **4.0** | 7.6 | 7.4 |
| `NIO_2022_UNNAMED_01` | UNNAMED | 3 | **1273.5** | 1629.5 | 1576.7 | 11.6 | 10.9 | **9.0** |
| `NIO_2022_UNNAMED_2_13_1`| UNNAMED | 1 | **238.7** | 928.7 | 746.5 | **6.6** | 11.3 | 9.0 |
| `NIO_2022_UNNAMED_2_15_1`| UNNAMED | 4 | 1486.3 | **970.0** | 1153.8 | **2.6** | 9.0 | 6.2 |
| `NIO_2022_UNNAMED_2_16_1`| UNNAMED | 5 | **207.9** | 754.4 | 625.9 | **6.6** | 10.5 | 8.9 |
| `NIO_2022_UNNAMED_2_2_1` | UNNAMED | 5 | **527.7** | 941.1 | 689.4 | **8.5** | 10.8 | 8.8 |
| `NIO_2022_UNNAMED_2_9_1` | UNNAMED | 1 | 638.2 | **254.4** | 400.0 | 3.1 | 1.9 | **0.7** |
| `NIO_2023_BIPARJOY` | BIPARJOY | 81 | **756.1** | 1554.8 | 1431.4 | **23.2** | 38.2 | 38.7 |
| `NIO_2023_HAMOON` | HAMOON | 11 | 1357.3 | **872.9** | 1036.1 | 21.7 | 7.9 | **6.7** |
| `NIO_2023_MICHAUNG` | MICHAUNG | 16 | **223.0** | 736.6 | 585.7 | 6.3 | 7.6 | **5.6** |
| `NIO_2023_MIDHILI` | MIDHILI | 4 | 1191.1 | **467.9** | 825.5 | 23.4 | **6.5** | 6.6 |
| `NIO_2023_MOCHA` | MOCHA | 20 | 1472.2 | **992.5** | 1054.2 | **23.0** | 23.7 | 23.6 |
| `NIO_2023_TEJ` | TEJ | 16 | **1888.9** | 2312.8 | 2239.1 | **20.7** | 30.9 | 30.9 |
| `NIO_2023_UNNAMED_2_2_1` | UNNAMED | 6 | **468.8** | 928.0 | 735.7 | **5.2** | 10.5 | 8.3 |
| `NIO_2024_ASNA` | ASNA | 28 | **1288.3** | 1562.7 | 1529.8 | 3.6 | 4.1 | **3.4** |
| `NIO_2024_DANA` | DANA | 4 | 2023.5 | **1024.3** | 1381.5 | 29.3 | **7.3** | 7.6 |
| `NIO_2024_FENGAL` | FENGAL | 20 | **334.5** | 846.4 | 589.8 | **4.1** | 8.1 | 5.3 |
| `NIO_2024_LAT` | UNNAMED | 3 | 854.4 | **524.8** | 878.6 | 14.3 | 9.7 | **9.2** |
| `NIO_2024_NO` | UNNAMED | 10 | 747.8 | **671.7** | 945.7 | **2.4** | 3.6 | 2.8 |
| `NIO_2024_REMAL` | REMAL | 13 | **689.5** | 755.0 | 1251.9 | 10.9 | 8.8 | **8.4** |
| `NIO_2024_UNNAMED_2_4_1` | UNNAMED | 6 | **555.1** | 745.3 | 945.5 | 4.0 | 4.2 | **3.8** |
| `NIO_2024_UNNAMED_2_7_1` | UNNAMED | 2 | **211.8** | 578.7 | 656.4 | **1.2** | 8.5 | 5.3 |

**Summary of Storm Outcomes:**
- **M1A achieves lowest Center DPE** on 19 of 31 test storms ($61.3\%$). Notably, on large long-duration systems such as **BIPARJOY** ($N=81$), M1A achieved $756.1 \text{ km}$ vs. $1554.8 \text{ km}$ (M1B) and $1431.4 \text{ km}$ (M1C).
- **M1B achieves lowest Center DPE** on 12 of 31 storms ($38.7\%$), such as **YAAS** ($N=15$, $488.4 \text{ km}$ vs $934.3 \text{ km}$).
- **M1C does not achieve lowest Center DPE** on any individual test storm.

---

## 6. Scientific Interpretation & Methodological Governance

### 6.1 Strictly Evidence-Based Interpretation
- **Empirical Result:** The tested naive late-feature fusion architecture (M1C) did not improve multi-task performance over the unimodal GridSat baseline (M1A).
- **Absence of Causal Attribution Evidence:** Modality-specific gradient norms, layer-wise gradient contributions, and attention/attribution weights were not measured during EXP-M1 training. Therefore, hypotheses such as "gradient dilution" or "modality interference" remain unverified conjectures rather than experimentally demonstrated mechanisms.
- **Architectural Observations:**
  - Concatenating two 64-dimensional spatial vectors and projecting down via a simple linear layer before an un-pretrained temporal GRU provides no explicit inductive bias to select or prioritize informative modalities.
  - Precipitation fields are sparse and intermittent across oceanic domains ($>85\%$ zero-valued pixels), whereas infrared cloud top temperatures provide continuous synoptic coverage.

---

## 7. Operational & Scientific Disclaimer

> [!WARNING]  
> **EXP-M1 IS A CONTROLLED RESEARCH ABLATION ONLY.**  
> None of the models (M1A, M1B, M1C) produced in this experiment are approved or intended for operational tropical cyclone forecasting, civil protection, or early warning advisories. The operational VAYU-NET production inference pipeline and its validated checkpoints remain untouched and strictly isolated.

---

## 8. Integrity Audit Verification

All 15 mandatory integrity audit criteria were verified programmatically via `scripts/audit/validate_exp_m1_integrity.py`:
- [x] 1. Same Train Sample IDs across M1A/B/C ($N=696$)
- [x] 2. Same Validation Sample IDs ($N=252$)
- [x] 3. Same Test Sample IDs ($N=371$)
- [x] 4. Same 31 Test Storm IDs
- [x] 5. Zero Train/Test Storm Leakage
- [x] 6. Train-Only Normalization (`imerg_train_normalization_stats.json`)
- [x] 7. Zero Future Leakage (Strict Causality $\Delta t = 0$)
- [x] 8. Six Input Frames per Modality
- [x] 9. Correct Tensor Shapes ($[6, 72, 116]$)
- [x] 10. Multi-Task Loss Finite & Non-NaN
- [x] 11. Gradients Finite & Bounded
- [x] 12. Checkpoints Saved on Disk (`m1a_gridsat_only.pt`, `m1b_imerg_only.pt`, `m1c_gridsat_imerg.pt`)
- [x] 13. Model Selection strictly Validation-driven
- [x] 14. Evaluation Metrics from Test Partition Only
- [x] 15. Production Backend/Frontend/Checkpoints Unchanged
- [x] Pre-existing Test Suite Regression: 73/73 tests passed.

---

## 9. Recommendations for EXP-M2

Before proceeding to EXP-M2:
1. **Target Formulation Consistency:** Maintain absolute coordinate regression or explicitly benchmark against displacement targets if multi-horizon tracking is evaluated.
2. **Explicit Cross-Modal Gating:** Instead of naive linear concatenation, evaluate gated fusion or cross-attention mechanisms where the network can dynamically weight precipitation features.
3. **Modality Contribution Logging:** Log gradient norms for each encoder branch to test whether one modality dominates during backpropagation.
