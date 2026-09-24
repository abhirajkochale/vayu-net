# VAYU-NET Phase 4B: Hybrid Kinematic + Satellite Residual Track Prediction

## 1. Executive Summary & Objective

The primary objective of **VAYU-NET Phase 4B** is to investigate whether satellite temporal information (a six-frame, 3-hourly sequence of GridSat-B1 IR observations spanning $t_{-15\text{h}}$ to $t_0$) can improve future cyclone track forecasts over deterministic kinematic baselines by learning residual deviations rather than predicting absolute geographic coordinates from scratch.

In Phase 4A, a standalone temporal GRU directly regressing absolute latitude and longitude coordinates achieved a test mean track Great Circle Displacement (DPE) of **848.5 km**, failing to beat the simple deterministic constant-velocity kinematic baseline (**189.1 km**). Phase 4B implements a physically grounded hybrid architecture:
$$\hat{\mathbf{P}}_h = \mathbf{P}_{\text{kinematic}, h} + \hat{\mathbf{R}}_h, \quad h \in \{+12\text{h}, +24\text{h}, +48\text{h}\}$$
where $\mathbf{P}_{\text{kinematic}, h}$ is an extrapolated constant-velocity trajectory and $\hat{\mathbf{R}}_h = [\Delta \text{North}_h, \Delta \text{East}_h]$ is a learned residual displacement predicted by a recurrent temporal neural network.

Two strictly separated experimental variants were evaluated:
- **Variant A (Observed-Center Hybrid):** Kinematic motion derived from exact ground-truth IMD cyclone centers ($t_{-3\text{h}}, t_0$). Evaluates the incremental predictive value of satellite imagery when recent storm positions are known.
- **Variant B (Satellite-Derived Hybrid):** Kinematic motion derived from satellite-localized centers estimated by the frozen Phase 3C CNN. Evaluates an end-to-end, satellite-only operational configuration without ground-truth center inputs.

---

## 2. Motivation from Phase 4A & Scientific Hypothesis

### 2.1 Findings from Phase 4A
Phase 4A established that:
1. Spatial coordinate collapse was avoided (variance ratios $> 0.85$), and temporal ordering was validated.
2. Direct regression of absolute lat/lon coordinates from high-dimensional satellite imagery suffers from an accuracy floor constrained by satellite-only localization resolution (~810 km in Phase 3C).
3. The deterministic constant-velocity baseline is remarkably strong across the North Indian Ocean ($189.1\text{ km}$ mean track DPE).
4. Direct spatial regression cannot easily discover the smooth, persistent physical inertia of tropical cyclones without an explicit kinematic inductive bias.

### 2.2 Scientific Hypothesis
- **First-Order Kinematics:** Cyclone steering flow and atmospheric momentum ensure that short-term motion ($+12\text{h}$ to $+48\text{h}$) is dominantly linear and continuous.
- **Second-Order Satellite Signatures:** Changes in cloud convective asymmetry, eye eyewall tilts, spiral band curvature, and environmental shearing visible in temporal IR sequences correlate with track curvature, recurvature, or deceleration.
- **Residual Formulation:** Constraining the neural network to output only local displacements $[\Delta \text{North}, \Delta \text{East}]$ around an inertia-based trajectory frees the network from relearning basic planetary geography and focuses capacity on predicting dynamic deviations.

---

## 3. Dataset & Causal Sequence Specification

The locked VAYU-NET dataset invariants were strictly preserved:
- **Dataset Lock:** Zero modifications to `data/raw/imd/`, `data/processed/`, `data/interim/gridsat/`, or `data/manifests/`.
- **Temporal Sequence:** 6 native 3-hourly frames per sample:
  $$[t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0]$$
- **Input Dimensions:** $[B, 6, 1, 572, 929]$ (single-channel GridSat-B1 IRWIN CDR, calibrated brightness temperatures in Kelvin, normalized using train-only statistics: $\mu = 265.45\text{ K}, \sigma = 24.58\text{ K}$).
- **Split Partitions (Event-Level):**
  - **TRAIN:** 673 candidate $t_0$ samples (17 historical storms)
  - **VALIDATION:** 275 candidate $t_0$ samples (7 historical storms)
  - **TEST:** 371 candidate $t_0$ samples (9 historical storms)
  - **Total:** 1,319 samples
- **Strict Anti-Leakage:** Only observations at $t \le t_0$ are used in feature extraction. Future targets ($t > t_0$) are strictly masked and used exclusively as training supervisory targets or validation/test evaluation references.

---

## 4. Coordinate System & Local Tangent-Plane Geodesic Math

Directly differencing longitude degrees introduces severe latitude-dependent metric distortion ($\Delta \text{lon} \times \cos(\text{lat})$). To maintain physical consistency across the domain ($0^\circ\text{N}$ to $35^\circ\text{N}$, $45^\circ\text{E}$ to $115^\circ\text{E}$), all kinematic forecasts, neural predictions, and residuals are computed in a local tangent-plane geodesic coordinate system anchored at the cyclone position at $t_0$: $\mathbf{P}_0 = (\phi_0, \lambda_0)$.

### 4.1 Forward Transformation: Geographic to Local Tangent-Plane Residuals
Given an anchor position $(\phi_{\text{ref}}, \lambda_{\text{ref}})$ and a target position $(\phi, \lambda)$:
$$\Delta \phi = \frac{\pi}{180} (\phi - \phi_{\text{ref}})$$
$$\Delta \lambda = \frac{\pi}{180} (\lambda - \lambda_{\text{ref}})$$
$$\bar{\phi} = \frac{\pi}{180} \left( \frac{\phi + \phi_{\text{ref}}}{2} \right)$$
$$\Delta \text{North (km)} = R_{\text{earth}} \cdot \Delta \phi$$
$$\Delta \text{East (km)} = R_{\text{earth}} \cdot \Delta \lambda \cdot \cos(\bar{\phi})$$
where $R_{\text{earth}} = 6371.0\text{ km}$.

### 4.2 Inverse Transformation: Local Residuals to Geographic Position
$$\phi = \phi_{\text{ref}} + \frac{180}{\pi} \left( \frac{\Delta \text{North}}{R_{\text{earth}}} \right)$$
$$\bar{\phi} = \frac{\pi}{180} \left( \frac{\phi + \phi_{\text{ref}}}{2} \right)$$
$$\lambda = \lambda_{\text{ref}} + \frac{180}{\pi} \left( \frac{\Delta \text{East}}{R_{\text{earth}} \cdot \cos(\bar{\phi})} \right)$$

This transformation is closed-form, numerically stable, and exact to machine precision. Numerical invertibility was verified during audit tests (round-trip coordinate error $< 10^{-12\circ}$).

---

## 5. Deterministic Kinematic Baselines & Baseline Reproduction

### 5.1 Constant-Velocity Formulation
Using positions at $t_{-3\text{h}} = (\phi_{-3}, \lambda_{-3})$ and $t_0 = (\phi_0, \lambda_0)$:
1. Compute the 3-hour displacement vector in tangent plane anchored at $\mathbf{P}_{-3}$:
   $$\mathbf{v}_{3\text{h}} = [v_{\text{north}}, v_{\text{east}}] = \text{Forward}(\mathbf{P}_{-3}, \mathbf{P}_0)$$
2. Extrapolate linear motion from $\mathbf{P}_0$:
   $$\mathbf{P}_{\text{kin}, +12\text{h}} = \text{Inverse}(\mathbf{P}_0, 4 \times \mathbf{v}_{3\text{h}})$$
   $$\mathbf{P}_{\text{kin}, +24\text{h}} = \text{Inverse}(\mathbf{P}_0, 8 \times \mathbf{v}_{3\text{h}})$$
   $$\mathbf{P}_{\text{kin}, +48\text{h}} = \text{Inverse}(\mathbf{P}_0, 16 \times \mathbf{v}_{3\text{h}})$$

### 5.2 Verification Against Locked Historical Baselines
Before training the hybrid models, the locked baseline implementation was independently verified on the held-out TEST set:

| Baseline Model | +12h DPE | +24h DPE | +48h DPE | Mean Track DPE | Locked Benchmark Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Persistence** | 132.67 km | 263.37 km | 530.81 km | **308.95 km** | Exact match |
| **Constant Velocity (Exact)** | 71.28 km | 150.74 km | 345.15 km | **189.06 km** | Exact match |

---

## 6. Hybrid Architecture & Transfer Learning

```
[Satellite Sequence: 6 frames]
  B x [6, 1, 572, 929]
         │
         ▼
[Frozen Phase 3C Spatial Backbone] (11.26M params, ResNet-18)
         │  Layer-4 activations [B*6, 512, 18, 30]
         ▼
[Spatial Adaptive Pooling + Projection]
         │  Compact visual embedding [B, 6, 64]
         ▼
[Temporal Motion Fusion] (Concatenate recent center coordinates & velocities)
         │  Input dimension: 64 + 5 = 69
         ▼
[2-Layer Bidirectional/Causal GRU] (Hidden dim = 128, Dropout = 0.2)
         │  Terminal hidden state h(t0) [B, 128]
         ▼
[Multi-Horizon Residual Regression Heads]
         ├── Linear(128, 64) -> GELU -> Linear(64, 2) ==> R_hat(+12h) [km]
         ├── Linear(128, 64) -> GELU -> Linear(64, 2) ==> R_hat(+24h) [km]
         └── Linear(128, 64) -> GELU -> Linear(64, 2) ==> R_hat(+48h) [km]
         │
         ▼
[Geodesic Reconstruction: P_hat_h = P_kin_h + R_hat_h]
```

### 6.1 Model Components & Parameter Distribution
- **Spatial Encoder:** Conv1 through Layer4 transferred intact from `best_center_localization_cnn.pt` (Phase 3C). **All 11,263,777 parameters frozen**.
- **Spatial Projection:** AdaptiveAvgPool2d(2, 2) + Linear(2048, 64) + LayerNorm + GELU.
- **Kinematic Context Vector (5 dims):** $[\phi_0 / 90.0, \lambda_0 / 180.0, v_{\text{north}} / 100.0, v_{\text{east}} / 100.0, \|\mathbf{v}\| / 100.0]$.
- **Temporal Backbone:** 2-layer causal GRU, `input_size=69`, `hidden_size=128`, `dropout=0.2`.
- **Residual Heads:** Three independent MLP heads outputting $[\Delta \text{North}_h, \Delta \text{East}_h]$ scaled by $100.0\text{ km}$.
- **Trainable Parameters:** 324,486.
- **Total Parameters:** 11,588,263.

---

## 7. Experimental Variants: Variant A vs Variant B

```
VARIANT A (Observed-Center Hybrid)
   Exact IMD (t-3h, t0) ──> Exact Kinematic Forecast ──┐
                                                        ├──> Final Forecast A
   Satellite Frames (t-15h..t0) ──> Predicted Residual ──┘
   [Measures incremental satellite utility when past track is known]

VARIANT B (Satellite-Derived Hybrid)
   Phase 3C CNN (t-15h..t0) ──> Estimated Centers
                                      │
                                      ▼
                               Satellite Kinematic Forecast ──┐
                                                               ├──> Final Forecast B
   Satellite Frames (t-15h..t0) ──> Predicted Residual ────────┘
   [Measures end-to-end satellite operational track capability]
```

### 7.1 Variant A — Observed-Center Hybrid
- Inputs to kinematic extrapolation: Exact IMD positions at $t_{-3\text{h}}$ and $t_0$.
- Neural network receives: 6 satellite frames + exact recent motion context up to $t_0$.
- Target: Correct the exact constant-velocity trajectory.

### 7.2 Variant B — Satellite-Derived Hybrid
- Inputs to kinematic extrapolation: Phase 3C estimated centers at $t_{-3\text{h}}$ and $t_0$.
- Neural network receives: 6 satellite frames + estimated center motion context up to $t_0$.
- Ground-truth IMD positions are **never** provided as inputs (targets only).

---

## 8. Training & Model Selection

### 8.1 Objective Function
Multi-horizon Smooth L1 loss on normalized residual deviations:
$$\mathcal{L} = \sum_{h \in \{12, 24, 48\}} M_h \cdot \text{SmoothL1} \left( \frac{\hat{\mathbf{R}}_h - \mathbf{R}_h}{100.0\text{ km}} \right)$$
where $M_h \in \{0, 1\}$ is the validity mask indicating if the cyclone persisted to horizon $h$.

### 8.2 Model Selection Criterion
Checkpoints were evaluated at each epoch against the **VALIDATION set** using the **Mean Track DPE of the final hybrid forecast** ($\hat{\mathbf{P}}_h$ converted back to lat/lon and evaluated via great-circle distance). Checkpoint selection was **strictly forbidden** from utilizing test set metrics.

- **Variant A Best Val Epoch:** Epoch 1 (Val Mean Track DPE = **224.44 km**, checkpoint: `best_phase4b_variant_a.pt`).
- **Variant B Best Val Epoch:** Epoch 2 (Val Mean Track DPE = **1,841.75 km**, checkpoint: `best_phase4b_variant_b.pt`).

---

## 9. Comprehensive Benchmark Results on Held-Out TEST Set

The frozen checkpoints were evaluated strictly once against the locked TEST partition ($N=371$ candidate samples, 9 distinct storms).

### 9.1 Overall Benchmark Comparison Table

| Model / Configuration | Input Data Mode | +12h DPE (Mean / Med / P90) | +24h DPE (Mean / Med / P90) | +48h DPE (Mean / Med / P90) | Mean Track DPE | Comparison vs Own Kinematic Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Persistence** | Exact IMD $t_0$ | 132.7 / 130.1 / 211.7 km | 263.4 / 266.2 / 398.4 km | 530.8 / 534.8 / 785.1 km | **308.95 km** | N/A |
| **Constant Velocity (Exact)** | Exact IMD $t_{-3\text{h}}, t_0$ | 71.3 / 58.2 / 133.9 km | 150.7 / 124.1 / 285.9 km | 345.1 / 295.0 / 625.6 km | **189.06 km** | Baseline Reference |
| **Phase 4A Satellite-Only GRU** | 6 Satellite Frames (Direct Reg) | 823.0 / 628.0 / 1695.5 km | 832.5 / 658.8 / 1668.6 km | 890.0 / 732.8 / 1730.2 km | **848.50 km** | N/A |
| **Satellite Kinematic Base** | Phase 3C Centers ($t_{-3\text{h}}, t_0$) | 2271.8 / 1217.1 / 5396.2 km | 3512.1 / 1668.1 / 8243.6 km | 4557.0 / 2385.9 / 10642.1 km | **3,447.00 km** | Baseline Reference |
| **Phase 4B: Variant A (Hybrid)** | Exact Centers + 6 Sat Frames | 72.6 / 60.5 / 135.7 km | 151.1 / 124.5 / 288.0 km | 345.4 / 297.6 / 626.2 km | **189.71 km** | **-0.65 km** (0.3% degradation) |
| **Phase 4B: Variant B (Hybrid)** | Phase 3C Centers + 6 Sat Frames | 1619.8 / 1195.6 / 3941.7 km | 2000.4 / 1595.6 / 4599.3 km | 2504.0 / 2176.3 / 4982.8 km | **2,041.43 km** | **+1,405.57 km** (**40.8% error reduction**) |

---

## 10. Residual Dynamics & Horizon Analysis

### 10.1 Residual Magnitude and Bias Statistics on TEST

| Variant | Horizon | Mean Predicted $\|\hat{\mathbf{R}}\|$ | Median Predicted $\|\hat{\mathbf{R}}\|$ | North Bias ($\Delta \text{N}$) | East Bias ($\Delta \text{E}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | **+12h** | 19.42 km | 19.54 km | +11.22 km | -15.82 km |
| **Variant A** | **+24h** | 17.87 km | 17.92 km | +6.12 km | -16.67 km |
| **Variant A** | **+48h** | 27.15 km | 27.07 km | +26.65 km | -5.14 km |
| **Variant B** | **+12h** | 55.94 km | 54.57 km | +46.45 km | +14.75 km |
| **Variant B** | **+24h** | 62.07 km | 60.58 km | +44.78 km | +27.90 km |
| **Variant B** | **+48h** | 65.33 km | 68.37 km | +58.97 km | +5.46 km |

### 10.2 Analysis of Residual Behavior
1. **Conservative Corrections in Variant A:** When conditioned on exact historical center motion, the network learns that the kinematic baseline is already near-optimal. It outputs very small residual vectors (mean $17.8\text{--}27.1\text{ km}$). Because the true standard deviation of trajectory curvature is large compared to the weak satellite steering signal, any aggressive correction degrades validation loss. As a result, the network acts as a near-identity operator around the kinematic base.
2. **Horizon Growth:** In Variant A, the residual magnitude increases from $19.4\text{ km}$ at $+12\text{h}$ to $27.1\text{ km}$ at $+48\text{h}$, reflecting greater expected trajectory divergence at longer forecast lead times.
3. **Damping Action in Variant B:** Differencing two noisy single-frame satellite center predictions separated by only 3 hours creates severe velocity noise. Multiplying this noisy velocity vector by 16 for a 48-hour forecast causes catastrophic linear divergence ($4,557\text{ km}$ error). The GRU in Variant B learns to systematically counter and dampen this runaway linear extrapolation, reducing mean track error by **1,405.6 km**.

---

## 11. Geographic Error Stratification: Arabian Sea vs. Bay of Bengal

The test dataset was partitioned by geographic basin using longitude ($77.5^\circ\text{E}$ boundary):
- **Arabian Sea (AS):** $\lambda \le 77.5^\circ\text{E}$ ($N = 168$ test samples)
- **Bay of Bengal (BoB):** $\lambda > 77.5^\circ\text{E}$ ($N = 203$ test samples)

| Basin | Split Count | Exact Kinematic Base | Hybrid Variant A | Satellite Kinematic Base | Hybrid Variant B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabian Sea** | $N = 168$ | **153.84 km** | 155.14 km (+1.30 km) | 3,609.13 km | 2,172.09 km (**-1,437.04 km**) |
| **Bay of Bengal** | $N = 203$ | **218.20 km** | 218.32 km (+0.12 km) | 3,312.82 km | 1,933.29 km (**-1,379.53 km**) |

### Observations:
- **Baseline Disparity:** Cyclones in the Arabian Sea exhibit significantly lower kinematic error ($153.8\text{ km}$) than those in the Bay of Bengal ($218.2\text{ km}$), because Arabian Sea storms typically move along steady west-northwest tracks toward Oman or Gujarat, whereas Bay of Bengal storms frequently undergo recurvature toward Bangladesh/Myanmar upon interacting with mid-latitude troughs.
- **Uniformity of Residual Performance:** In both basins, Variant A tracks the exact kinematic baseline within $\pm 1.3\text{ km}$. Variant B achieves substantial, balanced error reduction in both basins ($-1437.0\text{ km}$ in AS, $-1379.5\text{ km}$ in BoB).

---

## 12. Representative Case Studies & Failure Analysis

All representative cases were selected deterministically based on sorting error differentials ($\Delta \text{DPE} = \text{DPE}_{\text{kin}} - \text{DPE}_{\text{hybrid}}$):

### 12.1 Variant A Case Studies
- **Top 3 Successful Corrections:**
  1. `NIO_2024_REMAL` (2024-05-25 06:00 UTC): Kinematic DPE = $99.9\text{ km} \to$ Hybrid DPE = **$75.1\text{ km}$** (Improvement: **+24.8 km**). Satellite imagery captured northward acceleration into the Bengal delta ahead of the linear extrapolation.
  2. `NIO_2022_ASANI` (2022-05-07 06:00 UTC): Kinematic DPE = $462.4\text{ km} \to$ Hybrid DPE = **$440.1\text{ km}$** (Improvement: **+22.3 km**). Successfully predicted early westward deceleration.
  3. `NIO_2021_YAAS` (2021-05-24 00:00 UTC): Kinematic DPE = $293.8\text{ km} \to$ Hybrid DPE = **$271.8\text{ km}$** (Improvement: **+22.0 km**).
- **Top 3 Degradations (Over-Correction):**
  1. `NIO_2022_ASANI` (2022-05-08 15:00 UTC): Kinematic DPE = $185.0\text{ km} \to$ Hybrid DPE = **$207.0\text{ km}$** (Degradation: **-22.0 km**). Recurvature sharply accelerated faster than predicted.
  2. `NIO_2023_BIPARJOY` (2023-06-10 15:00 UTC): Kinematic DPE = $45.9\text{ km} \to$ Hybrid DPE = **$67.2\text{ km}$** (Degradation: **-21.3 km**). Overpredicted northward deflection while the storm stalled in the central Arabian Sea.
  3. `NIO_2022_ASANI` (2022-05-08 21:00 UTC): Kinematic DPE = $239.9\text{ km} \to$ Hybrid DPE = **$260.9\text{ km}$** (Degradation: **-21.0 km**).

### 12.2 Variant B Case Studies
- **Top Improvements:** Extreme velocity corrections occurred for `NIO_2024_ASNA` and `NIO_2023_BIPARJOY` where 3-hour center jitter initially yielded spurious velocity vectors $> 100\text{ km/h}$. The GRU compressed these paths back toward reality, improving DPE by up to **+10,708.2 km**.
- **Worst Degraded Cases:** For storms where the satellite-derived center velocity happened to be fortuitously accurate (`NIO_2022_MANDOUS`, `NIO_2024_FENGAL`, `NIO_2023_MOCHA`), the network's default damping action caused minor degradations of **-56 to -59 km**.

---

## 13. Audit & Verification

### 13.1 Strict Anti-Leakage Audit
An audit of [`scripts/audit/test_hybrid_kinematic_satellite.py`](file:///c:/GitHub/vayu-net/scripts/audit/test_hybrid_kinematic_satellite.py) verified:
1. Feature sequence strictly terminates at $t_0$ ($t_{-15\text{h}}$ to $t_0$).
2. Kinematic inputs $v_{\text{north}}, v_{\text{east}}$ depend strictly on $(t_{-3\text{h}}, t_0)$.
3. No future coordinates ($+12\text{h}, +24\text{h}, +48\text{h}$) enter the network or feature cache.
4. Normalization constants are train-only.

### 13.2 Geographic & Physical Sanity Checks
- No predicted latitudes exceeded $[0^\circ, 40^\circ\text{N}]$.
- No predicted longitudes exceeded $[40^\circ, 110^\circ\text{E}]$.
- No NaN or Inf values occurred across all 1,319 dataset samples.
- Zero-residual reconstruction test: When $\hat{\mathbf{R}} = \mathbf{0}$, $\hat{\mathbf{P}} \equiv \mathbf{P}_{\text{kin}}$ to machine precision ($0.00\text{ km}$ difference).

---

## 14. Visual Diagnostics

All 10 diagnostic figures were generated and saved to `docs/figures/phase4b_hybrid/`:
1. `baseline_comparison.png`: Comprehensive bar chart contrasting Persistence, Constant Velocity, Phase 4A GRU, Variant A, and Variant B.
2. `validation_track_dpe.png`: Validation track DPE progression across training epochs.
3. `horizon_dpe_comparison.png`: Horizon-wise error breakdown (+12h, +24h, +48h).
4. `residual_distributions.png`: Histograms of predicted residual magnitudes across forecast horizons.
5. `predicted_vs_actual_residuals.png`: Scatter plots of predicted vs actual residual deviations.
6. `hybrid_vs_kinematic_tracks.png`: Paired sample comparisons of ground-truth tracks, kinematic baselines, and hybrid predictions.
7. `representative_variant_a.png`: Top 3 improved and top 3 degraded cases for Variant A.
8. `representative_variant_b.png`: Top 3 improved and top 3 degraded cases for Variant B.
9. `worst_case_hybrid_tracks.png`: Worst 6 overall absolute track predictions for Variant A.
10. `arabian_sea_vs_bob_comparison.png`: Regional error breakdown between the Arabian Sea and Bay of Bengal.

---

## 15. Scientific Conclusions & Decision Gate

1. **Does satellite temporal information improve an exact-center kinematic forecast (Variant A)?**
   **No.** When accurate recent cyclone positions are available, the constant-velocity baseline is exceptionally strong ($189.06\text{ km}$ mean track DPE). The satellite residual model learns conservative corrections that preserve this strong foundation ($189.71\text{ km}$ test DPE, a negligible difference of $-0.65\text{ km}$), but satellite imagery alone does not provide sufficient steering-flow signal to improve upon linear inertia.
2. **Does satellite temporal information improve a satellite-derived kinematic forecast (Variant B)?**
   **Yes, substantially within its domain.** Differencing single-frame satellite center estimates creates noisy velocities that diverge severely under linear extrapolation ($3,447.0\text{ km}$ DPE). The neural network successfully acts as a regularizing filter, cutting error by **40.8% (1,405.6 km improvement)** down to $2,041.43\text{ km}$. However, because center localization carries an inherent ~800 km uncertainty floor, Variant B remains far less accurate than exact kinematics.
3. **Scientific Implication:**
   Atmospheric steering flow is governed primarily by large-scale synoptic pressure fields (subtropical ridges, mid-latitude troughs, monsoon depressions) that are largely invisible in single-channel satellite IR cloud-top patterns alone. Future phases must incorporate environmental steering context (e.g., numerical reanalysis wind fields or multi-spectral soundings) to capture track curvature.
