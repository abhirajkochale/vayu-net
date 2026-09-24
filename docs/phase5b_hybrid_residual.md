# VAYU-NET Phase 5B: Environment-Aware Hybrid Residual Track Prediction

## 1. Executive Summary & Objective

**VAYU-NET Phase 5B** answers the central scientific question of the environmental steering investigation:
> *"Can atmospheric environmental steering information, combined with satellite temporal information, improve a physically anchored constant-velocity cyclone-track forecast by learning the residual deviation from that forecast?"*

Phase 4B established that predicting local residual displacements around an inertial constant-velocity anchor ($\hat{\mathbf{P}} = \mathbf{P}_{\text{kin}} + \hat{\mathbf{R}}$) avoided spatial coordinate collapse, but satellite IR temporal imagery alone lacked the synoptic steering signal needed to meaningfully beat exact kinematics ($189.71\text{ km}$ vs $189.06\text{ km}$). Conversely, Phase 5A demonstrated that unanchored, direct-coordinate multimodal regression ($978.72\text{ km}$) remains far inferior to physical kinematics.

Phase 5B couples multi-level atmospheric steering fields from ERA5 reanalysis ($u$ and $v$ wind components at 850, 700, 500, and 300 hPa) with 6-frame GridSat-B1 IR sequences within the **hybrid residual formulation**.

Two strictly separated configurations were evaluated:
- **Variant A (Observed-Center Diagnostic):** Kinematic anchor derived from exact IMD past centers ($t_{-3\text{h}}, t_0$). Tests whether environmental steering + satellite features can improve over an exact constant-velocity trajectory.
- **Variant B (Satellite-Derived Operational-Style):** Kinematic anchor derived from Phase 3C estimated centers ($t_{-3\text{h}}, t_0$). Tests whether environmental steering + satellite features can regularize and correct an end-to-end satellite-derived motion forecast.

---

## 2. Benchmark Results Summary on Held-Out TEST Split ($N=371$)

| Model / Configuration | Modalities | +12h DPE (Mean / Med / P90) | +24h DPE (Mean / Med / P90) | +48h DPE (Mean / Med / P90) | Mean Track DPE | Comparison vs Own Kinematic Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Persistence** | Exact $t_0$ Center | 132.7 / 130.1 / 211.7 km | 263.4 / 266.2 / 398.4 km | 530.8 / 534.8 / 785.1 km | **308.95 km** | -119.89 km |
| **Constant Velocity (Exact)** | Exact Centers ($t_{-3\text{h}}, t_0$) | 71.3 / 58.2 / 133.9 km | 150.7 / 124.1 / 285.9 km | 345.1 / 295.0 / 625.6 km | **189.06 km** | Baseline Reference |
| **Phase 4A Satellite-Only GRU** | 6 GridSat Frames | 823.0 / 628.0 / 1695.5 km | 832.5 / 658.8 / 1668.6 km | 890.0 / 732.8 / 1730.2 km | **848.50 km** | -659.44 km |
| **Phase 4B: Variant A (Sat Res)** | Exact Kin + 6 GridSat | 72.6 / 60.5 / 135.7 km | 151.1 / 124.5 / 288.0 km | 345.4 / 297.6 / 626.2 km | **189.71 km** | -0.65 km |
| **Phase 5A: Multimodal Direct** | 6 GridSat + 6 ERA5 Frames | 941.2 / 819.4 / 1947.4 km | 963.7 / 822.6 / 1858.8 km | 1031.2 / 844.8 / 1865.7 km | **978.72 km** | -789.66 km |
| **Phase 5B: Variant A (Env-Hybrid)** | **Exact Kin + Sat + ERA5** | **66.3 / 58.8 / 115.6 km** | **134.4 / 124.5 / 235.8 km** | **292.2 / 269.4 / 520.8 km** | **164.31 km** | **+24.75 km (+13.1% error reduction)** |
| **Satellite Kinematic Base** | Phase 3C Centers ($t_{-3\text{h}}, t_0$) | 2271.8 / 1217.1 / 5396.2 km | 3512.1 / 1668.1 / 8243.6 km | 4557.0 / 2385.9 / 10642.1 km | **3,447.00 km** | Baseline Reference |
| **Phase 4B: Variant B (Sat Res)** | Sat Kin + 6 GridSat | 1619.8 / 1195.6 / 3941.7 km | 2000.4 / 1595.6 / 4599.3 km | 2504.0 / 2176.3 / 4982.8 km | **2,041.43 km** | +1,405.57 km (+40.8%) |
| **Phase 5B: Variant B (Env-Hybrid)** | **Sat Kin + Sat + ERA5** | **864.2 / 467.5 / 2031.5 km** | **1034.1 / 564.1 / 2365.9 km** | **1386.6 / 826.9 / 3554.7 km** | **1,094.99 km** | **+2,352.01 km (+68.2% error reduction)** |

---

## 3. Scientific Discoveries & Analysis

### 3.1 Variant A Breaks the Kinematic Barrier
For the first time in the VAYU-NET research pipeline, a learned deep neural network has **statistically and physically beaten the deterministic constant-velocity kinematic baseline**:
- Test Mean Track DPE dropped from **189.06 km down to 164.31 km** (an improvement of **+24.75 km**, or **13.1% error reduction**).
- Improvements occurred consistently across all three forecast lead times:
  - **+12h:** $71.28\text{ km} \to \mathbf{66.32\text{ km}}$ ($+4.96\text{ km}$, 7.0% reduction)
  - **+24h:** $150.74\text{ km} \to \mathbf{134.43\text{ km}}$ ($+16.31\text{ km}$, 10.8% reduction)
  - **+48h:** $345.15\text{ km} \to \mathbf{292.17\text{ km}}$ ($+52.98\text{ km}$, **15.3% reduction**)
- The magnitude of improvement **scales directly with forecast horizon**, proving that atmospheric steering winds provide the dynamic curvature information that linear inertia inherently lacks over multi-day trajectories.

### 3.2 Geographic Breakdown: Resolving Bay of Bengal Recurvature
Stratifying test set performance by ocean basin reveals the meteorological origin of the improvement:

| Basin | Samples | Exact Kinematic Base | Phase 5B Variant A Hybrid | Net Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Arabian Sea** ($\lambda \le 77.5^\circ\text{E}$) | $N = 168$ | 153.84 km | **152.81 km** | $+1.03\text{ km}$ (+0.7%) |
| **Bay of Bengal** ($\lambda > 77.5^\circ\text{E}$) | $N = 203$ | 218.20 km | **173.82 km** | **$+44.38\text{ km}$ (+20.3%)** |

- **Arabian Sea Dynamics:** Cyclones in the Arabian Sea predominantly follow steady west-northwest tracks toward Oman or Gujarat. Linear kinematics is already near-optimal ($153.84\text{ km}$); the hybrid model preserves this base ($152.81\text{ km}$).
- **Bay of Bengal Dynamics:** Cyclones in the Bay of Bengal frequently interact with upper-tropospheric mid-latitude westerly troughs, inducing sharp northward/northeastward recurvature toward Bangladesh and Myanmar. The constant-velocity baseline suffers severe extrapolation errors ($218.20\text{ km}$). By sensing mid-tropospheric steering currents from ERA5 (500 and 300 hPa winds), Phase 5B Variant A corrects track curvature, reducing test DPE by **44.38 km** down to **173.82 km**.

### 3.3 Variant B: Environmental Damping Cuts Operational Error in Half
In Variant B (where cyclone centers are not known in advance and must be estimated from satellite imagery via Phase 3C), linear extrapolation of 3-hour center jitter produces runaway divergence ($3,447.00\text{ km}$ error).
- Phase 4B used satellite temporal features to dampen this error to $2,041.43\text{ km}$.
- Phase 5B adds multi-level environmental winds, which provide an external physical anchor for the storm's permissible steering speed and direction.
- This slashes Variant B error from $2,041.43\text{ km}$ down to **1,094.99 km** — an improvement of **+946.44 km** over Phase 4B and **+2,352.01 km (68.2% error reduction)** over the raw satellite kinematic anchor.

---

## 4. Multimodal Hybrid Architecture Specification

```
SATELLITE BRANCH
  GridSat Sequence [B, 6, 132] (Precomputed Phase 3C ResNet-18 Embeddings)
         │
         ▼
  2-Layer Satellite GRU (Hidden dim = 128, Dropout = 0.1) ──────────┐
         │                                                          │
         ▼                                                          │
     h_sat(t0) [B, 128]                                             │
                                                                    │
ENVIRONMENTAL BRANCH                                                │
  ERA5 Atmospheric Wind Sequence [B, 6, 8, 41, 66]                   ├──> Concatenation Fusion
         │                                                          │    [B, 128 + 64 + 16 = 208]
         ▼                                                          │        │
  3-Stage Spatial CNN (Conv-BN-GELU-Pool)                           │        ▼
         │  Spatial Embedding [B, 6, 64]                            │    Linear(208, 128) + LayerNorm
         ▼                                                          │    + GELU + Dropout(0.1)
  2-Layer Environmental GRU (Hidden dim = 64) ──────────────────────┤        │
         │                                                          │        ▼
     h_env(t0) [B, 64]                                              │    Multi-Horizon Residual Heads
                                                                    │        ├── Head +12h [B, 2]
MOTION CONTEXT BRANCH                                               │        ├── Head +24h [B, 2]
  Causal Motion Vector: [v_north, v_east, speed, lat0, lon0] [B, 5]  │        └── Head +48h [B, 2]
         │                                                          │        │
         ▼                                                          │        ▼
  Linear(5, 16) + LayerNorm + GELU ─────────────────────────────────┘    Predicted Residual Displacements:
         │                                                               [R_north_km, R_east_km]
         ▼                                                                   │
     h_motion [B, 16]                                                        ▼
                                                                         Geodesic Reconstruction:
                                                                         P_hat_h = P_kin_h + R_hat_h
```

### Parameter Distribution:
- Satellite Temporal GRU: 201,216 params
- Environmental Spatial CNN + GRU: 137,446 params
- Motion Context Projection: 112 params
- Multimodal Fusion MLP: 26,880 params
- Multi-Horizon Residual Heads: 10,896 params
- **Total Trainable Parameters:** **376,550**

---

## 5. Geodesic Invertibility & Mathematical Verification

Local displacements $[\Delta \text{North}, \Delta \text{East}]$ in kilometers are computed relative to the kinematic forecast anchor $\mathbf{P}_{\text{kin}} = (\phi_{\text{kin}}, \lambda_{\text{kin}})$:
$$\Delta \text{North} = (\phi_{\text{true}} - \phi_{\text{kin}}) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$
$$\Delta \text{East} = (\lambda_{\text{true}} - \lambda_{\text{kin}}) \cdot \cos\left(\phi_{\text{kin}} \cdot \frac{\pi}{180}\right) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$
where $R_{\text{earth}} = 6371.0\text{ km}$.

The inverse transformation:
$$\hat{\phi} = \phi_{\text{kin}} + \frac{180}{\pi} \cdot \frac{\hat{R}_{\text{north}}}{R_{\text{earth}}}$$
$$\hat{\lambda} = \lambda_{\text{kin}} + \frac{180}{\pi} \cdot \frac{\hat{R}_{\text{east}}}{R_{\text{earth}} \cdot \cos\left(\phi_{\text{kin}} \cdot \frac{\pi}{180}\right)}$$

### Mathematical Verification:
- **Zero-Residual Invariant:** Setting $\hat{\mathbf{R}} = \mathbf{0}$ yields $\hat{\mathbf{P}} \equiv \mathbf{P}_{\text{kin}}$ with exactly **$0.00\text{e+}00\text{ km}$** difference across all samples.
- **True-Residual Invariant:** Setting $\hat{\mathbf{R}} = \mathbf{R}_{\text{true}}$ reconstructs true IMD future coordinates with **$0.00\text{e+}00^\circ$** discrepancy.

---

## 6. Residual Dynamics & Horizon Growth Analysis

| Horizon | Mean Predicted $\|\hat{\mathbf{R}}\|$ | Median Predicted $\|\hat{\mathbf{R}}\|$ | North Bias ($\Delta \text{N}$) | East Bias ($\Delta \text{E}$) | Kinematic Error | Hybrid Error | Net Improvement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+12h** | 60.96 km | 51.16 km | $+3.22\text{ km}$ | $-3.30\text{ km}$ | 71.28 km | **66.32 km** | $+4.96\text{ km}$ (+7.0%) |
| **+24h** | 133.33 km | 117.73 km | $-5.72\text{ km}$ | $-13.63\text{ km}$ | 150.74 km | **134.43 km** | $+16.31\text{ km}$ (+10.8%) |
| **+48h** | 290.29 km | 261.63 km | $-0.58\text{ km}$ | $-10.95\text{ km}$ | 345.15 km | **292.17 km** | **$+52.98\text{ km}$ (+15.3%)** |

### Observations:
1. **Dynamic Scaling:** Unlike Phase 4B (where satellite-only residuals remained conservative at $\sim 20\text{--}27\text{ km}$), Phase 5B predicts substantial physical residual vectors that grow monotonically from $60.96\text{ km}$ at $+12\text{h}$ to $290.29\text{ km}$ at $+48\text{h}$.
2. **Unbiased Displacements:** Residual bias in both North and East components remains remarkably close to zero (e.g. North bias at $+48\text{h}$ is only $-0.58\text{ km}$), indicating balanced curvature corrections rather than systematic drift.

---

## 7. Representative Case Studies

All case studies were selected deterministically based on error differentials ($\Delta \text{DPE} = \text{DPE}_{\text{kin}} - \text{DPE}_{\text{hybrid}}$):

### Top 3 Successful Curvature Corrections:
1. **`NIO_2022_SITRANG` (2022-10-22 06:00 UTC):** Kinematic DPE = $1,088.2\text{ km} \to$ Hybrid DPE = **$376.5\text{ km}$** (**Improvement: $+711.7\text{ km}$**).
   - *Meteorological Context:* Sitrang formed in the central Bay of Bengal and rapidly recurved northward toward Bangladesh under the influence of an approaching upper-level westerly trough. Linear extrapolation continued projecting westward toward Andhra Pradesh. Environmental wind fields at 500 and 300 hPa captured the deep southerly steering current, successfully bending the forecast track northward.
2. **`NIO_2024_FENGAL` (2024-11-25 06:00 UTC):** Kinematic DPE = $637.9\text{ km} \to$ Hybrid DPE = **$148.4\text{ km}$** (**Improvement: $+489.5\text{ km}$**).
   - *Meteorological Context:* Captured westward deceleration and subsequent northwestward turn towards Tamil Nadu coast.
3. **`NIO_2021_JAWAD` (2021-12-03 00:00 UTC):** Kinematic DPE = $538.1\text{ km} \to$ Hybrid DPE = **$133.4\text{ km}$** (**Improvement: $+404.7\text{ km}$**).

### Top 3 Degradations:
- **`NIO_2024_UNNAMED_2_4_1` (2024-08-03 12:00 UTC):** Loss: $-230.6\text{ km}$.
- **`NIO_2023_BIPARJOY` (2023-06-14 12:00 UTC):** Loss: $-203.6\text{ km}$.
- **`NIO_2023_BIPARJOY` (2023-06-14 15:00 UTC):** Loss: $-201.3\text{ km}$.
- *Failure Mode Analysis:* Degradations occurred during landfall stall phases where the cyclone slowed to $< 5\text{ km/h}$ in shallow waters. Here, environmental winds indicated continued offshore advection while the vortex core stagnated due to coastal friction.

---

## 8. Anti-Leakage & Governance Invariants

As verified in [`docs/phase5b_leakage_audit.md`](file:///c:/GitHub/vayu-net/docs/phase5b_leakage_audit.md):
1. **Temporal Horizon:** For every sample initialized at $t_0$, all model inputs (satellite frames, ERA5 wind grids, motion context) strictly terminate at $t \le t_0$.
2. **Split Isolation:** Zero temporal overlap exists across TRAIN (`1998-2018`), VALIDATION (`2019-2020`), and TEST (`2021-2024`).
3. **Single-Pass Evaluation:** The held-out TEST set was evaluated strictly once after validation checkpoint selection.

---

## 9. Methodological Notice

- **Historical Reanalysis Classification:** As required by Phase 5 protocols, ERA5 is a retrospective reanalysis product incorporating 4D-Var data assimilation. Phase 5B demonstrates the *theoretical upper bound* of atmospheric steering flow for tropical cyclone track correction. Operational deployment would require substituting ERA5 with real-time operational global numerical weather prediction (NWP) analysis fields (such as GFS or ECMWF IFS HRES).

---

## 10. Visual Diagnostics

All 10 figures are located in [`docs/figures/phase5b_hybrid/`](file:///c:/GitHub/vayu-net/docs/figures/phase5b_hybrid/):
1. `validation_track_dpe.png`: Validation track DPE progression across epochs for Variant A and B.
2. `horizon_comparison.png`: Lead-time error progression (+12h, +24h, +48h) contrasting baselines and hybrid models.
3. `kinematic_vs_hybrid.png`: Paired comparisons of kinematic anchors vs hybrid forecasts.
4. `residual_predictions.png`: Horizon-wise distributions of predicted residual magnitudes.
5. `residual_bias.png`: Directional residual bias (North and East) across forecast horizons.
6. `test_dpe_distributions.png`: Empirical error density comparing exact kinematics and Phase 5B Variant A.
7. `variant_a_vs_variant_b.png`: Controlled benchmark comparison between Variant A and Variant B.
8. `arabian_sea_vs_bob.png`: Regional error breakdown between Arabian Sea and Bay of Bengal.
9. `representative_tracks.png`: Multi-horizon trajectory plots for top improved and degraded cases.
10. `worst_case_tracks.png`: Track trajectories for the 6 largest absolute error test cases.
