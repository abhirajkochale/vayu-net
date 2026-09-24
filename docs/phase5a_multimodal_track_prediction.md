# VAYU-NET Phase 5A: Environmental Steering + Satellite Track Prediction

## 1. Executive Summary & Objective

**VAYU-NET Phase 5A** investigates whether integrating atmospheric environmental steering flow from European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 reanalysis into the satellite temporal representation can improve future cyclone track prediction beyond standalone satellite-only models and deterministic kinematic baselines.

In Phase 4A and 4B, single-band satellite infrared sequences alone proved insufficient to beat an exact constant-velocity baseline (**189.06 km** test mean track DPE). Phase 5A constructs a rigorous, multimodal deep learning framework incorporating multi-level tropospheric wind fields ($u$ and $v$ components at 850, 700, 500, and 300 hPa) alongside GridSat-B1 IR sequences.

Two controlled primary experiments were executed:
- **Experiment B (Environment-Only Model):** Standalone 6-frame ERA5 temporal GRU predicting future track coordinates from atmospheric wind fields alone.
- **Experiment C (Multimodal Model):** Joint model combining the frozen Phase 3C satellite spatial backbone and environmental temporal GRU via late feature fusion.

---

## 2. Scientific Question & Controlled Hypotheses

### Primary Scientific Question:
> *"Does adding atmospheric environmental steering information to the satellite temporal representation improve cyclone track prediction beyond satellite-only and kinematic baselines?"*

### Hypotheses Evaluated:
1. **Hypothesis 1 (Steering Information Utility):** Environmental wind fields across the lower, middle, and upper troposphere capture broad synoptic pressure ridges, monsoonal troughs, and mid-latitude westerly troughs that dictate cyclone advection, providing essential steering context absent in cloud-top IR brightness temperatures.
2. **Hypothesis 2 (Resolution / Vortex Localization Trade-off):** Without explicit center tracking, a 1.0° resolution environmental wind field might lack localized vortex-core identity, making pure environmental models too coarse for sub-200 km track forecasting.

---

## 3. ERA5 Atmospheric Data Provenance & Invariants

All environmental data were acquired from the official ECMWF/Copernicus ERA5 hourly reanalysis on pressure levels via the cloud-optimized Google Cloud Public Datasets (`gcp-public-data-arco-era5`):
- **Official DOI:** [10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6)
- **Domain Extent:** $-5.0^\circ\text{N}$ to $+35.0^\circ\text{N}$, $40.0^\circ\text{E}$ to $105.0^\circ\text{E}$ (North Indian Ocean basin).
- **Target Levels:** 850 hPa, 700 hPa, 500 hPa, 300 hPa (4 pressure levels).
- **Variables:** $u$-component of wind (eastward), $v$-component of wind (northward) (8 channels total).
- **Spatial Resolution:** Raw $0.25^\circ \times 0.25^\circ$ ($161 \times 261$) downsampled via area-preserving bilinear interpolation to $\sim 1.0^\circ \times 1.0^\circ$ ($41 \times 66$ grid cells).
- **Temporal Alignment:** 6 native 3-hourly frames matching GridSat: $[t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0]$.
- **Storage & Integrity:** Raw slices stored in `data/raw/era5/` ($73\text{ KB}$ per timestamp); SHA256 checksums recorded in `data/manifests/era5_sha256_manifest.csv`. Feature cache packaged at `data/interim/ml/cache/era5_environment_features.pt`.

---

## 4. Coverage Audit & Anti-Leakage Verification

### 4.1 Temporal & Split Isolation Audit
- Total candidate samples: **1,319** across 33 storms.
- Total unique required ERA5 timestamps: **2,353**.
  - **TRAIN (1998–2018):** 1,304 unique timestamps (17 storms).
  - **VALIDATION (2019–2020):** 378 unique timestamps (7 storms).
  - **TEST (2021–2024):** 671 unique timestamps (9 storms).
  - **Temporal Overlap Across Splits:** **0 timestamps** (strictly isolated).

### 4.2 Anti-Leakage Invariants
- For every sample at initialization time $t_0$, model inputs strictly satisfy $t \le t_0$.
- Normalization statistics (`data/interim/ml/era5_train_normalization_stats.json`) were computed **strictly on TRAIN** samples. Zero validation or test data were utilized during feature normalization.
- Future track ground truths ($t_0 + 12\text{h}, +24\text{h}, +48\text{h}$) were reserved exclusively as supervised regression targets and evaluation references.

---

## 5. Model Architectures

```
SATELLITE BRANCH
  GridSat IR Sequence [B, 6, 1, 572, 929]
         │
         ▼
  Frozen Phase 3C Spatial Backbone (ResNet-18, 11.26M params)
         │  Layer4 Activations [B*6, 512, 18, 30]
         ▼
  Adaptive Pooling + Linear Projection -> [B, 6, 64]
         │
         ▼
  2-Layer Satellite GRU (Hidden dim = 128) ──> h_sat(t0) [B, 128]
                                                       │
                                                       ├──> Late Fusion [B, 192]
ENVIRONMENTAL BRANCH                                   │        │
  ERA5 Multi-Level Wind Sequence [B, 6, 8, 41, 66]      │        ▼
         │                                             │    Linear(192, 128) + GELU
         ▼                                             │        │
  3-Stage Environmental CNN (Conv-BN-GELU-Pool)        │        ▼
         │  Spatial Embedding [B, 6, 64]               │    Track Regression Heads
         ▼                                             │        ├── Head +12h [B, 2]
  2-Layer Environmental GRU (Hidden dim = 64) ─────────┘        ├── Head +24h [B, 2]
         │                                                      └── Head +48h [B, 2]
         ▼
     h_env(t0) [B, 64]
```

### Parameter Distribution:
- **Environment-Only Model:** 137,446 trainable parameters.
- **Multimodal Model:** 11,649,766 total parameters (11,170,240 frozen satellite backbone parameters + 479,526 trainable temporal fusion parameters).

---

## 6. Training & Checkpoint Selection Protocol

- **Objective Function:** Multi-horizon Smooth L1 loss on future (lat, lon) coordinates:
  $$\mathcal{L} = \sum_{h \in \{12, 24, 48\}} M_h \cdot \text{SmoothL1}(\hat{\mathbf{P}}_h, \mathbf{P}_h)$$
- **Optimizer:** AdamW ($\text{lr}=10^{-3}$, $\text{weight decay}=10^{-4}$, batch size 16).
- **Selection Criterion:** Checkpoints were evaluated at each epoch against the **VALIDATION set** using the **Mean Track DPE** (great-circle error).
  - **Environment-Only Best Checkpoint:** Epoch 5 (Val Track DPE: **1,373.33 km**).
  - **Multimodal Best Checkpoint:** Epoch 13 (Val Track DPE: **836.31 km**).
- **Strict Test Evaluation:** The held-out TEST set ($N=371$) was evaluated **strictly once** on the frozen checkpoints.

---

## 7. Comprehensive Benchmark Results on Held-Out TEST Set

| Model / Configuration | Modalities | +12h DPE (Mean / Med / P90) | +24h DPE (Mean / Med / P90) | +48h DPE (Mean / Med / P90) | Mean Track DPE | Comparison vs Kinematic Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Persistence** | Exact $t_0$ Center | 132.7 / 130.1 / 211.7 km | 263.4 / 266.2 / 398.4 km | 530.8 / 534.8 / 785.1 km | **308.95 km** | -119.89 km |
| **Constant Velocity (Exact)** | Exact Centers ($t_{-3\text{h}}, t_0$) | 71.3 / 58.2 / 133.9 km | 150.7 / 124.1 / 285.9 km | 345.1 / 295.0 / 625.6 km | **189.06 km** | **Baseline Ref** |
| **Phase 4A Satellite-Only GRU** | 6 GridSat Frames | 823.0 / 628.0 / 1695.5 km | 832.5 / 658.8 / 1668.6 km | 890.0 / 732.8 / 1730.2 km | **848.50 km** | -659.44 km |
| **Phase 5A: Environment-Only** | 6 ERA5 Wind Frames | 1220.2 / 985.9 / 2052.0 km | 1212.5 / 955.8 / 2111.5 km | 1215.7 / 1170.3 / 2175.3 km | **1,216.12 km** | -1,027.06 km |
| **Phase 5A: Multimodal (Sat+Env)** | 6 GridSat + 6 ERA5 Frames | 941.2 / 819.4 / 1947.4 km | 963.7 / 822.6 / 1858.8 km | 1031.2 / 844.8 / 1865.7 km | **978.72 km** | -789.66 km |

---

## 8. Scientific Interpretation & Key Findings

1. **Environmental Fields Alone Cannot Localize Storm Centers:**
   The Environment-Only model achieves a test mean DPE of **1,216.12 km**. Large-scale reanalysis wind fields at 1.0° resolution capture general monsoonal surges and broad westerly flow, but because the cyclone vortex itself is smoothed at this scale, the network cannot pinpoint the exact vortex center.
2. **Multimodal Fusion Substantially Improves Over Environment-Only:**
   Adding satellite imagery to the environmental fields reduces mean track error by **237.40 km** (from $1,216.12\text{ km}$ down to $978.72\text{ km}$, a $19.5\%$ error reduction). The satellite branch restores high-resolution cloud-top convective structure and eye/eyewall spatial signatures.
3. **Multimodal Direct Regression Does Not Beat Standalone Satellite-Only GRU:**
   The multimodal model (**978.72 km**) performs worse than the Phase 4A satellite-only model (**848.50 km** by $+130.22\text{ km}$). The additional 8 environmental channels increase feature dimensionality and optimization complexity without resolving the fundamental limitation of direct coordinate regression from unanchored representations.
4. **The Kinematic Baseline Remains Undefeated by Direct Neural Regression:**
   Both neural models remain drastically inferior to the deterministic constant-velocity kinematic baseline (**189.06 km**). Direct regression of future geographic coordinates from continuous rasters cannot discover the smooth inertial trajectory that simple physical differencing provides.

---

## 9. Methodological Limitations

- **Reanalysis vs. Operational NWP:** ERA5 is an offline reanalysis product with full four-dimensional variational data assimilation (4D-Var). It is not available in real-time operational environments. Thus, Phase 5A measures the *theoretical upper bound* of historical environmental steering utility, not real-time operational readiness.
- **Direct vs. Residual Formulation:** In Phase 4B, we proved that predicting residual deviations around a kinematic base ($\hat{\mathbf{P}} = \mathbf{P}_{\text{kin}} + \hat{\mathbf{R}}$) is vastly superior to predicting absolute coordinates from scratch. Phase 5A evaluated direct multimodal regression to establish an unconstrained baseline; future phases should combine environmental steering with residual kinematic formulations.

---

## 10. Visual Diagnostics

All 8 figures were generated and saved to `docs/figures/phase5a_environment/`:
1. `era5_wind_examples.png`: Wind speed contours and direction quiver vectors across 850, 500, and 300 hPa.
2. `environment_feature_distributions.png`: Per-channel mean and standard deviation from TRAIN normalization statistics.
3. `validation_dpe_comparison.png`: Validation track DPE progression and model selection checkpoints.
4. `horizon_dpe_comparison.png`: Breakdown of +12h, +24h, and +48h errors across all models.
5. `test_dpe_distributions.png`: Empirical error density histograms comparing baselines and neural variants.
6. `predicted_vs_actual_tracks.png`: Coordinate trajectories for historical cyclones (Remal, Mocha, Biparjoy).
7. `satellite_vs_environment_vs_multimodal.png`: High-level benchmark summary across all VAYU-NET models.
8. `failure_cases.png`: Analysis of large-error cases exhibiting track recurvature or shearing.
