# VAYU-NET Phase 4A — Temporal Cyclone Track Prediction

**Status:** COMPLETE & RIGOROUSLY EVALUATED  
**Author:** Antigravity AI & VAYU-NET Research Team  
**Date:** September 2026  
**Primary Research Question:** *"Does a six-frame temporal satellite sequence improve future cyclone track prediction over deterministic persistence and constant-velocity baselines?"*

---

## 1. Executive Summary & Core Scientific Findings

Phase 4A introduces the first temporal sequence model in the VAYU-NET intelligence project. Using a shared, frozen spatial representation transferred directly from the validated Phase 3C checkpoint ([`best_center_localization_cnn.pt`](file:///c:/GitHub/vayu-net/data/interim/ml/checkpoints/best_center_localization_cnn.pt)), the **`TemporalTrackGRU`** processes a 6-frame causal sequence of GridSat-B1 infrared observations spanning $t_{-15\text{h}}$ to $t_0$ (at native 3-hour cadence) to forecast cyclone center positions across three distinct horizons: **+12h, +24h, and +48h**.

### Key Findings & Honest Scientific Reality:
1. **The Temporal Model Does NOT Beat Deterministic Baselines:**
   - On the held-out TEST set ($N=371$ samples across 31 unseen storms), `TemporalTrackGRU` achieves a **Mean Track DPE of 848.5 km** and a **Median Track DPE of 661.9 km**.
   - By comparison, the deterministic **Constant-Velocity Baseline** achieves a **Mean Track DPE of 189.1 km** (71.3 km at +12h, 150.7 km at +24h, 345.1 km at +48h).
   - The deterministic **Stationary Persistence Baseline** achieves a **Mean Track DPE of 308.9 km** (132.7 km at +12h, 263.4 km at +24h, 530.8 km at +48h).
2. **The Root Scientific Cause (The Single-Frame Spatial Uncertainty Floor):**
   - In physical reality, tropical cyclones move slowly (~10–25 km/h), typically displacing only ~120–250 km over 12 hours and ~500–1000 km over 48 hours.
   - Deterministic baselines start from the **exact, verified ground-truth IMD center at $t_0$**; thus their initial error is zero ($0\text{ km}$ at $t_0$).
   - In contrast, pure satellite neural models must infer both where the storm is at $t_0$ and where it will travel. The Phase 3C spatial center-localization baseline established that single-frame infrared satellite observation carries an inherent mean localization uncertainty of **810.3 km** (median 407.4 km).
   - Because the neural model operates without ground-truth IMD coordinate anchoring, its future forecast inherits this ~800 km spatial localization error floor.
3. **What the Temporal Sequence Learned:**
   - **Monotonic Forecast Error Growth:** Predictions exhibit physically realistic temporal dispersion: $+12\text{h} (823.0\text{ km}) \to +24\text{h} (832.5\text{ km}) \to +48\text{h} (889.9\text{ km})$.
   - **Stable Convergence:** The multi-horizon loss decreased steadily from $0.0380$ to $0.0074$, reducing Validation Mean Track DPE from $1,245.5\text{ km}$ down to **$724.9\text{ km}$** (Epoch 13).
   - **Absence of Spatial Collapse:** Trajectories actively track storm movement across the Arabian Sea and Bay of Bengal, rather than collapsing to the regional centroid.

---

## 2. Multi-Horizon Benchmark Comparison (Strictly Held-Out TEST Split)

All evaluations below reflect the strictly held-out **TEST Split ($N=371$ samples across 31 storms)**:

| Model / Baseline | Input Modality | Ground-Truth t0 Anchor? | +12h DPE (Mean / Med / P90) | +24h DPE (Mean / Med / P90) | +48h DPE (Mean / Med / P90) | Mean Track DPE (Across Horizons) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Persistence** | IMD Ground-Truth $t_0$ | **YES** | 132.7 / 130.1 / 211.7 km | 263.4 / 266.2 / 398.4 km | 530.8 / 534.8 / 785.1 km | **308.9 km** |
| **Constant Velocity** | IMD Ground-Truth $t_{-3h}, t_0$ | **YES** | **71.3 / 58.2 / 133.9 km** | **150.7 / 124.1 / 285.9 km** | **345.1 / 295.0 / 625.6 km** | **189.1 km** |
| **TemporalTrackGRU (Phase 4A)** | 6-Frame IR ($t_{-15h} \dots t_0$) | **NO (Satellite-only)** | 823.0 / 628.0 / 1695.5 km | 832.5 / 658.8 / 1668.6 km | 890.0 / 732.8 / 1730.2 km | **848.5 km** |

---

## 3. Dataset & Causal Sequence Construction

The experiment strictly preserves all locked dataset governance protocols:
- **Total Valid Dataset Samples:** 1,319
  - **TRAIN Split:** 696 samples / 81 distinct storms
  - **VALIDATION Split:** 252 samples / 14 distinct storms
  - **TEST Split:** 371 samples / 31 distinct storms (Zero storm overlap)
- **Temporal Sequence:** Exactly 6 source-native GridSat-B1 frames per sample:
  $$[t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0]$$
  Cadence: 3 hours. Spatial resolution: $572 \times 929$.
- **Normalization Invariant:** Per-pixel standardization using TRAIN-only statistics:
  $z = (x - 265.4856\text{ K}) / 21.0559\text{ K}$.
- **Zero Future Leakage:** The input strictly terminates at $t_0$. No future satellite observations ($t_{+3\text{h}}$ onwards), no future coordinates, and no future intensity labels enter the model.

---

## 4. Architecture & Spatial-Temporal Formulation

Implemented in [`ml/models/temporal_track_gru.py`](file:///c:/GitHub/vayu-net/ml/models/temporal_track_gru.py) as `TemporalTrackGRU`:

```
Satellite Sequence: [B, 6, 572, 929]
        │
        ▼ (Shared across all 6 time steps)
Frozen Phase 3C Spatial Encoder (DedicatedCenterLocalizationResNet, 11.26M params)
  ├─► layer2 Feature Map: [B, 128, 72, 117]
  ├─► Spatial Adaptive AvgPool2d((2, 3)) + Linear(768, 128) + BN + ReLU ──► Spatial Embedding [B, 128]
  └─► CoordConv + Center Decoder + Soft-Argmax ──► Estimated Center [B, 2] in [0, 1]
        │
        ▼
Per-Step Feature Assembly:
  x_t = [ spatial_embedding (128),
          norm_center_lat (1), norm_center_lon (1),
          delta_lat (1), delta_lon (1) ] ──► [B, 132]
        │
        ▼
Temporal Sequence Tensor: [B, 6, 132]
        │
        ▼
2-Layer GRU (input_size=132, hidden_size=128, dropout=0.1, batch_first=True)
        │
        ▼
Final Hidden State at t0: h_6 [B, 128]
        │
        ├─► Head +12h: Linear(128, 64) -> ReLU -> Linear(64, 2) -> Sigmoid ──► [B, 2] in [0, 1]
        ├─► Head +24h: Linear(128, 64) -> ReLU -> Linear(64, 2) -> Sigmoid ──► [B, 2] in [0, 1]
        └─► Head +48h: Linear(128, 64) -> ReLU -> Linear(64, 2) -> Sigmoid ──► [B, 2] in [0, 1]
        │
        ▼
Physical Coordinate Denormalization:
  lat = -5.0° + u_lat * 40.0°
  lon = 40.0° + u_lon * 65.0°
```

### Parameter Breakdown:
- **Total Parameters:** 11,587,303
- **Frozen Spatial Parameters:** 11,263,777 (Phase 3C weights strictly preserved)
- **Trainable Temporal Parameters:** 323,526 (Spatial projection, 2-layer GRU, forecast heads)

---

## 5. Loss Formulation & Training Dynamics

### Multi-Horizon Coordinate Loss:
$$\mathcal{L}_{\text{track}} = \text{SmoothL1}(\hat{c}_{12}, c^*_{12}) + \text{SmoothL1}(\hat{c}_{24}, c^*_{24}) + \text{SmoothL1}(\hat{c}_{48}, c^*_{48})$$
where $\hat{c}_h$ and $c^*_h$ are the predicted and ground-truth normalized coordinate pairs in $[0, 1]$.

### Training Regimen:
- **Optimizer:** AdamW ($\text{LR} = 10^{-4}$, Weight Decay = $10^{-4}$)
- **Batch Size:** 16 (deterministic seed 42)
- **Early Stopping & Model Selection:** Lowest **Validation Mean Track DPE (km)**, patience = 4.
- **High-Throughput Caching:** To prevent CPU bottlenecking, frame-level features across all 2,353 unique NPZ satellite frames were extracted and cached to [`data/interim/ml/cache/temporal_track_features.pt`](file:///c:/GitHub/vayu-net/data/interim/ml/cache/temporal_track_features.pt) (4.18 MB).

### Epoch-by-Epoch Validation Progression:

| Epoch | Train Loss | Val Loss | Val Mean Track DPE | Val +12h DPE | Val +24h DPE | Val +48h DPE | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0380 | 0.0340 | 1,245.5 km | 1,214.2 km | 1,239.8 km | 1,282.5 km | New Best |
| 2 | 0.0302 | 0.0302 | 1,134.9 km | 1,104.6 km | 1,133.6 km | 1,166.4 km | New Best |
| 3 | 0.0225 | 0.0214 | 895.8 km | 888.3 km | 897.1 km | 902.0 km | New Best |
| 4 | 0.0157 | 0.0188 | 829.9 km | 844.0 km | 826.0 km | 819.8 km | New Best |
| 6 | 0.0104 | 0.0184 | 820.8 km | 830.7 km | 810.7 km | 821.0 km | New Best |
| 7 | 0.0097 | 0.0177 | 797.2 km | 806.2 km | 784.7 km | 800.8 km | New Best |
| 10 | 0.0084 | 0.0167 | 761.9 km | 762.0 km | 748.5 km | 775.3 km | New Best |
| **13** | **0.0078** | **0.0162** | **724.9 km** | **713.3 km** | **708.8 km** | **752.5 km** | **BEST CHECKPOINT SAVED** |
| 14 | 0.0079 | 0.0178 | 782.3 km | 773.3 km | 763.1 km | 810.4 km | Counter: 1/4 |
| 15 | 0.0074 | 0.0176 | 778.8 km | 765.9 km | 760.8 km | 809.7 km | Counter: 2/4 |

Checkpoint at **Epoch 13** achieved the lowest Validation Mean Track DPE ($724.9\text{ km}$) and was frozen for single-pass evaluation.

---

## 6. Final Evaluation Breakdown Across Splits

| Split | Sample Count | Mean Track DPE | +12h DPE (Mean / Med / P90) | +24h DPE (Mean / Med / P90) | +48h DPE (Mean / Med / P90) | Lat / Lon MAE (+24h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 696 | 514.9 km | 484.0 / 374.8 / 985.7 km | 496.3 / 390.1 / 967.5 km | 564.3 / 452.4 / 1086.4 km | 1.76° / 3.88° |
| **VALIDATION** | 252 | 724.9 km | 713.3 / 509.0 / 1651.8 km | 708.8 / 518.2 / 1515.3 km | 752.5 / 532.2 / 1519.5 km | 2.21° / 5.51° |
| **TEST** | 371 | **848.5 km** | **823.0 / 628.0 / 1695.5 km** | **832.5 / 658.8 / 1668.6 km** | **890.0 / 732.8 / 1730.2 km** | **2.55° / 6.87°** |

---

## 7. Diagnostic Visualizations

Generated and stored under [`docs/figures/temporal_track/`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/):

1. **[`train_loss.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/train_loss.png):** Multi-horizon Smooth L1 loss curve illustrating smooth convergence.
2. **[`validation_track_dpe.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/validation_track_dpe.png):** Mean and median track DPE progression during training, highlighting the selection at Epoch 13.
3. **[`horizon_dpe_comparison.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/horizon_dpe_comparison.png):** Bar chart comparing Stationary Persistence, Constant Velocity, and TemporalTrackGRU across +12h, +24h, and +48h.
4. **[`test_dpe_distributions.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/test_dpe_distributions.png):** Histograms of position errors on the held-out TEST set for all three forecast horizons.
5. **[`predicted_vs_actual_tracks.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/predicted_vs_actual_tracks.png):** Basin-wide geographic trajectory map comparing true and predicted tracks across the North Indian Ocean.
6. **[`representative_tracks.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/representative_tracks.png):** Case studies of low-error and median-error forecasts demonstrating coherent trajectory alignment.
7. **[`worst_case_tracks.png`](file:///c:/GitHub/vayu-net/docs/figures/temporal_track/worst_case_tracks.png):** Detailed inspection of top worst-case track errors showing basin-switching anomalies during recurvature.

---

## 8. Limitations & Scientific Analysis

1. **The Representation Gap between Satellite Imagery and Point Kinematics:**
   - Predicting future trajectory purely from unanchored satellite pixels requires the model to solve two concatenated problems: (1) localize the storm's current center, and (2) estimate its future advection.
   - When operational meteorologists or deterministic track models forecast tracks, they anchor the trajectory to the current best-estimate center coordinate ($t_0$).
2. **Kinematic Baseline Superiority:**
   - Because cyclones possess substantial inertia, constant-velocity extrapolation ($\vec{v} = (\vec{x}_{t_0} - \vec{x}_{t_{-3\text{h}}}) / 3\text{h}$) is exceptionally difficult to beat within 12–24 hours when given the exact IMD ground truth.
   - A satellite-only model that lacks ground-truth coordinate anchoring cannot beat a 71 km baseline when its own center detection error is ~400–800 km.
3. **Implications for Phase 4B:**
   - Future track prediction architectures should evaluate a **hybrid kinematic-satellite formulation**, where the model receives both the known historical coordinates at $t_{-3\text{h}}, t_0$ (as in operational forecasting) and uses satellite features to predict the **deviation from constant velocity** ($\Delta \vec{v}$).

---

## 9. Decision Gate Block

```text
============================================================
PHASE 4A STATUS
============================================================

1. Best validation +12h DPE:              713.3 km
2. Best validation +24h DPE:              708.8 km
3. Best validation +48h DPE:              752.5 km
4. Best validation mean track DPE:        724.9 km
5. Selected epoch:                        13

6. TEST +12h mean/median/P90:             823.0 / 628.0 / 1695.5 km
7. TEST +24h mean/median/P90:             832.5 / 658.8 / 1668.6 km
8. TEST +48h mean/median/P90:             890.0 / 732.8 / 1730.2 km

9. TEST mean track DPE:                   848.5 km

10. Constant-velocity baseline:           189.1 km Mean (71.3 / 150.7 / 345.1 km)
11. Temporal GRU result:                  848.5 km Mean (823.0 / 832.5 / 890.0 km)

12. Beats persistence?:                   NO (-539.5 km Mean Track DPE vs 308.9 km)
13. Beats constant velocity?:             NO (-659.4 km Mean Track DPE vs 189.1 km)

14. Forecast error grows with horizon?:   YES (+12h: 823.0 km -> +24h: 832.5 km -> +48h: 890.0 km)

15. Future leakage detected?:             NO (Strictly causal, inputs end at t0)
16. Spatial collapse detected?:           NO (Active dynamic multi-horizon tracks)

17. Temporal model sufficiently
    validated for Phase 4B?               YES (Establishes the satellite-only track benchmark;
                                          Phase 4B should explore kinematic-residual track modeling)

18. Recommended next step:                Conclude Phase 4A and formulate Phase 4B
                                          (residual motion modeling around operational coordinates
                                          or temporal intensity forecasting).
============================================================
```
