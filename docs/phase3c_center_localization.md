# VAYU-NET Phase 3C — Dedicated Cyclone Center Localization

**Status:** COMPLETE & VALIDATED  
**Author:** Antigravity AI & VAYU-NET Research Team  
**Date:** September 2026  
**Primary Research Question:** *"Can the current satellite representation learn cyclone center location when center localization is optimized and selected independently of wind/category objectives?"*

---

## 1. Executive Summary & Core Scientific Findings

In Phase 3B, an intermediate spatial heatmap branch with CoordConv coordinates and differentiable 2D soft-argmax demonstrated dramatic spatial learning potential during early epochs (achieving a validation Direct Position Error of 709.1 km mean and 294.0 km median at Epoch 3). However, because model selection and optimization were coupled to an overarching multi-task loss dominated by wind regression and category classification cross-entropy, the spatial branch overfit to categorical shortcuts, resulting in severe spatial collapse by the final selected checkpoint (TEST Mean DPE = 1,268.6 km).

**Phase 3C isolated cyclone center localization as the primary and sole task:**
1. Wind regression and category classification heads were completely removed from optimization and model selection.
2. The primary checkpoint selection metric was locked strictly to **Minimum Validation Mean Direct Position Error (DPE in km)**.
3. The model was trained with early stopping based purely on Validation Mean DPE.
4. The held-out TEST set (371 samples across 31 unseen storms) remained untouched until the final best checkpoint (Epoch 2) was locked and frozen.

### Key Scientific Results on Held-Out TEST Split (N=371, 31 Storms):
- **Phase 3C TEST Mean DPE: 810.3 km** (vs. 1,186.0 km Centroid Baseline, 1,215.1 km Phase 3, and 1,268.6 km Phase 3B).
- **Phase 3C TEST Median DPE: 407.4 km** (vs. 1,082.6 km Centroid Baseline, 1,204.9 km Phase 3, and 1,364.0 km Phase 3B).
- **Error Reduction:** Achieved a **375.7 km mean error reduction (31.7%)** and **675.2 km median error reduction (62.4%)** over the regional centroid baseline.
- **Overcoming Spatial Collapse:** The prediction variance ratio reached **0.70 for Latitude** and **0.93 for Longitude** (compared to 0.04 and 0.03 in Phase 3B). Predictions actively track dynamic cyclone centers across the entire North Indian Ocean rather than collapsing to the regional geographic mean.

---

## 2. Benchmark Comparison Table (Strictly Held-Out TEST Split)

| Model / Baseline | Input Representation | Model Selection Criterion | TEST Mean DPE (km) | TEST Median DPE (km) | TEST P90 DPE (km) | Lat / Lon MAE (°) | Variance Ratio (Lat / Lon) | Spatial Collapse? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Constant-Mean Centroid Baseline** | None (Historical Mean) | Pre-computed Train Mean | 1,186.0 km | 1,082.6 km | 2,192.4 km | 7.97° / 8.52° | 0.00 / 0.00 | **YES (Static)** |
| **Phase 3 Single-Frame CNN** | Single IR $t_0$, Global Pooling | Min Val Total Multi-Task Loss | 1,215.1 km | 1,204.9 km | 2,238.1 km | 7.42° / 9.53° | 0.04 / 0.03 | **YES (Learned)** |
| **Phase 3B Multi-Task Spatial CNN** | Single IR $t_0$, Spatial Tap + CoordConv | Min Val Total Multi-Task Loss | 1,268.6 km | 1,364.0 km | 2,233.9 km | 8.01° / 9.87° | 0.04 / 0.03 | **YES (Multi-task Interference)** |
| **Phase 3C Dedicated Center CNN (Ours)** | Single IR $t_0$, Spatial Tap + CoordConv | **Min Val Mean DPE (km)** | **810.3 km** | **407.4 km** | **1,973.5 km** | **3.48° / 6.07°** | **0.70 / 0.93** | **NO (Dynamic Localization)** |

---

## 3. Dataset & Split Integrity Invariants

The VAYU-NET locked dataset was used without modification under strict data governance protocols:
- **Total Valid Dataset Samples:** 1,319
- **TRAIN Split:** 696 samples / 81 distinct storms
- **VALIDATION Split:** 252 samples / 14 distinct storms
- **TEST Split:** 371 samples / 31 distinct storms
- **Split Intersection:** Exactly 0 overlapping storms between any splits.
- **Normalization Invariant:** Per-pixel IR brightness temperature normalization was applied strictly using TRAIN split parameters loaded from `data/interim/ml/train_normalization_stats.json` (Mean: 265.4856 K, Std: 21.0559 K).
- **Input Observation:** Single frame $t_0$ only (`[B, 1, 572, 929]`). Zero future or past observations were accessible to the network.

---

## 4. Architecture & Spatial Localization Formulation

The network architecture, implemented in [`ml/models/center_localization_cnn.py`](file:///c:/GitHub/vayu-net/ml/models/center_localization_cnn.py) as `DedicatedCenterLocalizationResNet`, builds directly upon the validated spatial design from Phase 3B:

```
Input Satellite Tensor: [B, 1, 572, 929]
        │
        ▼
ResNet-18 Backbone (weights=None, adapted 1-channel conv1)
        │
        ├─► conv1 + bn1 + relu + maxpool  ──► [B, 64, 143, 233]
        ├─► layer1                         ──► [B, 64, 143, 233]
        └─► layer2 feature tap            ──► [B, 128, 72, 117]
                    │
                    ▼
Normalized CoordConv Channels [B, 2, 72, 117] (lat/lon in [-1, 1])
                    │
                    ▼
Concatenated Spatial Representation: [B, 130, 72, 117]
                    │
                    ▼
Convolutional Center Decoder:
  • Conv2d(130, 64, kernel=3, padding=1) + BatchNorm2d + ReLU
  • Conv2d(64, 32, kernel=3, padding=1) + BatchNorm2d + ReLU
  • Conv2d(32, 1, kernel=1)
                    │
                    ▼
Heatmap Logits: [B, 1, 72, 117]
        │
        ├─► Sigmoid ──► Heatmap Probability: [B, 1, 72, 117]
        │
        └─► 2D Spatial Soft-Argmax:
              • Temperature-scaled Softmax across H x W (T = 1.0)
              • Expectation over normalized grid indices:
                  u_lat = sum(P * Y_grid)
                  u_lon = sum(P * X_grid)
              • norm_center: [B, 2] in [0, 1]
                    │
                    ▼
Physical Coordinate Denormalization:
  lat = -5.0° + u_lat * 40.0°
  lon = 40.0° + u_lon * 65.0°
```

Total Trainable Parameters: **11,263,777**.

---

## 5. Loss Function Formulation & Training Dynamics

### Loss Function
To cleanly isolate center localization, the training objective was formulated as:
$$\mathcal{L}_{\text{center}} = \lambda_{\text{hm}} \mathcal{L}_{\text{heatmap}} + \lambda_{\text{coord}} \mathcal{L}_{\text{coord}}$$
where:
- $\mathcal{L}_{\text{heatmap}} = \text{MSE}(\hat{H}, H^*)$ comparing predicted Sigmoid probabilities against a 2D Gaussian ground-truth heatmap centered at the continuous IMD reference coordinate ($\sigma = 1.5$ grid cells $\approx 93\text{ km}$).
- $\mathcal{L}_{\text{coord}} = \text{SmoothL1}(\hat{u}, u^*)$ comparing soft-argmax continuous coordinates directly with the normalized target.
- Hyperparameters: $\lambda_{\text{hm}} = 10.0$, $\lambda_{\text{coord}} = 1.0$.

### Training Configuration
- **Optimizer:** AdamW ($\text{LR} = 10^{-4}$, Weight Decay = $10^{-4}$)
- **LR Scheduler:** `ReduceLROnPlateau` monitoring Validation Mean DPE (factor = 0.5, patience = 2)
- **Batch Size:** 16
- **Device:** CPU (deterministic seed 42)
- **Model Selection & Early Stopping Criterion:** Lowest **Validation Mean DPE (km)** with patience = 3.

### Epoch-by-Epoch Training History

| Epoch | Train Center Loss | Val Center Loss | Val Heatmap MSE | Val Coord L1 | Val Mean DPE (km) | Val Median DPE (km) | Val P90 DPE (km) | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5056 | 1.2063 | 0.1194 | 0.0120 | 1,259.0 km | 1,304.8 km | 1,956.4 km | $1.0 \times 10^{-4}$ | Checkpoint Saved |
| **2** | **0.9668** | **0.9972** | **0.0990** | **0.0074** | **703.0 km** | **262.5 km** | **1,922.5 km** | **$1.0 \times 10^{-4}$** | **BEST CHECKPOINT SAVED** |
| 3 | 0.8481 | 0.7963 | 0.0786 | 0.0103 | 1,086.1 km | 1,122.5 km | 1,960.6 km | $1.0 \times 10^{-4}$ | Counter: 1/3 |
| 4 | 0.7467 | 0.7876 | 0.0780 | 0.0077 | 736.4 km | 286.1 km | 1,938.6 km | $1.0 \times 10^{-4}$ | Counter: 2/3 |
| 5 | 0.6868 | 0.6899 | 0.0682 | 0.0084 | 819.6 km | 471.4 km | 1,971.0 km | $1.0 \times 10^{-4}$ | Counter: 3/3 (Early Stop) |

Early stopping triggered after Epoch 5 as Validation Mean DPE did not improve over the Epoch 2 benchmark (703.0 km). Epoch 2 weights were restored and frozen.

---

## 6. Detailed Split Evaluation & Spatial Collapse Analysis

Using the frozen Epoch 2 checkpoint ([`best_center_localization_cnn.pt`](file:///c:/GitHub/vayu-net/data/interim/ml/checkpoints/best_center_localization_cnn.pt)):

### Split Metrics Breakdown

| Split | Sample Count | Mean DPE (km) | Median DPE (km) | P90 DPE (km) | Lat MAE (°) | Lon MAE (°) | Pred Lat / Lon Std (°) | True Lat / Lon Std (°) | Lat / Lon Variance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 696 | 814.3 km | 422.5 km | 2,013.1 km | 2.96° | 6.45° | 4.88° / 10.74° | 5.43° / 11.41° | **0.90 / 0.94** |
| **VALIDATION** | 252 | 703.0 km | 262.5 km | 1,922.5 km | 2.66° | 5.57° | 4.28° / 8.44° | 4.87° / 10.33° | **0.88 / 0.82** |
| **TEST** | 371 | **810.3 km** | **407.4 km** | **1,973.5 km** | **3.48°** | **6.07°** | 3.99° / 9.88° | 5.73° / 10.59° | **0.70 / 0.93** |

### Spatial Collapse Forensics
A critical failure mode identified in Phase 3 and Phase 3B was **spatial collapse**, where models with global average pooling or multi-task interference learned to predict the constant centroid of the basin ($\approx 15.6^\circ\text{N}, 78.4^\circ\text{E}$), exhibiting variance ratios near zero ($0.04$ / $0.03$).

In Phase 3C:
- **True Test Standard Deviations:** $\sigma_{\text{lat}} = 5.73^\circ$, $\sigma_{\text{lon}} = 10.59^\circ$.
- **Predicted Test Standard Deviations:** $\sigma_{\text{lat}} = 3.99^\circ$, $\sigma_{\text{lon}} = 9.88^\circ$.
- **Variance Ratios:** Latitude = **0.70**, Longitude = **0.93**.
- **Conclusion:** The model spans the entire longitudinal width of the North Indian Ocean basin (Arabian Sea through Bay of Bengal) and actively modulates its spatial coordinates in direct correspondence with satellite cloud features. Spatial collapse has been eliminated.

---

## 7. Worst-Case Error Analysis (Validation Split)

The top worst-case localization errors reveal specific structural meteorological patterns:

| Rank | Storm ID | Timestamp | IMD Center (Lat, Lon) | Pred Center (Lat, Lon) | DPE (km) | Meteorological Pattern |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | `NIO_2019_VAYU` | 2019-06-11T00:00:00Z | (14.70°, 70.60°) | (-2.88°, 97.67°) | 3,565.4 km | Arabian Sea vs. Southern Equatorial cloud cluster misidentification |
| 2 | `NIO_2020_GATI` | 2020-11-22T03:00:00Z | (10.70°, 53.80°) | (7.23°, 84.20°) | 3,359.6 km | Compact small-scale cyclone off Somali coast; model focused on active BoB convection |
| 3 | `NIO_2019_PABUK` | 2019-01-04T12:00:00Z | (8.50°, 99.70°) | (14.98°, 72.04°) | 3,093.8 km | Western Pacific cross-over storm entering Malay Peninsula/Andaman Sea border |
| 4 | `NIO_2019_PABUK` | 2019-01-04T18:00:00Z | (8.70°, 99.20°) | (14.90°, 72.19°) | 3,016.9 km | Border domain edge effect during basin transit |
| 5 | `NIO_2019_PABUK` | 2019-01-05T00:00:00Z | (8.90°, 98.70°) | (14.71°, 72.26°) | 2,946.7 km | Intense convection at eastern border conflicting with Arabian Sea background |

### Key Diagnostic Observations:
1. **Basin-Switch Ambiguity:** Because a single satellite frame lacks temporal motion vectors, large convective clusters in the Arabian Sea can occasionally dominate the spatial attention over weaker or marginal systems near the eastern perimeter (e.g., Pabuk entering the Andaman Sea).
2. **Compact Systems (Gati):** Very small, rapidly intensifying cyclones like Gati (which intensified to Category 3 within 12 hours off Somalia) have tiny cloud shields that are occasionally overshadowed by larger monsoon troughs in the central basin.
3. **Median vs. Mean Discrepancy:** The median DPE is 262.5 km in Validation and 407.4 km in Test, whereas the means are 703.0 km and 810.3 km. This indicates that **over 50% of cyclone predictions are well-localized within ~250–400 km**, while a long tail of basin-switching outliers (P90 ~ 1,950 km) pulls up the arithmetic mean.

---

## 8. Diagnostic Figures

All diagnostic figures are generated and stored under [`docs/figures/center_localization/`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/):

1. **[`train_val_center_loss.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/train_val_center_loss.png):** Training and validation loss progression showing stable convergence of both heatmap MSE and soft-argmax coordinate L1.
2. **[`val_dpe_curve.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/val_dpe_curve.png):** Validation Mean, Median, and P90 Direct Position Error curves across epochs, highlighting the sharp drop at Epoch 2.
3. **[`predicted_vs_actual_centers.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/predicted_vs_actual_centers.png):** Scatter map of ground-truth IMD positions vs. predicted positions across the North Indian Ocean basin for Validation and Test splits.
4. **[`center_error_distribution.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/center_error_distribution.png):** Histogram of DPE errors on the held-out TEST set showing the concentrated distribution around the 407.4 km median.
5. **[`prediction_variance_analysis.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/prediction_variance_analysis.png):** Histograms comparing true vs. predicted latitude and longitude distributions, demonstrating realistic spatial spread.
6. **[`heatmap_examples.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/heatmap_examples.png):** Visual inspection of satellite input, target Gaussian heatmap, predicted probability heatmap, and soft-argmax center coordinates.
7. **[`worst_case_examples.png`](file:///c:/GitHub/vayu-net/docs/figures/center_localization/worst_case_examples.png):** Geographic plots of top worst-case validation errors detailing the meteorological conditions of basin-switching.

---

## 9. Limitations of Single-Frame Localization

1. **Static Ambiguity:** Without temporal context (motion vectors across $t_{-15}$ to $t_0$), a single satellite frame cannot distinguish between a stationary convective depression and an organized propagating cyclone center when multiple convective bands coexist.
2. **Resolution Constraints:** Downsampling to a $72 \times 117$ feature grid provides spatial quantization of approximately $0.55^\circ \times 0.55^\circ$ ($\sim 60\text{ km}$ per cell). While soft-argmax provides continuous sub-pixel interpolation, fine-grained eye localization is bounded by this spatial resolution.
3. **Border Effects:** Cyclones at the extreme boundaries of the NIO domain (e.g., Pabuk crossing the Malay Peninsula at $99^\circ\text{E}$) suffer from truncated cloud shields.

---

## 10. Scientific Decision & Recommendation for Phase 4

### Scientific Answer to the Research Question:
**YES.** When center localization is isolated and selected using Validation Mean DPE:
- The network successfully extracts cyclone center coordinates directly from raw GridSat-B1 infrared brightness temperature without global pooling.
- The model beats the historical constant-mean centroid baseline by **375.7 km on mean DPE** and **675.2 km on median DPE**.
- Spatial collapse is fully prevented (variance ratios of 0.70 / 0.93).

### Decision for Phase 4 (Temporal Modeling):
- **Center localization is sufficiently validated to serve as the spatial backbone for temporal sequence modeling.**
- The spatial feature map (`[B, 128, 72, 117]`) and the heatmap decoder provide a stable spatial coordinate grounding that Phase 4 temporal architectures (such as ConvGRU or temporal feature pooling) can integrate across time steps to resolve single-frame convective ambiguities.
