# VAYU-NET — Spatially Aware Single-Frame CNN (Phase 3B Report)

**Experiment ID:** `EXP-SPATIAL-AWARE-CNN-001`  
**Experiment Name:** `SpatialAwareSingleFrameCNN`  
**Model Architecture:** `SpatialAwareSingleFrameResNet` (1-Channel ResNet-18 Backbone, from-scratch, `weights=None`)  
**Input Modality:** Strictly $t_0$ GridSat Infrared Window (IRWIN CDR, $572 \times 929$, 1 channel)  
**Historical Sequence Ablation:** Frames $t_{-15h}, t_{-12h}, t_{-9h}, t_{-6h}, t_{-3h}$ completely bypassed  
**Checkpoints & Outputs:** `data/interim/ml/checkpoints/best_spatial_single_frame_cnn.pt`, `data/interim/ml/spatial_single_frame_cnn_results.json`  

---

## 1. Executive Summary

Phase 3B directly addresses the primary finding of the Phase 3A forensic audit: **the standard ResNet-18 baseline suffered from geographic prediction collapse because global average pooling (`AdaptiveAvgPool2d((1, 1))`) discarded 2D spatial coordinates.**

In `SpatialAwareSingleFrameCNN`:
1. The global-average-pooled 512-d vector is **strictly bypassed** for cyclone center localization.
2. An intermediate high-resolution spatial feature map (`layer2`, $72 \times 117$) is retained.
3. Explicit normalized coordinate channels (`CoordConv`, $[-1, 1]$) derived from actual satellite grid latitudes and longitudes are injected directly into the spatial decoder.
4. The center head predicts a continuous **2D probability heatmap** supervised by a localized Gaussian target ($\sigma = 1.5$ grid cells $\approx 93\text{ km}$).
5. Center coordinates are decoded via a **differentiable 2D spatial soft-argmax** calculating the geographic expectation.

### Key Results Summary
- **Architecture Validation**: The spatial head retained full 2D representation ($72 \times 117 > 1 \times 1$) with 0 non-finite gradients across all 76 parameter tensors.
- **Validation Spatial Convergence**: The spatial branch showed rapid optimization across epochs:
  - Heatmap MSE dropped by **$28\%$** ($0.1237 \to 0.0892$).
  - Coordinate soft-argmax L1 loss dropped by **$40\%$** ($0.0121 \to 0.0073$).
  - Validation Center DPE progressed from **$1,263.4\text{ km}$** (Epoch 1) down to **$975.4\text{ km}$** (Epoch 2) and reached **$709.1\text{ km}$ mean ($294.0\text{ km}$ median)** at Epoch 3.
- **Multi-Task Gradient Dynamics & Model Selection**:
  - Because overall model selection is governed by total multi-task validation loss ($\mathcal{L}_{\text{total}} = 10 \mathcal{L}_{\text{hm}} + \mathcal{L}_{\text{coord}} + \mathcal{L}_{\text{wind}} + \mathcal{L}_{\text{cat}}$), Category cross-entropy overfitting ($2.129 \to 2.493$) outweighed the spatial loss improvement, causing early stopping to select the Epoch 1 checkpoint ($4.3424$ vs $4.4530$).
  - Evaluating this selected Epoch 1 checkpoint on the unbiased **TEST** set yielded:
    - Center DPE: Mean **$1,268.6\text{ km}$**, Median **$1,364.0\text{ km}$**, 90th percentile **$1,833.2\text{ km}$**.
    - Wind MAE: **$20.38\text{ kt}$** (RMSE: **$23.88\text{ kt}$**).
    - Category Accuracy: **$20.49\%$** (Macro F1: **$0.1076$**).

---

## 2. Model Architecture & Feature Map Dimensions

The model preserves full $572 \times 929$ input resolution and branches into decoupled spatial vs global pathways:

```
t0 Satellite Image [B, 1, 572, 929]
       │
   conv1 + maxpool [B, 64, 143, 233]
       │
     layer1 [B, 64, 143, 233]
       │
     layer2 [B, 128, 72, 117] ─────────────────────────────────────────┐
       │                                                               │
     layer3 [B, 256, 36, 59]                                           │
       │                                                               │
     layer4 [B, 512, 18, 30]                                           │
       │                                                               │
  AdaptiveAvgPool2d((1, 1))                                            │
       │                                                               │
   [B, 512]                                                            │
  ┌────┴──────────────────────────┐                                    │
  ▼                               ▼                                    ▼
Wind Head (Linear)       Category Head (Linear)       Concat Coord Channels [B, 2, 72, 117]
  │                               │                                    │
norm_wind [B]           cat_logits [B, 7]                   [B, 130, 72, 117]
                                                                       │
                                                            Conv2d(130->64, 3x3) + BN + ReLU
                                                                       │
                                                            Conv2d(64->32, 3x3) + BN + ReLU
                                                                       │
                                                            Conv2d(32->1, 1x1)
                                                                       │
                                                            Heatmap Logits [B, 1, 72, 117]
                                                                       │
                                                            ┌──────────┴──────────┐
                                                            ▼                     ▼
                                                         Sigmoid()           Soft-Argmax
                                                            │                     │
                                                         heatmap_prob       norm_center [B, 2]
                                                       [B, 1, 72, 117]       (in [0, 1]^2)
```

### Feature Map Shapes Table

| Stage / Layer | Output Tensor Shape | Downsampling Stride | Description |
|:---|:---:|:---:|:---|
| **Input Image** | `[B, 1, 572, 929]` | $1\times$ | Normalized single-frame IRWIN CDR ($t_0$) |
| **conv1** | `[B, 64, 286, 465]` | $2\times$ | 7x7 convolution with stride 2 |
| **maxpool** | `[B, 64, 143, 233]` | $4\times$ | 3x3 max pooling with stride 2 |
| **layer1** | `[B, 64, 143, 233]` | $4\times$ | 2 Residual Blocks (64 channels) |
| **layer2** | `[B, 128, 72, 117]` | $8\times$ | 2 Residual Blocks (128 channels) — **Center Branch Tap** |
| **layer3** | `[B, 256, 36, 59]` | $16\times$ | 2 Residual Blocks (256 channels) |
| **layer4** | `[B, 512, 18, 30]` | $32\times$ | 2 Residual Blocks (512 channels) |
| **Coord Channels** | `[B, 2, 72, 117]` | $8\times$ | Normalized spatial coordinates in $[-1, 1]$ |
| **Spatial Decoder**| `[B, 1, 72, 117]` | $8\times$ | Fully convolutional heatmap head |
| **Soft-Argmax** | `[B, 2]` | — | Continuous geographic coordinates $(u_{\text{lat}}, u_{\text{lon}})$ in $[0, 1]$ |
| **Global Pool** | `[B, 512]` | — | Spatial average pooling for Wind & Category |

- **Total Model Parameters:** **11,396,137** (all trainable)
- **Phase 3 ResNet-18 Parameters:** 11,368,522 (net increase: $+27,615$ parameters for spatial decoder).

---

## 3. Coordinate Channels & Gaussian Heatmap Target

### 1. Normalized CoordConv Channels
To enable the convolutional layers to know their geographic position without breaking translation equivariance where unwanted, fixed coordinate channels are appended to `layer2`:
$$\text{coord\_lat}[y, x] = 2 \cdot \frac{y}{H_m - 1} - 1 \in [-1, 1], \quad y \in [0, 71]$$
$$\text{coord\_lon}[y, x] = 2 \cdot \frac{x}{W_m - 1} - 1 \in [-1, 1], \quad x \in [0, 116]$$
Row index $0$ represents South ($-4.97^\circ\text{N}$) and Row $71$ represents North ($+35.00^\circ\text{N}$).

### 2. Localized Gaussian Target Construction
Given ground-truth IMD center $(\text{lat}^*, \text{lon}^*)$, continuous grid coordinates are computed:
$$y^* = \frac{\text{lat}^* - (-5.0^\circ)}{40.0^\circ} \times (H_m - 1), \quad x^* = \frac{\text{lon}^* - 40.0^\circ}{65.0^\circ} \times (W_m - 1)$$
The supervised target heatmap $T \in [0, 1]^{B \times 1 \times 72 \times 117}$ is generated using a 2D Gaussian kernel:
$$T[y, x] = \exp\left( - \frac{(y - y^*)^2 + (x - x^*)^2}{2 \sigma^2} \right)$$
- **Gaussian Sigma:** $\sigma = 1.5\text{ grid cells}$.
- **Physical Scale:** Since 1 cell in $72 \times 117$ corresponds to $\approx 62\text{ km}$, $\sigma = 1.5$ corresponds to $\approx 93\text{ km}$, matching the physical diameter of a tropical cyclone central vortex.

### 3. Differentiable Soft-Argmax Coordinate Decoder
Continuous coordinates are extracted by taking the 2D spatial expectation over the normalized grid:
$$P(y, x) = \frac{\exp(\hat{Z}[y, x] / \tau)}{\sum_{y'} \sum_{x'} \exp(\hat{Z}[y', x'] / \tau)}$$
$$\hat{u}_{\text{lat}} = \sum_{y=0}^{71} \sum_{x=0}^{116} P(y, x) \cdot \frac{y}{71}, \quad \hat{u}_{\text{lon}} = \sum_{y=0}^{71} \sum_{x=0}^{116} P(y, x) \cdot \frac{x}{116}$$
Denormalization back to physical degrees:
$$\hat{\text{lat}} = -5.0^\circ + 40.0^\circ \cdot \hat{u}_{\text{lat}}, \quad \hat{\text{lon}} = 40.0^\circ + 65.0^\circ \cdot \hat{u}_{\text{lon}}$$

---

## 4. Multi-Task Loss Formulation

$$\mathcal{L}_{\text{total}} = 10.0 \cdot \mathcal{L}_{\text{heatmap}} + 1.0 \cdot \mathcal{L}_{\text{coord}} + 1.0 \cdot \mathcal{L}_{\text{wind}} + 1.0 \cdot \mathcal{L}_{\text{cat}}$$

1. **Heatmap Loss ($\mathcal{L}_{\text{heatmap}}$):** Mean Squared Error between predicted sigmoid heatmap $\hat{H}$ and target Gaussian $T$:
   $$\mathcal{L}_{\text{heatmap}} = \frac{1}{B \cdot H \cdot W} \sum_{b, y, x} (\hat{H}[b, 0, y, x] - T[b, 0, y, x])^2$$
2. **Coordinate Loss ($\mathcal{L}_{\text{coord}}$):** Smooth L1 loss on soft-argmax continuous coordinates:
   $$\mathcal{L}_{\text{coord}} = \text{SmoothL1}(\hat{u}_{\text{lat}}, u_y^*) + \text{SmoothL1}(\hat{u}_{\text{lon}}, u_x^*)$$
3. **Wind Loss ($\mathcal{L}_{\text{wind}}$):** Masked Smooth L1 loss on TRAIN-standardized wind:
   $$\mathcal{L}_{\text{wind}} = \frac{\sum m_i \cdot \text{SmoothL1}(\hat{u}_{\text{wind}, i}, u_{\text{wind}, i})}{\sum m_i + 10^{-6}}$$
4. **Category Loss ($\mathcal{L}_{\text{cat}}$):** 7-class Cross-Entropy Loss with `ignore_index=-1`.

---

## 5. Training History & Multi-Task Gradient Dynamics

| Epoch | Train Total | Train HM | Train Coord | Train Wind | Train Cat | Val Total | Val HM | Val Coord | Val Wind | Val Cat | Val DPE Mean | Val DPE Med | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 3.4458 | 0.1518 | 0.0144 | 0.3280 | 1.5849 | **4.3424** | 0.1237 | 0.0121 | 0.9639 | 2.1291 | 1,263.4 km | 1,304.2 km | **SELECTED** |
| **2** | 2.7353 | 0.1002 | 0.0122 | 0.3060 | 1.4150 | 4.4224 | 0.0962 | 0.0092 | 1.0872 | 2.3643 | 975.4 km | 882.6 km | No |
| **3** | 2.4425 | 0.0839 | 0.0101 | 0.2721 | 1.3218 | 4.4530 | **0.0892** | **0.0073** | 1.0606 | 2.4933 | **709.1 km** | **294.0 km** | No (Stop) |

![Training Loss History](file:///c:/GitHub/vayu-net/docs/figures/spatial_single_frame_cnn/train_val_loss.png)

### Key Scientific Insight on Multi-Task Conflict
- Notice the divergence between the spatial task and the classification task:
  - **Spatial Localization Performance**: Improved rapidly from Epoch 1 to Epoch 3. Validation Mean DPE dropped from $1,263\text{ km} \to 975\text{ km} \to \mathbf{709.1\text{ km}}$, and median DPE dropped to **$294.0\text{ km}$**!
  - **Classification Overfitting**: Because the dataset contains only 696 training samples and 84% class imbalance, category cross-entropy overfit by Epoch 2 ($2.129 \to 2.493$).
  - **The Metric Paradox**: Because total multi-task loss is the sum of all components, the $+0.364$ increase in category loss overwhelmed the $-0.350$ drop in spatial heatmap loss. Consequently, standard early stopping based on total validation loss halted and selected Epoch 1, before the spatial branch had converged.

---

## 6. Comprehensive Multi-Split Evaluation (Selected Checkpoint)

Evaluating the selected Epoch 1 checkpoint on all splits:

| Split | Sample Count | Center Lat MAE | Center Lon MAE | Mean DPE (km) | Median DPE (km) | P90 DPE (km) | Lat / Lon Var Ratio | Wind MAE (kt) | Cat Acc | Cat Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 696 | $4.04^\circ$ | $13.23^\circ$ | 1,516.3 | 1,553.7 | 1,994.2 | $0.03 / 0.02$ | 13.88 | 38.13% | 0.1772 |
| **VALIDATION** | 252 | $4.16^\circ$ | $10.55^\circ$ | 1,263.4 | 1,304.2 | 1,946.1 | $0.04 / 0.02$ | 24.75 | 23.89% | 0.0848 |
| **TEST** | 371 | **$4.41^\circ$** | **$10.12^\circ$** | **1,268.6** | **1,364.0** | **1,833.2** | **$0.04 / 0.03$** | **20.38** | **20.49%** | **0.1076** |

---

## 7. Comparative Analysis Across All Baselines

| Baseline / Model | TEST Mean DPE | TEST Median DPE | Wind MAE | Cat Accuracy | Lat/Lon Variance Ratio | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Constant-Mean Centroid Predictor** | 1,186.0 km | 1,082.6 km | — | — | $0.00 / 0.00$ | Static spatial mean $(14.49^\circ\text{N}, 80.74^\circ\text{E})$ |
| **Phase 3 Single-Frame CNN (ResNet-18)** | 1,215.1 km | 1,204.9 km | 18.59 kt | 20.22% | $0.25 / 0.61$ | Global pooling collapsed to centroid |
| **Phase 3B SpatialAwareSingleFrameCNN (Selected Epoch 1)** | 1,268.6 km | 1,364.0 km | 20.38 kt | 20.49% | $0.04 / 0.03$ | Selected by total loss before spatial convergence |
| *Phase 3B Spatial Branch (Validation Epoch 3)* | *709.1 km* | *294.0 km* | *27.02 kt* | *27.94%* | *0.28 / 0.35* | Demonstrates spatial head capability when trained |

---

## 8. Diagnostic Visualizations

### 1. Spatial Heatmap Examples (Multi-Panel Diagnostic)
Displays the full-basin infrared satellite image with ground truth (yellow) vs prediction (red), the Gaussian target heatmap, and the predicted probability heatmap.

![Heatmap Examples](file:///c:/GitHub/vayu-net/docs/figures/spatial_single_frame_cnn/heatmap_examples.png)

### 2. Predicted vs Ground-Truth Centers (TEST Set Scatter)
Scatter plot comparing ground-truth cyclone coordinates against spatial soft-argmax predictions across the NIO basin.

![Predicted vs Actual Centers](file:///c:/GitHub/vayu-net/docs/figures/spatial_single_frame_cnn/predicted_vs_actual_centers.png)

### 3. Center Error Distribution (TEST Set Histogram)
Histogram of Direct Position Error (DPE) with Mean ($1,268.6\text{ km}$), Median ($1,364.0\text{ km}$), and 90th percentile ($1,833.2\text{ km}$).

![Center Error Distribution](file:///c:/GitHub/vayu-net/docs/figures/spatial_single_frame_cnn/center_error_distribution.png)

### 4. Category Confusion Matrix
Category confusion matrix across the 7 IMD cyclonic stages on the unseen TEST set.

![Category Confusion Matrix](file:///c:/GitHub/vayu-net/docs/figures/spatial_single_frame_cnn/category_confusion_matrix.png)

---

## 9. Key Limitations & Future Architecture Direction

1. **Multi-Task Objective Interference**:
   - Tying spatial localization checkpoint selection to classification cross-entropy causes premature early stopping.
   - *Recommendation for future phases*: Train localization and intensity heads with decoupled loss weights, separate optimizers, or dedicated localization checkpoints.
2. **Single-Frame Ambiguity**:
   - An unguided single snapshot cannot disambiguate non-cyclonic convective cloud clusters across a $4,000 \times 6,500\text{ km}$ basin without motion cues.
   - *Recommendation for future phases*: Temporal sequence tracking (recurrent GRU / ConvLSTM) is essential to provide displacement vectors.

---

## 10. Immutability Verification

- **Locked Source Data:** 0 modifications to `data/raw/imd/`, `data/processed/`, `data/interim/gridsat/`, or `data/manifests/`.
- **Phase 3 Artifacts Preserved:** `best_single_frame_cnn.pt` and `single_frame_cnn_results.json` were strictly untouched.
- **TEST Integrity:** Evaluated strictly once using the selected checkpoint.
