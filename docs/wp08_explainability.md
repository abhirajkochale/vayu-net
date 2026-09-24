# VAYU-NET WP-08 — Grad-CAM Saliency Explainability

## 1. Executive Summary

Work Package 08 (`WP-08`) implements visual model interpretability for VAYU-NET's operational cyclone intelligence pipeline. Adhering to the locked SIH 2026 Problem Statement 26070 and Blueprint requirements, this module applies Gradient-Weighted Class Activation Mapping (Grad-CAM) to the terminal observation frame ($t_0$) of the 6-frame satellite sequence feeding the multi-task intensity classification head.

> [!CAUTION]
> ### MANDATORY SCIENTIFIC INTERPRETATION NOTICE
> **The saliency/Grad-CAM output is an interpretation aid and must NOT be described or presented as a causal meteorological explanation.**
> Attributions reflect the spatial features within the convolutional representation that contributed most strongly to the model's categorical classification score. They do not imply that individual cloud bands caused cyclone intensification or cyclogenesis.

---

## 2. Theoretical Formulation & Architecture

### 2.1 Model Representation & Tap Layer
The VAYU-NET architecture utilizes a shared convolutional backbone derived from the dedicated center localization model (`best_center_localization_cnn.pt`), transferred into the temporal track prediction network (`best_temporal_track_gru.pt`), and fused with ERA5 environmental reanalysis in the multi-task intensity classifier (`best_phase6_intensity_wind.pt`).

| Property | Value / Description |
| :--- | :--- |
| **Model Checkpoint** | `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt` |
| **Spatial Backbone** | 1-channel ResNet-18 (Phase 3C Dedicated Architecture) |
| **Target Layer** | `spatial_encoder.layer2` |
| **Feature Map Tensor** | $[B, 128, 72, 117]$ (72 rows $\times$ 117 cols, 128 channels) |
| **Input Satellite Frame** | Single-band GridSat-B1 IRWIN CDR ($572 \times 929$) |
| **Observation Slot** | Terminal step $t_0$ (Frame index 5 of sequence $[t_{-15\text{h}}, \dots, t_0]$) |
| **Target Classification** | IMD 7-Class Intensity Logits: $\mathbf{y} \in \mathbb{R}^7$ |

### 2.2 Mathematical Derivation
Let $A_{i,j}^k$ denote the activation of feature channel $k \in \{1, \dots, 128\}$ at spatial location $(i, j)$ in `layer2`, where $i \in \{1, \dots, 72\}$ and $j \in \{1, \dots, 117\}$. Let $y^c$ denote the unnormalized classification logit for the predicted class $c = \arg\max_k y^k$.

1. **Gradient Computation**:
   $$\frac{\partial y^c}{\partial A^k_{i,j}}$$
   computed via automatic differentiation backpropagated through the spatial pooling, projection, GRU sequence encoder, and classification head.

2. **Global Average Pooling Weights**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{72} \sum_{j=1}^{117} \frac{\partial y^c}{\partial A^k_{i,j}}$$
   where $Z = 72 \times 117 = 8,424$.

3. **Rectified Linear Activation Map**:
   $$L_{\text{Grad-CAM}}^c(i, j) = \text{ReLU}\left( \sum_{k=1}^{128} \alpha_k^c A^k_{i, j} \right)$$
   The $\text{ReLU}$ non-linearity strictly captures features that positively correlate with the predicted cyclonic category.

4. **Bilinear Spatial Upsampling & Normalization**:
   $$M^c(u, v) = \text{BilinearUpsample}\left( L_{\text{Grad-CAM}}^c \right) \in \mathbb{R}^{572 \times 929}$$
   $$M_{\text{norm}}^c(u, v) = \frac{M^c(u, v) - \min(M^c)}{\max(M^c) - \min(M^c)}$$

---

## 3. Spatial Alignment & Preprocessing

The spatial pipeline preserves 100% geometric alignment with the North Indian Ocean basin ($40^\circ\text{E} - 105^\circ\text{E}$, $-5^\circ\text{N} - 35^\circ\text{N}$):

```
Original GridSat-B1 IRWIN CDR Frame [572, 929]
                  ↓
Standardization: z = (x - 279.37) / 22.79 (TRAIN-only stats)
                  ↓
ResNet-18 Backbone: conv1 → bn1 → relu → maxpool → layer1 → layer2
                  ↓
Hooked Feature Map: [1, 128, 72, 117]
                  ↓
Grad-CAM Weighting: α_k = GAP(∂y^c / ∂A^k)
                  ↓
Coarse Saliency Map: [72, 117]
                  ↓
Bilinear Upsampling to [572, 929] & Min-Max Normalization to [0, 1]
                  ↓
Alpha-Blended False-Color Overlay (Turbo Colormap, α = 0.45) on Grayscale Satellite Image
```

---

## 4. Shortlist Reference Cases & Generated Artifacts

Explanations have been generated and validated for the five locked reference cyclones across both basins:

| Cyclone | Year | Basin | Ground Truth | Predicted Class | Confidence | Max Activation Loc $(r, c)$ | Centroid $(\bar{r}, \bar{c})$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FANI** | 2019 | Bay of Bengal | ESCS (115 kt) | DD | 43.22% | $(91, 853)$ | $(340.8, 538.9)$ |
| **AMPHAN** | 2020 | Bay of Bengal | SuCS (120 kt) | VSCS | 27.51% | $(242, 647)$ | $(235.3, 605.9)$ |
| **TAUKTAE** | 2021 | Arabian Sea | ESCS (100 kt) | D | 43.41% | $(218, 488)$ | $(283.0, 449.2)$ |
| **BIPARJOY** | 2023 | Arabian Sea | ESCS (90 kt) | DD | 24.64% | $(321, 392)$ | $(289.8, 419.7)$ |
| **REMAL** | 2024 | Bay of Bengal | SCS (50 kt) | DD | 41.19% | $(194, 520)$ | $(250.6, 544.1)$ |

### Artifact Registry
All explanation artifacts are stored under `data/interim/ml/explainability/`:
- **Original Grayscale**: `original_{storm}_{timestamp}.png`
- **Normalized Heatmap**: `heatmap_{storm}_{timestamp}.png`
- **Blended Overlay**: `overlay_{storm}_{timestamp}.png`
- **Audit Metadata**: `metadata_{storm}_{timestamp}.json`
- **Summary Index**: `explainability_manifest.json`

---

## 5. API-Ready Structured Payload

The explainability engine outputs an audit-compliant JSON dictionary designed for direct ingestion by the subsequent backend service:

```json
{
  "sample_id": "NIO_2020_AMPHAN_20200518_0600Z",
  "storm_id": "NIO_2020_AMPHAN",
  "storm_name": "AMPHAN",
  "t0": "2020-05-18T06:00:00+00:00",
  "source": "GridSat-B1 IRWIN CDR",
  "split": "VALIDATION",
  "frame_index": 5,
  "ground_truth_category": "SuCS",
  "ground_truth_wind_kt": 120.0,
  "predicted_class": "VSCS",
  "predicted_class_idx": 4,
  "confidence": 0.2751,
  "method": "Grad-CAM",
  "target_layer": "spatial_encoder.layer2",
  "image_height": 572,
  "image_width": 929,
  "max_activation_loc": [242, 647],
  "activation_centroid": [235.3, 605.9],
  "heatmap_path": "data/interim/ml/explainability/heatmap_amphan_20200518T060000+0000.png",
  "overlay_path": "data/interim/ml/explainability/overlay_amphan_20200518T060000+0000.png",
  "original_path": "data/interim/ml/explainability/original_amphan_20200518T060000+0000.png",
  "model_checkpoint": "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt",
  "spatial_checkpoint": "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
  "temporal_checkpoint": "data/interim/ml/checkpoints/best_temporal_track_gru.pt",
  "dataset_version": "v1.0-locked",
  "preprocessing_version": "TRAIN_zscore_standardization",
  "interpretation_note": "Interpretation aid showing regions that contributed most strongly to the model's IMD intensity classification. Must not be presented or interpreted as a causal meteorological explanation."
}
```

---

## 6. Scientific Limitations

1. **Resolution Granularity**: The convolutional feature map at `layer2` has a spatial resolution of $72 \times 117$, corresponding to an effective receptive field of ~8 pixels (~64 km) per feature cell. Fine-scale eye-wall structures are smoothed during bilinear interpolation.
2. **Post-Hoc Attribution**: Grad-CAM is a gradient-weighted attribution method reflecting first-order local sensitivity of the model; it does not account for complex non-linear feature interactions within the sequence GRU or fusion layers.
3. **Non-Causality**: Saliency does not prove physical causation; cloud regions highlighting high attribution indicate predictive reliance by the network, not necessarily dynamical forcing mechanisms.
