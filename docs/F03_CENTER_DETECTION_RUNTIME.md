# VAYU-NET — F03 Center Detection Runtime Integration Specification

## 1. Executive Summary & Objective

**Requirement F03:** *"Center detection: Predict center latitude/longitude."*

This document formalizes the production runtime execution of the dedicated cyclone center localization model (**Phase 3C**), establishing how the AI-estimated center is generated, validated, and exposed via the VAYU-NET API, while preserving the validated forecaster-in-the-loop hybrid architecture of **Phase 5B Variant A**.

### Valid Scientific Description
> **"AI center localization combined with an observed-anchor hybrid track forecast."**
> 
> *Do NOT claim "fully autonomous cyclone tracking", "satellite-only end-to-end forecasting", or "189.71 km autonomous performance".*

---

## 2. Model & Checkpoint Architecture

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | `DedicatedCenterLocalizationResNet` |
| **Source Implementation** | [`ml/models/center_localization_cnn.py`](file:///c:/GitHub/vayu-net/ml/models/center_localization_cnn.py) |
| **Runtime Service** | [`ml/inference/model_service.py`](file:///c:/GitHub/vayu-net/ml/inference/model_service.py) via `ModelService.predict_center()` |
| **Checkpoint Path** | `data/interim/ml/checkpoints/best_center_localization_cnn.pt` |
| **Checkpoint Identity** | `best_center_localization_cnn.pt` |
| **Model Version** | `phase3c-v1.0-center-localization` |
| **File Size** | 51,406,686 bytes (~49.02 MB) |
| **Cryptographic SHA-256** | `5ef11b596ef94ece71079c1329eab1e50191127eb7b9b1ff13f7dcdd3185f1c9` |
| **Backbone** | 1-channel ResNet-18 (trained from scratch, weights=None) |
| **Spatial Feature Tap** | `spatial_encoder.layer2` feature map $[B, 128, 72, 117]$ |
| **Coordinate Conditioning** | 2D normalized CoordConv channels $[B, 2, 72, 117]$ in $[-1, 1]$ |
| **Heatmap Decoder Head** | Fully convolutional decoder $[B, 130, 72, 117] \to [B, 1, 72, 117]$ |
| **Decoding Operator** | Differentiable spatial soft-argmax for continuous geographic expectation $[B, 2]$ |
| **Diagnostic Peak** | Discrete 2D argmax $[B, 2]$ |

---

## 3. Data Flow & Input Provenance

### Input Specifications
- **Input Tensor:** $x \in \mathbb{R}^{B \times 1 \times 572 \times 929}$ (single-frame GridSat-B1 IRWIN CDR infrared brightness temperature at $t_0$).
- **Source Origin:** Raw NetCDF/NPZ observation identified by `frame_t0` in [`data/manifests/vayu_net_sample_index.csv`](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_sample_index.csv).
- **Preprocessing & Normalization:**
  1. Missing/corrupted pixels ($<100$ K, $>380$ K, or NaN) are imputed with training baseline mean: `mean_kelvin = 279.367674 K`.
  2. Z-score standardization strictly using TRAIN partition normalization parameters from [`data/interim/ml/train_normalization_stats.json`](file:///c:/GitHub/vayu-net/data/interim/ml/train_normalization_stats.json):
     $$x_{\text{norm}} = \frac{x - 279.367674}{22.790485}$$
- **Zero Ground-Truth Contamination:**
  The center model strictly receives satellite pixel intensities. It does **NOT** read, ingest, or condition on:
  - `imd_lat_t0`, `imd_lon_t0`
  - `imd_lat_12h`, `imd_lon_12h`
  - `imd_lat_24h`, `imd_lon_24h`
  - `imd_lat_48h`, `imd_lon_48h`
  The output center is 100% neural network forward pass output.

---

## 4. Architectural Separation: Center Detection vs Track Forecast

```
[ Raw Satellite Frame at t0 (572x929) ]
                │
                ▼
┌────────────────────────────────────────┐
│  Phase 3C Localization CNN             │
│  best_center_localization_cnn.pt       │
└────────────────────────────────────────┘
                │
                ▼
       [ ai_detected_center ]  ─────────► Exposed as F03 Center Detection
        (e.g., 9.72°N, 85.33°E)            (center_detection_source: MODEL_INFERENCE)


[ Observed IMD t0 Center ] (e.g., 11.50°N, 86.00°E)
        + Observed Prior Motion
        + Frozen Satellite Features [6, 132]
        + ERA5 Environmental Wind [6, 8, 41, 66]
                │
                ▼
┌────────────────────────────────────────┐
│  Phase 5B Variant A Hybrid Residual    │
│  best_phase5b_variant_a.pt             │
└────────────────────────────────────────┘
                │
                ▼
   [ +12h / +24h / +48h Forecast ] ─────► Forecast initialized from:
                                           forecast_anchor_type: OBSERVED_IMD_T0
```

### Critical Design Decision
1. **F03 Center Detection Output:** Independent AI prediction representing satellite computer vision estimation of cyclone eye/circulation center.
2. **Phase 5B Variant A Track Forecast:** Retains official observed IMD synoptic analysis at $t_0$ as its kinematic anchor. This preserves the operational forecaster-in-the-loop paradigm and the verified 189.71 km test DPE benchmark.
3. **No Cross-Contamination:** The AI-detected center is **NOT** fed into Variant A as an anchor (which would degrade test DPE to 1048.57 km as seen in Variant B). Variant A and Phase 3C operate in parallel.

---

## 5. API Response Contract

Endpoints returning forecast payloads (`GET /api/cyclones/{id}/forecast` and `POST /api/predict`) explicitly return:

```json
{
  "observed_reference_center": {
    "latitude": 11.5,
    "longitude": 86.0
  },
  "ai_detected_center": {
    "latitude": 9.72,
    "longitude": 85.33
  },
  "center_detection_source": "MODEL_INFERENCE",
  "forecast_anchor_type": "OBSERVED_IMD_T0",
  "prediction_source": "MODEL_INFERENCE"
}
```

### Semantic Provenance Definitions
- `observed_reference_center`: Official IMD synoptic best-track analysis position at $t_0$. **Never labeled as AI-predicted.**
- `ai_detected_center`: Pure neural inference output from `best_center_localization_cnn.pt` on single-frame GridSat IR.
- `center_detection_source`: Provenance tag: `"MODEL_INFERENCE"` (live PyTorch forward pass) or `"PRECOMPUTED_DEMO"`.
- `forecast_anchor_type`: Explicit declaration that track extrapolation initializes from `"OBSERVED_IMD_T0"`.
- `current_center`: Array `[lat, lon]` retained strictly for backwards compatibility with legacy UI consumers, semantically equivalent to `observed_reference_center`.

---

## 6. Live Smoke Test Results Across Reference Cyclones

Results from live execution of `POST /api/predict` in `MODEL_INFERENCE` mode:

| Cyclone | Sample ID | AI Detected Center (Phase 3C) | Observed IMD Reference ($t_0$) | Offset (km) | Forecast Anchor Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AMPHAN** | `NIO_2020_AMPHAN_20200517_0600Z` | **9.72°N, 85.33°E** | 11.50°N, 86.00°E | 210.91 km | `OBSERVED_IMD_T0` |
| **FANI** | `NIO_2019_FANI_20190429_1200Z` | **9.05°N, 85.74°E** | 10.10°N, 86.70°E | 157.03 km | `OBSERVED_IMD_T0` |
| **TAUKTAE** | `NIO_2021_TAUKTAE_20210515_1800Z` | **14.29°N, 71.05°E** | 14.50°N, 72.60°E | 168.53 km | `OBSERVED_IMD_T0` |
| **BIPARJOY** | `NIO_2023_BIPARJOY_20230611_0300Z` | **15.73°N, 65.12°E** | 18.00°N, 67.60°E | 364.66 km | `OBSERVED_IMD_T0` |
| **REMAL** | `NIO_2024_REMAL_20240525_0600Z` | **13.72°N, 83.54°E** | 18.20°N, 89.70°E | 825.33 km | `OBSERVED_IMD_T0` |

---

## 7. Operational Limitations & Scientific Honesty

1. **Epoch 2 Checkpoint Horizon:** `best_center_localization_cnn.pt` was checkpointed at Epoch 2 (validation mean DPE: 703.0 km; test mean DPE: 810.3 km). It provides coarse spatial localization and eye region detection, but exhibits significant displacement errors for asymmetric or sheared systems (e.g., REMAL offset: 825.33 km).
2. **Not Suitable for Autonomous Extrapolation:** Because single-frame center localization error (~150–800 km) is compounding, using this center directly as a kinematic anchor (Variant B) results in unacceptably high track errors (1048.57 km).
3. **Operational Recommendation:** Forecasters should utilize `ai_detected_center` as an automated computer-vision visual aid on infrared imagery, while continuing to rely on official synoptic fixes (`observed_reference_center`) as the initialized anchor for numerical and hybrid track guidance.
