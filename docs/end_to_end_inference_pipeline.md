# VAYU-NET End-to-End Operational Inference Pipeline
## Architectural Specification, Dataflow, & Operational Contracts

**Document Version:** 1.0.0  
**Date:** September 2026  
**System:** VAYU-NET (SIH 2026 Problem Statement 26070)  
**Governing Agencies:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)  

---

### Executive Overview

The VAYU-NET End-to-End Inference Pipeline operationalizes validated deep learning models, empirical calibration databases, and interpretability assets into a unified forecasting service for tropical cyclones in the North Indian Ocean (Bay of Bengal and Arabian Sea).

The pipeline strictly enforces two foundational architectural boundaries:
1. **Prediction Path vs Verification Path:** Future ground-truth coordinates and observations are never accessed during feature extraction, center detection, intensity classification, or trajectory forecasting. Ground truth is accessed strictly downstream inside the verification engine.
2. **Primary GridSat Satellite ML vs IMERG Secondary Context:** Controlled ablation experiments (EXP-M1, EXP-M2, EXP-M3) conclusively established that multimodal precipitation fusion does not improve forecast skill over unimodal infrared observations. GridSat-B1 IRWIN (plus ERA5 atmospheric fields) drives all predictive heads. Calibrated precipitation from NASA GPM IMERG Final Run V07B is ingested and served purely as a secondary synchronized meteorological observation context layer with 0% predictive weight.

---

### 1. Architectural Dataflow Diagram

```
[Client Request: event_id, t0_utc]
                 │
                 ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Event & Sample Manifest Resolution                  │
  │    (Resolve storm metadata, t0 issue timestamp,        │
  │     sample_id from vayu_net_sample_index.csv)          │
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Causal 6-Frame Observation Preparation              │
  │    (Load t-15h, t-12h, t-9h, t-6h, t-3h, t0)           │
  │    Verify zero future frames, zero NaNs                │
  └────────────────────────┬───────────────────────────────┘
                           │
      ┌────────────────────┴────────────────────┐
      ▼                                         ▼
┌───────────────────────────────┐     ┌────────────────────────────────┐
│ PRIMARY PREDICTION PATH       │     │ SECONDARY OBSERVATION CONTEXT  │
│ (GridSat-B1 IRWIN + ERA5)     │     │ (NASA GPM IMERG V07B)          │
└──────────────┬────────────────┘     └────────────────┬───────────────┘
               │                                       │
               ├─► [3. Phase 3C Center Localization]  ├─► [10. Synchronized Frames]
               │     Model: Dedicated ResNet-18        │     - Variable: precipCal (mm/h)
               │     Output: AI Center [lat, lon]      │     - 6 frames (t-15h to t0)
               │                                       │     - Predictive Role: NONE (0%)
               ├─► [4. Phase 6 Intensity & Wind]       │
               │     Model: Multi-Task Conv-GRU        │
               │     Output: Category, Wind, Pressure  │
               │                                       │
               ├─► [5. Phase 5B Track Forecasting]     │
               │     Model: Hybrid Residual Model      │
               │     Anchor: Observed IMD t0           │
               │     Output: +12h, +24h, +48h [lat,lon]│
               │                                       │
               ├─► [6. Empirical Uncertainty Cones]    │
               │     Source: Validation Residuals      │
               │     Output: P80 Radii & Polygon Cones │
               │                                       │
               ├─► [8. Top-2 Analog Retrieval]         │
               │     Source: 7-D Vector (TRAIN split)  │
               │     Output: Nearest historical storms │
               │                                       │
               └─► [9. Grad-CAM Vision Interpretability]
                     Layer: Phase 3C spatial layer2
                     Output: Saliency, Overlay, Raw
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 7. Trajectory Verification Path (Strictly Downstream)   │
  │    Input: Predicted Coordinates (+12h, +24h, +48h)     │
  │    Target: Held-Out IMD Best Track Ground Truth        │
  │    Output: Direct Position Error (DPE in km), Δlat/lon │
  └────────────────────────────┬───────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 11. Canonical Inference Contract Response Object       │
  │     (Validated single schema for API & UI dashboard)   │
  └────────────────────────────────────────────────────────┘
```

---

### 2. Detailed Pipeline Execution Stages

#### Stage 1: Event & Sample Resolution
- Input: `event_id` (e.g., `"AMPHAN"`, `"NIO_2020_AMPHAN"`), optional `t0_utc`.
- Matches against `data/manifests/storm_event_manifest_v2.csv` and `data/manifests/vayu_net_sample_index.csv`.
- If `t0_utc` is omitted, defaults deterministically to the representative mature observation sequence.

#### Stage 2: 6-Frame Observation Preparation
- Validates the existence and completeness of exactly 6 causal observations:
  - $t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0$.
- Checks that all frame timestamps satisfy $t \le t_0$ (guaranteeing zero future frame leakage).
- Standardizes raw GridSat-B1 IRWIN brightness temperatures using train-only statistics ($\mu = 265.41\text{ K}, \sigma = 24.89\text{ K}$).

#### Stage 3: Phase 3C Cyclone Center Localization
- Checkpoint: `data/interim/ml/checkpoints/best_center_localization_cnn.pt` (Dedicated ResNet-18).
- Forward Pass: Processes single-frame normalized satellite tensor $[1, 1, 572, 929]$ through spatial convolutional residual blocks to produce a 2D spatial heatmap.
- Soft-Argmax: Computes normalized coordinate expectation $[u_{\text{lat}}, u_{\text{lon}}] \in [0, 1]^2$ mapped continuously to geographic domain $[-5^\circ\text{S}, 35^\circ\text{N}] \times [40^\circ\text{E}, 105^\circ\text{E}]$.
- Contract Output: `center.ai_lat`, `center.ai_lon`, `center.confidence`.
- Provenance Guarantee: Operates strictly on satellite visual features without reading synoptic center labels.

#### Stage 4: Phase 6 Multi-Task Intensity Classification & Wind Regression
- Checkpoint: `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt` (Multi-Task Conv-GRU).
- Forward Pass: Ingests 6-frame satellite sequence $[1, 6, 132]$ and ERA5 environmental fields $[1, 6, 8, 41, 66]$.
- Outputs:
  - IMD 7-class intensity classification logits: `D`, `DD`, `CS`, `SCS`, `VSCS`, `ESCS`, `SuCS`.
  - Softmax probabilities labeled explicitly as `model_probability`.
  - Continuous maximum sustained wind speed (`wind_kt`) and central pressure (`pressure_hpa`).
  - Empirical uncertainty threshold ($\pm\text{kt}$ at P80).
- Disclaimer: Non-certified research guidance; does not replace IMD Advanced Dvorak Technique bulletins.

#### Stage 5: Phase 5B Variant A Multi-Horizon Track Forecasting
- Checkpoint: `data/interim/ml/checkpoints/best_phase5b_variant_a.pt` (Hybrid Residual Model).
- Anchor: Observed IMD $t_0$ reference position.
- Forward Pass: Integrates kinematic momentum extrapolation with deep spatio-temporal residual corrections:
  $$\hat{\mathbf{x}}_{t+h} = \mathbf{x}_{\text{anchor}} + \mathbf{v}_{\text{kinematic}} \cdot h + \Delta\mathbf{x}_{\text{residual}}(h)$$
- Outputs: Predicted multi-horizon coordinates for $+12\text{h}$, $+24\text{h}$, and $+48\text{h}$.

#### Stage 6: Empirical Forecast Uncertainty Cones
- Service: `apps/backend/services/uncertainty_service.py` (`UncertaintyService`).
- Source: `data/interim/ml/uncertainty_parameters.json` derived strictly from held-out Phase 5B validation residuals.
- Radii: $+12\text{h}$ ($97.5\text{ km}$), $+24\text{h}$ ($197.0\text{ km}$), $+48\text{h}$ ($449.6\text{ km}$) at P80.
- Geometry: Produces GeoJSON-compatible closed spherical polygon rings around destination points.

#### Stage 7: Deterministic Trajectory Verification (Strictly Downstream)
- Service: `apps/backend/services/verification_service.py` (`VerificationService`).
- Execution: Evaluated strictly *after* forecast generation.
- Calculation: Direct Position Error (DPE) in km via spherical haversine formula against actual future IMD coordinates:
  $$\text{DPE}_h = 2 R_{\text{earth}} \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$
- If future ground truth is unobserved (e.g. storm dissipated or future point), returns clean status `"UNAVAILABLE"` without raising pipeline exceptions.

#### Stage 8: Top-2 Historical Analog Retrieval
- Engine: `ml/forecast/analog_retrieval.py` (`AnalogRetriever`).
- Database: `data/interim/ml/analog_retrieval_cache.json` containing standardized 7-D vectors strictly from the TRAIN partition:
  $$\mathbf{z} = \left[ \frac{\text{lat} - \mu_1}{\sigma_1}, \frac{\text{lon} - \mu_2}{\sigma_2}, \frac{w - \mu_3}{\sigma_3}, \frac{p - \mu_4}{\sigma_4}, \frac{\Delta x_{12}}{\sigma_5}, \frac{\Delta y_{12}}{\sigma_6}, \frac{\Delta w_{12}}{\sigma_7} \right]$$
- Excludes same-storm observations to guarantee distinct historical comparisons. Returns exactly top-2 nearest neighbors.

#### Stage 9: Grad-CAM Explainability & Saliency
- Layer: Intermediate layer `spatial_encoder.layer2` of the Phase 3C feature extractor.
- Provides visual overlay and raw normalized activation heatmaps showing regions of high influence during intensity classification.
- Interpretation notice attached: interpretability diagnostic only; not a causal atmospheric dynamic claim.

#### Stage 10: Secondary IMERG Observation Context
- Synchronized half-hourly calibrated precipitation fields (NASA GPM IMERG Final Run V07B) aligned to the 6-frame observation sequence.
- Labeled explicitly with 0% predictive weight. Served to dashboard clients for multi-sensor visualization.

#### Stage 11: Canonical Response Construction
- Assembles all sub-dictionaries into the canonical schema and caches the result by `inference_id`.

---

### 3. Canonical Inference JSON Contract Schema

```json
{
  "event": {
    "event_id": "NIO_2020_AMPHAN",
    "storm_name": "AMPHAN",
    "year": 2020,
    "split": "VALIDATION",
    "basin": "North Indian Ocean (Bay of Bengal / Arabian Sea)",
    "peak_category": "SuCS",
    "max_wind_kt": 130.0
  },
  "observation": {
    "sample_id": "NIO_2020_AMPHAN_20200518_0600Z",
    "t0_utc": "2020-05-18T06:00:00+00:00",
    "reference_center": {"lat": 13.4, "lon": 86.2},
    "reference_wind_kt": 120.0,
    "reference_category": "SuCS",
    "gridsat_frames": [
      {"step": "t_minus_15h", "offset_hours": -15, "timestamp_utc": "2020-05-17T15:00:00+00:00", "status": "VALIDATED"},
      {"step": "t0", "offset_hours": 0, "timestamp_utc": "2020-05-18T06:00:00+00:00", "status": "VALIDATED"}
    ]
  },
  "center": {
    "ai_lat": 13.5682,
    "ai_lon": 86.3211,
    "confidence": 0.95,
    "method": "Phase 3C Dedicated ResNet-18 (2D Soft-Argmax)"
  },
  "intensity": {
    "category": "VSCS",
    "category_index": 4,
    "confidence": 0.3842,
    "confidence_label": "model_probability",
    "category_probabilities": {"D": 0.02, "DD": 0.05, "CS": 0.12, "SCS": 0.18, "VSCS": 0.38, "ESCS": 0.16, "SuCS": 0.09}
  },
  "wind": {
    "wind_kt": 92.4,
    "confidence": 15.0,
    "confidence_metric": "empirical_p80_kt",
    "central_pressure_hpa": 948.0
  },
  "forecast": {
    "plus_12h": {"lat": 15.2140, "lon": 86.8210},
    "plus_24h": {"lat": 17.4102, "lon": 87.4120},
    "plus_48h": {"lat": 22.1804, "lon": 88.6210},
    "anchor_type": "observed_center"
  },
  "uncertainty": {
    "plus_12h_km": 97.5,
    "plus_24h_km": 197.0,
    "plus_48h_km": 449.6,
    "percentile": "P80",
    "label": "empirical",
    "cone_geometries": {"plus_12h": [[86.8, 15.2], ...]}
  },
  "verification": {
    "status": "AVAILABLE",
    "aggregate_dpe_km": 84.3,
    "horizons": {
      "plus_12h": {"status": "VERIFIED", "predicted": {"lat": 15.214, "lon": 86.821}, "actual": {"lat": 15.4, "lon": 87.0}, "dpe_km": 28.4},
      "plus_24h": {"status": "VERIFIED", "predicted": {"lat": 17.410, "lon": 87.412}, "actual": {"lat": 17.8, "lon": 87.5}, "dpe_km": 44.1},
      "plus_48h": {"status": "VERIFIED", "predicted": {"lat": 22.180, "lon": 88.621}, "actual": {"lat": 23.2, "lon": 88.9}, "dpe_km": 118.2}
    }
  },
  "analogs": [
    {"storm_id": "NIO_2013_PHAILIN", "storm_name": "PHAILIN", "year": 2013, "distance": 1.42, "split": "TRAIN"},
    {"storm_id": "NIO_1999_ORISSA_SUPER_CYCLONE", "storm_name": "ORISSA_SUPER_CYCLONE", "year": 1999, "distance": 1.78, "split": "TRAIN"}
  ],
  "saliency": {
    "status": "AVAILABLE",
    "method": "Grad-CAM",
    "target_head_explained": "intensity_classification",
    "target_layer": "spatial_encoder.layer2",
    "heatmap_url": "/static/explainability/heatmap_amphan.png",
    "overlay_url": "/static/explainability/overlay_amphan.png"
  },
  "secondary_observation": {
    "source": "NASA GPM IMERG Final Run V07B",
    "frames": [
      {"step": "t_minus_15h", "timestamp_utc": "2020-05-17T15:00:00+00:00", "local_status": "VALIDATED_HDF5"},
      {"step": "t0", "timestamp_utc": "2020-05-18T06:00:00+00:00", "local_status": "VALIDATED_HDF5"}
    ],
    "timestamp_alignment": "EXACT_3H_SYNCHRONIZED",
    "display_ready": true,
    "predictive_fusion_role": "NONE (Diagnostic / Observation Context Only)"
  },
  "metadata": {
    "inference_id": "vayu_inf_a1b2c3d4e5f6",
    "execution_timestamp_utc": "2026-09-26T01:00:00+00:00",
    "prediction_source": "MODEL_INFERENCE"
  }
}
```

---

### 4. API Specification & Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service liveness, status, and checkpoint file presence. |
| `POST` | `/api/inference/run` | Executes canonical end-to-end multi-task inference pipeline. |
| `GET` | `/api/events` | Catalog of all 126 historical cyclone events. |
| `GET` | `/api/events/{event_id}` | Detailed observation history and timestamps for an event. |
| `GET` | `/api/inference/{inference_id}` | Retrieves cached inference results by unique execution ID. |
| `GET` | `/api/cyclones` | Legacy shortlist endpoint preserved for backwards compatibility. |
| `POST` | `/api/predict` | Legacy consolidated predict endpoint preserved for backwards compatibility. |

---

### 5. Primary SIH Rehearsal Demo Scenario
- **Event ID:** `AMPHAN` (`NIO_2020_AMPHAN`)
- **Observation Timestamp ($t_0$):** `2020-05-18T06:00:00+00:00`
- **One-Click UI Trigger:** Click **"⚡ SIH REHEARSAL DEMO"** button in header.
- **Expected Results:**
  - AI Detected Center: Soft-argmax localized near Bay of Bengal eye.
  - Category / Wind: Very Severe Cyclonic Storm (VSCS), continuous wind estimation.
  - Multi-Horizon Forecast: Curved northeastward trajectory toward West Bengal/Bangladesh.
  - Verification: DPE computed against held-out IMD ground truth.
  - Analogs: Top-2 historical analogs retrieved strictly from TRAIN split.
  - Saliency: Grad-CAM overlay highlighting inner eyewall convection.
  - Secondary Context: 6 synchronized NASA GPM IMERG precipitation frames displayed.
