# VAYU-NET — Operational REST API Specification

**Service:** FastAPI Cyclone Intelligence Backend  
**Specification Baseline:** SIH 2026 Problem Statement 26070 — Blueprint / SRS V1.1 (P0/P1 Remediated)  
**Default Local Port:** `8000`  
**Production Host:** Render (`0.0.0.0:$PORT`)  
**CORS Policy:** Strict allowed origins (`http://localhost:5173,https://vayu-net.vercel.app`). Wildcard `*` disallowed.

---

## 1. Overview & Anti-Leakage Service Invariant

The VAYU-NET REST API provides locked endpoints delivering AI-driven cyclone pattern recognition, multi-horizon trajectory forecasting, empirical uncertainty estimation, historical analog retrieval, and Grad-CAM interpretability.

> [!IMPORTANT]
> ### STRICT ANTI-LEAKAGE & SOURCE PROVENANCE INVARIANT
> 1. **Zero Ground-Truth Fallback:** Missing predictions NEVER fall back to ground-truth coordinates. If a requested precomputed prediction is unavailable, the API returns a structured HTTP `503 Service Unavailable` with `error_code: "PREDICTION_UNAVAILABLE"`.
> 2. **Explicit Prediction Provenance:** Every forecast response includes `prediction_source`:
>    - `"MODEL_INFERENCE"`: Indicates live execution of a PyTorch checkpoint.
>    - `"PRECOMPUTED_DEMO"`: Indicates reproducible benchmark results from the held-out test evaluation.
> 3. **Input vs Target Isolation:** Predictions only ingest observed features up to $t_0$. Future coordinates ($t_{+12h}, t_{+24h}, t_{+48h}$) are accessed strictly post-prediction in the verification endpoint.

---

## 2. Endpoints Summary Table

| Method | Endpoint | Description | Modes | Runtime Dependency |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health & runtime artifact status check | Standard | File existence check |
| `GET` | `/api/cyclones` | List reference demo cyclones or full catalog | Catalog | `storm_event_manifest_v2.csv` |
| `GET` | `/api/cyclones/{id}` | Detailed storm metadata & observation timestamps | Catalog | `vayu_net_sample_index.csv` |
| `GET` | `/api/cyclones/{id}/forecast` | +12h, +24h, +48h predictions with P80 cones | `MODEL_INFERENCE`, `PRECOMPUTED_DEMO` | `ModelService` (`best_phase5b_variant_a.pt`) |
| `GET` | `/api/cyclones/{id}/verification` | Verification against held-out IMD ground truth | `MODEL_INFERENCE`, `PRECOMPUTED_DEMO` | `ForecastVerifier` |
| `GET` | `/api/cyclones/{id}/analogs` | Top-2 historical analog cyclones (TRAIN-only) | TRAIN | `analog_retrieval_cache.json` |
| `GET` | `/api/cyclones/{id}/explainability` | Grad-CAM saliency overlay and metadata | Interpretability | `explainability_manifest.json` |
| `GET` | `/api/explainability/{run_id}` | Lookup explainability record by sample ID | Interpretability | `explainability_manifest.json` |
| `POST` | `/api/predict` | Consolidated prediction, intensity & explanation | `MODEL_INFERENCE`, `PRECOMPUTED_DEMO` | Full runtime stack |

---

## 3. Detailed Endpoint Contracts

### 3.1 `GET /health`
Verifies backend process responsiveness and checks the presence of runtime weights without performing heavy inference.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "service": "vayu-net-backend",
  "version": "1.0.0",
  "environment": "production",
  "runtime_artifacts": {
    "best_center_localization_cnn.pt": true,
    "best_temporal_track_gru.pt": true,
    "best_phase5b_variant_a.pt": true,
    "best_phase5b_variant_b.pt": true,
    "best_phase6_intensity_wind.pt": true,
    "uncertainty_parameters.json": true,
    "analog_retrieval_cache.json": true,
    "vayu_net_sample_index.csv": true
  }
}
```

---

### 3.2 `GET /api/cyclones`
Returns catalog of cyclone events.

**Query Parameters:**
- `shortlist_only` (bool, default `true`): When true, restricts list to FANI, AMPHAN, TAUKTAE, BIPARJOY, REMAL.

**Response `200 OK`:**
```json
[
  {
    "cyclone_id": "NIO_2020_AMPHAN",
    "storm_name": "AMPHAN",
    "year": 2020,
    "split": "VALIDATION",
    "peak_category": "SuCS",
    "max_wind_kt": 130.0,
    "samples_count": 20,
    "is_demo_shortlist": true
  }
]
```

---

### 3.3 `GET /api/cyclones/{id}/forecast`
Returns multi-horizon (+12h, +24h, +48h) track forecasts and empirical uncertainty cones.

**Query Parameters:**
- `t0` (string, optional): Observation timestamp in ISO format. If omitted, defaults to mature sample.
- `percentile` (string, default `"p80"`): Empirical uncertainty level (`"p80"` or `"p90"`).
- `mode` (string, default `"MODEL_INFERENCE"`): 
  - `"MODEL_INFERENCE"`: Executes live PyTorch checkpoint.
  - `"PRECOMPUTED_DEMO"`: Reads precomputed benchmark results.

**Response `200 OK` (Live Model Inference):**
```json
{
  "cyclone_id": "NIO_2020_AMPHAN",
  "sample_id": "NIO_2020_AMPHAN_20200517_0600Z",
  "mode": "MODEL_INFERENCE",
  "prediction_source": "MODEL_INFERENCE",
  "center_detection_source": "MODEL_INFERENCE",
  "forecast_anchor_type": "OBSERVED_IMD_T0",
  "model_version": "phase5b-v1.0-variant-a",
  "checkpoint_identity": "best_phase5b_variant_a.pt",
  "anchor_type": "observed_center",
  "issue_timestamp_utc": "2020-05-17T06:00:00+00:00",
  "observed_reference_center": {
    "latitude": 11.5,
    "longitude": 86.0
  },
  "ai_detected_center": {
    "latitude": 9.72,
    "longitude": 85.33
  },
  "current_center": [11.5, 86.0],
  "current_wind_kt": 55.0,
  "current_category": "SCS",
  "forecast_horizons": [
    {
      "horizon": "+12h",
      "target_timestamp_utc": "2020-05-17T18:00:00+00:00",
      "latitude": 12.3698,
      "longitude": 85.6707,
      "empirical_uncertainty": {
        "radius_km": 97.51,
        "percentile": "p80",
        "derivation_split": "VALIDATION",
        "label": "empirical"
      }
    },
    {
      "horizon": "+24h",
      "target_timestamp_utc": "2020-05-18T06:00:00+00:00",
      "latitude": 13.2450,
      "longitude": 85.4567,
      "empirical_uncertainty": {
        "radius_km": 197.04,
        "percentile": "p80",
        "derivation_split": "VALIDATION",
        "label": "empirical"
      }
    },
    {
      "horizon": "+48h",
      "target_timestamp_utc": "2020-05-19T06:00:00+00:00",
      "latitude": 15.5183,
      "longitude": 85.1316,
      "empirical_uncertainty": {
        "radius_km": 449.65,
        "percentile": "p80",
        "derivation_split": "VALIDATION",
        "label": "empirical"
      }
    }
  ],
  "uncertainty_summary": {
    "derivation_split": "VALIDATION",
    "percentile": "P80",
    "label": "empirical",
    "is_probabilistic_confidence_interval": false
  }
}
```

**Error Response `503 Service Unavailable` (When Demo Result is Missing):**
```json
{
  "detail": {
    "error_code": "PREDICTION_UNAVAILABLE",
    "message": "Precomputed demo forecast unavailable for sample 'NIO_2020_AMPHAN_20200517_0600Z'. Precomputed demo results are archived only for held-out TEST partition storms. To execute real runtime model inference, request mode='MODEL_INFERENCE'.",
    "sample_id": "NIO_2020_AMPHAN_20200517_0600Z",
    "status_code": 503
  }
}
```

---

### 3.4 `GET /api/cyclones/{id}/verification`
Verifies forecast trajectory against held-out IMD ground truth observations strictly post-prediction.

**Query Parameters:**
- `t0` (string, optional): Observation timestamp in ISO format.
- `mode` (string, default `"MODEL_INFERENCE"`): Mode used for forecast generation.

**Response `200 OK`:**
```json
{
  "cyclone_id": "NIO_2020_AMPHAN",
  "sample_id": "NIO_2020_AMPHAN_20200517_0600Z",
  "split": "VALIDATION",
  "forecast_mode": "MODEL_INFERENCE",
  "prediction_source": "MODEL_INFERENCE",
  "model_version": "phase5b-v1.0-variant-a",
  "horizons": {
    "12h": {
      "horizon": "+12h",
      "forecast": { "latitude": 12.3698, "longitude": 85.6707 },
      "actual": { "latitude": 12.5, "longitude": 86.1, "source": "IMD_BEST_TRACK_V2" },
      "error": { "dpe_km": 48.81, "delta_lat_deg": -0.1302, "delta_lon_deg": -0.4293 }
    }
  },
  "summary": {
    "mean_dpe_km": 116.44,
    "status": "VERIFIED_AGAINST_IMD_TRUTH"
  }
}
```

---

### 3.5 `GET /api/cyclones/{id}/analogs`
Retrieves top-$k$ historical analog storms strictly from the 696 candidate observations in the **TRAIN partition (1998–2018)**.

**Query Parameters:**
- `t0` (string, optional): Observation timestamp.
- `k` (int, default `2`, range `1..5`): Number of analog storms to return.

**Response `200 OK`:**
```json
{
  "selected_case": { "storm_id": "NIO_2020_AMPHAN" },
  "features_used": ["lat", "lon", "wind_kt", "pressure_hpa", "dx_12h", "dy_12h", "dwind_12h"],
  "distance_metric": "Standardized Euclidean Distance (TRAIN baseline)",
  "candidate_pool_provenance": "Strictly TRAIN partition (1998–2018), 696 candidate snapshots from 81 historical storms",
  "candidate_split_verified": true,
  "k": 2,
  "label": "HISTORICAL ANALOG",
  "analogs": [
    {
      "rank": 1,
      "storm_name": "NARGIS",
      "year": 2008,
      "candidate_split": "TRAIN",
      "standardized_distance": 0.6778,
      "label": "HISTORICAL ANALOG (TRAIN ONLY)"
    }
  ]
}
```

---

### 3.6 `POST /api/predict`
Consolidated operational endpoint returning track forecast, Phase 6 multi-task intensity/wind prediction, deterministic verification, analogs, and Grad-CAM interpretability.

**Request Body:**
```json
{
  "cyclone_id": "AMPHAN",
  "t0": "2020-05-17T06:00:00+00:00",
  "percentile": "p80",
  "mode": "MODEL_INFERENCE"
}
```

**Response `200 OK`:**
```json
{
  "status": "success",
  "sample_id": "NIO_2020_AMPHAN_20200517_0600Z",
  "mode": "MODEL_INFERENCE",
  "prediction_source": "MODEL_INFERENCE",
  "center_detection_source": "MODEL_INFERENCE",
  "forecast_anchor_type": "OBSERVED_IMD_T0",
  "observed_reference_center": {
    "latitude": 11.5,
    "longitude": 86.0
  },
  "ai_detected_center": {
    "latitude": 9.72,
    "longitude": 85.33
  },
  "model_version": "phase5b-v1.0-variant-a",
  "checkpoint_identity": "best_phase5b_variant_a.pt",
  "forecast": { ... },
  "intensity_prediction": {
    "prediction_source": "MODEL_INFERENCE",
    "model_version": "phase6-v1.0-multimodal-era5",
    "checkpoint_identity": "best_phase6_intensity_wind.pt",
    "predicted_category": "CS",
    "predicted_category_confidence": 0.3925,
    "predicted_wind_kt": 42.9,
    "wind_uncertainty_p80_kt": 46.8,
    "scientific_disclaimer": "Phase 6 multi-task intensity/wind neural network is an experimental research model (validation macro F1: 0.2006, accuracy: 25.10%, wind MAE: 25.44 kt). It is NOT a certified operational replacement for the IMD Advanced Dvorak Technique (ADT)."
  },
  "verification": { ... },
  "analogs": { ... },
  "explainability": {
    "storm_name": "AMPHAN",
    "target_layer": "spatial_encoder.layer2",
    "interpretation_note": "Saliency maps visualize activations of intermediate layer 'spatial_encoder.layer2' in Phase 3C localization encoder. This is an interpretability diagnostic, not a causal meteorological explanation."
  },
  "scientific_disclaimer": "VAYU-NET operational intelligence aid. Observed actual track is separated from AI forecast to prevent leakage. Phase 6 intensity/wind predictions are experimental research estimates (macro F1 0.20, accuracy 25.1%) and not certified operational replacements for IMD ADT. Grad-CAM visual saliency is an interpretation aid only."
}
```
