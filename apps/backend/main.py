"""VAYU-NET — FastAPI Production Backend Service
=============================================
SIH 2026 Problem Statement 26070
India Meteorological Department (IMD) / Ministry of Earth Sciences (MoES)

Endpoints:
  GET  /health
  GET  /api/cyclones
  GET  /api/cyclones/{id}
  GET  /api/cyclones/{id}/forecast
  GET  /api/cyclones/{id}/verification
  GET  /api/cyclones/{id}/analogs
  GET  /api/cyclones/{id}/explainability
  GET  /api/explainability/{run_id}
  POST /api/predict
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.forecast.api_contracts import VayuForecastService, PredictionUnavailableError

app = FastAPI(
    title="VAYU-NET Cyclone Intelligence API",
    version="1.0.0",
    description="Operational North Indian Ocean Tropical Cyclone Intelligence & Multi-Horizon Forecasting Engine",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration (Strict Origins, Wildcard Disallowed)
raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,https://vayu-net.vercel.app")
origins = [o.strip() for o in raw_origins.split(",") if o.strip() and o.strip() != "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(PredictionUnavailableError)
def handle_prediction_unavailable(request, exc: PredictionUnavailableError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.to_dict()})

# Mount static explainability assets
EXPLAINABILITY_DIR = PROJECT_ROOT / "data/interim/ml/explainability"
if EXPLAINABILITY_DIR.exists():
    app.mount("/static/explainability", StaticFiles(directory=str(EXPLAINABILITY_DIR)), name="explainability")

# Lazy-loaded service instance
forecast_service: Optional[VayuForecastService] = None


def get_service() -> VayuForecastService:
    global forecast_service
    if forecast_service is None:
        forecast_service = VayuForecastService()
    return forecast_service


# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    cyclone_id: str = Field(..., description="Cyclone identifier or name (e.g., AMPHAN, FANI, TAUKTAE)")
    t0: Optional[str] = Field(None, description="Observation timestamp in ISO format (e.g., 2020-05-18T06:00:00+00:00)")
    percentile: str = Field("p80", description="Empirical uncertainty percentile ('p80' or 'p90')")
    mode: str = Field("MODEL_INFERENCE", description="Prediction mode: 'MODEL_INFERENCE' (runs real checkpoint) or 'PRECOMPUTED_DEMO'")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    runtime_artifacts: Dict[str, bool]


# ------------------------------------------------------------------------------
# Route Handlers
# ------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Verifies service responsiveness and runtime artifact presence without running heavy inference."""
    checkpoints = [
        "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
        "data/interim/ml/checkpoints/best_temporal_track_gru.pt",
        "data/interim/ml/checkpoints/best_phase5b_variant_a.pt",
        "data/interim/ml/checkpoints/best_phase5b_variant_b.pt",
        "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt",
    ]
    metadata_files = [
        "data/interim/ml/uncertainty_parameters.json",
        "data/interim/ml/analog_retrieval_cache.json",
        "data/manifests/vayu_net_sample_index.csv",
    ]

    artifact_status = {}
    for p in checkpoints + metadata_files:
        path = PROJECT_ROOT / p
        artifact_status[Path(p).name] = path.exists()

    all_critical_present = all(artifact_status[Path(p).name] for p in metadata_files)

    return HealthResponse(
        status="healthy" if all_critical_present else "degraded",
        service="vayu-net-backend",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "production"),
        runtime_artifacts=artifact_status,
    )


@app.get("/api/cyclones")
def list_cyclones(shortlist_only: bool = Query(True, description="Filter to locked 5 reference demo cyclones")):
    """Returns catalog of historical cyclone events."""
    service = get_service()
    return service.list_cyclones(shortlist_only=shortlist_only)


@app.get("/api/cyclones/{cyclone_id}")
def get_cyclone_details(cyclone_id: str):
    """Returns details and available observation timestamps for a cyclone."""
    service = get_service()
    try:
        return service.get_cyclone_details(cyclone_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/cyclones/{cyclone_id}/forecast")
def get_forecast(
    cyclone_id: str,
    t0: Optional[str] = Query(None, description="Observation timestamp"),
    percentile: str = Query("p80", description="Empirical error percentile (p80, p90)"),
    mode: str = Query("MODEL_INFERENCE", description="Forecast mode: 'MODEL_INFERENCE' or 'PRECOMPUTED_DEMO'"),
):
    """Returns multi-horizon (+12h, +24h, +48h) track predictions and empirical uncertainty cones."""
    service = get_service()
    try:
        return service.get_forecast(cyclone_id=cyclone_id, t0=t0, percentile=percentile, mode=mode)
    except PredictionUnavailableError as e:
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/cyclones/{cyclone_id}/verification")
def get_verification(
    cyclone_id: str,
    t0: Optional[str] = Query(None),
    mode: str = Query("MODEL_INFERENCE", description="Forecast mode used for verification"),
):
    """Verifies forecast trajectory against held-out IMD ground truth observations."""
    service = get_service()
    try:
        return service.get_verification(cyclone_id=cyclone_id, t0=t0, mode=mode)
    except PredictionUnavailableError as e:
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/cyclones/{cyclone_id}/analogs")
def get_analogs(cyclone_id: str, t0: Optional[str] = Query(None), k: int = Query(2, ge=1, le=5)):
    """Retrieves top-k historical analog storms using 7-dimensional standardized nearest-neighbor retrieval."""
    service = get_service()
    try:
        return service.get_analogs(cyclone_id=cyclone_id, t0=t0, k=k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/cyclones/{cyclone_id}/explainability")
def get_cyclone_explainability(cyclone_id: str):
    """Returns Grad-CAM explainability metadata and visual asset URLs for a reference cyclone."""
    manifest_path = EXPLAINABILITY_DIR / "explainability_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Explainability manifest not found.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    for case in manifest.get("cases", []):
        if cyclone_id.lower() in case["storm_name"].lower() or cyclone_id.lower() in case["storm_id"].lower():
            # Build accessible URLs
            base_url = "/static/explainability"
            case_resp = dict(case)
            case_resp["heatmap_url"] = f"{base_url}/{Path(case['heatmap_path']).name}"
            case_resp["overlay_url"] = f"{base_url}/{Path(case['overlay_path']).name}"
            case_resp["original_url"] = f"{base_url}/{Path(case['original_path']).name}"
            case_resp["interpretation_note"] = (
                f"{case.get('interpretation_note', '')} Saliency maps visualize activations of "
                f"intermediate layer '{case.get('target_layer', 'spatial_encoder.layer2')}' in Phase 3C localization encoder. "
                "This is an interpretability diagnostic, not a causal meteorological explanation."
            )
            return case_resp

    raise HTTPException(status_code=404, detail=f"No explainability record found for cyclone '{cyclone_id}'.")


@app.get("/api/explainability/{run_id}")
def get_explainability_by_id(run_id: str):
    """Returns explainability record by sample ID or run identifier."""
    manifest_path = EXPLAINABILITY_DIR / "explainability_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Explainability manifest not found.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    for case in manifest.get("cases", []):
        if run_id == case["sample_id"] or run_id in case["heatmap_path"]:
            base_url = "/static/explainability"
            case_resp = dict(case)
            case_resp["heatmap_url"] = f"{base_url}/{Path(case['heatmap_path']).name}"
            case_resp["overlay_url"] = f"{base_url}/{Path(case['overlay_path']).name}"
            case_resp["original_url"] = f"{base_url}/{Path(case['original_path']).name}"
            case_resp["interpretation_note"] = (
                f"{case.get('interpretation_note', '')} Saliency maps visualize activations of "
                f"intermediate layer '{case.get('target_layer', 'spatial_encoder.layer2')}' in Phase 3C localization encoder. "
                "This is an interpretability diagnostic, not a causal meteorological explanation."
            )
            return case_resp

    raise HTTPException(status_code=404, detail=f"No explainability record found for identifier '{run_id}'.")


@app.post("/api/predict")
def predict_cyclone(req: PredictRequest):
    """Consolidated endpoint delivering forecast, uncertainty, verification, analogs, and explainability."""
    service = get_service()
    try:
        sample_id = service.resolve_sample(req.cyclone_id, req.t0)
        forecast = service.get_forecast(cyclone_id=req.cyclone_id, t0=req.t0, percentile=req.percentile, mode=req.mode)
        verification = service.get_verification(cyclone_id=req.cyclone_id, t0=req.t0, mode=req.mode)
        analogs = service.get_analogs(cyclone_id=req.cyclone_id, t0=req.t0, k=2)

        # Intensity and wind prediction from real checkpoint
        intensity_pred = None
        try:
            intensity_pred = service.model_service.predict_intensity_and_wind(sample_id)
        except Exception as e:
            intensity_pred = {
                "error": f"Intensity inference unavailable: {str(e)}",
                "disclaimer": "Phase 6 model inference not available for this sample."
            }

        # Explainability lookup
        explainability = None
        manifest_path = EXPLAINABILITY_DIR / "explainability_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            for c in manifest.get("cases", []):
                if req.cyclone_id.lower() in c["storm_name"].lower() or req.cyclone_id.lower() in c["storm_id"].lower():
                    base_url = "/static/explainability"
                    explainability = dict(c)
                    explainability["heatmap_url"] = f"{base_url}/{Path(c['heatmap_path']).name}"
                    explainability["overlay_url"] = f"{base_url}/{Path(c['overlay_path']).name}"
                    explainability["original_url"] = f"{base_url}/{Path(c['original_path']).name}"
                    explainability["interpretation_note"] = (
                        f"{c.get('interpretation_note', '')} Saliency maps visualize activations of "
                        f"intermediate layer '{c.get('target_layer', 'spatial_encoder.layer2')}' in Phase 3C localization encoder. "
                        "This is an interpretability diagnostic, not a causal meteorological explanation."
                    )
                    break

        return {
            "status": "success",
            "sample_id": sample_id,
            "mode": req.mode.upper(),
            "prediction_source": forecast["prediction_source"],
            "center_detection_source": forecast["center_detection_source"],
            "forecast_anchor_type": forecast["forecast_anchor_type"],
            "observed_reference_center": forecast["observed_reference_center"],
            "ai_detected_center": forecast["ai_detected_center"],
            "model_version": forecast["model_version"],
            "checkpoint_identity": forecast.get("checkpoint_identity"),
            "forecast": forecast,
            "intensity_prediction": intensity_pred,
            "verification": verification,
            "analogs": analogs,
            "explainability": explainability,
            "scientific_disclaimer": (
                "VAYU-NET operational intelligence aid. Observed actual track is separated from "
                "AI forecast to prevent leakage. AI detected center is produced by Phase 3C CNN "
                "(best_center_localization_cnn.pt). Multi-horizon track forecast is initialized from the "
                "Observed IMD t0 reference (Phase 5B Variant A hybrid). Phase 6 intensity/wind predictions "
                "are experimental research estimates (macro F1 0.20, accuracy 25.1%) and not certified "
                "operational replacements for IMD ADT. Grad-CAM visual saliency is an interpretation aid only."
            ),
        }
    except PredictionUnavailableError as e:
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("apps.backend.main:app", host=host, port=port, reload=False)
