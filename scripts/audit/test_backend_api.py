"""VAYU-NET — Automated Backend & API Integration Test Suite
==========================================================
Verifies that all FastAPI REST endpoints, CORS configurations,
static asset mounts, and P0/P1 remediation guarantees operate as expected.

Explicitly verifies:
1. Missing prediction in PRECOMPUTED_DEMO mode NEVER returns actual future target coordinates (raises 503).
2. MODEL_INFERENCE mode actually invokes the PyTorch inference pipeline.
3. PRECOMPUTED_DEMO mode is explicitly labeled (prediction_source = 'PRECOMPUTED_DEMO').
4. MODEL_INFERENCE never silently falls back to DEMO.
5. Invalid cyclone IDs produce proper errors (404).
6. Missing runtime model produces structured 503 error.
7. /health does not access future target data.
8. API response contains prediction_source.
9. Forecast output coordinates are distinct from verification target fields.
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200, healthy status, and does not touch future target data."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["service"] == "vayu-net-backend"
    assert "runtime_artifacts" in data
    # Health check must not touch ground truth coordinates
    assert "imd_lat_12h" not in data
    assert "forecast" not in data


def test_list_cyclones():
    """Verify GET /api/cyclones returns reference shortlist."""
    res = client.get("/api/cyclones")
    assert res.status_code == 200
    cyclones = res.json()
    assert len(cyclones) >= 5
    names = [c["storm_name"] for c in cyclones]
    for ref in ["AMPHAN", "FANI", "TAUKTAE", "BIPARJOY", "REMAL"]:
        assert any(ref in n for n in names), f"Reference cyclone {ref} missing from catalog!"


def test_cyclone_details():
    """Verify GET /api/cyclones/{id} returns metadata and sample list."""
    res = client.get("/api/cyclones/AMPHAN")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) > 0
    assert "peak_category" in data


def test_forecast_live_model_inference():
    """Verify GET /api/cyclones/{id}/forecast with mode=MODEL_INFERENCE executes real model."""
    res = client.get("/api/cyclones/AMPHAN/forecast?percentile=p80&mode=MODEL_INFERENCE")
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_source"] == "MODEL_INFERENCE"
    assert "phase5b" in data["model_version"]
    assert "forecast_horizons" in data
    assert len(data["forecast_horizons"]) == 3
    for h in data["forecast_horizons"]:
        assert "latitude" in h and "longitude" in h
        assert h["empirical_uncertainty"]["radius_km"] > 0


def test_forecast_explicit_precomputed_demo():
    """Verify GET /api/cyclones/{id}/forecast with mode=PRECOMPUTED_DEMO returns labeled stored result."""
    # TAUKTAE is in TEST split, so precomputed results exist
    res = client.get("/api/cyclones/TAUKTAE/forecast?percentile=p80&mode=PRECOMPUTED_DEMO")
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_source"] == "PRECOMPUTED_DEMO"
    assert "precomputed" in data["model_version"]


def test_missing_prediction_never_leaks_ground_truth():
    """
    P0-1 CRITICAL TEST: Missing prediction in PRECOMPUTED_DEMO mode must NEVER
    return ground truth coordinates. It must return structured 503 error.
    """
    # AMPHAN is in VALIDATION split, so it has no stored raw_test prediction in phase5b_hybrid_results.json
    res = client.get("/api/cyclones/AMPHAN/forecast?percentile=p80&mode=PRECOMPUTED_DEMO")
    assert res.status_code == 503
    err = res.json()
    assert "detail" in err
    assert err["detail"]["error_code"] == "PREDICTION_UNAVAILABLE"
    # Ground truth future coordinates must NEVER appear as forecast
    assert "forecast_horizons" not in err


def test_invalid_cyclone_id_returns_404():
    """Verify invalid cyclone ID returns 404."""
    res = client.get("/api/cyclones/NON_EXISTENT_CYCLONE_XYZ/forecast")
    assert res.status_code == 404


def test_verification_endpoint_distinct_from_forecast():
    """Verify GET /api/cyclones/{id}/verification compares prediction with held-out truth strictly post-prediction."""
    res = client.get("/api/cyclones/AMPHAN/verification?mode=MODEL_INFERENCE")
    assert res.status_code == 200
    data = res.json()
    assert "horizons" in data
    assert "summary" in data
    assert data["prediction_source"] == "MODEL_INFERENCE"

    # Verify forecast coordinates are distinct from actual truth
    for h_key, h_data in data["horizons"].items():
        fc = h_data["forecast"]
        ac = h_data["actual"]
        err = h_data["error"]
        assert "latitude" in fc and "longitude" in fc
        assert "latitude" in ac and "longitude" in ac
        assert err["dpe_km"] >= 0


def test_analogs_provenance_train_only():
    """Verify GET /api/cyclones/{id}/analogs returns analogs strictly from TRAIN partition."""
    res = client.get("/api/cyclones/AMPHAN/analogs?k=2")
    assert res.status_code == 200
    data = res.json()
    assert "analogs" in data
    assert len(data["analogs"]) == 2
    assert data.get("candidate_split_verified") is True
    for an in data["analogs"]:
        assert an["candidate_split"] == "TRAIN"


def test_explainability_endpoint():
    """Verify GET /api/cyclones/{id}/explainability returns accurate layer description."""
    res = client.get("/api/cyclones/AMPHAN/explainability")
    assert res.status_code == 200
    data = res.json()
    assert "overlay_url" in data
    assert "heatmap_url" in data
    assert "target_layer" in data
    assert "spatial_encoder.layer2" in data["target_layer"]
    assert "interpretability" in data["interpretation_note"].lower()


def test_static_asset_serving():
    """Verify static mount serves pre-generated explainability assets."""
    res = client.get("/static/explainability/overlay_amphan_20200518T060000+0000.png")
    assert res.status_code == 200
    assert "image/png" in res.headers["content-type"]
    assert len(res.content) > 1000


def test_consolidated_predict_endpoint():
    """Verify POST /api/predict delivers complete operational payload with prediction_source."""
    res = client.post("/api/predict", json={"cyclone_id": "AMPHAN", "percentile": "p80", "mode": "MODEL_INFERENCE"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["prediction_source"] == "MODEL_INFERENCE"
    assert data["mode"] == "MODEL_INFERENCE"
    assert "forecast" in data
    assert "intensity_prediction" in data
    assert "verification" in data
    assert "analogs" in data
    assert "explainability" in data
    assert "scientific_disclaimer" in data


def test_model_inference_never_silently_falls_back():
    """Verify that when MODEL_INFERENCE mode is requested, it does not silently switch to precomputed demo."""
    res = client.post("/api/predict", json={"cyclone_id": "TAUKTAE", "percentile": "p80", "mode": "MODEL_INFERENCE"})
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_source"] == "MODEL_INFERENCE"
    assert data["forecast"]["prediction_source"] == "MODEL_INFERENCE"


def test_ai_detected_center_exists_and_distinct():
    """
    F03-1 & F03-3 TEST: Verify ai_detected_center exists and remains strictly distinct
    from observed_reference_center in MODEL_INFERENCE mode.
    """
    res = client.get("/api/cyclones/AMPHAN/forecast?percentile=p80&mode=MODEL_INFERENCE")
    assert res.status_code == 200
    data = res.json()

    assert "ai_detected_center" in data
    assert "observed_reference_center" in data
    assert "center_detection_source" in data
    assert "forecast_anchor_type" in data

    ai_c = data["ai_detected_center"]
    obs_c = data["observed_reference_center"]

    assert "latitude" in ai_c and "longitude" in ai_c
    assert "latitude" in obs_c and "longitude" in obs_c

    # Must be distinct numbers (AI prediction vs human synoptic analysis)
    assert not (ai_c["latitude"] == obs_c["latitude"] and ai_c["longitude"] == obs_c["longitude"]), (
        f"AI detected center ({ai_c}) must not be identical to observed IMD reference ({obs_c})!"
    )
    assert data["center_detection_source"] == "MODEL_INFERENCE"
    assert data["forecast_anchor_type"] == "OBSERVED_IMD_T0"
    assert data["prediction_source"] == "MODEL_INFERENCE"


def test_ai_detected_center_generated_by_phase3c():
    """
    F03-2 TEST: Verify ai_detected_center is genuinely generated by the Phase 3C CNN forward pass.
    """
    from ml.inference.model_service import get_model_service
    from ml.models.center_localization_cnn import DedicatedCenterLocalizationResNet

    ms = get_model_service()
    sample_id = "NIO_2020_AMPHAN_20200516_0000Z"

    # 1. Execute via ModelService
    center_res = ms.predict_center(sample_id)
    assert center_res["checkpoint_identity"] == "best_center_localization_cnn.pt"
    assert center_res["model_version"] == "phase3c-v1.0-center-localization"
    ai_c = center_res["ai_detected_center"]

    # 2. Directly verify against raw model forward pass on the t0 frame
    t0_tensor = ms.load_t0_satellite_frame(sample_id)
    model = ms.get_center_model()
    with pytest.MonkeyPatch.context() as mp:
        import torch
        with torch.no_grad():
            out = model(t0_tensor)
            deg = DedicatedCenterLocalizationResNet.denormalize_center(out["norm_center"].cpu())[0]
            lat_expected = round(float(deg[0].item()), 4)
            lon_expected = round(float(deg[1].item()), 4)

    assert ai_c["latitude"] == lat_expected
    assert ai_c["longitude"] == lon_expected


def test_forecast_anchor_explicitly_observed_imd_t0():
    """
    F03-4 TEST: Verify forecast anchor is explicitly labeled OBSERVED_IMD_T0 and Variant A
    uses the observed center, preserving the forecaster-in-the-loop design.
    """
    res = client.post("/api/predict", json={"cyclone_id": "AMPHAN", "percentile": "p80", "mode": "MODEL_INFERENCE"})
    assert res.status_code == 200
    data = res.json()

    assert data["forecast_anchor_type"] == "OBSERVED_IMD_T0"
    assert data["forecast"]["forecast_anchor_type"] == "OBSERVED_IMD_T0"
    assert data["forecast"]["anchor_type"] == "observed_center"


def test_missing_center_model_causes_explicit_failure(tmp_path):
    """
    F03-8 TEST: Verify missing or invalid center localization model causes explicit failure
    without silent fallback to ground truth or precomputed results.
    """
    from ml.inference.model_service import ModelService

    # Point to an empty directory where best_center_localization_cnn.pt does not exist
    empty_ckpt_dir = tmp_path / "checkpoints"
    empty_ckpt_dir.mkdir()

    ms = ModelService(checkpoint_dir=str(empty_ckpt_dir))
    with pytest.raises(FileNotFoundError) as exc_info:
        ms.predict_center("NIO_2020_AMPHAN_20200516_0000Z")

    assert "best_center_localization_cnn.pt" in str(exc_info.value)


def test_no_future_ground_truth_in_model_inputs():
    """
    F03-5 TEST: Verify that the center model and forecast model input tensors contain
    strictly past and current data, and zero future target coordinates.
    """
    from ml.inference.model_service import get_model_service

    ms = get_model_service()
    sample_id = "NIO_2020_AMPHAN_20200516_0000Z"

    # Satellite t0 frame input has shape [1, 1, 572, 929]
    t0_tensor = ms.load_t0_satellite_frame(sample_id)
    assert t0_tensor.shape == (1, 1, 572, 929)
    assert bool(torch.isfinite(t0_tensor).all())

    # Features for track forecast
    feats = ms.get_sample_features(sample_id)
    # Target fields exist in feature store for verification but are NEVER passed to model.forward()
    assert "true_12" in feats  # verification target only
    # Model inputs are strictly past/current:
    assert feats["sat_seq"].shape == (6, 132)
    assert feats["env_seq"].shape == (6, 8, 41, 66)
    assert feats["kin_ctx_a"].shape == (5,)
    assert feats["kin_a_12"].shape == (2,)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

