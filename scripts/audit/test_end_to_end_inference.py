"""VAYU-NET — End-to-End Inference Pipeline Audit & Verification Suite
=====================================================================
SIH 2026 Problem Statement 26070
India Meteorological Department (IMD) / Ministry of Earth Sciences (MoES)

Tests the operational inference pipeline on known historical cyclones (e.g. AMPHAN, FANI, TAUKTAE).
Verifies:
1. Event loads from catalog
2. Six causal GridSat frames exist
3. Inference completes end-to-end
4. Center localization exists and is finite
5. Intensity classification exists and is valid IMD category
6. Wind speed exists and is finite
7. +12h forecast coordinates exist
8. +24h forecast coordinates exist
9. +48h forecast coordinates exist
10. Empirical uncertainty exists or returns explicit unavailable status
11. Verification does not modify predictions
12. Top-2 analogs returned from TRAIN archive
13. Saliency returned or explicit unavailable status
14. IMERG secondary context loads cleanly
15. Final response strictly matches canonical schema contract
16. Zero NaN/Inf anywhere in response
17. No future frame enters prediction path
18. Handles scenario where future verification ground truth is unavailable
"""

import math
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.services.inference_pipeline import get_inference_pipeline


@pytest.fixture(scope="module")
def pipeline():
    return get_inference_pipeline()


def check_no_nan_or_inf(data, path=""):
    """Recursively checks that no float is NaN or Inf."""
    if isinstance(data, dict):
        for k, v in data.items():
            check_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            check_no_nan_or_inf(item, f"{path}[{idx}]")
    elif isinstance(data, float):
        assert not math.isnan(data), f"NaN found at {path}"
        assert not math.isinf(data), f"Inf found at {path}"


def test_1_event_loads_from_catalog(pipeline):
    events = pipeline.list_events()
    assert len(events) > 0, "No events returned from catalog"
    amphan = next((e for e in events if "AMPHAN" in e["storm_name"]), None)
    assert amphan is not None, "AMPHAN event not found in catalog"
    assert amphan["year"] == 2020


def test_2_six_causal_gridsat_frames_exist(pipeline):
    event_details = pipeline.get_event_details("AMPHAN")
    assert event_details["observations_count"] > 0
    first_obs = event_details["observations"][0]
    sample_id = first_obs["sample_id"]

    row = pipeline._resolve_sample_row("AMPHAN", first_obs["t0_utc"])
    frames = pipeline._load_and_validate_gridsat_sequence(row)
    assert len(frames) == 6, f"Expected 6 frames, got {len(frames)}"
    assert frames[0]["step"] == "t_minus_15h"
    assert frames[5]["step"] == "t0"
    for f in frames:
        assert Path(f["file_path"]).exists(), f"Frame file missing: {f['file_path']}"


def test_3_inference_completes_end_to_end(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    assert result is not None
    assert isinstance(result, dict)


def test_4_center_exists_and_finite(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    center = result["center"]
    assert "ai_lat" in center and "ai_lon" in center
    assert -5.0 <= center["ai_lat"] <= 35.0
    assert 40.0 <= center["ai_lon"] <= 105.0
    assert center["confidence"] > 0.0


def test_5_intensity_exists_and_valid_imd_category(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    intensity = result["intensity"]
    valid_cats = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]
    assert intensity["category"] in valid_cats
    assert 0 <= intensity["category_index"] <= 6
    assert 0.0 <= intensity["confidence"] <= 1.0


def test_6_wind_exists_and_finite(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    wind = result["wind"]
    assert "wind_kt" in wind
    assert 10.0 <= wind["wind_kt"] <= 200.0


def test_7_8_9_multi_horizon_forecast_exists(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    fc = result["forecast"]
    for horizon in ["plus_12h", "plus_24h", "plus_48h"]:
        assert horizon in fc
        pt = fc[horizon]
        assert "lat" in pt and "lon" in pt
        assert -5.0 <= pt["lat"] <= 35.0
        assert 40.0 <= pt["lon"] <= 105.0


def test_10_uncertainty_exists_or_returns_explicit_unavailable(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    u = result["uncertainty"]
    if u.get("status") != "UNAVAILABLE":
        assert u["plus_12h_km"] is not None and u["plus_12h_km"] > 0
        assert u["plus_24h_km"] is not None and u["plus_24h_km"] > 0
        assert u["plus_48h_km"] is not None and u["plus_48h_km"] > 0
        assert "cone_geometries" in u


def test_11_verification_does_not_modify_predictions(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    fc_lat = result["forecast"]["plus_12h"]["lat"]
    fc_lon = result["forecast"]["plus_12h"]["lon"]

    # Re-verify independently
    verif = result["verification"]
    assert verif["status"] in ("AVAILABLE", "UNAVAILABLE")
    if verif["status"] == "AVAILABLE":
        verif_pred = verif["horizons"]["plus_12h"]["predicted"]
        assert verif_pred["lat"] == fc_lat
        assert verif_pred["lon"] == fc_lon


def test_12_top_2_analogs_returned_from_train_archive(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    analogs = result["analogs"]
    assert len(analogs) == 2, f"Expected exactly 2 analogs, got {len(analogs)}"
    for a in analogs:
        assert a["split"] == "TRAIN", f"Analog {a['storm_name']} must be from TRAIN split"
        assert "AMPHAN" not in a["storm_name"], "Self-match must be excluded"
        assert a["distance"] >= 0.0


def test_13_saliency_returned_or_explicit_unavailable(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    sal = result["saliency"]
    assert sal["status"] in ("AVAILABLE", "UNAVAILABLE")
    if sal["status"] == "AVAILABLE":
        assert sal["method"] == "Grad-CAM"
        assert "heatmap_url" in sal
        assert "overlay_url" in sal


def test_14_imerg_secondary_context_loads(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    imerg = result["secondary_observation"]
    assert imerg["source"] == "NASA GPM IMERG Final Run V07B"
    assert imerg["display_ready"] is True
    assert len(imerg["frames"]) == 6


def test_15_final_response_matches_canonical_schema(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    required_top_keys = [
        "event", "observation", "center", "intensity", "wind",
        "forecast", "uncertainty", "verification", "analogs", "saliency",
        "secondary_observation", "metadata"
    ]
    for key in required_top_keys:
        assert key in result, f"Missing required canonical key: '{key}'"


def test_16_no_nan_or_inf_in_response(pipeline):
    result = pipeline.run_inference("AMPHAN", "2020-05-18T06:00:00+00:00")
    check_no_nan_or_inf(result)


def test_17_no_future_frame_enters_prediction(pipeline):
    row = pipeline._resolve_sample_row("AMPHAN", "2020-05-18T06:00:00+00:00")
    t0_dt = row["t0"]
    frames = pipeline._load_and_validate_gridsat_sequence(row)
    for f in frames:
        f_dt = f["timestamp_utc"]
        assert f_dt <= t0_dt, f"Observation frame timestamp {f_dt} is in future relative to t0 {t0_dt}"


def test_18_verification_unavailable_scenario(pipeline):
    """
    Simulates scenario where future ground-truth points are missing or untracked
    (e.g. final observation before dissipation).
    Verifies that pipeline does not crash and returns status='UNAVAILABLE'.
    """
    synthetic_fc = {
        "plus_12h": {"lat": 25.0, "lon": 90.0},
        "plus_24h": {"lat": 27.0, "lon": 92.0},
        "plus_48h": {"lat": 30.0, "lon": 95.0},
    }
    # Query non-existent sample id
    verif = pipeline.verification_service.verify_forecast("NON_EXISTENT_SAMPLE_ID", synthetic_fc)
    assert verif["status"] == "UNAVAILABLE"
    assert "reason" in verif
