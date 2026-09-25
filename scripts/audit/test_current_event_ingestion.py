"""
VAYU-NET — Current Event Ingestion & Readiness Audit Suite
==========================================================
Verifies the operational current-event ingestion layer, causal frame selection,
kinematic history derivation, readiness state transitions, and endpoint contracts.

Test Points:
1. current-event request schema
2. timestamp validation
3. six-frame completeness
4. no future frame selection
5. correct ordering t-15h -> t0
6. preprocessing consistency
7. readiness states (READY, WAITING_FOR_FRAMES, INSUFFICIENT_HISTORY, SOURCE_UNAVAILABLE, INFERENCE_READY)
8. missing-frame behavior
9. insufficient-history behavior
10. verification remains unavailable without future ground truth
11. historical endpoint still works (/api/inference/run)
12. current endpoint returns canonical schema (/api/inference/current)
13. no NaN/Inf across all generated tensors and results
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.services.satellite_ingestion_service import (
    SatelliteIngestionService,
    SatelliteIngestionError,
)
from apps.backend.services.current_event_service import (
    CurrentEventService,
    CurrentEventError,
)
from apps.backend.services.inference_pipeline import get_inference_pipeline
from fastapi.testclient import TestClient
from apps.backend.main import app

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. Request Schema & Parameter Validation
# ------------------------------------------------------------------------------
def test_01_current_event_request_schema():
    """Verifies that POST /api/inference/current accepts valid request and rejects invalid bodies."""
    # Missing required event_id
    res_bad = client.post("/api/inference/current", json={})
    assert res_bad.status_code == 422

    # Valid schema structure
    res_good = client.post(
        "/api/inference/current",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res_good.status_code == 200
    data = res_good.json()
    assert "event" in data
    assert "readiness" in data
    assert data["event"]["mode"] == "CURRENT_EVENT"


# ------------------------------------------------------------------------------
# 2. Timestamp Validation
# ------------------------------------------------------------------------------
def test_02_timestamp_validation():
    """Verifies strict ISO-8601 parsing and UTC normalization."""
    ingest = SatelliteIngestionService()

    # Valid UTC strings
    dt1 = ingest.parse_utc_timestamp("2023-06-11T00:00:00Z")
    assert dt1.tzinfo == timezone.utc
    assert dt1.hour == 0

    dt2 = ingest.parse_utc_timestamp("2020-05-18T06:00:00+00:00")
    assert dt2.tzinfo == timezone.utc
    assert dt2.hour == 6

    # Malformed timestamp
    with pytest.raises(SatelliteIngestionError):
        ingest.parse_utc_timestamp("not-a-timestamp")


# ------------------------------------------------------------------------------
# 3. Six-Frame Completeness
# ------------------------------------------------------------------------------
def test_03_six_frame_completeness():
    """Verifies that exactly 6 causal frames are checked and expected."""
    ingest = SatelliteIngestionService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    expected = ingest.get_expected_frame_timestamps(t0_dt)

    assert len(expected) == 6
    offsets = [item[1] for item in expected]
    assert offsets == [-15, -12, -9, -6, -3, 0]


# ------------------------------------------------------------------------------
# 4. Zero Future Frame Selection
# ------------------------------------------------------------------------------
def test_04_no_future_frame_selection():
    """Verifies that frames at t > t0 are never selected or allowed in causal window."""
    ingest = SatelliteIngestionService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    readiness = ingest.check_sequence_readiness(t0_dt)

    for f in readiness["available_frames"]:
        f_dt = datetime.fromisoformat(f["timestamp_utc"])
        assert f_dt <= t0_dt, f"Future frame detected: {f['timestamp_utc']} > {t0_dt.isoformat()}"


# ------------------------------------------------------------------------------
# 5. Chronological Ordering (t-15h -> t0)
# ------------------------------------------------------------------------------
def test_05_chronological_ordering():
    """Verifies causal frames are strictly ordered chronologically from t-15h to t0."""
    ingest = SatelliteIngestionService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    seq = ingest.ingest_causal_sequence(t0_dt)

    frames = seq["frames"]
    assert len(frames) == 6
    offsets = [f["offset_hours"] for f in frames]
    assert offsets == [-15, -12, -9, -6, -3, 0]
    assert frames[-1]["step"] == "t0"


# ------------------------------------------------------------------------------
# 6. Preprocessing Consistency
# ------------------------------------------------------------------------------
def test_06_preprocessing_consistency():
    """Verifies brightness temperature normalization and shape contracts."""
    ingest = SatelliteIngestionService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    seq = ingest.ingest_causal_sequence(t0_dt)

    t0_t = seq["t0_tensor"]
    seq_t = seq["seq_tensor"]

    assert t0_t.shape == (1, 1, 572, 929)
    assert seq_t.shape == (1, 6, 72, 116)
    assert t0_t.dtype == torch.float32
    assert seq_t.dtype == torch.float32

    # Check reasonable normalized range (typically within [-5, 5])
    assert float(t0_t.mean().abs()) < 2.0
    assert float(seq_t.mean().abs()) < 2.0


# ------------------------------------------------------------------------------
# 7. Readiness States
# ------------------------------------------------------------------------------
def test_07_readiness_states():
    """Verifies distinct operational readiness states."""
    ingest = SatelliteIngestionService()

    # 1. Complete historical rehearsal window -> READY
    t0_ready = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    res_ready = ingest.check_sequence_readiness(t0_ready)
    assert res_ready["status"] == "READY"
    assert res_ready["is_complete"] is True

    # 2. Far future date with no files -> SOURCE_UNAVAILABLE
    t0_empty = datetime(2035, 1, 1, 0, 0, tzinfo=timezone.utc)
    res_empty = ingest.check_sequence_readiness(t0_empty)
    assert res_empty["status"] == "SOURCE_UNAVAILABLE"
    assert res_empty["available_count"] == 0


# ------------------------------------------------------------------------------
# 8. Missing-Frame Behavior
# ------------------------------------------------------------------------------
def test_08_missing_frame_behavior():
    """Verifies that when frames are missing, pipeline returns WAITING_FOR_FRAMES / SOURCE_UNAVAILABLE without crashing."""
    pipeline = get_inference_pipeline()
    # Call with a timestamp that has no satellite files
    res = pipeline.execute_current_pipeline(
        event_id="NIO_2020_AMPHAN",
        t0_utc="2099-01-01T00:00:00+00:00"
    )
    assert res["readiness"]["status"] in ("SOURCE_UNAVAILABLE", "WAITING_FOR_FRAMES")
    assert res["center"] is None
    assert res["forecast"] is None
    assert res["verification"]["status"] == "UNAVAILABLE"


# ------------------------------------------------------------------------------
# 9. Insufficient History Behavior
# ------------------------------------------------------------------------------
def test_09_insufficient_history_behavior():
    """Verifies that an event with < 2 historical fixes returns INSUFFICIENT_HISTORY for track."""
    event_svc = CurrentEventService()
    t0_dt = datetime(2023, 6, 11, 0, 0, tzinfo=timezone.utc)

    # Event with only 1 fix at t0
    single_fix_event = {
        "event_id": "TEST_STORM_SINGLE_FIX",
        "storm_name": "SINGLE_FIX",
        "observation_time": t0_dt.isoformat(),
        "center_history": [
            {"timestamp_utc": t0_dt.isoformat(), "lat": 15.0, "lon": 85.0}
        ],
        "latest_center": {"lat": 15.0, "lon": 85.0}
    }

    kin_info = event_svc.derive_kinematic_inputs(single_fix_event, t0_dt)
    assert kin_info["track_readiness"] == "INSUFFICIENT_HISTORY"
    assert "at least 2 causal observed fixes" in kin_info["reason"]
    assert kin_info["motion_ctx"] is None


# ------------------------------------------------------------------------------
# 10. Verification Remains Unavailable Without Future Ground Truth
# ------------------------------------------------------------------------------
def test_10_verification_strictly_unavailable_in_current_mode():
    """Verifies that verification is strictly marked UNAVAILABLE in current event mode."""
    res = client.post(
        "/api/inference/current",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["verification"]["status"] == "UNAVAILABLE"
    assert "Future ground truth is strictly unavailable" in data["verification"]["message"]
    assert data["verification"].get("metrics") is None


# ------------------------------------------------------------------------------
# 11. Historical Endpoint Coexistence
# ------------------------------------------------------------------------------
def test_11_historical_endpoint_still_works():
    """Verifies that POST /api/inference/run remains 100% operational alongside /api/inference/current."""
    res = client.post(
        "/api/inference/run",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res.status_code == 200
    data = res.json()
    # In historical mode, verification IS available
    assert data["verification"]["status"] == "AVAILABLE"
    assert data["verification"]["horizons"]["plus_12h"]["dpe_km"] > 0
    assert data["forecast"]["plus_12h"]["lat"] is not None


# ------------------------------------------------------------------------------
# 12. Current Endpoint Canonical Schema
# ------------------------------------------------------------------------------
def test_12_current_endpoint_canonical_schema():
    """Verifies that POST /api/inference/current produces complete canonical dictionary."""
    res = client.post(
        "/api/inference/current",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res.status_code == 200
    data = res.json()

    required_keys = [
        "event", "observation", "readiness", "center", "intensity",
        "wind", "forecast", "uncertainty", "verification", "analogs",
        "saliency", "secondary_observation", "metadata"
    ]
    for key in required_keys:
        assert key in data, f"Missing canonical key '{key}' in current inference response."

    assert data["readiness"]["status"] in ("INFERENCE_READY", "READY")
    assert data["center"]["ai_lat"] is not None
    assert data["intensity"]["category"] is not None
    assert data["wind"]["wind_kt"] is not None


# ------------------------------------------------------------------------------
# 13. Numerical Stability (No NaN / Inf)
# ------------------------------------------------------------------------------
def test_13_no_nan_or_inf():
    """Verifies that all tensors and numeric outputs in the canonical result are finite."""
    res = client.post(
        "/api/inference/current",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res.status_code == 200
    data = res.json()

    def check_finite(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_finite(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                check_finite(v, f"{path}[{idx}]")
        elif isinstance(obj, float):
            assert not (np.isnan(obj) or np.isinf(obj)), f"Non-finite value {obj} at {path}"

    check_finite(data)
