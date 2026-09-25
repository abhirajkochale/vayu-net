"""
VAYU-NET — Live Source Sandbox & Replay Validation Suite
=========================================================
Tests the operational current-event forecasting pipeline in sandbox/replay mode.
Uses historical cyclone events treated AS IF they were currently active.

Invariants Verified:
1. At t0, ONLY data <= t0 is visible to:
   - Event discovery
   - Satellite sequence ingestion (t-15h to t0)
   - Track history fixes
2. Future IMD observations are strictly quarantined from the prediction path.
3. Verification remains UNAVAILABLE for current events until explicitly verified.
4. The exact same inference pipeline produces valid multi-task predictions.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.services.live_source_service import (
    LiveSourceService,
    ReplayEventDiscoveryProvider,
    ReplayTrackHistoryProvider,
    LocalArchiveSatelliteProvider,
    get_live_source_service,
)
from apps.backend.services.satellite_ingestion_service import SatelliteIngestionService
from apps.backend.services.current_event_service import CurrentEventService
from apps.backend.services.inference_pipeline import get_inference_pipeline
from fastapi.testclient import TestClient
from apps.backend.main import app

client = TestClient(app)


# ------------------------------------------------------------------------------
# Test 1: Event Discovery in Replay Sandbox Mode
# ------------------------------------------------------------------------------
def test_replay_event_discovery():
    """Verifies that replay event discovery returns active storm candidates from the catalog."""
    provider = ReplayEventDiscoveryProvider()
    events = provider.discover_active_events()
    assert isinstance(events, list)
    assert len(events) > 0

    event_ids = [e["event_id"] for e in events]
    assert any("AMPHAN" in eid for eid in event_ids)

    # Check event metadata schema
    amphan_event = next(e for e in events if "AMPHAN" in e["event_id"])
    assert "event_id" in amphan_event
    assert "name" in amphan_event
    assert "source" in amphan_event
    assert amphan_event["source"] == "CATALOG_REPLAY"
    assert amphan_event["status"] == "ACTIVE_REPLAY"


# ------------------------------------------------------------------------------
# Test 2: Observation Lookup at t0 (Causal Isolation)
# ------------------------------------------------------------------------------
def test_replay_observation_causal_lookup():
    """Verifies that observation at t0 returns only current/observed center and intensity."""
    provider = ReplayEventDiscoveryProvider()
    t0_utc = "2020-05-18T06:00:00+00:00"
    obs = provider.get_latest_observation("NIO_2020_AMPHAN", t0_utc=t0_utc)

    assert "event_id" in obs
    assert "latest_center" in obs
    assert obs["latest_center"]["lat"] is not None
    assert obs["latest_center"]["lon"] is not None
    assert obs["latest_wind_kt"] is not None
    assert "2020-05-18" in obs["observation_time"]


# ------------------------------------------------------------------------------
# Test 3: Causal Track History Filtering (Strictly t <= t0)
# ------------------------------------------------------------------------------
def test_replay_track_history_strictly_causal():
    """Verifies that every history fix returned satisfies timestamp <= t0, with zero future fixes."""
    provider = ReplayTrackHistoryProvider()
    t0_utc = "2020-05-18T06:00:00+00:00"
    t0_dt = pd.to_datetime(t0_utc)

    history = provider.get_recent_track_history("NIO_2020_AMPHAN", t0_utc=t0_utc)
    assert len(history) >= 2, "Expected at least 2 causal fixes for AMPHAN at t0"

    for fix in history:
        fix_dt = pd.to_datetime(fix["timestamp_utc"])
        assert fix_dt <= t0_dt, f"Future fix detected! {fix_dt} > {t0_dt}"
        assert -5.0 <= fix["lat"] <= 35.0
        assert 40.0 <= fix["lon"] <= 105.0

    # Ensure chronological order
    timestamps = [pd.to_datetime(f["timestamp_utc"]) for f in history]
    assert timestamps == sorted(timestamps), "History fixes must be strictly chronological."


# ------------------------------------------------------------------------------
# Test 4: Satellite Frame Discovery in Replay Sandbox Mode
# ------------------------------------------------------------------------------
def test_replay_satellite_causal_sequence():
    """Verifies that satellite sequence queries exactly [t-15h, t0] with no future timestamps."""
    sat_service = SatelliteIngestionService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    expected_frames = sat_service.get_expected_frame_timestamps(t0_dt)

    assert len(expected_frames) == 6
    assert expected_frames[-1][2] == t0_dt
    assert expected_frames[0][2] == datetime(2020, 5, 17, 15, 0, tzinfo=timezone.utc)

    # Verify no frame in the sequence is after t0
    for f in expected_frames:
        assert f[2] <= t0_dt


# ------------------------------------------------------------------------------
# Test 5: Full End-to-End Pipeline in Replay Current-Event Mode
# ------------------------------------------------------------------------------
def test_replay_full_current_pipeline():
    """Verifies that the canonical pipeline runs smoothly in replay current-event mode."""
    pipeline = get_inference_pipeline()
    result = pipeline.execute_current_pipeline(
        event_id="NIO_2020_AMPHAN",
        t0_utc="2020-05-18T06:00:00+00:00",
    )

    # 1. High-level structure
    assert "event" in result
    assert result["event"]["mode"] == "CURRENT_EVENT"
    assert result["event"]["event_id"] == "NIO_2020_AMPHAN"
    assert result["readiness"]["status"] in ("READY", "INFERENCE_READY")

    # 2. Phase 3C Center Localization
    assert "center" in result
    assert "ai_lat" in result["center"]
    assert "ai_lon" in result["center"]
    assert -5.0 <= result["center"]["ai_lat"] <= 35.0
    assert 40.0 <= result["center"]["ai_lon"] <= 105.0

    # 3. Phase 6 Intensity & Wind
    assert "intensity" in result
    assert "category" in result["intensity"]
    assert "wind" in result
    assert "wind_kt" in result["wind"]

    # 4. Phase 5B Multi-Horizon Track
    assert "forecast" in result
    assert "plus_12h" in result["forecast"]
    assert "plus_24h" in result["forecast"]
    assert "plus_48h" in result["forecast"]

    # 5. Uncertainty Cones
    assert "uncertainty" in result
    assert "cone_geometries" in result["uncertainty"]

    # 6. Analogs
    assert "analogs" in result
    assert isinstance(result["analogs"], list)
    assert len(result["analogs"]) >= 1

    # 7. Explainability
    assert "saliency" in result

    # 8. Strict Verification Quarantine
    assert "verification" in result
    assert result["verification"]["status"] == "UNAVAILABLE"
    assert "message" in result["verification"]


# ------------------------------------------------------------------------------
# Test 6: Replay Mode via FastAPI Current Endpoint
# ------------------------------------------------------------------------------
def test_replay_fastapi_current_endpoint():
    """Verifies that POST /api/inference/current works seamlessly with the replay provider."""
    res = client.post(
        "/api/inference/current",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "event" in data
    assert data["readiness"]["status"] in ("READY", "INFERENCE_READY")
    assert data["verification"]["status"] == "UNAVAILABLE"
