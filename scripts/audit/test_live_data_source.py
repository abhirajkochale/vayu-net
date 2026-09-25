"""
VAYU-NET — Live Data Source Integrity Validation Suite
======================================================
Tests the 13 foundational integrity and safety requirements for operational
real/current cyclone data adapters:

1. Provider initialization & env var overrides
2. Event discovery schema & fields
3. No credentials or secrets logged or stored
4. Timestamp validity & strict UTC parsing
5. Causal-only filtering
6. Six-frame causal discovery (t-15h to t0)
7. Missing-frame handling (WAITING_FOR_FRAMES)
8. Satellite integrity check (corrupt/empty files rejected)
9. Track-history causal filtering (t <= t0)
10. Insufficient-history behavior (<2 fixes -> INSUFFICIENT_HISTORY)
11. No future observations exposed to prediction path
12. Current endpoint schemas (GET /api/current/events, POST /api/inference/current)
13. Graceful provider failure handling (SOURCE_UNAVAILABLE)
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.backend.services.live_source_service import (
    LiveSourceService,
    BaseEventDiscoveryProvider,
    BaseTrackHistoryProvider,
    BaseSatelliteProvider,
    ReplayEventDiscoveryProvider,
    ReplayTrackHistoryProvider,
    NOAAAwsS3SatelliteProvider,
    LocalArchiveSatelliteProvider,
    get_live_source_service,
)
from apps.backend.services.satellite_ingestion_service import (
    SatelliteIngestionService,
    SatelliteIngestionError,
)
from apps.backend.services.current_event_service import CurrentEventService
from apps.backend.services.inference_pipeline import get_inference_pipeline
from fastapi.testclient import TestClient
from apps.backend.main import app

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. Provider Initialization & Configuration
# ------------------------------------------------------------------------------
def test_01_provider_initialization(monkeypatch):
    """Verifies provider initialization and environment-variable configuration."""
    monkeypatch.setenv("CYCLONE_EVENT_PROVIDER", "replay")
    monkeypatch.setenv("TRACK_HISTORY_PROVIDER", "replay")
    monkeypatch.setenv("SATELLITE_PROVIDER", "local_archive")

    service = LiveSourceService()
    assert isinstance(service.event_provider, BaseEventDiscoveryProvider)
    assert isinstance(service.track_provider, BaseTrackHistoryProvider)
    assert isinstance(service.satellite_provider, BaseSatelliteProvider)
    assert service.event_provider_name == "replay"
    assert service.satellite_provider_name == "local_archive"

    # Test NOAA AWS S3 configuration
    monkeypatch.setenv("SATELLITE_PROVIDER", "noaa_aws_s3")
    service_s3 = LiveSourceService()
    assert isinstance(service_s3.satellite_provider, NOAAAwsS3SatelliteProvider)


# ------------------------------------------------------------------------------
# 2. Event Discovery Schema
# ------------------------------------------------------------------------------
def test_02_event_schema():
    """Verifies that discovered events adhere to the mandatory schema."""
    service = LiveSourceService()
    events = service.discover_active_events()
    assert len(events) > 0, "Expected at least one discoverable event."

    required_fields = ["event_id", "name", "source", "status"]
    for ev in events:
        for field in required_fields:
            assert field in ev, f"Missing required field '{field}' in event: {ev}"


# ------------------------------------------------------------------------------
# 3. No Credentials in Logs or Source
# ------------------------------------------------------------------------------
def test_03_no_credentials_in_logs(caplog):
    """Verifies that initialization and calls never log secrets or credentials."""
    caplog.set_level(logging.DEBUG)
    service = LiveSourceService()
    _ = service.discover_active_events()

    forbidden_patterns = ["password", "secret", "aws_secret_access_key", "bearer ", "token "]
    for record in caplog.records:
        msg = record.getMessage().lower()
        for forbidden in forbidden_patterns:
            assert forbidden not in msg, f"Potential credential leak detected in log: '{msg}'"


# ------------------------------------------------------------------------------
# 4. Timestamp Validity & Strict UTC Parsing
# ------------------------------------------------------------------------------
def test_04_timestamp_validity():
    """Verifies strict ISO-8601 validation and timezone handling."""
    sat_service = SatelliteIngestionService()

    dt = sat_service.parse_utc_timestamp("2020-05-18T06:00:00Z")
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2020
    assert dt.month == 5
    assert dt.day == 18
    assert dt.hour == 6

    with pytest.raises(SatelliteIngestionError):
        sat_service.parse_utc_timestamp("invalid-date-format")


# ------------------------------------------------------------------------------
# 5. Causal-Only Filtering
# ------------------------------------------------------------------------------
def test_05_causal_only_filtering():
    """Verifies that get_available_timestamps returns only timestamps <= t0."""
    service = LiveSourceService()
    t0_utc = "2020-05-18T06:00:00+00:00"
    t0_dt = pd.to_datetime(t0_utc)

    timestamps = service.get_available_timestamps(t0_utc, window_hours=15)
    assert len(timestamps) == 6

    for ts in timestamps:
        dt = pd.to_datetime(ts)
        assert dt <= t0_dt, f"Timestamp {dt} is in the future relative to t0 {t0_dt}"


# ------------------------------------------------------------------------------
# 6. Six-Frame Discovery
# ------------------------------------------------------------------------------
def test_06_six_frame_discovery():
    """Verifies that the expected sequence has exactly 6 frames at 3-hour intervals."""
    sat_service = SatelliteIngestionService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)
    expected = sat_service.get_expected_frame_timestamps(t0_dt)

    assert len(expected) == 6
    offsets = [(expected[i][2] - expected[i - 1][2]).total_seconds() / 3600 for i in range(1, 6)]
    assert all(o == 3.0 for o in offsets), f"Expected 3-hour intervals, got {offsets}"


# ------------------------------------------------------------------------------
# 7. Missing-Frame Handling
# ------------------------------------------------------------------------------
def test_07_missing_frame_handling(tmp_path):
    """Verifies that missing frames trigger WAITING_FOR_FRAMES without crashing."""
    empty_provider = LocalArchiveSatelliteProvider(interim_dir=tmp_path)
    ok, path, msg = empty_provider.acquire_satellite_frame("2020-05-18T06:00:00+00:00")
    assert not ok
    assert path is None
    assert "FRAME_NOT_IN_LOCAL_ARCHIVE" in msg

    sat_service = SatelliteIngestionService(gridsat_dir=tmp_path)
    readiness = sat_service.check_sequence_readiness(datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc))
    assert readiness["status"] == "SOURCE_UNAVAILABLE" or readiness["status"] == "WAITING_FOR_FRAMES"
    assert not readiness["is_complete"]
    assert readiness["available_count"] == 0


# ------------------------------------------------------------------------------
# 8. Satellite Integrity Check
# ------------------------------------------------------------------------------
def test_08_satellite_integrity_check(tmp_path):
    """Verifies that corrupt or undersized satellite files are detected and rejected."""
    corrupt_file = tmp_path / "corrupt_frame.npz"
    corrupt_file.write_bytes(b"garbage content")

    sat_service = SatelliteIngestionService()
    is_valid, _ = sat_service.verify_frame_integrity(corrupt_file)
    assert not is_valid, "Corrupt file should fail integrity check."


# ------------------------------------------------------------------------------
# 9. Track-History Causal Filtering
# ------------------------------------------------------------------------------
def test_09_track_history_causal_filtering():
    """Verifies that track history fixes strictly satisfy timestamp <= t0."""
    service = LiveSourceService()
    t0_utc = "2020-05-18T06:00:00+00:00"
    t0_dt = pd.to_datetime(t0_utc)

    history = service.get_recent_track_history("NIO_2020_AMPHAN", t0_utc)
    assert len(history) >= 2

    for fix in history:
        f_dt = pd.to_datetime(fix["timestamp_utc"])
        assert f_dt <= t0_dt, f"Non-causal fix detected: {fix}"


# ------------------------------------------------------------------------------
# 10. Insufficient-History Behavior
# ------------------------------------------------------------------------------
def test_10_insufficient_history_behavior():
    """Verifies that events with fewer than 2 fixes declare INSUFFICIENT_HISTORY."""
    event_service = CurrentEventService()
    t0_dt = datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc)

    # Empty history
    kin_insufficient = event_service.derive_kinematic_inputs(
        {"event_id": "test_storm", "center_history": [], "latest_center": None},
        t0_dt,
    )
    assert kin_insufficient["track_readiness"] == "INSUFFICIENT_HISTORY"


# ------------------------------------------------------------------------------
# 11. No Future Observations Exposed
# ------------------------------------------------------------------------------
def test_11_no_future_observations():
    """Verifies that the inference pipeline quarantines future observations in current mode."""
    pipeline = get_inference_pipeline()
    res = pipeline.execute_current_pipeline("NIO_2020_AMPHAN", "2020-05-18T06:00:00+00:00")

    # Verification must be UNAVAILABLE in current mode
    assert res["verification"]["status"] == "UNAVAILABLE"
    assert "Future ground truth is strictly unavailable" in res["verification"]["message"]


# ------------------------------------------------------------------------------
# 12. Current Endpoint Schemas
# ------------------------------------------------------------------------------
def test_12_current_endpoint_schema():
    """Verifies schemas of GET /api/current/events and POST /api/inference/current."""
    # 1. GET /api/current/events
    res_events = client.get("/api/current/events")
    assert res_events.status_code == 200
    events_data = res_events.json()
    assert isinstance(events_data, list)
    if events_data:
        ev0 = events_data[0]
        for key in ["event_id", "name", "source", "status", "readiness"]:
            assert key in ev0

    # 2. POST /api/inference/current
    res_inf = client.post(
        "/api/inference/current",
        json={"event_id": "NIO_2020_AMPHAN", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    assert res_inf.status_code == 200
    inf_data = res_inf.json()
    assert "event" in inf_data
    assert "readiness" in inf_data
    assert "forecast" in inf_data
    assert "center" in inf_data
    assert "verification" in inf_data


# ------------------------------------------------------------------------------
# 13. Graceful Provider Failure Handling
# ------------------------------------------------------------------------------
def test_13_graceful_provider_failure():
    """Verifies graceful handling when an unknown event or unavailable source is requested."""
    res = client.post(
        "/api/inference/current",
        json={"event_id": "NON_EXISTENT_CYCLONE_XYZ_999", "t0_utc": "2020-05-18T06:00:00+00:00"},
    )
    # Pipeline should return 400 or 404 with structured error, not unhandled 500 crash
    assert res.status_code in (400, 404)
    data = res.json()
    assert "detail" in data
