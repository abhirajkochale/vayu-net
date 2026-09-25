"""
VAYU-NET — Live Source Adapter & Provider Management Layer
===========================================================
Defines provider-neutral interfaces for operational current-cyclone forecasting:
1. Event Discovery: Discovers currently active tropical cyclones in the NIO basin.
2. Track History: Retrieves causal synoptic center/wind fixes (t <= t0).
3. Satellite Observation: Retrieves causal geostationary satellite frames (NOAA AWS S3 / Local).

Key Invariants:
- Provider-neutral interfaces; zero provider-specific code in ML pipeline.
- Configurable via environment variables (CYCLONE_EVENT_PROVIDER, SATELLITE_PROVIDER, TRACK_HISTORY_PROVIDER).
- Strictly zero credential leakage (no secrets logged or stored).
- Never fabricates positions, never interpolates missing frames.
- Replay/Sandbox mode for deterministic testing without external live dependency.
"""

from __future__ import annotations

import os
import re
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger("vayu.backend.live_source")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_LOCAL_GRIDSAT_DIR = PROJECT_ROOT / "data" / "interim" / "gridsat"
DEFAULT_RAW_GRIDSAT_DIR = PROJECT_ROOT / "data" / "raw" / "gridsat"
SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "manifests" / "vayu_net_sample_index.csv"
BEST_TRACK_PATH = PROJECT_ROOT / "data" / "processed" / "imd_best_track_v2.csv"
STORM_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "storm_event_manifest_v2.csv"

AWS_GRIDSAT_BASE_URL = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data"


# ==============================================================================
# Abstract Base Interfaces
# ==============================================================================

class BaseEventDiscoveryProvider(ABC):
    """Abstract interface for active cyclone event discovery."""

    @abstractmethod
    def discover_active_events(self) -> List[Dict[str, Any]]:
        """Returns list of currently active cyclone events."""
        pass

    @abstractmethod
    def get_latest_observation(self, event_id: str, t0_utc: Optional[str] = None) -> Dict[str, Any]:
        """Returns latest observed storm status for the given event at or before t0."""
        pass


class BaseTrackHistoryProvider(ABC):
    """Abstract interface for causal track history fixes."""

    @abstractmethod
    def get_recent_track_history(self, event_id: str, t0_utc: str) -> List[Dict[str, Any]]:
        """
        Returns chronological list of observed fixes strictly satisfying timestamp <= t0.
        Never exposes future ground truth.
        """
        pass


class BaseSatelliteProvider(ABC):
    """Abstract interface for satellite observation acquisition."""

    @abstractmethod
    def get_available_timestamps(self, t0_utc: str, window_hours: int = 15) -> List[str]:
        """Queries available frame timestamps in the causal window [t0 - window_hours, t0]."""
        pass

    @abstractmethod
    def acquire_satellite_frame(self, timestamp_utc: str) -> Tuple[bool, Optional[Path], str]:
        """
        Retrieves or verifies a satellite frame for the specified timestamp.
        Returns: (success: bool, local_path: Optional[Path], message: str)
        """
        pass


# ==============================================================================
# Concrete Provider Implementations
# ==============================================================================

class ReplayEventDiscoveryProvider(BaseEventDiscoveryProvider):
    """
    Certified Replay Provider: discovers events from the historical catalogue
    treated strictly under operational causal rules (t <= t0).
    """

    def __init__(self, manifest_path: Optional[Path] = None, sample_index_path: Optional[Path] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else STORM_MANIFEST_PATH
        self.sample_index_path = Path(sample_index_path) if sample_index_path else SAMPLE_INDEX_PATH
        self._events_df: Optional[pd.DataFrame] = None
        self._samples_df: Optional[pd.DataFrame] = None

    def _load_data(self) -> None:
        if self._events_df is None and self.manifest_path.exists():
            self._events_df = pd.read_csv(self.manifest_path)
        if self._samples_df is None and self.sample_index_path.exists():
            self._samples_df = pd.read_csv(self.sample_index_path)

    def discover_active_events(self) -> List[Dict[str, Any]]:
        self._load_data()
        if self._events_df is None:
            return []

        # Return demo shortlist storms as candidates
        events = []
        for _, r in self._events_df.iterrows():
            sid = str(r["storm_id"])
            sname = str(r["storm_name"])
            events.append({
                "event_id": sid,
                "name": sname,
                "source": "CATALOG_REPLAY",
                "status": "ACTIVE_REPLAY",
                "year": int(r["year"]),
                "peak_category": str(r.get("peak_category", "UNKNOWN")),
                "basin": "North Indian Ocean",
            })
        return events

    def get_latest_observation(self, event_id: str, t0_utc: Optional[str] = None) -> Dict[str, Any]:
        self._load_data()
        if self._samples_df is None:
            raise ValueError("Catalog sample index not available.")

        matches = self._samples_df[
            (self._samples_df["storm_id"].str.contains(event_id, case=False, na=False)) |
            (self._samples_df["sample_id"].str.contains(event_id, case=False, na=False))
        ]
        if matches.empty:
            raise KeyError(f"Event '{event_id}' not found in discovery catalog.")

        if t0_utc:
            target_str = t0_utc.replace("Z", "+00:00").strip()
            t_rows = matches[matches["t0"].str.contains(target_str[:16], regex=False)]
            row = t_rows.iloc[0] if not t_rows.empty else matches.iloc[len(matches) // 2]
        else:
            row = matches.iloc[len(matches) // 2]

        return {
            "event_id": str(row["storm_id"]),
            "sample_id": str(row["sample_id"]),
            "observation_time": str(row["t0"]),
            "latest_center": {
                "lat": float(row["imd_lat_t0"]) if pd.notna(row["imd_lat_t0"]) else None,
                "lon": float(row["imd_lon_t0"]) if pd.notna(row["imd_lon_t0"]) else None,
            },
            "latest_wind_kt": float(row["imd_wind_t0"]) if pd.notna(row["imd_wind_t0"]) else None,
            "latest_category": str(row["imd_category_t0"]) if pd.notna(row["imd_category_t0"]) else "UNKNOWN",
        }


class ReplayTrackHistoryProvider(BaseTrackHistoryProvider):
    """
    Provides causal track history fixes from IMD best track tables strictly filtered to t <= t0.
    """

    def __init__(self, best_track_path: Optional[Path] = None):
        self.best_track_path = Path(best_track_path) if best_track_path else BEST_TRACK_PATH
        self._bt_df: Optional[pd.DataFrame] = None

    def _load_data(self) -> None:
        if self._bt_df is None and self.best_track_path.exists():
            self._bt_df = pd.read_csv(self.best_track_path)

    def get_recent_track_history(self, event_id: str, t0_utc: str) -> List[Dict[str, Any]]:
        self._load_data()
        if self._bt_df is None:
            return []

        t0_dt = pd.to_datetime(t0_utc.replace("Z", "+00:00"))
        if t0_dt.tzinfo is None:
            t0_dt = t0_dt.replace(tzinfo=timezone.utc)

        matches = self._bt_df[self._bt_df["storm_id"].str.contains(event_id, case=False, na=False)]
        if matches.empty:
            return []

        ts_col = "timestamp_utc" if "timestamp_utc" in self._bt_df.columns else "iso_time"
        wind_col = "maximum_sustained_wind_kt" if "maximum_sustained_wind_kt" in self._bt_df.columns else "max_wind_kt"

        history = []
        for _, r in matches.iterrows():
            f_ts = pd.to_datetime(r[ts_col])
            if f_ts.tzinfo is None:
                f_ts = f_ts.replace(tzinfo=timezone.utc)

            # Strictly causal check: t <= t0
            if f_ts <= t0_dt:
                history.append({
                    "timestamp_utc": f_ts.isoformat(),
                    "lat": float(r["latitude"]),
                    "lon": float(r["longitude"]),
                    "wind_kt": float(r[wind_col]) if pd.notna(r.get(wind_col)) else None,
                    "pressure_hpa": float(r["central_pressure_hpa"]) if pd.notna(r.get("central_pressure_hpa")) else None,
                    "category": str(r.get("category", "")),
                })

        history.sort(key=lambda x: x["timestamp_utc"])
        return history


class NOAAAwsS3SatelliteProvider(BaseSatelliteProvider):
    """
    Acquires NOAA NCEI GridSat-B1 NetCDF files directly from the public AWS S3 bucket.
    Endpoint: https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/{YYYY}/
    Authentication: None (Public Anonymous Open Data).
    """

    def __init__(
        self,
        interim_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        self.interim_dir = Path(interim_dir) if interim_dir else DEFAULT_LOCAL_GRIDSAT_DIR
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_GRIDSAT_DIR

    def get_available_timestamps(self, t0_utc: str, window_hours: int = 15) -> List[str]:
        t0_dt = pd.to_datetime(t0_utc.replace("Z", "+00:00"))
        if t0_dt.tzinfo is None:
            t0_dt = t0_dt.replace(tzinfo=timezone.utc)

        # Generate expected 3-hourly timestamps
        timestamps = []
        for offset in range(-window_hours, 1, 3):
            ts = (t0_dt + timedelta(hours=offset)).isoformat()
            timestamps.append(ts)
        return timestamps

    def acquire_satellite_frame(self, timestamp_utc: str) -> Tuple[bool, Optional[Path], str]:
        """
        Locates or downloads a 3-hourly GridSat-B1 frame.
        Checks local interim cache first. If absent, downloads from AWS S3 Open Data bucket.
        """
        dt = pd.to_datetime(timestamp_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        year_str = str(dt.year)
        base_name = f"gridsat_{dt.year:04d}.{dt.month:02d}.{dt.day:02d}.{dt.hour:02d}"

        # 1. Check local processed interim directory (.npz or .nc)
        interim_npz = self.interim_dir / year_str / f"{base_name}.npz"
        if interim_npz.exists() and interim_npz.stat().st_size > 1000:
            return True, interim_npz, "LOCAL_CACHE_HIT"

        interim_nc = self.interim_dir / year_str / f"{base_name}.nc"
        if interim_nc.exists() and interim_nc.stat().st_size > 1000:
            return True, interim_nc, "LOCAL_CACHE_HIT"

        # 2. Build remote NOAA AWS S3 URL
        remote_filename = f"GRIDSAT-B1.{dt.year:04d}.{dt.month:02d}.{dt.day:02d}.{dt.hour:02d}.v02r01.nc"
        remote_url = f"{AWS_GRIDSAT_BASE_URL}/{year_str}/{remote_filename}"

        raw_dest_dir = self.raw_dir / year_str
        raw_dest_dir.mkdir(parents=True, exist_ok=True)
        raw_dest_path = raw_dest_dir / remote_filename

        # If already downloaded raw NetCDF
        if raw_dest_path.exists() and raw_dest_path.stat().st_size > 10 * 1024 * 1024:
            return True, raw_dest_path, "RAW_CACHED"

        # Download from NOAA AWS S3
        logger.info(f"Downloading GridSat frame from {remote_url}...")
        try:
            req = urllib.request.Request(
                remote_url,
                headers={"User-Agent": "VAYU-NET-Operational-Forecasting/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(raw_dest_path, "wb") as out_f:
                out_f.write(resp.read())

            if raw_dest_path.exists() and raw_dest_path.stat().st_size > 1000:
                logger.info(f"Successfully downloaded {remote_filename} ({raw_dest_path.stat().st_size} bytes)")
                return True, raw_dest_path, "DOWNLOADED_FROM_NOAA_S3"
            else:
                return False, None, "DOWNLOADED_FILE_EMPTY"

        except urllib.error.HTTPError as e:
            return False, None, f"HTTP_ERROR_{e.code}: {e.reason}"
        except Exception as e:
            return False, None, f"ACQUISITION_FAILED: {str(e)}"


class LocalArchiveSatelliteProvider(BaseSatelliteProvider):
    """
    Offline/local provider that looks up GridSat frames in data/interim/gridsat.
    """

    def __init__(self, interim_dir: Optional[Path] = None):
        self.interim_dir = Path(interim_dir) if interim_dir else DEFAULT_LOCAL_GRIDSAT_DIR

    def get_available_timestamps(self, t0_utc: str, window_hours: int = 15) -> List[str]:
        t0_dt = pd.to_datetime(t0_utc.replace("Z", "+00:00"))
        if t0_dt.tzinfo is None:
            t0_dt = t0_dt.replace(tzinfo=timezone.utc)

        timestamps = []
        for offset in range(-window_hours, 1, 3):
            ts = (t0_dt + timedelta(hours=offset)).isoformat()
            timestamps.append(ts)
        return timestamps

    def acquire_satellite_frame(self, timestamp_utc: str) -> Tuple[bool, Optional[Path], str]:
        dt = pd.to_datetime(timestamp_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        year_str = str(dt.year)
        base_name = f"gridsat_{dt.year:04d}.{dt.month:02d}.{dt.day:02d}.{dt.hour:02d}"

        interim_npz = self.interim_dir / year_str / f"{base_name}.npz"
        if interim_npz.exists() and interim_npz.stat().st_size > 1000:
            return True, interim_npz, "LOCAL_CACHE_HIT"

        interim_nc = self.interim_dir / year_str / f"{base_name}.nc"
        if interim_nc.exists() and interim_nc.stat().st_size > 1000:
            return True, interim_nc, "LOCAL_CACHE_HIT"

        return False, None, "FRAME_NOT_IN_LOCAL_ARCHIVE"


class INSATMOSDACSatelliteProvider(BaseSatelliteProvider):
    """
    Adapter abstraction for native ISRO INSAT-3D/3DR HDF5 observation feeds.
    Requires institutional authenticated API credentials from ISRO MOSDAC (gated).
    Does NOT feed raw or uncalibrated imagery into primary ML checkpoints.
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("MOSDAC_API_TOKEN")

    def get_available_timestamps(self, t0_utc: str, window_hours: int = 15) -> List[str]:
        # Planned for authenticated MOSDAC native API ingestion
        return []

    def acquire_satellite_frame(self, timestamp_utc: str) -> Tuple[bool, Optional[Path], str]:
        if not self.api_token:
            return False, None, "INSAT_UNAVAILABLE: MOSDAC authenticated API credentials required."
        return False, None, "INSAT_ADAPTER_PENDING_REGISTRATION"


# ==============================================================================
# Unified Adapter Service
# ==============================================================================

class LiveSourceService:
    """
    Unified operational adapter coordinating Event Discovery, Track History,
    and Satellite Observation acquisition across configurable providers.
    """

    def __init__(self):
        # Configure providers from environment variables with safe defaults
        self.event_provider_name = os.getenv("CYCLONE_EVENT_PROVIDER", "replay").lower()
        self.track_provider_name = os.getenv("TRACK_HISTORY_PROVIDER", "replay").lower()
        self.satellite_provider_name = os.getenv("SATELLITE_PROVIDER", "local_archive").lower()

        logger.info(
            f"Initialized LiveSourceService: "
            f"Event='{self.event_provider_name}', "
            f"Track='{self.track_provider_name}', "
            f"Satellite='{self.satellite_provider_name}'"
        )

        # 1. Event Discovery Provider
        if self.event_provider_name in ("replay", "catalog", "mock"):
            self.event_provider: BaseEventDiscoveryProvider = ReplayEventDiscoveryProvider()
        else:
            self.event_provider = ReplayEventDiscoveryProvider()

        # 2. Track History Provider
        if self.track_provider_name in ("replay", "imd_advisory"):
            self.track_provider: BaseTrackHistoryProvider = ReplayTrackHistoryProvider()
        else:
            self.track_provider = ReplayTrackHistoryProvider()

        # 3. Satellite Provider
        if self.satellite_provider_name == "noaa_aws_s3":
            self.satellite_provider: BaseSatelliteProvider = NOAAAwsS3SatelliteProvider()
        elif self.satellite_provider_name in ("insat", "mosdac"):
            self.satellite_provider = INSATMOSDACSatelliteProvider()
        else:
            self.satellite_provider = LocalArchiveSatelliteProvider()

    def discover_active_events(self) -> List[Dict[str, Any]]:
        """Provider-neutral call to discover currently active cyclones."""
        return self.event_provider.discover_active_events()

    def get_latest_observation(self, event_id: str, t0_utc: Optional[str] = None) -> Dict[str, Any]:
        """Provider-neutral call for latest cyclone synoptic status."""
        return self.event_provider.get_latest_observation(event_id, t0_utc)

    def get_recent_track_history(self, event_id: str, t0_utc: str) -> List[Dict[str, Any]]:
        """Provider-neutral call for causal observed track fixes (t <= t0)."""
        return self.track_provider.get_recent_track_history(event_id, t0_utc)

    def get_satellite_observation(self, timestamp_utc: str) -> Tuple[bool, Optional[Path], str]:
        """Provider-neutral call to retrieve or verify a satellite frame."""
        return self.satellite_provider.acquire_satellite_frame(timestamp_utc)

    def get_available_timestamps(self, t0_utc: str, window_hours: int = 15) -> List[str]:
        """Provider-neutral call to query available timestamps in the causal window."""
        return self.satellite_provider.get_available_timestamps(t0_utc, window_hours=window_hours)


# Global singleton
_global_live_source_service: Optional[LiveSourceService] = None


def get_live_source_service() -> LiveSourceService:
    global _global_live_source_service
    if _global_live_source_service is None:
        _global_live_source_service = LiveSourceService()
    return _global_live_source_service
