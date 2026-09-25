"""
VAYU-NET — Current Event Discovery & Kinematic History Service
==============================================================
Responsible for managing active / current cyclone event registries,
tracking observed synoptic fixes (center coordinates, wind, pressure),
and deriving causal kinematic motion anchors for Phase 5B.

Key Invariants:
1. Event Discovery is strictly isolated from Satellite Ingestion.
2. Does NOT invent synthetic cyclone tracks.
3. Evaluates only historical / current observations (t <= t0).
4. If fewer than 2 observed center fixes exist, declares
   track_readiness = "INSUFFICIENT_HISTORY" rather than fabricating motion.
"""

import math
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import torch

logger = logging.getLogger("vayu.backend.current_event")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "manifests" / "vayu_net_sample_index.csv"
BEST_TRACK_PATH = PROJECT_ROOT / "data" / "processed" / "imd_best_track_v2.csv"

# Earth and geographic domain constants
R_EARTH_KM = 6371.0
DEG_TO_RAD = math.pi / 180.0
LAT_MIN, LAT_SPAN = -5.0, 40.0
LON_MIN, LON_SPAN = 40.0, 65.0


class CurrentEventError(Exception):
    """Exception raised for current event service errors."""
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class CurrentEventService:
    """
    Manages current cyclone event discovery, track history, and kinematic preparation.
    """

    def __init__(
        self,
        sample_index_path: Optional[Path] = None,
        best_track_path: Optional[Path] = None,
    ):
        self.sample_index_path = Path(sample_index_path) if sample_index_path else SAMPLE_INDEX_PATH
        self.best_track_path = Path(best_track_path) if best_track_path else BEST_TRACK_PATH

        # In-memory registry for live / registered current events
        self._live_events: Dict[str, Dict[str, Any]] = {}
        self._sample_index_df: Optional[pd.DataFrame] = None
        self._best_track_df: Optional[pd.DataFrame] = None

    def _ensure_catalog_loaded(self) -> None:
        """Loads historical catalogue for fallback/demonstration support."""
        if self._sample_index_df is None and self.sample_index_path.exists():
            self._sample_index_df = pd.read_csv(self.sample_index_path)
        if self._best_track_df is None and self.best_track_path.exists():
            self._best_track_df = pd.read_csv(self.best_track_path)

    def register_current_event(
        self,
        event_id: str,
        storm_name: str,
        observation_time: str,
        source: str = "MANUAL_ENTRY",
        center_history: Optional[List[Dict[str, Any]]] = None,
        latest_center: Optional[Dict[str, float]] = None,
        latest_wind_kt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Registers or updates a live/current event in the operational registry."""
        event_record = {
            "event_id": event_id,
            "storm_name": storm_name,
            "source": source,
            "status": "ACTIVE",
            "observation_time": observation_time,
            "center_history": center_history or [],
            "latest_center": latest_center,
            "latest_wind_kt": latest_wind_kt,
            "storm_history_available": bool(center_history and len(center_history) >= 2),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._live_events[event_id] = event_record
        logger.info(f"Registered current event '{event_id}' ({storm_name}) with {len(event_record['center_history'])} history fixes.")
        return event_record

    def list_active_current_events(self) -> List[Dict[str, Any]]:
        """
        Discovers active cyclone events through LiveSourceService
        and computes operational readiness (satellite frames + causal history).
        """
        from apps.backend.services.live_source_service import get_live_source_service
        from apps.backend.services.satellite_ingestion_service import SatelliteIngestionService
        live_source = get_live_source_service()
        sat_service = SatelliteIngestionService()

        events_raw = live_source.discover_active_events()
        active_list = []

        for ev in events_raw:
            eid = ev["event_id"]
            ename = ev.get("name", eid)
            esource = ev.get("source", "UNKNOWN")
            estatus = ev.get("status", "ACTIVE")

            try:
                latest_obs = live_source.get_latest_observation(eid)
                obs_time = latest_obs["observation_time"]
                t0_dt = pd.to_datetime(obs_time.replace("Z", "+00:00"))
                if t0_dt.tzinfo is None:
                    t0_dt = t0_dt.replace(tzinfo=timezone.utc)

                sat_readiness = sat_service.check_sequence_readiness(t0_dt)
                history = live_source.get_recent_track_history(eid, obs_time)

                is_sat_ready = sat_readiness["is_complete"]
                is_history_ready = len(history) >= 2

                if not is_sat_ready:
                    readiness_state = sat_readiness["status"]
                elif not is_history_ready:
                    readiness_state = "INSUFFICIENT_HISTORY"
                else:
                    readiness_state = "READY"

                active_list.append({
                    "event_id": eid,
                    "name": ename,
                    "source": esource,
                    "status": estatus,
                    "latest_observation": obs_time,
                    "latest_center": latest_obs.get("latest_center"),
                    "history_fix_count": len(history),
                    "satellite_frame_count": sat_readiness["available_count"],
                    "readiness": readiness_state,
                })
            except Exception as e:
                logger.warning(f"Error evaluating readiness for current event {eid}: {e}")
                active_list.append({
                    "event_id": eid,
                    "name": ename,
                    "source": esource,
                    "status": estatus,
                    "latest_observation": None,
                    "latest_center": None,
                    "history_fix_count": 0,
                    "satellite_frame_count": 0,
                    "readiness": "SOURCE_UNAVAILABLE",
                })

        return active_list

    def get_current_event(self, event_id: str, t0_utc: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves event record by event_id.
        Priority:
        1. Live registered events in memory.
        2. Fallback to historical catalogue evaluated strictly at t <= t0.
        """
        # 1. Check live in-memory registry
        if event_id in self._live_events:
            ev = dict(self._live_events[event_id])
            if t0_utc:
                ev["observation_time"] = t0_utc
            return ev

        # 2. Historical fallback for demonstration
        self._ensure_catalog_loaded()
        if self._sample_index_df is not None:
            matches = self._sample_index_df[
                (self._sample_index_df["storm_id"].str.contains(event_id, case=False, na=False)) |
                (self._sample_index_df["sample_id"].str.contains(event_id, case=False, na=False))
            ]
            if not matches.empty:
                sid = matches.iloc[0]["storm_id"]
                # Resolve t0
                if t0_utc:
                    t0_target = t0_utc.replace("Z", "+00:00").strip()
                    t_rows = matches[matches["t0"].str.contains(t0_target[:16], regex=False)]
                    active_row = t_rows.iloc[0] if not t_rows.empty else matches.iloc[len(matches) // 2]
                else:
                    active_row = matches.iloc[len(matches) // 2]

                t0_str = str(active_row["t0"])
                t0_dt = pd.to_datetime(t0_str)

                # Extract observed center history up to t0 (STRICTLY CAUSAL: t <= t0)
                history = []
                if self._best_track_df is not None:
                    ts_col = "timestamp_utc" if "timestamp_utc" in self._best_track_df.columns else "iso_time"
                    wind_col = "maximum_sustained_wind_kt" if "maximum_sustained_wind_kt" in self._best_track_df.columns else "max_wind_kt"
                    bt_matches = self._best_track_df[self._best_track_df["storm_id"] == sid]
                    for _, btr in bt_matches.iterrows():
                        b_ts = pd.to_datetime(btr[ts_col])
                        if b_ts.tzinfo is None:
                            b_ts = b_ts.replace(tzinfo=timezone.utc)
                        t0_cmp = t0_dt if t0_dt.tzinfo is not None else t0_dt.replace(tzinfo=timezone.utc)
                        if b_ts <= t0_cmp:
                            history.append({
                                "timestamp_utc": b_ts.isoformat(),
                                "lat": float(btr["latitude"]),
                                "lon": float(btr["longitude"]),
                                "wind_kt": float(btr[wind_col]) if pd.notna(btr.get(wind_col)) else None,
                                "pressure_hpa": float(btr["central_pressure_hpa"]) if pd.notna(btr.get("central_pressure_hpa")) else None,
                            })
                    history.sort(key=lambda x: x["timestamp_utc"])

                latest_center = {
                    "lat": float(active_row["imd_lat_t0"]) if pd.notna(active_row["imd_lat_t0"]) else None,
                    "lon": float(active_row["imd_lon_t0"]) if pd.notna(active_row["imd_lon_t0"]) else None,
                }
                latest_wind = float(active_row["imd_wind_t0"]) if pd.notna(active_row["imd_wind_t0"]) else None

                return {
                    "event_id": sid,
                    "storm_name": sid.replace("NIO_", "").split("_")[-1],
                    "source": "HISTORICAL_REHEARSAL_CATALOG",
                    "status": "DEMO_REHEARSAL",
                    "observation_time": t0_str,
                    "sample_id": str(active_row["sample_id"]),
                    "center_history": history,
                    "latest_center": latest_center,
                    "latest_wind_kt": latest_wind,
                    "storm_history_available": len(history) >= 2,
                    "fixes_count": len(history),
                }

        raise CurrentEventError("EVENT_NOT_FOUND", f"Current event identifier '{event_id}' not found in active or fallback registries.", status_code=404)

    def derive_kinematic_inputs(
        self,
        event_data: Dict[str, Any],
        t0_dt: datetime,
    ) -> Dict[str, Any]:
        """
        Derives the 5-dim motion context and multi-horizon kinematic anchors from observed history.
        
        Requires:
        - Storm center at t0
        - At least 1 prior observed center at t_prior <= t0 - 3h
        
        If fewer than 2 fixes exist:
        Returns track_readiness = "INSUFFICIENT_HISTORY" with clear documentation of what is missing.
        """
        history = event_data.get("center_history", [])
        latest_c = event_data.get("latest_center")

        if not history and not latest_c:
            return {
                "track_readiness": "INSUFFICIENT_HISTORY",
                "reason": "Zero historical or current center observations available for event.",
                "motion_ctx": None,
                "kin_12": None,
                "kin_24": None,
                "kin_48": None,
            }

        # Filter history strictly up to t0 (zero future leakage)
        causal_fixes = []
        for h in history:
            h_dt = pd.to_datetime(h["timestamp_utc"])
            if h_dt.tzinfo is None:
                h_dt = h_dt.replace(tzinfo=timezone.utc)
            if h_dt <= t0_dt:
                causal_fixes.append((h_dt, float(h["lat"]), float(h["lon"])))

        causal_fixes.sort(key=lambda x: x[0])

        if len(causal_fixes) < 2:
            return {
                "track_readiness": "INSUFFICIENT_HISTORY",
                "reason": (
                    f"Phase 5B Variant A requires at least 2 causal observed fixes (prior and t0) "
                    f"to construct its physical inertial kinematic anchor. Available causal fixes: {len(causal_fixes)}."
                ),
                "causal_fixes_count": len(causal_fixes),
                "motion_ctx": None,
                "kin_12": None,
                "kin_24": None,
                "kin_48": None,
            }

        t_curr, lat0, lon0 = causal_fixes[-1]
        t_prior, lat_prev, lon_prev = causal_fixes[-2]

        delta_h = (t_curr - t_prior).total_seconds() / 3600.0
        if delta_h <= 0.1:
            delta_h = 3.0  # Safe minimum step if identical timestamps

        # Velocity in degrees per hour
        v_lat = (lat0 - lat_prev) / delta_h
        d_lon = lon0 - lon_prev
        if d_lon > 180.0:
            d_lon -= 360.0
        elif d_lon < -180.0:
            d_lon += 360.0
        v_lon = d_lon / delta_h

        # Kinematic forecast anchors in degrees
        kin_12 = [round(lat0 + v_lat * 12.0, 4), round(lon0 + v_lon * 12.0, 4)]
        kin_24 = [round(lat0 + v_lat * 24.0, 4), round(lon0 + v_lon * 24.0, 4)]
        kin_48 = [round(lat0 + v_lat * 48.0, 4), round(lon0 + v_lon * 48.0, 4)]

        # Physical velocity (km/h)
        v_north = v_lat * DEG_TO_RAD * R_EARTH_KM
        v_east = v_lon * math.cos(lat0 * DEG_TO_RAD) * DEG_TO_RAD * R_EARTH_KM
        speed = math.sqrt(v_north**2 + v_east**2)

        # 5-dimensional motion context tensor [1, 5]
        motion_ctx = torch.tensor([
            v_north / 50.0,
            v_east / 50.0,
            speed / 50.0,
            (lat0 - LAT_MIN) / LAT_SPAN,
            (lon0 - LON_MIN) / LON_SPAN
        ], dtype=torch.float32)

        return {
            "track_readiness": "READY",
            "anchor_type": "observed_center_history",
            "lat0": lat0,
            "lon0": lon0,
            "v_lat_deg_per_h": round(v_lat, 4),
            "v_lon_deg_per_h": round(v_lon, 4),
            "speed_km_per_h": round(speed, 2),
            "delta_h": delta_h,
            "motion_ctx": motion_ctx,
            "kin_12": torch.tensor(kin_12, dtype=torch.float32),
            "kin_24": torch.tensor(kin_24, dtype=torch.float32),
            "kin_48": torch.tensor(kin_48, dtype=torch.float32),
            "kin_coords": {
                "plus_12h": {"lat": kin_12[0], "lon": kin_12[1]},
                "plus_24h": {"lat": kin_24[0], "lon": kin_24[1]},
                "plus_48h": {"lat": kin_48[0], "lon": kin_48[1]},
            }
        }
