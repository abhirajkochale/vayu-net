"""
VAYU-NET — Lightweight Historical Demo Service
===============================================
SIH 2026 Problem Statement 26070
India Meteorological Department (IMD) / Ministry of Earth Sciences (MoES)

Lightweight, zero-PyTorch catalog and canonical demo inference orchestrator
specifically designed for memory-constrained deployments (e.g. Render Free 512 MB).

Architecture & Guarantees:
1. Zero PyTorch / Torchvision imports on startup or during canonical demo inference.
2. Serves pre-validated canonical AMPHAN historical rehearsal inference output generated
   from real VAYU-NET model checkpoints.
3. Preserves full catalog and metadata query performance via lightweight manifest parsing.
4. Seamlessly delegates to the full PyTorch InferencePipeline when executing in local
   research or full-model environments with available memory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("vayu.services.demo_service")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "manifests" / "vayu_net_sample_index.csv"
STORM_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "storm_event_manifest_v2.csv"
CANONICAL_AMPHAN_PATH = PROJECT_ROOT / "data" / "processed" / "canonical_amphan_demo_result.json"

DEMO_SHORTLIST = ["FANI", "AMPHAN", "TAUKTAE", "BIPARJOY", "REMAL"]


class InferencePipelineError(Exception):
    """Structured error raised during inference pipeline execution."""
    def __init__(self, error_code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }


class HistoricalDemoService:
    """
    Lightweight, zero-PyTorch catalog and canonical demo inference provider.
    """

    def __init__(self):
        logger.info("Initializing VAYU-NET Lightweight Historical Demo Service (Zero-PyTorch)...")
        if not SAMPLE_INDEX_PATH.exists():
            raise FileNotFoundError(f"Sample index manifest not found at {SAMPLE_INDEX_PATH}")
        self.sample_index_df = pd.read_csv(SAMPLE_INDEX_PATH)

        if not STORM_MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Storm manifest not found at {STORM_MANIFEST_PATH}")
        self.storm_manifest_df = pd.read_csv(STORM_MANIFEST_PATH)

        self._canonical_amphan_cache: Optional[Dict[str, Any]] = None
        self._inference_cache: Dict[str, Dict[str, Any]] = {}

    def get_canonical_amphan_result(self) -> Dict[str, Any]:
        """Loads and returns the pre-validated canonical AMPHAN historical rehearsal inference output."""
        if self._canonical_amphan_cache is None:
            if not CANONICAL_AMPHAN_PATH.exists():
                raise FileNotFoundError(
                    f"Canonical AMPHAN demo result artifact not found at {CANONICAL_AMPHAN_PATH}."
                )
            with open(CANONICAL_AMPHAN_PATH, "r", encoding="utf-8") as f:
                self._canonical_amphan_cache = json.load(f)
            logger.info("Loaded pre-validated canonical AMPHAN demo result from disk.")
        return self._canonical_amphan_cache

    def is_demo_event(self, event_id: str) -> bool:
        """Determines if an event request matches the canonical AMPHAN demo."""
        clean = (event_id or "").strip().upper()
        return "AMPHAN" in clean or clean == "NIO_2020_AMPHAN"

    def list_events(self) -> List[Dict[str, Any]]:
        """Returns catalog of all available cyclone events without loading ML dependencies."""
        events = []
        for _, r in self.storm_manifest_df.iterrows():
            sid = str(r["storm_id"])
            sname = str(r["storm_name"])
            year = int(r["year"])
            samples_sub = self.sample_index_df[self.sample_index_df["storm_id"] == sid]
            events.append({
                "event_id": sid,
                "storm_name": sname,
                "year": year,
                "split": str(r["split"]),
                "peak_category": str(r["peak_category"]),
                "max_wind_kt": float(r["max_wind_kt"]) if pd.notna(r["max_wind_kt"]) else None,
                "min_pressure_hpa": float(r["min_pressure_hpa"]) if pd.notna(r["min_pressure_hpa"]) else None,
                "observation_count": len(samples_sub),
                "start_timestamp_utc": str(r["start_timestamp_utc"]) if pd.notna(r["start_timestamp_utc"]) else None,
                "end_timestamp_utc": str(r["end_timestamp_utc"]) if pd.notna(r["end_timestamp_utc"]) else None,
            })
        return events

    def get_event_details(self, event_id: str) -> Dict[str, Any]:
        """Returns detailed metadata and candidate observation timestamps for an event."""
        matches = self.storm_manifest_df[
            (self.storm_manifest_df["storm_id"].str.contains(event_id, case=False, na=False)) |
            (self.storm_manifest_df["storm_name"].str.contains(event_id, case=False, na=False))
        ]
        if matches.empty:
            raise InferencePipelineError("EVENT_NOT_FOUND", f"Cyclone event '{event_id}' not found in catalog.", status_code=404)

        st_row = matches.iloc[0]
        sid = str(st_row["storm_id"])
        samples = self.sample_index_df[self.sample_index_df["storm_id"] == sid]

        observations = []
        for _, sr in samples.iterrows():
            observations.append({
                "sample_id": str(sr["sample_id"]),
                "t0_utc": str(sr["t0"]),
                "reference_center": {
                    "lat": float(sr["imd_lat_t0"]) if pd.notna(sr["imd_lat_t0"]) else None,
                    "lon": float(sr["imd_lon_t0"]) if pd.notna(sr["imd_lon_t0"]) else None,
                },
                "reference_wind_kt": float(sr["imd_wind_t0"]) if pd.notna(sr["imd_wind_t0"]) else None,
                "reference_category": str(sr["imd_category_t0"]) if pd.notna(sr["imd_category_t0"]) else None,
            })

        return {
            "event_id": sid,
            "storm_name": str(st_row["storm_name"]),
            "year": int(st_row["year"]),
            "split": str(st_row["split"]),
            "peak_category": str(st_row["peak_category"]),
            "max_wind_kt": float(st_row["max_wind_kt"]) if pd.notna(st_row["max_wind_kt"]) else None,
            "min_pressure_hpa": float(st_row["min_pressure_hpa"]) if pd.notna(st_row["min_pressure_hpa"]) else None,
            "observations_count": len(observations),
            "observations": observations,
        }

    def list_cyclones(self, shortlist_only: bool = True) -> List[Dict[str, Any]]:
        """Returns cyclone events, defaulting to the reference demo shortlist."""
        results = []
        for _, r in self.storm_manifest_df.iterrows():
            sid = str(r["storm_id"])
            sname = str(r["storm_name"])
            year = int(r["year"])

            is_in_shortlist = any(sl.lower() in sname.lower() or sl.lower() in sid.lower() for sl in DEMO_SHORTLIST)
            if shortlist_only and not is_in_shortlist:
                continue

            n_samples = len(self.sample_index_df[self.sample_index_df["storm_id"] == sid])

            results.append({
                "cyclone_id": sid,
                "storm_name": sname,
                "year": year,
                "split": str(r["split"]),
                "peak_category": str(r["peak_category"]),
                "max_wind_kt": float(r["max_wind_kt"]) if pd.notna(r["max_wind_kt"]) else None,
                "samples_count": n_samples,
                "is_demo_shortlist": is_in_shortlist
            })
        return results

    def get_cyclone_details(self, cyclone_id: str) -> Dict[str, Any]:
        """Returns metadata and candidate sample timestamps for a given cyclone."""
        matches = self.storm_manifest_df[
            (self.storm_manifest_df["storm_id"].str.contains(cyclone_id, case=False, na=False)) |
            (self.storm_manifest_df["storm_name"].str.contains(cyclone_id, case=False, na=False))
        ]
        if matches.empty:
            raise KeyError(f"Cyclone identifier '{cyclone_id}' not found.")
        st_row = matches.iloc[0]
        sid = str(st_row["storm_id"])

        samples = self.sample_index_df[self.sample_index_df["storm_id"] == sid]
        sample_list = []
        for _, sr in samples.iterrows():
            sample_list.append({
                "sample_id": str(sr["sample_id"]),
                "t0": str(sr["t0"]),
                "center": [
                    float(sr["imd_lat_t0"]) if pd.notna(sr["imd_lat_t0"]) else 0.0,
                    float(sr["imd_lon_t0"]) if pd.notna(sr["imd_lon_t0"]) else 0.0,
                ],
                "wind_kt": float(sr["imd_wind_t0"]) if pd.notna(sr["imd_wind_t0"]) else None,
                "category": str(sr["imd_category_t0"]) if pd.notna(sr["imd_category_t0"]) else "",
            })

        return {
            "cyclone_id": sid,
            "storm_name": str(st_row["storm_name"]),
            "year": int(st_row["year"]),
            "split": str(st_row["split"]),
            "peak_category": str(st_row["peak_category"]),
            "max_wind_kt": float(st_row["max_wind_kt"]) if pd.notna(st_row["max_wind_kt"]) else None,
            "samples": sample_list
        }

    def list_current_events(self) -> List[Dict[str, Any]]:
        """
        Discovers active cyclone events and evaluates operational readiness
        without loading heavy ML models into memory.
        """
        try:
            from apps.backend.services.live_source_service import get_live_source_service
            live_source = get_live_source_service()
            events_raw = live_source.discover_active_events()
            active_list = []
            for ev in events_raw:
                eid = ev["event_id"]
                ename = ev.get("name", eid)
                esource = ev.get("source", "UNKNOWN")
                estatus = ev.get("status", "ACTIVE")
                try:
                    latest_obs = live_source.get_latest_observation(eid)
                    obs_time = latest_obs.get("observation_time")
                    history = live_source.get_recent_track_history(eid, obs_time) if obs_time else []
                    active_list.append({
                        "event_id": eid,
                        "name": ename,
                        "source": esource,
                        "status": estatus,
                        "latest_observation": obs_time,
                        "latest_center": latest_obs.get("latest_center"),
                        "history_fix_count": len(history),
                        "satellite_frame_count": 6 if "AMPHAN" in eid.upper() else 0,
                        "readiness": "DEMO_AVAILABLE" if "AMPHAN" in eid.upper() else "INSUFFICIENT_FRAMES",
                    })
                except Exception:
                    # Storm without active live observations in local catalog
                    continue
            return active_list
        except Exception as e:
            logger.warning(f"Could not discover live events in demo mode: {e}")
            return []

    def run_inference(self, event_id: str, t0_utc: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes inference request:
        - If event matches AMPHAN: returns the pre-validated canonical historical demo result (Zero-PyTorch).
        - If custom event requested:
          - If model checkpoints exist and memory allows: delegates to full InferencePipeline.
          - Otherwise: returns structured error explaining Free tier limitation.
        """
        if self.is_demo_event(event_id):
            result = self.get_canonical_amphan_result()
            self._inference_cache[result.get("metadata", {}).get("inference_id", "demo_amphan")] = result
            return result

        # Check if full models exist locally
        ckpt_path = PROJECT_ROOT / "data" / "interim" / "ml" / "checkpoints" / "best_center_localization_cnn.pt"
        if not ckpt_path.exists() or os.getenv("VAYU_PUBLIC_DEMO_MODE", "0") == "1":
            raise InferencePipelineError(
                error_code="LIVE_INFERENCE_UNAVAILABLE_ON_FREE_TIER",
                message=(
                    f"Operational live model inference for '{event_id}' is unavailable on the Free Public tier (512 MB). "
                    "The full validated Historical Replay Demo is available for cyclone AMPHAN (2020-05-18T06:00:00Z)."
                ),
                status_code=503,
            )

        # In full local/research environment, lazy-import and delegate to full pipeline
        from apps.backend.services.inference_pipeline import get_inference_pipeline
        pipeline = get_inference_pipeline()
        return pipeline.run_inference(event_id=event_id, t0_utc=t0_utc)

    def get_cached_inference(self, inference_id: str) -> Dict[str, Any]:
        """Retrieves cached canonical inference result."""
        if inference_id in self._inference_cache:
            return self._inference_cache[inference_id]
        if "amphan" in inference_id.lower():
            return self.get_canonical_amphan_result()
        raise InferencePipelineError(
            "INFERENCE_NOT_FOUND",
            f"Inference run '{inference_id}' not found in cache.",
            status_code=404,
        )


# Singleton demo service
_demo_service: Optional[HistoricalDemoService] = None


def get_demo_service() -> HistoricalDemoService:
    global _demo_service
    if _demo_service is None:
        _demo_service = HistoricalDemoService()
    return _demo_service
