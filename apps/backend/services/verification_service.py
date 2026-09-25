"""VAYU-NET — Trajectory Verification Service
===========================================
SIH 2026 Problem Statement 26070
India Meteorological Department (IMD) / Ministry of Earth Sciences (MoES)

Computes deterministic verification metrics (Direct Position Error in km)
comparing predicted multi-horizon trajectory coordinates against held-out IMD ground truth.

Strict Operational Rule:
Verification executes strictly downstream of prediction.
Future ground-truth observations must NEVER enter the prediction path.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger("vayu.services.verification")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_INDEX_PATH = PROJECT_ROOT / "data/manifests/vayu_net_sample_index.csv"
EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(float(EARTH_RADIUS_KM * c), 2)


def compute_directional_displacement(
    pred_lat: float, pred_lon: float, true_lat: float, true_lon: float
) -> Dict[str, float]:
    """Computes directional error components in degrees and local kilometers."""
    delta_lat = round(float(pred_lat - true_lat), 4)
    delta_lon = round(float(pred_lon - true_lon), 4)
    mean_lat_rad = math.radians((pred_lat + true_lat) / 2.0)
    error_north_km = round(float(delta_lat * (math.pi / 180.0) * EARTH_RADIUS_KM), 2)
    error_east_km = round(float(delta_lon * math.cos(mean_lat_rad) * (math.pi / 180.0) * EARTH_RADIUS_KM), 2)
    return {
        "delta_lat_deg": delta_lat,
        "delta_lon_deg": delta_lon,
        "error_north_km": error_north_km,
        "error_east_km": error_east_km,
    }


class VerificationService:
    """Production service for verifying cyclone forecast trajectories against IMD observations."""

    def __init__(self, sample_index_path: Optional[Path] = None):
        self.sample_index_path = sample_index_path or DEFAULT_SAMPLE_INDEX_PATH
        self._sample_df: Optional[pd.DataFrame] = None
        self._load_index()

    def _load_index(self) -> None:
        if not self.sample_index_path.exists():
            logger.warning(f"Sample index not found at {self.sample_index_path}")
            return
        try:
            self._sample_df = pd.read_csv(self.sample_index_path)
            logger.info(f"Loaded {len(self._sample_df)} sample records for trajectory verification.")
        except Exception as e:
            logger.error(f"Failed to load sample index for verification: {e}")
            self._sample_df = None

    def verify_forecast(
        self,
        sample_id: str,
        forecast_coords: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Compares forecast coordinates against held-out IMD ground truth.
        Args:
            sample_id: Unique sequence identifier
            forecast_coords: e.g. {
                'plus_12h': {'lat': 16.5, 'lon': 88.2},
                'plus_24h': {'lat': 18.1, 'lon': 89.0},
                'plus_48h': {'lat': 21.4, 'lon': 89.8}
            }
        """
        if self._sample_df is None:
            return {
                "status": "UNAVAILABLE",
                "reason": "Sample index not loaded",
                "horizons": {},
                "aggregate_dpe_km": None,
            }

        matches = self._sample_df[self._sample_df["sample_id"] == sample_id]
        if matches.empty:
            return {
                "status": "UNAVAILABLE",
                "reason": f"Sample '{sample_id}' not found in verification catalog",
                "horizons": {},
                "aggregate_dpe_km": None,
            }

        row = matches.iloc[0]
        horizons_eval: Dict[str, Any] = {}
        dpe_list = []

        horizon_mapping = [
            ("plus_12h", "12h", "imd_lat_12h", "imd_lon_12h"),
            ("plus_24h", "24h", "imd_lat_24h", "imd_lon_24h"),
            ("plus_48h", "48h", "imd_lat_48h", "imd_lon_48h"),
        ]

        for canonical_key, short_key, lat_col, lon_col in horizon_mapping:
            pred_pt = forecast_coords.get(canonical_key)
            if not pred_pt or "lat" not in pred_pt or "lon" not in pred_pt:
                continue

            true_lat = row.get(lat_col)
            true_lon = row.get(lon_col)

            if pd.isna(true_lat) or pd.isna(true_lon):
                horizons_eval[canonical_key] = {
                    "status": "GROUND_TRUTH_UNAVAILABLE",
                    "reason": f"Storm dissipated or untracked at +{short_key}",
                    "predicted": pred_pt,
                    "actual": None,
                    "dpe_km": None,
                }
                continue

            true_lat_f = float(true_lat)
            true_lon_f = float(true_lon)
            pred_lat_f = float(pred_pt["lat"])
            pred_lon_f = float(pred_pt["lon"])

            dpe = haversine_distance_km(pred_lat_f, pred_lon_f, true_lat_f, true_lon_f)
            directional = compute_directional_displacement(pred_lat_f, pred_lon_f, true_lat_f, true_lon_f)
            dpe_list.append(dpe)

            horizons_eval[canonical_key] = {
                "status": "VERIFIED",
                "predicted": {"lat": round(pred_lat_f, 4), "lon": round(pred_lon_f, 4)},
                "actual": {"lat": round(true_lat_f, 4), "lon": round(true_lon_f, 4)},
                "dpe_km": dpe,
                "directional_error": directional,
            }

        agg_dpe = round(float(sum(dpe_list) / len(dpe_list)), 2) if dpe_list else None

        return {
            "status": "AVAILABLE" if dpe_list else "UNAVAILABLE",
            "sample_id": sample_id,
            "horizons": horizons_eval,
            "aggregate_dpe_km": agg_dpe,
            "verified_horizons_count": len(dpe_list),
            "verification_source": "IMD Best Track Archive",
            "evaluation_notice": (
                "Verification is evaluated strictly downstream of model inference. "
                "Predicted trajectory is compared against actual observed future coordinates."
            ),
        }
