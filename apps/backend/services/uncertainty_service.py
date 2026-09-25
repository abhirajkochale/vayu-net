"""VAYU-NET — Empirical Uncertainty Service
==========================================
SIH 2026 Problem Statement 26070
India Meteorological Department (IMD) / Ministry of Earth Sciences (MoES)

Derives empirical forecast uncertainty radii and cones from Phase 5B validation residuals.
Guarantees:
- Strict empirical derivation: No test data used for calibration
- No fabricated probabilistic uncertainty
- Explicit radii for +12h, +24h, +48h in kilometers
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("vayu.services.uncertainty")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNCERTAINTY_PATH = PROJECT_ROOT / "data/interim/ml/uncertainty_parameters.json"
EARTH_RADIUS_KM = 6371.0


def compute_destination_point(lat_deg: float, lon_deg: float, distance_km: float, bearing_deg: float) -> Tuple[float, float]:
    """Computes destination coordinates along a spherical great circle."""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)
    angular_dist = distance_km / EARTH_RADIUS_KM

    dest_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_dist) +
        math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
    )

    dest_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
        math.cos(angular_dist) - math.sin(lat_rad) * math.sin(dest_lat_rad)
    )

    dest_lat = math.degrees(dest_lat_rad)
    dest_lon = math.degrees(dest_lon_rad)
    dest_lon = (dest_lon + 540.0) % 360.0 - 180.0
    return round(dest_lat, 4), round(dest_lon, 4)


def generate_circle_polygon(lat_deg: float, lon_deg: float, radius_km: float, num_points: int = 32) -> List[List[float]]:
    """Generates a GeoJSON-compatible closed polygon ring approximating a circular uncertainty boundary."""
    coordinates = []
    bearings = np.linspace(0, 360, num_points + 1)
    for b in bearings:
        pt_lat, pt_lon = compute_destination_point(lat_deg, lon_deg, radius_km, float(b))
        coordinates.append([pt_lon, pt_lat])
    return coordinates


class UncertaintyService:
    """Production service for empirical forecast uncertainty estimation."""

    def __init__(self, calibration_path: Optional[Path] = None):
        self.calibration_path = calibration_path or DEFAULT_UNCERTAINTY_PATH
        self.calibration_data: Optional[Dict[str, Any]] = None
        self._load_calibration()

    def _load_calibration(self) -> None:
        if not self.calibration_path.exists():
            logger.warning(f"Uncertainty calibration parameters file not found at {self.calibration_path}")
            return
        try:
            with open(self.calibration_path, "r") as f:
                self.calibration_data = json.load(f)
            logger.info(f"Loaded uncertainty calibration from {self.calibration_path} (split={self.calibration_data.get('derivation_split')})")
        except Exception as e:
            logger.error(f"Failed to load uncertainty calibration: {e}")
            self.calibration_data = None

    def estimate_uncertainty(
        self,
        forecast: Dict[str, Any],
        percentile: str = "p80",
    ) -> Dict[str, Any]:
        """
        Estimates empirical uncertainty for a forecast dictionary.
        Forecast is expected to contain 'plus_12h', 'plus_24h', 'plus_48h' coordinates.
        """
        if self.calibration_data is None:
            return {
                "status": "UNAVAILABLE",
                "reason": "Calibration parameters file not available",
                "plus_12h_km": None,
                "plus_24h_km": None,
                "plus_48h_km": None,
                "percentile": percentile,
                "label": "empirical",
                "is_probabilistic": False,
            }

        horizons = self.calibration_data.get("horizons", {})
        p_key = percentile.lower()
        p_km_key = f"{p_key}_km"

        h12 = horizons.get("12h", {})
        h24 = horizons.get("24h", {})
        h48 = horizons.get("48h", {})

        r12 = h12.get(p_km_key, h12.get(p_key))
        r24 = h24.get(p_km_key, h24.get(p_key))
        r48 = h48.get(p_km_key, h48.get(p_key))

        cones = {}
        if "plus_12h" in forecast and r12 is not None:
            lat12 = forecast["plus_12h"].get("lat")
            lon12 = forecast["plus_12h"].get("lon")
            if lat12 is not None and lon12 is not None:
                cones["plus_12h"] = generate_circle_polygon(lat12, lon12, float(r12))

        if "plus_24h" in forecast and r24 is not None:
            lat24 = forecast["plus_24h"].get("lat")
            lon24 = forecast["plus_24h"].get("lon")
            if lat24 is not None and lon24 is not None:
                cones["plus_24h"] = generate_circle_polygon(lat24, lon24, float(r24))

        if "plus_48h" in forecast and r48 is not None:
            lat48 = forecast["plus_48h"].get("lat")
            lon48 = forecast["plus_48h"].get("lon")
            if lat48 is not None and lon48 is not None:
                cones["plus_48h"] = generate_circle_polygon(lat48, lon48, float(r48))

        return {
            "status": "AVAILABLE",
            "plus_12h_km": round(float(r12), 1) if r12 is not None else None,
            "plus_24h_km": round(float(r24), 1) if r24 is not None else None,
            "plus_48h_km": round(float(r48), 1) if r48 is not None else None,
            "percentile": percentile.upper(),
            "derivation_split": self.calibration_data.get("derivation_split", "VALIDATION"),
            "label": "empirical",
            "is_probabilistic": False,
            "cone_geometries": cones,
        }
