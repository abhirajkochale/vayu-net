"""
VAYU-NET WP-06 — EMPIRICAL UNCERTAINTY MODULE
==============================================
Derives empirical forecast uncertainty radii and cones from VALIDATION residuals (Phase 5B Variant A).
Guarantees:
  - No TEST data used for uncertainty derivation
  - Strictly empirical provenance (labeled 'empirical', not probabilistic CI)
  - Output radii for +12h, +24h, +48h
  - Polygon cone geometry construction around forecast centers
"""

import os
import json
import math
import numpy as np

UNCERTAINTY_PARAMS_PATH = "data/interim/ml/uncertainty_parameters.json"
EARTH_RADIUS_KM = 6371.0


def compute_destination_point(lat_deg, lon_deg, distance_km, bearing_deg):
    """
    Computes destination latitude and longitude given distance and bearing along a great circle.
    """
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
    # Normalize longitude to [-180, 180]
    dest_lon = (dest_lon + 540.0) % 360.0 - 180.0
    return round(dest_lat, 4), round(dest_lon, 4)


def generate_circle_polygon(lat_deg, lon_deg, radius_km, num_points=32):
    """
    Generates a GeoJSON-compatible closed polygon ring approximating a circular
    uncertainty circle of given radius_km on the spherical Earth.
    """
    coordinates = []
    bearings = np.linspace(0, 360, num_points + 1)
    for b in bearings:
        pt_lat, pt_lon = compute_destination_point(lat_deg, lon_deg, radius_km, b)
        coordinates.append([pt_lon, pt_lat])  # GeoJSON convention: [lon, lat]
    return coordinates


class EmpiricalUncertainty:
    """
    Manages empirical uncertainty parameters derived from Phase 5B validation residuals.
    """
    def __init__(self, params_path=UNCERTAINTY_PARAMS_PATH):
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Uncertainty parameters file not found at: {params_path}")
        with open(params_path, "r") as f:
            self.data = json.load(f)
        self.horizons = self.data["horizons"]
        self.derivation_split = self.data["derivation_split"]

    def get_radius_km(self, horizon, percentile="p80"):
        """Returns empirical radius in km for a given horizon ('12h', '24h', '48h') and percentile."""
        h_key = str(horizon).replace("+", "").replace("h", "") + "h"
        if h_key not in self.horizons:
            raise KeyError(f"Invalid horizon: {horizon}. Expected one of {list(self.horizons.keys())}")
        
        pct_key = percentile.lower()
        if not pct_key.endswith("_km"):
            if pct_key in ["mean", "std", "median"]:
                candidate_keys = [f"{pct_key}_dpe_km", f"{pct_key}_km", pct_key]
            else:
                if not pct_key.startswith("p"):
                    pct_key = f"p{pct_key}"
                candidate_keys = [f"{pct_key}_km", pct_key]
        else:
            candidate_keys = [pct_key]

        metric_dict = self.horizons[h_key]
        matched_key = None
        for ck in candidate_keys:
            if ck in metric_dict:
                matched_key = ck
                break

        if matched_key is None:
            raise KeyError(f"Percentile/metric {percentile} not found for horizon {h_key}. Available: {list(metric_dict.keys())}")
        return float(metric_dict[matched_key])

    def get_uncertainty_object(self, forecast_lat, forecast_lon, horizon, percentile="p80", include_polygon=True):
        """
        Constructs product-ready uncertainty payload for a given forecast point.
        """
        h_key = str(horizon).replace("+", "").replace("h", "") + "h"
        radius_km = self.get_radius_km(h_key, percentile)
        
        payload = {
            "horizon": f"+{h_key}",
            "center": {
                "latitude": round(float(forecast_lat), 4),
                "longitude": round(float(forecast_lon), 4)
            },
            "radius_km": round(radius_km, 2),
            "derivation_split": self.derivation_split,
            "derivation_metric": percentile.upper(),
            "label": "empirical",
            "is_probabilistic_confidence_interval": False,
            "description": f"Empirical {percentile.upper()} track error radius derived from {self.derivation_split} residuals"
        }

        if include_polygon:
            payload["geometry"] = {
                "type": "Polygon",
                "coordinates": [generate_circle_polygon(forecast_lat, forecast_lon, radius_km)]
            }

        return payload

    def construct_track_cone(self, forecast_points, percentile="p80"):
        """
        Builds complete multi-horizon uncertainty cone across a forecast track.
        Args:
            forecast_points: dict with keys '12h', '24h', '48h', each having {'lat': ..., 'lon': ...}
        """
        cones = {}
        for h in ["12h", "24h", "48h"]:
            if h in forecast_points:
                pt = forecast_points[h]
                cones[h] = self.get_uncertainty_object(pt["lat"], pt["lon"], h, percentile=percentile)
        return {
            "derivation_split": self.derivation_split,
            "derivation_source": self.data.get("derivation_model", "Phase 5B Variant A Hybrid"),
            "percentile": percentile.upper(),
            "label": "empirical",
            "horizons": cones
        }
