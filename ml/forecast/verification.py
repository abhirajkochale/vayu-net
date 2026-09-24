"""
VAYU-NET WP-06 — FORECAST VERIFICATION MODULE
==============================================
Provides deterministic, mathematically rigorous verification comparing model forecast
coordinates (+12h, +24h, +48h) against verified IMD ground-truth coordinates.
Clearly distinguishes:
  - FORECAST
  - ACTUAL
  - ERROR (DPE in km, directional error vectors, North/East components)
"""

import os
import math
import json
import pandas as pd
import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between two geographic coordinates in kilometers.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(float(EARTH_RADIUS_KM * c), 2)


def compute_directional_errors(pred_lat, pred_lon, true_lat, true_lon):
    """
    Computes delta latitude, delta longitude, and local tangent-plane displacement in km.
    Positive error_north means forecast was north of actual.
    Positive error_east means forecast was east of actual.
    """
    delta_lat = round(float(pred_lat - true_lat), 4)
    delta_lon = round(float(pred_lon - true_lon), 4)
    
    mean_lat_rad = math.radians((pred_lat + true_lat) / 2.0)
    error_north_km = round(float(delta_lat * (math.pi / 180.0) * EARTH_RADIUS_KM), 2)
    error_east_km = round(float(delta_lon * math.cos(mean_lat_rad) * (math.pi / 180.0) * EARTH_RADIUS_KM), 2)
    
    return {
        "delta_lat_deg": delta_lat,
        "delta_lon_deg": delta_lon,
        "error_north_km": error_north_km,
        "error_east_km": error_east_km
    }


class ForecastVerifier:
    """
    Deterministic verification engine for VAYU-NET multi-horizon forecasts.
    """
    def __init__(self, sample_index_path="data/manifests/vayu_net_sample_index.csv",
                 results_path="data/interim/ml/phase5b_hybrid_results.json"):
        if not os.path.exists(sample_index_path):
            raise FileNotFoundError(f"Sample index not found at: {sample_index_path}")
        self.sample_index_df = pd.read_csv(sample_index_path)
        
        # Load precomputed predictions from Phase 5B Variant A if available
        self.predictions_lookup = {}
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                res_data = json.load(f)
            raw = res_data.get("variant_a_observed_center", {}).get("raw_test", {})
            sids = raw.get("sample_ids", [])
            p12 = raw.get("pred_12", [])
            p24 = raw.get("pred_24", [])
            p48 = raw.get("pred_48", [])
            for i, sid in enumerate(sids):
                self.predictions_lookup[sid] = {
                    "12h": p12[i],
                    "24h": p24[i],
                    "48h": p48[i]
                }

    def verify_sample(self, sample_id, predicted_coords=None):
        """
        Verifies forecast against held-out actual observation for a given sample_id.
        Args:
            sample_id: str (e.g. 'NIO_2021_TAUKTAE_20210515_0000Z')
            predicted_coords: optional dict with keys '12h', '24h', '48h',
                              each being [lat, lon] or {'lat': ..., 'lon': ...}
        """
        match = self.sample_index_df[self.sample_index_df["sample_id"] == sample_id]
        if len(match) == 0:
            raise KeyError(f"Sample ID {sample_id} not found in master sample index.")
        row = match.iloc[0]

        cyclone_id = str(row["storm_id"])
        t0_str = str(row["t0"])
        t0_dt = pd.to_datetime(t0_str)

        # Retrieve predictions
        if predicted_coords is None:
            if sample_id not in self.predictions_lookup:
                raise ValueError(f"No stored predictions for {sample_id}; please provide predicted_coords.")
            stored_p = self.predictions_lookup[sample_id]
            pred_dict = {
                "12h": {"lat": stored_p["12h"][0], "lon": stored_p["12h"][1]},
                "24h": {"lat": stored_p["24h"][0], "lon": stored_p["24h"][1]},
                "48h": {"lat": stored_p["48h"][0], "lon": stored_p["48h"][1]}
            }
        else:
            pred_dict = {}
            for h in ["12h", "24h", "48h"]:
                val = predicted_coords[h]
                if isinstance(val, (list, tuple)):
                    pred_dict[h] = {"lat": float(val[0]), "lon": float(val[1])}
                elif isinstance(val, dict):
                    pred_dict[h] = {"lat": float(val["lat"]), "lon": float(val["lon"])}

        # Build verification per horizon
        horizons_verification = {}
        dpes = []

        horizon_hours = {"12h": 12, "24h": 24, "48h": 48}
        for h, hours in horizon_hours.items():
            true_lat = float(row[f"imd_lat_{hours}h"])
            true_lon = float(row[f"imd_lon_{hours}h"])
            plat = pred_dict[h]["lat"]
            plon = pred_dict[h]["lon"]

            dpe = haversine_km(plat, plon, true_lat, true_lon)
            dpes.append(dpe)
            dir_errors = compute_directional_errors(plat, plon, true_lat, true_lon)

            target_ts = (t0_dt + pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

            horizons_verification[h] = {
                "horizon": f"+{h}",
                "forecast": {
                    "latitude": round(float(plat), 4),
                    "longitude": round(float(plon), 4),
                    "issue_timestamp_utc": t0_str,
                    "target_timestamp_utc": target_ts
                },
                "actual": {
                    "latitude": round(float(true_lat), 4),
                    "longitude": round(float(true_lon), 4),
                    "timestamp_utc": target_ts,
                    "source": "IMD_BEST_TRACK_V2"
                },
                "error": {
                    "dpe_km": dpe,
                    "delta_lat_deg": dir_errors["delta_lat_deg"],
                    "delta_lon_deg": dir_errors["delta_lon_deg"],
                    "error_north_km": dir_errors["error_north_km"],
                    "error_east_km": dir_errors["error_east_km"],
                    "metric": "Direct Position Error (Great-Circle Distance)"
                }
            }

        payload = {
            "cyclone_id": cyclone_id,
            "sample_id": sample_id,
            "split": str(row["split"]),
            "t0": t0_str,
            "horizons": horizons_verification,
            "summary": {
                "mean_dpe_km": round(float(np.mean(dpes)), 2),
                "median_dpe_km": round(float(np.median(dpes)), 2),
                "p90_dpe_km": round(float(np.percentile(dpes, 90)), 2),
                "dpe_12h_km": horizons_verification["12h"]["error"]["dpe_km"],
                "dpe_24h_km": horizons_verification["24h"]["error"]["dpe_km"],
                "dpe_48h_km": horizons_verification["48h"]["error"]["dpe_km"],
                "status": "VERIFIED_AGAINST_IMD_HELD_OUT_TRUTH"
            }
        }
        return payload
