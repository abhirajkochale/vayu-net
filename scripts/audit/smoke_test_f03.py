"""
VAYU-NET — F03 Center Detection & Model Inference Smoke Test
============================================================
Runs live MODEL_INFERENCE for the 5 reference cyclones (AMPHAN, FANI, TAUKTAE, BIPARJOY, REMAL).
Prints:
- AI Detected Center (Phase 3C DedicatedCenterLocalizationResNet)
- Observed IMD Reference Center (IMD best-track analysis at t0)
- Distance between AI center and observed IMD center (km)
- Forecast Anchor Type (OBSERVED_IMD_T0)
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from apps.backend.main import app

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))

def run_smoke_test():
    client = TestClient(app)
    storms = ["AMPHAN", "FANI", "TAUKTAE", "BIPARJOY", "REMAL"]

    print("=" * 105)
    print(f"{'CYCLONE':<10} | {'SAMPLE ID':<35} | {'AI DETECTED CENTER':<20} | {'OBSERVED IMD REF':<18} | {'OFFSET km':<10} | {'ANCHOR TYPE'}")
    print("=" * 105)

    results = []
    for name in storms:
        res = client.post("/api/predict", json={"cyclone_id": name, "percentile": "p80", "mode": "MODEL_INFERENCE"})
        assert res.status_code == 200, f"Inference failed for {name}: {res.text}"
        d = res.json()
        sid = d["sample_id"]
        ai = d["ai_detected_center"]
        obs = d["observed_reference_center"]
        anchor = d["forecast_anchor_type"]
        dist = haversine_km(ai["latitude"], ai["longitude"], obs["latitude"], obs["longitude"])
        ai_str = f"{ai['latitude']:.2f}°N, {ai['longitude']:.2f}°E"
        obs_str = f"{obs['latitude']:.2f}°N, {obs['longitude']:.2f}°E"
        print(f"{name:<10} | {sid:<35} | {ai_str:<20} | {obs_str:<18} | {dist:<10.2f} | {anchor}")
        results.append({
            "name": name,
            "sample_id": sid,
            "ai_center": ai,
            "obs_center": obs,
            "offset_km": dist,
            "anchor": anchor
        })

    print("=" * 105)
    return results

if __name__ == "__main__":
    run_smoke_test()
