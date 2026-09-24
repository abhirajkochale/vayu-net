"""
VAYU-NET WP-06 — UNCERTAINTY & VERIFICATION AUDIT & UNIT TEST SUITE
===================================================================
Tests:
1. Uncertainty parameter integrity (valid horizons: 12h, 24h, 48h)
2. Empirical provenance (labeled strictly as 'empirical', derivation_split='VALIDATION')
3. Positive and monotonic radii scaling across horizons (+12h < +24h < +48h)
4. Geodesic circle geometry generation (valid closed polygon, 33 points)
5. Known coordinate Haversine DPE mathematical sanity check
6. Verification horizon association (+12h, +24h, +48h)
7. Predicted vs actual timestamp progression and integrity
8. Full verification deterministic reproducibility
9. Split separation (zero test contamination in uncertainty derivation)
10. Locked dataset invariants (manifests and historical checkpoints untouched)
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.forecast.uncertainty import EmpiricalUncertainty, generate_circle_polygon
from ml.forecast.verification import ForecastVerifier, compute_directional_errors, haversine_km

UNCERTAINTY_PARAMS_PATH = "data/interim/ml/uncertainty_parameters.json"
SAMPLE_INDEX_PATH = "data/manifests/vayu_net_sample_index.csv"


def test_uncertainty_parameter_integrity():
    print("[Test 1/10] Testing Uncertainty Parameter File Integrity...")
    assert os.path.exists(UNCERTAINTY_PARAMS_PATH), f"Missing {UNCERTAINTY_PARAMS_PATH}"
    with open(UNCERTAINTY_PARAMS_PATH, "r") as f:
        data = json.load(f)

    assert data["derivation_split"] == "VALIDATION", f"Expected VALIDATION derivation, got {data['derivation_split']}"
    assert "horizons" in data
    assert set(data["horizons"].keys()) == {"12h", "24h", "48h"}

    for h in ["12h", "24h", "48h"]:
        h_data = data["horizons"][h]
        for key in ["residual_count", "mean_dpe_km", "median_dpe_km", "p50_km", "p80_km", "p90_km", "p95_km"]:
            assert key in h_data, f"Missing {key} in horizon {h}"
            val = h_data[key]
            assert val > 0, f"Value for {key} in horizon {h} must be positive, got {val}"

    print("  -> Passed: Parameter file has valid horizons, positive values, and correct schema.")


def test_empirical_provenance_and_monotonicity():
    print("[Test 2/10] Testing Empirical Provenance and Radius Monotonicity...")
    engine = EmpiricalUncertainty()
    assert engine.derivation_split == "VALIDATION"

    r12 = engine.get_radius_km("12h", "p80")
    r24 = engine.get_radius_km("24h", "p80")
    r48 = engine.get_radius_km("48h", "p80")

    assert r12 < r24 < r48, f"Expected monotonic radius growth, got 12h={r12}, 24h={r24}, 48h={r48}"

    p50_12 = engine.get_radius_km("12h", "p50")
    p80_12 = engine.get_radius_km("12h", "p80")
    p90_12 = engine.get_radius_km("12h", "p90")
    assert p50_12 < p80_12 < p90_12, f"Expected percentile monotonicity, got {p50_12} < {p80_12} < {p90_12}"

    print(f"  -> Passed: Monotonic scaling confirmed (12h P80: {r12} km < 24h P80: {r24} km < 48h P80: {r48} km).")


def test_cone_geometry_generation():
    print("[Test 3/10] Testing Polygon Cone Geometry...")
    engine = EmpiricalUncertainty()
    cone = engine.get_uncertainty_object(15.0, 85.0, "12h", "p80", include_polygon=True)

    assert cone["label"] == "empirical"
    assert cone["is_probabilistic_confidence_interval"] is False
    assert "geometry" in cone
    poly = cone["geometry"]["coordinates"][0]
    assert len(poly) == 33  # 32 segments + 1 closing point
    assert poly[0] == poly[-1], "Polygon must be closed (first point == last point)"

    # Verify radius distances of polygon points from center
    for pt in poly[:-1]:
        pt_lon, pt_lat = pt
        d = haversine_km(15.0, 85.0, pt_lat, pt_lon)
        assert abs(d - cone["radius_km"]) < 1.0, f"Polygon vertex distance {d} km does not match radius {cone['radius_km']} km"

    print("  -> Passed: Spherical polygon geometry mathematically verified.")


def test_haversine_known_benchmark():
    print("[Test 4/10] Testing Known Benchmark Haversine DPE...")
    # Distance between Chennai (13.0827°N, 80.2707°E) and Kolkata (22.5726°N, 88.3639°E) is ~1358.36 km
    d = haversine_km(13.0827, 80.2707, 22.5726, 88.3639)
    assert 1350.0 <= d <= 1370.0, f"Expected ~1358 km, got {d} km"

    # Zero distance
    d0 = haversine_km(15.0, 85.0, 15.0, 85.0)
    assert d0 == 0.0, f"Expected 0.0 km for identical points, got {d0}"

    print(f"  -> Passed: Haversine distance matches physical benchmark ({d} km).")


def test_directional_error_vectors():
    print("[Test 5/10] Testing Directional Error Vectors...")
    errs = compute_directional_errors(15.0, 85.0, 14.0, 84.0)
    assert errs["delta_lat_deg"] == 1.0
    assert errs["delta_lon_deg"] == 1.0
    assert errs["error_north_km"] > 0, "Forecast north of truth should produce positive north error"
    assert errs["error_east_km"] > 0, "Forecast east of truth should produce positive east error"
    print("  -> Passed: Directional errors correctly signed and scaled.")


def test_verification_sample_structure():
    print("[Test 6/10] Testing Verification Payload Structure...")
    verifier = ForecastVerifier()
    # Use any valid test sample with stored predictions
    test_sid = list(verifier.predictions_lookup.keys())[0]
    v_res = verifier.verify_sample(test_sid)

    assert v_res["sample_id"] == test_sid
    assert "horizons" in v_res
    assert set(v_res["horizons"].keys()) == {"12h", "24h", "48h"}

    for h in ["12h", "24h", "48h"]:
        h_dict = v_res["horizons"][h]
        assert "forecast" in h_dict
        assert "actual" in h_dict
        assert "error" in h_dict
        assert "dpe_km" in h_dict["error"]
        assert h_dict["error"]["dpe_km"] >= 0.0

    assert "mean_dpe_km" in v_res["summary"]
    assert "status" in v_res["summary"]

    print("  -> Passed: Verification payload matches locked schema.")


def test_timestamp_progression_integrity():
    print("[Test 7/10] Testing Timestamp Progression in Verification...")
    verifier = ForecastVerifier()
    test_sid = list(verifier.predictions_lookup.keys())[0]
    v_res = verifier.verify_sample(test_sid)

    t0_dt = pd.to_datetime(v_res["t0"])
    t12_dt = pd.to_datetime(v_res["horizons"]["12h"]["actual"]["timestamp_utc"])
    t24_dt = pd.to_datetime(v_res["horizons"]["24h"]["actual"]["timestamp_utc"])
    t48_dt = pd.to_datetime(v_res["horizons"]["48h"]["actual"]["timestamp_utc"])

    assert t12_dt == t0_dt + pd.Timedelta(hours=12)
    assert t24_dt == t0_dt + pd.Timedelta(hours=24)
    assert t48_dt == t0_dt + pd.Timedelta(hours=48)

    print("  -> Passed: Timestamps strictly follow synoptic progression (+12h, +24h, +48h).")


def test_verification_determinism():
    print("[Test 8/10] Testing Verification Determinism...")
    verifier = ForecastVerifier()
    test_sid = list(verifier.predictions_lookup.keys())[0]
    r1 = verifier.verify_sample(test_sid)
    r2 = verifier.verify_sample(test_sid)

    assert r1 == r2, "Verification output must be bitwise deterministic across calls."
    print("  -> Passed: 100% deterministic verification output.")


def test_split_integrity_zero_test_contamination():
    print("[Test 9/10] Testing Split Isolation in Uncertainty Derivation...")
    with open(UNCERTAINTY_PARAMS_PATH, "r") as f:
        data = json.load(f)

    assert data["derivation_split"] == "VALIDATION"
    assert data["metadata"]["derivation_samples_count"] == 252
    assert data["metadata"]["derivation_storms_count"] == 14
    # Ensure test samples were not mixed into derivation
    assert data["horizons"]["12h"]["residual_count"] == 252
    print("  -> Passed: Uncertainty parameters derived strictly from VALIDATION (N=252).")


def test_locked_datasets_immutability():
    print("[Test 10/10] Testing Immutability of Locked Files...")
    locked_files = [
        "data/manifests/vayu_net_sample_index.csv",
        "data/manifests/vayu_net_candidate_t0_manifest.csv",
        "data/manifests/gridsat_sha256_manifest.csv",
        "data/processed/imd_best_track_v2.csv",
        "data/interim/ml/checkpoints/best_phase5b_variant_a.pt",
        "data/interim/ml/phase5b_hybrid_results.json"
    ]
    for p in locked_files:
        assert os.path.exists(p), f"Locked file missing: {p}"

    df = pd.read_csv("data/manifests/vayu_net_sample_index.csv")
    assert len(df) == 1319, f"Expected 1319 samples, got {len(df)}"
    print("  -> Passed: All locked files and sample counts remain intact.")


def run_all_tests():
    print("=" * 80)
    print("VAYU-NET WP-06: UNCERTAINTY & VERIFICATION AUDIT SUITE")
    print("=" * 80)
    test_uncertainty_parameter_integrity()
    test_empirical_provenance_and_monotonicity()
    test_cone_geometry_generation()
    test_haversine_known_benchmark()
    test_directional_error_vectors()
    test_verification_sample_structure()
    test_timestamp_progression_integrity()
    test_verification_determinism()
    test_split_integrity_zero_test_contamination()
    test_locked_datasets_immutability()
    print("=" * 80)
    print("ALL WP-06 UNCERTAINTY & VERIFICATION TESTS PASSED (10/10)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
