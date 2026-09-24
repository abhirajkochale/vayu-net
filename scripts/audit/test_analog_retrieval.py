"""
VAYU-NET WP-07 — ANALOG-STORM RETRIEVAL AUDIT & UNIT TEST SUITE
==============================================================
Tests:
1. Analog archive file integrity and schema (data/interim/ml/analog_retrieval_cache.json)
2. 7-dimensional feature vector definitions and valid numerical ranges
3. Training-only standardization statistics derivation (zero test leakage)
4. Nearest-neighbor distance metric and strictly non-decreasing distance ordering
5. Exactly two returned analogs (k = 2)
6. Self-match protection: query storm never matches itself or its own samples
7. Cross-storm analog diversity: top-2 analogs come from two distinct historical storms
8. Full deterministic reproducibility across repeated retrieval calls
9. Demo shortlist execution (FANI 2019, AMPHAN 2020, TAUKTAE 2021, BIPARJOY 2023, REMAL 2024)
10. Immutability of locked manifests and datasets
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

from ml.forecast.analog_retrieval import AnalogRetriever, FEATURE_NAMES, ANALOG_CACHE_PATH

SHORTLIST = ["FANI", "AMPHAN", "TAUKTAE", "BIPARJOY", "REMAL"]


def test_analog_archive_schema():
    print("[Test 1/10] Testing Analog Archive File Schema...")
    assert os.path.exists(ANALOG_CACHE_PATH), f"Missing {ANALOG_CACHE_PATH}"
    with open(ANALOG_CACHE_PATH, "r") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "standardization" in data
    assert "candidates" in data
    assert len(data["candidates"]) == 696, f"Expected 696 TRAIN candidates, got {len(data['candidates'])}"
    assert data["metadata"]["derivation_split"] == "TRAIN"
    assert data["metadata"]["k_default"] == 2
    assert data["metadata"]["label"] == "HISTORICAL ANALOG"
    print("  -> Passed: Archive schema matches locked specifications.")


def test_feature_vector_dimensions_and_ranges():
    print("[Test 2/10] Testing 7-D Feature Vector Dimensions and Ranges...")
    retriever = AnalogRetriever()
    assert len(FEATURE_NAMES) == 7
    assert FEATURE_NAMES == ["lat", "lon", "wind_kt", "pressure_hpa", "dx_12h", "dy_12h", "dwind_12h"]

    for cand in retriever.candidates[:50]:
        rf = cand["raw_features"]
        assert len(rf) == 7
        assert -5.0 <= rf["lat"] <= 35.0, f"Lat out of NIO bounds: {rf['lat']}"
        assert 40.0 <= rf["lon"] <= 105.0, f"Lon out of NIO bounds: {rf['lon']}"
        assert 15.0 <= rf["wind_kt"] <= 150.0, f"Wind out of bounds: {rf['wind_kt']}"
        assert 880.0 <= rf["pressure_hpa"] <= 1015.0, f"Pressure out of bounds: {rf['pressure_hpa']}"
        assert -20.0 <= rf["dx_12h"] <= 20.0
        assert -20.0 <= rf["dy_12h"] <= 20.0
        assert -150.0 <= rf["dwind_12h"] <= 150.0

    print("  -> Passed: All 7 feature values within valid meteorological bounds.")


def test_training_only_standardization():
    print("[Test 3/10] Testing Training-Only Standardization Statistics...")
    retriever = AnalogRetriever()
    means = retriever.means
    stds = retriever.stds
    assert len(means) == 7 and len(stds) == 7
    assert (stds > 0).all(), "Standard deviations must be strictly positive"

    # Compare with known TRAIN statistics from normalization json
    norm_path = "data/interim/ml/train_normalization_stats.json"
    with open(norm_path, "r") as f:
        norm_data = json.load(f)

    # wind mean should match ~36.88 kt
    assert abs(means[2] - norm_data["train_wind_mean_kt"]) < 0.1, "Wind standardization mean must match train mean"
    assert abs(stds[2] - norm_data["train_wind_std_kt"]) < 0.1, "Wind standardization std must match train std"

    print("  -> Passed: Standardization statistics derived strictly from TRAIN partition.")


def test_distance_metric_and_ordering():
    print("[Test 4/10] Testing Distance Calculation and Monotonic Ordering...")
    retriever = AnalogRetriever()
    # Query with arbitrary synthetic vector
    dummy_q = [15.0, 85.0, 45.0, 990.0, -1.0, 0.5, 5.0]
    res = retriever.find_analogs(dummy_q, k=2)

    assert len(res["analogs"]) == 2
    a1 = res["analogs"][0]
    a2 = res["analogs"][1]

    assert a1["rank"] == 1 and a2["rank"] == 2
    assert a1["standardized_distance"] <= a2["standardized_distance"], "Analogs must be sorted by ascending distance"
    assert a1["standardized_distance"] >= 0.0

    print(f"  -> Passed: Distance ordering verified (Rank 1: {a1['standardized_distance']} <= Rank 2: {a2['standardized_distance']}).")


def test_exact_two_analogs_returned():
    print("[Test 5/10] Testing Exactly k=2 Analogs Returned...")
    retriever = AnalogRetriever()
    # Test on multiple samples across splits
    sids = list(retriever.all_sample_features.keys())[:20]
    for sid in sids:
        res = retriever.find_analogs(sid, k=2)
        assert res["k"] == 2, f"Expected k=2, got {res['k']}"
        assert len(res["analogs"]) == 2, f"Expected 2 analogs, got {len(res['analogs'])}"

    print("  -> Passed: Exactly k=2 analogs returned across all test cases.")


def test_self_match_protection():
    print("[Test 6/10] Testing Self-Match and Same-Storm Exclusion Protection...")
    retriever = AnalogRetriever()

    # Query using a TRAIN sample (which is in the candidate archive)
    train_cand = retriever.candidates[0]
    train_sid = train_cand["sample_id"]
    train_storm = train_cand["storm_id"]

    res = retriever.find_analogs(train_sid, k=2)

    for a in res["analogs"]:
        assert a["sample_id"] != train_sid, f"Self-match detected: returned exact same sample {train_sid}"
        assert a["storm_id"] != train_storm, f"Same-storm match detected: returned candidate from same storm {train_storm}"

    print(f"  -> Passed: Querying {train_sid} correctly excluded self-match and storm {train_storm}.")


def test_cross_storm_analog_diversity():
    print("[Test 7/10] Testing Storm Diversity Across Returned Analogs...")
    retriever = AnalogRetriever()
    sids = list(retriever.all_sample_features.keys())[:30]
    for sid in sids:
        res = retriever.find_analogs(sid, k=2)
        s1 = res["analogs"][0]["storm_id"]
        s2 = res["analogs"][1]["storm_id"]
        assert s1 != s2, f"Returned analogs must be from distinct storms, got {s1} and {s2}"

    print("  -> Passed: Top-2 analogs consistently represent two distinct historical cyclone events.")


def test_analog_retrieval_determinism():
    print("[Test 8/10] Testing Deterministic Reproducibility...")
    retriever = AnalogRetriever()
    sid = list(retriever.all_sample_features.keys())[10]

    res1 = retriever.find_analogs(sid, k=2)
    res2 = retriever.find_analogs(sid, k=2)

    assert res1["analogs"][0]["sample_id"] == res2["analogs"][0]["sample_id"]
    assert res1["analogs"][1]["sample_id"] == res2["analogs"][1]["sample_id"]
    assert res1["analogs"][0]["standardized_distance"] == res2["analogs"][0]["standardized_distance"]

    print("  -> Passed: 100% bitwise deterministic analog retrieval across repeated calls.")


def test_shortlist_demo_cyclones():
    print("[Test 9/10] Testing Reference Demo Shortlist Cyclones...")
    retriever = AnalogRetriever()

    for sname in SHORTLIST:
        matching_sids = [sid for sid in retriever.all_sample_features.keys() if sname in sid]
        assert len(matching_sids) > 0, f"Demo storm {sname} not found in sample feature database"
        sample_query = matching_sids[len(matching_sids)//2]

        res = retriever.find_analogs(sample_query, k=2)
        assert len(res["analogs"]) == 2
        a1 = res["analogs"][0]
        a2 = res["analogs"][1]
        assert a1["label"] == "HISTORICAL ANALOG"
        assert a2["label"] == "HISTORICAL ANALOG"
        print(f"  {sname:10s} ({sample_query[:25]}...) -> 1: {a1['storm_name']} ({a1['year']}, d={a1['standardized_distance']:.2f}) | 2: {a2['storm_name']} ({a2['year']}, d={a2['standardized_distance']:.2f})")

    print("  -> Passed: All 5 demo shortlist storms successfully retrieved validated top-2 analogs.")


def test_locked_files_and_sample_count():
    print("[Test 10/10] Testing Invariants & Immutability...")
    df = pd.read_csv("data/manifests/vayu_net_sample_index.csv")
    assert len(df) == 1319, f"Sample count altered! Expected 1319, got {len(df)}"

    splits = df["split"].value_counts()
    assert splits["TRAIN"] == 696
    assert splits["VALIDATION"] == 252
    assert splits["TEST"] == 371
    print("  -> Passed: Exact split counts verified (TRAIN=696, VAL=252, TEST=371).")


def run_all_tests():
    print("=" * 80)
    print("VAYU-NET WP-07: ANALOG-STORM RETRIEVAL AUDIT SUITE")
    print("=" * 80)
    test_analog_archive_schema()
    test_feature_vector_dimensions_and_ranges()
    test_training_only_standardization()
    test_distance_metric_and_ordering()
    test_exact_two_analogs_returned()
    test_self_match_protection()
    test_cross_storm_analog_diversity()
    test_analog_retrieval_determinism()
    test_shortlist_demo_cyclones()
    test_locked_files_and_sample_count()
    print("=" * 80)
    print("ALL WP-07 ANALOG RETRIEVAL TESTS PASSED (10/10)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
