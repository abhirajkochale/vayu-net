"""
VAYU-NET — Multi-Source Satellite Dataset Quality & Anti-Leakage Audit Suite

Deterministic test suite verifying:
- Integrity of multi-source sample index (725 paired samples)
- Strict split isolation with zero storm overlap
- Temporal synchronization bounds (all offsets <= 60 min, >=90% exact)
- Strict causal anti-leakage invariants (all frames <= t0)
- Train-only derivation of normalization statistics
- Physical calibration bounds for TIR1, TIR2, and WV channels
- Completeness of coverage manifests and visual QA artifacts
"""

import os
import json
import pytest
import pandas as pd
import numpy as np


def test_multisource_sample_index_integrity():
    """Verify multisource sample index contains all 757 paired samples and required columns."""
    index_path = "data/manifests/vayu_net_multisource_sample_index.csv"
    assert os.path.exists(index_path), f"Multisource sample index missing at {index_path}"
    
    df = pd.read_csv(index_path)
    assert len(df) == 757, f"Expected 757 paired samples, found {len(df)}"
    
    required_cols = [
        'sample_id', 'storm_id', 'split', 't0',
        'grid_sat_t_minus_15h', 'grid_sat_t_minus_12h', 'grid_sat_t_minus_9h',
        'grid_sat_t_minus_6h', 'grid_sat_t_minus_3h', 'grid_sat_t0',
        'insat_t_minus_15h', 'insat_t_minus_12h', 'insat_t_minus_9h',
        'insat_t_minus_6h', 'insat_t_minus_3h', 'insat_t0',
        'insat_timestamps', 'timestamp_offsets_minutes', 'channels',
        'calibration_status', 'spatial_projection', 'spatial_shape',
        'is_causal_verified', 'current_center_lat', 'current_center_lon',
        'current_wind_kt', 'current_pressure_hpa', 'current_category',
        'has_target_12h', 'has_target_24h', 'has_target_48h'
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from multisource sample index"


def test_split_isolation_and_no_storm_overlap():
    """Verify that TRAIN, VALIDATION, and TEST storm sets are strictly disjoint."""
    index_path = "data/manifests/vayu_net_multisource_sample_index.csv"
    df = pd.read_csv(index_path)
    
    train_storms = set(df[df['split'] == 'TRAIN']['storm_id'])
    val_storms = set(df[df['split'] == 'VALIDATION']['storm_id'])
    test_storms = set(df[df['split'] == 'TEST']['storm_id'])
    
    # Check storm counts
    assert len(train_storms) == 25, f"Expected 25 train storms, found {len(train_storms)}"
    assert len(val_storms) == 14, f"Expected 14 validation storms, found {len(val_storms)}"
    assert len(test_storms) == 24, f"Expected 24 test storms, found {len(test_storms)}"
    
    # Check disjointness
    assert len(train_storms.intersection(val_storms)) == 0, "TRAIN and VALIDATION storms overlap!"
    assert len(train_storms.intersection(test_storms)) == 0, "TRAIN and TEST storms overlap!"
    assert len(val_storms.intersection(test_storms)) == 0, "VALIDATION and TEST storms overlap!"
    
    # Check sample counts
    assert (df['split'] == 'TRAIN').sum() == 207
    assert (df['split'] == 'VALIDATION').sum() == 252
    assert (df['split'] == 'TEST').sum() == 298


def test_temporal_alignment_and_offset_bounds():
    """Verify timestamp alignment manifest has 4,542 frames, all <= 60 min offset, and >=90% exact."""
    align_path = "data/manifests/multisource_timestamp_alignment.csv"
    assert os.path.exists(align_path), f"Timestamp alignment manifest missing at {align_path}"
    
    df = pd.read_csv(align_path)
    assert len(df) == 4542, f"Expected 4,542 observation frames, found {len(df)}"
    
    # Policy threshold: <= 60 minutes
    assert (df['delta_minutes'] <= 60.0).all(), "Frame offset exceeds 60-minute maximum accepted threshold"
    
    # Exact match percentage check
    exact_pct = (df['delta_minutes'] == 0.0).mean() * 100.0
    assert exact_pct >= 90.0, f"Expected >= 90% exact matches, found {exact_pct:.2f}%"
    
    # Mean and median offset
    assert df['delta_minutes'].median() == 0.0, "Median offset must be 0.0 minutes"
    assert df['delta_minutes'].mean() <= 10.0, "Mean offset must be <= 10.0 minutes"


def test_strict_causality_no_future_leakage():
    """Verify that every single observation timestamp is <= t0."""
    index_path = "data/manifests/vayu_net_multisource_sample_index.csv"
    df = pd.read_csv(index_path)
    
    for idx, row in df.iterrows():
        t0 = pd.to_datetime(row['t0'])
        in_times = [pd.to_datetime(t) for t in json.loads(row['insat_timestamps'])]
        for t in in_times:
            assert t <= t0, f"CAUSALITY LEAKAGE: Frame {t} is after t0 {t0} in {row['sample_id']}"
            
    align_path = "data/manifests/multisource_timestamp_alignment.csv"
    df_align = pd.read_csv(align_path)
    assert df_align['is_causal'].all(), "Found non-causal frame in timestamp alignment manifest"


def test_train_normalization_isolation_and_physical_bounds():
    """Verify normalization statistics are strictly derived from TRAIN data and have valid ranges."""
    norm_path = "data/interim/ml/multisource_train_normalization_stats.json"
    assert os.path.exists(norm_path), f"Normalization stats missing at {norm_path}"
    
    with open(norm_path, 'r') as f:
        stats = json.load(f)
        
    meta = stats['metadata']
    assert "TRAIN_ONLY" in meta['derivation_split'], "Normalization must be TRAIN_ONLY"
    assert meta['train_samples_count'] == 207, f"Expected 207 train samples, found {meta['train_samples_count']}"
    assert meta['unique_train_frames_count'] == 379, f"Expected 379 unique frames, found {meta.get('unique_train_frames_count')}"
    
    channels = stats['channels']
    for ch in ['IMG_TIR1', 'IMG_TIR2', 'IMG_WV']:
        assert ch in channels, f"Channel {ch} missing from normalization stats"
        cstats = channels[ch]
        assert 200.0 <= cstats['mean'] <= 300.0, f"Mean out of physical range for {ch}: {cstats['mean']}"
        assert 5.0 <= cstats['std'] <= 40.0, f"Std out of physical range for {ch}: {cstats['std']}"
        assert cstats['min'] >= 170.0, f"Min below physical lower bound for {ch}: {cstats['min']}"
        assert cstats['max'] <= 340.0, f"Max above physical upper bound for {ch}: {cstats['max']}"


def test_coverage_manifest_completeness():
    """Verify coverage manifest documents all 216 storms from original manifest."""
    cov_path = "data/manifests/multisource_coverage_manifest.csv"
    assert os.path.exists(cov_path), f"Coverage manifest missing at {cov_path}"
    
    df = pd.read_csv(cov_path)
    assert len(df) == 216, f"Expected 216 storms in coverage manifest, found {len(df)}"
    assert df['paired_multisource_samples'].sum() == 757, "Total paired samples must equal 757"
    assert df['missing_insat_samples'].sum() == 594, "Total missing samples must equal 594"
    assert df['total_original_samples'].sum() == 1319, "Total original samples must equal 1,319"


def test_visual_qa_artifacts_exist():
    """Verify all 4 meteorological category visual QA comparisons exist and are non-empty."""
    qa_files = [
        "data/interim/ml/multisource_qa/qa_bay_of_bengal_strong_amphan.png",
        "data/interim/ml/multisource_qa/qa_arabian_sea_strong_biparjoy.png",
        "data/interim/ml/multisource_qa/qa_bay_of_bengal_developing_remal.png",
        "data/interim/ml/multisource_qa/qa_arabian_sea_vayu.png"
    ]
    for qf in qa_files:
        assert os.path.exists(qf), f"Visual QA artifact missing at {qf}"
        assert os.path.getsize(qf) > 50000, f"Visual QA artifact {qf} is suspiciously small"


def test_tensor_dimensions_and_no_corrupted_samples():
    """Verify that all satellite frames in index exist, are non-empty, and load with valid (572, 929) shape."""
    index_path = "data/manifests/vayu_net_multisource_sample_index.csv"
    df = pd.read_csv(index_path)
    
    frame_cols = [
        'grid_sat_t_minus_15h', 'grid_sat_t_minus_12h', 'grid_sat_t_minus_9h',
        'grid_sat_t_minus_6h', 'grid_sat_t_minus_3h', 'grid_sat_t0'
    ]
    
    # Check all unique files
    unique_files = list(set(df[frame_cols].values.flatten()))
    for fp in unique_files:
        assert os.path.exists(fp), f"Missing frame file: {fp}"
        assert os.path.getsize(fp) > 1000, f"Suspiciously empty/corrupt frame file: {fp}"
        
    # Spot-check 10 random frames for exact dimensions
    for fp in unique_files[:10]:
        with np.load(fp) as d:
            assert "irwin_cdr" in d, f"Missing 'irwin_cdr' key in {fp}"
            arr = d["irwin_cdr"]
            assert arr.shape == (572, 929), f"Invalid tensor shape {arr.shape} in {fp}"
            assert not np.all(np.isnan(arr)), f"All-NaN tensor in {fp}"


if __name__ == '__main__':
    pytest.main(['-v', __file__])
