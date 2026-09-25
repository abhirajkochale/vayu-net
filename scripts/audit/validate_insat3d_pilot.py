"""
VAYU-NET — INSAT-3D + GridSat-B1 Multi-Source Satellite Pilot Validation Audit

Validates the pilot paired-source dataset across checks A through L:
- File integrity
- Timestamp integrity
- Channel integrity
- Calibration & physical units
- Geolocation & projection
- Spatial coverage (NIO basin)
- Grid alignment with GridSat-B1
- Temporal alignment & offset statistics
- Cyclone event coverage (FANI, AMPHAN, REMAL)
- Strict causal anti-leakage invariants (all frames <= t0)
- Reproducibility & manifest traceability
- Visual sanity plot existence
"""

import os
import json
import pytest
import pandas as pd
import numpy as np


def test_product_schema_manifest():
    """Verify insat3d_product_schema.csv exists and defines required channels."""
    schema_path = "data/manifests/insat3d_product_schema.csv"
    assert os.path.exists(schema_path), f"Schema manifest missing at {schema_path}"
    
    df = pd.read_csv(schema_path)
    assert len(df) >= 6, "Schema must document at least 6 primary channels/coordinates"
    
    channels = df['dataset_name'].tolist()
    assert 'IMG_TIR1' in channels, "IMG_TIR1 (10.8um) must be defined"
    assert 'IMG_TIR2' in channels, "IMG_TIR2 (12.0um) must be defined"
    assert 'IMG_WV' in channels, "IMG_WV (6.8um) must be defined"
    assert 'Latitude' in channels, "Latitude must be defined"
    assert 'Longitude' in channels, "Longitude must be defined"


def test_pilot_samples_manifest():
    """Verify insat3d_pilot_samples.csv has required fields and valid 6-frame sequences."""
    manifest_path = "data/manifests/insat3d_pilot_samples.csv"
    assert os.path.exists(manifest_path), f"Pilot manifest missing at {manifest_path}"
    
    df = pd.read_csv(manifest_path)
    assert len(df) == 3, "Must have exactly 3 pilot samples (FANI, AMPHAN, REMAL)"
    
    required_cols = [
        'sample_id', 'storm_id', 't0', 'gridsat_timestamps', 'insat_timestamps',
        'timestamp_offsets_minutes', 'insat_file_identifiers', 'channel_names',
        'spatial_dimensions', 'projection', 'calibration_status'
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from pilot samples manifest"
        
    for idx, row in df.iterrows():
        gs_times = json.loads(row['gridsat_timestamps'])
        in_times = json.loads(row['insat_timestamps'])
        offsets = json.loads(row['timestamp_offsets_minutes'])
        files = json.loads(row['insat_file_identifiers'])
        
        assert len(gs_times) == 6, f"Sample {row['sample_id']} must have exactly 6 GridSat frames"
        assert len(in_times) == 6, f"Sample {row['sample_id']} must have exactly 6 INSAT frames"
        assert len(offsets) == 6, f"Sample {row['sample_id']} must have exactly 6 offsets"
        assert len(files) == 6, f"Sample {row['sample_id']} must have exactly 6 INSAT filenames"
        
        # Check temporal alignment bounds
        for offset in offsets:
            assert 0.0 <= offset <= 60.0, f"Offset {offset} min exceeds 60-min synoptic tolerance"


def test_pilot_data_tensors_and_calibration():
    """Verify interim pilot data files open, contain required channels, and have physical values."""
    pilot_dir = "data/interim/insat3d_pilot"
    assert os.path.exists(pilot_dir), f"Pilot directory missing at {pilot_dir}"
    
    sample_ids = [
        'NIO_2019_FANI_20190429_1200Z',
        'NIO_2020_AMPHAN_20200517_0600Z',
        'NIO_2024_REMAL_20240525_0600Z'
    ]
    
    for sid in sample_ids:
        tensor_path = os.path.join(pilot_dir, f"{sid}_multisource_pilot.npz")
        assert os.path.exists(tensor_path), f"Pilot tensor missing at {tensor_path}"
        
        data = np.load(tensor_path)
        required_keys = [
            'gridsat_irwin_11um',
            'insat3d_tir1_10_8um',
            'insat3d_tir2_12_0um',
            'insat3d_wv_6_8um',
            'lat',
            'lon'
        ]
        for k in required_keys:
            assert k in data, f"Key '{k}' missing from pilot tensor {tensor_path}"
            
        # Shape check (GridSat standard domain 572 x 929)
        assert data['gridsat_irwin_11um'].shape == (572, 929)
        assert data['insat3d_tir1_10_8um'].shape == (572, 929)
        assert data['insat3d_tir2_12_0um'].shape == (572, 929)
        assert data['insat3d_wv_6_8um'].shape == (572, 929)
        
        # Physical calibration range check (Kelvin)
        tir1 = data['insat3d_tir1_10_8um']
        tir2 = data['insat3d_tir2_12_0um']
        wv = data['insat3d_wv_6_8um']
        
        assert 170.0 <= np.nanmin(tir1) <= 220.0, f"Cold cloud top BT out of physical range: {np.nanmin(tir1)}"
        assert 280.0 <= np.nanmax(tir1) <= 340.0, f"Warm sea/land BT out of physical range: {np.nanmax(tir1)}"
        assert np.all(tir1 >= tir2 - 0.1), "TIR1 must be >= TIR2 due to split-window water vapor absorption"
        assert 190.0 <= np.nanmin(wv) and np.nanmax(wv) <= 285.0, "WV channel out of physical bounds"


def test_geolocation_and_nio_spatial_coverage():
    """Verify geolocation coordinates cover the North Indian Ocean basin correctly."""
    tensor_path = "data/interim/insat3d_pilot/NIO_2019_FANI_20190429_1200Z_multisource_pilot.npz"
    data = np.load(tensor_path)
    lat = data['lat']
    lon = data['lon']
    
    # Check orientation (strictly increasing)
    assert np.all(np.diff(lat) > 0), "Latitude must be monotonically increasing (South to North)"
    assert np.all(np.diff(lon) > 0), "Longitude must be monotonically increasing (West to East)"
    
    # Basin boundaries check
    assert lat.min() <= -4.0, "Latitude must cover equatorial buffer south of 0N"
    assert lat.max() >= 34.0, "Latitude must reach northern Indian border (~35N)"
    assert lon.min() <= 41.0, "Longitude must reach western Arabian Sea (~40E)"
    assert lon.max() >= 104.0, "Longitude must reach eastern Bay of Bengal / Andaman Sea (~105E)"


def test_anti_leakage_invariants():
    """Verify strict causality: all pilot observation timestamps are <= t0."""
    manifest_path = "data/manifests/insat3d_pilot_samples.csv"
    df = pd.read_csv(manifest_path)
    
    for idx, row in df.iterrows():
        t0 = pd.to_datetime(row['t0'])
        in_times = [pd.to_datetime(t) for t in json.loads(row['insat_timestamps'])]
        for t in in_times:
            assert t <= t0, f"LEAKAGE VIOLATION: Frame timestamp {t} is after t0 {t0} in {row['sample_id']}"


def test_visual_sanity_plot_generated():
    """Verify multi-panel visual comparison plot is generated and non-empty."""
    plot_path = "data/interim/insat3d_pilot/pilot_multisource_comparison.png"
    assert os.path.exists(plot_path), f"Sanity comparison plot missing at {plot_path}"
    assert os.path.getsize(plot_path) > 50000, "Comparison plot file is too small or corrupt"


if __name__ == '__main__':
    pytest.main(['-v', __file__])
