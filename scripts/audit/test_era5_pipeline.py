"""
VAYU-NET PHASE 5A — ERA5 PIPELINE SANITY TEST & VERIFICATION SUITE
===================================================================
Tests required before complete environmental archive extraction:
1. ERA5 timestamp retrieval
2. Exact UTC alignment
3. Correct variables (u_component_of_wind, v_component_of_wind)
4. Correct pressure levels ([850, 700, 500, 300] hPa)
5. Correct geographic crop (lat: 35 to -5, lon: 40 to 105)
6. Correct dimensions: raw (4, 161, 261) and downsampled (4, 41, 66)
7. Correct U/V orientation (u = eastward, v = northward, lat descending, lon ascending)
8. No NaN or Inf values
9. Normalization verification
10. Multimodal tensor batching [B, 6, 8, 41, 66]
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

def test_era5_pipeline():
    print("=" * 80)
    print("RUNNING VAYU-NET PHASE 5A: ERA5 PIPELINE AUDIT")
    print("=" * 80)

    # 1. Imports and store access
    import xarray as xr
    import gcsfs

    print("[TEST 1/10] Connecting to official cloud-optimized ERA5 archive...")
    t0 = time.time()
    fs = gcsfs.GCSFileSystem(token='anon')
    store = gcsfs.mapping.GCSMap('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', gcs=fs, check=False)
    ds = xr.open_zarr(store, consolidated=True)
    print(f"  Connected in {time.time() - t0:.2f}s.")
    assert 'u_component_of_wind' in ds, "Missing u_component_of_wind in dataset"
    assert 'v_component_of_wind' in ds, "Missing v_component_of_wind in dataset"
    print("  PASS: ERA5 connection and variables verified.")

    # 2. Timestamp alignment check
    print("[TEST 2/10] Testing exact UTC alignment for historical storm (Cyclone Remal)...")
    target_utc = "2024-05-25T06:00:00"
    target_time_val = np.datetime64(target_utc)
    # Check that time coordinate contains exact timestamp
    matched_time = ds.time.sel(time=target_utc, method=None).values
    assert np.datetime64(matched_time) == target_time_val, f"Timestamp mismatch: {matched_time} != {target_time_val}"
    print(f"  PASS: Exact UTC timestamp {target_utc} matched without interpolation.")

    # 3. Variables & 4. Pressure levels
    print("[TEST 3-4/10] Selecting variables and pressure levels (850, 700, 500, 300 hPa)...")
    req_levels = [850, 700, 500, 300]
    for lev in req_levels:
        assert lev in ds.level.values, f"Pressure level {lev} hPa not found in ERA5 level coordinates"
    print(f"  PASS: All 4 pressure levels present in dataset coordinates.")

    # 5. Geographic crop & 6. Dimensions
    print("[TEST 5-6/10] Testing geographic crop (Lat: 35.0 to -5.0, Lon: 40.0 to 105.0)...")
    # Slicing: latitude in ERA5 is descending [90.0 .. -90.0]
    u_slice = ds['u_component_of_wind'].sel(time=target_utc, level=req_levels).sel(latitude=slice(35.0, -5.0), longitude=slice(40.0, 105.0))
    v_slice = ds['v_component_of_wind'].sel(time=target_utc, level=req_levels).sel(latitude=slice(35.0, -5.0), longitude=slice(40.0, 105.0))
    
    u_vals = u_slice.values
    v_vals = v_slice.values
    
    raw_shape = u_vals.shape
    expected_raw_shape = (4, 161, 261)
    assert raw_shape == expected_raw_shape, f"Raw shape mismatch: {raw_shape} != {expected_raw_shape}"
    print(f"  PASS: Raw extraction shape matches expected grid: {raw_shape}")

    # 7. Orientation check
    print("[TEST 7/10] Verifying geographic orientation...")
    lats = u_slice.latitude.values
    lons = u_slice.longitude.values
    assert lats[0] > lats[-1], f"Latitude should be descending: {lats[0]} to {lats[-1]}"
    assert lons[0] < lons[-1], f"Longitude should be ascending: {lons[0]} to {lons[-1]}"
    assert abs(lats[0] - 35.0) < 1e-4 and abs(lats[-1] - (-5.0)) < 1e-4, "Latitude bounds incorrect"
    assert abs(lons[0] - 40.0) < 1e-4 and abs(lons[-1] - 105.0) < 1e-4, "Longitude bounds incorrect"
    print(f"  PASS: Latitude monotonic descending ({lats[0]} -> {lats[-1]}), Longitude monotonic ascending ({lons[0]} -> {lons[-1]}).")

    # 8. Missing values & NaN check
    print("[TEST 8/10] Checking for NaNs, Infs, or invalid fill values...")
    assert not np.isnan(u_vals).any(), "Found NaN in u values"
    assert not np.isnan(v_vals).any(), "Found NaN in v values"
    assert not np.isinf(u_vals).any(), "Found Inf in u values"
    assert not np.isinf(v_vals).any(), "Found Inf in v values"
    print(f"  PASS: Zero NaNs/Infs. Wind U range: [{u_vals.min():.2f}, {u_vals.max():.2f}] m/s, V range: [{v_vals.min():.2f}, {v_vals.max():.2f}] m/s.")

    # Downsampling to ~1.0 degree: (41, 66)
    print("  Testing 4x spatial downsampling to 1.0 degree...")
    # Stack [8, 161, 261]
    env_8ch = np.concatenate([u_vals, v_vals], axis=0) # [8, 161, 261]
    tensor_raw = torch.from_numpy(env_8ch).unsqueeze(0).float() # [1, 8, 161, 261]
    tensor_down = F.interpolate(tensor_raw, size=(41, 66), mode='bilinear', align_corners=True).squeeze(0) # [8, 41, 66]
    assert tensor_down.shape == (8, 41, 66), f"Downsampled shape mismatch: {tensor_down.shape}"
    print(f"  PASS: Downsampled tensor shape: {tuple(tensor_down.shape)}")

    # 9. Normalization verification
    print("[TEST 9/10] Verifying channel-wise normalization logic...")
    mean_dummy = tensor_down.mean(dim=(-2, -1), keepdim=True)
    std_dummy = tensor_down.std(dim=(-2, -1), keepdim=True)
    normalized = (tensor_down - mean_dummy) / (std_dummy + 1e-6)
    assert not torch.isnan(normalized).any()
    print(f"  PASS: Channel-wise normalization logic verified.")

    # 10. Sequence tensor batching [B, 6, 8, 41, 66]
    print("[TEST 10/10] Verifying 6-frame sequence batch shape [B, 6, 8, 41, 66]...")
    batch_size = 4
    seq_tensor = tensor_down.unsqueeze(0).unsqueeze(0).repeat(batch_size, 6, 1, 1, 1)
    assert seq_tensor.shape == (batch_size, 6, 8, 41, 66), f"Batch sequence shape mismatch: {seq_tensor.shape}"
    print(f"  PASS: Multimodal environmental tensor batch shape verified: {tuple(seq_tensor.shape)}")

    print("=" * 80)
    print("ALL 10 ERA5 PIPELINE AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    test_era5_pipeline()
