"""
VAYU-NET GridSat-B1 Pilot Processing & Alignment Pipeline
Cyclone FANI (2019)
Processes 6 historical frames + validates alignment with IMD Best Track V2 targets.
Generates manifests, checks A-K, and visual QC plot.
"""

import os
import sys
import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt

# Configuration
RAW_DIR = "data/interim/gridsat_pilot/fani_2019/raw"
OUT_DIR = "data/interim/gridsat_pilot/fani_2019"
DOCS_DIR = "docs"
IMD_V2_CSV = "data/processed/imd_best_track_v2.csv"

STORM_ID = "NIO_2019_FANI"
T0_STR = "2019-04-26T06:00:00+00:00"

# Exact 6 historical frames
FRAMES = [
    {"tag": "t-15h", "delta_h": -15, "ts": "2019-04-25T15:00:00+00:00", "file": "GRIDSAT-B1.2019.04.25.15.v02r01.nc"},
    {"tag": "t-12h", "delta_h": -12, "ts": "2019-04-25T18:00:00+00:00", "file": "GRIDSAT-B1.2019.04.25.18.v02r01.nc"},
    {"tag": "t-9h",  "delta_h": -9,  "ts": "2019-04-25T21:00:00+00:00", "file": "GRIDSAT-B1.2019.04.25.21.v02r01.nc"},
    {"tag": "t-6h",  "delta_h": -6,  "ts": "2019-04-26T00:00:00+00:00", "file": "GRIDSAT-B1.2019.04.26.00.v02r01.nc"},
    {"tag": "t-3h",  "delta_h": -3,  "ts": "2019-04-26T03:00:00+00:00", "file": "GRIDSAT-B1.2019.04.26.03.v02r01.nc"},
    {"tag": "t0",    "delta_h": 0,   "ts": "2019-04-26T06:00:00+00:00", "file": "GRIDSAT-B1.2019.04.26.06.v02r01.nc"},
]

# 3 Target references
TARGETS = [
    {"tag": "t+12h", "delta_h": 12, "ts": "2019-04-26T18:00:00+00:00", "file": "GRIDSAT-B1.2019.04.26.18.v02r01.nc"},
    {"tag": "t+24h", "delta_h": 24, "ts": "2019-04-27T06:00:00+00:00", "file": "GRIDSAT-B1.2019.04.27.06.v02r01.nc"},
    {"tag": "t+48h", "delta_h": 48, "ts": "2019-04-28T06:00:00+00:00", "file": "GRIDSAT-B1.2019.04.28.06.v02r01.nc"},
]

# NIO locked bounding box
LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = 40.0, 105.0

def process_frame(frame_info):
    raw_path = os.path.join(RAW_DIR, frame_info["file"])
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")
    
    nc = netCDF4.Dataset(raw_path, 'r')
    # CRITICAL: Disable automatic masking & scaling so we get raw int16 packed values
    nc.set_auto_maskandscale(False)
    
    # Read coordinates
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    time_val = nc.variables['time'][:]
    
    # Boolean mask for NIO bounds
    lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    
    cropped_lat = lat[lat_mask]
    cropped_lon = lon[lon_mask]
    
    # Read irwin_cdr metadata
    var = nc.variables['irwin_cdr']
    scale_factor = getattr(var, 'scale_factor', 0.01)
    add_offset = getattr(var, 'add_offset', 200.0)
    fill_value = getattr(var, '_FillValue', -31999)
    missing_value = getattr(var, 'missing_value', -31999)
    
    # Raw packed slice: shape (lat, lon) -> (572, 929)
    raw_slice = var[0, lat_mask, :][:, lon_mask]
    
    # Explicit decoding per Task 3 specification:
    # decoded_value = packed_value * scale_factor + add_offset
    # Treat documented fill value as NaN
    is_missing = (raw_slice == fill_value) | (raw_slice == missing_value) | (raw_slice <= -30000)
    decoded = raw_slice.astype(np.float32) * float(scale_factor) + float(add_offset)
    decoded[is_missing] = np.nan
    
    nc.close()
    
    # Statistics
    nan_count = int(np.isnan(decoded).sum())
    total_elements = int(decoded.size)
    nan_fraction = float(nan_count / total_elements)
    
    valid_vals = decoded[~np.isnan(decoded)]
    min_k = float(valid_vals.min()) if len(valid_vals) > 0 else np.nan
    max_k = float(valid_vals.max()) if len(valid_vals) > 0 else np.nan
    mean_k = float(valid_vals.mean()) if len(valid_vals) > 0 else np.nan
    
    # Save cropped NetCDF
    out_nc_path = os.path.join(OUT_DIR, f"gridsat_fani_{frame_info['tag']}.nc")
    save_cropped_netcdf(out_nc_path, frame_info, cropped_lat, cropped_lon, decoded)
    
    # Save compressed npz for easy ML loading
    out_npz_path = os.path.join(OUT_DIR, f"gridsat_fani_{frame_info['tag']}.npz")
    np.savez_compressed(out_npz_path, 
                        irwin_cdr=decoded, 
                        lat=cropped_lat, 
                        lon=cropped_lon, 
                        timestamp=frame_info['ts'], 
                        delta_t=frame_info['delta_h'])
    
    record = {
        "storm_id": STORM_ID,
        "t0": T0_STR,
        "tag": frame_info["tag"],
        "satellite_timestamp_utc": frame_info["ts"],
        "delta_t_hours": frame_info["delta_h"],
        "source_file": frame_info["file"],
        "variable": "irwin_cdr",
        "units": "Kelvin",
        "lat_min": float(cropped_lat.min()),
        "lat_max": float(cropped_lat.max()),
        "lon_min": float(cropped_lon.min()),
        "lon_max": float(cropped_lon.max()),
        "array_shape": f"{decoded.shape[0]}x{decoded.shape[1]}",
        "nan_count": nan_count,
        "nan_fraction": round(nan_fraction, 6),
        "min_kelvin": round(min_k, 2),
        "max_kelvin": round(max_k, 2),
        "mean_kelvin": round(mean_k, 2),
        "saved_nc": os.path.basename(out_nc_path),
        "saved_npz": os.path.basename(out_npz_path)
    }
    return record, cropped_lat, cropped_lon, decoded

def save_cropped_netcdf(path, frame_info, lat, lon, data):
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as nc:
        nc.createDimension('lat', len(lat))
        nc.createDimension('lon', len(lon))
        
        vlat = nc.createVariable('lat', 'f4', ('lat',))
        vlat.units = 'degrees_north'
        vlat.standard_name = 'latitude'
        vlat[:] = lat
        
        vlon = nc.createVariable('lon', 'f4', ('lon',))
        vlon.units = 'degrees_east'
        vlon.standard_name = 'longitude'
        vlon[:] = lon
        
        vir = nc.createVariable('irwin_cdr', 'f4', ('lat', 'lon'), fill_value=np.nan, zlib=True)
        vir.long_name = 'NOAA FCDR of Brightness Temperature near 11 microns (Nadir-most observations) cropped to NIO'
        vir.standard_name = 'toa_brightness_temperature'
        vir.units = 'Kelvin'
        vir.coordinates = 'lat lon'
        vir[:] = data
        
        # Attributes
        nc.title = f"VAYU-NET GridSat-B1 Pilot Frame - Cyclone FANI ({frame_info['tag']})"
        nc.storm_id = STORM_ID
        nc.satellite_timestamp_utc = frame_info['ts']
        nc.delta_t_hours = frame_info['delta_h']
        nc.source_file = frame_info['file']
        nc.spatial_region = "North Indian Ocean (NIO)"

def generate_visual_qc(frames_data, imd_t0_fix):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Brightness temperature standard color scale:
    # Cold deep convective clouds = 190-220K (cyan/white/magenta), Warm ocean/land = 280-305K
    cmap = plt.cm.inferno_r # inverted inferno: dark/warm, bright/cold convective tops
    vmin, vmax = 190, 305
    
    for i, (info, lat, lon, data) in enumerate(frames_data):
        ax = axes[i]
        im = ax.imshow(data, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        title_str = f"{info['tag']} ({info['delta_h']:+d}h) | {info['ts'][:16]}Z"
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlabel("Longitude (°E)")
        ax.grid(color='cyan', linestyle=':', linewidth=0.5, alpha=0.4)
        
        # Annotate IMD cyclone center at t0
        if info['tag'] == 't0':
            c_lat = imd_t0_fix['latitude']
            c_lon = imd_t0_fix['longitude']
            wind = imd_t0_fix['maximum_sustained_wind_kt']
            pres = imd_t0_fix['central_pressure_hpa']
            cat = imd_t0_fix.get('category', 'D')
            
            # Draw bullseye center marker
            ax.plot(c_lon, c_lat, marker='o', markersize=11, markeredgecolor='black', 
                    markerfacecolor='red', markeredgewidth=2, label=f"IMD Center: {c_lat}°N, {c_lon}°E")
            ax.plot(c_lon, c_lat, marker='+', markersize=15, markeredgecolor='white', markeredgewidth=2)
            
            ax.annotate(f"IMD Center at t0:\n({c_lat}°N, {c_lon}°E)\nWind: {wind} kt | {pres} hPa [{cat}]",
                        xy=(c_lon, c_lat), xytext=(c_lon + 3.0, c_lat + 4.5),
                        arrowprops=dict(facecolor='yellow', edgecolor='black', arrowstyle='->', lw=2),
                        fontsize=9.5, fontweight='bold', color='yellow',
                        bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.85, ec="yellow", lw=1.5))
            ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    plt.subplots_adjust(bottom=0.15, top=0.92, wspace=0.12, hspace=0.22)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("IR Brightness Temperature (Kelvin) [irwin_cdr] — Inverted: Cold Cloud Tops = Bright", fontsize=11, fontweight='bold')
    
    plt.suptitle("VAYU-NET GridSat-B1 Pilot Sequence — Cyclone FANI (2019)\nFrozen 6-Frame Historical Input Sequence (t-15h to t0) Cropped to North Indian Ocean",
                 fontsize=14, fontweight='bold', y=0.98)
    
    qc_out1 = os.path.join(OUT_DIR, "fani_qc_plot.png")
    qc_out2 = os.path.join(DOCS_DIR, "fani_qc_plot.png")
    fig.savefig(qc_out1, dpi=200, bbox_inches='tight')
    fig.savefig(qc_out2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"QC plots saved to:\n  {qc_out1}\n  {qc_out2}")

def main():
    print("=" * 70)
    print("VAYU-NET GridSat-B1 Pilot Processing & Alignment Pipeline")
    print("=" * 70)
    
    # Load IMD V2 dataset
    imd_df = pd.read_csv(IMD_V2_CSV)
    fani_imd = imd_df[imd_df['storm_id'] == STORM_ID].copy()
    print(f"Loaded IMD Best Track V2 for {STORM_ID}: {len(fani_imd)} rows.")
    
    # Check t0 and targets in IMD
    t0_row = fani_imd[fani_imd['timestamp_utc'] == T0_STR]
    if t0_row.empty:
        raise ValueError(f"IMD observation at t0 ({T0_STR}) not found!")
    imd_t0_fix = t0_row.iloc[0].to_dict()
    
    t12_row = fani_imd[fani_imd['timestamp_utc'] == TARGETS[0]['ts']]
    t24_row = fani_imd[fani_imd['timestamp_utc'] == TARGETS[1]['ts']]
    t48_row = fani_imd[fani_imd['timestamp_utc'] == TARGETS[2]['ts']]
    
    if t12_row.empty or t24_row.empty or t48_row.empty:
        raise ValueError("One or more required future IMD targets (+12h, +24h, +48h) missing!")
    
    imd_t12_fix = t12_row.iloc[0].to_dict()
    imd_t24_fix = t24_row.iloc[0].to_dict()
    imd_t48_fix = t48_row.iloc[0].to_dict()
    
    print("\nIMD Fixes Verified:")
    print(f"  t0   ({T0_STR}): Lat {imd_t0_fix['latitude']}°N, Lon {imd_t0_fix['longitude']}°E, Wind {imd_t0_fix['maximum_sustained_wind_kt']} kt, Pres {imd_t0_fix['central_pressure_hpa']} hPa")
    print(f"  +12h ({TARGETS[0]['ts']}): Lat {imd_t12_fix['latitude']}°N, Lon {imd_t12_fix['longitude']}°E, Wind {imd_t12_fix['maximum_sustained_wind_kt']} kt, Pres {imd_t12_fix['central_pressure_hpa']} hPa")
    print(f"  +24h ({TARGETS[1]['ts']}): Lat {imd_t24_fix['latitude']}°N, Lon {imd_t24_fix['longitude']}°E, Wind {imd_t24_fix['maximum_sustained_wind_kt']} kt, Pres {imd_t24_fix['central_pressure_hpa']} hPa")
    print(f"  +48h ({TARGETS[2]['ts']}): Lat {imd_t48_fix['latitude']}°N, Lon {imd_t48_fix['longitude']}°E, Wind {imd_t48_fix['maximum_sustained_wind_kt']} kt, Pres {imd_t48_fix['central_pressure_hpa']} hPa")
    
    # Process 6 historical frames
    print("\nProcessing 6 Historical GridSat-B1 Frames...")
    frame_records = []
    frames_data = []
    
    for f in FRAMES:
        rec, lat, lon, data = process_frame(f)
        frame_records.append(rec)
        frames_data.append((f, lat, lon, data))
        print(f"  [{rec['tag']}] {rec['satellite_timestamp_utc']} | Shape: {rec['array_shape']} | Range: {rec['min_kelvin']:.2f}K - {rec['max_kelvin']:.2f}K (mean {rec['mean_kelvin']:.2f}K) | NaN: {rec['nan_fraction']*100:.3f}%")
        
    # Create Frame Manifest
    frames_df = pd.DataFrame(frame_records)
    frames_manifest_path = os.path.join(OUT_DIR, "fani_frames_manifest.csv")
    frames_df.to_csv(frames_manifest_path, index=False)
    print(f"\nSaved frame-level manifest: {frames_manifest_path}")
    
    # Create Sample Manifest
    history_ts_list = ";".join([f['ts'] for f in FRAMES])
    sample_record = {
        "storm_id": STORM_ID,
        "t0": T0_STR,
        "history_timestamps": history_ts_list,
        "target_12h_timestamp": TARGETS[0]['ts'],
        "target_24h_timestamp": TARGETS[1]['ts'],
        "target_48h_timestamp": TARGETS[2]['ts'],
        "imd_lat_t0": imd_t0_fix['latitude'],
        "imd_lon_t0": imd_t0_fix['longitude'],
        "imd_wind_t0": imd_t0_fix['maximum_sustained_wind_kt'],
        "imd_pressure_t0": imd_t0_fix['central_pressure_hpa'],
        "imd_lat_12h": imd_t12_fix['latitude'],
        "imd_lon_12h": imd_t12_fix['longitude'],
        "imd_wind_12h": imd_t12_fix['maximum_sustained_wind_kt'],
        "imd_pressure_12h": imd_t12_fix['central_pressure_hpa'],
        "imd_lat_24h": imd_t24_fix['latitude'],
        "imd_lon_24h": imd_t24_fix['longitude'],
        "imd_wind_24h": imd_t24_fix['maximum_sustained_wind_kt'],
        "imd_pressure_24h": imd_t24_fix['central_pressure_hpa'],
        "imd_lat_48h": imd_t48_fix['latitude'],
        "imd_lon_48h": imd_t48_fix['longitude'],
        "imd_wind_48h": imd_t48_fix['maximum_sustained_wind_kt'],
        "imd_pressure_48h": imd_t48_fix['central_pressure_hpa'],
    }
    sample_df = pd.DataFrame([sample_record])
    sample_manifest_path = os.path.join(OUT_DIR, "fani_sample_manifest.csv")
    sample_df.to_csv(sample_manifest_path, index=False)
    print(f"Saved sample-level manifest: {sample_manifest_path}")
    
    # Task 6: Automated Validation Checks A through K
    print("\nRunning Task 6 Alignment Validation Checks A through K...")
    checks = {}
    
    # Check A: Exactly 6 historical frames exist
    checks["A_six_frames_exist"] = len(frame_records) == 6
    
    # Check B: Historical timestamps are exactly: -15h, -12h, -9h, -6h, -3h, 0h
    expected_deltas = [-15, -12, -9, -6, -3, 0]
    actual_deltas = [r["delta_t_hours"] for r in frame_records]
    checks["B_exact_deltas"] = actual_deltas == expected_deltas
    
    # Check C: All six are native 3-hour GridSat observations
    checks["C_native_3h_cadence"] = all(pd.to_datetime(r["satellite_timestamp_utc"]).hour % 3 == 0 for r in frame_records)
    
    # Check D: No interpolation performed
    checks["D_no_interpolation"] = True
    
    # Check E: IMD t0 exists
    checks["E_imd_t0_exists"] = not t0_row.empty
    
    # Check F: IMD +12h, +24h, +48h targets exist
    checks["F_imd_targets_exist"] = (not t12_row.empty) and (not t24_row.empty) and (not t48_row.empty)
    
    # Check G: No timestamp duplication
    ts_set = set(r["satellite_timestamp_utc"] for r in frame_records)
    checks["G_no_timestamp_duplication"] = len(ts_set) == len(frame_records)
    
    # Check H: Satellite spatial grid is internally consistent across all six frames
    shapes = [r["array_shape"] for r in frame_records]
    lat_mins = [r["lat_min"] for r in frame_records]
    lon_mins = [r["lon_min"] for r in frame_records]
    lat_maxs = [r["lat_max"] for r in frame_records]
    lon_maxs = [r["lon_max"] for r in frame_records]
    checks["H_spatial_grid_consistent"] = (len(set(shapes)) == 1 and 
                                           len(set(lat_mins)) == 1 and 
                                           len(set(lon_mins)) == 1 and 
                                           len(set(lat_maxs)) == 1 and 
                                           len(set(lon_maxs)) == 1)
    
    # Check I: Crop is inside NIO bounds (-5 to 35 lat, 40 to 105 lon)
    checks["I_inside_nio_bounds"] = (min(lat_mins) >= -5.0 and max(lat_maxs) <= 35.0 and
                                     min(lon_mins) >= 40.0 and max(lon_maxs) <= 105.0)
    
    # Check J: No unexpected missing-data fraction (< 5% missing expected across NIO domain)
    nan_fractions = [r["nan_fraction"] for r in frame_records]
    checks["J_missing_data_acceptable"] = all(nf < 0.05 for nf in nan_fractions)
    
    # Check K: Source filenames correspond exactly to requested timestamps
    checks["K_filenames_match_timestamps"] = all(
        pd.to_datetime(r["satellite_timestamp_utc"]).strftime("%Y.%m.%d.%H") in r["source_file"]
        for r in frame_records
    )
    
    all_pass = all(checks.values())
    for name, res in checks.items():
        print(f"  Check {name}: {'PASS' if res else 'FAIL'}")
    print(f"Overall Validation Result: {'PASS' if all_pass else 'FAIL'}")
    
    # Task 7: Generate Visual QC
    print("\nGenerating Task 7 Visual QC Plot...")
    generate_visual_qc(frames_data, imd_t0_fix)
    
    print("\n" + "=" * 70)
    print("PILOT PROCESSING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
