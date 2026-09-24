"""
VAYU-NET — Stream-and-Crop Production GridSat-B1 Ingestion Pipeline
Executes resumable, host-storage-safe acquisition of all required GridSat files.

Architecture:
1. Reads data/manifests/gridsat_required_file_manifest.csv.
2. Identifies PENDING files.
3. For each file:
   - Downloads to transient staging folder (data/raw/staging/)
   - Validates NetCDF-4 integrity & irwin_cdr metadata
   - Physical decoding: Kelvin = packed * scale_factor + add_offset (set_auto_maskandscale(False))
   - Immediate North Indian Ocean crop: Lat [-5, 35], Lon [40, 105] (572x929)
   - Saves processed cropped tensor to data/interim/gridsat/{year}/ (.npz and .nc)
   - Removes temporary global raw file immediately
   - Updates manifest status: download_status=DOWNLOADED, file_size_bytes, validation_status=PASSED
4. Supports multi-threaded parallel execution.
5. Real-time manifest persistence every batch.
"""

import os
import sys
import time
import shutil
import subprocess
import threading
import numpy as np
import pandas as pd
import netCDF4
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST_PATH = "data/manifests/gridsat_required_file_manifest.csv"
STAGING_DIR = "data/raw/staging"
INTERIM_DIR = "data/interim/gridsat"
PILOT_RAW_DIR = "data/raw/gridsat"

LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = 40.0, 105.0

manifest_lock = threading.Lock()

def get_processed_paths(year, fname):
    base_name = fname.replace(".v02r01.nc", "").replace("GRIDSAT-B1.", "gridsat_")
    out_dir_year = os.path.join(INTERIM_DIR, str(year))
    out_nc = os.path.join(out_dir_year, f"{base_name}.nc")
    out_npz = os.path.join(out_dir_year, f"{base_name}.npz")
    return out_nc, out_npz

def download_file(url, dest_path, max_retries=5):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 20 * 1024 * 1024:
        return True, os.path.getsize(dest_path), "EXISTS_OK"
        
    cmd = [
        "curl.exe",
        "-C", "-",
        "--connect-timeout", "15",
        "--max-time", "180",
        "--retry", str(max_retries),
        "--retry-delay", "2",
        "--retry-all-errors",
        "-s", "-S",
        "-o", dest_path,
        url
    ]
    ret = subprocess.run(cmd)
    if ret.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True, os.path.getsize(dest_path), "DOWNLOADED_OK"
    else:
        return False, 0, f"CURL_ERROR_{ret.returncode}"

def process_and_crop(raw_path, year, fname, timestamp_utc):
    out_nc, out_npz = get_processed_paths(year, fname)
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    
    if os.path.exists(out_nc) and os.path.exists(out_npz):
        return True, "ALREADY_PROCESSED", out_nc, out_npz, (572, 929), 0.0
        
    try:
        nc = netCDF4.Dataset(raw_path, 'r')
        nc.set_auto_maskandscale(False)
        
        if 'irwin_cdr' not in nc.variables:
            nc.close()
            return False, "MISSING_IRWIN_CDR", None, None, None, None
            
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        
        lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
        lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
        
        cropped_lat = lat[lat_mask]
        cropped_lon = lon[lon_mask]
        
        var = nc.variables['irwin_cdr']
        scale_factor = getattr(var, 'scale_factor', 0.01)
        add_offset = getattr(var, 'add_offset', 200.0)
        fill_val = getattr(var, '_FillValue', -31999)
        miss_val = getattr(var, 'missing_value', -31999)
        
        raw_slice = var[0, lat_mask, :][:, lon_mask]
        nc.close()
        
        is_missing = (raw_slice == fill_val) | (raw_slice == miss_val) | (raw_slice <= -30000)
        decoded = raw_slice.astype(np.float32) * float(scale_factor) + float(add_offset)
        decoded[is_missing] = np.nan
        
        nan_fraction = float(np.isnan(decoded).sum() / decoded.size)
        grid_shape = decoded.shape
        
        # Save NetCDF-4
        with netCDF4.Dataset(out_nc, 'w', format='NETCDF4') as out_ds:
            out_ds.createDimension('lat', len(cropped_lat))
            out_ds.createDimension('lon', len(cropped_lon))
            
            vlat = out_ds.createVariable('lat', 'f4', ('lat',))
            vlat.units = 'degrees_north'
            vlat[:] = cropped_lat
            
            vlon = out_ds.createVariable('lon', 'f4', ('lon',))
            vlon.units = 'degrees_east'
            vlon[:] = cropped_lon
            
            vir = out_ds.createVariable('irwin_cdr', 'f4', ('lat', 'lon'), fill_value=np.nan, zlib=True)
            vir.units = 'Kelvin'
            vir[:] = decoded
            
            out_ds.timestamp_utc = timestamp_utc
            out_ds.source_filename = fname
            out_ds.scale_factor = float(scale_factor)
            out_ds.add_offset = float(add_offset)
            out_ds.crop_bounds = f"lat:[{LAT_MIN},{LAT_MAX}], lon:[{LON_MIN},{LON_MAX}]"

        # Save NPZ
        np.savez_compressed(out_npz,
                            irwin_cdr=decoded,
                            lat=cropped_lat,
                            lon=cropped_lon,
                            timestamp_utc=timestamp_utc)
                            
        return True, "PROCESSED_OK", out_nc, out_npz, grid_shape, nan_fraction
        
    except Exception as e:
        return False, f"PROCESSING_ERROR: {str(e)}", None, None, None, None

def process_single_file(item, is_pilot=False):
    idx, row = item
    yr = row['year']
    fname = row['source_filename']
    url = row['source_url']
    ts = row['satellite_timestamp_utc']
    
    out_nc, out_npz = get_processed_paths(yr, fname)
    if os.path.exists(out_nc) and os.path.exists(out_npz):
        sz = os.path.getsize(out_nc)
        return idx, "DOWNLOADED", sz, "PASSED", "STREAM_CROPPED_AND_VALIDATED"
        
    # Check if raw pilot file exists in data/raw/gridsat/{yr}/
    pilot_p = os.path.join(PILOT_RAW_DIR, str(yr), fname)
    raw_p = os.path.join(STAGING_DIR, f"thread_{threading.get_ident()}_{fname}")
    
    if os.path.exists(pilot_p) and os.path.getsize(pilot_p) > 20 * 1024 * 1024:
        src_for_crop = pilot_p
        raw_sz = os.path.getsize(pilot_p)
        dl_ok = True
        remove_raw = False # Keep pilot raw file per instructions
    else:
        dl_ok, raw_sz, dl_msg = download_file(url, raw_p)
        src_for_crop = raw_p
        remove_raw = True
        if not dl_ok:
            if os.path.exists(raw_p): os.remove(raw_p)
            return idx, "FAILED", 0, "FAILED", dl_msg
            
    # Process & Crop
    p_ok, p_msg, nc_p, npz_p, shape, nan_f = process_and_crop(src_for_crop, yr, fname, ts)
    
    # Remove transient raw file if not pilot
    if remove_raw and os.path.exists(raw_p):
        try:
            os.remove(raw_p)
        except Exception:
            pass
            
    if p_ok:
        return idx, "DOWNLOADED", raw_sz, "PASSED", "STREAM_CROPPED_AND_VALIDATED"
    else:
        return idx, "FAILED", raw_sz, "FAILED", p_msg

def main(max_workers=6, limit=None):
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    
    df = pd.read_csv(MANIFEST_PATH)
    
    pending_items = []
    for idx, row in df.iterrows():
        yr = row['year']
        fname = row['source_filename']
        out_nc, out_npz = get_processed_paths(yr, fname)
        if not (os.path.exists(out_nc) and os.path.exists(out_npz)):
            pending_items.append((idx, row))
            
    print(f"Total files in manifest: {len(df)}", flush=True)
    print(f"Already processed: {len(df) - len(pending_items)}", flush=True)
    print(f"Pending to stream-and-crop: {len(pending_items)}", flush=True)
    
    if limit is not None:
        pending_items = pending_items[:limit]
        print(f"Processing limit applied: {len(pending_items)} files", flush=True)
        
    if not pending_items:
        print("All files are already processed!", flush=True)
        return
        
    start_time = time.time()
    completed = 0
    total = len(pending_items)
    
    print(f"Launching Stream-and-Crop with {max_workers} worker threads...", flush=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, item): item for item in pending_items}
        
        for future in as_completed(futures):
            item = futures[future]
            try:
                idx, status, sz, v_status, v_msg = future.result()
                
                with manifest_lock:
                    df.at[idx, 'download_status'] = status
                    df.at[idx, 'file_size_bytes'] = sz
                    df.at[idx, 'validation_status'] = v_status
                    df.at[idx, 'validation_message'] = v_msg
                    
                completed += 1
                if completed % 5 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{total} ({completed*100/total:.1f}%) | {rate:.2f} files/s | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m", flush=True)
                    with manifest_lock:
                        df.to_csv(MANIFEST_PATH, index=False)
                        
            except Exception as e:
                print(f"Error on {item[1]['source_filename']}: {e}", flush=True)
                
    with manifest_lock:
        df.to_csv(MANIFEST_PATH, index=False)
        
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print(f"\nStream-and-crop completed in {time.time() - start_time:.1f}s!", flush=True)

if __name__ == "__main__":
    workers = 6
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_workers=workers, limit=lim)
