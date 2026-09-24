"""
VAYU-NET PHASE 5A — HIGH-PERFORMANCE ERA5 CLOUD STREAMING ACQUISITION
======================================================================
Streams official ECMWF ERA5 pressure-level U and V wind components from
Google Cloud Public Datasets (gcp-public-data-arco-era5), crops to the
North Indian Ocean domain (-5 to 35 N, 40 to 105 E) across 4 pressure
levels (850, 700, 500, 300 hPa), downsamples to (41, 66), and stores
each timestamp as an 86 KB NPZ array in data/raw/era5/.

Memory-bounded, disk-safe, fully resumable, and multi-threaded.
"""

import os
import sys
import time
import hashlib
import requests
import numcodecs
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_TIME = np.datetime64("1900-01-01T00:00:00")
URL_TEMPLATE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/{var}/{idx}.0.0.0"

# Target levels: 850 (idx 30), 700 (idx 25), 500 (idx 21), 300 (idx 17)
LEVEL_INDICES = [30, 25, 21, 17]
# Lat: 35.0 to -5.0 (idx 220:381)
LAT_SLICE = slice(220, 381)
# Lon: 40.0 to 105.0 (idx 160:421)
LON_SLICE = slice(160, 421)

CODEC = numcodecs.Blosc(cname='lz4', clevel=5)

def timestamp_to_idx(iso_ts):
    clean_ts = iso_ts.replace("Z", "")
    dt = np.datetime64(clean_ts)
    diff_h = int((dt - BASE_TIME) / np.timedelta64(1, 'h'))
    return diff_h

def fetch_and_slice_var(var_name, time_idx, session=None):
    url = URL_TEMPLATE.format(var=var_name, idx=time_idx)
    s = session or requests
    r = s.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url}: HTTP {r.status_code}")
    raw = np.frombuffer(CODEC.decode(r.content), dtype='<f4').reshape(37, 721, 1440)
    sliced = raw[LEVEL_INDICES, LAT_SLICE, LON_SLICE] # [4, 161, 261]
    return sliced

def process_single_timestamp(iso_ts, out_dir):
    # iso_ts e.g. 1998-10-06T12:00:00Z
    dt_str = iso_ts.replace("Z", "").replace(":", "").replace("-", "")
    # Format: YYYY.MM.DD.HH
    y = dt_str[:4]
    m = dt_str[4:6]
    d = dt_str[6:8]
    h = dt_str[9:11]
    filename = f"era5_{y}.{m}.{d}.{h}.npz"
    filepath = os.path.join(out_dir, filename)

    if os.path.exists(filepath):
        # Already exists, verify integrity
        try:
            data = np.load(filepath)
            if 'env' in data and data['env'].shape == (8, 41, 66):
                return iso_ts, filepath, filename, True, "EXISTS"
        except Exception:
            pass # corrupted, re-download

    time_idx = timestamp_to_idx(iso_ts)
    
    with requests.Session() as s:
        u_sliced = fetch_and_slice_var('u_component_of_wind', time_idx, session=s)
        v_sliced = fetch_and_slice_var('v_component_of_wind', time_idx, session=s)

    # Combine into 8 channels [8, 161, 261]
    # Channels 0-3: U at 850, 700, 500, 300 hPa
    # Channels 4-7: V at 850, 700, 500, 300 hPa
    env_raw = np.concatenate([u_sliced, v_sliced], axis=0) # [8, 161, 261]
    
    # Bilinear downsample to [8, 41, 66]
    tensor_raw = torch.from_numpy(env_raw).unsqueeze(0).float()
    tensor_down = F.interpolate(tensor_raw, size=(41, 66), mode='bilinear', align_corners=True).squeeze(0)
    env_down = tensor_down.numpy().astype(np.float32)

    # Save to disk
    np.savez_compressed(filepath, env=env_down, timestamp=iso_ts, levels=np.array([850, 700, 500, 300]))
    
    # Compute SHA256
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return iso_ts, filepath, filename, True, file_hash

def run_streaming_acquisition(manifest_path, out_dir, max_workers=6, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} entries.")
    
    # Filter for pending
    pending = []
    for _, row in df.iterrows():
        ts = row['timestamp']
        dt_str = ts.replace("Z", "").replace(":", "").replace("-", "")
        y, m, d, h = dt_str[:4], dt_str[4:6], dt_str[6:8], dt_str[9:11]
        fname = f"era5_{y}.{m}.{d}.{h}.npz"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            pending.append(ts)
            
    print(f"Pending timestamps: {len(pending)} / {len(df)}")
    if limit is not None:
        pending = pending[:limit]
        print(f"Limiting execution to {limit} timestamps.")

    if not pending:
        print("All requested timestamps already acquired!")
        return

    t0_start = time.time()
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_timestamp, ts, out_dir): ts for ts in pending}
        for future in as_completed(futures):
            ts = futures[future]
            try:
                iso_ts, fpath, fname, ok, info = future.result()
                completed += 1
                if completed % 5 == 0 or completed == len(pending):
                    elapsed = time.time() - t0_start
                    rate = completed / elapsed
                    eta_sec = (len(pending) - completed) / rate if rate > 0 else 0
                    print(f"[{completed}/{len(pending)}] Acquired {fname} ({info[:8]}) | {rate:.2f} ts/s | ETA: {eta_sec/60:.1f} min")
            except Exception as e:
                errors += 1
                print(f"ERROR on {ts}: {e}")

    print(f"Completed run: {completed} acquired, {errors} errors in {time.time() - t0_start:.2f} s.")

if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_streaming_acquisition(
        manifest_path='data/manifests/era5_timestamp_manifest.csv',
        out_dir='data/raw/era5',
        max_workers=6,
        limit=limit
    )
