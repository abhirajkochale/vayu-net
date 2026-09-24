"""
Download GridSat-B1 NetCDF files for Cyclone AMPHAN (2020) pilot.
6 historical frames + 3 target reference frames.
Uses curl.exe for high throughput and resume resilience.
"""
import os
import subprocess
import sys

FILES = [
    # 6 historical frames
    ("t-15h", "2020-05-15T09:00:00Z", "GRIDSAT-B1.2020.05.15.09.v02r01.nc"),
    ("t-12h", "2020-05-15T12:00:00Z", "GRIDSAT-B1.2020.05.15.12.v02r01.nc"),
    ("t-9h",  "2020-05-15T15:00:00Z", "GRIDSAT-B1.2020.05.15.15.v02r01.nc"),
    ("t-6h",  "2020-05-15T18:00:00Z", "GRIDSAT-B1.2020.05.15.18.v02r01.nc"),
    ("t-3h",  "2020-05-15T21:00:00Z", "GRIDSAT-B1.2020.05.15.21.v02r01.nc"),
    ("t0",    "2020-05-16T00:00:00Z", "GRIDSAT-B1.2020.05.16.00.v02r01.nc"),
    # 3 target reference frames
    ("t+12h", "2020-05-16T12:00:00Z", "GRIDSAT-B1.2020.05.16.12.v02r01.nc"),
    ("t+24h", "2020-05-17T00:00:00Z", "GRIDSAT-B1.2020.05.17.00.v02r01.nc"),
    ("t+48h", "2020-05-18T00:00:00Z", "GRIDSAT-B1.2020.05.18.00.v02r01.nc"),
]

BASE_URL = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/2020/"
DEST_DIR = "data/interim/gridsat_pilot/amphan_2020/raw"

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading {len(FILES)} GridSat-B1 NetCDF files for Cyclone AMPHAN to {DEST_DIR}...")
    
    for tag, ts, filename in FILES:
        url = BASE_URL + filename
        dest = os.path.join(DEST_DIR, filename)
        
        if os.path.exists(dest) and os.path.getsize(dest) > 35 * 1024 * 1024:
            print(f"[{tag}] {filename} already exists ({os.path.getsize(dest):,} bytes). Skipping.")
            continue
            
        print(f"[{tag}] Downloading {filename} via curl.exe...")
        cmd = [
            "curl.exe",
            "-C", "-",
            "--connect-timeout", "15",
            "--max-time", "180",
            "--retry", "5",
            "--retry-delay", "2",
            "--retry-all-errors",
            "-s", "-S",
            "-o", dest,
            url
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Error downloading {filename}: exit code {ret.returncode}")
            sys.exit(1)
        size = os.path.getsize(dest)
        print(f"[{tag}] Successfully downloaded {filename} ({size:,} bytes).")

    print("\nAll 9 AMPHAN GridSat files verified/downloaded!")

if __name__ == "__main__":
    main()
