"""
Fast and robust downloader using curl.exe for GridSat-B1 pilot files.
"""
import os
import subprocess
import sys

FILES = [
    # 6 historical frames
    ("t-15h", "2019-04-25T15:00:00Z", "GRIDSAT-B1.2019.04.25.15.v02r01.nc"),
    ("t-12h", "2019-04-25T18:00:00Z", "GRIDSAT-B1.2019.04.25.18.v02r01.nc"),
    ("t-9h",  "2019-04-25T21:00:00Z", "GRIDSAT-B1.2019.04.25.21.v02r01.nc"),
    ("t-6h",  "2019-04-26T00:00:00Z", "GRIDSAT-B1.2019.04.26.00.v02r01.nc"),
    ("t-3h",  "2019-04-26T03:00:00Z", "GRIDSAT-B1.2019.04.26.03.v02r01.nc"),
    ("t0",    "2019-04-26T06:00:00Z", "GRIDSAT-B1.2019.04.26.06.v02r01.nc"),
    # 3 target reference frames
    ("t+12h", "2019-04-26T18:00:00Z", "GRIDSAT-B1.2019.04.26.18.v02r01.nc"),
    ("t+24h", "2019-04-27T06:00:00Z", "GRIDSAT-B1.2019.04.27.06.v02r01.nc"),
    ("t+48h", "2019-04-28T06:00:00Z", "GRIDSAT-B1.2019.04.28.06.v02r01.nc"),
]

BASE_URL = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/2019/"
DEST_DIR = "data/interim/gridsat_pilot/fani_2019/raw"

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print("Checking / downloading files using curl.exe...")
    
    for tag, ts, filename in FILES:
        url = BASE_URL + filename
        dest = os.path.join(DEST_DIR, filename)
        
        if os.path.exists(dest) and os.path.getsize(dest) > 35 * 1024 * 1024:
            print(f"[{tag}] {filename} exists ({os.path.getsize(dest):,} bytes). OK.")
            continue
            
        print(f"[{tag}] Downloading {filename} via curl.exe...")
        cmd = [
            "curl.exe",
            "-C", "-",
            "--retry", "5",
            "--retry-delay", "2",
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

    print("\nAll 9 files downloaded and verified!")

if __name__ == "__main__":
    main()
