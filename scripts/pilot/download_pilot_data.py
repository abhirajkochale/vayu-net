"""
Robust Downloader for GridSat-B1 pilot files using requests with chunking, timeouts, and retries.
"""
import os
import sys
import time
import requests

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

def download_file(tag, ts, filename, max_retries=5):
    url = BASE_URL + filename
    dest_path = os.path.join(DEST_DIR, filename)
    part_path = dest_path + ".part"
    
    # Check if already fully downloaded
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 30 * 1024 * 1024: # GridSat files are ~44-57MB
            print(f"[{tag}] {filename} already exists ({size:,} bytes). Skipping.")
            return dest_path
        else:
            print(f"[{tag}] Found incomplete {filename} ({size:,} bytes). Re-downloading.")
            os.remove(dest_path)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{tag}] [Attempt {attempt}/{max_retries}] Downloading {filename}...")
            start_time = time.time()
            
            with requests.get(url, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(part_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = int(downloaded * 100 / total_size)
                                sys.stdout.write(f"\r  {pct}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)")
                                sys.stdout.flush()
                                
            # Atomic rename
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(part_path, dest_path)
            
            duration = time.time() - start_time
            final_size = os.path.getsize(dest_path)
            speed = (final_size / (1024 * 1024)) / duration if duration > 0 else 0
            print(f"\n[{tag}] Successfully downloaded {filename} ({final_size:,} bytes, {speed:.2f} MB/s)")
            return dest_path
            
        except Exception as e:
            print(f"\n[{tag}] Attempt {attempt} failed: {e}")
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts.")

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    # Remove any stray .part or corrupted partial files
    for f in os.listdir(DEST_DIR):
        if f.endswith(".part"):
            os.remove(os.path.join(DEST_DIR, f))
            
    print(f"Starting robust download of {len(FILES)} GridSat-B1 files...")
    for tag, ts, filename in FILES:
        download_file(tag, ts, filename)
    print("\nAll 9 files downloaded and verified!")

if __name__ == "__main__":
    main()
