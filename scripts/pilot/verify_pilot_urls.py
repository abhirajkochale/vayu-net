import requests

base_url = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/2019/"
files = [
    "GRIDSAT-B1.2019.04.25.15.v02r01.nc",
    "GRIDSAT-B1.2019.04.25.18.v02r01.nc",
    "GRIDSAT-B1.2019.04.25.21.v02r01.nc",
    "GRIDSAT-B1.2019.04.26.00.v02r01.nc",
    "GRIDSAT-B1.2019.04.26.03.v02r01.nc",
    "GRIDSAT-B1.2019.04.26.06.v02r01.nc",
    "GRIDSAT-B1.2019.04.26.18.v02r01.nc",
    "GRIDSAT-B1.2019.04.27.06.v02r01.nc",
    "GRIDSAT-B1.2019.04.28.06.v02r01.nc"
]

all_ok = True
for fn in files:
    r = requests.head(base_url + fn, timeout=10)
    sz = int(r.headers.get("Content-Length", 0))
    print(f"{fn}: status={r.status_code}, size={sz:,} bytes")
    if r.status_code != 200:
        all_ok = False

print(f"\nAll 9 files available: {all_ok}")
