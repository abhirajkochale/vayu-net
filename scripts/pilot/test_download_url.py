import urllib.request
import urllib.error

urls = [
    "https://noaa-gridsat-pds.s3.amazonaws.com/data/v02r01/2019/GridSat-B1.2019.04.25.15.v02r01.nc",
    "https://noaa-gridsat-pds.s3.amazonaws.com/2019/GridSat-B1.2019.04.25.15.v02r01.nc",
    "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/2019/GridSat-B1.2019.04.25.15.v02r01.nc"
]

for url in urls:
    req = urllib.request.Request(url, method='HEAD')
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"SUCCESS: {url} -> Status: {resp.status}, Size: {resp.headers.get('Content-Length')}")
    except Exception as e:
        print(f"FAILED: {url} -> {e}")
