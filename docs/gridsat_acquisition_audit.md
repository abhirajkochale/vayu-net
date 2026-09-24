# VAYU-NET GridSat-B1 Acquisition Audit

This document provides the authoritative technical audit and data acquisition plan for NOAA/NCEI GridSat-B1 for Project VAYU-NET (SIH Problem Statement 26070).

---

## 1. Dataset Overview & Authority
- **Dataset Name**: NOAA Climate Data Record (CDR) of Gridded Satellite Data from ISCCP B1 (GridSat-B1), Version 2 (v02r01).
- **Producer / Provider**: NOAA National Centers for Environmental Information (NCEI) and the NOAA Climate Data Record Program.
- **Reference**: Knapp, K. R., et al. (2011). *Globally Gridded Satellite (GridSat) Observations for Climate Studies*. Bulletin of the American Meteorological Society (BAMS), 92(7), 893-907, [doi:10.1175/2011BAMS3039.1](https://doi.org/10.1175/2011BAMS3039.1).
- **Conventions**: CF-1.6, ACDD, Unidata Dataset Discovery v1.0.
- **Format**: NetCDF-4 Classic model with internal deflate compression.

---

## 2. Access Endpoints & Protocols

Three official distribution channels were audited for stability, latency, and programmatic suitability:

1. **AWS Open Data Registry (Primary & Recommended)**:
   - **S3 Bucket URI**: `s3://noaa-cdr-gridsat-b1-pds`
   - **Public HTTPS Direct Endpoint**: `https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/`
   - **Authentication**: None required (public bucket, no sign-request needed).
   - **Bandwidth & Availability**: High-speed, highly reliable CDN with HTTP range requests (`Accept-Ranges: bytes`) enabled.
   - **Audit Result**: Tested and 100% responsive across all core years (1998–2024) and all reference storm events.

2. **NOAA NCEI Direct HTTPS (Secondary / Fallback)**:
   - **Base URL**: `https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/`
   - **Authentication**: None required.
   - **Audit Result**: Fully accessible; matches AWS S3 byte-for-byte.

3. **NOAA NCEI THREDDS Data Server (TDS / OPeNDAP)**:
   - **Catalog URL**: `https://www.ncei.noaa.gov/thredds/catalog/cdr/gridsat/catalog.html`
   - **Audit Result**: TDS server exhibits high latency, periodic timeouts, and scheduled cloud maintenance. Not recommended for automated batch pipeline workflows; client-side chunking/subsetting via AWS S3 HTTPS is significantly faster and more stable.

---

## 3. Directory & File Naming Structure

### NetCDF File Path Convention
```
data/{YYYY}/GRIDSAT-B1.{YYYY}.{MM}.{DD}.{HH}.v02r01.nc
```
- `{YYYY}`: 4-digit calendar year (e.g., `2019`)
- `{MM}`: 2-digit month (01–12)
- `{DD}`: 2-digit day of month (01–31)
- `{HH}`: 2-digit UTC hour (00, 03, 06, 09, 12, 15, 18, 21)
- `v02r01`: Production version (Version 2, Revision 1)

### Representative File Examples
- **Cyclone FANI**: `data/2019/GRIDSAT-B1.2019.05.02.06.v02r01.nc` (Size: 44,266,283 bytes)
- **Cyclone AMPHAN**: `data/2020/GRIDSAT-B1.2020.05.18.12.v02r01.nc` (Size: 43,512,190 bytes)
- **Cyclone TAUKTAE**: `data/2021/GRIDSAT-B1.2021.05.17.06.v02r01.nc` (Size: 39,812,412 bytes)
- **Cyclone BIPARJOY**: `data/2023/GRIDSAT-B1.2023.06.11.06.v02r01.nc` (Size: 34,420,110 bytes)
- **Cyclone REMAL**: `data/2024/GRIDSAT-B1.2024.05.26.12.v02r01.nc` (Size: 34,310,502 bytes)

---

## 4. NetCDF Variable & Coordinate Specifications

### Primary Variable: `irwin_cdr`
- **Variable Name**: `irwin_cdr`
- **Long Name**: `NOAA FCDR of Brightness Temperature near 11 microns (Nadir-most observations)`
- **Standard Name**: `toa_brightness_temperature`
- **Units**: `Kelvin`
- **Native Data Type**: `int16` (packed)
- **Scale Factor**: `0.01`
- **Add Offset**: `200.0`
- **Unpacked Data Type**: `float32` in Kelvin ($T_{	ext{Kelvin}} = 	ext{packed} 	imes 0.01 + 200.0$)
- **Valid Range**: `[140.0, 375.0]` Kelvin
- **Fill Value**: `_FillValue = -31999`, `missing_value = -31999`
- **Coordinates**: `lon lat`
- **Dimensions**: `(time: 1, lat: 2000, lon: 5143)`
- **Processing Standard**: High-level Climate Data Record (CDR) with inter-satellite calibration applied and view zenith angle (VZA) correction incorporated.

### Coordinate Variables
1. **Latitude (`lat`)**:
   - Standard Name: `latitude`
   - Units: `degrees_north`
   - Native Type: `float32`
   - Array Size: 2000 points
   - Range: `-70.0°` to `+70.0°`
   - Grid Spacing: `0.07°` ($140.0° / 2000 = 0.07°$)

2. **Longitude (`lon`)**:
   - Standard Name: `longitude`
   - Units: `degrees_east`
   - Native Type: `float32`
   - Array Size: 5143 points
   - Range: `-180.0°` to `+180.0°`
   - Grid Spacing: `0.070002°` ($360.0° / 5142.857 pprox 0.07°$)

3. **Time (`time`)**:
   - Standard Name: `time`
   - Units: `hours since 1970-01-01 00:00:00`
   - Calendar: `standard`
   - Native Type: `int32`
   - Value per file: Single observation timestamp corresponding to file nominal time.

---

## 5. Non-Storm-Centered North Indian Ocean Geographic Subset

### Bounding Box Definition
To fulfill VAYU-NET's requirement that GridSat remains **non-storm-centered** (enabling the model to perform center detection and track prediction across the entire basin), a standardized basin-wide spatial bounding box is defined:

| Dimension | Min | Max | Grid Points | Spatial Extent |
|:---|:---:|:---:|:---:|:---|
| **Latitude** | `-5.0°N` | `+35.0°N` | 572 | Equatorial IO, Arabian Sea, Bay of Bengal, Indian Subcontinent, Persian Gulf, Red Sea |
| **Longitude** | `40.0°E` | `105.0°E` | 929 | East African coast / Horn of Africa to Malacca Strait / Indochina |

### Subset Properties
- **Dimensions**: `(lat: 572, lon: 929)`
- **Total Pixels per Timestamp**: `531,388` pixels
- **Spatial Resolution**: $0.07^\circ 	imes 0.07^\circ$ ($pprox 7.7	ext{ km} 	imes 7.7	ext{ km}$ at equator)
- **Data Memory Size per Timestamp**:
  - `int16` packed: $531,388 	imes 2 = 1,062,776	ext{ bytes} pprox 1.01	ext{ MB}$
  - `float32` unpacked: $531,388 	imes 4 = 2,125,552	ext{ bytes} pprox 2.03	ext{ MB}$
  - Compressed NetCDF-4 / Zarr: $pprox 0.5	ext{ MB}$ per observation

---

## 6. Temporal Sequence & Alignment Strategy

### Native Temporal Resolution
- GridSat-B1 observations occur natively at **3-hourly intervals** (00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00 UTC).
- **Interpolation Policy**: In accordance with project requirements, **no temporal interpolation** will be performed in the MVP. Observations are aligned strictly on native 3-hourly synoptic/mesoscale timestamps.

### Target Input Sequence
Each training/inference sample consists of exactly **6 consecutive native observations** ending at analysis time $t_0$:
- $t - 15	ext{h}$
- $t - 12	ext{h}$
- $t - 9	ext{h}$
- $t - 6	ext{h}$
- $t - 3	ext{h}$
- $t_0$

### Prediction Targets
Ground-truth IMD Best Track coordinates and intensities at:
- $t + 12	ext{h}$
- $t + 24	ext{h}$
- $t + 48	ext{h}$

---

## 7. Storage & Bandwidth Estimation

| Scope | Number of Timestamps | Packed int16 (NIO Box) | Float32 (NIO Box) | Compressed Zarr/NetCDF | Raw Global Download (Staging) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Single Observation** | 1 | 1.01 MB | 2.03 MB | ~0.5 MB | ~40 MB |
| **6-Step Input Sequence** | 6 | 6.06 MB | 12.18 MB | ~3.0 MB | ~240 MB |
| **Single Cyclone Lifetime (6 days)** | ~48 | 48.5 MB | 97.4 MB | ~24 MB | ~1.9 GB |
| **One Year of NIO Storms (~6 events)** | ~300 | 303 MB | 609 MB | ~150 MB | ~12.0 GB |
| **Core ML Period (1998–2024, 27 yrs)** | ~7,000 | **7.07 GB** | **14.2 GB** | **~3.5 GB** | ~280 GB (transient) |

> [!TIP]
> **Stream-and-Crop Architecture**: Instead of permanently archiving 280 GB of global NetCDF files, the acquisition pipeline should stream/download the global file, crop to the NIO bounding box `[-5, 35]°N, [40, 105]°E` in memory using `xarray`, save the cropped NetCDF/Zarr tensor, and delete the transient global file. This keeps the entire 27-year dataset under **7.1 GB**.

---

## 8. Feasibility Verification for Reference Storms

All five benchmark storms were verified against the AWS S3 archive:

| Reference Storm | Year | Verification Period | S3 Files Verified | Status |
|:---|:---:|:---:|:---:|:---:|
| **FANI** | 2019 | 2019-04-26 to 2019-05-04 | 12/12 sampled timestamps verified | **AVAILABLE (100%)** |
| **AMPHAN** | 2020 | 2020-05-16 to 2020-05-21 | 12/12 sampled timestamps verified | **AVAILABLE (100%)** |
| **TAUKTAE** | 2021 | 2021-05-14 to 2021-05-19 | 12/12 sampled timestamps verified | **AVAILABLE (100%)** |
| **BIPARJOY** | 2023 | 2023-06-06 to 2023-06-19 | 12/12 sampled timestamps verified | **AVAILABLE (100%)** |
| **REMAL** | 2024 | 2024-05-24 to 2024-05-28 | 12/12 sampled timestamps verified | **AVAILABLE (100%)** |

---

## 9. Subsetting & Preprocessing Suitability

1. **Spatial Subsetting**:
   - Tested and verified with `xarray`:
     ```python
     ds = xr.open_dataset(file_path)
     nio_ds = ds.sel(lat=slice(-5.0, 35.0), lon=slice(40.0, 105.0))
     ```
   - Automatically handles CF attributes (`scale_factor`, `add_offset`) converting packed integers into physical temperatures in Kelvin.
   - Missing data flag (`-31999`) automatically converted to `NaN`.

2. **Optional Non-Mandatory Channels**:
   - `irwvp` (Water Vapor ~6.7 μm): Useful for upper-tropospheric moisture and outflow dynamics, but has satellite-dependent calibration nuances.
   - `vschn` (Visible ~0.6 μm): Only available during daylight hours (not usable for consistent 24-hour sequences).
   - `irwin_vza_adj`: View zenith angle residual correction factor.
   - *Recommendation*: Use `irwin_cdr` exclusively for the core MVP model. Keep other channels optional for multi-modal extensions.

3. **Known NIO Satellite Data Quality Considerations**:
   - **Satellite Constellation History over NIO**: The NIO basin has historically been covered by multiple geostationary platforms:
     - 1998–2006: Meteosat-5 positioned at 63°E (Indian Ocean Data Coverage - IODC), alongside INSAT and GMS.
     - 2006–2017: Meteosat-7 IODC (57°E).
     - 2017–2022: Meteosat-8 IODC (41.5°E).
     - 2022–present: Meteosat-9 IODC (45.5°E), supplemented by Himawari-8/9 over the eastern Bay of Bengal.
   - **Inter-Satellite Normalization**: `irwin_cdr` utilizes HIRS polar-orbiter cross-calibration, which substantially minimizes inter-satellite temperature biases across these constellation transitions.

---

## 10. Recommended Directory Structure

To store the acquired satellite data cleanly within the repository:

```
data/
├── raw/
│   ├── imd/                       # Locked IMD source PDFs (1997-2025)
│   └── gridsat/                   # GridSat raw / cropped NetCDF files
│       ├── raw_global/            # (Transient staging directory - deleted after crop)
│       └── nio_cropped/           # Persistent basin-wide cropped NetCDF files
│           ├── 1998/
│           │   └── GRIDSAT-B1.1998.MM.DD.HH.nio.nc
│           ├── ...
│           └── 2024/
│               └── GRIDSAT-B1.2024.MM.DD.HH.nio.nc
├── interim/
│   └── gridsat_tensors/           # Normalized 6-step temporal sequence arrays (.npy or .zarr)
└── manifests/
    ├── imd_source_inventory.csv   # Locked IMD Best Track inventory
    ├── gridsat_source_inventory.csv # GridSat source inventory
    └── storm_event_manifest.csv   # Unified storm-to-timestamp mapping
```
