# VAYU-NET GridSat-B1 Acquisition — Executive Summary

This document provides a concise summary of the GridSat-B1 satellite data audit and acquisition plan for Project VAYU-NET (SIH Problem Statement 26070).

---

## Audit Key Facts

| Item | Specification | Operational Recommendation |
|:---|:---|:---|
| **Dataset** | NOAA GridSat-B1 Climate Data Record (v02r01) | Primary historical satellite source for VAYU-NET |
| **Primary Variable** | `irwin_cdr` (Brightness Temperature near 11 μm) | Standardized, inter-calibrated IR window channel |
| **Variable Units** | Kelvin (stored as int16, unpacked to float32) | $T = 	ext{packed} 	imes 0.01 + 200.0$ |
| **Fill Value** | `-31999` | Automatically masked to `NaN` in xarray |
| **Spatial Resolution** | 0.07° equal-angle grid (~7.7 km at equator) | Native cylindrical projection |
| **Temporal Resolution** | 3-hourly native (00, 03, 06, 09, 12, 15, 18, 21 UTC) | Exact match for VAYU-NET temporal sequences |
| **Temporal Stride** | 6 consecutive timesteps ($t-15	ext{h}$ to $t_0$) | No temporal interpolation required in MVP |
| **Forecast Horizons** | $+12	ext{h}$, $+24	ext{h}$, $+48	ext{h}$ | Aligned with IMD Best Track targets |
| **Geographic Box** | Lat: `[-5.0°N, +35.0°N]`, Lon: `[40.0°E, 105.0°E]` | Non-storm-centered basin box (Arabian Sea & Bay of Bengal) |
| **Grid Size (NIO)** | $572 	imes 929$ pixels ($531,388$ cells) | Fixed bounding box for full basin center detection |
| **Years Available** | 1980–2026 (47 continuous years) | 100% coverage across core period 1998–2024 |
| **Acquisition Source** | AWS Open Data (`s3://noaa-cdr-gridsat-b1-pds`) | Free, high-speed public HTTPS, no authentication |
| **Storage per Event** | ~24 MB (compressed) / ~48 MB (packed int16) | Cropped NIO box for average 6-day storm |
| **Storage per Year** | ~150 MB (compressed) / ~300 MB (packed int16) | ~6 cyclonic storms per year (~300 timestamps) |
| **Total Core Storage** | **~3.5 GB (compressed)** / **~7.1 GB (packed int16)** | Complete 1998–2024 cyclone-focused dataset (27 years) |
| **Reference Storms** | FANI, AMPHAN, TAUKTAE, BIPARJOY, REMAL | 100% verified accessible on AWS S3 |

---

## Chronological Split Feasibility

- **TRAIN (1998–2018)**: 21 years available. Full 3-hourly satellite coverage spanning 1998–2004 (scanned IMD era) and 2005–2018 (digital IMD era).
- **VALIDATION (2019–2020)**: 2 years available. Includes benchmark storms **FANI (2019)** and **AMPHAN (2020)**.
- **TEST (2021–2024)**: 4 years available. Includes benchmark storms **TAUKTAE (2021)**, **MANDOUS (2022)**, **BIPARJOY (2023)**, and **REMAL (2024)**.
- **BLIND (2025)**: Available on AWS S3 / NCEI; preserved exclusively for future blind testing.

---

## Action Plan & Next Steps

1. **Do NOT bulk download global data**: Global files are ~40 MB each (280 GB total).
2. **Execute Event-Driven Acquisition**:
   - Use the locked IMD Best Track manifest to extract the exact start/end timestamps for each cyclone event.
   - Download only the specific 3-hourly timestamps associated with active cyclone periods.
   - Immediately crop each downloaded global file to the NIO box `[-5.0, 35.0]°N, [40.0, 105.0]°E` and save to `data/raw/gridsat/nio_cropped/`.
   - Remove transient global files to maintain storage under ~7.1 GB.
