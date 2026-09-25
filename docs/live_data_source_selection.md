# VAYU-NET — Live Data Source Selection & Provider Architecture Audit

**Audit Date:** September 2026  
**Document Target:** `docs/live_data_source_selection.md`  
**System Version:** VAYU-NET Operational Ingestion Layer (SIH Problem Statement 26070)

---

## 1. Executive Summary

Transitioning VAYU-NET from an offline retrospective evaluation pipeline to an operational current-event inference system requires connecting three distinct external data domains:
1. **Cyclone Event Discovery:** Detection of active tropical disturbances, depressions, and named cyclonic storms over the North Indian Ocean basin.
2. **Causal Track History:** Sequential synoptic center fixes $(\text{lat}, \text{lon}, \text{wind}, \text{pressure})$ observed up to the current issue time $t_0$.
3. **Geostationary Satellite Imagery:** A six-frame causal infrared brightness temperature sequence $[t_{-15\text{h}} \dots t_0]$.

Under non-negotiable scientific constraints, **we do not fake live data**, **we do not interpolate missing frames**, and **we do not silently substitute unadapted satellite sensors into models trained on GridSat-B1**.

---

## 2. Comparative Evaluation of Available Providers

| Feature / Metric | Provider 1: NOAA/NCEI AWS S3 Open Data (GridSat-B1 CDR) | Provider 2: IMD RSMC New Delhi (Official Bulletins / Digital Advisories) | Provider 3: ISRO MOSDAC (INSAT-3D/3DR Real-Time) | Provider 4: NASA GES DISC (GPM IMERG Final/Early) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Role** | Satellite Observation ($t_{-15\text{h}} \dots t_0$) | Event Discovery & Causal Track History | High-frequency Regional Secondary Context | Secondary Context (Precipitation) |
| **Target Endpoint** | `https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/{YYYY}/` | `https://rsmcnewdelhi.imd.gov.in/` (Bulletins / Best Track Archive) | `https://mosdac.gov.in/download_api/` | `https://gpm1.gesdisc.eosdis.nasa.gov/data/` |
| **Authentication** | **None (Public Open Access)** | Public Web Access / Scraped Synoptic Tables | Bearer Token (`MOSDAC_USERNAME` / `PASSWORD`) | Earthdata URS Auth (`EARTHDATA_USER` / `PASSWORD`) |
| **Latency / Refresh** | Archive delay (3–12 months); NRT stream variable | 3-hourly to 6-hourly operational bulletins during active storms | 15–30 minutes (Near-real-time) | 3.5 months (Final Run V07B); ~4 hours (Early Run) |
| **Spatial Coverage** | Global ($70^\circ\text{S} - 70^\circ\text{N}$, $180^\circ\text{W} - 180^\circ\text{E}$) | North Indian Ocean (Arabian Sea & Bay of Bengal) | South Asia / North Indian Ocean (Asia Mercator) | Global ($60^\circ\text{S} - 60^\circ\text{N}$) |
| **Temporal Resolution**| 3-hourly ($00, 03, 06, 09, 12, 15, 18, 21\text{ UTC}$) | 3-hourly to 6-hourly fixes | 15 to 30 minutes | 30 minutes |
| **Data Format** | NetCDF-4 (`.nc`) packed int16 ($\text{Kelvin} = x \times 0.01 + 200.0$) | Tabular advisory text / CSV synoptic bulletins | HDF5 (`.h5`) Asian Mercator rasters | HDF5 (`.HDF5`) precipitation rate (mm/h) |
| **Cyclone Discovery?** | **NO** (Physical radiance only) | **YES (Authoritative for NIO)** | **NO** (Radiance only) | **NO** |
| **Track History?** | **NO** | **YES (Authoritative synoptic center/wind fixes)**| **NO** | **NO** |
| **Primary ML Ready?** | **YES (Exact match to Phase 3C/5B/6 training)** | N/A (Feeds kinematic anchor) | **NO (Severe domain shift; requires adaptation)**| **NO (Contextual layer only)** |

---

## 3. Selected Operational Provider Architecture

To maximize robustness and eliminate artificial dependencies, VAYU-NET implements an **Adapter Architecture** configured via environment variables:

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │              LiveSourceService (Adapter)               │
                                      └────────────────────────────────────────────────────────┘
                                               │                       │                       │
                                               ▼                       ▼                       ▼
                              ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
                              │ CycloneEventProvider   │  │  TrackHistoryProvider  │  │   SatelliteProvider    │
                              ├────────────────────────┤  ├────────────────────────┤  ├────────────────────────┤
                              │ • IMD RSMC Official    │  │ • IMD Advisory Fixes   │  │ • NOAA AWS Open Data   │
                              │ • JTWC Automated NIO   │  │ • JTWC Causal Bestrack │  │ • Local Staged Archive │
                              │ • Historical Replay    │  │ • Historical Replay    │  │ • Historical Replay    │
                              └────────────────────────┘  └────────────────────────┘  └────────────────────────┘
```

### Provider Details

### 1. Satellite Provider (`SATELLITE_PROVIDER="noaa_aws_s3" | "local_archive"`)
- **Primary Source:** AWS S3 Public Bucket `noaa-cdr-gridsat-b1-pds`.
- **URL Pattern:** `https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/{YYYY}/GRIDSAT-B1.{YYYY}.{MM}.{DD}.{HH}.v02r01.nc`
- **Authentication:** None required.
- **Integrity Validation:** Validates NetCDF-4 container, variable `irwin_cdr`, shape, and physical range $[175\text{ K}, 335\text{ K}]$.
- **Fallback / Replay:** Local pre-staged directory `data/interim/gridsat/{YYYY}/`.

### 2. Cyclone Event Provider (`CYCLONE_EVENT_PROVIDER="imd_rsmc" | "replay"`)
- **Primary Source:** IMD RSMC Tropical Cyclone Advisories & Tropical Disturbance Bulletins.
- **Identifiers:** Standardized NIO storm codes (e.g. `BOB/01/2024`, `ARB/02/2023`, `NIO_2023_BIPARJOY`).
- **Replay / Sandbox:** Certified historical storms evaluated under real-time causal conditions ($t \le t_0$).

### 3. Track History Provider (`TRACK_HISTORY_PROVIDER="imd_advisory" | "replay"`)
- **Primary Source:** Synoptic center fixes published in IMD advisories at $t \le t_0$.
- **Attributes per Fix:** `timestamp_utc`, `latitude`, `longitude`, `wind_kt`, `pressure_hpa`.
- **Constraint:** At least 2 causal fixes ($t_{\text{prior}}$ and $t_0$) are required to derive velocity $v_{\text{lat}}, v_{\text{lon}}$ for the Phase 5B Variant A kinematic anchor.

---

## 4. GridSat Sensor Compatibility & Domain Shift Lock

Trained VAYU-NET models expect:
- Spatial Grid: Monotonic $0.07^\circ$ Mercator grid $[-5.0^\circ, 35.0^\circ\text{N}] \times [40.0^\circ, 105.0^\circ\text{E}]$ ($572 \times 929$).
- Radiometric Calibration: GridSat-B1 intercalibrated IR brightness temperature with training normalization constants ($\mu = 265.41\text{ K}, \sigma = 24.89\text{ K}$).

> [!CAUTION]
> Feeding raw INSAT-3D/3DR or GFS forecast radiances directly into the frozen ResNet-18 spatial backbone without fine-tuned cross-sensor calibration causes domain shift and coordinate divergence. All primary ML predictions remain strictly GridSat-B1 based. Other sources (INSAT, IMERG) are exposed solely as secondary contextual layers.
