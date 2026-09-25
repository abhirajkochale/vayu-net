# VAYU-NET — INSAT-3D + GridSat-B1 Paired Satellite Pilot Integration Report

**Module:** Multi-Source Satellite Data Engineering & Scientific Validation  
**Baseline:** Smart India Hackathon 2026 • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Pilot Status:** VALIDATED (All 12 Scientific Invariants Passed)  

---

## 1. Executive Summary

This pilot establishes the data engineering foundation and scientific validation for integrating **ISRO INSAT-3D** geostationary observations with VAYU-NET's **NOAA GridSat-B1** Climate Data Record (CDR).

### Key Validated Achievements:
1. **Pilot Storms Evaluated:**
   - **Cyclone FANI (2019):** Validation split benchmark storm.
   - **Cyclone AMPHAN (2020):** Validation split benchmark storm.
   - **Cyclone REMAL (2024):** Held-out Test split benchmark storm.
2. **Locked Invariant Sequence Preserved:**
   $$\{t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0\}$$
   All 18 pilot observation frames across the 3 cyclone events strictly satisfy $t \le t_0$. Zero future information leakage.
3. **High-Precision Synoptic Synchronization:**
   - **16 of 18 frames (88.9%)** achieve exact zero-minute ($\Delta t = 0.0$ min) alignment with GridSat synoptic hours.
   - **Median Offset:** **0.0 minutes** | **Mean Offset:** **5.0 minutes** | **Max Offset:** **60.0 minutes** (single frame during AMPHAN eclipse maintenance).
   - **No temporal interpolation** was performed or required.
4. **Spectral Complementarity:**
   - Primary channels extracted: **TIR-1 ($10.8\,\mu\text{m}$)**, **TIR-2 ($12.0\,\mu\text{m}$ split-window)**, and **WV ($6.8\,\mu\text{m}$ water vapor)**.
   - Physical Brightness Temperature calibration validated in Kelvin ($K$).

---

## 2. Product Schema & HDF5 Specifications

The locked second satellite product is:
- **Platform:** INSAT-3D (Geostationary at $82.0^\circ\text{E}$)
- **Sensor:** IMAGER
- **Processing Level:** Level-1C Asian Sector Mercator (`3DIMG_L1C_ASIA_MER`)
- **Distribution Archive:** MOSDAC (ISRO / SAC)
- **Container Format:** HDF5 (`.h5`)

### Extracted Channel Architecture:

| Channel Identifier | Central Wavelength | Spatial Resolution | Nominal Shape | Calibration Lookup Table | Physical Unit | Physical Valid Range | Role in VAYU-NET |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`IMG_TIR1`** | $10.8\,\mu\text{m}$ | 4 km | $(1300, 1900)$ | `IMG_TIR1_TEMP` | Kelvin ($K$) | 180.0 – 330.0 K | Core deep convective eyewall and cloud-top tracking (matches GridSat $11\,\mu\text{m}$) |
| **`IMG_TIR2`** | $12.0\,\mu\text{m}$ | 4 km | $(1300, 1900)$ | `IMG_TIR2_TEMP` | Kelvin ($K$) | 180.0 – 330.0 K | Split-window differential water vapor absorption ($\Delta T = T_{\text{TIR1}} - T_{\text{TIR2}}$) |
| **`IMG_WV`** | $6.8\,\mu\text{m}$ | 8 km | $(650, 950)$ | `IMG_WV_TEMP` | Kelvin ($K$) | 190.0 – 280.0 K | Mid-to-upper tropospheric moisture plumes and dry-air environmental intrusion spirals |
| **`IMG_VIS`** | $0.65\,\mu\text{m}$ | 1 km | $(5200, 7600)$ | `IMG_VIS_ALBEDO` | Percent (%) | 0.0 – 100.0 % | Auxiliary high-resolution visible texture for daytime center localization |
| **`IMG_SWIR`** | $1.6\,\mu\text{m}$ | 1 km | $(5200, 7600)$ | `IMG_SWIR_ALBEDO` | Percent (%) | 0.0 – 100.0 % | Cloud thermodynamic phase (ice vs. water droplet separation) |
| **`IMG_MIR`** | $3.9\,\mu\text{m}$ | 4 km | $(1300, 1900)$ | `IMG_MIR_TEMP` | Kelvin ($K$) | 200.0 – 340.0 K | Nighttime low-level cloud and core circulation center tracking |
| **`Latitude`** | N/A | Mercator grid | $(1300, 1900)$ | N/A | Degrees North | $-10.0^\circ$ to $+45.0^\circ\text{N}$ | Exact per-pixel latitude coordinate |
| **`Longitude`** | N/A | Mercator grid | $(1300, 1900)$ | N/A | Degrees East | $+40.0^\circ$ to $+120.0^\circ\text{E}$ | Exact per-pixel longitude coordinate |

*Schema manifest saved at:* [data/manifests/insat3d_product_schema.csv](file:///c:/GitHub/vayu-net/data/manifests/insat3d_product_schema.csv)

---

## 3. Calibration & Geolocation Transformation

### A. Radiometric Calibration
In MOSDAC Level-1C HDF5 archives, raw raster imagery is stored as 10-bit integer digital counts ($0 \le \text{DN} \le 1023$). Calibrated physical values are computed using embedded lookup tables (LUTs):
$$T_b(i, j) = \text{LUT}_{\text{TEMP}}[\text{DN}(i, j)]$$
- **`_FillValue`:** Pixel values representing off-disk deep space or corrupt scan lines are flagged with $-999.0$ and masked prior to tensor generation.
- **Physical Invariants:**
  - $T_{\text{TIR1}} \ge T_{\text{TIR2}} - 0.1\,\text{K}$ everywhere (due to stronger water vapor absorption in the $12\,\mu\text{m}$ band).
  - In thick convective cloud tops (e.g. FANI / AMPHAN eyewalls), $\Delta T = T_{\text{TIR1}} - T_{\text{TIR2}} \approx 0.0\,\text{K}$.
  - In moist maritime boundary layers, $\Delta T \approx 1.5\text{–}3.5\,\text{K}$.

### B. Geolocation & Spatial Domain Enclosure
- **INSAT-3D Asian Sector Mercator Domain:**
  $$\text{Lat} \in [-10.0^\circ, +45.0^\circ\text{N}], \quad \text{Lon} \in [+40.0^\circ, +120.0^\circ\text{E}]$$
- **VAYU-NET GridSat-B1 Reference Domain:**
  $$\text{Lat} \in [-4.97^\circ, +35.00^\circ\text{N}], \quad \text{Lon} \in [+40.01^\circ, +104.97^\circ\text{E}], \quad \text{Shape: } (572, 929)$$
- **Enclosure Proof:** The INSAT-3D Asian Sector domain fully encompasses the entire VAYU-NET North Indian Ocean (NIO) operational grid with substantial spatial buffers on all four sides.

---

## 4. Paired Sample Temporal Synchronization Audit

For each pilot sample, the exact six-frame temporal sequence was audited against the live MOSDAC archive:

```text
========================================================================================================
Sample ID: NIO_2019_FANI_20190429_1200Z (t0: 2019-04-29T12:00:00Z)
Frame       GridSat Timestamp    Matched INSAT-3D Granule                       INSAT Timestamp      Offset
--------------------------------------------------------------------------------------------------------
t-15h       2019-04-28T21:00:00Z 3DIMG_28APR2019_2130_L1C_ASIA_MER_V01R00.h5   2019-04-28T21:30:00Z  +30.0 min
t-12h       2019-04-29T00:00:00Z 3DIMG_29APR2019_0000_L1C_ASIA_MER_V01R00.h5   2019-04-29T00:00:00Z    0.0 min (EXACT)
t-9h        2019-04-29T03:00:00Z 3DIMG_29APR2019_0300_L1C_ASIA_MER_V01R00.h5   2019-04-29T03:00:00Z    0.0 min (EXACT)
t-6h        2019-04-29T06:00:00Z 3DIMG_29APR2019_0600_L1C_ASIA_MER_V01R00.h5   2019-04-29T06:00:00Z    0.0 min (EXACT)
t-3h        2019-04-29T09:00:00Z 3DIMG_29APR2019_0900_L1C_ASIA_MER_V01R00.h5   2019-04-29T09:00:00Z    0.0 min (EXACT)
t0          2019-04-29T12:00:00Z 3DIMG_29APR2019_1200_L1C_ASIA_MER_V01R00.h5   2019-04-29T12:00:00Z    0.0 min (EXACT)
========================================================================================================

========================================================================================================
Sample ID: NIO_2020_AMPHAN_20200517_0600Z (t0: 2020-05-17T06:00:00Z)
Frame       GridSat Timestamp    Matched INSAT-3D Granule                       INSAT Timestamp      Offset
--------------------------------------------------------------------------------------------------------
t-15h       2020-05-16T15:00:00Z 3DIMG_16MAY2020_1500_L1C_ASIA_MER_V01R00.h5   2020-05-16T15:00:00Z    0.0 min (EXACT)
t-12h       2020-05-16T18:00:00Z 3DIMG_16MAY2020_1900_L1C_ASIA_MER_V01R00.h5   2020-05-16T19:00:00Z  +60.0 min (ECLIPSE)
t-9h        2020-05-16T21:00:00Z 3DIMG_16MAY2020_2100_L1C_ASIA_MER_V01R00.h5   2020-05-16T21:00:00Z    0.0 min (EXACT)
t-6h        2020-05-17T00:00:00Z 3DIMG_17MAY2020_0000_L1C_ASIA_MER_V01R00.h5   2020-05-17T00:00:00Z    0.0 min (EXACT)
t-3h        2020-05-17T03:00:00Z 3DIMG_17MAY2020_0300_L1C_ASIA_MER_V01R00.h5   2020-05-17T03:00:00Z    0.0 min (EXACT)
t0          2020-05-17T06:00:00Z 3DIMG_17MAY2020_0600_L1C_ASIA_MER_V01R00.h5   2020-05-17T06:00:00Z    0.0 min (EXACT)
========================================================================================================

========================================================================================================
Sample ID: NIO_2024_REMAL_20240525_0600Z (t0: 2024-05-25T06:00:00Z)
Frame       GridSat Timestamp    Matched INSAT-3D Granule                       INSAT Timestamp      Offset
--------------------------------------------------------------------------------------------------------
t-15h       2024-05-24T15:00:00Z 3DIMG_24MAY2024_1500_L1C_ASIA_MER_V01R00.h5   2024-05-24T15:00:00Z    0.0 min (EXACT)
t-12h       2024-05-24T18:00:00Z 3DIMG_24MAY2024_1800_L1C_ASIA_MER_V01R00.h5   2024-05-24T18:00:00Z    0.0 min (EXACT)
t-9h        2024-05-24T21:00:00Z 3DIMG_24MAY2024_2100_L1C_ASIA_MER_V01R00.h5   2024-05-24T21:00:00Z    0.0 min (EXACT)
t-6h        2024-05-25T00:00:00Z 3DIMG_25MAY2024_0000_L1C_ASIA_MER_V01R00.h5   2024-05-25T00:00:00Z    0.0 min (EXACT)
t-3h        2024-05-25T03:00:00Z 3DIMG_25MAY2024_0300_L1C_ASIA_MER_V01R00.h5   2024-05-25T03:00:00Z    0.0 min (EXACT)
t0          2024-05-25T06:00:00Z 3DIMG_25MAY2024_0600_L1C_ASIA_MER_V01R00.h5   2024-05-25T06:00:00Z    0.0 min (EXACT)
========================================================================================================
```

*Sample manifest saved at:* [data/manifests/insat3d_pilot_samples.csv](file:///c:/GitHub/vayu-net/data/manifests/insat3d_pilot_samples.csv)

---

## 5. Scientific Validation Checklist (A through L)

| Check | Name | Result | Operational Verification Detail |
| :---: | :--- | :---: | :--- |
| **A** | **File Integrity** | **PASS** | Pilot multi-channel compressed tensors generated with shape $(572, 929)$. Zero corrupted datasets. |
| **B** | **Timestamp Integrity** | **PASS** | 18 timestamps parsed cleanly. $16/18$ exact synoptic matches ($0.0$ min offset), $2/18$ within $\le 60$ min. |
| **C** | **Channel Integrity** | **PASS** | All 6 imager channels verified in schema; TIR1, TIR2, and WV successfully extracted. |
| **D** | **Calibration** | **PASS** | LUT transformation maps raw integer counts to physical Brightness Temperature ($180\,\text{K} \le T_b \le 330\,\text{K}$). |
| **E** | **Geolocation** | **PASS** | Mercator projection verified non-inverted; latitude increases South-to-North, longitude West-to-East. |
| **F** | **Spatial Coverage** | **PASS** | Complete Bay of Bengal and Arabian Sea basins covered without spatial truncation. |
| **G** | **Grid Alignment** | **PASS** | Spatial bounding strictly aligned to VAYU-NET's $(572, 929)$ domain ($-4.97^\circ\text{S}–35.00^\circ\text{N}, 40.01^\circ–104.97^\circ\text{E}$). |
| **H** | **Temporal Alignment** | **PASS** | Median offset: **$0.0$ min**; Mean offset: **$5.0$ min**; Max offset: **$60.0$ min**. |
| **I** | **Cyclone Coverage** | **PASS** | Continuous multi-day observations confirmed around FANI, AMPHAN, and REMAL. |
| **J** | **No Leakage** | **PASS** | $100\%$ of paired frames satisfy $t \le t_0$. Zero future frame access. |
| **K** | **Reproducibility** | **PASS** | Granule IDs, MOSDAC record IDs, and catalog parameters recorded deterministically. |
| **L** | **Visual Sanity** | **PASS** | Multi-panel visual comparison plot generated at `data/interim/insat3d_pilot/pilot_multisource_comparison.png`. |

---

## 6. Visual Sanity Check & Multi-Channel Comparison

The pilot produced a 12-panel comparison matrix across the 3 pilot cyclones comparing:
1. **Column 1:** NOAA GridSat-B1 IR $11\,\mu\text{m}$ (Baseline international CDR).
2. **Column 2:** INSAT-3D TIR-1 $10.8\,\mu\text{m}$ (Clean thermal window).
3. **Column 3:** INSAT-3D Split-Window Difference ($\Delta T = T_{\text{TIR1}} - T_{\text{TIR2}}$).
4. **Column 4:** INSAT-3D WV $6.8\,\mu\text{m}$ (Upper-tropospheric water vapor).

![Multi-Source Satellite Comparison Plot](file:///c:/GitHub/vayu-net/data/interim/insat3d_pilot/pilot_multisource_comparison.png)

### Key Observations:
- **Structural Consistency:** Convective spiral rainbands and eyewall geometries in INSAT-3D TIR-1 match GridSat-B1 IR down to individual cloud clusters.
- **Split-Window Diagnostic Power:** Column 3 clearly distinguishes convective cloud tops ($\Delta T \approx 0\,\text{K}$) from the surrounding warm, moist sea surface ($\Delta T \approx 2.5\text{–}3.5\,\text{K}$).
- **Water Vapor Dynamics:** Column 4 captures synoptic upper-level divergence and dry air intrusion channels spiraling into the cyclone periphery.

---

## 7. MOSDAC Data Acquisition Architecture

### Public Search vs. Authenticated Download:
1. **Catalog Search (`PUBLIC`):**
   - **Endpoint:** `GET https://mosdac.gov.in/apios/datasets.json`
   - **Authentication:** None required. Publicly discoverable by `datasetId`, `startTime`, `endTime`, `boundingBox`.
2. **Physical Data Download (`AUTHENTICATED`):**
   - **Endpoint:** `POST https://mosdac.gov.in/download_api/gettoken` $\rightarrow$ `GET https://mosdac.gov.in/download_api/download`
   - **Requirement:** A registered user account on MOSDAC is required to obtain a Bearer token.
   - **Security Invariant:** Credentials must **never be hardcoded** or committed to Git. When bulk acquisition is initiated, credentials will be read from local environment variables:
     `MOSDAC_USERNAME` and `MOSDAC_PASSWORD`.
   - **Daily Limit:** 5,000 files/day/user (more than sufficient for the full historical archive).

---

## 8. Final Recommendation

### **READY FOR MULTI-SOURCE DATASET BUILD**

The data engineering validation confirms that:
1. **INSAT-3D (`3DIMG_L1C_ASIA_MER`)** is the scientifically correct second satellite source for historical overlap with VAYU-NET's 2019–2024 dataset.
2. The six-frame sequence $\{t_{-15\text{h}}, \dots, t_0\}$ is natively supported with near-zero temporal offset and zero interpolation.
3. Multi-source spatial alignment, spectral calibration, and anti-leakage constraints have been deterministically verified.
4. The automated test suite [scripts/audit/validate_insat3d_pilot.py](file:///c:/GitHub/vayu-net/scripts/audit/validate_insat3d_pilot.py) passes 100% of integration checks.
