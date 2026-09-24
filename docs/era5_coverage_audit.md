# VAYU-NET Phase 5A: ERA5 Coverage and Alignment Audit

## 1. Audit Overview & Purpose
In accordance with Phase 5A specifications, this audit inspects the locked VAYU-NET sample index (`data/manifests/vayu_net_sample_index.csv`) to determine the exact temporal and geographic requirements for acquiring external atmospheric reanalysis data (ERA5 hourly pressure-level products from ECMWF/Copernicus).

The goal is to eliminate unnecessary data acquisition, verify exact temporal alignment with the 6-frame satellite sequences ($t_{-15\text{h}}$ to $t_0$), audit split separation, and guarantee that no future data leakage can occur.

---

## 2. Sample Index & Temporal Sequence Structure

The locked VAYU-NET dataset defines 1,319 total candidate samples across 33 tropical cyclones in the North Indian Ocean (Arabian Sea and Bay of Bengal) spanning 1998 through 2024.

Each candidate sample is defined by a reference forecast initialization time ($t_0$) and requires a backward-looking, 6-frame sequence of 3-hourly observations:
$$\mathcal{T}_{\text{seq}} = [t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0]$$

| Metric | Value | Audit Verification |
| :--- | :---: | :--- |
| **Total Candidate Samples ($t_0$)** | 1,319 | Matches locked manifest |
| **Unique $t_0$ Timestamps** | 1,301 | Verified (18 multi-storm co-occurring timesteps) |
| **Unique Satellite Frame Paths** | 2,353 | Verified from `frame_t_minus_*` columns |
| **Total Unique Required ERA5 Timestamps** | **2,353** | Exact 1-to-1 match with satellite frames |
| **Temporal Granularity** | 3-hourly | GridSat native intervals (00, 03, 06, 09, 12, 15, 18, 21 UTC) |
| **Earliest Required Timestamp** | `1998-10-06T12:00:00Z` | $t_{-15\text{h}}$ for earliest sample (`1998-10-07T03:00:00Z`) |
| **Latest Required Timestamp** | `2024-11-29T18:00:00Z` | $t_0$ for Cyclone Fengal (`2024-11-29T18:00:00Z`) |

---

## 3. Split Coverage & Cross-Partition Leakage Verification

The VAYU-NET dataset utilizes an **event-level chronological partition** by cyclone season. The audit evaluated timestamp assignments across splits:

| Split | Number of Samples | Distinct Storms | Unique ERA5 Timestamps | Earliest Timestamp | Latest Timestamp | Cross-Split Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 673 | 17 | **1,304** | `1998-10-06T12:00:00Z` | `2018-12-14T06:00:00Z` | **0** |
| **VALIDATION** | 275 | 7 | **378** | `2019-01-03T21:00:00Z` | `2020-12-03T03:00:00Z` | **0** |
| **TEST** | 371 | 9 | **671** | `2021-05-13T15:00:00Z` | `2024-11-29T18:00:00Z` | **0** |
| **TOTAL** | **1,319** | **33** | **2,353** | `1998-10-06T12:00:00Z` | `2024-11-29T18:00:00Z` | **0** |

### Key Audit Finding on Data Isolation:
There is **strictly zero temporal overlap** between the timestamps required for TRAIN, VALIDATION, and TEST. All split boundaries are chronologically separated by seasons, ensuring that environmental normalization statistics computed on TRAIN (`1998-2018`) cannot leak into VALIDATION (`2019-2020`) or TEST (`2021-2024`).

---

## 4. Atmospheric Variable & Pressure Level Invariants

The environmental representation is strictly isolated to atmospheric steering wind components:
- **Pressure Levels (4):**
  - **850 hPa:** Lower-tropospheric steering flow / monsoon surge
  - **700 hPa:** Mid-lower tropospheric steering
  - **500 hPa:** Mid-tropospheric deep steering flow / subtropical ridge interaction
  - **300 hPa:** Upper-tropospheric outflow and westerly trough shear
- **Variables (2 per level):**
  - $u$-component of wind (eastward component, $\text{m/s}$)
  - $v$-component of wind (northward component, $\text{m/s}$)
- **Total Channels:** 8 channels ($4 \times 2$).
- **Excluded Variables in Phase 5A:** Geopotential height, temperature, humidity, SST, precipitation, surface pressure.

---

## 5. Geographic Domain & Resolution Alignment

- **Extraction Bounding Box:**
  - Latitude: $-5.0^\circ$ to $+35.0^\circ$ ($40.0^\circ$ span)
  - Longitude: $40.0^\circ$ to $105.0^\circ$ ($65.0^\circ$ span)
- **Native ERA5 Grid (0.25°):**
  - Latitude grid: $161$ points ($35.0^\circ, 34.75^\circ, \dots, -5.0^\circ$)
  - Longitude grid: $261$ points ($40.0^\circ, 40.25^\circ, \dots, 105.0^\circ$)
- **Downsampled ML Representation (~1.0°):**
  - Spatial 4x downsampling yields approximately $41 \times 66$ grid cells.
  - Matches the physical cyclone domain of GridSat-B1 without redundant raster overhead.

---

## 6. Manifest Generation

The complete list of required timestamps has been compiled into:
`data/manifests/era5_timestamp_manifest.csv`

Each entry specifies:
1. `timestamp`: Full ISO-8601 UTC timestamp
2. `split`: TRAIN, VALIDATION, or TEST
3. `num_frame_refs`: Count of candidate sequences referencing this timestamp
4. `frame_references`: Relative paths to corresponding GridSat NPZ files
5. `is_t0`: Boolean indicating if this timestamp acts as a forecast initialization time
6. `storms`: Storm IDs active during this timestamp
7. `era5_availability`: Verified AVAILABLE across ECMWF/Copernicus reanalysis
8. `extraction_status`: PENDING extraction
