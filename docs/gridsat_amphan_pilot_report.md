# VAYU-NET GridSat-B1 Second Pilot Report — Cyclone AMPHAN (2020)

**Authoritative Technical Documentation**  
**Dataset:** NOAA/NCEI GridSat-B1 Climate Data Record (CDR) v02r01  
**Ground-Truth Source:** IMD Best Track Dataset V2 (`data/processed/imd_best_track_v2.csv`)  
**Pilot Target Storm:** `NIO_2020_AMPHAN`  
**Execution Date:** 2026-09-23  

---

## 1. Executive Summary

Following the successful execution of the FANI (2019) pilot, the satellite ingestion, decoding, geographic cropping, and temporal alignment pipeline was independently evaluated against **Cyclone AMPHAN (2020)**.

Cyclone AMPHAN represents the second benchmark storm in the VAYU-NET validation split (`VAL`). Unlike FANI, which exhibited a gradual initial intensification curve, AMPHAN underwent explosive rapid intensification (RI) over the Bay of Bengal, transitioning from a Depression ($25\text{ kt}$) to an Extremely Severe Cyclonic Storm ($100\text{ kt}$) within 48 hours of initial classification.

All automated alignment checks (**Checks A through K**) passed with 100% compliance.

---

## 2. Candidate Analysis & $t_0$ Selection

### Candidate Identification
Using the identical protocol established for FANI:
1. Native 3-hourly synoptic GridSat timestamp (`hour % 3 == 0`).
2. Exact matching IMD observations at $t_0 + 12\text{h}$, $t_0 + 24\text{h}$, and $t_0 + 48\text{h}$.
3. Exactly six consecutive 3-hourly historical steps: $t - 15\text{h}, t - 12\text{h}, t - 9\text{h}, t - 6\text{h}, t - 3\text{h}, t_0$.

Across AMPHAN's 36 IMD Best Track observations, **20 valid candidate $t_0$ timestamps** satisfied all alignment constraints:
```
1.  2020-05-16T00:00:00+00:00  [SELECTED - EARLIEST VALID CANDIDATE]
2.  2020-05-16T03:00:00+00:00
3.  2020-05-16T06:00:00+00:00
4.  2020-05-16T09:00:00+00:00
5.  2020-05-16T12:00:00+00:00
6.  2020-05-16T15:00:00+00:00
7.  2020-05-16T18:00:00+00:00
8.  2020-05-16T21:00:00+00:00
9.  2020-05-17T00:00:00+00:00
10. 2020-05-17T03:00:00+00:00
11. 2020-05-17T06:00:00+00:00
12. 2020-05-17T09:00:00+00:00
13. 2020-05-17T12:00:00+00:00
14. 2020-05-17T15:00:00+00:00
15. 2020-05-17T18:00:00+00:00
16. 2020-05-17T21:00:00+00:00
17. 2020-05-18T00:00:00+00:00
18. 2020-05-18T03:00:00+00:00
19. 2020-05-18T06:00:00+00:00
20. 2020-05-18T09:00:00+00:00
```

### Selected Analysis Timestamp ($t_0$)
- **Selected $t_0$:** `2020-05-16T00:00:00+00:00` (2020-05-16 00:00 UTC)
- **Phase at $t_0$:** Depression (D)
- **Initial Fix at $t_0$:** Latitude `10.4°N`, Longitude `87.0°E`, Wind `25.0 kt`, Pressure `1000.0 hPa`.

---

## 3. Sequence Timestamps & Ground-Truth Alignment

### 6-Step Historical Satellite Input Sequence
| Step | Nominal Offset | Timestamp (UTC) | GridSat-B1 Source NetCDF File | File Size (bytes) | Status |
|:---|:---:|:---:|:---|:---:|:---:|
| 1 | $t - 15\text{h}$ | `2020-05-15T09:00:00Z` | `GRIDSAT-B1.2020.05.15.09.v02r01.nc` | 45,293,452 | Verified & Cropped |
| 2 | $t - 12\text{h}$ | `2020-05-15T12:00:00Z` | `GRIDSAT-B1.2020.05.15.12.v02r01.nc` | 44,528,979 | Verified & Cropped |
| 3 | $t - 9\text{h}$  | `2020-05-15T15:00:00Z` | `GRIDSAT-B1.2020.05.15.15.v02r01.nc` | 45,817,293 | Verified & Cropped |
| 4 | $t - 6\text{h}$  | `2020-05-15T18:00:00Z` | `GRIDSAT-B1.2020.05.15.18.v02r01.nc` | 45,157,667 | Verified & Cropped |
| 5 | $t - 3\text{h}$  | `2020-05-15T21:00:00Z` | `GRIDSAT-B1.2020.05.15.21.v02r01.nc` | 45,525,039 | Verified & Cropped |
| 6 | $t_0$            | `2020-05-16T00:00:00Z` | `GRIDSAT-B1.2020.05.16.00.v02r01.nc` | 47,294,169 | Verified & Cropped |

### Future Target Timestamps & Ground-Truth Reference
| Step | Nominal Offset | Timestamp (UTC) | GridSat Reference File | IMD Lat (°N) | IMD Lon (°E) | IMD Wind (kt) | IMD Pressure (hPa) | IMD Category |
|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| Target 1 | $t + 12\text{h}$ | `2020-05-16T12:00:00Z` | `GRIDSAT-B1.2020.05.16.12.v02r01.nc` | 10.9 | 86.3 | 35.0 | 996.0 | Cyclonic Storm (CS) |
| Target 2 | $t + 24\text{h}$ | `2020-05-17T00:00:00Z` | `GRIDSAT-B1.2020.05.17.00.v02r01.nc` | 11.4 | 86.0 | 45.0 | 992.0 | Cyclonic Storm (CS) |
| Target 3 | $t + 48\text{h}$ | `2020-05-18T00:00:00Z` | `GRIDSAT-B1.2020.05.18.00.v02r01.nc` | 13.2 | 86.3 | 100.0 | 952.0 | Extremely Severe CS (ESCS) |

---

## 4. Decoding, Domain Geometry & Frame Statistics

- **Variable:** `irwin_cdr` (Brightness Temperature in Kelvin)
- **Decoding Equation:** $\text{Kelvin} = \text{packed\_int16} \times 0.01 + 200.0$ (`set_auto_maskandscale(False)` enforced)
- **Spatial Bounding Box:** Latitude `[-5.0°N, +35.0°N]`, Longitude `[40.0°E, 105.0°E]`
- **Array Shape:** Exactly $572 \times 929$ pixels ($531,388$ cells per frame)

### Frame-by-Frame Physical Statistics:
| Tag | Timestamp (UTC) | Shape | Min (K) | Max (K) | Mean (K) | NaN Count | NaN Fraction (%) | Saved NetCDF | Saved NPZ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| `t-15h` | `2020-05-15T09:00:00Z` | $572 \times 929$ | 183.52 | 333.95 | 283.50 | 0 | 0.000% | `gridsat_amphan_t-15h.nc` | `gridsat_amphan_t-15h.npz` |
| `t-12h` | `2020-05-15T12:00:00Z` | $572 \times 929$ | 179.70 | 326.77 | 279.54 | 0 | 0.000% | `gridsat_amphan_t-12h.nc` | `gridsat_amphan_t-12h.npz` |
| `t-9h`  | `2020-05-15T15:00:00Z` | $572 \times 929$ | 179.77 | 313.65 | 277.33 | 0 | 0.000% | `gridsat_amphan_t-9h.nc` | `gridsat_amphan_t-9h.npz` |
| `t-6h`  | `2020-05-15T18:00:00Z` | $572 \times 929$ | 179.79 | 308.37 | 277.58 | 0 | 0.000% | `gridsat_amphan_t-6h.nc` | `gridsat_amphan_t-6h.npz` |
| `t-3h`  | `2020-05-15T21:00:00Z` | $572 \times 929$ | 179.58 | 304.73 | 276.80 | 0 | 0.000% | `gridsat_amphan_t-3h.nc` | `gridsat_amphan_t-3h.npz` |
| `t0`    | `2020-05-16T00:00:00Z` | $572 \times 929$ | 179.90 | 303.27 | 276.66 | 0 | 0.000% | `gridsat_amphan_t0.nc` | `gridsat_amphan_t0.npz` |

---

## 5. Automated Validation Results (Task 9 Checks A–K)

| Check | Requirement | Result | Status |
|:---|:---|:---|:---:|
| **Check A** | Exactly 6 historical frames exist | 6 frames verified | **PASS** |
| **Check B** | Historical deltas are exactly $[-15, -12, -9, -6, -3, 0]\text{h}$ | Deltas match $[-15, -12, -9, -6, -3, 0]\text{h}$ | **PASS** |
| **Check C** | All 6 observations are native 3-hourly cadence | All timestamps have UTC hours modulo 3 == 0 | **PASS** |
| **Check D** | No temporal interpolation performed | Raw discrete observations retained | **PASS** |
| **Check E** | Official IMD Best Track observation exists at $t_0$ | Verified in V2 (`10.4°N, 87.0°E, 25 kt`) | **PASS** |
| **Check F** | IMD $+12\text{h}, +24\text{h}, +48\text{h}$ targets exist | All 3 future targets verified in V2 | **PASS** |
| **Check G** | No timestamp duplication | 6 unique timestamps across 6 frames | **PASS** |
| **Check H** | Satellite spatial grid is internally consistent | All frames have shape $(572, 929)$ and identical bounds | **PASS** |
| **Check I** | Crop is inside locked NIO bounding box | Lat $[-4.97^\circ, 35.00^\circ]$, Lon $[40.01^\circ, 104.97^\circ]$ | **PASS** |
| **Check J** | Missing data fraction is acceptable | 0.000% NaN (well below 5% threshold) | **PASS** |
| **Check K** | Source filenames correspond exactly to timestamps | Filename YYYY.MM.DD.HH matches each UTC timestamp | **PASS** |

**Overall Alignment Verdict: PASS (11/11 Checks Passed)**

---

## 6. Comparative Analysis: AMPHAN (2020) vs. FANI (2019)

| Parameter | Cyclone FANI (2019) | Cyclone AMPHAN (2020) | Comparative Finding |
|:---|:---:|:---:|:---|
| **Total IMD Observations** | 64 rows | 36 rows | AMPHAN was faster-moving and shorter-lived post-genesis |
| **Valid Candidate $t_0$ Count** | 47 candidates | 20 candidates | Both offer ample valid training/evaluation sample sequences |
| **Selected $t_0$** | `2019-04-26T06:00:00Z` | `2020-05-16T00:00:00Z` | Both selected at earliest initial Depression stage |
| **Initial $t_0$ Location** | `3.0°N, 89.4°E` (Equatorial) | `10.4°N, 87.0°E` (Central BoB) | AMPHAN formed further north in the Bay of Bengal |
| **Initial $t_0$ Intensity** | $25\text{ kt}$, $998\text{ hPa}$ (D) | $25\text{ kt}$, $1000\text{ hPa}$ (D) | Identical baseline genesis intensity |
| **$+12\text{h}$ Target Intensity** | $25\text{ kt}$, $998\text{ hPa}$ (D) | $35\text{ kt}$, $996\text{ hPa}$ (CS) | AMPHAN intensified to Cyclonic Storm within 12h |
| **$+24\text{h}$ Target Intensity** | $35\text{ kt}$, $995\text{ hPa}$ (CS) | $45\text{ kt}$, $992\text{ hPa}$ (CS) | AMPHAN maintained higher acceleration |
| **$+48\text{h}$ Target Intensity** | $45\text{ kt}$, $992\text{ hPa}$ (CS) | **$100\text{ kt}$, $952\text{ hPa}$ (ESCS)** | **Explosive Rapid Intensification ($+75\text{ kt}$ in 48h)** |
| **GridSat Files Downloaded** | 9 files | 9 files | Identical 6 historical + 3 target reference files |
| **Cropped Dimensions** | $572 \times 929$ | $572 \times 929$ | 100% spatial consistency across years |
| **Missing Data (NaN %)** | 0.000% | 0.000% | Flawless satellite coverage in both 2019 and 2020 |
| **Validation Verdict** | **PASS (11/11)** | **PASS (11/11)** | Both pilots pass all automated integrity checks |

### Storm-Specific Findings:
1. **Intensification Dynamics:** AMPHAN represents an ideal stress test for VAYU-NET's future intensity head because of its massive pressure drop ($-48\text{ hPa}$ in 48 hours) and rapid wind ramp ($25 \to 100\text{ kt}$).
2. **Convective Asymmetry:** The visual QC plot (`docs/amphan_qc_plot.png`) reveals broad monsoonal convective banding extending southwestward during genesis, followed by rapid symmetric vortex consolidation by $t_0$.

---

## 7. Artifacts Created

1. `data/interim/gridsat_pilot/amphan_2020/amphan_frames_manifest.csv`
2. `data/interim/gridsat_pilot/amphan_2020/amphan_sample_manifest.csv`
3. `data/interim/gridsat_pilot/amphan_2020/amphan_qc_plot.png` (and `docs/amphan_qc_plot.png`)
4. `data/interim/gridsat_pilot/amphan_2020/gridsat_amphan_*.nc` (6 NetCDF-4 cropped files)
5. `data/interim/gridsat_pilot/amphan_2020/gridsat_amphan_*.npz` (6 compressed NumPy files)
6. `data/interim/gridsat_pilot/amphan_2020/raw/` (9 raw NetCDF files)
