# VAYU-NET Phase 5A: ERA5 Data Provenance & Invariants

## 1. Dataset Provenance Specification

In accordance with Phase 5A protocols, all external atmospheric environmental data used for steering flow augmentation must be fully documented with formal provenance tracking and cryptographic hash verification.

| Property | Detail |
| :--- | :--- |
| **Official Dataset Name** | ERA5 hourly data on pressure levels from 1940 to present |
| **Originating Agency** | European Centre for Medium-Range Weather Forecasts (ECMWF) / Copernicus Climate Change Service (C3S) |
| **Official DOI** | [10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6) |
| **Access Pipeline** | Analysis-Ready Cloud-Optimized (ARCO) ERA5 on Google Cloud Public Datasets (`gcp-public-data-arco-era5`) |
| **Access Endpoint** | `https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/` |
| **Retrieval Date** | 2026-09-24 / 2026-09-25 UTC |
| **Temporal Frequency** | 1-hourly (sampled at native 3-hourly intervals matching GridSat) |
| **Variables Extracted** | Eastward wind ($u$, `u_component_of_wind`), Northward wind ($v$, `v_component_of_wind`) |
| **Pressure Levels** | 850 hPa, 700 hPa, 500 hPa, 300 hPa (4 levels) |
| **Spatial Bounding Box** | Latitude: $-5.0^\circ\text{N}$ to $+35.0^\circ\text{N}$, Longitude: $40.0^\circ\text{E}$ to $105.0^\circ\text{E}$ |
| **Raw Spatial Resolution** | $0.25^\circ \times 0.25^\circ$ ($161 \times 261$ grid points) |
| **Target ML Resolution** | $\sim 1.0^\circ \times 1.0^\circ$ ($41 \times 66$ grid cells via bilinear downsampling) |
| **Local Storage Path** | `data/raw/era5/` |
| **File Format** | Compressed NumPy array files (`.npz`) containing float32 arrays `[8, 41, 66]` |
| **Cryptographic Manifest** | `data/manifests/era5_sha256_manifest.csv` |

---

## 2. Channel Ordering & Units

Each extracted timestamp is saved as an 8-channel array of dimensions `(8, 41, 66)`:

| Channel Index | Variable Name | Pressure Level | Unit | Physical Role |
| :---: | :--- | :---: | :---: | :--- |
| **0** | $u$-component of wind | 850 hPa | $\text{m/s}$ | Low-level cyclonic circulation & monsoon inflow |
| **1** | $u$-component of wind | 700 hPa | $\text{m/s}$ | Mid-lower steering layer |
| **2** | $u$-component of wind | 500 hPa | $\text{m/s}$ | Mid-tropospheric steering flow (subtropical ridge boundary) |
| **3** | $u$-component of wind | 300 hPa | $\text{m/s}$ | Upper-tropospheric jet & shear environment |
| **4** | $v$-component of wind | 850 hPa | $\text{m/s}$ | Low-level meridional flow |
| **5** | $v$-component of wind | 700 hPa | $\text{m/s}$ | Mid-lower meridional steering |
| **6** | $v$-component of wind | 500 hPa | $\text{m/s}$ | Mid-tropospheric meridional steering |
| **7** | $v$-component of wind | 300 hPa | $\text{m/s}$ | Upper-level meridional outflow |

---

## 3. Storage Invariants & Verification

1. **Separation from Locked Datasets:** ERA5 files are stored strictly under `data/raw/era5/` and cached under `data/interim/ml/cache/`. Zero files in `data/raw/imd/`, `data/processed/`, or `data/interim/gridsat/` are altered.
2. **Deterministic Byte Verification:** All extracted `.npz` files have their SHA256 checksums recorded in `data/manifests/era5_sha256_manifest.csv`.
3. **No Future Leakage:** For each forecast initialization time $t_0$, the environmental sequence strictly encompasses:
   $$\{t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0\}$$
   No ERA5 data after $t_0$ is ever read, transformed, or supplied to the network.
4. **Historical Reanalysis Notice:** As per Section 30 of the Phase 5A specification, ERA5 is a retrospective reanalysis product. This experiment evaluates the scientific limit of environmental steering information on cyclone track predictability, not real-time operational NWP latency.
