# GridSat-B1 NIO Cropped File Schema Specification

**Document Version:** 1.0  
**Scope:** Authoritative structural specification of the NetCDF-4 (`.nc`) and compressed NumPy (`.npz`) file pairs in `data/interim/gridsat/`.  
**Archive Size:** 2,353 `.nc` files and 2,353 `.npz` files (4,706 total files across 1998–2024).  

---

## 1. Executive Summary & Format Comparison

Each native 3-hourly GridSat-B1 observation is pre-processed into two formats: NetCDF-4 (`.nc`) and NumPy compressed archive (`.npz`). Both formats store the North Indian Ocean (NIO) spatial crop spanning $[-5^\circ\text{N}, 35^\circ\text{N}]$ and $[40^\circ\text{E}, 105^\circ\text{E}]$.

| Attribute / Property | NetCDF-4 (`.nc`) | NumPy Archive (`.npz`) | Equivalence / Parity |
| :--- | :--- | :--- | :---: |
| **Primary Variable** | `irwin_cdr` | `irwin_cdr` | **100% Bit-Identical** |
| **Data Type** | `float32` | `float32` | **Identical** |
| **Grid Dimensions** | `lat`: 572, `lon`: 929 | `(572, 929)` | **Identical** |
| **Units** | Kelvin ($\text{K}$) | Kelvin ($\text{K}$) | **Identical** |
| **Physical Value Range** | $\sim 166.06\text{ K}$ to $\sim 345.70\text{ K}$ | $\sim 166.06\text{ K}$ to $\sim 345.70\text{ K}$ | **Max Abs Diff = 0.0** |
| **Missing / Fill Value** | `NaN` (`_FillValue: nan`) | `NaN` (`np.nan`) | **Identical Mask** |
| **Coordinate Variables** | `lat` (572,), `lon` (929,) | `lat` (572,), `lon` (929,) | **Identical** |
| **Metadata** | NetCDF attributes | Key `'timestamp_utc'` (`<U25`) | **Identical** |
| **Primary ML Role** | Archival & GIS Interoperability | **High-Throughput ML Data Loading** | **Recommended: NPZ** |

---

## 2. Detailed NetCDF-4 Schema (`.nc`)

### Dimensions
- `lat`: 572 (spatial latitude grid, resolution $0.07^\circ$)
- `lon`: 929 (spatial longitude grid, resolution $0.07^\circ$)

### Coordinate Variables
1. **`lat`**:
   - `dtype`: `float32`
   - `shape`: `(572,)`
   - `units`: `degrees_north`
   - `range`: `[-4.970001, 35.0]`
2. **`lon`**:
   - `dtype`: `float32`
   - `shape`: `(929,)`
   - `units`: `degrees_east`
   - `range`: `[40.009995, 104.97000]`

### Data Variable: `irwin_cdr`
- `dtype`: `float32`
- `shape`: `(572, 929)`
- `units`: `Kelvin`
- `_FillValue`: `NaN`
- **Decoding Transformation Applied at Ingestion**:
  $$\text{Kelvin} = \text{packed} \times \text{scale\_factor} + \text{add\_offset}$$
  where `scale_factor = 0.01` and `add_offset = 200.0`. Packed missing values ($\le -30000$) were mapped to `NaN`.

### Global Attributes
- `timestamp_utc`: ISO-8601 timestamp string (e.g. `1998-10-07T03:00:00+00:00`)
- `source_filename`: Original NOAA/NCEI CDR filename (e.g. `GRIDSAT-B1.1998.10.07.03.v02r01.nc`)
- `crop_bounds`: `"lat:[-5.0,35.0], lon:[40.0,105.0]"`
- `scale_factor`: `0.01`
- `add_offset`: `200.0`

---

## 3. Detailed NumPy Archive Schema (`.npz`)

An inspection of the compressed `.npz` file reveals the following dictionary keys:

| Key Name | Array Type | Shape | Physical Meaning | Value Bounds |
| :--- | :--- | :---: | :--- | :--- |
| **`irwin_cdr`** | `float32` | `(572, 929)` | Calibrated IR Brightness Temperature | $\approx 166.06\text{ K}$ to $345.70\text{ K}$ |
| **`lat`** | `float32` | `(572,)` | Latitude coordinates | $-4.97^\circ$ to $+35.0^\circ$ |
| **`lon`** | `float32` | `(929,)` | Longitude coordinates | $+40.01^\circ$ to $+104.97^\circ$ |
| **`timestamp_utc`** | String (`<U25`) | Scalar `()` | Synoptic observation timestamp | ISO-8601 UTC string |

---

## 4. Architectural Decision: NPZ for ML Data Loading

The VAYU-NET data loader uses `.npz` files exclusively during model training and evaluation for the following reasons:

1. **Elimination of HDF5 File Handle Contention**:
   NetCDF-4 relies on the underlying C-based HDF5 library. In PyTorch `DataLoader` with multi-process workers (`num_workers > 0`), concurrent file-access via libnetcdf/libhdf5 can cause threading locks, segmentation faults, or silent worker stalling. `np.load` on uncompressed/compressed NumPy archives uses thread-safe, pure Python/C file streams without library-level locks.
2. **I/O Latency**:
   `np.load(path)['irwin_cdr']` achieves sub-millisecond memory-mapped reads, delivering superior throughput during batch generation.
3. **Bitwise Fidelity**:
   Comparative verification confirmed that the NumPy array is bit-identical to the NetCDF dataset, guaranteeing zero loss of scientific fidelity.
