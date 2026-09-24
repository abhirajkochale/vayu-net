# VAYU-NET WP-06: Empirical Uncertainty & Forecast Verification

**Module:** Work Package 06 (WP-06) — Track Forecast Uncertainty & Deterministic Verification  
**Specification Baseline:** SIH 2026 Problem Statement 26070 — Blueprint / SRS V1.1  
**Authoritative Derivation Split:** `VALIDATION` partition ($N = 252$ samples, 14 storms)  
**Verification Target:** Verified IMD Best Track Coordinates at $+12\text{h}$, $+24\text{h}$, $+48\text{h}$  
**Status:** **PRODUCT-READY & FROZEN**  

---

## 1. Executive Summary

Work Package 06 converts model track error residuals into a clean, scientifically governed, and reproducible empirical uncertainty representation and provides deterministic verification against held-out ground truth:

1. **Empirical Track Error Cone:**
   - Radii derived strictly from the **VALIDATION** partition of the winning Phase 5B Variant A (Observed-Center Hybrid Residual Model).
   - Zero test data used in uncertainty parameter construction.
   - Distinct, horizon-specific empirical radii for $+12\text{h}$, $+24\text{h}$, and $+48\text{h}$.
   - Stored in machine-readable format: [`data/interim/ml/uncertainty_parameters.json`](file:///c:/GitHub/vayu-net/data/interim/ml/uncertainty_parameters.json).
2. **Explicit Scientific Nomenclature:**
   - All uncertainty objects are labeled strictly as `"label": "empirical"`, with `"is_probabilistic_confidence_interval": false`.
   - Radii represent empirical quantile bounds ($\text{P80}$, $\text{P90}$) of historical track Great-Circle Direct Position Errors (DPE).
3. **Deterministic Verification Module:**
   - Compares predicted forecast coordinates against verified IMD ground truth at each synoptic horizon.
   - Outputs horizon-specific DPE (km), mean/median/P90 DPE, and directional error vectors ($\Delta\text{lat}, \Delta\text{lon}$, $\text{Error}_{\text{north}}$, $\text{Error}_{\text{east}}$).
   - Clearly delineates `FORECAST`, `ACTUAL`, and `ERROR`.

---

## 2. Empirical Uncertainty Formulation & Parameter Table

### 2.1 Derivation Discipline
In strict compliance with VAYU-NET governance rules, uncertainty radii must not be fitted from TEST data to avoid data snooping or artificial test tuning. The empirical radii were derived from the complete VALIDATION split ($N = 252$ samples across 14 distinct cyclone events from 2019 to 2020).

### 2.2 Authoritative Empirical Radii Table

| Horizon | Sample Count | Mean DPE | Median DPE (P50) | P80 Radius | P90 Radius | P95 Radius | TEST P80 Coverage ($N=371$) | TEST P90 Coverage ($N=371$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+12h** | 252 | 75.37 km | 63.55 km | **97.51 km** | **117.01 km** | 134.56 km | **80.9%** (target 80%) | **90.6%** (target 90%) |
| **+24h** | 252 | 150.41 km | 119.01 km | **197.04 km** | **238.44 km** | 279.81 km | **81.4%** (target 80%) | **90.3%** (target 90%) |
| **+48h** | 252 | 323.88 km | 257.77 km | **449.65 km** | **554.03 km** | 622.26 km | **81.1%** (target 80%) | **92.2%** (target 90%) |

*Validation Confirmation: When evaluated on the completely held-out TEST partition ($N = 371$ samples across 31 storms from 2021 to 2024), the empirical P80 radii achieved 80.9%–81.4% coverage and the P90 radii achieved 90.3%–92.2% coverage, proving robust empirical calibration across multi-year operational regimes.*

---

## 3. Geometric Cone Construction

Every uncertainty object exposes:
- `horizon`: e.g. `"+12h"`, `"+24h"`, `"+48h"`
- `center`: `[latitude, longitude]`
- `radius_km`: Float radius in kilometers
- `derivation_split`: `"VALIDATION"`
- `derivation_metric`: `"P80"` or `"P90"`
- `label`: `"empirical"`
- `geometry`: GeoJSON-compliant polygon ring with 32 boundary vertices computed along spherical great circles using forward geodesic formulas.

### Product Uncertainty Payload Schema:
```json
{
  "horizon": "+12h",
  "center": {
    "latitude": 14.5000,
    "longitude": 85.0000
  },
  "radius_km": 97.51,
  "derivation_split": "VALIDATION",
  "derivation_metric": "P80",
  "label": "empirical",
  "is_probabilistic_confidence_interval": false,
  "description": "Empirical P80 track error radius derived from VALIDATION residuals",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [ [85.0000, 15.3773], [85.1764, 15.3582], ..., [85.0000, 15.3773] ]
    ]
  }
}
```

---

## 4. Deterministic Verification Architecture

The verification engine ([`ml/forecast/verification.py`](file:///c:/GitHub/vayu-net/ml/forecast/verification.py)) compares predicted forecast coordinates ($\hat{\phi}_h, \hat{\lambda}_h$) with ground-truth IMD best-track coordinates ($\phi_h, \lambda_h$).

### Metrics Computed:
1. **Direct Position Error (DPE):**
   $$\text{DPE}_h = 2 R_{\text{earth}} \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\hat{\phi}_h)\cos(\phi_h)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$
2. **Directional Error Vector:**
   - $\Delta\phi = \hat{\phi}_h - \phi_h$
   - $\Delta\lambda = \hat{\lambda}_h - \lambda_h$
   - $\text{Error}_{\text{north}} = \Delta\phi \cdot \frac{\pi}{180} R_{\text{earth}}$
   - $\text{Error}_{\text{east}} = \Delta\lambda \cos\left(\frac{\hat{\phi}_h + \phi_h}{2}\right) \cdot \frac{\pi}{180} R_{\text{earth}}$

### Product Verification Payload Example:
```json
{
  "cyclone_id": "NIO_2021_TAUKTAE",
  "sample_id": "NIO_2021_TAUKTAE_20210515_0000Z",
  "split": "TEST",
  "t0": "2021-05-15T00:00:00+00:00",
  "horizons": {
    "12h": {
      "horizon": "+12h",
      "forecast": {
        "latitude": 13.5210,
        "longitude": 72.8420,
        "issue_timestamp_utc": "2021-05-15T00:00:00+00:00",
        "target_timestamp_utc": "2021-05-15T12:00:00+00:00"
      },
      "actual": {
        "latitude": 13.8000,
        "longitude": 72.7000,
        "timestamp_utc": "2021-05-15T12:00:00+00:00",
        "source": "IMD_BEST_TRACK_V2"
      },
      "error": {
        "dpe_km": 34.62,
        "delta_lat_deg": -0.2790,
        "delta_lon_deg": 0.1420,
        "error_north_km": -31.02,
        "error_east_km": 15.37,
        "metric": "Direct Position Error (Great-Circle Distance)"
      }
    },
    "24h": { ... },
    "48h": { ... }
  },
  "summary": {
    "mean_dpe_km": 142.15,
    "median_dpe_km": 118.40,
    "p90_dpe_km": 284.10,
    "status": "VERIFIED_AGAINST_IMD_HELD_OUT_TRUTH"
  }
}
```

---

## 5. API Endpoints Fed by WP-06

1. **`GET /api/cyclones/{id}/forecast`**
   - Feeds: Model forecast trajectory $+12\text{h}, +24\text{h}, +48\text{h}$ along with empirical uncertainty cones and radii.
2. **`GET /api/cyclones/{id}/verification`**
   - Feeds: Comprehensive verification payload showing forecast vs actual coordinates, DPEs, and directional errors.

---

## 6. Limitations & Governance Notes

1. **Empirical vs Parametric:** Radii represent empirical quantile boundaries from the historical validation distribution. They are not Gaussian standard deviations and do not model heteroscedastic storm-specific spread (e.g. sheared vs symmetrical storms).
2. **Derivation Immutability:** Any re-tuning or re-derivation of `uncertainty_parameters.json` must be performed on validation/training splits only; the held-out TEST partition is permanently reserved for benchmark reporting.
