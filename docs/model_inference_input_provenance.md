# VAYU-NET — Model Inference Input Provenance & Ground-Truth Isolation Audit

**Audit Date:** September 25, 2026  
**Subject:** Phase 5B Production Model Inference (`best_phase5b_variant_a.pt` & `best_center_localization_cnn.pt` via `ModelService`)  
**Verdict:** **YELLOW** (Current center at $t_0$ is an observed IMD synoptic analysis input for Variant A track forecast; AI center detection is purely model-derived from Phase 3C; future ground truth is strictly isolated and never ingested).

---

## 1. Input Provenance Table

| Input Tensor / Feature | Dimension / Shape | Immediate Source | Actual Upstream Origin | Model-Derived? | Ground Truth? | Temporal Status | Allowed in Operational Production? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frame_t0` (IRWIN CDR) | `[1, 1, 572, 929]` | GridSat-B1 `.npz` archive | Geostationary infrared satellite observation at $t_0$ | **NO** (Raw remote sensing observation) | **NO** (Physical satellite measurement) | Current ($t_0$) | **YES (Input to Phase 3C Center Localization CNN)** |
| `sat_seq` | `[1, 6, 132]` | `phase5b_hybrid_features.pt` | Phase 4A Spatial Projection on GridSat-B1 IRWIN CDR frames ($t_{-15h} \dots t_0$) | **YES** (Phase 3C ResNet + Phase 4A Linear Projection) | **NO** | Past & Current ($t_{-15h} \dots t_0$) | **YES** |
| `env_seq` | `[1, 6, 8, 41, 66]` | `phase5b_hybrid_features.pt` | ERA5 Reanalysis 8-channel wind fields ($u, v$ at 850, 700, 500, 200 hPa) ($t_{-15h} \dots t_0$) | **NO** (Physical atmospheric reanalysis) | **NO** (Environmental observation) | Past & Current ($t_{-15h} \dots t_0$) | **YES** |
| `kin_ctx_a` | `[1, 5]` | `phase5b_hybrid_features.pt` | Derived from `imd_lat_t0`, `imd_lon_t0` and prior IMD best-track fix ($t \le t_0$) | **NO** | **YES (Past/Current Ground Truth)** | Past & Current ($t_{-3h} \dots t_0$) | **YES (Forecaster-in-the-loop / Observed-anchor mode only)** |
| `kin_a_12` | `[1, 2]` | `phase5b_hybrid_features.pt` | Linear extrapolation from `[imd_lat_t0, imd_lon_t0]` at velocity $[v_{lat}, v_{lon}]$ over +12h | **NO** (Mathematical extrapolation from truth at $t_0$) | **DERIVED FROM $t_0$ TRUTH** | Target $+12h$ (Extrapolated from $t_0$) | **YES (Forecaster-in-the-loop / Observed-anchor mode only)** |
| `kin_a_24` | `[1, 2]` | `phase5b_hybrid_features.pt` | Linear extrapolation from `[imd_lat_t0, imd_lon_t0]` at velocity $[v_{lat}, v_{lon}]$ over +24h | **NO** (Mathematical extrapolation from truth at $t_0$) | **DERIVED FROM $t_0$ TRUTH** | Target $+24h$ (Extrapolated from $t_0$) | **YES (Forecaster-in-the-loop / Observed-anchor mode only)** |
| `kin_a_48` | `[1, 2]` | `phase5b_hybrid_features.pt` | Linear extrapolation from `[imd_lat_t0, imd_lon_t0]` at velocity $[v_{lat}, v_{lon}]$ over +48h | **NO** (Mathematical extrapolation from truth at $t_0$) | **DERIVED FROM $t_0$ TRUTH** | Target $+48h$ (Extrapolated from $t_0$) | **YES (Forecaster-in-the-loop / Observed-anchor mode only)** |
| `true_12` | `[1, 2]` | `vayu_net_sample_index.csv` | IMD Best Track verified center at $t_{+12h}$ | **NO** | **YES (Future Target Ground Truth)** | Future ($t_{+12h}$) | **STRICTLY VERIFICATION ONLY** (Never enters prediction) |
| `true_24` | `[1, 2]` | `vayu_net_sample_index.csv` | IMD Best Track verified center at $t_{+24h}$ | **NO** | **YES (Future Target Ground Truth)** | Future ($t_{+24h}$) | **STRICTLY VERIFICATION ONLY** (Never enters prediction) |
| `true_48` | `[1, 2]` | `vayu_net_sample_index.csv` | IMD Best Track verified center at $t_{+48h}$ | **NO** | **YES (Future Target Ground Truth)** | Future ($t_{+48h}$) | **STRICTLY VERIFICATION ONLY** (Never enters prediction) |

---

## 2. Key Audit Proofs & File References

1. **Origin of `kin_ctx_a` and `kin_a_12/24/48`:**
   - Script: `ml/train/train_hybrid_kinematic_satellite.py`, Lines 122–150.
   - Code:
     ```python
     lat0, lon0 = float(row["imd_lat_t0"]), float(row["imd_lon_t0"])
     s_obs = storm_obs_sorted[sid]
     priors = s_obs[s_obs["dt"] < t0_dt]
     ...
     kin_a_12 = np.array([lat0 + v_lat_a * 12.0, lon0 + v_lon_a * 12.0], dtype=np.float32)
     ```
   - **Finding:** Kinematic anchors in Variant A are computed directly from `imd_lat_t0` and `imd_lon_t0`.

2. **Phase 3C Center Localization Execution in `ModelService`:**
   - File: `ml/inference/model_service.py`, `predict_center()`.
   - **Finding:** `best_center_localization_cnn.pt` is **EXECUTED LIVE** by `ModelService.predict_center()` on the normalized $t_0$ satellite image `[1, 1, 572, 929]`.
   - Decodes soft-argmax continuous coordinates into `ai_detected_center` with zero ground-truth conditioning.
   - Distinct from track prediction: Phase 5B Variant A remains an observed-anchor hybrid model initializing from `OBSERVED_IMD_T0`.

3. **Origin of `forecast.observed_reference_center` vs `forecast.ai_detected_center`:**
   - File: `ml/forecast/api_contracts.py`, `get_forecast()`.
   - `observed_reference_center`: Classified as `OBSERVED_REFERENCE` (ground-truth IMD best track at $t_0$, `imd_lat_t0`, `imd_lon_t0`).
   - `ai_detected_center`: Classified as `MODEL_PREDICTED` (output of `best_center_localization_cnn.pt`).
   - `current_center`: Array retained for backwards compatibility, strictly representing the observed reference position.

4. **Future Ground-Truth Isolation:**
   - In `ModelService.predict_track()`: Only `sat_seq`, `env_seq`, `kin_ctx_a`, and `kin_a_12/24/48` are passed into `Phase5BHybridResidualModel`.
   - In `ModelService.predict_center()`: Only `frame_t0` normalized satellite pixel tensor `[1, 1, 572, 929]` is passed into `DedicatedCenterLocalizationResNet`.
   - Fields `imd_lat_12h`, `imd_lat_24h`, `imd_lat_48h` are accessed **strictly inside `ForecastVerifier.verify_sample()`** after prediction coordinates have been computed.
