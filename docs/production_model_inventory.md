# VAYU-NET Production Model Inventory
## Complete Component Audit & Runtime Integration Status

**Audit Date:** September 2026  
**System Version:** 1.0.0 (SIH 2026 Problem Statement 26070)  
**Governing Standard:** Production End-to-End Inference Architecture  

---

### Executive Summary

All required models, runtime checkpoints, feature stores, calibration databases, and interpretability artifacts required for the VAYU-NET operational inference pipeline have been located, inspected, and verified on disk. No components are missing.

---

### Component Inventory Table

| Component | Source / Checkpoint Path | Architecture Class | Input Contract | Output Contract | Validation Status | Production Readiness |
|---|---|---|---|---|---|---|
| **A. Center Localization** | `data/interim/ml/checkpoints/best_center_localization_cnn.pt` (SHA-256: `5ef11b...`) | `DedicatedCenterLocalizationResNet` (`ml/models/center_localization_cnn.py`) | Single-frame GridSat-B1 IRWIN CDR `[1, 1, 572, 929]`, standardized with train mean (265.41 K) and std (24.89 K). | Normalized center coordinates $[u_{\text{lat}}, u_{\text{lon}}] \in [0, 1]$ mapped to $[-5^\circ, 35^\circ \text{N}] \times [40^\circ, 105^\circ \text{E}]$ via 2D soft-argmax. | **VALIDATED** (Phase 3C: Validation Mean DPE = 703.0 km; Held-out TEST Mean DPE = 810.3 km, Median DPE = 407.4 km; Lat/Lon MAE = 3.48° / 6.07°). | **READY** |
| **B. Multi-Horizon Track Forecast** | `data/interim/ml/checkpoints/best_phase5b_variant_a.pt` (SHA-256: `c1a51f...`) & `best_phase5b_variant_b.pt` (SHA-256: `2f0ec1...`) | `Phase5BHybridResidualModel` (`ml/models/phase5b_hybrid_residual.py`) | 6-frame satellite sequence `[1, 6, 132]`, ERA5 sequence `[1, 6, 8, 41, 66]`, motion context `[1, 5]`, kinematic anchor points `[1, 2]`. | Predicted geographic coordinates $[lat, lon]$ for +12h, +24h, +48h derived via kinematic anchor + residual offset $(\Delta x, \Delta y)$ in km. | **VALIDATED** (Phase 5B: Aggregate Track DPE ~117.8 km across +12h, +24h, +48h). | **READY** |
| **C. Intensity & Wind Speed** | `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt` (SHA-256: `a014e0...`) | `Phase6IntensityWindModel` (`ml/models/phase6_intensity_wind.py`) | 6-frame satellite sequence `[1, 6, 132]` + 6-frame ERA5 sequence `[1, 6, 8, 41, 66]`. | IMD 7-class intensity classification logits + calibrated softmax probabilities (`D`, `DD`, `CS`, `SCS`, `VSCS`, `ESCS`, `SuCS`), continuous max sustained wind (kt), central pressure (hPa), empirical p80 wind uncertainty. | **VALIDATED** (Phase 6: Macro F1 0.2006, accuracy 25.10%, wind MAE 25.44 kt). Research estimate disclaimer attached. | **READY** |
| **D. Empirical Uncertainty Cones** | `data/interim/ml/uncertainty_parameters.json` | `EmpiricalUncertainty` (`ml/forecast/uncertainty.py`) | Forecast coordinates `{"12h": [lat, lon], "24h": [lat, lon], "48h": [lat, lon]}` + percentile selector (`p80`, `p90`). | Empirical displacement error radii (km) for +12h, +24h, +48h + closed GeoJSON-compatible spherical polygon coordinates. Derived strictly from validation residuals. | **VALIDATED** (WP-06: Strict empirical derivation from validation partition; no test leakage). | **READY** |
| **E. Trajectory Verification** | `data/manifests/vayu_net_sample_index.csv` | `ForecastVerifier` (`ml/forecast/verification.py`) | Forecast coordinates + sample ID (strictly evaluated after prediction). | Direct Position Error (DPE in km via haversine) and directional displacement vectors ($\Delta lat, \Delta lon, \text{Error}_{\text{North}}, \text{Error}_{\text{East}}$) against held-out IMD ground truth. | **VALIDATED** (WP-06: Evaluated strictly downstream of prediction path). | **READY** |
| **F. Analog Cyclone Retrieval** | `data/interim/ml/analog_retrieval_cache.json` | `AnalogRetriever` (`ml/forecast/analog_retrieval.py`) | 7-dimensional standardized vector: $[lat, lon, wind\_kt, pressure\_hpa, dx\_12h, dy\_12h, dwind\_12h]$ with $k=2$. | Top-2 historical analog storms from TRAIN partition with same-storm exclusion logic. | **VALIDATED** (WP-07: Provenance audit passed; candidate library restricted to TRAIN). | **READY** |
| **G. Saliency & Explainability** | `data/interim/ml/explainability/explainability_manifest.json` + `ml/explainability/gradcam.py` | `GradCAMExplainer` (`ml/explainability/gradcam.py`) | Single-frame observation tensor or precomputed run ID. | Saliency activation heatmap, overlay image, original satellite frame, target layer identifier (`spatial_encoder.layer2`), and non-causal interpretability notice. | **VALIDATED** (WP-08: Precomputed assets and live gradient hook both functional). | **READY** |
| **H. Secondary Observation Context (IMERG)** | `data/interim/ml/exp_m1_tensor_cache.pt` / `data/manifests/gridsat_imerg_sequence_pairing_manifest.csv` | Data provider (`ml/data/multimodal_dataset.py`) | Sequence sample ID ($t_{-15\text{h}}$ to $t_0$). | 6 synchronized precipitation frames from NASA GPM IMERG Final Run V07B as an observational visualization context layer (0 predictive weight). | **VALIDATED** (EXP-M1/M2/M3: Verified zero predictive contribution to operational heads). | **READY** |

---

### Non-Locatable Components
**None.** All components are present, checksummed, and operational.
