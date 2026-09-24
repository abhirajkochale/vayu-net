# VAYU-NET — Runtime Artifacts Specification & Integrity Manifest

**Module:** DevOps & ML Deployment Integration  
**Project:** VAYU-NET — North Indian Ocean Tropical Cyclone Intelligence System  
**Baseline:** SIH 2026 Problem Statement 26070  
**Storage Provider Status:** Local Checkpoints Verified (Remote Storage: `MANUAL STEP — NOT CONFIGURED`)

> [!NOTE]
> **Cloud Deployment Status:** Cloud deployment has not yet been performed. Manual deployment is pending.
> All 5 runtime models, calibration metadata, and explainability assets are **VERIFIED LOCALLY**.

---

## 1. Overview & Separation of Concerns

VAYU-NET strictly isolates its operational inference runtime from its scientific retraining pipelines. The deployed backend application requires only verified runtime checkpoints and metadata assets (< 150 MB total), not the multi-gigabyte raw satellite netCDF and reanalysis archives.

| Category | Typical Location | Git Tracked? | Local Presence | Remote Status | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Runtime Models** | `data/interim/ml/checkpoints/` | NO | **VERIFIED (100%)** | `NOT_PROVISIONED` | 5 frozen PyTorch checkpoints needed for inference (~100 MB) |
| **Runtime Metadata** | `data/interim/ml/` | **YES** | **VERIFIED (100%)** | Committed | Normalization stats, P80 error cones, analog vectors |
| **Demo Assets** | `data/interim/ml/explainability/` | **YES** | **VERIFIED (100%)** | Committed | Pre-generated Grad-CAM visualizations for 5 shortlist storms |
| **Manifests & Track** | `data/manifests/`, `data/processed/` | **YES** | **VERIFIED (100%)** | Committed | Storm event catalog, sample index, verified best-track |
| **Research Datasets** | `data/raw/`, `data/interim/gridsat/` | NO | Local Only | N/A | Raw satellite netCDF and ERA5 reanalysis training data |
| **Feature Caches** | `data/interim/ml/cache/` | NO | Local Only | N/A | Cached sequence tensors for multi-epoch training/inference |

---

## 2. Mandatory Runtime Model Checkpoints

All 5 checkpoints are present locally and have been cryptographically verified via SHA-256:

### 1. Dedicated Center Localization CNN (Phase 3C)
- **Path:** `data/interim/ml/checkpoints/best_center_localization_cnn.pt`
- **Size:** 49.03 MB (51,406,686 bytes)
- **SHA-256:** `5ef11b596ef94ece71079c1329eab1e50191127eb7b9b1ff13f7dcdd3185f1c9`
- **Role:** Pinpoints current ($t_0$) cyclone center coordinates $[\text{lat}, \text{lon}]$ directly from the full-basin satellite frame.

### 2. Temporal Track GRU (Phase 4A)
- **Path:** `data/interim/ml/checkpoints/best_temporal_track_gru.pt`
- **Size:** 46.17 MB (48,407,839 bytes)
- **SHA-256:** `7973c12a1d1b15fcb4c201deb9491bdb51d4ae9022881aff5233d85606676b46`
- **Role:** Provides 132-dimensional spatial visual embeddings across the 6-step observation sequence ($t_{-15\text{h}}$ to $t_0$).

### 3. Hybrid Residual Track Model — Variant A (Observed Center Anchor)
- **Path:** `data/interim/ml/checkpoints/best_phase5b_variant_a.pt`
- **Size:** 1.67 MB (1,754,009 bytes)
- **SHA-256:** `c1a51fb9d2e55a6043c4b903a4b4696d4c98cb9034082e4825fb6682fefd9216`
- **Validation DPE:** **183.22 km** | **Held-Out Test DPE:** **189.71 km**
- **Anchor:** Observed IMD past center fixes.
- **Role:** Canonical production checkpoint for `MODEL_INFERENCE` track forecasting.

### 4. Hybrid Residual Track Model — Variant B (Satellite-Derived Anchor)
- **Path:** `data/interim/ml/checkpoints/best_phase5b_variant_b.pt`
- **Size:** 1.67 MB (1,754,009 bytes)
- **SHA-256:** `2f0ec1e1e136c552a1ab4a4a8002e1af3293489d5fb649d95b30c2ca5e4a9533`
- **Validation DPE:** **1048.57 km** | **Held-Out Test DPE:** **1023.77 km**
- **Anchor:** Phase 3C satellite-derived center fixes.
- **Role:** Reference autonomous ablation model (retained for research transparency).

### 5. Multi-Task Intensity Classification & Wind Speed Model (Phase 6)
- **Path:** `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt`
- **Size:** 1.44 MB (1,506,213 bytes)
- **SHA-256:** `a014e06dbf6fc17f37d04bb545a58746e57be6cb309808735660989009a460ef`
- **Validation Metrics:** Macro F1: 0.2006, Accuracy: 25.10%, Wind MAE: 25.44 kt.
- **Role:** Estimates cyclone intensity category (7 classes) and sustained wind speed (kt).

---

## 3. Remote Storage & Download Script Semantics

In `artifacts/manifest.json`, remote external source URLs are explicitly marked as:
`source_status = "NOT_PROVISIONED"`

The download utility `scripts/setup/download_runtime_artifacts.py` operates under strict semantics:
- `python scripts/setup/download_runtime_artifacts.py --check`:
  Verifies local artifact presence and SHA-256 integrity only. Never initiates network calls. Exits `0` if all local files are present and match checksums.
- `python scripts/setup/download_runtime_artifacts.py`:
  Attempts retrieval of missing checkpoints from configured remote sources (`MODEL_STORAGE_BASE_URL`). If sources are `NOT_PROVISIONED` or unconfigured, it **fails clearly** with exit code `1`. It never silently fabricates or substitutes artifacts.
