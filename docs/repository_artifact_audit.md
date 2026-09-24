# VAYU-NET — Repository & Artifact Audit Report

**Date:** 2026-09-25  
**Baseline:** SIH 2026 Problem Statement 26070 — North Indian Ocean Tropical Cyclone Intelligence System  
**Branch:** `main` (Clean initial repository staging)  
**Remote:** `https://github.com/abhirajkochale/vayu-net.git`

---

## 1. Executive Summary & Disk Footprint

The VAYU-NET codebase encompasses the full research and development lifecycle across Phases 1 through 6, Work Packages 06 (Uncertainty & Verification), 07 (Analog-Storm Retrieval), and 08 (Grad-CAM Explainability).

### Overall Directory Breakdown

| Directory | Total Size (MB) | Purpose | Action for Git |
| :--- | :--- | :--- | :--- |
| `apps/` | 0.00 MB | Frontend (React/Vite) & Backend (FastAPI) applications | **KEEP IN GIT** |
| `configs/` | 0.00 MB | Runtime artifact manifests and system configurations | **KEEP IN GIT** |
| `data/manifests/` | 3.78 MB | Sample indices, candidate manifests, SHA-256 manifests | **KEEP IN GIT** |
| `data/processed/` | 2.33 MB | Cleaned IMD best-track tables (`imd_best_track_v2.csv`) | **KEEP IN GIT** |
| `data/interim/ml/*.json` | 5.48 MB | Experiment results, normalization stats, uncertainty, audits | **KEEP IN GIT** |
| `data/interim/ml/explainability/` | 6.19 MB | Reference Grad-CAM overlays, heatmaps, and metadata JSON | **KEEP IN GIT** (demo assets) |
| `data/interim/ml/checkpoints/` | 455.59 MB | PyTorch model weights (.pt files) | **EXCLUDE FROM GIT** (Supabase/S3 Storage) |
| `data/interim/ml/cache/` | 1,674.35 MB | Pre-computed multimodal sequence tensors (.pt files) | **EXCLUDE FROM GIT** (Local research only) |
| `data/interim/gridsat/` | 3,044.20 MB | Preprocessed 572x929 satellite arrays (thousands of .npz) | **EXCLUDE FROM GIT** (Local research only) |
| `data/interim/gridsat_pilot/` | 844.74 MB | Raw pilot netCDF files (.nc) | **EXCLUDE FROM GIT** (Local research only) |
| `data/raw/imd/` | 1,676.94 MB | 15 RSMC / IMD Annual Cyclone Reports (PDFs) | **EXCLUDE FROM GIT** (Archival only) |
| `docs/` | 38.14 MB | Architectural specifications, blueprints, figures, audits | **KEEP IN GIT** |
| `ml/` | 0.60 MB | Model definitions, data loaders, explainability, forecast modules | **KEEP IN GIT** |
| `scripts/` | 0.90 MB | Audit suites, production generation, training pipelines | **KEEP IN GIT** |
| `scratch/` | 2.46 MB | Intermediate validation scripts and QA dumps | **EXCLUDE FROM GIT** |
| `.pytest_cache/` | 0.00 MB | Python test cache | **EXCLUDE FROM GIT** |
| **Total Workspace** | **7,781.83 MB** | Entire local research workspace | **Repository size: ~55 MB** |

---

## 2. Large File Inventory (Files $\ge$ 10 MB)

A total of **61 files** exceed 10 MB in the current workspace. None of these files should be committed directly to GitHub; they belong in `.gitignore` while remaining preserved in the local environment.

| PATH | TYPE | SIZE (MB) | PURPOSE | KEEP IN GIT? | RUNTIME REQUIRED? | EXTERNAL STORAGE? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `data/interim/ml/cache/phase5b_hybrid_features.pt` | PyTorch Cache | 665.48 | Pre-extracted sequence features for Phase 5B training | NO | NO | Optional (Re-computable) |
| `data/interim/ml/cache/phase6_intensity_features.pt` | PyTorch Cache | 661.25 | Pre-extracted multimodal features for Phase 6 training | NO | NO | Optional (Re-computable) |
| `data/interim/ml/cache/era5_environment_features.pt` | PyTorch Cache | 329.77 | Environmental 8-channel wind fields for sequence training | NO | NO | Optional (Re-computable) |
| `data/interim/ml/checkpoints/best_spatial_single_frame_cnn.pt` | PyTorch Model | 130.63 | Phase 3B spatial ablation checkpoint | NO | NO | Yes (Model Archive) |
| `data/interim/ml/checkpoints/best_single_frame_cnn.pt` | PyTorch Model | 130.24 | Phase 3A baseline CNN checkpoint | NO | NO | Yes (Model Archive) |
| `data/raw/imd/27_c8dbd0_0_COVER_PAGE_RSMC_Report_2025.pdf` | Document | 93.55 | IMD 2025 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_7d3be4_Final_upload RSMC Report 2024 (1).pdf` | Document | 82.81 | IMD 2024 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_26e77b_rsmc-2020 with damage.pdf` | Document | 80.12 | IMD 2020 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_a7fb96_27_ef4e32_RSMC-Report2024-for upload.pdf` | Document | 79.04 | IMD 2024 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_501da8_RSMC full report 2022 13 Jan.pdf` | Document | 77.91 | IMD 2022 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_fddc6c_rsmc2020.pdf` | Document | 76.98 | IMD 2020 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_60dae9_rsmc-2018.pdf` | Document | 76.59 | IMD 2018 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_81bdf0_RSMC Report 2021.pdf` | Document | 57.17 | IMD 2021 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_ad292c_rsmc-2016.pdf` | Document | 56.95 | IMD 2016 cyclone report | NO | NO | External Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.25.18.v02r01.nc` | NetCDF4 | 54.70 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.25.18.v02r01.nc` | NetCDF4 | 54.70 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.26.18.v02r01.nc` | NetCDF4 | 54.66 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.26.18.v02r01.nc` | NetCDF4 | 54.66 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.26.00.v02r01.nc` | NetCDF4 | 50.16 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.26.00.v02r01.nc` | NetCDF4 | 50.16 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.25.21.v02r01.nc` | NetCDF4 | 50.02 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.25.21.v02r01.nc` | NetCDF4 | 50.02 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.26.03.v02r01.nc` | NetCDF4 | 49.46 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.26.03.v02r01.nc` | NetCDF4 | 49.46 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/interim/ml/checkpoints/best_center_localization_cnn.pt` | PyTorch Model | 49.03 | **Phase 3C Center Localization Backbone** | NO | **YES** | **Supabase Storage** |
| `data/interim/ml/checkpoints/best_phase4b_variant_a.pt` | PyTorch Model | 46.30 | Phase 4B Variant A (Observed Center Hybrid) | NO | NO | Yes (Model Archive) |
| `data/interim/ml/checkpoints/best_phase4b_variant_b.pt` | PyTorch Model | 46.30 | Phase 4B Variant B (Estimated Center Hybrid) | NO | NO | Yes (Model Archive) |
| `data/interim/ml/checkpoints/best_temporal_track_gru.pt` | PyTorch Model | 46.17 | **Phase 4A Shared Spatial Projection + GRU** | NO | **YES** | **Supabase Storage** |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.16.00.v02r01.nc` | NetCDF4 | 45.10 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.16.00.v02r01.nc` | NetCDF4 | 45.10 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.18.00.v02r01.nc` | NetCDF4 | 44.17 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.18.00.v02r01.nc` | NetCDF4 | 44.17 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.15.15.v02r01.nc` | NetCDF4 | 43.69 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.15.15.v02r01.nc` | NetCDF4 | 43.69 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.15.21.v02r01.nc` | NetCDF4 | 43.42 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.15.21.v02r01.nc` | NetCDF4 | 43.42 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.15.09.v02r01.nc` | NetCDF4 | 43.20 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.15.09.v02r01.nc` | NetCDF4 | 43.20 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.15.18.v02r01.nc` | NetCDF4 | 43.07 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.15.18.v02r01.nc` | NetCDF4 | 43.07 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.17.00.v02r01.nc` | NetCDF4 | 43.02 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.17.00.v02r01.nc` | NetCDF4 | 43.02 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.16.12.v02r01.nc` | NetCDF4 | 42.48 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.16.12.v02r01.nc` | NetCDF4 | 42.48 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2020/GRIDSAT-B1.2020.05.15.12.v02r01.nc` | NetCDF4 | 42.47 | Raw GridSat file (Amphan) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/amphan_2020/raw/GRIDSAT-B1.2020.05.15.12.v02r01.nc` | NetCDF4 | 42.47 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.25.15.v02r01.nc` | NetCDF4 | 42.25 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.25.15.v02r01.nc` | NetCDF4 | 42.25 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.28.06.v02r01.nc` | NetCDF4 | 42.01 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.28.06.v02r01.nc` | NetCDF4 | 42.01 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.27.06.v02r01.nc` | NetCDF4 | 41.96 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.27.06.v02r01.nc` | NetCDF4 | 41.96 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/gridsat/2019/GRIDSAT-B1.2019.04.26.06.v02r01.nc` | NetCDF4 | 41.89 | Raw GridSat file (Fani) | NO | NO | NOAA NCEI Archive |
| `data/interim/gridsat_pilot/fani_2019/raw/GRIDSAT-B1.2019.04.26.06.v02r01.nc` | NetCDF4 | 41.89 | Raw pilot netCDF | NO | NO | NOAA NCEI Archive |
| `data/raw/imd/27_14ab8f_rsmc-2013.pdf` | Document | 34.60 | IMD 2013 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_4e1280_rsmc-2014.pdf` | Document | 26.98 | IMD 2014 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_bbaf11_rsmc-2017.pdf` | Document | 19.86 | IMD 2017 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_9bbd0d_RSMC-2015.pdf` | Document | 15.34 | IMD 2015 cyclone report | NO | NO | External Archive |
| `data/raw/imd/27_2f6165_rsmc-2011.pdf` | Document | 14.82 | IMD 2011 cyclone report | NO | NO | External Archive |
| `data/interim/ml/cache/phase4b_hybrid_features.pt` | PyTorch Cache | 10.53 | Pre-extracted sequence features for Phase 4B | NO | NO | Optional |
| `data/raw/imd/27_2138f3_rsmc-2012.pdf` | Document | 10.52 | IMD 2012 cyclone report | NO | NO | External Archive |

---

## 3. Four-Tier Artifact Categorization

To maintain a clean, lightweight Git repository without sacrificing local scientific integrity, all files are categorized into four tiers:

### Tier A: Repository-Safe Files (Included in Git)
- **Source Code**: `apps/`, `ml/`, `scripts/`
- **Documentation & Specifications**: `docs/` (including all figures and markdown reports)
- **Manifests & Index Metadata**:
  - `data/manifests/vayu_net_sample_index.csv` (1,319 samples)
  - `data/manifests/storm_event_manifest_v2.csv` (126 storms)
  - `data/manifests/vayu_net_candidate_t0_manifest.csv`
  - `data/manifests/gridsat_sha256_manifest.csv`
  - `data/processed/imd_best_track_v2.csv`
- **Calibration & Metric Reports**:
  - `data/interim/ml/train_normalization_stats.json`
  - `data/interim/ml/era5_train_normalization_stats.json`
  - `data/interim/ml/uncertainty_parameters.json`
  - `data/interim/ml/analog_retrieval_cache.json`
  - `data/interim/ml/*_results.json` (Phase 3C, 4A, 4B, 5A, 5B, 6 result JSONs)
- **Demo Reference Assets**:
  - `data/interim/ml/explainability/` (PNG overlays and metadata JSONs for the 5 reference storms)

### Tier B: Runtime Artifacts (Managed via Storage / Artifact Script)
These 4 trained model checkpoints (~98.3 MB) are needed for full AI inference:
1. `data/interim/ml/checkpoints/best_center_localization_cnn.pt` (49.03 MB, SHA-256: `5ef11b59...`)
2. `data/interim/ml/checkpoints/best_temporal_track_gru.pt` (46.17 MB, SHA-256: `7973c12a...`)
3. `data/interim/ml/checkpoints/best_phase5b_variant_b.pt` (1.67 MB, SHA-256: `2f0ec1e1...`)
4. `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt` (1.44 MB, SHA-256: `a014e06d...`)

*Action:* Excluded from Git via `.gitignore`. Downloaded on-demand via `scripts/setup/download_runtime_artifacts.py` or served from Supabase Storage.

### Tier C: Research & Training Data (Local Research Workspace Only)
- `data/raw/imd/*.pdf` (733 MB)
- `data/raw/gridsat/*.nc` and `data/interim/gridsat_pilot/` (1.5 GB)
- `data/interim/gridsat/*.npz` (3.04 GB)
- Historical ablation checkpoints: `best_single_frame_cnn.pt`, `best_spatial_single_frame_cnn.pt`, `best_phase4b_variant_a.pt` (353 MB)

*Action:* Excluded from Git. Preserved on local disks for scientific audit and retraining.

### Tier D: Generated Cache & Ephemeral Files (Excluded from Git)
- `data/interim/ml/cache/*.pt` (1.67 GB feature caches)
- `scratch/` (2.46 MB)
- `__pycache__/`, `.pytest_cache/`, `node_modules/`

*Action:* Excluded from Git via `.gitignore`.
