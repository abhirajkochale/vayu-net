# VAYU-NET PRE-FLIGHT RED-TEAM AUDIT MASTER REPORT

**Project:** VAYU-NET - North Indian Ocean Tropical Cyclone Intelligence System
**SIH Problem:** 26070 - MoES / India Meteorological Department
**Audit Type:** Independent Pre-Flight Red-Team
**Audit Date:** 2026-09-25
**Mandate:** Find what is wrong. Do not make the project look good.

> WARNING: This report was produced by INDEPENDENT DIRECT INSPECTION of files, checksums, model weights,
> and code paths - NOT by trusting prior self-reports. All claims are backed by executable evidence.

## 1. AUDIT SCOPE

| Area | Covered |
|---|---|
| Model checkpoint integrity (SHA-256, content inspection) | YES |
| Data split leakage (event-level, row-level, fallback paths) | YES |
| Label integrity in sample index | YES |
| Uncertainty derivation split provenance | YES |
| Analog retrieval split provenance | YES |
| Explainability manifest consistency | YES |
| Deployment config (render.yaml, vercel.json) | YES |
| CORS security | YES |
| API contract code paths | YES |
| Scientific performance claims vs checkpoint contents | YES |
| Runtime artifact URL validity | YES |

## 2. VERIFICATION BASELINE

| Check | Result |
|---|---|
| SHA-256 for all 4 model checkpoints | ALL MATCH manifest.json |
| torch.load inspection of all 4 checkpoints | COMPLETED |
| Storm-level split isolation | PASS - no storm in multiple splits |
| TEST predictions_lookup coverage | 371/371 TEST samples stored |
| Analog candidate split cross-reference | 696 candidates all TRAIN storms |
| Explainability assets (15 files) | All present and non-zero |
| Sample index label completeness | 6 missing imd_category_t0 values |

## 3. FINDINGS SUMMARY

| ID | Severity | Category | Title |
|---|---|---|---|
| F-01 | HIGH | Model Integrity | Phase 5B checkpoint DPE=1048 km - contradicts Variant A benchmark |
| F-02 | MEDIUM | Data Leakage | Silent ground-truth fallback in get_forecast() |
| F-03 | MEDIUM | Model Performance | Phase 6 intensity accuracy 25.1% - below persistence |
| F-04 | MEDIUM | Deployment | All artifact source URLs are placeholder |
| F-05 | MEDIUM | Security | Default CORS allows wildcard origin |
| F-06 | LOW | Data Integrity | Analog cache candidate split field is UNKNOWN |
| F-07 | LOW | Explainability | Grad-CAM targets intermediate layer not classifier head |
| F-08 | LOW | Model Integrity | Phase 5B residual magnitudes over 1700 km at p90 |
| F-09 | INFO | Reproducibility | Phase 3C checkpoint at epoch=2 only |
| F-10 | INFO | Deployment | render.yaml build step checks but does not download artifacts |
| F-11 | INFO | Scientific Integrity | 3/5 Grad-CAM cases are TEST split - unlabeled |
| F-12 | INFO | Deployment | vercel.json missing installCommand |

## 4. DETAILED FINDINGS

### F-01 - HIGH: Phase 5B Checkpoint Identity Mismatch

Component: data/interim/ml/checkpoints/best_phase5b_variant_b.pt

Evidence (directly measured via torch.load):
  val_mean_track_dpe = 1048.5696 km
  12h mean = 800.64 km, median = 427.28 km, p90 = 2133.20 km
  24h mean = 943.39 km, median = 553.57 km, p90 = 2210.39 km
  48h mean = 1401.68 km, median = 780.48 km, p90 = 3774.35 km
  epoch = 14

Published benchmark: Variant A TEST mean DPE = 189.71 km

Problem: The deployed checkpoint has VAL DPE=1048 km, exceeding even Phase 4A baseline
(VAL=724.87 km). The backend ForecastVerifier.predictions_lookup is loaded from
phase5b_hybrid_results.json (371 pre-stored Variant A TEST predictions), NOT from running
this checkpoint at inference time. Displayed forecasts come from the JSON results file.
The checkpoint identity is a reproducibility concern, not an active accuracy bug in the demo.

Impact: If checkpoint is ever used for live inference the output will be 5x worse than
the claimed 189.71 km benchmark. Checkpoint identity must be clarified.

Recommendation:
  1. Confirm whether best_phase5b_variant_b.pt is genuinely Variant B
  2. Confirm best_phase5b_variant_a.pt exists with VAL DPE ~189 km
  3. Add variant and experiment_id keys to checkpoint metadata

### F-02 - MEDIUM: Silent Ground-Truth Fallback in get_forecast()

Component: ml/forecast/api_contracts.py lines 154-159

Problem: When sample_id not in predictions_lookup the code falls back to reading
imd_lat_12h / imd_lon_12h ground-truth coordinates and presents them as forecast outputs
without any label or warning.

Evidence:
  All 371 TEST samples ARE in predictions_lookup (verified)
  VALIDATION samples: 252 total, 0 stored -> all trigger fallback
  TRAIN samples: 696 total, 0 stored -> all trigger fallback

Current exposure: Demo shortlist (FANI/AMPHAN/TAUKTAE/BIPARJOY/REMAL) are TEST cyclones
with stored predictions - no active leakage for the demo. But VALIDATION/TRAIN queries
silently return ground truth labeled as model forecast.

Recommendation: Add label source=FALLBACK_GROUND_TRUTH and WARNING log if fallback triggers.

### F-03 - MEDIUM: Phase 6 Intensity Classifier Below Persistence Baseline

Component: data/interim/ml/checkpoints/best_phase6_intensity_wind.pt

Evidence (directly measured):
  accuracy  = 0.2510 (25.1%)
  macro_f1  = 0.2006
  per_class_f1 = [0.255, 0.167, 0.368, 0.158, 0.218, 0.238, 0.0]
  wind_mae  = 25.44 kt
  wind_rmse = 34.30 kt
  temperature (calibration) = 1.725

One class has F1=0.0 (never correctly predicted). Temperature > 1.0 indicates
significant overconfidence requiring post-hoc calibration.

Recommendation: Dashboard must show explicit disclaimer that intensity classification
is experimental and sub-persistence.

### F-04 - MEDIUM: Artifact Source URLs Are Placeholder

Component: artifacts/manifest.json

Evidence: All 4 model source_url fields contain placeholder.vayu-net.gov.in (fictional domain).
render.yaml buildCommand uses --check mode which only verifies presence, does not download.
On fresh Render instance models would be absent and check would fail.

Recommendation:
  1. Upload weights to Supabase Storage
  2. Update source_url with real URLs
  3. Change buildCommand to download without --check

### F-05 - MEDIUM: Default CORS Includes Wildcard

Component: apps/backend/main.py line 47

Evidence: Default CORS_ORIGINS env var includes * making all origins allowed in dev/CI.
Production render.yaml overrides this safely. Local dev without .env uses wildcard.

Recommendation: Remove * from default CORS_ORIGINS string.

### F-06 - LOW: Analog Cache Candidate split Field is UNKNOWN

Evidence: All 696 candidates have split=UNKNOWN. Cross-ref confirms all 81 candidate
storm_ids are TRAIN. No leakage but self-certification absent.

### F-07 - LOW: Grad-CAM Targets Intermediate Layer Not Classifier Head

Evidence: target_layer=spatial_encoder.layer2 in explainability manifest.
Phase 6 model state dict has no CNN layer parameters. Saliency targets ResNet layer
in the Phase 4A spatial encoder. Scientifically valid but differs from WP-08 spec.

### F-08 - LOW: Phase 5B Residual Magnitudes Implausibly Large

Evidence: 12h mean_mag=1701.97 km, 48h mean_mag=4391.04 km, p90 DPE=3774 km.
uncertainty_parameters.json uses Variant A residuals (reasonable: 12h p80=97.51 km).
Checkpoint-stored Variant B residuals are extreme.

### F-09 - INFO: Phase 3C Checkpoint at Epoch 2

Evidence: epoch=2 in checkpoint. val_mean_dpe_km=703.0. TEST DPE=810.3 km.
Early stopping likely triggered. Training convergence cannot be verified.

### F-10 - INFO: render.yaml Build Step Does Not Download

Evidence: --check flag in buildCommand only verifies, does not download.
Combined with F-04 (placeholder URLs), fresh cloud deployment would fail.

### F-11 - INFO: Grad-CAM Cases Include TEST Split Without Labeling

Evidence: TAUKTAE, BIPARJOY, REMAL are TEST split. No leakage risk (post-hoc).
Should be labeled in dashboard.

### F-12 - INFO: vercel.json Missing installCommand

Evidence: No installCommand in vercel.json. Vercel default is npm install instead of npm ci.

## 5. ITEMS VERIFIED CLEAN

| Check | Result |
|---|---|
| All 4 checkpoint SHA-256 checksums | MATCH manifest exactly |
| Event-level split isolation | CLEAN - no storm in 2+ splits |
| TEST predictions coverage | CLEAN - 371/371 stored |
| Analog candidate pool | CLEAN - all TRAIN storms confirmed |
| Uncertainty derivation split | CLEAN - VALIDATION only |
| Track coordinate labels (lat/lon) | CLEAN - 0 missing rows |
| Explainability assets (15 files) | CLEAN - all present |
| phase5b_hybrid_results.json | CLEAN - Variant A+B, 371 test IDs |

## 6. DEPLOYMENT READINESS

| Component | Status | Blocker? |
|---|---|---|
| Backend FastAPI (local) | Structurally sound | No |
| Model checkpoints (local) | Present + checksum-valid | No |
| Model download on cloud | NOT FUNCTIONAL (placeholder URLs) | YES |
| Build command on Render | CHECK-ONLY - no download | YES |
| Frontend vercel.json | Functional, not hardened | No |
| CORS (production) | Safe (render.yaml overrides) | No |
| Grad-CAM assets | 15 files present | No |
| Intensity classifier disclaimer | Missing from UI | No |

VERDICT: Locally functional. NOT cloud-deployment-ready (F-04 + F-10 are production blockers).

## 7. SCIENTIFIC INTEGRITY

| Claim | Verification Method | Status |
|---|---|---|
| Phase 3C TEST DPE = 810.3 km | Checkpoint VAL=703 km; consistent gap | PLAUSIBLE |
| Phase 4A TEST DPE = 848.5 km | Checkpoint VAL=724.87 km; consistent | PLAUSIBLE |
| Phase 5B Variant A TEST DPE = 189.71 km | 371 stored predictions in results JSON | VERIFIABLE |
| Phase 6 accuracy = 25.1% | Directly measured from checkpoint | CONFIRMED |
| Uncertainty from VALIDATION only | derivation_split=VALIDATION confirmed | CONFIRMED CLEAN |
| Analog pool from TRAIN only | Storm manifest cross-reference | CONFIRMED CLEAN |

IMPORTANT: The 189.71 km result is from STORED PRE-COMPUTED PREDICTIONS in
phase5b_hybrid_results.json, not from running the deployed checkpoint at inference time.
The system serves a lookup-based demo, not live model inference.

## 8. DELIVERABLES

| File | Status |
|---|---|
| docs/AUDIT_MASTER_REPORT.md | THIS FILE |
| docs/AUDIT_FINDINGS.csv | COMPLETE - 12 findings |
| docs/AUDIT_TRACEABILITY_MATRIX.md | COMPLETE |

All evidence is from direct file inspection, Python script execution, SHA-256 verification,
and model weight inspection via torch.load. No prior audit reports trusted without verification.
