# VAYU-NET AUDIT TRACEABILITY MATRIX

**Project:** VAYU-NET SIH 2026 (Problem 26070)
**Document Version:** 1.0
**Date:** 2026-09-25
**Purpose:** Maps each audit finding to the specific files, code lines, and evidence used to verify it.

---

## TRACEABILITY TABLE

| Finding ID | Severity | Finding Title | Primary Evidence File(s) | Evidence Type | Code Location | Verification Method | Finding Status |
|---|---|---|---|---|---|---|---|
| F-01 | HIGH | Phase 5B checkpoint DPE=1048 km | best_phase5b_variant_b.pt | torch.load inspection | N/A (binary file) | Python torch.load; val_mean_track_dpe field | OPEN |
| F-02 | MEDIUM | Silent ground-truth fallback in get_forecast | ml/forecast/api_contracts.py | Source code + data cross-ref | L154-159 | Code review + verify 371 TEST IDs in predictions_lookup | OPEN |
| F-03 | MEDIUM | Phase 6 intensity accuracy 25.1% | best_phase6_intensity_wind.pt | torch.load inspection | N/A (binary file) | Python torch.load; val_metrics.accuracy field | OPEN |
| F-04 | MEDIUM | Artifact source URLs are placeholder | artifacts/manifest.json | JSON inspection | N/A | grep + direct read of source_url fields | OPEN |
| F-05 | MEDIUM | Default CORS includes wildcard | apps/backend/main.py | Source code | L47, L51 | Code review of CORS_ORIGINS default value | OPEN |
| F-06 | LOW | Analog cache split is UNKNOWN | data/interim/ml/analog_retrieval_cache.json | JSON inspection + cross-ref | N/A | Python: cand_splits set + storm manifest cross-reference | OPEN |
| F-07 | LOW | Grad-CAM targets intermediate layer | explainability_manifest.json + best_phase6_intensity_wind.pt | Manifest + checkpoint | N/A | Read target_layer field + inspect Phase 6 state dict keys | OPEN |
| F-08 | LOW | Phase 5B residuals >1700 km at p90 | best_phase5b_variant_b.pt | torch.load inspection | N/A | Python torch.load; val_metrics.residuals field | OPEN |
| F-09 | INFO | Phase 3C at epoch=2 | best_center_localization_cnn.pt | torch.load inspection | N/A | Python torch.load; epoch field | OPEN |
| F-10 | INFO | render.yaml build step does not download | render.yaml + scripts/setup/download_runtime_artifacts.py | Config + script | buildCommand line | Read render.yaml buildCommand; read download script --check mode | OPEN |
| F-11 | INFO | Grad-CAM cases include TEST split unlabeled | explainability_manifest.json | JSON inspection | N/A | Read split field for each of 5 cases | OPEN |
| F-12 | INFO | vercel.json missing installCommand | apps/frontend/vercel.json | JSON inspection | N/A | Read vercel.json and confirm absence of installCommand | OPEN |

---

## REQUIREMENT-TO-FINDING TRACEABILITY

| Requirement | Requirement Source | Related Finding(s) | Compliance |
|---|---|---|---|
| No TEST data used for uncertainty derivation | SIH 26070 Blueprint; Phase 5B spec | (none - PASSED) | PASS |
| Event-level train/validation/test split | Phase 3C-6 methodology | (none - PASSED) | PASS |
| All model checkpoints must be SHA-256 verified | Repository audit spec | (none - PASSED) | PASS |
| Ground truth must not enter production forecast | Leakage audit directive | F-02 | PARTIAL (TEST OK; VAL/TRAIN fallback leaks) |
| Production deployment must be functional | Deployment audit scope | F-04, F-10 | FAIL (cloud deployment blocked) |
| Intensity classification must be clearly disclaimed | Scientific integrity spec | F-03 | OPEN |
| Analog retrieval pool must be TRAIN-only | WP-07 spec | F-06 | PASS (verified via cross-ref despite UNKNOWN label) |
| Grad-CAM saliency on at least one vision frame | WP-08 acceptance criterion | F-07 | PARTIAL (saliency exists; wrong layer per spec) |
| CORS must be restricted in production | Security baseline | F-05 | PARTIAL (production safe; dev wildcard) |
| All runtime artifact URLs must be resolvable | Deployment spec | F-04 | FAIL |

---

## EVIDENCE ARTIFACTS

| Script / Tool | Purpose | Findings Supported |
|---|---|---|
| scratch/check_artifacts.py | SHA-256 verification of all 4 checkpoints + runtime files | Baseline (all clean) |
| scratch/deep_audit2.py | Split leakage check; uncertainty provenance; analog cache; fallback analysis | F-02, F-06 |
| scratch/audit3.py | CORS check; source URL check; vercel.json check; split case sensitivity | F-05, F-04, F-12 |
| scratch/audit4.py | torch.load of all 4 checkpoints; render.yaml full content; download script | F-01, F-04, F-09, F-10 |
| scratch/audit5b.py | Explainability manifest; Phase 5B/6 val_metrics; Grad-CAM target layer | F-03, F-07, F-08, F-11 |

---

## PHASE-LEVEL RESULT TRACEABILITY

| Phase | Published Result | Stored in Checkpoint | Cross-Checked Against | Consistent? |
|---|---|---|---|---|
| Phase 3C | TEST mean DPE = 810.3 km | VAL mean DPE = 703.0 km (epoch=2) | phase3c_center_localization.md | PLAUSIBLE (expected generalization gap) |
| Phase 4A | TEST mean DPE = 848.5 km | VAL mean DPE = 724.87 km (epoch=13) | phase4a_temporal_track_prediction.md | PLAUSIBLE |
| Phase 5B Var A | TEST mean DPE = 189.71 km | phase5b_hybrid_results.json raw_test, 371 IDs | docs/phase5b_hybrid_residual.md | VERIFIABLE (JSON stores predictions) |
| Phase 5B Var B | Checkpoint DPE = 1048.57 km | best_phase5b_variant_b.pt val_metrics | N/A | FINDING F-01 (identity ambiguity) |
| Phase 6 | Accuracy = 25.1% | best_phase6_intensity_wind.pt val_metrics | docs/phase6_intensity_wind.md | CONFIRMED |

---

## DATA SPLIT TRACEABILITY

| Dataset Component | TRAIN | VALIDATION | TEST | Verified? |
|---|---|---|---|---|
| Sample index (vayu_net_sample_index.csv) | 696 samples | 252 samples | 371 samples | YES |
| Storm event manifest (storm_event_manifest_v2.csv) | 81 storms | N/A | N/A | YES |
| Analog candidate pool | 696 (all TRAIN) | 0 | 0 | YES (cross-ref) |
| Uncertainty parameters (uncertainty_parameters.json) | 0 | 252 residuals | 0 | YES (derivation_split=VALIDATION) |
| Stored predictions (phase5b_hybrid_results.json) | 0 | 0 | 371 | YES |
| Explainability cases (explainability_manifest.json) | 0 | 2 (FANI, AMPHAN) | 3 (TAUKTAE, BIPARJOY, REMAL) | YES |

---

## DEPLOYMENT ARTIFACT TRACEABILITY

| Artifact | Local Present | SHA-256 Matches Manifest | Source URL Valid | Cloud-Ready? |
|---|---|---|---|---|
| best_center_localization_cnn.pt (51.4 MB) | YES | YES | NO (placeholder) | NO |
| best_temporal_track_gru.pt (48.4 MB) | YES | YES | NO (placeholder) | NO |
| best_phase5b_variant_b.pt (1.75 MB) | YES | YES | NO (placeholder) | NO |
| best_phase6_intensity_wind.pt (1.51 MB) | YES | YES | NO (placeholder) | NO |
| uncertainty_parameters.json (1.97 KB) | YES | YES | N/A (no source_url) | YES (small, can commit) |
| analog_retrieval_cache.json (651 KB) | YES | YES | N/A | YES (small enough) |
| vayu_net_sample_index.csv (653 KB) | YES | YES | N/A | YES |
| storm_event_manifest_v2.csv (29 KB) | YES | YES | N/A | YES |
| train_normalization_stats.json (544 B) | YES | YES | N/A | YES |
| era5_train_normalization_stats.json (1.8 KB) | YES | YES | N/A | YES |
| explainability_manifest.json (8.6 KB) | YES | N/A | N/A | YES |
| Grad-CAM assets (5 x 3 files) | YES | N/A | N/A | YES (if committed) |

---

## AUDIT COMPLETION CHECKLIST

| Audit Item | Status |
|---|---|
| All 4 model checkpoints SHA-256 verified | DONE |
| All model checkpoints torch.load inspected | DONE |
| Sample index split integrity verified | DONE |
| Fallback leakage path analyzed | DONE |
| Uncertainty derivation provenance verified | DONE |
| Analog candidate pool split verified | DONE |
| Explainability manifest and assets verified | DONE |
| Backend source code (api_contracts.py) reviewed | DONE |
| Deployment configs (render.yaml, vercel.json) reviewed | DONE |
| CORS configuration reviewed | DONE |
| Artifact manifest source URL audit | DONE |
| Performance claims vs checkpoint cross-check | DONE |
| AUDIT_MASTER_REPORT.md generated | DONE |
| AUDIT_FINDINGS.csv generated | DONE |
| AUDIT_TRACEABILITY_MATRIX.md generated | DONE |
