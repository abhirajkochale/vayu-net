# VAYU-NET WP-08 — Explainability Audit & Data Integrity Report

## 1. Audit Overview

This audit document certifies that Work Package 08 (`WP-08: Grad-CAM / Saliency Explainability`) adheres strictly to all governance, anti-leakage, reproducibility, and architectural constraints established by the VAYU-NET project for SIH 2026 Problem Statement 26070.

> [!CAUTION]
> ### LOCKED INTERPRETATION DISCLAIMER
> **The saliency/Grad-CAM output is an interpretation aid and must not be presented as a causal meteorological explanation.**
> All user-facing representations, documentation, and API contracts must display this disclaimer prominently.

---

## 2. Invariant Verification Matrix

| Audit Dimension | Requirement | Audit Finding | Status |
| :--- | :--- | :--- | :--- |
| **Model Weight Freezing** | Zero weights modified; no fine-tuning or retraining | Verified: Weights loaded in `eval()` mode. Zero optimizer calls. | **PASS** ✅ |
| **Model Architectures** | Reuse Phase 3C spatial encoder & Phase 6 multi-task model | Verified: Direct hook on `DedicatedCenterLocalizationResNet.layer2` feeding `Phase6IntensityWindModel`. | **PASS** ✅ |
| **Temporal Anti-Leakage** | Zero access to future observations or labels | Verified: Sequence strictly terminates at $t_0$ (frame index 5). Zero future frames ($t_{+12\text{h}}$, etc.) accessed. | **PASS** ✅ |
| **Split Isolation** | Explanations generated for evaluation/demo only; no test tuning | Verified: Explanations generated on representative cases without threshold optimization or hyperparameter search. | **PASS** ✅ |
| **Spatial Alignment** | Heatmap dimensions match input frame ($572 \times 929$) | Verified: Bilinear interpolation produces exact $572 \times 929$ tensor aligned to North Indian Ocean grid. | **PASS** ✅ |
| **Finite Bounds** | No NaNs, Infs, or all-zero heatmaps | Verified: All heatmaps strictly non-empty, finite, and min-max scaled to $[0.0, 1.0]$. | **PASS** ✅ |
| **Deterministic Output** | Identical inputs produce identical attributions | Verified: Maximum absolute difference across repeated runs $< 10^{-5}$. | **PASS** ✅ |
| **Artifact Integrity** | All locked historical checkpoints and manifests untouched | Verified: MD5 / SHA-256 and byte sizes of all historical checkpoints verified. | **PASS** ✅ |

---

## 3. Detailed Audit Findings

### 3.1 Model Frozen State Verification
The explainability engine (`GradCAMExplainer`) instantiates the frozen models:
- `data/interim/ml/checkpoints/best_center_localization_cnn.pt` (Phase 3C)
- `data/interim/ml/checkpoints/best_temporal_track_gru.pt` (Phase 4A)
- `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt` (Phase 6)

Forward execution runs with `torch.no_grad()` for reference evaluation or with backward hooks isolated strictly to `spatial_encoder.layer2` during attribution calculation. No model parameters are updated (`requires_grad=False` on backbone parameters).

### 3.2 Anti-Leakage Verification
Satellite observation input sequences are restricted to:
1. $t_{-15\text{h}}$
2. $t_{-12\text{h}}$
3. $t_{-9\text{h}}$
4. $t_{-6\text{h}}$
5. $t_{-3\text{h}}$
6. $t_0$ (Terminal observation)

The backward target is the predicted categorical logit $\mathbf{y}[c]$ evaluated at step $t_0$. No future ground-truth trajectory ($t_{+12\text{h}}$, $t_{+24\text{h}}$, $t_{+48\text{h}}$) or future intensity values enter the computational graph.

### 3.3 Historical Artifact Integrity Checks
All critical historical project artifacts have been verified as intact and non-empty:
- `data/manifests/vayu_net_sample_index.csv` (1,319 rows)
- `data/manifests/vayu_net_candidate_t0_manifest.csv`
- `data/manifests/gridsat_sha256_manifest.csv`
- `data/processed/imd_best_track_v2.csv`
- `data/interim/ml/checkpoints/best_center_localization_cnn.pt`
- `data/interim/ml/phase3c_center_localization_results.json`
- `data/interim/ml/checkpoints/best_phase4b_variant_a.pt`
- `data/interim/ml/checkpoints/best_phase4b_variant_b.pt`
- `data/interim/ml/phase4b_hybrid_results.json`
- `data/interim/ml/phase5a_results.json`
- `data/interim/ml/phase5b_hybrid_results.json`
- `data/interim/ml/checkpoints/best_phase6_intensity_wind.pt`
- `data/interim/ml/phase6_intensity_wind_results.json`

### 3.4 Audit Test Execution
Audit test suite `scripts/audit/test_gradcam_explainability.py` executed:
```
============================= test session starts =============================
platform win32 -- Python 3.13.13, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\GitHub\vayu-net
collected 13 items

scripts/audit/test_gradcam_explainability.py::test_1_checkpoints_load PASSED [  7%]
scripts/audit/test_gradcam_explainability.py::test_2_forward_inference PASSED [ 15%]
scripts/audit/test_gradcam_explainability.py::test_3_target_class_extraction PASSED [ 23%]
scripts/audit/test_gradcam_explainability.py::test_4_gradcam_saliency_generation PASSED [ 30%]
scripts/audit/test_gradcam_explainability.py::test_5_heatmap_finite_values PASSED [ 38%]
scripts/audit/test_gradcam_explainability.py::test_6_normalization_bounds PASSED [ 46%]
scripts/audit/test_gradcam_explainability.py::test_7_spatial_alignment_dimensions PASSED [ 53%]
scripts/audit/test_gradcam_explainability.py::test_8_overlay_generation PASSED [ 61%]
scripts/audit/test_gradcam_explainability.py::test_9_metadata_completeness PASSED [ 69%]
scripts/audit/test_gradcam_explainability.py::test_10_deterministic_reproduction PASSED [ 76%]
scripts/audit/test_gradcam_explainability.py::test_11_t0_timestamp_correctness PASSED [ 84%]
scripts/audit/test_gradcam_explainability.py::test_12_no_future_dependency PASSED [ 92%]
scripts/audit/test_gradcam_explainability.py::test_13_locked_dataset_and_historical_artifacts_integrity PASSED [100%]

============================= 13 passed in 7.33s ==============================
```

Combined regression testing with WP-06 (`test_uncertainty_verification.py`) and WP-07 (`test_analog_retrieval.py`):
**33 passed in 7.32s (100% pass rate).**

---

## 4. Certification

The VAYU-NET WP-08 Grad-CAM Explainability implementation meets all acceptance criteria:
1. At least one inference returns a viewable saliency overlay and metadata (all 5 shortlist cases generated).
2. The target layer is a real convolutional layer of the existing vision encoder (`spatial_encoder.layer2`).
3. Spatial alignment matches the source satellite frame ($572 \times 929$).
4. The overlay is documented strictly as an interpretation aid and not a causal meteorological explanation.
5. All locked historical models, datasets, and manifests remain intact.
