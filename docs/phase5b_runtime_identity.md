# VAYU-NET — Phase 5B Checkpoint Identity & Runtime Compatibility Resolution

**Problem Statement:** SIH 2026 PS 26070  
**Ministry / Department:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)  
**Investigation Date:** September 2026  
**Auditor / Implementer:** Lead DevOps & ML Deployment Engineer  

---

## 1. Executive Summary & Definitve Findings

Through direct inspection of PyTorch checkpoint state dictionaries, model architecture code (`ml/models/phase5b_hybrid_residual.py`), training scripts (`ml/train/train_phase5b_hybrid_residual.py`), evaluation scripts (`scripts/production/evaluate_phase5b_and_generate_figures.py`), and serialized results (`data/interim/ml/phase5b_hybrid_results.json`), the identities of the Phase 5B checkpoints have been definitively resolved:

1. **`best_phase5b_variant_a.pt` is Variant A:**
   - **Kinematic Anchor:** Exact historical observed center from IMD best track (`kin_ctx_a`, `kin_a_12/24/48`).
   - **Validation DPE:** **183.22 km** (checkpoint epoch 6).
   - **Held-Out Test DPE:** **189.71 km** (evaluated strictly once).
   - **Architecture:** `Phase5BHybridResidualModel` (345,638 parameters).

2. **`best_phase5b_variant_b.pt` is Variant B:**
   - **Kinematic Anchor:** Autonomous satellite-derived estimated center from Phase 3C localization CNN (`kin_ctx_b`, `kin_b_12/24/48`).
   - **Validation DPE:** **1048.57 km** (checkpoint epoch 14).
   - **Held-Out Test DPE:** **1023.77 km** (evaluated strictly once).
   - **Architecture:** Identical `Phase5BHybridResidualModel` (345,638 parameters).

---

## 2. Checkpoint Identity & Compatibility Matrix

| Artifact | Variant | Anchor Source | Inputs | Outputs | Validation DPE | Held-Out Test DPE | Runtime-Compatible | Production Candidate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `best_phase5b_variant_a.pt` | **Variant A** | **Observed IMD Center** | Satellite sequence [6, 132], ERA5 sequence [6, 8, 41, 66], Observed motion context [5], Exact kinematic anchors | Geodesic residual displacements & +12h, +24h, +48h lat/lon | **183.22 km** | **189.71 km** | **YES** (Native PyTorch forward pass) | **YES (Forecaster-in-the-loop / Observed-anchor mode)** |
| `best_phase5b_variant_b.pt` | **Variant B** | **Phase 3C Satellite-Derived Center** | Satellite sequence [6, 132], ERA5 sequence [6, 8, 41, 66], Phase 3C motion context [5], Satellite kinematic anchors | Geodesic residual displacements & +12h, +24h, +48h lat/lon | **1048.57 km** | **1023.77 km** | **YES** (Native PyTorch forward pass) | **NO (Fails operational error threshold)** |

---

## 3. Scientific & Operational Integrity Clarification

### 3.1 The 189.71 km Benchmark Provenance
The **189.71 km** mean track error benchmark reported in historical project documentation is strictly the test performance of **Variant A**.
Variant A relies on **exact past centers observed at $t_{-15h}, \dots, t_0$** by IMD radar/best-track/bulletin observations to construct its initial kinematic trajectory anchor. The neural network then learns the atmospheric steering residual ($R_{\Delta t}$).

### 3.2 Operational Limitation Notice
- In an operational emergency setting, if no verified human or radar center observation is available and the system must initialize its kinematic anchor purely from noisy single-frame satellite center predictions (Variant B), kinematic errors compound dramatically over 48 hours ($1048.57\text{ km}$).
- Therefore, **the 189.71 km figure must NOT be advertised as fully autonomous satellite-only operational performance**. It represents a **"Forecaster-in-the-Loop" / Observed-Anchor Hybrid Model**, which is standard operational practice at IMD where synoptic forecasters provide verified past fixes before running guidance models.

---

## 4. Runtime Inference Implementation

Both checkpoints share the exact same model definition:
```python
model = Phase5BHybridResidualModel(
    sat_feat_dim=132,
    sat_hidden_dim=128,
    env_channels=8,
    env_spatial_dim=64,
    env_hidden_dim=64,
    motion_in_dim=5,
    motion_emb_dim=16,
    fusion_dim=128,
    dropout=0.1
)
```
The newly created `ml/inference/model_service.py` provides clean, isolated loading of both checkpoints on CPU:
- When `predict_track(sample_id, variant="A")` is called, `best_phase5b_variant_a.pt` is executed.
- When `predict_track(sample_id, variant="B")` is called, `best_phase5b_variant_b.pt` is executed.
- Responses explicitly return `checkpoint_identity`, `model_version`, and `anchor_type`.
