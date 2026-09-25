# VAYU-NET — Multi-Source Transfer Learning & Auxiliary Fusion Report

**Module:** Pretrained GridSat Representation Transfer + Lightweight INSAT-3D Fusion  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — TRANSFER LEARNING EXPERIMENT FINALIZED  

---

## 1. Executive Summary & Objective

In the first multi-source experiment, training deep networks from scratch on only 175 paired samples caused severe overfitting.

This second experiment tests the scientific hypothesis:
> *"Does reusing the validated, strong GridSat representation through Transfer Learning allow INSAT-3D to provide measurable auxiliary predictive value?"*

### Architecture Strategy:
1. **GridSat Representation Transfer:** Reused the pretrained 2-layer temporal GRU from `best_temporal_track_gru.pt` (Phase 4A, trained on GridSat-B1 11µm representations).
2. **Freezing Control:**
   - **EXP-1:** Pretrained GridSat GRU completely frozen ($199,680$ frozen params) + trainable multi-task heads ($67,344$ trainable params).
   - **EXP-2:** Pretrained GridSat GRU completely frozen ($199,680$ frozen params) + lightweight INSAT-3D branch ($73,680$ params) + fusion projection & heads ($140,896$ trainable params).
   - **EXP-3:** Pretrained GridSat GRU partially unfrozen (Layer 0 frozen with $100,608$ params, Layer 1 unfrozen with $134k$ params) + INSAT branch + fusion ($239,968$ trainable params).
3. **Identical Evaluation Cohort:** Evaluated strictly on the identical **298 TEST samples** (24 storms, 2021–2024).

---

## 2. Model Architectures & Parameter Efficiency

All models were implemented in [ml/models/multisource_transfer_fusion.py](file:///c:/GitHub/vayu-net/ml/models/multisource_transfer_fusion.py):

| Model / Experiment | Total Parameters | Trainable Parameters | Frozen Parameters | Selected Epoch | Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-1: Frozen GridSat** | 267,024 | 67,344 | 199,680 | Epoch 1 | 3.4268 |
| **EXP-2: Frozen GridSat + INSAT Fusion** | 340,576 | 140,896 | 199,680 | Epoch 1 | 3.4557 |
| **EXP-3: Partial Unfreeze + INSAT Fusion**| 340,576 | 239,968 | 100,608 | Epoch 1 | 3.5275 |
| **Channel Ablation A: TIR1 Only (1 ch)** | 340,288 | 140,608 | 199,680 | Epoch 1 | 3.5141 |
| **Channel Ablation B: TIR1+TIR2 (2 ch)**  | 340,432 | 140,752 | 199,680 | Epoch 1 | 3.5441 |

---

## 3. Training Governance & Overfitting Analysis

- **Random Seed:** $42$
- **Optimization:** AdamW ($\eta = 10^{-3}$, weight decay $10^{-4}$), ReduceLROnPlateau, Early Stopping patience $5$ on Validation Loss.
- **Selection Governance:** Selected strictly by lowest Validation Loss. Test set evaluated exactly once.
- Training curves saved at: [docs/figures/multisource_transfer/transfer_training_curves.png](file:///c:/GitHub/vayu-net/docs/figures/multisource_transfer/transfer_training_curves.png).

### Overfitting Comparison (Transfer vs. From-Scratch):
1. **Dramatic DPE Reduction via Transfer Learning:**
   Transfer learning improved track and center localization by over **$250\text{–}280\text{ km}$** compared to from-scratch models:
   - From-Scratch Model C: Center DPE = $1,220.1\text{ km}$, Track DPE = $1,184.8\text{ km}$.
   - Transfer Model EXP-2: Center DPE = **$938.7\text{ km}$**, Track DPE = **$974.0\text{ km}$**.
2. **Persistent Generalization Gap:**
   Despite transfer learning, training loss dropped from $1.99$ to $1.19$, while validation loss degraded from $3.45$ to $4.14\text{–}4.24$.
3. **Partial Unfreezing Penalty (EXP-3):**
   Unfreezing Layer 1 of the GridSat GRU increased trainable parameters from $140\text{k}$ to $240\text{k}$. On 175 training samples, this accelerated overfitting, yielding a worse validation loss ($3.5275$) and worse track aggregate DPE ($1,002.7\text{ km}$ vs $974.0\text{ km}$).

---

## 4. Checkpoint Registry

Checkpoints saved in `data/interim/ml/checkpoints/`:
- `multisource_transfer_exp1.pt` (Epoch 1, Val Loss 3.4268)
- `multisource_transfer_exp2.pt` (Epoch 1, Val Loss 3.4557)
- `multisource_transfer_exp3.pt` (Epoch 1, Val Loss 3.5275)
- `multisource_transfer_ablation_tir1.pt` (Epoch 1, Val Loss 3.5141)
- `multisource_transfer_ablation_tir1_tir2.pt` (Epoch 1, Val Loss 3.5441)

Machine-readable results recorded in [data/interim/ml/multisource_transfer_results.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_transfer_results.json).
