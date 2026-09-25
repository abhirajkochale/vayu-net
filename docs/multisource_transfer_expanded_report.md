# VAYU-NET — Multi-Source Transfer Learning on Expanded TRAIN Dataset

**Module:** Multi-Source Deep Transfer Learning & Channel Ablation Suite  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — 100% REPRODUCIBLE (Deterministic Seed = 42, Invariant In-Place Audit Passed)  

---

## 1. Executive Summary & Objective

This research experiment evaluates whether expanding the paired multi-source satellite training population from **175 samples (24 storms)** to **207 samples (25 storms, +18.29%)** improves the transfer-learning capabilities of dual-satellite architectures combining **NOAA GridSat-B1** and **ISRO INSAT-3D** (`3DIMG_L1C_ASIA_MER`).

### Strict Scientific Constraints Maintained:
- **Locked Evaluation Splits:** The evaluation set remains **strictly 298 TEST samples across 24 test storms (2021–2024)**, completely identical to all prior multi-source and GridSat baseline evaluations.
- **Validation Isolation:** The validation set remains **strictly 252 VALIDATION samples across 14 storms (2019–2020)**.
- **Normalization Derivation:** INSAT-3D channel parameters were derived **strictly on the 207 TRAIN samples across 379 unique satellite frames**, preventing data leakage.
- **System Isolation:** Production code in `apps/backend/`, `apps/frontend/`, production checkpoints, and deployment configurations remained **100% untouched**.

---

## 2. Experimental Setup & Model Architectures

Three transfer-learning configurations were evaluated under an identical training protocol:
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **LR Policy:** ReduceLROnPlateau ($\text{factor} = 0.5$, $\text{patience} = 2$)
- **Batch Size:** 16
- **Epochs:** 20 with early stopping patience of 5 epochs
- **Multi-Task Objective:** $\mathcal{L} = \mathcal{L}_{\text{center}} + \mathcal{L}_{\text{class}} + \mathcal{L}_{\text{wind}} + \mathcal{L}_{\text{track}}$

### Evaluated Configurations:
1. **EXP1 — GridSat-Only Transfer Baseline (`mode='exp1_gridsat_frozen'`):**
   - Frozen pretrained 2-layer temporal GRU encoder (132-dim features) from Phase 4A.
   - Trainable multi-task prediction heads (Center, Category, Wind, +12h/+24h/+48h Track).
   - Parameters: 267,024 Total | 67,344 Trainable | 199,680 Frozen.
2. **EXP2 — GridSat + INSAT TIR1 + TIR2 (`mode='exp2_fusion_frozen'`, $C=2$):**
   - Reuses frozen GridSat representation.
   - Fuses a lightweight 3-stage CNN + GRU auxiliary branch processing INSAT `IMG_TIR1` (10.8µm) and `IMG_TIR2` (12.0µm split-window).
   - Late feature fusion via projection to 128-dim shared latent space.
   - Parameters: 340,432 Total | 140,752 Trainable | 199,680 Frozen.
3. **EXP3 — GridSat + INSAT TIR1 + TIR2 + WV (`mode='exp2_fusion_frozen'`, $C=3$):**
   - Reuses frozen GridSat representation.
   - Fuses auxiliary branch processing all 3 INSAT channels: `IMG_TIR1`, `IMG_TIR2`, and `IMG_WV` (6.8µm upper-tropospheric water vapor).
   - Parameters: 340,576 Total | 140,896 Trainable | 199,680 Frozen.

---

## 3. Comprehensive Test Set Evaluation Results (298 Test Samples)

### Primary Comparison Table: Expanded TRAIN (N = 207)

| Metric | EXP1: GridSat Frozen | EXP2: GridSat + INSAT (TIR1+TIR2) | EXP3: GridSat + INSAT (TIR1+TIR2+WV) | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **Best Val Epoch** | Epoch 1 | Epoch 1 | Epoch 4 | EXP3 |
| **Validation Loss** | 3.5760 | 3.7232 | **3.4665** | **EXP3** |
| **Test Loss** | 2.8818 | 3.1301 | **2.7490** | **EXP3** |
| **Center Mean DPE (km)** | **954.28** | 1033.29 | 1048.57 | **EXP1** |
| **Center Median DPE (km)** | 921.69 | 1036.04 | **832.55** | **EXP3** |
| **Center P90 DPE (km)** | **1699.84** | 1759.76 | 2119.97 | **EXP1** |
| **Latitude MAE (°)** | 3.804 | 4.496 | **3.027** | **EXP3** |
| **Longitude MAE (°)** | **7.121** | 7.286 | 8.295 | **EXP1** |
| **Category Accuracy** | 23.83% | 23.49% | **28.19%** | **EXP3** |
| **Category Macro-F1** | 0.0885 | 0.0883 | **0.1671** | **EXP3** |
| **Wind MAE (kt)** | 21.12 | 21.97 | **19.26** | **EXP3** |
| **Wind RMSE (kt)** | 28.59 | 29.63 | **24.82** | **EXP3** |
| **Wind Median AE (kt)** | 14.77 | 14.86 | **14.71** | **EXP3** |
| **Wind P90 AE (kt)** | 52.64 | 53.39 | **44.37** | **EXP3** |
| **Wind Bias (kt)** | -18.20 | -19.66 | **-9.81** | **EXP3** |
| **Wind Pearson $r$** | 0.2799 | 0.2619 | **0.3128** | **EXP3** |
| **Track +12h DPE (km)** | **908.72** | 922.25 | 1014.05 | **EXP1** |
| **Track +24h DPE (km)** | 954.42 | **923.41** | 1133.88 | **EXP2** |
| **Track +48h DPE (km)** | **955.44** | 1024.97 | 1189.32 | **EXP1** |
| **Track Aggregate DPE (km)**| **939.52** | 956.88 | 1112.42 | **EXP1** |

---

## 4. Comparison with Previous Transfer-Learning Experiment (Old TRAIN N = 175)

| Metric | EXP1 (N=175) | EXP1 (N=207) | Delta | EXP2/ChB (N=175, 2ch) | EXP2 (N=207, 2ch) | Delta | EXP2 (N=175, 3ch) | EXP3 (N=207, 3ch) | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Center Mean DPE (km)** | 959.41 | 954.28 | **-5.13** | 1048.47 | 1033.29 | **-15.18** | 938.66 | 1048.57 | +109.91 |
| **Center Median DPE (km)**| 956.78 | 921.69 | **-35.09** | 963.68 | 1036.04 | +72.36 | 876.51 | 832.55 | **-43.96** |
| **Wind MAE (kt)** | 21.44 | 21.12 | **-0.32** | 20.85 | 21.97 | +1.12 | 20.93 | 19.26 | **-1.67** |
| **Wind RMSE (kt)** | 29.11 | 28.59 | **-0.52** | 27.78 | 29.63 | +1.85 | 27.82 | 24.82 | **-3.00** |
| **Wind Bias (kt)** | -19.24 | -18.20 | **+1.04** | -18.42 | -19.66 | -1.24 | -18.52 | -9.81 | **+8.71** |
| **Wind Pearson $r$** | 0.3106 | 0.2799 | -0.0307 | 0.3207 | 0.2619 | -0.0588 | 0.3082 | 0.3128 | **+0.0046** |
| **Track +12h DPE (km)** | 885.09 | 908.72 | +23.63 | 891.81 | 922.25 | +30.44 | 889.07 | 1014.05 | +124.98 |
| **Track +24h DPE (km)** | 908.76 | 954.42 | +45.66 | 890.49 | 923.41 | +32.92 | 1014.37 | 1133.88 | +119.51 |
| **Track +48h DPE (km)** | 1006.54 | 955.44 | **-51.10** | 994.98 | 1024.97 | +29.99 | 1018.55 | 1189.32 | +170.77 |
| **Track Aggregate DPE (km)**| 933.46 | 939.52 | +6.06 | 925.76 | 956.88 | +31.12 | 974.00 | 1112.42 | +138.42 |

---

## 5. Answers to Mandatory Research Questions

### A. Did expanding TRAIN from 175 to 207 samples improve generalization?
- **Yes, partially.** For the GridSat-only transfer baseline (EXP1), expanding the training set improved center localization (Median DPE reduced from 956.8 km to 921.7 km; Mean DPE from 959.4 km to 954.3 km) and wind estimation (MAE reduced from 21.44 kt to 21.12 kt).
- For the 3-channel fusion architecture (EXP3), expanding the dataset allowed the network to learn deeper representations before early stopping (best epoch improved from Epoch 1 to Epoch 4), resulting in the lowest overall test loss (2.7490) and a substantial reduction in wind error (MAE reduced from 20.93 kt to 19.26 kt; RMSE from 27.82 kt to 24.82 kt; bias reduced by nearly half).
- However, for track prediction, the expanded training set did not prevent spatial displacement errors when fusing INSAT channels.

### B. Did INSAT improve center localization?
- **No for Mean DPE, but mixed for Median DPE.**
  - EXP1 (GridSat only): Mean DPE = **954.28 km**, Median = 921.69 km, P90 = **1699.84 km**.
  - EXP2 (GridSat + TIR1/TIR2): Mean DPE = 1033.29 km, Median = 1036.04 km, P90 = 1759.76 km.
  - EXP3 (GridSat + TIR1/TIR2/WV): Mean DPE = 1048.57 km, Median = **832.55 km**, P90 = 2119.97 km.
  - While EXP3 achieved a lower median error (832.55 km), it experienced significant tail divergence on difficult storms (P90 exceeding 2,100 km), degrading the mean DPE.

### C. Did INSAT improve wind regression?
- **Yes, decisively for the 3-channel model (EXP3).**
  - EXP3 achieved **19.26 kt MAE** and **24.82 kt RMSE**, representing the best intensity performance observed across all multi-source experiments.
  - Wind bias decreased sharply from -18.20 kt (EXP1) down to **-9.81 kt** (EXP3).
  - Upper-tropospheric water vapor (WV 6.8µm) captures convective cloud-top pumping and radial outflow channels, providing physical signal that complements thermal IR window channels.

### D. Did INSAT improve +12/+24/+48 track prediction?
- **No.** Track forecasting performance consistently favored the GridSat-only representation:
  - EXP1 (GridSat): Aggregate DPE = **939.52 km** (+12h: 908.7 km, +24h: 954.4 km, +48h: 955.4 km).
  - EXP2 (TIR1+TIR2): Aggregate DPE = **956.88 km** (+12h: 922.3 km, +24h: **923.4 km**, +48h: 1025.0 km).
  - EXP3 (3-channel): Aggregate DPE = **1112.42 km** (+12h: 1014.1 km, +24h: 1133.9 km, +48h: 1189.3 km).
  - The auxiliary INSAT branch lacks the deep historical pretraining of the 1,319-sample GridSat GRU, leading to error accumulation over long multi-horizon rollouts.

### E. Does TIR1+TIR2 outperform the 3-channel version?
- **Yes for Track and Center Mean DPE; No for Wind and Classification.**
  - **Track:** EXP2 (TIR1+TIR2) achieved 956.88 km Aggregate DPE vs. 1112.42 km for EXP3 (+155.5 km advantage for 2-channel).
  - **Center:** EXP2 achieved lower Mean DPE (1033.29 km vs. 1048.57 km) and lower P90 error (1759.76 km vs. 2119.97 km).
  - **Wind:** EXP3 (3-channel) decisively outperformed EXP2 (19.26 kt vs. 21.97 kt MAE, 24.82 kt vs. 29.63 kt RMSE).
  - The WV channel introduces higher spatial variability that disrupts geometric trajectory heads but provides thermodynamic information for intensity estimation.

### F. Is there now enough evidence to consider INSAT for the production architecture?
- **NO.** Although expanding the training set to 207 samples reduced wind MAE to 19.26 kt in EXP3, all multi-source transfer models remain severely uncompetitive compared to VAYU-NET's existing production models:
  - **Production Center Localization CNN:** **106.3 km Mean DPE** (vs. 954.3 km for EXP1, 1048.6 km for EXP3).
  - **Production Temporal Track GRU:** **168.3 km Mean DPE** (vs. 939.5 km for EXP1, 1112.4 km for EXP3).
  - **Production Phase 5B Hybrid Kinematic Track:** **142.1 km Mean DPE**.
  - **Production Phase 6 Intensity Model:** **11.2 kt Wind MAE** (vs. 19.3 kt for EXP3).
- Deploying the current multisource transfer architecture to production would degrade center and track accuracy by ~800 km and intensity accuracy by ~8 kt.
- **Strategic Recommendation:** Continued research only. The next phase must explore cross-attention spatial fusion or pretraining the multi-source encoder across the entire INSAT archive before attempting production integration.

---

## 6. Generated Artifacts & File Inventory

| Artifact Name | Path | Description |
| :--- | :--- | :--- |
| **EXP1 Model Checkpoint** | [multisource_transfer_exp1_expanded.pt](file:///c:/GitHub/vayu-net/data/interim/ml/checkpoints/multisource_transfer_exp1_expanded.pt) | Best weights for GridSat-only transfer baseline (N=207) |
| **EXP2 Model Checkpoint** | [multisource_transfer_exp2_expanded.pt](file:///c:/GitHub/vayu-net/data/interim/ml/checkpoints/multisource_transfer_exp2_expanded.pt) | Best weights for GridSat + INSAT TIR1+TIR2 (N=207) |
| **EXP3 Model Checkpoint** | [multisource_transfer_exp3_expanded.pt](file:///c:/GitHub/vayu-net/data/interim/ml/checkpoints/multisource_transfer_exp3_expanded.pt) | Best weights for GridSat + INSAT TIR1+TIR2+WV (N=207) |
| **Expanded Transfer Cache** | `data/interim/ml/cache/multisource_transfer_cache_expanded.pt` | Unified multimodal cache for all 757 samples |
| **Results Summary JSON** | [multisource_transfer_expanded_results.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_transfer_expanded_results.json) | Complete numerical metrics, training history, and metadata |
| **Comparison CSV** | [multisource_transfer_expanded_comparison.csv](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_transfer_expanded_comparison.csv) | Tabular metric comparison across EXP1, EXP2, EXP3 |
| **Training Curves Plot** | [transfer_training_curves_expanded.png](file:///c:/GitHub/vayu-net/docs/figures/multisource_transfer_expanded/transfer_training_curves_expanded.png) | 4-panel visual comparison of Loss, Center DPE, Wind MAE, Track DPE |
| **Channel Ablation Report** | [multisource_transfer_expanded_channel_ablation.md](file:///c:/GitHub/vayu-net/docs/multisource_transfer_expanded_channel_ablation.md) | Dedicated analysis of 2-channel vs. 3-channel INSAT contributions |
