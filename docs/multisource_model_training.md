# VAYU-NET — Multi-Source Satellite Model Training & Governance Report

**Module:** Multi-Source Deep Learning Architecture, Training Governance & Overfitting Audit  
**Baseline:** Smart India Hackathon (SIH 2026) • Problem Statement SIH26070  
**Repository Working Directory:** `C:\GitHub\vayu-net`  
**Date:** 2026-09-25  
**Audit Status:** COMPLETE — FAIR THREE-MODEL EXPERIMENT TRAINED & EVALUATED  

---

## 1. Executive Summary & Experiment Purpose

This milestone executes the empirical training of multi-source deep learning models on the frozen, verified **NOAA GridSat-B1 + ISRO INSAT-3D** paired dataset ($N = 725$ samples).

The objective is to evaluate:
> *"Does adding INSAT-3D Imager (`3DIMG_L1C_ASIA_MER`) observations provide measurable predictive value beyond the GridSat-B1 baseline?"*

To isolate the contribution of the second satellite source without confounding variables, three models were constructed and trained under strictly identical governance:
1. **Model A (GridSat Only):** 6-frame sequence of GridSat-B1 IRWIN 11µm brightness temperatures.
2. **Model B (INSAT Only):** 6-frame sequence of INSAT-3D 3-channel observations (`IMG_TIR1` 10.8µm, `IMG_TIR2` 12.0µm split-window, and `IMG_WV` 6.8µm upper tropospheric water vapor).
3. **Model C (GridSat + INSAT Fusion):** Joint multi-source sequence modeling fusing GridSat and INSAT-3D temporal latents.

---

## 2. Frozen Dataset & Strict Causal Governance

All experiments operated on the frozen paired manifest [data/manifests/vayu_net_multisource_sample_index.csv](file:///c:/GitHub/vayu-net/data/manifests/vayu_net_multisource_sample_index.csv) under the validated $\Delta t \le 60\text{ min}$ temporal tolerance:
- **Total Paired Samples:** **725**
- **TRAIN Split:** **175 samples** (24 storms, 2014–2018)
- **VALIDATION Split:** **252 samples** (14 storms, 2019–2020)
- **TEST Split:** **298 samples** (24 storms, 2021–2024)
- **Storm Disjointness:** $100\%$ split isolation with zero storm overlap.
- **Strict Anti-Leakage Invariants:** All sequence frames satisfy $t \le t_0$; zero future observation or ground-truth label enters any input pipeline.

### Feature Normalization Isolation:
- **GridSat Normalization:** TRAIN-only statistics ($\mu = 279.37\,\text{K}, \sigma = 22.79\,\text{K}$).
- **INSAT-3D Normalization:** Strictly derived from [data/interim/ml/multisource_train_normalization_stats.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_train_normalization_stats.json) on the 175 TRAIN samples:
  - `IMG_TIR1`: $\mu = 279.41\,\text{K}, \sigma = 24.26\,\text{K}$
  - `IMG_TIR2`: $\mu = 277.49\,\text{K}, \sigma = 23.58\,\text{K}$
  - `IMG_WV`: $\mu = 261.83\,\text{K}, \sigma = 15.96\,\text{K}$

---

## 3. Lightweight Model Architecture & Parameter Budgets

Because the paired training set contains only $175$ samples, deploying heavy backbone networks would cause severe overfitting. A lightweight, modular architecture was implemented in [ml/models/multisource_fusion.py](file:///c:/GitHub/vayu-net/ml/models/multisource_fusion.py):

```
                        [GridSat-B1 6-Frame Seq]           [INSAT-3D 6-Frame Seq]
                          [B, 6, 1, 72, 116]                [B, 6, 3, 72, 116]
                                  │                                  │
                                  ▼                                  ▼
                         CompactSpatialCNN                  CompactSpatialCNN
                         (Conv2d -> GELU)                   (Conv2d -> GELU)
                                  │                                  │
                                  ▼                                  ▼
                            Temporal GRU                       Temporal GRU
                             (hidden=64)                        (hidden=64)
                                  │                                  │
                          Latent A [B, 64]                   Latent B [B, 64]
                                  └───────────────┬──────────────────┘
                                                  ▼
                                          Concat & Projection
                                          (LayerNorm -> Linear -> GELU)
                                                  │
                                                  ▼
                                      Shared Latent [B, 64]
                                                  │
            ┌─────────────────────┬───────────────┴───────────────┬─────────────────────┐
            ▼                     ▼                               ▼                     ▼
     Center Localizer      Intensity Classifier             Wind Regressor        Track Heads
     [u_lat, u_lon]        7 IMD Logits                     Normalized Wind       +12h, +24h, +48h
```

### Parameter Budgets:

| Component | Model A (GridSat) | Model B (INSAT) | Model C (Fusion) |
| :--- | :---: | :---: | :---: |
| **Spatial Encoder** | 48,432 | 48,720 | 97,152 |
| **Temporal Sequence GRU** | 24,960 | 24,960 | 49,920 |
| **Fusion Projection** | 4,288 | 4,288 | 8,384 |
| **Multi-Task Heads** | 4,560 | 4,560 | 4,560 |
| **Total Parameter Count** | **82,240** | **82,528** | **151,696** |
| **Trainable Parameter Count** | **82,240** | **82,528** | **151,696** |
| **Frozen Parameter Count** | **0** | **0** | **0** |

---

## 4. Training Governance & Protocol

- **Random Seed:** $42$ (`torch.manual_seed(42)`, `np.random.seed(42)`)
- **Optimizer:** `AdamW` (learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$)
- **Batch Size:** $16$ ($11$ batches per epoch on the 175-sample TRAIN set)
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor $0.5$, patience $2$ epochs)
- **Early Stopping:** Strict patience of $5$ epochs monitoring Validation Loss
- **Checkpoint Selection:** Checkpoint selected **strictly by lowest Validation Loss**; test split evaluated **exactly once** upon final selection.

---

## 5. Training History & Overfitting Audit

Training was conducted deterministically. The full training trajectories are plotted in:  
[docs/figures/multisource_model/multisource_training_curves.png](file:///c:/GitHub/vayu-net/docs/figures/multisource_model/multisource_training_curves.png)

### Training Trajectory Table:

#### Model A (GridSat Only)
| Epoch | Train Loss | Val Loss | Val Center DPE (km) | Val Wind MAE (kt) | Val Track Agg DPE (km) | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.0743 | **3.4011** | 1,244.3 | 30.4 | 1,279.1 | 0.0010 | **BEST CHECKPOINT** |
| 2 | 1.7317 | 3.9943 | 1,169.8 | 33.0 | 1,239.5 | 0.0010 | Degraded |
| 3 | 1.5388 | 4.5304 | 1,441.7 | 34.7 | 1,267.4 | 0.0010 | Degraded |
| 4 | 1.3712 | 4.8983 | 1,499.3 | 32.5 | 1,399.3 | 0.0005 | Degraded |
| 5 | 1.2534 | 3.8246 | 1,008.2 | 27.8 | 1,011.7 | 0.0005 | Degraded |
| 6 | 1.1572 | 3.9532 | 1,008.0 | 27.8 | 1,041.0 | 0.0005 | **Early Stop Triggered** |

#### Model B (INSAT Only)
| Epoch | Train Loss | Val Loss | Val Center DPE (km) | Val Wind MAE (kt) | Val Track Agg DPE (km) | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.0306 | **3.2789** | 1,267.1 | 29.3 | 1,279.0 | 0.0010 | **BEST CHECKPOINT** |
| 2 | 1.7610 | 3.6087 | 1,190.4 | 29.7 | 1,184.9 | 0.0010 | Degraded |
| 3 | 1.5727 | 3.9930 | 1,084.7 | 31.3 | 1,094.8 | 0.0010 | Degraded |
| 4 | 1.4014 | 4.3338 | 1,124.6 | 32.5 | 1,106.1 | 0.0005 | Degraded |
| 5 | 1.2032 | 3.7430 | 1,155.8 | 27.1 | 1,283.8 | 0.0005 | Degraded |
| 6 | 1.1883 | 4.9725 | 1,165.4 | 32.9 | 1,088.7 | 0.0005 | **Early Stop Triggered** |

#### Model C (GridSat + INSAT Fusion)
| Epoch | Train Loss | Val Loss | Val Center DPE (km) | Val Wind MAE (kt) | Val Track Agg DPE (km) | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.9914 | **3.6562** | 1,234.9 | 34.7 | 1,283.0 | 0.0010 | **BEST CHECKPOINT** |
| 2 | 1.6420 | 4.7207 | 1,288.4 | 41.1 | 1,312.8 | 0.0010 | Degraded |
| 3 | 1.4321 | 5.3350 | 1,744.0 | 37.9 | 1,673.5 | 0.0010 | Degraded |
| 4 | 1.2673 | 4.8695 | 1,063.3 | 31.0 | 1,056.7 | 0.0005 | Degraded |
| 5 | 1.0405 | 4.3123 | 1,043.4 | 30.1 | 1,126.3 | 0.0005 | Degraded |
| 6 | 0.9300 | 4.6984 | 985.2 | 30.0 | 1,058.0 | 0.0005 | **Early Stop Triggered** |

### Rigorous Overfitting Diagnosis:
1. **Severe Generalization Divergence:**
   In all three models, Train Loss declined rapidly by over $44\%\text{–}53\%$ (e.g. from $1.99$ to $0.93$ in Fusion), while Validation Loss increased from $3.65$ to $4.70\text{–}5.33$.
2. **Root Cause Analysis:**
   The training set contains only **175 samples across 24 historical storms**. With only 24 independent cyclone life histories, spatiotemporal neural networks readily memorize storm-specific convective textures rather than learning generalizable meteorological dynamics.
3. **Capacity Penalty on Fusion:**
   Model C has $151,696$ parameters ($1.8\times$ Model A). Because of this increased capacity, Model C overfit faster and achieved a worse best validation loss ($3.6562$) than either GridSat alone ($3.4011$) or INSAT alone ($3.2789$).
4. **Majority-Class Collapse:**
   Across all three models, intensity classification collapsed to the majority class (`D`, Depression), producing a classification accuracy of $22.1\%$ and Macro-F1 of $0.0604$.

---

## 6. Checkpoint Registry

The following reproducible checkpoints have been saved and validated:

| Checkpoint File | Modality | Best Epoch | Val Loss | File Size | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `multisource_gridsat_model_a.pt` | GridSat Only | 1 | 3.4011 | 336 KB | Recorded in results JSON |
| `multisource_insat_model_b.pt` | INSAT Only | 1 | 3.2789 | 337 KB | Recorded in results JSON |
| `multisource_fusion_model_c.pt` | GridSat + INSAT | 1 | 3.6562 | 618 KB | Recorded in results JSON |

*Checkpoints directory:* [data/interim/ml/checkpoints/](file:///c:/GitHub/vayu-net/data/interim/ml/checkpoints/)  
*Machine-readable metrics:* [data/interim/ml/multisource_model_results.json](file:///c:/GitHub/vayu-net/data/interim/ml/multisource_model_results.json)
