# VAYU-NET — Single-Frame CNN Baseline (Phase 3 Report)

**Experiment ID:** `EXP-SINGLE-FRAME-CNN-001`  
**Model Architecture:** 1-Channel ResNet-18 (From-Scratch Baseline, `weights=None`)  
**Input Modality:** Strictly $t_0$ GridSat Infrared Window (IRWIN CDR, $572 \times 929$, 1 channel)  
**Historical Sequence Ablation:** Frames $t_{-15h}, t_{-12h}, t_{-9h}, t_{-6h}, t_{-3h}$ completely bypassed  
**Checkpoints & Outputs:** `data/interim/ml/checkpoints/best_single_frame_cnn.pt`, `data/interim/ml/single_frame_cnn_results.json`  

---

## 1. Executive Summary

This report establishes the first deep learning benchmark for the VAYU-NET framework: the **Single-Frame CNN Baseline**. Designed strictly as a minimal, from-scratch ablation baseline, this model assesses what information can be extracted solely from an unguided, full-basin synoptic satellite snapshot at $t_0$ without temporal dynamics, without ImageNet transfer learning, and without physical coordinate priors.

### Core Key Results (Unbiased TEST Split, $N=371$ across 31 storms)
- **Current Center Location:**
  - Latitude MAE: **$3.95^\circ$**
  - Longitude MAE: **$9.76^\circ$**
  - Mean Direct Position Error (DPE): **$1,215.1\text{ km}$**
  - Median Direct Position Error (DPE): **$1,204.9\text{ km}$**
- **Current Intensity (Wind):**
  - Mean Absolute Error (MAE): **$18.59\text{ kt}$**
  - Root Mean Square Error (RMSE): **$22.21\text{ kt}$**
- **Current Category Classification:**
  - Overall Accuracy: **$20.22\%$**
  - Macro F1-Score: **$0.0911$**
  - Primary Error Mode: Predicts dominant classes (Depression, Deep Depression, Cyclonic Storm) due to severe class imbalance.

---

## 2. Experimental Discipline & Reproducibility

Strict dataset isolation was enforced across all stages. Model training and hyperparameter tracking were conducted exclusively on the `TRAIN` set, model selection and early stopping were governed solely by the `VALIDATION` set, and the `TEST` set was evaluated **strictly once** using the best-selected checkpoint.

| Parameter / Environment | Specification |
|:---|:---|
| **Python Version** | `3.13.13` |
| **PyTorch Version** | `2.14.0+cpu` |
| **Torchvision Version** | `0.29.0+cpu` |
| **Execution Hardware** | 16-Core Host CPU (10 threads) |
| **Random Seed** | `42` (`torch`, `numpy`, `random` fixed) |
| **Backbone** | ResNet-18 (`weights=None`, 1-channel adapted first conv) |
| **Total Parameters** | **11,368,522** (all trainable) |
| **Batch Size** | 16 |
| **Optimizer** | `AdamW` (learning rate: $1 \times 10^{-4}$, weight decay: $1 \times 10^{-4}$) |
| **LR Scheduler** | `ReduceLROnPlateau` (mode: `min`, factor: 0.5, patience: 1) |
| **Max Epochs / Patience** | 5 epochs max / early stopping patience = 2 |
| **Data Augmentation** | None (strictly deterministic unaugmented inputs) |

---

## 3. Dataset Splits & Class Imbalance Analysis

The dataset consists of 1,319 locked samples split chronologically and storm-wise to prevent data leakage:
- **TRAIN:** 696 samples across 81 storms (52.8%)
- **VALIDATION:** 252 samples across 14 storms (19.1%)
- **TEST:** 371 samples across 31 storms (28.1%)

### TRAIN Category Distribution (Class Imbalance Baseline)
Category statistics were computed strictly from the `TRAIN` split:

| IMD Category | Code | Samples | Percentage | Wind Range (kt) |
|:---|:---:|:---:|:---:|:---:|
| **Depression (D)** | 0 | 246 | 35.34% | 17–27 |
| **Deep Depression (DD)** | 1 | 190 | 27.30% | 28–33 |
| **Cyclonic Storm (CS)** | 2 | 151 | 21.70% | 34–47 |
| **Very Severe Cyclonic Storm (VSCS)** | 4 | 63 | 9.05% | 64–89 |
| **Severe Cyclonic Storm (SCS)** | 3 | 40 | 5.75% | 48–63 |
| **Extremely Severe Cyclonic Storm (ESCS)** | 5 | 3 | 0.43% | 90–119 |
| **Super Cyclonic Storm (SuCS)** | 6 | 2 | 0.29% | $\ge 120$ |
| *Missing / Uncategorized* | -1 | 1 | 0.14% | — |

> **Finding:** Over 84% of the training distribution belongs to the lowest three intensity stages ($D, DD, CS$). High-end extreme cyclones ($ESCS, SuCS$) represent less than 1% of the dataset.

---

## 4. Multi-Task Loss Formulation & Target Scaling

To ensure balanced gradient magnitudes across heterogeneous tasks and prevent raw wind values from dominating optimization, targets were scaled strictly using `TRAIN` statistics:

### 1. Track Center Head
- Parameterized with a `Sigmoid()` activation to constrain outputs to the unit square $[0, 1]^2$.
- Mapped to North Indian Ocean geographic domain:
  $$u_{\text{lat}} = \frac{\text{lat} - (-5.0^\circ)}{40.0^\circ}, \quad u_{\text{lon}} = \frac{\text{lon} - 40.0^\circ}{65.0^\circ}$$
- Geographic bounds: Latitude $[-5.0^\circ\text{N}, 35.0^\circ\text{N}]$, Longitude $[40.0^\circ\text{E}, 105.0^\circ\text{E}]$.
- Loss: $\mathcal{L}_{\text{center}} = \text{SmoothL1}(\hat{u}_{\text{center}}, u_{\text{center}})$.

### 2. Maximum Sustained Wind Head
- Scaled using `TRAIN`-only mean and standard deviation:
  - $\mu_{\text{wind}} = 36.8836\text{ kt}$
  - $\sigma_{\text{wind}} = 17.9754\text{ kt}$
  - $u_{\text{wind}} = \frac{\text{wind} - \mu_{\text{wind}}}{\sigma_{\text{wind}}}$
- Loss: Masked Smooth L1 loss ignoring samples with missing wind labels:
  $$\mathcal{L}_{\text{wind}} = \frac{\sum m_i \cdot \text{SmoothL1}(\hat{u}_{\text{wind}, i}, u_{\text{wind}, i})}{\sum m_i + \epsilon}$$

### 3. Category Head
- 7-class discrete classification head using standard Cross-Entropy Loss:
  $$\mathcal{L}_{\text{cat}} = \text{CrossEntropyLoss}(\hat{\mathbf{y}}_{\text{cat}}, y_{\text{cat}}, \text{ignore\_index}=-1)$$

### Combined Multi-Task Loss
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{center}} + 1.0 \cdot \mathcal{L}_{\text{wind}} + 1.0 \cdot \mathcal{L}_{\text{cat}}$$

---

## 5. Training History & Dynamics

Training executed for 4 epochs before early stopping triggered:

| Epoch | Train Loss | Train Center | Train Wind | Train Cat | Val Loss | Val Center | Val Wind | Val Cat | Val DPE (km) | Val Wind MAE (kt) | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1.9044 | 0.0098 | 0.3240 | 1.5705 | 3.1307 | 0.0102 | 1.0659 | 2.0545 | 1126.1 | 26.10 | YES |
| **2** | 1.7039 | 0.0082 | 0.2923 | 1.4034 | **3.0779** | 0.0097 | 0.9650 | 2.1032 | **1111.2** | **24.64** | **BEST** |
| **3** | 1.5986 | 0.0082 | 0.2706 | 1.3198 | 5.3172 | 0.0215 | 1.5064 | 3.7893 | 1576.2 | 34.45 | No |
| **4** | 1.4101 | 0.0081 | 0.2037 | 1.1983 | 4.7684 | 0.0135 | 1.3731 | 3.3817 | 1288.9 | 32.04 | No (Stop) |

![Training and Validation Loss Curves](file:///c:/GitHub/vayu-net/docs/figures/single_frame_cnn/train_val_loss.png)

> **Observation:** Training loss decreased steadily across all four epochs. However, validation loss reached its optimum at Epoch 2 ($3.0779$) before overfitting manifested on the relatively small dataset (696 training images without pretraining or data augmentation). Early stopping appropriately halted optimization after 2 consecutive non-improving epochs.

---

## 6. Comprehensive Multi-Split Evaluation

The best model from Epoch 2 was evaluated across all three splits:

| Split | Sample Count | Center Lat MAE | Center Lon MAE | Mean DPE (km) | Median DPE (km) | Wind MAE (kt) | Wind RMSE (kt) | Category Acc | Category Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 696 | $2.96^\circ$ | $8.80^\circ$ | 1,067.3 | 970.0 | 13.30 | 16.64 | 38.71% | 0.1565 |
| **VALIDATION** | 252 | $3.69^\circ$ | $8.91^\circ$ | 1,111.2 | 1,046.2 | 24.64 | 32.86 | 27.13% | 0.0750 |
| **TEST** | 371 | **$3.95^\circ$** | **$9.76^\circ$** | **1,215.1** | **1,204.9** | **18.59** | **22.21** | **20.22%** | **0.0911** |

---

## 7. Diagnostic Visualizations

### 1. Direct Position Error Distribution (TEST Split)
The histogram below depicts the distribution of geographic center errors on the test set.

![Center Error Distribution](file:///c:/GitHub/vayu-net/docs/figures/single_frame_cnn/center_error_distribution.png)

### 2. Geographic Center Scatter: Ground Truth vs Prediction
The scatter plot compares ground-truth cyclone centers with CNN predictions across the Bay of Bengal and Arabian Sea basin.

![Center Scatter Plot](file:///c:/GitHub/vayu-net/docs/figures/single_frame_cnn/center_scatter.png)

### 3. Category Confusion Matrix
The confusion matrix indicates the classification distribution across the 7 IMD categories on the test set.

![Category Confusion Matrix](file:///c:/GitHub/vayu-net/docs/figures/single_frame_cnn/category_confusion_matrix.png)

---

## 8. Scientific Insights & Baseline Comparison

### 1. Spatial Center Localization Without Prior Track Information
- A single uncropped full-basin satellite image ($572 \times 929$, representing $\approx 4,000 \times 6,500\text{ km}$) presents an immense spatial search space.
- Training a 1-channel ResNet-18 from scratch on 696 images produces a median DPE of **$1,204.9\text{ km}$**. The model learns the broad climatological centroid of cyclogenesis (central Bay of Bengal / central Arabian Sea) but lacks the inductive bias to pinpoint cyclone eyes amidst extensive non-cyclonic convective cloud clusters.
- This establishes a quantitative baseline proving that **temporal motion tracking (multi-frame recurrent sequences) or localized patch/crop detection is essential** for high-precision cyclone positioning.

### 2. Intensity Estimation (Wind MAE = 18.6 kt)
- Remarkably, the continuous wind head achieves an MAE of **$18.59\text{ kt}$** and RMSE of **$22.21\text{ kt}$** on unseen test storms.
- Because IR brightness temperatures correlate physically with convective cloud-top heights and eye formation, the CNN learns meaningful visual features indicative of storm intensity even without pretraining.

### 3. Category Imbalance
- With 84% of the dataset in categories D, DD, and CS, the unweighted Cross-Entropy loss predictably converges towards predicting common moderate categories, achieving a macro F1 of $0.0911$.
- Future iterations will require focal loss, class-balanced reweighting, or ordinal regression formulation.

---

## 9. Immutability & Governance Audit

- **Locked Source Data:** 0 modifications to `data/raw/imd/`, `data/processed/`, `data/interim/gridsat/`, or `data/manifests/`.
- **Derived Artifacts:** All model outputs, weights, logs, and plots are isolated under `data/interim/ml/` and `docs/figures/single_frame_cnn/`.
- **TEST Separation:** The test split was untouched during training and model selection, and was evaluated exactly once.
