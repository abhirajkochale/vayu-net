# VAYU-NET ML Dataset and Data Loading Architecture

**Document Version:** 1.0  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Architectural Overview

The VAYU-NET data loading pipeline transforms the locked multi-modal dataset (`data/manifests/vayu_net_sample_index.csv`) into structured, normalized PyTorch tensors for multi-task cyclone intensity estimation and forecasting.

```
+-----------------------------------------------------------------------------------+
|                           VAYU-NET SAMPLE INDEX (1,319)                           |
|       (TRAIN: 696 samples | VALIDATION: 252 samples | TEST: 371 samples)          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         VayuSatelliteDataset (__getitem__)                        |
|                                                                                   |
|  1. Frozen Satellite Sequence (Lazy Loading from 6 NPZ files):                    |
|     t-15h  --->  t-12h  --->  t-9h  --->  t-6h  --->  t-3h  --->  t0              |
|                                                                                   |
|  2. Satellite Missing Pixel Imputation:                                           |
|     NaN / Invalid Pixels  --> Imputed with TRAIN Mean (279.37 K)                  |
|                                                                                   |
|  3. Training-Only Z-Score Normalization:                                          |
|     Normalized Pixel = (Pixel_K - 279.3677) / 22.7905  --> (Imputed pixels = 0.0) |
|                                                                                   |
|  4. Ground-Truth Target Extraction (t0, +12h, +24h, +48h):                         |
|     - Track Centers (lat, lon)                                                    |
|     - Maximum Sustained Wind (kt) + Explicit Validity Mask                        |
|     - Central Pressure (hPa) + Explicit Validity Mask                             |
|     - IMD Category (7 classes: D..SuCS) + Explicit Validity Mask                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       PyTorch DataLoader (Batch Generation)                       |
|         Batch Shape: (B, 6, 572, 929) float32 | Deterministic Seeding             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Satellite Input Sequence Specification

For every sample, the model receives an ordered temporal tensor of shape `(6, 572, 929)` representing native 3-hourly GridSat-B1 IR brightness temperatures:
1. `frame_t_minus_15h` ($t_0 - 15\text{h}$)
2. `frame_t_minus_12h` ($t_0 - 12\text{h}$)
3. `frame_t_minus_9h` ($t_0 - 9\text{h}$)
4. `frame_t_minus_6h` ($t_0 - 6\text{h}$)
5. `frame_t_minus_3h` ($t_0 - 3\text{h}$)
6. `frame_t0` ($t_0$)

### Strict Temporal Causality
- No frame with timestamp $> t_0$ is accessible to the input.
- Frame temporal intervals are verified to be strictly $3\text{ hours}$ apart.
- Lazy loading ensures only the requested 6 frames are loaded into memory per sample.

---

## 3. Satellite Missing-Value Handling Strategy

Across the 27-year GridSat-B1 archive, missing pixels can occur due to scan boundaries or telemetry dropouts.
To avoid propagating `NaN`s into downstream convolution operations:

1. **Identification**:
   Pixels where $\text{arr} \text{ is NaN}$ or $\text{arr} \le 100\text{ K}$ or $\text{arr} \ge 380\text{ K}$ are flagged as invalid.
2. **Imputation**:
   All invalid pixels are imputed with the exact **Training Mean Brightness Temperature** ($\mu = 279.367674\text{ K}$).
3. **Standardization**:
   $$z = \frac{x - \mu}{\sigma}$$
   Because $x = \mu$ for imputed pixels, the resulting standardized value is **exactly $0.0$**.
   In deep learning feature spaces, a $0.0$ input represents a neutral, uninformative baseline that does not distort gradient calculations.
4. **Underlying Data Immutability**:
   This imputation is performed purely in-memory during tensor construction. The underlying locked `.npz` and `.nc` files on disk are **never modified**.

---

## 4. Training-Only Normalization Governance

To prevent data leakage from validation or test splits, normalization statistics were computed **strictly and exclusively** from the 696 `TRAIN` samples (1,304 unique satellite frames):

- **Baseline File**: `data/interim/ml/train_normalization_stats.json`
- **Mean ($\mu$)**: `279.3676741882571` Kelvin
- **Standard Deviation ($\sigma$)**: `22.790485281834904` Kelvin
- **Rule**: Validation and test splits **must always load and reuse** this precomputed JSON file. They are strictly prohibited from calculating their own statistics.

---

## 5. Ground-Truth Target Representation & Loss Masking

| Target Variable | Measurement Units | Tensor Dtype | Representation / Range | Missing Value Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Center Location** | Degrees Lat/Lon | `torch.float32` | `(2,)` $[\text{lat}, \text{lon}]$ | Always present ($100\%$ valid) |
| **Maximum Sustained Wind** | Knots ($\text{kt}$) | `torch.float32` | Scalar (e.g. $25.0$ to $140.0$) | If `NaN`, tensor = $0.0$, `wind_mask = 0.0`. Valid: `wind_mask = 1.0` |
| **Central Pressure** | Hectopascals ($\text{hPa}$) | `torch.float32` | Scalar (e.g. $900.0$ to $1010.0$) | If `NaN`, tensor = $0.0$, `pressure_mask = 0.0`. Valid: `pressure_mask = 1.0` |
| **Cyclone Category** | Ordinal Class | `torch.int64` | Discrete: $0 \dots 6$ | If `NaN`, tensor = $-1$, `category_mask = 0.0`. Valid: `category_mask = 1.0` |

### Category Class Mapping:
$$\mathcal{C} = \{\text{'D'}: 0, \text{'DD'}: 1, \text{'CS'}: 2, \text{'SCS'}: 3, \text{'VSCS'}: 4, \text{'ESCS'}: 5, \text{'SuCS'}: 6\}$$

### Explicit Loss Masking:
Downstream loss functions must apply the returned masks:
$$\mathcal{L}_{\text{wind}} = \frac{\sum_i \text{mask}_i \cdot \ell(y_i, \hat{y}_i)}{\sum_i \text{mask}_i}$$
This guarantees that legitimate dissipation `NaN` observations do not produce artificial zero-wind loss penalties.
