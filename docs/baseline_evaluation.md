# VAYU-NET Deterministic Baseline Forecasting Evaluation

**Document Version:** 1.0  
**Evaluation Date:** 2026-09-24  
**Scope:** Rigorous evaluation of non-ML persistence baselines across the locked VAYU-NET multi-modal dataset.  
**Primary Benchmark:** TEST Split (371 samples across 31 independent cyclones, 2021–2024).  
**Secondary Split:** VALIDATION Split (252 samples across 14 independent cyclones, 2019–2020).  

---

## 1. Executive Summary

To establish scientifically valid benchmarks for future deep learning architectures (e.g. spatio-temporal neural networks, vision transformers, and multi-modal fusion heads), three deterministic, non-ML forecasting baselines were implemented and evaluated:
1. **Stationary Position Persistence**: Forecast assumes the cyclone remains fixed at its current location ($p_{t_0}$).
2. **Constant-Velocity Track Extrapolation**: Forecast projects the historical 3-hour displacement forward linearly at constant velocity.
3. **Intensity Persistence**: Forecast assumes current maximum sustained wind, central pressure, and cyclonic category remain constant over the forecast horizon.

All evaluations strictly enforce zero temporal leakage: predictions are computed using solely information available at or prior to $t_0$, without accessing future satellite frames or future IMD observations.

---

## 2. Baseline Mathematical Definitions

### A. Stationary Position Persistence
Assumes zero translation velocity between $t_0$ and the forecast horizon $t_0 + H$:
$$\hat{\mathbf{x}}(t_0 + H) = \mathbf{x}(t_0) = [\text{lat}(t_0), \text{lon}(t_0)] \quad \forall H \in \{12, 24, 48\}\text{ hours}$$

### B. Constant-Velocity Track Extrapolation
Estimates translational velocity vector $\mathbf{v} = [v_{\text{lat}}, v_{\text{lon}}]$ from the most recent synoptic observations prior to or at $t_0$:
$$\mathbf{v} = \frac{\mathbf{x}(t_0) - \mathbf{x}(t_{\text{prev}})}{\Delta t}$$
where $t_{\text{prev}}$ is the latest known IMD observation strictly prior to $t_0$ (native $\Delta t = 3\text{h}$ when available; $\Delta t = 6\text{h}$ for older synoptic records; $\mathbf{v} = \mathbf{0}$ if $t_0$ is the cyclone's initial detection observation).

Linear extrapolation over horizon $H \in \{12, 24, 48\}\text{ hours}$ is computed as:
$$\hat{\mathbf{x}}(t_0 + H) = [\text{lat}(t_0) + H \cdot v_{\text{lat}}, \quad \text{lon}(t_0) + H \cdot v_{\text{lon}}]$$

### C. Intensity Persistence
Assumes meteorological stationarity across intensity metrics:
$$\hat{w}(t_0 + H) = w(t_0) \quad (\text{Maximum Sustained Wind in knots})$$
$$\hat{p}(t_0 + H) = p(t_0) \quad (\text{Central Pressure in hPa})$$
$$\hat{c}(t_0 + H) = c(t_0) \quad (\text{IMD Cyclonic Category Class } 0 \dots 6)$$

---

## 3. Evaluation Metrics & Formulations

### Track Error: Direct Position Error (DPE)
Computed as the great-circle distance between the predicted coordinate $(\hat{\phi}, \hat{\lambda})$ and ground-truth IMD coordinate $(\phi, \lambda)$ using the Haversine formula:
$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi)\cos(\hat{\phi})\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
$$d = 2 R \cdot \arctan2(\sqrt{a}, \sqrt{1 - a})$$
where $R = 6371.0\text{ km}$.

Reported statistical aggregates:
- **Mean DPE**: $\frac{1}{N}\sum_{i=1}^N d_i$
- **Median DPE**: 50th percentile of $\{d_i\}$
- **RMSE DPE**: $\sqrt{\frac{1}{N}\sum_{i=1}^N d_i^2}$
- **90th Percentile DPE**: 90th percentile of $\{d_i\}$
- **Maximum DPE**: $\max(\{d_i\})$
- **Coordinate MAE**: $\text{MAE}_{\text{lat}} = \frac{1}{N}\sum |\hat{\phi}_i - \phi_i|$, $\text{MAE}_{\text{lon}} = \frac{1}{N}\sum |\hat{\lambda}_i - \lambda_i|$

### Intensity Error Metrics
- **Wind Speed**: Mean Absolute Error ($\text{MAE} = \frac{1}{N}\sum |w - \hat{w}|$), Root Mean Square Error ($\text{RMSE} = \sqrt{\frac{1}{N}\sum (w - \hat{w})^2}$)
- **Central Pressure**: Mean Absolute Error ($\text{MAE} = \frac{1}{N}\sum |p - \hat{p}|$), RMSE
- **Category Classification**: Overall Accuracy, Macro-averaged F1 Score ($\text{Macro-F1} = \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} F1_c$), and 7-class Confusion Matrix

---

## 4. Test Split Evaluation Results (Primary Benchmark)

**Test Set Coverage:** 371 samples across 31 distinct cyclones (2021–2024).

### 4.1 Track Forecasting Performance Comparison

| Forecast Horizon | Track Baseline Model | Mean DPE (km) | Median DPE (km) | RMSE DPE (km) | P90 DPE (km) | Max DPE (km) | Lat MAE (°) | Lon MAE (°) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+12h** | **Stationary Position** | 132.67 | 130.07 | 145.43 | 211.70 | 388.66 | 0.81° | 0.81° |
| **+12h** | **Constant Velocity (All)** | **71.28** | **58.24** | **88.63** | **133.87** | 438.98 | **0.42°** | **0.45°** |
| **+12h** | *Constant Velocity (Strict $t_{-3\text{h}}$)* | 68.42 | 56.63 | 85.06 | 126.93 | 438.98 | 0.40° | 0.43° |
| **+24h** | **Stationary Position** | 263.37 | 266.16 | 282.72 | 398.41 | 667.69 | 1.62° | 1.63° |
| **+24h** | **Constant Velocity (All)** | **150.74** | **124.09** | **184.64** | **285.91** | 880.62 | **0.88°** | **0.97°** |
| **+24h** | *Constant Velocity (Strict $t_{-3\text{h}}$)* | 143.20 | 119.12 | 177.46 | 264.45 | 880.62 | 0.83° | 0.92° |
| **+48h** | **Stationary Position** | 530.81 | 534.83 | 566.99 | 785.09 | 1311.89 | 3.23° | 3.39° |
| **+48h** | **Constant Velocity (All)** | **345.15** | **294.97** | **415.34** | **625.58** | 1945.10 | **1.96°** | **2.28°** |
| **+48h** | *Constant Velocity (Strict $t_{-3\text{h}}$)* | 327.73 | 279.28 | 403.19 | 607.15 | 1945.10 | 1.83° | 2.18° |

### 4.2 Intensity Forecasting Performance (Persistence Baseline)

| Horizon | Wind MAE (kt) | Wind RMSE (kt) | Valid / Masked Wind | Pressure MAE (hPa) | Pressure RMSE (hPa) | Category Accuracy | Category Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+12h** | 5.93 | 8.01 | 371 / 0 | 3.33 | 4.88 | 56.33% | 0.5657 |
| **+24h** | 11.43 | 15.03 | 371 / 0 | 6.44 | 9.07 | 36.93% | 0.3546 |
| **+48h** | 18.40 | 23.36 | 371 / 0 | 10.32 | 13.56 | 27.22% | 0.2332 |

---

## 5. Validation Split Performance (Secondary Reference)

**Validation Set Coverage:** 252 samples across 14 distinct cyclones (2019–2020).

| Horizon | Stationary DPE (km) | Constant-Velocity DPE (km) | Wind MAE (kt) | Pressure MAE (hPa) | Category Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **+12h** | 127.42 | 70.96 | 6.75 | 4.60 | 50.79% |
| **+24h** | 258.91 | 148.96 | 13.06 | 8.87 | 34.13% |
| **+48h** | 525.68 | 329.83 | 23.49 | 15.35 | 20.63% |

---

## 6. Per-Storm Error Distribution across 31 TEST Storms

Below is the complete per-storm breakdown for all 31 cyclones evaluated in the independent TEST split:

| Storm Identifier | Total Samples | Mean Vel DPE +12h (km) | Mean Vel DPE +24h (km) | Mean Vel DPE +48h (km) | Mean Wind MAE +24h (kt) | Category Acc +24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `NIO_2021_GULAB` | 9 | 100.9 | 200.0 | 449.6 | 9.4 kt | 55.6% |
| `NIO_2021_JAWAD` | 7 | 181.7 | 382.9 | 751.4 | 10.0 kt | 57.1% |
| `NIO_2021_SHAHEEN` | 15 | 67.8 | 140.2 | 349.5 | 11.7 kt | 26.7% |
| `NIO_2021_TAUKTAE` | 23 | 64.9 | 141.0 | 338.5 | 26.1 kt | 26.1% |
| `NIO_2021_UNNAMED_2_5_1` | 2 | 84.1 | 188.4 | 338.2 | 5.0 kt | 50.0% |
| `NIO_2021_YAAS` | 15 | 58.7 | 127.3 | 283.4 | 20.3 kt | 26.7% |
| `NIO_2022_ASANI` | 19 | 74.3 | 145.9 | 281.3 | 8.2 kt | 36.8% |
| `NIO_2022_MANDOUS` | 10 | 60.1 | 124.8 | 284.2 | 10.5 kt | 40.0% |
| `NIO_2022_NO` | 8 | 80.2 | 174.3 | 425.2 | 3.1 kt | 62.5% |
| `NIO_2022_SITRANG` | 4 | 188.8 | 362.1 | 757.7 | 8.8 kt | 25.0% |
| `NIO_2022_UNNAMED_01` | 3 | 55.4 | 111.4 | 225.8 | 5.0 kt | 66.7% |
| `NIO_2022_UNNAMED_2_13_1`| 1 | 127.3 | 254.5 | 509.0 | 0.0 kt | 100.0% |
| `NIO_2022_UNNAMED_2_15_1`| 4 | 55.7 | 111.4 | 222.8 | 2.5 kt | 75.0% |
| `NIO_2022_UNNAMED_2_16_1`| 5 | 167.1 | 334.1 | 668.2 | 0.0 kt | 100.0% |
| `NIO_2022_UNNAMED_2_2_1` | 5 | 79.8 | 159.6 | 319.2 | 3.0 kt | 60.0% |
| `NIO_2022_UNNAMED_2_9_1` | 1 | 205.5 | 410.9 | 821.9 | 0.0 kt | 100.0% |
| `NIO_2023_BIPARJOY` | 81 | 48.7 | 105.1 | 259.0 | 11.0 kt | 33.3% |
| `NIO_2023_HAMOON` | 11 | 82.5 | 172.9 | 360.7 | 13.2 kt | 36.4% |
| `NIO_2023_MICHAUNG` | 16 | 66.5 | 139.6 | 309.8 | 9.7 kt | 31.2% |
| `NIO_2023_MIDHILI` | 4 | 92.4 | 190.7 | 450.7 | 5.0 kt | 50.0% |
| `NIO_2023_MOCHA` | 20 | 73.1 | 155.0 | 361.3 | 24.0 kt | 25.0% |
| `NIO_2023_TEJ` | 16 | 93.3 | 193.9 | 437.5 | 28.1 kt | 25.0% |
| `NIO_2023_UNNAMED_2_2_1` | 6 | 60.9 | 121.9 | 243.8 | 0.0 kt | 100.0% |
| `NIO_2024_ASNA` | 28 | 51.6 | 109.6 | 281.3 | 3.0 kt | 60.7% |
| `NIO_2024_DANA` | 4 | 82.5 | 173.9 | 364.5 | 16.2 kt | 0.0% |
| `NIO_2024_FENGAL` | 20 | 80.4 | 165.0 | 344.9 | 4.5 kt | 60.0% |
| `NIO_2024_LAT` | 3 | 100.2 | 200.3 | 400.7 | 3.3 kt | 33.3% |
| `NIO_2024_NO` | 10 | 89.2 | 181.8 | 382.4 | 1.5 kt | 80.0% |
| `NIO_2024_REMAL` | 13 | 64.1 | 132.7 | 304.5 | 15.0 kt | 15.4% |
| `NIO_2024_UNNAMED_2_4_1` | 6 | 107.5 | 214.9 | 429.8 | 4.2 kt | 50.0% |
| `NIO_2024_UNNAMED_2_7_1` | 2 | 46.8 | 93.7 | 187.3 | 5.0 kt | 50.0% |

---

## 7. Diagnostic Visualizations

Diagnostic plots have been generated and saved under `docs/figures/baselines/`:

1. **Direct Position Error vs Forecast Horizon** (`docs/figures/baselines/dpe_vs_horizon.png`):
   Demonstrates linear error growth across horizons and quantifies the substantial reduction in DPE achieved by accounting for initial translational motion (Constant Velocity: $71.28\text{ km}$, $150.74\text{ km}$, $345.15\text{ km}$ vs Stationary: $132.67\text{ km}$, $263.37\text{ km}$, $530.81\text{ km}$).

2. **DPE Distribution Spread** (`docs/figures/baselines/dpe_distributions.png`):
   Depicts the error distribution percentiles, interquartile ranges, and outliers at +12h, +24h, and +48h.

3. **Historical Track Forecast Benchmark** (`docs/figures/baselines/example_track_comparison.png`):
   Illustrates actual ground truth track versus linear extrapolation and stationary persistence for Extremely Severe Cyclonic Storm BIPARJOY (2023).

---

## 8. Physical Interpretation & Baseline Limitations

1. **Short-Range vs Medium-Range Dynamics**:
   - At $+12\text{h}$, Constant Velocity track persistence achieves an average error of $71.28\text{ km}$ ($58.24\text{ km}$ median), proving that short-term inertial motion is the dominant factor in immediate track trajectory.
   - At $+48\text{h}$, mean DPE rises to $345.15\text{ km}$ ($415.34\text{ km}$ RMSE) because linear extrapolation cannot account for environmental steering flow changes, recurvature, or landfall interactions.
2. **Intensity Non-Stationarity**:
   - Intensity persistence degrades markedly at $+48\text{h}$ (Wind MAE $18.40\text{ kt}$, Category accuracy $27.22\%$), reflecting rapid intensification (RI) and post-landfall decay phases that cannot be anticipated without satellite atmospheric imagery.
3. **Role for Deep Learning**:
   - Any proposed neural network architecture for VAYU-NET must convincingly outperform these deterministic thresholds ($< 71.28\text{ km}$ at +12h, $< 150.74\text{ km}$ at +24h, $< 345.15\text{ km}$ at +48h; Wind MAE $< 5.93\text{ kt}$ at +12h, $< 11.43\text{ kt}$ at +24h, $< 18.40\text{ kt}$ at +48h) to justify model complexity.
