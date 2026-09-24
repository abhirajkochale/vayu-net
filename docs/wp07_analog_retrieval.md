# VAYU-NET WP-07: Historical Analog-Storm Retrieval

**Module:** Work Package 07 (WP-07) — 7-Dimensional Nearest-Neighbor Analog Cyclone Retrieval  
**Specification Baseline:** SIH 2026 Problem Statement 26070 — Blueprint / SRS V1.1  
**Candidate Archive:** Locked `TRAIN` partition ($N = 696$ candidate observations across 81 historical storms, 1998–2018)  
**Distance Metric:** Euclidean distance in TRAIN-standardized feature space  
**Return Cardinality:** Exactly $k = 2$ distinct historical analog storms  
**Status:** **PRODUCT-READY & FROZEN**  

---

## 1. Executive Summary

Analog storm retrieval provides meteorological decision-makers with historical situational context by identifying past tropical cyclones in the North Indian Ocean (NIO) that exhibited the most similar spatial location, intensity, core pressure, and 12-hour motion/intensification trajectory.

### Key Operational Invariants:
1. **Explicit UI Labeling:** Every retrieved match is labeled strictly as:
   `"label": "HISTORICAL ANALOG"`
   The engine carries an explicit system disclaimer that historical analogs serve solely as situational context and **never as a direct numerical forecast source**.
2. **Deterministic Top-2 Retrieval ($k=2$):** Returns exactly two analog storms ranked by ascending standardized Euclidean distance.
3. **Self-Match & Same-Storm Protection:** Prevents query cases from matching against themselves or against subsequent observations of the same storm event.
4. **Zero Test-Set Standardization Contamination:** Standardization means and standard deviations are derived exclusively from the historical TRAIN archive ($N=696$).

---

## 2. 7-Dimensional Feature Vector Specification

For any selected synoptic snapshot at reference time $t_0$, the analog retrieval vector $\mathbf{x} \in \mathbb{R}^7$ is defined strictly from causal information ($t \le t_0$):

$$\mathbf{x} = \begin{bmatrix} \text{lat} & \text{lon} & \text{wind\_kt} & \text{pressure\_hpa} & \Delta x_{12\text{h}} & \Delta y_{12\text{h}} & \Delta \text{wind}_{12\text{h}} \end{bmatrix}^T$$

| Dimension | Feature Name | Unit | Meteorological Definition | Physical Significance |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `lat` | degrees N | Cyclone center latitude at $t_0$ | Geographic latitude / Coriolis forcing |
| 2 | `lon` | degrees E | Cyclone center longitude at $t_0$ | Geographic basin location (Arabian Sea vs Bay of Bengal) |
| 3 | `wind_kt` | knots | Maximum sustained wind speed at $t_0$ | Kinetic intensity tier |
| 4 | `pressure_hpa` | hPa | Central surface pressure at $t_0$ | Thermodynamic central depression depth |
| 5 | `dx_12h` | degrees | Past 12-hour zonal displacement: $\lambda(t_0) - \lambda(t_0 - 12\text{h})$ | Zonal translational speed & direction |
| 6 | `dy_12h` | degrees | Past 12-hour meridional displacement: $\phi(t_0) - \phi(t_0 - 12\text{h})$ | Meridional translational speed & poleward recurvature |
| 7 | `dwind_12h` | knots | Past 12-hour intensification rate: $w(t_0) - w(t_0 - 12\text{h})$ | Rapid intensification vs steady state vs decay |

---

## 3. Standardization & Distance Metric

Because meteorological features operate on vastly different numerical scales (pressure $\approx 1000\text{ hPa}$, displacements $\approx 1^\circ$), raw Euclidean distance would be dominated by pressure and wind.

### 3.1 Training Archive Standardization Statistics ($N = 696$)
Features are standardized using parameters derived exclusively from the locked TRAIN partition:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}, \quad j \in \{1, \dots, 7\}$$

| Feature | TRAIN Mean ($\mu$) | TRAIN Std ($\sigma$) | Range in Archive |
| :--- | :---: | :---: | :---: |
| `lat` | **$14.49^\circ\text{N}$** | **$4.70^\circ$** | $5.5^\circ$ to $24.0^\circ\text{N}$ |
| `lon` | **$80.74^\circ\text{E}$** | **$11.41^\circ$** | $52.0^\circ$ to $98.5^\circ\text{E}$ |
| `wind_kt` | **$36.88\text{ kt}$** | **$17.97\text{ kt}$** | $15.0$ to $127.0\text{ kt}$ |
| `pressure_hpa` | **$993.73\text{ hPa}$** | **$10.98\text{ hPa}$** | $920.0$ to $1006.0\text{ hPa}$ |
| `dx_12h` | **$-0.73^\circ$** | **$0.93^\circ$** | $-6.4^\circ$ to $+11.8^\circ$ |
| `dy_12h` | **$+0.52^\circ$** | **$0.72^\circ$** | $-2.0^\circ$ to $+9.0^\circ$ |
| `dwind_12h` | **$+4.76\text{ kt}$** | **$8.37\text{ kt}$** | $-100.0$ to $+50.0\text{ kt}$ |

### 3.2 Distance Metric
For query $\mathbf{z}_q$ and historical candidate $\mathbf{z}_c$:
$$d(q, c) = \sqrt{\sum_{j=1}^7 (z_{q, j} - z_{c, j})^2}$$

---

## 4. Self-Match Protection & Diversity Enforcement

In an operational deployment, retrieving an analog from the exact same storm (e.g. 3 hours prior) provides no historical analog value.

The retrieval engine ([`ml/forecast/analog_retrieval.py`](file:///c:/GitHub/vayu-net/ml/forecast/analog_retrieval.py)) strictly enforces:
1. `candidate.sample_id != query.sample_id` (Prevents identity self-match).
2. `candidate.storm_id != query.storm_id` (Prevents matching against the query storm's own trajectory).
3. `candidate_1.storm_id != candidate_2.storm_id` (Guarantees that Analog 1 and Analog 2 come from two distinct historical cyclone events).

---

## 5. Reference Demo Case Validation

Running analog retrieval on the five locked VAYU-NET reference demo storms produces verified, meteorologically sound historical analogs:

| Query Storm | Reference Snapshot $t_0$ | Query Location | Query Wind | Analog 1 (Rank 1) | Distance 1 | Analog 2 (Rank 2) | Distance 2 |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :---: |
| **FANI (2019)** | `2019-04-29 12:00Z` | $8.7^\circ\text{N}, 86.8^\circ\text{E}$ | 65 kt | **MADI (2013)** | $1.02$ | **WARD (2009)** | $1.12$ |
| **AMPHAN (2020)** | `2020-05-17 00:00Z` | $11.5^\circ\text{N}, 86.0^\circ\text{E}$ | 55 kt | **NARGIS (2008)** | $0.68$ | **MADI (2013)** | $0.78$ |
| **TAUKTAE (2021)** | `2021-05-15 00:00Z` | $12.3^\circ\text{N}, 72.8^\circ\text{E}$ | 45 kt | **NIO_2007_ARB** | $1.13$ | **NARGIS (2008)** | $1.39$ |
| **BIPARJOY (2023)**| `2023-06-11 00:00Z` | $18.6^\circ\text{N}, 67.7^\circ\text{E}$ | 90 kt | **MEKUNU (2018)** | $2.04$ | **NIO_2007_ARB** | $2.53$ |
| **REMAL (2024)** | `2024-05-25 00:00Z` | $17.6^\circ\text{N}, 89.6^\circ\text{E}$ | 35 kt | **NIO_2007_BOB** | $1.02$ | **KYANT (2016)** | $1.22$ |

*Key Meteorological Realism:*
- **AMPHAN (2020) matched NARGIS (2008):** Both were catastrophic pre-monsoon May severe cyclones originating in the south-central Bay of Bengal.
- **BIPARJOY (2023) matched MEKUNU (2018):** Both were extremely severe Arabian Sea cyclones with prolonged northwestward tracks.

---

## 6. Product Output Payload Schema

Exposed via `GET /api/cyclones/{id}/analogs`:
```json
{
  "selected_case": {
    "sample_id": "NIO_2021_TAUKTAE_20210515_0000Z",
    "storm_id": "NIO_2021_TAUKTAE",
    "raw_features": {
      "lat": 12.3,
      "lon": 72.8,
      "wind_kt": 45.0,
      "pressure_hpa": 990.0,
      "dx_12h": -0.4,
      "dy_12h": 1.2,
      "dwind_12h": 10.0
    }
  },
  "features_used": ["lat", "lon", "wind_kt", "pressure_hpa", "dx_12h", "dy_12h", "dwind_12h"],
  "distance_metric": "Standardized Euclidean Distance (TRAIN baseline)",
  "k": 2,
  "label": "HISTORICAL ANALOG",
  "warning": "Historical analogs provide situational context only and are not a direct forecast source.",
  "analogs": [
    {
      "rank": 1,
      "storm_name": "NIO_2007_UNNAMED_2_3_1",
      "year": 2007,
      "season": "Post-monsoon",
      "storm_id": "NIO_2007_UNNAMED_2_3_1",
      "sample_id": "NIO_2007_UNNAMED_2_3_1_20070603_1200Z",
      "standardized_distance": 1.1301,
      "matched_snapshot": {
        "timestamp_utc": "2007-06-03T12:00:00+00:00",
        "category": "VSCS",
        "latitude": 13.0,
        "longitude": 70.5,
        "wind_kt": 65.0,
        "pressure_hpa": 980.0
      },
      "feature_snapshot": { ... },
      "label": "HISTORICAL ANALOG"
    },
    {
      "rank": 2,
      "storm_name": "NARGIS",
      "year": 2008,
      "season": "Pre-monsoon",
      "storm_id": "NIO_2008_NARGIS",
      "sample_id": "NIO_2008_NARGIS_20080428_1200Z",
      "standardized_distance": 1.3898,
      "matched_snapshot": { ... },
      "label": "HISTORICAL ANALOG"
    }
  ]
}
```

---

## 7. Limitations

1. **Equal Weighting:** Euclidean distance treats all 7 standardized dimensions equally. While meteorologically sound, weighting displacement $\Delta x, \Delta y$ higher than central pressure could alter the ranking for rapidly recurving storms.
2. **Candidate Pool Finite Horizon:** Candidates are drawn from the historical archive (1998–2018). As additional years are verified, expanding the candidate pool will yield even closer geographic analogs.
