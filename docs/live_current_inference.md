# VAYU-NET — Live & Current Cyclone Operational Inference

## 1. System Architecture Overview

VAYU-NET's operational current-event inference architecture connects external observation and cyclone discovery feeds to the production multimodal deep learning pipeline through a provider-neutral adapter layer.

```
+-----------------------------------------------------------------------------------+
|                            EXTERNAL OBSERVATION LAYERS                            |
+-----------------------------------------------------------------------------------+
|  A. Event Discovery Provider    B. Satellite Ingestion Provider    C. Track Provider  |
|  - Active cyclone detection     - NOAA AWS S3 / MOSDAC / Local   - Causal synoptic |
|  - Official advisory / bulletin - 3-hourly NetCDF-4 IRWIN          fixes (t <= t0)|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        PROVIDER-NEUTRAL ADAPTER LAYER                             |
|                  (apps/backend/services/live_source_service.py)                   |
|  - BaseEventDiscoveryProvider: discover_active_events(), get_latest_observation() |
|  - BaseSatelliteProvider: acquire_satellite_frame(), get_available_timestamps()    |
|  - BaseTrackHistoryProvider: get_recent_track_history()                           |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        CAUSAL SIX-FRAME & MOTION BUILDER                          |
|               (satellite_ingestion_service.py & current_event_service.py)         |
|  - Strictly causal window: [t-15h, t-12h, t-9h, t-6h, t-3h, t0]                  |
|  - Crop to NIO bounding box (572 x 929) at 0.07° resolution                       |
|  - Impute invalid brightness temperature (<100K or >380K) using training mean     |
|  - Normalize with training split constants: mean=268.34 K, std=28.16 K            |
|  - Derive velocity vector [v_lat, v_lon] from >= 2 observed fixes                 |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                               VAYU-NET ML INFERENCE                               |
|                     (apps/backend/services/model_service.py)                      |
|  1. Phase 3C: Center Localization CNN (ResNet-18 + 2D soft-argmax)                |
|  2. Phase 6: Multi-Task Intensity Classification & 1-min Max Wind Regression      |
|  3. Phase 5B: Variant A Hybrid Residual Track Forecast (+12h, +24h, +48h)        |
|  4. Uncertainty: Conformal empirical quantile cones (p80, p90)                    |
|  5. Explainability: Phase 3C Grad-CAM saliency maps                               |
|  6. Analogs: 7D standardized nearest-neighbor retrieval                          |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                                 FORECAST DELIVERY                                 |
|  - Verification: UNAVAILABLE (Quarantined; ground truth does not exist at t0)     |
|  - Secondary Context: IMERG precipitation layer (visual only, 0% predictive role)|
|  - Canonical REST Contract: POST /api/inference/current                           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Distinguishing Live Current Event vs. Historical Replay

| Dimension | Live Current Event (`CURRENT EVENT`) | Historical Replay (`HISTORICAL`) |
| :--- | :--- | :--- |
| **Primary Goal** | Operational real-time cyclogenesis & forecasting | Rehearsal, model validation, and retrospective audit |
| **Observation Timestamp $t_0$** | Live operational UTC timestamp or latest advisory | Pre-indexed historical observation timestamp |
| **Causal Boundary** | Strictly $t \le t_0$; future ground truth does not exist | Strictly $t \le t_0$; future ground truth exists but is quarantined |
| **Satellite Sequence** | Causal 6 frames dynamically queried & downloaded | Preprocessed local archive frames from manifest |
| **Track History** | Real causal advisory fixes; requires $\ge 2$ fixes | Best-track records filtered strictly to $t \le t_0$ |
| **Verification** | **UNAVAILABLE** (`"status": "UNAVAILABLE"`) | **AVAILABLE** (calculated against future IMD ground truth) |
| **API Route** | `POST /api/inference/current` | `POST /api/inference/run` |

---

## 3. Operational Readiness States

Each active cyclone is continuously evaluated across two independent operational dimensions:
1. **Satellite Sequence Readiness:**
   - Requires all 6 frames $[t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0]$.
   - `6 / 6` frames available $\to$ Complete.
   - Missing $\ge 1$ frame $\to$ `WAITING_FOR_FRAMES`.
   - `0 / 6` frames available $\to$ `SOURCE_UNAVAILABLE`.
2. **Track History Readiness:**
   - Requires center fix at $t_0$ and $\ge 1$ causal prior fix at $t_{\text{prior}} \le t_0 - 3\text{h}$.
   - $\ge 2$ valid fixes $\to$ `READY`.
   - $< 2$ fixes $\to$ `INSUFFICIENT_HISTORY`.

**Overall Readiness Gate:**
- `READY` / `INFERENCE_READY`: All 6 causal satellite frames present AND $\ge 2$ causal track history fixes present.
- `WAITING_FOR_FRAMES`: Awaiting satellite frame delivery from external observation provider.
- `INSUFFICIENT_HISTORY`: Cyclone detected, but insufficient history to establish motion anchor velocity.
- `SOURCE_UNAVAILABLE`: External data provider endpoint unreachable or catalog empty.

> **Operational Safety Rule:** The "RUN FORECAST" action is disabled in the frontend unless the overall state is `READY` or `INFERENCE_READY`. VAYU-NET strictly prohibits synthetic frame interpolation or coordinate hallucination.

---

## 4. Provider Configuration

Data providers are dynamically instantiated via environment variables with safe fallback defaults:

```bash
# Cyclone Event Discovery Provider
# Options: 'replay' (default), 'imd_rsmc', 'catalog'
CYCLONE_EVENT_PROVIDER=replay

# Satellite Observation Provider
# Options: 'local_archive' (default), 'noaa_aws_s3'
SATELLITE_PROVIDER=local_archive

# Track History Provider
# Options: 'replay' (default), 'imd_advisory'
TRACK_HISTORY_PROVIDER=replay
```

### Credential & Security Guarantees
- NOAA AWS S3 Open Data utilizes public anonymous HTTPS requests (`https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/`). Zero AWS access keys, secret tokens, or passwords are used, stored, or logged.
- All provider logs are audited by automated tests (`test_03_no_credentials_in_logs`) to prevent credential leakage.

---

## 5. Instrument Compatibility & Domain Shift Notice

### Primary Model Constraint: GridSat-B1 Geostationary CDR
The primary trained deep learning checkpoints:
- Phase 3C Center Localization: `best_center_localization_cnn.pt`
- Phase 5B Variant A Hybrid Track: `best_phase5b_variant_a.pt`
- Phase 6 Intensity & Wind: `best_phase6_intensity_wind.pt`

were trained exclusively on 3-hourly $11\,\mu\text{m}$ infrared window brightness temperatures from the NOAA NCEI GridSat-B1 Climate Data Record ($0.07^\circ$ equal-angle projection, decadal calibration across GOES, METEOSAT, and INSAT sensors).

### Domain Shift Policy
- Direct feeding of raw Level-1B / Level-2 INSAT-3D/3DR or Meteosat data into the primary checkpoints without instrument-specific transfer calibration will cause severe distributional shift.
- Raw operational feeds may only be rendered as secondary contextual imagery until sensor-specific domain adaptation is validated.

---

## 6. Real-World Provider Latency & Operational Limitation

| Provider | Data Stream | Access | Latency / Availability | Operational Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA AWS S3** | GridSat-B1 NetCDF-4 CDR | Public Anonymous | **Retrospective CDR** (several months latency) | Functional for replay/sandbox & historical assimilation. Cannot provide near-real-time observations within minutes of $t_0$. |
| **IMD RSMC New Delhi** | Synoptic Bulletins & Advisories | Public Web Feed | Quasi-real-time (3-hourly synoptic issuances) | Primary operational candidate for Event Discovery & Track History. |
| **ISRO MOSDAC** | INSAT-3D/3DR Imager | Authenticated (Token) | Near-real-time (15-30 min) | High spatial/temporal capability, but spectrally different from GridSat-B1 (requires transfer adapter before ML inference). |

### Transparent Operational Readiness Declaration
- **Current-event orchestration:** READY
- **Live event discovery:** READY (Operational adapter implemented; replay sandbox validated)
- **Live satellite acquisition:** READY (AWS S3 downloader & local cache adapter implemented)
- **ML inference:** READY (All production checkpoints verified with 0 NaN/Inf)
- **Fully autonomous current forecasting:** **NOT READY** (Blocked by external availability of near-real-time GridSat-B1 CDR feed; requires near-real-time satellite proxy or calibrated INSAT-3D transfer model).
