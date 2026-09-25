# VAYU-NET — Current Inference Input Audit & Phase 5B Kinematic Contract

**Audit Date:** September 2026  
**Document Target:** `docs/current_inference_input_audit.md`  
**Focus:** Phase 5B Multi-Horizon Track Model (`best_phase5b_variant_a.pt` & `best_phase5b_variant_b.pt`) & Transition to Current/Live Cyclone Ingestion.

---

## 1. Executive Summary

This audit establishes the exact mathematical, architectural, and operational input requirements of the production Phase 5B hybrid residual track model (`Phase5BHybridResidualModel`). 

A key finding of this audit is that **Phase 5B Variant A (the production benchmark with 189.71 km test DPE)** is an *observed-anchor hybrid model*. It strictly requires a causal observed storm center at $t_0$ and at least one preceding historical center fix ($t_{-3\text{h}}$ or earlier) to construct its constant-velocity extrapolation anchor and motion context. 

For a current/new cyclone where historical IMD synoptic bulletins or observed track fixes have not yet been established or ingested, the pipeline **cannot execute Variant A track prediction without observed history**. Under no circumstances will future ground truth be back-propagated or synthetic coordinates invented. In such cases, the system must explicitly declare:
$$\text{track\_readiness} = \texttt{"INSUFFICIENT\_HISTORY"}$$

---

## 2. Phase 5B Architecture & Input Contract

The production model is implemented in [`ml/models/phase5b_hybrid_residual.py`](file:///c:/GitHub/vayu-net/ml/models/phase5b_hybrid_residual.py):

```python
class Phase5BHybridResidualModel(nn.Module):
    def forward(self, sat_seq, env_seq, motion_ctx, kin_12=None, kin_24=None, kin_48=None):
        ...
```

### Detailed Tensor Specifications

| Input Tensor / Feature | Dimension / Shape | Representation & Origin | Timestep Range | Upstream Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **1. Satellite Sequence (`sat_seq`)** | `[B, 6, 132]` | 6 causal GridSat-B1 IR observations processed through frozen Phase 3C spatial encoder (128 features) + spatial coordinates/intensity summary (4 features). | $t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0$ | GridSat-B1 IRWIN CDR archive / live feed. |
| **2. Environmental Sequence (`env_seq`)** | `[B, 6, 8, 41, 66]` | ERA5 8-channel wind fields ($u, v$ at 850, 700, 500, 200 hPa) on a $0.25^\circ$ regional grid. | $t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0$ | Atmospheric reanalysis / numerical weather prediction wind analysis. |
| **3. Motion Context (`motion_ctx`)** | `[B, 5]` | 5-dimensional normalized motion and position vector. | Evaluated between prior fix and $t_0$ | Causal track observations ($t_{\text{prior}} \to t_0$). |
| **4. Kinematic Anchors (`kin_12`, `kin_24`, `kin_48`)** | `[B, 2]` each | Linear constant-velocity extrapolation coordinates $[\phi_{h}, \lambda_{h}]$ in degrees. | Targets $+12\text{h}, +24\text{h}, +48\text{h}$ | Linear extrapolation from $t_0$ center. |

---

## 3. Dissection of Motion Context & Kinematics

In the training pipeline ([`ml/train/train_hybrid_kinematic_satellite.py`](file:///c:/GitHub/vayu-net/ml/train/train_hybrid_kinematic_satellite.py), lines 121–150):

### The 4 Observed Track State Variables:
Across historical storm track records, every observation timestep contains 4 primary kinematic variables:
1. **`lat` (Latitude in degrees North)**
2. **`lon` (Longitude in degrees East)**
3. **`wind_kt` (Continuous maximum sustained wind in knots)**
4. **`pressure_hpa` (Estimated central pressure in hPa)**

### Derivation of Velocity and Motion Context:
Let $\mathbf{P}_0 = (\phi_0, \lambda_0)$ be the storm center at $t_0$, and $\mathbf{P}_{\text{prior}} = (\phi_{\text{prior}}, \lambda_{\text{prior}})$ be the latest preceding verified center at $t_{\text{prior}} = t_0 - \Delta h$ ($\Delta h \ge 3\text{ hours}$):

1. **Angular Velocity:**
   $$v_{\text{lat}} = \frac{\phi_0 - \phi_{\text{prior}}}{\Delta h}, \quad v_{\text{lon}} = \frac{\lambda_0 - \lambda_{\text{prior}}}{\Delta h}$$
2. **Physical Velocity (km/h):**
   $$v_{\text{north}} = v_{\text{lat}} \times \frac{\pi}{180} \times R_{\text{earth}}$$
   $$v_{\text{east}} = v_{\text{lon}} \times \cos\left(\phi_0 \times \frac{\pi}{180}\right) \times \frac{\pi}{180} \times R_{\text{earth}}$$
   $$\text{speed} = \sqrt{v_{\text{north}}^2 + v_{\text{east}}^2}$$
3. **5-Dimensional Motion Context Vector (`motion_ctx`):**
   $$\text{motion\_ctx} = \begin{bmatrix} 
   v_{\text{north}} / 50.0 \\ 
   v_{\text{east}} / 50.0 \\ 
   \text{speed} / 50.0 \\ 
   (\phi_0 - \text{LAT\_MIN}) / \text{LAT\_SPAN} \\ 
   (\lambda_0 - \text{LON\_MIN}) / \text{LON\_SPAN} 
   \end{bmatrix}$$
   *(where $\text{LAT\_MIN}=-5.0^\circ$, $\text{LAT\_SPAN}=40.0^\circ$, $\text{LON\_MIN}=40.0^\circ$, $\text{LON\_SPAN}=65.0^\circ$)*
4. **Extrapolated Kinematic Anchors:**
   $$\mathbf{P}_{\text{kin}, 12} = \left[\phi_0 + v_{\text{lat}} \cdot 12.0, \; \lambda_0 + v_{\text{lon}} \cdot 12.0\right]$$
   $$\mathbf{P}_{\text{kin}, 24} = \left[\phi_0 + v_{\text{lat}} \cdot 24.0, \; \lambda_0 + v_{\text{lon}} \cdot 24.0\right]$$
   $$\mathbf{P}_{\text{kin}, 48} = \left[\phi_0 + v_{\text{lat}} \cdot 48.0, \; \lambda_0 + v_{\text{lon}} \cdot 48.0\right]$$

---

## 4. Specific Data Dependencies Audit

### Does the production pipeline require IMD best-track historical positions?
- **For Variant A (`best_phase5b_variant_a.pt`):** **YES.** To achieve its verified **189.71 km test DPE**, Variant A requires the human-observed / synoptically analyzed center at $t_0$ and at least one prior fix at $t \le t_0 - 3\text{h}$ to establish the physical inertial base.
- **For Variant B (`best_phase5b_variant_b.pt`):** **NO.** Variant B computes $v_{\text{lat}}, v_{\text{lon}}$ purely from the AI-detected centers at $t_{-3\text{h}}$ and $t_0$ output by Phase 3C. However, because single-frame center localization error (~400–800 km) is noisy, differencing these centers causes severe velocity error, resulting in **1048.57 km validation DPE**. Hence Variant B is an ablation model, not operational standard.

### Does the production pipeline require observed storm center?
- **YES.** Variant A requires the observed center at $t_0$ as the anchor point for trajectory reconstruction:
  $$\hat{\mathbf{P}}_h = \mathbf{P}_{\text{kin}, h} + \hat{\mathbf{R}}_h$$
  where $\hat{\mathbf{R}}_h = [\Delta\text{North}_h, \Delta\text{East}_h]$ is the learned geodesic residual displacement in km.

### Does the production pipeline require observed wind?
- **For Track (Phase 5B):** **NO.** The track model does not ingest wind speed directly.
- **For Analog Retrieval (k=2):** **YES.** The 7D analog retrieval vector uses $t_0$ observed wind:
  $$\mathbf{x}_{\text{analog}} = [\text{lat}, \text{lon}, \text{wind\_kt}, \text{pressure\_hpa}, \Delta x_{12\text{h}}, \Delta y_{12\text{h}}, \Delta\text{wind}_{12\text{h}}]$$
  If observed wind is unavailable at $t_0$, the AI-predicted wind from Phase 6 can be used with explicit notation, or analog retrieval returns `UNAVAILABLE`.

### Does the production pipeline require other historical track information?
- **YES.** The timestamp difference $\Delta h$ between prior and current observations is required to normalize velocity to degrees per hour.

---

## 5. Current Event Readiness Rules

For live or newly forming cyclonic systems:
1. **If 6 GridSat frames exist AND at least 2 historical center observations ($t_{-3h}, t_0$) exist:**
   - Center localization: `READY`
   - Intensity & wind: `READY`
   - Track forecast: `READY` (using observed history)
   - Trajectory verification: `UNAVAILABLE` (no future ground truth exists yet)
   - Readiness status: `INFERENCE_READY`
2. **If 6 GridSat frames exist BUT storm track history is absent ($< 2$ fixes):**
   - Center localization: `READY` (single frame $t_0$ operates independently)
   - Intensity & wind: `READY` (operates on satellite sequence)
   - Track forecast: `INSUFFICIENT_HISTORY` (cannot construct Variant A kinematic anchor)
   - Trajectory verification: `UNAVAILABLE`
   - Readiness status: `READY_PARTIAL`
3. **If $< 6$ satellite frames are available:**
   - Readiness status: `WAITING_FOR_FRAMES`
   - Missing frames must NOT be linearly interpolated or fabricated.
