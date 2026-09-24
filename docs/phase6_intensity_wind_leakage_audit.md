# VAYU-NET Phase 6: Anti-Leakage & Causal Integrity Audit
## Multi-Task Cyclone Intensity Classification + Maximum Sustained Wind Speed Regression

**Date:** 2026-09-25  
**Audit Status:** **VERIFIED & CERTIFIED (ZERO LEAKAGE)**  
**Governing Standard:** SIH 2026 Problem Statement 26070 Research Governance  

---

## 1. Audit Scope & Mandate
Phase 6 builds the real-time cyclone intensity and maximum sustained wind intelligence module for VAYU-NET. This audit provides an authoritative, mathematical, and forensic verification that:
1. **Zero Future Information enters the model:** Inputs are strictly causally bounded by $t \le t_0$.
2. **Zero Information Leakage occurs across dataset splits:** Complete event-level storm separation is preserved.
3. **Zero Contamination in Feature Normalization:** All continuous normalization parameters and loss class weights are derived exclusively from the TRAIN split.
4. **Zero Missing-Label Poisoning:** Legitimate missing wind and category observations are loss-masked and never converted into artificial real labels.
5. **Zero Data Snooping in Model Selection & Calibration:** Model checkpoint selection and probability calibration are strictly governed by the VALIDATION split; held-out TEST is evaluated strictly once.

---

## 2. Invariant Checklist: Causal Horizon Bounding

For every candidate sample initialized at synoptic reference time $t_0$, the input feature space is causally bounded by:
$$\mathcal{T}_{\text{inputs}} = \{t \in \mathcal{T} \mid t \le t_0\} = \{t-15\text{h}, t-12\text{h}, t-9\text{h}, t-6\text{h}, t-3\text{h}, t_0\}$$

Predictions are strictly focused on the current cyclone state at reference time $t_0$:
- Current IMD Intensity Category $\hat{C}(t_0) \in \{0, 1, \dots, 6\}$
- Current Maximum Sustained Wind Speed $\hat{w}(t_0) \in [15, 150]\text{ kt}$

### 2.1 Component-by-Component Forensic Audit

| Pipeline Component | Temporal Boundary | Audit Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **GridSat Satellite Sequences** | $t-15\text{h} \le t \le t_0$ | Source NPZ file paths from `data/manifests/vayu_net_sample_index.csv`. Monotonically non-decreasing timestamps terminating strictly at $t_0$. | **PASSED** |
| **ERA5 Multimodal Wind Fields** | $t-15\text{h} \le t \le t_0$ | Multi-level 8-channel wind frames extracted strictly at synoptic hours $\le t_0$. No ERA5 frames after $t_0$ are accessed. | **PASSED** |
| **Future Track / Intensity Ground Truths** | $t_0 + 12\text{h}, +24\text{h}, +48\text{h}$ | Future positions and future winds from the sample index are completely excluded from the Phase 6 feature cache and model forward pass. | **PASSED** |
| **Current Ground Truth Targets** | $t_0$ strictly | Ground truth intensity category and wind speed at $t_0$ are used exclusively in loss calculation ($\mathcal{L}_{\text{multi-task}}$) and metric evaluation. | **PASSED** |
| **Temporal Recurrent State** | Causal Unidirectional GRU | 2-layer GRU consumes sequence causally from $t-15\text{h} \to t_0$. Terminal hidden state represents $h(t_0)$. No reverse-time or bidirectional pass across timesteps. | **PASSED** |
| **Multimodal Representation Fusion** | Fusion at $t_0$ | Satellite hidden state $h_{\text{sat}}(t_0)$ and environmental hidden state $h_{\text{env}}(t_0)$ are concatenated and projected at $t_0$. | **PASSED** |

---

## 3. Event-Level Split Disjointness & Spatial Isolation

The VAYU-NET dataset partitions all candidate sequences at the **cyclone storm event level**, ensuring that no frames, storm dynamics, or life cycles from the same cyclone ever co-occur across different splits:

$$\mathcal{S}_{\text{TRAIN}} \cap \mathcal{S}_{\text{VAL}} = \emptyset, \quad \mathcal{S}_{\text{TRAIN}} \cap \mathcal{S}_{\text{TEST}} = \emptyset, \quad \mathcal{S}_{\text{VAL}} \cap \mathcal{S}_{\text{TEST}} = \emptyset$$

### Split Inventory & Disjointness Audit:
- **TRAIN Split:** **81 storms** (696 samples, $52.8\%$)
- **VALIDATION Split:** **14 storms** (252 samples, $19.1\%$)
- **TEST Split:** **31 storms** (371 samples, $28.1\%$)
- **Total Validated Samples:** **1,319 samples** across 126 storms (1998–2024)
- **Storm-Level Intersections:** Exactly **0 overlapping storms** across all split combinations.

---

## 4. Normalization Statistics & Loss Class Weight Integrity

To guarantee that no future or held-out test distribution information leaks into model parameter training:

1. **Continuous Target Normalization:**
   - Training Wind Mean: $\mu_{\text{wind}} = 36.8836\text{ kt}$
   - Training Wind Std: $\sigma_{\text{wind}} = 17.9754\text{ kt}$
   - Derived strictly from the 696 TRAIN samples. Zero VALIDATION or TEST labels were included.

2. **CrossEntropy Class Weights:**
   - Derived strictly from valid TRAIN category observations ($N=695$):
     - `D` (246 samples): $w = 0.4048$
     - `DD` (190 samples): $w = 0.5242$
     - `CS` (151 samples): $w = 0.6596$
     - `SCS` (40 samples): $w = 2.4897$
     - `VSCS` (63 samples): $w = 1.5808$
     - `ESCS` (3 samples): $w = 33.1966$
     - `SuCS` (2 samples): $w = 49.7949$
   - Smoothed and clipped to $[0.25, 5.0]$ with mean $1.0$ to ensure numerical stability during optimization.
   - Zero VALIDATION or TEST class frequencies were used in weight calculation.

---

## 5. Missing Label Masking Integrity

1. **Intensity Category:**
   - Exactly 6 samples across the entire NIO archive have missing category values (1 in TRAIN, 5 in VAL, 0 in TEST).
   - In all instances, these samples are encoded with index `-1` and mask `0.0`.
   - `CrossEntropyLoss(ignore_index=-1)` strictly excludes them from loss computation and gradient updates.
   - Unit test verified: zero gradient flows to the classification head from masked category positions.

2. **Maximum Sustained Wind:**
   - The 10 legitimate source dissipation NaN positions identified in QA are loss-masked via $M_{\text{wind}} \in \{0, 1\}$.
   - Loss formulation:
     $$\mathcal{L}_{\text{wind}} = \frac{\sum_{i=1}^B M_{\text{wind}, i} \cdot \text{SmoothL1}(\hat{z}_i, z_i)}{\max\left(1, \sum_{i=1}^B M_{\text{wind}, i}\right)}$$
   - Missing winds are never silently imputed or filled with zeros.

---

## 6. Model Selection & Probability Calibration Discipline

1. **Validation-Governed Model Selection:**
   - Hyperparameter tuning, early stopping, and checkpoint selection were governed strictly by the **Validation Composite Error Index**:
     $$\text{Composite Error} = (1.0 - \text{Macro F1}_{\text{val}}) + \frac{\text{MAE}_{\text{wind, val}}}{\sigma_{\text{train, wind}}}$$
   - The TEST split was **completely held out** during model selection.

2. **Calibration Parameter Fitting:**
   - Temperature scaling scalar $T$ was fitted strictly on VALIDATION logits to minimize validation cross-entropy loss.
   - The optimal temperature $T$ was frozen and subsequently applied to TEST exactly once.

3. **Empirical Uncertainty Quantiles:**
   - Empirical error bounds (Median AE, P80 AE, P90 AE) were computed strictly from VALIDATION residuals.
   - Evaluated on TEST to assess empirical coverage without post-hoc tuning.

---

## 7. Cryptographic & Immutability Verification
All authoritative locked files remain bit-identical to their locked specifications:
- `data/processed/imd_best_track_v2.csv` (SHA-256 verified)
- `data/manifests/vayu_net_sample_index.csv` (SHA-256 verified)
- `data/manifests/gridsat_sha256_manifest.csv` (SHA-256 verified)
- Historical experiment checkpoints (Phases 3A through 5B) intact and unmodified.

**Audit Conclusion:** Phase 6 satisfies all scientific, causal, and anti-leakage invariants. The model outputs are certified free from temporal and target leakage.
