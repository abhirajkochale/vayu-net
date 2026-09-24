# VAYU-NET Phase 5A: Anti-Leakage & Causal Integrity Audit

## 1. Audit Scope & Mandate
Phase 5A introduces multi-level atmospheric wind fields from ERA5 reanalysis into the VAYU-NET temporal track prediction framework. In accordance with Section 20 of the research specifications, this audit provides a rigorous, mathematical, and forensic verification that **zero future information** enters model training, feature extraction, normalization, caching, or evaluation.

---

## 2. Invariant Checklist: Timestamp Causal Boundary

For every candidate sample initialized at reference time $t_0$, the input sequence is causally bounded by:
$$\mathcal{T}_{\text{inputs}} = \{t \in \mathcal{T} \mid t \le t_0\}$$
Specifically:
$$\mathcal{T}_{\text{inputs}} = \{t_{-15\text{h}}, t_{-12\text{h}}, t_{-9\text{h}}, t_{-6\text{h}}, t_{-3\text{h}}, t_0\}$$

Future cyclone positions are strictly target-only references:
$$\mathcal{T}_{\text{targets}} = \{t_0 + 12\text{h}, t_0 + 24\text{h}, t_0 + 48\text{h}\} > t_0$$

### 2.1 Component-by-Component Causal Audit

| Pipeline Component | Temporal Boundary | Audit Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **GridSat Satellite Frames** | $t_{-15\text{h}} \le t \le t_0$ | Source NPZ file paths inspected from `vayu_net_sample_index.csv`. Maximum timestamp index is $t_0$. | **PASSED** |
| **ERA5 Atmospheric Wind Frames** | $t_{-15\text{h}} \le t \le t_0$ | Timestamp extraction in `ml/data/era5_loader.py` matches GridSat frame times. ERA5 frames after $t_0$ are never requested. | **PASSED** |
| **Normalization Statistics** | $t \le \text{2018-12-14}$ (TRAIN only) | `data/interim/ml/era5_train_normalization_stats.json` computed strictly from TRAIN samples. Zero VALIDATION (`2019-2020`) or TEST (`2021-2024`) pixels included. | **PASSED** |
| **Feature Cache Storage** | $t_{-15\text{h}} \le t \le t_0$ | `data/interim/ml/cache/era5_environment_features.pt` inspects each sample dictionary. Input tensors contain only $T \le t_0$. Targets stored in distinct target tensors. | **PASSED** |
| **Environmental Encoder** | Causal GRU | 2-layer GRU consumes sequence $t_{-15\text{h}} \to t_0$. Terminal hidden state represents $h(t_0)$. No bidirectional sequence processing across future steps. | **PASSED** |
| **Multimodal Fusion** | Concatenation at $t_0$ | Satellite hidden state $h_{\text{sat}}(t_0)$ and environmental hidden state $h_{\text{env}}(t_0)$ fused at $t_0$. | **PASSED** |
| **Target Ground Truths** | $t_0 + 12\text{h}, +24\text{h}, +48\text{h}$ | Supervised targets used solely in loss computation ($\mathcal{L}_{\text{track}}$) and evaluation metrics (DPE). Zero gradients or features flow back to inputs. | **PASSED** |

---

## 3. Split Isolation & Historical Reanalysis Provenance

1. **Chronological Event-Level Split:**
   - **TRAIN (673 samples / 17 storms):** `1998-10-06` to `2018-12-14`
   - **VALIDATION (275 samples / 7 storms):** `2019-01-03` to `2020-12-03`
   - **TEST (371 samples / 9 storms):** `2021-05-13` to `2024-11-29`
   - **Overlap:** Exactly 0 co-occurring timesteps across split boundaries.

2. **Reanalysis Invariant (Section 30 Notice):**
   ERA5 is an atmospheric reanalysis produced with full data assimilation across historical periods. While retrospective reanalysis provides an ideal scientific testbed for evaluating the upper bound of steering flow utility, it is explicitly classified as **historical reanalysis-augmented forecasting**, distinct from real-time operational NWP forecasts.
