# CEFM Step 4 — Production Feature Distribution and Clinical Scoring Integrity Audit

**Document ID:** `CEFM_STEP4_PRODUCTION_FEATURE_DISTRIBUTION_AUDIT`  
**Date:** November 2026  
**Status:** Complete Forensic Investigation (Read-Only)  
**Target Case:** `data/uploads/5ef6da25eecc47c6a05bee924b2aa3f8.jpg`  
**Target Case SHA-256:** `a6ebb0d5869e833f874f776a442278288521bc605950bdf599494e1766fb4bba`  
**Scope Boundaries:** Read-only forensic investigation. Zero model retraining, zero checkpoint modification, zero threshold recalibration, zero production code modification.

---

## Executive Summary

This forensic audit investigates the clinical feature pipeline, training-to-production distribution shifts, model compatibility, and clinical risk scoring across the CEFM platform. It addresses the core operational phenomenon where dermoscopic images exhibiting high physical or ABCDE suspicion (e.g. diameter $> 11\text{ mm}$, color diversity) produce confident `NON_MEL` predictions in deep multimodal classifiers.

### Key Forensic Discoveries:

1. **Severe Out-of-Distribution (OOD) Shift at Indices 11 and 12:**  
   In the HAM10000 dataset used to train all Phase 1 and Phase 2 models ($N=10,015$ across train, val, and test splits), **zero physical calibration reference existed**. Consequently, Dimension 11 (`diameter_mm_norm`) and Dimension 12 (`calibration_available`) were **strictly constant zero** ($0.0000 \pm 0.0000$) across 100% of training samples. In contrast, in production user uploads, **88.9% of cases feature millimeter rulers**, resulting in non-zero physical diameters ($d_{\text{norm}} \approx 0.10 - 0.60$) and `calibration_available = 1.0`.

2. **Astronomical Feature Scaling in M6 Normalization Pipeline:**  
   During M6 training cache preparation, near-zero standard deviations were clamped to `1.0e-6` (`clinical_norm_std[11] = 1.0e-6`, `clinical_norm_std[12] = 1.0e-6`). When production passes calibrated diameter values ($x_{11} = 0.598, x_{12} = 1.0$), the normalization formula `(x - mean) / std` produces:
   $$\hat{x}_{11} = \frac{0.598 - 0.0}{1.0 \times 10^{-6}} = \mathbf{598,000.0}$$
   $$\hat{x}_{12} = \frac{1.0 - 0.0}{1.0 \times 10^{-6}} = \mathbf{1,000,000.0}$$
   This exposes the M6 `clinical_encoder` Linear layer to inputs six orders of magnitude outside its training domain.

3. **LayerNorm Containment Prevents Numerical Explosion:**  
   Despite inputs exceeding $10^5-10^6$, M6 does not output `NaN` or `Inf`. The first layer of `clinical_encoder` contains `LayerNorm((64,))`, which re-standardizes the 64-D activations back to zero mean and unit variance. However, because the input is astronomically large, the linear projection is dominated entirely by the random initial weights of dimensions 11 and 12, saturating downstream GELU units.

4. **Counterfactual Invariance Confirms Visual Dominance:**  
   Controlled ablation across four counterfactual input conditions (actual calibrated, index 11 zeroed, index 12 zeroed, both zeroed) proves that zeroing out the OOD diameter features alters melanoma probability by only **$0.01\%$ in Phase 1 Multimodal** ($0.02\% - 0.03\%$) and **$0.16\%$ in M6** ($0.71\% - 0.87\%$). In all four conditions, both models predict `non_mel`. The deep visual representations ($z_g, z_l$) overwhelmingly dominate the final classification logits.

5. **Reconciliation of ABCDE Discrepancy ($0.1182$ vs $0.3884$):**  
   The numerical discrepancy between $0.1182$ and $0.3884$ is conclusively resolved. Passing the raw nested dictionary from `extract_abcde_features()` into `compute_abcde_score()` caused color and diameter to default silently to zero ($0.1182$). Passing the production flattened dictionary (`flat`) correctly captures color index ($0.5003$) and diameter ($11.96\text{ mm}$), yielding $0.3884$. Both evaluate to categorical risk label `LOW` ($< 0.45$).

---

## Section A: Training versus Production Feature Distributions

A comprehensive analysis of the 16-dimensional clinical feature vector was conducted across five independent cohorts:
1. **HAM10000 Training Cache** ($N=6,978$)
2. **HAM10000 Validation Cache** ($N=1,533$)
3. **HAM10000 Test Cache** ($N=1,504$)
4. **PH² External Cohort** ($N=200$)
5. **Production Uploads Cohort** ($N=9$ user cases in `data/uploads`)

The summary statistics are recorded in [`reports/CEFM_STEP4_FEATURE_DISTRIBUTION.csv`](file:///D:/pytest_cache/reports/CEFM_STEP4_FEATURE_DISTRIBUTION.csv) and presented below:

### Table 1: Complete 16-Dimensional Clinical Feature Distribution Across Cohorts

| Idx | Feature Name | Train Mean $\pm$ Std | Val Mean $\pm$ Std | Test Mean $\pm$ Std | PH² Mean $\pm$ Std | Prod Mean $\pm$ Std | Prod % Non-zero | Distribution Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `asymmetry_score` | $0.2441 \pm 0.1232$ | $0.2414 \pm 0.1219$ | $0.2435 \pm 0.1235$ | $0.2194 \pm 0.1070$ | $0.2376 \pm 0.0612$ | 100.0% | Moderate Shift ($p<0.05$) |
| **1** | `horizontal_asymmetry` | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |
| **2** | `vertical_asymmetry` | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |
| **3** | `border_irregularity` | $0.4155 \pm 0.1332$ | $0.4144 \pm 0.1301$ | $0.4160 \pm 0.1311$ | $0.4631 \pm 0.1129$ | $0.3428 \pm 0.0981$ | 100.0% | Extreme Segmentation Shift |
| **4** | `circularity` | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | 100.0% | Constant One In-Distribution |
| **5** | `radial_distance_var` | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |
| **6** | `curvature_variation` | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |
| **7** | `color_h_norm` | $0.6076 \pm 0.3120$ | $0.6117 \pm 0.3131$ | $0.6014 \pm 0.3135$ | $0.1829 \pm 0.2638$ | $0.0184 \pm 0.0211$ | 100.0% | Extreme Optical Shift ($p < 10^{-60}$) |
| **8** | `color_s_norm` | $0.2042 \pm 0.0800$ | $0.2039 \pm 0.0753$ | $0.2037 \pm 0.0788$ | $0.1991 \pm 0.0612$ | $0.2082 \pm 0.0410$ | 100.0% | In-Distribution |
| **9** | `color_v_norm` | $0.1760 \pm 0.0891$ | $0.1766 \pm 0.0887$ | $0.1738 \pm 0.0875$ | $0.2430 \pm 0.0892$ | $0.1155 \pm 0.0520$ | 100.0% | Extreme Optical Shift ($p < 10^{-18}$) |
| **10** | `color_diversity` | $0.9950 \pm 0.0630$ | $0.9984 \pm 0.0285$ | $0.9984 \pm 0.0329$ | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | 100.0% | In-Distribution |
| **11** | `diameter_mm_norm` | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.4212 \pm 0.2311}$ | **88.9%** | **Severe OOD Shift** |
| **12** | `calibration_avail` | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.0000 \pm 0.0000}$ | $\mathbf{0.8889 \pm 0.3333}$ | **88.9%** | **Severe OOD Shift** |
| **13** | `evolution_change` | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |
| **14** | `evolution_magnitude`| $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |
| **15** | `evolution_available`| $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | 0.0% | Constant Zero In-Distribution |

---

## Section B: Model Compatibility Investigation

### 1. Training Cache Reality vs Production Reality
- Across the entire HAM10000 dataset ($10,015$ images), **9 out of 16 feature dimensions** (1, 2, 5, 6, 11, 12, 13, 14, 15) were **constant zero**. Dimension 4 was constant 1.0.
- Only **6 dimensions** (0, 3, 7, 8, 9, 10) exhibited training variance.
- Because dimensions 11 and 12 were constant zero during training:
  $$\frac{\partial \mathcal{L}}{\partial W_{:, 11}} = 0, \quad \frac{\partial \mathcal{L}}{\partial W_{:, 12}} = 0$$
  The neural network weights connecting inputs 11 and 12 to the 64 hidden units in both Phase 1 Multimodal (`clinical_encoder[0].weight[:, 11:13]`) and M6 (`clinical_encoder[0].weight[:, 11:13]`) were **never updated by gradient descent**. They remain at their random PyTorch initialization values ($\mathcal{U}(-1/\sqrt{16}, 1/\sqrt{16}) \approx [-0.25, 0.25]$).

### 2. The Numerical Catastrophe: Epsilon Clamping to $1.0 \times 10^{-6}$
In `models/experiments/M6/config.json`, the empirical standard deviations calculated during training feature extraction were saved:
```json
"clinical_norm_std": [
  0.1232572, 1.0e-6, 1.0e-6, 0.1331989, 1.0e-6, 1.0e-6, 1.0e-6,
  0.3119767, 0.0800194, 0.0891368, 0.0629591, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6
]
```
When `app/services/classification.py::predict_m6_challenger()` runs:
```python
c_norm = (c_tensor - norm_mean) / norm_std
```
For target image `5ef6da25eecc47c6a05bee924b2aa3f8.jpg`:
- `c_tensor[11] = 0.598` (normalized diameter $11.96\text{ mm} / 20.0$)
- `c_tensor[12] = 1.0` (calibration available)
- `c_norm[11] = 0.598 / 1.0e-6 = 598,000.0`
- `c_norm[12] = 1.0 / 1.0e-6 = 1,000,000.0`

### 3. LayerNorm Resilience Mechanism
Why does M6 not crash with `NaN` or `Inf`?
The first layer of `clinical_encoder` is:
```
Linear(16, 64) -> LayerNorm(64) -> GELU() -> Dropout(0.15)
```
1. In `Linear(16, 64)`, the dot product $W \cdot c_{\text{norm}}$ yields huge intermediate activations: $\approx \pm 250,000.0$.
2. Immediately following, `LayerNorm((64,))` calculates:
   $$\mu = \frac{1}{64} \sum_{i=1}^{64} z_i, \quad \sigma = \sqrt{\frac{1}{64} \sum_{i=1}^{64} (z_i - \mu)^2 + \epsilon}, \quad \hat{z}_i = \frac{z_i - \mu}{\sigma}$$
3. This mathematically forces the 64 output activations back to mean $0.0$ and variance $1.0$, which is why downstream components receive finite values ($c_{\text{mod}} \approx 0.50$, amplification $\approx 1.50$).
4. **However, from a machine learning perspective, this projection is pure pseudorandom noise**, dominated entirely by unoptimized initialization weights rather than learned clinical intelligence.

---

## Section C: Counterfactual Sensitivity Analysis

To determine the empirical impact of this out-of-distribution feature scaling on classification probabilities, a controlled counterfactual ablation experiment was conducted on the target production case (`5ef6da25eecc47c6a05bee924b2aa3f8.jpg`).

Four counterfactual conditions were evaluated:
- **Condition 1 (Actual Calibrated):** $x_{11} = 0.598, x_{12} = 1.0$ (production inference state)
- **Condition 2 (Index 11 Zeroed):** $x_{11} = 0.000, x_{12} = 1.0$ (calibration flagged, zero diameter)
- **Condition 3 (Index 12 Zeroed):** $x_{11} = 0.598, x_{12} = 0.0$ (diameter present, calibration unflagged)
- **Condition 4 (Both Zeroed):** $x_{11} = 0.000, x_{12} = 0.0$ (exact HAM10000 training distribution state)

The results are recorded in [`reports/CEFM_STEP4_COUNTERFACTUAL_INPUT_ANALYSIS.csv`](file:///D:/pytest_cache/reports/CEFM_STEP4_COUNTERFACTUAL_INPUT_ANALYSIS.csv) and summarized in Table 2:

### Table 2: Counterfactual Sensitivity Results on Target Case

| Metric / Dimension | Condition 1: Actual Calibrated | Condition 2: Index 11 Zeroed | Condition 3: Index 12 Zeroed | Condition 4: Both Zeroed (Training State) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Input $x_{11}$ (`diameter_norm`)** | $0.598$ | $0.000$ | $0.598$ | $0.000$ |
| **Raw Input $x_{12}$ (`calibration_avail`)** | $1.000$ | $1.000$ | $0.000$ | $0.000$ |
| **M6 Normalized Input $\hat{x}_{11}$** | **$598,000.0$** | **$0.0$** | **$598,000.0$** | **$0.0$** |
| **M6 Normalized Input $\hat{x}_{12}$** | **$1,000,000.0$** | **$1,000,000.0$** | **$0.0$** | **$0.0$** |
| **P1 Clinical Embedding Norm $\|z_c\|$** | $4.0151$ | $3.9382$ | $3.7154$ | $3.5274$ |
| **P1 Raw Logits $[z_0, z_1]$** | $[4.1776, -4.0866]$ | $[4.1958, -4.1511]$ | $[4.1728, -4.1151]$ | $[4.1757, -4.1946]$ |
| **P1 Melanoma Probability** | **$0.0258\%$** | **$0.0237\%$** | **$0.0251\%$** | **$0.0232\%$** |
| **P1 Prediction (Threshold 0.36)** | **`non_mel`** | **`non_mel`** | **`non_mel`** | **`non_mel`** |
| **M6 Clinical Embedding Norm $\|z_c\|$** | $5.1614$ | $5.0504$ | $4.7798$ | $5.3190$ |
| **M6 RCAG Modulation Mean $c_{\text{mod}}$** | $0.5007$ | $0.4998$ | $0.5078$ | $0.4947$ |
| **M6 RCAG Amplification Mean $(1 + c_{\text{mod}})$**| $1.5007$ | $1.4998$ | $1.5078$ | $1.4947$ |
| **M6 Raw Logits $[z_0, z_1]$** | $[2.4401, -2.2953]$ | $[2.6370, -2.3021]$ | $[2.5071, -2.2654]$ | $[2.6544, -2.2630]$ |
| **M6 Melanoma Probability** | **$0.8702\%$** | **$0.7110\%$** | **$0.8388\%$** | **$0.7265\%$** |
| **M6 Prediction (F1 Calibrated 0.690)** | **`non_mel`** | **`non_mel`** | **`non_mel`** | **`non_mel`** |
| **M6 Prediction (Matched Spec 0.625)** | **`non_mel`** | **`non_mel`** | **`non_mel`** | **`non_mel`** |

### Critical Findings from Counterfactual Probing:
1. **Probability Invariance:** Zeroing both OOD diameter features drops M6 melanoma probability from $0.87\%$ to $0.73\%$ ($\Delta = -0.14\%$). In Phase 1 Multimodal, probability shifts by less than $0.003\%$ ($0.026\%$ vs $0.023\%$).
2. **Prediction Robustness:** Across all four conditions, both Phase 1 Multimodal and M6 predict `non_mel` with overwhelming confidence. The decision margins are massive ($> 0.35$ below threshold in P1; $> 0.60$ below threshold in M6).
3. **Amplification Stability:** In M6, the RCAG residual gating multiplier $(1 + c_{\text{mod}})$ evaluates to almost exactly $1.500\times$ across all conditions ($1.5007\times$ in Cond 1 vs $1.4947\times$ in Cond 4), demonstrating that LayerNorm completely absorbs the $10^6$ input scale.

---

## Section D: ABCDE Scoring Interface Reconciliation

As documented in [`reports/CEFM_STEP4_ABCDE_CALLER_AUDIT.md`](file:///D:/pytest_cache/reports/CEFM_STEP4_ABCDE_CALLER_AUDIT.md):

1. **Exact Origin of 0.1182 vs 0.3884:**
   - In `app/services/analysis.py`, `extract_abcde_features()` returns a **nested** dictionary with `features["color"]["index"]` and `features["diameter"]["diameter_mm"]`.
   - `compute_abcde_score()` expects a **flattened** dictionary with top-level keys `color_index` and `diameter_mm`.
   - When called with the raw nested dictionary, `.get("color_index", 0.0)` and `.get("diameter_mm")` return zero, yielding **0.1182**.
   - When called with the flattened dictionary `flat` generated at line 377, color index ($0.5003$) and physical diameter ($11.96\text{ mm}$) are evaluated, yielding **0.3884**.
2. **Production Integrity:**
   - In production (`app/services/analysis.py:425`), the call is `abcde_risk = compute_abcde_score(flat)`. Production has **always** computed and displayed **0.3884**.
3. **Risk Label Invariance:**
   - Because both $0.1182 < 0.45$ and $0.3884 < 0.45$, both map strictly to categorical risk label **`LOW`**.

---

## Section E: Scientific Interpretation & Limitations of Counterfactual Probing

### 1. Why Did Distribution Shift Occur?
The distribution shift between training and production stems from dataset provenance:
- **HAM10000** was curated from dermatological centers where lesion dermoscopy was captured without standardized physical metric markers or ruler scales. Training caches zero-imputed diameter because physical measurements were completely absent.
- **Production Demands** encourage clinicians to include calibration markers (rulers, standard coins) to assess lesion enlargement under Rule 15/44. The preprocessing pipeline appropriately detects rulers ($88.9\%$ detection rate in user uploads).
- However, because the neural network architectures were trained exclusively on zeroed clinical diameter caches, **they never learned the predictive association between large physical diameter ($>6\text{ mm}$) and melanoma malignancy**.

### 2. Why Counterfactual Manipulation Cannot Establish Ground-Truth
In machine learning forensics, setting input features to counterfactual values (e.g. forcing $x_{11}=0$) demonstrates **model sensitivity**, not **biological or clinical truth**:
- Forcing $x_{11}=0$ shows that the model's prediction is insensitive to diameter variations because the linear weights are unoptimized random noise.
- It does **not** prove that the lesion is biologically benign or malignant.
- Medical deep learning systems must never be judged solely on counterfactual ablations when the underlying models were trained with uninformative or zero-imputed priors.

### 3. Concrete Architectural Roadmap for Future Model Versions
*(Note: These are recommendations for future training cycles; no code or weights have been modified during this audit).*
1. **Dynamic Schema Normalizer:** Implement a unified dictionary adapter that gracefully unpacks both nested and flattened ABCDE dictionaries without silent zero-fallback.
2. **Missing Feature Indicator Channel:** When retraining multimodal models, supply an explicit `diameter_missing` boolean mask, and train with random feature dropout on diameter so the network learns to operate under both calibrated and uncalibrated clinical settings.
3. **Normalization Guardrails:** Never clamp `norm_std` to $10^{-6}$ for constant training features. Constant features should either be dropped from the clinical vector entirely or assigned unit standard deviation ($\sigma = 1.0$) to prevent runaway division at inference time.

---

## Conclusion and Audit Sign-Off

The CEFM Step 4 Production Feature Distribution and Clinical Scoring Integrity Audit is complete:
- **Table 1** fully documents the 16-D feature vectors across HAM10000 train, val, test, PH2, and Production.
- **OOD Shift** at indices 11 and 12 is fully quantified, explaining the $598,000$ and $1,000,000$ input scaling in M6.
- **LayerNorm Containment** explains why M6 remains numerically stable despite astronomical inputs.
- **Counterfactual Sensitivity** proves that visual features dominate classification ($0.02\%$ P1, $0.87\%$ M6) and that both models robustly predict `non_mel` across all conditions.
- **ABCDE Discrepancy** ($0.1182$ vs $0.3884$) is fully reconciled with zero impact on categorical risk labels (`LOW`).
- **Zero code, weights, configs, or thresholds were modified**, strictly maintaining system integrity.

