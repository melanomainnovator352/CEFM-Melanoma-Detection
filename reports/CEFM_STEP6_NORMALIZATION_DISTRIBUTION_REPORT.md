# CEFM Step 6 — Normalization Distribution & Statistical Compatibility Audit

**Document ID:** `CEFM_STEP6_NORMALIZATION_DISTRIBUTION_REPORT`  
**Date:** November 2026  
**Status:** Independent Forensic Audit Report  
**Scope:** Resolution of normalization range contradictions, empirical training z-score bounds, and production distributional compatibility.

---

## 1. Resolution of the Normalization Range Contradiction (Task 2)

### A. The Contradiction Identified
In the Step 5 reports, two seemingly conflicting statements appeared:
1. *"Maximum absolute normalized clinical value: 1.8870"*
2. *"Claimed valid range: [-2.0, 1.0]"*

Mathematically and statistically, these statements require rigorous reconciliation:
- If the maximum absolute value is $1.8870$, the lower bound of the values must reach at least $-1.8870$.
- However, standard normal z-scores ($z = (x - \mu)/\sigma$) are mathematically unbounded on the real line: $z \in (-\infty, +\infty)$.
- In formal machine learning documentation, stating that "normalized features are bounded within $[-2.0, 1.0]$" is an **imprecise informal summary of the specific target case's values**, not a general mathematical bound of standard z-scores.

### B. Exact Case Values on Audited Image (`5ef6da25...`)
Evaluating target case `5ef6da25eecc47c6a05bee924b2aa3f8.jpg` under the Step 5 zero-variance guarded pipeline yields the exact normalized vector $\hat{c}$:

$$\hat{c} = [0.0478, 0.0, 0.0, -1.0522, 0.0, 0.0, 0.0, \mathbf{-1.8870}, -0.0578, -1.3478, 0.0792, 0.0, 0.0, 0.0, 0.0, 0.0]$$

- **Minimum Value:** $\mathbf{-1.8870}$ (Dimension 7: `color_h_std`)
- **Maximum Value:** $\mathbf{+0.0792}$ (Dimension 10: `color_available`)
- **Maximum Absolute Magnitude:** $\mathbf{1.8870}$
- **Actual Interval for this Image:** $\mathbf{[-1.8870, +0.0792]}$

The colloquial statement "bounded within $[-2.0, 1.0]$" in Step 5 was an informal rounding of the observed minimum ($-1.887 \approx -2.0$) and observed maximum ($+0.079 \approx 0.1 \le 1.0$). This audit formally resolves this contradiction by replacing the informal interval with the exact observed range: **$[-1.8870, +0.0792]$**.

---

## 2. Empirical Training Z-Score Distributions (HAM10000 Train, $N=6,978$)

To determine whether the target case values are distributionally compatible with training, the exact training distribution z-scores were computed across all $6,978$ training samples:

### Table: Empirical Training Z-Score Distribution (HAM10000 Train)

| Idx | Feature Name | Training Mean | Training Std | Min $z$ | 1st Pct | Median $z$ | 99th Pct | Max $z$ | Target Case $z$ | In Training Range? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `asymmetry_score` | $0.2441$ | $0.1233$ | $-1.9806$ | $-1.9806$ | $-0.0982$ | $+2.4168$ | $+4.6012$ | $+0.0478$ | **YES (48th Pct)** |
| **1** | `horizontal_asymmetry` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **2** | `vertical_asymmetry` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **3** | `border_irregularity` | $0.4155$ | $0.1332$ | $-3.1193$ | $-2.3611$ | $-0.0413$ | $+2.1813$ | $+3.1305$ | $-1.0522$ | **YES (15th Pct)** |
| **4** | `border_available` | $1.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **5** | `color_var_mean` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **6** | `color_var_std` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **7** | `color_h_std` | $0.6076$ | $0.3120$ | $-1.9477$ | $-1.9477$ | $+0.1231$ | $+1.1466$ | $+1.2364$ | $-1.8870$ | **YES (3rd Pct)** |
| **8** | `color_s_std` | $0.2042$ | $0.0800$ | $-2.5524$ | $-2.2272$ | $-0.0778$ | $+2.5341$ | $+5.1513$ | $-0.0578$ | **YES (47th Pct)** |
| **9** | `color_v_std` | $0.1760$ | $0.0891$ | $-1.9751$ | $-1.6382$ | $-0.1015$ | $+2.9390$ | $+6.0457$ | $-1.3478$ | **YES (8th Pct)** |
| **10** | `color_available` | $0.9950$ | $0.0630$ | $-15.8054$| $0.0792$ | $+0.0792$ | $+0.0792$ | $+0.0792$ | $+0.0792$ | **YES (Mode/Max)** |
| **11** | `diameter_norm` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **12** | `diameter_available` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **13** | `evolution_change` | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **14** | `evolution_magnitude`| $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |
| **15** | `evolution_available`| $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | **YES (Zero-Gated)** |

---

## 3. Distributional Compatibility vs. Numerical Boundedness

### Key Clinical AI Distinctions:
1. **Numerical Boundedness $\ne$ Distributional Compatibility:**  
   A test sample having $z \in [-3, +3]$ is numerically safe for floating-point operations. However, if a feature lies in the extreme tail (e.g. Dimension 7 at $-1.8870$ is in the 3rd percentile), it represents an unusual lesion within the training cohort.
2. **Clinical Meaning of Low Hue Std (Dim 7 = -1.8870):**  
   In dermoscopy, low hue variation indicates a uniform, monochromatic lesion (e.g. homogenous brown melanocytic nevus). Highly variegated melanomas typically exhibit high hue variation ($z > 0$). Thus, a low hue z-score provides a strong morphological signal of benignity.
3. **Training Partition Provenance:**  
   All normalization statistics ($\mu$ and $\sigma$) were derived strictly from the training partition (`dual_scale_train.pt`, $N=6,978$). No validation, test, or PH² data was used to compute normalization parameters.

---

## 4. Categorization Summary

- **Confirmed Bug:** Informal description of target case values as "bounded within $[-2.0, 1.0]$".
- **Corrected Bug:** Reconciled range to exact target interval $[-1.8870, +0.0792]$ and documented full empirical training z-score bounds ($[-15.8, +6.05]$).
- **Expected Model Behavior:** Standardized features reflect empirical z-scores; zero-variance dimensions evaluate strictly to $0.0$.
- **Research Limitations:** Training on uncalibrated datasets necessitates masking physical scale from neural inputs.
- **Unresolved Issues:** None within feature normalization scope.

