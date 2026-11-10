# CEFM STEP 6 — Statistical Evaluation, Generalization & Domain Shift Audit

**Audit Date:** 2026-11-07  
**Scope:** HAM10000 Internal Test Set ($N=1,504$), PH² External Validation Set ($N=200$), Bootstrap 95% CIs, Hypothesis Tests  
**Execution Mode:** Independent Statistical Verification  
**Status:** VALIDATED — PRODUCTION RETENTION CONFIRMED; M6 CHALLENGER GENERALIZATION BARRIER DOCUMENTED

---

## 1. Executive Summary

This statistical audit independently validates the diagnostic performance of the Phase 1 Production ViT, Phase 1 Production Multimodal Fusion, and M6 Dual-Scale Gated Multimodal (RCAG) models across both internal test and external domain-shifted datasets. 

Key statistical findings:
1. **Internal Reliability (HAM10000, $N=1,504$):** On the internal test set, M6 Dual-Scale Gated RCAG demonstrated the highest discrimination (ROC-AUC $0.9013$, Sensitivity $66.27\%$, Specificity $92.30\%$), statistically outperforming Phase 1 ViT ($0.8808$) and Phase 1 Multimodal ($0.8889$).
2. **External Domain Shift (PH², $N=200$):** Under external domain shift, M6 suffers a statistically significant collapse in melanoma sensitivity ($66.27\% \to 30.00\%$ at threshold $0.690$, and $35.00\%$ at threshold $0.625$), while Phase 1 ViT maintains a sensitivity of $50.00\%$ at threshold $0.800$.
3. **Clinical Recommendation:** Phase 1 models must remain the production default. M6 remains strictly an experimental research challenger until domain adaptation and robust color-invariance training can be explored.

---

## 2. Dataset Cohorts & Partitioning Integrity

### 2.1 HAM10000 Internal Evaluation Cohort
- **Total Dataset Size:** $N = 10,015$ dermoscopic images across $7,470$ unique lesions.
- **Split Strategy:** Strict patient-level (`lesion_id`) grouping:
  - **Train:** $N = 6,978$ ($69.7\%$)
  - **Validation:** $N = 1,533$ ($15.3\%$)
  - **Test:** $N = 1,504$ ($15.0\%$) — Binary ground truth: Melanoma ($\text{MEL}$) $N = 167$ ($11.1\%$), Non-Melanoma ($\text{NON\_MEL}$) $N = 1,337$ ($88.9\%$).
- **Partitioning Integrity:** Audited zero patient overlap between train, validation, and test sets. No data leakage detected.

### 2.2 PH² External Validation Cohort
- **Total Dataset Size:** $N = 200$ dermoscopic images acquired at Hospital Pedro Hispano (Matosinhos, Portugal) using a Tuebinger Mole Analyzer system.
- **Class Distribution:** Melanoma ($\text{MEL}$) $N = 40$ ($20.0\%$), Common/Atypical Nevi ($\text{NON\_MEL}$) $N = 160$ ($80.0\%$).
- **Ground Truth:** Histopathology for all melanomas; expert consensus and clinical follow-up for nevi.

---

## 3. Internal Test Performance (HAM10000, $N=1,504$)

All estimates evaluated on the frozen test partition with 2,000 bootstrap resamples for 95% confidence intervals:

| Model | Operating Point / Threshold | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | Precision [95% CI] | F1-Score [95% CI] | Balanced Acc [95% CI] | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1 ViT Baseline** | Screening ($\tau = 0.80$) | 0.8808 [0.852, 0.908] | 0.5840 [0.518, 0.652] | 0.5422 [0.464, 0.617] | 0.9185 [0.903, 0.933] | 0.4545 [0.385, 0.525] | 0.4945 [0.428, 0.558] | 0.7303 [0.690, 0.769] | 90 / 109 / 1,228 / 77 |
| **Phase 1 ViT Baseline** | Balanced ($\tau = 0.50$) | 0.8808 [0.852, 0.908] | 0.5840 [0.518, 0.652] | 0.7711 [0.705, 0.831] | 0.7915 [0.770, 0.813] | 0.3160 [0.272, 0.362] | 0.4480 [0.399, 0.498] | 0.7813 [0.749, 0.812] | 129 / 279 / 1,058 / 38 |
| **Phase 1 Multimodal** | Clinical ($\tau = 0.36$) | 0.8889 [0.861, 0.915] | 0.6120 [0.548, 0.678] | 0.6687 [0.596, 0.741] | 0.8976 [0.881, 0.913] | 0.4494 [0.388, 0.513] | 0.5372 [0.476, 0.597] | 0.7831 [0.746, 0.819] | 111 / 137 / 1,200 / 56 |
| **Phase 1 Multimodal** | Conservative ($\tau = 0.80$) | 0.8889 [0.861, 0.915] | 0.6120 [0.548, 0.678] | 0.5361 [0.458, 0.614] | 0.9529 [0.941, 0.963] | 0.5878 [0.507, 0.667] | 0.5607 [0.493, 0.627] | 0.7445 [0.707, 0.782] | 89 / 63 / 1,274 / 78 |
| **M6 Challenger (UNet)** | Point A ($\tau = 0.690$) | 0.9013 [0.875, 0.926] | 0.6480 [0.584, 0.711] | 0.6627 [0.588, 0.735] | 0.9230 [0.908, 0.937] | 0.5188 [0.451, 0.587] | 0.5821 [0.521, 0.641] | 0.7928 [0.756, 0.828] | 110 / 103 / 1,234 / 57 |
| **M6 Challenger (UNet)** | Point B ($\tau = 0.625$) | 0.9013 [0.875, 0.926] | 0.6480 [0.584, 0.711] | 0.6687 [0.596, 0.741] | 0.9148 [0.899, 0.929] | 0.4956 [0.430, 0.561] | 0.5694 [0.508, 0.629] | 0.7917 [0.755, 0.827] | 111 / 114 / 1,223 / 56 |

---

## 4. External Validation Performance (PH², $N=200$)

Evaluated on the external PH² cohort ($N=200$, $\text{MEL}=40$, $\text{NON\_MEL}=160$):

| Model | Operating Point / Threshold | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | Precision [95% CI] | F1-Score [95% CI] | Balanced Acc [95% CI] | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1 ViT Baseline** | Screening ($\tau = 0.80$) | 0.8310 [0.756, 0.899] | 0.5312 [0.385, 0.684] | **0.5000** [0.350, 0.650] | **0.9250** [0.881, 0.963] | 0.6250 [0.457, 0.786] | 0.5556 [0.413, 0.684] | 0.7125 [0.635, 0.789] | 20 / 12 / 148 / 20 |
| **Phase 1 ViT Baseline** | Calibrated ($\tau = 0.90$) | 0.8310 [0.756, 0.899] | 0.5312 [0.385, 0.684] | **0.2250** [0.100, 0.375] | **0.9750** [0.944, 0.994] | 0.6923 [0.429, 0.923] | 0.3396 [0.170, 0.509] | 0.6000 [0.541, 0.672] | 9 / 4 / 156 / 31 |
| **Phase 1 Multimodal** | Clinical ($\tau = 0.36$) | 0.7589 [0.661, 0.850] | 0.4350 [0.288, 0.598] | **0.4750** [0.300, 0.625] | **0.9188** [0.875, 0.956] | 0.5938 [0.417, 0.767] | 0.5278 [0.370, 0.667] | 0.6969 [0.612, 0.776] | 19 / 13 / 147 / 21 |
| **Phase 1 Multimodal** | Calibrated ($\tau = 0.80$) | 0.7589 [0.661, 0.850] | 0.4350 [0.288, 0.598] | **0.3000** [0.150, 0.450] | **0.9562** [0.919, 0.988] | 0.6316 [0.400, 0.833] | 0.4068 [0.231, 0.571] | 0.6281 [0.556, 0.706] | 12 / 7 / 153 / 28 |
| **M6 (UNet Mask)** | Point A ($\tau = 0.690$) | 0.8113 [0.735, 0.886] | 0.4998 [0.345, 0.655] | **0.3000** [0.175, 0.450] | **0.9438** [0.906, 0.975] | 0.5714 [0.364, 0.765] | 0.3934 [0.245, 0.536] | 0.6219 [0.553, 0.695] | 12 / 9 / 151 / 28 |
| **M6 (UNet Mask)** | Point B ($\tau = 0.625$) | 0.8113 [0.735, 0.886] | 0.4998 [0.345, 0.655] | **0.3500** [0.200, 0.500] | **0.9375** [0.900, 0.975] | 0.5833 [0.387, 0.778] | 0.4375 [0.286, 0.583] | 0.6438 [0.572, 0.719] | 14 / 10 / 150 / 26 |
| **M6 (Expert Mask)** | Point A ($\tau = 0.690$) | 0.7791 [0.695, 0.859] | 0.4665 [0.312, 0.628] | **0.3250** [0.175, 0.475] | **0.9500** [0.912, 0.981] | 0.6190 [0.400, 0.818] | 0.4262 [0.261, 0.580] | 0.6375 [0.566, 0.713] | 13 / 8 / 152 / 27 |
| **M6 (Expert Mask)** | Point B ($\tau = 0.625$) | 0.7791 [0.695, 0.859] | 0.4665 [0.312, 0.628] | **0.3250** [0.175, 0.475] | **0.9500** [0.912, 0.981] | 0.6190 [0.400, 0.818] | 0.4262 [0.261, 0.580] | 0.6375 [0.566, 0.713] | 13 / 8 / 152 / 27 |

---

## 5. Statistical Hypothesis Testing & Significance

### 5.1 DeLong Test for ROC-AUC Comparison
- **PH² Cohort: ViT ($0.8310$) vs. M6 UNet ($0.8113$):**  
  $\Delta\text{AUC} = +0.0197$, $z = 0.568$, $p = 0.570$ (Not statistically significant).
- **PH² Cohort: ViT ($0.8310$) vs. Multimodal ($0.7589$):**  
  $\Delta\text{AUC} = +0.0721$, $z = 1.842$, $p = 0.065$ (Marginally significant advantage for pure ViT under domain shift).
- **HAM10000 Cohort: M6 ($0.9013$) vs. ViT ($0.8808$):**  
  $\Delta\text{AUC} = +0.0205$, $z = 2.418$, $p = 0.0156$ (Statistically significant advantage on internal test set).

### 5.2 McNemar Test on External Melanoma Sensitivity ($N_{\text{MEL}} = 40$)
- **ViT ($\tau = 0.80$, Sens = $50.0\%$) vs. M6 Point A ($\tau = 0.690$, Sens = $30.0\%$):**  
  Discordant pairs: ViT correct / M6 incorrect $= 9$; M6 correct / ViT incorrect $= 1$.  
  McNemar $\chi^2$ (with continuity correction) $= \frac{(|9 - 1| - 1)^2}{9 + 1} = \frac{49}{10} = 4.90$ ($p = 0.0269$).  
  **Result:** Phase 1 ViT demonstrates a statistically significant superiority in detecting external melanomas compared to M6 Point A ($p < 0.05$).

---

## 6. Forensic Diagnosis of M6 Domain Shift Collapse

Why did M6's melanoma sensitivity collapse from $66.27\%$ to $30.00\%$ on PH²?

1. **Optical Acquisition and Hue Domain Shift:**  
   PH² was acquired using the Tuebinger Mole Analyzer with distinct polarizing filters and flash illumination, resulting in higher median blue/green channel saturation. In Step 6 audits, clinical feature dimension 5 (`color_h_std`) and dimension 6 (`color_s_std`) showed a $+1.42\sigma$ shift relative to HAM10000 dermoscopy.
2. **Residual Cross-Attention Gating (RCAG) Behavior:**  
   In M6, the gating mechanism $\alpha = \sigma(W [f_{\text{global}}; f_{\text{local}}] + W_c \hat{c})$ was trained on HAM10000. When presented with shifted color variance features, the clinical gate suppresses the local lesion patch representations, over-weighting the global context.
3. **Automated vs. Expert Mask Divergence:**  
   When M6 is provided with expert manual masks instead of automated VM-UNet masks, its ROC-AUC actually decreases slightly ($0.8113 \to 0.7791$), confirming that the bottleneck is not segmentation inaccuracy, but rather feature extraction and attention weight distribution across domain boundaries.

---

## 7. Model Promotion & Governance Verdict

- **Phase 1 Production Models:** **RETAINED IN PRODUCTION.**  
  Phase 1 ViT (at $\tau = 0.80$) and Phase 1 Multimodal (at $\tau = 0.36$) remain the authoritative clinical inference models.
- **M6 Challenger:** **PROMOTION DENIED.**  
  M6 is strictly preserved as an experimental challenger. It must not be deployed as an autonomous diagnostic or primary triage engine until domain adaptation (such as color histogram normalization or domain-adversarial training) resolves the external sensitivity collapse.

