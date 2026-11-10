# CEFM Phase 3 Experimental Protocol: Image-Level Visual Domain Adaptation & Color Constancy

**Protocol Version:** 1.0  
**Phase Target:** CEFM Phase 3 Research Program  
**Objective:** Resolve external domain shift between acquisition devices (FotoFinder vs. Tuebinger Mole Analyzer) to restore external melanoma sensitivity while preserving internal test discrimination.

---

## 1. Scientific Rationale & Hypotheses

### 1.1 Problem Statement
In Phase 2 Step 5–7 audits, M6 Dual-Scale Gated RCAG demonstrated high internal discrimination on HAM10000 (ROC-AUC $0.9013$, Sensitivity $66.27\%$, Specificity $92.30\%$), but suffered an acute sensitivity drop to $30.00\%$ on external PH² dermoscopy. Step 7 causal analysis proved that this collapse is driven by **visual patch embedding divergence** in the Vision Transformer backbone rather than clinical feature shift.

### 1.2 Hypotheses
- **Primary Hypothesis ($H_1$):** Applying image-level optical normalization (e.g., Shades of Gray color constancy or Reinhard perceptual color transfer) prior to patch tiling will align external visual patch embeddings with the HAM10000 training manifold, increasing external melanoma sensitivity on PH² by $\ge 15.0\%$ absolute ($30.0\% \to \ge 45.0\%$) without decreasing internal specificity by more than $2.0\%$.
- **Null Hypothesis ($H_0$):** Image-level color normalization produces no statistically significant improvement in external melanoma sensitivity ($p \ge 0.05$ via McNemar test).

---

## 2. Experimental Arms & Ablation Matrix

We define five controlled experimental variants evaluated under strictly identical model architectures:

| Arm ID | Intervention Category | Algorithm / Transformation | Pipeline Stage | Description |
| :---: | :--- | :--- | :--- | :--- |
| **Arm 0** | **Negative Control** | Standard Bilinear Resize | Input Preprocessing | Frozen Phase 1 / M6 baseline with ImageNet standardization only. |
| **Arm 1** | **Shades of Gray ($p=6$)** | Minkowski Norm Color Constancy | Input Preprocessing | Estimates scene illuminant via $e(x) = \left(\int |\nabla f(x)|^p dx\right)^{1/p}$ and transforms RGB channels to neutral gray. |
| **Arm 2** | **Reinhard Color Transfer** | Perceptual LAB Statistical Transfer | Input Preprocessing | Shifts image color distribution in Lab space to match a canonical HAM10000 dermoscopic reference centroid. |
| **Arm 3** | **Contrast Limited AHE (CLAHE)**| Adaptive Histogram Equalization | Input Preprocessing | Enhances local edge contrast in luminance (L) channel with $2.0$ clip limit. |
| **Arm 4** | **Test-Time Adaptation (TTA)** | Tent (Entropy Minimization) | Inference Pass | Updates BatchNorm/LayerNorm affine parameters on unlabeled target batches via Shannon entropy minimization. |

---

## 3. Training & Validation Protocol

1. **Cohort Partitions:**
   - **Training:** HAM10000 Train ($N = 6,978$)
   - **Validation:** HAM10000 Val ($N = 1,533$)
   - **Patient-Level Zero Leakage:** Preserved strictly via `lesion_id` grouping.
2. **External Test Cohort Protection:**
   - **PH² Cohort ($N = 200$, $40$ melanomas):** Strictly held out as a zero-touch external validation benchmark.
   - **Prohibition:** No training, no fine-tuning, no threshold tuning, and no hyperparameter search may access PH² images or labels.
3. **Operating Thresholds:**
   - Operating thresholds must be determined strictly on the HAM10000 validation split before evaluating on external test data.

---

## 4. Evaluation Metrics & Statistical Testing

All metrics will be reported with 2,000 bootstrap resamples for 95% confidence intervals:
- **Primary Discriminative Metrics:** ROC-AUC, PR-AUC.
- **Operating Point Metrics:** Sensitivity, Specificity, Precision, F1-score, Balanced Accuracy.
- **Significance Testing:**
  - DeLong test for paired ROC-AUC comparisons.
  - McNemar test (with continuity correction) on paired 2x2 contingency tables for melanoma classification ($N_{\text{MEL}} = 40$).
  - Significance criterion: $\alpha = 0.05$ (two-tailed).

---

## 5. Strict Production Safety & Rollback Conditions

Phase 1 models remain the production default throughout Phase 3 experimentation. An experimental arm may be considered for challenger promotion only if:
1. **External Generalization Gate:** Melanoma sensitivity on PH² reaches $\ge 45.0\%$ at specificity $\ge 90.0\%$.
2. **Internal Integrity Gate:** Internal test ROC-AUC on HAM10000 ($N=1,504$) does not decrease by more than $0.010$ relative to Phase 1 baselines ($0.8808$ for ViT, $0.8889$ for Multimodal).
3. **Rollback Trigger:** If any experimental arm causes internal test sensitivity to fall below $54.0\%$ or specificity below $90.0\%$, the arm is disqualified and rolled back immediately.

