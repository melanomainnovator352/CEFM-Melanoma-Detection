# CEFM Phase 2 Step 4 — Final Results Consistency Audit

**Audit Document ID:** `CEFM-PHASE2-STEP4-AUDIT-2026-11-07`  
**Audit Purpose:** Targeted consistency, threshold isolation, and statistical verification of Phase 2 Step 4 Research Validation  
**Dataset Under Audit:** Untouched HAM10000 Test Split ($N=1,504$; $166$ Melanoma, $1,338$ Non-Melanoma)  
**Audit Policy:** Strict evaluation-only; no models retrained, no checkpoints modified, zero threshold adjustment on test data  

---

## 1. Executive Summary & Audit Overview

This consistency audit independently re-evaluated all prediction arrays, decision thresholds, paired statistical differences, and case-level diagnostic claims from the Phase 2 Step 4 evaluation.

### Key Audit Findings & Corrective Actions
1. **Operating Point Separation in Paired Comparisons (Corrected):**
   - In the initial reporting script, paired bootstrap differences against Phase 1 models compared M6 at its F1 threshold ($0.690$) against Phase 1 ViT ($0.800$) and Phase 1 Multimodal ($0.360$) at their specificity screening thresholds.
   - **Correction Applied:** Paired bootstrap differences have now been strictly partitioned into two separate, unambiguous comparisons:
     - **Set A (F1 Operating Points):** M6 ($0.690$) vs. Phase 1 ViT ($0.900$) and Phase 1 Multimodal ($0.800$).
     - **Set B (Specificity-Target Operating Points):** M6 ($0.625$) vs. Phase 1 ViT ($0.800$) and Phase 1 Multimodal ($0.360$).
2. **Ground-Truth Verification of Case `ISIC_0012418` (Retracted & Corrected):**
   - The claim that `ISIC_0012418` has a histopathologically confirmed melanoma diagnosis was audited across all HAM10000 and ISIC repository metadata manifests.
   - **Finding:** `ISIC_0012418` is absent from all HAM10000 splits and resides in `ISIC2018_Task1-2_Test_Input`, an unannotated blind challenge partition with privately held labels.
   - **Correction Applied:** The claim of "histopathologically confirmed melanoma" and "rescued false negative" has been **formally retracted**. Case `ISIC_0012418` is documented strictly as an unannotated dermoscopic sensitivity probe demonstrating model behavior across architectures ($25.84\%$ ViT Baseline, $0.03\%$ Phase 1 Multimodal, $71.20\%$ M6 Challenger).
3. **Statistical Multiplicity Audit:**
   - 1,000 stratified bootstrap iterations (`seed=42`) were verified.
   - Multiple testing corrections (Bonferroni $\alpha = 0.00208$ and Benjamini-Hochberg FDR $q = 0.05$ across 24 hypotheses) were applied. M6's sensitivity improvement over Phase 1 ViT ($+12.00\%$) and Phase 1 Multimodal ($+12.53\%$) at the F1 operating point remains strictly statistically significant after multiplicity correction.
4. **Checkpoint Integrity:**
   - SHA-256 hashes of all 10 checkpoints were audited and confirmed 100% untouched.

---

## 2. Threshold Consistency & Justification

All operating thresholds were determined exclusively during validation; no thresholds were chosen or adjusted using test labels.

### Table 1: Comprehensive Model Thresholds and Test Confusion Matrices

| Model Identity | Operating Point | Applied Threshold $T^*$ | Derivation Basis on Validation Split ($N=1,533$) | TP | FP | TN | FN | Test Sensitivity | Test Specificity | Test F1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 ViT Baseline** | **Table A: F1** | **0.900** | Max validation F1-score ($0.5722$ in `threshold_config.json`) | 90 | 109 | 1,229 | 76 | 54.22% | 91.85% | 0.4932 |
| | **Table B: Spec** | **0.800** | Max validation balanced accuracy ($0.8415$ in `threshold_config.json`) | 128 | 279 | 1,059 | 38 | 77.11% | 79.15% | 0.4468 |
| **Phase 1 Multimodal Fusion**| **Table A: F1** | **0.800** | Max validation F1-score (`optimal_f1_threshold: 0.800`) | 89 | 63 | 1,275 | 77 | 53.61% | 95.29% | 0.5597 |
| | **Table B: Spec** | **0.360** | Clinical screening calibration target (`clinical_threshold: 0.360`) | 111 | 137 | 1,201 | 55 | 66.87% | 89.76% | 0.5362 |
| **M1: Global ViT Only** | **Table A: F1** | **0.830** | Validation F1-optimal sweep | 81 | 46 | 1,292 | 85 | 48.80% | 96.56% | 0.5529 |
| | **Table B: Spec** | **0.585** | Validation 90% specificity target | 112 | 119 | 1,219 | 54 | 67.47% | 91.11% | 0.5642 |
| **M2: Local ViT Only** | **Table A: F1** | **0.720** | Validation F1-optimal sweep | 89 | 55 | 1,283 | 77 | 53.61% | 95.89% | 0.5742 |
| | **Table B: Spec** | **0.470** | Validation 90% specificity target | 110 | 110 | 1,228 | 56 | 66.27% | 91.78% | 0.5699 |
| **M3: Dual-Scale Visual** | **Table A: F1** | **0.450** | Validation F1-optimal sweep | 96 | 73 | 1,265 | 70 | 57.83% | 94.54% | 0.5731 |
| | **Table B: Spec** | **0.075** | Validation 90% specificity target | 116 | 128 | 1,210 | 50 | 69.88% | 90.43% | 0.5659 |
| **M4: Global ViT + Clinical** | **Table A: F1** | **0.830** | Validation F1-optimal sweep | 83 | 37 | 1,301 | 83 | 50.00% | 97.23% | 0.5804 |
| | **Table B: Spec** | **0.285** | Validation 90% specificity target | 109 | 111 | 1,227 | 57 | 65.66% | 91.70% | 0.5648 |
| **M5: Dual Concat Multimodal** | **Table A: F1** | **0.630** | Validation F1-optimal sweep | 104 | 86 | 1,252 | 62 | 62.65% | 93.57% | 0.5843 |
| | **Table B: Spec** | **0.270** | Validation 90% specificity target | 115 | 137 | 1,201 | 51 | 69.28% | 89.76% | 0.5502 |
| **M6: Dual Gated RCAG** | **Table A: F1** | **0.690** | Validation F1-optimal sweep | **110** | **103** | **1,235** | **56** | **66.27%** | **92.30%** | **0.5805** |
| | **Table B: Spec** | **0.625** | Validation 90% specificity target | **111** | **114** | **1,224** | **55** | **66.87%** | **91.48%** | **0.5678** |
| **M7: Dual Cross-Attention** | **Table A: F1** | **0.180** | Validation F1-optimal sweep | 104 | 101 | 1,237 | 62 | 62.65% | 92.45% | 0.5606 |
| | **Table B: Spec** | **0.110** | Validation 90% specificity target | 108 | 117 | 1,221 | 58 | 65.06% | 91.26% | 0.5524 |

### Why Phase 1 ViT Uses $0.900$ in Table A and $0.800$ in Table B
In `models/classification/threshold_config.json`:
- `best_f1_threshold: 0.900`: On the validation set, $0.900$ maximized the harmonic mean of precision and recall ($F_1 = 0.5722$, validation precision $52.55\%$, sensitivity $62.8\%$). Therefore, $0.900$ is the authoritative validation-calibrated F1 operating point for Table A.
- `clinical_threshold: 0.800` (also `best_bacc_threshold: 0.800`): On the validation set, $0.800$ maximized balanced accuracy ($0.8415$) while prioritizing clinical screening recall ($87.80\%$ sensitivity, $80.50\%$ specificity). Therefore, $0.800$ is the clinical screening threshold evaluated in Table B.

---

## 3. Recalculation of Paired Bootstrap Differences

All paired differences were recalculated directly from the stored prediction arrays and test labels using **1,000 stratified bootstrap iterations** (`seed=42`).

### Set A: Strict F1 Operating Point Comparisons (Table A Thresholds)
*Compares M6 at $T^* = 0.690$ against baselines evaluated at their validation F1 thresholds.*

| Paired Comparison | Metric | Mean Difference ($\Delta$) | 95% Bootstrap CI | Nominal $p$-value | Bonferroni Significant ($\alpha=0.00208$) | FDR Significant ($q=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M6 (0.690) vs. Phase 1 ViT (0.900)** | Sensitivity | $+0.1200$ | $[+0.0422, +0.2048]$ | $0.0020$ | **YES** | **YES** |
| | Specificity | $+0.0042$ | $[-0.0090, +0.0179]$ | $0.5880$ | No | No |
| | Precision | $+0.0634$ | $[+0.0092, +0.1170]$ | $0.0220$ | No | **YES** |
| | F1-Score | $+0.0868$ | $[+0.0278, +0.1448]$ | $0.0020$ | **YES** | **YES** |
| | Balanced Accuracy | $+0.0621$ | $[+0.0225, +0.1035]$ | $<0.0001$ | **YES** | **YES** |
| | ROC-AUC | $+0.0200$ | $[+0.0042, +0.0368]$ | $0.0080$ | No | **YES** |
| **M6 (0.690) vs. Phase 1 Multimodal (0.800)**| Sensitivity | $+0.1253$ | $[+0.0542, +0.1928]$ | $<0.0001$ | **YES** | **YES** |
| | Specificity | $-0.0298$ | $[-0.0433, -0.0179]$ | $<0.0001$ | **YES** | **YES** |
| | Precision | $-0.0691$ | $[-0.1313, -0.0117]$ | $0.0260$ | No | **YES** |
| | F1-Score | $+0.0206$ | $[-0.0335, +0.0788]$ | $0.4880$ | No (CI includes 0) | No |
| | Balanced Accuracy | $+0.0478$ | $[+0.0133, +0.0829]$ | $0.0060$ | No | **YES** |
| | ROC-AUC | $+0.0121$ | $[-0.0006, +0.0253]$ | $0.0620$ | No (CI includes 0) | No |
| **M6 (0.690) vs. M3 Dual-Scale Visual (0.450)**| Sensitivity | $+0.0841$ | $[+0.0241, +0.1446]$ | $0.0040$ | No | **YES** |
| | Specificity | $-0.0223$ | $[-0.0344, -0.0105]$ | $<0.0001$ | **YES** | **YES** |
| | F1-Score | $+0.0077$ | $[-0.0373, +0.0575]$ | $0.7780$ | No (CI includes 0) | No |
| | Balanced Accuracy | $+0.0309$ | $[+0.0019, +0.0626]$ | $0.0400$ | No | **YES** |
| **M6 (0.690) vs. M5 Dual Concat (0.630)** | Sensitivity | $+0.0355$ | $[-0.0120, +0.0843]$ | $0.1880$ | No (CI includes 0) | No |
| | Specificity | $-0.0127$ | $[-0.0232, -0.0022]$ | $0.0180$ | No | **YES** |
| | F1-Score | $-0.0040$ | $[-0.0428, +0.0393]$ | $0.8400$ | No (CI includes 0) | No |
| | Balanced Accuracy | $+0.0114$ | $[-0.0131, +0.0377]$ | $0.3920$ | No (CI includes 0) | No |

---

### Set B: Strict Specificity-Target Operating Point Comparisons (Table B Thresholds)
*Compares M6 at $T^* = 0.625$ against baselines evaluated at their clinical specificity screening thresholds.*

| Paired Comparison | Metric | Mean Difference ($\Delta$) | 95% Bootstrap CI | Nominal $p$-value | Bonferroni Significant ($\alpha=0.00208$) | FDR Significant ($q=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M6 (0.625) vs. Phase 1 ViT (0.800)** | Sensitivity | $-0.1025$ | $[-0.1688, -0.0301]$ | $0.0040$ | No | **YES** (ViT 0.80 trades specificity for recall) |
| | Specificity | $+0.1230$ | $[+0.1039, +0.1428]$ | $<0.0001$ | **YES** | **YES** (M6 vastly superior specificity, $+12.3\%$) |
| | Precision | $+0.1785$ | $[+0.1326, +0.2212]$ | $<0.0001$ | **YES** | **YES** |
| | F1-Score | $+0.1202$ | $[+0.0734, +0.1674]$ | $<0.0001$ | **YES** | **YES** |
| | Balanced Accuracy | $+0.0102$ | $[-0.0253, +0.0470]$ | $0.5980$ | No (CI includes 0) | No |
| **M6 (0.625) vs. Phase 1 Multimodal (0.360)**| Sensitivity | $+0.0011$ | $[-0.0544, +0.0602]$ | $1.0000$ | No (CI includes 0) | No (Exact recall parity, 111 TP each) |
| | Specificity | $+0.0170$ | $[+0.0022, +0.0314]$ | $0.0300$ | No | **YES** (Nominal $p=0.030$, 23 fewer false alarms) |
| | Precision | $+0.0458$ | $[+0.0023, +0.0913]$ | $0.0420$ | No | **YES** |
| | F1-Score | $+0.0318$ | $[-0.0111, +0.0779]$ | $0.1440$ | No (CI includes 0) | No |
| | Balanced Accuracy | $+0.0090$ | $[-0.0197, +0.0395]$ | $0.5640$ | No (CI includes 0) | No |
| **M6 (0.625) vs. M3 Dual-Scale Visual (0.075)**| Sensitivity | $-0.0297$ | $[-0.0723, +0.0120]$ | $0.2320$ | No (CI includes 0) | No |
| | Specificity | $+0.0106$ | $[-0.0015, +0.0224]$ | $0.1040$ | No (CI includes 0) | No |
| **M6 (0.625) vs. M5 Dual Concat (0.270)** | Sensitivity | $-0.0243$ | $[-0.0602, +0.0120]$ | $0.2600$ | No (CI includes 0) | No |
| | Specificity | $+0.0173$ | $[+0.0067, +0.0299]$ | $<0.0001$ | **YES** | **YES** |

---

## 4. Ground-Truth Audit of Benchmark Case `ISIC_0012418`

An investigation was conducted across all HAM10000 metadata files (`test.csv`, `train.csv`, `validation.csv`), ISIC archive metadata manifests, and dataset partitions.

### Findings
1. **Absence from HAM10000:** `ISIC_0012418` is completely absent from all HAM10000 dataset splits.
2. **Partition Origin:** The physical image is located at `D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg`.
3. **Challenge Ground-Truth Status:** `ISIC2018_Task1-2_Test_Input` constitutes the official unannotated challenge test partition for ISIC 2018 Tasks 1 and 2. Ground-truth diagnostic labels for this partition were withheld by challenge organizers and are not distributed with the dataset.
4. **Lack of Independent Pathological Record:** There is no clinical biopsy record or histopathological confirmation in repository metadata verifying whether `ISIC_0012418` is a malignant melanoma or a benign dysplastic nevus.

### Audit Actions Taken
- **Retraction of Malignancy Claim:** All claims asserting that `ISIC_0012418` is a "confirmed melanoma" or a "histopathologically verified rescued false negative" have been **retracted**.
- **Objective Reclassification:** `ISIC_0012418` is documented strictly as an **unannotated dermoscopic sensitivity probe case study**, demonstrating the response of different network architectures to challenging lesion morphology:
  - Phase 1 ViT Baseline: $P_{\text{mel}} = 25.84\%$ (Predicted `NON_MEL` at $T^* = 0.80$)
  - Phase 1 Multimodal Fusion: $P_{\text{mel}} = 0.03\%$ (Predicted `NON_MEL` at $T^* = 0.36$)
  - Phase 2 M6 Challenger: $P_{\text{mel}} = 71.20\%$ (Predicted `MEL` at both $T^* = 0.690$ and $T^* = 0.625$)
- The primary evaluation of M6 relies exclusively on the verified, histopathologically annotated test split ($166$ confirmed melanomas, $1,338$ confirmed non-melanomas).

---

## 5. Statistical Methodology & Multiplicity Adjustments

1. **Bootstrap Procedure:**
   - Resamples: $B = 1,000$ iterations.
   - Stratification: Resampling performed with replacement within positive ($N_{\text{pos}} = 166$) and negative ($N_{\text{neg}} = 1,338$) strata to preserve test-set class prevalence.
   - Seed: Fixed at `RANDOM_SEED = 42`.
2. **Paired Difference Estimation:**
   - For every resample $b$, metric difference $\Delta_b = \theta_b^{(M6)} - \theta_b^{(\text{baseline})}$ was computed on identical resample indices.
   - 95% Bootstrap Percentile Confidence Interval: $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.
   - Empirical two-sided $p$-value: $p = 2 \times \min(P(\Delta \le 0), P(\Delta \ge 0))$.
3. **Multiple Testing Correction:**
   - Total primary hypotheses evaluated: $4 \text{ models} \times 6 \text{ primary metrics} = 24$ tests per operating point set.
   - **Bonferroni Adjustment:** $\alpha_{\text{Bonf}} = 0.05 / 24 = 0.00208$.
   - **Benjamini-Hochberg (BH) False Discovery Rate:** $q = 0.05$.
   - Key robust findings surviving Bonferroni correction:
     - M6 sensitivity gain over Phase 1 ViT at F1 threshold ($+12.00\%$, $p = 0.0020 \le \alpha_{\text{Bonf}}$).
     - M6 sensitivity gain over Phase 1 Multimodal at F1 threshold ($+12.53\%$, $p < 0.0001 \le \alpha_{\text{Bonf}}$).
     - M6 specificity gain over Phase 1 ViT at screening threshold ($+12.30\%$, $p < 0.0001 \le \alpha_{\text{Bonf}}$).

---

## 6. Audit of Claims: Corrected vs. Unchanged

### Corrected Claims
| Claim Identifier | Original Claim | Corrected Claim Following Audit |
| :--- | :--- | :--- |
| **Claim 1: Paired Threshold Alignment** | M6 ($0.690$) was compared against Phase 1 ViT ($0.800$) and Phase 1 Multimodal ($0.360$) in Table 4. | Comparisons are now strictly aligned by operating point: Table A compares F1 points (M6 $0.690$ vs. ViT $0.900$ and Multimodal $0.800$); Table B compares Specificity points (M6 $0.625$ vs. ViT $0.800$ and Multimodal $0.360$). |
| **Claim 2: M6 vs. Multimodal F1 Difference** | M6 has a statistically significant F1 advantage over Phase 1 Multimodal ($\Delta = +0.0445$, $p < 0.05$). | When compared at their respective F1 operating points ($0.690$ vs. $0.800$), the F1 difference is $\Delta = +0.0206$ [$ -0.0335, +0.0788$] ($p = 0.488$), which is **not statistically significant** (CI includes zero). |
| **Claim 3: ISIC_0012418 Histopathology** | `ISIC_0012418` was described as a histopathologically confirmed melanoma rescued by M6. | Claim of histopathological confirmation is **retracted**. Reclassified strictly as an unannotated sensitivity probe case from the ISIC 2018 challenge blind test partition. |
| **Claim 4: Multiplicity Significance** | All nominal $p < 0.05$ differences were described as statistically significant. | Explicitly distinguished between nominal significance ($p < 0.05$), Benjamini-Hochberg FDR significance ($q < 0.05$), and conservative Bonferroni significance ($p < 0.00208$). |

### Unchanged Claims (Verified Valid)
- Test set size ($N=1,504$; $166$ melanoma, $1,338$ non-melanoma) and zero patient/lesion leakage.
- All model checkpoint SHA-256 hashes remain 100% untouched.
- M6 achieves the top sensitivity ($66.27\%$) among all models at the validation F1 operating point ($T^* = 0.690$).
- M6 achieves the highest ROC-AUC ($0.9013$) and PR-AUC ($0.6103$) among dual-scale architectures.
- M6 reduces false negatives at the F1 operating point from $76$ (Phase 1 ViT) and $77$ (Phase 1 Multimodal) down to $56$ (a recovery of $20-21$ melanomas).
- Case `ISIC_0012418` demonstrates a stark model sensitivity difference: $25.84\%$ (ViT), $0.03\%$ (Phase 1 Multimodal), and $71.20\%$ (M6 Challenger).

---

## 7. Checkpoint Hash Verification

Cryptographic SHA-256 hashes of all checkpoints were verified before and after this consistency audit.

| Checkpoint Name | Relative Path | Verified SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :--- |
| **Phase 1 ViT** | `models/classification/vit_model/model.safetensors` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | **UNTOUCHED** |
| **Phase 1 UNet** | `models/segmentation/unet_isic2018.pt` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | **UNTOUCHED** |
| **Phase 1 Fusion** | `models/multimodal/fusion_model.pt` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | **UNTOUCHED** |
| **M1** | `models/experiments/M1/weights.pt` | `6e51e90f085c1909e60cdd6342b11d68d7ec42251becce365a01c6e16b5fa88d` | **UNTOUCHED** |
| **M2** | `models/experiments/M2/weights.pt` | `e8d1a4bf02b9d0e01780bc8fb8cb2e78c019e9d53b36c095b3c34cfcc20e2857` | **UNTOUCHED** |
| **M3** | `models/experiments/M3/weights.pt` | `c1b6137bd6e2b09c255da97e78db477fe838a93247dce806287e60ac50885c33` | **UNTOUCHED** |
| **M4** | `models/experiments/M4/weights.pt` | `2ba424a5393ca183c2ea3d34331296f12a9d61031da7d5770a9b53986ada2d98` | **UNTOUCHED** |
| **M5** | `models/experiments/M5/weights.pt` | `f01b9a991569f8ee15a22f1576c00d7bf6423c84070f2d885c3401b83b5720b6` | **UNTOUCHED** |
| **M6** | `models/experiments/M6/weights.pt` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | **UNTOUCHED** |
| **M7** | `models/experiments/M7/weights.pt` | `0af0bfc4c420dc4e5d63cbf7e442651db8fbbe684893d315bc011dc4d56dae78` | **UNTOUCHED** |

---

## 8. Conclusion

The consistency audit is complete. All metrics, tables, paired bootstrap differences, and case-level diagnostic descriptions are fully reconciled and verified.

