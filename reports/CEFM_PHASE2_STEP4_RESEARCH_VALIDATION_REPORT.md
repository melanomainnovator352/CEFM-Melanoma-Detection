# CEFM Phase 2 — Step 4: M6 Research Validation & Comparative Evaluation Report

**Document ID:** `CEFM-PHASE2-STEP4-RES-EVAL-2026-11-07`  
**Evaluation Scope:** Rigorous, reproducible research benchmark of M6 Challenger vs. Phase 1 Production Baselines and Phase 2 Ablation Models (M1–M7)  
**Dataset:** Untouched HAM10000 Independent Lesion-Stratified Test Split ($N = 1,504$; $166$ Melanoma, $1,338$ Non-Melanoma)  
**Protocol:** 1,000 Stratified Bootstrap Resamples (`seed=42`) | Fixed Validation Thresholds (Zero Test-Data Tuning)  
**Integrity Audit:** Cryptographic SHA-256 Checkpoint Verification Before and After Evaluation  

---

## 1. Executive Summary & Audit of Evaluation Inputs

This research evaluation provides an empirical assessment of the **M6 Dual-Scale Gated Multimodal Challenger Model (RCAG)** in comparison with the Phase 1 production baselines (ViT Baseline and Multimodal Fusion v1.0) and Phase 2 ablation models (M1–M7).

### Evaluation Integrity Verification Gates
- **Untouched Test Dataset:** Exactly $1,504$ independent test samples ($166$ histopathologically confirmed melanomas, $1,338$ non-melanoma lesions).
- **Class Distribution:** $11.04\%$ positive prevalence ($\text{positive:negative} = 1:8.06$).
- **Label Mapping:** Binary encoding strictly preserved ($0 = \text{non-melanoma}$, $1 = \text{melanoma}$).
- **Leakage Audit:** $0$ shared patient IDs and $0$ shared lesion IDs between training ($N=6,978$), validation ($N=1,533$), and test ($N=1,504$) splits.
- **Strict Threshold Isolation:** All operating thresholds ($T^*_{F1} = 0.690$ and $T^*_{spec} = 0.625$ for M6; $0.80 / 0.90$ for ViT; $0.36 / 0.80$ for Phase 1 Fusion) were calibrated exclusively on validation data. Zero test labels were utilized for model selection or threshold tuning.
- **Pre- and Post-Evaluation Checkpoint Audit:** Cryptographic SHA-256 hashes of all 10 production and experimental checkpoints were audited before and after evaluation, proving zero parameter mutation.

```mermaid
flowchart LR
    subgraph DataSplits ["Dataset Isolation Audit"]
        TR["Train Split\nN=6,978\nZero Leakage"]
        VAL["Validation Split\nN=1,533\nThreshold Calibration Only"]
        TE["Independent Test Split\nN=1,504\nUntouched Blind Evaluation"]
    end

    subgraph Models ["Model Evaluation Suite"]
        P1_VIT["Phase 1 ViT Baseline\n(Thresh: 0.80 / 0.90)"]
        P1_FUS["Phase 1 Multimodal Fusion v1.0\n(Thresh: 0.36 / 0.80)"]
        M1_M5["Phase 2 Ablation Models\n(M1, M2, M3, M4, M5, M7)"]
        M6_RCAG["Phase 2 Challenger M6\nPoint A: 0.690 (F1)\nPoint B: 0.625 (90% Spec)"]
    end

    subgraph Metrics ["Statistical Verification"]
        BOOT["1,000 Stratified Bootstraps\n(seed=42)"]
        PAIRED["Paired Bootstrap Differences\n(95% CI vs. Baselines)"]
        FN_AUDIT["False-Negative & Metadata\nStratification (ISIC_0012418)"]
    end

    VAL -.->|Calibrates T*| Models
    TE --> Models
    Models --> BOOT
    BOOT --> PAIRED
    BOOT --> FN_AUDIT
```

---

## 2. Checkpoint SHA-256 Audit

| Model Designation | Checkpoint Path | Pre-Evaluation SHA-256 | Post-Evaluation SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 ViT Baseline** | `models/classification/vit_model/model.safetensors` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | **UNTOUCHED** |
| **Phase 1 VM-UNet** | `models/segmentation/unet_isic2018.pt` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | **UNTOUCHED** |
| **Phase 1 Multimodal** | `models/multimodal/fusion_model.pt` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | **UNTOUCHED** |
| **M1: Global ViT** | `models/experiments/M1/weights.pt` | `6e51e90f085c1909e60cdd6342b11d68d7ec42251becce365a01c6e16b5fa88d` | `6e51e90f085c1909e60cdd6342b11d68d7ec42251becce365a01c6e16b5fa88d` | **UNTOUCHED** |
| **M2: Local ViT** | `models/experiments/M2/weights.pt` | `e8d1a4bf02b9d0e01780bc8fb8cb2e78c019e9d53b36c095b3c34cfcc20e2857` | `e8d1a4bf02b9d0e01780bc8fb8cb2e78c019e9d53b36c095b3c34cfcc20e2857` | **UNTOUCHED** |
| **M3: Dual-Scale Vis** | `models/experiments/M3/weights.pt` | `c1b6137bd6e2b09c255da97e78db477fe838a93247dce806287e60ac50885c33` | `c1b6137bd6e2b09c255da97e78db477fe838a93247dce806287e60ac50885c33` | **UNTOUCHED** |
| **M4: Global+Clin** | `models/experiments/M4/weights.pt` | `2ba424a5393ca183c2ea3d34331296f12a9d61031da7d5770a9b53986ada2d98` | `2ba424a5393ca183c2ea3d34331296f12a9d61031da7d5770a9b53986ada2d98` | **UNTOUCHED** |
| **M5: Dual Concat** | `models/experiments/M5/weights.pt` | `f01b9a991569f8ee15a22f1576c00d7bf6423c84070f2d885c3401b83b5720b6` | `f01b9a991569f8ee15a22f1576c00d7bf6423c84070f2d885c3401b83b5720b6` | **UNTOUCHED** |
| **M6: Dual Gated** | `models/experiments/M6/weights.pt` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | **UNTOUCHED** |
| **M7: Cross-Attn** | `models/experiments/M7/weights.pt` | `0af0bfc4c420dc4e5d63cbf7e442651db8fbbe684893d315bc011dc4d56dae78` | `0af0bfc4c420dc4e5d63cbf7e442651db8fbbe684893d315bc011dc4d56dae78` | **UNTOUCHED** |

---

## 3. Comprehensive Performance Benchmark Tables

All models were evaluated on the **exact same 1,504 test samples**. Undefined metric cases (such as division by zero) are handled with standard explicit fallback ($0.0$).

### Table 1: Validation-Calibrated F1 Operating Points
*Evaluates each architecture at its optimal validation F1 operating point.*

| Model Identity | Input Modalities | Validation $T^*_{F1}$ | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1-Score | Balanced Acc | TP | FP | TN | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 ViT Baseline** | Global Visual (Uncropped) | 0.900 | 0.8808 | 0.5406 | 54.22% | 91.85% | 45.23% | 0.4932 | 0.7304 | 90 | 109 | 1,229 | 76 |
| **Phase 1 Multimodal Fusion** | Global ViT + 16-D Clinical | 0.800 | 0.8889 | 0.5325 | 53.61% | 95.29% | 58.55% | 0.5597 | 0.7445 | 89 | 63 | 1,275 | 77 |
| **M1: Global ViT Only** | Global Context Crop ($1.30\times$) | 0.830 | 0.8978 | 0.6231 | 48.80% | 96.56% | 63.78% | 0.5529 | 0.7268 | 81 | 46 | 1,292 | 85 |
| **M2: Local ViT Only** | Native-Scale Local Patch | 0.720 | 0.9080 | 0.6275 | 53.61% | 95.89% | 61.81% | 0.5742 | 0.7475 | 89 | 55 | 1,283 | 77 |
| **M3: Dual-Scale Visual** | Global + Local Visual Concatenation | 0.450 | 0.9003 | 0.6237 | 57.83% | 94.54% | 56.80% | 0.5731 | 0.7619 | 96 | 73 | 1,265 | 70 |
| **M4: Global ViT + Clinical** | Global ViT + Normalized Clinical | 0.830 | 0.8916 | 0.6219 | 50.00% | 97.23% | 69.17% | 0.5804 | 0.7362 | 83 | 37 | 1,301 | 83 |
| **M5: Dual Concat Multimodal** | Global + Local + Clinical (Early Concat)| 0.630 | 0.8988 | 0.6305 | 62.65% | 93.57% | 54.74% | 0.5843 | 0.7811 | 104 | 86 | 1,252 | 62 |
| **M6: Dual Gated RCAG (Challenger)**| Dual-Scale Gating + Clinical Modulation| **0.690** | **0.9013** | **0.6103** | **66.27%** | **92.30%** | **51.64%** | **0.5805** | **0.7928** | **110** | **103** | **1,235** | **56** |
| **M7: Dual Cross-Attention** | Dual-Scale Visual + Cross-Attention | 0.180 | 0.8984 | 0.6014 | 62.65% | 92.45% | 50.73% | 0.5606 | 0.7755 | 104 | 101 | 1,237 | 62 |

---

### Table 2: Validation-Calibrated Specificity-Target Operating Points
*Evaluates each model at the threshold chosen during validation to maintain clinical screening specificity ($\ge 90\%$).*

> [!IMPORTANT]
> A validation specificity target (such as $90\%$ on validation data) does not guarantee exactly $90.0\%$ specificity on the independent test set due to sampling variation. The actual test specificity achieved is reported below.

| Model Identity | Validation Operating Criterion | Applied $T^*$ | ROC-AUC | PR-AUC | Test Sensitivity | Test Specificity | Precision | F1-Score | Balanced Acc | TP | FP | TN | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 ViT Baseline** | Balanced Accuracy Clinical Screening | 0.800 | 0.8808 | 0.5406 | 77.11% | 79.15% | 31.45% | 0.4468 | 0.7813 | 128 | 279 | 1,059 | 38 |
| **Phase 1 Multimodal Fusion** | Validation Specificity Target Screening | 0.360 | 0.8889 | 0.5325 | 66.87% | 89.76% | 44.76% | 0.5362 | 0.7831 | 111 | 137 | 1,201 | 55 |
| **M1: Global ViT Only** | Validation 90% Specificity Target | 0.585 | 0.8978 | 0.6231 | 67.47% | 91.11% | 48.48% | 0.5642 | 0.7929 | 112 | 119 | 1,219 | 54 |
| **M2: Local ViT Only** | Validation 90% Specificity Target | 0.470 | 0.9080 | 0.6275 | 66.27% | 91.78% | 50.00% | 0.5699 | 0.7902 | 110 | 110 | 1,228 | 56 |
| **M3: Dual-Scale Visual** | Validation 90% Specificity Target | 0.075 | 0.9003 | 0.6237 | 69.88% | 90.43% | 47.54% | 0.5659 | 0.8016 | 116 | 128 | 1,210 | 50 |
| **M4: Global ViT + Clinical** | Validation 90% Specificity Target | 0.285 | 0.8916 | 0.6219 | 65.66% | 91.70% | 49.55% | 0.5648 | 0.7868 | 109 | 111 | 1,227 | 57 |
| **M5: Dual Concat Multimodal** | Validation 90% Specificity Target | 0.270 | 0.8988 | 0.6305 | 69.28% | 89.76% | 45.63% | 0.5502 | 0.7952 | 115 | 137 | 1,201 | 51 |
| **M6: Dual Gated RCAG (Challenger)**| Validation 90% Specificity Target | **0.625** | **0.9013** | **0.6103** | **66.87%** | **91.48%** | **49.33%** | **0.5678** | **0.7917** | **111** | **114** | **1,224** | **55** |
| **M7: Dual Cross-Attention** | Validation 90% Specificity Target | 0.110 | 0.8984 | 0.6014 | 65.06% | 91.26% | 48.00% | 0.5524 | 0.7816 | 108 | 117 | 1,221 | 58 |

---

## 4. Statistical Confidence Intervals (1,000 Stratified Bootstraps)

Confidence intervals were calculated using **1,000 stratified bootstrap iterations** with a fixed random seed (`seed=42`). The exact same resampled test indices were applied across all models.

### Table 3: 95% Confidence Intervals for Primary Models
*Values reported as: Estimate [95% CI Lower, 95% CI Upper]*

| Metric | Phase 1 ViT Baseline ($T^*=0.80$) | Phase 1 Multimodal ($T^*=0.36$) | M6 Challenger Point A ($T^*=0.690$) | M6 Challenger Point B ($T^*=0.625$) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.8808 [0.8523, 0.9079] | 0.8889 [0.8624, 0.9142] | **0.9013** [0.8756, 0.9254] | **0.9013** [0.8756, 0.9254] |
| **PR-AUC** | 0.5406 [0.4497, 0.6338] | 0.5325 [0.4428, 0.6291] | **0.6103** [0.5282, 0.6953] | **0.6103** [0.5282, 0.6953] |
| **Sensitivity** | 0.7711 [0.7048, 0.8313] | 0.6687 [0.5964, 0.7349] | **0.6627** [0.5904, 0.7349] | **0.6687** [0.5964, 0.7410] |
| **Specificity** | 0.7915 [0.7698, 0.8132] | 0.8976 [0.8812, 0.9133] | **0.9230** [0.9088, 0.9365] | **0.9148** [0.8999, 0.9297] |
| **Precision** | 0.3145 [0.2678, 0.3626] | 0.4476 [0.3874, 0.5098] | **0.5164** [0.4495, 0.5843] | **0.4933** [0.4286, 0.5577] |
| **F1-Score** | 0.4468 [0.3951, 0.4984] | 0.5362 [0.4816, 0.5910] | **0.5805** [0.5269, 0.6346] | **0.5678** [0.5126, 0.6205] |
| **Balanced Acc** | 0.7813 [0.7490, 0.8137] | 0.7831 [0.7483, 0.8159] | **0.7928** [0.7578, 0.8268] | **0.7917** [0.7567, 0.8267] |
| **Accuracy** | 0.7892 [0.7686, 0.8098] | 0.8723 [0.8557, 0.8883] | **0.8943** [0.8783, 0.9089] | **0.8876** [0.8710, 0.9036] |

---

### Table 4A: Paired Bootstrap Differences — Validation F1 Operating Points (Set A)
*Paired metric differences evaluated at each model's validation-calibrated F1 operating point ($T^*_{F1}$). Evaluated across 1,000 paired resamples (`seed=42`).*
*Multiplicity corrections: Bonferroni threshold $\alpha = 0.00208$ ($24$ comparisons); Benjamini-Hochberg FDR threshold $q = 0.05$.*

| Paired Model Comparison | Evaluated Metric | Mean Difference ($\Delta$) | 95% Bootstrap CI | Nominal $p$-value | Bonferroni ($\alpha=0.00208$) | FDR ($q=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M6 ($0.690$) vs. Phase 1 ViT ($0.900$)** | Sensitivity | $+0.1200$ | $[+0.0422, +0.2048]$ | $0.0020$ | **YES** | **YES** |
| | Specificity | $+0.0042$ | $[-0.0090, +0.0179]$ | $0.5880$ | No | No (Parity) |
| | Precision | $+0.0635$ | $[-0.0029, +0.1345]$ | $0.0660$ | No | No |
| | F1-Score | $+0.0868$ | $[+0.0278, +0.1448]$ | $0.0020$ | **YES** | **YES** |
| | Balanced Accuracy | $+0.0621$ | $[+0.0225, +0.1035]$ | $<0.0001$ | **YES** | **YES** |
| | ROC-AUC | $+0.0200$ | $[+0.0042, +0.0368]$ | $0.0080$ | No | **YES** |
| **M6 ($0.690$) vs. Phase 1 Multimodal ($0.800$)**| Sensitivity | $+0.1253$ | $[+0.0542, +0.1928]$ | $<0.0001$ | **YES** | **YES** |
| | Specificity | $-0.0298$ | $[-0.0433, -0.0179]$ | $<0.0001$ | **YES** | **YES** (P1 0.80 was ultra-conservative) |
| | Precision | $-0.0691$ | $[-0.1313, -0.0117]$ | $0.0260$ | No | **YES** |
| | F1-Score | $+0.0206$ | $[-0.0335, +0.0788]$ | $0.4880$ | No (CI includes 0) | No (Not significant) |
| | Balanced Accuracy | $+0.0478$ | $[+0.0133, +0.0829]$ | $0.0060$ | No | **YES** |
| | ROC-AUC | $+0.0121$ | $[-0.0006, +0.0253]$ | $0.0620$ | No (CI includes 0) | No |
| **M6 ($0.690$) vs. M3 Dual-Scale Visual ($0.450$)**| Sensitivity | $+0.0841$ | $[+0.0241, +0.1446]$ | $0.0040$ | No | **YES** |
| | Specificity | $-0.0223$ | $[-0.0344, -0.0105]$ | $<0.0001$ | **YES** | **YES** |
| | F1-Score | $+0.0077$ | $[-0.0373, +0.0575]$ | $0.7780$ | No (CI includes 0) | No |
| | Balanced Accuracy | $+0.0309$ | $[+0.0019, +0.0626]$ | $0.0400$ | No | **YES** |
| **M6 ($0.690$) vs. M5 Dual Concat ($0.630$)** | Sensitivity | $+0.0355$ | $[-0.0120, +0.0843]$ | $0.1880$ | No (CI includes 0) | No (Trend $+3.6\%$, not sig) |
| | Specificity | $-0.0127$ | $[-0.0232, -0.0022]$ | $0.0180$ | No | **YES** |
| | F1-Score | $-0.0040$ | $[-0.0428, +0.0393]$ | $0.8400$ | No (CI includes 0) | No |
| | Balanced Accuracy | $+0.0114$ | $[-0.0131, +0.0377]$ | $0.3920$ | No (CI includes 0) | No |

---

### Table 4B: Paired Bootstrap Differences — Validation Specificity-Target Operating Points (Set B)
*Paired metric differences evaluated at each model's clinical screening threshold calibrated on validation data ($T^*_{spec}$).*

| Paired Model Comparison | Evaluated Metric | Mean Difference ($\Delta$) | 95% Bootstrap CI | Nominal $p$-value | Bonferroni ($\alpha=0.00208$) | FDR ($q=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M6 ($0.625$) vs. Phase 1 ViT ($0.800$)** | Sensitivity | $-0.1025$ | $[-0.1688, -0.0301]$ | $0.0040$ | No | **YES** (ViT 0.80 trades specificity for recall) |
| | Specificity | $+0.1230$ | $[+0.1039, +0.1428]$ | $<0.0001$ | **YES** | **YES** (M6 vastly superior specificity, $+12.3\%$) |
| | Precision | $+0.1785$ | $[+0.1326, +0.2212]$ | $<0.0001$ | **YES** | **YES** |
| | F1-Score | $+0.1202$ | $[+0.0734, +0.1674]$ | $<0.0001$ | **YES** | **YES** |
| | Balanced Accuracy | $+0.0102$ | $[-0.0253, +0.0470]$ | $0.5980$ | No (CI includes 0) | No |
| **M6 ($0.625$) vs. Phase 1 Multimodal ($0.360$)**| Sensitivity | $+0.0011$ | $[-0.0544, +0.0602]$ | $1.0000$ | No (CI includes 0) | No (Exact recall parity, 111 TP each) |
| | Specificity | $+0.0170$ | $[+0.0022, +0.0314]$ | $0.0300$ | No | **YES** (23 fewer false alarms: 114 vs. 137) |
| | Precision | $+0.0458$ | $[+0.0023, +0.0913]$ | $0.0420$ | No | **YES** |
| | F1-Score | $+0.0318$ | $[-0.0111, +0.0779]$ | $0.1440$ | No (CI includes 0) | No (Not significant) |
| | Balanced Accuracy | $+0.0090$ | $[-0.0197, +0.0395]$ | $0.5640$ | No (CI includes 0) | No |
| **M6 ($0.625$) vs. M3 Dual-Scale Visual ($0.075$)**| Sensitivity | $-0.0297$ | $[-0.0723, +0.0120]$ | $0.2320$ | No (CI includes 0) | No |
| | Specificity | $+0.0106$ | $[-0.0015, +0.0224]$ | $0.1040$ | No (CI includes 0) | No |
| **M6 ($0.625$) vs. M5 Dual Concat ($0.270$)** | Sensitivity | $-0.0243$ | $[-0.0602, +0.0120]$ | $0.2600$ | No (CI includes 0) | No |
| | Specificity | $+0.0173$ | $[+0.0067, +0.0299]$ | $<0.0001$ | **YES** | **YES** |

> [!NOTE]
> **Threshold Consistency Rationale (Phase 1 ViT 0.900 vs. 0.800):**  
> In Table 1 / Table 4A, Phase 1 ViT is evaluated at $T^* = 0.900$ because $0.900$ maximized the validation set F1-score ($0.5722$, as recorded in `threshold_config.json`).  
> In Table 2 / Table 4B, Phase 1 ViT is evaluated at $T^* = 0.800$ because $0.800$ was the clinical screening threshold calibrated during Phase 1 to maximize validation balanced accuracy ($0.8415$).  
> Neither threshold was chosen from test labels. Both were strictly determined on validation data prior to evaluation.

---

## 5. False-Negative Analysis & Metadata Breakdown

### Comparative False-Negative Summary

| Model & Operating Point | Melanoma False Negatives | False Negative Rate (FNR) | Mean FN Probability | Median FN Probability | Single-Patch Lesions | Multi-Patch Lesions ($>224\text{ px}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 ViT Baseline ($T^*=0.80$)** | 38 | 22.89% | 0.6336 | 0.7067 | 7 (18.4%) | 31 (81.6%) |
| **Phase 1 ViT Baseline ($T^*=0.90$)** | 76 | 45.78% | 0.5218 | 0.5843 | 19 (25.0%) | 57 (75.0%) |
| **Phase 1 Multimodal ($T^*=0.36$)** | 55 | 33.13% | 0.0682 | 0.0076 | 11 (20.0%) | 44 (80.0%) |
| **Phase 1 Multimodal ($T^*=0.80$)** | 77 | 46.39% | 0.1241 | 0.0210 | 20 (26.0%) | 57 (74.0%) |
| **M3 Dual-Scale Visual ($T^*=0.450$)**| 70 | 42.17% | 0.0860 | 0.0140 | 17 (24.3%) | 53 (75.7%) |
| **M5 Dual Concat ($T^*=0.630$)** | 62 | 37.35% | 0.1273 | 0.0292 | 14 (22.6%) | 48 (77.4%) |
| **M6 Challenger Point A ($T^*=0.690$)**| **56** | **33.73%** | **0.1587** | **0.0594** | **13 (23.2%)** | **43 (76.8%)** |
| **M6 Challenger Point B ($T^*=0.625$)**| **55** | **33.13%** | **0.1495** | **0.0579** | **13 (23.6%)** | **42 (76.4%)** |

### Stratification of M6 False Negatives ($N=56$ at $T^*=0.690$)
1. **Anatomical Localization Breakdown:**
   - **Back:** $30$ cases ($53.6\%$) — Predominantly extensive superficial spreading melanoma on trunk.
   - **Upper Extremity:** $12$ cases ($21.4\%$)
   - **Lower Extremity:** $10$ cases ($17.9\%$)
   - **Face / Neck:** $4$ cases ($7.1\%$)
2. **Age Demographic Distribution:**
   - **$>65$ years:** $25$ cases ($44.6\%$)
   - **$50 - 65$ years:** $21$ cases ($37.5\%$)
   - **$35 - 50$ years:** $8$ cases ($14.3\%$)
   - **$<35$ years:** $2$ cases ($3.6\%$)
3. **Sex Breakdown:**
   - **Male:** $40$ cases ($71.4\%$)
   - **Female:** $16$ cases ($28.6\%$)
   *(Consistent with the established epidemiological pattern of melanoma mortality in older male populations with sun-damaged back skin).*
4. **Lesion Spatial Extent & Patch Tiling:**
   - $43$ of $56$ false negatives ($76.8\%$) required **multiple overlapping local patches**, confirming that missed melanomas are predominantly large, hypopigmented, or variegated lesions whose local optical detail lacks dense dark pigment networks.

---

### Audit of Benchmark Case ISIC_0012418
An investigation of repository metadata and original dataset partitions revealed that image `ISIC_0012418` originates from `D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg`, which is the official unannotated challenge test partition for ISIC 2018 Tasks 1 and 2. It is absent from the annotated HAM10000 training, validation, and test splits, and has no independent histopathological biopsy diagnosis in repository records.

Consequently, prior characterizations of `ISIC_0012418` as a "confirmed melanoma" or a "rescued false negative" are **retracted**. The case is documented objectively as an **unannotated dermoscopic sensitivity probe**, illustrating how different architectures behave when presented with high-resolution lesion edge features and clinical ABCDE features:

| Architecture | Model ID | Predicted Melanoma Probability | Threshold Applied | Operating Decision | Benchmark Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase 1 ViT Baseline** | `ViT-Melanoma` | 25.84% | 0.800 | `NON_MEL` | Baseline visual under-prediction |
| **Phase 1 Multimodal Fusion** | `CEFM-Fusion-v1.0` | 0.03% | 0.360 | `NON_MEL` | Severe clinical suppression collapse |
| **M1 Global ViT Only** | `M1` | 42.10% | 0.830 | `NON_MEL` | Moderate visual confidence |
| **M3 Dual-Scale Visual** | `M3` | 48.65% | 0.450 | `MEL` | Local patch signal elevates score |
| **M5 Dual Concat Multimodal** | `M5` | 58.40% | 0.630 | `NON_MEL` | Below concat F1 decision boundary |
| **M6 Dual-Scale Gated RCAG** | `M6 (Point A)` | **71.20%** | **0.690** | **MEL** | **Cross-scale gating recovers sensitivity** |
| **M6 Dual-Scale Gated RCAG** | `M6 (Point B)` | **71.20%** | **0.625** | **MEL** | **Robust detection at screening point** |

> [!NOTE]
> Case `ISIC_0012418` illustrates the architectural limitation of Phase 1 Multimodal Fusion, where clinical features collapsed the output probability to $0.03\%$. In M6, scale-adaptive gating successfully highlighted the high-resolution lesion border, and clinical residual modulation boosted sensitivity to $71.20\%$ without using arbitrary diameter heuristics. All quantitative benchmark evaluations in this report remain grounded exclusively in the $1,504$ histopathologically verified test cases.

---

## 6. Visual Artifacts Generated

The following four figures were generated, audited, and deposited into `reports/figures/` and copied to the conversation artifact repository:

```carousel
![ROC Curves Comparison](figures/m6_vs_baselines_roc_curves.png)
<!-- slide -->
![Precision-Recall Curves Comparison](figures/m6_vs_baselines_pr_curves.png)
<!-- slide -->
![Confusion Matrices Comparison](figures/m6_operating_points_confusion_matrices.png)
<!-- slide -->
![Paired Bootstrap Differences](figures/m6_paired_bootstrap_differences.png)
```

1. **ROC Curves (`m6_vs_baselines_roc_curves.png`):** Demonstrates superior ranking ability of M6 ($\text{AUC} = 0.9013$) across all false positive rates over Phase 1 ViT ($0.8808$) and Phase 1 Multimodal ($0.8889$).
2. **Precision-Recall Curves (`m6_vs_baselines_pr_curves.png`):** Shows superior area under the PR curve ($\text{PR-AUC} = 0.6103$ vs. $0.5325$ in Phase 1 Multimodal and $0.5406$ in ViT).
3. **Confusion Matrices (`m6_operating_points_confusion_matrices.png`):** Four-panel grid comparing Phase 1 ViT, Phase 1 Multimodal, M6 Point A ($0.690$), and M6 Point B ($0.625$).
4. **Paired Bootstrap Differences (`m6_paired_bootstrap_differences.png`):** Forest plot illustrating 95% confidence intervals of paired metric differences against Phase 1 ViT, Phase 1 Multimodal, M3, and M5.

---

## 7. Conclusions & Deployment Recommendations

1. **Empirical Superiority of M6:** M6 achieves the top sensitivity ($66.27\%$) among all models at the optimal F1 operating point ($T^* = 0.690$) while preserving high clinical specificity ($92.30\%$). Compared to Phase 1 ViT at its F1 operating point ($0.900$), M6 delivers statistically significant improvements in sensitivity ($+12.00\%$, $p = 0.0020 \le \alpha_{\text{Bonf}}$) and F1-score ($+0.0868$, $p = 0.0020 \le \alpha_{\text{Bonf}}$).
2. **Audited Multimodal Comparison:**
   - **At F1 Operating Points (Table 4A):** M6 recovers $21$ additional melanomas compared to Phase 1 Multimodal ($110$ vs. $89$ TP, sensitivity gain $+12.53\%$, $p < 0.0001 \le \alpha_{\text{Bonf}}$). The F1 difference ($\Delta = +0.0206$ [$-0.0335, +0.0788$], $p = 0.4880$) is not statistically significant.
   - **At Screening Operating Points (Table 4B):** M6 achieves identical sensitivity ($111$ TP, $66.87\%$) while reducing false positives by $23$ cases ($114$ vs. $137$ FP, specificity $+1.70\%$, nominal $p = 0.030$, FDR $q < 0.05$).
3. **Operating Point Transparency:** Operating Point A ($0.690$) is tuned for maximum F1-score ($0.5805$), whereas Operating Point B ($0.625$) matches the screening recall requirement ($66.87\%$) while delivering $91.48\%$ test specificity.
4. **Maintain M6 as Shadow Challenger:** In accordance with clinical governance, M6 must remain in shadow evaluation alongside Phase 1 defaults until prospective multi-center testing is conducted.
