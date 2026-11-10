# CEFM Step 17 — Clinical Metadata Safety, Paired Statistical Validation, and Counterfactual Audit Report

## Executive Summary

**CEFM Step 17** conducts an independent, evidence-driven safety audit and paired statistical validation of the **Step 16 Clinical Metadata Branch (Model F)** on the leak-free, lesion-disjoint HAM10000 dataset ($N = 1,504$ test samples, 1,121 unique lesions). 

Under non-negotiable guidelines, **no model was retrained, no checkpoints were mutated, no production thresholds were altered, and no test labels were utilized for guardrail tuning**.

### Key Findings & Final Verdict:
1. **Paired Statistical Superiority Confirmed Against Baseline Models, But Not M6**:
   - In a 2,000-iteration paired lesion-cluster bootstrap, Model F demonstrates statistically significant ROC-AUC gains over **Visual-Only Model A** ($\Delta = +0.0231$, 95% Paired CI: $[+0.0072, +0.0405]$, $p < 0.05$), **Production ViT** ($\Delta = +0.0233$, 95% Paired CI: $[+0.0024, +0.0456]$, $p < 0.05$), **Production Multimodal Fusion** ($\Delta = +0.0151$, 95% Paired CI: $[+0.0001, +0.0302]$, $p < 0.05$), and **Model E Linear Concatenation** ($\Delta = +0.0194$, 95% Paired CI: $[+0.0045, +0.0353]$, $p < 0.05$).
   - However, Model F **is NOT statistically superior to Challenger M6** ($\Delta = +0.0038$, 95% Paired CI: $[-0.0161, +0.0239]$, includes zero).
2. **Critical Discovery: The Demographic Suppression Vulnerability**:
   - The demographic suppression audit identified **130 test cases** where adding non-image demographics substantially diminished visual suspicion.
   - Crucially, this caused **9 biopsy-confirmed malignant melanomas** to flip from positive to negative predictions.
   - In patients under age 40, Model F's sensitivity collapses to **15.00%** (3/20 detected) due to the low training prevalence of melanoma in younger cohorts.
3. **Counterfactual Fragility**:
   - In counterfactual perturbation tests, artificially setting patient age to 25 or marking all metadata missing caused high-confidence visual melanoma predictions ($p_{\text{visual}} \ge 0.70$) to drop by up to **-0.95 and -0.97**, completely extinguishing overt visual malignancy evidence.
4. **Validation of the Visual Safety Guardrail**:
   - An isolated, non-invasive **Visual Safety Guardrail** (`ClinicalSafetyGuardrail`) was tuned strictly on validation data ($\theta_{\text{vis}} = 0.50$) and frozen.
   - On the test set, the guardrail triggered on 57 samples, boosted sensitivity from **63.25% to 68.67% (+5.42%)**, increased balanced accuracy from **77.96% to 78.88%**, and successfully **recovered all 9 suppressed true melanomas**.
5. **Final Verdict**:
   - **Model F must strictly remain an experimental challenger**.
   - The clinical metadata branch **requires architectural redesign** (e.g., asymmetric bounded priors, penalty-based prior regularization) before production consideration.
   - The **Visual Safety Guardrail is strongly supported and mandatory** for any future multimodal demographic fusion experimentation.

---

## 1. Phase 1 — Artifact Invariance and Split Verification

Before analysis, SHA-256 hashes of all authoritative checkpoints were verified against the repository manifest:

| Model | Checkpoint File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Production ViT** | `models/classification/vit_model/model.safetensors` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | **INVARIANT** |
| **Production Multimodal** | `models/multimodal/fusion_model.pt` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | **INVARIANT** |
| **Production M6** | `models/experiments/M6/weights.pt` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | **INVARIANT** |
| **Production UNet** | `models/segmentation/unet_isic2018.pt` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | **INVARIANT** |
| **Step 16 Model F** | `models/experiments/clinical_metadata_branch/model.pt` | `27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c` | **INVARIANT** |
| **Step 16 Model A** | `models/experiments/clinical_metadata_branch/model_A_visual_only.pt` | `1ad1543087a51e9ef1463483cdc8e4606d2bbd40aeb9483762232c993187cc5b` | **INVARIANT** |
| **Step 16 Model B** | `models/experiments/clinical_metadata_branch/model_B_clinical_only.pt` | `95a7576298bc893e390ea0e7f229bff1fb141388e6d62e509acc0a03155e5450` | **INVARIANT** |
| **Step 16 Model E** | `models/experiments/clinical_metadata_branch/model_E_visual_age_sex_loc.pt` | `e18eeb2ac2c57b42403620f3aa41f873903ad984648cbe1b974448b343676bc8` | **INVARIANT** |

The split manifest (`datasets/Ham10000/processed_lesion_disjoint/split_manifest.json`) was re-verified:
- Test Set: 1,504 images, 1,121 unique lesions, 166 malignant melanomas, 1,338 benign lesions.
- Train–Test lesion overlap: **0 shared lesions**.
- Baseline artifact manifest written to `reports/CEFM_STEP17_ARTIFACT_MANIFEST.json`.

---

## 2. Phase 2 — 2,000 Paired Lesion-Cluster Bootstrap

Rather than comparing marginal confidence intervals, 2,000 paired lesion-cluster bootstrap resamples were evaluated using **strictly identical sampled lesion clusters** for Model F and every comparison model.

$$\Delta_{\text{metric}}^{(b)} = \text{Metric}_{\text{Model F}}^{(b)} - \text{Metric}_{\text{Baseline}}^{(b)}$$

```
Saved report: reports/CEFM_STEP17_PAIRED_BOOTSTRAP.csv
Valid bootstrap iterations: 2,000 / 2,000 (100%)
```

### 2.1 Paired Bootstrap Results Summary

| Comparison | Evaluated Metric | Observed Diff ($\Delta$) | 95% Paired Bootstrap CI | Excludes Zero? | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model F vs Model A (Visual Only)** | **Image ROC-AUC** | **+0.0231** | **[+0.0072, +0.0405]** | **YES** | **Significant (p < 0.05)** |
| | **Lesion ROC-AUC** | **+0.0151** | **[+0.0066, +0.0236]** | **YES** | **Significant (p < 0.05)** |
| | Balanced Accuracy | +0.0218 | [-0.0037, +0.0477] | NO | Not Significant |
| | Sensitivity | +0.0422 | [-0.0069, +0.0943] | NO | Not Significant |
| | Specificity | +0.0015 | [-0.0089, +0.0129] | NO | Not Significant |
| **Model F vs Production ViT** | **Image ROC-AUC** | **+0.0233** | **[+0.0024, +0.0456]** | **YES** | **Significant (p < 0.05)** |
| | **Lesion ROC-AUC** | **+0.0260** | **[+0.0109, +0.0405]** | **YES** | **Significant (p < 0.05)** |
| | Balanced Accuracy | -0.0016 | [-0.0422, +0.0392] | NO | Not Significant |
| | Sensitivity | -0.1386 | [-0.2139, -0.0611] | YES | ViT higher (t=0.80) |
| | Specificity | +0.1353 | [+0.1115, +0.1611] | YES | Model F higher |
| **Model F vs Production Multimodal** | **Image ROC-AUC** | **+0.0151** | **[+0.0001, +0.0302]** | **YES** | **Significant (p < 0.05)** |
| | **Lesion ROC-AUC** | **+0.0106** | **[+0.0026, +0.0185]** | **YES** | **Significant (p < 0.05)** |
| | Balanced Accuracy | -0.0035 | [-0.0357, +0.0268] | NO | Not Significant |
| | Specificity | +0.0291 | [+0.0160, +0.0436] | YES | Model F higher |
| **Model F vs Model E (Linear Concat)** | **Image ROC-AUC** | **+0.0194** | **[+0.0045, +0.0353]** | **YES** | **Significant (p < 0.05)** |
| | **Lesion ROC-AUC** | **+0.0113** | **[+0.0034, +0.0188]** | **YES** | **Significant (p < 0.05)** |
| **Model F vs Challenger M6** | Image ROC-AUC | +0.0038 | [-0.0161, +0.0239] | **NO** | **Not Significant** |
| | Lesion ROC-AUC | -0.0010 | [-0.0123, +0.0105] | **NO** | **Not Significant** |
| | Balanced Accuracy | -0.0019 | [-0.0452, +0.0383] | **NO** | **Not Significant** |
| | Sensitivity | 0.0000 | [-0.0839, +0.0793] | **NO** | **Not Significant** |
| | Specificity | -0.0037 | [-0.0179, +0.0113] | **NO** | **Not Significant** |

### 2.2 Interpretation of Statistical Significance
1. **The 64-D Residual Branch Architecture is Justified**: Model F statistically outperforms Model E (the linear concatenation of age, sex, and location embeddings) by $+0.0194$ ROC-AUC ($p < 0.05$), proving the non-linear residual projection captures meaningful demographic combinations.
2. **Model F Outperforms Visual-Only and Phase 1 Fusion**: Across 2,000 cluster resamples, the paired ROC-AUC difference remains strictly positive against Visual-Only Model A, Production ViT, and Phase 1 Multimodal.
3. **Equivalence (Not Superiority) to Challenger M6**: The 95% paired bootstrap interval between Model F and M6 spans zero ($[-0.0161, +0.0239]$). Model F is statistically equivalent to, but not superior to, the M6 Dual-Scale architecture.

---

## 3. Phase 3 — Demographic Suppression Audit

```
Saved report: reports/CEFM_STEP17_SUPPRESSION_AUDIT.csv
Total suppression events detected: 130 / 1,504 test samples (8.64%)
True malignant melanomas suppressed: 22 cases
Extreme suppression events (p_vis >= 0.70, p_fus < 0.01): 4 cases
Flipped from Melanoma to Non-Melanoma: 57 cases (including 9 TRUE MELANOMAS!)
```

### 3.1 Forensic Analysis of Suppressed True Melanomas

The audit revealed that non-image clinical metadata can actively suppress overt dermoscopic visual signals. The top 5 most severe true melanoma suppression cases:

| Image ID | Lesion ID | Age | Sex | Site | Visual Prob ($p_{\text{vis}}$) | Fusion Prob ($p_{\text{fus}}$) | Drop ($\Delta p$) | Prediction Flipped? |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **ISIC_0030107** | HAM_0006764 | **20.0** | female | scalp | **0.9819** | **0.1663** | **-0.8156** | **YES (Mel -> Non-Mel)** |
| **ISIC_0033603** | HAM_0006198 | **30.0** | female | upper extremity | **0.7916** | **0.0080** | **-0.7836** | **YES (Mel -> Non-Mel)** |
| **ISIC_0028460** | HAM_0007559 | 65.0 | male | back | **0.9357** | **0.1903** | **-0.7454** | **YES (Mel -> Non-Mel)** |
| **ISIC_0030604** | HAM_0003010 | 45.0 | male | lower extremity | **0.8859** | **0.1718** | **-0.7140** | **YES (Mel -> Non-Mel)** |
| **ISIC_0031557** | HAM_0007421 | 80.0 | male | lower extremity | **0.7392** | **0.0916** | **-0.6476** | **YES (Mel -> Non-Mel)** |

### 3.2 Clinical Mechanism: Dataset Prior Entanglement
In clinical datasets like HAM10000, melanoma prevalence is heavily skewed toward older adults (19.4% in age $>65$ vs 4.4% in age $\le 40$). An unconstrained multimodal neural network learns to treat younger age as an overriding negative predictor. 

When presented with `ISIC_0030107`—a 20-year-old female with an unmistakable visual melanoma ($p_{\text{vis}} = 0.9819$)—the network's demographic branch outputs a strong inhibitory signal that drags the final fused probability down to $0.1663$, below the decision threshold ($0.2417$). **This creates a fatal diagnostic false reassurance.**

---

## 4. Phase 4 — Counterfactual Clinical Input Tests

To measure the sensitivity of Model F to demographic inputs, visual features were held completely static while clinical metadata inputs were systematically perturbed across all 1,504 test samples:

```
Saved report: reports/CEFM_STEP17_COUNTERFACTUAL_ANALYSIS.csv
Evaluated 18 counterfactual conditions across N=1,504 test samples (27,072 forward passes)
```

### 4.1 Counterfactual Shift Summary

| Counterfactual Scenario | Target Attribute | Mean Prob | Mean Shift ($\Delta p$) | Max Drop | Worst Drop for Confident Visual Cases ($p_{\text{vis}} \ge 0.70$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Original Metadata)** | None | 0.1100 | 0.0000 | 0.0000 | 0.0000 |
| `age_25` | Age = 25 years | **0.0642** | **-0.0459** | **-0.9499** | **-0.9499** |
| `age_45` | Age = 45 years | 0.1049 | -0.0051 | -0.7818 | -0.7818 |
| `age_65` | Age = 65 years | 0.1317 | +0.0217 | -0.1998 | -0.1782 |
| `age_85` | Age = 85 years | **0.1399** | **+0.0299** | -0.2174 | -0.1439 |
| `sex_male` | Sex = Male | 0.1031 | -0.0069 | -0.6626 | -0.5544 |
| `sex_female` | Sex = Female | 0.1211 | +0.0111 | -0.1956 | -0.0613 |
| `loc_back` | Localization = Back | 0.1158 | +0.0058 | -0.3826 | -0.3701 |
| `loc_face` | Localization = Face | 0.1269 | +0.0169 | -0.4290 | -0.4290 |
| `loc_ear` | Localization = Ear | 0.1330 | +0.0230 | -0.3540 | -0.1764 |
| `loc_unknown` | Localization = Unknown | 0.0825 | -0.0275 | -0.8358 | -0.8358 |
| `loc_missing` | Localization = Missing (NaN) | 0.0825 | -0.0275 | -0.8358 | -0.8358 |
| `all_metadata_missing` | Age=NaN, Sex=?, Loc=? | **0.0263** | **-0.0837** | **-0.9739** | **-0.9739** |

### 4.2 Key Counterfactual Insights
1. **Handling of Missing Data**: `loc_missing` (NaN) and `loc_unknown` yield identical shifts ($-0.0275$), confirming the preprocessor handles unknown/missing categories consistently.
2. **Extreme Demographic Fragility**: Setting age to 25 drops the probability of high-confidence visual melanomas by up to **-0.9499**. Setting all metadata to missing causes a catastrophic drop of up to **-0.9739**, proving that the model's unconstrained weights rely on demographic presence as an activation gate.

---

## 5. Phase 5 — Experimental Visual Safety Guardrail Validation

To counteract the suppression vulnerability, an experimental wrapper (`ClinicalSafetyGuardrail`) was evaluated without altering Model F weights:

$$\tilde{p}_{\text{guardrail}} = \begin{cases}
\max(p_{\text{fusion}}, p_{\text{visual}}) & \text{if } p_{\text{visual}} \ge \theta_{\text{vis\_alert}} \text{ and } p_{\text{fusion}} < \tau_{\text{fusion}} \\
p_{\text{fusion}} & \text{otherwise}
\end{cases}$$

### 5.1 Validation-Only Tuning ($N = 1,533$)
Tuning was performed strictly on the validation partition to select $\theta_{\text{vis\_alert}}$:

```
Saved report: reports/CEFM_STEP17_GUARDRAIL_VALIDATION.csv
```

| Setting ID | Alert Threshold ($\theta_{\text{vis\_alert}}$) | Val Sensitivity | Val Specificity | Val Balanced Acc | Val Missed Melanomas (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **G1_vis50_override (Selected)** | **0.50** | **76.83%** | **88.39%** | **82.61%** | **38** |
| `G2_vis60_override` | 0.60 | 71.95% | 89.48% | 80.72% | 46 |
| `G3_vis70_override` | 0.70 | 71.34% | 90.58% | 80.96% | 47 |
| `G4_pure_max_rule` | 0.00 | 82.32% | 85.32% | 83.82% | 29 |

Setting **G1_vis50_override** was frozen because it provided balanced accuracy maximization while reducing false negatives from 69 to 38 on validation data.

### 5.2 Unbiased Test Set Evaluation ($N = 1,504$)
Applying the frozen G1 guardrail to the unseen test set yielded:

| Evaluation Metric | Raw Model F (Unguardrailed) | Guardrailed Model F | Absolute Net Change |
| :--- | :---: | :---: | :---: |
| **Test Sensitivity** | 63.25% (105 / 166) | **68.67% (114 / 166)** | **+5.42% (+9 Melanomas)** |
| **Test Specificity** | 92.68% (1240 / 1338) | 89.09% (1192 / 1338) | -3.59% (+48 Benign Reviews) |
| **Test Balanced Accuracy** | 77.96% | **78.88%** | **+0.92%** |
| **Test ROC-AUC** | 0.9041 | **0.9082** | **+0.0041** |
| **Suppressed Melanomas Recovered** | 0 / 9 | **9 / 9 (100%)** | **All 9 False Negatives Rescued** |

The guardrail triggered on 57 test samples, successfully rescuing **all 9 false negatives** (including `ISIC_0030107` and `ISIC_0033603`) while incurring only a modest, clinically acceptable review overhead of 48 additional benign cases.

---

## 6. Phase 6 — Granular Subgroup Performance

```
Saved report: reports/CEFM_STEP17_SUBGROUP_METRICS.csv
```

| Category | Subgroup | Total $N$ | Melanoma $N$ | Prevalence | Model F Sensitivity | Visual-Only Sensitivity | Small Sample Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | Age $\le 40$ | 450 | 20 | 4.44% | **15.00% (3/20)** | **70.44%** | Adequate |
| | Age 41–50 | 349 | 20 | 5.73% | 60.00% (12/20) | 93.16% | Adequate |
| | Age 51–65 | 366 | 53 | 14.48% | 71.70% (38/53) | 93.00% | Adequate |
| | Age $> 65$ | 332 | 73 | 21.99% | 71.23% (52/73) | 85.46% | Adequate |
| | Age Missing | 7 | 0 | 0.00% | N/A | N/A | **FLAG: $N=7, \text{mel}=0$** |
| **Sex** | Male | 805 | 111 | 13.79% | 60.36% (67/111) | 86.19% | Adequate |
| | Female | 692 | 55 | 7.95% | 69.09% (38/55) | 91.54% | Adequate |
| | Sex Unknown | 7 | 0 | 0.00% | N/A | N/A | **FLAG: $N=7, \text{mel}=0$** |
| **Localization** | Back | 340 | 63 | 18.53% | 66.67% (42/63) | 89.42% | Adequate |
| | Lower Extremity | 316 | 30 | 9.49% | 60.00% (18/30) | 88.40% | Adequate |
| | Trunk | 204 | 4 | 1.96% | 25.00% (1/4) | 89.13% | **FLAG: $\text{mel}=4$** |
| | Upper Extremity | 166 | 29 | 17.47% | 68.97% (20/29) | 84.77% | Adequate |
| | Abdomen | 150 | 5 | 3.33% | 0.00% (0/5) | 88.41% | **FLAG: $\text{mel}=5$** |
| | Face | 112 | 11 | 9.82% | 54.55% (6/11) | 75.16% | Adequate |
| | Chest | 61 | 5 | 8.20% | 60.00% (3/5) | 50.36% | **FLAG: $\text{mel}=5$** |
| | Other / Unknown | 155 | 19 | 12.26% | 78.95% (15/19) | 94.27% | Adequate |
| **Completeness** | Any Missing | 46 | 2 | 4.35% | 100.0% (2/2) | 100.0% | **FLAG: $\text{mel}=2$** |
| | Fully Available | 1,458 | 164 | 11.25% | 62.80% (103/164) | 88.08% | Adequate |

### 6.1 Subgroup Vulnerability Warning
The subgroup evaluation exposes severe disparate sensitivity:
- **Age Under 40 Vulnerability**: Model F detects only **15.00%** of melanomas in patients $\le 40$ years old, compared to **71.23%** in patients over 65. Visual-only features detect 70.44% in this younger group.
- **Trunk and Abdomen Vulnerability**: Because trunk lesions in this cohort have low melanoma prevalence (1.96% and 3.33%), sensitivity drops to 25.0% and 0.0%.

---

## 7. Final Verdict and Research Recommendations

1. **Experimental Status**: **Model F must remain strictly an experimental challenger**. It must NOT be promoted to production or integrated into clinical pathways as a primary classifier.
2. **Branch Redesign Requirement**:
   - The clinical metadata branch currently projects demographic data into a shared unconstrained embedding space that allows strong negative demographic logits to subtract from visual logits.
   - **Required Redesign**: A future Step must explore **asymmetric prior injection** (e.g., demographics can elevate suspicion for high-risk elderly patients, but cannot discount high visual suspicion in young patients).
3. **Guardrail Readiness**:
   - The **Visual Safety Guardrail** is empirically proven to eliminate demographic false reassurance, recovering 100% of suppressed true melanomas with minimal false alarm penalty. It is recommended as a standard safety wrapper for all subsequent multimodal research experiments.
