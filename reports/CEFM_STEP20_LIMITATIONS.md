# CEFM STEP 20: SYSTEMIC LIMITATIONS, FAILURE MODES, AND CLINICAL EVIDENCE GAPS
## Forensic Analysis of Known Vulnerabilities and Barriers to Clinical Translation

**Document ID:** CEFM-STEP20-LIMITATIONS-20261107  
**Execution Timestamp:** 2026-11-07T22:33:00Z  
**Scope:** Critical Assessment of CEFM Pipeline, Datasets, and Generalization  
**Status:** OFFICIAL EPISTEMIC DEFENSE & LIMITATIONS REGISTRY  

---

> [!WARNING]
> **EXPLICIT REGULATORY & CLINICAL BOUNDARY**  
> CEFM is an experimental decision-support research prototype developed for forensic auditing and multi-agent systems research. It has **NOT** undergone prospective clinical trials, blinded reader studies, or regulatory review by the US FDA, European Medicines Agency (CE mark), or any national medical authority. It must **NEVER** be used to guide autonomous clinical diagnosis, biopsy decisions, patient discharge, or treatment planning.

---

## 1. Primary Empirical Failure Modes

### 1.1 Remaining External False Negatives on PH² ($19 / 40$ Missed Melanomas)
While Challenger Model F-G successfully rescued unconstrained fusion from a near-total collapse on PH² ($5.00\% \to 52.50\%$ sensitivity), **19 out of 40 confirmed melanomas ($47.50\%$) were still missed** at the standard operating threshold ($\tau = 0.21032$).
- **Root Cause Analysis:**  
  The Visual Evidence Preserver operates as an override when the visual backbone signals high suspicion ($p_{\text{visual}} \ge 0.50$). However, on PH², Tuebinger fluid immersion optics (oil/gel reflections, vignetting, blue-dominant spectrum) caused the fine-tuned ViT backbone itself to output low visual scores ($p_{\text{visual}} < 0.50$) on 19 melanomas. Because the visual model failed to recognize suspicion, the guardrail did not trigger, and the lesion was demoted to benign.
- **Clinical Implication:** Software safety guardrails cannot correct for visual blindness in the foundational image encoder. If the underlying vision model is misled by foreign optical physics, post-hoc logic cannot rescue the detection.

### 1.2 Training Data Demographic Confounding in Young Patients
In the HAM10000 training population, cutaneous melanoma is heavily age-skewed towards older individuals ($60+$ years). In patients aged $<40$ years, melanoma prevalence in the dataset is extremely low ($13 / 286 = 4.5\%$).
- **Observed Manifestation:**  
  Subgroup audit on the lesion-disjoint test set revealed that in the `<40 Years` demographic, Model F-G achieved a sensitivity of only **$23.08\%$** ($3 / 13$ melanomas detected), compared to **$75.76\%$** in patients aged $60+$.
- **Clinical Risk:** Dermatological AI models trained on epidemiological cohorts inherit the population's baseline incidence rates as strong statistical priors. In younger patients—where melanomas are rare but devastating—models exhibit systemic under-sensitivity unless explicit visual overrides are enforced.

---

## 2. Dataset and Evidence Infrastructure Gaps

### 2.1 The Small-Cohort External Validation Problem ($N=200$)
- The primary external validation cohort available locally (PH²) contains only **$200$ total images and $40$ melanomas**.
- **Statistical Uncertainty:**  
  With only 40 positive events, the 95% bootstrap confidence interval for external sensitivity spans from **$35.00\%$ to $67.50\%$** ($\pm 16.25\%$). A sample of 40 cases is statistically insufficient to power non-inferiority trials or establish clinical reliability.

### 2.2 Complete Absence of External Multimodal Demographic Metadata
- The PH² benchmark records histopathological diagnosis and dermoscopic ABCD criteria, but **completely omits patient age, biological sex, and anatomical location**.
- **Evidence Gap:**  
  Because demographic metadata was missing, our external evaluation of Model F and Model F-G was conducted under the standardized *missing-data protocol* (`all_metadata_missing`). While this provided an outstanding test of missingness robustness, **true external multi-center validation of the combined visual-plus-demographic tabular fusion branch remains unexecuted**. Claiming that CEFM's multimodal fusion has been clinically validated on external cohorts would be scientifically false.

### 2.3 Optical Modality and Camera Physics Discrepancies
- **HAM10000:** Captured using DermLite (cross-polarized contact dermoscopy) and Heine Delta 20 (contact dermoscopy with alcohol interface).
- **PH²:** Captured using a Tuebinger Mole Analyzer with fluid immersion gel, strong optical vignetting, and a distinct spectral response (blue channel attenuation $-21.2\text{ px}$, saturation $+8.7$).
- **Generalization Barrier:** Dermoscopic deep neural networks are hypersensitive to microscopic immersion physics and polarization states. Hardware-agnostic domain adaptation (e.g. histogram matching, Shades of Gray color constancy) remains an untested hypothesis in this pipeline.

### 2.4 Fitzpatrick Skin Phototype Underrepresentation (Phototypes IV–VI)
- Both HAM10000 and PH² were collected primarily in European dermatology clinics (Vienna, Austria; Graz, Austria; Queensland, Australia; Matosinhos, Portugal) from predominantly fair-skinned populations (Fitzpatrick Skin Types I–III).
- **Critical Equity Gap:** The model has zero verified performance data on richly pigmented skin (Fitzpatrick Types IV–VI), acral lentiginous melanomas on dark skin, or nail apparatus lesions. Deploying this system in diverse populations without targeted validation risks severe diagnostic disparities.

---

## 3. Translation and Workflow Gaps

### 3.1 Lack of Prospective Clinical Workflow Evaluation
- All evaluations in Steps 1–20 were retrospective benchmark evaluations on pre-cropped, expert-curated image files.
- In actual clinical practice, dermatologists do not encounter isolated, pre-centered lesions. They evaluate lesions in real-time, considering clinical evolution, palpation (firmness, elevation), lesion comparison across the whole body ("ugly duckling" sign), and patient anxiety. Retrospective accuracy metrics do not predict clinical efficacy or physician-AI interaction dynamics.

### 3.2 Model Calibration and Probability Sharpness
- On the external PH² benchmark, Brier scores ranged from $0.1313$ to $0.1879$. Raw model logits exhibit uncalibrated overconfidence on in-distribution data and underconfidence on out-of-distribution optical data.
- The system lacks temperature scaling or conformal prediction sets with guaranteed error-coverage bounds.

---

## 4. Epistemic Conclusion

The findings of this audit demonstrate that **CEFM has achieved significant scientific progress in identifying and neutralizing demographic suppression via software guardrailing**, but **it remains far from clinical readiness**. 

Prior to any prospective clinical trial, the following mandatory steps must occur:
1. Multi-center external validation on large cohorts ($N > 5,000$) possessing verified histopathology AND complete patient demographic metadata (e.g. curated ISIC 2020 cohorts).
2. Prospective evaluation across diverse skin phototypes (Fitzpatrick I–VI).
3. Incorporation of image-level optical normalization (color constancy / immersion invariant embeddings).
4. Blinded multi-reader clinical utility trials comparing clinician accuracy with versus without model assistance.

