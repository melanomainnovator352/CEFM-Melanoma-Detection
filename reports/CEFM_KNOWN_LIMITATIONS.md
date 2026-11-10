# CEFM — Known Limitations & Research Outline

**Document ID:** `CEFM-LIMITATIONS-2026-11-07`  
**System Designation:** Cross-Modal Explainable Framework for Melanoma Assessment (CEFM)  
**Classification:** Investigational Clinical Decision Support System (CDSS)  

---

## 1. Regulatory & Clinical Decision Support Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL DEVICE DISCLAIMER:**  
> The CEFM software and its constituent machine learning models (ViT Baseline, Multimodal Fusion v1.0, and M6 Dual-Scale Gated RCAG) are designed **strictly for research and clinical decision support purposes**.  
> - CEFM is **NOT** cleared or approved by the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), or equivalent regulatory bodies as an autonomous diagnostic medical device.  
> - Predictions, risk scores, and generated clinical reports must **NOT** be used as a standalone basis for diagnostic or therapeutic decisions.  
> - A certified dermatologist, pathologist, or qualified physician must review physical clinical presentation, full dermoscopic exam, and histopathological biopsy before initiating any clinical intervention.

---

## 2. Dataset & Demographic Limitations

### 2.1 Fitzpatrick Skin Phototype Skew
- The primary training and validation dataset (HAM10000) was acquired from patients in Vienna, Austria, and Queensland, Australia.
- The population skews overwhelmingly toward **Fitzpatrick Skin Types I, II, and III** (light, fair skin prone to sunburn).
- Performance on Fitzpatrick Skin Types IV, V, and VI (brown, deeply pigmented skin) has **not been independently validated** due to dataset sparsity. Diagnostic accuracy for melanoma in darkly pigmented skin cannot be inferred from these results.

### 2.2 Melanoma Subtype Representation
- The training cohort is dominated by **Superficial Spreading Melanoma (SSM)** and **Lentigo Maligna (LM)**.
- Rare and aggressive melanoma subtypes are significantly under-represented:
  - **Acral Lentiginous Melanoma (ALM):** Melanomas of the palms, soles, and nail beds are poorly represented.
  - **Nodular Melanoma:** Fast-growing invasive nodular lesions that lack prominent lateral ABCDE border features may receive lower risk scores.
  - **Amelanotic / Hypomelanotic Melanoma:** Lesions lacking dark pigmentation networks require careful dermatoscopic examination.

---

## 3. External Generalization & Domain Shift Findings

### 3.1 Atypical Nevus Specificity Trade-Off
- In routine clinical screening, common nevi comprise $>80\%$ of benign encounters.
- External validation on the **PH² dataset** demonstrated that when an external cohort is heavily enriched with **atypical/dysplastic nevi ($40\%$ of cohort; $50\%$ of non-melanomas)**, sensitivity was reduced across all tested models ($30\% - 35\%$ for M6, $47.5\%$ for Multimodal, $50\%$ for ViT).
- Dysplastic nevi share structural features (asymmetry, multifocal pigmentation) with early melanomas, requiring clinical biopsy for definitive ruling.

### 3.2 Missing Clinical Metadata Vulnerability
- In clinical practice, multimodal architectures depend on complete patient records (age, sex, anatomical location, evolution).
- When clinical metadata is missing (as in unannotated databases or incomplete patient intake), models must fall back to zero-imputation.
- Experimental evidence shows that missing demographic features lower multimodal detection confidence, reducing sensitivity for subtle lesions.

---

## 4. Formal Research Paper Outline

*Based strictly on verified experimental data from CEFM Phase 1, Phase 2 (Steps 1–4), and Step 5 external validation.*

### Title
**Dual-Scale Cross-Attention Gated Multimodal Learning for Clinically Explainable Melanoma Assessment: Bridging Global Context and Local Micro-Structures**

### Abstract
- **Context:** Automated melanoma screening often struggles with false negatives in large or variegated lesions due to trade-offs between global lesion context and local dermoscopic micro-structure resolution.
- **Methods:** We introduce a dual-scale multimodal framework (M6) combining a 768-D global context view ($1.30\times$), high-resolution native-scale local patch tiling guided by semantic segmentation, deterministic 16-D ABCDE feature extraction, and a Residual Cross-Attention Gating (RCAG) network.
- **Results:** Evaluated on an independent, lesion-stratified test split ($N = 1,504$; $166$ melanomas) without parameter tuning on test data, M6 achieved $\text{ROC-AUC} = 0.9013$ [$0.8756, 0.9254$] and $\text{PR-AUC} = 0.6103$ [$0.5282, 0.6953$]. At the validation-calibrated F1 operating point ($T^* = 0.690$), M6 reduced melanoma false negatives from $76$ (ViT baseline) and $77$ (Multimodal v1.0) down to $56$ ($+12.00\%$ and $+12.53\%$ sensitivity, $p \le 0.0020$).
- **External Validation:** On the external PH² benchmark ($N = 200$), M6 preserved high clinical specificity ($93.75\% - 94.38\%$) against a $50\%$ dysplastic nevi challenge, outperforming the Phase 1 multimodal baseline in cross-center ranking ($\text{ROC-AUC} = 0.8112$ vs. $0.7589$).
- **Significance:** Demonstrates the efficacy of scale-adaptive visual gating and clinical residual modulation for robust, explainable dermatological AI.

### Outline Structure
1. **Introduction:**
   - Epidemiology of malignant melanoma and diagnostic challenges in primary care.
   - Limitations of standard Vision Transformers (downsampling loss of fine pigment network details).
   - The multimodal alignment gap: over-reliance on visual representations vs. clinical ABCDE features.
2. **Related Work:**
   - Deep learning in dermoscopy (ISIC benchmarks, ViT vs. CNN backbones).
   - Multi-scale and patch-based dermatological analysis.
   - Multimodal fusion and explainability in clinical decision support.
3. **Methodology:**
   - Semantic lesion segmentation via VM-UNet and scale-adaptive local patch tiling.
   - Dual-scale visual feature encoding (frozen ViT backbone, global context vs. local boundary patches).
   - Deterministic ABCDE clinical vector extraction and non-leaking feature normalization.
   - Residual Cross-Attention Gated (RCAG) architecture and clinical residual modulation.
   - Operating threshold calibration protocol (validation-only F1-optimal and 90% specificity targets).
4. **Experimental Setup & Dataset Curation:**
   - Lesion-stratified partitioning of HAM10000 ($N=10,015$; 6,978 train, 1,533 val, 1,504 test) with zero patient overlap.
   - Independent external cohort: PH² database ($N=200$, Hospital Pedro Hispano, Portugal).
   - Statistical evaluation protocol (1,000 stratified bootstrap resamples, paired difference estimation, Bonferroni & FDR multiplicity adjustments).
5. **Results & Benchmark Evaluation:**
   - In-distribution performance: Comparative evaluation of M1–M7 against Phase 1 baselines.
   - Statistical significance audit of paired bootstrap metric differences.
   - External validation results on the PH² database.
   - Segmentation robustness and Dice score analysis (mean Dice: $0.8069$).
6. **Clinical Discussion & Interpretability:**
   - Qualitative analysis of false-negative recovery (case study analysis).
   - Self-attention rollout heatmaps and clinical discordance alerting.
   - Domain shift analysis: Atypical nevi confounding and missing demographic metadata.
7. **Limitations & Future Directions:**
   - Skin phototype representation, rare melanoma subtypes, and prospective multi-center clinical trials.
8. **Conclusion & Code Availability.**

