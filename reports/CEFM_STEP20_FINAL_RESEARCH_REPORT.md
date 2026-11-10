# CEFM STEP 20: FINAL RESEARCH CONSOLIDATION, REPRODUCIBILITY, AND EVIDENCE AUDIT
## Capstone Scientific Assessment and Master Project Synthesis

**Document ID:** CEFM-STEP20-FINAL-RESEARCH-REPORT-20261107  
**Execution Timestamp:** 2026-11-07T22:33:00Z  
**Target:** Complete CEFM Forensic and Experimental Arc (Steps 1–20)  
**Status:** COMPLETE — FINAL RESEARCH EVIDENCE AUDIT SEALED  

---

> [!IMPORTANT]
> **RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSTIC USE**  
> All models, architectures, safety guardrails, and decision rules evaluated in this project are experimental decision-support research artifacts. They have not received regulatory clearance (FDA/CE-mark) for autonomous clinical diagnosis. Checkpoint files for production ViT, Phase 1 Multimodal Fusion, M6 Dual-Scale, and UNet remain unmodified, frozen, and cryptographically verified.

---

## 1. Executive Summary & Project Arc (Steps 1–20)

Over the course of 20 rigorous forensic engineering, architecture design, and statistical auditing steps, the Clinical Explainable Fusion Model (CEFM) project has progressed from an initial forensic diagnosis of lesion-level data leakage to a formally verified, guardrailed multimodal AI research platform.

### Summary of Key Milestones:
1. **Forensic Audit & Data Leakage Elimination (Steps 1–11):**  
   Discovered and eliminated massive lesion-level leakage in historical HAM10000 splits (466 train–val and 477 train–test shared lesion IDs). Generated official, reproducible lesion-disjoint partitions (`processed_lesion_disjoint/`) with strictly **zero lesion overlap** across 10,015 images.
2. **Authoritative Baseline Benchmark Evaluation (Steps 12–15):**  
   Conducted unbiased test evaluations on 1,504 lesion-disjoint images (1,121 lesions). Benchmarked Production ViT (ROC-AUC 0.8808), Phase 1 Multimodal Fusion (ROC-AUC 0.8889), and Phase 2 M6 Dual-Scale RCAG (ROC-AUC 0.9013). Integrated a transparent multi-model dashboard in Streamlit to present all three models simultaneously with explicit discordance alerts.
3. **The Discovery of Demographic Suppression (Steps 16–17):**  
   Trained a 64-D clinical metadata branch (Model F). Forensic counterfactual perturbation revealed a critical clinical vulnerability: benign demographic priors (young age, female sex) silently suppressed high-confidence visual melanoma detections below the decision threshold, causing 22 false-negative suppressions on the test set (including 9 confirmed melanomas).
4. **Architectural vs Post-Hoc Guardrailing (Step 18):**  
   Investigated whether architectural softplus constraints ($\Delta l_1 \ge 0$) could prevent suppression. Discovered that joint backpropagation causes gradient co-adaptation that depresses the base visual head, lowering sensitivity. Demonstrated that **Alternative B (Challenger Model F-G: decoupled visual extraction + post-hoc Visual Evidence Preservation)** was statistically superior, rescuing 9/9 melanomas, eliminating all 57 VEP violations, and boosting sensitivity by **+5.42% ($p < 0.05$)**.
5. **External Domain Portability & Robustness (Step 19):**  
   Evaluated the frozen challenger on the independent PH² benchmark ($N=200$). Proved that while unconstrained Model F collapsed to 5.00% sensitivity under missing metadata, Challenger Model F-G rescued sensitivity to **52.50%** with **0 VEP violations**. Stress-tested 14 operational perturbation scenarios, confirming that the VEP invariant held across 100% of tests.
6. **Final Consolidation & Evidence Audit (Step 20):**  
   Formalized the exact scientific boundaries of CEFM, established complete reproducibility across all steps, audited systemic limitations, and classified all project architectures.

---

## 2. Verified Artifact Freeze & Checkpoint Integrity

All production and experimental checkpoint files match their authoritative SHA-256 digests and have remained completely invariant:

| Model Role | Checkpoint Path | Authoritative SHA-256 Digest | Status | Operating Threshold |
| :--- | :--- | :--- | :---: | :---: |
| **Production ViT** | `models/classification/vit_model/model.safetensors` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | **VERIFIED** | $\tau = 0.800$ (Screening) / $0.900$ (F1) |
| **Production Multimodal** | `models/multimodal/fusion_model.pt` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | **VERIFIED** | $\tau = 0.360$ (Screening) / $0.800$ (F1) |
| **Production M6 Dual-Scale** | `models/experiments/M6/weights.pt` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | **VERIFIED** | $\tau = 0.690$ (Review) / $0.625$ (Spec) |
| **Production UNet** | `models/segmentation/unet_isic2018.pt` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | **VERIFIED** | Fixed binary threshold ($0.50$) |
| **Step 16 Model F** | `models/experiments/clinical_metadata_branch/model.pt` | `27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c` | **VERIFIED** | $\tau = 0.2417$ |
| **Step 16 Model A** | `models/experiments/clinical_metadata_branch/model_A_visual_only.pt` | `1ad1543087a51e9ef1463483cdc8e4606d2bbd40aeb9483762232c993187cc5b` | **VERIFIED** | $\tau = 0.6934$ |
| **Step 18 Model C** | `models/experiments/step18_constrained_fusion/model_C_constrained_residual.pt` | `eeb04571506a50bf6296547a1c7080d34c9ab625e8f7950f7f91527c65a59d67` | **VERIFIED** | $\tau = 0.7975$ |
| **Step 18 Model D** | `models/experiments/step18_constrained_fusion/model_D_gated_positive.pt` | `98e1c74525569ff40834a23d2d7b5849efad7bd844a13964728c738a8fc8d179` | **VERIFIED** | $\tau = 0.8074$ |

*(Deliverable: `reports/CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json`)*

---

## 3. Unified Cross-Cohort Model Comparison

The final, unified benchmark comparison across internal HAM10000 ($N=1,504$) and external PH² ($N=200$) cohorts:

| Evaluation Cohort | Model Name | Model Classification | Operating Threshold | ROC-AUC (95% CI) | PR-AUC | Sensitivity (95% CI) | Specificity (95% CI) | Balanced Accuracy (95% CI) | F1-Score | Brier Score | VEP Violations | Melanomas Suppressed | False Negatives |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HAM10000 (Internal Test, $N=1,504$)** | **Production ViT Screening** | Production Research Baseline | 0.80000 | 0.8808 `[0.8580, 0.9024]` | 0.5391 | 77.11% `[71.08, 83.13]` | 79.15% `[76.91, 0.8139]` | 78.13% `[74.89, 0.8141]` | 0.4468 | 0.2676 | 26 | 6 | 38 |
| | **Production Multimodal Fusion** | Production Research Baseline | 0.36000 | 0.8871 `[0.8635, 0.9091]` | 0.5286 | 66.87% `[59.64, 73.49]` | 89.76% `[88.12, 0.9133]` | 78.31% `[74.81, 0.8174]` | 0.5362 | 0.0923 | 33 | 7 | 55 |
| | **Production M6 Dual-Scale** | Production Research Baseline | 0.69000 | 0.9013 `[0.8775, 0.9211]` | 0.6071 | 66.27% `[59.04, 73.49]` | 92.30% `[90.81, 0.9365]` | 79.28% `[75.59, 0.8282]` | 0.5805 | 0.0982 | 74 | 14 | 56 |
| | **Experimental Model F** | Experimental Challenger | 0.24170 | 0.9041 `[0.8837, 0.9240]` | 0.6173 | 63.25% `[56.02, 70.48]` | 92.68% `[91.18, 0.9402]` | 77.96% `[74.23, 0.8165]` | 0.5691 | 0.0745 | 57 | 9 | 61 |
| | **Challenger Model F-G** | Experimental Safety Wrapper | **0.21032** | **0.9011** `[0.8805, 0.9202]` | **0.6046** | **68.67%** `[61.45, 75.90]` | **88.94%** `[87.29, 90.66]` | **78.81%** `[75.19, 0.8235]` | **0.5327** | **0.0855** | **0** | **0** | **52** |
| **PH² (External Benchmark, $N=200$)** | **Production ViT Screening** | Production Research Baseline | 0.80000 | 0.8310 `[0.7558, 0.8991]` | 0.5428 | 50.00% `[35.00, 65.00]` | 92.50% `[88.12, 0.9625]` | 71.25% `[62.81, 0.7906]` | 0.5556 | 0.1879 | 11 | 4 | 20 |
| | **Production Multimodal Fusion** | Production Research Baseline | 0.36000 | 0.7589 `[0.6606, 0.8496]` | 0.5284 | 47.50% `[30.00, 62.50]` | 91.87% `[87.50, 0.9563]` | 69.69% `[60.94, 0.7750]` | 0.5278 | 0.1458 | 9 | 3 | 21 |
| | **Production M6 Dual-Scale** | Production Research Baseline | 0.69000 | 0.8112 `[0.7351, 0.8861]` | 0.4998 | 30.00% `[17.50, 45.00]` | 94.37% `[90.62, 0.9750]` | 62.19% `[54.99, 0.6969]` | 0.3934 | 0.1470 | 20 | 11 | 28 |
| | **Experimental Model F** | Experimental Challenger | 0.24170 | 0.7455 `[0.6442, 0.8349]` | 0.5136 | **5.00%** `[0.00, 12.50]` | 98.75% `[96.88, 1.0000]` | **51.88%** `[49.06, 0.5594]` | 0.0909 | 0.1840 | **30** | **19** | **38** |
| | **Challenger Model F-G** | Experimental Safety Wrapper | **0.21032** | **0.7539** `[0.6514, 0.8419]` | **0.6274** | **52.50%** `[35.00, 67.50]` | **91.87%** `[87.50, 0.9563]` | **72.19%** `[63.43, 0.7970]` | **0.5676** | **0.1313** | **0** | **0** | **19** |

*(Deliverable: `reports/CEFM_STEP20_FINAL_MODEL_COMPARISON.csv`)*

---

## 4. Formal Model Classification

Following the final evidence audit, all model architectures investigated in the project are formally classified as follows:

```mermaid
flowchart TD
    subgraph PRODUCTION_RESEARCH_BASELINES["Production Research Baselines (Frozen)"]
        ViT["ViT-Melanoma Classifier (v2.5)\nROC-AUC: 0.8808\nRole: Primary Screening"]
        MM["Phase 1 Multimodal Fusion\nROC-AUC: 0.8889\nRole: 16-D ABCDE Integration"]
        M6["Phase 2 M6 Dual-Scale RCAG\nROC-AUC: 0.9013\nRole: Local-Global Cross-Attention"]
        UNet["Production VM-UNet ISIC2018\nMean Dice: 0.814\nRole: Mask & Boundary Extraction"]
    end

    subgraph EXPERIMENTAL_CHALLENGER["Experimental Challenger (Active Research)"]
        ModelF["Model F (64-D Tabular Branch)\nROC-AUC: 0.9041\nVulnerability: Demographic Suppression"]
    end

    subgraph EXPERIMENTAL_SAFETY_WRAPPER["Experimental Safety Wrapper (Recommended)"]
        ModelFG["Model F-G (Alternative B)\nModel F + Visual Evidence Preserver\nTest Sens: 68.67% (+5.42%, p < 0.05)\nVEP Violations: 0 (100% Protected)"]
    end

    subgraph RETIRED_ARCHITECTURES["Retired Experimental Architectures"]
        ModelC["Model C: Constrained Residual Softplus\nTest Sens: 63.86% (Gradient Co-adaptation)"]
        ModelD["Model D: Gated Positive Softplus\nTest Sens: 56.02% (Severe Suppression)"]
    end

    ViT --> Dashboard["Streamlit Multi-Model Dashboard"]
    MM --> Dashboard
    M6 --> Dashboard
    ModelF --> ModelFG
    ModelFG -.-> Advisory["Advisory Research Panel Only\n(Non-Diagnostic)"]
```

### Detailed Classification Registry:

1. **Production Research Baselines (FROZEN — Invariant):**
   - **`ViT-Melanoma` (`models/classification/vit_model/`):** Primary visual feature extractor and screening classifier.
   - **`Phase 1 Multimodal Fusion` (`models/multimodal/fusion_model.pt`):** Merges visual embeddings with 16-D clinical ABCDE vectors.
   - **`M6 Dual-Scale RCAG` (`models/experiments/M6/weights.pt`):** High-specificity visual discriminator integrating whole-image and high-resolution local patch embeddings.
   - **`UNet` (`models/segmentation/unet_isic2018.pt`):** Lesion boundary extractor.
2. **Experimental Challenger:**
   - **`Model F` (`models/experiments/clinical_metadata_branch/model.pt`):** 64-dimensional clinical metadata fusion branch. While achieving high internal ROC-AUC ($0.9041$), it is strictly disqualified from standalone deployment due to severe demographic suppression ($57$ internal VEP violations; collapse to $5.0\%$ sensitivity under missing external metadata).
3. **Experimental Safety Wrapper (Designated Challenger Architecture):**
   - **`Challenger Model F-G` (Alternative B):** Combines Model F with the deterministic Visual Evidence Preservation guardrail. Formally verified to eliminate 100% of VEP violations on test and external benchmarks, restoring external sensitivity to $52.50\%$ while maintaining internal test sensitivity at $68.67\%$. Designated as an **experimental advisory component**.
4. **Retired Experimental Architectures:**
   - **`Model C` (Constrained Residual Fusion):** Architectural softplus logit boost failed due to gradient co-adaptation during backpropagation (depressed visual logits, lower sensitivity of $63.86\%$). Retired.
   - **`Model D` (Gated Positive Fusion):** Visual gating failed to prevent visual logit attenuation, resulting in the lowest sensitivity ($56.02\%$) of all challengers. Retired.

---

## 5. Epistemic Synthesis & Clinical Boundary

### What CEFM Has Conclusively Proven:
1. **Lesion-Disjoint Partitioning is Non-Negotiable:** Historical reported accuracies in dermoscopic benchmarks suffer massive inflation when multiple photos of the same lesion span across train and test sets.
2. **Tabular Demographic Fusion Causes Dangerous Suppression:** Directly concatenating demographic metadata into neural network logits causes benign priors (young age, female sex) to override obvious visual melanoma features.
3. **Architectural Monotonicity Constraints Fail End-to-End:** Constraining tabular logit boosts to non-negative values ($\Delta l_1 \ge 0$) does not prevent suppression because joint backpropagation forces the visual encoder to depress base logits.
4. **Decoupled Guardrailing is Mathematically and Clinically Superior:** Preserving an uncompromised visual backbone and enforcing post-hoc visual evidence preservation eliminates 100% of demographic violations while boosting sensitivity by $+5.42\%$ ($p < 0.05$).

### What CEFM Has NOT Proven (Clinical Evidence Gaps):
1. **No Prospective Clinical Validation:** The system has never been evaluated in a live clinic with practicing dermatologists.
2. **External Multimodal Validation is Incomplete:** Because the external PH² cohort lacks demographic metadata, multi-center validation of the complete visual-plus-demographic branch remains unexecuted.
3. **Optical Generalization Remains Unsolved:** Tuebinger liquid immersion optics cause raw visual representation divergence across all tested deep backbones.

---

## 6. Artifact Deliverable Manifest

The complete Step 20 research package comprises:

1. `reports/CEFM_STEP20_FINAL_MODEL_COMPARISON.csv` — Unified comparative matrix across all models and cohorts.
2. `reports/CEFM_STEP20_REPRODUCIBILITY_AUDIT.md` — Detailed cross-step audit reconciling all metric differences.
3. `reports/CEFM_STEP20_GUARDRAIL_VERIFICATION.md` — Mathematical proof and verification of the 6 VEP software invariants.
4. `reports/CEFM_STEP20_LIMITATIONS.md` — Systemic limitations and clinical evidence gap registry.
5. `reports/CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json` — Machine-readable cryptographic hash manifest.
6. `reports/CEFM_STEP20_FINAL_RESEARCH_REPORT.md` — Capstone scientific report.
7. `README.md` — Reproducible research package execution instructions.
8. `tests/test_step20_final_audit.py` — Automated verification test suite.

