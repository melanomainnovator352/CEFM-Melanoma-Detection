# CEFM STEP 19: EXTERNAL DATASET SUITABILITY AUDIT
## Systematic Evaluation of Independent Dermoscopic Cohorts for Multimodal Model Validation

**Document ID:** CEFM-STEP19-SUITABILITY-20261107  
**Execution Timestamp:** 2026-11-07T22:20:00Z  
**Target:** Challenger Model F-G Generalization & Domain Portability  
**Status:** OFFICIAL BENCHMARK SUITABILITY ASSESSMENT  

---

> [!IMPORTANT]
> **EPISTEMIC INTEGRITY MANDATE: INTERNAL VALIDATION IS NOT EXTERNAL VALIDATION**  
> In accordance with strict machine learning and clinical audit standards, an internal test split—regardless of whether it is lesion-disjoint, patient-disjoint, or center-stratified—originates from the same underlying distribution as the training data (identical camera hardware, clinic patient demographics, clinical photography protocols, and institutional biopsying practices). It can never serve as proof of real-world domain generalization. Independent external validation requires distinct clinical institutions, unseen acquisition hardware, and geographically separate patient populations.

---

## 1. Candidate Dataset Evaluation Matrix

To establish a principled external validation framework for CEFM Challenger Model F-G (Alternative B: Model F + Visual Evidence Preservation Guardrail), we systematically audited 8 public dermatological benchmarks against CEFM's technical and clinical requirements:

| Benchmark Name | Host Institution / Origin | License / Research Terms | Sample Size ($N$) | Diagnostic Ground Truth | Acquisition Hardware & Optics | Lesion / Patient IDs | Tabular Metadata (Age, Sex, Site) | Modality & CEFM Target Compatibility | Local Disk Status | Suitability Verdict |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **HAM10000** | Medical Univ. Vienna & Cliff Rosendahl Clinic (Austria/Australia) | CC-BY-NC 4.0 | 10,015 | 7 classes (histopathology & expert consensus) | Contact non-polarized & polarized dermoscopy (DermLite, Heine) | Yes (`lesion_id`, 7,470 unique) | Complete (Age, Sex, 15 Anatomical Sites) | Dermoscopic; Binary ($1=\text{mel}$, $0=\text{non-mel}$) | Present (`datasets/HAM10000/`) | **Internal Baseline Only** (Training source) |
| **PH²** | Pedro Hispano Hospital (Matosinhos, Portugal) | Open Academic / Free for Research | 200 | Histopathology: 80 Common Nevi, 80 Atypical Nevi, 40 Melanomas | Tuebinger Mole Analyzer, contact liquid immersion (gel/fluid) | No (1 image per lesion, no patient ID) | **ABSENT** (No Age, Sex, or Anatomical Site) | Dermoscopic; Compatible ($1=\text{mel}$, $0=\text{non-mel}$) | Present (`D:\PH2\`) | **Approved External Optical Benchmark** (Missing metadata mode) |
| **ISIC 2018 (Task 1)** | ISIC / HAM10000 / MSKCC | CC-BY-NC 4.0 | 2,594 train, 100 val, 1,000 test | Task 1 binary segmentation masks | Dermoscopic | No diagnostic labels on local disk | N/A (Masks only on disk) | Segmentation Only | Present (`datasets/ISIC2018/`) | **Ineligible for Classification** (Lacks diagnostic CSV) |
| **ISIC 2019 / BCN20000** | Hospital Clínic de Barcelona & ViDIR (Spain/Austria) | CC-BY-NC 4.0 | 25,331 | 8 classes (histopathology confirmed) | DermLite & Heine dermoscopes | Partial (`lesion_id` for subset) | Yes (Age, Sex, Anatomical Site ~95%) | Dermoscopic; Multi-class reducible to binary | Not present locally | **Ineligible Without Deduplication** (10,015 HAM images overlap) |
| **ISIC 2020** | Melanoma Institute Australia, Univ. Sydney, MSKCC | CC-BY-NC 4.0 | 33,126 | Histopathology: 584 Melanomas, 32,542 Benign | Predominantly wide-field & handheld dermoscopy | Yes (`patient_id`) | Yes (Age approx, Sex, Anatomical Site) | Dermoscopic; Extreme imbalance (1.76% melanoma) | Not present locally | **Future Candidate** (Requires class rebalancing) |
| **PAD-UFES-20** | Federal Univ. of Espírito Santo (Brazil) | CC-BY 4.0 | 2,298 | Biopsy: 6 classes (BCC, MEL, SCC, ACK, SEK, NEV) | Smartphone cameras (clinical macroscopic photography) | Yes (`patient_id`) | Complete (Age, Sex, Anatomical Site, Fitspatrick) | **Clinical Photography** (Non-dermoscopic macro) | Not present locally | **Incompatible Modality** (Clinical vs Dermoscopic) |
| **MED-NODE** | Univ. Medical Center Groningen (Netherlands) | Open Research | 170 | 70 Melanomas, 100 Nevi | Macroscopic digital camera images (Nikon Coolpix) | No | **ABSENT** | **Clinical Photography** (Non-dermoscopic macro) | Not present locally | **Incompatible Modality** (No dermoscopy, no metadata) |
| **Derm7pt** | Interactive Dermoscopy Atlas (Argenziano et al.) | Academic DUA | 1,011 pairs | 7-point criteria & histopathology | Dermoscopic & clinical paired images | No patient grouping | Complete (Age, Sex, Location, 7-point scores) | Compatible dermoscopy | Not present locally | **Candidate for Future Multi-Center Study** |

---

## 2. In-Depth Suitability Analysis of Candidates

### 2.1 PH² Dermoscopy Benchmark ($N=200$)
- **Origin & Clinical Setting:** Collected under the supervision of dermatologists at the Dermatology Service of Pedro Hispano Hospital in Matosinhos, Portugal.
- **Hardware & Optics:** Captured using a Tuebinger Mole Analyzer system with a magnification of 20x. The system uses **fluid immersion dermoscopy** (liquid interface with oil or gel) and non-polarized illumination.
- **Diagnostic Distribution:**
  - $40$ Melanomas ($20.0\%$, histologically confirmed).
  - $80$ Atypical / Dysplastic Nevi ($40.0\%$, clinically and histologically evaluated).
  - $80$ Common Melanocytic Nevi ($40.0\%$).
- **Tabular Metadata Audit:** **ZERO** patient demographic metadata is provided. Patient age, biological sex, and anatomical location are unrecorded in the benchmark release.
- **Lesion / Patient Identifiers:** None. Each image represents an isolated cutaneous lesion; patient-level clustering cannot be performed.
- **Domain Generalization Value:**
  1. It provides an uncompromised test of **optical domain shift** (Tuebinger liquid immersion vs HAM10000 contact/polarized dermoscopy).
  2. Because all demographic metadata is missing, it provides the ultimate real-world test of **multimodal missingness robustness**: evaluating whether Model F-G preserves visual diagnostic fidelity when clinical metadata is completely absent.

### 2.2 ISIC 2018 (Local Directory Audit)
- **Forensic Inspection of `datasets/ISIC2018/`:**
  - Files present: `ISIC2018_Task1-2_Training_Input` (2,594 images), `ISIC2018_Task1_Training_GroundTruth` (2,594 segmentation masks), `ISIC2018_Task1-2_Validation_Input` (100 images), `ISIC2018_Task1-2_Test_Input` (1,000 images).
  - Diagnostic Metadata: `metadata/` directory is completely empty; zero classification ground-truth CSVs exist on disk.
  - Training Contamination: The ISIC 2018 Task 3 classification challenge was built by pooling HAM10000 with MSKCC archives. Even if the labels were downloaded, over 70% of the training images are identical to HAM10000, creating severe training contamination.
  - **Verdict:** Local ISIC 2018 files are suitable strictly for segmentation (UNet validation), not for classification external validation.

### 2.3 ISIC 2019 / BCN20000
- **Contamination Risk:** The ISIC 2019 challenge dataset contains 25,331 images. However, **10,015 of those images are the exact HAM10000 dataset**. Evaluating a HAM10000-trained model on ISIC 2019 without cryptographic hash deduplication would result in evaluating training data as test data.
- **Verdict:** Ineligible for out-of-the-box validation without rigorous image-hash deduplication against the internal HAM10000 partition.

### 2.4 ISIC 2020 Tele-Dermatology Benchmark
- **Characteristics:** 33,126 dermoscopic images from 2,056 patients across Australia and the US. Contains complete metadata (`patient_id`, `age_approx`, `sex`, `anatom_site_general_challenge`).
- **Clinical Nuance:** The prevalence of melanoma is extremely low ($584 / 33,126 = 1.76\%$). Evaluating a model calibrated at an operating point for a secondary screening clinic ($\sim 11\%$ prevalence) without prevalence re-weighting or threshold adjustment would lead to high false-positive call rates.
- **Verdict:** Prime candidate for future large-scale tele-triage validation, but requires explicit Bayesian prevalence adjustment.

### 2.5 PAD-UFES-20 and MED-NODE
- **Modality Mismatch:** Both datasets consist of macroscopic clinical photographs taken with consumer digital cameras or smartphones.
- **Pathological Rationale:** Dermoscopic models are trained on microscopic transillumination patterns (pigment network atypicality, streaks, blue-white veil) that are invisible in macroscopic clinical photographs. Applying a dermoscopic ViT backbone to clinical macroscopic images represents an invalid out-of-domain modality violation.
- **Verdict:** Disqualified due to optical imaging modality mismatch.

---

## 3. Epistemic Protocol for Step 19 Validation

Based on the suitability audit:
1. **PH² is the sole external dermoscopic cohort present on disk** with verified diagnostic ground truth.
2. Because PH² lacks patient age, sex, and anatomical location, **evaluating Model F-G on PH² must strictly execute under the standardized missing metadata protocol** (`all_metadata_missing`: $age=0, missing=1, sex=[0,0,1], loc=0$).
3. PH² results must be transparently disclosed as an **optical and missing-data generalization stress test**, not as an external validation of complete multimodal fusion.
4. The internal HAM10000 lesion-disjoint test set ($N=1,504$) must be preserved as the **internal multi-modal benchmark**, where complete patient metadata is available.
5. All evaluations must freeze model weights and decision thresholds ($\tau = 0.21032$). No post-hoc recalibration is permitted.

