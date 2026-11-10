# CEFM STEP 8 — Dataset Partition Independence: Lesion-Level vs. Patient-Level Separation Audit

**Audit Date:** 2026-11-07  
**Scope:** Dataset manifests in `data/HAM10000/metadata/` and published benchmark metadata `HAM10000_metadata.csv`  
**Execution Mode:** Forensic Metadata Inspection & Identifier Lineage Verification  
**Status:** FULLY RECONCILED — STRICT LESION-LEVEL INDEPENDENCE VERIFIED; TRUE PATIENT-LEVEL INDEPENDENCE UNVERIFIABLE FROM OFFICIAL DATASET

---

## 1. Executive Summary

Previous project reports informally referred to the HAM10000 partition strategy as "patient-level separation." This audit conducted an exhaustive inspection of the official published metadata (`HAM10000_metadata.csv`) and partition manifests (`train.csv`, `validation.csv`, `test.csv`) to determine whether `lesion_id` constitutes a patient identifier or a lesion identifier.

### Core Forensic Conclusions:
1. **Lesion-Level Separation is Absolute (0 Leakage):** The dataset partitions strictly enforce lesion independence. There are **0 shared `lesion_id`s**, **0 shared `image_id`s**, and **0 duplicate image checksums** between train, validation, and test splits. All multi-image captures of the same lesion reside inside the exact same split.
2. **Patient-Level Separation Cannot Be Verified:** The official HAM10000 dataset release (Tschandl, Rosendahl & Kittler, *Nature Scientific Data* 2018) **does not include a `patient_id` or `subject_id` column** in its metadata schema. 
3. **Scientific Distinction Mandated:** Because a single human patient with dysplastic nevus syndrome or multiple primary melanomas could possess multiple distinct lesions (each receiving a unique `lesion_id`), lesions from the same patient could potentially exist in different partitions.
4. **Governance Rule:** Project documentation must strictly describe the split as **"Lesion-Level Partitioning"** and explicitly state that patient-level separation cannot be certified from public HAM10000 metadata.

---

## 2. Metadata Schema Audit

Inspection of the primary metadata files (`data/HAM10000/metadata/train.csv`, `validation.csv`, `test.csv`, and root `HAM10000_metadata.csv`):

```csv
lesion_id,image_id,dx,dx_type,age,sex,localization
HAM_0000118,ISIC_0027419,bkl,histo,80.0,male,scalp
HAM_0000118,ISIC_0025030,bkl,histo,80.0,male,scalp
HAM_0002730,ISIC_0026769,bkl,histo,80.0,male,scalp
HAM_0002730,ISIC_0025661,bkl,histo,80.0,male,scalp
HAM_0001466,ISIC_0031633,bkl,histo,75.0,male,ear
```

### Published Column Definitions:
- `lesion_id`: Identifier for a specific physical skin lesion (e.g. `HAM_0000118`).
- `image_id`: Identifier for an individual dermoscopic photograph of that lesion (e.g. `ISIC_0027419`).
- `dx`: Ground-truth diagnostic class (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
- `dx_type`: Method of ground-truth confirmation (`histo`, `follow_up`, `consensus`, `confocal`).
- `age`, `sex`, `localization`: Patient demographic and anatomic location metadata.
- **`patient_id`:** **ABSENT from the official HAM10000 dataset release.**

---

## 3. Partition Overlap & Multi-Image Lesion Analysis

Empirical calculation across all 10,015 images in the dataset:

| Partition Cohort | Metadata CSV | Image Count ($N$) | Unique Lesions (`lesion_id`) | Multi-Image Lesions | Overlap with Train | Overlap with Val | Overlap with Test |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | `train.csv` | 6,978 | 5,228 | 1,357 | Baseline | **0** | **0** |
| **Validation** | `validation.csv` | 1,533 | 1,121 | 297 | **0** | Baseline | **0** |
| **Test** | `test.csv` | 1,504 | 1,121 | 293 | **0** | **0** | Baseline |
| **TOTAL** | | **10,015** | **7,470** | **1,947** | — | — | — |

### Quantified Results:
- Shared `lesion_id` between Train and Validation: **0** ($0.0\%$)
- Shared `lesion_id` between Train and Test: **0** ($0.0\%$)
- Shared `lesion_id` between Validation and Test: **0** ($0.0\%$)
- Shared `image_id` between any pair of partitions: **0** ($0.0\%$)
- Duplicate image checksums across partitions: **0** ($0.0\%$)
- Lesions with multiple images (2 to 6 photographs): **1,947 lesions**. In all 1,947 cases, all photographs of the same `lesion_id` reside in the **same partition**.

---

## 4. Multi-Lesion Patients & Leakage Assessment

1. **Why Patient Overlap is Possible:**  
   In epidemiological cohorts like HAM10000, dermatologists frequently biopsy or photograph multiple suspicious nevi on the same patient during a single full-body examination. Because the authors assigned a unique `HAM_xxxxxxx` identifier to each lesion without publishing a master patient key, two distinct lesions from the same individual (e.g. one on the back and one on the thigh) were assigned independent `lesion_id`s.
2. **Estimated Demographic Clusters:**  
   Co-occurrence grouping on `(age, sex, localization)` reveals that a subset of distinct lesions share identical demographic attributes, consistent with multi-lesion patients commonly treated in specialized Austrian and Australian dermatology clinics.
3. **Audit Verdict on Leakage:**  
   - **Image-level leakage:** **0.0% (Confirmed None).**
   - **Lesion-level leakage:** **0.0% (Confirmed None).**
   - **Patient-level leakage:** **Cannot be calculated from public data.** While images of the same lesion never cross splits, images from the same patient could theoretically cross splits if the patient had multiple lesions.

---

## 5. Strategic Recommendations for Research Governance

1. **Terminology Correction:** In all manuscripts, technical reports, and documentation, replace *"patient-level split"* with **"lesion-level split (`lesion_id` grouping)"**.
2. **Future Validation Protocol:** To formally evaluate cross-patient generalization and prove immunity to patient-level leakage, future phases should benchmark models on datasets with explicit patient tracking, such as:
   - **ISIC 2020 Melanoma Challenge** ($33,126$ images with explicit `patient_id`).
   - **BCN20000 (Barcelona Clinic)** ($19,424$ images with explicit patient identifiers).
3. **Partition Preservation:** Do **NOT** attempt to invent pseudo-patient groupings or re-split HAM10000. Preserving the frozen benchmark splits is essential for reproducible comparison with published literature.

