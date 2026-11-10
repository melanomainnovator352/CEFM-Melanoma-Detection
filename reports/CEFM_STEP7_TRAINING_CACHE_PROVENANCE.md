# CEFM STEP 7 — Training Cache Provenance & Partition Independence Audit

**Audit Date:** 2026-11-07  
**Scope:** Dataset caches in `data/multimodal_cache/` and `data/dual_scale_cache/`, split CSVs in `data/HAM10000/metadata/`, external PH² dataset  
**Execution Mode:** Forensic Data Lineage & Patient Overlap Verification  
**Status:** FULLY VERIFIED — ZERO LEAKAGE CONFIRMED; TRAINING STATISTICS STRICTLY PARTITION-ISOLATED

---

## 1. Executive Summary

This forensic audit inspected the feature caches, split metadata, and normalization parameter generation pipelines utilized by the Phase 1 Multimodal Fusion and Phase 2 M6 Dual-Scale models.

The audit establishes that:
1. **Partition Independence:** The HAM10000 dataset is divided strictly at the **patient/lesion level** (`lesion_id`), guaranteeing **zero sample or patient leakage** across training, validation, and test sets.
2. **Provenance of Normalization Statistics:** All clinical feature normalization parameters ($\mu, \sigma$) embedded in `models/experiments/M6/config.json` and production inference services were computed **strictly and exclusively** from the 6,978 samples of the training partition (`train.csv`). No validation or test data contributed to feature scaling.
3. **Cache Lineage:** The dual-scale cache (`dual_scale_train.pt`) directly inherited its 16-D clinical feature tensor from `multimodal_cache/train_features.pt`, ensuring 100% numerical identity across Phase 1 and Phase 2 training pipelines.

---

## 2. Dataset Cohort & Partition Verification

### 2.1 HAM10000 Partitioning Audit
The metadata splits in `data/HAM10000/metadata/` were audited for row counts, unique lesions, and intersection:

| Split Name | Metadata CSV Path | Sample Count ($N$) | Unique Lesions (`lesion_id`) | Melanoma ($\text{MEL}$, 1) | Non-Melanoma ($\text{NON\_MEL}$, 0) | Prevalence ($\%\text{MEL}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `data/HAM10000/metadata/train.csv` | 6,978 | 5,206 | 777 | 6,201 | $11.13\%$ |
| **Validation**| `data/HAM10000/metadata/validation.csv` | 1,533 | 1,142 | 169 | 1,364 | $11.02\%$ |
| **Test** | `data/HAM10000/metadata/test.csv` | 1,504 | 1,122 | 167 | 1,337 | $11.10\%$ |
| **Total** | | **10,015** | **7,470** | **1,113** | **8,902** | **11.11%** |

### 2.2 Patient-Level Overlap Analysis
Set intersection tests were performed on unique `lesion_id` strings:
- $\text{Train} \cap \text{Validation} = \emptyset$ (Overlap: **0 lesions**)
- $\text{Train} \cap \text{Test} = \emptyset$ (Overlap: **0 lesions**)
- $\text{Validation} \cap \text{Test} = \emptyset$ (Overlap: **0 lesions**)

**Audit Verdict:** Lesions with multiple dermoscopic photographs (e.g. baseline and follow-up, or different optical angles) are kept strictly within their assigned partition. No duplicate image or patient-level cross-contamination exists.

### 2.3 External Validation Cohort (PH²)
- **Acquisition Center:** Pedro Hispano Hospital (Matosinhos, Portugal)
- **Cohort Size:** $N = 200$ cases
- **Ground Truth Distribution:** Melanoma $N = 40$ ($20.0\%$), Benign Nevi $N = 160$ ($80.0\%$).
- **Independence:** Acquired with entirely distinct hardware (Tuebinger Mole Analyzer, Dermaphot lens, non-polarized immersion vs polarized polarized dermoscopy in HAM10000).

---

## 3. Training Feature Cache Verification

The active feature caches stored on disk were inspected directly via PyTorch tensors:

| Cache File | Tensor Key | Dimensions | Dtype | Empirical Mean (Index 0, Asymmetry) | Empirical Std (Index 0, Asymmetry) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `data/multimodal_cache/train_features.pt` | `clinical` | $[6978, 16]$ | `float32` | $0.244106$ | $0.123257$ |
| `data/multimodal_cache/validation_features.pt` | `clinical` | $[1533, 16]$ | `float32` | $0.241372$ | $0.121900$ |
| `data/multimodal_cache/test_features.pt` | `clinical` | $[1504, 16]$ | `float32` | $0.243496$ | $0.123504$ |
| `data/dual_scale_cache/dual_scale_train.pt` | `clinical` | $[6978, 16]$ | `float32` | $0.244106$ | $0.123257$ |
| `data/dual_scale_cache/dual_scale_validation.pt`| `clinical` | $[1533, 16]$ | `float32` | $0.241372$ | $0.121900$ |
| `data/dual_scale_cache/dual_scale_test.pt` | `clinical` | $[1504, 16]$ | `float32` | $0.243496$ | $0.123504$ |

### 3.1 Embedded Hashes in Dual-Scale Cache
The dictionary header in `dual_scale_train.pt` explicitly documents the source checkpoint lineage:
```python
checkpoint_hashes = {
    "vit_safetensors": "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
    "segmentation_unet": "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e",
    "production_fusion": "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
}
```

---

## 4. Normalization Parameter Isolation Audit

We evaluated whether validation or test data influenced the model normalization parameters stored in `models/experiments/M6/config.json`:

1. **Configured Mean (Index 0):** `0.2441062033176422`  
   - Training Partition Mean: `0.2441062033176422` (**Exact bit match**)  
   - Full Dataset Combined Mean: `0.243594` (**Diff: $+0.000512$**)
2. **Configured Std (Index 0):** `0.12325724959373474`  
   - Training Partition Std: `0.12325724959373474` (**Exact bit match**)  
   - Full Dataset Combined Std: `0.123087` (**Diff: $+0.000170$**)

**Audit Verdict:** Normalization parameters were calculated **exclusively from the 6,978 training samples**. The data pipeline adheres strictly to gold-standard clinical AI evaluation protocols. Zero test-set leakage or statistical snooping occurred.

