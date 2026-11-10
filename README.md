# Clinical Explainable Fusion Model (CEFM)
## Forensic Research Package & Reproducible Evaluation Harness

**Version:** 2.0 (Steps 1–20 Research Consolidation)  
**Status:** Research Decision Support Prototype (Non-Diagnostic)  
**Notice:** Research use only. Not approved for autonomous clinical diagnosis.

---

## 1. Environment Setup

### System Prerequisites
- **Python:** 3.10.x
- **OS:** Windows / Linux / macOS
- **Hardware Acceleration:** CUDA 11.8+ / 12.x compatible GPU recommended (CPU execution supported).

### Virtual Environment & Dependencies
```bash
# Clone or navigate to the repository root
cd d:/pytest_cache

# Activate Python 3.10 virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python pandas numpy scikit-learn pytest streamlit
```

---

## 2. Dataset Organization

All evaluation pipelines strictly enforce **lesion-disjoint partitioning** to eliminate data leakage.

```
d:/pytest_cache/
├── datasets/
│   ├── HAM10000/
│   │   ├── HAM10000_images_part_1/
│   │   ├── HAM10000_images_part_2/
│   │   ├── HAM10000_metadata.csv
│   │   └── processed_lesion_disjoint/
│   │       ├── train.csv         # 6,978 images (5,188 unique lesions)
│   │       ├── validation.csv    # 1,533 images (1,141 unique lesions)
│   │       └── test.csv          # 1,504 images (1,121 unique lesions)
│   └── ISIC2018/
│       ├── ISIC2018_Task1-2_Training_Input/
│       └── ISIC2018_Task1_Training_GroundTruth/
└── D:/PH2/                       # Independent External Benchmark
    ├── PH2 Dataset images/       # 200 high-resolution dermoscopic folders
    ├── PH2_dataset.txt           # Verified histopathological ground truth
    └── Readme.txt
```

---

## 3. Checkpoint Verification

Before running inference, verify that all production baselines and experimental models match their authoritative SHA-256 digests:

```bash
# Run automated checkpoint verification test
pytest tests/test_step20_final_audit.py -k test_checkpoint_invariance -v
```

### Authoritative Checkpoint Registry:
| Model Name | Checkpoint File | SHA-256 Digest |
| :--- | :--- | :--- |
| **ViT Classifier** | `models/classification/vit_model/model.safetensors` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` |
| **Multimodal Fusion** | `models/multimodal/fusion_model.pt` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` |
| **M6 Dual-Scale** | `models/experiments/M6/weights.pt` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` |
| **UNet Segmenter** | `models/segmentation/unet_isic2018.pt` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` |
| **Model F (Tabular)** | `models/experiments/clinical_metadata_branch/model.pt` | `27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c` |
| **Model A (Visual)** | `models/experiments/clinical_metadata_branch/model_A_visual_only.pt` | `1ad1543087a51e9ef1463483cdc8e4606d2bbd40aeb9483762232c993187cc5b` |

---

## 4. Evaluation Commands

### Step 12: Baseline Benchmark Evaluation
```bash
python scripts/evaluate_lesion_disjoint_models.py
```
Evaluates frozen Production ViT, Phase 1 Multimodal Fusion, and M6 Dual-Scale on the 1,504 lesion-disjoint test set.

### Step 18: Constrained Residual & Gated Fusion Training
```bash
python scripts/run_step18_experiments.py
```
Trains Alternative C (Constrained Residual) and Alternative D (Gated Positive), validates against Alternative B (Model F + Guardrail), and executes 2,000 paired bootstrap resamples.

### Step 19: External Benchmark & Robustness Evaluation
```bash
python scripts/run_step19_external_validation.py
```
Evaluates all models on the external PH² cohort ($N=200$) under standardized missing-data conditions, and stress-tests 14 operational perturbation scenarios.

### Step 20: Final Consolidation & Evidence Audit
```bash
python scripts/build_step20_deliverables.py
```
Generates `reports/CEFM_STEP20_FINAL_MODEL_COMPARISON.csv` and `reports/CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json`.

---

## 5. Automated Regression Test Suite

Execute the entire project test suite across Steps 15 through 20:

```bash
pytest tests/test_step20_final_audit.py \
       tests/test_step19_external_validation.py \
       tests/test_step18_constrained_fusion.py \
       tests/test_step17_clinical_safety.py \
       tests/test_step16_clinical_metadata.py \
       tests/test_step15_multi_model_dashboard.py -v
```

---

## 6. Multi-Model Interactive Dashboard

Launch the clinical decision-support interface:

```bash
streamlit run streamlit_app1.py
```
The dashboard displays:
- Tri-model assessment panel (ViT, Multimodal Fusion, M6 Dual-Scale).
- Model discordance warning banners.
- Automated data quality and segmentation boundary validation.
- Multi-photo lesion aggregation logic.

---

## 7. Regulatory & Clinical Boundary

> **NON-DIAGNOSTIC RESEARCH PROTOTYPE**  
> CEFM has been designed for forensic validation and multi-agent machine learning research. It has not undergone prospective clinical trials, blinded reader studies, or regulatory review. It must not be used for autonomous clinical diagnosis or treatment planning.
