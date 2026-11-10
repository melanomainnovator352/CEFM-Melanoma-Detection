# CEFM — Reproducibility Guide

**Document ID:** `CEFM-REPRODUCIBILITY-2026-11-07`  
**Purpose:** Deterministic, end-to-end instructions for reproducing all CEFM Phase 1 and Phase 2 experimental benchmarks, feature caches, model checkpoints, and evaluation tables.  

---

## 1. Environment & Determinism Setup

### System Specifications
- **Operating System:** Windows 11 / Ubuntu 22.04 LTS
- **Python Version:** Python 3.10.0+
- **CUDA Toolkit:** CUDA 11.8 / 12.1 with cuDNN 8.x
- **PyTorch Version:** PyTorch 2.1.2+cu118

### Environment Setup
```bash
git clone https://github.com/your-org/cefm-melanoma.git
cd cefm-melanoma

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install exact pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
```

### Random Seed Configuration
All experiments and bootstrap procedures enforce deterministic pseudo-random seeds:
- `RANDOM_SEED = 42`
- PyTorch Determinism:
  ```python
  import torch, numpy as np, random
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  np.random.seed(42)
  random.seed(42)
  torch.backends.cudnn.deterministic = True
  ```

---

## 2. Dataset Partitioning & Leakage Safeguards

The HAM10000 benchmark ($N = 10,015$) was partitioned using strict **lesion-stratified grouping**:
- `data/HAM10000/metadata/train.csv`: $6,978$ samples ($771$ melanoma, $6,207$ non-melanoma)
- `data/HAM10000/metadata/validation.csv`: $1,533$ samples ($176$ melanoma, $1,357$ non-melanoma)
- `data/HAM10000/metadata/test.csv`: $1,504$ samples ($166$ melanoma, $1,338$ non-melanoma)
- **Zero Leakage Gate:** Zero shared `lesion_id` or `patient_id` across splits. Test set was completely unblinded only for Step 4 and Step 5 evaluations.

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Dual-Scale Feature Extraction & Caching
Extracts 768-D global context embeddings, 768-D local high-res patch embeddings, and 16-D clinical ABCDE vectors:
```bash
python scripts/cache_dual_scale_features.py
python scripts/verify_dual_scale_cache.py
```
*Expected Outputs in `data/dual_scale_cache/`:*
- `train_dual_scale.pt` ($N = 6,978$)
- `val_dual_scale.pt` ($N = 1,533$)
- `test_dual_scale.pt` ($N = 1,504$)

### Step 2: Training Phase 2 Dual-Scale Ablation Models (M1–M7)
Trains the 7 experimental architectures with early stopping based on validation AUC:
```bash
python scripts/train_phase2_experiments.py
```
*Trained Architectures Saved to `models/experiments/`:*
- `M1/weights.pt`: Global ViT Only
- `M2/weights.pt`: Local ViT Only
- `M3/weights.pt`: Dual-Scale Visual Concatenation
- `M4/weights.pt`: Global ViT + Clinical Features
- `M5/weights.pt`: Dual-Scale Early Concatenation
- `M6/weights.pt`: Dual-Scale Gated RCAG (Challenger)
- `M7/weights.pt`: Dual-Scale Cross-Attention

### Step 3: Threshold Calibration on Validation Split
Calibrates optimal F1 and 90% specificity operating points exclusively on validation data:
```bash
python scripts/calibrate_validation_thresholds.py
```
*Verified M6 Thresholds in `models/experiments/M6/config.json`:*
- $T^*_{F1} = 0.690$
- $T^*_{spec} = 0.625$

### Step 4: Independent Research Validation (Step 4)
Runs 1,000 stratified bootstrap resamples on the untouched test split ($N = 1,504$):
```bash
python scripts/evaluate_phase2_research_validation.py
python scripts/audit_step4_consistency.py
```
*Expected Outputs:*
- `reports/CEFM_PHASE2_STEP4_RESEARCH_VALIDATION_REPORT.md`
- `reports/CEFM_PHASE2_STEP4_FINAL_CONSISTENCY_AUDIT.md`
- `reports/figures/m6_vs_baselines_roc_curves.png`
- `reports/figures/m6_vs_baselines_pr_curves.png`

### Step 5: Independent External Validation on PH² (Step 5)
Evaluates frozen models on the external PH² dermoscopy database ($N = 200$):
```bash
python scripts/evaluate_ph2_external_validation.py
```
*Expected Outputs:*
- `reports/CEFM_PHASE2_STEP5_EXTERNAL_VALIDATION.md`
- `reports/CEFM_PHASE2_STEP5_METRICS.csv`
- `reports/figures/phase2_external_validation/ph2_external_roc_curves.png`
- `reports/figures/phase2_external_validation/ph2_external_pr_curves.png`

### Step 6: Automated Regression Test Suite
Executes all 79 unit, integration, and UI regression tests:
```bash
pytest tests/ -v
```
*Passing Standard: 79 passed, 0 failed.*

