# CEFM Phase 2 — Reproducibility Manifest & Artifact Provenance

**Project:** Clinical-Expert-Feature-Modulated (CEFM) Multimodal Skin Lesion Classification  
**Phase:** Phase 2 Dual-Scale Fusion Research  
**Document Type:** Reproducibility Manifest, Environment Specification, and Checkpoint Provenance  
**Date:** November 7, 2026  
**Status:** COMPLETE (Reproducibility Standard Verified)  

---

## 1. System Execution Environment

All Phase 2 training, validation, testing, external evaluation, and ablation scripts were executed and verified within the following environment:

| Component | Specification |
|:----------|:--------------|
| **Operating System** | Microsoft Windows 11 Enterprise (64-bit) |
| **Python Runtime** | Python 3.10.0 (`C:\Program Files\Python310\python.exe`) |
| **PyTorch Version** | PyTorch 2.5.1+cu121 (CUDA 12.1 enabled, cuDNN 9.1 supported) |
| **HuggingFace Transformers** | Transformers 4.49.0 |
| **PyTorch Image Models (timm)**| timm 1.0.14 |
| **Computer Vision (OpenCV)** | OpenCV-Python 4.11.0.86 (headless / GUI compliant) |
| **Scientific Computing** | NumPy 1.26.4, SciPy 1.15.1, Pandas 2.2.3, Scikit-learn 1.6.1 |
| **Web Framework & UI** | FastAPI 0.115.8, Uvicorn 0.34.0, Streamlit 1.42.0 |
| **Database & Caching** | SQLite 3.39.2 (cefm.db), PyTorch Tensor Cache (`.pt`) |
| **Primary GPU** | NVIDIA GeForce RTX Laptop GPU (Compute Capability 8.6, 8,192 MB VRAM) |

---

## 2. Checkpoint SHA-256 Hash Manifest

To ensure absolute integrity, every model checkpoint is version-controlled with its SHA-256 hash. The table below represents the verified state:

| Model ID | Functional Role | File Path | File Size (Bytes) | SHA-256 Checksum | Production Status |
|:---------|:----------------|:----------|:------------------:|:-----------------|:-----------------:|
| **Phase 1 ViT** | Vision Baseline | `models/classification/vit_model/model.safetensors` | 343,223,968 | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | **ACTIVE DEFAULT** |
| **Phase 1 Fusion** | Multimodal Baseline | `models/multimodal/fusion_model.pt` | 505,629 | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | **ACTIVE DEFAULT** |
| **VM-UNet** | Lesion Segmentation | `models/segmentation/unet_isic2018.pt` | 7,822,971 | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | **ACTIVE DEFAULT** |
| **M1** | Global ViT Challenger | `models/experiments/M1/weights.pt` | 398,613 | `6e51e90f085c1909e60cdd6342b11d68d7ec42251becce365a01c6e16b5fa88d` | Research Challenger |
| **M2** | Local ViT Challenger | `models/experiments/M2/weights.pt` | 398,613 | `e8d1a4bf02b9d0e01780bc8fb8cb2e78c019e9d53b36c095b3c34cfcc20e2857` | Research Challenger |
| **M3** | Dual-Scale ViT | `models/experiments/M3/weights.pt` | 928,249 | `c1b6137bd6e2b09c255da97e78db477fe838a93247dce806287e60ac50885c33` | Research Challenger |
| **M4** | Global + Clinical | `models/experiments/M4/weights.pt` | 505,657 | `2ba424a5393ca183c2ea3d34331296f12a9d61031da7d5770a9b53986ada2d98` | Research Challenger |
| **M5** | Concat Multimodal | `models/experiments/M5/weights.pt` | 967,213 | `f01b9a991569f8ee15a22f1576c00d7bf6423c84070f2d885c3401b83b5720b6` | Research Challenger |
| **M6** | **Gated Dual-Scale (RCAG)**| `models/experiments/M6/weights.pt` | 1,268,559 | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | **Experimental Challenger** |
| **M7** | Cross-Attention | `models/experiments/M7/weights.pt` | 1,474,694 | `0af0bfc4c420dc4e5d63cbf7e442651db8fbbe684893d315bc011dc4d56dae78` | Research Challenger |

---

## 3. Preprocessing & Extraction Specifications

### 1. Dual-Scale Spatial View Generation (`app/services/dual_scale.py`)
- **Global Context View:**
  - Context expansion margin: **$1.30\times$** around lesion bounding box ($[ymin, ymax, xmin, xmax]$).
  - Target resolution: **$224 \times 224$ pixels** (rescaled via `cv2.INTER_AREA`).
- **Local Native-Resolution Patches:**
  - Native spatial resolution: **$224 \times 224$ pixels** per patch.
  - Overlap ratio: **$25\%$**.
  - Maximum patch limit: **$5$ patches**.
  - Patch pooling strategy: Weighted pooling giving **$50\%$ weight to the center patch** ($w_0 = 2.0$) and distributing $50\%$ equally among peripheral patches ($w_i = 1.0$).
- **ViT Embedding Extraction:**
  - Standard path applies post-layer-normalization CLS token:
    $$\text{emb} = \text{LayerNorm}(\text{ViTBlock}_{12}(\text{tokens}))_{[:, 0, :]}$$

### 2. Clinical Feature Normalization (`app/ml/multimodal/dual_scale_models.py`)
- Standardized 16-D ABCDE feature vector normalized with training split parameters:
  $$\tilde{c}_i = \frac{c_i - \mu_{\text{train}, i}}{\sigma_{\text{train}, i} + \epsilon}$$
  - $\epsilon = 1 \times 10^{-6}$ for invariant features.

---

## 4. Dataset Splits & Manifest Inventory

| Dataset Cohort | Purpose | Total Samples | Melanoma Cases (1) | Non-Melanoma Cases (0) | Split Mechanism |
|:---------------|:--------|:-------------:|:------------------:|:----------------------:|:----------------|
| **HAM10000 Train** | Model Training | 6,978 | 777 (11.1%) | 6,201 (88.9%) | Patient-level lesion hashing |
| **HAM10000 Val** | Model Selection & Calibration | 1,533 | 166 (10.8%) | 1,367 (89.2%) | Zero lesion overlap with train/test |
| **HAM10000 Test** | Independent Internal Test | 1,504 | 168 (11.2%) | 1,336 (88.8%) | Zero lesion overlap with train/val |
| **PH² External** | Out-of-Distribution Benchmark | 200 | 40 (20.0%) | 160 (80.0%) | Complete external cohort |

---

## 5. End-to-End Reproduction Instructions

To reproduce all findings from Phase 2 Steps 1–8:

1. **Verify Checkpoints & Dependencies:**
   ```powershell
   pytest tests/ -v
   ```
   *Expected:* 79 passed, 0 failed.

2. **Feature Extraction & Verification (Step 1):**
   ```powershell
   python scripts/cache_dual_scale_features.py
   python scripts/final_step1_verification.py
   ```

3. **Train Dual-Scale Models M1–M7 (Step 2):**
   ```powershell
   python scripts/train_phase2_experiments.py
   ```

4. **Internal Test Evaluation (Step 4):**
   ```powershell
   python scripts/evaluate_phase2_research_validation.py
   python scripts/audit_step4_consistency.py
   ```

5. **External Validation on PH² (Step 5):**
   ```powershell
   python scripts/evaluate_ph2_external_validation.py
   ```

6. **Domain-Shift Audit (Step 6):**
   ```powershell
   python scripts/audit_ph2_domain_shift.py
   ```

7. **Controlled Counterfactual Ablations (Step 7):**
   ```powershell
   python scripts/run_step7_domain_shift_ablation.py
   ```

8. **Final Consistency & Statistical Audit (Step 8):**
   ```powershell
   python scripts/run_step8_final_audit.py
   python scripts/build_master_results_csv.py
   ```

---

## 6. Production Safety Guarantee

- **Default Runtime:** FastAPI backend (`app/main.py`) and Streamlit frontend (`streamlit_app1.py`) default to `models/multimodal/fusion_model.pt` (Phase 1 Baseline).
- **Challenger Status:** M6 is gated behind `ENABLE_CHALLENGER_M6=true` and operates as a non-promoted challenger model.
- **Zero Inversion:** No production weights have been modified or overwritten.
