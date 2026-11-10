# CEFM Step 5 — Training–Inference Parity Validation Report

**Document ID:** `CEFM_TRAINING_INFERENCE_PARITY_REPORT`  
**Date:** November 2026  
**Status:** Validated and Verified Technical Report  
**Author:** Deep Learning Research Architect & Medical Machine Learning Engineer  
**Scope:** Verification of training cache manifolds, inference feature vectors, normalization statistics, and checkpoint compatibility.

---

## 1. Executive Summary & Forensic Audit Objective

A fundamental requirement for reliable medical machine learning systems is **strict parity between the data distribution seen during model training and the input tensors presented during production inference**.

This audit verifies that the corrections made in Step 5 restore complete mathematical parity between the frozen checkpoints trained on HAM10000 and the live inference pipeline.

---

## 2. Complete Cross-Split Feature Manifold Audit

The 16-dimensional clinical feature vector was audited across all splits:
- **Training Cache:** `data/dual_scale_cache/dual_scale_train.pt` ($N=6,978$) & `data/multimodal_cache/train_features.pt`
- **Validation Cache:** `data/dual_scale_cache/dual_scale_validation.pt` ($N=1,533$)
- **Test Cache:** `data/dual_scale_cache/dual_scale_test.pt` ($N=1,504$)
- **Production Preprocessing (Before):** Unshielded raw inputs with $1e-6$ division
- **Production Preprocessing (After):** Zero-variance gated standardization

### Table: 16-Dimensional Training–Inference Manifold Parity

| Idx | Feature Name | Training Mean | Training Std | Training % NZ | Production Raw | Production Normalized (Before) | Production Normalized (After) | Parity Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `asymmetry_score` | $0.2441$ | $0.1233$ | $99.7\%$ | $0.2500$ | $+0.0478$ | $+0.0478$ | **Exact Parity** |
| **1** | `horizontal_asymmetry` | $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **2** | `vertical_asymmetry` | $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **3** | `border_irregularity` | $0.4155$ | $0.1332$ | $99.7\%$ | $0.2753$ | $-1.0522$ | $-1.0522$ | **Exact Parity** |
| **4** | `border_available` | $1.0000$ | $0.0000$ | $100.0\%$ | $1.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **5** | `color_var_mean` | $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **6** | `color_var_std` | $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **7** | `color_h_std` | $0.6076$ | $0.3120$ | $99.7\%$ | $0.0189$ | $-1.8870$ | $-1.8870$ | **Exact Parity** |
| **8** | `color_s_std` | $0.2042$ | $0.0800$ | $99.7\%$ | $0.1996$ | $-0.0578$ | $-0.0578$ | **Exact Parity** |
| **9** | `color_v_std` | $0.1760$ | $0.0891$ | $99.7\%$ | $0.0559$ | $-1.3478$ | $-1.3478$ | **Exact Parity** |
| **10** | `color_available` | $0.9950$ | $0.0630$ | $99.7\%$ | $1.0000$ | $+0.0792$ | $+0.0792$ | **Exact Parity** |
| **11** | `diameter_norm` | $\mathbf{0.0000}$ | $\mathbf{0.0000}$ | $\mathbf{0.0\%}$ | $0.5980$ | $\mathbf{+598,000.0}$ | $\mathbf{0.0000}$ | **Restored to Training Manifold** |
| **12** | `diameter_available` | $\mathbf{0.0000}$ | $\mathbf{0.0000}$ | $\mathbf{0.0\%}$ | $1.0000$ | $\mathbf{+1,000,000.0}$ | $\mathbf{0.0000}$ | **Restored to Training Manifold** |
| **13** | `evolution_change` | $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **14** | `evolution_magnitude`| $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |
| **15** | `evolution_available`| $0.0000$ | $0.0000$ | $0.0\%$ | $0.0000$ | $0.0000$ | $0.0000$ | **Exact Parity** |

---

## 3. Checkpoint Integrity & Hash Verification

To guarantee that no model parameters, architectures, or weights were modified, all checkpoints were audited against the original Phase 2 Reproducibility Manifest:

| Model Component | Local File Path | Checkpoint Type | SHA-256 Hash Verified | Manifest Match |
| :--- | :--- | :--- | :--- | :---: |
| **ViT Backbone** | `models/classification/vit_model/model.safetensors` | Safetensors | `5e911c7503ad56db740ad622a55928d2d6c141d8e136b8c9197c88b64e8e19c3` | **100% Match** |
| **Phase 1 Multimodal**| `models/multimodal/fusion_model.pt` | PyTorch Binary | `97ff3c69024f2b1c4e7fa090b830e55ee141d06b72d250882e9dff774138e6a1` | **100% Match** |
| **M6 Challenger** | `models/experiments/M6/weights.pt` | PyTorch Binary | `7bc2eec4c274fe0f0556f0e57ba5791c855a88a0e3635905ea6a6c4b2a3a5f32` | **100% Match** |
| **UNet Segmentation** | `models/segmentation/unet_isic2018.pt` | PyTorch Binary | `e837cb0d0e65eb635db15f5c0934e622b794f92bc319520cb95df5050f28e67a` | **100% Match** |

**Zero weights have been retrained or altered.** The improvements are achieved strictly by repairing the feature normalization and inference pipelines to match the data distributions on which these checkpoints were trained.

---

## 4. Categorization Summary

- **Confirmed Bug:** Disconnect between training cache manifold ($x_{11}=0, x_{12}=0$) and unshielded production inputs ($5.98 \times 10^5, 1.0 \times 10^6$).
- **Corrected Bug:** Restored training–inference parity by gating zero-variance dimensions for neural network tensors while preserving physical scale for clinical screening.
- **Expected Model Behavior:** Checkpoint weights produce mathematically stable, reproducible predictions across all splits.
- **Research Limitations:** Future multimodal models require calibrated training sets to learn diameter end-to-end.
- **Unresolved Issues:** None within frozen checkpoint scope.

