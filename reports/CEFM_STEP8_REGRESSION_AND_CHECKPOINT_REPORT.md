# CEFM STEP 8 — Regression Test Suite & Checkpoint Integrity Audit Report

**Audit Date:** 2026-11-07  
**Test Harness:** `pytest-9.1.1` (Python 3.10.0, Pluggy 1.6.0, AnyIO 4.13.0)  
**Execution Scope:** Complete System Automated Test Suite (`tests/`)  
**Status:** 100% PASSING — 90/90 TESTS PASSED, ZERO FAILURES, ZERO SKIPPED

---

## 1. Executive Summary

As the final validation gate of Step 8, the complete CEFM automated test suite was executed in Standard Sandbox Mode. Following the addition of dedicated Dimension-4 training parity tests and re-entrant adapter validation tests, the test suite expanded from 88 to **90 automated tests**.

All **90 tests passed with 100% success** in **24.53 seconds**. Exactly zero tests failed, and exactly zero tests were skipped.

Simultaneously, live byte-level SHA-256 checksums were re-verified across all four authoritative production and challenger model checkpoints, confirming **100% bit-for-bit identity** with the Phase 2 Reproducibility Manifest.

---

## 2. Comprehensive Test Suite Execution Breakdown

| Test Module File | Test Count | Passed | Failed | Skipped | Core Capabilities Verified |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `tests/test_abcde.py` | 4 | 4 | 0 | 0 | Uncalibrated diameter handling, evolution requirement, asymmetry/color scoring, composite risk ascent. |
| `tests/test_api_endpoints.py` | 5 | 5 | 0 | 0 | API health check, endpoint routing registry, 404 handling on invalid IDs. |
| `tests/test_calibration_ruler.py` | 1 | 1 | 0 | 0 | Metric ruler edge detection and pixel-to-millimeter ratio estimation. |
| `tests/test_classification.py` | 7 | 7 | 0 | 0 | ViT backbone validation, melanoma class index mapping, borderline lesion classification, no diameter override, live dermoscopy classification. |
| `tests/test_dual_scale.py` | 6 | 6 | 0 | 0 | Empty mask fallback, malformed masks, boundary lesions, multi-patch tiling, dual-scale embedding dimensions. |
| `tests/test_dual_scale_fusion.py` | 4 | 4 | 0 | 0 | Concat fusion, gated fusion, cross-attention fusion forward passes and backprop gradient flow. |
| `tests/test_dual_scale_models.py` | 1 | 1 | 0 | 0 | Instantiation and forward passes of all dual-scale model variants. |
| `tests/test_evolution_demo.py` | 3 | 3 | 0 | 0 | Baseline without prior image, stable lesion, meaningful longitudinal change detection. |
| `tests/test_explainability.py` | 5 | 5 | 0 | 0 | ViT attention map generation, mask alignment, PNG byte encoding, view registry, text explanation synthesis. |
| `tests/test_image_quality_gate.py` | 4 | 4 | 0 | 0 | Marker calibration, missing marker rejection, boundary segmentation rejection. |
| `tests/test_m6_integration.py` | 9 | 9 | 0 | 0 | M6 checkpoint integrity, dual threshold loading, model loading, feature extraction, challenger schema. |
| `tests/test_model_adapters.py` | 3 | 3 | 0 | 0 | Vision model adapter, segmentation adapter fallback, CLIP/LLM adapter configurability. |
| `tests/test_multimodal_architecture.py` | 2 | 2 | 0 | 0 | Multimodal stage exposure, real image execution pipeline. |
| `tests/test_multimodal_fusion.py` | 4 | 4 | 0 | 0 | Architecture tensor shapes, 16-D clinical vector, graceful fallback, no hardcoded overrides. |
| `tests/test_multimodal_integration.py` | 10 | 10 | 0 | 0 | Checkpoint loading, feature dimensions, normalization stability, CPU/GPU parity, UI backend consistency. |
| `tests/test_multimodal_pipeline.py` | 2 | 2 | 0 | 0 | Contrastive aligner similarity, explainability summary ABCDE mention. |
| `tests/test_project_accuracy.py` | 2 | 2 | 0 | 0 | Accuracy calculation with known labels, zero accuracy handling for all-wrong predictions. |
| `tests/test_step5_parity.py` | **11** | **11** | 0 | 0 | Schema ordering, dual aliases, vector bounds, ABCDE caller parity, zero-variance protection, **Dimension 4 Multimodal/M6 parity**, **adapter re-entrant idempotency**. |
| `tests/test_ui_and_prediction_regression.py` | 5 | 5 | 0 | 0 | Class mapping integrity, threshold logic, HTML rendering, discordance alert generation. |
| `tests/test_vit_training_config.py` | 2 | 2 | 0 | 0 | Training configuration accuracy target, backbone freeze option. |
| **TOTAL** | **90** | **90** | **0** | **0** | **100% Suite Passing (Execution Time: 24.53s, 0 Skipped)** |

---

## 3. Checkpoint SHA-256 Checksum Verification

Byte-level SHA-256 hashes recalculated live from physical disk:

| Model Subsystem | Active Disk File Path | Actual On-Disk SHA-256 | Phase 2 Manifest SHA-256 | Bit-for-Bit Match |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 1 ViT Baseline** | `models/classification/vit_model/model.safetensors` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | `1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db` | **100% MATCH** |
| **Phase 1 Multimodal** | `models/multimodal/fusion_model.pt` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | `04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422` | **100% MATCH** |
| **M6 Challenger** | `models/experiments/M6/weights.pt` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | `573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32` | **100% MATCH** |
| **VM-UNet Segmentation**| `models/segmentation/unet_isic2018.pt` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | `7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e` | **100% MATCH** |

---

## 4. Certification Verdict

The CEFM software repository passes all regression criteria. Model weights, thresholds, schemas, and API endpoints are 100% verified, stable, and ready for Phase 3 domain-adaptation experimentation.

