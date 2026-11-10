"""
Tests for M6 Dual-Scale Experimental Challenger Integration.
Validates:
1. Checkpoint and config integrity.
2. Architecture and dual-scale feature extraction.
3. Dual validation-calibrated thresholds (0.690 and 0.625).
4. Fallback on missing clinical features.
5. Configuration toggle (ENABLE_CHALLENGER_M6).
6. End-to-end FastAPI endpoint and schema compatibility.
7. Preservation of Phase 1 production checkpoints.
"""

import os
import io
import json
import hashlib
import numpy as np
import cv2
import torch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import CLASSIFICATION_MODEL_DIR
from app.ml.multimodal.dual_scale_models import M6_DualGated
from app.services.classification import (
    _load_m6_challenger_model,
    predict_m6_challenger,
    predict_m6_challenger_from_image,
    classify,
)
from app.services.dual_scale import (
    extract_dual_scale_views,
    extract_dual_scale_embeddings_single,
)
from app.main import app

client = TestClient(app)

PHASE1_HASHES = {
    "vit": "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
    "unet": "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e",
    "fusion": "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
}


def test_m6_checkpoint_and_phase1_integrity():
    """Verify M6 checkpoint exists and all Phase 1 checkpoints remain strictly untouched."""
    m6_dir = CLASSIFICATION_MODEL_DIR.parent / "experiments" / "M6"
    weights_path = m6_dir / "weights.pt"
    config_path = m6_dir / "config.json"

    assert weights_path.exists(), "M6 weights.pt missing"
    assert config_path.exists(), "M6 config.json missing"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Check M6 hash
    actual_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert actual_hash == cfg["weights_sha256"], f"M6 weights hash mismatch: {actual_hash} vs {cfg['weights_sha256']}"

    # Verify Phase 1 checkpoints are untouched
    vit_path = CLASSIFICATION_MODEL_DIR / "vit_model" / "model.safetensors"
    unet_path = CLASSIFICATION_MODEL_DIR.parent / "segmentation" / "unet_isic2018.pt"
    fusion_path = CLASSIFICATION_MODEL_DIR.parent / "multimodal" / "fusion_model.pt"

    assert hashlib.sha256(vit_path.read_bytes()).hexdigest() == PHASE1_HASHES["vit"]
    assert hashlib.sha256(unet_path.read_bytes()).hexdigest() == PHASE1_HASHES["unet"]
    assert hashlib.sha256(fusion_path.read_bytes()).hexdigest() == PHASE1_HASHES["fusion"]


def test_m6_config_thresholds_and_normalization():
    """Verify M6 config contains explicit calibrated thresholds and clinical normalization vectors."""
    m6_dir = CLASSIFICATION_MODEL_DIR.parent / "experiments" / "M6"
    with open(m6_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["calibrated_threshold_f1"] == 0.690
    assert cfg["calibrated_threshold_matched_spec"] == 0.625
    assert len(cfg["clinical_norm_mean"]) == 16
    assert len(cfg["clinical_norm_std"]) == 16
    assert cfg["architecture"] == "GatedDualScaleFusion"


def test_m6_model_loading():
    """Test loading M6 model onto CPU and checking parameter dimensions."""
    model = _load_m6_challenger_model(device="cpu")
    assert model is not None
    assert isinstance(model, M6_DualGated)

    # Check forward pass with dummy tensors
    g_dummy = torch.randn(2, 768)
    l_dummy = torch.randn(2, 768)
    c_dummy = torch.randn(2, 16)
    out = model(g_dummy, l_dummy, c_dummy)
    assert out.shape == (2, 2)


def test_dual_scale_views_extraction():
    """Verify dual-scale spatial extraction pipeline correctly bounds lesion."""
    dummy_img = np.zeros((400, 500, 3), dtype=np.uint8)
    dummy_mask = np.zeros((400, 500), dtype=np.uint8)
    # Draw lesion in center
    cv2.circle(dummy_mask, (250, 200), 50, 1, -1)
    dummy_img[dummy_mask > 0] = [180, 80, 60]

    views = extract_dual_scale_views(dummy_img, dummy_mask)
    assert "global_image" in views
    assert "local_patches" in views
    assert views["global_image"].shape == (224, 224, 3)
    assert len(views["local_patches"]) >= 1
    assert views["local_patches"][0]["image"].shape == (224, 224, 3)


def test_predict_m6_challenger_dual_thresholds():
    """Verify M6 prediction returns explicit dual operating points and margins."""
    g_emb = np.random.randn(768).astype(np.float32)
    l_emb = np.random.randn(768).astype(np.float32)
    features = {
        "asymmetry": 0.35,
        "border_irregularity": 0.42,
        "color_h": 0.15,
        "color_s": 0.30,
        "color_v": 0.50,
        "color_index": 0.85,
        "diameter_mm": 7.5,
    }

    res = predict_m6_challenger(
        global_embedding=g_emb,
        local_embedding=l_emb,
        features=features,
        device="cpu",
    )

    assert res is not None
    assert res["challenger_model_id"] == "M6"
    assert res["challenger_status"] == "experimental_challenger"
    assert res["challenger_calibrated_f1_threshold"] == 0.690
    assert res["challenger_matched_spec_threshold"] == 0.625
    assert res["challenger_predicted_class"] in ("mel", "non_mel")
    assert res["challenger_predicted_class_matched_spec"] in ("mel", "non_mel")
    assert 0.0 <= res["challenger_melanoma_probability"] <= 1.0
    assert 0.0 <= res["challenger_non_melanoma_probability"] <= 1.0


def test_m6_missing_clinical_features_fallback():
    """Verify M6 challenger handles missing/None clinical features gracefully."""
    g_emb = np.random.randn(768).astype(np.float32)
    l_emb = np.random.randn(768).astype(np.float32)

    res = predict_m6_challenger(
        global_embedding=g_emb,
        local_embedding=l_emb,
        features=None,
        device="cpu",
    )

    assert res is not None
    assert "challenger_melanoma_probability" in res
    assert 0.0 <= res["challenger_melanoma_probability"] <= 1.0


def test_m6_configuration_toggle():
    """Verify ENABLE_CHALLENGER_M6 toggle cleanly disables challenger inference."""
    import app.services.classification as cl
    orig_toggle = cl.ENABLE_CHALLENGER_M6
    try:
        cl.ENABLE_CHALLENGER_M6 = False
        res = cl.predict_m6_challenger(
            global_embedding=np.zeros(768),
            local_embedding=np.zeros(768),
            features={},
        )
        assert res is not None
        assert res["challenger_enabled"] is False
        assert res["challenger_status"] == "disabled"
    finally:
        cl.ENABLE_CHALLENGER_M6 = orig_toggle


def test_classify_includes_challenger_and_phase1():
    """Verify classify() produces both Phase 1 and challenger predictions simultaneously."""
    dummy_path = "data/uploads/test_classify_tmp.jpg"
    os.makedirs("data/uploads", exist_ok=True)
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    cv2.circle(img, (150, 150), 40, (50, 40, 120), -1)
    cv2.imwrite(dummy_path, img)

    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, (150, 150), 40, 1, -1)

    features = {
        "asymmetry": 0.20,
        "border_irregularity": 0.30,
        "color_h": 0.1,
        "color_s": 0.2,
        "color_v": 0.6,
        "color_index": 0.5,
        "diameter_mm": 5.0,
    }

    try:
        res = classify(dummy_path, features=features, mask=mask)
        # Check Phase 1 keys
        assert "predicted_class" in res
        assert "melanoma_probability" in res
        assert "multimodal_predicted_class" in res
        assert "multimodal_melanoma_probability" in res

        # Check Challenger keys
        assert "challenger" in res
        assert "challenger_predicted_class" in res
        assert "challenger_melanoma_probability" in res
        assert res["challenger_calibrated_f1_threshold"] == 0.690
        assert res["challenger_matched_spec_threshold"] == 0.625
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)


def test_fastapi_analyze_challenger_schema():
    """Verify FastAPI AnalyzeResponse schema validates challenger_prediction alongside Phase 1."""
    from app.schemas.response import (
        AnalyzeResponse,
        ViTPredictionResult,
        MultimodalPredictionResult,
        ChallengerPredictionResult,
    )
    from app.api.routes import router

    # Verify router registration
    route_paths = {r.path for r in router.routes}
    assert "/analyze" in route_paths

    sample_payload = {
        "status": "success",
        "analysis_id": "test_analysis_m6",
        "case_id": "test_case_m6",
        "image_url": "/api/v1/images/test_analysis_m6",
        "segmentation_url": "/api/v1/images/test_analysis_m6/segmentation",
        "attention_url": "/api/v1/images/test_analysis_m6/attention",
        "filename": "ISIC_0012418.jpg",
        "validation": {"valid": True, "reason": "All checks passed"},
        "image_type": {"image_type": "dermoscopic", "confidence": 0.99, "supported": True},
        "calibration": {"valid": False},
        "abcde": {
            "asymmetry": 0.35,
            "border_irregularity": 0.45,
            "color_h": 0.20,
            "color_s": 0.30,
            "color_v": 0.40,
            "color_index": 0.30,
            "diameter_pixels": 180.0,
            "diameter_mm": 12.0,
        },
        "classification": {
            "predicted_class": "non_mel",
            "melanoma_probability": 0.258,
            "non_melanoma_probability": 0.742,
            "decision_threshold": 0.80,
            "uncertainty_status": "confident",
            "multimodal_predicted_class": "non_mel",
            "multimodal_melanoma_probability": 0.0003,
            "multimodal_non_melanoma_probability": 0.9997,
            "multimodal_decision_threshold": 0.36,
            "challenger_predicted_class": "mel",
            "challenger_melanoma_probability": 0.712,
            "challenger_non_melanoma_probability": 0.288,
            "challenger_decision_threshold": 0.690,
        },
        "vit_prediction": {
            "predicted_class": "non_mel",
            "melanoma_probability": 0.258,
            "non_melanoma_probability": 0.742,
            "decision_threshold": 0.80,
            "uncertainty_status": "confident",
            "model_version": "v2.5",
            "model_name": "ViT-Melanoma",
            "backbone": "google/vit-base-patch16-224",
        },
        "multimodal_prediction": {
            "predicted_class": "non_mel",
            "melanoma_probability": 0.0003,
            "non_melanoma_probability": 0.9997,
            "decision_threshold": 0.36,
            "uncertainty_status": "confident",
            "margin_to_threshold": -0.3597,
            "risk_level": "low",
            "model_version": "CEFM-Fusion-v1.0",
            "model_architecture": "CEFMMultimodalFusion",
            "status": "available",
        },
        "challenger_prediction": {
            "model_id": "M6",
            "model_name": "Dual-Scale Gated Multimodal (RCAG)",
            "model_version": "v2.0-rcag-challenger",
            "model_architecture": "GatedDualScaleFusion",
            "status": "experimental_challenger",
            "enabled": True,
            "predicted_class": "mel",
            "melanoma_probability": 0.712,
            "non_melanoma_probability": 0.288,
            "decision_threshold": 0.690,
            "operating_point": "calibrated_f1",
            "calibrated_f1_threshold": 0.690,
            "matched_spec_threshold": 0.625,
            "predicted_class_matched_spec": "mel",
            "margin_to_threshold": 0.022,
            "uncertainty_status": "confident",
            "risk_level": "high",
            "local_patch_count": 1,
            "patch_coordinates": [{"ymin": 50, "ymax": 274, "xmin": 80, "xmax": 304}],
            "note": "Experimental challenger model. Not used for primary clinical diagnosis.",
        },
    }

    parsed = AnalyzeResponse(**sample_payload)
    assert parsed.status == "success"
    assert parsed.vit_prediction.decision_threshold == 0.80
    assert parsed.multimodal_prediction.decision_threshold == 0.36
    assert parsed.challenger_prediction.calibrated_f1_threshold == 0.690
    assert parsed.challenger_prediction.matched_spec_threshold == 0.625
    assert parsed.challenger_prediction.predicted_class == "mel"
    assert parsed.challenger_prediction.model_id == "M6"
    assert parsed.challenger_prediction.status == "experimental_challenger"
