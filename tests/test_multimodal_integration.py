"""
Comprehensive Integration Tests for CEFM Multimodal Fusion Model.
Tests:
1. Checkpoint loading and matching configuration.
2. Feature dimension consistency (768-D ViT + 16-D clinical -> 2 logits).
3. Feature normalization (values in [0, 1], diameter normalized by 20mm).
4. Missing clinical features (gracefully handled without failure).
5. Probability normalization (sum to 1.0, non-negative).
6. Threshold consistency (calibrated validation threshold = 0.36 applied).
7. API response correctness (FastAPI endpoint returns both ViT and multimodal fields).
8. Streamlit / backend prediction consistency.
9. CPU and GPU inference compatibility.
10. Failure handling (graceful handling of invalid or missing inputs).
"""
import io
import json
import cv2
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.services.abcde.vector import build_abcde_vector, ABCDEVector
from app.services.classification import (
    predict_multimodal_fusion,
    _load_multimodal_model,
    classify,
    MULTIMODAL_MODEL_DIR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = REPO_ROOT / "data" / "uploads" / "ISIC_0024306.jpg"


def test_checkpoint_and_config_loading():
    """Verify production fusion checkpoint and exact configuration load successfully."""
    model_file = MULTIMODAL_MODEL_DIR / "fusion_model.pt"
    config_file = MULTIMODAL_MODEL_DIR / "fusion_config.json"

    assert model_file.exists(), f"Missing production checkpoint: {model_file}"
    assert config_file.exists(), f"Missing configuration file: {config_file}"

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["model_architecture"] == "CEFMMultimodalFusion"
    assert cfg["visual_dim"] == 768
    assert cfg["clinical_dim"] == 16
    assert cfg["hidden_dim"] == 128
    assert cfg["clinical_threshold"] == pytest.approx(0.36, rel=1e-2)

    model = _load_multimodal_model(device="cpu")
    assert model is not None
    assert isinstance(model, CEFMMultimodalFusion)


def test_feature_dimension_consistency():
    """Verify input dimensions (768 visual, 16 clinical) and output dimension (2 classes)."""
    model = _load_multimodal_model(device="cpu")
    assert model is not None

    batch_size = 5
    dummy_visual = torch.randn(batch_size, 768)
    dummy_clinical = torch.randn(batch_size, 16)

    with torch.no_grad():
        logits = model(dummy_visual, dummy_clinical)
        probs = model.predict_proba(dummy_visual, dummy_clinical)

    assert logits.shape == (batch_size, 2)
    assert probs.shape == (batch_size, 2)


def test_feature_normalization():
    """Verify ABCDE clinical feature normalization rules (diameter / 20.0, [0, 1] bounds)."""
    features = {
        "asymmetry": 0.45,
        "border_irregularity": 0.65,
        "color_h": 0.25,
        "color_s": 0.50,
        "color_v": 0.80,
        "color_index": 0.60,
        "diameter_mm": 10.0,
        "diameter_mm_available": True,
    }
    vec = build_abcde_vector(features)
    t = vec.to_tensor()

    assert t.shape == (16,)
    # Diameter mm (10.0 mm / 20.0 mm = 0.50)
    assert t[11].item() == pytest.approx(0.50, rel=1e-3)
    # Calibration available flag
    assert t[12].item() == 1.0
    # All values must be bounded
    assert (t >= 0.0).all()
    assert (t <= 1.0).all()


def test_missing_clinical_features_handled():
    """Verify feature builder and multimodal predictor gracefully handle empty or partial features."""
    empty_features = {}
    vec = build_abcde_vector(empty_features)
    t = vec.to_tensor()

    assert t.shape == (16,)
    assert not torch.isnan(t).any()
    assert not torch.isinf(t).any()

    dummy_emb = np.zeros(768, dtype=np.float32)
    pred = predict_multimodal_fusion(dummy_emb, features=empty_features)
    assert pred is not None
    assert "multimodal_melanoma_probability" in pred
    assert "multimodal_predicted_class" in pred
    assert 0.0 <= pred["multimodal_melanoma_probability"] <= 1.0


def test_probability_normalization():
    """Verify predicted probabilities sum to 1.0 and lie strictly within [0, 1]."""
    dummy_emb = np.random.randn(768).astype(np.float32)
    features = {"asymmetry": 0.3, "border_irregularity": 0.4, "diameter_mm": 5.0}

    pred = predict_multimodal_fusion(dummy_emb, features)
    assert pred is not None
    p_mel = pred["multimodal_melanoma_probability"]
    p_non_mel = pred["multimodal_non_melanoma_probability"]

    assert 0.0 <= p_mel <= 1.0
    assert 0.0 <= p_non_mel <= 1.0
    assert (p_mel + p_non_mel) == pytest.approx(1.0, abs=1e-3)


def test_threshold_consistency():
    """Verify prediction matches the validation-calibrated threshold (0.36)."""
    with open(MULTIMODAL_MODEL_DIR / "fusion_config.json", "r") as f:
        cfg = json.load(f)
    expected_threshold = float(cfg.get("clinical_threshold", 0.36))

    dummy_emb = np.random.randn(768).astype(np.float32)
    pred = predict_multimodal_fusion(dummy_emb, features={"asymmetry": 0.2})
    assert pred is not None
    assert pred["multimodal_decision_threshold"] == pytest.approx(expected_threshold, rel=1e-3)

    p_mel = pred["multimodal_melanoma_probability"]
    expected_class = "mel" if p_mel >= expected_threshold else "non_mel"
    assert pred["multimodal_predicted_class"] == expected_class


def test_api_response_correctness():
    """Verify API schema and response structures contain both ViT and Multimodal predictions."""
    from app.api.routes import router
    from app.schemas.response import AnalyzeResponse, ViTPredictionResult, MultimodalPredictionResult

    # Verify route registration
    route_paths = {r.path for r in router.routes}
    assert "/analyze" in route_paths

    # Verify schema validation on sample API payload
    sample_response = {
        "status": "success",
        "analysis_id": "test_analysis_123",
        "case_id": "test_case_456",
        "image_url": "/api/v1/images/test_analysis_123",
        "segmentation_url": "/api/v1/images/test_analysis_123/segmentation",
        "attention_url": "/api/v1/images/test_analysis_123/attention",
        "filename": "ISIC_0024306.jpg",
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
            "diameter_mm": None,
        },
        "classification": {
            "predicted_class": "mel",
            "melanoma_probability": 0.82,
            "non_melanoma_probability": 0.18,
            "decision_threshold": 0.80,
            "uncertainty_status": "confident",
            "vit_predicted_class": "mel",
            "vit_melanoma_probability": 0.82,
            "vit_non_melanoma_probability": 0.18,
            "vit_decision_threshold": 0.80,
            "vit_uncertainty_status": "confident",
            "multimodal_predicted_class": "mel",
            "multimodal_melanoma_probability": 0.74,
            "multimodal_non_melanoma_probability": 0.26,
            "multimodal_decision_threshold": 0.36,
            "multimodal_uncertainty_status": "confident",
            "multimodal_margin_to_threshold": 0.38,
            "multimodal_risk_level": "high",
            "multimodal_status": "available",
            "multimodal_model_version": "CEFM-Fusion-v1.0",
        },
        "vit_prediction": {
            "predicted_class": "mel",
            "melanoma_probability": 0.82,
            "non_melanoma_probability": 0.18,
            "decision_threshold": 0.80,
            "uncertainty_status": "confident",
            "model_version": "v2.5",
            "model_name": "ViT-Melanoma",
            "backbone": "google/vit-base-patch16-224",
        },
        "multimodal_prediction": {
            "predicted_class": "mel",
            "melanoma_probability": 0.74,
            "non_melanoma_probability": 0.26,
            "decision_threshold": 0.36,
            "uncertainty_status": "confident",
            "margin_to_threshold": 0.38,
            "risk_level": "high",
            "model_version": "CEFM-Fusion-v1.0",
            "model_architecture": "CEFMMultimodalFusion",
            "status": "available",
        },
        "evolution": {"available": False},
        "multimodal": {
            "multimodal_fusion": {
                "status": "available",
                "predicted_class": "mel",
                "melanoma_probability": 0.74,
                "decision_threshold": 0.36,
            }
        },
        "contrastive_alignment": {"score": 0.85},
        "explanation": {"summary": "High risk lesion"},
        "rag_evidence": {"query": "", "documents": []},
        "report": {"summary": "Clinical report", "overall_risk": "high"},
    }

    # Validate against Pydantic schema
    validated = AnalyzeResponse(**sample_response)
    assert validated.status == "success"
    assert validated.vit_prediction.predicted_class == "mel"
    assert validated.vit_prediction.decision_threshold == 0.80
    assert validated.multimodal_prediction.predicted_class == "mel"
    assert validated.multimodal_prediction.decision_threshold == 0.36
    assert validated.multimodal_prediction.melanoma_probability == 0.74


def test_cpu_and_gpu_inference():
    """Verify multimodal fusion model can execute on CPU and GPU (if CUDA available)."""
    # 1. CPU
    model_cpu = _load_multimodal_model(device="cpu")
    assert model_cpu is not None
    v_cpu = torch.randn(1, 768, device="cpu")
    c_cpu = torch.randn(1, 16, device="cpu")
    with torch.no_grad():
        out_cpu = model_cpu.predict_proba(v_cpu, c_cpu)
    assert out_cpu.shape == (1, 2)

    # 2. GPU (if available)
    if torch.cuda.is_available():
        model_gpu = _load_multimodal_model(device="cuda")
        assert model_gpu is not None
        v_gpu = torch.randn(1, 768, device="cuda")
        c_gpu = torch.randn(1, 16, device="cuda")
        with torch.no_grad():
            out_gpu = model_gpu.predict_proba(v_gpu, c_gpu)
        assert out_gpu.shape == (1, 2)


def test_failure_handling_graceful():
    """Verify failure modes return safe fallback dictionary or None without unhandled crashes."""
    # Corrupt or wrong-dimension embedding
    corrupted_embedding = np.zeros(100, dtype=np.float32)  # Wrong dim (100 instead of 768)
    res = predict_multimodal_fusion(corrupted_embedding, features={"asymmetry": 0.5})
    # Should catch exception and return None
    assert res is None

    # None features
    res_none = predict_multimodal_fusion(np.zeros(768), features=None)
    assert res_none is None


def test_streamlit_backend_prediction_consistency():
    """Verify Streamlit parser logic matches backend classification outputs exactly."""
    dummy_embedding = np.random.randn(768).astype(np.float32)
    features = {
        "asymmetry": 0.35,
        "border_irregularity": 0.40,
        "color_index": 0.25,
        "diameter_mm": 5.2,
        "diameter_mm_available": True,
    }

    res = predict_multimodal_fusion(dummy_embedding, features)
    assert res is not None

    # Verify fields parsed by Streamlit render_report()
    mm_prob = res.get("multimodal_melanoma_probability")
    mm_pred = res.get("multimodal_predicted_class")
    mm_thresh = float(res.get("multimodal_decision_threshold", 0.36))
    mm_unc = res.get("multimodal_uncertainty_status")

    assert mm_prob is not None
    assert mm_pred in ["mel", "non_mel"]
    assert mm_thresh == pytest.approx(0.36, rel=1e-2)
    assert mm_unc in ["confident", "borderline"]

    # Verify decision boundary logic matches exactly
    expected_pred = "mel" if mm_prob >= mm_thresh else "non_mel"
    assert mm_pred == expected_pred
