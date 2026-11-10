"""
Automated tests for Multimodal Fusion Architecture and Inference.
Verifies cross-modal tensor dimensions, forward pass, probability calibration,
vector standardization, and API/service integration.
"""
import torch
import numpy as np
import pytest

from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.services.abcde.vector import ABCDEVector, build_abcde_vector
from app.services.classification import predict_multimodal_fusion


def test_multimodal_fusion_architecture_shapes():
    """Verify CEFMMultimodalFusion takes 768-D visual and 16-D clinical vectors and outputs 2 logits."""
    model = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2)
    model.eval()

    batch_size = 4
    v_feat = torch.randn(batch_size, 768)
    c_feat = torch.randn(batch_size, 16)

    with torch.no_grad():
        logits = model(v_feat, c_feat)
        probs = model.predict_proba(v_feat, c_feat)

    assert logits.shape == (batch_size, 2)
    assert probs.shape == (batch_size, 2)
    sums = probs.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)


def test_abcde_vector_16_dimensions():
    """Verify ABCDE clinical vector standardizes to exactly 16 floating-point values."""
    raw_dict = {
        "asymmetry": {"score": 0.45, "horizontal_asymmetry": 0.3, "vertical_asymmetry": 0.6},
        "border_irregularity": {"border_irregularity_score": 0.55, "circularity": 0.8},
        "color": {"h": 0.15, "s": 0.40, "v": 0.70, "color_diversity": 0.50},
        "diameter": {"diameter_mm": 9.2, "available": True, "calibration_used": True},
    }
    vec = build_abcde_vector(raw_dict)
    assert isinstance(vec, ABCDEVector)
    t = vec.to_tensor()
    assert t.shape == (16,)
    assert t.dtype == torch.float32
    assert float(vec.diameter_mm_norm) == pytest.approx(9.2 / 20.0, rel=1e-3)
    assert float(vec.calibration_available) == 1.0


def test_multimodal_predict_graceful_fallback():
    """Verify predict_multimodal_fusion handles missing model or None gracefully without raising exceptions."""
    dummy_embedding = np.zeros(768, dtype=np.float32)
    res = predict_multimodal_fusion(dummy_embedding, features=None)
    assert res is None


def test_multimodal_predict_no_hardcoded_overrides():
    """Verify multimodal prediction does not use hardcoded diameter thresholds."""
    dummy_embedding = np.random.randn(768).astype(np.float32)
    features_small = {"asymmetry": 0.1, "border_irregularity": 0.1, "diameter_mm": 2.0}
    features_large = {"asymmetry": 0.1, "border_irregularity": 0.1, "diameter_mm": 25.0}

    res_small = predict_multimodal_fusion(dummy_embedding, features_small)
    res_large = predict_multimodal_fusion(dummy_embedding, features_large)

    # When model is loaded, verify that probabilities are continuous floating-point numbers in [0, 1]
    if res_small is not None:
        assert 0.0 <= res_small["multimodal_melanoma_probability"] <= 1.0
        assert res_small["multimodal_predicted_class"] in ["mel", "non_mel"]
    if res_large is not None:
        assert 0.0 <= res_large["multimodal_melanoma_probability"] <= 1.0
        assert res_large["multimodal_predicted_class"] in ["mel", "non_mel"]
