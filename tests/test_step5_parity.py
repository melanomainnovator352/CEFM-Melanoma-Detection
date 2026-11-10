"""
Comprehensive validation test suite for CEFM Step 5.
Covers:
  - Canonical 16-D clinical feature schema & dual-alias accessors
  - Robust clinical normalization & zero-variance handling
  - ABCDE composite scoring: nested vs. flat parity, missing components, weight re-normalization
  - Multimodal & M6 inference parity with zero-variance training constraints
  - Softmax and threshold verification
"""

import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F

from app.services.abcde.vector import ABCDEVector, build_abcde_vector
from app.services.abcde.composite import compute_abcde_score, adapt_abcde_features
from app.services.classification import (
    _prepare_abcde_tensor,
    predict_multimodal_fusion,
    predict_m6_challenger,
    _load_multimodal_model,
    _load_m6_challenger_model,
    _challenger_m6_norm_mean,
    _challenger_m6_norm_std,
)


# =====================================================================
# 1. CANONICAL FEATURE SCHEMA & VECTOR VALIDATION
# =====================================================================

def test_canonical_schema_dimensions_and_ordering():
    """Verify exact 16-D ordering and canonical names."""
    vec = ABCDEVector()
    lst = vec.to_list()
    assert len(lst) == 16, f"Vector must be exactly 16 dimensions, got {len(lst)}"
    
    # Check default values
    assert vec.asymmetry_score == 0.0
    assert vec.border_irregularity == 0.0
    assert vec.border_available == 1.0
    assert vec.circularity == 1.0
    assert vec.diameter_norm == 0.0
    assert vec.diameter_mm_norm == 0.0
    assert vec.diameter_available == 0.0
    assert vec.calibration_available == 0.0


def test_dual_alias_property_synchronization():
    """Verify that updating a canonical property updates the legacy alias and vice versa."""
    vec = ABCDEVector()
    
    # Border available <-> circularity
    vec.border_available = 0.0
    assert vec.circularity == 0.0
    vec.circularity = 1.0
    assert vec.border_available == 1.0
    
    # Diameter norm <-> diameter_mm_norm
    vec.diameter_norm = 0.598
    assert vec.diameter_mm_norm == pytest.approx(0.598)
    vec.diameter_mm_norm = 0.400
    assert vec.diameter_norm == pytest.approx(0.400)
    
    # Diameter available <-> calibration_available
    vec.diameter_available = 1.0
    assert vec.calibration_available == 1.0
    vec.calibration_available = 0.0
    assert vec.diameter_available == 0.0


def test_vector_validation_catches_invalid_values():
    """Verify that non-finite values (NaN, Inf) raise ValueError."""
    vec = ABCDEVector(asymmetry_score=float("nan"))
    with pytest.raises(ValueError, match="invalid/non-finite"):
        vec.validate()


# =====================================================================
# 2. ABCDE COMPOSITE SCORING & PARITY
# =====================================================================

def test_abcde_scoring_nested_vs_flat_parity():
    """
    Verify that nested dictionaries (from extract_abcde_features) and flattened
    dictionaries (from analysis.py) produce bit-for-bit identical composite scores.
    """
    flat_input = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2753,
        "color_index": 0.5003,
        "diameter_mm": 11.96,
        "diameter_mm_available": True,
        "evolution_change_detected": False,
        "evolution_available": False,
    }
    nested_input = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2753,
        "color": {"index": 0.5003},
        "diameter": {"diameter_mm": 11.96, "available": True, "calibration_used": True},
        "evolution": {"available": False},
    }

    s_flat = compute_abcde_score(flat_input)
    s_nested = compute_abcde_score(nested_input)

    assert s_flat["score"] == pytest.approx(s_nested["score"], abs=1e-4)
    assert s_flat["label"] == s_nested["label"]
    assert s_flat["score"] == pytest.approx(0.4764, abs=1e-3)
    assert s_flat["label"] == "moderate"


def test_abcde_uncalibrated_weight_renormalization():
    """
    When diameter is uncalibrated, active weights (A, B, C) should re-normalize
    to 1.0 without falsely penalizing the lesion as 0 mm.
    """
    uncalibrated = {
        "asymmetry": 0.80,
        "border_irregularity": 0.80,
        "color_index": 0.80,
        "diameter_mm": None,
        "diameter_mm_available": False,
        "evolution_available": False,
    }
    score = compute_abcde_score(uncalibrated)

    # Active weight sum is 0.24 + 0.22 + 0.18 = 0.64
    assert score["active_weight_sum"] == pytest.approx(0.64)
    assert score["weight_normalized"] is True
    # Average of available components: (0.24*0.8 + 0.22*0.8 + 0.18*0.8) / 0.64 = 0.80
    assert score["score"] == pytest.approx(0.8000, abs=1e-3)
    assert score["label"] == "high"


def test_abcde_missing_all_components_returns_safe_zero():
    """Verify that empty or invalid feature input safely returns score 0.0, label 'low'."""
    assert compute_abcde_score({})["score"] == 0.0
    assert compute_abcde_score(None)["score"] == 0.0


# =====================================================================
# 3. ROBUST NORMALIZATION & ZERO-VARIANCE HANDLING
# =====================================================================

def test_normalization_never_produces_huge_values():
    """
    Verify that even with non-zero physical diameter (e.g. 0.598, 1.0),
    robust normalization does NOT divide by 1e-6 epsilon to produce 598,000.
    """
    device = torch.device("cpu")
    # Load model config tensors
    _load_m6_challenger_model(device=device)
    import app.services.classification as cl
    norm_mean = cl._challenger_m6_norm_mean.to(device)
    norm_std = cl._challenger_m6_norm_std.to(device)

    # Input with non-zero diameter
    c_tensor = torch.tensor([[0.25, 0.0, 0.0, 0.2753, 1.0, 0.0, 0.0, 0.0189, 0.1996, 0.0559, 1.0, 0.598, 1.0, 0.0, 0.0, 0.0]])

    # Robust normalization logic
    valid_variance_mask = (norm_std >= 1e-4)
    safe_std = torch.where(valid_variance_mask, norm_std, torch.ones_like(norm_std))
    c_norm = torch.where(valid_variance_mask, (c_tensor - norm_mean) / safe_std, torch.zeros_like(c_tensor))

    # Assert that all values are bounded strictly within normal z-score ranges
    max_val = c_norm.abs().max().item()
    assert max_val < 10.0, f"Normalized value exploded to {max_val}!"
    assert c_norm[0, 11].item() == 0.0, "Zero-variance diameter must be normalized to 0.0!"
    assert c_norm[0, 12].item() == 0.0, "Zero-variance calibration must be normalized to 0.0!"


def test_multimodal_model_inference_isolates_untrained_dimensions():
    """
    Verify that Phase 1 Multimodal receives a safe model input where
    untrained constant-zero dimensions (11 & 12) are zeroed, while the original
    ABCDE vector retains the physical measurement.
    """
    features = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2753,
        "color_index": 0.5003,
        "diameter_mm": 11.96,
        "diameter_mm_available": True,
    }
    vec = build_abcde_vector(features)
    assert vec.diameter_norm > 0.0, "Clinical vector must retain calibrated diameter!"
    
    # Model tensor must isolate untrained dimensions
    model_t = vec.to_model_tensor(isolate_unsupported=True)
    assert model_t[11].item() == 0.0
    assert model_t[12].item() == 0.0


# =====================================================================
# 4. END-TO-END INFERENCE CONSISTENCY
# =====================================================================

def test_m6_challenger_produces_finite_probabilities_and_valid_predictions():
    """Verify that M6 challenger inference produces valid softmax probabilities."""
    device = torch.device("cpu")
    _load_m6_challenger_model(device=device)

    # Dummy visual embeddings
    g_emb = torch.randn(1, 768)
    l_emb = torch.randn(1, 768)
    features = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2753,
        "color_index": 0.5003,
        "diameter_mm": 11.96,
        "diameter_mm_available": True,
    }

    pred = predict_m6_challenger(
        global_embedding=g_emb,
        local_embedding=l_emb,
        features=features,
        device=device,
    )

    assert pred is not None
    assert "challenger_melanoma_probability" in pred
    assert "challenger_non_melanoma_probability" in pred
    prob_sum = pred["challenger_melanoma_probability"] + pred["challenger_non_melanoma_probability"]
    assert prob_sum == pytest.approx(1.0, abs=1e-3)
    assert pred["challenger_predicted_class"] in ["mel", "non_mel"]


def test_dim4_training_inference_parity_multimodal_and_m6():
    """Verify that Dimension 4 evaluates to 1.0 for Multimodal and 0.0 for M6, exactly matching training caches."""
    features = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2753,
        "color_index": 0.5003,
    }
    
    # 1. Phase 1 Multimodal input tensor
    c_raw = _prepare_abcde_tensor(features, device="cpu")
    mm_tensor = c_raw.clone()
    mm_tensor[:, 11] = 0.0
    mm_tensor[:, 12] = 0.0
    assert mm_tensor[0, 4].item() == 1.0, "Multimodal must receive 1.0 for Dim 4 matching training cache"
    
    # 2. M6 Challenger input tensor
    _load_m6_challenger_model(device="cpu")
    from app.services.classification import _challenger_m6_norm_mean, _challenger_m6_norm_std
    valid_mask = (_challenger_m6_norm_std >= 1e-4)
    safe_std = torch.where(valid_mask, _challenger_m6_norm_std, torch.ones_like(_challenger_m6_norm_std))
    m6_tensor = torch.where(valid_mask, (c_raw - _challenger_m6_norm_mean) / safe_std, torch.zeros_like(c_raw))
    assert m6_tensor[0, 4].item() == 0.0, "M6 must receive 0.0 for Dim 4 matching standardized training cache"


def test_adapt_abcde_features_reentrant_idempotency():
    """Verify that adapt_abcde_features is idempotent and does not drop color when re-adapted."""
    raw = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2753,
        "color": {"index": 0.5014, "color_diversity": 1.0},
    }
    adapted_1 = adapt_abcde_features(raw)
    assert adapted_1["color_available"] is True
    assert adapted_1["color"] == pytest.approx(0.5014, abs=1e-4)
    
    adapted_2 = adapt_abcde_features(adapted_1)
    assert adapted_2["color_available"] is True
    assert adapted_2["color"] == pytest.approx(0.5014, abs=1e-4)
    assert adapted_1 == adapted_2

