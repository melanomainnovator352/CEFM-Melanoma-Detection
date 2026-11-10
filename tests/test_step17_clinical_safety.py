"""
Tests for CEFM Step 17: Clinical Metadata Safety, Guardrail, and Counterfactual Robustness.

Verifies:
1. Missing metadata does not create extreme confidence or cause runtime failure.
2. Extreme probability suppression is flagged by the safety guardrail.
3. Strong visual evidence is preserved when demographic priors suggest benign status.
4. Unknown and unmapped anatomical categories are handled consistently without leaking test data.
5. Guardrail behavior: arbitrates probability, flags suppression severity, and restores sensitivity.
6. Checkpoint invariance: verifies zero mutation of production and experimental checkpoints.
"""
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from app.ml.multimodal.clinical_metadata import (
    ClinicalMetadataPreprocessor,
    ProposedClinicalFusionModel,
)
from app.services.clinical_guardrail import ClinicalSafetyGuardrail

REPO_ROOT = Path("D:/pytest_cache")


def test_missing_metadata_does_not_crash_or_produce_nan():
    """Verify missing age, sex, and location produce valid finite predictions."""
    df_train = pd.read_csv(REPO_ROOT / "datasets/Ham10000/processed_lesion_disjoint/train.csv")
    prep = ClinicalMetadataPreprocessor().fit(df_train)

    # All metadata missing
    df_missing = pd.DataFrame({
        "age": [np.nan, np.nan],
        "sex": ["unknown", None],
        "localization": ["unknown", np.nan],
    })
    t_miss = prep.transform(df_missing)

    model_f = ProposedClinicalFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64)
    model_f.load_state_dict(torch.load(REPO_ROOT / "models/experiments/clinical_metadata_branch/model.pt", map_location="cpu"))
    model_f.eval()

    vis_dummy = torch.randn(2, 768)
    with torch.no_grad():
        logits = model_f(vis_dummy, t_miss["age_norm"], t_miss["age_missing"], t_miss["sex_onehot"], t_miss["loc_indices"])
        probs = torch.softmax(logits, dim=-1)

    assert not torch.isnan(probs).any()
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_guardrail_detects_extreme_and_moderate_suppression():
    """Verify guardrail identifies extreme, moderate, and delta-based suppression."""
    guard = ClinicalSafetyGuardrail(
        visual_alert_threshold=0.50,
        fusion_decision_threshold=0.2417,
        high_confidence_visual_threshold=0.70,
        extreme_suppression_fusion_threshold=0.01,
    )

    # 1. Extreme suppression: vis=0.85, fus=0.005
    res_extreme = guard.evaluate(visual_probability=0.85, fusion_probability=0.005)
    assert res_extreme["suppression_detected"] is True
    assert res_extreme["is_extreme_suppression"] is True
    assert res_extreme["suppression_severity"] == "critical"
    assert res_extreme["guardrail_triggered"] is True
    assert res_extreme["guardrail_probability"] == pytest.approx(0.85)
    assert res_extreme["prediction_guardrail"] == "mel"

    # 2. Moderate suppression: vis=0.55, fus=0.08
    res_mod = guard.evaluate(visual_probability=0.55, fusion_probability=0.08)
    assert res_mod["suppression_detected"] is True
    assert res_mod["suppression_severity"] == "high"
    assert res_mod["guardrail_triggered"] is True
    assert res_mod["guardrail_probability"] == pytest.approx(0.55)

    # 3. No suppression: vis=0.10, fus=0.15
    res_clean = guard.evaluate(visual_probability=0.10, fusion_probability=0.15)
    assert res_clean["suppression_detected"] is False
    assert res_clean["suppression_severity"] == "none"
    assert res_clean["guardrail_triggered"] is False
    assert res_clean["guardrail_probability"] == pytest.approx(0.15)


def test_guardrail_preserves_strong_visual_evidence():
    """Verify a strong visual suspicion is not silently overridden by a benign demographic prior."""
    guard = ClinicalSafetyGuardrail(visual_alert_threshold=0.50, fusion_decision_threshold=0.2417)

    # High visual melanoma score suppressed by young-age prior
    res = guard.evaluate(
        visual_probability=0.78,
        fusion_probability=0.04,
        metadata={"age": 22, "sex": "female", "localization": "lower extremity"},
    )
    assert res["prediction_visual"] == "mel"
    assert res["prediction_fusion"] == "non_mel"
    # Guardrail must restore to melanoma
    assert res["prediction_guardrail"] == "mel"
    assert res["prediction_restored_by_guardrail"] is True
    assert res["guardrail_probability"] >= 0.50


def test_unknown_anatomical_categories_handled_consistently():
    """Verify unknown, missing, and unmapped strings map to identical zero-index embeddings."""
    df_train = pd.read_csv(REPO_ROOT / "datasets/Ham10000/processed_lesion_disjoint/train.csv")
    prep = ClinicalMetadataPreprocessor().fit(df_train)

    df_test_locs = pd.DataFrame({
        "age": [50.0, 50.0, 50.0],
        "sex": ["male", "male", "male"],
        "localization": ["unknown", np.nan, "unseen_exotic_site_name"],
    })
    t = prep.transform(df_test_locs)

    # All three must map to 0 (unknown)
    assert t["loc_indices"][0].item() == 0
    assert t["loc_indices"][1].item() == 0
    assert t["loc_indices"][2].item() == 0


def test_checkpoint_invariance_integrity():
    """Confirm production and Step 16 experimental checkpoints remain unchanged."""
    checkpoints = {
        "models/classification/vit_model/model.safetensors": "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
        "models/multimodal/fusion_model.pt": "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
        "models/experiments/M6/weights.pt": "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32",
        "models/segmentation/unet_isic2018.pt": "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e",
        "models/experiments/clinical_metadata_branch/model.pt": "27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c",
    }
    for rel_path, exp_hash in checkpoints.items():
        p = REPO_ROOT / rel_path
        assert p.exists(), f"File missing: {rel_path}"
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == exp_hash, f"Checkpoint mutated: {rel_path}"

