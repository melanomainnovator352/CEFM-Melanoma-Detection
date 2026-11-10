"""
Tests for CEFM Step 18: Visual Evidence Preservation and Constrained Clinical Metadata Fusion.

Verifies:
1. VisualEvidencePreserver logic, violation detection, and probability arbitration.
2. ConstrainedResidualFusionModel (Alternative C) non-negative logit boost guarantee (Delta l_1 >= 0).
3. GatedPositiveFusionModel (Alternative D) non-negative logit boost guarantee (Delta l_1 >= 0).
4. Robustness under missing, invalid, or extreme metadata inputs (no NaNs, finite bounds).
5. Exact checkpoint invariance across production baselines and experimental models.
6. Completeness of generated Step 18 evaluation reports and manifests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.multimodal.clinical_metadata import ClinicalMetadataPreprocessor
from app.ml.multimodal.constrained_fusion import (
    ConstrainedResidualFusionModel,
    GatedPositiveFusionModel,
    VisualEvidencePreserver,
)

REPO_ROOT = Path("D:/pytest_cache")

EXPECTED_HASHES = {
    "vit": (
        REPO_ROOT / "models/classification/vit_model/model.safetensors",
        "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
    ),
    "multimodal": (
        REPO_ROOT / "models/multimodal/fusion_model.pt",
        "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
    ),
    "m6": (
        REPO_ROOT / "models/experiments/M6/weights.pt",
        "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32",
    ),
    "unet": (
        REPO_ROOT / "models/segmentation/unet_isic2018.pt",
        "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e",
    ),
    "model_f": (
        REPO_ROOT / "models/experiments/clinical_metadata_branch/model.pt",
        "27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c",
    ),
    "model_c": (
        REPO_ROOT / "models/experiments/step18_constrained_fusion/model_C_constrained_residual.pt",
        "eeb04571506a50bf6296547a1c7080d34c9ab625e8f7950f7f91527c65a59d67",
    ),
    "model_d": (
        REPO_ROOT / "models/experiments/step18_constrained_fusion/model_D_gated_positive.pt",
        "98e1c74525569ff40834a23d2d7b5849efad7bd844a13964728c738a8fc8d179",
    ),
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_visual_evidence_preserver_violation_and_arbitration():
    """Verify VisualEvidencePreserver accurately flags suppression and arbitrates probabilities."""
    vep = VisualEvidencePreserver(visual_alert_threshold=0.50, decision_threshold=0.2417)

    # 1. Suspicious visual evidence suppressed by fusion below threshold
    res1 = vep.evaluate(visual_probability=0.72, fusion_probability=0.10)
    assert res1["is_violation"] is True
    assert res1["action"] == "visual_override_applied"
    assert res1["status"] == "VIOLATION_ARBITRATED"
    assert res1["p_guarded"] == pytest.approx(0.72)
    assert res1["suppression_delta"] == pytest.approx(0.62)

    # 2. Suspicious visual evidence concordant with fusion above threshold
    res2 = vep.evaluate(visual_probability=0.80, fusion_probability=0.65)
    assert res2["is_violation"] is False
    assert res2["action"] == "none"
    assert res2["status"] == "PRESERVED"
    assert res2["p_guarded"] == pytest.approx(0.65)

    # 3. Non-suspicious visual evidence (p_vis < 0.50)
    res3 = vep.evaluate(visual_probability=0.20, fusion_probability=0.05)
    assert res3["is_violation"] is False
    assert res3["action"] == "none"
    assert res3["status"] == "PRESERVED"
    assert res3["p_guarded"] == pytest.approx(0.05)


def test_constrained_residual_fusion_strictly_non_negative_boost():
    """Verify ConstrainedResidualFusionModel strictly enforces Delta l_1 >= 0."""
    torch.manual_seed(42)
    model = ConstrainedResidualFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64)
    model.eval()

    batch_size = 32
    vis_feats = torch.randn(batch_size, 768)
    age_norm = torch.randn(batch_size, 1)
    age_missing = torch.zeros(batch_size, 1)
    sex_onehot = torch.zeros(batch_size, 3)
    sex_onehot[:, 0] = 1.0  # e.g., female
    loc_indices = torch.randint(0, 15, (batch_size,))

    with torch.no_grad():
        components = model.get_components(vis_feats, age_norm, age_missing, sex_onehot, loc_indices)
        logits = model(vis_feats, age_norm, age_missing, sex_onehot, loc_indices)

    # delta_l1 must be strictly non-negative due to Softplus
    delta_l1 = components["delta_l1"]
    assert torch.all(delta_l1 >= 0.0), "delta_l1 contains negative values!"

    # Probability after fusion must be >= visual probability
    p_vis = components["p_visual"]
    p_fused = components["p_fused"]
    assert torch.all(p_fused >= p_vis - 1e-6), "Fusion probability dropped below visual probability!"

    # Forward logits check: l1 must be >= l1_base
    zv = model.visual_encoder(vis_feats)
    base_l1 = model.visual_head(zv)[:, 1]
    fused_l1 = logits[:, 1]
    assert torch.all(fused_l1 >= base_l1 - 1e-6)


def test_gated_positive_fusion_strictly_non_negative_boost():
    """Verify GatedPositiveFusionModel enforces Delta l_1 >= 0 through gate * Softplus."""
    torch.manual_seed(42)
    model = GatedPositiveFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64)
    model.eval()

    batch_size = 16
    vis_feats = torch.randn(batch_size, 768)
    age_norm = torch.randn(batch_size, 1)
    age_missing = torch.zeros(batch_size, 1)
    sex_onehot = torch.zeros(batch_size, 3)
    sex_onehot[:, 1] = 1.0  # male
    loc_indices = torch.randint(0, 15, (batch_size,))

    with torch.no_grad():
        zv = model.visual_encoder(vis_feats)
        vis_logits = model.visual_head(zv)
        gate = model.visual_gate(zv)
        zc = model.clinical_branch(age_norm, age_missing, sex_onehot, loc_indices)
        raw_boost = model.clinical_boost_head(zc)
        delta_l1 = gate * raw_boost

        logits = model(vis_feats, age_norm, age_missing, sex_onehot, loc_indices)

    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    assert torch.all(raw_boost >= 0.0)
    assert torch.all(delta_l1 >= 0.0)
    assert torch.all(logits[:, 1] >= vis_logits[:, 1] - 1e-6)


def test_missing_metadata_robustness_step18_models():
    """Verify models execute without NaN or crash when all metadata is missing or extreme."""
    df_train = pd.read_csv(REPO_ROOT / "datasets/Ham10000/processed_lesion_disjoint/train.csv")
    prep = ClinicalMetadataPreprocessor().fit(df_train)

    df_test_missing = pd.DataFrame({
        "age": [np.nan, 120.0, -10.0],
        "sex": ["unknown", None, "invalid_sex"],
        "localization": ["unknown", np.nan, "unseen_location"],
    })
    t_miss = prep.transform(df_test_missing)

    vis_dummy = torch.randn(3, 768)

    model_c = ConstrainedResidualFusionModel()
    model_c.load_state_dict(torch.load(EXPECTED_HASHES["model_c"][0], map_location="cpu"))
    model_c.eval()

    model_d = GatedPositiveFusionModel()
    model_d.load_state_dict(torch.load(EXPECTED_HASHES["model_d"][0], map_location="cpu"))
    model_d.eval()

    with torch.no_grad():
        logits_c = model_c(vis_dummy, t_miss["age_norm"], t_miss["age_missing"], t_miss["sex_onehot"], t_miss["loc_indices"])
        logits_d = model_d(vis_dummy, t_miss["age_norm"], t_miss["age_missing"], t_miss["sex_onehot"], t_miss["loc_indices"])

        probs_c = torch.softmax(logits_c, dim=-1)
        probs_d = torch.softmax(logits_d, dim=-1)

    assert not torch.isnan(probs_c).any()
    assert not torch.isnan(probs_d).any()
    assert torch.all(probs_c >= 0.0) and torch.all(probs_c <= 1.0)
    assert torch.all(probs_d >= 0.0) and torch.all(probs_d <= 1.0)


def test_checkpoint_invariance_step18():
    """Verify production baselines and Step 18 checkpoints match authoritative SHA-256 hashes."""
    for name, (path, expected_hash) in EXPECTED_HASHES.items():
        assert path.exists(), f"Checkpoint {name} not found at {path}"
        computed_hash = _sha256(path)
        assert (
            computed_hash == expected_hash
        ), f"Checkpoint {name} hash mismatch! Expected {expected_hash}, got {computed_hash}"


def test_step18_report_deliverables_exist():
    """Verify all Step 18 CSV deliverables and JSON manifest exist and contain data."""
    expected_files = [
        "CEFM_STEP18_ARCHITECTURE_COMPARISON.csv",
        "CEFM_STEP18_MISSING_METADATA_AUDIT.csv",
        "CEFM_STEP18_COUNTERFACTUAL_AUDIT.csv",
        "CEFM_STEP18_VISUAL_PRESERVATION.csv",
        "CEFM_STEP18_PAIRED_BOOTSTRAP.csv",
        "CEFM_STEP18_ARTIFACT_MANIFEST.json",
    ]
    for fname in expected_files:
        fpath = REPO_ROOT / "reports" / fname
        assert fpath.exists(), f"Missing Step 18 deliverable: {fname}"
        assert fpath.stat().st_size > 0, f"File {fname} is empty"
