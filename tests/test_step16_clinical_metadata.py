"""
Tests for CEFM Step 16: Clinical Metadata Branch and Preprocessing Pipeline.

Verifies:
1. Leak-free preprocessing: fits exclusively on train data; correct handling of missing age and unknown sex/location.
2. Clinical metadata branch forward pass and 64-D output tensor shapes.
3. Fusion architectures (Models A through F) forward pass and compatibility.
4. Checkpoint integrity and non-modification of existing models.
"""
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from app.ml.multimodal.clinical_metadata import (
    ClinicalMetadataPreprocessor,
    ClinicalMetadataBranch,
    ProposedClinicalFusionModel,
    VisualOnlyClassifier,
    ClinicalOnlyClassifier,
    VisualAgeClassifier,
    VisualAgeSexClassifier,
    VisualAgeSexLocClassifier,
    ANATOMICAL_LOCATIONS,
)

REPO_ROOT = Path("D:/pytest_cache")


def test_clinical_preprocessor_fitting_and_transformation():
    """Verify preprocessor computes training stats strictly and transforms unseen inputs."""
    # Dummy training data
    df_train = pd.DataFrame({
        "age": [20.0, 40.0, 60.0, np.nan],
        "sex": ["male", "female", "unknown", "female"],
        "localization": ["back", "face", "trunk", "unknown"],
    })
    prep = ClinicalMetadataPreprocessor().fit(df_train)

    assert prep.is_fitted is True
    assert prep.age_mean == pytest.approx(40.0)
    # std of [20, 40, 60] with ddof=0 is 16.3299
    assert prep.age_std > 0

    # Test transformation on unseen validation data
    df_val = pd.DataFrame({
        "age": [np.nan, 80.0],
        "sex": ["male", "unknown_other"],
        "localization": ["ear", "nonexistent_loc"],
    })
    t = prep.transform(df_val)

    assert t["age_norm"].shape == (2, 1)
    assert t["age_missing"].shape == (2, 1)
    assert t["sex_onehot"].shape == (2, 3)
    assert t["loc_indices"].shape == (2,)

    # Missing age check: normalized value should be 0.0, missing flag 1.0
    assert t["age_missing"][0].item() == 1.0
    assert t["age_norm"][0].item() == 0.0

    # Present age check
    assert t["age_missing"][1].item() == 0.0
    assert t["age_norm"][1].item() > 0.0

    # Unrecognized sex maps to unknown index [0, 0, 1]
    assert torch.equal(t["sex_onehot"][1], torch.tensor([0.0, 0.0, 1.0]))

    # Unrecognized location maps to 0 (unknown)
    assert t["loc_indices"][1].item() == 0


def test_clinical_metadata_branch_forward():
    """Verify 64-D residual clinical branch forward pass."""
    batch_size = 8
    branch = ClinicalMetadataBranch(num_locations=15, location_embed_dim=16, hidden_dim=64)

    age_norm = torch.randn(batch_size, 1)
    age_missing = torch.zeros(batch_size, 1)
    sex_onehot = torch.zeros(batch_size, 3)
    sex_onehot[:, 0] = 1.0
    loc_indices = torch.randint(0, 15, (batch_size,))

    zc = branch(age_norm, age_missing, sex_onehot, loc_indices)
    assert zc.shape == (batch_size, 64)
    assert not torch.isnan(zc).any()


def test_full_proposed_fusion_model_forward():
    """Verify Experiment F full proposed fusion model forward pass."""
    batch_size = 4
    model = ProposedClinicalFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64, num_classes=2)

    vis = torch.randn(batch_size, 768)
    age_norm = torch.randn(batch_size, 1)
    age_missing = torch.zeros(batch_size, 1)
    sex_onehot = torch.zeros(batch_size, 3)
    sex_onehot[:, 1] = 1.0
    loc_indices = torch.randint(0, 15, (batch_size,))

    logits = model(vis, age_norm, age_missing, sex_onehot, loc_indices)
    assert logits.shape == (batch_size, 2)
    assert not torch.isnan(logits).any()


def test_all_ablation_models_forward():
    """Verify Models A through E forward passes."""
    batch_size = 4
    vis = torch.randn(batch_size, 768)
    age_n = torch.randn(batch_size, 1)
    age_m = torch.zeros(batch_size, 1)
    sex = torch.zeros(batch_size, 3)
    sex[:, 0] = 1.0
    loc = torch.randint(0, 15, (batch_size,))

    m_a = VisualOnlyClassifier()
    assert m_a(vis).shape == (batch_size, 2)

    m_b = ClinicalOnlyClassifier()
    assert m_b(age_n, age_m, sex, loc).shape == (batch_size, 2)

    m_c = VisualAgeClassifier()
    assert m_c(vis, age_n, age_m).shape == (batch_size, 2)

    m_d = VisualAgeSexClassifier()
    assert m_d(vis, age_n, age_m, sex).shape == (batch_size, 2)

    m_e = VisualAgeSexLocClassifier()
    assert m_e(vis, age_n, age_m, sex, loc).shape == (batch_size, 2)


def test_production_checkpoints_unmodified():
    """Verify existing production checkpoints were not overwritten or modified."""
    import hashlib

    expected_hashes = {
        "models/classification/vit_model/model.safetensors": "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
        "models/multimodal/fusion_model.pt": "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
        "models/experiments/M6/weights.pt": "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32",
    }
    for path_str, exp_hash in expected_hashes.items():
        p = REPO_ROOT / path_str
        assert p.exists()
        cur_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        assert cur_hash == exp_hash, f"Checkpoint {path_str} was mutated!"

