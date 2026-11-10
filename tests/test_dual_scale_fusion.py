"""
Unit tests for Dual-Scale Multimodal Fusion Architectures.
Tests ConcatDualScaleFusion, GatedDualScaleFusion, and CrossAttentionDualScaleFusion.
"""
import pytest
import torch
from app.ml.multimodal.dual_scale_fusion import (
    ConcatDualScaleFusion,
    GatedDualScaleFusion,
    CrossAttentionDualScaleFusion,
)


@pytest.fixture
def dummy_inputs():
    batch_size = 4
    global_feat = torch.randn(batch_size, 768)
    local_feat = torch.randn(batch_size, 768)
    clinical_feat = torch.randn(batch_size, 16)
    return global_feat, local_feat, clinical_feat


def test_concat_dual_scale_fusion_forward(dummy_inputs):
    global_feat, local_feat, clinical_feat = dummy_inputs
    model = ConcatDualScaleFusion()
    logits = model(global_feat, local_feat, clinical_feat)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()

    probs = model.predict_proba(global_feat, local_feat, clinical_feat)
    assert probs.shape == (4, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_gated_dual_scale_fusion_forward(dummy_inputs):
    global_feat, local_feat, clinical_feat = dummy_inputs
    model = GatedDualScaleFusion()
    logits = model(global_feat, local_feat, clinical_feat)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()

    probs = model.predict_proba(global_feat, local_feat, clinical_feat)
    assert probs.shape == (4, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_cross_attention_dual_scale_fusion_forward(dummy_inputs):
    global_feat, local_feat, clinical_feat = dummy_inputs
    model = CrossAttentionDualScaleFusion()
    logits = model(global_feat, local_feat, clinical_feat)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()

    probs = model.predict_proba(global_feat, local_feat, clinical_feat)
    assert probs.shape == (4, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_fusion_backprop_gradients(dummy_inputs):
    global_feat, local_feat, clinical_feat = dummy_inputs
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    for model_cls in [ConcatDualScaleFusion, GatedDualScaleFusion, CrossAttentionDualScaleFusion]:
        model = model_cls()
        model.train()
        logits = model(global_feat, local_feat, clinical_feat)
        loss = criterion(logits, targets)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name} in {model_cls.__name__}"

