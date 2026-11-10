"""
Unit tests for all M1–M7 dual-scale models.
"""
import pytest
import torch
from app.ml.multimodal.dual_scale_models import MODEL_REGISTRY


@pytest.fixture
def dummy_inputs():
    batch_size = 4
    g = torch.randn(batch_size, 768)
    l = torch.randn(batch_size, 768)
    c = torch.randn(batch_size, 16)
    return g, l, c


def test_all_models_instantiation_and_forward(dummy_inputs):
    g, l, c = dummy_inputs
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    for m_id, info in MODEL_REGISTRY.items():
        model_cls = info["class"]
        model = model_cls()
        model.train()

        logits = model(g, l, c)
        assert logits.shape == (4, 2), f"Failed shape for {m_id}"
        assert not torch.isnan(logits).any(), f"NaN in logits for {m_id}"

        probs = model.predict_proba(g, l, c)
        assert probs.shape == (4, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

        loss = criterion(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name} in {m_id}"

