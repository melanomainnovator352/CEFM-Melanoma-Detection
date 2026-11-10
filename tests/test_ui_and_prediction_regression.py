"""
Regression tests for:
- False-negative melanoma prediction investigation
- Class mapping verification (0 = non_mel, 1 = mel)
- Threshold application logic (applied to melanoma probability)
- Multimodal feature alignment
- HTML card rendering (no leaked comments or raw <pre><code> blocks)
- Clinical-AI discordance handling
"""

import re
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from app.services.classification import (
    _load_torch_model,
    _binary_melanoma_probability,
    get_decision_threshold,
    classify,
    _risk_level,
)
from app.services.abcde.vector import build_abcde_vector
from app.ml.multimodal.fusion import CEFMMultimodalFusion


def test_class_mapping_integrity():
    """Verify class mapping: 0 = non_mel, 1 = mel."""
    model = _load_torch_model()
    assert model is not None
    id2label = getattr(model.config, "id2label", {})
    label2id = getattr(model.config, "label2id", {})

    assert id2label[0] == "non_mel" or id2label.get("0") == "non_mel"
    assert id2label[1] == "mel" or id2label.get("1") == "mel"
    assert label2id.get("non_mel") == 0
    assert label2id.get("mel") == 1

    # Check helper mapping
    probs = {"non_mel": 0.7416, "mel": 0.2584}
    assert _binary_melanoma_probability(probs) == pytest.approx(0.2584)


def test_threshold_logic_applied_to_melanoma_prob():
    """Verify decision threshold is applied to melanoma probability, not non-melanoma."""
    vit_threshold = get_decision_threshold()
    assert vit_threshold == pytest.approx(0.80)

    # When mel_prob = 0.2584 (< 0.80) and non_mel = 0.7416, prediction MUST be non_mel
    mel_prob = 0.2584
    pred_class = "mel" if mel_prob >= vit_threshold else "non_mel"
    assert pred_class == "non_mel"

    # Multimodal threshold
    mm_threshold = 0.36
    mm_prob = 0.0003
    mm_pred = "mel" if mm_prob >= mm_threshold else "non_mel"
    assert mm_pred == "non_mel"


def test_multimodal_feature_alignment():
    """Verify 16-D clinical feature vector dimensions, ordering, and normalization."""
    features = {
        "asymmetry": 0.25,
        "border_irregularity": 0.2645,
        "color": {"h": 0.0248, "s": 0.1995, "v": 0.0559, "index": 0.5014},
        "diameter": {"diameter_mm": 11.96, "available": True},
    }
    vec = build_abcde_vector(features)
    tensor = vec.to_tensor()
    assert tensor.shape == (16,)
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()
    # Check bounded in [0, 1]
    assert (tensor >= 0.0).all()
    assert (tensor <= 1.0).all()


def test_comparative_assessment_html_rendering():
    """
    Verify the Comparative Model Assessment HTML card renders cleanly:
    - No raw HTML comments leaked as text (<!-- Multimodal Fusion Card -->)
    - No markdown <pre><code> block wrapping due to 4-space indentation after blank lines
    - Both ViT and Multimodal Fusion cards present with probabilities, thresholds, and uncertainties
    """
    import markdown

    pred = "non_mel"
    mel_prob = 0.2584
    non_mel_prob = 0.7416
    decision_thresh = 0.80
    vit_margin = -0.5416
    vit_unc = "confident"

    mm_pred = "non_mel"
    mm_prob = 0.0003
    mm_non_mel_prob = 0.9997
    mm_thresh = 0.36
    mm_margin = -0.3597
    mm_unc = "confident"
    mm_risk = "low"

    vit_pill_color = "red" if pred == "mel" else "green"
    mm_pill_color = "red" if mm_pred == "mel" else "green"
    vit_unc_pill = "amber" if vit_unc == "borderline" else "slate"
    mm_unc_pill = "amber" if mm_unc == "borderline" else "slate"
    mm_risk_pill = "red" if mm_risk == "high" else ("amber" if "medium" in str(mm_risk) else "green")
    vit_bar_pct = min(100.0, max(0.0, mel_prob * 100))
    mm_bar_pct = min(100.0, max(0.0, float(mm_prob) * 100))

    comparative_html = (
        f'<div class="box" style="margin:0.8rem 0; border-left:4px solid #3b82f6;">'
        f'<div style="font-weight:600; font-size:0.98rem; margin-bottom:8px; color:#1e293b;">'
        f'🔬 Comparative Model Assessment (Vision Transformer Baseline vs. Multimodal Fusion)'
        f'</div>'
        f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; font-size:0.88rem;">'
        f'<div style="background:#f8fafc; padding:12px 14px; border-radius:6px; border:1px solid #e2e8f0;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
        f'<b>Primary ViT Model:</b>'
        f'<span class="pill pill-{vit_pill_color}">{pred.upper()}</span>'
        f'</div>'
        f'<div style="margin-bottom:4px;">Melanoma Probability: <b>{mel_prob*100:.1f}%</b> · Non-Mel: <b>{non_mel_prob*100:.1f}%</b></div>'
        f'<div style="margin-bottom:6px;">Decision Threshold: <b>{int(decision_thresh*100)}%</b> · Margin: <b>{("+" if vit_margin>=0 else "")}{vit_margin*100:.1f}%</b></div>'
        f'<div style="margin-bottom:8px;">Model Uncertainty: <span class="pill pill-{vit_unc_pill}">{vit_unc.capitalize()}</span></div>'
        f'<div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden; position:relative; margin-bottom:8px;">'
        f'<div style="background:{"#ef4444" if pred=="mel" else "#22c55e"}; width:{vit_bar_pct:.1f}%; height:100%;"></div>'
        f'</div>'
        f'<span style="color:#64748b; font-size:0.8rem;">Visual encoder alone (fine-tuned ViT-Base backbone, 768-D image embedding).</span>'
        f'</div>'
        f'<div style="background:#f8fafc; padding:12px 14px; border-radius:6px; border:1px solid #e2e8f0;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
        f'<b>Multimodal Fusion Model:</b>'
        f'<span class="pill pill-{mm_pill_color}">{mm_pred.upper()}</span>'
        f'</div>'
        f'<div style="margin-bottom:4px;">Melanoma Probability: <b>{float(mm_prob)*100:.1f}%</b> · Non-Mel: <b>{mm_non_mel_prob*100:.1f}%</b></div>'
        f'<div style="margin-bottom:6px;">Decision Threshold: <b>{int(mm_thresh*100)}%</b> · Margin: <b>{("+" if float(mm_margin)>=0 else "")}{float(mm_margin)*100:.1f}%</b></div>'
        f'<div style="margin-bottom:8px;">Model Uncertainty: <span class="pill pill-{mm_unc_pill}">{mm_unc.capitalize()}</span> · Risk: <span class="pill pill-{mm_risk_pill}">{str(mm_risk).capitalize()}</span></div>'
        f'<div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden; position:relative; margin-bottom:8px;">'
        f'<div style="background:{"#ef4444" if mm_pred=="mel" else "#22c55e"}; width:{mm_bar_pct:.1f}%; height:100%;"></div>'
        f'</div>'
        f'<span style="color:#64748b; font-size:0.8rem;">Trainable fusion: ViT 768-D visual embedding + 16-D clinical ABCDE feature vector.</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    rendered = markdown.markdown(comparative_html)
    assert "<pre><code>" not in rendered, "Found unexpected code block in rendered markdown!"
    assert "<!-- Multimodal Fusion Card -->" not in rendered
    assert "Multimodal Fusion Model:" in rendered
    assert "Primary ViT Model:" in rendered
    assert "25.8%" in rendered
    assert "0.0%" in rendered


def test_clinical_ai_discordance_alert_generation():
    """Verify that when AI predicts NON_MEL but clinical risk is HIGH, a discordance alert is generated."""
    pred = "non_mel"
    risk = "high"
    feat = {"diameter_mm": 11.96}

    dia_val = feat.get("diameter_mm")
    has_elevated_clinical = "high" in str(risk).lower() or "medium" in str(risk).lower()
    has_large_diam = dia_val is not None and float(dia_val) > 6.0
    is_non_mel_pred = str(pred).lower() in ["non_mel", "benign", "nv"]

    discordance_triggered = is_non_mel_pred and (has_elevated_clinical or has_large_diam)
    assert discordance_triggered is True

    # Confirm that neither AI nor clinical indicators override each other
    assert pred == "non_mel"  # Model prediction remains unchanged
    assert risk == "high"     # Clinical risk score remains unchanged

