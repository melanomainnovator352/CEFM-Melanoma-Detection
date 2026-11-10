"""
Automated tests for Vision Transformer explainability and attention visualization.
Verifies multi-head CLS attention extraction, overlay blending, heatmap normalization,
and visualization service integration.
"""
from pathlib import Path
import numpy as np
import pytest

from app.services.explainability import (
    generate_vit_attention_map,
    generate_vit_attention_bytes,
    generate_explanation,
)
from app.services.visualization import render_view, vis_attention


def get_sample_image_path() -> Path:
    candidates = [
        Path("data/uploads/f9a8de9c96734d41bb31cf20adc88567.jpg"),
        Path("data/uploads/debug_crop_mel.jpg"),
        Path("D:/datasets/HAM10000/HAM10000_images_part_1/ISIC_0024306.jpg"),
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip("No sample image found for explainability testing")


def test_vit_attention_map_generation():
    """Verify ViT attention overlay and heatmap have correct shapes and types."""
    img_path = get_sample_image_path()
    overlay, heatmap, peak = generate_vit_attention_map(img_path)

    assert isinstance(overlay, np.ndarray)
    assert isinstance(heatmap, np.ndarray)
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert heatmap.ndim == 2
    assert overlay.shape[:2] == heatmap.shape
    assert overlay.dtype == np.uint8
    assert heatmap.dtype == np.uint8
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 255.0
    assert peak > 0.0


def test_vit_attention_with_lesion_mask():
    """Verify attention map generates when mask is supplied."""
    img_path = get_sample_image_path()
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 1

    overlay, heatmap, peak = generate_vit_attention_map(img_path, mask=mask)
    assert overlay is not None
    assert peak > 0.0


def test_vit_attention_bytes_encoding():
    """Verify attention overlay encodes to valid JPEG/PNG bytes."""
    img_path = get_sample_image_path()
    jpg_bytes = generate_vit_attention_bytes(img_path, fmt=".jpg")
    assert isinstance(jpg_bytes, bytes)
    assert len(jpg_bytes) > 1000
    # JPEG magic header: FF D8
    assert jpg_bytes[:2] == b"\xff\xd8"


def test_visualization_view_registry_includes_attention():
    """Verify 'attention' and 'explainability' views are registered in visualization service."""
    img_path = get_sample_image_path()
    att_bytes = render_view("attention", str(img_path))
    assert isinstance(att_bytes, bytes)
    assert len(att_bytes) > 1000
    assert att_bytes[:2] == b"\xff\xd8"

    exp_bytes = render_view("explainability", str(img_path))
    assert isinstance(exp_bytes, bytes)
    assert len(exp_bytes) > 1000


def test_explanation_incorporates_vit_attention():
    """Verify textual explanation integrates attention peak saliency."""
    clf = {"melanoma_probability": 0.82, "risk_level": "high"}
    abc = {"asymmetry": 0.45, "border_irregularity": 0.52, "color_index": 0.60}
    exp = generate_explanation(clf, abc, attention_focus=0.032)

    assert "summary" in exp
    assert "vit_attention_peak" in exp
    assert exp["vit_attention_peak"] == 0.032
    assert "Vision Transformer" in exp["summary"]
