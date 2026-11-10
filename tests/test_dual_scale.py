"""
Unit tests for Dual-Scale Feature Extraction Engine (Phase 2).
Tests:
- Empty and None masks (clean fallback)
- Incorrect or malformed masks
- Boundary-adjacent lesions (clamping)
- Large lesions (multi-patch tiling)
- Small lesions (single centered patch)
- Spatial coordinate boundary validation
- ViT embedding dimensions (768-D global, 768-D local)
"""
import pytest
import numpy as np
import torch
from app.services.dual_scale import (
    compute_lesion_spatial_bounds,
    extract_global_crop,
    extract_local_patches,
    extract_dual_scale_views,
    extract_dual_scale_embeddings_single,
)
from app.services.classification import _load_torch_model, _processor


def test_empty_mask_fallback():
    """Verify that empty or None mask cleanly falls back to image center without crashing."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # None mask
    bounds_none = compute_lesion_spatial_bounds(None, img.shape)
    assert bounds_none["has_mask"] is False
    assert bounds_none["cy"] == 200
    assert bounds_none["cx"] == 300

    # All-zero mask
    zero_mask = np.zeros((400, 600), dtype=np.uint8)
    bounds_zero = compute_lesion_spatial_bounds(zero_mask, img.shape)
    assert bounds_zero["has_mask"] is False
    assert bounds_zero["area_px"] == 0

    views = extract_dual_scale_views(img, None)
    assert views["global_image"].shape == (224, 224, 3)
    assert len(views["local_patches"]) >= 1
    assert views["local_patches"][0]["image"].shape == (224, 224, 3)


def test_incorrect_and_malformed_masks():
    """Verify robust handling of non-standard or malformed masks."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Non-array object
    bounds = compute_lesion_spatial_bounds("not_a_mask", img.shape)
    assert bounds["has_mask"] is False

    # Mask with negative or non-boolean values
    weird_mask = np.full((300, 300), -5, dtype=np.int32)
    bounds_weird = compute_lesion_spatial_bounds(weird_mask, img.shape)
    assert bounds_weird["has_mask"] is False


def test_boundary_adjacent_lesions():
    """Verify that lesions touching or hugging image borders are clamped safely."""
    H, W = 300, 400
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Lesion flush with top-left corner
    mask_tl = np.zeros((H, W), dtype=np.uint8)
    mask_tl[0:50, 0:60] = 255
    bounds_tl = compute_lesion_spatial_bounds(mask_tl, img.shape)
    assert bounds_tl["ymin"] == 0
    assert bounds_tl["xmin"] == 0

    views_tl = extract_dual_scale_views(img, mask_tl)
    for p in views_tl["local_patches"]:
        b = p["bbox"]
        assert 0 <= b["ymin"] < b["ymax"] <= H
        assert 0 <= b["xmin"] < b["xmax"] <= W
        assert p["image"].shape == (224, 224, 3)

    # Lesion flush with bottom-right corner
    mask_br = np.zeros((H, W), dtype=np.uint8)
    mask_br[H-40:H, W-50:W] = 255
    views_br = extract_dual_scale_views(img, mask_br)
    for p in views_br["local_patches"]:
        b = p["bbox"]
        assert 0 <= b["ymin"] < b["ymax"] <= H
        assert 0 <= b["xmin"] < b["xmax"] <= W
        assert p["image"].shape == (224, 224, 3)


def test_large_lesion_multi_patch_tiling():
    """Verify that lesions larger than 224x224 trigger multiple overlapping native patches."""
    H, W = 1000, 1000
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    
    # Large 600x600 lesion
    mask_large = np.zeros((H, W), dtype=np.uint8)
    mask_large[200:800, 200:800] = 255

    views = extract_dual_scale_views(img, mask_large, patch_size=224)
    # Must have multiple patches to cover 600x600
    assert views["num_local_patches"] > 1
    assert views["num_local_patches"] <= 5

    # Center patch must be first
    assert views["local_patches"][0]["is_center"] is True

    # Validate all bboxes
    for p in views["local_patches"]:
        b = p["bbox"]
        assert b["ymax"] - b["ymin"] == 224
        assert b["xmax"] - b["xmin"] == 224
        assert 0 <= b["ymin"] <= H
        assert 0 <= b["xmin"] <= W
        assert p["image"].shape == (224, 224, 3)


def test_small_lesion_single_patch():
    """Verify that small lesion produces a single centered native patch."""
    H, W = 600, 600
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Small 40x40 lesion in center
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[280:320, 280:320] = 255

    views = extract_dual_scale_views(img, mask, patch_size=224)
    assert views["num_local_patches"] == 1
    assert views["local_patches"][0]["image"].shape == (224, 224, 3)


def test_dual_scale_embedding_dimensions():
    """Verify that dual-scale extraction outputs strictly 768-D global and local embeddings."""
    import app.services.classification as cl
    vit_model = cl._load_torch_model()
    processor = cl._processor
    assert vit_model is not None
    assert processor is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[150:250, 150:250] = 255

    res = extract_dual_scale_embeddings_single(img, mask, vit_model, processor, device)
    
    assert res["global_embedding"].shape == (768,)
    assert res["local_embedding"].shape == (768,)
    assert res["patch_embeddings"].ndim == 2
    assert res["patch_embeddings"].shape[1] == 768
    assert not torch.isnan(res["global_embedding"]).any()
    assert not torch.isnan(res["local_embedding"]).any()
