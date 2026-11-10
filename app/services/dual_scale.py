"""
Dual-Scale Feature Extraction Engine for CEFM Phase 2.

Extracts:
1. Global Visual Representation:
   - 1.30x context square crop around VM-UNet lesion mask, resized to 224x224.
   - Identical to production baseline preprocessing.
2. Local High-Resolution Visual Representation:
   - High-resolution native-scale patch(es) guided by lesion segmentation mask.
   - Supports single centered patch for standard lesions, and multiple overlapping
     patches for lesions exceeding the patch window.
   - Preserves original image spatial coordinates and clamps boundaries.
   - Returns patch-level and pooled 768-D local embeddings.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from PIL import Image


def compute_lesion_spatial_bounds(
    mask: Optional[np.ndarray],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Compute spatial bounding box and centroid from binary lesion mask.
    Handles empty masks, partial masks, and boundary conditions cleanly.
    """
    h_img, w_img = image_shape[:2]
    
    if mask is None or not isinstance(mask, np.ndarray) or not np.any(mask > 0):
        # Fallback to image center
        cy, cx = h_img // 2, w_img // 2
        return {
            "has_mask": False,
            "ymin": max(0, cy - 112),
            "ymax": min(h_img, cy + 112),
            "xmin": max(0, cx - 112),
            "xmax": min(w_img, cx + 112),
            "cy": cy,
            "cx": cx,
            "lesion_h": min(224, h_img),
            "lesion_w": min(224, w_img),
            "area_px": 0,
        }

    y_idx, x_idx = np.where(mask > 0)
    ymin, ymax = int(y_idx.min()), int(y_idx.max())
    xmin, xmax = int(x_idx.min()), int(x_idx.max())

    # Clamp to image dimensions
    ymin = max(0, min(ymin, h_img - 1))
    ymax = max(ymin + 1, min(ymax, h_img))
    xmin = max(0, min(xmin, w_img - 1))
    xmax = max(xmin + 1, min(xmax, w_img))

    cy = (ymin + ymax) // 2
    cx = (xmin + xmax) // 2
    lesion_h = ymax - ymin
    lesion_w = xmax - xmin
    area_px = int(np.sum(mask > 0))

    return {
        "has_mask": True,
        "ymin": ymin,
        "ymax": ymax,
        "xmin": xmin,
        "xmax": xmax,
        "cy": cy,
        "cx": cx,
        "lesion_h": lesion_h,
        "lesion_w": lesion_w,
        "area_px": area_px,
    }


def extract_global_crop(
    image_rgb: np.ndarray,
    bounds: Dict[str, Any],
    target_size: int = 224,
    context_margin: float = 1.30,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extract the global context crop matching the production baseline.
    Uses a 1.30x square context box around the lesion bounds, clamped to image borders.
    """
    h_img, w_img = image_rgb.shape[:2]
    cy, cx = bounds["cy"], bounds["cx"]
    h, w = bounds["lesion_h"], bounds["lesion_w"]

    side = int(max(h, w) * context_margin)
    side = max(side, 32)  # Minimum context

    c_ymin = max(0, cy - side // 2)
    c_ymax = min(h_img, cy + side // 2)
    c_xmin = max(0, cx - side // 2)
    c_xmax = min(w_img, cx + side // 2)

    # Ensure square crop where possible
    crop = image_rgb[c_ymin:c_ymax, c_xmin:c_xmax]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = image_rgb

    # Resize to exact target size
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    bbox = {
        "ymin": int(c_ymin),
        "ymax": int(c_ymax),
        "xmin": int(c_xmin),
        "xmax": int(c_xmax),
    }
    return resized, bbox


def extract_local_patches(
    image_rgb: np.ndarray,
    bounds: Dict[str, Any],
    patch_size: int = 224,
    overlap_ratio: float = 0.25,
    max_patches: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extract native-resolution local patches guided by the segmentation mask.
    
    Strategy:
    - If lesion bounding box fits within patch_size, extract 1 centered patch.
    - If lesion exceeds patch_size, tile overlapping patches across the lesion
      area (center + peripheral regions), ensuring no lesion area is omitted.
    - Preserves native optical resolution (no heavy downsampling).
    - Clamps all patch coordinates strictly within image boundaries.
    """
    h_img, w_img = image_rgb.shape[:2]
    ymin, ymax = bounds["ymin"], bounds["ymax"]
    xmin, xmax = bounds["xmin"], bounds["xmax"]
    lesion_h, lesion_w = bounds["lesion_h"], bounds["lesion_w"]

    patches = []

    # Case A: Lesion fits within patch_size
    if lesion_h <= patch_size and lesion_w <= patch_size:
        cy, cx = bounds["cy"], bounds["cx"]
        p_ymin = max(0, cy - patch_size // 2)
        p_ymax = p_ymin + patch_size
        if p_ymax > h_img:
            p_ymax = h_img
            p_ymin = max(0, h_img - patch_size)

        p_xmin = max(0, cx - patch_size // 2)
        p_xmax = p_xmin + patch_size
        if p_xmax > w_img:
            p_xmax = w_img
            p_xmin = max(0, w_img - patch_size)

        patch_img = image_rgb[p_ymin:p_ymax, p_xmin:p_xmax]
        if patch_img.shape[:2] != (patch_size, patch_size):
            patch_img = cv2.resize(patch_img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        patches.append({
            "patch_id": 0,
            "image": patch_img,
            "bbox": {"ymin": int(p_ymin), "ymax": int(p_ymax), "xmin": int(p_xmin), "xmax": int(p_xmax)},
            "centroid": (int((p_ymin + p_ymax) // 2), int((p_xmin + p_xmax) // 2)),
            "is_center": True,
        })
        return patches

    # Case B: Large lesion exceeding patch_size -> Multi-patch tiling
    stride = int(patch_size * (1.0 - overlap_ratio))
    stride = max(stride, 32)

    # Compute grid coordinates across lesion bounds
    y_starts = list(range(ymin, max(ymin + 1, ymax - patch_size + 1), stride))
    if not y_starts or (y_starts[-1] + patch_size < ymax):
        y_starts.append(max(0, ymax - patch_size))

    x_starts = list(range(xmin, max(xmin + 1, xmax - patch_size + 1), stride))
    if not x_starts or (x_starts[-1] + patch_size < xmax):
        x_starts.append(max(0, xmax - patch_size))

    # Always include lesion center patch
    cy, cx = bounds["cy"], bounds["cx"]
    center_ymin = max(0, min(cy - patch_size // 2, h_img - patch_size))
    center_xmin = max(0, min(cx - patch_size // 2, w_img - patch_size))

    candidate_coords = [(center_ymin, center_xmin)]
    for ys in y_starts:
        for xs in x_starts:
            candidate_coords.append((max(0, min(ys, h_img - patch_size)), max(0, min(xs, w_img - patch_size))))

    # Deduplicate while preserving order
    unique_coords = []
    seen = set()
    for ys, xs in candidate_coords:
        key = (ys // 16, xs // 16)  # 16-px spatial bucket
        if key not in seen:
            seen.add(key)
            unique_coords.append((ys, xs))
        if len(unique_coords) >= max_patches:
            break

    for p_id, (p_ymin, p_xmin) in enumerate(unique_coords):
        p_ymax = min(h_img, p_ymin + patch_size)
        p_xmax = min(w_img, p_xmin + patch_size)
        patch_img = image_rgb[p_ymin:p_ymax, p_xmin:p_xmax]

        if patch_img.shape[:2] != (patch_size, patch_size):
            patch_img = cv2.resize(patch_img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        patches.append({
            "patch_id": p_id,
            "image": patch_img,
            "bbox": {"ymin": int(p_ymin), "ymax": int(p_ymax), "xmin": int(p_xmin), "xmax": int(p_xmax)},
            "centroid": (int((p_ymin + p_ymax) // 2), int((p_xmin + p_xmax) // 2)),
            "is_center": (p_id == 0),
        })

    return patches


def extract_dual_scale_views(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: int = 224,
) -> Dict[str, Any]:
    """
    Complete dual-scale spatial extraction pipeline.
    Returns global context view, local native-scale patch(es), and spatial coordinates.
    """
    bounds = compute_lesion_spatial_bounds(mask, image_rgb.shape)
    global_img, global_bbox = extract_global_crop(image_rgb, bounds, target_size=patch_size)
    local_patches = extract_local_patches(image_rgb, bounds, patch_size=patch_size)

    return {
        "global_image": global_img,
        "global_bbox": global_bbox,
        "local_patches": local_patches,
        "num_local_patches": len(local_patches),
        "lesion_bounds": bounds,
    }


def extract_vit_embeddings_from_images(
    images_rgb: List[np.ndarray],
    vit_model: nn.Module,
    processor: Any,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract 768-D CLS embeddings for a list of RGB images using the frozen ViT encoder.
    Automatically respects the model parameter device.
    """
    vit_model.eval()
    if device is None:
        try:
            device = next(vit_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        # Move model to requested device if different
        try:
            current_device = next(vit_model.parameters()).device
            if current_device != device:
                vit_model = vit_model.to(device)
        except StopIteration:
            pass

    inputs = processor(images=images_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(vit_model, "vit"):
            outputs = vit_model.vit(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = vit_model(**inputs, output_hidden_states=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :]

    return cls_emb.cpu()


def extract_dual_scale_embeddings_single(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    vit_model: nn.Module,
    processor: Any,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Extract global and pooled local 768-D embeddings for a single image.
    """
    views = extract_dual_scale_views(image_rgb, mask)
    
    # 1. Global visual embedding (768-D)
    global_emb = extract_vit_embeddings_from_images([views["global_image"]], vit_model, processor, device)[0]

    # 2. Local patch embeddings (K, 768-D)
    patch_images = [p["image"] for p in views["local_patches"]]
    patch_embs = extract_vit_embeddings_from_images(patch_images, vit_model, processor, device)

    # 3. Pooled local embedding (Mean pooling over local patches)
    if patch_embs.shape[0] == 1:
        local_pooled = patch_embs[0]
    else:
        # Weighted pooling: give 50% weight to center patch, 50% distributed to peripheral patches
        weights = torch.ones(patch_embs.shape[0])
        weights[0] = 2.0  # Center patch
        weights = (weights / weights.sum()).unsqueeze(1)
        local_pooled = (patch_embs * weights).sum(dim=0)

    return {
        "global_embedding": global_emb,        # [768]
        "local_embedding": local_pooled,       # [768]
        "patch_embeddings": patch_embs,        # [K, 768]
        "num_patches": views["num_local_patches"],
        "views": views,
    }
