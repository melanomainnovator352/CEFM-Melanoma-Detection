"""
CEFM Data Quality and Segmentation Audit Service.

Detects and flags data quality issues:
1. Segmentation failure (empty or missing mask)
2. Boundary touch (lesion edge contacts image frame)
3. Abnormal mask area (area ratio < 1% or > 85%)
4. Uncalibrated physical diameter (missing calibration ruler)
5. Missing clinical features (e.g. longitudinal evolution)
6. Model inference failures
7. High multi-photo variance (delta_p > 0.35)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional


def audit_data_quality(
    mask: Optional[np.ndarray] = None,
    image_shape: Optional[tuple] = None,
    features: Optional[dict] = None,
    classification: Optional[dict] = None,
    validation: Optional[dict] = None,
    multi_photo_delta_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Performs comprehensive data quality audit and returns structured warnings.
    """
    warnings: List[Dict[str, Any]] = []

    # 1. Segmentation Failure
    has_valid_mask = False
    if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0 and np.any(mask > 0):
        has_valid_mask = True
    else:
        warnings.append({
            "code": "SEGMENTATION_FAILURE",
            "severity": "critical",
            "title": "Lesion Segmentation Failed",
            "message": "Automated segmentation could not identify lesion boundaries. Morphological ABCDE features are degraded or zero-imputed.",
        })

    # 2. Boundary Touch & Mask Area checks (only if mask is present)
    if has_valid_mask and image_shape is not None and len(image_shape) >= 2:
        h, w = image_shape[:2]
        y_indices, x_indices = np.where(mask > 0)
        ymin, ymax = int(y_indices.min()), int(y_indices.max())
        xmin, xmax = int(x_indices.min()), int(x_indices.max())

        # Boundary touch (within 2 pixels of edge)
        touches_top = ymin <= 2
        touches_bottom = ymax >= h - 3
        touches_left = xmin <= 2
        touches_right = xmax >= w - 3

        if touches_top or touches_bottom or touches_left or touches_right:
            warnings.append({
                "code": "BOUNDARY_TOUCH",
                "severity": "warning",
                "title": "Lesion Contacts Image Edge",
                "message": "The segmented lesion boundary extends to the image border. The lesion appears partially clipped; diameter and shape irregularity may be underestimated.",
            })

        # Abnormal Mask Area Ratio
        total_pixels = h * w
        mask_pixels = np.count_nonzero(mask > 0)
        area_ratio = mask_pixels / float(total_pixels) if total_pixels > 0 else 0.0

        if area_ratio < 0.01:
            warnings.append({
                "code": "ABNORMAL_MASK_AREA_SMALL",
                "severity": "warning",
                "title": "Atypically Small Lesion Mask",
                "message": f"Segmented lesion occupies only {area_ratio*100:.2f}% of the image frame (< 1%). Fine feature measurements may be subject to pixel noise.",
            })
        elif area_ratio > 0.85:
            warnings.append({
                "code": "ABNORMAL_MASK_AREA_LARGE",
                "severity": "warning",
                "title": "Atypically Large Lesion Mask",
                "message": f"Segmented lesion occupies {area_ratio*100:.1f}% of the image frame (> 85%). Surround margin is constrained; contrast features may be distorted.",
            })

    # 3. Physical Ruler Calibration
    feat = features or {}
    diam_available = feat.get("diameter_mm_available", False)
    diam_mm = feat.get("diameter_mm")
    if isinstance(feat.get("diameter"), dict):
        diam_available = diam_available or feat.get("diameter", {}).get("available", False)
        diam_mm = diam_mm or feat.get("diameter", {}).get("max_diameter_mm")

    if not diam_available or diam_mm is None:
        warnings.append({
            "code": "CALIBRATION_UNAVAILABLE",
            "severity": "info",
            "title": "Physical Scale Uncalibrated",
            "message": "Physical dermoscopic millimeter ruler was not detected. Diameter reported in pixels only; millimeter criteria (e.g. 6 mm) cannot be verified.",
        })

    # 4. Missing Clinical Features (Longitudinal Evolution)
    evolution = feat.get("evolution") or {}
    if not evolution.get("available", False):
        warnings.append({
            "code": "EVOLUTION_UNAVAILABLE",
            "severity": "info",
            "title": "Longitudinal Evolution Unavailable",
            "message": "Single clinical visit evaluated. Evolution assessment (E) requires comparative prior visit photography.",
        })

    # 5. Model Inference Failures
    clf = classification or {}
    if clf.get("method") in ("Not available", "failed"):
        warnings.append({
            "code": "VIT_INFERENCE_FAILED",
            "severity": "critical",
            "title": "ViT Model Unavailable",
            "message": "Vision Transformer classifier could not execute inference.",
        })

    # 6. High Multi-Photo Prediction Discrepancy
    if multi_photo_delta_p is not None and multi_photo_delta_p > 0.35:
        warnings.append({
            "code": "HIGH_MULTI_PHOTO_VARIANCE",
            "severity": "warning",
            "title": "High Inter-Photograph Variance",
            "message": f"Melanoma probabilities differ by {multi_photo_delta_p*100:.1f}% (> 35%) across multiple photographs of this lesion. Individual photos must be inspected for lighting artifacts or focal atypical features.",
        })

    return {
        "has_warnings": len(warnings) > 0,
        "warning_count": len(warnings),
        "warnings": warnings,
    }

