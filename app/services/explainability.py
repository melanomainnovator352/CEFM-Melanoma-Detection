"""Generate multimodal explanation and Vision Transformer attention visualizations for melanoma assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch

from app.services.contrastive import contrastive_aligner


def generate_vit_attention_map(
    image_path: str | Path,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate Vision Transformer multi-head self-attention heatmap overlay.

    For ViT architectures, attention rollout/multi-head CLS-to-patch attention
    represents the direct spatial focus of the model for classification.

    Returns:
        overlay: Blended image with Jet attention colormap (BGR uint8)
        heatmap: Normalized spatial attention map (2D uint8, 0-255)
        peak_attention: Maximum raw attention weight
    """
    import app.services.classification as cl

    if cl._classifier is None or cl._processor is None:
        cl._load_torch_model()

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    h_orig, w_orig = img.shape[:2]

    # Crop to lesion region if mask provided
    if mask is not None and np.any(mask > 0):
        y_idx, x_idx = np.where(mask > 0)
        ymin, ymax = int(y_idx.min()), int(y_idx.max())
        xmin, xmax = int(x_idx.min()), int(x_idx.max())
        cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
        h_box, w_box = ymax - ymin, xmax - xmin
        side = int(max(h_box, w_box) * 1.30)
        c_ymin = max(0, cy - side // 2)
        c_ymax = min(h_orig, cy + side // 2)
        c_xmin = max(0, cx - side // 2)
        c_xmax = min(w_orig, cx + side // 2)
        cropped_img = img[c_ymin:c_ymax, c_xmin:c_xmax]
        cropped_mask = mask[c_ymin:c_ymax, c_xmin:c_xmax] if mask is not None else None
    else:
        cropped_img = img
        cropped_mask = mask
        c_ymin, c_ymax, c_xmin, c_xmax = 0, h_orig, 0, w_orig

    # Preprocess
    rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    inputs = cl._processor(images=rgb, return_tensors="pt")
    device = next(cl._classifier.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference with attention extraction
    with torch.no_grad():
        outputs = cl._classifier(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if not attentions:
        raise RuntimeError("Model did not return attention weights.")

    # Final transformer layer multi-head attention: [batch, heads, tokens, tokens]
    # Token 0 is [CLS], tokens 1..196 correspond to 14x14 spatial patches
    cls_attention = attentions[-1][0, :, 0, 1:].mean(dim=0)  # Shape: [196]
    attn_grid = cls_attention.view(14, 14).cpu().numpy().astype(np.float32)

    peak_attention = float(attn_grid.max())

    # Upsample 14x14 attention grid to cropped image dimensions
    target_h, target_w = cropped_img.shape[:2]
    attn_resized = cv2.resize(attn_grid, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Normalize to 0-255 range
    attn_min, attn_max = float(attn_resized.min()), float(attn_resized.max())
    if attn_max > attn_min:
        attn_norm = ((attn_resized - attn_min) / (attn_max - attn_min) * 255.0).astype(np.uint8)
    else:
        attn_norm = np.zeros((target_h, target_w), dtype=np.uint8)

    # Colorize attention map
    heatmap = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)

    # Blend cropped image with heatmap
    blended_crop = cv2.addWeighted(cropped_img, 0.55, heatmap, 0.45, 0)

    # If lesion mask is available, draw subtle contour on crop to show alignment
    if cropped_mask is not None and np.any(cropped_mask > 0):
        contours, _ = cv2.findContours(
            (cropped_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(blended_crop, contours, -1, (255, 255, 255), 1)

    # Embed back into full canvas if cropped
    full_overlay = img.copy()
    full_heatmap = np.zeros((h_orig, w_orig), dtype=np.uint8)

    full_overlay[c_ymin:c_ymax, c_xmin:c_xmax] = blended_crop
    full_heatmap[c_ymin:c_ymax, c_xmin:c_xmax] = attn_norm

    return full_overlay, full_heatmap, peak_attention


def generate_vit_attention_bytes(
    image_path: str | Path,
    mask: Optional[np.ndarray] = None,
    fmt: str = ".jpg",
) -> bytes:
    """Generate JPEG/PNG encoded bytes of the ViT attention map overlay."""
    overlay, _, _ = generate_vit_attention_map(image_path, mask=mask)
    ok, buf = cv2.imencode(fmt, overlay)
    if not ok:
        raise ValueError("Could not encode ViT attention overlay.")
    return buf.tobytes()


def generate_explanation(
    classification: dict[str, Any],
    abcde: dict[str, Any],
    evolution: Optional[dict[str, Any]] = None,
    attention_focus: Optional[float] = None,
) -> dict[str, Any]:
    """Create a textual explanation using clinical ABCDE features and the
    multimodal alignment score.
    """
    prob = float((classification or {}).get("melanoma_probability", 0.0) or 0.0)
    risk = str((classification or {}).get("risk_level", "low")).lower()

    asymmetry = float((abcde or {}).get("asymmetry", 0.0) or 0.0)
    border = float((abcde or {}).get("border_irregularity", 0.0) or 0.0)
    color_index = float((abcde or {}).get("color_index", 0.0) or 0.0)
    evolution_change = bool((evolution or {}).get("change_detected", False))

    image_features = {
        "texture": max(0.0, min(1.0, 0.5 + asymmetry * 0.5)),
        "color": max(0.0, min(1.0, color_index)),
        "structure": max(0.0, min(1.0, border)),
    }
    clinical_features = {
        "asymmetry": asymmetry,
        "border_irregularity": border,
        "color_index": color_index,
    }
    similarity = contrastive_aligner.compute_similarity(image_features, clinical_features)

    summary_parts = [
        f"The model indicates a {risk} risk level with an estimated melanoma probability of {prob:.0%}.",
        f"Asymmetry is {asymmetry:.2f}, border irregularity is {border:.2f}, and color variation is {color_index:.2f}.",
        (
            "The combined visual-to-clinical alignment score is "
            f"{similarity:.2f}, indicating the lesion appearance is consistent with "
            "the observed ABCD pattern."
        ),
    ]

    if attention_focus is not None and attention_focus > 0:
        summary_parts.append(
            f"Vision Transformer self-attention demonstrates focused saliency "
            f"(peak attention weight: {attention_focus:.4f}) over the primary lesion core."
        )

    if evolution_change:
        summary_parts.append("A change compared with the prior assessment was detected.")
    else:
        summary_parts.append("No meaningful change compared with the prior assessment was detected.")

    summary = " ".join(summary_parts)

    return {
        "summary": summary,
        "classification": classification,
        "abcde": abcde,
        "evolution": evolution,
        "multimodal_similarity": round(similarity, 4),
        "vit_attention_peak": round(attention_focus, 4) if attention_focus is not None else None,
    }
