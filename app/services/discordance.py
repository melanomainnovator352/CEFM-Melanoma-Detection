"""
CEFM Three-Way Model Discordance Detection Service.

Compares predictions across:
1. ViT Baseline
2. Phase 1 Multimodal Fusion
3. Challenger M6 Dual-Scale RCAG

Detects and classifies consensus, visual-vs-multimodal discrepancy,
and partial model availability without suppressing disagreement.
"""

from __future__ import annotations
from typing import Dict, Any, Optional


def detect_three_way_discordance(
    vit_pred: Optional[str] = None,
    mm_pred: Optional[str] = None,
    m6_pred: Optional[str] = None,
    vit_prob: Optional[float] = None,
    mm_prob: Optional[float] = None,
    m6_prob: Optional[float] = None,
    vit_thresh: float = 0.80,
    mm_thresh: float = 0.36,
    m6_thresh: float = 0.690,
    m6_mode: str = "review",
) -> Dict[str, Any]:
    """
    Evaluates three-way agreement among ViT, Multimodal, and M6 predictions.

    Possible status codes:
    - 'all_mel': All active models predict MEL.
    - 'all_non_mel': All active models predict non-MEL.
    - 'vit_non_mel_m6_mel': ViT predicts non-MEL, but M6 predicts MEL.
    - 'vit_non_mel_mm_mel': ViT predicts non-MEL, but Multimodal predicts MEL.
    - 'm6_non_mel_other_mel': M6 predicts non-MEL, but ViT and/or Multimodal predicts MEL.
    - 'models_disagree': General model disagreement.
    - 'partial_availability': One or more models unavailable.
    """
    # Clean model inputs
    models_info = {
        "ViT": {
            "available": vit_pred is not None and str(vit_pred).lower() not in ("none", "not available", "—"),
            "pred": str(vit_pred).lower() if vit_pred is not None else None,
            "prob": round(float(vit_prob), 4) if vit_prob is not None else None,
            "thresh": round(float(vit_thresh), 4),
        },
        "Multimodal": {
            "available": mm_pred is not None and str(mm_pred).lower() not in ("none", "not available", "—"),
            "pred": str(mm_pred).lower() if mm_pred is not None else None,
            "prob": round(float(mm_prob), 4) if mm_prob is not None else None,
            "thresh": round(float(mm_thresh), 4),
        },
        "M6": {
            "available": m6_pred is not None and str(m6_pred).lower() not in ("none", "not available", "—"),
            "pred": str(m6_pred).lower() if m6_pred is not None else None,
            "prob": round(float(m6_prob), 4) if m6_prob is not None else None,
            "thresh": round(float(m6_thresh), 4),
            "mode": m6_mode,
        },
    }

    # Count active models
    active_models = {k: v for k, v in models_info.items() if v["available"]}
    total_active = len(active_models)

    if total_active < 2:
        return {
            "status": "partial_availability",
            "has_discordance": False,
            "severity": "info",
            "title": "Model Availability Note",
            "message": f"Only {total_active} model(s) available. Cross-model consensus requires at least 2 active models.",
            "active_model_count": total_active,
            "models": models_info,
        }

    # Classify predictions
    vit_is_mel = models_info["ViT"]["pred"] == "mel" if models_info["ViT"]["available"] else None
    mm_is_mel = models_info["Multimodal"]["pred"] == "mel" if models_info["Multimodal"]["available"] else None
    m6_is_mel = models_info["M6"]["pred"] == "mel" if models_info["M6"]["available"] else None

    mel_count = sum(1 for v in [vit_is_mel, mm_is_mel, m6_is_mel] if v is True)
    non_mel_count = sum(1 for v in [vit_is_mel, mm_is_mel, m6_is_mel] if v is False)

    # 1. Complete Consensus: All MEL
    if mel_count == total_active:
        return {
            "status": "all_mel",
            "has_discordance": False,
            "severity": "warning",
            "title": "Multi-Model Consensus: Elevated Malignancy Concern",
            "message": f"All {total_active} evaluated models (ViT, Multimodal, M6) predict MEL above their respective operating thresholds. Prioritized clinical evaluation recommended.",
            "active_model_count": total_active,
            "models": models_info,
        }

    # 2. Complete Consensus: All Non-MEL
    if non_mel_count == total_active:
        return {
            "status": "all_non_mel",
            "has_discordance": False,
            "severity": "info",
            "title": "Multi-Model Consensus: Lower Level of Concern",
            "message": f"All {total_active} evaluated models predict non-MEL below their respective thresholds. Standard monitoring recommended. (Note: automated screening does not rule out melanoma).",
            "active_model_count": total_active,
            "models": models_info,
        }

    # 3. Discordance: ViT non-mel, M6 mel
    if vit_is_mel is False and m6_is_mel is True:
        status_code = "vit_non_mel_m6_mel"
        title = "Model Discordance Alert: ViT Non-MEL vs. M6 Challenger MEL"
        message = (
            f"ViT baseline predicts NON_MEL ({models_info['ViT']['prob']*100:.1f}%), but M6 Dual-Scale RCAG predicts "
            f"MEL ({models_info['M6']['prob']*100:.1f}% at threshold {models_info['M6']['thresh']:.3f}). "
            "High-resolution local lesion features or clinical gating suggest suspicious patterns missed by the global ViT alone."
        )
    # 4. Discordance: ViT non-mel, Multimodal mel
    elif vit_is_mel is False and mm_is_mel is True:
        status_code = "vit_non_mel_mm_mel"
        title = "Model Discordance Alert: ViT Non-MEL vs. Multimodal Fusion MEL"
        message = (
            f"ViT baseline predicts NON_MEL ({models_info['ViT']['prob']*100:.1f}%), but Phase 1 Multimodal Fusion predicts "
            f"MEL ({models_info['Multimodal']['prob']*100:.1f}%). Clinical ABCDE features elevated the overall risk score."
        )
    # 5. Discordance: M6 non-mel, other mel
    elif m6_is_mel is False and (vit_is_mel is True or mm_is_mel is True):
        status_code = "m6_non_mel_other_mel"
        other_name = "ViT" if vit_is_mel is True else "Multimodal Fusion"
        title = f"Model Discordance Alert: M6 Non-MEL vs. {other_name} MEL"
        message = (
            f"M6 Dual-Scale predicts NON_MEL ({models_info['M6']['prob']*100:.1f}%), while {other_name} predicts "
            f"MEL. Scale-adaptive gating in M6 suppressed global visual suspicion due to reassuring local dermoscopic context."
        )
    else:
        status_code = "models_disagree"
        title = "Inter-Model Discordance Alert"
        message = "Evaluated models disagree on malignancy classification. Expert dermatological examination is required."

    return {
        "status": status_code,
        "has_discordance": True,
        "severity": "discordance",
        "title": title,
        "message": message,
        "active_model_count": total_active,
        "models": models_info,
    }

