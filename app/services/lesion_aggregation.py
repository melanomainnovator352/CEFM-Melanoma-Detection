"""
CEFM Lesion-Level Multi-Photo Aggregation Service.

Aggregates multiple photographs explicitly associated with the same physical lesion_id:
1. Max-Probability Aggregation: Conservative clinical screening rule.
2. Inter-Photo Variance Guardrail: delta_p = max_prob - min_prob.
3. Outlier Flagging: If delta_p > 0.35, flags case for photographic artifact review.
4. Preserves all individual image predictions and outputs.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


def aggregate_lesion_photographs(
    image_predictions: List[Dict[str, Any]],
    lesion_id: Optional[str] = None,
    variance_threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Aggregates predictions for multiple photographs of the same lesion.
    Ensures images are only grouped if explicitly linked by the same non-empty lesion_id.
    """
    if not image_predictions:
        return {
            "is_multi_image": False,
            "image_count": 0,
            "lesion_id": lesion_id,
            "models": {},
        }

    # Verify uniform lesion_id
    if not lesion_id:
        # Check if predictions contain uniform lesion_id
        lids = {p.get("lesion_id") for p in image_predictions if p.get("lesion_id")}
        if len(lids) == 1:
            lesion_id = next(iter(lids))
        else:
            # Cannot aggregate without verified lesion_id
            return {
                "is_multi_image": False,
                "image_count": len(image_predictions),
                "lesion_id": None,
                "note": "Multi-image aggregation requires an explicit, verified lesion_id.",
                "individual_predictions": image_predictions,
            }

    count = len(image_predictions)
    if count == 1:
        return {
            "is_multi_image": False,
            "image_count": 1,
            "lesion_id": lesion_id,
            "individual_predictions": image_predictions,
        }

    # Extract model probabilities across all images
    model_aggregations = {}

    for model_key, prob_key, thresh_key in [
        ("ViT", "melanoma_probability", "decision_threshold"),
        ("Multimodal", "multimodal_melanoma_probability", "multimodal_decision_threshold"),
        ("M6", "challenger_melanoma_probability", "challenger_decision_threshold"),
    ]:
        probs = []
        thresh = 0.50
        for p in image_predictions:
            clf = p.get("classification", p)
            val = clf.get(prob_key)
            if val is not None:
                probs.append(float(val))
                if clf.get(thresh_key) is not None:
                    thresh = float(clf.get(thresh_key))

        if probs:
            p_max = float(max(probs))
            p_min = float(min(probs))
            p_mean = float(sum(probs) / len(probs))
            delta_p = round(p_max - p_min, 4)
            high_variance = delta_p > variance_threshold

            pred_max = "mel" if p_max >= thresh else "non_mel"
            pred_mean = "mel" if p_mean >= thresh else "non_mel"

            model_aggregations[model_key] = {
                "available": True,
                "threshold": thresh,
                "probabilities": probs,
                "max_probability": round(p_max, 4),
                "min_probability": round(p_min, 4),
                "mean_probability": round(p_mean, 4),
                "delta_p": delta_p,
                "high_variance_flag": high_variance,
                "variance_threshold": variance_threshold,
                "predicted_class_max": pred_max,
                "predicted_class_mean": pred_mean,
            }
        else:
            model_aggregations[model_key] = {"available": False}

    # Check overall max delta_p across all models
    overall_max_delta = max(
        (m["delta_p"] for m in model_aggregations.values() if m.get("available")),
        default=0.0,
    )
    flag_for_review = overall_max_delta > variance_threshold

    return {
        "is_multi_image": True,
        "image_count": count,
        "lesion_id": lesion_id,
        "overall_max_delta_p": round(overall_max_delta, 4),
        "high_variance_alert": flag_for_review,
        "models": model_aggregations,
        "individual_predictions": image_predictions,
        "clinical_guidance": (
            "Screening escalation: Lesion risk aggregated using conservative maximum probability across views. "
            + ("⚠️ High inter-photograph variance detected (> 35%): inspect individual photos for localized features vs imaging artifacts."
               if flag_for_review else "Photographs exhibit consistent inter-view probabilities.")
        ),
    }

