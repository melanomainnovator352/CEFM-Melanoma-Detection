"""
CEFM Step 17: Experimental Visual Safety Guardrail Wrapper.

Provides safe clinical arbitration when non-image demographic metadata
diverges from or suppresses dermoscopic visual evidence.

Design Principles:
1. Preserves raw visual probability (p_visual) and raw fusion probability (p_fusion).
2. Never replaces or hides either raw prediction.
3. Detects demographic suppression (p_visual elevated, p_fusion suppressed).
4. Provides an explicit guardrail-arbitrated probability (p_guardrail) that prevents
   benign demographic priors from silently extinguishing visual malignancy risk.
5. Strictly isolated and labeled experimental.
"""
from __future__ import annotations

from typing import Dict, Any, Optional


class ClinicalSafetyGuardrail:
    """
    Experimental Visual Safety Guardrail for clinical metadata fusion.
    """

    def __init__(
        self,
        visual_alert_threshold: float = 0.50,
        fusion_decision_threshold: float = 0.2417,
        suppression_delta_threshold: float = 0.30,
        high_confidence_visual_threshold: float = 0.70,
        extreme_suppression_fusion_threshold: float = 0.01,
    ):
        self.visual_alert_threshold = visual_alert_threshold
        self.fusion_decision_threshold = fusion_decision_threshold
        self.suppression_delta_threshold = suppression_delta_threshold
        self.high_confidence_visual_threshold = high_confidence_visual_threshold
        self.extreme_suppression_fusion_threshold = extreme_suppression_fusion_threshold

    def evaluate(
        self,
        visual_probability: float,
        fusion_probability: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate visual vs fusion probability and apply safety arbitration.

        Args:
            visual_probability: Raw visual-only model probability in [0, 1]
            fusion_probability: Raw multimodal fusion probability in [0, 1]
            metadata: Optional patient demographics dict (age, sex, localization)

        Returns:
            Dict containing raw probabilities, arbitration status, suppression flags,
            and the guardrail-arbitrated probability.
        """
        metadata = metadata or {}
        p_vis = float(visual_probability)
        p_fus = float(fusion_probability)
        delta_p = p_vis - p_fus  # Positive means visual > fusion (suppression)

        # 1. Detection of suppression conditions
        is_extreme_suppression = bool(
            p_vis >= self.high_confidence_visual_threshold
            and p_fus < self.extreme_suppression_fusion_threshold
        )
        is_moderate_suppression = bool(
            p_vis >= self.visual_alert_threshold
            and p_fus < 0.10
        )
        is_delta_suppression = bool(delta_p >= self.suppression_delta_threshold)

        suppression_detected = is_extreme_suppression or is_moderate_suppression or is_delta_suppression

        if is_extreme_suppression:
            severity = "critical"
        elif is_moderate_suppression or (is_delta_suppression and p_vis >= self.visual_alert_threshold):
            severity = "high"
        elif is_delta_suppression:
            severity = "moderate"
        else:
            severity = "none"

        # 2. Guardrail arbitration logic
        # If visual evidence indicates high/suspicious risk, do not allow demographic prior
        # to pull the probability below visual suspicion.
        guardrail_triggered = False
        if p_vis >= self.visual_alert_threshold and p_fus < self.fusion_decision_threshold:
            guardrail_triggered = True
            p_guardrail = max(p_fus, p_vis)
            action = "visual_override_applied"
            recommendation = (
                "WARNING: Demographic prior suppressed dermoscopic visual suspicion. "
                "The safety guardrail has arbitrated to the visual probability. "
                "Urgent specialist dermoscopic / histopathological review is advised."
            )
        elif is_extreme_suppression:
            guardrail_triggered = True
            p_guardrail = max(p_fus, p_vis)
            action = "critical_suppression_override"
            recommendation = (
                "CRITICAL ALERT: Strong visual melanoma evidence was almost entirely extinguished "
                "by demographic metadata. Guardrail restored visual risk level."
            )
        else:
            p_guardrail = p_fus
            action = "standard_fusion_retained"
            recommendation = "Standard multimodal fusion assessment."

        # Classification calls
        pred_visual = "mel" if p_vis >= self.visual_alert_threshold else "non_mel"
        pred_fusion = "mel" if p_fus >= self.fusion_decision_threshold else "non_mel"
        pred_guardrail = "mel" if p_guardrail >= self.fusion_decision_threshold else "non_mel"

        prediction_changed_by_metadata = (pred_visual == "mel" and pred_fusion == "non_mel")
        prediction_restored_by_guardrail = (prediction_changed_by_metadata and pred_guardrail == "mel")

        return {
            "visual_probability": round(p_vis, 4),
            "fusion_probability": round(p_fus, 4),
            "guardrail_probability": round(p_guardrail, 4),
            "delta_probability": round(delta_p, 4),
            "suppression_detected": suppression_detected,
            "suppression_severity": severity,
            "is_extreme_suppression": is_extreme_suppression,
            "is_moderate_suppression": is_moderate_suppression,
            "guardrail_triggered": guardrail_triggered,
            "action_taken": action,
            "prediction_visual": pred_visual,
            "prediction_fusion": pred_fusion,
            "prediction_guardrail": pred_guardrail,
            "prediction_changed_by_metadata": prediction_changed_by_metadata,
            "prediction_restored_by_guardrail": prediction_restored_by_guardrail,
            "recommendation": recommendation,
            "metadata_context": metadata,
            "is_experimental": True,
        }

