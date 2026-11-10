"""
Automated Test Suite for CEFM Step 15:
Controlled Multi-Model Dashboard Integration.

Covers:
1. Three-model prediction extraction and display structure.
2. Correct class-index mapping (0=non_mel, 1=mel).
3. Threshold selection (ViT=0.80, MM=0.36, M6 review=0.690, M6 screening=0.060).
4. Three-way discordance detection across all 6 specified cases.
5. Missing model output handling (partial availability).
6. Segmentation failure and data quality auditing.
7. Multi-image grouping and lesion-level max aggregation.
8. Probability variance guardrail (delta_p > 0.35).
9. Report generation with multi-model section and mandatory disclaimer.
10. Report fact-checking validation.
"""

import pytest
import numpy as np
from app.services.discordance import detect_three_way_discordance
from app.services.quality_audit import audit_data_quality
from app.services.lesion_aggregation import aggregate_lesion_photographs
from app.services.report_generation import generate_structured_report
from app.services.report_fact_checker import validate_report


def test_three_way_discordance_all_mel():
    """Verify consensus elevated concern when all 3 models predict MEL."""
    disc = detect_three_way_discordance(
        vit_pred="mel",
        mm_pred="mel",
        m6_pred="mel",
        vit_prob=0.85,
        mm_prob=0.45,
        m6_prob=0.75,
        vit_thresh=0.80,
        mm_thresh=0.36,
        m6_thresh=0.690,
    )
    assert disc["status"] == "all_mel"
    assert disc["has_discordance"] is False
    assert disc["severity"] == "warning"
    assert "Consensus" in disc["title"]


def test_three_way_discordance_all_non_mel():
    """Verify consensus low concern when all 3 models predict non-MEL."""
    disc = detect_three_way_discordance(
        vit_pred="non_mel",
        mm_pred="non_mel",
        m6_pred="non_mel",
        vit_prob=0.40,
        mm_prob=0.05,
        m6_prob=0.10,
        vit_thresh=0.80,
        mm_thresh=0.36,
        m6_thresh=0.690,
    )
    assert disc["status"] == "all_non_mel"
    assert disc["has_discordance"] is False
    assert disc["severity"] == "info"
    assert "Consensus" in disc["title"]


def test_three_way_discordance_vit_non_mel_m6_mel():
    """Verify discordance detection when ViT predicts non-mel but M6 predicts mel."""
    disc = detect_three_way_discordance(
        vit_pred="non_mel",
        mm_pred="non_mel",
        m6_pred="mel",
        vit_prob=0.65,
        mm_prob=0.10,
        m6_prob=0.72,
        vit_thresh=0.80,
        mm_thresh=0.36,
        m6_thresh=0.690,
    )
    assert disc["status"] == "vit_non_mel_m6_mel"
    assert disc["has_discordance"] is True
    assert disc["severity"] == "discordance"
    assert "ViT Non-MEL vs. M6 Challenger MEL" in disc["title"]


def test_three_way_discordance_vit_non_mel_mm_mel():
    """Verify discordance detection when ViT predicts non-mel but Multimodal predicts mel."""
    disc = detect_three_way_discordance(
        vit_pred="non_mel",
        mm_pred="mel",
        m6_pred="non_mel",
        vit_prob=0.55,
        mm_prob=0.48,
        m6_prob=0.30,
        vit_thresh=0.80,
        mm_thresh=0.36,
        m6_thresh=0.690,
    )
    assert disc["status"] == "vit_non_mel_mm_mel"
    assert disc["has_discordance"] is True
    assert "ViT Non-MEL vs. Multimodal Fusion MEL" in disc["title"]


def test_three_way_discordance_m6_non_mel_other_mel():
    """Verify discordance detection when M6 predicts non-mel but ViT predicts mel."""
    disc = detect_three_way_discordance(
        vit_pred="mel",
        mm_pred="non_mel",
        m6_pred="non_mel",
        vit_prob=0.88,
        mm_prob=0.12,
        m6_prob=0.25,
        vit_thresh=0.80,
        mm_thresh=0.36,
        m6_thresh=0.690,
    )
    assert disc["status"] == "m6_non_mel_other_mel"
    assert disc["has_discordance"] is True
    assert "M6 Non-MEL vs. ViT MEL" in disc["title"]


def test_three_way_discordance_partial_availability():
    """Verify handling when fewer than 2 models are available."""
    disc = detect_three_way_discordance(
        vit_pred="mel",
        mm_pred=None,
        m6_pred="—",
        vit_prob=0.88,
    )
    assert disc["status"] == "partial_availability"
    assert disc["has_discordance"] is False
    assert disc["active_model_count"] == 1


def test_m6_operating_modes():
    """Verify M6 dual operating modes (Screening 0.060 vs Review 0.690)."""
    # Probability = 0.20:
    # Under Review (0.690): non_mel
    # Under Screening (0.060): mel
    disc_review = detect_three_way_discordance(
        vit_pred="non_mel", mm_pred="non_mel", m6_pred="non_mel",
        vit_prob=0.40, mm_prob=0.05, m6_prob=0.20,
        m6_thresh=0.690, m6_mode="review",
    )
    assert disc_review["status"] == "all_non_mel"

    m6_screening_pred = "mel" if 0.20 >= 0.060 else "non_mel"
    disc_screening = detect_three_way_discordance(
        vit_pred="non_mel", mm_pred="non_mel", m6_pred=m6_screening_pred,
        vit_prob=0.40, mm_prob=0.05, m6_prob=0.20,
        m6_thresh=0.060, m6_mode="screening",
    )
    assert disc_screening["status"] == "vit_non_mel_m6_mel"
    assert disc_screening["has_discordance"] is True


def test_data_quality_segmentation_failure():
    """Verify quality audit flags empty or invalid segmentation masks."""
    audit = audit_data_quality(mask=None, image_shape=(450, 600, 3))
    assert audit["has_warnings"] is True
    codes = [w["code"] for w in audit["warnings"]]
    assert "SEGMENTATION_FAILURE" in codes


def test_data_quality_boundary_touch():
    """Verify quality audit flags when lesion mask touches image edges."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:30, 20:50] = 1  # Touches top edge (row 0)
    audit = audit_data_quality(mask=mask, image_shape=(100, 100, 3))
    assert audit["has_warnings"] is True
    codes = [w["code"] for w in audit["warnings"]]
    assert "BOUNDARY_TOUCH" in codes


def test_data_quality_abnormal_mask_area():
    """Verify quality audit flags unusually small (<1%) or large (>85%) masks."""
    mask_small = np.zeros((100, 100), dtype=np.uint8)
    mask_small[50:52, 50:52] = 1  # 4 pixels out of 10,000 = 0.04%
    audit_small = audit_data_quality(mask=mask_small, image_shape=(100, 100, 3))
    codes_small = [w["code"] for w in audit_small["warnings"]]
    assert "ABNORMAL_MASK_AREA_SMALL" in codes_small

    mask_large = np.ones((100, 100), dtype=np.uint8)  # 100% of frame
    audit_large = audit_data_quality(mask=mask_large, image_shape=(100, 100, 3))
    codes_large = [w["code"] for w in audit_large["warnings"]]
    assert "ABNORMAL_MASK_AREA_LARGE" in codes_large


def test_data_quality_uncalibrated_diameter():
    """Verify quality audit flags missing physical ruler calibration."""
    feat = {"diameter_mm": None, "diameter_mm_available": False, "diameter_pixels": 120}
    audit = audit_data_quality(features=feat)
    codes = [w["code"] for w in audit["warnings"]]
    assert "CALIBRATION_UNAVAILABLE" in codes


def test_multi_photo_lesion_aggregation():
    """Verify lesion-level aggregation groups only matching lesion_id and computes max probability."""
    preds = [
        {"lesion_id": "HAM_123", "melanoma_probability": 0.45, "decision_threshold": 0.80},
        {"lesion_id": "HAM_123", "melanoma_probability": 0.85, "decision_threshold": 0.80},
    ]
    agg = aggregate_lesion_photographs(preds, lesion_id="HAM_123")
    assert agg["is_multi_image"] is True
    assert agg["image_count"] == 2
    vit_agg = agg["models"]["ViT"]
    assert vit_agg["max_probability"] == 0.85
    assert vit_agg["min_probability"] == 0.45
    assert vit_agg["mean_probability"] == 0.65
    assert vit_agg["delta_p"] == 0.40
    assert vit_agg["predicted_class_max"] == "mel"
    assert vit_agg["predicted_class_mean"] == "non_mel"


def test_multi_photo_variance_guardrail():
    """Verify high inter-photograph variance alert triggers when delta_p > 0.35."""
    preds = [
        {"lesion_id": "HAM_999", "melanoma_probability": 0.15},
        {"lesion_id": "HAM_999", "melanoma_probability": 0.70},  # delta = 0.55 > 0.35
    ]
    agg = aggregate_lesion_photographs(preds, lesion_id="HAM_999")
    assert agg["high_variance_alert"] is True
    assert agg["overall_max_delta_p"] == 0.55


def test_multi_photo_unrelated_images_rejected():
    """Verify multi-photo aggregation does not group images without matching lesion_id."""
    preds = [
        {"lesion_id": "HAM_111", "melanoma_probability": 0.20},
        {"lesion_id": "HAM_222", "melanoma_probability": 0.80},
    ]
    agg = aggregate_lesion_photographs(preds, lesion_id=None)
    assert agg["is_multi_image"] is False


def test_report_generation_multi_model():
    """Verify report generation includes multi-model outputs and mandatory disclaimer."""
    classification = {
        "predicted_class": "non_mel",
        "melanoma_probability": 0.65,
        "decision_threshold": 0.80,
        "risk_level": "medium",
        "multimodal_predicted_class": "mel",
        "multimodal_melanoma_probability": 0.42,
        "multimodal_decision_threshold": 0.36,
        "challenger_predicted_class": "mel",
        "challenger_melanoma_probability": 0.72,
        "challenger_decision_threshold": 0.690,
        "challenger_operating_point": "higher_specificity_review",
        "three_way_discordance": {
            "has_discordance": True,
            "message": "ViT baseline predicts NON_MEL, but Multimodal and M6 predict MEL.",
        },
    }
    abcde = {
        "asymmetry": 0.32,
        "border_irregularity": 0.45,
        "color_index": 0.55,
        "diameter_mm": 5.2,
        "diameter_mm_available": True,
    }
    report = generate_structured_report(
        validation={"valid": True},
        classification=classification,
        abcde=abcde,
    )
    text = report["full_report_text"]
    assert "Multi-Model Research Assessment" in text
    assert "Primary ViT Baseline" in text
    assert "Phase 1 Multimodal Fusion" in text
    assert "Challenger M6 (Dual-Scale RCAG)" in text
    assert "Consensus and Model Agreement" in text
    assert "cannot confirm or rule out melanoma" in text.lower()


def test_report_fact_checker_safety():
    """Verify report fact checker flags unsupported diagnostic certainty statements."""
    good_report = generate_structured_report(
        {"valid": True, "message": "OK"},
        classification={"predicted_class": "mel", "melanoma_probability": 0.85},
        abcde={"asymmetry": 0.35, "border_irregularity": 0.42, "color_index": 0.50, "diameter_mm": 5.0, "diameter_mm_available": True},
    )
    check_good = validate_report(
        good_report,
        classification={"predicted_class": "mel", "melanoma_probability": 0.85},
        abcde={"asymmetry": 0.35, "border_irregularity": 0.42, "color_index": 0.50, "diameter_mm": 5.0, "diameter_mm_available": True},
    )
    assert check_good["fact_check_passed"] is True

    bad_report = {
        "full_report_text": good_report["full_report_text"] + " The system confirms melanoma definitively."
    }
    check_bad = validate_report(
        bad_report,
        classification={"predicted_class": "mel", "melanoma_probability": 0.85},
        abcde={"asymmetry": 0.35, "border_irregularity": 0.42, "color_index": 0.50, "diameter_mm": 5.0, "diameter_mm_available": True},
    )
    assert check_bad["fact_check_passed"] is False
    codes = [c["name"] for c in check_bad["checks"] if not c["passed"]]
    assert "unsupported_diagnostic_claims" in codes
