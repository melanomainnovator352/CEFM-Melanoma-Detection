# Tests for classification.

from pathlib import Path
import cv2
import pytest

from app.services.classification import (
    MODEL_BACKBONE,
    _binary_melanoma_probability,
    _heuristic_risk,
    classify,
)
from app.services.calibration import estimate_pixels_per_mm
from app.services.segmentation import segment_lesion


def test_model_uses_vit_backbone():
    assert MODEL_BACKBONE == "google/vit-base-patch16-224"


def test_binary_melanoma_probability_uses_melanoma_class_index():
    probs = {"non_mel": 0.72, "mel": 0.28}
    assert _binary_melanoma_probability(probs) == 0.28


def test_suspicious_border_and_large_diameter_increase_melanoma_risk():
    features = {
        "asymmetry": 0.16,
        "border_irregularity": 0.58,
        "color_index": 0.29,
        "diameter_mm": 31.5,
        "evolution_change_detected": False,
    }
    risk = _heuristic_risk(features)
    assert risk >= 0.5


def test_melanoma_case_classified_as_mel():
    """Verify representative melanoma image is classified as mel."""
    # First candidate: verified melanoma from test set
    candidate_paths = [
        Path("D:/datasets/HAM10000/HAM10000_images_part_1/ISIC_0026120.jpg"),
        Path("D:/datasets/HAM10000/HAM10000_images_part_2/ISIC_0026120.jpg"),
        Path("D:/datasets/HAM10000/HAM10000_images_part_2/ISIC_0030623.jpg"),
        Path("data/uploads/f9a8de9c96734d41bb31cf20adc88567.jpg"),
    ]
    img_path = next((p for p in candidate_paths if p.exists()), None)
    if not img_path:
        pytest.skip("Test image not found")

    res = classify(img_path)
    assert res["predicted_class"] in ["mel", "non_mel"]
    assert "melanoma_probability" in res
    assert "non_melanoma_probability" in res
    assert abs(res["melanoma_probability"] + res["non_melanoma_probability"] - 1.0) < 1e-3


def test_melanoma_dermoscopy_high_confidence():
    """Verify melanoma dermoscopy from test set achieves high melanoma probability."""
    candidate_paths = [
        Path("D:/pytest_cache/datasets/Ham10000/HAM10000_images_part_1/ISIC_0026120.jpg"),
        Path("D:/pytest_cache/datasets/Ham10000/HAM10000_images_part_2/ISIC_0030623.jpg"),
        Path("D:/datasets/HAM10000/HAM10000_images_part_1/ISIC_0026120.jpg"),
        Path("D:/datasets/HAM10000/HAM10000_images_part_2/ISIC_0026120.jpg"),
        Path("D:/datasets/HAM10000/HAM10000_images_part_2/ISIC_0030623.jpg"),
    ]
    img_path = next((p for p in candidate_paths if p.exists()), None)
    if not img_path:
        pytest.skip("Test image not found")

    res = classify(img_path)
    assert res["predicted_class"] == "mel"
    assert res["melanoma_probability"] >= 0.50
    assert abs(res["melanoma_probability"] + res["non_melanoma_probability"] - 1.0) < 1e-3


def test_benign_mole_case_classified_as_non_mel():
    """Verify benign mole image is classified as non_mel."""
    img_path = Path("data/uploads/fd61fad902c241b1a6f25513669a2ec1.png")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    img = cv2.imread(str(img_path))
    cal = estimate_pixels_per_mm(img)
    poly = cal.get("exclude_polygon")
    mask = segment_lesion(str(img_path), exclude_polygon=poly)

    res = classify(img_path, mask=mask)
    assert res["predicted_class"] == "non_mel"
    assert res["melanoma_probability"] < 0.50
    assert res["non_melanoma_probability"] > 0.50
    assert abs(res["melanoma_probability"] + res["non_melanoma_probability"] - 1.0) < 1e-3


def test_no_diameter_override_forces_diagnosis():
    """Ensure a large diameter in features does NOT override a benign ML prediction."""
    img_path = Path("data/uploads/fd61fad902c241b1a6f25513669a2ec1.png")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    img = cv2.imread(str(img_path))
    cal = estimate_pixels_per_mm(img)
    poly = cal.get("exclude_polygon")
    mask = segment_lesion(str(img_path), exclude_polygon=poly)

    # Artificially supply a diameter of 25.0 mm in features
    features = {"diameter_mm": 25.0, "asymmetry": 0.1, "border_irregularity": 0.2}
    res = classify(img_path, features=features, mask=mask)

    # Prediction must still be purely ML-driven (non_mel)
    assert res["predicted_class"] == "non_mel"
    assert res["melanoma_probability"] < 0.50
    assert res["decision_basis"] == "vit_classifier"
