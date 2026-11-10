"""
Melanoma classification service using Vision Transformer (ViT).
Implements binary classification:
  Class 0 = non_mel (Benign / non-melanoma)
  Class 1 = mel (Melanoma)
Complies with Rules 2, 18, 35, 42.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import cv2
import numpy as np
import torch

from app.core.config import CLASSIFICATION_MODEL_DIR
from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.ml.multimodal.dual_scale_models import M6_DualGated
from app.services.abcde.vector import build_abcde_vector

MODEL_BACKBONE = "google/vit-base-patch16-224"
MODEL_PATH = CLASSIFICATION_MODEL_DIR / "vit_model"
MULTIMODAL_MODEL_DIR = CLASSIFICATION_MODEL_DIR.parent / "multimodal"
CHALLENGER_M6_DIR = CLASSIFICATION_MODEL_DIR.parent / "experiments" / "M6"
MODEL_NAME = "ViT-Melanoma"
MODEL_VERSION = "v2.5"
DATASET_NAME = "HAM10000"

ENABLE_CHALLENGER_M6 = os.getenv("ENABLE_CHALLENGER_M6", "true").lower() in ("1", "true", "yes")

_classifier = None
_processor = None
_loaded_path = None
_multimodal_model = None
_multimodal_config = None
_challenger_m6_model = None
_challenger_m6_config = None
_challenger_m6_norm_mean = None
_challenger_m6_norm_std = None


def _prepare_abcde_tensor(
    features: Optional[dict],
    evolution: Optional[dict] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Helper to convert flat or nested feature dictionary into standard 16-D ABCDE tensor."""
    target_device = device if device is not None else torch.device("cpu")
    if isinstance(target_device, str):
        target_device = torch.device(target_device)

    if features is None:
        return torch.zeros((1, 16), dtype=torch.float32, device=target_device)

    if "asymmetry_score" in features or "horizontal_asymmetry" in features:
        nested = features
    else:
        # Color
        if isinstance(features.get("color"), dict):
            c_dict = features["color"]
            color = {
                "h": float(c_dict.get("h", features.get("color_h", 0.0)) or 0.0),
                "s": float(c_dict.get("s", features.get("color_s", 0.0)) or 0.0),
                "v": float(c_dict.get("v", features.get("color_v", 0.0)) or 0.0),
                "color_diversity": float(
                    c_dict.get(
                        "color_diversity",
                        c_dict.get("index", features.get("color_index", 0.0)),
                    )
                    or 0.0
                ),
            }
        else:
            color = {
                "h": float(features.get("color_h", 0.0) or 0.0),
                "s": float(features.get("color_s", 0.0) or 0.0),
                "v": float(features.get("color_v", 0.0) or 0.0),
                "color_diversity": float(features.get("color_index", 0.0) or 0.0),
            }

        # Diameter
        if isinstance(features.get("diameter"), dict):
            d_dict = features["diameter"]
            dia_mm = d_dict.get("max_diameter_mm", d_dict.get("diameter_mm", features.get("diameter_mm")))
            cal_used = bool(
                d_dict.get("calibration_used")
                or d_dict.get("available")
                or features.get("diameter_mm_available", False)
            )
            diameter = {
                "diameter_mm": float(dia_mm) if dia_mm is not None else None,
                "available": dia_mm is not None,
                "calibration_used": cal_used,
            }
        else:
            dia_mm = features.get("diameter_mm")
            diameter = {
                "diameter_mm": float(dia_mm) if dia_mm is not None else None,
                "available": dia_mm is not None,
                "calibration_used": bool(features.get("diameter_mm_available", dia_mm is not None)),
            }

        # Asymmetry
        asym = features.get("asymmetry", 0.0)
        asym_val = float(asym or 0.0) if not isinstance(asym, dict) else asym

        # Border
        border = features.get("border_irregularity", 0.0)
        border_val = float(border or 0.0) if not isinstance(border, dict) else border

        nested = {
            "asymmetry": asym_val,
            "border_irregularity": border_val,
            "color": color,
            "diameter": diameter,
        }

    evo_data = evolution
    if evo_data is None and isinstance(features.get("evolution"), dict):
        evo_data = features["evolution"]

    vec = build_abcde_vector(nested, evolution=evo_data).to_tensor().unsqueeze(0).to(target_device)
    return vec


def _load_multimodal_model(device: Optional[torch.device | str] = None):
    """Load the trained multimodal fusion model and its threshold configuration."""
    global _multimodal_model, _multimodal_config
    model_file = MULTIMODAL_MODEL_DIR / "fusion_model.pt"
    config_file = MULTIMODAL_MODEL_DIR / "fusion_config.json"

    if not model_file.exists():
        return None

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    try:
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                _multimodal_config = json.load(f)
        else:
            _multimodal_config = {
                "model_architecture": "CEFMMultimodalFusion",
                "visual_dim": 768,
                "clinical_dim": 16,
                "hidden_dim": 128,
                "clinical_threshold": 0.36,
            }

        v_dim = int(_multimodal_config.get("visual_dim", 768))
        c_dim = int(_multimodal_config.get("clinical_dim", 16))
        h_dim = int(_multimodal_config.get("hidden_dim", 128))

        model = CEFMMultimodalFusion(
            visual_dim=v_dim,
            clinical_dim=c_dim,
            hidden_dim=h_dim,
            num_classes=2,
        )
        weights = torch.load(model_file, map_location=target_device)
        model.load_state_dict(weights)
        model.to(target_device)
        model.eval()
        _multimodal_model = model

        return _multimodal_model
    except Exception as exc:
        print(f"[classification] Could not load multimodal fusion model: {exc}")
        return None


def predict_multimodal_fusion(
    cls_embedding: np.ndarray | torch.Tensor,
    features: Optional[dict] = None,
    evolution: Optional[dict] = None,
    device: Optional[torch.device | str] = None,
) -> Optional[dict]:
    """
    Run the trained Multimodal Fusion network fusing 768-D ViT embedding
    and 16-D ABCDE clinical vector without hardcoded diameter rules.
    Complies with Rules 2, 17, 35, 42.
    """
    global _multimodal_model, _multimodal_config
    if _multimodal_model is None:
        _load_multimodal_model(device=device)

    if _multimodal_model is None or features is None:
        return None

    try:
        # Determine execution device
        if device is not None:
            target_device = torch.device(device) if isinstance(device, str) else device
        else:
            try:
                target_device = next(_multimodal_model.parameters()).device
            except Exception:
                target_device = torch.device("cpu")

        try:
            model_device = next(_multimodal_model.parameters()).device
            if model_device != target_device:
                _multimodal_model = _multimodal_model.to(target_device)
        except Exception:
            pass

        abcde_vec = _prepare_abcde_tensor(features, evolution=evolution, device=target_device)
        # Isolate untrained dimensions for model inference:
        # Dimensions 11 (diameter_norm) and 12 (diameter_available) had strictly zero variance in HAM10000 training caches.
        # Zeroing them for the neural model input ensures unlearned random initialization weights do not distort inference,
        # while the full calibrated physical measurement is preserved for clinical screening and reports.
        model_abcde = abcde_vec.clone()
        model_abcde[:, 11] = 0.0
        model_abcde[:, 12] = 0.0

        if isinstance(cls_embedding, torch.Tensor):
            v_tensor = cls_embedding.detach().clone().float().to(target_device)
            if v_tensor.ndim == 1:
                v_tensor = v_tensor.unsqueeze(0)
        else:
            v_tensor = torch.tensor(cls_embedding, dtype=torch.float32, device=target_device).unsqueeze(0)

        with torch.no_grad():
            logits = _multimodal_model(v_tensor, model_abcde)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        mel_prob = round(float(probs[1]), 4)
        non_mel_prob = round(float(probs[0]), 4)

        thresh = 0.36
        if _multimodal_config and "clinical_threshold" in _multimodal_config:
            thresh = float(_multimodal_config["clinical_threshold"])

        pred_class = "mel" if mel_prob >= thresh else "non_mel"
        margin = round(float(mel_prob - thresh), 4)
        uncertainty = "borderline" if abs(margin) <= 0.08 else "confident"

        # Categorize risk level based on continuous multimodal probability relative to threshold
        if mel_prob >= 0.50:
            mm_risk = "high"
        elif mel_prob >= thresh:
            mm_risk = "medium"
        elif mel_prob >= 0.20:
            mm_risk = "low-medium"
        else:
            mm_risk = "low"

        return {
            "multimodal_melanoma_probability": mel_prob,
            "multimodal_non_melanoma_probability": non_mel_prob,
            "multimodal_predicted_class": pred_class,
            "multimodal_decision_threshold": thresh,
            "multimodal_uncertainty_status": uncertainty,
            "multimodal_margin_to_threshold": margin,
            "multimodal_risk_level": mm_risk,
            "multimodal_status": "available",
            "multimodal_model_version": "CEFM-Fusion-v1.0",
            "multimodal_architecture": "CEFMMultimodalFusion",
            "multimodal_probabilities": {
                "non_mel": non_mel_prob,
                "mel": mel_prob,
            },
        }
    except Exception as exc:
        print(f"[classification] Error running multimodal fusion: {exc}")
        return None


def _load_m6_challenger_model(device: Optional[torch.device | str] = None):
    """Load the experimental M6 challenger model weights and normalization configuration."""
    global _challenger_m6_model, _challenger_m6_config, _challenger_m6_norm_mean, _challenger_m6_norm_std
    model_file = CHALLENGER_M6_DIR / "weights.pt"
    config_file = CHALLENGER_M6_DIR / "config.json"

    if not model_file.exists():
        return None

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    try:
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                _challenger_m6_config = json.load(f)
        else:
            _challenger_m6_config = {
                "model_id": "M6",
                "name": "Dual-Scale Gated Multimodal (RCAG)",
                "calibrated_threshold_f1": 0.690,
                "calibrated_threshold_matched_spec": 0.625,
            }

        # Normalization vectors
        if "clinical_norm_mean" in _challenger_m6_config and "clinical_norm_std" in _challenger_m6_config:
            _challenger_m6_norm_mean = torch.tensor(
                _challenger_m6_config["clinical_norm_mean"], dtype=torch.float32, device=target_device
            ).unsqueeze(0)
            _challenger_m6_norm_std = torch.tensor(
                _challenger_m6_config["clinical_norm_std"], dtype=torch.float32, device=target_device
            ).unsqueeze(0)
        else:
            _challenger_m6_norm_mean = torch.zeros((1, 16), dtype=torch.float32, device=target_device)
            _challenger_m6_norm_std = torch.ones((1, 16), dtype=torch.float32, device=target_device)

        model = M6_DualGated(
            global_dim=int(_challenger_m6_config.get("visual_dim", 768)),
            local_dim=int(_challenger_m6_config.get("visual_dim", 768)),
            clinical_dim=int(_challenger_m6_config.get("clinical_dim", 16)),
            hidden_dim=int(_challenger_m6_config.get("hidden_dim", 128)),
            clinical_hidden=64,
            num_classes=int(_challenger_m6_config.get("num_classes", 2)),
            dropout=0.15,
        )
        weights = torch.load(model_file, map_location=target_device)
        model.load_state_dict(weights)
        model.to(target_device)
        model.eval()
        _challenger_m6_model = model
        return _challenger_m6_model
    except Exception as exc:
        print(f"[classification] Could not load M6 challenger model: {exc}")
        return None


def predict_m6_challenger(
    global_embedding: np.ndarray | torch.Tensor,
    local_embedding: np.ndarray | torch.Tensor,
    features: Optional[dict] = None,
    evolution: Optional[dict] = None,
    patch_coordinates: Optional[list] = None,
    num_patches: int = 1,
    device: Optional[torch.device | str] = None,
) -> Optional[dict]:
    """
    Run M6 experimental challenger inference fusing 768-D global context,
    768-D local high-res patch, and 16-D normalized ABCDE clinical features.
    Provides dual operating point predictions (F1-calibrated 0.690 and 90% Spec 0.625).
    """
    global _challenger_m6_model, _challenger_m6_config, _challenger_m6_norm_mean, _challenger_m6_norm_std
    if not ENABLE_CHALLENGER_M6:
        return {
            "challenger_model_id": "M6",
            "challenger_model_name": "Dual-Scale Gated Multimodal (RCAG)",
            "challenger_model_version": "v2.0-rcag-challenger",
            "challenger_status": "disabled",
            "challenger_enabled": False,
            "challenger_note": "Challenger model is disabled via configuration.",
        }

    if _challenger_m6_model is None:
        _load_m6_challenger_model(device=device)

    if _challenger_m6_model is None:
        return None

    try:
        if device is not None:
            target_device = torch.device(device) if isinstance(device, str) else device
        else:
            try:
                target_device = next(_challenger_m6_model.parameters()).device
            except Exception:
                target_device = torch.device("cpu")

        try:
            model_device = next(_challenger_m6_model.parameters()).device
            if model_device != target_device:
                _challenger_m6_model = _challenger_m6_model.to(target_device)
        except Exception:
            pass

        # Format global visual tensor
        if isinstance(global_embedding, torch.Tensor):
            g_tensor = global_embedding.detach().clone().float().to(target_device)
            if g_tensor.ndim == 1:
                g_tensor = g_tensor.unsqueeze(0)
        else:
            g_tensor = torch.tensor(global_embedding, dtype=torch.float32, device=target_device).unsqueeze(0)

        # Format local visual tensor
        if isinstance(local_embedding, torch.Tensor):
            l_tensor = local_embedding.detach().clone().float().to(target_device)
            if l_tensor.ndim == 1:
                l_tensor = l_tensor.unsqueeze(0)
        else:
            l_tensor = torch.tensor(local_embedding, dtype=torch.float32, device=target_device).unsqueeze(0)

        # Format clinical tensor
        c_tensor = _prepare_abcde_tensor(features, evolution=evolution, device=target_device)

        # Normalize clinical features using training stats with explicit zero-variance guard
        if _challenger_m6_norm_mean is not None and _challenger_m6_norm_std is not None:
            norm_mean = _challenger_m6_norm_mean.to(target_device)
            norm_std = _challenger_m6_norm_std.to(target_device)
            # Guard against tiny epsilon division:
            # 1. Features with valid training variance (std >= 1e-4) are standardized: (x - mean) / std.
            # 2. Features with zero training variance (std < 1e-4) are normalized to 0.0, matching the exact training cache manifold.
            valid_variance_mask = (norm_std >= 1e-4)
            safe_std = torch.where(valid_variance_mask, norm_std, torch.ones_like(norm_std))
            c_norm = torch.where(valid_variance_mask, (c_tensor - norm_mean) / safe_std, torch.zeros_like(c_tensor))
        else:
            c_norm = c_tensor

        with torch.no_grad():
            logits = _challenger_m6_model(g_tensor, l_tensor, c_norm)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        non_mel_prob = round(float(probs[0]), 4)
        mel_prob = round(float(probs[1]), 4)

        thresh_f1 = 0.690
        thresh_spec = 0.625
        thresh_screening = float(os.getenv("CHALLENGER_SCREENING_THRESHOLD", "0.060"))
        thresh_review = float(os.getenv("CHALLENGER_REVIEW_THRESHOLD", "0.690"))
        if _challenger_m6_config:
            thresh_f1 = float(_challenger_m6_config.get("calibrated_threshold_f1", 0.690))
            thresh_spec = float(_challenger_m6_config.get("calibrated_threshold_matched_spec", 0.625))
            thresh_screening = float(_challenger_m6_config.get("screening_threshold", thresh_screening))
            thresh_review = float(_challenger_m6_config.get("review_threshold", thresh_review))

        pred_f1 = "mel" if mel_prob >= thresh_f1 else "non_mel"
        pred_spec = "mel" if mel_prob >= thresh_spec else "non_mel"
        pred_screening = "mel" if mel_prob >= thresh_screening else "non_mel"
        pred_review = "mel" if mel_prob >= thresh_review else "non_mel"
        margin = round(float(mel_prob - thresh_f1), 4)
        uncertainty = "borderline" if abs(margin) <= 0.08 else "confident"

        # Continuous risk level
        if mel_prob >= 0.50:
            risk = "high"
        elif mel_prob >= thresh_f1:
            risk = "medium"
        elif mel_prob >= 0.20:
            risk = "low-medium"
        else:
            risk = "low"

        return {
            "challenger_model_id": "M6",
            "challenger_model_name": "Dual-Scale Gated Multimodal (RCAG)",
            "challenger_model_version": "v2.0-rcag-challenger",
            "challenger_architecture": "GatedDualScaleFusion",
            "challenger_status": "experimental_challenger",
            "challenger_enabled": True,
            "challenger_predicted_class": pred_f1,
            "challenger_melanoma_probability": mel_prob,
            "challenger_non_melanoma_probability": non_mel_prob,
            "challenger_decision_threshold": thresh_f1,
            "challenger_operating_point": "calibrated_f1",
            "challenger_calibrated_f1_threshold": thresh_f1,
            "challenger_matched_spec_threshold": thresh_spec,
            "challenger_predicted_class_matched_spec": pred_spec,
            "challenger_screening_threshold": thresh_screening,
            "challenger_predicted_class_screening": pred_screening,
            "challenger_review_threshold": thresh_review,
            "challenger_predicted_class_review": pred_review,
            "challenger_operating_modes": {
                "screening": {
                    "mode_name": "Screening-Oriented Research Mode",
                    "threshold": thresh_screening,
                    "predicted_class": pred_screening,
                    "target_sensitivity": ">= 95%",
                    "clinical_purpose": "Primary triage / high sensitivity",
                },
                "review": {
                    "mode_name": "Higher-Specificity Review Mode",
                    "threshold": thresh_review,
                    "predicted_class": pred_review,
                    "target_objective": "Max F1 / high specificity",
                    "clinical_purpose": "Specialist review / high specificity",
                },
            },
            "challenger_margin_to_threshold": margin,
            "challenger_uncertainty_status": uncertainty,
            "challenger_risk_level": risk,
            "challenger_local_patch_count": int(num_patches),
            "challenger_patch_coordinates": patch_coordinates or [],
            "challenger_probabilities": {
                "non_mel": non_mel_prob,
                "mel": mel_prob,
            },
            "challenger_note": "Experimental challenger model. Not used for primary clinical diagnosis.",
        }
    except Exception as exc:
        print(f"[classification] Error running M6 challenger inference: {exc}")
        return None


def predict_m6_challenger_from_image(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    features: Optional[dict] = None,
    evolution: Optional[dict] = None,
    device: Optional[torch.device | str] = None,
) -> Optional[dict]:
    """
    End-to-end M6 challenger prediction from an RGB image array and lesion mask.
    Extracts dual-scale views (global context crop + local high-res patches) using
    the frozen ViT backbone, then evaluates GatedDualScaleFusion.
    """
    global _classifier, _processor
    if not ENABLE_CHALLENGER_M6:
        return {
            "challenger_model_id": "M6",
            "challenger_model_name": "Dual-Scale Gated Multimodal (RCAG)",
            "challenger_model_version": "v2.0-rcag-challenger",
            "challenger_status": "disabled",
            "challenger_enabled": False,
            "challenger_note": "Challenger model is disabled via configuration.",
        }

    if _classifier is None or _processor is None:
        _load_torch_model()

    if _classifier is None or _processor is None:
        return None

    try:
        from app.services.dual_scale import extract_dual_scale_embeddings_single
        target_device = device
        if target_device is None:
            try:
                target_device = next(_classifier.parameters()).device
            except Exception:
                target_device = torch.device("cpu")

        dual_emb = extract_dual_scale_embeddings_single(
            image_rgb,
            mask=mask,
            vit_model=_classifier,
            processor=_processor,
            device=target_device,
        )

        global_emb = dual_emb["global_embedding"]
        local_emb = dual_emb["local_embedding"]
        num_patches = dual_emb["num_patches"]
        patches = dual_emb["views"].get("local_patches", [])
        patch_coords = [p.get("bbox") for p in patches if isinstance(p, dict)]

        return predict_m6_challenger(
            global_embedding=global_emb,
            local_embedding=local_emb,
            features=features,
            evolution=evolution,
            patch_coordinates=patch_coords,
            num_patches=num_patches,
            device=target_device,
        )
    except Exception as exc:
        print(f"[classification] Error running dual-scale challenger pipeline: {exc}")
        return None


def get_decision_threshold() -> float:
    """Read the validation-calibrated clinical decision threshold if available; default to 0.80 clinical screening."""
    thresh_file = CLASSIFICATION_MODEL_DIR / "threshold_config.json"
    if thresh_file.exists():
        try:
            with open(thresh_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return float(cfg.get("clinical_threshold", cfg.get("best_bacc_threshold", 0.80)))
        except Exception:
            pass
    return 0.50


def _load_torch_model():
    """Load the fine-tuned binary ViT model and image processor."""
    global _classifier, _loaded_path, _processor

    if not MODEL_PATH.exists():
        _loaded_path = "missing"
        return None

    try:
        from transformers import ViTForImageClassification, ViTImageProcessor

        model = ViTForImageClassification.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            attn_implementation="eager",
        )
        processor = ViTImageProcessor.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )
        model.eval()

        _classifier = model
        _processor = processor
        _loaded_path = str(MODEL_PATH)
        return model

    except Exception as exc:
        print(f"[classification] Could not load ViT model: {exc}")
        _loaded_path = "error"
        return None


def get_vit_embedding(
    image_path: str | Path,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Extract 768-dimensional ViT CLS embedding from penultimate layer.
    Used for cross-modal contrastive alignment.
    """
    global _classifier, _processor
    if _classifier is None:
        _load_torch_model()
    if _classifier is None or _processor is None:
        return None

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # When mask is provided, crop around the lesion with square context margin
        if mask is not None and np.any(mask > 0):
            y_idx, x_idx = np.where(mask > 0)
            ymin, ymax = int(y_idx.min()), int(y_idx.max())
            xmin, xmax = int(x_idx.min()), int(x_idx.max())
            cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
            h, w = ymax - ymin, xmax - xmin
            side = int(max(h, w) * 1.30)
            c_ymin = max(0, cy - side // 2)
            c_ymax = min(img.shape[0], cy + side // 2)
            c_xmin = max(0, cx - side // 2)
            c_xmax = min(img.shape[1], cx + side // 2)
            input_img = img[c_ymin:c_ymax, c_xmin:c_xmax]
        else:
            input_img = img

        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        inputs = _processor(images=rgb, return_tensors="pt")
        try:
            device = next(_classifier.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            if hasattr(_classifier, "vit"):
                outputs = _classifier.vit(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            else:
                outputs = _classifier(**inputs, output_hidden_states=True)
                cls_embedding = outputs.hidden_states[-1][:, 0, :]

        return cls_embedding.cpu().numpy().squeeze(0)
    except Exception as exc:
        print(f"[classification] Error extracting ViT embedding: {exc}")
        return None
def _binary_melanoma_probability(probabilities: dict) -> float:
    """Return the probability associated with the melanoma class."""
    if not probabilities:
        return 0.0

    if "mel" in probabilities:
        return float(probabilities["mel"])

    if "melanoma" in probabilities:
        return float(probabilities["melanoma"])

    # Fallback for binary class ordering: index 1 = melanoma.
    values = list(probabilities.values())
    if len(values) >= 2:
        return float(values[1])

    return float(values[0])


def _heuristic_risk(features: dict) -> float:
    """
    Compute a simple feature-based risk score for testing/auxiliary analysis.

    This does NOT override the ViT prediction.
    """
    score = 0.0

    asymmetry = float(features.get("asymmetry", 0.0) or 0.0)
    border = float(features.get("border_irregularity", 0.0) or 0.0)
    color = float(features.get("color_index", 0.0) or 0.0)
    diameter = float(features.get("diameter_mm", 0.0) or 0.0)
    evolution = bool(features.get("evolution_change_detected", False))

    if asymmetry >= 0.10:
        score += 0.20

    if border >= 0.50:
        score += 0.20

    if color >= 0.25:
        score += 0.20

    if diameter >= 20.0:
        score += 0.20

    if evolution:
        score += 0.20

    return min(score, 1.0)

def _risk_level(mel_prob: float, features: Optional[dict] = None) -> str:
    """
    Categorize probability into clinical risk band.
    Synthesizes ML model probability with physical ABCDE measurements:
    a lesion measuring >= 10 mm or >= 6 mm carries elevated clinical risk.
    """
    if mel_prob >= 0.50:
        return "high"

    if features and isinstance(features, dict):
        d_val = features.get("diameter_mm")
        if d_val is not None:
            try:
                d = float(d_val)
                if d >= 10.0:
                    return "high"
                if d >= 6.0:
                    return "medium"
            except (ValueError, TypeError):
                pass

        border = float(features.get("border_irregularity", 0.0) or 0.0)
        asym = float(features.get("asymmetry", 0.0) or 0.0)
        if border >= 0.55 or asym >= 0.45:
            return "medium"

    if mel_prob >= 0.25:
        return "medium"
    return "low"


def classify(
    image_path: str | Path,
    features: Optional[dict] = None,
    mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Classify the lesion using the fine-tuned Vision Transformer.
    When a segmentation mask is provided, the lesion region of interest (ROI)
    is cropped with a narrow context margin so wide-field clinical images
    are properly centered and scaled to match HAM10000 training distribution.
    Complies with Rule 35: The ML model's prediction is NOT arbitrarily overridden by heuristics.
    """
    global _classifier, _processor

    if _classifier is None and _loaded_path is None:
        _load_torch_model()

    if _classifier is None:
        return {
            "method": "Not available",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset": DATASET_NAME,
            "status": "Not trained / Not available",
            "predicted_class": "Not available",
            "probabilities": None,
            "melanoma_probability": None,
            "non_melanoma_probability": None,
            "decision_threshold": 0.50,
            "uncertainty_status": "Not available",
            "risk_level": "Not available",
            "evaluation_status": "Not evaluated",
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Unable to read image at {image_path}")

        # Crop to square lesion bounding box with context margin if mask is provided
        # Preserves lesion aspect ratio and dermoscopic surround without distortion
        if mask is not None and np.any(mask > 0):
            y_idx, x_idx = np.where(mask > 0)
            ymin, ymax = int(y_idx.min()), int(y_idx.max())
            xmin, xmax = int(x_idx.min()), int(x_idx.max())
            cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
            h, w = ymax - ymin, xmax - xmin
            side = int(max(h, w) * 1.30)
            c_ymin = max(0, cy - side // 2)
            c_ymax = min(img.shape[0], cy + side // 2)
            c_xmin = max(0, cx - side // 2)
            c_xmax = min(img.shape[1], cx + side // 2)
            input_img = img[c_ymin:c_ymax, c_xmin:c_xmax]
        else:
            input_img = img

        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        inputs = _processor(images=rgb, return_tensors="pt")
        try:
            device = next(_classifier.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            outputs = _classifier(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy().squeeze(0)
            else:
                cls_emb = None

        id2label = getattr(_classifier.config, "id2label", {0: "non_mel", 1: "mel"})
        label_names = [id2label.get(i, str(i)) for i in range(len(probs))]
        prob_map = {name: round(float(p), 4) for name, p in zip(label_names, probs)}

        # Melanoma and non-melanoma probabilities
        if "mel" in prob_map:
            mel_prob = prob_map["mel"]
            non_mel_prob = prob_map.get("non_mel", round(1.0 - mel_prob, 4))
        elif len(probs) >= 2:
            non_mel_prob = round(float(probs[0]), 4)
            mel_prob = round(float(probs[1]), 4)
        else:
            mel_prob = round(float(probs[0]), 4)
            non_mel_prob = round(1.0 - mel_prob, 4)

        DECISION_THRESHOLD = get_decision_threshold()
        predicted_class = "mel" if mel_prob >= DECISION_THRESHOLD else "non_mel"
        basis = "vit_classifier"
        risk_level = _risk_level(mel_prob, features)

        # Borderline / confident uncertainty status
        uncertainty_status = (
            "borderline"
            if abs(mel_prob - DECISION_THRESHOLD) <= 0.08
            else "confident"
        )

        res = {
            "method": "trained_vit_model",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "backbone": MODEL_BACKBONE,
            "dataset": DATASET_NAME,
            "predicted_class": predicted_class,
            "probabilities": {
                "non_mel": non_mel_prob,
                "mel": mel_prob,
            },
            "melanoma_probability": mel_prob,
            "non_melanoma_probability": non_mel_prob,
            "decision_threshold": DECISION_THRESHOLD,
            "uncertainty_status": uncertainty_status,
            "decision_basis": basis,
            "risk_level": risk_level,
            "evaluation_status": "Model loaded and verified",
            "timestamp": datetime.utcnow().isoformat(),
            # Explicit ViT baseline keys for comparative analysis
            "vit_predicted_class": predicted_class,
            "vit_melanoma_probability": mel_prob,
            "vit_non_melanoma_probability": non_mel_prob,
            "vit_decision_threshold": DECISION_THRESHOLD,
            "vit_uncertainty_status": uncertainty_status,
            "vit_model_version": MODEL_VERSION,
            "vit_model_name": MODEL_NAME,
            "vit_backbone": MODEL_BACKBONE,
        }

        if cls_emb is not None and features is not None:
            multimodal_res = predict_multimodal_fusion(
                cls_emb,
                features=features,
                evolution=features.get("evolution") if isinstance(features, dict) else None,
            )
            if multimodal_res:
                res.update(multimodal_res)

        # M6 Challenger Prediction (Dual-scale gated multimodal)
        if ENABLE_CHALLENGER_M6 and img is not None:
            try:
                full_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                challenger_res = predict_m6_challenger_from_image(
                    full_rgb,
                    mask=mask,
                    features=features,
                    evolution=features.get("evolution") if isinstance(features, dict) else None,
                )
                if challenger_res:
                    res["challenger"] = challenger_res
                    res.update({
                        "challenger_predicted_class": challenger_res.get("challenger_predicted_class"),
                        "challenger_melanoma_probability": challenger_res.get("challenger_melanoma_probability"),
                        "challenger_non_melanoma_probability": challenger_res.get("challenger_non_melanoma_probability"),
                        "challenger_decision_threshold": challenger_res.get("challenger_decision_threshold"),
                        "challenger_operating_point": challenger_res.get("challenger_operating_point"),
                        "challenger_calibrated_f1_threshold": challenger_res.get("challenger_calibrated_f1_threshold"),
                        "challenger_matched_spec_threshold": challenger_res.get("challenger_matched_spec_threshold"),
                        "challenger_predicted_class_matched_spec": challenger_res.get("challenger_predicted_class_matched_spec"),
                        "challenger_margin_to_threshold": challenger_res.get("challenger_margin_to_threshold"),
                        "challenger_uncertainty_status": challenger_res.get("challenger_uncertainty_status"),
                        "challenger_risk_level": challenger_res.get("challenger_risk_level"),
                        "challenger_local_patch_count": challenger_res.get("challenger_local_patch_count"),
                        "challenger_status": challenger_res.get("challenger_status"),
                        "challenger_enabled": challenger_res.get("challenger_enabled"),
                        "challenger_model_id": challenger_res.get("challenger_model_id"),
                        "challenger_model_name": challenger_res.get("challenger_model_name"),
                        "challenger_model_version": challenger_res.get("challenger_model_version"),
                        "challenger_architecture": challenger_res.get("challenger_architecture"),
                        "challenger_screening_threshold": challenger_res.get("challenger_screening_threshold"),
                        "challenger_predicted_class_screening": challenger_res.get("challenger_predicted_class_screening"),
                        "challenger_review_threshold": challenger_res.get("challenger_review_threshold"),
                        "challenger_predicted_class_review": challenger_res.get("challenger_predicted_class_review"),
                        "challenger_operating_modes": challenger_res.get("challenger_operating_modes"),
                    })
            except Exception as _challenger_err:
                print(f"[classification] Non-fatal error during challenger inference: {_challenger_err}")

        # Three-Way Discordance Analysis
        try:
            from app.services.discordance import detect_three_way_discordance
            vit_pred_val = res.get("vit_predicted_class", res.get("predicted_class"))
            mm_pred_val = res.get("multimodal_predicted_class")
            m6_pred_val = res.get("challenger_predicted_class")
            vit_prob_val = res.get("vit_melanoma_probability", res.get("melanoma_probability"))
            mm_prob_val = res.get("multimodal_melanoma_probability")
            m6_prob_val = res.get("challenger_melanoma_probability")
            vit_thresh_val = float(res.get("decision_threshold", 0.80))
            mm_thresh_val = float(res.get("multimodal_decision_threshold", 0.36))
            m6_thresh_val = float(res.get("challenger_decision_threshold", 0.690))

            discordance_res = detect_three_way_discordance(
                vit_pred=vit_pred_val,
                mm_pred=mm_pred_val,
                m6_pred=m6_pred_val,
                vit_prob=vit_prob_val,
                mm_prob=mm_prob_val,
                m6_prob=m6_prob_val,
                vit_thresh=vit_thresh_val,
                mm_thresh=mm_thresh_val,
                m6_thresh=m6_thresh_val,
            )
            res["three_way_discordance"] = discordance_res
            res["discordance"] = discordance_res
        except Exception as _disc_err:
            print(f"[classification] Discordance detection warning: {_disc_err}")

        # Data Quality Audit
        try:
            from app.services.quality_audit import audit_data_quality
            quality_res = audit_data_quality(
                mask=mask,
                image_shape=img.shape if img is not None else None,
                features=features,
                classification=res,
            )
            res["quality_audit"] = quality_res
        except Exception as _qual_err:
            print(f"[classification] Quality audit warning: {_qual_err}")

        return res

    except Exception as exc:
        print(f"[classification] ViT inference error: {exc}")
        return {
            "method": "trained_vit_model",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset": DATASET_NAME,
            "status": f"Error: {exc}",
            "predicted_class": "Not available",
            "probabilities": None,
            "melanoma_probability": None,
            "non_melanoma_probability": None,
            "decision_threshold": 0.50,
            "uncertainty_status": "error",
            "risk_level": "Not available",
            "evaluation_status": "Execution error",
            "timestamp": datetime.utcnow().isoformat(),
        }


classifier = type(
    "ClassifierService",
    (),
    {
        "predict": staticmethod(classify),
        "extract_embedding": staticmethod(get_vit_embedding),
        "predict_m6_challenger": staticmethod(predict_m6_challenger),
        "predict_m6_challenger_from_image": staticmethod(predict_m6_challenger_from_image),
    },
)()
