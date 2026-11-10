from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ImageValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = None
    checks: Optional[Dict[str, Any]] = None

class ImageTypeResult(BaseModel):
    image_type: str
    confidence: float
    supported: bool
    message: Optional[str] = None

class ABCDEResult(BaseModel):
    asymmetry: Optional[float] = None
    border_irregularity: Optional[float] = None
    color_h: Optional[float] = None
    color_s: Optional[float] = None
    color_v: Optional[float] = None
    color_index: Optional[float] = None
    diameter_pixels: Optional[float] = None
    diameter_mm: Optional[float] = None
    evolution: Optional[Dict[str, Any]] = None

class ViTPredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    predicted_class: Optional[str] = None
    melanoma_probability: Optional[float] = None
    non_melanoma_probability: Optional[float] = None
    decision_threshold: float = 0.80
    uncertainty_status: str = "confident"
    model_version: str = "v2.5"
    model_name: str = "ViT-Melanoma"
    backbone: str = "google/vit-base-patch16-224"

class MultimodalPredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    predicted_class: Optional[str] = None
    melanoma_probability: Optional[float] = None
    non_melanoma_probability: Optional[float] = None
    decision_threshold: float = 0.36
    uncertainty_status: str = "confident"
    model_version: str = "CEFM-Fusion-v1.0"
    model_architecture: str = "CEFMMultimodalFusion"
    status: str = "available"
    margin_to_threshold: Optional[float] = None
    risk_level: Optional[str] = None

class ChallengerPredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str = "M6"
    model_name: str = "Dual-Scale Gated Multimodal (RCAG)"
    model_version: str = "v2.0-rcag-challenger"
    model_architecture: str = "GatedDualScaleFusion"
    status: str = "experimental_challenger"
    enabled: bool = True
    predicted_class: Optional[str] = None
    melanoma_probability: Optional[float] = None
    non_melanoma_probability: Optional[float] = None
    decision_threshold: float = 0.690
    operating_point: str = "calibrated_f1"
    calibrated_f1_threshold: float = 0.690
    matched_spec_threshold: float = 0.625
    predicted_class_matched_spec: Optional[str] = None
    margin_to_threshold: Optional[float] = None
    uncertainty_status: str = "confident"
    risk_level: Optional[str] = None
    local_patch_count: int = 1
    patch_coordinates: Optional[list] = None
    note: str = "Experimental challenger model. Not used for primary clinical diagnosis."

class AnalyzeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str = "success"
    analysis_id: Optional[str] = None
    case_id: Optional[str] = None
    image_url: Optional[str] = None
    segmentation_url: Optional[str] = None
    attention_url: Optional[str] = None
    filename: Optional[str] = None
    validation: Optional[ImageValidationResult] = None
    image_type: Optional[ImageTypeResult] = None
    calibration: Optional[Dict[str, Any]] = None
    abcde: Optional[ABCDEResult] = None
    classification: Optional[Dict[str, Any]] = None
    vit_prediction: Optional[ViTPredictionResult] = None
    multimodal_prediction: Optional[MultimodalPredictionResult] = None
    challenger_prediction: Optional[ChallengerPredictionResult] = None
    evolution: Optional[Dict[str, Any]] = None
    multimodal: Optional[Dict[str, Any]] = None
    contrastive_alignment: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    rag_evidence: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
