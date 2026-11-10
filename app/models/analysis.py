# Analysis model (MongoDB-backed).
# Lightweight Python dataclass placeholder matching the MongoDB document schema.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AnalysisRecord:
    """One melanoma analysis tied to a Case (mirrors the MongoDB 'analyses' document)."""
    _id: str = ""
    case_id: str = ""
    original_filename: Optional[str] = None
    stored_path: Optional[str] = None
    segmentation_path: Optional[str] = None
    is_valid: bool = True
    image_type: Optional[dict] = None

    # Classification
    predicted_class: Optional[str] = None
    melanoma_probability: Optional[float] = None
    non_melanoma_probability: Optional[float] = None
    decision_threshold: Optional[float] = None
    uncertainty_status: Optional[str] = None
    risk_level: Optional[str] = None

    # ABCDE
    asymmetry: Optional[float] = None
    border_irregularity: Optional[float] = None
    color_h: Optional[float] = None
    color_s: Optional[float] = None
    color_v: Optional[float] = None
    color_index: Optional[float] = None
    diameter_pixels: Optional[float] = None
    diameter_mm: Optional[float] = None

    # Evolution
    evolution_available: bool = False
    evolution_change_detected: bool = False
    evolution: Optional[dict] = None

    # Multimodal / RAG / Report
    multimodal: Optional[dict] = None
    contrastive_alignment: Optional[dict] = None
    rag_evidence: Optional[dict] = None
    explanation: Optional[dict] = None
    report: Optional[dict] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
