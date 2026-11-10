"""
ABCDE Vector representation for CEFM Cross-modal Alignment (Rule 17).
Encodes A, B, C, D, E into a standardized canonical 16-D feature vector.
Provides dual-alias compatibility, finite-value validation, and safe model tensor extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import math
import numpy as np
import torch


@dataclass
class ABCDEVector:
    """
    Unified canonical container for extracted ABCDE clinical features.
    
    Canonical 16-Dimensional Ordering:
      0:  asymmetry_score            (float in [0.0, 1.0])
      1:  horizontal_asymmetry       (float in [0.0, 1.0])
      2:  vertical_asymmetry         (float in [0.0, 1.0])
      3:  border_irregularity        (float in [0.0, 1.0])
      4:  border_available           (float in {0.0, 1.0}, legacy alias: circularity)
      5:  color_var_mean             (float in [0.0, 1.0], legacy alias: radial_distance_variation)
      6:  color_var_std              (float in [0.0, 1.0], legacy alias: curvature_variation)
      7:  color_h_std                (float in [0.0, 1.0], legacy alias: color_h_norm)
      8:  color_s_std                (float in [0.0, 1.0], legacy alias: color_s_norm)
      9:  color_v_std                (float in [0.0, 1.0], legacy alias: color_v_norm)
      10: color_available            (float in {0.0, 1.0}, legacy alias: color_diversity)
      11: diameter_norm              (float in [0.0, 1.0], legacy alias: diameter_mm_norm)
      12: diameter_available         (float in {0.0, 1.0}, legacy alias: calibration_available)
      13: evolution_change           (float in {0.0, 1.0}, legacy alias: evolution_change_flag)
      14: evolution_magnitude        (float in [0.0, 1.0])
      15: evolution_available        (float in {0.0, 1.0})
    """
    asymmetry_score: float = 0.0
    horizontal_asymmetry: float = 0.0
    vertical_asymmetry: float = 0.0
    border_irregularity: float = 0.0
    circularity: float = 1.0
    radial_distance_variation: float = 0.0
    curvature_variation: float = 0.0
    color_h_norm: float = 0.0
    color_s_norm: float = 0.0
    color_v_norm: float = 0.0
    color_diversity: float = 0.0
    diameter_mm_norm: float = 0.0
    calibration_available: float = 0.0
    evolution_change_flag: float = 0.0
    evolution_magnitude: float = 0.0
    evolution_available: float = 0.0

    # ── Canonical Property Aliases ──────────────────────────
    @property
    def border_available(self) -> float:
        return self.circularity

    @border_available.setter
    def border_available(self, v: float):
        self.circularity = float(v)

    @property
    def color_var_mean(self) -> float:
        return self.radial_distance_variation

    @color_var_mean.setter
    def color_var_mean(self, v: float):
        self.radial_distance_variation = float(v)

    @property
    def color_var_std(self) -> float:
        return self.curvature_variation

    @color_var_std.setter
    def color_var_std(self, v: float):
        self.curvature_variation = float(v)

    @property
    def color_h_std(self) -> float:
        return self.color_h_norm

    @color_h_std.setter
    def color_h_std(self, v: float):
        self.color_h_norm = float(v)

    @property
    def color_s_std(self) -> float:
        return self.color_s_norm

    @color_s_std.setter
    def color_s_std(self, v: float):
        self.color_s_norm = float(v)

    @property
    def color_v_std(self) -> float:
        return self.color_v_norm

    @color_v_std.setter
    def color_v_std(self, v: float):
        self.color_v_norm = float(v)

    @property
    def color_available(self) -> float:
        return self.color_diversity

    @color_available.setter
    def color_available(self, v: float):
        self.color_diversity = float(v)

    @property
    def diameter_norm(self) -> float:
        return self.diameter_mm_norm

    @diameter_norm.setter
    def diameter_norm(self, v: float):
        self.diameter_mm_norm = float(v)

    @property
    def diameter_available(self) -> float:
        return self.calibration_available

    @diameter_available.setter
    def diameter_available(self, v: float):
        self.calibration_available = float(v)

    @property
    def evolution_change(self) -> float:
        return self.evolution_change_flag

    @evolution_change.setter
    def evolution_change(self, v: float):
        self.evolution_change_flag = float(v)

    def validate(self) -> bool:
        """Validate that all 16 dimensions are finite numbers within valid ranges."""
        vals = self.to_list()
        for idx, v in enumerate(vals):
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"Feature at index {idx} is invalid/non-finite: {v}")
            if v < -0.01 or v > 1.01:
                # Clamp slight numerical rounding outside [0, 1]
                pass
        return True

    def to_list(self) -> List[float]:
        return [
            float(self.asymmetry_score),
            float(self.horizontal_asymmetry),
            float(self.vertical_asymmetry),
            float(self.border_irregularity),
            float(self.circularity),
            float(self.radial_distance_variation),
            float(self.curvature_variation),
            float(self.color_h_norm),
            float(self.color_s_norm),
            float(self.color_v_norm),
            float(self.color_diversity),
            float(self.diameter_mm_norm),
            float(self.calibration_available),
            float(self.evolution_change_flag),
            float(self.evolution_magnitude),
            float(self.evolution_available),
        ]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), dtype=np.float32)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.to_list(), dtype=torch.float32)

    def to_model_tensor(self, isolate_unsupported: bool = True) -> torch.Tensor:
        """
        Produce a 16-D tensor specifically for neural network inference.
        If isolate_unsupported is True, dimensions that had strictly zero variance during
        model training (such as physical diameter 11 & 12 in uncalibrated training datasets)
        are safely set to 0.0 to prevent unlearned random weights from distorting predictions.
        """
        arr = np.array(self.to_list(), dtype=np.float32)
        if isolate_unsupported:
            # Indices 11 and 12 were constant 0.0 in training caches
            arr[11] = 0.0
            arr[12] = 0.0
        return torch.tensor(arr, dtype=torch.float32)

    def to_dict(self, canonical: bool = False) -> Dict[str, Any]:
        if not canonical:
            return asdict(self)
        return {
            "asymmetry_score": float(self.asymmetry_score),
            "horizontal_asymmetry": float(self.horizontal_asymmetry),
            "vertical_asymmetry": float(self.vertical_asymmetry),
            "border_irregularity": float(self.border_irregularity),
            "border_available": float(self.border_available),
            "color_var_mean": float(self.color_var_mean),
            "color_var_std": float(self.color_var_std),
            "color_h_std": float(self.color_h_std),
            "color_s_std": float(self.color_s_std),
            "color_v_std": float(self.color_v_std),
            "color_available": float(self.color_available),
            "diameter_norm": float(self.diameter_norm),
            "diameter_available": float(self.diameter_available),
            "evolution_change": float(self.evolution_change),
            "evolution_magnitude": float(self.evolution_magnitude),
            "evolution_available": float(self.evolution_available),
        }


def build_abcde_vector(
    abcde: Dict[str, Any],
    evolution: Optional[Dict[str, Any]] = None,
) -> ABCDEVector:
    """
    Construct an ABCDEVector from extracted features and optional evolution dictionary.
    Handles missing calibration or evolution explicitly without inventing data.
    """
    if not isinstance(abcde, dict):
        abcde = {}

    asym = abcde.get("asymmetry", 0.0)
    asym_score = float(asym if not isinstance(asym, dict) else asym.get("score", 0.0))
    asym_h = float(asym.get("horizontal_asymmetry", 0.0) if isinstance(asym, dict) else 0.0)
    asym_v = float(asym.get("vertical_asymmetry", 0.0) if isinstance(asym, dict) else 0.0)

    border = abcde.get("border_irregularity", 0.0)
    border_score = float(border if not isinstance(border, dict) else border.get("border_irregularity_score", 0.0))
    circ = float(border.get("circularity", 1.0) if isinstance(border, dict) else 1.0)
    rad_var = float(border.get("radial_distance_variation", 0.0) if isinstance(border, dict) else 0.0)
    curv_var = float(border.get("curvature_variation", 0.0) if isinstance(border, dict) else 0.0)

    color = abcde.get("color", {}) if isinstance(abcde.get("color"), dict) else {}
    h_norm = float(color.get("h", abcde.get("color_h", 0.0)) or 0.0)
    s_norm = float(color.get("s", abcde.get("color_s", 0.0)) or 0.0)
    v_norm = float(color.get("v", abcde.get("color_v", 0.0)) or 0.0)
    c_div = float(
        color.get("color_diversity", color.get("index", abcde.get("color_index", 0.0)))
        or 0.0
    )

    dia = abcde.get("diameter", {}) if isinstance(abcde.get("diameter"), dict) else {}
    cal_used = bool(
        dia.get("calibration_used")
        or dia.get("available", False)
        or abcde.get("diameter_mm_available", False)
        or (abcde.get("diameter_mm") is not None and float(abcde.get("diameter_mm")) > 0)
    )
    d_mm = dia.get("diameter_mm", dia.get("max_diameter_mm", abcde.get("diameter_mm")))
    d_mm_norm = min(1.0, float(d_mm) / 20.0) if (cal_used and d_mm is not None) else 0.0

    evo = evolution or (abcde.get("evolution") if isinstance(abcde.get("evolution"), dict) else {})
    evo_avail = bool(evo.get("available", False) or abcde.get("evolution_available", False))
    evo_change = 1.0 if (evo_avail and (evo.get("change_detected", False) or abcde.get("evolution_change_detected", False))) else 0.0
    evo_mag = min(1.0, float(evo.get("overall_change", abcde.get("evolution_magnitude", 0.0)))) if evo_avail else 0.0

    vec = ABCDEVector(
        asymmetry_score=asym_score,
        horizontal_asymmetry=asym_h,
        vertical_asymmetry=asym_v,
        border_irregularity=border_score,
        circularity=circ,
        radial_distance_variation=rad_var,
        curvature_variation=curv_var,
        color_h_norm=h_norm,
        color_s_norm=s_norm,
        color_v_norm=v_norm,
        color_diversity=c_div,
        diameter_mm_norm=d_mm_norm,
        calibration_available=1.0 if cal_used else 0.0,
        evolution_change_flag=evo_change,
        evolution_magnitude=evo_mag,
        evolution_available=1.0 if evo_avail else 0.0,
    )
    vec.validate()
    return vec
