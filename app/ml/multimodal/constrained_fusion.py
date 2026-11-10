"""
CEFM Step 18: Constrained Clinical Metadata Fusion Architectures.

Implements fusion architectures with mathematical Visual Evidence Preservation (VEP):
1. ConstrainedResidualFusionModel (Alternative C):
   - Visual branch outputs base logits [l_0_vis, l_1_vis].
   - Clinical branch outputs a non-negative scalar melanoma logit boost:
       Delta_l1 = Softplus(Linear(64 -> 1)(z_clinical)) >= 0
   - Fused logits: l_1 = l_1_vis + Delta_l1, l_0 = l_0_vis.
   - Guarantees p_fusion(mel) >= p_visual(mel) strictly for all inputs.
   - Impossible for demographic metadata to subtract from visual suspicion.

2. GatedPositiveFusionModel (Alternative D):
   - Gated positive-only contribution modulated by visual representation:
       gate = Sigmoid(Linear(128 -> 1)(z_vis)) in [0, 1]
       Delta_l1 = gate * Softplus(Linear(64 -> 1)(z_clinical)) >= 0
   - Fused logits: l_1 = l_1_vis + Delta_l1, l_0 = l_0_vis.
   - Preserves non-negative property while allowing visual context to modulate clinical boost.

3. VisualEvidencePreserver:
   - Standalone evaluator computing visual probability, raw fusion probability,
     and guardrail-arbitrated probability with formal violation tracking.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.multimodal.clinical_metadata import ClinicalMetadataBranch


class ConstrainedResidualFusionModel(nn.Module):
    """
    Alternative C: Constrained Residual Fusion.
    The clinical branch contributes exclusively non-negative logit boosts.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        hidden_dim: int = 128,
        clinical_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes

        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Visual classifier head producing base visual logits [l_0, l_1]
        self.visual_head = nn.Linear(hidden_dim, num_classes)

        # 64-D Residual Clinical Metadata Branch
        self.clinical_branch = ClinicalMetadataBranch(
            num_locations=15,
            location_embed_dim=16,
            hidden_dim=clinical_dim,
            dropout=0.10,
        )

        # Positive-only projection: Linear -> LayerNorm -> SiLU -> Linear -> Softplus
        self.clinical_boost_head = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Strictly non-negative output >= 0
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        loc_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.
        Returns:
            logits: [B, 2] where logit for melanoma (index 1) is l_1_vis + delta_l1 (delta_l1 >= 0)
        """
        zv = self.visual_encoder(visual_features)  # [B, 128]
        vis_logits = self.visual_head(zv)  # [B, 2] -> [l_0_vis, l_1_vis]

        zc = self.clinical_branch(age_norm, age_missing, sex_onehot, loc_indices)  # [B, 64]
        delta_l1 = self.clinical_boost_head(zc)  # [B, 1] strictly >= 0

        l0 = vis_logits[:, 0:1]  # [B, 1]
        l1 = vis_logits[:, 1:2] + delta_l1  # [B, 1] strictly >= l_1_vis

        return torch.cat([l0, l1], dim=-1)  # [B, 2]

    def get_components(
        self,
        visual_features: torch.Tensor,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        loc_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Expose visual logits, clinical boost, and fused probabilities."""
        zv = self.visual_encoder(visual_features)
        vis_logits = self.visual_head(zv)
        p_vis = torch.softmax(vis_logits, dim=-1)[:, 1]

        zc = self.clinical_branch(age_norm, age_missing, sex_onehot, loc_indices)
        delta_l1 = self.clinical_boost_head(zc).squeeze(-1)

        l0 = vis_logits[:, 0]
        l1 = vis_logits[:, 1] + delta_l1
        fused_logits = torch.stack([l0, l1], dim=-1)
        p_fused = torch.softmax(fused_logits, dim=-1)[:, 1]

        return {
            "p_visual": p_vis,
            "delta_l1": delta_l1,
            "p_fused": p_fused,
        }


class GatedPositiveFusionModel(nn.Module):
    """
    Alternative D: Gated Positive-Only Clinical Contribution.
    Clinical non-negative boost is modulated by a visual uncertainty gate:
       gate = Sigmoid(Linear(128, 1)(z_vis)) in [0, 1]
       Delta_l1 = gate * Softplus(clinical_boost) >= 0
    """

    def __init__(
        self,
        visual_dim: int = 768,
        hidden_dim: int = 128,
        clinical_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes

        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Visual classifier head producing base visual logits [l_0, l_1]
        self.visual_head = nn.Linear(hidden_dim, num_classes)

        # Visual gate: controls how much demographic boost can penetrate
        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # 64-D Residual Clinical Metadata Branch
        self.clinical_branch = ClinicalMetadataBranch(
            num_locations=15,
            location_embed_dim=16,
            hidden_dim=clinical_dim,
            dropout=0.10,
        )

        # Raw positive clinical boost head
        self.clinical_boost_head = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Output >= 0
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        loc_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        zv = self.visual_encoder(visual_features)  # [B, 128]
        vis_logits = self.visual_head(zv)  # [B, 2]
        gate = self.visual_gate(zv)  # [B, 1] in [0, 1]

        zc = self.clinical_branch(age_norm, age_missing, sex_onehot, loc_indices)  # [B, 64]
        raw_boost = self.clinical_boost_head(zc)  # [B, 1] >= 0

        # Gated non-negative boost
        delta_l1 = gate * raw_boost  # [B, 1] strictly >= 0

        l0 = vis_logits[:, 0:1]
        l1 = vis_logits[:, 1:2] + delta_l1

        return torch.cat([l0, l1], dim=-1)


class VisualEvidencePreserver:
    """
    Transparent, deterministic visual evidence preservation rule evaluator.
    Preserves and outputs:
      1. Original visual probability
      2. Raw fusion probability
      3. Guardrail-adjusted probability
    Tracks violations where clinical metadata silently reduces risk below visual threshold.
    """

    def __init__(
        self,
        visual_alert_threshold: float = 0.50,
        decision_threshold: float = 0.2417,
    ):
        self.visual_alert_threshold = visual_alert_threshold
        self.decision_threshold = decision_threshold

    def evaluate(
        self,
        visual_probability: float,
        fusion_probability: float,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        p_vis = float(visual_probability)
        p_fus = float(fusion_probability)
        metadata = metadata or {}

        # A violation occurs if visual evidence is sufficiently suspicious (>= 0.50),
        # but fusion output drops below the decision threshold
        is_violation = bool(p_vis >= self.visual_alert_threshold and p_fus < self.decision_threshold)
        suppression_delta = max(0.0, p_vis - p_fus)

        if is_violation:
            p_guarded = max(p_fus, p_vis)
            action = "visual_override_applied"
            status = "VIOLATION_ARBITRATED"
        else:
            p_guarded = p_fus
            action = "none"
            status = "PRESERVED"

        return {
            "p_visual": round(p_vis, 4),
            "p_fusion": round(p_fus, 4),
            "p_guarded": round(p_guarded, 4),
            "is_violation": is_violation,
            "suppression_delta": round(suppression_delta, 4),
            "action": action,
            "status": status,
            "metadata": metadata,
            "is_experimental": True,
        }

