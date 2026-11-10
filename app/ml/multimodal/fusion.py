"""
Multimodal Fusion Architecture for Melanoma Assessment (CEFM).
Integrates 768-D Vision Transformer visual embeddings with 16-D ABCDE clinical feature vectors.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CEFMMultimodalFusion(nn.Module):
    """
    Multimodal fusion classifier combining deep ViT features and ABCDE clinical features.
    Architecture:
      - Visual branch: Linear(768, 128) -> LayerNorm -> GELU -> Dropout(0.15)
      - Clinical branch: Linear(16, 64) -> LayerNorm -> GELU -> Dropout(0.15)
      - Concatenated fusion: Linear(192, 128) -> LayerNorm -> GELU -> Dropout(0.20)
      - Classification head: Linear(128, 2)
    """
    def __init__(
        self,
        visual_dim: int = 768,
        clinical_dim: int = 16,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.clinical_dim = clinical_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        clinical_proj_dim = 64
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, clinical_proj_dim),
            nn.LayerNorm(clinical_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = hidden_dim + clinical_proj_dim  # 128 + 64 = 192
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        visual_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            visual_features: [batch_size, 768]
            clinical_features: [batch_size, 16]
        Returns:
            logits: [batch_size, 2]
        """
        zv = self.visual_encoder(visual_features)
        zc = self.clinical_encoder(clinical_features)
        z = torch.cat([zv, zc], dim=-1)
        fused = self.fusion(z)
        logits = self.classifier(fused)
        return logits

    def predict_proba(
        self,
        visual_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities [batch_size, 2]."""
        logits = self.forward(visual_features, clinical_features)
        return F.softmax(logits, dim=-1)
