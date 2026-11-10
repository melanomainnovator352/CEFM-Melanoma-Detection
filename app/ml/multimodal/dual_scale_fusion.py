"""
Dual-Scale Multimodal Fusion Architectures for CEFM Phase 2.

Implements three fusion paradigms:
  1. ConcatDualScaleFusion: Baseline concatenation of global ViT (768),
     local high-res patch ViT (768), and ABCDE clinical features (16).
  2. GatedDualScaleFusion: Reciprocal Channel Attention & Squeeze-and-Excitation
     Gating between global context, local lesion details, and clinical priors.
  3. CrossAttentionDualScaleFusion: Multi-head cross-attention mechanism where
     clinical descriptors and local visual features cross-attend to global context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class ConcatDualScaleFusion(nn.Module):
    """
    Concatenation-based Dual-Scale Multimodal Fusion.
    Architecture:
      - Global visual encoder: Linear(768, 128) -> LayerNorm -> GELU -> Dropout
      - Local visual encoder: Linear(768, 128) -> LayerNorm -> GELU -> Dropout
      - Clinical encoder: Linear(16, 64) -> LayerNorm -> GELU -> Dropout
      - Joint fusion: Linear(128 + 128 + 64 = 320, 128) -> LayerNorm -> GELU -> Dropout
      - Classification head: Linear(128, 2)
    """
    def __init__(
        self,
        global_dim: int = 768,
        local_dim: int = 768,
        clinical_dim: int = 16,
        hidden_dim: int = 128,
        clinical_hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.clinical_dim = clinical_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.local_encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, clinical_hidden_dim),
            nn.LayerNorm(clinical_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = hidden_dim + hidden_dim + clinical_hidden_dim  # 128 + 128 + 64 = 320
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            global_features: [batch_size, 768]
            local_features: [batch_size, 768]
            clinical_features: [batch_size, 16]
        Returns:
            logits: [batch_size, 2]
        """
        zg = self.global_encoder(global_features)
        zl = self.local_encoder(local_features)
        zc = self.clinical_encoder(clinical_features)

        z = torch.cat([zg, zl, zc], dim=-1)
        fused = self.fusion(z)
        logits = self.classifier(fused)
        return logits

    def predict_proba(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities [batch_size, 2]."""
        logits = self.forward(global_features, local_features, clinical_features)
        return F.softmax(logits, dim=-1)


class GatedDualScaleFusion(nn.Module):
    """
    Gated Dual-Scale Multimodal Fusion with Reciprocal Attention & Clinical Modulation.
    Architecture:
      1. Scale-Adaptive Visual Gating: Dynamically blends global context and local lesion
         micro-structures via an inter-scale gating unit.
      2. Clinical Contextual Modulation: Computes clinical gating coefficients to residuals
         and adaptively amplifies/attenuates visual channels based on ABCDE priors.
      3. Deep residual classification head.
    """
    def __init__(
        self,
        global_dim: int = 768,
        local_dim: int = 768,
        clinical_dim: int = 16,
        hidden_dim: int = 128,
        clinical_hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clinical_hidden_dim = clinical_hidden_dim

        # Encoders
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.local_encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, clinical_hidden_dim),
            nn.LayerNorm(clinical_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Scale-selection gate: dynamically weights global vs local features
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Clinical modulation gate: modulates visual representations with clinical context
        self.clinical_gate = nn.Sequential(
            nn.Linear(clinical_hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Fused projection
        fused_dim = hidden_dim * 2 + clinical_hidden_dim  # blended visual (128) + concat visual (128) + clinical (64)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        zg = self.global_encoder(global_features)   # [B, 128]
        zl = self.local_encoder(local_features)     # [B, 128]
        zc = self.clinical_encoder(clinical_features) # [B, 64]

        # 1. Scale-adaptive visual gating
        concat_v = torch.cat([zg, zl], dim=-1)      # [B, 256]
        scale_weights = self.scale_gate(concat_v)   # [B, 128]
        blended_v = scale_weights * zg + (1.0 - scale_weights) * zl # [B, 128]

        # 2. Clinical modulation gate (residual modulation: (1 + gate))
        c_mod = self.clinical_gate(zc)              # [B, 128]
        modulated_v = blended_v * (1.0 + c_mod)     # [B, 128]

        # 3. Concatenate modulated visual, local detail residual, and clinical representation
        combined = torch.cat([modulated_v, zl, zc], dim=-1) # [B, 128 + 128 + 64 = 320]
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        return logits

    def predict_proba(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(global_features, local_features, clinical_features)
        return F.softmax(logits, dim=-1)


class CrossAttentionDualScaleFusion(nn.Module):
    """
    Cross-Attention Dual-Scale Multimodal Fusion.
    Treats global context, local lesion patches, and clinical features as distinct tokens:
      - Visual tokens: [Global Token (128), Local Token (128)] -> 2 tokens of dim 128
      - Clinical query token: [Clinical Token (128)] -> 1 token of dim 128
      - Multi-Head Cross-Attention allows the clinical query to query visual features,
        identifying regions/scales that correlate with clinical risk signals (e.g., asymmetry, border).
      - Multi-Head Self-Attention allows global context and local details to interact.
    """
    def __init__(
        self,
        global_dim: int = 768,
        local_dim: int = 768,
        clinical_dim: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections into shared embedding dimension
        self.proj_global = nn.Sequential(
            nn.Linear(global_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.proj_local = nn.Sequential(
            nn.Linear(local_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.proj_clinical = nn.Sequential(
            nn.Linear(clinical_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Scale position encoding for the two visual tokens: global (0) and local (1)
        self.visual_scale_embed = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)

        # Cross-Attention: Clinical Query attends to Visual Keys/Values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_cross = nn.LayerNorm(embed_dim)

        # Visual Self-Attention: Global and Local tokens attend to each other
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_self = nn.LayerNorm(embed_dim)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        # Project inputs:
        tg = self.proj_global(global_features).unsqueeze(1)    # [B, 1, 128]
        tl = self.proj_local(local_features).unsqueeze(1)      # [B, 1, 128]
        tc = self.proj_clinical(clinical_features).unsqueeze(1) # [B, 1, 128]

        # Visual tokens sequence
        v_tokens = torch.cat([tg, tl], dim=1) + self.visual_scale_embed # [B, 2, 128]

        # 1. Visual self-attention
        v_attn, _ = self.self_attn(v_tokens, v_tokens, v_tokens)
        v_tokens = self.ln_self(v_tokens + v_attn) # [B, 2, 128]
        v_pooled = v_tokens.mean(dim=1)             # [B, 128]

        # 2. Clinical cross-attention over visual tokens
        c_attn, _ = self.cross_attn(query=tc, key=v_tokens, value=v_tokens)
        c_attended = self.ln_cross(tc + c_attn).squeeze(1) # [B, 128]

        # 3. Concatenate pooled visual and clinical-queried visual representations
        fused = self.fusion(torch.cat([v_pooled, c_attended], dim=-1)) # [B, 128]
        logits = self.classifier(fused)
        return logits

    def predict_proba(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(global_features, local_features, clinical_features)
        return F.softmax(logits, dim=-1)

