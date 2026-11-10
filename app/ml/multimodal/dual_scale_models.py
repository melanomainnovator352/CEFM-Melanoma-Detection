"""
CEFM Phase 2: Complete Suite of M1–M7 Experimental Models.

Models:
  M1: Global ViT Only (Linear Head on 768-D global context)
  M2: Local ViT Only (Linear Head on 768-D native-resolution patch)
  M3: Dual-Scale ViT (Visual Only: 768-D global + 768-D local -> 1536-D)
  M4: Global ViT + Clinical (Phase 1 Baseline: 768-D global + 16-D clinical)
  M5: Dual-Scale Concat Multimodal (768-D global + 768-D local + 16-D clinical)
  M6: Dual-Scale Gated Multimodal (RCAG: Scale-Adaptive Residual Gating & Clinical Modulation)
  M7: Dual-Scale Cross-Attention (Multi-Head Cross-Attention: Clinical Query attends to Visual Tokens)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class M1_GlobalViT(nn.Module):
    """M1: Global ViT Only (Linear classification head on 768-D global embedding)."""
    def __init__(self, global_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, global_visual: torch.Tensor, local_visual: Optional[torch.Tensor] = None, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(global_visual)
        return self.classifier(h)

    def predict_proba(self, global_visual: torch.Tensor, local_visual: Optional[torch.Tensor] = None, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


class M2_LocalViT(nn.Module):
    """M2: Local ViT Only (Linear classification head on 768-D local high-res patch embedding)."""
    def __init__(self, local_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, global_visual: Optional[torch.Tensor], local_visual: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(local_visual)
        return self.classifier(h)

    def predict_proba(self, global_visual: Optional[torch.Tensor], local_visual: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


class M3_DualVisual(nn.Module):
    """M3: Dual-Scale Visual Only (Concatenation of 768-D global and 768-D local embeddings)."""
    def __init__(self, global_dim: int = 768, local_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
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
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        zg = self.global_encoder(global_visual)
        zl = self.local_encoder(local_visual)
        z = torch.cat([zg, zl], dim=-1)
        fused = self.fusion(z)
        return self.classifier(fused)

    def predict_proba(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


class M4_GlobalClinical(nn.Module):
    """M4: Global ViT + Clinical (Phase 1 Baseline Architecture: 768-D global + 16-D clinical)."""
    def __init__(self, global_dim: int = 768, clinical_dim: int = 16, hidden_dim: int = 128, clinical_hidden: int = 64, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, clinical_hidden),
            nn.LayerNorm(clinical_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = hidden_dim + clinical_hidden  # 128 + 64 = 192
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, global_visual: torch.Tensor, local_visual: Optional[torch.Tensor], clinical: torch.Tensor) -> torch.Tensor:
        zg = self.global_encoder(global_visual)
        zc = self.clinical_encoder(clinical)
        z = torch.cat([zg, zc], dim=-1)
        fused = self.fusion(z)
        return self.classifier(fused)

    def predict_proba(self, global_visual: torch.Tensor, local_visual: Optional[torch.Tensor], clinical: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


class M5_DualConcat(nn.Module):
    """M5: Dual-Scale Concat Multimodal (768-D global + 768-D local + 16-D clinical)."""
    def __init__(self, global_dim: int = 768, local_dim: int = 768, clinical_dim: int = 16, hidden_dim: int = 128, clinical_hidden: int = 64, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
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
            nn.Linear(clinical_dim, clinical_hidden),
            nn.LayerNorm(clinical_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = hidden_dim + hidden_dim + clinical_hidden  # 128 + 128 + 64 = 320
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        zg = self.global_encoder(global_visual)
        zl = self.local_encoder(local_visual)
        zc = self.clinical_encoder(clinical)
        z = torch.cat([zg, zl, zc], dim=-1)
        fused = self.fusion(z)
        return self.classifier(fused)

    def predict_proba(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


class M6_DualGated(nn.Module):
    """M6: Dual-Scale Gated Multimodal (RCAG: Scale-Adaptive Residual Channel Gating & Clinical Context Modulation)."""
    def __init__(self, global_dim: int = 768, local_dim: int = 768, clinical_dim: int = 16, hidden_dim: int = 128, clinical_hidden: int = 64, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
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
            nn.Linear(clinical_dim, clinical_hidden),
            nn.LayerNorm(clinical_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Inter-scale gating between global context and local lesion details
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Clinical context modulation gate
        self.clinical_gate = nn.Sequential(
            nn.Linear(clinical_hidden, hidden_dim),
            nn.Sigmoid(),
        )
        fused_dim = hidden_dim * 2 + clinical_hidden  # modulated visual (128) + local residual (128) + clinical (64) = 320
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

    def forward(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        zg = self.global_encoder(global_visual)
        zl = self.local_encoder(local_visual)
        zc = self.clinical_encoder(clinical)

        # Scale-adaptive visual gating
        concat_v = torch.cat([zg, zl], dim=-1)
        scale_weights = self.scale_gate(concat_v)
        blended_v = scale_weights * zg + (1.0 - scale_weights) * zl

        # Clinical modulation gate: residual amplification
        c_mod = self.clinical_gate(zc)
        modulated_v = blended_v * (1.0 + c_mod)

        # Combined representation
        combined = torch.cat([modulated_v, zl, zc], dim=-1)
        fused = self.fusion(combined)
        return self.classifier(fused)

    def predict_proba(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


class M7_DualCrossAttn(nn.Module):
    """M7: Dual-Scale Cross-Attention (Clinical Query attends to Visual Tokens + Visual Self-Attention)."""
    def __init__(self, global_dim: int = 768, local_dim: int = 768, clinical_dim: int = 16, embed_dim: int = 128, num_heads: int = 4, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
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

        self.visual_scale_embed = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)

        # Cross-Attention: Clinical Query attends to Visual Keys/Values
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln_cross = nn.LayerNorm(embed_dim)

        # Visual Self-Attention: Global and Local tokens interact
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln_self = nn.LayerNorm(embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        tg = self.proj_global(global_visual).unsqueeze(1)
        tl = self.proj_local(local_visual).unsqueeze(1)
        tc = self.proj_clinical(clinical).unsqueeze(1)

        v_tokens = torch.cat([tg, tl], dim=1) + self.visual_scale_embed

        # Visual self-attention
        v_attn, _ = self.self_attn(v_tokens, v_tokens, v_tokens)
        v_tokens = self.ln_self(v_tokens + v_attn)
        v_pooled = v_tokens.mean(dim=1)

        # Clinical cross-attention over visual tokens
        c_attn, _ = self.cross_attn(query=tc, key=v_tokens, value=v_tokens)
        c_attended = self.ln_cross(tc + c_attn).squeeze(1)

        fused = self.fusion(torch.cat([v_pooled, c_attended], dim=-1))
        return self.classifier(fused)

    def predict_proba(self, global_visual: torch.Tensor, local_visual: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(global_visual, local_visual, clinical), dim=-1)


MODEL_REGISTRY: Dict[str, Any] = {
    "M1": {
        "name": "Global ViT Only",
        "class": M1_GlobalViT,
        "desc": "Baseline global context crop (768-D)",
        "modalities": ["global_visual"],
    },
    "M2": {
        "name": "Local ViT Only",
        "class": M2_LocalViT,
        "desc": "Native-resolution lesion micro-detail patch (768-D)",
        "modalities": ["local_visual"],
    },
    "M3": {
        "name": "Dual-Scale ViT (Visual Only)",
        "class": M3_DualVisual,
        "desc": "Global context + local micro-detail (1536-D)",
        "modalities": ["global_visual", "local_visual"],
    },
    "M4": {
        "name": "Global ViT + Clinical (Phase 1 Baseline)",
        "class": M4_GlobalClinical,
        "desc": "Global context (768-D) + 16-D ABCDE clinical features",
        "modalities": ["global_visual", "clinical"],
    },
    "M5": {
        "name": "Dual-Scale Concat Multimodal",
        "class": M5_DualConcat,
        "desc": "Global (768-D) + Local (768-D) + Clinical (16-D) linear concatenation",
        "modalities": ["global_visual", "local_visual", "clinical"],
    },
    "M6": {
        "name": "Dual-Scale Gated Multimodal (RCAG)",
        "class": M6_DualGated,
        "desc": "Dual-scale visual with reciprocal scale gating & clinical residual modulation",
        "modalities": ["global_visual", "local_visual", "clinical"],
    },
    "M7": {
        "name": "Dual-Scale Cross-Attention",
        "class": M7_DualCrossAttn,
        "desc": "Multi-head cross-modal attention between clinical query and dual-scale visual tokens",
        "modalities": ["global_visual", "local_visual", "clinical"],
    },
}

