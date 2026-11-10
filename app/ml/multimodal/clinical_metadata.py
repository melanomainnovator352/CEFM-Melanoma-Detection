"""
CEFM Step 16: Clinical Metadata Branch Architecture and Preprocessing Pipeline.

Implements:
1. Leakage-free clinical metadata preprocessing (age normalization, missing handling, 3-way sex, anatomical location vocab).
2. 64-dimensional residual clinical metadata branch:
   - Inputs: normalized age (1), missing age indicator (1), 3-way sex (3), 16-D learned location embedding (16) -> 21-D input.
   - Network: Linear(21, 64) -> LayerNorm(64) -> SiLU() -> Linear(64, 64) -> Residual Connection -> 64-D output.
3. Fusion architectures for controlled ablation experiments (A through F).
4. Strict experimental isolation: does not modify or overwrite production models or checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ANATOMICAL_LOCATIONS = [
    "unknown",
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "foot",
    "neck",
    "scalp",
    "hand",
    "ear",
    "genital",
    "acral",
]


class ClinicalMetadataPreprocessor:
    """
    Leak-free clinical metadata preprocessor.
    Computes summary statistics exclusively on the training set.
    """

    def __init__(self):
        self.age_mean: float = 0.0
        self.age_std: float = 1.0
        self.location_to_idx: Dict[str, int] = {loc: idx for idx, loc in enumerate(ANATOMICAL_LOCATIONS)}
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> ClinicalMetadataPreprocessor:
        """Fit normalization parameters strictly on training dataframe."""
        valid_ages = df_train["age"].dropna().to_numpy(dtype=np.float32)
        if len(valid_ages) > 0:
            self.age_mean = float(np.mean(valid_ages))
            self.age_std = float(np.std(valid_ages))
            if self.age_std == 0:
                self.age_std = 1.0
        else:
            self.age_mean = 50.0
            self.age_std = 15.0

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Transform raw clinical metadata into PyTorch tensors.
        Returns:
            age_norm: [N, 1] normalized age float tensor (0.0 if missing)
            age_missing: [N, 1] binary flag (1.0 if missing, 0.0 if present)
            sex_onehot: [N, 3] float tensor [male, female, unknown]
            loc_indices: [N] long tensor of location vocabulary indices
            clinical_21d: [N, 21] float tensor if location is one-hot or placeholders
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalMetadataPreprocessor must be fitted on training data first.")

        n_samples = len(df)

        # 1. Age normalization and missing indicator
        age_raw = df["age"].to_numpy()
        age_missing = np.isnan(age_raw).astype(np.float32)[:, np.newaxis]
        age_imputed = np.where(np.isnan(age_raw), self.age_mean, age_raw).astype(np.float32)[:, np.newaxis]
        age_norm = (age_imputed - self.age_mean) / self.age_std
        # Zero out missing age normalized value so it does not inject pseudo-signal
        age_norm = np.where(age_missing > 0.5, 0.0, age_norm).astype(np.float32)

        # 2. Three-way sex encoding [male, female, unknown]
        sex_raw = df["sex"].fillna("unknown").astype(str).str.lower().to_numpy()
        sex_onehot = np.zeros((n_samples, 3), dtype=np.float32)
        for i, s in enumerate(sex_raw):
            if s == "male":
                sex_onehot[i, 0] = 1.0
            elif s == "female":
                sex_onehot[i, 1] = 1.0
            else:
                sex_onehot[i, 2] = 1.0

        # 3. Anatomical location vocabulary index
        loc_raw = df["localization"].fillna("unknown").astype(str).str.lower().to_numpy()
        loc_indices = np.zeros(n_samples, dtype=np.int64)
        for i, loc in enumerate(loc_raw):
            loc_indices[i] = self.location_to_idx.get(loc, 0)  # default to 'unknown' (0)

        return {
            "age_norm": torch.from_numpy(age_norm),
            "age_missing": torch.from_numpy(age_missing),
            "sex_onehot": torch.from_numpy(sex_onehot),
            "loc_indices": torch.from_numpy(loc_indices),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration and normalization statistics to dictionary."""
        return {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "location_to_idx": self.location_to_idx,
            "anatomical_locations": ANATOMICAL_LOCATIONS,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClinicalMetadataPreprocessor:
        """Instantiate preprocessor from serialized dictionary."""
        prep = cls()
        prep.age_mean = float(data.get("age_mean", 0.0))
        prep.age_std = float(data.get("age_std", 1.0))
        prep.location_to_idx = data.get("location_to_idx", {loc: idx for idx, loc in enumerate(ANATOMICAL_LOCATIONS)})
        prep.is_fitted = bool(data.get("is_fitted", True))
        return prep


class ClinicalMetadataBranch(nn.Module):
    """
    Proposed 64-dimensional clinical metadata representation branch.

    Input Components:
      - Normalized age: 1-D float (z-score, 0.0 if missing)
      - Missing age indicator: 1-D binary flag (1 if missing, 0 if present)
      - Three-way sex representation: 3-D float vector [male, female, unknown]
      - Learned anatomical location embedding: 16-D float vector from nn.Embedding(15, 16)
      Total input vector dimension: 1 + 1 + 3 + 16 = 21 dimensions.

    Network Forward Pass:
      u = [age_norm || age_missing || sex_onehot || loc_emb]  in R^21
      h1 = SiLU(LayerNorm(Linear(21, 64)(u)))                  in R^64
      h2 = Linear(64, 64)(h1)                                  in R^64
      z_clinical = h1 + h2                                     in R^64 (Residual Connection)
    """

    def __init__(
        self,
        num_locations: int = 15,
        location_embed_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.num_locations = num_locations
        self.location_embed_dim = location_embed_dim
        self.hidden_dim = hidden_dim

        # 16-D learned location embedding
        self.location_embedding = nn.Embedding(num_locations, location_embed_dim)

        # Total input dimension: 1 (age_norm) + 1 (age_missing) + 3 (sex_onehot) + 16 (loc_emb) = 21
        self.input_dim = 1 + 1 + 3 + location_embed_dim  # 21

        self.linear1 = nn.Linear(self.input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        loc_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            age_norm: [B, 1]
            age_missing: [B, 1]
            sex_onehot: [B, 3]
            loc_indices: [B]
        Returns:
            z_clinical: [B, 64]
        """
        loc_emb = self.location_embedding(loc_indices)  # [B, 16]
        u = torch.cat([age_norm, age_missing, sex_onehot, loc_emb], dim=-1)  # [B, 21]

        h1 = self.act(self.norm1(self.linear1(u)))  # [B, 64]
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)  # [B, 64]
        z_clinical = h1 + h2  # [B, 64] (Residual connection)
        return z_clinical


# =====================================================================
# EXPERIMENTAL ABLATION ARCHITECTURES (A THROUGH F)
# =====================================================================

class VisualOnlyClassifier(nn.Module):
    """Experiment A: Visual-only baseline."""

    def __init__(self, visual_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, visual_features: torch.Tensor, **kwargs) -> torch.Tensor:
        zv = self.visual_encoder(visual_features)
        return self.classifier(zv)


class ClinicalOnlyClassifier(nn.Module):
    """Experiment B: Clinical-only branch."""

    def __init__(self, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.10):
        super().__init__()
        self.clinical_branch = ClinicalMetadataBranch(hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        loc_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        zc = self.clinical_branch(age_norm, age_missing, sex_onehot, loc_indices)
        return self.classifier(zc)


class VisualAgeClassifier(nn.Module):
    """Experiment C: Visual + Age (normalized + missing indicator)."""

    def __init__(self, visual_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.age_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        visual_features: torch.Tensor,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        zv = self.visual_encoder(visual_features)
        age_feat = torch.cat([age_norm, age_missing], dim=-1)
        za = self.age_encoder(age_feat)
        z = torch.cat([zv, za], dim=-1)
        fused = self.fusion(z)
        return self.classifier(fused)


class VisualAgeSexClassifier(nn.Module):
    """Experiment D: Visual + Age + Sex."""

    def __init__(self, visual_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Age (2) + Sex (3) = 5
        self.clin_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        visual_features: torch.Tensor,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        zv = self.visual_encoder(visual_features)
        clin_feat = torch.cat([age_norm, age_missing, sex_onehot], dim=-1)
        zc = self.clin_encoder(clin_feat)
        z = torch.cat([zv, zc], dim=-1)
        fused = self.fusion(z)
        return self.classifier(fused)


class VisualAgeSexLocClassifier(nn.Module):
    """Experiment E: Visual + Age + Sex + Anatomical Location (without residual block)."""

    def __init__(self, visual_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.location_embedding = nn.Embedding(15, 16)
        # 1 (age_norm) + 1 (age_missing) + 3 (sex) + 16 (loc) = 21
        self.clin_linear = nn.Sequential(
            nn.Linear(21, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        visual_features: torch.Tensor,
        age_norm: torch.Tensor,
        age_missing: torch.Tensor,
        sex_onehot: torch.Tensor,
        loc_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        zv = self.visual_encoder(visual_features)
        loc_emb = self.location_embedding(loc_indices)
        u = torch.cat([age_norm, age_missing, sex_onehot, loc_emb], dim=-1)
        zc = self.clin_linear(u)
        z = torch.cat([zv, zc], dim=-1)
        fused = self.fusion(z)
        return self.classifier(fused)


class ProposedClinicalFusionModel(nn.Module):
    """
    Experiment F: Full Proposed Fusion Model.
    Combines 768-D ViT Visual Representation and 64-D Residual Clinical Metadata Branch.
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

        # 64-D Residual Clinical Metadata Branch
        self.clinical_branch = ClinicalMetadataBranch(
            num_locations=15,
            location_embed_dim=16,
            hidden_dim=clinical_dim,
            dropout=0.10,
        )

        # Fusion network: hidden_dim (128) + clinical_dim (64) = 192
        fused_dim = hidden_dim + clinical_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

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
        Args:
            visual_features: [B, 768]
            age_norm: [B, 1]
            age_missing: [B, 1]
            sex_onehot: [B, 3]
            loc_indices: [B]
        Returns:
            logits: [B, 2]
        """
        zv = self.visual_encoder(visual_features)  # [B, 128]
        zc = self.clinical_branch(age_norm, age_missing, sex_onehot, loc_indices)  # [B, 64]
        z = torch.cat([zv, zc], dim=-1)  # [B, 192]
        fused = self.fusion(z)  # [B, 128]
        logits = self.classifier(fused)  # [B, 2]
        return logits

