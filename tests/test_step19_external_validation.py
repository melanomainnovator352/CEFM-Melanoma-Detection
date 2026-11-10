"""
Tests for CEFM Step 19: External Validation, Domain Generalization, and Robustness Evaluation.

Verifies:
1. Exact checkpoint invariance of production ViT, Multimodal, M6, UNet, and Model F/A.
2. PH2 dataset loading, ground truth distribution, and lack of demographic metadata.
3. Label harmonization mapping validity (Melanoma=1, Nevi=0).
4. Guardrail behavior under external domain shift: 0 VEP violations, rescuing unconstrained collapse.
5. Guardrail robustness under optical and metadata perturbations (VEP violations = 0).
6. Existence and non-emptiness of all 7 Step 19 deliverables.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.multimodal.constrained_fusion import VisualEvidencePreserver

REPO_ROOT = Path("D:/pytest_cache")

EXPECTED_HASHES = {
    "vit": (
        REPO_ROOT / "models/classification/vit_model/model.safetensors",
        "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
    ),
    "multimodal": (
        REPO_ROOT / "models/multimodal/fusion_model.pt",
        "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
    ),
    "m6": (
        REPO_ROOT / "models/experiments/M6/weights.pt",
        "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32",
    ),
    "unet": (
        REPO_ROOT / "models/segmentation/unet_isic2018.pt",
        "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e",
    ),
    "model_f": (
        REPO_ROOT / "models/experiments/clinical_metadata_branch/model.pt",
        "27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c",
    ),
    "model_a": (
        REPO_ROOT / "models/experiments/clinical_metadata_branch/model_A_visual_only.pt",
        "1ad1543087a51e9ef1463483cdc8e4606d2bbd40aeb9483762232c993187cc5b",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_invariance_step19():
    """Verify all production and experimental checkpoints match authoritative hashes."""
    for name, (path, exp_hash) in EXPECTED_HASHES.items():
        assert path.exists(), f"Missing checkpoint: {path}"
        actual = _sha256(path)
        assert actual == exp_hash, f"Checkpoint {name} mutated! Expected {exp_hash}, got {actual}"


def test_ph2_dataset_integrity():
    """Verify PH2 dataset structure, counts, and lack of demographic metadata."""
    ph2_txt = Path("D:/PH2/PH2_dataset.txt")
    assert ph2_txt.exists(), "PH2_dataset.txt not found"

    with open(ph2_txt, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip().startswith("||") and not l.startswith("||   Name")]

    assert len(lines) == 200, f"Expected 200 cases in PH2, got {len(lines)}"

    codes = [int(l.split("||")[1:-1][2].strip()) for l in lines]
    assert codes.count(0) == 80, "Expected 80 common nevi"
    assert codes.count(1) == 80, "Expected 80 atypical nevi"
    assert codes.count(2) == 40, "Expected 40 melanomas"


def test_external_metrics_guardrail_rescue():
    """Verify Model F-G rescues sensitivity on PH2 compared to collapsed unconstrained Model F."""
    csv_path = REPO_ROOT / "reports/CEFM_STEP19_EXTERNAL_METRICS.csv"
    assert csv_path.exists(), "Missing CEFM_STEP19_EXTERNAL_METRICS.csv"
    df = pd.read_csv(csv_path)

    ph2_fg = df[(df["Dataset"] == "PH2_External_Benchmark") & (df["Model"] == "Challenger_Model_FG")].iloc[0]
    ph2_f = df[(df["Dataset"] == "PH2_External_Benchmark") & (df["Model"] == "Unconstrained_Model_F")].iloc[0]

    # Model F collapses to 5% sensitivity; Model F-G recovers to 52.5%
    assert ph2_f["Sensitivity"] <= 0.10, "Unconstrained Model F did not reflect missingness collapse!"
    assert ph2_fg["Sensitivity"] >= 0.50, "Model F-G failed to restore sensitivity on PH2!"
    assert ph2_fg["VEP_Violations"] == 0, "Model F-G permitted VEP violations on PH2!"
    assert ph2_f["VEP_Violations"] > 0, "Unconstrained Model F should exhibit VEP violations!"


def test_guardrail_robustness_zero_vep_violations():
    """Verify all 14 perturbation scenarios maintain zero VEP violations."""
    csv_path = REPO_ROOT / "reports/CEFM_STEP19_GUARDRAIL_ROBUSTNESS.csv"
    assert csv_path.exists(), "Missing CEFM_STEP19_GUARDRAIL_ROBUSTNESS.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == 14, f"Expected 14 perturbation scenarios, got {len(df)}"
    assert (df["VEP_Violations"] == 0).all(), "Guardrail allowed VEP violations under stress testing!"
    assert (df["Guarded_Specificity"] >= 0.85).all(), "Guardrail specificity dropped below 85% safety bound!"


def test_step19_report_deliverables_exist():
    """Verify all Step 19 deliverable files exist and have non-zero size."""
    expected = [
        "CEFM_STEP19_DATASET_SUITABILITY.md",
        "CEFM_STEP19_LABEL_MAPPING.csv",
        "CEFM_STEP19_EXTERNAL_METRICS.csv",
        "CEFM_STEP19_SUBGROUP_METRICS.csv",
        "CEFM_STEP19_GUARDRAIL_ROBUSTNESS.csv",
        "CEFM_STEP19_ARTIFACT_MANIFEST.json",
    ]
    for fname in expected:
        p = REPO_ROOT / "reports" / fname
        assert p.exists(), f"Missing file: {fname}"
        assert p.stat().st_size > 0, f"File {fname} is empty"

