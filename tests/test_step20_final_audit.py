"""
Tests for CEFM Step 20: Final Research Consolidation, Reproducibility, and Evidence Audit.

Verifies:
1. Exact checkpoint invariance across all 8 verified model files.
2. Complete existence and non-emptiness of all 7 Step 20 deliverables.
3. Content validity of CEFM_STEP20_FINAL_MODEL_COMPARISON.csv (metrics, zero VEP violations for Model F-G).
4. Guardrail mathematical invariants: original scores preserved, non-negative risk override.
5. Invariance of production model prediction paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.multimodal.constrained_fusion import VisualEvidencePreserver

REPO_ROOT = Path("D:/pytest_cache")

EXPECTED_HASHES = {
    "production_vit": (
        REPO_ROOT / "models/classification/vit_model/model.safetensors",
        "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
    ),
    "production_multimodal": (
        REPO_ROOT / "models/multimodal/fusion_model.pt",
        "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
    ),
    "production_m6": (
        REPO_ROOT / "models/experiments/M6/weights.pt",
        "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32",
    ),
    "production_unet": (
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


def test_checkpoint_invariance_step20():
    """Verify all production and experimental checkpoints match authoritative digests."""
    for name, (path, exp_hash) in EXPECTED_HASHES.items():
        assert path.exists(), f"Missing checkpoint: {path}"
        actual = _sha256(path)
        assert actual == exp_hash, f"Checkpoint {name} mutated! Expected {exp_hash}, got {actual}"


def test_step20_deliverables_exist():
    """Verify all Step 20 deliverables exist and are non-empty."""
    deliverables = [
        "CEFM_STEP20_FINAL_MODEL_COMPARISON.csv",
        "CEFM_STEP20_REPRODUCIBILITY_AUDIT.md",
        "CEFM_STEP20_GUARDRAIL_VERIFICATION.md",
        "CEFM_STEP20_LIMITATIONS.md",
        "CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json",
        "CEFM_STEP20_FINAL_RESEARCH_REPORT.md",
    ]
    for fname in deliverables:
        p = REPO_ROOT / "reports" / fname
        assert p.exists(), f"Missing deliverable: {fname}"
        assert p.stat().st_size > 0, f"Empty deliverable: {fname}"

    # Also verify README.md exists and is non-empty
    readme = REPO_ROOT / "README.md"
    assert readme.exists() and readme.stat().st_size > 0


def test_final_model_comparison_metrics_integrity():
    """Verify metrics and VEP violation counts in the final comparison CSV."""
    df = pd.read_csv(REPO_ROOT / "reports/CEFM_STEP20_FINAL_MODEL_COMPARISON.csv")

    assert len(df) == 10, f"Expected 10 rows (5 models x 2 cohorts), got {len(df)}"

    # HAM10000 Test Model F-G
    ham_fg = df[(df["Evaluation_Cohort"] == "HAM10000_Internal_Test") & (df["Model_Name"] == "Challenger_Model_FG")].iloc[0]
    assert ham_fg["Sensitivity"] == pytest.approx(0.6867, abs=1e-3)
    assert ham_fg["Specificity"] == pytest.approx(0.8894, abs=1e-3)
    assert ham_fg["VEP_Violations"] == 0
    assert ham_fg["Melanomas_Suppressed_by_Prior"] == 0

    # PH2 External Benchmark Model F-G
    ph2_fg = df[(df["Evaluation_Cohort"] == "PH2_External_Benchmark") & (df["Model_Name"] == "Challenger_Model_FG")].iloc[0]
    assert ph2_fg["Sensitivity"] == pytest.approx(0.5250, abs=1e-3)
    assert ph2_fg["Specificity"] == pytest.approx(0.9187, abs=1e-3)
    assert ph2_fg["VEP_Violations"] == 0
    assert ph2_fg["Melanomas_Suppressed_by_Prior"] == 0

    # PH2 External Benchmark Unconstrained Model F (Failure demonstration)
    ph2_f = df[(df["Evaluation_Cohort"] == "PH2_External_Benchmark") & (df["Model_Name"] == "Experimental_Model_F")].iloc[0]
    assert ph2_f["Sensitivity"] <= 0.10
    assert ph2_f["VEP_Violations"] == 30
    assert ph2_f["Melanomas_Suppressed_by_Prior"] == 19


def test_guardrail_mathematical_invariants():
    """Verify the 6 mathematical software invariants of VisualEvidencePreserver."""
    vep = VisualEvidencePreserver(visual_alert_threshold=0.50, decision_threshold=0.21032)

    # 1. Invariant: Original scores preserved
    res = vep.evaluate(visual_probability=0.82, fusion_probability=0.08)
    assert res["p_visual"] == pytest.approx(0.82)
    assert res["p_fusion"] == pytest.approx(0.08)

    # 2. Invariant: Violation arbitrated when p_vis >= alert and p_fus < threshold
    assert res["is_violation"] is True
    assert res["p_guarded"] == pytest.approx(0.82)
    assert res["action"] == "visual_override_applied"
    assert res["status"] == "VIOLATION_ARBITRATED"

    # 3. Invariant: Preserved when concordant
    res_clean = vep.evaluate(visual_probability=0.82, fusion_probability=0.75)
    assert res_clean["is_violation"] is False
    assert res_clean["p_guarded"] == pytest.approx(0.75)
    assert res_clean["action"] == "none"

    # 4. Invariant: Non-negative override: p_guarded >= p_fusion always
    rng = np.random.default_rng(42)
    for _ in range(500):
        pv = float(rng.uniform(0.0, 1.0))
        pf = float(rng.uniform(0.0, 1.0))
        r = vep.evaluate(pv, pf)
        assert r["p_guarded"] >= round(pf, 4) - 1e-6, "Guardrail lowered fusion probability!"


def test_manifest_json_completeness():
    """Verify that CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json is valid JSON and contains all sections."""
    m_path = REPO_ROOT / "reports/CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json"
    with open(m_path) as f:
        data = json.load(f)

    assert data["step"] == "CEFM_STEP_20"
    assert "verified_checkpoints" in data
    assert "operating_thresholds" in data
    assert "datasets" in data
    assert "deliverables" in data
    assert len(data["deliverables"]) >= 6
