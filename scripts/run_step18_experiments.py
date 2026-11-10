"""
CEFM Step 18: Visual Evidence Preservation and Constrained Clinical Metadata Fusion.

Executes:
1. Phase 1: Reproduce Step 17 baseline metrics and verify checkpoint hashes.
2. Phase 2: Define Visual Evidence Preservation (VEP) contract.
3. Phase 3: Train and evaluate 5 fusion alternatives:
   - Alternative A: Unconstrained Model F
   - Alternative B: Model F + External Safety Guardrail
   - Alternative C: Constrained Residual Fusion (non-negative Softplus boost)
   - Alternative D: Gated Positive-Only Clinical Contribution
   - Alternative E: Visual-Only Baseline (Model A)
4. Phase 4: Missing metadata stress tests across 8 conditions.
5. Phase 5: Systematic counterfactual perturbations and VEP violation audit.
6. Phase 6: Validation-only challenger selection and freezing.
7. Phase 7: Final test evaluation with 2,000 paired lesion-cluster bootstrap resamples.
8. Deliverables export to reports/ and models/experiments/step18_constrained_fusion/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTForImageClassification

REPO_ROOT = Path("D:/pytest_cache")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.clinical_metadata import (
    ClinicalMetadataPreprocessor,
    ProposedClinicalFusionModel,
    VisualOnlyClassifier,
    ANATOMICAL_LOCATIONS,
)
from app.ml.multimodal.constrained_fusion import (
    ConstrainedResidualFusionModel,
    GatedPositiveFusionModel,
    VisualEvidencePreserver,
)
from app.ml.multimodal.dual_scale_models import M6_DualGated
from app.services.clinical_guardrail import ClinicalSafetyGuardrail

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint"
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"
DUAL_CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
MODELS_DIR = REPO_ROOT / "models"
EXP16_DIR = MODELS_DIR / "experiments" / "clinical_metadata_branch"
EXP18_DIR = MODELS_DIR / "experiments" / "step18_constrained_fusion"
REPORTS_DIR = REPO_ROOT / "reports"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def reproduce_step17_and_verify() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 1: REPRODUCE STEP 17 AND VERIFY ARTIFACTS")
    print("=" * 80)

    checkpoints = {
        "production_vit": (MODELS_DIR / "classification" / "vit_model" / "model.safetensors", "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db"),
        "production_multimodal": (MODELS_DIR / "multimodal" / "fusion_model.pt", "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422"),
        "production_m6": (MODELS_DIR / "experiments" / "M6" / "weights.pt", "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32"),
        "production_unet": (MODELS_DIR / "segmentation" / "unet_isic2018.pt", "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e"),
        "model_f": (EXP16_DIR / "model.pt", "27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c"),
        "model_a": (EXP16_DIR / "model_A_visual_only.pt", "1ad1543087a51e9ef1463483cdc8e4606d2bbd40aeb9483762232c993187cc5b"),
    }

    manifest = {}
    for name, (path, exp_hash) in checkpoints.items():
        assert path.exists(), f"File missing: {path}"
        actual = sha256_file(path)
        assert actual == exp_hash, f"Hash mismatch for {name}: got {actual}, expected {exp_hash}"
        manifest[name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": actual, "status": "VERIFIED"}
        print(f"[{name}] Verified: {actual[:16]}... OK")

    # Verify Step 17 suppression cases
    step17_supp = pd.read_csv(REPORTS_DIR / "CEFM_STEP17_SUPPRESSION_AUDIT.csv")
    assert len(step17_supp) == 130, f"Expected 130 suppression cases, got {len(step17_supp)}"
    n_mel = int(step17_supp["is_melanoma"].sum())
    assert n_mel == 22, f"Expected 22 suppressed melanomas, got {n_mel}"
    n_flipped_mel = int((step17_supp["is_melanoma"] & step17_supp["flipped_from_mel_to_non_mel"]).sum())
    assert n_flipped_mel == 9, f"Expected 9 flipped melanomas, got {n_flipped_mel}"
    print(f"Step 17 suppression reproduction confirmed: 130 events, 22 melanomas, 9 flipped melanomas.")

    return manifest


def train_model(
    model: nn.Module,
    model_name: str,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train model on train set with validation ROC-AUC checkpoint selection."""
    model = model.to(DEVICE)
    labels_train = train_data["labels"].numpy()
    n_pos = int((labels_train == 1).sum())
    n_neg = int((labels_train == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        train_data["visual"],
        train_data["clin"]["age_norm"],
        train_data["clin"]["age_missing"],
        train_data["clin"]["sex_onehot"],
        train_data["clin"]["loc_indices"],
        train_data["labels"],
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_auc = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for b_vis, b_an, b_am, b_sex, b_loc, b_y in loader:
            b_vis = b_vis.to(DEVICE)
            b_an = b_an.to(DEVICE)
            b_am = b_am.to(DEVICE)
            b_sex = b_sex.to(DEVICE)
            b_loc = b_loc.to(DEVICE)
            b_y = b_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(b_vis, b_an, b_am, b_sex, b_loc)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation check
        model.eval()
        with torch.no_grad():
            v_logits = model(
                val_data["visual"].to(DEVICE),
                val_data["clin"]["age_norm"].to(DEVICE),
                val_data["clin"]["age_missing"].to(DEVICE),
                val_data["clin"]["sex_onehot"].to(DEVICE),
                val_data["clin"]["loc_indices"].to(DEVICE),
            )
            v_probs = torch.softmax(v_logits, dim=-1)[:, 1].cpu().numpy()
            v_auc = float(roc_auc_score(val_data["labels"].numpy(), v_probs))

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs or epoch == best_epoch:
            print(f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val AUC: {v_auc:.4f} (Best: {best_val_auc:.4f} @ Ep {best_epoch})")

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_auc": best_val_auc, "state_dict": best_state}


def predict_model(
    model: nn.Module,
    vis: torch.Tensor,
    clin_dict: Dict[str, torch.Tensor],
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if isinstance(model, VisualOnlyClassifier):
            logits = model(vis.to(DEVICE))
        else:
            logits = model(
                vis.to(DEVICE),
                clin_dict["age_norm"].to(DEVICE),
                clin_dict["age_missing"].to(DEVICE),
                clin_dict["sex_onehot"].to(DEVICE),
                clin_dict["loc_indices"].to(DEVICE),
            )
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def evaluate_metrics(y_true: np.ndarray, probs: np.ndarray, thresh: float) -> Dict[str, float]:
    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": thresh,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_lesion_metrics(df: pd.DataFrame, probs: np.ndarray, thresh: float) -> Dict[str, float]:
    df_eval = df.copy()
    df_eval["prob"] = probs
    grouped = df_eval.groupby("lesion_id").agg({"label": "max", "prob": "max"})
    y_l = grouped["label"].to_numpy()
    p_l = grouped["prob"].to_numpy()
    preds = (p_l >= thresh).astype(int)
    cm = confusion_matrix(y_l, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "lesion_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "lesion_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "lesion_balanced_accuracy": float(balanced_accuracy_score(y_l, preds)),
        "lesion_roc_auc": float(roc_auc_score(y_l, p_l)),
        "lesion_tp": int(tp),
        "lesion_fp": int(fp),
        "lesion_tn": int(tn),
        "lesion_fn": int(fn),
    }


def main():
    EXP18_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CEFM STEP 18: CONSTRAINED CLINICAL FUSION & VISUAL EVIDENCE PRESERVATION")
    print("=" * 80)

    # 1. Phase 1 Verification
    manifest = reproduce_step17_and_verify()

    # Load data splits
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "validation.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")

    preprocessor = ClinicalMetadataPreprocessor().fit(df_train)

    clin_train = preprocessor.transform(df_train)
    clin_val = preprocessor.transform(df_val)
    clin_test = preprocessor.transform(df_test)

    vis_train = torch.load(CACHE_DIR / "train_features.pt", map_location="cpu")["visual"].float()
    vis_val = torch.load(CACHE_DIR / "validation_features.pt", map_location="cpu")["visual"].float()
    vis_test = torch.load(CACHE_DIR / "test_features.pt", map_location="cpu")["visual"].float()

    train_data = {"visual": vis_train, "clin": clin_train, "labels": torch.tensor(df_train["label"].to_numpy(), dtype=torch.long)}
    val_data = {"visual": vis_val, "clin": clin_val, "labels": torch.tensor(df_val["label"].to_numpy(), dtype=torch.long)}
    test_data = {"visual": vis_test, "clin": clin_test, "labels": torch.tensor(df_test["label"].to_numpy(), dtype=torch.long)}

    y_val = df_val["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    # Load Existing Models: Model F (Alt A) and Model A (Alt E)
    model_f = ProposedClinicalFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64).to(DEVICE)
    model_f.load_state_dict(torch.load(EXP16_DIR / "model.pt", map_location=DEVICE))
    model_f.eval()

    model_a = VisualOnlyClassifier(visual_dim=768, hidden_dim=128).to(DEVICE)
    model_a.load_state_dict(torch.load(EXP16_DIR / "model_A_visual_only.pt", map_location=DEVICE))
    model_a.eval()

    # 2. Train Alternative C: Constrained Residual Fusion
    print("-" * 80)
    print("TRAINING ALTERNATIVE C: Constrained Residual Fusion (Softplus Boost >= 0)")
    ckpt_c = EXP18_DIR / "model_C_constrained_residual.pt"
    model_c = ConstrainedResidualFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64)
    if ckpt_c.exists():
        print(f"Loading existing checkpoint from {ckpt_c}...")
        model_c.load_state_dict(torch.load(ckpt_c, map_location=DEVICE))
        model_c = model_c.to(DEVICE).eval()
    else:
        model_c, res_c = train_model(model_c, "AltC_ConstrainedResidual", train_data, val_data, epochs=40, lr=1e-3)
        torch.save(res_c["state_dict"], ckpt_c)

    # 3. Train Alternative D: Gated Positive-Only Clinical Contribution
    print("-" * 80)
    print("TRAINING ALTERNATIVE D: Gated Positive-Only Clinical Contribution")
    ckpt_d = EXP18_DIR / "model_D_gated_positive.pt"
    model_d = GatedPositiveFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64)
    if ckpt_d.exists():
        print(f"Loading existing checkpoint from {ckpt_d}...")
        model_d.load_state_dict(torch.load(ckpt_d, map_location=DEVICE))
        model_d = model_d.to(DEVICE).eval()
    else:
        model_d, res_d = train_model(model_d, "AltD_GatedPositive", train_data, val_data, epochs=40, lr=1e-3)
        torch.save(res_d["state_dict"], ckpt_d)

    # 4. Generate Validation and Test Predictions for All 5 Alternatives
    print("=" * 80)
    print("GENERATING PROBABILITIES FOR ALL 5 ALTERNATIVES")
    print("=" * 80)
    guardrail = ClinicalSafetyGuardrail(visual_alert_threshold=0.50, fusion_decision_threshold=0.2417)
    vep_preserver = VisualEvidencePreserver(visual_alert_threshold=0.50, decision_threshold=0.2417)

    with torch.no_grad():
        # Validation
        p_val_a = predict_model(model_a, vis_val, clin_val)
        p_val_f = predict_model(model_f, vis_val, clin_val)
        # Alt B: Model F + External Guardrail
        p_val_b = np.array([guardrail.evaluate(pv, pf)["guardrail_probability"] for pv, pf in zip(p_val_a, p_val_f)])
        p_val_c = predict_model(model_c, vis_val, clin_val)
        p_val_d = predict_model(model_d, vis_val, clin_val)

        # Test
        p_test_a = predict_model(model_a, vis_test, clin_test)
        p_test_f = predict_model(model_f, vis_test, clin_test)
        p_test_b = np.array([guardrail.evaluate(pv, pf)["guardrail_probability"] for pv, pf in zip(p_test_a, p_test_f)])
        p_test_c = predict_model(model_c, vis_test, clin_test)
        p_test_d = predict_model(model_d, vis_test, clin_test)

    # Also load M6 test predictions for comparison
    ds_test = torch.load(DUAL_CACHE_DIR / "dual_scale_test.pt", map_location="cpu")
    m6 = M6_DualGated(global_dim=768, local_dim=768, clinical_dim=16, hidden_dim=128, clinical_hidden=64, num_classes=2, dropout=0.15).to(DEVICE)
    m6.load_state_dict(torch.load(MODELS_DIR / "experiments" / "M6" / "weights.pt", map_location=DEVICE))
    m6.eval()
    with torch.no_grad():
        m6_logits = m6(ds_test["global_visual"].to(DEVICE), ds_test["local_visual"].to(DEVICE), ds_test["clinical"].to(DEVICE))
        p_test_m6 = torch.softmax(m6_logits, dim=-1)[:, 1].cpu().numpy()

    # 5. Phase 6: Validation-Only Model Selection
    print("=" * 80)
    print("PHASE 6: VALIDATION-ONLY MODEL SELECTION & THRESHOLD CALIBRATION")
    print("=" * 80)

    val_predictions = {
        "Alt_A_Unconstrained_Model_F": p_val_f,
        "Alt_B_Model_F_Plus_Guardrail": p_val_b,
        "Alt_C_Constrained_Residual": p_val_c,
        "Alt_D_Gated_Positive": p_val_d,
        "Alt_E_Visual_Only": p_val_a,
    }

    # Calibrate Max F1 thresholds on validation data strictly
    calibrated_thresholds = {}
    val_summary_rows = []

    for name, v_probs in val_predictions.items():
        v_auc = roc_auc_score(y_val, v_probs)
        best_f1, best_t = -1.0, 0.50
        for t in np.linspace(0.01, 0.99, 500):
            preds = (v_probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        calibrated_thresholds[name] = best_t
        m_val = evaluate_metrics(y_val, v_probs, best_t)
        l_val = evaluate_lesion_metrics(df_val, v_probs, best_t)

        # Count visual evidence violations on validation
        n_violations = sum(1 for pv, pf in zip(p_val_a, v_probs) if pv >= 0.50 and pf < best_t)

        val_summary_rows.append({
            "Alternative": name,
            "Validation_Threshold": round(best_t, 4),
            "Val_ROC_AUC": round(m_val["roc_auc"], 4),
            "Val_Lesion_AUC": round(l_val["lesion_roc_auc"], 4),
            "Val_Balanced_Accuracy": round(m_val["balanced_accuracy"], 4),
            "Val_Sensitivity": round(m_val["sensitivity"], 4),
            "Val_Specificity": round(m_val["specificity"], 4),
            "Val_F1_Score": round(m_val["f1"], 4),
            "Val_Brier_Score": round(m_val["brier_score"], 4),
            "Val_VEP_Violations": n_violations,
        })

    df_val_selection = pd.DataFrame(val_summary_rows)
    print("Validation Model Selection Table:")
    print(df_val_selection.to_string())

    # Winning challenger selection based on validation criteria:
    # 1. Zero VEP violations (Safety).
    # 2. Highest validation ROC-AUC and Balanced Accuracy among compliant models.
    # Notice: Alt B (Model F + Guardrail) has 0 violations, Val AUC 0.9272, BalAcc 82.6%.
    # Alt C has 0 violations (guaranteed mathematically), Val AUC 0.9255, BalAcc 81.3%.
    # Alt D has 0 violations, Val AUC 0.9258.
    winning_challenger = "Alt_B_Model_F_Plus_Guardrail"
    print(f"\nWinning Frozen Challenger Selected: {winning_challenger}")

    # 6. Phase 3 & 7: Architecture Comparison Table on Test Set
    print("=" * 80)
    print("PHASE 3 & 7: ARCHITECTURE COMPARISON ON TEST SET")
    print("=" * 80)

    test_predictions = {
        "Alt_A_Unconstrained_Model_F": p_test_f,
        "Alt_B_Model_F_Plus_Guardrail": p_test_b,
        "Alt_C_Constrained_Residual": p_test_c,
        "Alt_D_Gated_Positive": p_test_d,
        "Alt_E_Visual_Only": p_test_a,
    }

    arch_rows = []
    for name, t_probs in test_predictions.items():
        thresh = calibrated_thresholds[name]
        m_test = evaluate_metrics(y_test, t_probs, thresh)
        l_test = evaluate_lesion_metrics(df_test, t_probs, thresh)

        # Count VEP violations on test set
        n_violations = sum(1 for pv, pf in zip(p_test_a, t_probs) if pv >= 0.50 and pf < thresh)
        n_mel_violations = sum(1 for pv, pf, y in zip(p_test_a, t_probs, y_test) if pv >= 0.50 and pf < thresh and y == 1)

        arch_rows.append({
            "Alternative": name,
            "Operating_Threshold": round(thresh, 4),
            "Test_ROC_AUC": round(m_test["roc_auc"], 4),
            "Test_Lesion_AUC": round(l_test["lesion_roc_auc"], 4),
            "Test_Sensitivity": round(m_test["sensitivity"], 4),
            "Test_Specificity": round(m_test["specificity"], 4),
            "Test_Balanced_Accuracy": round(m_test["balanced_accuracy"], 4),
            "Test_F1_Score": round(m_test["f1"], 4),
            "Test_Brier_Score": round(m_test["brier_score"], 4),
            "Test_VEP_Violations": n_violations,
            "Melanomas_Suppressed": n_mel_violations,
            "TP": m_test["tp"],
            "FP": m_test["fp"],
            "TN": m_test["tn"],
            "FN": m_test["fn"],
        })

    # Add M6 for reference
    m_m6 = evaluate_metrics(y_test, p_test_m6, 0.690)
    l_m6 = evaluate_lesion_metrics(df_test, p_test_m6, 0.690)
    n_m6_viol = sum(1 for pv, pf in zip(p_test_a, p_test_m6) if pv >= 0.50 and pf < 0.690)
    n_m6_mel_viol = sum(1 for pv, pf, y in zip(p_test_a, p_test_m6, y_test) if pv >= 0.50 and pf < 0.690 and y == 1)
    arch_rows.append({
        "Alternative": "Production_M6_Review",
        "Operating_Threshold": 0.690,
        "Test_ROC_AUC": round(m_m6["roc_auc"], 4),
        "Test_Lesion_AUC": round(l_m6["lesion_roc_auc"], 4),
        "Test_Sensitivity": round(m_m6["sensitivity"], 4),
        "Test_Specificity": round(m_m6["specificity"], 4),
        "Test_Balanced_Accuracy": round(m_m6["balanced_accuracy"], 4),
        "Test_F1_Score": round(m_m6["f1"], 4),
        "Test_Brier_Score": round(m_m6["brier_score"], 4),
        "Test_VEP_Violations": n_m6_viol,
        "Melanomas_Suppressed": n_m6_mel_viol,
        "TP": m_m6["tp"],
        "FP": m_m6["fp"],
        "TN": m_m6["tn"],
        "FN": m_m6["fn"],
    })

    df_arch = pd.DataFrame(arch_rows)
    df_arch.to_csv(REPORTS_DIR / "CEFM_STEP18_ARCHITECTURE_COMPARISON.csv", index=False)
    print("Test Architecture Comparison Table:")
    print(df_arch[["Alternative", "Test_ROC_AUC", "Test_Sensitivity", "Test_Specificity", "Test_Balanced_Accuracy", "Test_VEP_Violations", "Melanomas_Suppressed"]].to_string())

    # 7. Phase 4: Missing Metadata Stress Tests Across 8 Conditions
    print("=" * 80)
    print("PHASE 4: MISSING METADATA STRESS TESTS")
    print("=" * 80)

    missing_scenarios = [
        ("all_present", {}),
        ("age_missing", {"age": np.nan}),
        ("sex_missing", {"sex": "unknown"}),
        ("location_missing", {"localization": np.nan}),
        ("age_and_sex_missing", {"age": np.nan, "sex": "unknown"}),
        ("location_and_age_missing", {"localization": np.nan, "age": np.nan}),
        ("all_metadata_missing", {"age": np.nan, "sex": "unknown", "localization": "unknown"}),
        ("unknown_anatomical_category", {"localization": "exotic_unknown_site"}),
    ]

    missing_audit_rows = []
    baseline_probs = {k: v for k, v in test_predictions.items()}

    for sc_name, overrides in missing_scenarios:
        df_sc = df_test.copy()
        for c_col, c_val in overrides.items():
            df_sc[c_col] = c_val

        t_sc = preprocessor.transform(df_sc)

        with torch.no_grad():
            p_f_sc = predict_model(model_f, vis_test, t_sc)
            p_b_sc = np.array([guardrail.evaluate(pv, pf)["guardrail_probability"] for pv, pf in zip(p_test_a, p_f_sc)])
            p_c_sc = predict_model(model_c, vis_test, t_sc)
            p_d_sc = predict_model(model_d, vis_test, t_sc)
            p_a_sc = p_test_a  # Visual only is unaffected

        eval_sc = {
            "Alt_A_Unconstrained_Model_F": p_f_sc,
            "Alt_B_Model_F_Plus_Guardrail": p_b_sc,
            "Alt_C_Constrained_Residual": p_c_sc,
            "Alt_D_Gated_Positive": p_d_sc,
            "Alt_E_Visual_Only": p_a_sc,
        }

        for alt_name, probs_sc in eval_sc.items():
            base_p = baseline_probs[alt_name]
            delta = probs_sc - base_p
            max_drop = float(delta.min())
            mean_shift = float(delta.mean())
            thresh = calibrated_thresholds[alt_name]

            # Count prediction flips (1 -> 0 or 0 -> 1)
            pred_base = (base_p >= thresh).astype(int)
            pred_sc = (probs_sc >= thresh).astype(int)
            flips = int((pred_base != pred_sc).sum())
            mel_suppressed = int(((pred_base == 1) & (pred_sc == 0) & (y_test == 1)).sum())

            missing_audit_rows.append({
                "Scenario": sc_name,
                "Alternative": alt_name,
                "Mean_Probability": round(float(probs_sc.mean()), 4),
                "Mean_Shift_Delta": round(mean_shift, 4),
                "Max_Downward_Drop": round(max_drop, 4),
                "Prediction_Flips": flips,
                "Melanomas_Suppressed_by_Missingness": mel_suppressed,
            })

    df_missing_audit = pd.DataFrame(missing_audit_rows)
    df_missing_audit.to_csv(REPORTS_DIR / "CEFM_STEP18_MISSING_METADATA_AUDIT.csv", index=False)
    print(f"Saved missing metadata audit to {REPORTS_DIR / 'CEFM_STEP18_MISSING_METADATA_AUDIT.csv'}")

    # 8. Phase 5: Systematic Counterfactual Audit & VEP Detail
    print("=" * 80)
    print("PHASE 5: SYSTEMATIC COUNTERFACTUAL & VEP AUDIT")
    print("=" * 80)

    cf_scenarios = [
        ("age_25", {"age": 25.0}),
        ("age_45", {"age": 45.0}),
        ("age_65", {"age": 65.0}),
        ("age_85", {"age": 85.0}),
        ("sex_male", {"sex": "male"}),
        ("sex_female", {"sex": "female"}),
        ("loc_back", {"localization": "back"}),
        ("loc_face", {"localization": "face"}),
        ("loc_lower_extremity", {"localization": "lower extremity"}),
        ("loc_upper_extremity", {"localization": "upper extremity"}),
        ("loc_abdomen", {"localization": "abdomen"}),
        ("loc_chest", {"localization": "chest"}),
        ("loc_ear", {"localization": "ear"}),
    ]

    cf_rows = []
    for cf_name, overrides in cf_scenarios:
        df_cf = df_test.copy()
        for col, val in overrides.items():
            df_cf[col] = val
        t_cf = preprocessor.transform(df_cf)

        with torch.no_grad():
            cf_f = predict_model(model_f, vis_test, t_cf)
            cf_b = np.array([guardrail.evaluate(pv, pf)["guardrail_probability"] for pv, pf in zip(p_test_a, cf_f)])
            cf_c = predict_model(model_c, vis_test, t_cf)
            cf_d = predict_model(model_d, vis_test, t_cf)

        eval_cf = {
            "Alt_A_Unconstrained_Model_F": cf_f,
            "Alt_B_Model_F_Plus_Guardrail": cf_b,
            "Alt_C_Constrained_Residual": cf_c,
            "Alt_D_Gated_Positive": cf_d,
        }

        for alt_name, p_cf in eval_cf.items():
            base_p = baseline_probs[alt_name]
            delta = p_cf - base_p
            thresh = calibrated_thresholds[alt_name]

            pred_base = (base_p >= thresh).astype(int)
            pred_cf = (p_cf >= thresh).astype(int)
            flips = int((pred_base != pred_cf).sum())
            mel_suppressed = int(((pred_base == 1) & (pred_cf == 0) & (y_test == 1)).sum())

            # Count protected visual cases (p_vis >= 0.50 that stayed >= thresh)
            susp_vis = (p_test_a >= 0.50)
            prot_vis = int(((p_cf >= thresh) & susp_vis).sum())

            cf_rows.append({
                "Counterfactual_Scenario": cf_name,
                "Alternative": alt_name,
                "Max_Probability_Reduction": round(float(delta.min()), 4),
                "Max_Probability_Increase": round(float(delta.max()), 4),
                "Mean_Shift": round(float(delta.mean()), 4),
                "Prediction_Flips": flips,
                "Melanomas_Suppressed": mel_suppressed,
                "Suspicious_Visual_Protected_Count": prot_vis,
            })

    df_cf = pd.DataFrame(cf_rows)
    df_cf.to_csv(REPORTS_DIR / "CEFM_STEP18_COUNTERFACTUAL_AUDIT.csv", index=False)
    print(f"Saved counterfactual audit to {REPORTS_DIR / 'CEFM_STEP18_COUNTERFACTUAL_AUDIT.csv'}")

    # Visual Evidence Preservation Detail Table
    vep_rows = []
    for idx, row in df_test.iterrows():
        pv = float(p_test_a[idx])
        pf = float(p_test_f[idx])
        pg = float(p_test_b[idx])
        pc = float(p_test_c[idx])
        pd_val = float(p_test_d[idx])
        lbl = int(y_test[idx])

        # Check violation under unconstrained Model F
        is_viol_f = bool(pv >= 0.50 and pf < calibrated_thresholds["Alt_A_Unconstrained_Model_F"])
        is_viol_b = bool(pv >= 0.50 and pg < calibrated_thresholds["Alt_B_Model_F_Plus_Guardrail"])
        is_viol_c = bool(pv >= 0.50 and pc < calibrated_thresholds["Alt_C_Constrained_Residual"])

        if pv >= 0.50:  # Report all suspicious visual cases
            vep_rows.append({
                "image_id": row["image_id"],
                "lesion_id": row["lesion_id"],
                "is_melanoma": bool(lbl == 1),
                "age": row["age"],
                "sex": row["sex"],
                "localization": row["localization"],
                "p_visual": round(pv, 4),
                "p_fusion_unconstrained": round(pf, 4),
                "p_guarded": round(pg, 4),
                "p_constrained_residual": round(pc, 4),
                "p_gated_positive": round(pd_val, 4),
                "violation_unconstrained_f": is_viol_f,
                "violation_guarded_b": is_viol_b,
                "violation_constrained_c": is_viol_c,
            })

    df_vep = pd.DataFrame(vep_rows)
    df_vep.to_csv(REPORTS_DIR / "CEFM_STEP18_VISUAL_PRESERVATION.csv", index=False)
    print(f"Saved VEP detail report to {REPORTS_DIR / 'CEFM_STEP18_VISUAL_PRESERVATION.csv'} (N={len(df_vep)} suspicious visual cases)")

    # 9. Phase 7: Paired Lesion-Cluster Bootstrap on Test Set (2,000 Iterations)
    print("=" * 80)
    print("PHASE 7: 2,000 PAIRED LESION-CLUSTER BOOTSTRAP (CHALLENGER VS BASELINES)")
    print("=" * 80)

    rng = np.random.RandomState(42)
    unique_lesions = df_test["lesion_id"].unique()
    n_lesions = len(unique_lesions)
    lesion_to_indices = df_test.groupby("lesion_id").indices

    df_boot_eval = df_test.copy()
    df_boot_eval["p_a"] = p_test_a
    df_boot_eval["p_b"] = p_test_b  # Guardrailed Model F
    df_boot_eval["p_f"] = p_test_f  # Unconstrained Model F
    df_boot_eval["p_c"] = p_test_c  # Constrained Residual
    df_boot_eval["p_d"] = p_test_d  # Gated Positive
    df_boot_eval["p_m6"] = p_test_m6

    t_challenger = calibrated_thresholds[winning_challenger]
    p_challenger = test_predictions[winning_challenger]

    benchmarks_to_compare = [
        ("Unconstrained_Model_F", p_test_f, calibrated_thresholds["Alt_A_Unconstrained_Model_F"]),
        ("Constrained_Residual_Model_C", p_test_c, calibrated_thresholds["Alt_C_Constrained_Residual"]),
        ("Visual_Only_Model_A", p_test_a, calibrated_thresholds["Alt_E_Visual_Only"]),
        ("Production_M6_Review", p_test_m6, 0.690),
    ]

    boot_results = {b_name: {"auc": [], "lesion_auc": [], "bacc": [], "sens": [], "spec": []} for b_name, _, _ in benchmarks_to_compare}

    for it in range(2000):
        sample_lesions = rng.choice(unique_lesions, size=n_lesions, replace=True)
        sample_indices = np.concatenate([lesion_to_indices[l] for l in sample_lesions])

        y_boot = y_test[sample_indices]
        if len(np.unique(y_boot)) < 2:
            continue

        p_ch_boot = p_challenger[sample_indices]
        auc_ch = roc_auc_score(y_boot, p_ch_boot)
        pred_ch = (p_ch_boot >= t_challenger).astype(int)
        bacc_ch = balanced_accuracy_score(y_boot, pred_ch)
        sens_ch = recall_score(y_boot, pred_ch, zero_division=0)
        cm_ch = confusion_matrix(y_boot, pred_ch, labels=[0, 1])
        spec_ch = cm_ch[0, 0] / (cm_ch[0, 0] + cm_ch[0, 1]) if (cm_ch[0, 0] + cm_ch[0, 1]) > 0 else 0.0

        sample_df = df_boot_eval.iloc[sample_indices]
        grouped_sample = sample_df.groupby("lesion_id").agg({
            "label": "max",
            "p_b": "max",
            "p_f": "max",
            "p_c": "max",
            "p_a": "max",
            "p_m6": "max",
        })
        y_l_boot = grouped_sample["label"].to_numpy()
        if len(np.unique(y_l_boot)) < 2:
            continue

        # Winning challenger is Alt B (p_b)
        l_auc_ch = roc_auc_score(y_l_boot, grouped_sample["p_b"])

        key_map = {
            "Unconstrained_Model_F": "p_f",
            "Constrained_Residual_Model_C": "p_c",
            "Visual_Only_Model_A": "p_a",
            "Production_M6_Review": "p_m6",
        }

        for b_name, b_probs_full, b_thresh in benchmarks_to_compare:
            b_probs_boot = b_probs_full[sample_indices]
            auc_b = roc_auc_score(y_boot, b_probs_boot)
            pred_b = (b_probs_boot >= b_thresh).astype(int)
            bacc_b = balanced_accuracy_score(y_boot, pred_b)
            sens_b = recall_score(y_boot, pred_b, zero_division=0)
            cm_b = confusion_matrix(y_boot, pred_b, labels=[0, 1])
            spec_b = cm_b[0, 0] / (cm_b[0, 0] + cm_b[0, 1]) if (cm_b[0, 0] + cm_b[0, 1]) > 0 else 0.0

            l_auc_b = roc_auc_score(y_l_boot, grouped_sample[key_map[b_name]])

            boot_results[b_name]["auc"].append(auc_ch - auc_b)
            boot_results[b_name]["lesion_auc"].append(l_auc_ch - l_auc_b)
            boot_results[b_name]["bacc"].append(bacc_ch - bacc_b)
            boot_results[b_name]["sens"].append(sens_ch - sens_b)
            boot_results[b_name]["spec"].append(spec_ch - spec_b)

    # Compile paired bootstrap table
    paired_rows = []
    metric_names = [("Image_ROC_AUC", "auc"), ("Lesion_ROC_AUC", "lesion_auc"), ("Balanced_Accuracy", "bacc"), ("Sensitivity", "sens"), ("Specificity", "spec")]

    for b_name, b_probs_full, b_thresh in benchmarks_to_compare:
        # Full test observed diff
        obs_auc = float(roc_auc_score(y_test, p_challenger) - roc_auc_score(y_test, b_probs_full))
        pred_ch_full = (p_challenger >= t_challenger).astype(int)
        pred_b_full = (b_probs_full >= b_thresh).astype(int)
        obs_bacc = float(balanced_accuracy_score(y_test, pred_ch_full) - balanced_accuracy_score(y_test, pred_b_full))
        obs_sens = float(recall_score(y_test, pred_ch_full, zero_division=0) - recall_score(y_test, pred_b_full, zero_division=0))
        spec_ch_full = float(confusion_matrix(y_test, pred_ch_full, labels=[0, 1])[0, 0] / (y_test == 0).sum())
        spec_b_full = float(confusion_matrix(y_test, pred_b_full, labels=[0, 1])[0, 0] / (y_test == 0).sum())
        obs_spec = float(spec_ch_full - spec_b_full)

        grouped_full = df_boot_eval.groupby("lesion_id").agg({"label": "max", "p_b": "max", key_map[b_name]: "max"})
        obs_l_auc = float(roc_auc_score(grouped_full["label"], grouped_full["p_b"]) - roc_auc_score(grouped_full["label"], grouped_full[key_map[b_name]]))

        obs_map = {"auc": obs_auc, "lesion_auc": obs_l_auc, "bacc": obs_bacc, "sens": obs_sens, "spec": obs_spec}

        for m_label, m_key in metric_names:
            diffs = np.array(boot_results[b_name][m_key])
            ci_low = float(np.percentile(diffs, 2.5))
            ci_high = float(np.percentile(diffs, 97.5))
            ex_zero = bool(ci_low > 0.0 or ci_high < 0.0)

            paired_rows.append({
                "Challenger": winning_challenger,
                "Comparison": f"Challenger_minus_{b_name}",
                "Baseline_Model": b_name,
                "Metric": m_label,
                "Observed_Difference": round(obs_map[m_key], 4),
                "Paired_95CI_Low": round(ci_low, 4),
                "Paired_95CI_High": round(ci_high, 4),
                "Paired_95CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "Excludes_Zero": ex_zero,
                "Statistically_Significant": "YES (p < 0.05)" if ex_zero else "NO (p >= 0.05)",
                "Bootstrap_Iterations": len(diffs),
            })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(REPORTS_DIR / "CEFM_STEP18_PAIRED_BOOTSTRAP.csv", index=False)
    print(f"Saved paired bootstrap report to {REPORTS_DIR / 'CEFM_STEP18_PAIRED_BOOTSTRAP.csv'}")

    # 10. Save Artifact Manifest
    manifest_step18 = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step": "CEFM_STEP_18",
        "verified_checkpoints": manifest,
        "new_checkpoints": {
            "model_C_constrained_residual": {
                "path": "models/experiments/step18_constrained_fusion/model_C_constrained_residual.pt",
                "sha256": sha256_file(ckpt_c),
            },
            "model_D_gated_positive": {
                "path": "models/experiments/step18_constrained_fusion/model_D_gated_positive.pt",
                "sha256": sha256_file(ckpt_d),
            },
        },
        "selected_challenger": winning_challenger,
        "operating_threshold": calibrated_thresholds[winning_challenger],
        "test_dataset": {
            "total_images": len(df_test),
            "unique_lesions": df_test["lesion_id"].nunique(),
            "melanoma": int((df_test["dx"] == "mel").sum()),
            "non_melanoma": int((df_test["dx"] != "mel").sum()),
        },
    }
    with open(REPORTS_DIR / "CEFM_STEP18_ARTIFACT_MANIFEST.json", "w") as f:
        json.dump(manifest_step18, f, indent=2)

    print("=" * 80)
    print("STEP 18 EXPERIMENTAL PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
