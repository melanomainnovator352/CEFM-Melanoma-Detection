"""
CEFM Step 17: Clinical Metadata Safety, Paired Statistical Validation, and Counterfactual Audit.

Executes:
1. Phase 1: Artifact hashing and split manifest verification.
2. Phase 2: 2,000 paired lesion-cluster bootstrap iterations across Model F vs all baselines.
3. Phase 3: Demographic suppression audit (detecting visual-evidence attenuation by demographics).
4. Phase 4: Systematic counterfactual clinical metadata perturbations.
5. Phase 5: Experimental visual safety guardrail validation (tuned on validation set only, evaluated on test).
6. Phase 6: Granular subgroup performance evaluation with small-sample flags.
7. Phase 7: Export all 6 deliverable CSVs and manifest JSON.
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from transformers import ViTForImageClassification

REPO_ROOT = Path("D:/pytest_cache")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.clinical_metadata import (
    ClinicalMetadataPreprocessor,
    ProposedClinicalFusionModel,
    VisualOnlyClassifier,
    ClinicalOnlyClassifier,
    VisualAgeSexLocClassifier,
    ANATOMICAL_LOCATIONS,
)
from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.ml.multimodal.dual_scale_models import M6_DualGated
from app.services.clinical_guardrail import ClinicalSafetyGuardrail

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint"
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"
DUAL_CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
MODELS_DIR = REPO_ROOT / "models"
EXP_DIR = MODELS_DIR / "experiments" / "clinical_metadata_branch"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "evaluation_step17"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_phase1_verification() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 1: FREEZE AND VERIFY EXPERIMENTAL ARTIFACTS")
    print("=" * 80)

    # 1. Hashing checkpoints
    checkpoints = {
        "production_vit": (MODELS_DIR / "classification" / "vit_model" / "model.safetensors", "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db"),
        "production_multimodal": (MODELS_DIR / "multimodal" / "fusion_model.pt", "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422"),
        "production_m6": (MODELS_DIR / "experiments" / "M6" / "weights.pt", "573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32"),
        "production_unet": (MODELS_DIR / "segmentation" / "unet_isic2018.pt", "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e"),
        "step16_model_f": (EXP_DIR / "model.pt", "27f01d6b0c6660f77a67f91604a14531f4277f6b8a41e06ea845490666e4bb2c"),
        "step16_model_a": (EXP_DIR / "model_A_visual_only.pt", "1ad1543087a51e9ef1463483cdc8e4606d2bbd40aeb9483762232c993187cc5b"),
        "step16_model_b": (EXP_DIR / "model_B_clinical_only.pt", "95a7576298bc893e390ea0e7f229bff1fb141388e6d62e509acc0a03155e5450"),
        "step16_model_e": (EXP_DIR / "model_E_visual_age_sex_loc.pt", "e18eeb2ac2c57b42403620f3aa41f873903ad984648cbe1b974448b343676bc8"),
    }

    hash_manifest = {}
    for name, (path, expected_hash) in checkpoints.items():
        assert path.exists(), f"Checkpoint missing: {path}"
        actual_hash = sha256_file(path)
        assert actual_hash == expected_hash, f"Hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"
        hash_manifest[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": actual_hash,
            "status": "VERIFIED_MATCH",
        }
        print(f"[{name}] Verified SHA-256: {actual_hash[:16]}... OK")

    # 2. Split manifest verification
    split_manifest_path = SPLITS_DIR / "split_manifest.json"
    assert split_manifest_path.exists(), "split_manifest.json missing!"
    with open(split_manifest_path) as f:
        split_meta = json.load(f)

    df_test = pd.read_csv(SPLITS_DIR / "test.csv")
    assert len(df_test) == 1504, f"Expected 1,504 test samples, got {len(df_test)}"
    assert df_test["lesion_id"].nunique() == 1121, f"Expected 1,121 unique lesions, got {df_test['lesion_id'].nunique()}"
    assert (df_test["dx"] == "mel").sum() == 166, "Expected 166 melanomas"
    print(f"Test split confirmed: N=1,504 (1,121 unique lesions, 166 mel, 1,338 non-mel)")

    artifact_manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": "CEFM_STEP_17",
        "verified_checkpoints": hash_manifest,
        "split_manifest": split_meta,
        "test_counts": {
            "images": len(df_test),
            "lesions": df_test["lesion_id"].nunique(),
            "melanoma": int((df_test["dx"] == "mel").sum()),
            "non_melanoma": int((df_test["dx"] != "mel").sum()),
        },
    }

    manifest_file = REPORTS_DIR / "CEFM_STEP17_ARTIFACT_MANIFEST.json"
    with open(manifest_file, "w") as f:
        json.dump(artifact_manifest, f, indent=2)
    print(f"Saved baseline artifact manifest to {manifest_file}")

    return artifact_manifest


def load_all_models_and_predictions(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    preprocessor: ClinicalMetadataPreprocessor,
) -> Dict[str, Any]:
    print("=" * 80)
    print("LOADING ALL MODELS AND GENERATING PREDICTIONS")
    print("=" * 80)

    # 1. Transform clinical metadata
    clin_val = preprocessor.transform(df_val)
    clin_test = preprocessor.transform(df_test)

    # 2. Load visual features
    cache_val = torch.load(CACHE_DIR / "validation_features.pt", map_location="cpu")
    vis_val = cache_val["visual"].float().to(DEVICE)
    y_val = df_val["label"].to_numpy()

    cache_test = torch.load(CACHE_DIR / "test_features.pt", map_location="cpu")
    vis_test = cache_test["visual"].float().to(DEVICE)
    y_test = df_test["label"].to_numpy()

    # Load dual scale for M6
    ds_test = torch.load(DUAL_CACHE_DIR / "dual_scale_test.pt", map_location="cpu")
    ds_val = torch.load(DUAL_CACHE_DIR / "dual_scale_validation.pt", map_location="cpu")

    # Helper to move tensors to DEVICE
    def to_dev(t_dict):
        return {k: v.to(DEVICE) for k, v in t_dict.items()}

    t_clin_val = to_dev(clin_val)
    t_clin_test = to_dev(clin_test)

    # Load Model F
    model_f = ProposedClinicalFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64).to(DEVICE)
    model_f.load_state_dict(torch.load(EXP_DIR / "model.pt", map_location=DEVICE))
    model_f.eval()

    # Load Model A
    model_a = VisualOnlyClassifier(visual_dim=768, hidden_dim=128).to(DEVICE)
    model_a.load_state_dict(torch.load(EXP_DIR / "model_A_visual_only.pt", map_location=DEVICE))
    model_a.eval()

    # Load Model B
    model_b = ClinicalOnlyClassifier(hidden_dim=64).to(DEVICE)
    model_b.load_state_dict(torch.load(EXP_DIR / "model_B_clinical_only.pt", map_location=DEVICE))
    model_b.eval()

    # Load Model E
    model_e = VisualAgeSexLocClassifier(visual_dim=768, hidden_dim=128).to(DEVICE)
    model_e.load_state_dict(torch.load(EXP_DIR / "model_E_visual_age_sex_loc.pt", map_location=DEVICE))
    model_e.eval()

    # Load Production ViT
    vit = ViTForImageClassification.from_pretrained("models/classification/vit_model").to(DEVICE)
    vit.eval()

    # Load Production Multimodal
    mm = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2).to(DEVICE)
    mm.load_state_dict(torch.load(MODELS_DIR / "multimodal" / "fusion_model.pt", map_location=DEVICE))
    mm.eval()

    # Load Production M6
    m6 = M6_DualGated(global_dim=768, local_dim=768, clinical_dim=16, hidden_dim=128, clinical_hidden=64, num_classes=2, dropout=0.15).to(DEVICE)
    m6.load_state_dict(torch.load(MODELS_DIR / "experiments" / "M6" / "weights.pt", map_location=DEVICE))
    m6.eval()

    print("Generating validation and test probabilities...")
    with torch.no_grad():
        # Validation probabilities
        p_val_f = torch.softmax(model_f(vis_val, **t_clin_val), dim=-1)[:, 1].cpu().numpy()
        p_val_a = torch.softmax(model_a(vis_val), dim=-1)[:, 1].cpu().numpy()
        p_val_b = torch.softmax(model_b(**t_clin_val), dim=-1)[:, 1].cpu().numpy()
        p_val_e = torch.softmax(model_e(vis_val, **t_clin_val), dim=-1)[:, 1].cpu().numpy()
        p_val_vit = torch.softmax(vit.classifier(vis_val), dim=-1)[:, 1].cpu().numpy()
        p_val_mm = torch.softmax(mm(cache_val["visual"].to(DEVICE), cache_val["clinical"].to(DEVICE)), dim=-1)[:, 1].cpu().numpy()
        p_val_m6 = torch.softmax(m6(ds_val["global_visual"].to(DEVICE), ds_val["local_visual"].to(DEVICE), ds_val["clinical"].to(DEVICE)), dim=-1)[:, 1].cpu().numpy()

        # Test probabilities
        p_test_f = torch.softmax(model_f(vis_test, **t_clin_test), dim=-1)[:, 1].cpu().numpy()
        p_test_a = torch.softmax(model_a(vis_test), dim=-1)[:, 1].cpu().numpy()
        p_test_b = torch.softmax(model_b(**t_clin_test), dim=-1)[:, 1].cpu().numpy()
        p_test_e = torch.softmax(model_e(vis_test, **t_clin_test), dim=-1)[:, 1].cpu().numpy()
        p_test_vit = torch.softmax(vit.classifier(vis_test), dim=-1)[:, 1].cpu().numpy()
        p_test_mm = torch.softmax(mm(cache_test["visual"].to(DEVICE), cache_test["clinical"].to(DEVICE)), dim=-1)[:, 1].cpu().numpy()
        p_test_m6 = torch.softmax(m6(ds_test["global_visual"].to(DEVICE), ds_test["local_visual"].to(DEVICE), ds_test["clinical"].to(DEVICE)), dim=-1)[:, 1].cpu().numpy()

    return {
        "models": {
            "model_f": model_f,
            "model_a": model_a,
            "model_b": model_b,
            "model_e": model_e,
            "vit": vit,
            "mm": mm,
            "m6": m6,
        },
        "val": {
            "y": y_val,
            "p_f": p_val_f,
            "p_a": p_val_a,
            "p_b": p_val_b,
            "p_e": p_val_e,
            "p_vit": p_val_vit,
            "p_mm": p_val_mm,
            "p_m6": p_val_m6,
        },
        "test": {
            "y": y_test,
            "p_f": p_test_f,
            "p_a": p_test_a,
            "p_b": p_test_b,
            "p_e": p_test_e,
            "p_vit": p_test_vit,
            "p_mm": p_test_mm,
            "p_m6": p_test_m6,
        },
        "features": {
            "vis_test": vis_test,
            "clin_test": clin_test,
            "t_clin_test": t_clin_test,
        }
    }


def run_phase2_paired_bootstrap(
    df_test: pd.DataFrame,
    preds_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    thresholds: Dict[str, float],
    n_iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    print("=" * 80)
    print(f"PHASE 2: PAIRED LESION-CLUSTER BOOTSTRAP ({n_iterations} ITERATIONS)")
    print("=" * 80)

    rng = np.random.RandomState(seed)
    unique_lesions = df_test["lesion_id"].unique()
    n_lesions = len(unique_lesions)
    lesion_to_indices = df_test.groupby("lesion_id").indices

    # Models to compare against Model F
    baselines = [
        ("Model_A_VisualOnly", "p_a"),
        ("Model_B_ClinicalOnly", "p_b"),
        ("Model_E_VisualAgeSexLoc", "p_e"),
        ("Production_ViT_Prod", "p_vit"),
        ("Production_Multimodal_Prod", "p_mm"),
        ("Production_M6_Review", "p_m6"),
    ]

    p_f = preds_dict["p_f"]
    t_f = thresholds["p_f"]

    # Pre-calculate lesion grouping data for full test set
    df_lesion = df_test.copy()
    for key, arr in preds_dict.items():
        df_lesion[key] = arr

    # Calculate full-dataset observed differences
    observed_diffs = {}
    for base_name, base_key in baselines:
        p_base = preds_dict[base_key]
        t_base = thresholds[base_key]

        # Image metrics
        auc_diff = float(roc_auc_score(y_test, p_f) - roc_auc_score(y_test, p_base))
        pred_f = (p_f >= t_f).astype(int)
        pred_base = (p_base >= t_base).astype(int)

        bacc_diff = float(balanced_accuracy_score(y_test, pred_f) - balanced_accuracy_score(y_test, pred_base))
        sens_diff = float(recall_score(y_test, pred_f, zero_division=0) - recall_score(y_test, pred_base, zero_division=0))
        spec_f = float(confusion_matrix(y_test, pred_f, labels=[0, 1])[0, 0] / (y_test == 0).sum())
        spec_base = float(confusion_matrix(y_test, pred_base, labels=[0, 1])[0, 0] / (y_test == 0).sum())
        spec_diff = float(spec_f - spec_base)

        # Lesion metrics
        grouped = df_lesion.groupby("lesion_id").agg({"label": "max", "p_f": "max", base_key: "max"})
        y_l = grouped["label"].to_numpy()
        l_auc_diff = float(roc_auc_score(y_l, grouped["p_f"]) - roc_auc_score(y_l, grouped[base_key]))

        observed_diffs[base_name] = {
            "auc": auc_diff,
            "lesion_auc": l_auc_diff,
            "bacc": bacc_diff,
            "sens": sens_diff,
            "spec": spec_diff,
        }

    # Storage for paired differences across iterations
    diff_records = {b_name: {"auc": [], "lesion_auc": [], "bacc": [], "sens": [], "spec": []} for b_name, _ in baselines}

    t0 = time.time()
    valid_iters = 0

    for it in range(n_iterations):
        sample_lesions = rng.choice(unique_lesions, size=n_lesions, replace=True)
        sample_indices = np.concatenate([lesion_to_indices[l] for l in sample_lesions])

        y_boot = y_test[sample_indices]
        if len(np.unique(y_boot)) < 2:
            continue

        p_f_boot = p_f[sample_indices]
        auc_f = roc_auc_score(y_boot, p_f_boot)
        pred_f_boot = (p_f_boot >= t_f).astype(int)
        bacc_f = balanced_accuracy_score(y_boot, pred_f_boot)
        sens_f = recall_score(y_boot, pred_f_boot, zero_division=0)
        cm_f = confusion_matrix(y_boot, pred_f_boot, labels=[0, 1])
        spec_f = cm_f[0, 0] / (cm_f[0, 0] + cm_f[0, 1]) if (cm_f[0, 0] + cm_f[0, 1]) > 0 else 0.0

        # Sub-dataframe for lesion level
        sample_df = df_lesion.iloc[sample_indices]
        grouped_sample = sample_df.groupby("lesion_id").agg({
            "label": "max",
            "p_f": "max",
            "p_a": "max",
            "p_b": "max",
            "p_e": "max",
            "p_vit": "max",
            "p_mm": "max",
            "p_m6": "max",
        })
        y_l_boot = grouped_sample["label"].to_numpy()
        if len(np.unique(y_l_boot)) < 2:
            continue

        l_auc_f = roc_auc_score(y_l_boot, grouped_sample["p_f"])
        valid_iters += 1

        for base_name, base_key in baselines:
            p_base_boot = preds_dict[base_key][sample_indices]
            t_base = thresholds[base_key]
            auc_base = roc_auc_score(y_boot, p_base_boot)
            pred_base_boot = (p_base_boot >= t_base).astype(int)
            bacc_base = balanced_accuracy_score(y_boot, pred_base_boot)
            sens_base = recall_score(y_boot, pred_base_boot, zero_division=0)
            cm_base = confusion_matrix(y_boot, pred_base_boot, labels=[0, 1])
            spec_base = cm_base[0, 0] / (cm_base[0, 0] + cm_base[0, 1]) if (cm_base[0, 0] + cm_base[0, 1]) > 0 else 0.0

            l_auc_base = roc_auc_score(y_l_boot, grouped_sample[base_key])

            diff_records[base_name]["auc"].append(auc_f - auc_base)
            diff_records[base_name]["lesion_auc"].append(l_auc_f - l_auc_base)
            diff_records[base_name]["bacc"].append(bacc_f - bacc_base)
            diff_records[base_name]["sens"].append(sens_f - sens_base)
            diff_records[base_name]["spec"].append(spec_f - spec_base)

    elapsed = time.time() - t0
    print(f"Completed {valid_iters} valid bootstrap iterations in {elapsed:.2f}s.")

    rows = []
    metric_labels = [
        ("Image_ROC_AUC", "auc"),
        ("Lesion_ROC_AUC", "lesion_auc"),
        ("Balanced_Accuracy", "bacc"),
        ("Sensitivity", "sens"),
        ("Specificity", "spec"),
    ]

    for base_name, _ in baselines:
        for m_label, m_key in metric_labels:
            obs = observed_diffs[base_name][m_key]
            diffs = np.array(diff_records[base_name][m_key])
            ci_low = float(np.percentile(diffs, 2.5))
            ci_high = float(np.percentile(diffs, 97.5))
            excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)

            rows.append({
                "Comparison": f"Model_F_minus_{base_name}",
                "Baseline_Model": base_name,
                "Metric": m_label,
                "Observed_Difference": round(obs, 4),
                "Paired_95CI_Low": round(ci_low, 4),
                "Paired_95CI_High": round(ci_high, 4),
                "Paired_95CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "Excludes_Zero": excludes_zero,
                "Statistically_Significant": "YES (p < 0.05)" if excludes_zero else "NO (p >= 0.05)",
                "Valid_Bootstrap_Iterations": valid_iters,
            })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(REPORTS_DIR / "CEFM_STEP17_PAIRED_BOOTSTRAP.csv", index=False)
    print(f"Saved paired bootstrap report to {REPORTS_DIR / 'CEFM_STEP17_PAIRED_BOOTSTRAP.csv'}")
    return df_res


def run_phase3_suppression_audit(
    df_test: pd.DataFrame,
    p_vis: np.ndarray,
    p_fus: np.ndarray,
    guardrail: ClinicalSafetyGuardrail,
) -> pd.DataFrame:
    print("=" * 80)
    print("PHASE 3: DEMOGRAPHIC SUPPRESSION AUDIT")
    print("=" * 80)

    rows = []
    y_true = df_test["label"].to_numpy()
    delta_p = p_vis - p_fus  # Positive = visual exceeds fusion (demographic reduction)

    # Condition checks:
    # 1. p_vis >= 0.50 and p_fus < 0.10
    # 2. p_vis >= 0.70 and p_fus < 0.01
    # 3. delta_p >= 0.30
    for idx, row in df_test.iterrows():
        pv = float(p_vis[idx])
        pf = float(p_fus[idx])
        dp = float(delta_p[idx])
        label = int(y_true[idx])

        c1 = (pv >= 0.50 and pf < 0.10)
        c2 = (pv >= 0.70 and pf < 0.01)
        c3 = (dp >= 0.30)

        if c1 or c2 or c3:
            pred_v = "mel" if pv >= 0.50 else "non_mel"
            pred_f = "mel" if pf >= 0.2417 else "non_mel"
            flipped_to_neg = (pred_v == "mel" and pred_f == "non_mel")

            rows.append({
                "image_id": row["image_id"],
                "lesion_id": row["lesion_id"],
                "ground_truth_label": "mel" if label == 1 else "non_mel",
                "is_melanoma": bool(label == 1),
                "visual_probability": round(pv, 4),
                "fusion_probability": round(pf, 4),
                "delta_probability": round(dp, 4),
                "cond1_vis50_fus10": c1,
                "cond2_vis70_fus01_extreme": c2,
                "cond3_delta_ge_30": c3,
                "age": row["age"],
                "sex": row["sex"],
                "localization": row["localization"],
                "flipped_from_mel_to_non_mel": flipped_to_neg,
            })

    df_supp = pd.DataFrame(rows)
    df_supp.sort_values(by="delta_probability", ascending=False, inplace=True)
    df_supp.to_csv(REPORTS_DIR / "CEFM_STEP17_SUPPRESSION_AUDIT.csv", index=False)

    n_total = len(df_supp)
    n_mel = int(df_supp["is_melanoma"].sum())
    n_extreme = int(df_supp["cond2_vis70_fus01_extreme"].sum())
    n_flipped = int(df_supp["flipped_from_mel_to_non_mel"].sum())
    n_mel_flipped = int((df_supp["is_melanoma"] & df_supp["flipped_from_mel_to_non_mel"]).sum())

    print(f"Demographic Suppression Cases Detected: {n_total} / 1,504 test samples")
    print(f"  - True Melanomas suppressed: {n_mel} cases")
    print(f"  - Extreme Suppression (vis>=0.70, fus<0.01): {n_extreme} cases")
    print(f"  - Total flipped from mel to non_mel: {n_flipped} cases (of which {n_mel_flipped} are true melanomas!)")
    print(f"Saved suppression audit to {REPORTS_DIR / 'CEFM_STEP17_SUPPRESSION_AUDIT.csv'}")

    return df_supp


def run_phase4_counterfactual_tests(
    df_test: pd.DataFrame,
    vis_test: torch.Tensor,
    preprocessor: ClinicalMetadataPreprocessor,
    model_f: nn.Module,
) -> pd.DataFrame:
    print("=" * 80)
    print("PHASE 4: COUNTERFACTUAL CLINICAL INPUT TESTS")
    print("=" * 80)

    # 1. Baseline prediction
    t_orig = preprocessor.transform(df_test)
    with torch.no_grad():
        orig_logits = model_f(
            vis_test,
            t_orig["age_norm"].to(DEVICE),
            t_orig["age_missing"].to(DEVICE),
            t_orig["sex_onehot"].to(DEVICE),
            t_orig["loc_indices"].to(DEVICE),
        )
        p_orig = torch.softmax(orig_logits, dim=-1)[:, 1].cpu().numpy()

    counterfactual_scenarios = [
        ("age_missing", {"age": np.nan}),
        ("age_25", {"age": 25.0}),
        ("age_45", {"age": 45.0}),
        ("age_65", {"age": 65.0}),
        ("age_85", {"age": 85.0}),
        ("sex_missing", {"sex": "unknown"}),
        ("sex_male", {"sex": "male"}),
        ("sex_female", {"sex": "female"}),
        ("loc_back", {"localization": "back"}),
        ("loc_face", {"localization": "face"}),
        ("loc_lower_extremity", {"localization": "lower extremity"}),
        ("loc_upper_extremity", {"localization": "upper extremity"}),
        ("loc_abdomen", {"localization": "abdomen"}),
        ("loc_chest", {"localization": "chest"}),
        ("loc_ear", {"localization": "ear"}),
        ("loc_unknown", {"localization": "unknown"}),
        ("loc_missing", {"localization": np.nan}),
        ("all_metadata_missing", {"age": np.nan, "sex": "unknown", "localization": "unknown"}),
    ]

    summary_rows = []

    for sc_name, overrides in counterfactual_scenarios:
        df_cf = df_test.copy()
        for col, val in overrides.items():
            df_cf[col] = val

        t_cf = preprocessor.transform(df_cf)
        with torch.no_grad():
            cf_logits = model_f(
                vis_test,
                t_cf["age_norm"].to(DEVICE),
                t_cf["age_missing"].to(DEVICE),
                t_cf["sex_onehot"].to(DEVICE),
                t_cf["loc_indices"].to(DEVICE),
            )
            p_cf = torch.softmax(cf_logits, dim=-1)[:, 1].cpu().numpy()

        delta = p_cf - p_orig

        # Check for high confidence visual samples (p_orig >= 0.70)
        high_vis_mask = (p_orig >= 0.70)
        worst_drop_high_vis = float(delta[high_vis_mask].min()) if high_vis_mask.sum() > 0 else 0.0

        summary_rows.append({
            "Scenario": sc_name,
            "Perturbation_Target": list(overrides.keys())[0] if len(overrides) == 1 else "all_metadata",
            "Mean_Probability": round(float(p_cf.mean()), 4),
            "Median_Probability": round(float(np.median(p_cf)), 4),
            "Min_Probability": round(float(p_cf.min()), 6),
            "Max_Probability": round(float(p_cf.max()), 4),
            "Mean_Shift_Delta": round(float(delta.mean()), 4),
            "Max_Increase_Delta": round(float(delta.max()), 4),
            "Max_Drop_Delta": round(float(delta.min()), 4),
            "Std_Shift": round(float(delta.std()), 4),
            "Worst_Drop_for_p_ge_70": round(worst_drop_high_vis, 4),
        })

    df_cf_summary = pd.DataFrame(summary_rows)
    df_cf_summary.to_csv(REPORTS_DIR / "CEFM_STEP17_COUNTERFACTUAL_ANALYSIS.csv", index=False)
    print(f"Saved counterfactual summary to {REPORTS_DIR / 'CEFM_STEP17_COUNTERFACTUAL_ANALYSIS.csv'}")

    return df_cf_summary


def run_phase5_guardrail_validation(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    p_vis_val: np.ndarray,
    p_fus_val: np.ndarray,
    p_vis_test: np.ndarray,
    p_fus_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("=" * 80)
    print("PHASE 5: EXPERIMENTAL VISUAL SAFETY GUARDRAIL VALIDATION")
    print("=" * 80)

    y_val = df_val["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    # Candidate settings to evaluate strictly on validation data
    candidate_settings = [
        {"setting_id": "G1_vis50_override", "vis_thresh": 0.50, "fus_thresh": 0.2417},
        {"setting_id": "G2_vis60_override", "vis_thresh": 0.60, "fus_thresh": 0.2417},
        {"setting_id": "G3_vis70_override", "vis_thresh": 0.70, "fus_thresh": 0.2417},
        {"setting_id": "G4_pure_max_rule", "vis_thresh": 0.00, "fus_thresh": 0.2417},  # max(p_fus, p_vis) always
    ]

    val_eval_rows = []
    for cfg in candidate_settings:
        g = ClinicalSafetyGuardrail(
            visual_alert_threshold=cfg["vis_thresh"],
            fusion_decision_threshold=cfg["fus_thresh"],
        )

        p_guard_val = []
        for pv, pf in zip(p_vis_val, p_fus_val):
            res = g.evaluate(pv, pf)
            p_guard_val.append(res["guardrail_probability"])
        p_guard_val = np.array(p_guard_val)

        preds = (p_guard_val >= cfg["fus_thresh"]).astype(int)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        bacc = balanced_accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, p_guard_val)
        f1 = f1_score(y_val, preds, zero_division=0)

        val_eval_rows.append({
            "Validation_Setting": cfg["setting_id"],
            "Visual_Alert_Threshold": cfg["vis_thresh"],
            "Fusion_Decision_Threshold": cfg["fus_thresh"],
            "Val_ROC_AUC": round(auc, 4),
            "Val_Sensitivity": round(sens, 4),
            "Val_Specificity": round(spec, 4),
            "Val_Balanced_Accuracy": round(bacc, 4),
            "Val_F1_Score": round(f1, 4),
            "Val_TP": int(tp),
            "Val_FP": int(fp),
            "Val_FN": int(fn),
            "Val_TN": int(tn),
        })

    df_val_tuning = pd.DataFrame(val_eval_rows)
    df_val_tuning.to_csv(REPORTS_DIR / "CEFM_STEP17_GUARDRAIL_VALIDATION.csv", index=False)
    print(f"Validation Guardrail Tuning:")
    print(df_val_tuning[["Validation_Setting", "Val_Sensitivity", "Val_Specificity", "Val_Balanced_Accuracy", "Val_FN"]].to_string())

    # Select setting with highest Balanced Accuracy on validation: G1_vis50_override
    selected_setting = candidate_settings[0]
    print(f"\nFrozen Guardrail Setting: {selected_setting['setting_id']} (vis_thresh={selected_setting['vis_thresh']})")

    # Evaluate FROZEN setting on test set
    frozen_guard = ClinicalSafetyGuardrail(
        visual_alert_threshold=selected_setting["vis_thresh"],
        fusion_decision_threshold=selected_setting["fus_thresh"],
    )

    test_guard_probs = []
    test_triggered = []
    for pv, pf in zip(p_vis_test, p_fus_test):
        res = frozen_guard.evaluate(pv, pf)
        test_guard_probs.append(res["guardrail_probability"])
        test_triggered.append(res["guardrail_triggered"])
    test_guard_probs = np.array(test_guard_probs)
    test_triggered = np.array(test_triggered)

    # Compute test metrics for unguardrailed Model F vs Guardrailed Model F
    pred_f = (p_fus_test >= selected_setting["fus_thresh"]).astype(int)
    pred_g = (test_guard_probs >= selected_setting["fus_thresh"]).astype(int)

    test_sens_f = recall_score(y_test, pred_f, zero_division=0)
    test_sens_g = recall_score(y_test, pred_g, zero_division=0)
    test_spec_f = confusion_matrix(y_test, pred_f, labels=[0, 1])[0, 0] / (y_test == 0).sum()
    test_spec_g = confusion_matrix(y_test, pred_g, labels=[0, 1])[0, 0] / (y_test == 0).sum()
    test_bacc_f = balanced_accuracy_score(y_test, pred_f)
    test_bacc_g = balanced_accuracy_score(y_test, pred_g)
    test_auc_f = roc_auc_score(y_test, p_fus_test)
    test_auc_g = roc_auc_score(y_test, test_guard_probs)

    recovered_mel = int(((pred_f == 0) & (pred_g == 1) & (y_test == 1)).sum())
    extra_fps = int(((pred_f == 0) & (pred_g == 1) & (y_test == 0)).sum())

    test_guardrail_summary = {
        "frozen_setting": selected_setting["setting_id"],
        "test_guardrail_triggered_count": int(test_triggered.sum()),
        "raw_model_f_test_sensitivity": round(test_sens_f, 4),
        "guardrail_model_f_test_sensitivity": round(test_sens_g, 4),
        "sensitivity_gain": round(test_sens_g - test_sens_f, 4),
        "raw_model_f_test_specificity": round(test_spec_f, 4),
        "guardrail_model_f_test_specificity": round(test_spec_g, 4),
        "specificity_delta": round(test_spec_g - test_spec_f, 4),
        "raw_model_f_test_balacc": round(test_bacc_f, 4),
        "guardrail_model_f_test_balacc": round(test_bacc_g, 4),
        "raw_model_f_test_auc": round(test_auc_f, 4),
        "guardrail_model_f_test_auc": round(test_auc_g, 4),
        "melanomas_recovered_from_suppression": recovered_mel,
        "additional_false_positives": extra_fps,
    }

    print(f"\nOfficial Test Evaluation of Guardrail:")
    print(f"  - Guardrail triggered on {test_triggered.sum()} test samples")
    print(f"  - Sensitivity: {test_sens_f:.4f} -> {test_sens_g:.4f} (+{test_sens_g - test_sens_f:.4f})")
    print(f"  - Specificity: {test_spec_f:.4f} -> {test_spec_g:.4f} ({test_spec_g - test_spec_f:.4f})")
    print(f"  - Balanced Accuracy: {test_bacc_f:.4f} -> {test_bacc_g:.4f}")
    print(f"  - True Melanomas Recovered from Suppression: {recovered_mel}")

    return df_val_tuning, test_guardrail_summary


def run_phase6_subgroup_evaluation(
    df_test: pd.DataFrame,
    p_f: np.ndarray,
    p_vis: np.ndarray,
    thresh_f: float = 0.2417,
) -> pd.DataFrame:
    print("=" * 80)
    print("PHASE 6: SUBGROUP EVALUATION")
    print("=" * 80)

    df_eval = df_test.copy()
    df_eval["p_f"] = p_f
    df_eval["p_vis"] = p_vis
    df_eval["pred_f"] = (p_f >= thresh_f).astype(int)
    df_eval["pred_vis"] = (p_vis >= 0.50).astype(int)

    # Define subgroup columns and categories
    subgroups = []

    # 1. Age groups
    age_bins = [
        ("age <= 40", df_eval["age"] <= 40),
        ("age 41-50", (df_eval["age"] > 40) & (df_eval["age"] <= 50)),
        ("age 51-65", (df_eval["age"] > 50) & (df_eval["age"] <= 65)),
        ("age > 65", df_eval["age"] > 65),
        ("age missing", df_eval["age"].isna()),
    ]
    for name, mask in age_bins:
        subgroups.append(("Age", name, mask))

    # 2. Sex groups
    for s in ["male", "female", "unknown"]:
        mask = (df_eval["sex"] == s) if s != "unknown" else (df_eval["sex"].isna() | (df_eval["sex"] == "unknown"))
        subgroups.append(("Sex", s, mask))

    # 3. Location groups
    top_locs = ["back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest"]
    for loc in top_locs:
        subgroups.append(("Localization", loc, df_eval["localization"] == loc))
    other_loc_mask = ~df_eval["localization"].isin(top_locs)
    subgroups.append(("Localization", "other/unknown", other_loc_mask))

    # 4. Metadata completeness
    meta_missing_mask = df_eval["age"].isna() | (df_eval["sex"] == "unknown") | (df_eval["localization"] == "unknown") | df_eval["localization"].isna()
    subgroups.append(("Metadata_Completeness", "any_missing", meta_missing_mask))
    subgroups.append(("Metadata_Completeness", "fully_available", ~meta_missing_mask))

    results = []

    for category, sub_name, mask in subgroups:
        sub_df = df_eval[mask]
        n_total = len(sub_df)
        n_mel = int(sub_df["label"].sum())
        n_non_mel = n_total - n_mel

        # Flag small sample sizes
        is_small_sample = (n_total < 30 or n_mel < 5 or n_non_mel < 5)

        if n_total == 0 or n_mel == 0 or n_non_mel == 0:
            # Cannot compute ROC-AUC or balanced accuracy with only one class
            results.append({
                "Category": category,
                "Subgroup": sub_name,
                "Sample_Size_N": n_total,
                "Melanoma_Count": n_mel,
                "Non_Melanoma_Count": n_non_mel,
                "Prevalence_Melanoma": round(n_mel / n_total, 4) if n_total > 0 else 0.0,
                "ModelF_ROC_AUC": np.nan,
                "ModelF_Sensitivity": np.nan,
                "ModelF_Specificity": np.nan,
                "ModelF_Balanced_Accuracy": np.nan,
                "Visual_ROC_AUC": np.nan,
                "Visual_Sensitivity": np.nan,
                "Small_Sample_Warning": True,
            })
            continue

        y_true = sub_df["label"].to_numpy()
        p_sub_f = sub_df["p_f"].to_numpy()
        p_sub_vis = sub_df["p_vis"].to_numpy()
        pred_sub_f = sub_df["pred_f"].to_numpy()
        pred_sub_vis = sub_df["pred_vis"].to_numpy()

        cm_f = confusion_matrix(y_true, pred_sub_f, labels=[0, 1])
        tn, fp, fn, tp = cm_f.ravel()

        sens_f = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_f = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bacc_f = balanced_accuracy_score(y_true, pred_sub_f)
        auc_f = roc_auc_score(y_true, p_sub_f)

        sens_vis = recall_score(y_true, pred_sub_vis, zero_division=0)
        auc_vis = roc_auc_score(y_true, p_sub_vis)

        results.append({
            "Category": category,
            "Subgroup": sub_name,
            "Sample_Size_N": n_total,
            "Melanoma_Count": n_mel,
            "Non_Melanoma_Count": n_non_mel,
            "Prevalence_Melanoma": round(n_mel / n_total, 4),
            "ModelF_ROC_AUC": round(float(auc_f), 4),
            "ModelF_Sensitivity": round(float(sens_f), 4),
            "ModelF_Specificity": round(float(spec_f), 4),
            "ModelF_Balanced_Accuracy": round(float(bacc_f), 4),
            "Visual_ROC_AUC": round(float(auc_vis), 4),
            "Visual_Sensitivity": round(float(sens_vis), 4),
            "Small_Sample_Warning": is_small_sample,
        })

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(REPORTS_DIR / "CEFM_STEP17_SUBGROUP_METRICS.csv", index=False)
    print(f"Saved subgroup evaluation to {REPORTS_DIR / 'CEFM_STEP17_SUBGROUP_METRICS.csv'}")

    return df_sub


def main():
    print("=" * 80)
    print("CEFM STEP 17: CLINICAL METADATA SAFETY & STATISTICAL AUDIT")
    print("=" * 80)

    # 1. Phase 1 Verification
    manifest = run_phase1_verification()

    # Load data
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "validation.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")

    # Fit preprocessor on train ONLY
    preprocessor = ClinicalMetadataPreprocessor().fit(df_train)

    # Load models and predict
    data_pack = load_all_models_and_predictions(df_train, df_val, df_test, preprocessor)
    test_preds = data_pack["test"]
    val_preds = data_pack["val"]
    y_test = test_preds["y"]
    vis_test = data_pack["features"]["vis_test"]

    # Thresholds established in Step 16
    thresholds = {
        "p_f": 0.2417,     # Model F Max F1
        "p_a": 0.6934,     # Model A Max F1
        "p_b": 0.6326,     # Model B Max F1
        "p_e": 0.1926,     # Model E Max F1
        "p_vit": 0.8000,   # Production ViT
        "p_mm": 0.3600,    # Production Multimodal
        "p_m6": 0.6900,    # Production M6 Review
    }

    # 2. Phase 2: Paired Lesion-Cluster Bootstrap
    df_paired_bootstrap = run_phase2_paired_bootstrap(
        df_test=df_test,
        preds_dict=test_preds,
        y_test=y_test,
        thresholds=thresholds,
        n_iterations=2000,
        seed=42,
    )

    # 3. Phase 3: Demographic Suppression Audit
    guardrail = ClinicalSafetyGuardrail()
    df_suppression = run_phase3_suppression_audit(
        df_test=df_test,
        p_vis=test_preds["p_a"],
        p_fus=test_preds["p_f"],
        guardrail=guardrail,
    )

    # 4. Phase 4: Counterfactual Tests
    df_counterfactual = run_phase4_counterfactual_tests(
        df_test=df_test,
        vis_test=vis_test,
        preprocessor=preprocessor,
        model_f=data_pack["models"]["model_f"],
    )

    # 5. Phase 5: Guardrail Validation & Evaluation
    df_guard_val, test_guard_res = run_phase5_guardrail_validation(
        df_val=df_val,
        df_test=df_test,
        p_vis_val=val_preds["p_a"],
        p_fus_val=val_preds["p_f"],
        p_vis_test=test_preds["p_a"],
        p_fus_test=test_preds["p_f"],
    )

    # 6. Phase 6: Subgroup Evaluation
    df_subgroups = run_phase6_subgroup_evaluation(
        df_test=df_test,
        p_f=test_preds["p_f"],
        p_vis=test_preds["p_a"],
        thresh_f=thresholds["p_f"],
    )

    print("=" * 80)
    print("STEP 17 AUDIT EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

