"""
CEFM Step 20: Final Research Consolidation, Reproducibility, and Evidence Audit.

Builds:
1. reports/CEFM_STEP20_FINAL_MODEL_COMPARISON.csv
2. reports/CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    balanced_accuracy_score,
    brier_score_loss,
)

REPO_ROOT = Path("D:/pytest_cache")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REPORTS_DIR = REPO_ROOT / "reports"
SPLITS_DIR = REPO_ROOT / "datasets/Ham10000/processed_lesion_disjoint"
CACHE_DIR = REPO_ROOT / "data/multimodal_cache"
DUAL_CACHE_DIR = REPO_ROOT / "data/dual_scale_cache"
MODELS_DIR = REPO_ROOT / "models"

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
    "model_c": (
        REPO_ROOT / "models/experiments/step18_constrained_fusion/model_C_constrained_residual.pt",
        "eeb04571506a50bf6296547a1c7080d34c9ab625e8f7950f7f91527c65a59d67",
    ),
    "model_d": (
        REPO_ROOT / "models/experiments/step18_constrained_fusion/model_D_gated_positive.pt",
        "98e1c74525569ff40834a23d2d7b5849efad7bd844a13964728c738a8fc8d179",
    ),
}

import hashlib
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float) -> dict:
    preds = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    bacc = float(balanced_accuracy_score(y_true, preds))
    brier = float(brier_score_loss(y_true, y_prob))

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(rec_arr, prec_arr))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": bacc,
        "brier_score": brier,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, thresh: float, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    samples = {"roc_auc": [], "sensitivity": [], "specificity": [], "balanced_accuracy": [], "f1": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True), rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        m = compute_metrics(y_true[idx], y_prob[idx], thresh)
        for k in samples:
            samples[k].append(m[k])

    return {k: f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]" for k, v in samples.items()}

def main():
    print("=" * 80)
    print("CEFM STEP 20: FINAL MODEL COMPARISON & ARTIFACT MANIFEST GENERATION")
    print("=" * 80)

    # 1. Verify Checkpoints
    manifest_checkpoints = {}
    for name, (path, exp_hash) in EXPECTED_HASHES.items():
        act_hash = _sha256(path)
        assert act_hash == exp_hash, f"Mismatch on {name}!"
        manifest_checkpoints[name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": act_hash, "status": "VERIFIED"}

    # 2. Load Step 12 Predictions for HAM10000 Test (Production ViT, Production Multimodal, Production M6)
    df_step12_preds = pd.read_csv(REPORTS_DIR / "CEFM_STEP12_TEST_PREDICTIONS.csv")
    y_ham = df_step12_preds["ground_truth"].to_numpy()
    p_ham_vit = df_step12_preds["ViT_probability"].to_numpy()
    p_ham_multimodal = df_step12_preds["Multimodal_probability"].to_numpy()
    p_ham_m6 = df_step12_preds["M6_probability"].to_numpy()

    # Load Step 18 / 19 Predictions for Model F and Model F-G on HAM10000
    from app.ml.multimodal.clinical_metadata import ClinicalMetadataPreprocessor, ProposedClinicalFusionModel, VisualOnlyClassifier
    from app.ml.multimodal.constrained_fusion import VisualEvidencePreserver

    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_test_ham = pd.read_csv(SPLITS_DIR / "test.csv")
    prep = ClinicalMetadataPreprocessor().fit(df_train)
    clin_ham = prep.transform(df_test_ham)
    vis_ham = torch.load(CACHE_DIR / "test_features.pt", map_location="cpu", weights_only=False)["visual"].float()

    m_a = VisualOnlyClassifier()
    m_a.load_state_dict(torch.load(EXPECTED_HASHES["model_a"][0], map_location="cpu", weights_only=False))
    m_a.eval()

    m_f = ProposedClinicalFusionModel()
    m_f.load_state_dict(torch.load(EXPECTED_HASHES["model_f"][0], map_location="cpu", weights_only=False))
    m_f.eval()

    vep = VisualEvidencePreserver(visual_alert_threshold=0.50, decision_threshold=0.21032)

    with torch.no_grad():
        p_ham_vis = torch.softmax(m_a(vis_ham), dim=-1)[:, 1].numpy()
        p_ham_f = torch.softmax(m_f(vis_ham, clin_ham["age_norm"], clin_ham["age_missing"], clin_ham["sex_onehot"], clin_ham["loc_indices"]), dim=-1)[:, 1].numpy()
        p_ham_fg = np.array([vep.evaluate(pv, pf)["p_guarded"] for pv, pf in zip(p_ham_vis, p_ham_f)])

    # 3. Load External PH2 Benchmark Data ($N=200$)
    df_ph2_preds = pd.read_csv(REPORTS_DIR / "CEFM_PHASE2_STEP5_PREDICTIONS.csv")
    y_ph2 = df_ph2_preds["ground_truth_label"].to_numpy()
    p_ph2_vit = df_ph2_preds["vit_probability"].to_numpy()
    p_ph2_multimodal = df_ph2_preds["multimodal_probability"].to_numpy()
    p_ph2_m6 = df_ph2_preds["m6_probability"].to_numpy()

    # Load Model F and Model F-G on PH2
    vis_ph2 = torch.load(CACHE_DIR / "ph2_vit_features.pt", map_location="cpu", weights_only=False)["visual"].float()
    df_ph2_meta = pd.DataFrame({"age": [np.nan] * 200, "sex": ["unknown"] * 200, "localization": ["unknown"] * 200})
    clin_ph2 = prep.transform(df_ph2_meta)

    with torch.no_grad():
        p_ph2_vis = torch.softmax(m_a(vis_ph2), dim=-1)[:, 1].numpy()
        p_ph2_f = torch.softmax(m_f(vis_ph2, clin_ph2["age_norm"], clin_ph2["age_missing"], clin_ph2["sex_onehot"], clin_ph2["loc_indices"]), dim=-1)[:, 1].numpy()
        p_ph2_fg = np.array([vep.evaluate(pv, pf)["p_guarded"] for pv, pf in zip(p_ph2_vis, p_ph2_f)])

    # 4. Assemble Unified Final Comparison Matrix
    comparison_entries = [
        # Internal HAM10000 Test (N=1,504)
        ("HAM10000_Internal_Test", "Production_ViT_Screening", "Production Research Baseline", p_ham_vit, y_ham, 0.800, p_ham_vis),
        ("HAM10000_Internal_Test", "Production_Multimodal_Fusion", "Production Research Baseline", p_ham_multimodal, y_ham, 0.360, p_ham_vis),
        ("HAM10000_Internal_Test", "Production_M6_DualScale", "Production Research Baseline", p_ham_m6, y_ham, 0.690, p_ham_vis),
        ("HAM10000_Internal_Test", "Experimental_Model_F", "Experimental Challenger (Unconstrained)", p_ham_f, y_ham, 0.2417, p_ham_vis),
        ("HAM10000_Internal_Test", "Challenger_Model_FG", "Experimental Safety Wrapper (Guarded)", p_ham_fg, y_ham, 0.21032, p_ham_vis),

        # External PH2 Benchmark (N=200)
        ("PH2_External_Benchmark", "Production_ViT_Screening", "Production Research Baseline", p_ph2_vit, y_ph2, 0.800, p_ph2_vis),
        ("PH2_External_Benchmark", "Production_Multimodal_Fusion", "Production Research Baseline", p_ph2_multimodal, y_ph2, 0.360, p_ph2_vis),
        ("PH2_External_Benchmark", "Production_M6_DualScale", "Production Research Baseline", p_ph2_m6, y_ph2, 0.690, p_ph2_vis),
        ("PH2_External_Benchmark", "Experimental_Model_F", "Experimental Challenger (Unconstrained)", p_ph2_f, y_ph2, 0.2417, p_ph2_vis),
        ("PH2_External_Benchmark", "Challenger_Model_FG", "Experimental Safety Wrapper (Guarded)", p_ph2_fg, y_ph2, 0.21032, p_ph2_vis),
    ]

    records = []
    for dset, model_name, model_role, probs, targets, thresh, vis_ref in comparison_entries:
        m = compute_metrics(targets, probs, thresh)
        ci = compute_bootstrap_ci(targets, probs, thresh, n_boot=1000, seed=42)

        # Count VEP violations (vis_ref >= 0.50 and prob < thresh)
        vep_viols = int(np.sum((vis_ref >= 0.50) & (probs < thresh)))
        mel_supp = int(np.sum((vis_ref >= 0.50) & (probs < thresh) & (targets == 1)))

        records.append({
            "Evaluation_Cohort": dset,
            "Model_Name": model_name,
            "Model_Role": model_role,
            "Cohort_Sample_Count": len(targets),
            "Melanoma_Count": int(np.sum(targets == 1)),
            "Operating_Threshold": thresh,
            "ROC_AUC": round(m["roc_auc"], 4),
            "ROC_AUC_95CI": ci["roc_auc"],
            "PR_AUC": round(m["pr_auc"], 4),
            "Sensitivity": round(m["sensitivity"], 4),
            "Sensitivity_95CI": ci["sensitivity"],
            "Specificity": round(m["specificity"], 4),
            "Specificity_95CI": ci["specificity"],
            "Balanced_Accuracy": round(m["balanced_accuracy"], 4),
            "Balanced_Accuracy_95CI": ci["balanced_accuracy"],
            "Precision": round(m["precision"], 4),
            "F1_Score": round(m["f1"], 4),
            "Brier_Score": round(m["brier_score"], 4),
            "VEP_Violations": vep_viols,
            "Melanomas_Suppressed_by_Prior": mel_supp,
            "False_Negatives_Missed": m["fn"],
            "True_Positives": m["tp"],
            "False_Positives": m["fp"],
            "True_Negatives": m["tn"],
        })

    df_comp = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "CEFM_STEP20_FINAL_MODEL_COMPARISON.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"Saved final unified model comparison to {out_csv}")

    # 5. Build Final Artifact Manifest JSON
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": "CEFM_STEP_20",
        "title": "Final Research Consolidation, Reproducibility, and Evidence Audit",
        "verified_checkpoints": manifest_checkpoints,
        "operating_thresholds": {
            "Production_ViT_Screening": 0.800,
            "Production_ViT_F1": 0.900,
            "Production_Multimodal_Screening": 0.360,
            "Production_Multimodal_F1": 0.800,
            "Production_M6_Review": 0.690,
            "Production_M6_Target_Spec": 0.625,
            "Visual_Only_Baseline": 0.6934,
            "Experimental_Model_F": 0.2417,
            "Challenger_Model_FG": 0.21032,
        },
        "datasets": {
            "HAM10000_Lesion_Disjoint_Partitions": {
                "train_images": 6978,
                "validation_images": 1533,
                "test_images": 1504,
                "unique_test_lesions": 1121,
                "lesion_overlap_across_splits": 0,
            },
            "PH2_External_Benchmark": {
                "total_images": 200,
                "melanoma": 40,
                "common_nevi": 80,
                "atypical_nevi": 80,
                "metadata_status": "Demographic metadata absent",
            },
        },
        "deliverables": [
            "reports/CEFM_STEP20_FINAL_MODEL_COMPARISON.csv",
            "reports/CEFM_STEP20_REPRODUCIBILITY_AUDIT.md",
            "reports/CEFM_STEP20_GUARDRAIL_VERIFICATION.md",
            "reports/CEFM_STEP20_LIMITATIONS.md",
            "reports/CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json",
            "reports/CEFM_STEP20_FINAL_RESEARCH_REPORT.md",
            "README.md",
        ],
        "regulatory_disclaimer": "CEFM is a non-diagnostic experimental research prototype. It has not received regulatory clearance and is not intended for clinical decision-making.",
    }

    out_json = REPORTS_DIR / "CEFM_STEP20_FINAL_ARTIFACT_MANIFEST.json"
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved final artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
