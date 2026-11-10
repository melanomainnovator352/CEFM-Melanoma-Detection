"""
CEFM Phase 2 Step 4 — Final Results Consistency Audit Script.

Conducts an exhaustive consistency audit of:
1. Exact paired bootstrap differences for both F1 operating points (Table A) and Specificity-target operating points (Table B).
2. Clarification of threshold selections (e.g. Phase 1 ViT at 0.900 in Table A vs. 0.800 in Table B).
3. Statistical audit including 1,000 bootstrap resamples (seed=42), paired sampling, empirical p-values, Bonferroni and Benjamini-Hochberg FDR adjustments.
4. Independent ground-truth investigation for ISIC_0012418.
5. Verification of all 10 model checkpoint hashes.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import ViTForImageClassification

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import CLASSIFICATION_MODEL_DIR
from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.ml.multimodal.dual_scale_models import (
    M1_GlobalViT,
    M2_LocalViT,
    M3_DualVisual,
    M4_GlobalClinical,
    M5_DualConcat,
    M6_DualGated,
    M7_DualCrossAttn,
)

RANDOM_SEED = 42
BOOTSTRAP_ROUNDS = 1000

CHECKPOINT_PATHS = {
    "Phase1_ViT": CLASSIFICATION_MODEL_DIR / "vit_model" / "model.safetensors",
    "Phase1_UNet": CLASSIFICATION_MODEL_DIR.parent / "segmentation" / "unet_isic2018.pt",
    "Phase1_Fusion": CLASSIFICATION_MODEL_DIR.parent / "multimodal" / "fusion_model.pt",
    "M1": REPO_ROOT / "models" / "experiments" / "M1" / "weights.pt",
    "M2": REPO_ROOT / "models" / "experiments" / "M2" / "weights.pt",
    "M3": REPO_ROOT / "models" / "experiments" / "M3" / "weights.pt",
    "M4": REPO_ROOT / "models" / "experiments" / "M4" / "weights.pt",
    "M5": REPO_ROOT / "models" / "experiments" / "M5" / "weights.pt",
    "M6": REPO_ROOT / "models" / "experiments" / "M6" / "weights.pt",
    "M7": REPO_ROOT / "models" / "experiments" / "M7" / "weights.pt",
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def paired_bootstrap_analysis(
    targets: np.ndarray,
    probs_a: np.ndarray,
    thresh_a: float,
    probs_b: np.ndarray,
    thresh_b: float,
    resample_indices: List[np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """Compute paired bootstrap difference metrics between model A and model B."""
    diffs = {
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1_score": [],
        "balanced_accuracy": [],
        "accuracy": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    for b_idx in resample_indices:
        b_targets = targets[b_idx]
        b_pa = probs_a[b_idx]
        b_pb = probs_b[b_idx]

        b_pred_a = (b_pa >= thresh_a).astype(int)
        b_pred_b = (b_pb >= thresh_b).astype(int)

        tn_a, fp_a, fn_a, tp_a = confusion_matrix(b_targets, b_pred_a).ravel()
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(b_targets, b_pred_b).ravel()

        sens_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        sens_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0

        spec_a = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0.0
        spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0

        prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0

        f1_a = (2 * prec_a * sens_a) / (prec_a + sens_a) if (prec_a + sens_a) > 0 else 0.0
        f1_b = (2 * prec_b * sens_b) / (prec_b + sens_b) if (prec_b + sens_b) > 0 else 0.0

        acc_a = (tp_a + tn_a) / len(b_targets)
        acc_b = (tp_b + tn_b) / len(b_targets)

        bacc_a = 0.5 * (sens_a + spec_a)
        bacc_b = 0.5 * (sens_b + spec_b)

        if len(np.unique(b_targets)) == 2:
            roc_a = roc_auc_score(b_targets, b_pa)
            roc_b = roc_auc_score(b_targets, b_pb)
            pr_a = average_precision_score(b_targets, b_pa)
            pr_b = average_precision_score(b_targets, b_pb)
        else:
            roc_a, roc_b = 0.5, 0.5
            pr_a, pr_b = 0.0, 0.0

        diffs["sensitivity"].append(sens_a - sens_b)
        diffs["specificity"].append(spec_a - spec_b)
        diffs["precision"].append(prec_a - prec_b)
        diffs["f1_score"].append(f1_a - f1_b)
        diffs["balanced_accuracy"].append(bacc_a - bacc_b)
        diffs["accuracy"].append(acc_a - acc_b)
        diffs["roc_auc"].append(roc_a - roc_b)
        diffs["pr_auc"].append(pr_a - pr_b)

    results = {}
    for met, vals in diffs.items():
        arr = np.array(vals)
        mean_d = float(np.mean(arr))
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        inc_zero = bool(low <= 0.0 <= high)

        # Empirical two-sided p-value: 2 * min(P(diff <= 0), P(diff >= 0))
        p_le_0 = np.mean(arr <= 0)
        p_ge_0 = np.mean(arr >= 0)
        p_val = float(min(1.0, 2.0 * min(p_le_0, p_ge_0)))

        results[met] = {
            "mean_difference": round(mean_d, 4),
            "ci_lower": round(low, 4),
            "ci_upper": round(high, 4),
            "includes_zero": inc_zero,
            "p_value": round(p_val, 4),
            "significant_nominal_05": not inc_zero,
        }
    return results


def run_audit():
    print("=" * 80)
    print("CEFM PHASE 2 STEP 4 — FINAL RESULTS CONSISTENCY AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Audit Checkpoints
    print("\n[Audit Item 1] Checkpoint Hashes Verification:")
    hashes = {}
    for k, p in CHECKPOINT_PATHS.items():
        h = compute_sha256(p)
        hashes[k] = h
        print(f"  {k:<16}: {h}")

    # 2. Load Evaluation Data
    test_cache = torch.load("data/dual_scale_cache/dual_scale_test.pt", map_location="cpu")
    train_cache = torch.load("data/dual_scale_cache/dual_scale_train.pt", map_location="cpu")
    p1_test = torch.load("data/multimodal_cache/test_features.pt", map_location="cpu")

    targets = test_cache["labels"].numpy()

    # Clinical normalization
    c_mean = train_cache["clinical"].mean(dim=0, keepdim=True)
    c_std = torch.clamp(train_cache["clinical"].std(dim=0, keepdim=True), min=1e-6)
    test_clin_norm = (test_cache["clinical"] - c_mean) / c_std

    g_vis = test_cache["global_visual"].to(device)
    l_vis = test_cache["local_visual"].to(device)
    c_norm = test_clin_norm.to(device)

    # 3. Model Predictions
    print("\n[Audit Item 2] Extracting Model Predictions...")
    # ViT Baseline
    vit_model = ViTForImageClassification.from_pretrained(
        str(CLASSIFICATION_MODEL_DIR / "vit_model"), local_files_only=True
    ).to(device)
    vit_model.eval()
    with torch.no_grad():
        v_logits = vit_model.classifier(p1_test["visual"].to(device))
        vit_probs = torch.softmax(v_logits, dim=-1)[:, 1].cpu().numpy()

    # Phase 1 Multimodal
    p1_fusion = CEFMMultimodalFusion(768, 16, 128, 2).to(device)
    p1_fusion.load_state_dict(torch.load(CHECKPOINT_PATHS["Phase1_Fusion"], map_location=device))
    p1_fusion.eval()
    with torch.no_grad():
        f_logits = p1_fusion(p1_test["visual"].to(device), p1_test["clinical"].to(device))
        p1_fusion_probs = torch.softmax(f_logits, dim=-1)[:, 1].cpu().numpy()

    # Phase 2 M1-M7
    phase2_models = {}
    arch_map = {
        "M1": M1_GlobalViT(),
        "M2": M2_LocalViT(),
        "M3": M3_DualVisual(),
        "M4": M4_GlobalClinical(),
        "M5": M5_DualConcat(),
        "M6": M6_DualGated(),
        "M7": M7_DualCrossAttn(),
    }
    for m_id, net in arch_map.items():
        net = net.to(device)
        net.load_state_dict(torch.load(CHECKPOINT_PATHS[m_id], map_location=device))
        net.eval()
        with torch.no_grad():
            out = net(g_vis, l_vis, c_norm)
            probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
        phase2_models[m_id] = probs

    m6_probs = phase2_models["M6"]
    m5_probs = phase2_models["M5"]
    m3_probs = phase2_models["M3"]

    # Generate Bootstrap Indices
    rng = np.random.RandomState(RANDOM_SEED)
    pos_idx = np.where(targets == 1)[0]
    neg_idx = np.where(targets == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    resample_indices = []
    for _ in range(BOOTSTRAP_ROUNDS):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        resample_indices.append(np.concatenate([b_pos, b_neg]))

    # 4. Rigorous Operating Point Comparison Audit
    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP AUDIT — STRICT OPERATING POINT COMPARISON")
    print("=" * 80)

    # SET A: F1 Operating Points
    print("\n--- SET A: F1 OPERATING POINTS (Table A Thresholds) ---")
    paired_f1 = {
        "M6(0.690)_vs_Phase1_ViT(0.900)": paired_bootstrap_analysis(
            targets, m6_probs, 0.690, vit_probs, 0.900, resample_indices
        ),
        "M6(0.690)_vs_Phase1_Fusion(0.800)": paired_bootstrap_analysis(
            targets, m6_probs, 0.690, p1_fusion_probs, 0.800, resample_indices
        ),
        "M6(0.690)_vs_M3(0.450)": paired_bootstrap_analysis(
            targets, m6_probs, 0.690, m3_probs, 0.450, resample_indices
        ),
        "M6(0.690)_vs_M5(0.630)": paired_bootstrap_analysis(
            targets, m6_probs, 0.690, m5_probs, 0.630, resample_indices
        ),
    }

    # SET B: Specificity-Target Operating Points
    print("\n--- SET B: SPECIFICITY-TARGET OPERATING POINTS (Table B Thresholds) ---")
    paired_spec = {
        "M6(0.625)_vs_Phase1_ViT(0.800)": paired_bootstrap_analysis(
            targets, m6_probs, 0.625, vit_probs, 0.800, resample_indices
        ),
        "M6(0.625)_vs_Phase1_Fusion(0.360)": paired_bootstrap_analysis(
            targets, m6_probs, 0.625, p1_fusion_probs, 0.360, resample_indices
        ),
        "M6(0.625)_vs_M3(0.075)": paired_bootstrap_analysis(
            targets, m6_probs, 0.625, m3_probs, 0.075, resample_indices
        ),
        "M6(0.625)_vs_M5(0.270)": paired_bootstrap_analysis(
            targets, m6_probs, 0.625, m5_probs, 0.270, resample_indices
        ),
    }

    # 5. Multiple Comparisons Correction
    # We test 4 models across 6 primary metrics (Sens, Spec, Prec, F1, BAcc, ROC-AUC) = 24 tests per set
    num_tests = 24
    alpha_bonferroni = 0.05 / num_tests

    def apply_bh_fdr(p_values: List[float], q: float = 0.05) -> List[bool]:
        """Benjamini-Hochberg FDR procedure."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        thresholds = [(i + 1) / n * q for i in range(n)]
        significant = np.zeros(n, dtype=bool)
        below = np.where(sorted_p <= thresholds)[0]
        if len(below) > 0:
            max_idx = np.max(below)
            significant[sorted_indices[:max_idx + 1]] = True
        return significant.tolist()

    # Gather all p-values for Set A
    all_p_set_a = []
    test_keys_a = []
    for comp, metrics in paired_f1.items():
        for met in ["sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "roc_auc"]:
            all_p_set_a.append(metrics[met]["p_value"])
            test_keys_a.append((comp, met))

    bh_sig_a = apply_bh_fdr(all_p_set_a, q=0.05)

    print("\n" + "=" * 95)
    print("SET A RESULTS: F1 OPERATING POINTS (M6 @ 0.690 vs. Baselines @ Their F1 Thresholds)")
    print("=" * 95)
    header = f"{'Comparison':<34} | {'Metric':<18} | {'Mean Diff':<10} | {'95% CI':<24} | {'Nominal p':<10} | {'Bonf (p<0.002)':<14} | {'FDR q<0.05'}"
    print(header)
    print("-" * 95)

    idx_test = 0
    for comp, metrics in paired_f1.items():
        for met in ["sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "roc_auc"]:
            r = metrics[met]
            bonf = "YES" if r["p_value"] < alpha_bonferroni else "No"
            bh = "YES" if bh_sig_a[idx_test] else "No"
            ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
            print(f"{comp:<34} | {met:<18} | {r['mean_difference']:<+10.4f} | {ci_str:<24} | {r['p_value']:<10.4f} | {bonf:<14} | {bh}")
            idx_test += 1
        print("-" * 95)

    # Gather all p-values for Set B
    all_p_set_b = []
    test_keys_b = []
    for comp, metrics in paired_spec.items():
        for met in ["sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "roc_auc"]:
            all_p_set_b.append(metrics[met]["p_value"])
            test_keys_b.append((comp, met))

    bh_sig_b = apply_bh_fdr(all_p_set_b, q=0.05)

    print("\n" + "=" * 95)
    print("SET B RESULTS: SPECIFICITY-TARGET OPERATING POINTS (M6 @ 0.625 vs. Baselines @ Their Spec Thresholds)")
    print("=" * 95)
    print(header)
    print("-" * 95)

    idx_test = 0
    for comp, metrics in paired_spec.items():
        for met in ["sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "roc_auc"]:
            r = metrics[met]
            bonf = "YES" if r["p_value"] < alpha_bonferroni else "No"
            bh = "YES" if bh_sig_b[idx_test] else "No"
            ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
            print(f"{comp:<34} | {met:<18} | {r['mean_difference']:<+10.4f} | {ci_str:<24} | {r['p_value']:<10.4f} | {bonf:<14} | {bh}")
            idx_test += 1
        print("-" * 95)

    # 6. Audit ISIC_0012418
    print("\n" + "=" * 80)
    print("[Audit Item 3] Independent Investigation of ISIC_0012418 Ground Truth")
    print("=" * 80)

    # Search HAM10000 metadata
    found_in_ham = False
    for csv_name in ["test.csv", "train.csv", "validation.csv"]:
        df_meta = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / csv_name)
        if (df_meta["image_id"] == "ISIC_0012418").any():
            found_in_ham = True
            print(f"  Found in HAM10000 {csv_name}")

    if not found_in_ham:
        print("  [-] ISIC_0012418 is NOT present in HAM10000 training, validation, or test splits.")

    # Check file origin
    isic_12418_path = Path("D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg")
    if isic_12418_path.exists():
        print(f"  [+] Physical file located at: {isic_12418_path}")
        print("  [*] Partition Origin: ISIC 2018 Challenge Task 1-2 Blind Test Set.")
        print("  [*] Official Challenge Status: Ground-truth diagnoses for this partition are held privately by ISIC archive organizers.")
        print("  [!] CONCLUSION: There is NO independently documented histopathological melanoma diagnosis in dataset metadata.")
        print("  [!] REQUIRED ACTION: Strip the claim of 'histopathologically confirmed melanoma' or 'rescued false negative'.")
        print("      Describe strictly as a challenging dermoscopic case comparison of model sensitivity.")

    audit_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_hashes": hashes,
        "f1_operating_point_paired_differences": paired_f1,
        "spec_operating_point_paired_differences": paired_spec,
        "multiple_testing": {
            "num_hypotheses": num_tests,
            "bonferroni_alpha": alpha_bonferroni,
            "fdr_q": 0.05,
        },
        "isic_0012418_audit": {
            "image_id": "ISIC_0012418",
            "in_ham10000_splits": found_in_ham,
            "source_directory": str(isic_12418_path),
            "independently_verified_ground_truth": False,
            "status": "Challenging dermoscopic evaluation case with unconfirmed histopathology; reported solely for model sensitivity comparison.",
        },
    }

    audit_file = REPO_ROOT / "reports" / "phase2_step4_final_consistency_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_output, f, indent=2)
    print(f"\n[+] Saved audit results JSON to {audit_file}")

    return audit_output


if __name__ == "__main__":
    run_audit()

