"""
CEFM Step 12: Official Lesion-Disjoint Model Evaluation.

Evaluates the three existing checkpoints:
1. ViT Classifier (models/classification/vit_model/model.safetensors)
2. Phase 1 Multimodal Fusion (models/multimodal/fusion_model.pt)
3. Challenger M6 Dual-Scale RCAG (models/experiments/M6/weights.pt)

On the leak-free test partition:
datasets/Ham10000/processed_lesion_disjoint/test.csv (N=1,504 samples, 1,121 lesions)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
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
from transformers import ViTForImageClassification, ViTImageProcessor

REPO_ROOT = Path("D:/pytest_cache")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.ml.multimodal.dual_scale_models import M6_DualGated

TEST_CSV_PATH = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint" / "test.csv"
META_CSV_PATH = REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_metadata.csv"
TRAIN_CACHE_PATH = REPO_ROOT / "data" / "dual_scale_cache" / "dual_scale_train.pt"
TEST_DUAL_SCALE_PATH = REPO_ROOT / "data" / "dual_scale_cache" / "dual_scale_test.pt"
TEST_MULTIMODAL_PATH = REPO_ROOT / "data" / "multimodal_cache" / "test_features.pt"

VIT_CHECKPOINT = REPO_ROOT / "models" / "classification" / "vit_model" / "model.safetensors"
MULTIMODAL_CHECKPOINT = REPO_ROOT / "models" / "multimodal" / "fusion_model.pt"
M6_CHECKPOINT = REPO_ROOT / "models" / "experiments" / "M6" / "weights.pt"
M6_CONFIG = REPO_ROOT / "models" / "experiments" / "M6" / "config.json"

REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluation_step12"
CONF_MAT_DIR = OUTPUT_DIR / "CEFM_STEP12_CONFUSION_MATRICES"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_integrity() -> Dict[str, Any]:
    print("=" * 80)
    print("VERIFICATION CHECKLIST (STEP 12 PRE-EVALUATION AUDIT)")
    print("=" * 80)

    # 1. Dataset existence & counts
    assert TEST_CSV_PATH.exists(), f"Test CSV not found: {TEST_CSV_PATH}"
    df_test = pd.read_csv(TEST_CSV_PATH)
    assert len(df_test) == 1504, f"Expected 1,504 test samples, got {len(df_test)}"
    assert df_test["lesion_id"].nunique() == 1121, f"Expected 1,121 unique lesions, got {df_test['lesion_id'].nunique()}"
    assert (df_test["dx"] == "mel").sum() == 166, f"Expected 166 melanomas, got {(df_test['dx'] == 'mel').sum()}"
    assert (df_test["dx"] != "mel").sum() == 1338, f"Expected 1,338 non-melanomas, got {(df_test['dx'] != 'mel').sum()}"
    print(f"[1] Test CSV integrity: PASSED (N=1,504, 1,121 lesions, 166 mel, 1,338 non-mel)")

    # 2. Verify all images exist on disk
    img_dirs = [
        REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_images_part_1",
        REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_images_part_2",
    ]
    missing = [img_id for img_id in df_test["image_id"] if not any((d / f"{img_id}.jpg").exists() for d in img_dirs)]
    assert len(missing) == 0, f"Missing images on disk: {missing}"
    print(f"[2] Image on-disk availability: PASSED (All 1,504 images verified)")

    # 3. Verify labels against metadata
    meta_df = pd.read_csv(META_CSV_PATH).set_index("image_id")
    for idx, row in df_test.iterrows():
        m_row = meta_df.loc[row["image_id"]]
        assert m_row["dx"] == row["dx"], f"Diagnosis mismatch on {row['image_id']}"
        assert m_row["lesion_id"] == row["lesion_id"], f"Lesion ID mismatch on {row['image_id']}"
    print(f"[3] Label metadata alignment: PASSED (100% concordance with HAM10000_metadata.csv)")

    # 4. Zero lesion overlap with training data
    train_cache = torch.load(TRAIN_CACHE_PATH, map_location="cpu")
    train_lesions = set(train_cache["lesion_ids"])
    test_lesions = set(df_test["lesion_id"])
    overlap_lesions = train_lesions & test_lesions
    assert len(overlap_lesions) == 0, f"Lesion leakage detected: {len(overlap_lesions)}"
    print(f"[4] Zero lesion overlap with historical training data: PASSED (0 shared lesions)")

    # 5. Checkpoint hashes
    expected_hashes = {
        "ViT": ("1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db", VIT_CHECKPOINT),
        "Multimodal": ("04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422", MULTIMODAL_CHECKPOINT),
        "M6": ("573ac6c73a856e30c74ca5b4f2e6d28c62632331bd83053d5aff27b5861b4a32", M6_CHECKPOINT),
    }
    for name, (exp_h, path) in expected_hashes.items():
        assert path.exists(), f"Checkpoint not found: {path}"
        cur_h = sha256_file(path)
        assert cur_h == exp_h, f"Hash mismatch for {name}: got {cur_h}, expected {exp_h}"
        print(f"[5] Checkpoint hash for {name}: PASSED ({cur_h[:16]}...)")

    # 6. Feature-cache image ordering verification
    ds_test = torch.load(TEST_DUAL_SCALE_PATH, map_location="cpu")
    mm_test = torch.load(TEST_MULTIMODAL_PATH, map_location="cpu")
    assert ds_test["image_ids"] == df_test["image_id"].tolist(), "Dual-scale cache image ordering mismatch!"
    assert torch.allclose(ds_test["clinical"], mm_test["clinical"], atol=1e-5), "Clinical feature tensor mismatch!"
    assert (mm_test["labels"].numpy() == df_test["label"].values).all(), "Multimodal cache label mismatch!"
    print(f"[6] Feature cache correspondence & ordering: PASSED (100% row-for-row match)")

    print("\nALL PRE-EVALUATION INTEGRITY GATES PASSED.\n")
    return {"status": "PASSED"}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1_score": float(f1),
        "balanced_accuracy": float(bacc),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run_evaluation():
    verify_integrity()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONF_MAT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(TEST_CSV_PATH)
    y_true = df_test["label"].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing inference on device: {device}")

    # =========================================================================
    # A. ViT Classifier Evaluation
    # =========================================================================
    print("Evaluating Model A: ViT Classifier (google/vit-base-patch16-224)...")
    vit = ViTForImageClassification.from_pretrained(str(VIT_CHECKPOINT.parent), local_files_only=True).to(device)
    vit.eval()

    # Load visual embedding from test_features.pt
    mm_ts = torch.load(TEST_MULTIMODAL_PATH, map_location="cpu")
    vis = mm_ts["visual"].to(device)
    clin = mm_ts["clinical"].to(device)

    with torch.no_grad():
        vit_logits = vit.classifier(vis)
        vit_probs = torch.softmax(vit_logits, dim=-1)[:, 1].cpu().numpy()

    del vit

    # =========================================================================
    # B. Phase 1 Multimodal Fusion Evaluation
    # =========================================================================
    print("Evaluating Model B: Phase 1 Multimodal Fusion (ViT + ABCDE)...")
    mm_model = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2).to(device)
    mm_model.load_state_dict(torch.load(MULTIMODAL_CHECKPOINT, map_location=device))
    mm_model.eval()

    with torch.no_grad():
        mm_logits = mm_model(vis, clin)
        mm_probs = torch.softmax(mm_logits, dim=-1)[:, 1].cpu().numpy()

    del mm_model

    # =========================================================================
    # C. Challenger M6 Evaluation
    # =========================================================================
    print("Evaluating Model C: Challenger M6 (Dual-Scale RCAG)...")
    with open(M6_CONFIG) as f:
        m6_cfg = json.load(f)

    m6_model = M6_DualGated(
        global_dim=768,
        local_dim=768,
        clinical_dim=16,
        hidden_dim=128,
        clinical_hidden=64,
        num_classes=2,
    ).to(device)
    m6_model.load_state_dict(torch.load(M6_CHECKPOINT, map_location=device))
    m6_model.eval()

    ds_ts = torch.load(TEST_DUAL_SCALE_PATH, map_location="cpu")
    g_v = ds_ts["global_visual"].to(device)
    l_v = ds_ts["local_visual"].to(device)
    c_v = ds_ts["clinical"].to(device)

    mean = torch.tensor(m6_cfg["clinical_norm_mean"], device=device).unsqueeze(0)
    std = torch.tensor(m6_cfg["clinical_norm_std"], device=device).unsqueeze(0)
    c_norm = (c_v - mean) / std

    with torch.no_grad():
        m6_probs = m6_model.predict_proba(g_v, l_v, c_norm)[:, 1].cpu().numpy()

    del m6_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Compute ROC-AUC and PR-AUC
    # =========================================================================
    vit_auc = float(roc_auc_score(y_true, vit_probs))
    vit_prauc = float(average_precision_score(y_true, vit_probs))

    mm_auc = float(roc_auc_score(y_true, mm_probs))
    mm_prauc = float(average_precision_score(y_true, mm_probs))

    m6_auc = float(roc_auc_score(y_true, m6_probs))
    m6_prauc = float(average_precision_score(y_true, m6_probs))

    # Production thresholds
    PROD_THRESH_VIT = 0.80
    PROD_THRESH_MM = 0.36
    PROD_THRESH_M6_F1 = 0.690
    PROD_THRESH_M6_SPEC = 0.625

    vit_prod_metrics = compute_metrics(y_true, vit_probs, PROD_THRESH_VIT)
    vit_prod_metrics["roc_auc"] = vit_auc
    vit_prod_metrics["pr_auc"] = vit_prauc

    mm_prod_metrics = compute_metrics(y_true, mm_probs, PROD_THRESH_MM)
    mm_prod_metrics["roc_auc"] = mm_auc
    mm_prod_metrics["pr_auc"] = mm_prauc

    m6_prod_f1_metrics = compute_metrics(y_true, m6_probs, PROD_THRESH_M6_F1)
    m6_prod_f1_metrics["roc_auc"] = m6_auc
    m6_prod_f1_metrics["pr_auc"] = m6_prauc

    m6_prod_spec_metrics = compute_metrics(y_true, m6_probs, PROD_THRESH_M6_SPEC)
    m6_prod_spec_metrics["roc_auc"] = m6_auc
    m6_prod_spec_metrics["pr_auc"] = m6_prauc

    # Analytical thresholds sweep
    sweep_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sweep_records = []
    for t in sweep_thresholds:
        v_m = compute_metrics(y_true, vit_probs, t)
        v_m["model"] = "ViT Classifier"
        v_m["roc_auc"] = vit_auc
        sweep_records.append(v_m)

        m_m = compute_metrics(y_true, mm_probs, t)
        m_m["model"] = "Multimodal Fusion"
        m_m["roc_auc"] = mm_auc
        sweep_records.append(m_m)

        m6_m = compute_metrics(y_true, m6_probs, t)
        m6_m["model"] = "M6 Dual-Scale RCAG"
        m6_m["roc_auc"] = m6_auc
        sweep_records.append(m6_m)

    # Save Confusion Matrices
    for model_name, prod_m, prod_t in [
        ("ViT_prod_0.80", vit_prod_metrics, 0.80),
        ("Multimodal_prod_0.36", mm_prod_metrics, 0.36),
        ("M6_prod_F1_0.690", m6_prod_f1_metrics, 0.690),
        ("M6_prod_Spec_0.625", m6_prod_spec_metrics, 0.625),
    ]:
        with open(CONF_MAT_DIR / f"{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(prod_m, f, indent=2)

    # =========================================================================
    # Deliverable 2: CEFM_STEP12_MODEL_COMPARISON.csv
    # =========================================================================
    comparison_rows = [
        {
            "Model": "ViT Classifier (Production)",
            "Operating Threshold": PROD_THRESH_VIT,
            "Accuracy": round(vit_prod_metrics["accuracy"], 4),
            "Precision": round(vit_prod_metrics["precision"], 4),
            "Sensitivity": round(vit_prod_metrics["sensitivity"], 4),
            "Specificity": round(vit_prod_metrics["specificity"], 4),
            "F1-score": round(vit_prod_metrics["f1_score"], 4),
            "ROC-AUC": round(vit_auc, 4),
            "Balanced Accuracy": round(vit_prod_metrics["balanced_accuracy"], 4),
            "True Positives (Correct Mel)": vit_prod_metrics["tp"],
            "False Negatives (Missed Mel)": vit_prod_metrics["fn"],
            "True Negatives (Correct Non-Mel)": vit_prod_metrics["tn"],
            "False Positives (False Mel)": vit_prod_metrics["fp"],
        },
        {
            "Model": "Multimodal Fusion (Production)",
            "Operating Threshold": PROD_THRESH_MM,
            "Accuracy": round(mm_prod_metrics["accuracy"], 4),
            "Precision": round(mm_prod_metrics["precision"], 4),
            "Sensitivity": round(mm_prod_metrics["sensitivity"], 4),
            "Specificity": round(mm_prod_metrics["specificity"], 4),
            "F1-score": round(mm_prod_metrics["f1_score"], 4),
            "ROC-AUC": round(mm_auc, 4),
            "Balanced Accuracy": round(mm_prod_metrics["balanced_accuracy"], 4),
            "True Positives (Correct Mel)": mm_prod_metrics["tp"],
            "False Negatives (Missed Mel)": mm_prod_metrics["fn"],
            "True Negatives (Correct Non-Mel)": mm_prod_metrics["tn"],
            "False Positives (False Mel)": mm_prod_metrics["fp"],
        },
        {
            "Model": "M6 Dual-Scale RCAG (F1 Threshold)",
            "Operating Threshold": PROD_THRESH_M6_F1,
            "Accuracy": round(m6_prod_f1_metrics["accuracy"], 4),
            "Precision": round(m6_prod_f1_metrics["precision"], 4),
            "Sensitivity": round(m6_prod_f1_metrics["sensitivity"], 4),
            "Specificity": round(m6_prod_f1_metrics["specificity"], 4),
            "F1-score": round(m6_prod_f1_metrics["f1_score"], 4),
            "ROC-AUC": round(m6_auc, 4),
            "Balanced Accuracy": round(m6_prod_f1_metrics["balanced_accuracy"], 4),
            "True Positives (Correct Mel)": m6_prod_f1_metrics["tp"],
            "False Negatives (Missed Mel)": m6_prod_f1_metrics["fn"],
            "True Negatives (Correct Non-Mel)": m6_prod_f1_metrics["tn"],
            "False Positives (False Mel)": m6_prod_f1_metrics["fp"],
        },
        {
            "Model": "M6 Dual-Scale RCAG (Spec 90% Threshold)",
            "Operating Threshold": PROD_THRESH_M6_SPEC,
            "Accuracy": round(m6_prod_spec_metrics["accuracy"], 4),
            "Precision": round(m6_prod_spec_metrics["precision"], 4),
            "Sensitivity": round(m6_prod_spec_metrics["sensitivity"], 4),
            "Specificity": round(m6_prod_spec_metrics["specificity"], 4),
            "F1-score": round(m6_prod_spec_metrics["f1_score"], 4),
            "ROC-AUC": round(m6_auc, 4),
            "Balanced Accuracy": round(m6_prod_spec_metrics["balanced_accuracy"], 4),
            "True Positives (Correct Mel)": m6_prod_spec_metrics["tp"],
            "False Negatives (Missed Mel)": m6_prod_spec_metrics["fn"],
            "True Negatives (Correct Non-Mel)": m6_prod_spec_metrics["tn"],
            "False Positives (False Mel)": m6_prod_spec_metrics["fp"],
        },
    ]
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(REPORTS_DIR / "CEFM_STEP12_MODEL_COMPARISON.csv", index=False)
    df_comparison.to_csv(OUTPUT_DIR / "CEFM_STEP12_MODEL_COMPARISON.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'CEFM_STEP12_MODEL_COMPARISON.csv'}")

    # =========================================================================
    # Deliverable 3: CEFM_STEP12_TEST_PREDICTIONS.csv & Error Analysis
    # =========================================================================
    vit_preds = (vit_probs >= PROD_THRESH_VIT).astype(int)
    mm_preds = (mm_probs >= PROD_THRESH_MM).astype(int)
    m6_preds_f1 = (m6_probs >= PROD_THRESH_M6_F1).astype(int)

    df_preds = pd.DataFrame({
        "image_id": df_test["image_id"],
        "lesion_id": df_test["lesion_id"],
        "dx": df_test["dx"],
        "ground_truth": y_true,
        "ViT_probability": np.round(vit_probs, 4),
        "ViT_prediction": vit_preds,
        "Multimodal_probability": np.round(mm_probs, 4),
        "Multimodal_prediction": mm_preds,
        "M6_probability": np.round(m6_probs, 4),
        "M6_prediction": m6_preds_f1,
    })
    df_preds.to_csv(REPORTS_DIR / "CEFM_STEP12_TEST_PREDICTIONS.csv", index=False)
    df_preds.to_csv(OUTPUT_DIR / "CEFM_STEP12_TEST_PREDICTIONS.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'CEFM_STEP12_TEST_PREDICTIONS.csv'}")

    # Error analysis categories
    mel_mask = y_true == 1
    non_mel_mask = y_true == 0

    missed_by_all = df_preds[mel_mask & (vit_preds == 0) & (mm_preds == 0) & (m6_preds_f1 == 0)]
    detected_by_all = df_preds[mel_mask & (vit_preds == 1) & (mm_preds == 1) & (m6_preds_f1 == 1)]
    vit_only = df_preds[mel_mask & (vit_preds == 1) & (mm_preds == 0) & (m6_preds_f1 == 0)]
    mm_only = df_preds[mel_mask & (vit_preds == 0) & (mm_preds == 1) & (m6_preds_f1 == 0)]
    m6_only = df_preds[mel_mask & (vit_preds == 0) & (mm_preds == 0) & (m6_preds_f1 == 1)]
    disagree = df_preds[~((vit_preds == mm_preds) & (mm_preds == m6_preds_f1))]
    fp_all = df_preds[non_mel_mask & (vit_preds == 1) & (mm_preds == 1) & (m6_preds_f1 == 1)]

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Melanoma Test Cases:                   {mel_mask.sum()}")
    print(f"  Melanomas detected by ALL models:          {len(detected_by_all)} ({len(detected_by_all)/mel_mask.sum()*100:.1f}%)")
    print(f"  Melanomas MISSED by ALL models:            {len(missed_by_all)} ({len(missed_by_all)/mel_mask.sum()*100:.1f}%)")
    print(f"  Melanomas detected ONLY by ViT:            {len(vit_only)}")
    print(f"  Melanomas detected ONLY by Multimodal:     {len(mm_only)}")
    print(f"  Melanomas detected ONLY by M6:             {len(m6_only)}")
    print(f"Total Disagreement Cases (any model differs): {len(disagree)} ({len(disagree)/len(df_preds)*100:.1f}%)")
    print(f"False Positives flagged by ALL models:       {len(fp_all)}")

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (PRODUCTION THRESHOLDS)")
    print("=" * 80)
    print(df_comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP COMPARISON (ANALYTICAL ONLY)")
    print("=" * 80)
    df_sweep = pd.DataFrame(sweep_records)
    cols_order = ["model", "threshold", "accuracy", "precision", "sensitivity", "specificity", "f1_score", "balanced_accuracy", "tp", "fn", "tn", "fp"]
    print(df_sweep[cols_order].to_string(index=False))

    return {
        "comparison": comparison_rows,
        "sweep": sweep_records,
        "error_analysis": {
            "missed_by_all": len(missed_by_all),
            "detected_by_all": len(detected_by_all),
            "vit_only": len(vit_only),
            "mm_only": len(mm_only),
            "m6_only": len(m6_only),
            "disagreements": len(disagree),
            "fp_all": len(fp_all),
        },
    }


if __name__ == "__main__":
    run_evaluation()

