"""
CEFM Step 13: Threshold Calibration, Lesion-Level Evaluation,
Statistical Reliability, and Forensic Audit.

Executes Phases 1 to 6 in a rigorous, reproducible manner.
STRICT RULES COMPLIANCE:
- Read-only on model checkpoints.
- Validation-only threshold selection (zero test leakage).
- Single unbiased test evaluation of selected operating points.
- Lesion-level cluster bootstrap for 95% confidence intervals.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
from transformers import ViTForImageClassification

REPO_ROOT = Path("D:/pytest_cache")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.ml.multimodal.dual_scale_models import M6_DualGated

# Paths
VAL_CSV_PATH = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint" / "validation.csv"
TEST_CSV_PATH = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint" / "test.csv"
META_CSV_PATH = REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_metadata.csv"

VAL_MM_CACHE = REPO_ROOT / "data" / "multimodal_cache" / "validation_features.pt"
VAL_DS_CACHE = REPO_ROOT / "data" / "dual_scale_cache" / "dual_scale_validation.pt"

TEST_PRED_PATH = REPO_ROOT / "reports" / "CEFM_STEP12_TEST_PREDICTIONS.csv"
TEST_DS_CACHE = REPO_ROOT / "data" / "dual_scale_cache" / "dual_scale_test.pt"

VIT_CHECKPOINT = REPO_ROOT / "models" / "classification" / "vit_model"
MM_CHECKPOINT = REPO_ROOT / "models" / "multimodal" / "fusion_model.pt"
M6_CHECKPOINT = REPO_ROOT / "models" / "experiments" / "M6" / "weights.pt"
M6_CONFIG = REPO_ROOT / "models" / "experiments" / "M6" / "config.json"

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    acc = (tp + tn) / len(y_true)

    return {
        "threshold": round(float(threshold), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "accuracy": round(float(acc), 4),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
    }


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    thresholds = [round(float(t), 4) for t in np.arange(0.02, 0.98, 0.01)]
    records = []
    for t in thresholds:
        m = calculate_metrics_at_threshold(y_true, y_prob, t)
        m["model"] = model_name
        records.append(m)

    df_sweep = pd.DataFrame(records)

    # Operating Points Selection:
    # 1. Max F1
    best_f1_row = df_sweep.loc[df_sweep["f1_score"].idxmax()]
    # 2. Max Balanced Accuracy
    best_bacc_row = df_sweep.loc[df_sweep["balanced_accuracy"].idxmax()]

    # 3. Target Sens >= 90% (maximize specificity)
    s90_candidates = df_sweep[df_sweep["sensitivity"] >= 0.9000]
    if not s90_candidates.empty:
        # Choose highest threshold (which maximizes specificity) among candidates
        s90_row = s90_candidates.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
    else:
        s90_row = df_sweep.sort_values(by="sensitivity", ascending=False).iloc[0]

    # 4. Target Sens >= 92%
    s92_candidates = df_sweep[df_sweep["sensitivity"] >= 0.9200]
    if not s92_candidates.empty:
        s92_row = s92_candidates.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
    else:
        s92_row = df_sweep.sort_values(by="sensitivity", ascending=False).iloc[0]

    # 5. Target Sens >= 95%
    s95_candidates = df_sweep[df_sweep["sensitivity"] >= 0.9500]
    if not s95_candidates.empty:
        s95_row = s95_candidates.sort_values(by=["specificity", "threshold"], ascending=[False, False]).iloc[0]
    else:
        s95_row = df_sweep.sort_values(by="sensitivity", ascending=False).iloc[0]

    selected_points = {
        "max_f1": best_f1_row.to_dict(),
        "max_bacc": best_bacc_row.to_dict(),
        "sens_90": s90_row.to_dict(),
        "sens_92": s92_row.to_dict(),
        "sens_95": s95_row.to_dict(),
    }

    return df_sweep, selected_points


def main():
    print("=" * 80)
    print("CEFM STEP 13: MASTER THRESHOLD CALIBRATION & LESION-LEVEL ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # =========================================================================
    # PHASE 1: VALIDATION-ONLY THRESHOLD CALIBRATION
    # =========================================================================
    print("\n[PHASE 1] Executing Validation-Only Threshold Calibration...")
    df_val = pd.read_csv(VAL_CSV_PATH)
    y_val = df_val["label"].values.astype(int)
    assert len(df_val) == 1533, f"Expected 1,533 validation samples, got {len(df_val)}"
    print(f"Validation dataset: N={len(df_val)} ({df_val['dx'].value_counts().to_dict()})")

    # A. ViT validation inference
    print("Running ViT on validation features...")
    vit = ViTForImageClassification.from_pretrained(str(VIT_CHECKPOINT), local_files_only=True).to(device)
    vit.eval()
    val_mm = torch.load(VAL_MM_CACHE, map_location="cpu")
    val_vis = val_mm["visual"].to(device)
    val_clin = val_mm["clinical"].to(device)

    with torch.no_grad():
        vit_val_logits = vit.classifier(val_vis)
        vit_val_probs = torch.softmax(vit_val_logits, dim=-1)[:, 1].cpu().numpy()
    del vit

    # B. Multimodal validation inference
    print("Running Phase 1 Multimodal on validation features...")
    mm_model = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2).to(device)
    mm_model.load_state_dict(torch.load(MM_CHECKPOINT, map_location=device))
    mm_model.eval()

    with torch.no_grad():
        mm_val_logits = mm_model(val_vis, val_clin)
        mm_val_probs = torch.softmax(mm_val_logits, dim=-1)[:, 1].cpu().numpy()
    del mm_model

    # C. M6 validation inference
    print("Running Challenger M6 on validation features...")
    with open(M6_CONFIG) as f:
        m6_cfg = json.load(f)

    m6_model = M6_DualGated(
        global_dim=768, local_dim=768, clinical_dim=16, hidden_dim=128, clinical_hidden=64, num_classes=2
    ).to(device)
    m6_model.load_state_dict(torch.load(M6_CHECKPOINT, map_location=device))
    m6_model.eval()

    val_ds = torch.load(VAL_DS_CACHE, map_location="cpu")
    val_gv = val_ds["global_visual"].to(device)
    val_lv = val_ds["local_visual"].to(device)
    val_cv = val_ds["clinical"].to(device)

    mean = torch.tensor(m6_cfg["clinical_norm_mean"], device=device).unsqueeze(0)
    std = torch.tensor(m6_cfg["clinical_norm_std"], device=device).unsqueeze(0)
    val_cnorm = (val_cv - mean) / std

    with torch.no_grad():
        m6_val_probs = m6_model.predict_proba(val_gv, val_lv, val_cnorm)[:, 1].cpu().numpy()
    del m6_model

    val_auc_vit = roc_auc_score(y_val, vit_val_probs)
    val_auc_mm = roc_auc_score(y_val, mm_val_probs)
    val_auc_m6 = roc_auc_score(y_val, m6_val_probs)

    print(f"Validation ROC-AUC: ViT={val_auc_vit:.4f}, MM={val_auc_mm:.4f}, M6={val_auc_m6:.4f}")

    # Threshold sweeps
    df_sweep_vit, vit_pts = sweep_thresholds(y_val, vit_val_probs, "ViT")
    df_sweep_mm, mm_pts = sweep_thresholds(y_val, mm_val_probs, "Multimodal")
    df_sweep_m6, m6_pts = sweep_thresholds(y_val, m6_val_probs, "M6")

    df_val_sweep_all = pd.concat([df_sweep_vit, df_sweep_mm, df_sweep_m6], ignore_index=True)
    df_val_sweep_all.to_csv(REPORTS_DIR / "CEFM_STEP13_VALIDATION_THRESHOLDS.csv", index=False)
    print(f"Saved full validation sweep to {REPORTS_DIR / 'CEFM_STEP13_VALIDATION_THRESHOLDS.csv'}")

    # Display selected validation operating points
    operating_points = {
        "ViT": vit_pts,
        "Multimodal": mm_pts,
        "M6": m6_pts,
    }

    op_summary_records = []
    for model_name, pts in operating_points.items():
        for pt_name, metrics in pts.items():
            r = dict(metrics)
            r["operating_point_type"] = pt_name
            r["dataset"] = "validation"
            op_summary_records.append(r)
    df_op_val = pd.DataFrame(op_summary_records)
    print("\n--- Validation Selected Operating Points ---")
    print(df_op_val[["model", "operating_point_type", "threshold", "sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy"]].to_string(index=False))

    # =========================================================================
    # PHASE 2: LESION-LEVEL EVALUATION ON TEST PARTITION
    # =========================================================================
    print("\n[PHASE 2] Executing Lesion-Level Evaluation on Test Partition...")
    df_test_preds = pd.read_csv(TEST_PRED_PATH)
    assert len(df_test_preds) == 1504, f"Expected 1,504 test predictions, got {len(df_test_preds)}"

    # Check multiple photographs per lesion
    lesion_counts = df_test_preds.groupby("lesion_id").size()
    print(f"Total unique lesions in test set: {len(lesion_counts)}")
    print(f"Lesions with 1 image:  {(lesion_counts == 1).sum()}")
    print(f"Lesions with 2 images: {(lesion_counts == 2).sum()}")
    print(f"Lesions with 3 images: {(lesion_counts == 3).sum()}")
    print(f"Lesions with >3 images: {(lesion_counts > 3).sum()}")

    # Verify ground truth consistency per lesion
    gt_check = df_test_preds.groupby("lesion_id")["ground_truth"].nunique()
    assert (gt_check == 1).all(), "Ground truth is not uniform per lesion!"

    # Lesion-level ground truth
    df_lesions = df_test_preds.groupby("lesion_id").agg({
        "ground_truth": "first",
        "dx": "first",
        "ViT_probability": ["max", "mean", "median"],
        "Multimodal_probability": ["max", "mean", "median"],
        "M6_probability": ["max", "mean", "median"],
        "image_id": "count",
    })
    df_lesions.columns = ["_".join(c).strip("_") for c in df_lesions.columns.values]
    df_lesions = df_lesions.rename(columns={"ground_truth_first": "ground_truth", "dx_first": "dx", "image_id_count": "num_images"})

    y_test_img = df_test_preds["ground_truth"].values.astype(int)
    y_test_lesion = df_lesions["ground_truth"].values.astype(int)

    print(f"Test image count: {len(y_test_img)} (Melanoma: {(y_test_img==1).sum()}, Non-Mel: {(y_test_img==0).sum()})")
    print(f"Test lesion count: {len(y_test_lesion)} (Melanoma: {(y_test_lesion==1).sum()}, Non-Mel: {(y_test_lesion==0).sum()})")

    # Evaluate Image-Level vs Lesion-Level across models and operating points
    # We test:
    # 1. Step 12 Production Thresholds: ViT=0.80, Multimodal=0.36, M6=0.690
    # 2. Validation Selected Thresholds: Max F1, Max BalAcc, Sens >= 90%, Sens >= 92%, Sens >= 95%
    lesion_level_records = []

    models_config = {
        "ViT": {
            "img_prob": df_test_preds["ViT_probability"].values,
            "lesion_max": df_lesions["ViT_probability_max"].values,
            "lesion_mean": df_lesions["ViT_probability_mean"].values,
            "lesion_median": df_lesions["ViT_probability_median"].values,
            "prod_thresh": 0.80,
            "val_points": vit_pts,
        },
        "Multimodal": {
            "img_prob": df_test_preds["Multimodal_probability"].values,
            "lesion_max": df_lesions["Multimodal_probability_max"].values,
            "lesion_mean": df_lesions["Multimodal_probability_mean"].values,
            "lesion_median": df_lesions["Multimodal_probability_median"].values,
            "prod_thresh": 0.36,
            "val_points": mm_pts,
        },
        "M6": {
            "img_prob": df_test_preds["M6_probability"].values,
            "lesion_max": df_lesions["M6_probability_max"].values,
            "lesion_mean": df_lesions["M6_probability_mean"].values,
            "lesion_median": df_lesions["M6_probability_median"].values,
            "prod_thresh": 0.690,
            "val_points": m6_pts,
        },
    }

    for model_name, cfg in models_config.items():
        # Evaluate operating points
        threshold_dict = {"production": cfg["prod_thresh"]}
        for pt_name, pt_val in cfg["val_points"].items():
            threshold_dict[pt_name] = pt_val["threshold"]

        for pt_name, thresh in threshold_dict.items():
            # 1. Image-level
            img_m = calculate_metrics_at_threshold(y_test_img, cfg["img_prob"], thresh)
            img_m.update({"model": model_name, "level": "image", "aggregation": "none", "point_name": pt_name})
            lesion_level_records.append(img_m)

            # 2. Lesion-level Max
            l_max_m = calculate_metrics_at_threshold(y_test_lesion, cfg["lesion_max"], thresh)
            l_max_m.update({"model": model_name, "level": "lesion", "aggregation": "max", "point_name": pt_name})
            lesion_level_records.append(l_max_m)

            # 3. Lesion-level Mean
            l_mean_m = calculate_metrics_at_threshold(y_test_lesion, cfg["lesion_mean"], thresh)
            l_mean_m.update({"model": model_name, "level": "lesion", "aggregation": "mean", "point_name": pt_name})
            lesion_level_records.append(l_mean_m)

            # 4. Lesion-level Median
            l_med_m = calculate_metrics_at_threshold(y_test_lesion, cfg["lesion_median"], thresh)
            l_med_m.update({"model": model_name, "level": "lesion", "aggregation": "median", "point_name": pt_name})
            lesion_level_records.append(l_med_m)

    df_lesion_eval = pd.DataFrame(lesion_level_records)
    df_lesion_eval.to_csv(REPORTS_DIR / "CEFM_STEP13_LESION_LEVEL_METRICS.csv", index=False)
    print(f"Saved lesion-level metrics to {REPORTS_DIR / 'CEFM_STEP13_LESION_LEVEL_METRICS.csv'}")

    # Display comparison at production thresholds
    prod_comp = df_lesion_eval[df_lesion_eval["point_name"] == "production"]
    print("\n--- Production Threshold Comparison: Image-Level vs Lesion Aggregations ---")
    print(prod_comp[["model", "level", "aggregation", "threshold", "sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "fn", "fp"]].to_string(index=False))

    # =========================================================================
    # PHASE 3: STATISTICAL RELIABILITY & CLUSTER BOOTSTRAP 95% CIs
    # =========================================================================
    print("\n[PHASE 3] Computing 95% Confidence Intervals via Lesion-Cluster Bootstrap...")
    np.random.seed(42)
    B = 1000
    unique_lesions = df_test_preds["lesion_id"].unique()
    num_unique_lesions = len(unique_lesions)

    # Pre-index images by lesion for speed
    lesion_to_rows = df_test_preds.groupby("lesion_id").indices

    bootstrap_results = []
    # We will compute CIs for each model at:
    # 1. Production threshold
    # 2. Validation Sens >= 90% threshold
    # 3. Validation Sens >= 95% threshold
    # For both image-level and lesion-level (max aggregation)

    eval_targets = []
    for model_name, cfg in models_config.items():
        eval_targets.append((model_name, "production", cfg["prod_thresh"]))
        eval_targets.append((model_name, "sens_90", cfg["val_points"]["sens_90"]["threshold"]))
        eval_targets.append((model_name, "sens_95", cfg["val_points"]["sens_95"]["threshold"]))

    for model_name, pt_name, thresh in eval_targets:
        print(f"Bootstrapping {model_name} at {pt_name} (thresh={thresh:.4f})...")
        boot_img_sens, boot_img_spec, boot_img_prec, boot_img_auc = [], [], [], []
        boot_lmax_sens, boot_lmax_spec, boot_lmax_prec, boot_lmax_auc = [], [], [], []

        prob_col = f"{model_name}_probability"

        for b in range(B):
            # Sample lesions with replacement
            sampled_lesions = np.random.choice(unique_lesions, size=num_unique_lesions, replace=True)

            # Pull all corresponding images
            sampled_row_indices = []
            for l in sampled_lesions:
                sampled_row_indices.extend(lesion_to_rows[l])

            sample_df = df_test_preds.iloc[sampled_row_indices]
            y_b_img = sample_df["ground_truth"].values
            p_b_img = sample_df[prob_col].values

            # Image metrics
            m_img = calculate_metrics_at_threshold(y_b_img, p_b_img, thresh)
            boot_img_sens.append(m_img["sensitivity"])
            boot_img_spec.append(m_img["specificity"])
            boot_img_prec.append(m_img["precision"])
            try:
                boot_img_auc.append(roc_auc_score(y_b_img, p_b_img))
            except ValueError:
                pass

            # Lesion Max metrics
            # Group by resampled lesion index (since sample has repeated lesions)
            # To handle repeated lesions correctly, groupby lesion instance
            lesion_max_probs = []
            lesion_gts = []
            for l in sampled_lesions:
                rows = lesion_to_rows[l]
                lesion_max_probs.append(df_test_preds[prob_col].iloc[rows].max())
                lesion_gts.append(df_test_preds["ground_truth"].iloc[rows[0]])

            y_b_lesion = np.array(lesion_gts)
            p_b_lesion = np.array(lesion_max_probs)

            m_lmax = calculate_metrics_at_threshold(y_b_lesion, p_b_lesion, thresh)
            boot_lmax_sens.append(m_lmax["sensitivity"])
            boot_lmax_spec.append(m_lmax["specificity"])
            boot_lmax_prec.append(m_lmax["precision"])
            try:
                boot_lmax_auc.append(roc_auc_score(y_b_lesion, p_b_lesion))
            except ValueError:
                pass

        # Calculate percentiles (2.5% and 97.5%)
        for level, s_list, sp_list, p_list, a_list in [
            ("image", boot_img_sens, boot_img_spec, boot_img_prec, boot_img_auc),
            ("lesion_max", boot_lmax_sens, boot_lmax_spec, boot_lmax_prec, boot_lmax_auc),
        ]:
            bootstrap_results.append({
                "model": model_name,
                "operating_point": pt_name,
                "threshold": thresh,
                "level": level,
                "sens_mean": round(float(np.mean(s_list)), 4),
                "sens_ci_low": round(float(np.percentile(s_list, 2.5)), 4),
                "sens_ci_high": round(float(np.percentile(s_list, 97.5)), 4),
                "sens_ci_width": round(float(np.percentile(s_list, 97.5) - np.percentile(s_list, 2.5)), 4),
                "spec_mean": round(float(np.mean(sp_list)), 4),
                "spec_ci_low": round(float(np.percentile(sp_list, 2.5)), 4),
                "spec_ci_high": round(float(np.percentile(sp_list, 97.5)), 4),
                "spec_ci_width": round(float(np.percentile(sp_list, 97.5) - np.percentile(sp_list, 2.5)), 4),
                "prec_mean": round(float(np.mean(p_list)), 4),
                "prec_ci_low": round(float(np.percentile(p_list, 2.5)), 4),
                "prec_ci_high": round(float(np.percentile(p_list, 97.5)), 4),
                "auc_mean": round(float(np.mean(a_list)), 4),
                "auc_ci_low": round(float(np.percentile(a_list, 2.5)), 4),
                "auc_ci_high": round(float(np.percentile(a_list, 97.5)), 4),
            })

    df_bootstrap = pd.DataFrame(bootstrap_results)
    df_bootstrap.to_csv(REPORTS_DIR / "CEFM_STEP13_BOOTSTRAP_CI.csv", index=False)
    print(f"Saved bootstrap CIs to {REPORTS_DIR / 'CEFM_STEP13_BOOTSTRAP_CI.csv'}")
    print("\n--- Bootstrap 95% Confidence Intervals Summary ---")
    print(df_bootstrap[["model", "operating_point", "threshold", "level", "sens_mean", "sens_ci_low", "sens_ci_high", "sens_ci_width", "spec_mean", "spec_ci_low", "spec_ci_high", "auc_mean"]].to_string(index=False))

    # =========================================================================
    # PHASE 4: DETAILED AUDIT OF THE 27 JOINTLY MISSED MELANOMA CASES
    # =========================================================================
    print("\n[PHASE 4] Auditing the 27 Jointly Missed Melanoma Cases...")
    mel_test = df_test_preds[df_test_preds["ground_truth"] == 1]
    missed_all = mel_test[
        (mel_test["ViT_prediction"] == 0) &
        (mel_test["Multimodal_prediction"] == 0) &
        (mel_test["M6_prediction"] == 0)
    ].copy()
    assert len(missed_all) == 27, f"Expected 27 missed cases, got {len(missed_all)}"

    meta_df = pd.read_csv(META_CSV_PATH).set_index("image_id")
    test_ds = torch.load(TEST_DS_CACHE, map_location="cpu")
    img_to_idx = {img_id: i for i, img_id in enumerate(test_ds["image_ids"])}

    missed_records = []
    for _, row in missed_all.iterrows():
        img_id = row["image_id"]
        les_id = row["lesion_id"]
        m_row = meta_df.loc[img_id]

        idx = img_to_idx[img_id]
        clin_vec = test_ds["clinical"][idx].numpy()

        # Check total images for this lesion
        total_les_imgs = len(df_test_preds[df_test_preds["lesion_id"] == les_id])
        # Check if other images of this lesion were detected by any model
        other_imgs = df_test_preds[(df_test_preds["lesion_id"] == les_id) & (df_test_preds["image_id"] != img_id)]
        any_other_vit = int((other_imgs["ViT_prediction"] == 1).any()) if not other_imgs.empty else 0
        any_other_mm = int((other_imgs["Multimodal_prediction"] == 1).any()) if not other_imgs.empty else 0
        any_other_m6 = int((other_imgs["M6_prediction"] == 1).any()) if not other_imgs.empty else 0

        missed_records.append({
            "image_id": img_id,
            "lesion_id": les_id,
            "dx_type": m_row["dx_type"],
            "age": m_row["age"],
            "sex": m_row["sex"],
            "localization": m_row["localization"],
            "ViT_probability": row["ViT_probability"],
            "Multimodal_probability": row["Multimodal_probability"],
            "M6_probability": row["M6_probability"],
            "num_images_in_lesion": total_les_imgs,
            "any_other_image_detected_ViT": any_other_vit,
            "any_other_image_detected_MM": any_other_mm,
            "any_other_image_detected_M6": any_other_m6,
            "asymmetry_score": round(float(clin_vec[0]), 4),
            "border_irregularity": round(float(clin_vec[3]), 4),
            "color_h_std": round(float(clin_vec[7]), 4),
            "color_s_std": round(float(clin_vec[8]), 4),
            "color_v_std": round(float(clin_vec[9]), 4),
            "color_available": round(float(clin_vec[10]), 4),
            "border_available": round(float(clin_vec[4]), 4),
        })

    df_missed_audit = pd.DataFrame(missed_records)
    df_missed_audit.to_csv(REPORTS_DIR / "CEFM_STEP13_MISSED_MELANOMA_ANALYSIS.csv", index=False)
    print(f"Saved missed melanoma audit to {REPORTS_DIR / 'CEFM_STEP13_MISSED_MELANOMA_ANALYSIS.csv'}")

    # Comparative analysis: Missed vs Detected Melanomas
    detected_mel = mel_test[~mel_test["image_id"].isin(missed_all["image_id"])]
    detected_indices = [img_to_idx[img_id] for img_id in detected_mel["image_id"]]
    missed_indices = [img_to_idx[img_id] for img_id in missed_all["image_id"]]

    detected_clin = test_ds["clinical"][detected_indices].numpy()
    missed_clin = test_ds["clinical"][missed_indices].numpy()

    print("\n--- Clinical Feature Comparison: Missed (N=27) vs Detected (N=139) Melanomas ---")
    feat_names = ["asymmetry_score", "border_irregularity", "color_h_std", "color_s_std", "color_v_std"]
    feat_dims = [0, 3, 7, 8, 9]
    for name, dim in zip(feat_names, feat_dims):
        m_mean, m_std = float(missed_clin[:, dim].mean()), float(missed_clin[:, dim].std())
        d_mean, d_std = float(detected_clin[:, dim].mean()), float(detected_clin[:, dim].std())
        print(f"  {name:<20}: Missed={m_mean:.4f} +- {m_std:.4f} | Detected={d_mean:.4f} +- {d_std:.4f}")

    # Demographic comparison
    missed_meta = meta_df.loc[missed_all["image_id"]]
    detected_meta = meta_df.loc[detected_mel["image_id"]]
    print(f"\nMean age: Missed={missed_meta['age'].mean():.1f} | Detected={detected_meta['age'].mean():.1f}")
    print(f"Sex: Missed=\n{missed_meta['sex'].value_counts(normalize=True).to_dict()}\nDetected=\n{detected_meta['sex'].value_counts(normalize=True).to_dict()}")
    print(f"Top localizations missed:\n{missed_meta['localization'].value_counts().head(5).to_dict()}")

    # =========================================================================
    # PHASE 5: CLINICAL METADATA FEASIBILITY INVESTIGATION
    # =========================================================================
    print("\n[PHASE 5] Investigating Clinical Metadata Feasibility (Age, Sex, Site)...")
    full_meta = pd.read_csv(META_CSV_PATH)
    print(f"Total HAM10000 metadata samples: {len(full_meta)}")
    print(f"Missing age: {full_meta['age'].isna().sum()} ({full_meta['age'].isna().mean()*100:.2f}%)")
    print(f"Missing sex: {(full_meta['sex'] == 'unknown').sum()} ({(full_meta['sex'] == 'unknown').mean()*100:.2f}%)")
    print(f"Missing localization: {(full_meta['localization'] == 'unknown').sum()} ({(full_meta['localization'] == 'unknown').mean()*100:.2f}%)")

    # Melanoma prevalence by metadata groups
    full_meta["is_mel"] = (full_meta["dx"] == "mel").astype(int)

    # 1. Sex
    sex_prev = full_meta.groupby("sex")["is_mel"].agg(["count", "sum", "mean"]).rename(columns={"count": "total_samples", "sum": "melanoma_count", "mean": "melanoma_prevalence"})
    print("\nMelanoma prevalence by Sex:")
    print(sex_prev)

    # 2. Age brackets
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "21-40", "41-60", "61-80", "81+"]
    full_meta["age_group"] = pd.cut(full_meta["age"], bins=bins, labels=labels)
    age_prev = full_meta.groupby("age_group")["is_mel"].agg(["count", "sum", "mean"]).rename(columns={"count": "total_samples", "sum": "melanoma_count", "mean": "melanoma_prevalence"})
    print("\nMelanoma prevalence by Age Group:")
    print(age_prev)

    # 3. Localization
    site_prev = full_meta.groupby("localization")["is_mel"].agg(["count", "sum", "mean"]).rename(columns={"count": "total_samples", "sum": "melanoma_count", "mean": "melanoma_prevalence"}).sort_values(by="melanoma_prevalence", ascending=False)
    print("\nMelanoma prevalence by Anatomical Site:")
    print(site_prev)

    # Save metadata feasibility summary
    metadata_feasibility = {
        "metadata_completeness": {
            "total_images": len(full_meta),
            "age_available_pct": round(float((1 - full_meta["age"].isna().mean()) * 100), 2),
            "sex_available_pct": round(float((full_meta["sex"] != "unknown").mean() * 100), 2),
            "localization_available_pct": round(float((full_meta["localization"] != "unknown").mean() * 100), 2),
        },
        "prevalence_by_sex": sex_prev.reset_index().to_dict(orient="records"),
        "prevalence_by_age": age_prev.reset_index().to_dict(orient="records"),
        "prevalence_by_site": site_prev.reset_index().to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "CEFM_STEP13_METADATA_FEASIBILITY.json", "w") as f:
        json.dump(metadata_feasibility, f, indent=2)
    print(f"Saved metadata feasibility to {REPORTS_DIR / 'CEFM_STEP13_METADATA_FEASIBILITY.json'}")

    # =========================================================================
    # PHASE 6: UNBIASED EVALUATION OF VALIDATION-SELECTED OPERATING POINTS ON TEST SET
    # =========================================================================
    print("\n[PHASE 6] Evaluating Validation-Selected Operating Points on Untouched Test Set...")
    # Single unbiased test evaluation using thresholds chosen in Phase 1
    unbiased_test_records = []
    for model_name, cfg in models_config.items():
        # Evaluate validation operating points on test set
        for pt_name, pt_info in cfg["val_points"].items():
            thresh = pt_info["threshold"]
            # Image-level
            img_m = calculate_metrics_at_threshold(y_test_img, cfg["img_prob"], thresh)
            img_m.update({
                "model": model_name,
                "operating_point": pt_name,
                "threshold_selected_on_val": thresh,
                "val_sensitivity": pt_info["sensitivity"],
                "val_specificity": pt_info["specificity"],
                "val_f1": pt_info["f1_score"],
                "level": "image",
                "aggregation": "none",
            })
            unbiased_test_records.append(img_m)

            # Lesion-level Max
            l_max_m = calculate_metrics_at_threshold(y_test_lesion, cfg["lesion_max"], thresh)
            l_max_m.update({
                "model": model_name,
                "operating_point": pt_name,
                "threshold_selected_on_val": thresh,
                "val_sensitivity": pt_info["sensitivity"],
                "val_specificity": pt_info["specificity"],
                "val_f1": pt_info["f1_score"],
                "level": "lesion",
                "aggregation": "max",
            })
            unbiased_test_records.append(l_max_m)

    df_unbiased_test = pd.DataFrame(unbiased_test_records)
    df_unbiased_test.to_csv(REPORTS_DIR / "CEFM_STEP13_TEST_UNBIASED_CALIBRATED.csv", index=False)
    print(f"Saved unbiased test results to {REPORTS_DIR / 'CEFM_STEP13_TEST_UNBIASED_CALIBRATED.csv'}")

    print("\n--- Unbiased Test Performance of Validation-Selected Operating Points ---")
    print(df_unbiased_test[["model", "operating_point", "threshold_selected_on_val", "level", "aggregation", "sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "fn", "fp"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL PHASES (1-6) EXECUTED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()

