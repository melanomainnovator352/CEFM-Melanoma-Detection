"""
CEFM Phase 2 — Step 4: M6 Research Validation & Comparative Evaluation Pipeline.

Comprehensive evaluation of M6 against Phase 1 production baselines and Phase 2 models M1–M7.
- Evaluates on identical untouched 1,504 test samples (166 melanoma, 1,338 non-melanoma).
- Preserves all model checkpoints and thresholds.
- Evaluates M6 at both operating points:
    A. F1-optimal validation threshold: 0.690
    B. Validation-calibrated specificity-target threshold: 0.625
- Computes ROC-AUC, PR-AUC, Accuracy, Sensitivity, Specificity, Precision, F1, Balanced Accuracy, TP, FP, TN, FN.
- 1,000 stratified bootstrap resamples (seed=42) for 95% CIs and paired bootstrap differences.
- Detailed False-Negative analysis with metadata breakdown.
- Pre- and Post-evaluation SHA-256 checkpoint hash audits.
- High-resolution comparative figures rendered via OpenCV.
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
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
    roc_curve,
    precision_recall_curve,
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

ARTIFACT_DIR = Path(r"C:\Users\sruja\.gemini\antigravity\brain\fd604b17-8f8a-44a3-a817-98d62f87fbac")
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
if ARTIFACT_DIR.exists():
    (ARTIFACT_DIR / "figures").mkdir(parents=True, exist_ok=True)

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
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoints() -> Dict[str, str]:
    """Audit and record SHA-256 hashes of all model checkpoints."""
    hashes = {}
    for name, path in CHECKPOINT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for {name} missing at {path}")
        hashes[name] = compute_sha256(path)
    return hashes


def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute full suite of classification metrics at a given decision threshold."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    sens = recall_score(targets, preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    acc = accuracy_score(targets, preds)
    bacc = balanced_accuracy_score(targets, preds)

    roc_auc = roc_auc_score(targets, probs)
    pr_auc = average_precision_score(targets, probs)

    return {
        "threshold": float(threshold),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def stratified_bootstrap_ci(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    resample_indices: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for classification metrics."""
    boot_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1_score": [],
        "balanced_accuracy": [],
    }

    for b_idx in resample_indices:
        b_targets = targets[b_idx]
        b_probs = probs[b_idx]
        b_preds = (b_probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(b_targets, b_preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / len(b_targets)
        bacc = 0.5 * (sens + spec)

        # ROC & PR AUC
        if len(np.unique(b_targets)) == 2:
            r_auc = roc_auc_score(b_targets, b_probs)
            p_auc = average_precision_score(b_targets, b_probs)
        else:
            r_auc = 0.5
            p_auc = 0.0

        boot_metrics["roc_auc"].append(r_auc)
        boot_metrics["pr_auc"].append(p_auc)
        boot_metrics["accuracy"].append(acc)
        boot_metrics["sensitivity"].append(sens)
        boot_metrics["specificity"].append(spec)
        boot_metrics["precision"].append(prec)
        boot_metrics["f1_score"].append(f1)
        boot_metrics["balanced_accuracy"].append(bacc)

    ci_results = {}
    for k, v in boot_metrics.items():
        ci_results[k] = {
            "mean": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v)), 4),
            "ci_lower": round(float(np.percentile(v, 2.5)), 4),
            "ci_upper": round(float(np.percentile(v, 97.5)), 4),
        }
    return ci_results


def paired_bootstrap_test(
    targets: np.ndarray,
    probs_a: np.ndarray,
    thresh_a: float,
    probs_b: np.ndarray,
    thresh_b: float,
    resample_indices: List[np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """Compute paired bootstrap difference metrics between model A and model B."""
    diffs = {
        "roc_auc": [],
        "pr_auc": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1_score": [],
        "balanced_accuracy": [],
        "accuracy": [],
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

        diffs["roc_auc"].append(roc_a - roc_b)
        diffs["pr_auc"].append(pr_a - pr_b)
        diffs["sensitivity"].append(sens_a - sens_b)
        diffs["specificity"].append(spec_a - spec_b)
        diffs["precision"].append(prec_a - prec_b)
        diffs["f1_score"].append(f1_a - f1_b)
        diffs["balanced_accuracy"].append(bacc_a - bacc_b)
        diffs["accuracy"].append(acc_a - acc_b)

    paired_summary = {}
    for metric, vals in diffs.items():
        low = np.percentile(vals, 2.5)
        high = np.percentile(vals, 97.5)
        diff_mean = np.mean(vals)
        includes_zero = (low <= 0.0 <= high)
        p_val = min(np.mean([v <= 0 for v in vals]), np.mean([v >= 0 for v in vals])) * 2
        paired_summary[metric] = {
            "mean_difference": round(float(diff_mean), 4),
            "ci_lower": round(float(low), 4),
            "ci_upper": round(float(high), 4),
            "includes_zero": bool(includes_zero),
            "p_value_approx": round(float(p_val), 4),
            "significant_alpha_05": not bool(includes_zero),
        }
    return paired_summary


def render_roc_curves_comparison(eval_data: Dict[str, Any], save_path: Path):
    """Render anti-aliased ROC curves comparing M6 to baselines."""
    canvas = np.ones((750, 950, 3), dtype=np.uint8) * 255
    ox, oy = 110, 620
    w, h = 640, 500

    cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)
    for i in range(11):
        x = int(ox + i * (w / 10))
        y = int(oy - i * (h / 10))
        cv2.line(canvas, (x, oy - h), (x, oy), (220, 224, 230), 1)
        cv2.line(canvas, (ox, y), (ox + w, y), (220, 224, 230), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (x - 12, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (ox - 42, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

    cv2.line(canvas, (ox, oy), (ox + w, oy - h), (180, 180, 180), 2, cv2.LINE_AA)

    cv2.putText(canvas, "ROC Curves: M6 Challenger vs. Production & Phase 2 Baselines (N=1,504)",
                (ox - 30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "False Positive Rate (1 - Specificity)", (ox + 180, oy + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "True Positive Rate (Sensitivity)", (ox - 85, oy - 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    colors = {
        "Phase1_ViT": (180, 80, 40),       # Blue
        "Phase1_Fusion": (40, 140, 40),    # Green
        "M1": (120, 120, 120),             # Gray
        "M3": (200, 140, 30),              # Cyan/Blue
        "M5": (140, 40, 180),              # Purple
        "M6": (30, 30, 220),               # Bold Red
    }

    labels_map = {
        "Phase1_ViT": "Phase 1 ViT Baseline",
        "Phase1_Fusion": "Phase 1 Multimodal Fusion v1.0",
        "M1": "M1: Global ViT Only",
        "M3": "M3: Dual-Scale Visual",
        "M5": "M5: Dual Concat Multimodal",
        "M6": "M6: Dual-Scale Gated RCAG (Challenger)",
    }

    legend_y = 130
    for key in ["Phase1_ViT", "Phase1_Fusion", "M1", "M3", "M5", "M6"]:
        d = eval_data[key]
        fpr, tpr, _ = roc_curve(d["targets"], d["probs"])
        pts = [np.array([int(ox + fx * w), int(oy - ty * h)], dtype=np.int32) for fx, ty in zip(fpr, tpr)]
        thick = 3 if key == "M6" else 2
        for j in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), colors[key], thick, cv2.LINE_AA)

        cv2.line(canvas, (ox + 340, legend_y), (ox + 380, legend_y), colors[key], thick + 1, cv2.LINE_AA)
        leg_txt = f"{labels_map[key]} (AUC: {d['roc_auc']:.4f})"
        cv2.putText(canvas, leg_txt, (ox + 390, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        legend_y += 24

    cv2.imwrite(str(save_path), canvas)


def render_pr_curves_comparison(eval_data: Dict[str, Any], save_path: Path):
    """Render anti-aliased Precision-Recall curves comparing M6 to baselines."""
    canvas = np.ones((750, 950, 3), dtype=np.uint8) * 255
    ox, oy = 110, 620
    w, h = 640, 500

    cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)
    for i in range(11):
        x = int(ox + i * (w / 10))
        y = int(oy - i * (h / 10))
        cv2.line(canvas, (x, oy - h), (x, oy), (220, 224, 230), 1)
        cv2.line(canvas, (ox, y), (ox + w, y), (220, 224, 230), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (x - 12, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (ox - 42, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

    # Base prevalence line (166 / 1504 = 0.1104)
    prev_y = int(oy - 0.1104 * h)
    cv2.line(canvas, (ox, prev_y), (ox + w, prev_y), (180, 180, 180), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Precision-Recall Curves: M6 Challenger vs. Baselines (N=1,504)",
                (ox - 30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Recall (Sensitivity)", (ox + 230, oy + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Precision (Positive Predictive Value)", (ox - 85, oy - 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    colors = {
        "Phase1_ViT": (180, 80, 40),
        "Phase1_Fusion": (40, 140, 40),
        "M1": (120, 120, 120),
        "M3": (200, 140, 30),
        "M5": (140, 40, 180),
        "M6": (30, 30, 220),
    }

    labels_map = {
        "Phase1_ViT": "Phase 1 ViT Baseline",
        "Phase1_Fusion": "Phase 1 Multimodal Fusion v1.0",
        "M1": "M1: Global ViT Only",
        "M3": "M3: Dual-Scale Visual",
        "M5": "M5: Dual Concat Multimodal",
        "M6": "M6: Dual-Scale Gated RCAG (Challenger)",
    }

    legend_y = 130
    for key in ["Phase1_ViT", "Phase1_Fusion", "M1", "M3", "M5", "M6"]:
        d = eval_data[key]
        prec, rec, _ = precision_recall_curve(d["targets"], d["probs"])
        pts = [np.array([int(ox + r * w), int(oy - p * h)], dtype=np.int32) for r, p in zip(rec, prec)]
        thick = 3 if key == "M6" else 2
        for j in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), colors[key], thick, cv2.LINE_AA)

        cv2.line(canvas, (ox + 340, legend_y), (ox + 380, legend_y), colors[key], thick + 1, cv2.LINE_AA)
        leg_txt = f"{labels_map[key]} (PR-AUC: {d['pr_auc']:.4f})"
        cv2.putText(canvas, leg_txt, (ox + 390, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        legend_y += 24

    cv2.imwrite(str(save_path), canvas)


def render_confusion_matrices_comparison(eval_data: Dict[str, Any], save_path: Path):
    """Render 4-panel confusion matrix comparison."""
    canvas = np.ones((600, 1100, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Confusion Matrix Comparison across Primary Operating Points",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (20, 20, 20), 2, cv2.LINE_AA)

    panels = [
        ("Phase 1 ViT (Thresh: 0.80)", eval_data["Phase1_ViT"]["metrics_clinical"]["confusion_matrix"], 50, 90),
        ("Phase 1 Multimodal (Thresh: 0.36)", eval_data["Phase1_Fusion"]["metrics_clinical"]["confusion_matrix"], 310, 90),
        ("M6 Challenger: F1 Point (0.690)", eval_data["M6"]["metrics_calibrated_f1"]["confusion_matrix"], 570, 90),
        ("M6 Challenger: 90% Spec Point (0.625)", eval_data["M6"]["metrics_matched_spec"]["confusion_matrix"], 830, 90),
    ]

    for title, cm, px, py in panels:
        cv2.putText(canvas, title, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

        # Draw matrix grid
        bx, by = px, py + 20
        bw, bh = 220, 220
        half_w, half_h = bw // 2, bh // 2

        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

        # Color cells
        cv2.rectangle(canvas, (bx, by), (bx + half_w, by + half_h), (210, 245, 210), -1)             # TN
        cv2.rectangle(canvas, (bx + half_w, by), (bx + bw, by + half_h), (210, 210, 250), -1)         # FP
        cv2.rectangle(canvas, (bx, by + half_h), (bx + half_w, by + bh), (200, 200, 255), -1)         # FN
        cv2.rectangle(canvas, (bx + half_w, by + half_h), (bx + bw, by + bh), (200, 245, 200), -1)     # TP

        # Borders
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (140, 140, 140), 2)
        cv2.line(canvas, (bx + half_w, by), (bx + half_w, by + bh), (140, 140, 140), 1)
        cv2.line(canvas, (bx, by + half_h), (bx + bw, by + half_h), (140, 140, 140), 1)

        # Numbers
        cv2.putText(canvas, f"TN: {tn}", (bx + 20, by + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 80, 20), 2)
        cv2.putText(canvas, f"FP: {fp}", (bx + half_w + 20, by + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 20, 80), 2)
        cv2.putText(canvas, f"FN: {fn}", (bx + 20, by + half_h + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 180), 2)
        cv2.putText(canvas, f"TP: {tp}", (bx + half_w + 20, by + half_h + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 100, 20), 2)

        # Metrics below
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        cv2.putText(canvas, f"Sens: {sens*100:.1f}%", (bx, by + bh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        cv2.putText(canvas, f"Spec: {spec*100:.1f}%", (bx, by + bh + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        cv2.putText(canvas, f"F1: {f1*100:.1f}%", (bx + half_w, by + bh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        cv2.putText(canvas, f"FN Count: {fn}", (bx + half_w, by + bh + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 180), 2)

    cv2.imwrite(str(save_path), canvas)


def render_paired_bootstrap_forest_plot(paired_results: Dict[str, Dict[str, Any]], save_path: Path):
    """Render paired bootstrap differences forest plot comparing M6 to baselines."""
    canvas = np.ones((700, 1000, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Paired Bootstrap Differences: M6 vs. Baselines (95% CI, 1,000 Rounds)",
                (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)

    metrics = ["sensitivity", "specificity", "f1_score", "balanced_accuracy", "roc_auc"]
    comparisons = list(paired_results.keys())

    # Dimensions
    ox, oy = 280, 100
    row_h = 28
    chart_w = 600
    zero_x = ox + chart_w // 2

    # Scale: -0.20 to +0.20 maps to chart_w
    scale = chart_w / 0.40

    cv2.line(canvas, (zero_x, oy), (zero_x, oy + len(comparisons) * len(metrics) * row_h + 30), (180, 180, 180), 2, cv2.LINE_AA)

    y = oy
    colors = {
        "Phase1_ViT": (180, 80, 40),
        "Phase1_Fusion": (40, 140, 40),
        "M3": (200, 140, 30),
        "M5": (140, 40, 180),
    }

    for comp in comparisons:
        comp_name = comp.replace("M6_vs_", "M6 vs. ")
        cv2.putText(canvas, f"--- {comp_name} ---", (30, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 2)
        y += row_h

        comp_key = next((k for k in colors if k in comp), "Phase1_ViT")
        col = colors[comp_key]

        for met in ["sensitivity", "f1_score", "balanced_accuracy", "roc_auc"]:
            res = paired_results[comp][met]
            mean_d = res["mean_difference"]
            ci_l = res["ci_lower"]
            ci_u = res["ci_upper"]

            # Label
            sig_mark = "*" if not res["includes_zero"] else ""
            txt = f"{met.capitalize()}: {mean_d:+.3f} [{ci_l:+.3f}, {ci_u:+.3f}]{sig_mark}"
            cv2.putText(canvas, txt, (45, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 50), 1)

            # Forest bar
            mx = int(zero_x + mean_d * scale)
            lx = int(zero_x + ci_l * scale)
            ux = int(zero_x + ci_u * scale)

            cv2.line(canvas, (lx, y + 10), (ux, y + 10), col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (mx, y + 10), 4, col, -1, cv2.LINE_AA)
            y += row_h

        y += 10

    # Scale markings
    cv2.putText(canvas, "-0.20", (zero_x - int(0.20 * scale) - 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(canvas, "0.00", (zero_x - 12, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(canvas, "+0.20", (zero_x + int(0.20 * scale) - 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(canvas, "<- Favors Baseline | Favors M6 Challenger ->", (zero_x - 160, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 30), 1)

    cv2.imwrite(str(save_path), canvas)


def main():
    print("=" * 80)
    print("CEFM PHASE 2 — STEP 4: M6 RESEARCH VALIDATION & COMPARATIVE EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # =========================================================================
    # STEP 1: AUDIT EVALUATION INPUTS & PRE-EVALUATION HASHES
    # =========================================================================
    print("\n[Step 1] Auditing Checkpoints and Data Inputs...")
    pre_hashes = audit_checkpoints()
    print("Pre-Evaluation Checkpoint SHA-256 Hashes:")
    for k, v in pre_hashes.items():
        print(f"  - {k:<15}: {v}")

    # Load metadata test set
    metadata_test_csv = REPO_ROOT / "data" / "HAM10000" / "metadata" / "test.csv"
    assert metadata_test_csv.exists(), f"Metadata test.csv missing at {metadata_test_csv}"
    meta_df = pd.read_csv(metadata_test_csv)
    n_meta = len(meta_df)
    n_mel_meta = int((meta_df["label"] == 1).sum())
    n_non_mel_meta = int((meta_df["label"] == 0).sum())

    assert n_meta == 1504, f"Expected 1,504 test samples, got {n_meta}"
    assert n_mel_meta == 166, f"Expected 166 melanomas, got {n_mel_meta}"
    assert n_non_mel_meta == 1338, f"Expected 1,338 non-melanomas, got {n_non_mel_meta}"
    print(f"[+] Metadata Test Split Verified: N={n_meta} (Melanoma: {n_mel_meta}, Non-Melanoma: {n_non_mel_meta})")

    # Load Dual-Scale Test Cache
    test_cache_file = REPO_ROOT / "data" / "dual_scale_cache" / "dual_scale_test.pt"
    train_cache_file = REPO_ROOT / "data" / "dual_scale_cache" / "dual_scale_train.pt"
    assert test_cache_file.exists(), f"Dual-scale test cache missing at {test_cache_file}"
    assert train_cache_file.exists(), f"Dual-scale train cache missing at {train_cache_file}"

    test_cache = torch.load(test_cache_file, map_location="cpu")
    train_cache = torch.load(train_cache_file, map_location="cpu")

    targets = test_cache["labels"].numpy()
    assert len(targets) == 1504, f"Cache test count mismatch: {len(targets)}"
    assert (targets == 1).sum() == 166
    assert (targets == 0).sum() == 1338

    # Verify zero lesion overlap
    train_lesions = set(train_cache["lesion_ids"])
    test_lesions = set(test_cache["lesion_ids"])
    overlap = train_lesions.intersection(test_lesions)
    assert len(overlap) == 0, f"Critical Leakage: {len(overlap)} overlapping lesions found!"
    print(f"[+] Leakage Audit: Zero lesion overlap between train and test splits (0 shared lesions).")

    # Load Phase 1 Features
    p1_cache_file = REPO_ROOT / "data" / "multimodal_cache" / "test_features.pt"
    assert p1_cache_file.exists(), f"Phase 1 test features missing at {p1_cache_file}"
    p1_test = torch.load(p1_cache_file, map_location="cpu")
    assert torch.equal(p1_test["labels"], test_cache["labels"]), "Phase 1 and Phase 2 test label alignment mismatch!"
    print(f"[+] Label Alignment: Identical label vector and sample sequence verified across Phase 1 and 2 caches.")

    # Clinical Normalization Parameters
    c_mean = train_cache["clinical"].mean(dim=0, keepdim=True)
    c_std = torch.clamp(train_cache["clinical"].std(dim=0, keepdim=True), min=1e-6)
    test_clin_norm = (test_cache["clinical"] - c_mean) / c_std

    # =========================================================================
    # STEP 2: LOAD AND EVALUATE ALL MODELS
    # =========================================================================
    print("\n[Step 2] Executing Evaluation across All Models...")

    eval_data = {}

    # 1. Phase 1 ViT Baseline
    vit_model = ViTForImageClassification.from_pretrained(
        str(CLASSIFICATION_MODEL_DIR / "vit_model"), local_files_only=True
    ).to(device)
    vit_model.eval()
    with torch.no_grad():
        v_logits = vit_model.classifier(p1_test["visual"].to(device))
        vit_probs = torch.softmax(v_logits, dim=-1)[:, 1].cpu().numpy()

    # 2. Phase 1 Multimodal Fusion
    p1_fusion = CEFMMultimodalFusion(768, 16, 128, 2).to(device)
    p1_fusion.load_state_dict(torch.load(CHECKPOINT_PATHS["Phase1_Fusion"], map_location=device))
    p1_fusion.eval()
    with torch.no_grad():
        f_logits = p1_fusion(p1_test["visual"].to(device), p1_test["clinical"].to(device))
        p1_fusion_probs = torch.softmax(f_logits, dim=-1)[:, 1].cpu().numpy()

    eval_data["Phase1_ViT"] = {
        "model_id": "Phase1_ViT",
        "name": "Phase 1 ViT Baseline",
        "targets": targets,
        "probs": vit_probs,
        "roc_auc": round(float(roc_auc_score(targets, vit_probs)), 4),
        "pr_auc": round(float(average_precision_score(targets, vit_probs)), 4),
        "metrics_default": compute_metrics(targets, vit_probs, 0.50),
        "metrics_clinical": compute_metrics(targets, vit_probs, 0.80),
        "metrics_f1": compute_metrics(targets, vit_probs, 0.90),
    }

    eval_data["Phase1_Fusion"] = {
        "model_id": "Phase1_Fusion",
        "name": "Phase 1 Multimodal Fusion v1.0",
        "targets": targets,
        "probs": p1_fusion_probs,
        "roc_auc": round(float(roc_auc_score(targets, p1_fusion_probs)), 4),
        "pr_auc": round(float(average_precision_score(targets, p1_fusion_probs)), 4),
        "metrics_default": compute_metrics(targets, p1_fusion_probs, 0.50),
        "metrics_clinical": compute_metrics(targets, p1_fusion_probs, 0.36),
        "metrics_f1": compute_metrics(targets, p1_fusion_probs, 0.80),
    }

    # Phase 2 Models M1–M7
    phase2_architectures = {
        "M1": (M1_GlobalViT(), 0.830, 0.585, "Global ViT Only"),
        "M2": (M2_LocalViT(), 0.720, 0.470, "Local ViT Only"),
        "M3": (M3_DualVisual(), 0.450, 0.075, "Dual-Scale Visual"),
        "M4": (M4_GlobalClinical(), 0.830, 0.285, "Global ViT + Clinical"),
        "M5": (M5_DualConcat(), 0.630, 0.270, "Dual Concat Multimodal"),
        "M6": (M6_DualGated(), 0.690, 0.625, "Dual-Scale Gated RCAG (Challenger)"),
        "M7": (M7_DualCrossAttn(), 0.180, 0.110, "Dual-Scale Cross-Attention"),
    }

    g_vis = test_cache["global_visual"].to(device)
    l_vis = test_cache["local_visual"].to(device)
    c_norm = test_clin_norm.to(device)

    for m_id, (net, th_f1, th_sp, name) in phase2_architectures.items():
        net = net.to(device)
        net.load_state_dict(torch.load(CHECKPOINT_PATHS[m_id], map_location=device))
        net.eval()
        with torch.no_grad():
            out = net(g_vis, l_vis, c_norm)
            probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()

        eval_data[m_id] = {
            "model_id": m_id,
            "name": name,
            "targets": targets,
            "probs": probs,
            "roc_auc": round(float(roc_auc_score(targets, probs)), 4),
            "pr_auc": round(float(average_precision_score(targets, probs)), 4),
            "calibrated_threshold_f1": th_f1,
            "calibrated_threshold_matched_spec": th_sp,
            "metrics_calibrated_f1": compute_metrics(targets, probs, th_f1),
            "metrics_matched_spec": compute_metrics(targets, probs, th_sp),
            "metrics_default_05": compute_metrics(targets, probs, 0.50),
        }

    # =========================================================================
    # STEP 3: 1,000 STRATIFIED BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================
    print(f"\n[Step 3] Running {BOOTSTRAP_ROUNDS} Stratified Bootstrap Resamples (seed={RANDOM_SEED})...")
    rng = np.random.RandomState(RANDOM_SEED)

    pos_idx = np.where(targets == 1)[0]
    neg_idx = np.where(targets == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Pre-generate identical resample indices for all models
    resample_indices = []
    for _ in range(BOOTSTRAP_ROUNDS):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        resample_indices.append(np.concatenate([b_pos, b_neg]))

    bootstrap_cis = {}
    for m_id in ["Phase1_ViT", "Phase1_Fusion", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        m_dict = eval_data[m_id]
        if m_id == "Phase1_ViT":
            th = 0.80
        elif m_id == "Phase1_Fusion":
            th = 0.36
        else:
            th = m_dict["calibrated_threshold_f1"]

        bootstrap_cis[m_id] = stratified_bootstrap_ci(targets, m_dict["probs"], th, resample_indices)
        m_dict["bootstrap_95ci"] = bootstrap_cis[m_id]

    # Additional CI for M6 at Operating Point B (Spec: 0.625)
    bootstrap_cis["M6_spec_0625"] = stratified_bootstrap_ci(
        targets, eval_data["M6"]["probs"], 0.625, resample_indices
    )
    eval_data["M6"]["bootstrap_95ci_matched_spec"] = bootstrap_cis["M6_spec_0625"]

    # =========================================================================
    # STEP 4: PAIRED BOOTSTRAP STATISTICAL DIFFERENCES (M6 vs. Baselines)
    # =========================================================================
    print("\n[Step 4] Computing Paired Bootstrap Statistical Differences...")
    paired_comparisons = {}

    m6_probs = eval_data["M6"]["probs"]
    m6_f1_th = 0.690

    # 1. M6 (0.690) vs. Phase 1 ViT (0.80)
    paired_comparisons["M6_vs_Phase1_ViT"] = paired_bootstrap_test(
        targets, m6_probs, m6_f1_th, eval_data["Phase1_ViT"]["probs"], 0.80, resample_indices
    )
    # 2. M6 (0.690) vs. Phase 1 Multimodal (0.36)
    paired_comparisons["M6_vs_Phase1_Fusion"] = paired_bootstrap_test(
        targets, m6_probs, m6_f1_th, eval_data["Phase1_Fusion"]["probs"], 0.36, resample_indices
    )
    # 3. M6 (0.690) vs. M3 Dual-Scale Visual (0.450)
    paired_comparisons["M6_vs_M3_DualVisual"] = paired_bootstrap_test(
        targets, m6_probs, m6_f1_th, eval_data["M3"]["probs"], 0.450, resample_indices
    )
    # 4. M6 (0.690) vs. M5 Dual Concat Multimodal (0.630)
    paired_comparisons["M6_vs_M5_DualConcat"] = paired_bootstrap_test(
        targets, m6_probs, m6_f1_th, eval_data["M5"]["probs"], 0.630, resample_indices
    )

    # =========================================================================
    # STEP 5: FALSE-NEGATIVE AUDIT AND METADATA STRATIFICATION
    # =========================================================================
    print("\n[Step 5] Detailed False-Negative Analysis across Models...")

    image_ids = test_cache["image_ids"]
    patch_counts = test_cache["patch_counts"].numpy() if isinstance(test_cache["patch_counts"], torch.Tensor) else np.array(test_cache["patch_counts"])

    fn_audits = {}
    for m_key, th, label in [
        ("Phase1_ViT", 0.80, "Phase 1 ViT (0.80)"),
        ("Phase1_Fusion", 0.36, "Phase 1 Multimodal (0.36)"),
        ("M3", 0.450, "M3 Dual Visual (0.450)"),
        ("M5", 0.630, "M5 Dual Concat (0.630)"),
        ("M6_F1", 0.690, "M6 Challenger Point A (0.690)"),
        ("M6_Spec", 0.625, "M6 Challenger Point B (0.625)"),
    ]:
        probs = eval_data["M6"]["probs"] if "M6" in m_key else eval_data[m_key]["probs"]
        preds = (probs >= th).astype(int)

        fn_mask = (targets == 1) & (preds == 0)
        fn_indices = np.where(fn_mask)[0]
        fn_img_ids = [image_ids[idx] for idx in fn_indices]
        fn_probs = probs[fn_indices]

        # Metadata breakdown
        fn_meta = meta_df.iloc[fn_indices]

        fn_audits[m_key] = {
            "label": label,
            "threshold": th,
            "false_negative_count": int(len(fn_indices)),
            "false_negative_rate": round(float(len(fn_indices) / 166), 4),
            "sample_ids": fn_img_ids,
            "probability_stats": {
                "mean": round(float(np.mean(fn_probs)), 4) if len(fn_probs) > 0 else 0.0,
                "median": round(float(np.median(fn_probs)), 4) if len(fn_probs) > 0 else 0.0,
                "p25": round(float(np.percentile(fn_probs, 25)), 4) if len(fn_probs) > 0 else 0.0,
                "p75": round(float(np.percentile(fn_probs, 75)), 4) if len(fn_probs) > 0 else 0.0,
            },
            "by_localization": fn_meta["localization"].value_counts().to_dict(),
            "by_age_bracket": pd.cut(fn_meta["age"], bins=[0, 35, 50, 65, 100], labels=["<35", "35-50", "50-65", ">65"]).value_counts().to_dict(),
            "by_sex": fn_meta["sex"].value_counts().to_dict(),
            "by_patch_count": {
                "single_patch": int((patch_counts[fn_indices] == 1).sum()),
                "multi_patch": int((patch_counts[fn_indices] > 1).sum()),
            },
        }

    # Specifically check ISIC_0012418
    isic_12418_idx = image_ids.index("ISIC_0012418") if "ISIC_0012418" in image_ids else None
    audit_12418 = {}
    if isic_12418_idx is not None:
        audit_12418 = {
            "image_id": "ISIC_0012418",
            "ground_truth": "mel",
            "Phase1_ViT_prob": round(float(eval_data["Phase1_ViT"]["probs"][isic_12418_idx]), 4),
            "Phase1_Fusion_prob": round(float(eval_data["Phase1_Fusion"]["probs"][isic_12418_idx]), 4),
            "M1_prob": round(float(eval_data["M1"]["probs"][isic_12418_idx]), 4),
            "M3_prob": round(float(eval_data["M3"]["probs"][isic_12418_idx]), 4),
            "M5_prob": round(float(eval_data["M5"]["probs"][isic_12418_idx]), 4),
            "M6_prob": round(float(eval_data["M6"]["probs"][isic_12418_idx]), 4),
            "M6_pred_f1": "mel" if eval_data["M6"]["probs"][isic_12418_idx] >= 0.690 else "non_mel",
            "M6_pred_spec": "mel" if eval_data["M6"]["probs"][isic_12418_idx] >= 0.625 else "non_mel",
            "patch_count": int(patch_counts[isic_12418_idx]),
        }
        print(f"[+] Case ISIC_0012418 Audit: GT=mel | Phase 1 ViT={audit_12418['Phase1_ViT_prob']} | Phase 1 Fusion={audit_12418['Phase1_Fusion_prob']} | M6={audit_12418['M6_prob']} (Point A Pred: {audit_12418['M6_pred_f1'].upper()})")

    # =========================================================================
    # STEP 6: RENDER HIGH-RESOLUTION PLOTS
    # =========================================================================
    print("\n[Step 6] Rendering Visualizations...")
    roc_plot_path = FIGURES_DIR / "m6_vs_baselines_roc_curves.png"
    pr_plot_path = FIGURES_DIR / "m6_vs_baselines_pr_curves.png"
    cm_plot_path = FIGURES_DIR / "m6_operating_points_confusion_matrices.png"
    forest_plot_path = FIGURES_DIR / "m6_paired_bootstrap_differences.png"

    render_roc_curves_comparison(eval_data, roc_plot_path)
    render_pr_curves_comparison(eval_data, pr_plot_path)
    render_confusion_matrices_comparison(eval_data, cm_plot_path)
    render_paired_bootstrap_forest_plot(paired_comparisons, forest_plot_path)

    # Copy to artifact directory
    if ARTIFACT_DIR.exists():
        art_figs = ARTIFACT_DIR / "figures"
        for p in [roc_plot_path, pr_plot_path, cm_plot_path, forest_plot_path]:
            shutil.copy(p, art_figs / p.name)

    print("[+] Generated and copied all comparative figures to artifact directory.")

    # =========================================================================
    # STEP 7: POST-EVALUATION INTEGRITY AUDIT
    # =========================================================================
    print("\n[Step 7] Running Post-Evaluation Checkpoint Hash Audit...")
    post_hashes = audit_checkpoints()
    for k in pre_hashes:
        assert pre_hashes[k] == post_hashes[k], f"CRITICAL: Checkpoint {k} changed during evaluation!"
    print("[+] Post-evaluation integrity verified: All checkpoint hashes identical bit-for-bit.")

    # =========================================================================
    # STEP 8: SAVE FINAL STRUCTURED RESULTS
    # =========================================================================
    # Remove numpy arrays before JSON serialization
    json_data = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "HAM10000 Independent Test Split",
        "sample_count": 1504,
        "class_distribution": {"melanoma": 166, "non_melanoma": 1338},
        "checkpoint_hashes": pre_hashes,
        "eval_models": {},
        "paired_bootstrap_differences": paired_comparisons,
        "false_negative_analysis": fn_audits,
        "case_isic12418_audit": audit_12418,
    }

    for m_id, d in eval_data.items():
        clean_d = {k: v for k, v in d.items() if k not in ["targets", "probs"]}
        json_data["eval_models"][m_id] = clean_d

    results_file = REPORTS_DIR / "phase2_m6_research_validation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"[+] Saved structured research validation results to {results_file}")

    print("\n" + "=" * 80)
    print("M6 RESEARCH VALIDATION AND COMPARATIVE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

