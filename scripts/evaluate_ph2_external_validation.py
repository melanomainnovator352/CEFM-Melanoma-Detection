"""
CEFM Phase 2 Step 5 — Independent External Validation on PH² Dataset.
Evaluates frozen M6 Challenger Model, Phase 1 ViT Baseline, and Phase 1 Multimodal Fusion
on the independent PH² dermoscopy benchmark (N=200).

Strict Evaluation Protocol:
- Zero retraining or parameter mutation.
- Zero threshold recalibration on external labels.
- Preserves validation operating points:
    M6: T* = 0.690 (F1) and T* = 0.625 (90% Spec)
    Phase 1 ViT: T* = 0.900 (F1) and T* = 0.800 (Screening)
    Phase 1 Multimodal: T* = 0.800 (F1) and T* = 0.360 (Screening)
- Computes ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1, Balanced Acc, Confusion Matrices.
- 1,000 Stratified Bootstrap Resamples (seed=42) for 95% Confidence Intervals.
- Renders publication-quality anti-aliased figures directly via OpenCV.
"""

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
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.classification import classify

PH2_DIR = Path("D:/PH2")
PH2_IMAGES_DIR = PH2_DIR / "PH2 Dataset images"
PH2_TXT = PH2_DIR / "PH2_dataset.txt"

OUTPUT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures" / "phase2_external_validation"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_ARTIFACT_DIR = Path("C:/Users/sruja/.gemini/antigravity/brain/fd604b17-8f8a-44a3-a817-98d62f87fbac")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_ph2_metadata() -> pd.DataFrame:
    with open(PH2_TXT, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    records = []
    for l in lines:
        if l.startswith("||") and not l.startswith("||   Name"):
            parts = [p.strip() for p in l.split("||")[1:-1]]
            name = parts[0]
            histo = parts[1]
            clin = int(parts[2])
            features = [f.strip() for f in parts[3].split("|")]
            colors = parts[4] if len(parts) > 4 else ""
            
            # Ground truth: 0 = Common Nevus, 1 = Atypical Nevus, 2 = Melanoma
            label = 1 if clin == 2 else 0
            diag_name = "melanoma" if clin == 2 else ("common_nevus" if clin == 0 else "atypical_nevus")
            
            records.append({
                "image_id": name,
                "histo_diagnosis": histo,
                "clinical_diagnosis_code": clin,
                "diagnosis_name": diag_name,
                "label": label,
                "asymmetry_gt": features[0],
                "pigment_network_gt": features[1],
                "dots_globules_gt": features[2],
                "streaks_gt": features[3],
                "regression_areas_gt": features[4],
                "blue_whitish_veil_gt": features[5],
                "colors_gt": colors,
            })

    return pd.DataFrame(records)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(rec_arr, prec_arr))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    acc = (tp + tn) / len(y_true)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_acc": bacc,
        "accuracy": acc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    metric_samples = {
        "roc_auc": [],
        "pr_auc": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "balanced_acc": [],
        "accuracy": [],
    }

    for _ in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sample_pos, sample_neg])
        
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        
        m = compute_metrics(yt_b, yp_b, threshold)
        for k in metric_samples:
            metric_samples[k].append(m[k])

    ci_dict = {}
    for k, v in metric_samples.items():
        ci_dict[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "ci_lower": float(np.percentile(v, 2.5)),
            "ci_upper": float(np.percentile(v, 97.5)),
        }
    return ci_dict


# =========================================================================
# OPENCV PUBLICATION FIGURE RENDERERS
# =========================================================================

def render_ph2_roc_curves(eval_data: Dict[str, Any], save_path: Path):
    """Render anti-aliased ROC curves for PH2 external benchmark."""
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

    cv2.putText(canvas, "PH2 Independent External Validation: ROC Curves (N=200)",
                (ox - 30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "False Positive Rate (1 - Specificity)", (ox + 180, oy + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "True Positive Rate (Sensitivity)", (ox - 85, oy - 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    colors = {
        "Phase1_ViT": (180, 80, 40),        # Deep Blue
        "Phase1_Fusion": (40, 140, 40),     # Dark Green
        "M6_UNet": (30, 30, 220),           # Crimson Red
        "M6_GT": (140, 40, 180),            # Purple
    }

    labels_map = {
        "Phase1_ViT": "Phase 1 ViT Baseline",
        "Phase1_Fusion": "Phase 1 Multimodal Fusion v1.0",
        "M6_UNet": "Phase 2 M6 Challenger (VM-UNet Mask)",
        "M6_GT": "Phase 2 M6 Challenger (Expert GT Mask)",
    }

    legend_y = 130
    for key in ["Phase1_ViT", "Phase1_Fusion", "M6_UNet", "M6_GT"]:
        d = eval_data[key]
        fpr, tpr, _ = roc_curve(d["targets"], d["probs"])
        pts = [np.array([int(ox + fx * w), int(oy - ty * h)], dtype=np.int32) for fx, ty in zip(fpr, tpr)]
        thick = 3 if "M6" in key else 2
        for j in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), colors[key], thick, cv2.LINE_AA)

        cv2.line(canvas, (ox + 310, legend_y), (ox + 350, legend_y), colors[key], thick + 1, cv2.LINE_AA)
        leg_txt = f"{labels_map[key]} (AUC: {d['roc_auc']:.4f})"
        cv2.putText(canvas, leg_txt, (ox + 360, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        legend_y += 24

    cv2.imwrite(str(save_path), canvas)


def render_ph2_pr_curves(eval_data: Dict[str, Any], save_path: Path):
    """Render anti-aliased Precision-Recall curves for PH2 benchmark."""
    canvas = np.ones((750, 950, 3), dtype=np.uint8) * 255
    ox, oy = 110, 620
    w, h = 640, 500

    cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)
    for i in range(11):
        x = int(ox + i * (w / 10))
        y = int(oy - i * (h / 10))
        cv2.line(canvas, (ox, y), (ox + w, y), (220, 224, 230), 1)
        cv2.line(canvas, (x, oy - h), (x, oy), (220, 224, 230), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (x - 12, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (ox - 42, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

    # Base prevalence line (40 / 200 = 0.20)
    prev_y = int(oy - 0.20 * h)
    cv2.line(canvas, (ox, prev_y), (ox + w, prev_y), (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Melanoma Prevalence (0.20)", (ox + w - 210, prev_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    cv2.putText(canvas, "PH2 Independent External Validation: PR Curves (N=200)",
                (ox - 30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Recall (Sensitivity)", (ox + 230, oy + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Precision (Positive Predictive Value)", (ox - 85, oy - 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    colors = {
        "Phase1_ViT": (180, 80, 40),
        "Phase1_Fusion": (40, 140, 40),
        "M6_UNet": (30, 30, 220),
        "M6_GT": (140, 40, 180),
    }

    labels_map = {
        "Phase1_ViT": "Phase 1 ViT Baseline",
        "Phase1_Fusion": "Phase 1 Multimodal Fusion v1.0",
        "M6_UNet": "Phase 2 M6 Challenger (VM-UNet Mask)",
        "M6_GT": "Phase 2 M6 Challenger (Expert GT Mask)",
    }

    legend_y = 130
    for key in ["Phase1_ViT", "Phase1_Fusion", "M6_UNet", "M6_GT"]:
        d = eval_data[key]
        prec, rec, _ = precision_recall_curve(d["targets"], d["probs"])
        pts = [np.array([int(ox + rx * w), int(oy - py * h)], dtype=np.int32) for rx, py in zip(rec, prec)]
        thick = 3 if "M6" in key else 2
        for j in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), colors[key], thick, cv2.LINE_AA)

        cv2.line(canvas, (ox + 310, legend_y), (ox + 350, legend_y), colors[key], thick + 1, cv2.LINE_AA)
        leg_txt = f"{labels_map[key]} (PR-AUC: {d['pr_auc']:.4f})"
        cv2.putText(canvas, leg_txt, (ox + 360, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        legend_y += 24

    cv2.imwrite(str(save_path), canvas)


def render_ph2_confusion_matrices(cm_dict: Dict[str, Dict[str, Any]], save_path: Path):
    """Render 4-panel confusion matrix grid."""
    canvas = np.ones((700, 950, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "PH2 External Validation: Confusion Matrices (N=200)",
                (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)

    panels = [
        ("Phase 1 ViT (T=0.800)", cm_dict["ViT_080"], 80, 90),
        ("Phase 1 Multimodal (T=0.360)", cm_dict["Multimodal_036"], 510, 90),
        ("M6 Point A: F1-Calibrated (T=0.690)", cm_dict["M6_0690"], 80, 390),
        ("M6 Point B: Matched Spec (T=0.625)", cm_dict["M6_0625"], 510, 390),
    ]

    for title, data, px, py in panels:
        cv2.putText(canvas, title, (px + 20, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 2, cv2.LINE_AA)
        
        # Grid box (2x2 cells, 120x100 each)
        bx, by = px + 60, py + 50
        cw, ch = 130, 85

        cm = data["cm"]
        # TP, FP, FN, TN
        cells = [
            (0, 0, cm[0, 0], f"TN: {cm[0, 0]}", (225, 240, 225)),
            (0, 1, cm[0, 1], f"FP: {cm[0, 1]}", (220, 220, 255)),
            (1, 0, cm[1, 0], f"FN: {cm[1, 0]}", (210, 210, 255)),
            (1, 1, cm[1, 1], f"TP: {cm[1, 1]}", (215, 235, 245)),
        ]

        for r, c, val, txt, bg in cells:
            cx0 = bx + c * cw
            cy0 = by + r * ch
            cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), bg, -1)
            cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), (180, 180, 180), 1)
            cv2.putText(canvas, txt, (cx0 + 25, cy0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)

        # Labels
        cv2.putText(canvas, "Pred Non-Mel", (bx + 15, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)
        cv2.putText(canvas, "Pred Mel", (bx + cw + 30, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)
        cv2.putText(canvas, "True Non", (bx - 55, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)
        cv2.putText(canvas, "True Mel", (bx - 55, by + ch + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)

        # Metrics subtitle
        sub = f"Sens: {data['sens']*100:.1f}% | Spec: {data['spec']*100:.1f}% | F1: {data['f1']:.3f}"
        cv2.putText(canvas, sub, (px + 30, py + 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)

    cv2.imwrite(str(save_path), canvas)


def render_segmentation_dice_histogram(dices: List[float], save_path: Path):
    """Render VM-UNet Dice score distribution on PH2 dataset."""
    canvas = np.ones((600, 850, 3), dtype=np.uint8) * 255
    ox, oy = 90, 480
    w, h = 680, 360

    cv2.putText(canvas, "PH2 Segmentation Fidelity: VM-UNet vs. Expert Ground Truth (N=200)",
                (ox + 20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)

    # 20 Bins from 0.0 to 1.0
    bins = [0] * 20
    for d in dices:
        b = min(19, max(0, int(d * 20)))
        bins[b] += 1
    max_count = max(bins) if max(bins) > 0 else 1

    # Draw grid
    for i in range(11):
        x = int(ox + i * (w / 10))
        cv2.line(canvas, (x, oy - h), (x, oy), (220, 224, 230), 1)
        cv2.putText(canvas, f"{i/10:.1f}", (x - 12, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)

    for c in [0, 20, 40, 60, 80, 100]:
        if c <= max_count + 10:
            y = int(oy - (c / (max_count + 10)) * h)
            cv2.line(canvas, (ox, y), (ox + w, y), (220, 224, 230), 1)
            cv2.putText(canvas, str(c), (ox - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)

    # Draw bars
    bw = int(w / 20)
    for i, count in enumerate(bins):
        bx0 = ox + i * bw + 2
        bh = int((count / (max_count + 10)) * h)
        by0 = oy - bh
        cv2.rectangle(canvas, (bx0, by0), (bx0 + bw - 4, oy), (200, 150, 40), -1)
        cv2.rectangle(canvas, (bx0, by0), (bx0 + bw - 4, oy), (150, 100, 20), 1)

    # Draw Mean and Median lines
    mean_d = float(np.mean(dices))
    med_d = float(np.median(dices))
    mx = int(ox + mean_d * w)
    medx = int(ox + med_d * w)

    cv2.line(canvas, (mx, oy - h), (mx, oy), (30, 30, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Mean Dice: {mean_d:.4f}", (mx - 60, oy - h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 220), 2, cv2.LINE_AA)

    cv2.line(canvas, (medx, oy - h), (medx, oy), (40, 140, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Median Dice: {med_d:.4f}", (medx + 10, oy - h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 140, 40), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Dice Similarity Coefficient", (ox + 230, oy + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Number of Images", (ox - 70, oy - 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 50, 50), 1, cv2.LINE_AA)

    cv2.imwrite(str(save_path), canvas)


# =========================================================================
# MAIN EVALUATION PIPELINE
# =========================================================================

def run_ph2_evaluation():
    print("=" * 80)
    print("CEFM PHASE 2 STEP 5 — INDEPENDENT EXTERNAL VALIDATION (PH²)")
    print("=" * 80)

    # 1. Audit Checkpoint Hashes
    checkpoints = {
        "Phase1_ViT": PROJECT_ROOT / "models" / "classification" / "vit_model" / "model.safetensors",
        "Phase1_UNet": PROJECT_ROOT / "models" / "segmentation" / "unet_isic2018.pt",
        "Phase1_Fusion": PROJECT_ROOT / "models" / "multimodal" / "fusion_model.pt",
        "M6_Challenger": PROJECT_ROOT / "models" / "experiments" / "M6" / "weights.pt",
    }
    hashes = {}
    for k, p in checkpoints.items():
        if p.exists():
            hashes[k] = sha256_file(p)
            print(f"[{k}] Checkpoint: {p.name} | SHA-256: {hashes[k]}")
        else:
            print(f"[{k}] MISSING: {p}")

    # 2. Load PH2 Metadata
    df_ph2 = load_ph2_metadata()
    print(f"\nLoaded {len(df_ph2)} cases from PH² dataset:")
    print(f" - Melanomas: {(df_ph2['label'] == 1).sum()} (20.0%)")
    print(f" - Non-Melanomas: {(df_ph2['label'] == 0).sum()} (80.0%)")
    print(f"   * Common Nevi: {(df_ph2['clinical_diagnosis_code'] == 0).sum()}")
    print(f"   * Atypical Nevi: {(df_ph2['clinical_diagnosis_code'] == 1).sum()}")

    # 3. Process every image through CEFM pipeline
    results = []
    print("\nRunning inference across all 200 PH² cases...")
    start_time = time.time()

    for i, row in df_ph2.iterrows():
        img_id = row["image_id"]
        img_path = PH2_IMAGES_DIR / img_id / f"{img_id}_Dermoscopic_Image" / f"{img_id}.bmp"
        gt_mask_path = PH2_IMAGES_DIR / img_id / f"{img_id}_lesion" / f"{img_id}_lesion.bmp"

        if not img_path.exists():
            print(f"Missing {img_path}")
            continue

        # Load GT mask
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask_bin = (gt_mask > 127).astype(np.uint8) if gt_mask is not None else None

        # 1. Production VM-UNet segmentation
        pred_mask = segment_lesion(img_path)

        # Compute segmentation Dice against GT
        if gt_mask_bin is not None and pred_mask is not None:
            inter = np.logical_and(pred_mask > 0, gt_mask_bin > 0).sum()
            dice = float((2.0 * inter) / ((pred_mask > 0).sum() + (gt_mask_bin > 0).sum() + 1e-8))
        else:
            dice = 0.0

        # 2. ABCDE Feature extraction from image and predicted mask
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        features = extract_abcde_features(img_rgb, pred_mask)

        # 3. Classify with all models
        clf_res = classify(img_path, features=features, mask=pred_mask)

        # Model outputs
        vit_prob = clf_res.get("vit_melanoma_probability", 0.0)
        multimodal_prob = clf_res.get("multimodal_melanoma_probability", 0.0)
        
        challenger = clf_res.get("challenger", {})
        m6_prob = challenger.get("challenger_melanoma_probability", 0.0)
        patch_count = challenger.get("challenger_local_patch_count", 1)

        # Also test M6 with GT mask to observe segmentation dependency
        clf_gt = classify(img_path, features=features, mask=gt_mask_bin)
        m6_prob_gt = clf_gt.get("challenger", {}).get("challenger_melanoma_probability", m6_prob)

        results.append({
            "image_id": img_id,
            "ground_truth_label": row["label"],
            "diagnosis_name": row["diagnosis_name"],
            "clinical_diagnosis_code": row["clinical_diagnosis_code"],
            "histo_diagnosis": row["histo_diagnosis"],
            "vit_probability": vit_prob,
            "multimodal_probability": multimodal_prob,
            "m6_probability": m6_prob,
            "m6_probability_gt_mask": m6_prob_gt,
            "unet_dice_vs_gt": dice,
            "local_patch_count": patch_count,
            "asymmetry": features.get("asymmetry", 0.0),
            "border_irregularity": features.get("border_irregularity", 0.0),
            "diameter_pixels": features.get("diameter_pixels", 0.0),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(df_ph2):
            elapsed = time.time() - start_time
            print(f"Processed {i + 1}/{len(df_ph2)} cases in {elapsed:.1f}s (Mean Dice: {np.mean([r['unet_dice_vs_gt'] for r in results]):.4f})")

    df_res = pd.DataFrame(results)
    
    # Save individual predictions
    predictions_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP5_PREDICTIONS.csv"
    df_res.to_csv(predictions_csv, index=False)
    print(f"\nSaved sample-level predictions to {predictions_csv}")

    # 4. Metrics Evaluation
    y_true = df_res["ground_truth_label"].values
    y_vit = df_res["vit_probability"].values
    y_multi = df_res["multimodal_probability"].values
    y_m6 = df_res["m6_probability"].values
    y_m6_gt = df_res["m6_probability_gt_mask"].values

    models_to_evaluate = [
        # (Model Name, Prob Array, Operating Point Name, Threshold)
        ("Phase 1 ViT Baseline", y_vit, "F1-Calibrated", 0.900),
        ("Phase 1 ViT Baseline", y_vit, "Clinical Screening", 0.800),
        ("Phase 1 Multimodal Fusion", y_multi, "F1-Calibrated", 0.800),
        ("Phase 1 Multimodal Fusion", y_multi, "Clinical Screening", 0.360),
        ("M6 Dual Gated (UNet Mask)", y_m6, "Point A: F1-Calibrated", 0.690),
        ("M6 Dual Gated (UNet Mask)", y_m6, "Point B: 90% Spec Target", 0.625),
        ("M6 Dual Gated (GT Mask)", y_m6_gt, "Point A: F1-Calibrated", 0.690),
        ("M6 Dual Gated (GT Mask)", y_m6_gt, "Point B: 90% Spec Target", 0.625),
    ]

    metrics_rows = []
    print("\n" + "=" * 98)
    print(f"{'Model':<28} | {'Operating Point':<24} | {'Thresh':<6} | {'AUC':<6} | {'PR-AUC':<6} | {'Sens':<7} | {'Spec':<7} | {'F1':<6} | {'BAcc':<6}")
    print("-" * 98)

    for name, probs, op_name, thresh in models_to_evaluate:
        m = compute_metrics(y_true, probs, thresh)
        ci = compute_bootstrap_ci(y_true, probs, thresh, n_boot=1000, seed=42)
        
        row = {
            "model": name,
            "operating_point": op_name,
            "threshold": thresh,
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "pr_auc_ci_lower": ci["pr_auc"]["ci_lower"],
            "pr_auc_ci_upper": ci["pr_auc"]["ci_upper"],
            "sensitivity": m["sensitivity"],
            "sensitivity_ci_lower": ci["sensitivity"]["ci_lower"],
            "sensitivity_ci_upper": ci["sensitivity"]["ci_upper"],
            "specificity": m["specificity"],
            "specificity_ci_lower": ci["specificity"]["ci_lower"],
            "specificity_ci_upper": ci["specificity"]["ci_upper"],
            "precision": m["precision"],
            "precision_ci_lower": ci["precision"]["ci_lower"],
            "precision_ci_upper": ci["precision"]["ci_upper"],
            "f1_score": m["f1"],
            "f1_ci_lower": ci["f1"]["ci_lower"],
            "f1_ci_upper": ci["f1"]["ci_upper"],
            "balanced_accuracy": m["balanced_acc"],
            "balanced_acc_ci_lower": ci["balanced_acc"]["ci_lower"],
            "balanced_acc_ci_upper": ci["balanced_acc"]["ci_upper"],
            "accuracy": m["accuracy"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
        }
        metrics_rows.append(row)

        print(f"{name:<28} | {op_name:<24} | {thresh:<6.3f} | {m['roc_auc']:<6.4f} | {m['pr_auc']:<6.4f} | {m['sensitivity']*100:<6.2f}% | {m['specificity']*100:<6.2f}% | {m['f1']:<6.4f} | {m['balanced_acc']:<6.4f}")

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP5_METRICS.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics summary to {metrics_csv}")

    # 5. Generate Visual Figures using OpenCV
    print("\nRendering external validation figures...")
    eval_chart_data = {
        "Phase1_ViT": {"targets": y_true, "probs": y_vit, "roc_auc": float(roc_auc_score(y_true, y_vit)), "pr_auc": float(auc(*precision_recall_curve(y_true, y_vit)[1::-1]))},
        "Phase1_Fusion": {"targets": y_true, "probs": y_multi, "roc_auc": float(roc_auc_score(y_true, y_multi)), "pr_auc": float(auc(*precision_recall_curve(y_true, y_multi)[1::-1]))},
        "M6_UNet": {"targets": y_true, "probs": y_m6, "roc_auc": float(roc_auc_score(y_true, y_m6)), "pr_auc": float(auc(*precision_recall_curve(y_true, y_m6)[1::-1]))},
        "M6_GT": {"targets": y_true, "probs": y_m6_gt, "roc_auc": float(roc_auc_score(y_true, y_m6_gt)), "pr_auc": float(auc(*precision_recall_curve(y_true, y_m6_gt)[1::-1]))},
    }

    roc_fig_path = FIGURES_DIR / "ph2_external_roc_curves.png"
    render_ph2_roc_curves(eval_chart_data, roc_fig_path)

    pr_fig_path = FIGURES_DIR / "ph2_external_pr_curves.png"
    render_ph2_pr_curves(eval_chart_data, pr_fig_path)

    # Confusion matrix data
    def get_cm_info(probs, th):
        preds = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        return {"cm": cm, "sens": sens, "spec": spec, "f1": f1}

    cm_chart_data = {
        "ViT_080": get_cm_info(y_vit, 0.800),
        "Multimodal_036": get_cm_info(y_multi, 0.360),
        "M6_0690": get_cm_info(y_m6, 0.690),
        "M6_0625": get_cm_info(y_m6, 0.625),
    }
    cm_fig_path = FIGURES_DIR / "ph2_external_confusion_matrices.png"
    render_ph2_confusion_matrices(cm_chart_data, cm_fig_path)

    dice_fig_path = FIGURES_DIR / "ph2_segmentation_dice_distribution.png"
    dices = df_res["unet_dice_vs_gt"].tolist()
    render_segmentation_dice_histogram(dices, dice_fig_path)

    print(f"Generated figures in {FIGURES_DIR}:")
    print(f" - {roc_fig_path.name}")
    print(f" - {pr_fig_path.name}")
    print(f" - {cm_fig_path.name}")
    print(f" - {dice_fig_path.name}")

    # Copy to brain artifact directory as well
    brain_fig_dir = BRAIN_ARTIFACT_DIR / "figures"
    brain_fig_dir.mkdir(parents=True, exist_ok=True)
    for f in [roc_fig_path, pr_fig_path, cm_fig_path, dice_fig_path]:
        shutil.copy(f, brain_fig_dir / f.name)
    print("Copied all figures to conversation artifact directory.")

    # 6. Save JSON Summary for Report
    json_summary = {
        "dataset": {
            "name": "PH² Dermoscopy Database",
            "provenance": "Hospital Pedro Hispano, Matosinhos, Portugal (Mendonça et al., 2013)",
            "total_samples": 200,
            "melanoma_samples": 40,
            "non_melanoma_samples": 160,
            "prevalence": 0.20,
            "mean_segmentation_dice": float(np.mean(dices)),
            "median_segmentation_dice": float(np.median(dices)),
        },
        "checkpoint_hashes": hashes,
        "metrics": metrics_rows,
    }
    with open(OUTPUT_DIR / "phase2_step5_external_validation.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    print("\nPH² external validation complete!")


if __name__ == "__main__":
    run_ph2_evaluation()
