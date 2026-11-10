"""
Comparative Evaluation: ViT-Only Baseline vs. Multimodal Fusion Model.
Evaluates both models on the identical independent test split (1,504 samples).
Computes Sensitivity, Specificity, Precision, F1, ROC-AUC, and PR-AUC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from transformers import ViTForImageClassification, ViTImageProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.fusion import CEFMMultimodalFusion
from scripts.train_multimodal_fusion import extract_dataset_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_MODEL_DIR = REPO_ROOT / "models" / "classification" / "vit_model"
FUSION_MODEL_PATH = REPO_ROOT / "models" / "multimodal" / "fusion_model.pt"
FUSION_CONFIG_PATH = REPO_ROOT / "models" / "multimodal" / "fusion_config.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "models" / "multimodal" / "multimodal_evaluation_results.json"


def evaluate_models():
    print("=" * 80)
    print("COMPARATIVE EVALUATION: ViT-Only Baseline vs. Multimodal Fusion Model")
    print("=" * 80)

    # 1. Load ViT model
    processor = ViTImageProcessor.from_pretrained(str(VIT_MODEL_DIR), local_files_only=True)
    vit_model = ViTForImageClassification.from_pretrained(
        str(VIT_MODEL_DIR), local_files_only=True, attn_implementation="eager"
    ).to(DEVICE)
    vit_model.eval()

    # 2. Load test split
    test_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "test.csv")
    test_v, test_c, test_y = extract_dataset_features(test_df, vit_model, processor, "test")

    # 3. Load Multimodal Fusion model
    fusion_model = CEFMMultimodalFusion(
        visual_dim=768,
        clinical_dim=16,
        hidden_dim=128,
        num_classes=2,
    ).to(DEVICE)
    fusion_weights = torch.load(FUSION_MODEL_PATH, map_location=DEVICE)
    fusion_model.load_state_dict(fusion_weights)
    fusion_model.eval()

    # 4. Compute ViT baseline predictions from visual features (using ViT classifier head)
    # The classification head of vit_model is applied directly to the [CLS] embedding
    with torch.no_grad():
        vit_logits = vit_model.classifier(test_v.to(DEVICE))
        vit_probs = torch.softmax(vit_logits, dim=-1)[:, 1].cpu().numpy()

    # 5. Compute Multimodal Fusion predictions
    with torch.no_grad():
        fusion_logits = fusion_model(test_v.to(DEVICE), test_c.to(DEVICE))
        fusion_probs = torch.softmax(fusion_logits, dim=-1)[:, 1].cpu().numpy()

    targets = test_y.numpy()

    # Threshold configs
    vit_thresh = 0.80
    fusion_thresh = 0.50
    if FUSION_CONFIG_PATH.exists():
        with open(FUSION_CONFIG_PATH) as f:
            cfg = json.load(f)
            fusion_thresh = cfg.get("clinical_threshold", 0.50)

    def compute_all_metrics(probs: np.ndarray, threshold: float):
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        sens = recall_score(targets, preds)
        spec = tn / (tn + fp)
        prec = precision_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        bacc = balanced_accuracy_score(targets, preds)
        acc = accuracy_score(targets, preds)
        roc_auc = roc_auc_score(targets, probs)
        pr_auc = average_precision_score(targets, probs)
        return {
            "threshold": float(threshold),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "balanced_accuracy": round(float(bacc), 4),
            "accuracy": round(float(acc), 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    vit_metrics_default = compute_all_metrics(vit_probs, 0.50)
    vit_metrics_calibrated = compute_all_metrics(vit_probs, vit_thresh)

    fusion_metrics_default = compute_all_metrics(fusion_probs, 0.50)
    fusion_metrics_calibrated = compute_all_metrics(fusion_probs, fusion_thresh)

    report = {
        "dataset": "HAM10000 Independent Test Split",
        "sample_count": len(targets),
        "class_distribution": {
            "non_melanoma": int((targets == 0).sum()),
            "melanoma": int((targets == 1).sum()),
        },
        "models": {
            "vit_baseline_default": vit_metrics_default,
            "vit_baseline_calibrated": vit_metrics_calibrated,
            "multimodal_fusion_default": fusion_metrics_default,
            "multimodal_fusion_calibrated": fusion_metrics_calibrated,
        },
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "-" * 80)
    print(f"{'Metric':<25} | {'ViT Baseline (T=0.80)':<25} | {'Multimodal Fusion (T=' + str(fusion_thresh) + ')':<25}")
    print("-" * 80)
    print(f"{'ROC-AUC':<25} | {vit_metrics_calibrated['roc_auc']:<25.4f} | {fusion_metrics_calibrated['roc_auc']:<25.4f}")
    print(f"{'PR-AUC':<25} | {vit_metrics_calibrated['pr_auc']:<25.4f} | {fusion_metrics_calibrated['pr_auc']:<25.4f}")
    print(f"{'Sensitivity (Recall)':<25} | {vit_metrics_calibrated['sensitivity']:<25.2%} | {fusion_metrics_calibrated['sensitivity']:<25.2%}")
    print(f"{'Specificity':<25} | {vit_metrics_calibrated['specificity']:<25.2%} | {fusion_metrics_calibrated['specificity']:<25.2%}")
    print(f"{'Precision':<25} | {vit_metrics_calibrated['precision']:<25.2%} | {fusion_metrics_calibrated['precision']:<25.2%}")
    print(f"{'F1-Score':<25} | {vit_metrics_calibrated['f1_score']:<25.2%} | {fusion_metrics_calibrated['f1_score']:<25.2%}")
    print(f"{'Balanced Accuracy':<25} | {vit_metrics_calibrated['balanced_accuracy']:<25.2%} | {fusion_metrics_calibrated['balanced_accuracy']:<25.2%}")
    print(f"{'Overall Accuracy':<25} | {vit_metrics_calibrated['accuracy']:<25.2%} | {fusion_metrics_calibrated['accuracy']:<25.2%}")
    cm_vit = vit_metrics_calibrated["confusion_matrix"]
    cm_fus = fusion_metrics_calibrated["confusion_matrix"]
    cm_vit_str = f"TN={cm_vit['tn']}, FP={cm_vit['fp']}, FN={cm_vit['fn']}, TP={cm_vit['tp']}"
    cm_fus_str = f"TN={cm_fus['tn']}, FP={cm_fus['fp']}, FN={cm_fus['fn']}, TP={cm_fus['tp']}"
    print(f"{'Confusion Matrix':<25} | {cm_vit_str:<25} | {cm_fus_str:<25}")
    print("-" * 80)
    print(f"Results saved to {OUTPUT_REPORT_PATH}")

    return report


if __name__ == "__main__":
    evaluate_models()
