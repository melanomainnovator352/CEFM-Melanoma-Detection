from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_TEST_CSVS = [
    BASE_DIR / "data" / "HAM10000" / "metadata" / "test.csv",
    Path("D:/datasets/HAM10000/processed/test.csv"),
    BASE_DIR / "datasets" / "Ham10000" / "processed" / "test.csv",
]
TEST_CSV = next((p for p in CANDIDATE_TEST_CSVS if p.exists()), CANDIDATE_TEST_CSVS[0])
MODEL_DIR = BASE_DIR / "models" / "classification" / "vit_model"
THRESH_CONFIG = BASE_DIR / "models" / "classification" / "threshold_config.json"

IMAGE_DIRS = [
    Path("D:/datasets/HAM10000/HAM10000_images_part_1"),
    Path("D:/datasets/HAM10000/HAM10000_images_part_2"),
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_1",
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_2",
    BASE_DIR / "data" / "HAM10000" / "images",
]

OUTPUT_DIR = BASE_DIR / "models" / "classification" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image(image_id: str) -> Path:
    clean_id = str(image_id).replace(".jpg", "")
    for directory in IMAGE_DIRS:
        if not directory.exists():
            continue
        path = directory / f"{clean_id}.jpg"
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found: {clean_id}.jpg")


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "decision_threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(prec),
        "sensitivity_recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    model_path = Path(args.model_dir)

    print("=" * 70)
    print("CEFM ViT — Full HAM10000 Independent Test Evaluation")
    print("=" * 70)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")
    if not model_path.exists():
        raise FileNotFoundError(f"ViT model not found: {model_path}")

    # Load calibrated threshold if available
    calib_threshold = 0.80
    thresh_source = "default 0.80"
    if args.threshold is not None:
        calib_threshold = args.threshold
        thresh_source = f"CLI override ({args.threshold:.2f})"
    elif THRESH_CONFIG.exists():
        try:
            with open(THRESH_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                calib_threshold = cfg.get("clinical_threshold", cfg.get("best_bacc_threshold", 0.80))
                thresh_source = f"Validation-calibrated (Clinical: {calib_threshold:.2f})"
        except Exception:
            pass

    print(f"Decision threshold to apply: {calib_threshold:.2f} [{thresh_source}]")

    df = pd.read_csv(TEST_CSV)
    print(f"\nTest set: {len(df)} samples")
    print(f"Non-melanoma (0): {(df['label'] == 0).sum()}")
    print(f"Melanoma (1):     {(df['label'] == 1).sum()}")

    processor = ViTImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = ViTForImageClassification.from_pretrained(str(model_path), local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_prob = []
    batch_size = 32

    print("\nRunning inference on test dataset...")
    with torch.no_grad():
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_rows = df.iloc[start_idx:end_idx]
            images = []
            labels = []
            for _, row in batch_rows.iterrows():
                image_path = find_image(str(row["image_id"]))
                images.append(Image.open(image_path).convert("RGB"))
                labels.append(int(row["label"]))

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            for j, lab in enumerate(labels):
                mel_prob = float(probs[j, 1].item())
                y_true.append(lab)
                y_prob.append(mel_prob)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Evaluate at calibrated threshold
    calib_metrics = compute_metrics_at_threshold(y_true, y_prob, calib_threshold)
    # Evaluate at baseline 0.50 threshold
    base_metrics = compute_metrics_at_threshold(y_true, y_prob, 0.50)

    # Compute ROC and PR curve data
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec_pts, rec_pts, pr_thresh = precision_recall_curve(y_true, y_prob)

    curves_data = {
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[:: max(1, len(fpr) // 50)]],
            "tpr": [round(float(x), 4) for x in tpr[:: max(1, len(tpr) // 50)]],
            "roc_auc": round(roc_auc, 4),
        },
        "pr_curve": {
            "precision": [round(float(x), 4) for x in prec_pts[:: max(1, len(prec_pts) // 50)]],
            "recall": [round(float(x), 4) for x in rec_pts[:: max(1, len(rec_pts) // 50)]],
            "pr_auc": round(pr_auc, 4),
        },
    }

    curves_path = OUTPUT_DIR / "vit_curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(curves_data, f, indent=2)

    results = {
        "model": "ViT-Melanoma",
        "version": "v3.0-finetuned",
        "backbone": "google/vit-base-patch16-224",
        "dataset": "HAM10000",
        "test_samples": int(len(y_true)),
        "non_melanoma_samples": int((y_true == 0).sum()),
        "melanoma_samples": int((y_true == 1).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "calibrated_evaluation": calib_metrics,
        "baseline_0_50_evaluation": base_metrics,
        # Standard flattened fields mapped to calibrated metrics for backward compatibility:
        "decision_threshold": calib_metrics["decision_threshold"],
        "accuracy": calib_metrics["accuracy"],
        "balanced_accuracy": calib_metrics["balanced_accuracy"],
        "precision": calib_metrics["precision"],
        "sensitivity_recall": calib_metrics["sensitivity_recall"],
        "specificity": calib_metrics["specificity"],
        "f1": calib_metrics["f1"],
        "true_negative": calib_metrics["true_negative"],
        "false_positive": calib_metrics["false_positive"],
        "false_negative": calib_metrics["false_negative"],
        "true_positive": calib_metrics["true_positive"],
    }

    results_path = OUTPUT_DIR / "vit_test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    prediction_df = df.copy()
    y_pred = (y_prob >= calib_threshold).astype(int)
    prediction_df["predicted_label"] = y_pred
    prediction_df["melanoma_probability"] = [round(float(p), 4) for p in y_prob]
    prediction_df["non_melanoma_probability"] = [round(float(1.0 - p), 4) for p in y_prob]

    predictions_path = OUTPUT_DIR / "vit_test_predictions.csv"
    prediction_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print("-" * 70)
    print(f"Performance at Calibrated Threshold ({calib_threshold:.2f}):")
    print(f"  Accuracy:           {calib_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {calib_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:          {calib_metrics['precision']:.4f}")
    print(f"  Sensitivity/Recall: {calib_metrics['sensitivity_recall']:.4f}")
    print(f"  Specificity:        {calib_metrics['specificity']:.4f}")
    print(f"  F1 Score:           {calib_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:   TN={calib_metrics['true_negative']} | FP={calib_metrics['false_positive']} | FN={calib_metrics['false_negative']} | TP={calib_metrics['true_positive']}")
    print("-" * 70)
    print(f"Performance at Default 0.50 Threshold:")
    print(f"  Accuracy:           {base_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {base_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:          {base_metrics['precision']:.4f}")
    print(f"  Sensitivity/Recall: {base_metrics['sensitivity_recall']:.4f}")
    print(f"  Specificity:        {base_metrics['specificity']:.4f}")
    print(f"  F1 Score:           {base_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:   TN={base_metrics['true_negative']} | FP={base_metrics['false_positive']} | FN={base_metrics['false_negative']} | TP={base_metrics['true_positive']}")
    print("=" * 70)
    print(f"Saved results: {results_path}")
    print(f"Saved curves:  {curves_path}")
    print(f"Saved CSV:     {predictions_path}")


if __name__ == "__main__":
    main()
