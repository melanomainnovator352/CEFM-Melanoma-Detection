import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_CSV = BASE_DIR / "data" / "HAM10000" / "metadata" / "validation.csv"
MODEL_DIR = BASE_DIR / "models" / "classification" / "vit_model"
OUT_FILE = BASE_DIR / "models" / "classification" / "threshold_config.json"
EVAL_DIR = BASE_DIR / "models" / "classification" / "evaluation"

IMAGE_DIRS = [
    Path("D:/datasets/HAM10000/HAM10000_images_part_1"),
    Path("D:/datasets/HAM10000/HAM10000_images_part_2"),
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_1",
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_2",
    BASE_DIR / "data" / "HAM10000" / "images",
]


def find_image(image_id: str) -> Path:
    clean_id = str(image_id).replace(".jpg", "")
    for d in IMAGE_DIRS:
        if d.exists():
            p = d / f"{clean_id}.jpg"
            if p.exists():
                return p
    raise FileNotFoundError(f"Image not found: {clean_id}")


def main():
    val_df = pd.read_csv(VAL_CSV)
    val_df["path"] = val_df["image_id"].apply(find_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation sweep on {device}...")

    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR), local_files_only=True).to(device)
    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.eval()

    y_true, y_prob = [], []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            batch = val_df.iloc[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch["path"]]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            y_true.extend(batch["label"].tolist())
            y_prob.extend(probs[:, 1].tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    val_auc = float(roc_auc_score(y_true, y_prob))
    val_prauc = float(average_precision_score(y_true, y_prob))

    print(f"\n=======================================================")
    print(f"Validation Performance: ROC-AUC = {val_auc:.4f} | PR-AUC = {val_prauc:.4f}")
    print(f"Total Validation Samples: {len(y_true)} (Mel: {(y_true==1).sum()}, Non-Mel: {(y_true==0).sum()})")
    print(f"=======================================================\n")

    print(f"{'Thresh':<7} | {'Sens/Recall':<11} | {'Specificity':<11} | {'Precision':<9} | {'F1-Score':<8} | {'Bal Acc':<7} | {'FNR':<6} | {'TP':<4} | {'FP':<4} | {'TN':<5} | {'FN':<3}")
    print("-" * 92)

    records = []
    best_f1, best_f1_t = -1.0, 0.50
    best_bacc, best_bacc_t = -1.0, 0.50
    clinical_t = 0.50
    best_clin_spec = -1.0

    thresholds = [round(float(t), 2) for t in np.arange(0.10, 0.95, 0.05)]

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        bacc = 0.5 * (sens + spec)
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"{t:<7.2f} | {sens:<11.4f} | {spec:<11.4f} | {prec:<9.4f} | {f1:<8.4f} | {bacc:<7.4f} | {fnr:<6.4f} | {tp:<4d} | {fp:<4d} | {tn:<5d} | {fn:<3d}")

        rec = {
            "threshold": t,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "balanced_accuracy": round(bacc, 4),
            "false_negative_rate": round(fnr, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        records.append(rec)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = t

        if bacc > best_bacc:
            best_bacc = bacc
            best_bacc_t = t

        if sens >= 0.80 and spec > best_clin_spec:
            best_clin_spec = spec
            clinical_t = t

    print("-" * 92)
    print(f"Optimal F1 Threshold (Validation):         {best_f1_t:.2f} (F1: {best_f1:.4f})")
    print(f"Optimal Balanced Acc Threshold (Validation):{best_bacc_t:.2f} (BAcc: {best_bacc:.4f})")
    print(f"Clinical Threshold (Sens >= 80%):            {clinical_t:.2f} (Spec: {best_clin_spec:.4f})")

    calib = {
        "best_f1_threshold": best_f1_t,
        "best_f1_score": round(best_f1, 4),
        "best_bacc_threshold": best_bacc_t,
        "best_bacc_score": round(best_bacc, 4),
        "clinical_threshold": clinical_t,
        "validation_roc_auc": round(val_auc, 4),
        "validation_pr_auc": round(val_prauc, 4),
        "sweep_records": records,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    with open(EVAL_DIR / "threshold_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    print(f"\nSaved calibration configuration to:\n  {OUT_FILE}\n  {EVAL_DIR / 'threshold_calibration.json'}")


if __name__ == "__main__":
    main()

