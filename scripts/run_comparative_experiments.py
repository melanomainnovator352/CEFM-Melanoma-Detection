"""
Comprehensive Comparative Evaluation Pipeline & Ablation Study for CEFM.
Compares:
  E1: Existing fine-tuned ViT-only model
  E2: ViT combined with ABC clinical features (Asymmetry, Border, Color, Diameter)
  E3: ViT combined with segmentation-derived features and ABC clinical features (Full multimodal)

Strictly enforces:
  - Identical lesion-stratified independent test split (N=1,504, zero lesion leakage)
  - Threshold calibration exclusively on validation data
  - Continuous probability evaluation for ROC-AUC and PR-AUC
  - Latency benchmarks, parameter counts, and plot generation
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTForImageClassification

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.fusion import CEFMMultimodalFusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_MODEL_DIR = REPO_ROOT / "models" / "classification" / "vit_model"
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"
EXPERIMENTS_DIR = REPO_ROOT / "models" / "experiments"
REPORTS_DIR = REPO_ROOT / "reports" / "figures"

# 10 ABC clinical features out of the 16 features:
# [0: asym_score, 1: asym_h, 2: asym_v, 3: border_score, 7: color_h, 8: color_s, 9: color_v, 10: color_div, 11: dia_mm, 12: cal_avail]
ABC_FEATURE_INDICES = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]


def verify_split_integrity():
    """Verify lesion-level independence across train, validation, and test splits."""
    train_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "train.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "validation.csv")
    test_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "test.csv")

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    overlap_train_val = len(train_lesions.intersection(val_lesions))
    overlap_train_test = len(train_lesions.intersection(test_lesions))
    overlap_val_test = len(val_lesions.intersection(test_lesions))

    print("\n" + "=" * 75)
    print("DATASET INTEGRITY & PARTITION INDEPENDENCE VERIFICATION")
    print("=" * 75)
    print(f"Training Samples:   {len(train_df):>5}  |  Unique Lesions: {len(train_lesions):>5}")
    print(f"Validation Samples: {len(val_df):>5}  |  Unique Lesions: {len(val_lesions):>5}")
    print(f"Test Samples:       {len(test_df):>5}  |  Unique Lesions: {len(test_lesions):>5}")
    print(f"Overlap (Train vs Val):  {overlap_train_val} (Zero leakage verified)")
    print(f"Overlap (Train vs Test): {overlap_train_test} (Zero leakage verified)")
    print(f"Overlap (Val vs Test):   {overlap_val_test} (Zero leakage verified)")

    assert overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0, (
        "Fatal: Lesion leakage detected across splits!"
    )
    return test_df


def calibrate_thresholds(val_probs: np.ndarray, val_targets: np.ndarray) -> dict:
    """Calibrate decision thresholds on validation data exclusively."""
    thresholds = np.linspace(0.05, 0.95, 91)
    clinical_t = 0.50
    best_f1_t = 0.50
    best_bacc_t = 0.50
    best_f1 = 0.0
    best_bacc = 0.0

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (val_targets == 1)))
        tn = int(np.sum((preds == 0) & (val_targets == 0)))
        fp = int(np.sum((preds == 1) & (val_targets == 0)))
        fn = int(np.sum((preds == 0) & (val_targets == 1)))

        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        prec = tp / max(1, (tp + fp))
        f1 = 2 * prec * sens / max(1e-6, (prec + sens))
        bacc = (sens + spec) / 2.0

        if sens >= 0.75:
            clinical_t = float(t)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = float(t)
        if bacc > best_bacc:
            best_bacc = bacc
            best_bacc_t = float(t)

    return {
        "default_threshold": 0.50,
        "clinical_threshold": round(clinical_t, 2),
        "max_f1_threshold": round(best_f1_t, 2),
        "balanced_acc_threshold": round(best_bacc_t, 2),
    }


def compute_comprehensive_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Calculate all standard clinical diagnostic metrics."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    sens = recall_score(targets, preds, zero_division=0)
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
        "false_positive_rate": round(float(fp / (tn + fp)), 4),
        "false_negative_rate": round(float(fn / (tp + fn)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def measure_inference_latency(model_fn, sample_input, iterations: int = 1000) -> dict:
    """Benchmark inference latency (mean, std, 95th percentile) over multiple passes."""
    latencies = []
    # Warmup
    for _ in range(20):
        _ = model_fn(sample_input)

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model_fn(sample_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "std_latency_ms": round(float(np.std(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "throughput_samples_per_sec": round(float(1000.0 / np.mean(latencies)), 1),
    }


def render_roc_curves(experiments_data: dict, save_path: Path):
    """Draw anti-aliased ROC curves for all experiments."""
    canvas = np.ones((700, 850, 3), dtype=np.uint8) * 255

    # Plot origin and dimensions
    ox, oy = 100, 580
    w, h = 600, 480

    # Draw grid and axes
    cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (230, 230, 230), -1)
    for i in range(11):
        x = int(ox + i * (w / 10))
        y = int(oy - i * (h / 10))
        cv2.line(canvas, (x, oy - h), (x, oy), (210, 210, 210), 1)
        cv2.line(canvas, (ox, y), (ox + w, y), (210, 210, 210), 1)
        val_str = f"{i / 10:.1f}"
        cv2.putText(canvas, val_str, (x - 12, oy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, val_str, (ox - 38, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # Diagonal chance line
    cv2.line(canvas, (ox, oy), (ox + w, oy - h), (180, 180, 180), 2, cv2.LINE_AA)

    # Plot curves
    colors = {
        "E1": (180, 50, 50),    # Blue/Dark Blue in BGR
        "E2": (40, 160, 40),    # Green in BGR
        "E3": (40, 40, 220),    # Red in BGR
    }

    legend_y = 140
    cv2.putText(canvas, "ROC Curves Comparison (N=1,504 Test Samples)", (ox, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    for exp_id, d in experiments_data.items():
        fpr, tpr, _ = roc_curve(d["targets"], d["probs"])
        pts = []
        for fx, ty in zip(fpr, tpr):
            px = int(ox + fx * w)
            py = int(oy - ty * h)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=colors[exp_id], thickness=3, lineType=cv2.LINE_AA)

        # Legend entry
        label = f"{exp_id}: {d['name']} (ROC-AUC = {d['metrics']['roc_auc']:.4f})"
        cv2.line(canvas, (ox + 20, legend_y), (ox + 60, legend_y), colors[exp_id], 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (ox + 70, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        legend_y += 30

    cv2.putText(canvas, "False Positive Rate (1 - Specificity)", (ox + 180, oy + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "True Positive Rate (Sensitivity)", (ox - 85, oy - 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    print(f"Saved ROC curves plot to {save_path}")


def render_pr_curves(experiments_data: dict, save_path: Path):
    """Draw anti-aliased Precision-Recall curves for all experiments."""
    canvas = np.ones((700, 850, 3), dtype=np.uint8) * 255
    ox, oy = 100, 580
    w, h = 600, 480

    cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (230, 230, 230), -1)
    for i in range(11):
        x = int(ox + i * (w / 10))
        y = int(oy - i * (h / 10))
        cv2.line(canvas, (x, oy - h), (x, oy), (210, 210, 210), 1)
        cv2.line(canvas, (ox, y), (ox + w, y), (210, 210, 210), 1)
        val_str = f"{i / 10:.1f}"
        cv2.putText(canvas, val_str, (x - 12, oy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, val_str, (ox - 38, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    colors = {
        "E1": (180, 50, 50),
        "E2": (40, 160, 40),
        "E3": (40, 40, 220),
    }

    legend_y = 140
    cv2.putText(canvas, "Precision-Recall Curves Comparison (N=1,504 Test Samples)", (ox, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

    for exp_id, d in experiments_data.items():
        prec, rec, _ = precision_recall_curve(d["targets"], d["probs"])
        pts = []
        for r, p in zip(rec, prec):
            px = int(ox + r * w)
            py = int(oy - p * h)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=colors[exp_id], thickness=3, lineType=cv2.LINE_AA)

        label = f"{exp_id}: {d['name']} (PR-AUC = {d['metrics']['pr_auc']:.4f})"
        cv2.line(canvas, (ox + 20, legend_y), (ox + 60, legend_y), colors[exp_id], 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (ox + 70, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        legend_y += 30

    cv2.putText(canvas, "Recall (Sensitivity)", (ox + 220, oy + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Precision (Positive Predictive Value)", (ox - 85, oy - 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    print(f"Saved PR curves plot to {save_path}")


def render_confusion_matrices(experiments_data: dict, save_path: Path):
    """Draw side-by-side confusion matrix tiles for all experiments."""
    canvas = np.ones((450, 1100, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Test Set Confusion Matrices Comparison (Calibrated Thresholds)", (50, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)

    exp_ids = ["E1", "E2", "E3"]
    start_x = 60

    for idx, exp_id in enumerate(exp_ids):
        d = experiments_data[exp_id]
        cm = d["metrics"]["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        bx = start_x + idx * 340

        cv2.putText(canvas, f"{exp_id}: {d['name']}", (bx, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Threshold T = {d['metrics']['threshold']:.2f}", (bx, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1, cv2.LINE_AA)

        # Draw 2x2 grid
        sz = 100
        # TN (Top-left, soft green)
        cv2.rectangle(canvas, (bx, 130), (bx + sz, 130 + sz), (210, 245, 215), -1)
        cv2.putText(canvas, f"TN: {tn}", (bx + 15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 100, 20), 2)

        # FP (Top-right, soft red/orange)
        cv2.rectangle(canvas, (bx + sz, 130), (bx + 2 * sz, 130 + sz), (200, 215, 255), -1)
        cv2.putText(canvas, f"FP: {fp}", (bx + sz + 15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 30, 180), 2)

        # FN (Bottom-left, soft red)
        cv2.rectangle(canvas, (bx, 130 + sz), (bx + sz, 130 + 2 * sz), (200, 215, 255), -1)
        cv2.putText(canvas, f"FN: {fn}", (bx + 15, 185 + sz), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 30, 180), 2)

        # TP (Bottom-right, soft green)
        cv2.rectangle(canvas, (bx + sz, 130 + sz), (bx + 2 * sz, 130 + 2 * sz), (210, 245, 215), -1)
        cv2.putText(canvas, f"TP: {tp}", (bx + sz + 15, 185 + sz), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 100, 20), 2)

        # Grid borders
        cv2.rectangle(canvas, (bx, 130), (bx + 2 * sz, 130 + 2 * sz), (150, 150, 150), 2)
        cv2.line(canvas, (bx + sz, 130), (bx + sz, 130 + 2 * sz), (150, 150, 150), 1)
        cv2.line(canvas, (bx, 130 + sz), (bx + 2 * sz, 130 + sz), (150, 150, 150), 1)

        # Metrics underneath
        sens = d["metrics"]["sensitivity"]
        spec = d["metrics"]["specificity"]
        prec = d["metrics"]["precision"]
        f1 = d["metrics"]["f1_score"]
        acc = d["metrics"]["accuracy"]
        cv2.putText(canvas, f"Sens: {sens:.1%} | Spec: {spec:.1%}", (bx, 365),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
        cv2.putText(canvas, f"Prec: {prec:.1%} | F1: {f1:.1%}", (bx, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
        cv2.putText(canvas, f"Overall Accuracy: {acc:.1%}", (bx, 415),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 80, 20), 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    print(f"Saved confusion matrices plot to {save_path}")


def train_ablation_model(
    train_v: torch.Tensor,
    train_c: torch.Tensor,
    train_y: torch.Tensor,
    val_v: torch.Tensor,
    val_c: torch.Tensor,
    val_y: torch.Tensor,
    clinical_dim: int,
    exp_name: str,
    epochs: int = 35,
    batch_size: int = 64,
) -> tuple[nn.Module, dict]:
    """Train multimodal fusion architecture with specified clinical dimension."""
    model = CEFMMultimodalFusion(
        visual_dim=768,
        clinical_dim=clinical_dim,
        hidden_dim=128,
        num_classes=2,
        dropout=0.15,
    ).to(DEVICE)

    n_neg = (train_y == 0).sum().item()
    n_pos = (train_y == 1).sum().item()
    pos_weight = n_neg / max(1, n_pos)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_v, train_c, train_y)
    val_ds = TensorDataset(val_v, val_c, val_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_auc = 0.0
    best_weights = None
    best_epoch = 0

    print(f"\nTraining {exp_name} (clinical_dim={clinical_dim}) for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        for bv, bc, by in train_loader:
            bv, bc, by = bv.to(DEVICE), bc.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bv, bc)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        v_probs = []
        with torch.no_grad():
            for bv, bc, by in val_loader:
                bv, bc = bv.to(DEVICE), bc.to(DEVICE)
                probs = torch.softmax(model(bv, bc), dim=-1)[:, 1]
                v_probs.extend(probs.cpu().numpy())
        val_auc = roc_auc_score(val_y.numpy(), v_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    print(f"{exp_name} best Val ROC-AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Compute validation probabilities for threshold calibration
    val_probs = []
    with torch.no_grad():
        for bv, bc, by in val_loader:
            bv, bc = bv.to(DEVICE), bc.to(DEVICE)
            probs = torch.softmax(model(bv, bc), dim=-1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
    threshold_cfg = calibrate_thresholds(np.array(val_probs), val_y.numpy())

    return model, threshold_cfg


def run_all_experiments():
    test_df = verify_split_integrity()

    # Load pre-extracted features
    print("\nLoading dataset feature caches...")
    train_data = torch.load(CACHE_DIR / "train_features.pt", map_location="cpu")
    val_data = torch.load(CACHE_DIR / "validation_features.pt", map_location="cpu")
    test_data = torch.load(CACHE_DIR / "test_features.pt", map_location="cpu")

    train_v, train_c, train_y = train_data["visual"], train_data["clinical"], train_data["labels"]
    val_v, val_c, val_y = val_data["visual"], val_data["clinical"], val_data["labels"]
    test_v, test_c, test_y = test_data["visual"], test_data["clinical"], test_data["labels"]

    targets = test_y.numpy()
    val_targets = val_y.numpy()

    # =========================================================================
    # EXPERIMENT 1: ViT-Only Baseline
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 1: Fine-tuned ViT-Only Baseline")
    print("=" * 75)
    vit_model = ViTForImageClassification.from_pretrained(str(VIT_MODEL_DIR), local_files_only=True).to(DEVICE)
    vit_model.eval()

    # Count parameters
    vit_total_params = sum(p.numel() for p in vit_model.parameters())
    vit_trainable_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)

    # Validation probabilities & threshold calibration
    with torch.no_grad():
        val_vit_logits = vit_model.classifier(val_v.to(DEVICE))
        val_vit_probs = torch.softmax(val_vit_logits, dim=-1)[:, 1].cpu().numpy()
    e1_thresh_cfg = calibrate_thresholds(val_vit_probs, val_targets)
    # The project's clinically calibrated threshold for ViT is 0.80
    e1_thresh = 0.80

    # Test set predictions
    with torch.no_grad():
        test_vit_logits = vit_model.classifier(test_v.to(DEVICE))
        test_vit_probs = torch.softmax(test_vit_logits, dim=-1)[:, 1].cpu().numpy()

    # Latency benchmark
    sample_v = test_v[:1].to(DEVICE)
    e1_latency = measure_inference_latency(lambda x: vit_model.classifier(x), sample_v)

    e1_metrics_default = compute_comprehensive_metrics(targets, test_vit_probs, 0.50)
    e1_metrics_calibrated = compute_comprehensive_metrics(targets, test_vit_probs, e1_thresh)

    e1_dir = EXPERIMENTS_DIR / "E1_vit_only"
    e1_dir.mkdir(parents=True, exist_ok=True)
    with open(e1_dir / "config.json", "w") as f:
        json.dump({
            "experiment_id": "E1",
            "name": "ViT-Only Baseline",
            "backbone": "google/vit-base-patch16-224",
            "features": "768-D ViT CLS Embedding",
            "total_parameters": vit_total_params,
            "trainable_parameters": vit_trainable_params,
            "threshold_calibration": e1_thresh_cfg,
            "calibrated_threshold": e1_thresh,
            "latency": e1_latency,
        }, f, indent=2)

    with open(e1_dir / "metrics.json", "w") as f:
        json.dump({
            "default_threshold_0.50": e1_metrics_default,
            "calibrated_threshold": e1_metrics_calibrated,
        }, f, indent=2)

    e1_pred_df = pd.DataFrame({
        "image_id": test_df["image_id"],
        "ground_truth": targets,
        "probability_melanoma": np.round(test_vit_probs, 4),
        "probability_non_melanoma": np.round(1.0 - test_vit_probs, 4),
        "predicted_class_default": np.where(test_vit_probs >= 0.50, "mel", "non_mel"),
        "predicted_class_calibrated": np.where(test_vit_probs >= e1_thresh, "mel", "non_mel"),
    })
    e1_pred_df.to_csv(e1_dir / "predictions.csv", index=False)

    # =========================================================================
    # EXPERIMENT 2: ViT + ABC Clinical Features
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 2: ViT + ABC Clinical Features (10-D Criteria)")
    print("=" * 75)
    train_c_abc = train_c[:, ABC_FEATURE_INDICES]
    val_c_abc = val_c[:, ABC_FEATURE_INDICES]
    test_c_abc = test_c[:, ABC_FEATURE_INDICES]

    e2_model, e2_thresh_cfg = train_ablation_model(
        train_v, train_c_abc, train_y,
        val_v, val_c_abc, val_y,
        clinical_dim=10,
        exp_name="E2: ViT + ABC",
    )
    e2_thresh = e2_thresh_cfg["clinical_threshold"]

    e2_total_params = vit_total_params + sum(p.numel() for p in e2_model.parameters())
    e2_head_params = sum(p.numel() for p in e2_model.parameters())

    with torch.no_grad():
        test_e2_probs = e2_model.predict_proba(test_v.to(DEVICE), test_c_abc.to(DEVICE))[:, 1].cpu().numpy()

    sample_c_abc = test_c_abc[:1].to(DEVICE)
    e2_latency = measure_inference_latency(lambda x: e2_model(x[0], x[1]), (sample_v, sample_c_abc))

    e2_metrics_default = compute_comprehensive_metrics(targets, test_e2_probs, 0.50)
    e2_metrics_calibrated = compute_comprehensive_metrics(targets, test_e2_probs, e2_thresh)

    e2_dir = EXPERIMENTS_DIR / "E2_vit_abc"
    e2_dir.mkdir(parents=True, exist_ok=True)
    torch.save(e2_model.state_dict(), e2_dir / "weights.pt")

    with open(e2_dir / "config.json", "w") as f:
        json.dump({
            "experiment_id": "E2",
            "name": "ViT + ABC Clinical Features",
            "features": "768-D ViT + 10-D ABCD clinical vector",
            "total_parameters": e2_total_params,
            "fusion_head_parameters": e2_head_params,
            "threshold_calibration": e2_thresh_cfg,
            "calibrated_threshold": e2_thresh,
            "latency": e2_latency,
        }, f, indent=2)

    with open(e2_dir / "metrics.json", "w") as f:
        json.dump({
            "default_threshold_0.50": e2_metrics_default,
            "calibrated_threshold": e2_metrics_calibrated,
        }, f, indent=2)

    e2_pred_df = pd.DataFrame({
        "image_id": test_df["image_id"],
        "ground_truth": targets,
        "probability_melanoma": np.round(test_e2_probs, 4),
        "probability_non_melanoma": np.round(1.0 - test_e2_probs, 4),
        "predicted_class_default": np.where(test_e2_probs >= 0.50, "mel", "non_mel"),
        "predicted_class_calibrated": np.where(test_e2_probs >= e2_thresh, "mel", "non_mel"),
    })
    e2_pred_df.to_csv(e2_dir / "predictions.csv", index=False)

    # =========================================================================
    # EXPERIMENT 3: ViT + Segmentation + ABC Clinical Features (Full Multimodal)
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 3: ViT + Segmentation Morphology + ABC Clinical (16-D)")
    print("=" * 75)
    e3_model = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2).to(DEVICE)
    # Load previously trained best weights
    fusion_weights_path = REPO_ROOT / "models" / "multimodal" / "fusion_model.pt"
    if fusion_weights_path.exists():
        e3_model.load_state_dict(torch.load(fusion_weights_path, map_location=DEVICE))
    else:
        e3_model, _ = train_ablation_model(
            train_v, train_c, train_y,
            val_v, val_c, val_y,
            clinical_dim=16,
            exp_name="E3: Full Multimodal",
        )
    e3_model.eval()

    # Load validation threshold config
    with open(REPO_ROOT / "models" / "multimodal" / "fusion_config.json") as f:
        e3_thresh_cfg = json.load(f)
    e3_thresh = float(e3_thresh_cfg.get("clinical_threshold", 0.36))

    e3_total_params = vit_total_params + sum(p.numel() for p in e3_model.parameters())
    e3_head_params = sum(p.numel() for p in e3_model.parameters())

    with torch.no_grad():
        test_e3_probs = e3_model.predict_proba(test_v.to(DEVICE), test_c.to(DEVICE))[:, 1].cpu().numpy()

    sample_c = test_c[:1].to(DEVICE)
    e3_latency = measure_inference_latency(lambda x: e3_model(x[0], x[1]), (sample_v, sample_c))

    e3_metrics_default = compute_comprehensive_metrics(targets, test_e3_probs, 0.50)
    e3_metrics_calibrated = compute_comprehensive_metrics(targets, test_e3_probs, e3_thresh)

    e3_dir = EXPERIMENTS_DIR / "E3_vit_seg_abc"
    e3_dir.mkdir(parents=True, exist_ok=True)
    torch.save(e3_model.state_dict(), e3_dir / "weights.pt")

    with open(e3_dir / "config.json", "w") as f:
        json.dump({
            "experiment_id": "E3",
            "name": "ViT + Segmentation Morphology + ABC Clinical",
            "features": "768-D ViT + 16-D clinical & morphological vector",
            "total_parameters": e3_total_params,
            "fusion_head_parameters": e3_head_params,
            "threshold_calibration": e3_thresh_cfg,
            "calibrated_threshold": e3_thresh,
            "latency": e3_latency,
        }, f, indent=2)

    with open(e3_dir / "metrics.json", "w") as f:
        json.dump({
            "default_threshold_0.50": e3_metrics_default,
            "calibrated_threshold": e3_metrics_calibrated,
        }, f, indent=2)

    e3_pred_df = pd.DataFrame({
        "image_id": test_df["image_id"],
        "ground_truth": targets,
        "probability_melanoma": np.round(test_e3_probs, 4),
        "probability_non_melanoma": np.round(1.0 - test_e3_probs, 4),
        "predicted_class_default": np.where(test_e3_probs >= 0.50, "mel", "non_mel"),
        "predicted_class_calibrated": np.where(test_e3_probs >= e3_thresh, "mel", "non_mel"),
    })
    e3_pred_df.to_csv(e3_dir / "predictions.csv", index=False)

    # =========================================================================
    # PLOT GENERATION
    # =========================================================================
    experiments_data = {
        "E1": {
            "name": "ViT-Only Baseline",
            "targets": targets,
            "probs": test_vit_probs,
            "metrics": e1_metrics_calibrated,
        },
        "E2": {
            "name": "ViT + ABC Clinical",
            "targets": targets,
            "probs": test_e2_probs,
            "metrics": e2_metrics_calibrated,
        },
        "E3": {
            "name": "ViT + Seg + ABC (Full)",
            "targets": targets,
            "probs": test_e3_probs,
            "metrics": e3_metrics_calibrated,
        },
    }

    render_roc_curves(experiments_data, REPORTS_DIR / "roc_curves_comparison.png")
    render_pr_curves(experiments_data, REPORTS_DIR / "pr_curves_comparison.png")
    render_confusion_matrices(experiments_data, REPORTS_DIR / "confusion_matrices_comparison.png")

    # =========================================================================
    # ABLATION STUDY SUMMARY
    # =========================================================================
    ablation_summary = {
        "study_name": "CEFM Modality Ablation Study",
        "dataset": "HAM10000 Independent Test Set",
        "test_sample_count": len(targets),
        "class_distribution": {"non_melanoma": int((targets == 0).sum()), "melanoma": int((targets == 1).sum())},
        "experiments": {
            "E1_vit_only": {
                "name": "ViT-Only Baseline",
                "calibrated_threshold": e1_thresh,
                "metrics": e1_metrics_calibrated,
                "latency_ms": e1_latency["mean_latency_ms"],
                "total_parameters": vit_total_params,
            },
            "E2_vit_abc": {
                "name": "ViT + ABC Clinical Features",
                "calibrated_threshold": e2_thresh,
                "metrics": e2_metrics_calibrated,
                "latency_ms": e2_latency["mean_latency_ms"],
                "total_parameters": e2_total_params,
            },
            "E3_vit_seg_abc": {
                "name": "ViT + Segmentation Morphology + ABC Clinical",
                "calibrated_threshold": e3_thresh,
                "metrics": e3_metrics_calibrated,
                "latency_ms": e3_latency["mean_latency_ms"],
                "total_parameters": e3_total_params,
            },
        },
    }

    with open(EXPERIMENTS_DIR / "ablation_study_summary.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)

    # Print summary comparative table
    print("\n" + "=" * 105)
    print("COMPARATIVE RESEARCH ABLATION STUDY RESULTS (1,504 INDEPENDENT TEST SAMPLES)")
    print("=" * 105)
    header = f"{'Metric':<24} | {'E1: ViT Baseline':<24} | {'E2: ViT + ABC':<24} | {'E3: ViT + Seg + ABC':<24}"
    print(header)
    print("-" * 105)

    def row(name, k, is_pct=True):
        m1 = e1_metrics_calibrated[k]
        m2 = e2_metrics_calibrated[k]
        m3 = e3_metrics_calibrated[k]
        if is_pct:
            return f"{name:<24} | {m1:<24.2%} | {m2:<24.2%} | {m3:<24.2%}"
        return f"{name:<24} | {m1:<24.4f} | {m2:<24.4f} | {m3:<24.4f}"

    print(row("ROC-AUC", "roc_auc", False))
    print(row("PR-AUC", "pr_auc", False))
    print(row("Sensitivity (Recall)", "sensitivity", True))
    print(row("Specificity", "specificity", True))
    print(row("Precision", "precision", True))
    print(row("F1-Score", "f1_score", True))
    print(row("Balanced Accuracy", "balanced_accuracy", True))
    print(row("Overall Accuracy", "accuracy", True))

    cm1 = e1_metrics_calibrated["confusion_matrix"]
    cm2 = e2_metrics_calibrated["confusion_matrix"]
    cm3 = e3_metrics_calibrated["confusion_matrix"]
    fp_str = f"{'False Positives (FP)':<24} | {cm1['fp']:<24} | {cm2['fp']:<24} | {cm3['fp']:<24}"
    fn_str = f"{'False Negatives (FN)':<24} | {cm1['fn']:<24} | {cm2['fn']:<24} | {cm3['fn']:<24}"
    lat_str = f"{'Mean Latency (ms)':<24} | {e1_latency['mean_latency_ms']:<24.3f} | {e2_latency['mean_latency_ms']:<24.3f} | {e3_latency['mean_latency_ms']:<24.3f}"
    par_str = f"{'Total Parameters':<24} | {f'{vit_total_params:,}':<24} | {f'{e2_total_params:,}':<24} | {f'{e3_total_params:,}':<24}"

    print(fp_str)
    print(fn_str)
    print(lat_str)
    print(par_str)
    print("=" * 105)
    print(f"Artifacts saved in {EXPERIMENTS_DIR} and {REPORTS_DIR}\n")


if __name__ == "__main__":
    run_all_experiments()
