"""
CEFM Phase 2: Systematic Training and Benchmarking Suite for Models M1–M7.

Experiments:
  M1: Global ViT Only (768-D)
  M2: Local ViT Only (768-D native-resolution patch)
  M3: Dual-Scale ViT (Visual Only: 768-D global + 768-D local -> 1536-D)
  M4: Global ViT + Clinical (Phase 1 Multimodal Architecture: 768-D + 16-D)
  M5: Dual-Scale Concat Multimodal (768-D + 768-D + 16-D)
  M6: Dual-Scale Gated Multimodal (RCAG / Gated Fusion)
  M7: Dual-Scale Cross-Attention (Multi-Head Cross-Attention)

Methodology:
  - Preserves all Phase 1 production checkpoints.
  - Training on 6,978 samples.
  - Validation (1,533 samples) used strictly for early stopping and threshold calibration.
  - Independent test set (1,504 samples) evaluated only after training.
  - 1,000 stratified bootstrap iterations for 95% confidence intervals.
  - Validation-derived matched specificity comparison.
  - Generation of comparative charts and summary reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.dual_scale_models import MODEL_REGISTRY

CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
EXPERIMENTS_DIR = REPO_ROOT / "models" / "experiments"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_cached_features() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print("Loading cached dual-scale feature datasets...")
    train_data = torch.load(CACHE_DIR / "dual_scale_train.pt", map_location="cpu")
    val_data = torch.load(CACHE_DIR / "dual_scale_validation.pt", map_location="cpu")
    test_data = torch.load(CACHE_DIR / "dual_scale_test.pt", map_location="cpu")
    return train_data, val_data, test_data


def compute_clinical_norm(train_clinical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_clinical.mean(dim=0, keepdim=True)
    std = train_clinical.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def apply_clinical_norm(clinical: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (clinical - mean) / std


def evaluate_split_predictions(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for g_v, l_v, c_v, y in loader:
            g_v = g_v.to(DEVICE)
            l_v = l_v.to(DEVICE)
            c_v = c_v.to(DEVICE)
            probs = model.predict_proba(g_v, l_v, c_v)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist())
    return np.array(all_targets, dtype=np.int32), np.array(all_probs, dtype=np.float32)


def calibrate_threshold_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    best_t = 0.50
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (p_val >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return float(np.round(best_t, 4))


def calibrate_threshold_matched_spec(y_val: np.ndarray, p_val: np.ndarray, target_spec: float) -> float:
    best_t = 0.50
    best_diff = 999.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (p_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        diff = abs(spec - target_spec)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)
    return float(np.round(best_t, 4))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
    acc = float(accuracy_score(y_true, preds))
    sens = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

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

    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        y_b = y_true[b_idx]
        p_b = probs[b_idx]
        preds_b = (p_b >= threshold).astype(int)

        try:
            ra = roc_auc_score(y_b, p_b)
            pa = average_precision_score(y_b, p_b)
        except Exception:
            continue

        acc = accuracy_score(y_b, preds_b)
        sens = recall_score(y_b, preds_b, pos_label=1, zero_division=0)
        prec = precision_score(y_b, preds_b, pos_label=1, zero_division=0)
        f1 = f1_score(y_b, preds_b, pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y_b, preds_b)

        tn, fp, fn, tp = confusion_matrix(y_b, preds_b, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        boot_metrics["roc_auc"].append(ra)
        boot_metrics["pr_auc"].append(pa)
        boot_metrics["accuracy"].append(acc)
        boot_metrics["sensitivity"].append(sens)
        boot_metrics["specificity"].append(spec)
        boot_metrics["precision"].append(prec)
        boot_metrics["f1_score"].append(f1)
        boot_metrics["balanced_accuracy"].append(bal)

    ci_results = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
            ci_results[k] = (round(low, 4), round(high, 4))
        else:
            ci_results[k] = (0.0, 0.0)

    return ci_results


def train_model(
    model_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    patience: int = 8,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    set_seed(42)
    meta = MODEL_REGISTRY[model_id]
    model_cls = meta["class"]
    model = model_cls().to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "lr": [],
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for g_v, l_v, c_v, y in train_loader:
            g_v, l_v, c_v, y = g_v.to(DEVICE), l_v.to(DEVICE), c_v.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(g_v, l_v, c_v)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            n_train += len(y)

        train_loss /= n_train
        scheduler.step()

        # Validation pass
        model.eval()
        val_loss = 0.0
        n_val = 0
        all_val_probs = []
        all_val_y = []

        with torch.no_grad():
            for g_v, l_v, c_v, y in val_loader:
                g_v, l_v, c_v, y = g_v.to(DEVICE), l_v.to(DEVICE), c_v.to(DEVICE), y.to(DEVICE)
                logits = model(g_v, l_v, c_v)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)

                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_val_probs.extend(probs.cpu().numpy().tolist())
                all_val_y.extend(y.cpu().numpy().tolist())

        val_loss /= n_val
        val_auc = roc_auc_score(all_val_y, all_val_probs)
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(ep)
        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_auc"].append(round(float(val_auc), 5))
        history["lr"].append(current_lr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            star = "*"
        else:
            epochs_no_improve += 1
            star = ""

        if ep % 5 == 0 or ep == epochs or star == "*":
            print(f"[{model_id}] Ep {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} {star}")

        if epochs_no_improve >= patience:
            print(f"[{model_id}] Early stopping triggered at epoch {ep} (Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}).")
            break

    elapsed = round(time.time() - t0, 1)
    print(f"[{model_id}] Training finished in {elapsed}s. Best epoch: {best_epoch} with Val AUC: {best_val_auc:.4f}")

    # Load best weights
    model.load_state_dict(best_state)

    train_summary = {
        "model_id": model_id,
        "name": meta["name"],
        "best_epoch": best_epoch,
        "best_val_auc": round(float(best_val_auc), 4),
        "total_epochs_trained": len(history["epoch"]),
        "training_time_seconds": elapsed,
        "batch_size": 64,
        "initial_lr": lr,
        "weight_decay": weight_decay,
    }

    return model, train_summary, history


# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS USING OPENCV & PIL
# -----------------------------------------------------------------------------

def plot_multicurve_roc(models_roc: Dict[str, Tuple[np.ndarray, np.ndarray, float]], out_path: Path):
    """Draw high-resolution comparative ROC curves."""
    W, H = 1000, 750
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    margin_l, margin_r, margin_t, margin_b = 90, 240, 70, 70
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    # Draw grid & axes
    cv2.rectangle(img, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (200, 200, 200), 1)
    for tick in np.linspace(0, 1, 6):
        x = int(margin_l + tick * plot_w)
        y = int(margin_t + (1.0 - tick) * plot_h)
        cv2.line(img, (x, margin_t), (x, margin_t + plot_h), (235, 235, 235), 1)
        cv2.line(img, (margin_l, y), (margin_l + plot_w, y), (235, 235, 235), 1)
        cv2.putText(img, f"{tick:.1f}", (x - 12, margin_t + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(img, f"{tick:.1f}", (margin_l - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # Diagonal chance line
    cv2.line(img, (margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t), (180, 180, 180), 1, cv2.LINE_AA)

    # Title & Axis labels
    cv2.putText(img, "CEFM Phase 2: ROC Curves Comparison (M1 - M7)", (margin_l, margin_t - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    cv2.putText(img, "False Positive Rate (1 - Specificity)", (margin_l + plot_w // 4, margin_t + plot_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(img, "True Positive Rate (Sensitivity)", (15, margin_t + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    # Colors for M1-M7
    palette = [
        (120, 120, 120),  # M1 gray
        (180, 100, 50),   # M2 brown
        (30, 150, 30),    # M3 green
        (0, 140, 230),    # M4 orange
        (200, 30, 150),   # M5 purple
        (220, 80, 20),    # M6 blue
        (20, 20, 220),    # M7 red (accent)
    ]

    leg_y = margin_t + 20
    for idx, (m_id, (fpr, tpr, auc)) in enumerate(models_roc.items()):
        color = palette[idx % len(palette)]
        # Downsample points for smooth plotting
        pts = []
        for fx, fy in zip(fpr, tpr):
            px = int(margin_l + fx * plot_w)
            py = int(margin_t + (1.0 - fy) * plot_h)
            pts.append((px, py))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

        # Legend entry
        lx = margin_l + plot_w + 15
        cv2.line(img, (lx, leg_y), (lx + 25, leg_y), color, 3)
        cv2.putText(img, f"{m_id}: {auc:.4f}", (lx + 32, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        leg_y += 28

    cv2.imwrite(str(out_path), img)


def plot_multicurve_pr(models_pr: Dict[str, Tuple[np.ndarray, np.ndarray, float]], out_path: Path):
    """Draw high-resolution Precision-Recall curves."""
    W, H = 1000, 750
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    margin_l, margin_r, margin_t, margin_b = 90, 240, 70, 70
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    cv2.rectangle(img, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (200, 200, 200), 1)
    for tick in np.linspace(0, 1, 6):
        x = int(margin_l + tick * plot_w)
        y = int(margin_t + (1.0 - tick) * plot_h)
        cv2.line(img, (x, margin_t), (x, margin_t + plot_h), (235, 235, 235), 1)
        cv2.line(img, (margin_l, y), (margin_l + plot_w, y), (235, 235, 235), 1)
        cv2.putText(img, f"{tick:.1f}", (x - 12, margin_t + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(img, f"{tick:.1f}", (margin_l - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.putText(img, "CEFM Phase 2: Precision-Recall Curves (M1 - M7)", (margin_l, margin_t - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    cv2.putText(img, "Recall (Sensitivity)", (margin_l + plot_w // 3, margin_t + plot_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(img, "Precision", (25, margin_t + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    palette = [
        (120, 120, 120),
        (180, 100, 50),
        (30, 150, 30),
        (0, 140, 230),
        (200, 30, 150),
        (220, 80, 20),
        (20, 20, 220),
    ]

    leg_y = margin_t + 20
    for idx, (m_id, (rec, prec, pr_auc)) in enumerate(models_pr.items()):
        color = palette[idx % len(palette)]
        pts = []
        for rx, py in zip(rec, prec):
            px = int(margin_l + rx * plot_w)
            py_pos = int(margin_t + (1.0 - py) * plot_h)
            pts.append((px, py_pos))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

        lx = margin_l + plot_w + 15
        cv2.line(img, (lx, leg_y), (lx + 25, leg_y), color, 3)
        cv2.putText(img, f"{m_id}: {pr_auc:.4f}", (lx + 32, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        leg_y += 28

    cv2.imwrite(str(out_path), img)


def plot_confusion_matrices(cms: Dict[str, Dict[str, int]], out_path: Path):
    """Draw a 4x2 grid of confusion matrix heatmaps."""
    W, H = 1200, 700
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    cv2.putText(img, "CEFM Phase 2: Confusion Matrices on Independent Test Set (1,504 samples)", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)

    coords = [
        (60, 80), (340, 80), (620, 80), (900, 80),
        (60, 380), (340, 380), (620, 380)
    ]

    for idx, (m_id, cm) in enumerate(cms.items()):
        if idx >= len(coords):
            break
        x0, y0 = coords[idx]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        total = tn + fp + fn + tp

        cv2.putText(img, f"[{m_id}]", (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)

        # 2x2 grid
        cell_size = 90
        gx, gy = x0, y0 + 30
        cv2.rectangle(img, (gx, gy), (gx + cell_size * 2, gy + cell_size * 2), (180, 180, 180), 2)
        cv2.line(img, (gx + cell_size, gy), (gx + cell_size, gy + cell_size * 2), (180, 180, 180), 1)
        cv2.line(img, (gx, gy + cell_size), (gx + cell_size * 2, gy + cell_size), (180, 180, 180), 1)

        # Labels
        cv2.putText(img, f"TN:{tn}", (gx + 12, gy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 120, 30), 2)
        cv2.putText(img, f"FP:{fp}", (gx + cell_size + 12, gy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        cv2.putText(img, f"FN:{fn}", (gx + 12, gy + cell_size + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        cv2.putText(img, f"TP:{tp}", (gx + cell_size + 12, gy + cell_size + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 120, 30), 2)

        # Sub-stats
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        cv2.putText(img, f"Sens: {sens*100:.1f}% | Spec: {spec*100:.1f}%", (gx, gy + cell_size * 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
        cv2.putText(img, f"False Negatives: {fn}", (gx, gy + cell_size * 2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)

    cv2.imwrite(str(out_path), img)


def plot_loss_curves(histories: Dict[str, Dict[str, Any]], out_path: Path):
    """Draw training & validation loss curves for all models."""
    W, H = 1000, 650
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    margin_l, margin_r, margin_t, margin_b = 80, 220, 60, 60
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    cv2.rectangle(img, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (200, 200, 200), 1)
    max_epochs = 30
    max_loss = 1.0

    cv2.putText(img, "CEFM Phase 2: Training & Validation Loss Curves", (margin_l, margin_t - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, "Epoch", (margin_l + plot_w // 2, margin_t + plot_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(img, "Loss (Cross-Entropy)", (15, margin_t + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

    palette = [
        (120, 120, 120), (180, 100, 50), (30, 150, 30),
        (0, 140, 230), (200, 30, 150), (220, 80, 20), (20, 20, 220)
    ]

    leg_y = margin_t + 20
    for idx, (m_id, hist) in enumerate(histories.items()):
        color = palette[idx % len(palette)]
        eps = hist["epoch"]
        t_losses = hist["train_loss"]
        v_losses = hist["val_loss"]

        # Train loss (solid)
        pts_t = []
        for ep, l in zip(eps, t_losses):
            px = int(margin_l + (ep / max_epochs) * plot_w)
            py = int(margin_t + (1.0 - min(l, max_loss) / max_loss) * plot_h)
            pts_t.append((px, py))
        for p1, p2 in zip(pts_t[:-1], pts_t[1:]):
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

        # Val loss (thicker)
        pts_v = []
        for ep, l in zip(eps, v_losses):
            px = int(margin_l + (ep / max_epochs) * plot_w)
            py = int(margin_t + (1.0 - min(l, max_loss) / max_loss) * plot_h)
            pts_v.append((px, py))
        for p1, p2 in zip(pts_v[:-1], pts_v[1:]):
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

        # Legend
        lx = margin_l + plot_w + 15
        cv2.line(img, (lx, leg_y), (lx + 25, leg_y), color, 3)
        cv2.putText(img, f"{m_id} (Val: {v_losses[-1]:.3f})", (lx + 32, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        leg_y += 28

    cv2.imwrite(str(out_path), img)


def plot_fn_comparison(fn_data: Dict[str, Tuple[int, float]], out_path: Path):
    """Draw a bar chart comparing false-negative counts and sensitivity."""
    W, H = 900, 500
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    margin_l, margin_r, margin_t, margin_b = 80, 50, 60, 80
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    cv2.putText(img, "CEFM Phase 2: False-Negative Melanoma Comparison (Test Set: 166 Melanomas)", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.line(img, (margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h), (180, 180, 180), 2)

    n_models = len(fn_data)
    bar_width = plot_w // (n_models * 2)
    max_fn = 80  # approx scale

    for idx, (m_id, (fn, sens)) in enumerate(fn_data.items()):
        bx = margin_l + int((idx + 0.5) * (plot_w / n_models))
        bh = int((fn / max_fn) * plot_h)
        by = margin_t + plot_h - bh

        color = (50, 50, 220) if fn <= 45 else (120, 120, 120)
        cv2.rectangle(img, (bx - bar_width // 2, by), (bx + bar_width // 2, margin_t + plot_h), color, -1)
        cv2.putText(img, f"{fn} FN", (bx - 22, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 2)
        cv2.putText(img, f"{sens*100:.1f}%", (bx - 22, by - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 120, 30), 1)
        cv2.putText(img, m_id, (bx - 15, margin_t + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2)

    cv2.imwrite(str(out_path), img)


# -----------------------------------------------------------------------------
# MAIN EXPERIMENT RUNNER
# -----------------------------------------------------------------------------

def run_all_experiments():
    set_seed(42)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CEFM PHASE 2: SYSTEMATIC TRAINING & BENCHMARKING (M1 - M7)")
    print("=" * 80)

    # 1. Load cached splits
    train_data, val_data, test_data = load_cached_features()

    # 2. Normalize clinical features strictly using train stats
    c_mean, c_std = compute_clinical_norm(train_data["clinical"])
    train_clin = apply_clinical_norm(train_data["clinical"], c_mean, c_std)
    val_clin = apply_clinical_norm(val_data["clinical"], c_mean, c_std)
    test_clin = apply_clinical_norm(test_data["clinical"], c_mean, c_std)

    # 3. Create TensorDatasets & DataLoaders
    batch_size = 64
    train_ds = TensorDataset(train_data["global_visual"], train_data["local_visual"], train_clin, train_data["labels"])
    val_ds = TensorDataset(val_data["global_visual"], val_data["local_visual"], val_clin, val_data["labels"])
    test_ds = TensorDataset(test_data["global_visual"], test_data["local_visual"], test_clin, test_data["labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4. Class weights from train labels
    n_total = len(train_data["labels"])
    n_mel = int((train_data["labels"] == 1).sum().item())
    n_non_mel = n_total - n_mel
    w_non_mel = n_total / (2.0 * n_non_mel)
    w_mel = n_total / (2.0 * n_mel)
    class_weights = torch.tensor([w_non_mel, w_mel], dtype=torch.float32).to(DEVICE)
    print(f"Class Weights (Strictly from Train): Non-Mel={w_non_mel:.3f}, Mel={w_mel:.3f}")

    # 5. Determine Validation-Derived Target Specificity from Baseline
    # Standard validation operating target: 90.0% validation specificity
    target_val_spec = 0.9000
    print(f"Validation Target Specificity for Matched Comparisons: {target_val_spec*100:.1f}%\n")

    trained_models = {}
    train_summaries = {}
    train_histories = {}
    val_predictions = {}
    test_predictions = {}

    models_to_run = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]

    # 6. Train all models
    for m_id in models_to_run:
        print("-" * 80)
        print(f"TRAINING MODEL {m_id}: {MODEL_REGISTRY[m_id]['name']}")
        print(f"Description: {MODEL_REGISTRY[m_id]['desc']}")
        print("-" * 80)

        model, summary, history = train_model(
            model_id=m_id,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            epochs=30,
            lr=1e-4,
            weight_decay=1e-3,
            patience=8,
        )

        trained_models[m_id] = model
        train_summaries[m_id] = summary
        train_histories[m_id] = history

        # Save model weights to separate experiment directory
        exp_dir = EXPERIMENTS_DIR / m_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        weights_path = exp_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)
        summary["weights_sha256"] = get_file_sha256(weights_path)

        with open(exp_dir / "config.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(exp_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Validation predictions
        y_val, p_val = evaluate_split_predictions(model, val_loader)
        val_predictions[m_id] = (y_val, p_val)

        # Test predictions
        y_test, p_test = evaluate_split_predictions(model, test_loader)
        test_predictions[m_id] = (y_test, p_test)

    # 7. Threshold Calibration & Test Evaluation
    print("\n" + "=" * 80)
    print("INDEPENDENT TEST SET EVALUATION & STATISTICAL BOOTSTRAPPING")
    print("=" * 80)

    final_results = {}
    models_roc = {}
    models_pr = {}
    models_cm = {}
    fn_summary = {}

    test_image_ids = list(test_data["image_ids"])
    isic_0012418_idx = test_image_ids.index("ISIC_0012418") if "ISIC_0012418" in test_image_ids else None

    for m_id in models_to_run:
        y_val, p_val = val_predictions[m_id]
        y_test, p_test = test_predictions[m_id]

        # 1. Validation-selected optimal F1 threshold
        t_f1 = calibrate_threshold_f1(y_val, p_val)

        # 2. Validation-derived matched specificity threshold
        t_spec = calibrate_threshold_matched_spec(y_val, p_val, target_val_spec)

        # 3. Evaluate on test set
        m_test_f1 = compute_metrics(y_test, p_test, t_f1)
        m_test_spec = compute_metrics(y_test, p_test, t_spec)
        m_test_05 = compute_metrics(y_test, p_test, 0.50)

        # 4. Bootstrap 95% Confidence Intervals at t_f1
        ci_f1 = compute_bootstrap_ci(y_test, p_test, t_f1, n_bootstraps=1000)

        # 5. Check ISIC_0012418
        case_info = None
        if isic_0012418_idx is not None:
            c_prob = float(p_test[isic_0012418_idx])
            case_info = {
                "image_id": "ISIC_0012418",
                "ground_truth": int(y_test[isic_0012418_idx]),
                "melanoma_probability": round(c_prob, 4),
                "predicted_class_f1": "melanoma" if c_prob >= t_f1 else "non_mel",
                "predicted_class_spec": "melanoma" if c_prob >= t_spec else "non_mel",
            }

        # Save curve points
        fpr, tpr, _ = roc_curve(y_test, p_test)
        prec, rec, _ = precision_recall_curve(y_test, p_test)
        models_roc[m_id] = (fpr, tpr, m_test_f1["roc_auc"])
        models_pr[m_id] = (rec, prec, m_test_f1["pr_auc"])
        models_cm[m_id] = m_test_f1["confusion_matrix"]
        fn_summary[m_id] = (m_test_f1["false_negatives"], m_test_f1["sensitivity"])

        res_entry = {
            "model_id": m_id,
            "name": MODEL_REGISTRY[m_id]["name"],
            "description": MODEL_REGISTRY[m_id]["desc"],
            "modalities": MODEL_REGISTRY[m_id]["modalities"],
            "train_summary": train_summaries[m_id],
            "calibrated_threshold_f1": t_f1,
            "calibrated_threshold_matched_spec": t_spec,
            "metrics_calibrated_f1": m_test_f1,
            "metrics_matched_spec": m_test_spec,
            "metrics_default_05": m_test_05,
            "bootstrap_95ci_f1": ci_f1,
            "isic_0012418_audit": case_info,
        }
        final_results[m_id] = res_entry

        print(f"[{m_id}] Calibrated Threshold: {t_f1:.4f} | ROC-AUC: {m_test_f1['roc_auc']:.4f} | PR-AUC: {m_test_f1['pr_auc']:.4f}")
        print(f"     Sens: {m_test_f1['sensitivity']*100:.1f}% [{ci_f1['sensitivity'][0]*100:.1f} - {ci_f1['sensitivity'][1]*100:.1f}%] | Spec: {m_test_f1['specificity']*100:.1f}% | F1: {m_test_f1['f1_score']*100:.1f}% | FN: {m_test_f1['false_negatives']}")
        if case_info:
            print(f"     ISIC_0012418 Melanoma Prob: {case_info['melanoma_probability']*100:.2f}% -> Pred: {case_info['predicted_class_f1']}")

    # 8. Render Visualizations & Plots
    print("\nRendering comparative evaluation figures...")
    plot_multicurve_roc(models_roc, FIGURES_DIR / "roc_curves_m1_m7.png")
    plot_multicurve_pr(models_pr, FIGURES_DIR / "pr_curves_m1_m7.png")
    plot_confusion_matrices(models_cm, FIGURES_DIR / "confusion_matrices_m1_m7.png")
    plot_loss_curves(train_histories, FIGURES_DIR / "loss_curves_m1_m7.png")
    plot_fn_comparison(fn_summary, FIGURES_DIR / "false_negative_comparison.png")
    print(f"Figures saved to {FIGURES_DIR}")

    # 9. Save full comparative report
    out_json = REPORTS_DIR / "phase2_m1_m7_comparative_benchmarks.json"
    with open(out_json, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Comprehensive benchmark report saved to {out_json}")

    # 10. Final Verification of Phase 1 Production Checkpoint Hashes
    print("\nVerifying Phase 1 production checkpoints integrity...")
    phase1_checks = {
        "vit_model": (REPO_ROOT / "models" / "classification" / "vit_model" / "model.safetensors", "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db"),
        "unet_model": (REPO_ROOT / "models" / "segmentation" / "unet_isic2018.pt", "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e"),
        "fusion_model": (REPO_ROOT / "models" / "multimodal" / "fusion_model.pt", "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422"),
    }
    for name, (p, exp_h) in phase1_checks.items():
        act_h = get_file_sha256(p)
        assert act_h == exp_h, f"CRITICAL: Phase 1 checkpoint {name} was modified!"
        print(f"  - {name}: MATCH ({act_h[:16]}...)")
    print("All Phase 1 production checkpoints verified unchanged.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()

