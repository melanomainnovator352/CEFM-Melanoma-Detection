"""
CEFM Phase 2: Dual-Scale Multimodal Fusion Model Training & Evaluation.

Trains and benchmarks:
  - concat: ConcatDualScaleFusion
  - gated: GatedDualScaleFusion (Residual Channel Attention & Clinical Modulation)
  - cross_attn: CrossAttentionDualScaleFusion (Cross-modal Token Attention)

Methodology & Leakage Guards:
  - Clinical features normalized strictly with training-set statistics.
  - Decision threshold calibrated exclusively on validation set to optimize F1 / clinical utility.
  - Test set evaluated independently with calibrated threshold.
  - Existing production checkpoints (models/multimodal/fusion_model.pt) remain untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

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
    confusion_matrix,
)

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.dual_scale_fusion import (
    ConcatDualScaleFusion,
    GatedDualScaleFusion,
    CrossAttentionDualScaleFusion,
)

CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
EXPERIMENTS_DIR = REPO_ROOT / "models" / "experiments"
REPORTS_DIR = REPO_ROOT / "reports"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cached_splits():
    """Load cached global, local, and clinical features for train, val, and test."""
    train_data = torch.load(CACHE_DIR / "dual_scale_train.pt", map_location="cpu")
    val_data = torch.load(CACHE_DIR / "dual_scale_validation.pt", map_location="cpu")
    test_data = torch.load(CACHE_DIR / "dual_scale_test.pt", map_location="cpu")

    return train_data, val_data, test_data


def compute_clinical_norm(train_clinical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std strictly on training clinical features."""
    mean = train_clinical.mean(dim=0, keepdim=True)
    std = train_clinical.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def apply_clinical_norm(feat: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize clinical tensor using precomputed train statistics."""
    return (feat - mean) / std


def calibrate_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Calibrate optimal decision threshold strictly on validation predictions.
    Maximizes balanced F1-score across threshold candidates [0.05, 0.95].
    """
    best_thresh = 0.5
    best_f1 = -1.0

    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        # Prioritize sensitivity if f1 ties
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return float(np.round(best_thresh, 4))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute comprehensive diagnostic metrics at specified threshold."""
    preds = (probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
    acc = float(accuracy_score(y_true, preds))
    sens = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract melanoma probabilities and true labels from a DataLoader."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for global_v, local_v, clin, y in loader:
            global_v = global_v.to(device)
            local_v = local_v.to(device)
            clin = clin.to(device)

            probs = model.predict_proba(global_v, local_v, clin)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def train_single_architecture(
    model_type: str,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
) -> Dict[str, Any]:
    """Train and evaluate one dual-scale fusion model architecture."""
    print(f"\n{'='*70}\n[Training Architecture: {model_type.upper()}]\n{'='*70}")

    # 1. Normalize clinical features strictly using train stats
    mean, std = compute_clinical_norm(train_data["clinical"])
    train_clin = apply_clinical_norm(train_data["clinical"], mean, std)
    val_clin = apply_clinical_norm(val_data["clinical"], mean, std)
    test_clin = apply_clinical_norm(test_data["clinical"], mean, std)

    # 2. Datasets & Loaders
    train_ds = TensorDataset(train_data["global_visual"], train_data["local_visual"], train_clin, train_data["labels"])
    val_ds = TensorDataset(val_data["global_visual"], val_data["local_visual"], val_clin, val_data["labels"])
    test_ds = TensorDataset(test_data["global_visual"], test_data["local_visual"], test_clin, test_data["labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Model instantiation
    if model_type == "concat":
        model = ConcatDualScaleFusion()
    elif model_type == "gated":
        model = GatedDualScaleFusion()
    elif model_type == "cross_attn":
        model = CrossAttentionDualScaleFusion()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(DEVICE)

    # 4. Class weighting for loss
    n_total = len(train_data["labels"])
    n_mel = int((train_data["labels"] == 1).sum().item())
    n_non_mel = n_total - n_mel
    weight_mel = n_total / (2.0 * n_mel)
    weight_non_mel = n_total / (2.0 * n_non_mel)
    class_weights = torch.tensor([weight_non_mel, weight_mel], dtype=torch.float32).to(DEVICE)
    print(f"Train samples: {n_total} (Non-Mel: {n_non_mel}, Mel: {n_mel}). Mel weight: {weight_mel:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training loop tracking best validation ROC-AUC
    best_val_auc = -1.0
    best_weights = None
    save_dir = EXPERIMENTS_DIR / f"dual_scale_{model_type}"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "weights.pt"
    config_path = save_dir / "config.json"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for g_v, l_v, c_v, y in train_loader:
            g_v, l_v, c_v, y = g_v.to(DEVICE), l_v.to(DEVICE), c_v.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(g_v, l_v, c_v)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y)

        scheduler.step()
        train_loss = total_loss / n_total

        # Validation evaluation
        val_y, val_probs = evaluate_model(model, val_loader, DEVICE)
        val_auc = roc_auc_score(val_y, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, weights_path)
            improved = "*"
        else:
            improved = ""

        if epoch % 5 == 0 or epoch == epochs or improved == "*":
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {train_loss:.4f} | Val ROC-AUC: {val_auc:.4f} {improved}")

    # 6. Load best model for validation threshold calibration & test evaluation
    model.load_state_dict(best_weights)
    val_y, val_probs = evaluate_model(model, val_loader, DEVICE)
    val_threshold = calibrate_threshold(val_y, val_probs)
    val_metrics = compute_metrics(val_y, val_probs, val_threshold)
    print(f"\n[Validation Set Calibration - {model_type.upper()}]")
    print(f"Best Val ROC-AUC: {best_val_auc:.4f} | Optimal Threshold (F1): {val_threshold:.4f}")
    print(f"Val Sens: {val_metrics['sensitivity']*100:.1f}%, Spec: {val_metrics['specificity']*100:.1f}%, F1: {val_metrics['f1_score']*100:.1f}%")

    # 7. Independent Test Set Evaluation
    test_y, test_probs = evaluate_model(model, test_loader, DEVICE)
    test_metrics = compute_metrics(test_y, test_probs, val_threshold)
    test_metrics_default = compute_metrics(test_y, test_probs, 0.50)

    print(f"\n[Independent Test Set Evaluation - {model_type.upper()}] (At Val Threshold {val_threshold:.4f})")
    print(f"ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:      {test_metrics['pr_auc']:.4f}")
    print(f"Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity: {test_metrics['sensitivity']*100:.2f}%")
    print(f"Specificity: {test_metrics['specificity']*100:.2f}%")
    print(f"Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"F1-score:    {test_metrics['f1_score']*100:.2f}%")
    print(f"Confusion:   {test_metrics['confusion_matrix']}")

    # 8. Check specific test case ISIC_0012418
    test_image_ids = list(test_data["image_ids"])
    isic_0012418_result = None
    if "ISIC_0012418" in test_image_ids:
        idx_case = test_image_ids.index("ISIC_0012418")
        case_prob = float(test_probs[idx_case])
        case_pred = int(case_prob >= val_threshold)
        isic_0012418_result = {
            "image_id": "ISIC_0012418",
            "ground_truth": int(test_y[idx_case]),
            "melanoma_probability": round(case_prob, 4),
            "predicted_class": "melanoma" if case_pred == 1 else "non_mel",
            "decision_threshold": val_threshold,
        }
        print(f"\n[Case ISIC_0012418 Diagnosis under {model_type.upper()}]: Prob={case_prob*100:.2f}%, Pred={isic_0012418_result['predicted_class']}")

    # 9. Save configuration & metrics
    config = {
        "model_type": model_type,
        "global_dim": 768,
        "local_dim": 768,
        "clinical_dim": 16,
        "hidden_dim": 128,
        "clinical_hidden_dim": 64,
        "num_classes": 2,
        "dropout": 0.15,
        "decision_threshold": val_threshold,
        "clinical_norm_mean": mean.squeeze().tolist(),
        "clinical_norm_std": std.squeeze().tolist(),
        "best_val_roc_auc": round(float(best_val_auc), 4),
        "test_metrics_calibrated": test_metrics,
        "test_metrics_default_05": test_metrics_default,
        "isic_0012418_assessment": isic_0012418_result,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Scale Multimodal Fusion Architectures")
    parser.add_argument("--model", type=str, default="all", choices=["concat", "gated", "cross_attn", "all"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("CEFM Phase 2: Dual-Scale Multimodal Fusion Training Suite")
    print("=" * 70)

    train_data, val_data, test_data = load_cached_splits()
    print(f"Loaded Cached Splits:")
    print(f"  Train:      {len(train_data['labels'])} samples (Mel: {(train_data['labels']==1).sum().item()})")
    print(f"  Validation: {len(val_data['labels'])} samples (Mel: {(val_data['labels']==1).sum().item()})")
    print(f"  Test:       {len(test_data['labels'])} samples (Mel: {(test_data['labels']==1).sum().item()})")

    models_to_train = ["concat", "gated", "cross_attn"] if args.model == "all" else [args.model]
    all_results = {}

    for m_type in models_to_train:
        res = train_single_architecture(
            model_type=m_type,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_results[m_type] = res

    # Save summary report
    summary_path = REPORTS_DIR / "dual_scale_multimodal_benchmarks.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("All Dual-Scale Architectures Successfully Trained & Evaluated!")
    print(f"Results summary saved to {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

