"""
CEFM Step 16: Complete Training, Controlled Ablation Experiments, and Evaluation Engine.

Executes:
1. Leak-free clinical metadata preprocessing (age normalization, missing handling, 3-way sex, anatomical location vocab).
2. Training of 6 controlled experimental ablation models:
   - Model A: Visual-only baseline
   - Model B: Clinical-only branch
   - Model C: Visual + Age
   - Model D: Visual + Age + Sex
   - Model E: Visual + Age + Sex + Anatomical Location
   - Model F: Full Proposed Fusion Model (Visual + 64-D Residual Clinical Metadata Branch)
3. Validation-only threshold selection (Max F1 and Screening >=95% Sensitivity).
4. Unbiased test evaluation with:
   - Accuracy, Precision, Sensitivity, Specificity, F1, Balanced Accuracy, ROC-AUC
   - Confusion matrices
   - Lesion-level sensitivity and specificity
   - 1,000-iteration lesion-cluster bootstrap 95% confidence intervals
5. Forensic error analysis on the 27 jointly missed melanoma cases.
6. Export of all required Step 16 deliverable CSVs and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path("D:/pytest_cache")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.clinical_metadata import (
    ClinicalMetadataPreprocessor,
    ClinicalMetadataBranch,
    VisualOnlyClassifier,
    ClinicalOnlyClassifier,
    VisualAgeClassifier,
    VisualAgeSexClassifier,
    VisualAgeSexLocClassifier,
    ProposedClinicalFusionModel,
)
from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.ml.multimodal.dual_scale_models import M6_DualGated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLITS_DIR = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint"
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"
DUAL_CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
EXP_DIR = REPO_ROOT / "models" / "experiments" / "clinical_metadata_branch"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "evaluation_step16"

MISSED_MEL_CSV = REPORTS_DIR / "CEFM_STEP13_MISSED_MELANOMA_ANALYSIS.csv"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_features(
    split_name: str,
    preprocessor: ClinicalMetadataPreprocessor,
) -> Tuple[pd.DataFrame, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Load disjoint CSV, visual features from cache, and process clinical metadata."""
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    df = pd.read_csv(csv_path)

    cache_path = CACHE_DIR / f"{split_name}_features.pt"
    cache = torch.load(cache_path, map_location="cpu")

    visual = cache["visual"].float()
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)

    # Process clinical metadata
    clin_tensors = preprocessor.transform(df)

    return df, visual, clin_tensors, labels


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train model on training data and select checkpoint strictly using validation ROC-AUC."""
    model = model.to(DEVICE)

    # Positive class weighting based strictly on training class ratio
    labels_train = train_data["labels"].numpy()
    n_pos = int((labels_train == 1).sum())
    n_neg = int((labels_train == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Prepare training dataset
    # Collect all needed tensors into a tuple
    train_dataset = TensorDataset(
        train_data["visual"],
        train_data["clin"]["age_norm"],
        train_data["clin"]["age_missing"],
        train_data["clin"]["sex_onehot"],
        train_data["clin"]["loc_indices"],
        train_data["labels"],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_auc = -1.0
    best_epoch = 0
    best_state_dict = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for b_vis, b_age_n, b_age_m, b_sex, b_loc, b_y in train_loader:
            b_vis = b_vis.to(DEVICE)
            b_age_n = b_age_n.to(DEVICE)
            b_age_m = b_age_m.to(DEVICE)
            b_sex = b_sex.to(DEVICE)
            b_loc = b_loc.to(DEVICE)
            b_y = b_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(
                visual_features=b_vis,
                age_norm=b_age_n,
                age_missing=b_age_m,
                sex_onehot=b_sex,
                loc_indices=b_loc,
            )
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation evaluation
        val_probs, val_targets = predict_model(model, val_data, batch_size=batch_size)
        val_auc = float(roc_auc_score(val_targets, val_probs))
        val_pr_auc = float(average_precision_score(val_targets, val_probs))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_pr_auc": val_pr_auc,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs or epoch == best_epoch:
            print(f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} (Best: {best_val_auc:.4f} @ Ep {best_epoch})")

    # Load best checkpoint
    model.load_state_dict(best_state_dict)
    model.eval()

    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "history": history,
        "state_dict": best_state_dict,
    }


def predict_model(
    model: nn.Module,
    data: Dict[str, Any],
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inference helper returning melanoma probabilities and targets."""
    model.eval()
    dataset = TensorDataset(
        data["visual"],
        data["clin"]["age_norm"],
        data["clin"]["age_missing"],
        data["clin"]["sex_onehot"],
        data["clin"]["loc_indices"],
        data["labels"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for b_vis, b_age_n, b_age_m, b_sex, b_loc, b_y in loader:
            b_vis = b_vis.to(DEVICE)
            b_age_n = b_age_n.to(DEVICE)
            b_age_m = b_age_m.to(DEVICE)
            b_sex = b_sex.to(DEVICE)
            b_loc = b_loc.to(DEVICE)

            logits = model(
                visual_features=b_vis,
                age_norm=b_age_n,
                age_missing=b_age_m,
                sex_onehot=b_sex,
                loc_indices=b_loc,
            )
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(b_y.numpy())

    return np.array(all_probs), np.array(all_targets)


def calibrate_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
) -> Dict[str, float]:
    """Find max F1 and screening (>=95% sensitivity) thresholds strictly on validation set."""
    candidate_thresholds = np.linspace(0.01, 0.99, 500)
    best_f1 = -1.0
    best_f1_thresh = 0.50

    screening_thresh = 0.01
    screening_found = False

    for t in candidate_thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(val_targets, preds, zero_division=0)
        sens = recall_score(val_targets, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = float(t)

        if sens >= 0.95 and (not screening_found or t > screening_thresh):
            screening_thresh = float(t)
            screening_found = True

    return {
        "max_f1_threshold": best_f1_thresh,
        "val_best_f1": best_f1,
        "screening_threshold": screening_thresh,
    }


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute standard classification metrics."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    sens = float(recall_score(y_true, preds, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, preds, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    auc = float(roc_auc_score(y_true, probs))

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_lesion_level_metrics(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Aggregate predictions to lesion level using max-probability aggregation."""
    df_eval = df.copy()
    df_eval["prob"] = probs
    grouped = df_eval.groupby("lesion_id").agg({
        "label": "max",
        "prob": "max",
    })
    y_true = grouped["label"].to_numpy()
    lesion_probs = grouped["prob"].to_numpy()
    preds = (lesion_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "lesion_sensitivity": float(recall_score(y_true, preds, zero_division=0)),
        "lesion_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "lesion_balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "lesion_roc_auc": float(roc_auc_score(y_true, lesion_probs)),
        "lesion_tp": int(tp),
        "lesion_fp": int(fp),
        "lesion_tn": int(tn),
        "lesion_fn": int(fn),
    }


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Calculate 95% confidence intervals using lesion-cluster bootstrap."""
    rng = np.random.RandomState(seed)
    df_eval = df.copy()
    df_eval["prob"] = probs

    unique_lesions = df_eval["lesion_id"].unique()
    n_lesions = len(unique_lesions)

    # Pre-group indices for fast sampling
    lesion_to_indices = df_eval.groupby("lesion_id").indices

    boot_auc = []
    boot_sens = []
    boot_spec = []
    boot_bal_acc = []
    boot_f1 = []
    boot_lesion_sens = []
    boot_lesion_spec = []

    for _ in range(n_iterations):
        sample_lesions = rng.choice(unique_lesions, size=n_lesions, replace=True)
        sample_indices = np.concatenate([lesion_to_indices[l] for l in sample_lesions])

        sample_df = df_eval.iloc[sample_indices]
        y_true = sample_df["label"].to_numpy()
        sample_probs = sample_df["prob"].to_numpy()

        if len(np.unique(y_true)) < 2:
            continue

        preds = (sample_probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        boot_auc.append(roc_auc_score(y_true, sample_probs))
        boot_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        boot_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        boot_bal_acc.append(balanced_accuracy_score(y_true, preds))
        boot_f1.append(f1_score(y_true, preds, zero_division=0))

        # Lesion level aggregation in bootstrap sample
        sample_lesion_grouped = sample_df.groupby("lesion_id").agg({"label": "max", "prob": "max"})
        l_true = sample_lesion_grouped["label"].to_numpy()
        l_preds = (sample_lesion_grouped["prob"].to_numpy() >= threshold).astype(int)
        l_cm = confusion_matrix(l_true, l_preds, labels=[0, 1])
        l_tn, l_fp, l_fn, l_tp = l_cm.ravel()

        boot_lesion_sens.append(l_tp / (l_tp + l_fn) if (l_tp + l_fn) > 0 else 0.0)
        boot_lesion_spec.append(l_tn / (l_tn + l_fp) if (l_tn + l_fp) > 0 else 0.0)

    def ci(arr: List[float]) -> Tuple[float, float]:
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "roc_auc_ci": ci(boot_auc),
        "sensitivity_ci": ci(boot_sens),
        "specificity_ci": ci(boot_spec),
        "balanced_accuracy_ci": ci(boot_bal_acc),
        "f1_ci": ci(boot_f1),
        "lesion_sensitivity_ci": ci(boot_lesion_sens),
        "lesion_specificity_ci": ci(boot_lesion_spec),
    }


def main():
    set_seed(42)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CEFM STEP 16 — CLINICAL METADATA BRANCH TRAINING & CONTROLLED EXPERIMENTS")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # 1. Dataset loading and preprocessor fitting
    df_train_raw = pd.read_csv(SPLITS_DIR / "train.csv")
    print(f"Fitting ClinicalMetadataPreprocessor on train set (N={len(df_train_raw)})...")
    preprocessor = ClinicalMetadataPreprocessor().fit(df_train_raw)
    norm_stats = preprocessor.to_dict()
    with open(EXP_DIR / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats: age_mean={preprocessor.age_mean:.2f}, age_std={preprocessor.age_std:.2f}")

    print("Loading features and preparing tensor datasets...")
    df_train, vis_train, clin_train, y_train = load_dataset_features("train", preprocessor)
    df_val, vis_val, clin_val, y_val = load_dataset_features("validation", preprocessor)
    df_test, vis_test, clin_test, y_test = load_dataset_features("test", preprocessor)

    train_data = {"visual": vis_train, "clin": clin_train, "labels": y_train}
    val_data = {"visual": vis_val, "clin": clin_val, "labels": y_val}
    test_data = {"visual": vis_test, "clin": clin_test, "labels": y_test}

    print(f"Data ready: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 2. Define Model Configurations A through F
    models_to_train = {
        "ExpA_VisualOnly": (VisualOnlyClassifier(visual_dim=768, hidden_dim=128), "model_A_visual_only.pt"),
        "ExpB_ClinicalOnly": (ClinicalOnlyClassifier(hidden_dim=64), "model_B_clinical_only.pt"),
        "ExpC_VisualAge": (VisualAgeClassifier(visual_dim=768, hidden_dim=128), "model_C_visual_age.pt"),
        "ExpD_VisualAgeSex": (VisualAgeSexClassifier(visual_dim=768, hidden_dim=128), "model_D_visual_age_sex.pt"),
        "ExpE_VisualAgeSexLoc": (VisualAgeSexLocClassifier(visual_dim=768, hidden_dim=128), "model_E_visual_age_sex_loc.pt"),
        "ExpF_FullProposedFusion": (ProposedClinicalFusionModel(visual_dim=768, hidden_dim=128, clinical_dim=64), "model.pt"),
    }

    trained_models = {}
    training_results = {}

    for name, (model_inst, ckpt_name) in models_to_train.items():
        print("-" * 80)
        print(f"CHECKING/TRAINING: {name}")
        ckpt_path = EXP_DIR / ckpt_name

        if ckpt_path.exists():
            print(f"Loading existing trained checkpoint for {name} from {ckpt_path}...")
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model_inst.load_state_dict(state_dict)
            model_inst.to(DEVICE)
            model_inst.eval()
            trained_models[name] = model_inst
            trained_m = model_inst
        else:
            trained_m, res = train_single_model(
                model=model_inst,
                model_name=name,
                train_data=train_data,
                val_data=val_data,
                epochs=40,
                batch_size=64,
                lr=1e-3,
            )
            torch.save(res["state_dict"], ckpt_path)
            print(f"Saved {name} checkpoint to {ckpt_path} (Val AUC: {res['best_val_auc']:.4f})")
            trained_models[name] = trained_m
            training_results[name] = res

    # Save training summary
    with open(EXP_DIR / "training_summary.json", "w") as f:
        summary_to_save = {
            k: {
                "best_epoch": v["best_epoch"],
                "best_val_auc": v["best_val_auc"],
            }
            for k, v in training_results.items()
        }
        json.dump(summary_to_save, f, indent=2)

    # 3. Validation Threshold Calibration
    print("=" * 80)
    print("VALIDATION THRESHOLD CALIBRATION")
    print("=" * 80)
    model_thresholds = {}
    val_predictions = {}

    for name, model in trained_models.items():
        v_probs, v_targets = predict_model(model, val_data)
        val_predictions[name] = v_probs
        calib = calibrate_thresholds(v_probs, v_targets)
        model_thresholds[name] = calib
        print(f"{name:25s} | Val AUC: {roc_auc_score(v_targets, v_probs):.4f} | Max F1 Thresh: {calib['max_f1_threshold']:.3f} (F1: {calib['val_best_f1']:.4f}) | Screening Thresh: {calib['screening_threshold']:.3f}")

    # Also load existing checkpoints for comparison
    print("-" * 80)
    print("EVALUATING EXISTING CHECKPOINTS (ViT, Multimodal, M6)")
    # Historical ViT:
    vit_probs_test = None
    mm_probs_test = None
    m6_probs_test = None

    # Load test dual scale for M6
    ds_test = torch.load(DUAL_CACHE_DIR / "dual_scale_test.pt", map_location="cpu")
    # ViT classifier on visual
    from transformers import ViTForImageClassification
    vit_model = ViTForImageClassification.from_pretrained("models/classification/vit_model").to(DEVICE)
    vit_model.eval()
    with torch.no_grad():
        vit_logits = vit_model.classifier(vis_test.to(DEVICE))
        vit_probs_test = torch.softmax(vit_logits, dim=-1)[:, 1].cpu().numpy()

    # Multimodal Phase 1
    mm_model = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2).to(DEVICE)
    mm_model.load_state_dict(torch.load("models/multimodal/fusion_model.pt", map_location=DEVICE))
    mm_model.eval()
    with torch.no_grad():
        mm_cache_test = torch.load(CACHE_DIR / "test_features.pt", map_location="cpu")
        mm_logits = mm_model(mm_cache_test["visual"].to(DEVICE), mm_cache_test["clinical"].to(DEVICE))
        mm_probs_test = torch.softmax(mm_logits, dim=-1)[:, 1].cpu().numpy()

    # M6 Dual Scale
    m6_model = M6_DualGated(
        global_dim=768,
        local_dim=768,
        clinical_dim=16,
        hidden_dim=128,
        clinical_hidden=64,
        num_classes=2,
        dropout=0.15,
    ).to(DEVICE)
    m6_model.load_state_dict(torch.load("models/experiments/M6/weights.pt", map_location=DEVICE))
    m6_model.eval()
    with torch.no_grad():
        m6_logits = m6_model(
            ds_test["global_visual"].to(DEVICE),
            ds_test["local_visual"].to(DEVICE),
            ds_test["clinical"].to(DEVICE),
        )
        m6_probs_test = torch.softmax(m6_logits, dim=-1)[:, 1].cpu().numpy()

    # 4. Official Test Evaluation & Comparison Table
    print("=" * 80)
    print("OFFICIAL UNBIASED TEST EVALUATION & LESION-CLUSTER BOOTSTRAP")
    print("=" * 80)

    y_test_np = y_test.numpy()
    comparison_rows = []
    test_probs_dict = {}

    # Evaluate new models A through F at Max F1 threshold and Screening threshold
    for name, model in trained_models.items():
        t_probs, _ = predict_model(model, test_data)
        test_probs_dict[name] = t_probs
        calib = model_thresholds[name]

        for mode, thresh in [("Max_F1", calib["max_f1_threshold"]), ("Screening", calib["screening_threshold"])]:
            m_metrics = compute_metrics(y_test_np, t_probs, thresh)
            l_metrics = compute_lesion_level_metrics(df_test, t_probs, thresh)
            cis = cluster_bootstrap_ci(df_test, t_probs, thresh, n_iterations=1000)

            row = {
                "Model": name,
                "Operating_Mode": mode,
                "Threshold": round(thresh, 4),
                "Accuracy": round(m_metrics["accuracy"], 4),
                "Precision": round(m_metrics["precision"], 4),
                "Sensitivity": round(m_metrics["sensitivity"], 4),
                "Sensitivity_95CI": f"[{cis['sensitivity_ci'][0]:.4f}, {cis['sensitivity_ci'][1]:.4f}]",
                "Specificity": round(m_metrics["specificity"], 4),
                "Specificity_95CI": f"[{cis['specificity_ci'][0]:.4f}, {cis['specificity_ci'][1]:.4f}]",
                "F1_Score": round(m_metrics["f1"], 4),
                "Balanced_Accuracy": round(m_metrics["balanced_accuracy"], 4),
                "BalAcc_95CI": f"[{cis['balanced_accuracy_ci'][0]:.4f}, {cis['balanced_accuracy_ci'][1]:.4f}]",
                "ROC_AUC": round(m_metrics["roc_auc"], 4),
                "ROC_AUC_95CI": f"[{cis['roc_auc_ci'][0]:.4f}, {cis['roc_auc_ci'][1]:.4f}]",
                "Lesion_Sensitivity": round(l_metrics["lesion_sensitivity"], 4),
                "Lesion_Sens_95CI": f"[{cis['lesion_sensitivity_ci'][0]:.4f}, {cis['lesion_sensitivity_ci'][1]:.4f}]",
                "Lesion_Specificity": round(l_metrics["lesion_specificity"], 4),
                "Lesion_Spec_95CI": f"[{cis['lesion_specificity_ci'][0]:.4f}, {cis['lesion_specificity_ci'][1]:.4f}]",
                "Lesion_BalAcc": round(l_metrics["lesion_balanced_accuracy"], 4),
                "Lesion_ROC_AUC": round(l_metrics["lesion_roc_auc"], 4),
                "TP": m_metrics["tp"],
                "FP": m_metrics["fp"],
                "TN": m_metrics["tn"],
                "FN": m_metrics["fn"],
                "Lesion_TP": l_metrics["lesion_tp"],
                "Lesion_FP": l_metrics["lesion_fp"],
                "Lesion_TN": l_metrics["lesion_tn"],
                "Lesion_FN": l_metrics["lesion_fn"],
            }
            comparison_rows.append(row)
            print(f"[{name}] {mode:10s} (t={thresh:.3f}) | AUC: {m_metrics['roc_auc']:.4f} | Sens: {m_metrics['sensitivity']:.4f} | Spec: {m_metrics['specificity']:.4f} | BalAcc: {m_metrics['balanced_accuracy']:.4f} | Lesion Sens: {l_metrics['lesion_sensitivity']:.4f}")

    # Also add ViT, Multimodal, and M6 for authoritative comparison
    historical_benchmarks = [
        ("ViT_Baseline_Prod", vit_probs_test, 0.800, "Production"),
        ("ViT_Baseline_Calib", vit_probs_test, 0.730, "Calibrated"),
        ("Phase1_Multimodal_Prod", mm_probs_test, 0.360, "Production"),
        ("Phase1_Multimodal_Calib", mm_probs_test, 0.020, "Calibrated"),
        ("M6_Challenger_Review", m6_probs_test, 0.690, "Review"),
        ("M6_Challenger_Screening", m6_probs_test, 0.060, "Screening"),
    ]

    for b_name, b_probs, b_thresh, b_mode in historical_benchmarks:
        m_metrics = compute_metrics(y_test_np, b_probs, b_thresh)
        l_metrics = compute_lesion_level_metrics(df_test, b_probs, b_thresh)
        cis = cluster_bootstrap_ci(df_test, b_probs, b_thresh, n_iterations=1000)
        row = {
            "Model": b_name,
            "Operating_Mode": b_mode,
            "Threshold": round(b_thresh, 4),
            "Accuracy": round(m_metrics["accuracy"], 4),
            "Precision": round(m_metrics["precision"], 4),
            "Sensitivity": round(m_metrics["sensitivity"], 4),
            "Sensitivity_95CI": f"[{cis['sensitivity_ci'][0]:.4f}, {cis['sensitivity_ci'][1]:.4f}]",
            "Specificity": round(m_metrics["specificity"], 4),
            "Specificity_95CI": f"[{cis['specificity_ci'][0]:.4f}, {cis['specificity_ci'][1]:.4f}]",
            "F1_Score": round(m_metrics["f1"], 4),
            "Balanced_Accuracy": round(m_metrics["balanced_accuracy"], 4),
            "BalAcc_95CI": f"[{cis['balanced_accuracy_ci'][0]:.4f}, {cis['balanced_accuracy_ci'][1]:.4f}]",
            "ROC_AUC": round(m_metrics["roc_auc"], 4),
            "ROC_AUC_95CI": f"[{cis['roc_auc_ci'][0]:.4f}, {cis['roc_auc_ci'][1]:.4f}]",
            "Lesion_Sensitivity": round(l_metrics["lesion_sensitivity"], 4),
            "Lesion_Sens_95CI": f"[{cis['lesion_sensitivity_ci'][0]:.4f}, {cis['lesion_sensitivity_ci'][1]:.4f}]",
            "Lesion_Specificity": round(l_metrics["lesion_specificity"], 4),
            "Lesion_Spec_95CI": f"[{cis['lesion_specificity_ci'][0]:.4f}, {cis['lesion_specificity_ci'][1]:.4f}]",
            "Lesion_BalAcc": round(l_metrics["lesion_balanced_accuracy"], 4),
            "Lesion_ROC_AUC": round(l_metrics["lesion_roc_auc"], 4),
            "TP": m_metrics["tp"],
            "FP": m_metrics["fp"],
            "TN": m_metrics["tn"],
            "FN": m_metrics["fn"],
            "Lesion_TP": l_metrics["lesion_tp"],
            "Lesion_FP": l_metrics["lesion_fp"],
            "Lesion_TN": l_metrics["lesion_tn"],
            "Lesion_FN": l_metrics["lesion_fn"],
        }
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(REPORTS_DIR / "CEFM_STEP16_MODEL_COMPARISON.csv", index=False)
    df_comparison.to_csv(OUTPUTS_DIR / "CEFM_STEP16_MODEL_COMPARISON.csv", index=False)
    print(f"Saved comparison table to {REPORTS_DIR / 'CEFM_STEP16_MODEL_COMPARISON.csv'}")

    # 5. Forensic Error Analysis on 27 Jointly Missed Melanoma Cases
    print("=" * 80)
    print("FORENSIC ERROR ANALYSIS: 27 JOINTLY MISSED MELANOMA CASES")
    print("=" * 80)
    df_missed = pd.read_csv(MISSED_MEL_CSV)
    print(f"Loaded {len(df_missed)} jointly missed melanoma records from Step 13.")

    # Match each image_id in df_missed with its index in df_test
    test_id_to_idx = {img_id: idx for idx, img_id in enumerate(df_test["image_id"])}

    p_model_f = test_probs_dict["ExpF_FullProposedFusion"]
    p_model_b = test_probs_dict["ExpB_ClinicalOnly"]
    p_model_a = test_probs_dict["ExpA_VisualOnly"]
    f_max_f1_t = model_thresholds["ExpF_FullProposedFusion"]["max_f1_threshold"]
    f_screen_t = model_thresholds["ExpF_FullProposedFusion"]["screening_threshold"]

    missed_analysis_rows = []
    for _, row in df_missed.iterrows():
        img_id = row["image_id"]
        t_idx = test_id_to_idx.get(img_id)
        if t_idx is None:
            continue

        prob_f = float(p_model_f[t_idx])
        prob_b = float(p_model_b[t_idx])
        prob_a = float(p_model_a[t_idx])
        prob_vit = float(vit_probs_test[t_idx])
        prob_mm = float(mm_probs_test[t_idx])
        prob_m6 = float(m6_probs_test[t_idx])

        detected_f_max_f1 = bool(prob_f >= f_max_f1_t)
        detected_f_screen = bool(prob_f >= f_screen_t)

        out_row = dict(row)
        out_row["ModelF_probability"] = round(prob_f, 4)
        out_row["ClinicalOnly_probability"] = round(prob_b, 4)
        out_row["VisualOnly_probability"] = round(prob_a, 4)
        out_row["ModelF_detected_MaxF1"] = detected_f_max_f1
        out_row["ModelF_detected_Screening"] = detected_f_screen
        out_row["Delta_Prob_ModelF_minus_ViT"] = round(prob_f - prob_vit, 4)
        out_row["Delta_Prob_ModelF_minus_MM"] = round(prob_f - prob_mm, 4)
        out_row["Delta_Prob_ModelF_minus_M6"] = round(prob_f - prob_m6, 4)
        missed_analysis_rows.append(out_row)

    df_missed_analysis = pd.DataFrame(missed_analysis_rows)
    df_missed_analysis.to_csv(REPORTS_DIR / "CEFM_STEP16_MISSED_CASE_ANALYSIS.csv", index=False)
    df_missed_analysis.to_csv(OUTPUTS_DIR / "CEFM_STEP16_MISSED_CASE_ANALYSIS.csv", index=False)
    print(f"Saved missed case analysis to {REPORTS_DIR / 'CEFM_STEP16_MISSED_CASE_ANALYSIS.csv'}")

    n_recovered_max_f1 = sum(r["ModelF_detected_MaxF1"] for r in missed_analysis_rows)
    n_recovered_screen = sum(r["ModelF_detected_Screening"] for r in missed_analysis_rows)
    print(f"Recovered melanomas out of 27: Max F1 threshold ({f_max_f1_t:.3f}) = {n_recovered_max_f1}/27 | Screening threshold ({f_screen_t:.3f}) = {n_recovered_screen}/27")

    print("=" * 80)
    print("STEP 16 EXPERIMENTAL RUN COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
