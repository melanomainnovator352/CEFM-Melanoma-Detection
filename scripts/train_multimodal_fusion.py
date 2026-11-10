"""
Train Multimodal Fusion Classifier (ViT Visual Embedding + ABCDE Clinical Features).
Uses the frozen fine-tuned ViT backbone to extract 768-D image features and
the validated ABCDE extraction pipeline to extract 16-D clinical feature vectors.
Preserves the existing ViT model checkpoint completely.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# Ensure repository root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.fusion import CEFMMultimodalFusion
from app.services.abcde.vector import build_abcde_vector
from app.services.analysis import extract_abcde_features
from app.services.segmentation import segment_lesion
from scripts.evaluate_vit import find_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_MODEL_DIR = REPO_ROOT / "models" / "classification" / "vit_model"
OUTPUT_DIR = REPO_ROOT / "models" / "multimodal"
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"


def extract_dataset_features(
    df: pd.DataFrame,
    vit_model: nn.Module,
    processor: ViTImageProcessor,
    split_name: str,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract 768-D ViT visual embeddings and 16-D ABCDE clinical vectors.
    Caches features to disk to accelerate repeated training experiments.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{split_name}_features.pt"

    if cache_file.exists():
        print(f"Loading cached {split_name} features from {cache_file}...")
        data = torch.load(cache_file, map_location="cpu")
        return data["visual"], data["clinical"], data["labels"]

    print(f"Extracting features for {split_name} ({len(df)} samples)...")
    vit_model.eval()

    chunk_file = CACHE_DIR / f"{split_name}_checkpoint.pt"
    visual_features = []
    clinical_features = []
    labels = []
    processed_ids = set()

    if chunk_file.exists():
        try:
            chunk_data = torch.load(chunk_file, map_location="cpu")
            visual_features = list(chunk_data["visual"])
            clinical_features = list(chunk_data["clinical"])
            labels = list(chunk_data["labels"])
            processed_ids = set(chunk_data.get("image_ids", []))
            print(f"Resuming {split_name} from checkpoint: {len(visual_features)} samples already extracted.")
        except Exception as exc:
            print(f"Could not load checkpoint: {exc}")

    # Process samples
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"ABCDE + ViT [{split_name}]"):
        image_id = str(row["image_id"])
        if image_id in processed_ids:
            continue
        label = int(row["label"])
        image_path = find_image(image_id)

        try:
            # 1. Load image
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 2. Segment lesion & extract ABCDE clinical features
            mask = segment_lesion(str(image_path))
            abcde_dict = extract_abcde_features(img_rgb, mask)
            abcde_vec = build_abcde_vector(abcde_dict).to_tensor()

            # 3. Square ROI crop for ViT
            if mask is not None and np.any(mask > 0):
                y_idx, x_idx = np.where(mask > 0)
                ymin, ymax = int(y_idx.min()), int(y_idx.max())
                xmin, xmax = int(x_idx.min()), int(x_idx.max())
                cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
                h, w = ymax - ymin, xmax - xmin
                side = int(max(h, w) * 1.30)
                c_ymin = max(0, cy - side // 2)
                c_ymax = min(img_bgr.shape[0], cy + side // 2)
                c_xmin = max(0, cx - side // 2)
                c_xmax = min(img_bgr.shape[1], cx + side // 2)
                crop_rgb = img_rgb[c_ymin:c_ymax, c_xmin:c_xmax]
            else:
                crop_rgb = img_rgb

            inputs = processor(images=crop_rgb, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                if hasattr(vit_model, "vit"):
                    out = vit_model.vit(**inputs)
                    cls_emb = out.last_hidden_state[:, 0, :].cpu()
                else:
                    out = vit_model(**inputs, output_hidden_states=True)
                    cls_emb = out.hidden_states[-1][:, 0, :].cpu()

            visual_features.append(cls_emb.squeeze(0))
            clinical_features.append(abcde_vec)
            labels.append(label)
            processed_ids.add(image_id)

            if len(visual_features) % 200 == 0:
                torch.save({
                    "visual": visual_features,
                    "clinical": clinical_features,
                    "labels": labels,
                    "image_ids": list(processed_ids),
                }, chunk_file)

        except Exception as exc:
            print(f"Warning: sample {image_id} failed: {exc}")
            continue

    visual_tensor = torch.stack(visual_features)
    clinical_tensor = torch.stack(clinical_features)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    torch.save({
        "visual": visual_tensor,
        "clinical": clinical_tensor,
        "labels": label_tensor,
    }, cache_file)
    if chunk_file.exists():
        chunk_file.unlink()
    print(f"Saved {split_name} features cache ({len(label_tensor)} samples) to {cache_file}.")

    return visual_tensor, clinical_tensor, label_tensor


def train_fusion_model(
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
):
    print("=" * 70)
    print("CEFM Multimodal Fusion Model Training")
    print("=" * 70)

    # 1. Load ViT model for feature extraction
    print(f"Loading ViT checkpoint from {VIT_MODEL_DIR}...")
    processor = ViTImageProcessor.from_pretrained(str(VIT_MODEL_DIR), local_files_only=True)
    vit_model = ViTForImageClassification.from_pretrained(
        str(VIT_MODEL_DIR), local_files_only=True, attn_implementation="eager"
    ).to(DEVICE)
    vit_model.eval()

    # 2. Load splits
    train_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "train.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "validation.csv")
    test_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "test.csv")

    test_v, test_c, test_y = extract_dataset_features(test_df, vit_model, processor, "test")
    val_v, val_c, val_y = extract_dataset_features(val_df, vit_model, processor, "validation")
    train_v, train_c, train_y = extract_dataset_features(train_df, vit_model, processor, "train")

    # Clean up ViT from GPU to make room for training
    del vit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_ds = TensorDataset(train_v, train_c, train_y)
    val_ds = TensorDataset(val_v, val_c, val_y)
    test_ds = TensorDataset(test_v, test_c, test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Build Multimodal Fusion Network
    fusion_model = CEFMMultimodalFusion(
        visual_dim=768,
        clinical_dim=16,
        hidden_dim=128,
        num_classes=2,
        dropout=0.15,
    ).to(DEVICE)

    # 4. Class weighting for imbalance
    n_neg = (train_y == 0).sum().item()
    n_pos = (train_y == 1).sum().item()
    pos_weight = n_neg / max(1, n_pos)
    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_weights = None
    best_epoch = 0

    print(f"\nTraining on {len(train_ds)} samples, validating on {len(val_ds)} samples...")
    print(f"Class imbalance: {n_neg} non-melanoma, {n_pos} melanoma (weight={pos_weight:.2f})")

    for epoch in range(1, epochs + 1):
        fusion_model.train()
        train_loss = 0.0
        for batch_v, batch_c, batch_y in train_loader:
            batch_v = batch_v.to(DEVICE)
            batch_c = batch_c.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = fusion_model(batch_v, batch_c)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        fusion_model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for batch_v, batch_c, batch_y in val_loader:
                batch_v = batch_v.to(DEVICE)
                batch_c = batch_c.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = fusion_model(batch_v, batch_c)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_ds)
        val_auc = roc_auc_score(val_targets, val_probs)
        val_pr_auc = average_precision_score(val_targets, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == best_epoch:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} (Best: {best_val_auc:.4f} @ Epoch {best_epoch})")

    # Load best weights
    fusion_model.load_state_dict(best_weights)
    fusion_model.to(DEVICE)
    fusion_model.eval()

    # Save checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = OUTPUT_DIR / "fusion_model.pt"
    torch.save(best_weights, model_save_path)
    print(f"\nSaved best multimodal fusion model to {model_save_path} (Val ROC-AUC: {best_val_auc:.4f})")

    # 5. Calibrate decision threshold on validation set
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for batch_v, batch_c, batch_y in val_loader:
            batch_v = batch_v.to(DEVICE)
            batch_c = batch_c.to(DEVICE)
            probs = torch.softmax(fusion_model(batch_v, batch_c), dim=-1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Search for clinical threshold (target sensitivity >= 75%) and max F1 threshold
    thresholds = np.linspace(0.1, 0.95, 86)
    clinical_t = 0.50
    best_f1_t = 0.50
    best_f1 = 0.0

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (val_targets == 1))
        fp = np.sum((preds == 1) & (val_targets == 0))
        fn = np.sum((preds == 0) & (val_targets == 1))
        sens = tp / max(1, (tp + fn))
        prec = tp / max(1, (tp + fp))
        f1 = 2 * prec * sens / max(1e-6, (prec + sens))

        if sens >= 0.75:
            clinical_t = float(t)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = float(t)

    config_info = {
        "model_architecture": "CEFMMultimodalFusion",
        "visual_dim": 768,
        "clinical_dim": 16,
        "hidden_dim": 128,
        "best_epoch": best_epoch,
        "val_roc_auc": round(float(best_val_auc), 4),
        "clinical_threshold": round(clinical_t, 2),
        "optimal_f1_threshold": round(best_f1_t, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    config_path = OUTPUT_DIR / "fusion_config.json"
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)
    print(f"Saved threshold configuration to {config_path}: clinical_threshold={clinical_t:.2f}, max_f1_threshold={best_f1_t:.2f}")

    # 6. Evaluate on Test Set
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for batch_v, batch_c, batch_y in test_loader:
            batch_v = batch_v.to(DEVICE)
            batch_c = batch_c.to(DEVICE)
            probs = torch.softmax(fusion_model(batch_v, batch_c), dim=-1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(batch_y.numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    test_auc = roc_auc_score(test_targets, test_probs)
    test_pr_auc = average_precision_score(test_targets, test_probs)

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS (1,504 independent samples)")
    print("=" * 70)
    print(f"ROC-AUC: {test_auc:.4f}")
    print(f"PR-AUC:  {test_pr_auc:.4f}")

    for t in [0.50, clinical_t, best_f1_t]:
        preds = (test_probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (test_targets == 1)))
        tn = int(np.sum((preds == 0) & (test_targets == 0)))
        fp = int(np.sum((preds == 1) & (test_targets == 0)))
        fn = int(np.sum((preds == 0) & (test_targets == 1)))
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        prec = tp / max(1, (tp + fp))
        f1 = 2 * prec * sens / max(1e-6, (prec + sens))
        acc = (tp + tn) / len(test_targets)
        print(f"\nAt Threshold T={t:.2f}:")
        print(f"  Sensitivity (Recall): {sens:.2%} ({tp}/{tp+fn})")
        print(f"  Specificity:          {spec:.2%} ({tn}/{tn+fp})")
        print(f"  Precision:            {prec:.2%} ({tp}/{tp+fp})")
        print(f"  F1-Score:             {f1:.2%}")
        print(f"  Accuracy:             {acc:.2%}")
        print(f"  Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return fusion_model


if __name__ == "__main__":
    train_fusion_model()
