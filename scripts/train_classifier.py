"""
Train the melanoma classifier on HAM10000 using a fine-tuned Vision Transformer.

Usage:
    python scripts/train_classifier.py --epochs 6 --batch-size 16 --binary-melanoma

Saves the fine-tuned ViT weights under models/classification/vit_model,
threshold calibration under models/classification/threshold_config.json,
and evaluation logs under models/classification/evaluation/.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

IMAGE_SIZE = 224
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_BACKBONE = "google/vit-base-patch16-224"
TARGET_ACCURACY = 0.40
DEFAULT_EPOCHS = 6
DEFAULT_LR = 2e-5
DEFAULT_HEAD_LR = 1e-4


def set_trainable_parameters(model, freeze_backbone: bool = True):
    """Set parameter trainability for head-only or full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    if freeze_backbone:
        if hasattr(model, "vit"):
            for param in model.vit.parameters():
                param.requires_grad = False
        elif hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False

    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    return model

IMAGE_DIRS = [
    Path("D:/datasets/HAM10000/HAM10000_images_part_1"),
    Path("D:/datasets/HAM10000/HAM10000_images_part_2"),
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_1",
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_2",
    BASE_DIR / "data" / "HAM10000" / "images",
]


def find_image(image_id: str) -> Path:
    """Locate an image by its image_id across known candidate directories."""
    clean_id = str(image_id).replace(".jpg", "")
    for directory in IMAGE_DIRS:
        if not directory.exists():
            continue
        path = directory / f"{clean_id}.jpg"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate image: {clean_id}.jpg")


def load_dataset_splits(val_split=0.15, seed=42):
    """
    Load predefined leak-free lesion-level splits if available;
    otherwise load metadata and perform lesion-level stratified split.
    """
    train_csv = BASE_DIR / "data" / "HAM10000" / "metadata" / "train.csv"
    val_csv = BASE_DIR / "data" / "HAM10000" / "metadata" / "validation.csv"

    if train_csv.exists() and val_csv.exists():
        print(f"Loading existing leak-free splits from {train_csv.parent}")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        train_df["path"] = train_df["image_id"].apply(find_image)
        val_df["path"] = val_df["image_id"].apply(find_image)
        return train_df, val_df

    print("Predefined split CSVs not found. Loading raw HAM10000 metadata...")
    meta_path = Path("D:/datasets/HAM10000/HAM10000_metadata.csv")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    df = pd.read_csv(meta_path)
    lookup = {}
    for d in IMAGE_DIRS:
        if d.exists():
            for p in d.glob("*.jpg"):
                lookup[p.stem] = p

    df = df[df["image_id"].isin(lookup.keys())].copy()
    df["path"] = df["image_id"].map(lookup)
    df["label"] = (df["dx"] == "mel").astype(int)

    # Group by lesion_id to guarantee zero lesion leakage
    rng = np.random.RandomState(seed)
    lesions = df.groupby("lesion_id").agg({"label": "max"}).reset_index()
    val_lesions = []
    for lab in lesions["label"].unique():
        les_list = lesions[lesions["label"] == lab]["lesion_id"].tolist()
        rng.shuffle(les_list)
        n_val = max(1, int(round(len(les_list) * val_split)))
        val_lesions.extend(les_list[:n_val])

    val_set = set(val_lesions)
    val_mask = df["lesion_id"].isin(val_set)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    return train_df, val_df


class HAMDataset(Dataset):
    """PyTorch Dataset with dermoscopy-specific data augmentations."""

    def __init__(self, frame, augment=True):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=45),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        if self.augment and self.transform is not None:
            img = self.transform(img)

        return img, int(row["label"])


def make_balanced_sampler(labels, num_classes=2, samples_per_epoch=None):
    """Create a WeightedRandomSampler that samples classes evenly."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.max(counts) / np.clip(counts, 1.0, None)
    sample_weights = weights[labels]

    num_samples = int(samples_per_epoch) if samples_per_epoch is not None else int(len(sample_weights))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def build_model(num_classes=2, class_names=None):
    """Load pretrained Vision Transformer and replace the classification head."""
    class_names = class_names or ["non_mel", "mel"]
    label2id = {label: idx for idx, label in enumerate(class_names)}
    id2label = {idx: label for idx, label in enumerate(class_names)}

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        model = ViTForImageClassification.from_pretrained(
            MODEL_BACKBONE,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    except Exception:
        model = ViTForImageClassification.from_pretrained(
            MODEL_BACKBONE,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    model.config.problem_type = "single_label_classification"
    return model


def calibrate_thresholds(y_true, y_prob):
    """
    Sweep decision thresholds on validation data to identify:
    1. Threshold maximizing F1 score
    2. Threshold maximizing Balanced Accuracy
    3. Clinical threshold prioritizing Sensitivity >= 80% while maximizing Specificity
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.arange(0.10, 0.92, 0.02)
    records = []

    best_f1 = -1.0
    best_f1_thresh = 0.50

    best_bacc = -1.0
    best_bacc_thresh = 0.50

    clinical_thresh = 0.50
    best_clinical_spec = -1.0

    for t in thresholds:
        t = round(float(t), 2)
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        bacc = 0.5 * (sens + spec)
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        rec = {
            "threshold": t,
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "balanced_accuracy": round(float(bacc), 4),
            "false_negative_rate": round(float(fnr), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        records.append(rec)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t

        if bacc > best_bacc:
            best_bacc = bacc
            best_bacc_thresh = t

        # Clinical operating point: Sensitivity >= 80% with highest specificity
        if sens >= 0.79:
            if spec > best_clinical_spec:
                best_clinical_spec = spec
                clinical_thresh = t

    return {
        "best_f1_threshold": best_f1_thresh,
        "best_f1_score": round(float(best_f1), 4),
        "best_bacc_threshold": best_bacc_thresh,
        "best_bacc_score": round(float(best_bacc), 4),
        "clinical_threshold": clinical_thresh,
        "clinical_sensitivity": round(float(next((r["sensitivity"] for r in records if r["threshold"] == clinical_thresh), 0.8)), 4),
        "clinical_specificity": round(float(next((r["specificity"] for r in records if r["threshold"] == clinical_thresh), 0.7)), 4),
        "sweep_records": records,
    }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("CEFM — Vision Transformer Melanoma Classifier Training")
    print("=" * 70)

    train_df, val_df = load_dataset_splits(val_split=args.val_split, seed=args.seed)

    if args.binary_melanoma:
        class_names = ["non_mel", "mel"]
    else:
        class_names = HAM_CLASSES

    print(f"Train set: {len(train_df)} samples | Non-Mel: {(train_df['label']==0).sum()}, Mel: {(train_df['label']==1).sum()}")
    print(f"Val set:   {len(val_df)} samples | Non-Mel: {(val_df['label']==0).sum()}, Mel: {(val_df['label']==1).sum()}")

    processor = ViTImageProcessor.from_pretrained(MODEL_BACKBONE, local_files_only=True)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    if args.balanced_sampling:
        # Balanced sampling: equal chance of picking melanoma vs non-melanoma
        train_sampler = make_balanced_sampler(
            train_df["label"].to_numpy(),
            num_classes=len(class_names),
            samples_per_epoch=len(train_df) if args.full_epoch_samples else (2 * (train_df["label"] == 1).sum()),
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        HAMDataset(train_df, augment=True),
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        HAMDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(num_classes=len(class_names), class_names=class_names).to(device)

    # Optimizer with differential learning rates:
    # Gentle LR for pretrained transformer backbone, higher LR for classifier head
    if args.freeze_backbone:
        print("Backbone frozen: training classifier head only.")
        for param in model.vit.parameters():
            param.requires_grad = False
        param_groups = [{"params": model.classifier.parameters(), "lr": args.head_lr}]
    else:
        print("Full fine-tuning with differential learning rates:")
        print(f"  Backbone LR: {args.lr} | Classifier Head LR: {args.head_lr} | Weight Decay: 0.01")
        param_groups = [
            {"params": model.vit.parameters(), "lr": args.lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_training_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if args.balanced_sampling:
        # Unweighted loss because sampler already balances class exposure
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # Smooth square-root class weighting
        counts = np.bincount(train_df["label"].to_numpy(), minlength=2).astype(np.float32)
        weights = np.sqrt(np.max(counts) / np.clip(counts, 1.0, None))
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    out_dir = BASE_DIR / "models" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "vit_model"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_bacc": [],
        "val_auc": [],
        "val_pr_auc": [],
    }

    best_val_auc = 0.0
    best_threshold_calibration = None

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_steps and batch_idx >= args.max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(**batch)
                logits = outputs.logits
                loss = criterion(logits, batch["labels"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * len(batch["labels"])
            correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            seen += len(batch["labels"])

        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        # Validation phase
        model.eval()
        v_total_loss, v_seen = 0.0, 0
        val_y_true, val_y_prob = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = criterion(logits, batch["labels"])

                v_total_loss += loss.item() * len(batch["labels"])
                v_seen += len(batch["labels"])
                probs = torch.softmax(logits, dim=-1)

                val_y_true.extend(batch["labels"].cpu().tolist())
                val_y_prob.extend(probs[:, 1].cpu().tolist())

        val_loss = v_total_loss / max(v_seen, 1)
        val_y_true = np.asarray(val_y_true)
        val_y_prob = np.asarray(val_y_prob)

        # Metrics at 0.5 baseline
        val_acc = accuracy_score(val_y_true, (val_y_prob >= 0.5).astype(int))
        val_bacc = balanced_accuracy_score(val_y_true, (val_y_prob >= 0.5).astype(int))
        try:
            val_auc = roc_auc_score(val_y_true, val_y_prob)
            val_pr_auc = average_precision_score(val_y_true, val_y_prob)
        except Exception:
            val_auc = 0.5
            val_pr_auc = 0.1

        history["train_loss"].append(round(float(train_loss), 4))
        history["train_acc"].append(round(float(train_acc), 4))
        history["val_loss"].append(round(float(val_loss), 4))
        history["val_acc"].append(round(float(val_acc), 4))
        history["val_bacc"].append(round(float(val_bacc), 4))
        history["val_auc"].append(round(float(val_auc), 4))
        history["val_pr_auc"].append(round(float(val_pr_auc), 4))

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} BAcc: {val_bacc:.4f} "
            f"ROC-AUC: {val_auc:.4f} PR-AUC: {val_pr_auc:.4f}"
        )

        # Calibrate thresholds on validation data
        thresh_info = calibrate_thresholds(val_y_true, val_y_prob)

        # Save checkpoint if ROC-AUC is new best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold_calibration = thresh_info
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(
                f"  --> Saved new best checkpoint (ROC-AUC={val_auc:.4f}, Best F1-Thresh={thresh_info['best_f1_threshold']} with F1={thresh_info['best_f1_score']})"
            )

    # Save training history and threshold configuration
    history_path = eval_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if best_threshold_calibration is not None:
        thresh_file = out_dir / "threshold_config.json"
        with open(thresh_file, "w", encoding="utf-8") as f:
            json.dump(best_threshold_calibration, f, indent=2)

        calib_file = eval_dir / "threshold_calibration.json"
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump(best_threshold_calibration, f, indent=2)

        print("\nThreshold Calibration on Validation Data:")
        print(f"  Best F1 Threshold:       {best_threshold_calibration['best_f1_threshold']} (F1: {best_threshold_calibration['best_f1_score']})")
        print(f"  Best BAcc Threshold:     {best_threshold_calibration['best_bacc_threshold']} (BAcc: {best_threshold_calibration['best_bacc_score']})")
        print(f"  Clinical Threshold:      {best_threshold_calibration['clinical_threshold']} (Sens: {best_threshold_calibration['clinical_sensitivity']}, Spec: {best_threshold_calibration['clinical_specificity']})")

    print("\n" + "=" * 70)
    print(f"Training completed. Best Validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-epoch-samples", action="store_true", default=False)
    parser.add_argument("--binary-melanoma", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
