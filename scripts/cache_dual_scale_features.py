"""
High-Performance Dual-Scale Feature Extraction & Caching Engine (Phase 2).

Extracts:
1. Global Visual Representation (768-D ViT CLS embedding, 1.30x context crop at 224x224).
2. Local High-Resolution Representation (768-D ViT CLS embedding, native-scale lesion core patch(es) at 224x224).
3. 16-D Clinical Feature Vector.
4. Spatial Coordinates, Patch Counts, and Checkpoint Hashes.

Optimizations:
- Batched UNet segmentation on CUDA (batch size 32)
- Batched ViT feature extraction on CUDA (batch size 32)
- Incremental chunk checkpointing every 500 samples with automatic resume
- Verification of zero-leakage lesion-level partitioning
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.dual_scale import (
    compute_lesion_spatial_bounds,
    extract_global_crop,
    extract_local_patches,
    extract_dual_scale_views,
)
from app.services.segmentation import _get_unet_model
from app.services.calibration import estimate_pixels_per_mm
from scripts.evaluate_vit import find_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_MODEL_DIR = REPO_ROOT / "models" / "classification" / "vit_model"
OUTPUT_CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
VIZ_OUTPUT_DIR = REPO_ROOT / "reports" / "figures" / "dual_scale_visualizations"


def get_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_partition_independence() -> Dict[str, Any]:
    """Verify lesion-level zero leakage across train, val, and test splits."""
    train_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "train.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "validation.csv")
    test_df = pd.read_csv(REPO_ROOT / "data" / "HAM10000" / "metadata" / "test.csv")

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    overlap_train_val = len(train_lesions.intersection(val_lesions))
    overlap_train_test = len(train_lesions.intersection(test_lesions))
    overlap_val_test = len(val_lesions.intersection(test_lesions))

    assert overlap_train_val == 0, f"Leakage detected between Train and Val: {overlap_train_val}"
    assert overlap_train_test == 0, f"Leakage detected between Train and Test: {overlap_train_test}"
    assert overlap_val_test == 0, f"Leakage detected between Val and Test: {overlap_val_test}"

    print(f"[Split Verification] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"[Split Verification] Unique Lesions: Train={len(train_lesions)}, Val={len(val_lesions)}, Test={len(test_lesions)}")
    print("[Split Verification] Lesion-level zero-leakage verified across all splits.")

    return {
        "train_len": len(train_df),
        "val_len": len(val_df),
        "test_len": len(test_df),
        "train_lesions": len(train_lesions),
        "val_lesions": len(val_lesions),
        "test_lesions": len(test_lesions),
    }


def save_patch_visualization(
    img_rgb: np.ndarray,
    views: Dict[str, Any],
    image_id: str,
    output_dir: Path,
):
    """Save multi-panel visualization showing bounding box and extracted dual-scale views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = img_rgb.copy()
    gb = views["global_bbox"]
    cv2.rectangle(annotated, (gb["xmin"], gb["ymin"]), (gb["xmax"], gb["ymax"]), (0, 120, 255), 3)

    for p in views["local_patches"]:
        b = p["bbox"]
        cv2.rectangle(annotated, (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"]), (0, 255, 0), 2)
        cv2.putText(annotated, f"P{p['patch_id']}", (b["xmin"] + 4, b["ymin"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    h_target = 300
    scale = h_target / annotated.shape[0]
    w_ann = int(annotated.shape[1] * scale)
    ann_resized = cv2.resize(annotated, (w_ann, h_target))
    g_resized = cv2.resize(views["global_image"], (h_target, h_target))
    l_resized = cv2.resize(views["local_patches"][0]["image"], (h_target, h_target))

    def add_title(img, title):
        banner = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return np.vstack([banner, img])

    panel1 = add_title(ann_resized, f"{image_id} (Full Image + BBoxes)")
    panel2 = add_title(g_resized, "Global Context Crop (1.30x)")
    panel3 = add_title(l_resized, f"Local Native Patch (N={views['num_local_patches']})")

    composite = np.hstack([panel1, panel2, panel3])
    out_path = output_dir / f"{image_id}_dual_scale_viz.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


def segment_batch_unet(
    images_rgb: List[np.ndarray],
    unet_model: nn.Module,
    device: torch.device,
) -> List[np.ndarray]:
    """Segment a batch of RGB images using UNet on GPU."""
    unet_model.eval()
    tensors = []
    shapes = []

    for img in images_rgb:
        h, w = img.shape[:2]
        shapes.append((h, w))
        resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensors.append(tensor)

    batch_t = torch.stack(tensors).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(unet_model(batch_t)).squeeze(1).cpu().numpy()

    masks = []
    for pred, (h, w) in zip(preds, shapes):
        resized_pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (resized_pred >= 0.70).astype(np.uint8)
        if not np.any(mask > 0):
            mask = (resized_pred >= 0.50).astype(np.uint8)
        masks.append(mask)

    return masks


def extract_vit_batch(
    images_rgb: List[np.ndarray],
    vit_model: nn.Module,
    processor: Any,
    device: torch.device,
) -> torch.Tensor:
    """Extract ViT CLS embeddings for a batch of RGB images."""
    vit_model.eval()
    inputs = processor(images=images_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if hasattr(vit_model, "vit"):
            outputs = vit_model.vit(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = vit_model(**inputs, output_hidden_states=True)
            cls_emb = outputs.hidden_states[-1][:, 0, :]

    return cls_emb.cpu()


def process_split_batched(
    split_name: str,
    vit_model: nn.Module,
    processor: Any,
    unet_model: nn.Module,
    batch_size: int = 32,
    chunk_interval: int = 500,
    save_viz_count: int = 4,
) -> Dict[str, Any]:
    """Process an entire split using GPU batching and incremental checkpointing."""
    OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_CACHE_DIR / f"dual_scale_{split_name}.pt"
    chunk_file = OUTPUT_CACHE_DIR / f"dual_scale_{split_name}_checkpoint.pt"

    if out_file.exists():
        print(f"Loading existing dual-scale cache from {out_file}...")
        data = torch.load(out_file, map_location="cpu")
        print(f"Loaded {split_name}: {data['global_visual'].shape[0]} samples.")
        return data

    csv_path = REPO_ROOT / "data" / "HAM10000" / "metadata" / f"{split_name}.csv"
    df = pd.read_csv(csv_path)
    print(f"\n--- Processing Split: {split_name.upper()} ({len(df)} samples) ---")

    # Load pre-aligned 16-D clinical vectors
    existing_cache_file = REPO_ROOT / "data" / "multimodal_cache" / f"{split_name}_features.pt"
    existing_clinical = None
    if existing_cache_file.exists():
        exist_data = torch.load(existing_cache_file, map_location="cpu")
        existing_clinical = exist_data["clinical"]  # [N, 16]
        print(f"Reusing aligned 16-D clinical features from {existing_cache_file.name}")

    checkpoint_hashes = {
        "vit_safetensors": get_file_sha256(VIT_MODEL_DIR / "model.safetensors"),
        "segmentation_unet": get_file_sha256(REPO_ROOT / "models" / "segmentation" / "unet_isic2018.pt"),
        "production_fusion": get_file_sha256(REPO_ROOT / "models" / "multimodal" / "fusion_model.pt"),
    }

    # Resume from checkpoint if present
    global_features: List[torch.Tensor] = []
    local_features: List[torch.Tensor] = []
    patch_counts: List[int] = []
    image_ids: List[str] = []
    lesion_ids: List[str] = []
    labels: List[int] = []
    failed_images: List[Dict[str, str]] = []

    processed_ids = set()
    if chunk_file.exists():
        try:
            chunk = torch.load(chunk_file, map_location="cpu")
            global_features = list(chunk["global_visual"])
            local_features = list(chunk["local_visual"])
            patch_counts = list(chunk["patch_counts"].numpy())
            image_ids = list(chunk["image_ids"])
            lesion_ids = list(chunk["lesion_ids"])
            labels = list(chunk["labels"].numpy())
            failed_images = list(chunk.get("failed_images", []))
            processed_ids = set(image_ids)
            print(f"Resuming {split_name} from checkpoint: {len(image_ids)} samples already processed.")
        except Exception as exc:
            print(f"Could not read chunk file: {exc}")

    # Remaining rows to process
    rows_to_process = [row for _, row in df.iterrows() if str(row["image_id"]) not in processed_ids]
    viz_saved = 0
    t0 = time.time()

    # Process in batches
    for b_start in tqdm(range(0, len(rows_to_process), batch_size), desc=f"Dual-Scale Batch [{split_name}]"):
        batch_rows = rows_to_process[b_start : b_start + batch_size]
        batch_images = []
        valid_rows = []

        for row in batch_rows:
            img_id = str(row["image_id"])
            try:
                img_path = find_image(img_id)
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    failed_images.append({"image_id": img_id, "reason": "cv2.imread failed"})
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                batch_images.append(img_rgb)
                valid_rows.append(row)
            except Exception as e:
                failed_images.append({"image_id": img_id, "reason": str(e)})

        if not batch_images:
            continue

        # 1. Segment batch
        masks = segment_batch_unet(batch_images, unet_model, DEVICE)

        # 2. Extract dual-scale crops
        global_crops = []
        local_crops_flat = []
        sample_patch_counts = []

        for img_rgb, mask, row in zip(batch_images, masks, valid_rows):
            views = extract_dual_scale_views(img_rgb, mask, patch_size=224)
            global_crops.append(views["global_image"])
            
            p_imgs = [p["image"] for p in views["local_patches"]]
            local_crops_flat.extend(p_imgs)
            sample_patch_counts.append(len(p_imgs))

            # Save representative visualisations
            img_id = str(row["image_id"])
            lbl = int(row["label"])
            if viz_saved < save_viz_count or (lbl == 1 and viz_saved < save_viz_count + 2):
                save_patch_visualization(img_rgb, views, img_id, VIZ_OUTPUT_DIR)
                viz_saved += 1

        # 3. ViT Forward Pass on Global Crops
        g_embs = extract_vit_batch(global_crops, vit_model, processor, DEVICE)  # [B, 768]

        # 4. ViT Forward Pass on Local Crops
        l_embs_flat = extract_vit_batch(local_crops_flat, vit_model, processor, DEVICE)  # [Total_patches, 768]

        # 5. Pool Local Embeddings per Sample
        offset = 0
        for i, count in enumerate(sample_patch_counts):
            embs = l_embs_flat[offset : offset + count]
            offset += count

            if count == 1:
                pooled = embs[0]
            else:
                w = torch.ones(count)
                w[0] = 2.0
                w = (w / w.sum()).unsqueeze(1)
                pooled = (embs * w).sum(dim=0)

            global_features.append(g_embs[i])
            local_features.append(pooled)
            patch_counts.append(count)
            row = valid_rows[i]
            image_ids.append(str(row["image_id"]))
            lesion_ids.append(str(row["lesion_id"]))
            labels.append(int(row["label"]))

        # Checkpoint save
        if len(global_features) % chunk_interval < batch_size and len(global_features) > 0:
            torch.save({
                "global_visual": torch.stack(global_features),
                "local_visual": torch.stack(local_features),
                "patch_counts": torch.tensor(patch_counts, dtype=torch.int32),
                "image_ids": image_ids,
                "lesion_ids": lesion_ids,
                "labels": torch.tensor(labels, dtype=torch.long),
                "failed_images": failed_images,
            }, chunk_file)

    elapsed = time.time() - t0
    print(f"Completed {split_name}: {len(global_features)}/{len(df)} samples in {elapsed:.1f}s. Failed: {len(failed_images)}")

    global_tensor = torch.stack(global_features)
    local_tensor = torch.stack(local_features)
    patch_counts_tensor = torch.tensor(patch_counts, dtype=torch.int32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    if existing_clinical is not None and existing_clinical.shape[0] == len(labels):
        clinical_tensor = existing_clinical
    else:
        clinical_tensor = torch.zeros((len(labels), 16), dtype=torch.float32)

    cache_payload = {
        "split": split_name,
        "image_ids": image_ids,
        "lesion_ids": lesion_ids,
        "labels": labels_tensor,
        "global_visual": global_tensor,       # [N, 768]
        "local_visual": local_tensor,         # [N, 768]
        "clinical": clinical_tensor,          # [N, 16]
        "patch_counts": patch_counts_tensor,  # [N]
        "checkpoint_hashes": checkpoint_hashes,
        "preprocessing_config": {
            "global_crop_margin": 1.30,
            "global_size": 224,
            "local_patch_size": 224,
            "local_overlap_ratio": 0.25,
            "max_patches": 5,
            "vit_processor_mean": [0.5, 0.5, 0.5],
            "vit_processor_std": [0.5, 0.5, 0.5],
        },
        "failed_images": failed_images,
    }

    torch.save(cache_payload, out_file)
    if chunk_file.exists():
        chunk_file.unlink()

    print(f"Saved {split_name} dual-scale cache to {out_file} ({out_file.stat().st_size / 1e6:.2f} MB).")
    return cache_payload


def process_case_isic12418(vit_model: nn.Module, processor: Any, unet_model: nn.Module):
    """Process ISIC_0012418.jpg and save visualization."""
    p = Path("D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg")
    if not p.exists():
        return
    img = cv2.imread(str(p))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cal = estimate_pixels_per_mm(img_rgb)
    exclude_poly = cal.get("exclude_polygon") or cal.get("corners")
    masks = segment_batch_unet([img_rgb], unet_model, DEVICE)
    views = extract_dual_scale_views(img_rgb, masks[0])
    save_patch_visualization(img_rgb, views, "ISIC_0012418", VIZ_OUTPUT_DIR)
    print(f"[Special Case] Saved ISIC_0012418 dual-scale visualization to {VIZ_OUTPUT_DIR}.")


def main():
    print("=" * 80)
    print("CEFM Phase 2: Batched Dual-Scale Feature Extraction Engine")
    print("=" * 80)

    # 1. Verify split integrity
    split_info = verify_partition_independence()

    # 2. Load frozen ViT backbone and trained UNet on CUDA
    print(f"Loading ViT backbone from {VIT_MODEL_DIR} onto {DEVICE}...")
    vit_model = ViTForImageClassification.from_pretrained(
        str(VIT_MODEL_DIR),
        local_files_only=True,
    ).to(DEVICE)
    processor = ViTImageProcessor.from_pretrained(str(VIT_MODEL_DIR), local_files_only=True)
    vit_model.eval()

    unet_model = _get_unet_model()
    assert unet_model is not None, "Failed to load UNet model!"
    unet_model = unet_model.to(DEVICE)
    unet_model.eval()

    # 3. Process Test Split (1,504 samples)
    test_data = process_split_batched("test", vit_model, processor, unet_model, batch_size=32, save_viz_count=4)

    # 4. Process Validation Split (1,533 samples)
    val_data = process_split_batched("validation", vit_model, processor, unet_model, batch_size=32, save_viz_count=3)

    # 5. Process Training Split (6,978 samples)
    train_data = process_split_batched("train", vit_model, processor, unet_model, batch_size=32, save_viz_count=3)

    # 6. Process ISIC_0012418 for visual diagnostic verification
    process_case_isic12418(vit_model, processor, unet_model)

    print("\n" + "=" * 80)
    print("DUAL-SCALE CACHING COMPLETE & VERIFIED")
    print("=" * 80)
    print(f"Test Set:       {test_data['global_visual'].shape} global, {test_data['local_visual'].shape} local")
    print(f"Validation Set: {val_data['global_visual'].shape} global, {val_data['local_visual'].shape} local")
    print(f"Training Set:   {train_data['global_visual'].shape} global, {train_data['local_visual'].shape} local")


if __name__ == "__main__":
    main()
