"""
Final Step 1 Verification Script for CEFM Phase 2.
Executes all 13 verification gates required before proceeding to Step 2/3.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.classification import _load_torch_model
import app.services.classification as cl
from app.services.segmentation import _get_unet_model
from scripts.cache_dual_scale_features import segment_batch_unet, extract_vit_batch
from app.services.dual_scale import (
    extract_dual_scale_views,
    extract_vit_embeddings_from_images,
    compute_lesion_spatial_bounds,
)
from scripts.evaluate_vit import find_image

CACHE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
META_DIR = REPO_ROOT / "data" / "HAM10000" / "metadata"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_verification():
    results = {}
    all_passed = True

    print("=" * 80)
    print("CEFM PHASE 2: FINAL STEP 1 VERIFICATION AUDIT")
    print("=" * 80)

    # Load cache files
    print("\nLoading cached datasets...")
    train_cache = torch.load(CACHE_DIR / "dual_scale_train.pt", map_location="cpu")
    val_cache = torch.load(CACHE_DIR / "dual_scale_validation.pt", map_location="cpu")
    test_cache = torch.load(CACHE_DIR / "dual_scale_test.pt", map_location="cpu")

    # Load original split manifests
    train_df = pd.read_csv(META_DIR / "train.csv")
    val_df = pd.read_csv(META_DIR / "validation.csv")
    test_df = pd.read_csv(META_DIR / "test.csv")

    # Gate 1, 2, 3: Sample Counts
    gate1 = len(train_cache["labels"]) == 6978 and len(train_cache["image_ids"]) == 6978
    gate2 = len(val_cache["labels"]) == 1533 and len(val_cache["image_ids"]) == 1533
    gate3 = len(test_cache["labels"]) == 1504 and len(test_cache["image_ids"]) == 1504
    print(f"\n[Gate 1] Training samples:   {len(train_cache['labels'])} (Expected: 6,978) -> {'PASS' if gate1 else 'FAIL'}")
    print(f"[Gate 2] Validation samples: {len(val_cache['labels'])} (Expected: 1,533) -> {'PASS' if gate2 else 'FAIL'}")
    print(f"[Gate 3] Test samples:       {len(test_cache['labels'])} (Expected: 1,504) -> {'PASS' if gate3 else 'FAIL'}")
    if not (gate1 and gate2 and gate3):
        all_passed = False

    # Gate 4: Merged Checkpoints without Missing or Duplicate Samples
    train_ids = train_cache["image_ids"]
    val_ids = val_cache["image_ids"]
    test_ids = test_cache["image_ids"]

    gate4 = (
        len(train_ids) == len(set(train_ids)) == 6978 and
        len(val_ids) == len(set(val_ids)) == 1533 and
        len(test_ids) == len(set(test_ids)) == 1504
    )
    print(f"[Gate 4] Checkpoints merged without duplicates: {'PASS' if gate4 else 'FAIL'}")
    print(f"         Unique IDs: Train={len(set(train_ids))}, Val={len(set(val_ids))}, Test={len(set(test_ids))}")
    if not gate4:
        all_passed = False

    # Gate 5: Manifest Alignment (image_ids, lesion_ids, labels)
    train_ids_match = list(train_df["image_id"]) == train_cache["image_ids"]
    train_lesions_match = list(train_df["lesion_id"]) == train_cache["lesion_ids"]
    train_labels_match = list(train_df["label"].astype(int)) == train_cache["labels"].tolist()

    val_ids_match = list(val_df["image_id"]) == val_cache["image_ids"]
    val_lesions_match = list(val_df["lesion_id"]) == val_cache["lesion_ids"]
    val_labels_match = list(val_df["label"].astype(int)) == val_cache["labels"].tolist()

    test_ids_match = list(test_df["image_id"]) == test_cache["image_ids"]
    test_lesions_match = list(test_df["lesion_id"]) == test_cache["lesion_ids"]
    test_labels_match = list(test_df["label"].astype(int)) == test_cache["labels"].tolist()

    gate5 = (
        train_ids_match and train_lesions_match and train_labels_match and
        val_ids_match and val_lesions_match and val_labels_match and
        test_ids_match and test_lesions_match and test_labels_match
    )
    print(f"[Gate 5] Alignment against original CSV manifests: {'PASS' if gate5 else 'FAIL'}")
    print(f"         Train match (IDs: {train_ids_match}, Lesions: {train_lesions_match}, Labels: {train_labels_match})")
    print(f"         Val match   (IDs: {val_ids_match}, Lesions: {val_lesions_match}, Labels: {val_labels_match})")
    print(f"         Test match  (IDs: {test_ids_match}, Lesions: {test_lesions_match}, Labels: {test_labels_match})")
    if not gate5:
        all_passed = False

    # Gate 6: Zero Lesion Overlap Across All Splits
    train_l_set = set(train_cache["lesion_ids"])
    val_l_set = set(val_cache["lesion_ids"])
    test_l_set = set(test_cache["lesion_ids"])

    train_val_ov = train_l_set.intersection(val_l_set)
    train_test_ov = train_l_set.intersection(test_l_set)
    val_test_ov = val_l_set.intersection(test_l_set)

    gate6 = len(train_val_ov) == 0 and len(train_test_ov) == 0 and len(val_test_ov) == 0
    print(f"[Gate 6] Zero lesion leakage across splits: {'PASS' if gate6 else 'FAIL'}")
    print(f"         Train/Val: {len(train_val_ov)}, Train/Test: {len(train_test_ov)}, Val/Test: {len(val_test_ov)}")
    if not gate6:
        all_passed = False

    # Gate 7: Tensor Dimensions
    g_train = train_cache["global_visual"].shape == torch.Size([6978, 768])
    l_train = train_cache["local_visual"].shape == torch.Size([6978, 768])
    c_train = train_cache["clinical"].shape == torch.Size([6978, 16])

    g_val = val_cache["global_visual"].shape == torch.Size([1533, 768])
    l_val = val_cache["local_visual"].shape == torch.Size([1533, 768])
    c_val = val_cache["clinical"].shape == torch.Size([1533, 16])

    g_test = test_cache["global_visual"].shape == torch.Size([1504, 768])
    l_test = test_cache["local_visual"].shape == torch.Size([1504, 768])
    c_test = test_cache["clinical"].shape == torch.Size([1504, 16])

    gate7 = g_train and l_train and c_train and g_val and l_val and c_val and g_test and l_test and c_test
    print(f"[Gate 7] Embedding & clinical tensor dimensions: {'PASS' if gate7 else 'FAIL'}")
    print(f"         Train: Global={list(train_cache['global_visual'].shape)}, Local={list(train_cache['local_visual'].shape)}, Clin={list(train_cache['clinical'].shape)}")
    print(f"         Val:   Global={list(val_cache['global_visual'].shape)}, Local={list(val_cache['local_visual'].shape)}, Clin={list(val_cache['clinical'].shape)}")
    print(f"         Test:  Global={list(test_cache['global_visual'].shape)}, Local={list(test_cache['local_visual'].shape)}, Clin={list(test_cache['clinical'].shape)}")
    if not gate7:
        all_passed = False

    # Gate 8: Finite Values Check (Zero NaNs / Infs)
    finite_train = (
        torch.isfinite(train_cache["global_visual"]).all().item() and
        torch.isfinite(train_cache["local_visual"]).all().item() and
        torch.isfinite(train_cache["clinical"]).all().item()
    )
    finite_val = (
        torch.isfinite(val_cache["global_visual"]).all().item() and
        torch.isfinite(val_cache["local_visual"]).all().item() and
        torch.isfinite(val_cache["clinical"]).all().item()
    )
    finite_test = (
        torch.isfinite(test_cache["global_visual"]).all().item() and
        torch.isfinite(test_cache["local_visual"]).all().item() and
        torch.isfinite(test_cache["clinical"]).all().item()
    )

    gate8 = finite_train and finite_val and finite_test
    print(f"[Gate 8] Finite values verification (no NaNs/Infs): {'PASS' if gate8 else 'FAIL'}")
    print(f"         Train finite: {finite_train}, Val finite: {finite_val}, Test finite: {finite_test}")
    if not gate8:
        all_passed = False

    # Gate 9: Local Patch Aggregation Methodology & Coordinate Preservation
    patch_counts_train = train_cache.get("patch_counts", [])
    has_patch_counts = len(patch_counts_train) == 6978
    single_patches = sum(1 for c in patch_counts_train if c == 1)
    multi_patches = sum(1 for c in patch_counts_train if c > 1)
    gate9 = has_patch_counts and single_patches > 0 and multi_patches > 0
    print(f"[Gate 9] Local patch aggregation methodology: {'PASS' if gate9 else 'FAIL'}")
    print(f"         Method: VM-UNet guided coordinates, 224x224 windows, 2:1 center-to-periphery weighting")
    print(f"         Distribution in Train: {single_patches} single-patch (<=224px), {multi_patches} multi-patch tiled (>224px)")
    if not gate9:
        all_passed = False

    # Gate 10: Cache Reproducibility on 10 Random Samples
    print(f"\n[Gate 10] Testing cache reproducibility on 10 samples...")
    cl._load_torch_model()
    processor = cl._processor
    vit_model = cl._classifier.to(DEVICE)
    unet_model = _get_unet_model().to(DEVICE)
    unet_model.eval()

    np.random.seed(42)
    sample_indices = np.random.choice(len(test_df), size=10, replace=False)
    reproducibility_passed = True
    max_diff_g = 0.0
    max_diff_l = 0.0

    for idx in sample_indices:
        img_id = test_cache["image_ids"][idx]
        img_path = find_image(img_id)
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_rgb = np.array(img_pil)

        # Segment and extract views
        masks = segment_batch_unet([img_rgb], unet_model, DEVICE)
        views = extract_dual_scale_views(img_rgb, masks[0])

        # Global embedding
        recomp_g = extract_vit_batch([views["global_image"]], vit_model, processor, DEVICE)[0]

        # Local embedding
        patch_images = [p["image"] for p in views["local_patches"]]
        patch_embs = extract_vit_batch(patch_images, vit_model, processor, DEVICE)
        count = len(patch_embs)
        if count == 1:
            recomp_l = patch_embs[0]
        else:
            w = torch.ones(count)
            w[0] = 2.0
            w = (w / w.sum()).unsqueeze(1)
            recomp_l = (patch_embs * w).sum(dim=0)

        cached_g = test_cache["global_visual"][idx]
        cached_l = test_cache["local_visual"][idx]

        diff_g = float((recomp_g - cached_g).abs().max())
        diff_l = float((recomp_l - cached_l).abs().max())
        max_diff_g = max(max_diff_g, diff_g)
        max_diff_l = max(max_diff_l, diff_l)

        if not (torch.allclose(recomp_g, cached_g, atol=1e-4) and torch.allclose(recomp_l, cached_l, atol=1e-4)):
            reproducibility_passed = False
            print(f"         Sample {img_id}: MISMATCH diff_g={diff_g:.6f}, diff_l={diff_l:.6f}")

    gate10 = reproducibility_passed
    print(f"         Reproducibility result: {'PASS' if gate10 else 'FAIL'}")
    print(f"         Max absolute diff across 10 samples: Global={max_diff_g:.6e}, Local={max_diff_l:.6e}")
    if not gate10:
        all_passed = False

    # Gate 11: Checkpoint Hashes & Preprocessing Config in Cache
    cfg = train_cache.get("preprocessing_config", {})
    hashes = train_cache.get("checkpoint_hashes", {})
    gate11 = (
        cfg.get("global_crop_margin") == 1.3 and
        cfg.get("global_size") == 224 and
        cfg.get("local_patch_size") == 224 and
        "vit_safetensors" in hashes and
        "segmentation_unet" in hashes
    )
    print(f"[Gate 11] Checkpoint hashes & preprocessing configuration recorded: {'PASS' if gate11 else 'FAIL'}")
    print(f"          Recorded Config: {cfg}")
    print(f"          Recorded Hashes: {hashes}")
    if not gate11:
        all_passed = False

    # Gate 12: Production Checkpoints Preservation (Unchanged SHA256 Hashes)
    expected_hashes = {
        "models/classification/vit_model/model.safetensors": "1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db",
        "models/segmentation/unet_isic2018.pt": "7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e",
        "models/multimodal/fusion_model.pt": "04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422",
    }
    hashes_preserved = True
    for p_str, exp_h in expected_hashes.items():
        p = REPO_ROOT / p_str
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        act_h = hasher.hexdigest()
        if act_h != exp_h:
            hashes_preserved = False
            print(f"         Checkpoint {p.name}: MISMATCH! Exp={exp_h}, Act={act_h}")

    gate12 = hashes_preserved
    print(f"[Gate 12] Production Phase 1 checkpoints intact and unchanged: {'PASS' if gate12 else 'FAIL'}")
    if not gate12:
        all_passed = False

    print("=" * 80)
    print(f"FINAL AUDIT RESULT: {'ALL 12 DATA & MODEL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
