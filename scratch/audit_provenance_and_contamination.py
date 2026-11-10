"""
Step 11 Forensic Audit: Checkpoint Training Provenance and Contamination Analysis.
Analyzes exact image and lesion IDs for:
1. ViT Classifier
2. Multimodal Fusion
3. M6 Challenger
Against:
- New lesion-disjoint test.csv (N=1,504, unique lesions=1,121)
- New lesion-disjoint validation.csv (N=1,533, unique lesions=1,121)
"""

import json
from pathlib import Path
import pandas as pd
import torch

REPO_ROOT = Path("D:/pytest_cache")
META_PATH = REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_metadata.csv"
NEW_SPLIT_DIR = REPO_ROOT / "datasets" / "Ham10000" / "processed_lesion_disjoint"
ORIG_SPLIT_DIR = REPO_ROOT / "datasets" / "Ham10000" / "processed"
DUAL_SCALE_DIR = REPO_ROOT / "data" / "dual_scale_cache"
MULTIMODAL_DIR = REPO_ROOT / "data" / "multimodal_cache"

meta_df = pd.read_csv(META_PATH)
img_to_lesion = dict(zip(meta_df["image_id"], meta_df["lesion_id"]))

# Load New Partitions
new_test = pd.read_csv(NEW_SPLIT_DIR / "test.csv")
new_val = pd.read_csv(NEW_SPLIT_DIR / "validation.csv")
new_train = pd.read_csv(NEW_SPLIT_DIR / "train.csv")

new_test_imgs = set(new_test["image_id"])
new_test_lesions = set(new_test["lesion_id"])

new_val_imgs = set(new_val["image_id"])
new_val_lesions = set(new_val["lesion_id"])

new_train_imgs = set(new_train["image_id"])
new_train_lesions = set(new_train["lesion_id"])

# Load Historical Partitions
orig_train = pd.read_csv(ORIG_SPLIT_DIR / "train.csv")
orig_val = pd.read_csv(ORIG_SPLIT_DIR / "validation.csv")
orig_test = pd.read_csv(ORIG_SPLIT_DIR / "test.csv")

orig_train_imgs = set(orig_train["image_id"])
orig_train_lesions = set(orig_train["lesion_id"])

orig_val_imgs = set(orig_val["image_id"])
orig_val_lesions = set(orig_val["lesion_id"])

orig_test_imgs = set(orig_test["image_id"])
orig_test_lesions = set(orig_test["lesion_id"])

# Load Cached Training IDs (from dual_scale_cache)
ds_train = torch.load(DUAL_SCALE_DIR / "dual_scale_train.pt", map_location="cpu")
ds_train_imgs = set(ds_train["image_ids"])
ds_train_lesions = set(ds_train["lesion_ids"])

ds_val = torch.load(DUAL_SCALE_DIR / "dual_scale_validation.pt", map_location="cpu")
ds_val_imgs = set(ds_val["image_ids"])
ds_val_lesions = set(ds_val["lesion_ids"])

ds_test = torch.load(DUAL_SCALE_DIR / "dual_scale_test.pt", map_location="cpu")
ds_test_imgs = set(ds_test["image_ids"])
ds_test_lesions = set(ds_test["lesion_ids"])

print("=" * 80)
print("1. PROVENANCE VERIFICATION ACROSS ARTIFACTS")
print("=" * 80)
print(f"Total HAM10000 images: {len(meta_df)}, Total unique lesions: {meta_df['lesion_id'].nunique()}")
print(f"New Lesion-Disjoint Split: Train={len(new_train_imgs)}, Val={len(new_val_imgs)}, Test={len(new_test_imgs)}")
print(f"Original Processed Split:   Train={len(orig_train_imgs)}, Val={len(orig_val_imgs)}, Test={len(orig_test_imgs)}")
print(f"Dual-Scale Cache Split:     Train={len(ds_train_imgs)}, Val={len(ds_val_imgs)}, Test={len(ds_test_imgs)}")

print(f"\nMatch between New Train and Dual-Scale Cache Train:")
print(f"  Image sets identical: {new_train_imgs == ds_train_imgs}")
print(f"  Lesion sets identical: {new_train_lesions == ds_train_lesions}")

print("\n" + "=" * 80)
print("2. CONTAMINATION AUDIT FOR EACH CHECKPOINT")
print("=" * 80)

def audit_split_contamination(train_source_name, train_imgs, train_lesions):
    print(f"\n--- Training Source: {train_source_name} ({len(train_imgs)} images, {len(train_lesions)} lesions) ---")
    
    # Check against New Test Set
    test_img_overlap = train_imgs & new_test_imgs
    test_les_overlap = train_lesions & new_test_lesions
    unseen_test_les = new_test_lesions - train_lesions
    unseen_test_img = new_test_imgs - train_imgs
    
    print(f"  [New Test Partition - N={len(new_test_imgs)} images, {len(new_test_lesions)} lesions]:")
    print(f"    Overlapping Test Images:  {len(test_img_overlap)} ({len(test_img_overlap)/len(new_test_imgs)*100:.2f}%)")
    print(f"    Overlapping Test Lesions: {len(test_les_overlap)} ({len(test_les_overlap)/len(new_test_lesions)*100:.2f}%)")
    print(f"    Genuinely Unseen Lesions: {len(unseen_test_les)} ({len(unseen_test_les)/len(new_test_lesions)*100:.2f}%)")
    print(f"    Genuinely Unseen Images:  {len(unseen_test_img)} ({len(unseen_test_img)/len(new_test_imgs)*100:.2f}%)")
    if len(test_les_overlap) == 0:
        print("    >> TEST SET STATUS: 100% UNCONTAMINATED (GENUINELY INDEPENDENT)")
    else:
        print("    >> TEST SET STATUS: CONTAMINATED (LESION LEAKAGE DETECTED)")
        
    # Check against New Validation Set
    val_img_overlap = train_imgs & new_val_imgs
    val_les_overlap = train_lesions & new_val_lesions
    unseen_val_les = new_val_lesions - train_lesions
    unseen_val_img = new_val_imgs - train_imgs
    
    print(f"  [New Validation Partition - N={len(new_val_imgs)} images, {len(new_val_lesions)} lesions]:")
    print(f"    Overlapping Val Images:   {len(val_img_overlap)} ({len(val_img_overlap)/len(new_val_imgs)*100:.2f}%)")
    print(f"    Overlapping Val Lesions:  {len(val_les_overlap)} ({len(val_les_overlap)/len(new_val_lesions)*100:.2f}%)")
    print(f"    Genuinely Unseen Lesions: {len(unseen_val_les)} ({len(unseen_val_les)/len(new_val_lesions)*100:.2f}%)")
    print(f"    Genuinely Unseen Images:  {len(unseen_val_img)} ({len(unseen_val_img)/len(new_val_imgs)*100:.2f}%)")
    if len(val_les_overlap) == 0:
        print("    >> VALIDATION SET STATUS: 100% UNCONTAMINATED (SUITABLE FOR CALIBRATION)")
    else:
        print("    >> VALIDATION SET STATUS: CONTAMINATED")

# 1. Models trained on the authoritative 6,978 split (M6, Multimodal Fusion, ViT production)
audit_split_contamination("Phase 2 Authoritative Split (N=6,978 images, dual_scale_train.pt)", ds_train_imgs, ds_train_lesions)

# 2. What if a model was trained on orig processed/train.csv (N=7,010 images)?
audit_split_contamination("Historical Naive Image Split (N=7,010 images, processed/train.csv)", orig_train_imgs, orig_train_lesions)

print("\n" + "=" * 80)
print("3. EXACT TRAINING PROVENANCE EVIDENCE FOR EACH MODEL")
print("=" * 80)

# A. ViT Classifier Checkpoint Evidence
with open(REPO_ROOT / "models" / "classification" / "threshold_config.json") as f:
    vit_thresh_cfg = json.load(f)
vit_val_recs = vit_thresh_cfg["sweep_records"][0]
vit_val_total = vit_val_recs["tp"] + vit_val_recs["fp"] + vit_val_recs["tn"] + vit_val_recs["fn"]
vit_val_pos = vit_val_recs["tp"] + vit_val_recs["fn"]
vit_val_neg = vit_val_recs["fp"] + vit_val_recs["tn"]
print(f"ViT Validation Tuning Records in threshold_config.json:")
print(f"  Total validation samples: {vit_val_total} (Matches N=1,533 exactly: {vit_val_total == 1533})")
print(f"  Melanoma samples:         {vit_val_pos}   (Matches N=164 exactly: {vit_val_pos == 164})")
print(f"  Non-melanoma samples:     {vit_val_neg}  (Matches N=1,369 exactly: {vit_val_neg == 1369})")

# B. Multimodal Fusion Checkpoint Evidence
with open(REPO_ROOT / "models" / "multimodal" / "fusion_config.json") as f:
    mm_cfg = json.load(f)
with open(REPO_ROOT / "models" / "multimodal" / "multimodal_evaluation_results.json") as f:
    mm_eval = json.load(f)
print(f"\nMultimodal Fusion Metadata:")
print(f"  Config: {mm_cfg}")
print(f"  Evaluation dataset: {mm_eval['dataset']}, sample_count: {mm_eval['sample_count']} (Matches N=1,504 exactly: {mm_eval['sample_count'] == 1504})")

# C. Challenger M6 Checkpoint Evidence
with open(REPO_ROOT / "models" / "experiments" / "M6" / "config.json") as f:
    m6_cfg = json.load(f)
print(f"\nChallenger M6 Metadata:")
print(f"  Weights SHA256: {m6_cfg['weights_sha256']}")
print(f"  Best Val AUC:   {m6_cfg['best_val_auc']}")

print("\n" + "=" * 80)
print("4. MULTIMODAL CACHE ALIGNMENT AUDIT")
print("=" * 80)
mm_tr = torch.load(MULTIMODAL_DIR / "train_features.pt", map_location="cpu")
mm_va = torch.load(MULTIMODAL_DIR / "validation_features.pt", map_location="cpu")
mm_ts = torch.load(MULTIMODAL_DIR / "test_features.pt", map_location="cpu")

print("multimodal_cache keys:")
print("  train_features.pt:     ", list(mm_tr.keys()))
print("  validation_features.pt:", list(mm_va.keys()))
print("  test_features.pt:      ", list(mm_ts.keys()))

print("dual_scale_cache keys:")
print("  dual_scale_train.pt:   ", list(ds_train.keys()))

print(f"\nExact Label Alignment between multimodal_cache and dual_scale_cache:")
print(f"  Train labels match: {torch.equal(mm_tr['labels'], ds_train['labels'])}")
print(f"  Val labels match:   {torch.equal(mm_va['labels'], ds_val['labels'])}")
print(f"  Test labels match:  {torch.equal(mm_ts['labels'], ds_test['labels'])}")

print(f"\nLabel Alignment between multimodal_cache and new lesion_disjoint CSVs:")
print(f"  Train labels match CSV: {torch.equal(mm_tr['labels'], torch.tensor(new_train['label'].values, dtype=torch.long))}")
print(f"  Val labels match CSV:   {torch.equal(mm_va['labels'], torch.tensor(new_val['label'].values, dtype=torch.long))}")
print(f"  Test labels match CSV:  {torch.equal(mm_ts['labels'], torch.tensor(new_test['label'].values, dtype=torch.long))}")

print(f"\nExact Visual Feature Parity between multimodal_cache and ViT extraction:")
# Check if visual feature in multimodal cache matches dual_scale_cache or raw ViT
cos_sim = torch.nn.functional.cosine_similarity(mm_tr['visual'], ds_train['global_visual'], dim=-1)
print(f"  Cosine similarity mean: {cos_sim.mean().item():.4f}, min: {cos_sim.min().item():.4f}, max: {cos_sim.max().item():.4f}")
