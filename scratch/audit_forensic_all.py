import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch

REPO_ROOT = Path("D:/pytest_cache")
sys.path.insert(0, str(REPO_ROOT))

ham_dir = REPO_ROOT / "datasets" / "Ham10000"
isic_dir = REPO_ROOT / "datasets" / "ISIC2018"
processed_dir = ham_dir / "processed"

print("="*80)
print("SECTION A & B: DATASET FORENSIC AUDIT")
print("="*80)

# 1. HAM10000 Metadata & Image Counts
metadata_path = ham_dir / "HAM10000_metadata.csv"
meta_df = pd.read_csv(metadata_path)
print(f"Total rows in HAM10000_metadata.csv: {len(meta_df)}")
print(f"Columns: {list(meta_df.columns)}")
print(f"Unique image_ids in metadata: {meta_df['image_id'].nunique()}")
print(f"Unique lesion_ids in metadata: {meta_df['lesion_id'].nunique()}")

part1_dir = ham_dir / "HAM10000_images_part_1"
part2_dir = ham_dir / "HAM10000_images_part_2"

part1_files = list(part1_dir.glob("*.jpg"))
part2_files = list(part2_dir.glob("*.jpg"))
print(f"HAM10000_images_part_1 *.jpg count: {len(part1_files)}")
print(f"HAM10000_images_part_2 *.jpg count: {len(part2_files)}")
total_jpg = len(part1_files) + len(part2_files)
print(f"Total HAM10000 *.jpg images: {total_jpg}")

# Non-jpg files if any
all_p1 = list(part1_dir.iterdir())
all_p2 = list(part2_dir.iterdir())
print(f"Total items in part 1: {len(all_p1)} (non-jpg: {len(all_p1) - len(part1_files)})")
print(f"Total items in part 2: {len(all_p2)} (non-jpg: {len(all_p2) - len(part2_files)})")

# Check for missing images
image_lookup = {}
for p in part1_files:
    image_lookup[p.stem] = p
for p in part2_files:
    image_lookup[p.stem] = p

missing_images = [img_id for img_id in meta_df['image_id'] if img_id not in image_lookup]
print(f"Missing images (metadata vs disk): {len(missing_images)}")

# Check duplicate filenames
p1_stems = set(p.stem for p in part1_files)
p2_stems = set(p.stem for p in part2_files)
overlap_stems = p1_stems.intersection(p2_stems)
print(f"Overlap between part 1 and part 2: {len(overlap_stems)}")

# Overall HAM10000 Diagnosis distribution
print("\nHAM10000 Diagnosis (dx) Distribution:")
dx_counts = meta_df['dx'].value_counts()
for dx, count in dx_counts.items():
    print(f"  {dx:6s}: {count:5d} ({count/len(meta_df)*100:5.2f}%)")

# Binary distribution: mel vs non-mel
mel_count = (meta_df['dx'] == 'mel').sum()
non_mel_count = len(meta_df) - mel_count
print(f"Binary Melanoma Distribution: MEL={mel_count} ({mel_count/len(meta_df)*100:.2f}%), NON_MEL={non_mel_count} ({non_mel_count/len(meta_df)*100:.2f}%)")
print(f"Imbalance ratio: {non_mel_count/mel_count:.2f} : 1")

# Splits in processed/
train_df = pd.read_csv(processed_dir / "train.csv")
val_df = pd.read_csv(processed_dir / "validation.csv")
test_df = pd.read_csv(processed_dir / "test.csv")
print(f"\nSplit Sizes:")
print(f"  Train:      {len(train_df):5d} ({len(train_df)/len(meta_df)*100:5.2f}%)")
print(f"  Validation: {len(val_df):5d} ({len(val_df)/len(meta_df)*100:5.2f}%)")
print(f"  Test:       {len(test_df):5d} ({len(test_df)/len(meta_df)*100:5.2f}%)")
print(f"  Sum:        {len(train_df) + len(val_df) + len(test_df)}")

# Verify patient/lesion leakage between splits
train_lesions = set(train_df['lesion_id'])
val_lesions = set(val_df['lesion_id'])
test_lesions = set(test_df['lesion_id'])
print(f"Lesion ID Overlap Train-Val:  {len(train_lesions.intersection(val_lesions))}")
print(f"Lesion ID Overlap Train-Test: {len(train_lesions.intersection(test_lesions))}")
print(f"Lesion ID Overlap Val-Test:   {len(val_lesions.intersection(test_lesions))}")

train_imgs = set(train_df['image_id'])
val_imgs = set(val_df['image_id'])
test_imgs = set(test_df['image_id'])
print(f"Image ID Overlap Train-Val:   {len(train_imgs.intersection(val_imgs))}")
print(f"Image ID Overlap Train-Test:  {len(train_imgs.intersection(test_imgs))}")
print(f"Image ID Overlap Val-Test:    {len(val_imgs.intersection(test_imgs))}")

# Split Class Breakdown
for split_name, s_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    print(f"\n{split_name} Diagnosis Breakdown:")
    for dx in ['mel', 'nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df']:
        c = (s_df['dx'] == dx).sum()
        print(f"  {dx:6s}: {c:5d} ({c/len(s_df)*100:5.2f}%)")
    mel_c = (s_df['dx'] == 'mel').sum()
    print(f"  Binary: MEL={mel_c} ({mel_c/len(s_df)*100:.2f}%), NON_MEL={len(s_df)-mel_c} ({(len(s_df)-mel_c)/len(s_df)*100:.2f}%)")

# Check HMNIST CSVs
print("\nHMNIST CSV Analysis:")
for csv_name in ['hmnist_8_8_L.csv', 'hmnist_8_8_RGB.csv', 'hmnist_28_28_L.csv', 'hmnist_28_28_RGB.csv']:
    p = ham_dir / csv_name
    if p.exists():
        with open(p, 'r') as f:
            header = f.readline().strip().split(',')
            num_cols = len(header)
        # Count lines
        line_count = sum(1 for _ in open(p, 'r')) - 1
        print(f"  {csv_name:20s}: {line_count:6d} rows, {num_cols:5d} columns (header starts: {header[:4]}... last: {header[-1]})")

# Check ISIC 2018 Task 1
print("\nISIC 2018 Task 1 Integrity Analysis:")
isic_train_img_dir = isic_dir / "ISIC2018_Task1-2_Training_Input"
isic_train_mask_dir = isic_dir / "ISIC2018_Task1_Training_GroundTruth"
isic_val_img_dir = isic_dir / "ISIC2018_Task1-2_Validation_Input"
isic_test_img_dir = isic_dir / "ISIC2018_Task1-2_Test_Input"

isic_train_imgs = sorted(list(isic_train_img_dir.glob("*.jpg")))
isic_train_masks = sorted(list(isic_train_mask_dir.glob("*.png")))
print(f"  ISIC2018 Task 1-2 Training Images: {len(isic_train_imgs)}")
print(f"  ISIC2018 Task 1 Training Masks:    {len(isic_train_masks)}")
print(f"  ISIC2018 Validation Input Images:  {len(list(isic_val_img_dir.glob('*.jpg')))}")
print(f"  ISIC2018 Test Input Images:        {len(list(isic_test_img_dir.glob('*.jpg')))}")

# Verify 1-to-1 matching between training images and masks
img_bases = {p.stem: p for p in isic_train_imgs}
mask_bases = {p.stem.replace("_segmentation", ""): p for p in isic_train_masks}
unmatched_imgs = set(img_bases.keys()) - set(mask_bases.keys())
unmatched_masks = set(mask_bases.keys()) - set(img_bases.keys())
print(f"  Unmatched training images without mask: {len(unmatched_imgs)}")
print(f"  Unmatched training masks without image: {len(unmatched_masks)}")

# Check sample mask values and dimensions
sample_mask = cv2.imread(str(isic_train_masks[0]), cv2.IMREAD_GRAYSCALE)
print(f"  Sample mask shape: {sample_mask.shape}, unique values: {np.unique(sample_mask)}")

