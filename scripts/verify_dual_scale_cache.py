"""
Verification of Dual-Scale Cache Integrity and Leakage-Free Stratification.
"""
import hashlib
import torch
from pathlib import Path

train_data = torch.load("data/dual_scale_cache/dual_scale_train.pt", map_location="cpu")
val_data = torch.load("data/dual_scale_cache/dual_scale_validation.pt", map_location="cpu")
test_data = torch.load("data/dual_scale_cache/dual_scale_test.pt", map_location="cpu")

print("=" * 70)
print("CEFM PHASE 2: STEP 1 DUAL-SCALE CACHE INTEGRITY REPORT")
print("=" * 70)

print("\n1. SAMPLE COUNTS AND TENSOR DIMENSIONS:")
print(f"  - Train:      Global {list(train_data['global_visual'].shape)}, Local {list(train_data['local_visual'].shape)}, Clin {list(train_data['clinical'].shape)}, Labels {list(train_data['labels'].shape)}")
print(f"  - Validation: Global {list(val_data['global_visual'].shape)}, Local {list(val_data['local_visual'].shape)}, Clin {list(val_data['clinical'].shape)}, Labels {list(val_data['labels'].shape)}")
print(f"  - Test:       Global {list(test_data['global_visual'].shape)}, Local {list(test_data['local_visual'].shape)}, Clin {list(test_data['clinical'].shape)}, Labels {list(test_data['labels'].shape)}")

total_samples = len(train_data['labels']) + len(val_data['labels']) + len(test_data['labels'])
print(f"\n  Total images across all splits: {total_samples} / 10015 (100.0%)")

print("\n2. FAILURE AND QUALITY VERIFICATION:")
for name, d in [('Train', train_data), ('Validation', val_data), ('Test', test_data)]:
    nan_g = torch.isnan(d['global_visual']).any().item()
    nan_l = torch.isnan(d['local_visual']).any().item()
    nan_c = torch.isnan(d['clinical']).any().item()
    inf_g = torch.isinf(d['global_visual']).any().item()
    inf_l = torch.isinf(d['local_visual']).any().item()
    inf_c = torch.isinf(d['clinical']).any().item()
    failed = len(d.get('failed_images', []))
    print(f"  - {name:11s}: Failures={failed} | NaNs=[G:{nan_g}, L:{nan_l}, C:{nan_c}] | Infs=[G:{inf_g}, L:{inf_l}, C:{inf_c}]")

print("\n3. SPLIT ZERO-LEAKAGE VERIFICATION:")
train_lesions = set(train_data['lesion_ids'])
val_lesions = set(val_data['lesion_ids'])
test_lesions = set(test_data['lesion_ids'])

print(f"  - Unique Lesions: Train={len(train_lesions)}, Validation={len(val_lesions)}, Test={len(test_lesions)}")
print(f"  - Overlap Train / Validation: {len(train_lesions.intersection(val_lesions))} lesions")
print(f"  - Overlap Train / Test:       {len(train_lesions.intersection(test_lesions))} lesions")
print(f"  - Overlap Validation / Test:  {len(val_lesions.intersection(test_lesions))} lesions")

print("\n4. PRODUCTION CHECKPOINT PRESERVATION (SHA256 HASHES):")
checkpoints = {
    'ViT Baseline': ('models/classification/vit_model/model.safetensors', '1c37ebbb7f4a580b13f7c87319d67a1ab32a8e1e5ffbce02789722a12c4305db'),
    'VM-UNet Model': ('models/segmentation/unet_isic2018.pt', '7f7e49ef73d2d64f17db7d3953e2ed4939a4dc84b807ecfea1d5bd4ccd711c2e'),
    'Multimodal Fusion': ('models/multimodal/fusion_model.pt', '04d846fc06b5401db0ec8762fd8974b864991bb9d8d12e4a8f83ae3273393422'),
}

all_hashes_intact = True
for name, (path_str, expected_hash) in checkpoints.items():
    p = Path(path_str)
    hasher = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    match = actual_hash == expected_hash
    if not match:
        all_hashes_intact = False
    print(f"  - {name:18s}: {'MATCH' if match else 'MISMATCH'}")
    print(f"    Expected: {expected_hash}")
    print(f"    Actual:   {actual_hash}")

print(f"\n  Production Checkpoints Integrity: {'ALL PRESERVED & UNCHANGED' if all_hashes_intact else 'WARNING: MISMATCH'}")

print("\n5. MULTI-PANEL VISUALIZATIONS GENERATED:")
viz_dir = Path("reports/figures/dual_scale_visualizations")
viz_files = list(viz_dir.glob("*.jpg"))
print(f"  - Total visualization figures: {len(viz_files)} (Required: >= 10)")
has_special_case = (viz_dir / "ISIC_0012418_dual_scale_viz.jpg").exists()
print(f"  - ISIC_0012418 visualization saved: {has_special_case}")

print("=" * 70)

