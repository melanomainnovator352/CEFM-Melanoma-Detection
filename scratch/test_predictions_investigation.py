import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path

REPO_ROOT = Path("D:/pytest_cache")
sys.path.insert(0, str(REPO_ROOT))

from app.services.classification import classify
from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.abcde.composite import compute_abcde_score, adapt_abcde_features

# Load test dataset
test_csv_path = REPO_ROOT / "datasets" / "Ham10000" / "processed" / "test.csv"
test_df = pd.read_csv(test_csv_path)

img_dirs = [
    REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_images_part_1",
    REPO_ROOT / "datasets" / "Ham10000" / "HAM10000_images_part_2",
]

def find_img(img_id):
    clean = str(img_id).replace(".jpg", "")
    for d in img_dirs:
        p = d / f"{clean}.jpg"
        if p.exists():
            return p
    raise FileNotFoundError(f"{img_id} not found")

# Select cases
samples = []
for dx in ['mel', 'nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df']:
    sub = test_df[test_df['dx'] == dx]
    n = 6 if dx == 'mel' else (4 if dx == 'nv' else 2)
    samples.append(sub.head(n))

cohort_df = pd.concat(samples, ignore_index=True)
print(f"Total cohort samples: {len(cohort_df)}")

results = []
for idx, row in cohort_df.iterrows():
    img_id = row['image_id']
    gt_dx = row['dx']
    gt_mel = 1 if gt_dx == 'mel' else 0
    img_path = find_img(img_id)
    
    # Read image in RGB
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Segmentation
    mask = segment_lesion(str(img_path))
    
    # 2. ABCDE Features
    abcde_raw = extract_abcde_features(img_rgb, mask)
    color = abcde_raw["color"]
    diameter = abcde_raw["diameter"]
    flat = {
        "asymmetry": float(abcde_raw["asymmetry"]),
        "border_irregularity": float(abcde_raw["border_irregularity"]),
        "color_h": float(color.get("h", 0.0)),
        "color_s": float(color.get("s", 0.0)),
        "color_v": float(color.get("v", 0.0)),
        "color_index": float(color.get("index", 0.0)),
        "diameter_pixels": float(abcde_raw["diameter_pixels"]),
        "diameter_mm": float(diameter.get("max_diameter_mm")) if diameter.get("max_diameter_mm") is not None else None,
        "diameter_mm_available": bool(diameter.get("available", False)),
    }
    abcde_risk = compute_abcde_score(flat)
    abcde_val = abcde_risk.get("score")
    abcde_label = abcde_risk.get("label", "unknown")
    
    # 3. Classification with full multimodal and challenger
    clf_res = classify(str(img_path), features=flat, mask=mask)
    
    vit_prob = clf_res.get('melanoma_probability', 0.0)
    vit_thresh = clf_res.get('decision_threshold', 0.80)
    vit_pred = clf_res.get('predicted_class', 'unknown')
    
    mm_prob = clf_res.get('multimodal_melanoma_probability')
    mm_thresh = clf_res.get('multimodal_decision_threshold')
    mm_pred = clf_res.get('multimodal_predicted_class')
    
    m6_prob = clf_res.get('challenger_melanoma_probability')
    m6_thresh = clf_res.get('challenger_decision_threshold')
    m6_pred = clf_res.get('challenger_predicted_class')
    
    results.append({
        'image_id': img_id,
        'gt_dx': gt_dx,
        'gt_mel': gt_mel,
        'vit_prob': float(vit_prob),
        'vit_thresh': float(vit_thresh),
        'vit_pred': vit_pred,
        'mm_prob': float(mm_prob) if mm_prob is not None else None,
        'mm_thresh': float(mm_thresh) if mm_thresh is not None else None,
        'mm_pred': mm_pred,
        'm6_prob': float(m6_prob) if m6_prob is not None else None,
        'm6_thresh': float(m6_thresh) if m6_thresh is not None else None,
        'm6_pred': m6_pred,
        'abcde_score': float(abcde_val) if abcde_val is not None else None,
        'abcde_label': abcde_label,
    })

res_df = pd.DataFrame(results)

print("\n" + "="*110)
print(f"{'Image ID':13s} | {'GT':5s} | {'ViT Prob':8s} {'ViT Pred':8s} | {'MM Prob':8s} {'MM Pred':8s} | {'M6 Prob':8s} {'M6 Pred':8s} | {'ABCDE':15s} | {'ViT Match'}")
print("="*110)
for _, r in res_df.iterrows():
    vit_match = "CORRECT" if (r['gt_mel'] == 1 and r['vit_pred'] == 'mel') or (r['gt_mel'] == 0 and r['vit_pred'] == 'non_mel') else "MISMATCH"
    mm_p_str = f"{r['mm_prob']:.4f}" if r['mm_prob'] is not None else "N/A"
    m6_p_str = f"{r['m6_prob']:.4f}" if r['m6_prob'] is not None else "N/A"
    abcde_str = f"{r['abcde_score']:.3f} ({r['abcde_label']})" if r['abcde_score'] is not None else "N/A"
    print(f"{r['image_id']:13s} | {r['gt_dx']:5s} | {r['vit_prob']:8.4f} {r['vit_pred']:8s} | {mm_p_str:8s} {str(r['mm_pred']):8s} | {m6_p_str:8s} {str(r['m6_pred']):8s} | {abcde_str:15s} | {vit_match}")

print("\n" + "="*110)
print("ANALYSIS OF MELANOMA CASES (Ground Truth = MEL):")
mel_cases = res_df[res_df['gt_mel'] == 1]
for _, r in mel_cases.iterrows():
    print(f"Case {r['image_id']}: ViT P={r['vit_prob']:.4f} (Thresh={r['vit_thresh']:.2f}) -> Pred={r['vit_pred']} | MM P={r['mm_prob']} (Thresh={r['mm_thresh']}) -> Pred={r['mm_pred']} | M6 P={r['m6_prob']} -> Pred={r['m6_pred']} | ABCDE={r['abcde_score']} ({r['abcde_label']})")
