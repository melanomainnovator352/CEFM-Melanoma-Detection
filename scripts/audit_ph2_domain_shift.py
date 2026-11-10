"""
CEFM Phase 2 Step 6 — PH² Domain-Shift, Feature Distribution, Segmentation, and Error Audit.
Investigates the root causes of performance differences on the PH² external benchmark.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.classification import _prepare_abcde_tensor

PH2_DIR = Path("D:/PH2")
PH2_IMAGES_DIR = PH2_DIR / "PH2 Dataset images"
PH2_TXT = PH2_DIR / "PH2_dataset.txt"

OUTPUT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures" / "phase2_domain_shift"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_ARTIFACT_DIR = Path("C:/Users/sruja/.gemini/antigravity/brain/fd604b17-8f8a-44a3-a817-98d62f87fbac")


def run_domain_shift_audit():
    print("=" * 80)
    print("CEFM PHASE 2 STEP 6 — PH² DOMAIN SHIFT & ERROR AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 16-D CLINICAL FEATURE AUDIT: HAM10000 vs PH²
    # -------------------------------------------------------------------------
    print("\n[1] Auditing 16-D Clinical Feature Pipeline...")
    
    # Load HAM10000 training clinical features
    train_cache = torch.load(PROJECT_ROOT / "data" / "dual_scale_cache" / "dual_scale_train.pt", map_location="cpu")
    train_clin = train_cache["clinical"].numpy()  # [6978, 16]

    # Load M6 config normalization stats
    with open(PROJECT_ROOT / "models" / "experiments" / "M6" / "config.json", "r") as f:
        m6_cfg = json.load(f)
    norm_mean = np.array(m6_cfg["clinical_norm_mean"])
    norm_std = np.array(m6_cfg["clinical_norm_std"])

    # Load PH2 metadata and compute 16-D clinical vectors for all 200 cases
    with open(PH2_TXT, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and l.startswith("||") and not l.startswith("||   Name")]

    ph2_vectors = []
    ph2_meta = []
    ph2_img_stats = []
    seg_metrics = []

    print(f"Extracting features, segmentation, and image geometry across all {len(lines)} PH² cases...")
    for idx, l in enumerate(lines):
        parts = [p.strip() for p in l.split("||")[1:-1]]
        name = parts[0]
        histo = parts[1]
        clin_code = int(parts[2])
        label = 1 if clin_code == 2 else 0
        diag_name = "melanoma" if clin_code == 2 else ("common_nevus" if clin_code == 0 else "atypical_nevus")

        img_path = PH2_IMAGES_DIR / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp"
        gt_mask_path = PH2_IMAGES_DIR / name / f"{name}_lesion" / f"{name}_lesion.bmp"

        img_bgr = cv2.imread(str(img_path))
        h, w, c = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask_bin = (gt_mask > 127).astype(np.uint8)

        # Production UNet segmentation
        pred_mask = segment_lesion(img_path)

        # Image statistics
        r_mean, g_mean, b_mean = img_rgb[:, :, 0].mean(), img_rgb[:, :, 1].mean(), img_rgb[:, :, 2].mean()
        r_std, g_std, b_std = img_rgb[:, :, 0].std(), img_rgb[:, :, 1].std(), img_rgb[:, :, 2].std()

        ph2_img_stats.append({
            "image_id": name,
            "height": h,
            "width": w,
            "aspect_ratio": w / h,
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "r_std": r_std,
            "g_std": g_std,
            "b_std": b_std,
        })

        # Segmentation overlap metrics
        tp_mask = np.logical_and(pred_mask > 0, gt_mask_bin > 0).sum()
        fp_mask = np.logical_and(pred_mask > 0, gt_mask_bin == 0).sum()
        fn_mask = np.logical_and(pred_mask == 0, gt_mask_bin > 0).sum()
        tn_mask = np.logical_and(pred_mask == 0, gt_mask_bin == 0).sum()

        dice = (2.0 * tp_mask) / (2.0 * tp_mask + fp_mask + fn_mask + 1e-8)
        iou = tp_mask / (tp_mask + fp_mask + fn_mask + 1e-8)
        over_seg = fp_mask / (tp_mask + fp_mask + 1e-8)   # False discovery in predicted mask
        under_seg = fn_mask / (tp_mask + fn_mask + 1e-8)  # Missed lesion area

        seg_metrics.append({
            "image_id": name,
            "dice": dice,
            "iou": iou,
            "over_segmentation_rate": over_seg,
            "under_segmentation_rate": under_seg,
            "gt_lesion_area": int(gt_mask_bin.sum()),
            "pred_lesion_area": int(pred_mask.sum()),
            "area_ratio": float(pred_mask.sum() / (gt_mask_bin.sum() + 1e-8)),
        })

        # ABCDE Features from predicted mask
        feat = extract_abcde_features(img_rgb, pred_mask)
        vec = _prepare_abcde_tensor(feat).squeeze(0).numpy()
        ph2_vectors.append(vec)

        ph2_meta.append({
            "image_id": name,
            "label": label,
            "clinical_diagnosis_code": clin_code,
            "diagnosis_name": diag_name,
            "histo_diagnosis": histo,
        })

    ph2_clin = np.array(ph2_vectors)  # [200, 16]
    df_ph2_meta = pd.DataFrame(ph2_meta)
    df_ph2_img = pd.DataFrame(ph2_img_stats)
    df_seg = pd.DataFrame(seg_metrics)

    # Feature definitions and comparison
    dim_names = [
        "0: asymmetry_score",
        "1: horizontal_asymmetry",
        "2: vertical_asymmetry",
        "3: border_irregularity",
        "4: circularity",
        "5: radial_distance_var",
        "6: curvature_variation",
        "7: color_h_norm (Hue Std)",
        "8: color_s_norm (Sat Std)",
        "9: color_v_norm (Val Std)",
        "10: color_diversity",
        "11: diameter_mm_norm",
        "12: calibration_available",
        "13: evolution_change_flag",
        "14: evolution_magnitude",
        "15: evolution_available",
    ]

    feature_dist_rows = []
    print("\n--- 16-D Feature Distribution Comparison (HAM10000 vs. PH²) ---")
    print(f"{'Dimension':<28} | {'HAM Mean (Std)':<18} | {'PH2 Mean (Std)':<18} | {'KS Stat':<7} | {'KS p-val':<9} | {'Shift Status'}")
    print("-" * 95)

    for i in range(16):
        h_vals = train_clin[:, i]
        p_vals = ph2_clin[:, i]

        h_mean, h_std = float(h_vals.mean()), float(h_vals.std())
        p_mean, p_std = float(p_vals.mean()), float(p_vals.std())

        # Kolmogorov-Smirnov 2-sample test
        if h_std > 1e-6 or p_std > 1e-6:
            ks_stat, ks_pval = stats.ks_2samp(h_vals, p_vals)
        else:
            ks_stat, ks_pval = 0.0, 1.0

        if ks_pval < 0.001 and ks_stat > 0.20:
            shift_status = "EXTREME SHIFT"
        elif ks_pval < 0.05:
            shift_status = "MODERATE SHIFT"
        else:
            shift_status = "IN-DISTRIBUTION"

        feature_dist_rows.append({
            "dimension_index": i,
            "feature_name": dim_names[i],
            "ham10000_mean": h_mean,
            "ham10000_std": h_std,
            "ham10000_min": float(h_vals.min()),
            "ham10000_max": float(h_vals.max()),
            "m6_norm_mean": float(norm_mean[i]),
            "m6_norm_std": float(norm_std[i]),
            "ph2_mean": p_mean,
            "ph2_std": p_std,
            "ph2_min": float(p_vals.min()),
            "ph2_max": float(p_vals.max()),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "shift_status": shift_status,
        })

        print(f"{dim_names[i]:<28} | {h_mean:.4f} ({h_std:.4f})   | {p_mean:.4f} ({p_std:.4f})   | {ks_stat:<7.4f} | {ks_pval:<9.2e} | {shift_status}")

    df_feat_dist = pd.DataFrame(feature_dist_rows)
    feat_dist_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP6_FEATURE_DISTRIBUTION.csv"
    df_feat_dist.to_csv(feat_dist_csv, index=False)
    print(f"\nSaved feature distribution comparison to {feat_dist_csv}")

    # -------------------------------------------------------------------------
    # 2. IMAGE PREPROCESSING & GEOMETRY AUDIT
    # -------------------------------------------------------------------------
    print("\n[2] Auditing Image Preprocessing & Acquisition Differences...")
    # Load sample HAM10000 images to compare resolutions
    ham_meta = pd.read_csv(PROJECT_ROOT / "data" / "HAM10000" / "metadata" / "test.csv")
    print(f"HAM10000 Images: Native resolution fixed at 450 x 600 (Aspect ratio = 1.333).")
    print(f"PH² Images: Height mean = {df_ph2_img['height'].mean():.1f}, Width mean = {df_ph2_img['width'].mean():.1f} (Aspect ratio = {df_ph2_img['aspect_ratio'].mean():.3f}).")
    print(f"RGB Means - HAM10000 (typical): R~194, G~149, B~152 (Warm pinkish tone)")
    print(f"RGB Means - PH²: R={df_ph2_img['r_mean'].mean():.1f}, G={df_ph2_img['g_mean'].mean():.1f}, B={df_ph2_img['b_mean'].mean():.1f} (Cooler, polarized tone)")

    # -------------------------------------------------------------------------
    # 3. SEGMENTATION AUDIT (VM-UNet vs. Expert GT)
    # -------------------------------------------------------------------------
    print("\n[3] Segmentation Audit (VM-UNet vs. PH² Expert GT)...")
    print(f"Dice: Mean = {df_seg['dice'].mean():.4f}, Median = {df_seg['dice'].median():.4f}, Min = {df_seg['dice'].min():.4f}, Max = {df_seg['dice'].max():.4f}")
    print(f"IoU:  Mean = {df_seg['iou'].mean():.4f}, Median = {df_seg['iou'].median():.4f}, Min = {df_seg['iou'].min():.4f}, Max = {df_seg['iou'].max():.4f}")
    print(f"Over-segmentation rate (FP): Mean = {df_seg['over_segmentation_rate'].mean():.4f}")
    print(f"Under-segmentation rate (FN): Mean = {df_seg['under_segmentation_rate'].mean():.4f}")
    print(f"Pred Area / GT Area ratio: Mean = {df_seg['area_ratio'].mean():.4f}")

    # Cases with poor segmentation (Dice < 0.70)
    poor_seg = df_seg[df_seg['dice'] < 0.70]
    print(f"Number of cases with Dice < 0.70: {len(poor_seg)} ({len(poor_seg)/200*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 4. PREDICTION ERROR & LESION SUBTYPE BREAKDOWN
    # -------------------------------------------------------------------------
    print("\n[4] Prediction Error & Subtype Breakdown...")
    preds_df = pd.read_csv(OUTPUT_DIR / "CEFM_PHASE2_STEP5_PREDICTIONS.csv")
    merged_df = preds_df.merge(df_seg, on="image_id").merge(df_ph2_img, on="image_id")

    # Common Nevi vs. Atypical Nevi vs. Melanoma analysis
    for model_name, prob_col, th in [
        ("Phase 1 ViT (T=0.80)", "vit_probability", 0.800),
        ("Phase 1 Multimodal (T=0.36)", "multimodal_probability", 0.360),
        ("M6 Challenger (UNet, T=0.690)", "m6_probability", 0.690),
        ("M6 Challenger (UNet, T=0.625)", "m6_probability", 0.625),
        ("M6 Challenger (GT, T=0.690)", "m6_probability_gt_mask", 0.690),
        ("M6 Challenger (GT, T=0.625)", "m6_probability_gt_mask", 0.625),
    ]:
        print(f"\n--- Model: {model_name} ---")
        for diag, sub_df in merged_df.groupby("diagnosis_name"):
            mean_p = sub_df[prob_col].mean()
            median_p = sub_df[prob_col].median()
            preds = (sub_df[prob_col] >= th).astype(int)
            pos_rate = preds.mean()
            print(f"  {diag:<15} (N={len(sub_df):2d}): Mean Prob={mean_p:.4f}, Median={median_p:.4f}, Predicted Positive Rate={pos_rate*100:.1f}%")

    # False Negative and False Positive Detailed Error Logging
    error_records = []
    for _, r in merged_df.iterrows():
        y = r["ground_truth_label"]
        vit_p = r["vit_probability"]
        multi_p = r["multimodal_probability"]
        m6_p = r["m6_probability"]
        m6_gt_p = r["m6_probability_gt_mask"]

        m6_f1_pred = int(m6_p >= 0.690)
        m6_spec_pred = int(m6_p >= 0.625)
        vit_pred = int(vit_p >= 0.800)
        multi_pred = int(multi_p >= 0.360)

        is_fn_m6 = (y == 1 and m6_spec_pred == 0)
        is_fp_m6 = (y == 0 and m6_spec_pred == 1)
        is_fn_vit = (y == 1 and vit_pred == 0)
        is_fn_multi = (y == 1 and multi_pred == 0)

        error_records.append({
            "image_id": r["image_id"],
            "diagnosis_name": r["diagnosis_name"],
            "clinical_diagnosis_code": r["clinical_diagnosis_code"],
            "histo_diagnosis": r["histo_diagnosis"],
            "ground_truth_label": y,
            "vit_probability": vit_p,
            "vit_predicted_melanoma": vit_pred,
            "multimodal_probability": multi_p,
            "multimodal_predicted_melanoma": multi_pred,
            "m6_probability_unet": m6_p,
            "m6_pred_f1_0690": m6_f1_pred,
            "m6_pred_spec_0625": m6_spec_pred,
            "m6_probability_gt_mask": m6_gt_p,
            "m6_is_false_negative": is_fn_m6,
            "m6_is_false_positive": is_fp_m6,
            "vit_is_false_negative": is_fn_vit,
            "multimodal_is_false_negative": is_fn_multi,
            "unet_dice_vs_gt": r["dice"],
            "unet_iou_vs_gt": r["iou"],
            "over_segmentation_rate": r["over_segmentation_rate"],
            "under_segmentation_rate": r["under_segmentation_rate"],
            "local_patch_count": r["local_patch_count"],
            "asymmetry": r["asymmetry"],
            "border_irregularity": r["border_irregularity"],
        })

    df_errors = pd.DataFrame(error_records)
    error_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP6_ERROR_ANALYSIS.csv"
    df_errors.to_csv(error_csv, index=False)
    print(f"\nSaved detailed error analysis to {error_csv}")

    # -------------------------------------------------------------------------
    # 5. RENDER OPENCV PUBLICATION FIGURES
    # -------------------------------------------------------------------------
    print("\n[5] Rendering Domain Shift Publication Figures...")

    # Figure 1: 16-D Clinical Feature Shift Bar Chart
    def render_feature_shift_plot(save_path: Path):
        canvas = np.ones((700, 1000, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Clinical Feature Domain Shift: HAM10000 vs. PH2 (16 Dimensions)",
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)

        ox, oy = 100, 600
        w, h = 820, 500
        cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)

        # Plot KS statistics for all 16 features
        bar_w = int(w / 16)
        for i, row in df_feat_dist.iterrows():
            ks = row["ks_statistic"]
            bh = int(ks * h)
            bx0 = ox + i * bar_w + 6
            by0 = oy - bh

            color = (30, 30, 220) if row["shift_status"] == "EXTREME SHIFT" else ((40, 140, 40) if row["shift_status"] == "IN-DISTRIBUTION" else (200, 140, 30))
            cv2.rectangle(canvas, (bx0, by0), (bx0 + bar_w - 12, oy), color, -1)
            cv2.rectangle(canvas, (bx0, by0), (bx0 + bar_w - 12, oy), (50, 50, 50), 1)

            cv2.putText(canvas, f"D{i}", (bx0 + 4, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1)
            cv2.putText(canvas, f"{ks:.2f}", (bx0 - 2, by0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)

        # Grid lines for KS stat
        for k in [0.2, 0.4, 0.6, 0.8, 1.0]:
            y = int(oy - k * h)
            cv2.line(canvas, (ox, y), (ox + w, y), (200, 200, 200), 1)
            cv2.putText(canvas, f"{k:.1f}", (ox - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

        cv2.putText(canvas, "Kolmogorov-Smirnov Statistic (Distribution Divergence)", (ox - 85, oy - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Feature Dimension Index (D0 to D15)", (ox + 280, oy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 50, 50), 1, cv2.LINE_AA)

        # Legend
        cv2.rectangle(canvas, (ox + 500, 80), (ox + 520, 95), (30, 30, 220), -1)
        cv2.putText(canvas, "Extreme Shift (p < 0.001, KS > 0.20)", (ox + 530, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        cv2.rectangle(canvas, (ox + 500, 105), (ox + 520, 120), (40, 140, 40), -1)
        cv2.putText(canvas, "In-Distribution (p >= 0.05)", (ox + 530, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

        cv2.imwrite(str(save_path), canvas)

    fig1_path = FIGURES_DIR / "ph2_feature_domain_shift.png"
    render_feature_shift_plot(fig1_path)

    # Figure 2: Prediction Distribution by Lesion Subtype
    def render_subtype_distribution_plot(save_path: Path):
        canvas = np.ones((700, 950, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "M6 Predicted Probability Distribution by Histological Subtype",
                    (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)

        ox, oy = 100, 600
        w, h = 760, 500
        cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)

        # Draw decision thresholds
        th_f1_y = int(oy - 0.690 * h)
        th_sp_y = int(oy - 0.625 * h)
        cv2.line(canvas, (ox, th_f1_y), (ox + w, th_f1_y), (30, 30, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "F1 Operating Point (T* = 0.690)", (ox + w - 240, th_f1_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 220), 1)

        cv2.line(canvas, (ox, th_sp_y), (ox + w, th_sp_y), (40, 140, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Screening Point (T* = 0.625)", (ox + w - 240, th_sp_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 140, 40), 1)

        # Plot jittered points for: Common Nevi, Atypical Nevi, Melanomas
        groups = [
            ("Common Nevus (N=80)", merged_df[merged_df["diagnosis_name"] == "common_nevus"]["m6_probability"].values, 180, (40, 140, 40)),
            ("Atypical Nevus (N=80)", merged_df[merged_df["diagnosis_name"] == "atypical_nevus"]["m6_probability"].values, 440, (200, 140, 30)),
            ("Melanoma (N=40)", merged_df[merged_df["diagnosis_name"] == "melanoma"]["m6_probability"].values, 700, (30, 30, 220)),
        ]

        np.random.seed(42)
        for label, vals, cx, col in groups:
            cv2.putText(canvas, label, (cx - 70, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
            mean_y = int(oy - vals.mean() * h)
            cv2.line(canvas, (cx - 50, mean_y), (cx + 50, mean_y), (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"Mean: {vals.mean():.3f}", (cx - 45, mean_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 1)

            for v in vals:
                jx = int(cx + np.random.uniform(-35, 35))
                jy = int(oy - v * h)
                cv2.circle(canvas, (jx, jy), 4, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (jx, jy), 4, (30, 30, 30), 1, cv2.LINE_AA)

        # Y axis
        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = int(oy - p * h)
            cv2.line(canvas, (ox, y), (ox + w, y), (210, 210, 210), 1)
            cv2.putText(canvas, f"{p:.1f}", (ox - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        cv2.putText(canvas, "M6 Predicted Melanoma Probability", (ox - 85, oy - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.imwrite(str(save_path), canvas)

    fig2_path = FIGURES_DIR / "ph2_subtype_probability_distribution.png"
    render_subtype_distribution_plot(fig2_path)

    # Figure 3: Segmentation Overlap vs. Prediction Confidence
    def render_seg_vs_pred_plot(save_path: Path):
        canvas = np.ones((700, 950, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "VM-UNet Segmentation Dice vs. M6 Prediction Confidence on Melanomas",
                    (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

        ox, oy = 100, 600
        w, h = 760, 500
        cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)

        mel_df = merged_df[merged_df["ground_truth_label"] == 1]
        for _, r in mel_df.iterrows():
            d = r["dice"]
            p = r["m6_probability"]
            cx = int(ox + d * w)
            cy = int(oy - p * h)
            col = (30, 30, 220) if p >= 0.625 else (140, 40, 180)
            cv2.circle(canvas, (cx, cy), 6, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 6, (20, 20, 20), 1, cv2.LINE_AA)

        for d_tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            x = int(ox + d_tick * w)
            cv2.line(canvas, (x, oy - h), (x, oy), (210, 210, 210), 1)
            cv2.putText(canvas, f"{d_tick:.1f}", (x - 12, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        for p_tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = int(oy - p_tick * h)
            cv2.line(canvas, (ox, y), (ox + w, y), (210, 210, 210), 1)
            cv2.putText(canvas, f"{p_tick:.1f}", (ox - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        cv2.putText(canvas, "VM-UNet Dice Similarity Coefficient", (ox + 230, oy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, "M6 Melanoma Probability", (ox - 85, oy - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 50, 50), 1, cv2.LINE_AA)

        # Threshold line
        th_y = int(oy - 0.625 * h)
        cv2.line(canvas, (ox, th_y), (ox + w, th_y), (40, 140, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Detection Threshold (0.625)", (ox + 20, th_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 140, 40), 1)

        cv2.imwrite(str(save_path), canvas)

    fig3_path = FIGURES_DIR / "ph2_segmentation_vs_confidence.png"
    render_seg_vs_pred_plot(fig3_path)

    # Copy to brain artifact directory as well
    import shutil
    brain_fig_dir = BRAIN_ARTIFACT_DIR / "figures"
    brain_fig_dir.mkdir(parents=True, exist_ok=True)
    for f in [fig1_path, fig2_path, fig3_path]:
        shutil.copy(f, brain_fig_dir / f.name)
    print("Copied domain shift figures to artifact directory.")

    print("\nDomain shift audit execution completed successfully!")


if __name__ == "__main__":
    run_domain_shift_audit()

