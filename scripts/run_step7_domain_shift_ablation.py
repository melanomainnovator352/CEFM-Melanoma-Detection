"""
CEFM Phase 2 Step 7 — Controlled Domain-Shift Ablation Study.
Diagnostic research experiment investigating M6 clinical gate activations,
controlled feature ablations (Counterfactuals A-F), crop geometry, and optical color shift.
"""

import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.classification import _prepare_abcde_tensor
from app.services.dual_scale import (
    compute_lesion_spatial_bounds,
    extract_global_crop,
    extract_local_patches,
    extract_dual_scale_views,
)
from app.ml.multimodal.dual_scale_models import M6_DualGated

PH2_DIR = Path("D:/PH2")
PH2_IMAGES_DIR = PH2_DIR / "PH2 Dataset images"
PH2_TXT = PH2_DIR / "PH2_dataset.txt"

OUTPUT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures" / "phase2_feature_ablation"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_ARTIFACT_DIR = Path("C:/Users/sruja/.gemini/antigravity/brain/fd604b17-8f8a-44a3-a817-98d62f87fbac")


def run_ablation_study():
    print("=" * 80)
    print("CEFM PHASE 2 STEP 7 — CONTROLLED DOMAIN-SHIFT ABLATION STUDY")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. LOAD MODEL & CONFIGURATION
    # -------------------------------------------------------------------------
    weights_path = PROJECT_ROOT / "models" / "experiments" / "M6" / "weights.pt"
    cfg_path = PROJECT_ROOT / "models" / "experiments" / "M6" / "config.json"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    norm_mean = torch.tensor(cfg["clinical_norm_mean"], dtype=torch.float32)
    norm_std = torch.tensor(cfg["clinical_norm_std"], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M6_DualGated(
        global_dim=768, local_dim=768, clinical_dim=16,
        hidden_dim=128, clinical_hidden=64, num_classes=2, dropout=0.15
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("Loaded frozen M6 Challenger Model.")

    # -------------------------------------------------------------------------
    # 2. LOAD PREDICTIONS & EXTRACT GATE ACTIVATIONS ACROSS ALL 200 PH2 CASES
    # -------------------------------------------------------------------------
    print("\n[Step 2] Extracting Clinical and Visual Gate Activations on PH² (N=200)...")

    gate_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP7_GATE_ACTIVATIONS.csv"
    ablation_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP7_FEATURE_ABLATION.csv"
    crop_csv = OUTPUT_DIR / "CEFM_PHASE2_STEP7_CROP_ANALYSIS.csv"

    def extract_all_records():
        # Load ViT for feature extraction
        from transformers import ViTForImageClassification, ViTImageProcessor
        vit_path = PROJECT_ROOT / "models" / "classification" / "vit_model"
        vit_model = ViTForImageClassification.from_pretrained(str(vit_path), local_files_only=True)
        vit_model.to(device)
        vit_model.eval()
        processor = ViTImageProcessor.from_pretrained(str(vit_path), local_files_only=True)

        with open(PH2_TXT, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and l.startswith("||") and not l.startswith("||   Name")]

        gate_records = []
        ablation_records = []
        crop_records = []

        start_time = time.time()
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
            h_img, w_img, _ = img_bgr.shape
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            gt_mask_bin = (gt_mask > 127).astype(np.uint8)

            # Automated VM-UNet segmentation
            pred_mask = segment_lesion(img_path)

            # --- A. Crop & Patch Geometry Analysis (UNet vs GT) ---
            bounds_unet = compute_lesion_spatial_bounds(pred_mask, (h_img, w_img))
            bounds_gt = compute_lesion_spatial_bounds(gt_mask_bin, (h_img, w_img))

            views_unet = extract_dual_scale_views(img_rgb, pred_mask, patch_size=224)
            views_gt = extract_dual_scale_views(img_rgb, gt_mask_bin, patch_size=224)

            # Measure lesion coverage in global crop
            # UNet global crop
            gb_u = views_unet["global_bbox"]
            g_mask_unet_crop = gt_mask_bin[gb_u["ymin"]:gb_u["ymax"], gb_u["xmin"]:gb_u["xmax"]]
            lesion_in_unet_crop = g_mask_unet_crop.sum()
            total_lesion = gt_mask_bin.sum()
            lesion_coverage_unet = float(lesion_in_unet_crop / (total_lesion + 1e-8))
            context_area_unet = (gb_u["ymax"] - gb_u["ymin"]) * (gb_u["xmax"] - gb_u["xmin"]) - lesion_in_unet_crop

            # GT global crop
            gb_g = views_gt["global_bbox"]
            g_mask_gt_crop = gt_mask_bin[gb_g["ymin"]:gb_g["ymax"], gb_g["xmin"]:gb_g["xmax"]]
            lesion_in_gt_crop = g_mask_gt_crop.sum()
            lesion_coverage_gt = float(lesion_in_gt_crop / (total_lesion + 1e-8))
            context_area_gt = (gb_g["ymax"] - gb_g["ymin"]) * (gb_g["xmax"] - gb_g["xmin"]) - lesion_in_gt_crop

            side_unet = max(gb_u["ymax"] - gb_u["ymin"], gb_u["xmax"] - gb_u["xmin"])
            side_gt = max(gb_g["ymax"] - gb_g["ymin"], gb_g["xmax"] - gb_g["xmin"])

            crop_records.append({
                "image_id": name,
                "diagnosis_name": diag_name,
                "ground_truth_label": label,
                "unet_side_length": side_unet,
                "gt_side_length": side_gt,
                "side_length_ratio_unet_over_gt": float(side_unet / (side_gt + 1e-8)),
                "unet_patch_count": views_unet["num_local_patches"],
                "gt_patch_count": views_gt["num_local_patches"],
                "lesion_coverage_unet": lesion_coverage_unet,
                "lesion_coverage_gt": lesion_coverage_gt,
                "context_skin_area_unet_pixels": int(context_area_unet),
                "context_skin_area_gt_pixels": int(context_area_gt),
                "context_expansion_ratio": float(context_area_unet / (context_area_gt + 1e-8)),
            })

            # --- B. ViT Feature Embeddings ---
            # 1. Global embedding
            inputs_g = processor(images=views_unet["global_image"], return_tensors="pt")
            inputs_g = {k: v.to(device) for k, v in inputs_g.items()}
            with torch.no_grad():
                outputs_g = vit_model(**inputs_g, output_hidden_states=True)
                emb_g = outputs_g.hidden_states[-1][:, 0, :]  # [1, 768]

            # 2. Local patch embeddings
            p_imgs = [p["image"] for p in views_unet["local_patches"]]
            inputs_l = processor(images=p_imgs, return_tensors="pt")
            inputs_l = {k: v.to(device) for k, v in inputs_l.items()}
            with torch.no_grad():
                outputs_l = vit_model(**inputs_l, output_hidden_states=True)
                embs_l = outputs_l.hidden_states[-1][:, 0, :]  # [K, 768]
                if len(p_imgs) == 1:
                    emb_l = embs_l[0:1]
                else:
                    w = torch.ones(len(p_imgs), device=device)
                    w[0] = 2.0
                    w = (w / w.sum()).unsqueeze(1)
                    emb_l = (embs_l * w).sum(dim=0, keepdim=True)

            # 3. Clinical feature vector
            feat = extract_abcde_features(img_rgb, pred_mask)
            raw_clin_t = _prepare_abcde_tensor(feat).to(device)  # [1, 16]
            norm_mean_d = norm_mean.to(device).unsqueeze(0)
            norm_std_d = norm_std.to(device).unsqueeze(0)
            c_norm_orig = (raw_clin_t - norm_mean_d) / norm_std_d

            # --- C. Inspect M6 Internal Gate Activations ---
            with torch.no_grad():
                zg = model.global_encoder(emb_g)
                zl = model.local_encoder(emb_l)
                zc = model.clinical_encoder(c_norm_orig)

                concat_v = torch.cat([zg, zl], dim=-1)
                scale_weights = model.scale_gate(concat_v)   # [1, 128]
                c_mod = model.clinical_gate(zc)              # [1, 128]

                blended_v = scale_weights * zg + (1.0 - scale_weights) * zl
                modulated_v = blended_v * (1.0 + c_mod)
                combined = torch.cat([modulated_v, zl, zc], dim=-1)
                fused = model.fusion(combined)
                logits_orig = model.classifier(fused)
                p_orig = F.softmax(logits_orig, dim=-1)[0, 1].item()

            gate_records.append({
                "image_id": name,
                "diagnosis_name": diag_name,
                "clinical_diagnosis_code": clin_code,
                "ground_truth_label": label,
                "scale_gate_mean": scale_weights.mean().item(),
                "scale_gate_min": scale_weights.min().item(),
                "scale_gate_max": scale_weights.max().item(),
                "clinical_gate_mean": c_mod.mean().item(),
                "clinical_gate_min": c_mod.min().item(),
                "clinical_gate_max": c_mod.max().item(),
                "clinical_embedding_norm": zc.norm().item(),
                "raw_hue_std": raw_clin_t[0, 7].item(),
                "norm_hue_std": c_norm_orig[0, 7].item(),
                "raw_val_std": raw_clin_t[0, 9].item(),
                "norm_val_std": c_norm_orig[0, 9].item(),
                "raw_border_score": raw_clin_t[0, 3].item(),
                "norm_border_score": c_norm_orig[0, 3].item(),
                "raw_asymmetry": raw_clin_t[0, 0].item(),
                "norm_asymmetry": c_norm_orig[0, 0].item(),
                "original_melanoma_prob": p_orig,
            })

            # --- D. Controlled Diagnostic Counterfactuals (A to F) ---
            # A: Original
            p_A = p_orig

            # B: Hue Variation set to 0.0 (training mean)
            c_B = c_norm_orig.clone()
            c_B[0, 7] = 0.0
            with torch.no_grad():
                p_B = F.softmax(model(emb_g, emb_l, c_B), dim=-1)[0, 1].item()

            # C: Value Variation set to 0.0 (training mean)
            c_C = c_norm_orig.clone()
            c_C[0, 9] = 0.0
            with torch.no_grad():
                p_C = F.softmax(model(emb_g, emb_l, c_C), dim=-1)[0, 1].item()

            # D: Both Hue and Value Variation set to 0.0
            c_D = c_norm_orig.clone()
            c_D[0, 7] = 0.0
            c_D[0, 9] = 0.0
            with torch.no_grad():
                p_D = F.softmax(model(emb_g, emb_l, c_D), dim=-1)[0, 1].item()

            # E: All clinical inputs set to training means (all zeros in c_norm)
            c_E = torch.zeros_like(c_norm_orig)
            with torch.no_grad():
                p_E = F.softmax(model(emb_g, emb_l, c_E), dim=-1)[0, 1].item()

            # F: Visual inputs with clinical modulation disabled (c_mod=0, zc=0)
            with torch.no_grad():
                zg_f = model.global_encoder(emb_g)
                zl_f = model.local_encoder(emb_l)
                concat_vf = torch.cat([zg_f, zl_f], dim=-1)
                scale_w_f = model.scale_gate(concat_vf)
                blended_vf = scale_w_f * zg_f + (1.0 - scale_w_f) * zl_f
                
                # Clinical modulation disabled: c_mod = 0.0 -> modulated_v = blended_v * 1.0
                # zc = zeros
                zc_zero = torch.zeros((1, 64), device=device)
                combined_f = torch.cat([blended_vf, zl_f, zc_zero], dim=-1)
                fused_f = model.fusion(combined_f)
                logits_f = model.classifier(fused_f)
                p_F = F.softmax(logits_f, dim=-1)[0, 1].item()

            ablation_records.append({
                "image_id": name,
                "diagnosis_name": diag_name,
                "ground_truth_label": label,
                "prob_A_orig": p_A,
                "prob_B_hue_zero": p_B,
                "delta_B_minus_A": p_B - p_A,
                "prob_C_val_zero": p_C,
                "delta_C_minus_A": p_C - p_A,
                "prob_D_both_zero": p_D,
                "delta_D_minus_A": p_D - p_A,
                "prob_E_all_zero": p_E,
                "delta_E_minus_A": p_E - p_A,
                "prob_F_vis_only": p_F,
                "delta_F_minus_A": p_F - p_A,
            })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(lines):
                print(f"Processed {idx + 1}/{len(lines)} cases in {time.time() - start_time:.1f}s")

        df_g = pd.DataFrame(gate_records)
        df_a = pd.DataFrame(ablation_records)
        df_c = pd.DataFrame(crop_records)

        df_g.to_csv(gate_csv, index=False)
        df_a.to_csv(ablation_csv, index=False)
        df_c.to_csv(crop_csv, index=False)

        print(f"\nSaved gate activations to {gate_csv}")
        print(f"Saved feature ablation data to {ablation_csv}")
        print(f"Saved crop analysis to {crop_csv}")
        return df_g, df_a, df_c

    if gate_csv.exists() and ablation_csv.exists() and crop_csv.exists():
        print(f"Loading existing gate, ablation, and crop records from {OUTPUT_DIR}...")
        df_gates = pd.read_csv(gate_csv)
        df_ablation = pd.read_csv(ablation_csv)
        df_crops = pd.read_csv(crop_csv)
    else:
        df_gates, df_ablation, df_crops = extract_all_records()

    # -------------------------------------------------------------------------
    # 3. STATISTICAL ANALYSIS OF GATE ACTIVATIONS
    # -------------------------------------------------------------------------
    print("\n--- Gate Activations Analysis by Lesion Subtype ---")
    print(f"{'Subtype':<16} | {'Count':<5} | {'Scale Gate Mean (Std)':<24} | {'Clinical Gate Mean (Std)':<26} | {'Melanoma Prob'}")
    print("-" * 90)
    for diag, grp in df_gates.groupby("diagnosis_name"):
        sg_m, sg_s = grp["scale_gate_mean"].mean(), grp["scale_gate_mean"].std()
        cg_m, cg_s = grp["clinical_gate_mean"].mean(), grp["clinical_gate_mean"].std()
        p_m = grp["original_melanoma_prob"].mean()
        print(f"{diag:<16} | {len(grp):<5} | {sg_m:.4f} ({sg_s:.4f})           | {cg_m:.4f} ({cg_s:.4f})             | {p_m:.4f}")

    # Correlation between gate activation and hue z-score
    corr_hue_gate, pval_hue = stats.pearsonr(df_gates["norm_hue_std"], df_gates["clinical_gate_mean"])
    corr_val_gate, pval_val = stats.pearsonr(df_gates["norm_val_std"], df_gates["clinical_gate_mean"])
    print(f"\nPearson Correlation: norm_hue_std vs clinical_gate_mean: r = {corr_hue_gate:.4f} (p = {pval_hue:.2e})")
    print(f"Pearson Correlation: norm_val_std vs clinical_gate_mean: r = {corr_val_gate:.4f} (p = {pval_val:.2e})")

    # -------------------------------------------------------------------------
    # 4. COUNTERFACTUAL ABLATION BENCHMARK EVALUATION (A to F)
    # -------------------------------------------------------------------------
    y_true = df_ablation["ground_truth_label"].values
    variants = [
        ("A: Original M6", "prob_A_orig"),
        ("B: Hue Feature Zeroed", "prob_B_hue_zero"),
        ("C: Value Feature Zeroed", "prob_C_val_zero"),
        ("D: Hue & Value Zeroed", "prob_D_both_zero"),
        ("E: All Clinical Zeroed", "prob_E_all_zero"),
        ("F: Clinical Mod Disabled", "prob_F_vis_only"),
    ]

    print("\n--- Counterfactual Ablation Results Across Thresholds (N=200) ---")
    print(f"{'Variant':<26} | {'Mean Prob (Mel)':<16} | {'Sens (0.690)':<12} | {'Spec (0.690)':<12} | {'Sens (0.625)':<12} | {'Spec (0.625)':<12} | {'ROC-AUC':<8}")
    print("-" * 115)

    ablation_summary = []
    for v_name, col in variants:
        probs = df_ablation[col].values
        auc_val = float(roc_auc_score(y_true, probs))
        mel_mean_prob = float(df_ablation[df_ablation["ground_truth_label"] == 1][col].mean())

        # At threshold 0.690
        preds_690 = (probs >= 0.690).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds_690, labels=[0, 1]).ravel()
        sens_690 = tp / (tp + fn)
        spec_690 = tn / (tn + fp)

        # At threshold 0.625
        preds_625 = (probs >= 0.625).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_true, preds_625, labels=[0, 1]).ravel()
        sens_625 = tp2 / (tp2 + fn2)
        spec_625 = tn2 / (tn2 + fp2)

        ablation_summary.append({
            "variant": v_name,
            "melanoma_mean_prob": mel_mean_prob,
            "roc_auc": auc_val,
            "sensitivity_0690": sens_690,
            "specificity_0690": spec_690,
            "tp_0690": int(tp),
            "fp_0690": int(fp),
            "sensitivity_0625": sens_625,
            "specificity_0625": spec_625,
            "tp_0625": int(tp2),
            "fp_0625": int(fp2),
        })

        print(f"{v_name:<26} | {mel_mean_prob:<16.4f} | {sens_690*100:<10.1f}% | {spec_690*100:<10.1f}% | {sens_625*100:<10.1f}% | {spec_625*100:<10.1f}% | {auc_val:<8.4f}")

    # -------------------------------------------------------------------------
    # 5. CROP & CONTEXT INVESTIGATION (UNet vs GT)
    # -------------------------------------------------------------------------
    print("\n--- Crop Geometry & Context Analysis (UNet vs. GT) ---")
    print(f"Mean UNet Crop Side Length: {df_crops['unet_side_length'].mean():.1f} px vs. GT Crop Side: {df_crops['gt_side_length'].mean():.1f} px (Ratio: {df_crops['side_length_ratio_unet_over_gt'].mean():.3f})")
    print(f"Mean Perilesional Context Skin Area: UNet = {df_crops['context_skin_area_unet_pixels'].mean():.0f} px vs. GT = {df_crops['context_skin_area_gt_pixels'].mean():.0f} px (Context Expansion: {df_crops['context_expansion_ratio'].mean():.3f}x)")
    print(f"Mean Lesion Coverage: UNet = {df_crops['lesion_coverage_unet'].mean()*100:.2f}% vs. GT = {df_crops['lesion_coverage_gt'].mean()*100:.2f}%")
    print(f"Mean Patch Count: UNet = {df_crops['unet_patch_count'].mean():.2f} vs. GT = {df_crops['gt_patch_count'].mean():.2f}")

    # -------------------------------------------------------------------------
    # 6. OPTICAL COLOR-SHIFT & CHANNEL RATIO AUDIT (HAM10000 vs PH2)
    # -------------------------------------------------------------------------
    print("\n--- Optical Color-Shift & Channel Ratio Audit ---")
    # Sample 200 images from HAM10000 test set
    ham_test_df = pd.read_csv(PROJECT_ROOT / "data" / "HAM10000" / "metadata" / "test.csv")
    from scripts.evaluate_vit import find_image

    ham_stats = []
    for _, r in ham_test_df.head(200).iterrows():
        p = find_image(r["image_id"])
        img_b = cv2.imread(str(p))
        if img_b is not None:
            r_m = img_b[:, :, 2].mean()
            g_m = img_b[:, :, 1].mean()
            b_m = img_b[:, :, 0].mean()
            hsv = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
            h_m = hsv[:, :, 0].mean()
            s_m = hsv[:, :, 1].mean()
            v_m = hsv[:, :, 2].mean()
            ham_stats.append({
                "r_mean": r_m, "g_mean": g_m, "b_mean": b_m,
                "h_mean": h_m, "s_mean": s_m, "v_mean": v_m,
                "rb_ratio": r_m / (b_m + 1e-6),
                "rg_ratio": r_m / (g_m + 1e-6),
                "gb_ratio": g_m / (b_m + 1e-6),
            })
    df_ham_col = pd.DataFrame(ham_stats)

    ph2_col_stats = []
    for _, r in df_crops.iterrows():
        p = PH2_IMAGES_DIR / r["image_id"] / f"{r['image_id']}_Dermoscopic_Image" / f"{r['image_id']}.bmp"
        img_b = cv2.imread(str(p))
        if img_b is not None:
            r_m = img_b[:, :, 2].mean()
            g_m = img_b[:, :, 1].mean()
            b_m = img_b[:, :, 0].mean()
            hsv = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
            h_m = hsv[:, :, 0].mean()
            s_m = hsv[:, :, 1].mean()
            v_m = hsv[:, :, 2].mean()
            ph2_col_stats.append({
                "r_mean": r_m, "g_mean": g_m, "b_mean": b_m,
                "h_mean": h_m, "s_mean": s_m, "v_mean": v_m,
                "rb_ratio": r_m / (b_m + 1e-6),
                "rg_ratio": r_m / (g_m + 1e-6),
                "gb_ratio": g_m / (b_m + 1e-6),
            })
    df_ph2_col = pd.DataFrame(ph2_col_stats)

    print(f"Red Channel Mean:   HAM10000 = {df_ham_col['r_mean'].mean():.1f} vs. PH² = {df_ph2_col['r_mean'].mean():.1f}")
    print(f"Green Channel Mean: HAM10000 = {df_ham_col['g_mean'].mean():.1f} vs. PH² = {df_ph2_col['g_mean'].mean():.1f}")
    print(f"Blue Channel Mean:  HAM10000 = {df_ham_col['b_mean'].mean():.1f} vs. PH² = {df_ph2_col['b_mean'].mean():.1f} (Shift: {df_ph2_col['b_mean'].mean() - df_ham_col['b_mean'].mean():.1f})")
    print(f"R/B Color Ratio:    HAM10000 = {df_ham_col['rb_ratio'].mean():.2f} vs. PH² = {df_ph2_col['rb_ratio'].mean():.2f} (Shift: +{(df_ph2_col['rb_ratio'].mean()/df_ham_col['rb_ratio'].mean() - 1)*100:.1f}%)")
    print(f"Mean Hue (HSV):     HAM10000 = {df_ham_col['h_mean'].mean():.1f} vs. PH² = {df_ph2_col['h_mean'].mean():.1f}")
    print(f"Mean Saturation:    HAM10000 = {df_ham_col['s_mean'].mean():.1f} vs. PH² = {df_ph2_col['s_mean'].mean():.1f}")

    # -------------------------------------------------------------------------
    # 7. OPENCV PUBLICATION FIGURES GENERATION
    # -------------------------------------------------------------------------
    print("\nRendering publication figures...")

    # Figure 1: Gate Activations Box/Scatter
    def render_gate_activations_plot(save_path: Path):
        canvas = np.ones((700, 950, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "M6 Gate Activations by Subtype: Clinical vs. Visual Gating",
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)

        ox, oy = 100, 600
        w, h = 760, 500
        cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)

        groups = [
            ("Common Nevi", df_gates[df_gates["diagnosis_name"] == "common_nevus"], 180, (40, 140, 40)),
            ("Atypical Nevi", df_gates[df_gates["diagnosis_name"] == "atypical_nevus"], 440, (200, 140, 30)),
            ("Melanomas", df_gates[df_gates["diagnosis_name"] == "melanoma"], 700, (30, 30, 220)),
        ]

        np.random.seed(42)
        for label, grp, cx, col in groups:
            cv2.putText(canvas, label, (cx - 50, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

            # Plot Clinical Gate (Left offset -20)
            cg_vals = grp["clinical_gate_mean"].values
            mean_cg_y = int(oy - cg_vals.mean() * h)
            cv2.line(canvas, (cx - 45, mean_cg_y), (cx - 5, mean_cg_y), col, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"Clin: {cg_vals.mean():.2f}", (cx - 50, mean_cg_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

            for v in cg_vals:
                jx = int(cx - 25 + np.random.uniform(-10, 10))
                jy = int(oy - v * h)
                cv2.circle(canvas, (jx, jy), 3, col, -1, cv2.LINE_AA)

            # Plot Scale Gate (Right offset +20)
            sg_vals = grp["scale_gate_mean"].values
            mean_sg_y = int(oy - sg_vals.mean() * h)
            cv2.line(canvas, (cx + 5, mean_sg_y), (cx + 45, mean_sg_y), (100, 100, 100), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"Scale: {sg_vals.mean():.2f}", (cx + 5, mean_sg_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1)

            for v in sg_vals:
                jx = int(cx + 25 + np.random.uniform(-10, 10))
                jy = int(oy - v * h)
                cv2.circle(canvas, (jx, jy), 3, (100, 100, 100), -1, cv2.LINE_AA)

        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = int(oy - p * h)
            cv2.line(canvas, (ox, y), (ox + w, y), (210, 210, 210), 1)
            cv2.putText(canvas, f"{p:.1f}", (ox - 35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        cv2.putText(canvas, "Gate Mean Activation (Sigmoid Output [0.0 - 1.0])", (ox - 85, oy - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
        
        # Legend
        cv2.rectangle(canvas, (ox + 500, 75), (ox + 520, 90), (30, 30, 220), -1)
        cv2.putText(canvas, "Clinical Modulation Gate (c_mod)", (ox + 530, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1)
        cv2.rectangle(canvas, (ox + 500, 100), (ox + 520, 115), (100, 100, 100), -1)
        cv2.putText(canvas, "Scale Weighting Gate (Global vs Local)", (ox + 530, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1)

        cv2.imwrite(str(save_path), canvas)

    fig1_path = FIGURES_DIR / "ph2_gate_activations_by_subtype.png"
    render_gate_activations_plot(fig1_path)

    # Figure 2: Counterfactual Ablation ROC Curves
    def render_ablation_roc_plot(save_path: Path):
        canvas = np.ones((750, 950, 3), dtype=np.uint8) * 255
        ox, oy = 110, 620
        w, h = 640, 500

        cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)
        for i in range(11):
            x = int(ox + i * (w / 10))
            y = int(oy - i * (h / 10))
            cv2.line(canvas, (x, oy - h), (x, oy), (220, 224, 230), 1)
            cv2.line(canvas, (ox, y), (ox + w, y), (220, 224, 230), 1)
            cv2.putText(canvas, f"{i/10:.1f}", (x - 12, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
            cv2.putText(canvas, f"{i/10:.1f}", (ox - 42, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

        cv2.line(canvas, (ox, oy), (ox + w, oy - h), (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Counterfactual Feature Ablation: ROC Curves on PH2 (N=200)",
                    (ox - 30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, "False Positive Rate (1 - Specificity)", (ox + 180, oy + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, "True Positive Rate (Sensitivity)", (ox - 85, oy - 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

        cols_map = {
            "A: Original M6": ((30, 30, 220), 3),
            "B: Hue Feature Zeroed": ((200, 140, 30), 2),
            "C: Value Feature Zeroed": ((40, 140, 40), 2),
            "D: Hue & Value Zeroed": ((140, 40, 180), 2),
            "E: All Clinical Zeroed": ((180, 80, 40), 2),
            "F: Clinical Mod Disabled": ((80, 80, 80), 2),
        }

        legend_y = 130
        for v_name, col_name in variants:
            probs = df_ablation[col_name].values
            fpr, tpr, _ = roc_curve(y_true, probs)
            pts = [np.array([int(ox + fx * w), int(oy - ty * h)], dtype=np.int32) for fx, ty in zip(fpr, tpr)]
            c_rgb, thick = cols_map[v_name]
            for j in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]), c_rgb, thick, cv2.LINE_AA)

            score = roc_auc_score(y_true, probs)
            cv2.line(canvas, (ox + 310, legend_y), (ox + 350, legend_y), c_rgb, thick + 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{v_name} (AUC: {score:.4f})", (ox + 360, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)
            legend_y += 24

        cv2.imwrite(str(save_path), canvas)

    fig2_path = FIGURES_DIR / "ph2_counterfactual_ablation_roc.png"
    render_ablation_roc_plot(fig2_path)

    # Figure 3: Color Channel Shift Comparison (HAM10000 vs PH2)
    def render_color_shift_plot(save_path: Path):
        canvas = np.ones((650, 950, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Optical Color-Channel Shift: HAM10000 vs. PH2 Cohorts",
                    (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)

        ox, oy = 100, 560
        w, h = 760, 440
        cv2.rectangle(canvas, (ox, oy - h), (ox + w, oy), (242, 244, 247), -1)

        # Plot 4 Metrics: Red Mean, Green Mean, Blue Mean, R/B Ratio
        metrics_plot = [
            ("Red Mean", df_ham_col["r_mean"].mean(), df_ph2_col["r_mean"].mean(), 255.0, 100),
            ("Green Mean", df_ham_col["g_mean"].mean(), df_ph2_col["g_mean"].mean(), 255.0, 300),
            ("Blue Mean", df_ham_col["b_mean"].mean(), df_ph2_col["b_mean"].mean(), 255.0, 500),
            ("R / B Ratio", df_ham_col["rb_ratio"].mean(), df_ph2_col["rb_ratio"].mean(), 3.0, 700),
        ]

        for m_name, ham_v, ph2_v, max_v, cx in metrics_plot:
            cv2.putText(canvas, m_name, (cx - 40, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)

            # HAM bar
            bh_ham = int((ham_v / max_v) * h)
            cv2.rectangle(canvas, (cx - 45, oy - bh_ham), (cx - 10, oy), (180, 80, 40), -1)
            cv2.rectangle(canvas, (cx - 45, oy - bh_ham), (cx - 10, oy), (40, 40, 40), 1)
            cv2.putText(canvas, f"{ham_v:.1f}", (cx - 48, oy - bh_ham - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 80, 40), 1)

            # PH2 bar
            bh_ph2 = int((ph2_v / max_v) * h)
            cv2.rectangle(canvas, (cx + 5, oy - bh_ph2), (cx + 40, oy), (30, 30, 220), -1)
            cv2.rectangle(canvas, (cx + 5, oy - bh_ph2), (cx + 40, oy), (40, 40, 40), 1)
            cv2.putText(canvas, f"{ph2_v:.1f}", (cx + 5, oy - bh_ph2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 220), 1)

        # Legend
        cv2.rectangle(canvas, (ox + 480, 80), (ox + 500, 95), (180, 80, 40), -1)
        cv2.putText(canvas, "HAM10000 Multi-Center Cohort (N=200)", (ox + 510, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        cv2.rectangle(canvas, (ox + 480, 105), (ox + 500, 120), (30, 30, 220), -1)
        cv2.putText(canvas, "PH2 Polarized Tuebinger Cohort (N=200)", (ox + 510, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)

        cv2.imwrite(str(save_path), canvas)

    fig3_path = FIGURES_DIR / "ph2_optical_color_channel_shift.png"
    render_color_shift_plot(fig3_path)

    # Copy to brain artifact directory
    brain_fig_dir = BRAIN_ARTIFACT_DIR / "figures"
    brain_fig_dir.mkdir(parents=True, exist_ok=True)
    for f in [fig1_path, fig2_path, fig3_path]:
        shutil.copy(f, brain_fig_dir / f.name)
    print("Copied all Step 7 figures to artifact directory.")

    print("\nControlled domain-shift ablation study completed successfully!")


if __name__ == "__main__":
    run_ablation_study()
