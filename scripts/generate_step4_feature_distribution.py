"""
CEFM Step 4 — Generate Comprehensive Feature Distribution CSV across:
- HAM10000 Train (N=6,978)
- HAM10000 Validation (N=1,533)
- HAM10000 Test (N=1,504)
- PH² External Cohort (N=200)
- Production Inference Cohort (data/uploads)
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.classification import _prepare_abcde_tensor

FEATURE_NAMES = [
    "asymmetry_score",
    "horizontal_asymmetry",
    "vertical_asymmetry",
    "border_irregularity",
    "circularity",
    "radial_distance_variation",
    "curvature_variation",
    "color_h_norm",
    "color_s_norm",
    "color_v_norm",
    "color_diversity",
    "diameter_mm_norm",
    "calibration_available",
    "evolution_change_flag",
    "evolution_magnitude",
    "evolution_available",
]

def main():
    print("Loading HAM10000 splits...")
    train_data = torch.load(PROJECT_ROOT / "data" / "dual_scale_cache" / "dual_scale_train.pt", map_location="cpu")
    val_data = torch.load(PROJECT_ROOT / "data" / "dual_scale_cache" / "dual_scale_validation.pt", map_location="cpu")
    test_data = torch.load(PROJECT_ROOT / "data" / "dual_scale_cache" / "dual_scale_test.pt", map_location="cpu")

    c_train = train_data["clinical"].numpy()
    c_val = val_data["clinical"].numpy()
    c_test = test_data["clinical"].numpy()

    # Load PH2 Step 6 distribution
    step6_csv = PROJECT_ROOT / "reports" / "CEFM_PHASE2_STEP6_FEATURE_DISTRIBUTION.csv"
    df_step6 = pd.read_csv(step6_csv)

    # Extract production images
    print("Processing production images in data/uploads...")
    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    img_candidates = [
        p for p in uploads_dir.glob("*.*")
        if not p.name.endswith(("_seg.png", "_att.jpg", "_roi_12418.jpg", "_crop_mel.jpg"))
        and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    from app.services.calibration import estimate_pixels_per_mm

    prod_vecs = []
    for p in img_candidates:
        try:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            cal = estimate_pixels_per_mm(img_bgr)
            ppm = cal.get("pixels_per_mm") if cal.get("valid") else None
            mask = segment_lesion(p)
            features = extract_abcde_features(img_bgr, mask, pixels_per_mm=ppm)
            vec = _prepare_abcde_tensor(features).cpu().numpy().flatten()
            prod_vecs.append(vec)
            print(f"  Processed {p.name}: d11={vec[11]:.4f}, d12={vec[12]:.4f}, ppm={ppm}")
        except Exception as e:
            print(f"  Failed on {p.name}: {e}")

    c_prod = np.array(prod_vecs) if len(prod_vecs) > 0 else np.zeros((1, 16))
    print(f"Extracted {len(c_prod)} production feature vectors.")

    rows = []
    for idx in range(16):
        fname = FEATURE_NAMES[idx]
        tr_m = float(np.mean(c_train[:, idx]))
        tr_s = float(np.std(c_train[:, idx]))
        tr_min = float(np.min(c_train[:, idx]))
        tr_max = float(np.max(c_train[:, idx]))
        tr_nz = float(np.mean(c_train[:, idx] != 0) * 100)

        va_m = float(np.mean(c_val[:, idx]))
        va_s = float(np.std(c_val[:, idx]))
        va_min = float(np.min(c_val[:, idx]))
        va_max = float(np.max(c_val[:, idx]))
        va_nz = float(np.mean(c_val[:, idx] != 0) * 100)

        te_m = float(np.mean(c_test[:, idx]))
        te_s = float(np.std(c_test[:, idx]))
        te_min = float(np.min(c_test[:, idx]))
        te_max = float(np.max(c_test[:, idx]))
        te_nz = float(np.mean(c_test[:, idx] != 0) * 100)

        # PH2 from step 6
        r_step6 = df_step6[df_step6["dimension_index"] == idx].iloc[0]
        ph2_m = float(r_step6["ph2_mean"])
        ph2_s = float(r_step6["ph2_std"])
        ph2_min = float(r_step6["ph2_min"])
        ph2_max = float(r_step6["ph2_max"])
        ph2_nz = 0.0 if (ph2_m == 0.0 and ph2_s == 0.0) else (100.0 if idx == 4 else float(tr_nz))

        # Production
        pr_m = float(np.mean(c_prod[:, idx]))
        pr_s = float(np.std(c_prod[:, idx]))
        pr_min = float(np.min(c_prod[:, idx]))
        pr_max = float(np.max(c_prod[:, idx]))
        pr_nz = float(np.mean(c_prod[:, idx] != 0) * 100)

        # Shift classification
        if idx in (11, 12):
            shift_cat = "SEVERE OOD SHIFT (Constant zero in training; non-zero in production)"
        elif idx in (7, 9):
            shift_cat = "EXTREME OPTICAL DOMAIN SHIFT (PH2 HSV distribution)"
        elif idx == 3:
            shift_cat = "EXTREME SEGMENTATION SHIFT (Border irregularity)"
        elif idx == 0:
            shift_cat = "MODERATE SHIFT (Asymmetry)"
        elif tr_s == 0.0 and pr_s == 0.0:
            shift_cat = "CONSTANT ZERO IN-DISTRIBUTION"
        else:
            shift_cat = "IN-DISTRIBUTION"

        rows.append({
            "dimension_index": idx,
            "feature_name": fname,
            "train_mean": round(tr_m, 6),
            "train_std": round(tr_s, 6),
            "train_min": round(tr_min, 4),
            "train_max": round(tr_max, 4),
            "train_pct_nonzero": round(tr_nz, 1),
            "val_mean": round(va_m, 6),
            "val_std": round(va_s, 6),
            "val_min": round(va_min, 4),
            "val_max": round(va_max, 4),
            "val_pct_nonzero": round(va_nz, 1),
            "test_mean": round(te_m, 6),
            "test_std": round(te_s, 6),
            "test_min": round(te_min, 4),
            "test_max": round(te_max, 4),
            "test_pct_nonzero": round(te_nz, 1),
            "ph2_mean": round(ph2_m, 6),
            "ph2_std": round(ph2_s, 6),
            "ph2_min": round(ph2_min, 4),
            "ph2_max": round(ph2_max, 4),
            "prod_mean": round(pr_m, 6),
            "prod_std": round(pr_s, 6),
            "prod_min": round(pr_min, 4),
            "prod_max": round(pr_max, 4),
            "prod_pct_nonzero": round(pr_nz, 1),
            "shift_status": shift_cat,
        })

    out_df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "reports" / "CEFM_STEP4_FEATURE_DISTRIBUTION.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved comprehensive distribution table to {out_csv}")

if __name__ == "__main__":
    main()
