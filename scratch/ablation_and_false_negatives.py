import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

# 1. Load test features & metadata
test_features = torch.load('data/multimodal_cache/test_features.pt', map_location='cpu', weights_only=False)
val_features = torch.load('data/multimodal_cache/validation_features.pt', map_location='cpu', weights_only=False)
test_df = pd.read_csv('data/HAM10000/metadata/test.csv')
val_df = pd.read_csv('data/HAM10000/metadata/validation.csv')

y_val = val_features['labels'].numpy()
y_test = test_features['labels'].numpy()

from app.ml.multimodal.fusion import CEFMMultimodalFusion

def calc_metrics(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    acc = (tp + tn) / len(y_true)
    roc = roc_auc_score(y_true, y_prob)
    p_prec, p_rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(p_rec, p_prec)
    return {
        'thresh': round(thresh, 4), 'acc': round(acc, 4), 'sens': round(sens, 4), 'spec': round(spec, 4),
        'prec': round(prec, 4), 'f1': round(f1, 4), 'roc_auc': round(roc, 4), 'pr_auc': round(pr_auc, 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def find_best_val_threshold(y_v, p_v):
    best_t = 0.5
    best_bacc = 0.0
    for t in np.linspace(0.1, 0.9, 81):
        m = calc_metrics(y_v, p_v, t)
        bacc = (m['sens'] + m['spec']) / 2.0
        if bacc > best_bacc:
            best_bacc = bacc
            best_t = t
    return best_t

# Experiment 1: ViT Only (Full Image)
df_vit_full = pd.read_csv('models/classification/evaluation/vit_test_predictions.csv')
p_test_e1_full = df_vit_full['melanoma_probability'].values

# Experiment 2: ViT Only (Segmentation ROI Crop)
df_vit_crop = pd.read_csv('models/experiments/E1_vit_only/predictions.csv')
p_test_e1_crop = df_vit_crop['probability_melanoma'].values

# Experiment 3: ViT + ABCDE (E2)
df_e2 = pd.read_csv('models/experiments/E2_vit_abc/predictions.csv')
p_test_e2 = df_e2['probability_melanoma'].values

# Experiment 4: ViT + Seg + ABCDE Multimodal Fusion (E3 / Production)
df_e3 = pd.read_csv('models/experiments/E3_vit_seg_abc/predictions.csv')
p_test_e3 = df_e3['probability_melanoma'].values

print("=== ABLATION EXPERIMENTS EVALUATION (Independent Test Set: 1504 Samples) ===")
# Calibrated thresholds from validation:
# ViT Full: 0.80
# ViT Crop: 0.80
# ViT + ABC: 0.38
# ViT + Seg + ABC (Multimodal): 0.36
res_full = calc_metrics(y_test, p_test_e1_full, 0.80)
res_crop = calc_metrics(y_test, p_test_e1_crop, 0.80)
res_e2   = calc_metrics(y_test, p_test_e2, 0.38)
res_e3   = calc_metrics(y_test, p_test_e3, 0.36)

print(f"(A) ViT Only (Full Image, T=0.80):       Acc={res_full['acc']*100:.2f}%, Sens={res_full['sens']*100:.2f}%, Spec={res_full['spec']*100:.2f}%, Prec={res_full['prec']*100:.2f}%, F1={res_full['f1']:.4f}, ROC={res_full['roc_auc']:.4f}, PR={res_full['pr_auc']:.4f}, FN={res_full['fn']}")
print(f"(B) ViT Only (ROI Crop, T=0.80):         Acc={res_crop['acc']*100:.2f}%, Sens={res_crop['sens']*100:.2f}%, Spec={res_crop['spec']*100:.2f}%, Prec={res_crop['prec']*100:.2f}%, F1={res_crop['f1']:.4f}, ROC={res_crop['roc_auc']:.4f}, PR={res_crop['pr_auc']:.4f}, FN={res_crop['fn']}")
print(f"(C) ViT + ABCDE (No Seg Crop, T=0.38):   Acc={res_e2['acc']*100:.2f}%, Sens={res_e2['sens']*100:.2f}%, Spec={res_e2['spec']*100:.2f}%, Prec={res_e2['prec']*100:.2f}%, F1={res_e2['f1']:.4f}, ROC={res_e2['roc_auc']:.4f}, PR={res_e2['pr_auc']:.4f}, FN={res_e2['fn']}")
print(f"(D) ViT + Seg ROI + ABCDE (Prod, T=0.36): Acc={res_e3['acc']*100:.2f}%, Sens={res_e3['sens']*100:.2f}%, Spec={res_e3['spec']*100:.2f}%, Prec={res_e3['prec']*100:.2f}%, F1={res_e3['f1']:.4f}, ROC={res_e3['roc_auc']:.4f}, PR={res_e3['pr_auc']:.4f}, FN={res_e3['fn']}")

# False Negative Analysis on Test Set
print("\n=== FALSE NEGATIVE SUBTYPE & DEMOGRAPHIC ANALYSIS ===")
test_df['y_true'] = y_test
test_df['p_vit'] = p_test_e1_crop
test_df['p_mm'] = p_test_e3
test_df['fn_vit'] = (test_df['y_true'] == 1) & (test_df['p_vit'] < 0.80)
test_df['fn_mm'] = (test_df['y_true'] == 1) & (test_df['p_mm'] < 0.36)

print(f"Total Test Melanomas: {y_test.sum()}")
print(f"ViT False Negatives: {test_df['fn_vit'].sum()}")
print(f"Multimodal False Negatives: {test_df['fn_mm'].sum()}")

# Breakdown by dx_type
print("\nFalse Negatives by dx_type:")
for dxt in test_df[test_df['y_true'] == 1]['dx_type'].unique():
    sub = test_df[(test_df['y_true'] == 1) & (test_df['dx_type'] == dxt)]
    fn_v = sub['fn_vit'].sum()
    fn_m = sub['fn_mm'].sum()
    print(f"  {dxt:15s}: N={len(sub)}, ViT FN={fn_v} ({fn_v/len(sub)*100:.1f}%), MM FN={fn_m} ({fn_m/len(sub)*100:.1f}%)")

# Breakdown by localization
print("\nFalse Negatives by Anatomical Site:")
for loc, grp in test_df[test_df['y_true'] == 1].groupby('localization'):
    fn_v = grp['fn_vit'].sum()
    fn_m = grp['fn_mm'].sum()
    print(f"  {loc:20s}: N={len(grp)}, ViT FN={fn_v} ({fn_v/len(grp)*100:.1f}%), MM FN={fn_m} ({fn_m/len(grp)*100:.1f}%)")

# Age distribution of FN vs TP
mel_df = test_df[test_df['y_true'] == 1]
print("\nAge Comparison:")
print(f"  True Positives Mean Age: {mel_df[~mel_df['fn_mm']]['age'].mean():.1f} yrs")
print(f"  False Negatives Mean Age: {mel_df[mel_df['fn_mm']]['age'].mean():.1f} yrs")

