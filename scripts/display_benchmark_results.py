"""
Display Phase 2 M1–M7 Benchmark Results and Ablations.
"""
import json
from pathlib import Path

with open("reports/phase2_m1_m7_comparative_benchmarks.json") as f:
    data = json.load(f)

print("=" * 115)
print(f"{'Model':<5} | {'Architecture':<35} | {'ROC-AUC':<7} | {'PR-AUC':<7} | {'Sens (%) [95% CI]':<20} | {'Spec (%)':<8} | {'F1 (%)':<8} | {'FN':<4}")
print("=" * 115)

for m_id, res in data.items():
    name = res["name"]
    m = res["metrics_calibrated_f1"]
    ci = res["bootstrap_95ci_f1"]
    sens_str = f"{m['sensitivity']*100:.1f}% [{ci['sensitivity'][0]*100:.1f}-{ci['sensitivity'][1]*100:.1f}]"
    print(f"{m_id:<5} | {name:<35} | {m['roc_auc']:<7.4f} | {m['pr_auc']:<7.4f} | {sens_str:<20} | {m['specificity']*100:<8.1f} | {m['f1_score']*100:<8.1f} | {m['false_negatives']:<4}")

print("=" * 115)
print("\nMATCHED SPECIFICITY COMPARISON (Validation Target: 90.0% Specificity):")
print("=" * 115)
print(f"{'Model':<5} | {'Architecture':<35} | {'Thresh':<7} | {'Sens (%)':<9} | {'Spec (%)':<9} | {'Prec (%)':<9} | {'F1 (%)':<8} | {'FN':<4}")
print("=" * 115)
for m_id, res in data.items():
    name = res["name"]
    m = res["metrics_matched_spec"]
    print(f"{m_id:<5} | {name:<35} | {m['threshold']:<7.4f} | {m['sensitivity']*100:<9.2f} | {m['specificity']*100:<9.2f} | {m['precision']*100:<9.2f} | {m['f1_score']*100:<8.2f} | {m['false_negatives']:<4}")

print("=" * 115)
print("\nDEFAULT OPERATING POINT COMPARISON (Threshold: 0.50):")
print("=" * 115)
print(f"{'Model':<5} | {'Architecture':<35} | {'Sens (%)':<9} | {'Spec (%)':<9} | {'Prec (%)':<9} | {'F1 (%)':<8} | {'FN':<4}")
print("=" * 115)
for m_id, res in data.items():
    name = res["name"]
    m = res["metrics_default_05"]
    print(f"{m_id:<5} | {name:<35} | {m['sensitivity']*100:<9.2f} | {m['specificity']*100:<9.2f} | {m['precision']*100:<9.2f} | {m['f1_score']*100:<8.2f} | {m['false_negatives']:<4}")

print("=" * 115)
print("\nCASE AUDIT: ISIC_0012418 PREDICTIONS ACROSS MODELS (Ground Truth: Melanoma):")
print("=" * 115)
for m_id, res in data.items():
    audit = res.get("isic_0012418_audit")
    if audit:
        print(f"  [{m_id}] {res['name']:<35}: Prob = {audit['melanoma_probability']*100:6.2f}% | Calibrated Pred = {audit['predicted_class_f1']:<8} | Matched Spec Pred = {audit['predicted_class_spec']:<8}")

print("=" * 115)

