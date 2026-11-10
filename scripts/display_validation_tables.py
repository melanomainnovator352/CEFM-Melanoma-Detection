"""Display clean formatted tables of research validation results."""
import json
from pathlib import Path

results_file = Path("reports/phase2_m6_research_validation_results.json")
data = json.load(open(results_file, encoding="utf-8"))
models = data["eval_models"]

print("=" * 115)
print("TABLE 1: VALIDATION-CALIBRATED F1 OPERATING POINTS (N=1,504 Test Samples: 166 Mel, 1,338 Non-Mel)")
print("=" * 115)
header = f"{'Model Identity':<25} | {'Thresh':<6} | {'ROC-AUC':<7} | {'PR-AUC':<7} | {'Sens':<7} | {'Spec':<7} | {'Prec':<7} | {'F1':<7} | {'BAcc':<7} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4}"
print(header)
print("-" * 115)

model_order = ["Phase1_ViT", "Phase1_Fusion", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
for k in model_order:
    d = models[k]
    m = d.get("metrics_f1") or d.get("metrics_calibrated_f1")
    cm = m["confusion_matrix"]
    name = d["name"][:25]
    print(f"{name:<25} | {m['threshold']:<6.3f} | {d['roc_auc']:<7.4f} | {d['pr_auc']:<7.4f} | {m['sensitivity']*100:<6.2f}% | {m['specificity']*100:<6.2f}% | {m['precision']*100:<6.2f}% | {m['f1_score']:<7.4f} | {m['balanced_accuracy']:<7.4f} | {cm['tp']:<4} | {cm['fp']:<4} | {cm['tn']:<4} | {cm['fn']:<4}")

print("\n" + "=" * 115)
print("TABLE 2: VALIDATION-CALIBRATED SPECIFICITY-TARGET OPERATING POINTS")
print("=" * 115)
print(header)
print("-" * 115)

for k in model_order:
    d = models[k]
    m = d.get("metrics_clinical") or d.get("metrics_matched_spec")
    cm = m["confusion_matrix"]
    name = d["name"][:25]
    print(f"{name:<25} | {m['threshold']:<6.3f} | {d['roc_auc']:<7.4f} | {d['pr_auc']:<7.4f} | {m['sensitivity']*100:<6.2f}% | {m['specificity']*100:<6.2f}% | {m['precision']*100:<6.2f}% | {m['f1_score']:<7.4f} | {m['balanced_accuracy']:<7.4f} | {cm['tp']:<4} | {cm['fp']:<4} | {cm['tn']:<4} | {cm['fn']:<4}")

print("\n" + "=" * 115)
print("TABLE 3: PAIRED BOOTSTRAP STATISTICAL DIFFERENCES vs. M6 (1,000 Resamples, Seed=42, 95% CI)")
print("=" * 115)
paired = data["paired_bootstrap_differences"]
p_header = f"{'Comparison':<30} | {'Metric':<18} | {'Mean Diff':<10} | {'95% CI Lower':<12} | {'95% CI Upper':<12} | {'Sig (p<0.05)'}"
print(p_header)
print("-" * 115)
for comp_name, comp_dict in paired.items():
    print(f"[{comp_name}]")
    for met in ["sensitivity", "specificity", "f1_score", "balanced_accuracy", "roc_auc", "pr_auc"]:
        r = comp_dict[met]
        sig_str = "YES (p<0.05)" if r["significant_alpha_05"] else "No (CI includes 0)"
        print(f"  {'':<28} | {met:<18} | {r['mean_difference']:<+10.4f} | {r['ci_lower']:<+12.4f} | {r['ci_upper']:<+12.4f} | {sig_str}")
    print("-" * 115)

print("\n" + "=" * 115)
print("TABLE 4: FALSE-NEGATIVE AUDIT COMPARISON ACROSS MODELS")
print("=" * 115)
fn_data = data["false_negative_analysis"]
fn_header = f"{'Model & Operating Point':<38} | {'FN Count':<8} | {'FN Rate':<8} | {'Mean FN Prob':<12} | {'Median FN Prob':<14} | {'Single Patch':<12} | {'Multi Patch'}"
print(fn_header)
print("-" * 115)
for k, v in fn_data.items():
    lbl = v["label"][:38]
    p_stats = v["probability_stats"]
    p_counts = v["by_patch_count"]
    print(f"{lbl:<38} | {v['false_negative_count']:<8} | {v['false_negative_rate']*100:<7.2f}% | {p_stats['mean']:<12.4f} | {p_stats['median']:<14.4f} | {p_counts['single_patch']:<12} | {p_counts['multi_patch']}")

