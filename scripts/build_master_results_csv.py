"""
Script to compile reports/CEFM_PHASE2_EXPERIMENTAL_RESULTS_MASTER.csv
Aggregates all verified experimental metrics across Phase 2 Steps 1-8.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

records = []

# 1. Step 2: Internal Validation (M1 - M7 on HAM10000 Validation Split N=1533)
with open(REPORTS_DIR / "phase2_m1_m7_comparative_benchmarks.json", "r") as f:
    m1_m7_val = json.load(f)

for m_id, data in m1_m7_val.items():
    # Calibrated F1
    m_f1 = data.get("metrics_calibrated_f1", {})
    records.append({
        "phase_step": "Step 2",
        "experiment_id": f"{m_id}_val_f1",
        "model_id": m_id,
        "model_name": data["name"],
        "dataset": "HAM10000",
        "split": "validation",
        "sample_count": 1533,
        "operating_point": "Validation F1-Calibrated",
        "threshold": data["calibrated_threshold_f1"],
        "roc_auc": m_f1.get("roc_auc"),
        "pr_auc": m_f1.get("pr_auc"),
        "sensitivity": m_f1.get("sensitivity"),
        "specificity": m_f1.get("specificity"),
        "precision": m_f1.get("precision"),
        "f1_score": m_f1.get("f1_score"),
        "accuracy": m_f1.get("accuracy"),
        "balanced_accuracy": m_f1.get("balanced_accuracy"),
        "tp": m_f1.get("confusion_matrix", {}).get("tp"),
        "fp": m_f1.get("confusion_matrix", {}).get("fp"),
        "tn": m_f1.get("confusion_matrix", {}).get("tn"),
        "fn": m_f1.get("confusion_matrix", {}).get("fn"),
        "statistical_status": "Internal Validation Benchmark",
    })

    # Matched Specificity (90% Spec Target)
    m_spec = data.get("metrics_matched_spec", {})
    records.append({
        "phase_step": "Step 2",
        "experiment_id": f"{m_id}_val_spec90",
        "model_id": m_id,
        "model_name": data["name"],
        "dataset": "HAM10000",
        "split": "validation",
        "sample_count": 1533,
        "operating_point": "Validation Matched 90% Specificity",
        "threshold": data["calibrated_threshold_matched_spec"],
        "roc_auc": m_spec.get("roc_auc"),
        "pr_auc": m_spec.get("pr_auc"),
        "sensitivity": m_spec.get("sensitivity"),
        "specificity": m_spec.get("specificity"),
        "precision": m_spec.get("precision"),
        "f1_score": m_spec.get("f1_score"),
        "accuracy": m_spec.get("accuracy"),
        "balanced_accuracy": m_spec.get("balanced_accuracy"),
        "tp": m_spec.get("confusion_matrix", {}).get("tp"),
        "fp": m_spec.get("confusion_matrix", {}).get("fp"),
        "tn": m_spec.get("confusion_matrix", {}).get("tn"),
        "fn": m_spec.get("confusion_matrix", {}).get("fn"),
        "statistical_status": "Internal Validation Benchmark",
    })

# 2. Step 4: Internal Test Evaluation (Phase 1 Baselines & M1-M7 on HAM10000 Test Split N=1504)
with open(REPORTS_DIR / "phase2_m6_research_validation_results.json", "r") as f:
    res_val = json.load(f)

for m_id, data in res_val["eval_models"].items():
    # If M1-M7
    if "metrics_calibrated_f1" in data:
        mf1 = data["metrics_calibrated_f1"]
        records.append({
            "phase_step": "Step 4",
            "experiment_id": f"{m_id}_test_f1",
            "model_id": m_id,
            "model_name": data["name"],
            "dataset": "HAM10000",
            "split": "test",
            "sample_count": 1504,
            "operating_point": "F1-Calibrated",
            "threshold": data.get("calibrated_threshold_f1", 0.69),
            "roc_auc": data.get("roc_auc"),
            "pr_auc": data.get("pr_auc"),
            "sensitivity": mf1.get("sensitivity"),
            "specificity": mf1.get("specificity"),
            "precision": mf1.get("precision"),
            "f1_score": mf1.get("f1_score"),
            "accuracy": mf1.get("accuracy"),
            "balanced_accuracy": mf1.get("balanced_accuracy"),
            "tp": mf1.get("confusion_matrix", {}).get("tp"),
            "fp": mf1.get("confusion_matrix", {}).get("fp"),
            "tn": mf1.get("confusion_matrix", {}).get("tn"),
            "fn": mf1.get("confusion_matrix", {}).get("fn"),
            "statistical_status": "Internal Test Benchmark",
        })
    if "metrics_matched_spec" in data:
        mspec = data["metrics_matched_spec"]
        records.append({
            "phase_step": "Step 4",
            "experiment_id": f"{m_id}_test_spec90",
            "model_id": m_id,
            "model_name": data["name"],
            "dataset": "HAM10000",
            "split": "test",
            "sample_count": 1504,
            "operating_point": "Matched 90% Specificity",
            "threshold": data.get("calibrated_threshold_matched_spec", 0.625),
            "roc_auc": data.get("roc_auc"),
            "pr_auc": data.get("pr_auc"),
            "sensitivity": mspec.get("sensitivity"),
            "specificity": mspec.get("specificity"),
            "precision": mspec.get("precision"),
            "f1_score": mspec.get("f1_score"),
            "accuracy": mspec.get("accuracy"),
            "balanced_accuracy": mspec.get("balanced_accuracy"),
            "tp": mspec.get("confusion_matrix", {}).get("tp"),
            "fp": mspec.get("confusion_matrix", {}).get("fp"),
            "tn": mspec.get("confusion_matrix", {}).get("tn"),
            "fn": mspec.get("confusion_matrix", {}).get("fn"),
            "statistical_status": "Internal Test Benchmark",
        })
    # If Phase1 models
    if "metrics_clinical" in data:
        mclin = data["metrics_clinical"]
        records.append({
            "phase_step": "Step 4",
            "experiment_id": f"{m_id}_test_clinical",
            "model_id": m_id,
            "model_name": data["name"],
            "dataset": "HAM10000",
            "split": "test",
            "sample_count": 1504,
            "operating_point": "Clinical Screening Operating Point",
            "threshold": mclin.get("threshold"),
            "roc_auc": data.get("roc_auc"),
            "pr_auc": data.get("pr_auc"),
            "sensitivity": mclin.get("sensitivity"),
            "specificity": mclin.get("specificity"),
            "precision": mclin.get("precision"),
            "f1_score": mclin.get("f1_score"),
            "accuracy": mclin.get("accuracy"),
            "balanced_accuracy": mclin.get("balanced_accuracy"),
            "tp": mclin.get("confusion_matrix", {}).get("tp"),
            "fp": mclin.get("confusion_matrix", {}).get("fp"),
            "tn": mclin.get("confusion_matrix", {}).get("tn"),
            "fn": mclin.get("confusion_matrix", {}).get("fn"),
            "statistical_status": "Internal Test Production Baseline",
        })
    if "metrics_f1" in data:
        mf1 = data["metrics_f1"]
        records.append({
            "phase_step": "Step 4",
            "experiment_id": f"{m_id}_test_f1",
            "model_id": m_id,
            "model_name": data["name"],
            "dataset": "HAM10000",
            "split": "test",
            "sample_count": 1504,
            "operating_point": "F1-Calibrated",
            "threshold": mf1.get("threshold"),
            "roc_auc": data.get("roc_auc"),
            "pr_auc": data.get("pr_auc"),
            "sensitivity": mf1.get("sensitivity"),
            "specificity": mf1.get("specificity"),
            "precision": mf1.get("precision"),
            "f1_score": mf1.get("f1_score"),
            "accuracy": mf1.get("accuracy"),
            "balanced_accuracy": mf1.get("balanced_accuracy"),
            "tp": mf1.get("confusion_matrix", {}).get("tp"),
            "fp": mf1.get("confusion_matrix", {}).get("fp"),
            "tn": mf1.get("confusion_matrix", {}).get("tn"),
            "fn": mf1.get("confusion_matrix", {}).get("fn"),
            "statistical_status": "Internal Test Production Baseline",
        })

# 3. Step 5: External Validation on PH2 Benchmark (N=200)
df_step5_metrics = pd.read_csv(REPORTS_DIR / "CEFM_PHASE2_STEP5_METRICS.csv")
for _, r in df_step5_metrics.iterrows():
    m_name = r.get("model", r.get("model_name"))
    op_name = str(r["operating_point"])
    records.append({
        "phase_step": "Step 5",
        "experiment_id": f"PH2_{str(m_name).replace(' ', '_')}_{op_name.replace(' ', '_')}",
        "model_id": m_name,
        "model_name": m_name,
        "dataset": "PH2",
        "split": "external_test",
        "sample_count": 200,
        "operating_point": op_name,
        "threshold": r["threshold"],
        "roc_auc": r["roc_auc"],
        "pr_auc": r["pr_auc"],
        "sensitivity": r["sensitivity"],
        "specificity": r["specificity"],
        "precision": r["precision"],
        "f1_score": r["f1_score"],
        "accuracy": r["accuracy"],
        "balanced_accuracy": r["balanced_accuracy"],
        "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
        "statistical_status": "External Validation Benchmark",
    })

# 4. Step 7 & 8: Reconciled Step 7 Ablation Variants (A to F on PH2 N=200)
# Load df_reconciled
df_rec = pd.read_csv(REPORTS_DIR / "CEFM_PHASE2_STEP8_RECONCILED_METRICS.csv")
for _, r in df_rec.iterrows():
    records.append({
        "phase_step": "Step 8 Audit",
        "experiment_id": f"Reconciled_{r['evaluation_source'][:15].replace(' ', '_')}_{r['threshold']}",
        "model_id": "M6",
        "model_name": f"M6 ({r['mask_configuration']})",
        "dataset": "PH2",
        "split": "external_test",
        "sample_count": 200,
        "operating_point": f"Threshold {r['threshold']} ({r['feature_pipeline']})",
        "threshold": r["threshold"],
        "roc_auc": r["roc_auc"],
        "pr_auc": r["pr_auc"],
        "sensitivity": r["sensitivity"],
        "specificity": r["specificity"],
        "precision": None,
        "f1_score": r["f1_score"],
        "accuracy": None,
        "balanced_accuracy": r["balanced_accuracy"],
        "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
        "statistical_status": "Reconciled Audit Result",
    })

df_master = pd.DataFrame(records)
out_csv = REPORTS_DIR / "CEFM_PHASE2_EXPERIMENTAL_RESULTS_MASTER.csv"
df_master.to_csv(out_csv, index=False)
print(f"Generated {out_csv} with {len(df_master)} master experiment rows.")
