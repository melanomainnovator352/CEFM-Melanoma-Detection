"""
CEFM Step 4 — Production Feature Distribution and Clinical Scoring Integrity Audit
Counterfactual Input Sensitivity Analysis on target production image:
  data/uploads/5ef6da25eecc47c6a05bee924b2aa3f8.jpg
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.classification import (
    _prepare_abcde_tensor,
    _load_torch_model,
    get_vit_embedding,
    _load_multimodal_model,
    _load_m6_challenger_model,
)
from app.services.dual_scale import extract_dual_scale_views

def run_counterfactual_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing counterfactual analysis on {device}...")

    img_path = PROJECT_ROOT / "data" / "uploads" / "5ef6da25eecc47c6a05bee924b2aa3f8.jpg"
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = segment_lesion(img_path)
    features = extract_abcde_features(img_bgr, mask, pixels_per_mm=16.72)

    # Load models
    vit_model = _load_torch_model()
    multimodal_model = _load_multimodal_model(device=device)
    m6_model = _load_m6_challenger_model(device=device)

    # Import norm tensors
    import app.services.classification as cl
    norm_mean = cl._challenger_m6_norm_mean.to(device)
    norm_std = cl._challenger_m6_norm_std.to(device)

    from app.services.dual_scale import extract_dual_scale_embeddings_single
    dual_emb = extract_dual_scale_embeddings_single(
        img_rgb,
        mask=mask,
        vit_model=cl._classifier,
        processor=cl._processor,
        device=device,
    )
    g_emb = dual_emb["global_embedding"].to(device)
    if g_emb.ndim == 1:
        g_emb = g_emb.unsqueeze(0)
    l_emb = dual_emb["local_embedding"].to(device)
    if l_emb.ndim == 1:
        l_emb = l_emb.unsqueeze(0)

    # Base ABCDE tensor
    base_abcde = _prepare_abcde_tensor(features, device=device)
    print("Base ABCDE raw vector:", [round(x, 4) for x in base_abcde[0].tolist()])

    conditions = [
        ("Condition 1: Actual Calibrated (x11=0.598, x12=1.0)", 0.598, 1.0),
        ("Condition 2: Index 11 Zeroed (x11=0.0, x12=1.0)", 0.0, 1.0),
        ("Condition 3: Index 12 Zeroed (x11=0.598, x12=0.0)", 0.598, 0.0),
        ("Condition 4: Both Zeroed (x11=0.0, x12=0.0)", 0.0, 0.0),
    ]

    records = []

    for label, val11, val12 in conditions:
        c_vec = base_abcde.clone()
        c_vec[0, 11] = val11
        c_vec[0, 12] = val12

        # 1. Phase 1 Multimodal Fusion (unnormalized raw vector input)
        with torch.no_grad():
            p1_zc = multimodal_model.clinical_encoder(c_vec)
            p1_logits = multimodal_model(g_emb, c_vec)
            p1_probs = F.softmax(p1_logits, dim=-1)[0].cpu().numpy()
            p1_mel_prob = float(p1_probs[1])
            p1_pred = "mel" if p1_mel_prob >= 0.36 else "non_mel"

        # 2. M6 Dual-Scale Gated RCAG (normalized vector input)
        c_norm = (c_vec - norm_mean) / norm_std
        with torch.no_grad():
            zg = m6_model.global_encoder(g_emb)
            zl = m6_model.local_encoder(l_emb)
            zc = m6_model.clinical_encoder(c_norm)
            c_mod = m6_model.clinical_gate(zc)
            scale_weights = m6_model.scale_gate(torch.cat([zg, zl], dim=-1))
            blended_v = scale_weights * zg + (1.0 - scale_weights) * zl
            modulated_v = blended_v * (1.0 + c_mod)
            combined = torch.cat([modulated_v, zl, zc], dim=-1)
            fused = m6_model.fusion(combined)
            m6_logits = m6_model.classifier(fused)
            m6_probs = F.softmax(m6_logits, dim=-1)[0].cpu().numpy()
            m6_mel_prob = float(m6_probs[1])
            m6_pred_f1 = "mel" if m6_mel_prob >= 0.690 else "non_mel"
            m6_pred_spec = "mel" if m6_mel_prob >= 0.625 else "non_mel"

        record = {
            "condition": label,
            "index_11_raw": float(val11),
            "index_12_raw": float(val12),
            "m6_index_11_normalized": float(c_norm[0, 11].item()),
            "m6_index_12_normalized": float(c_norm[0, 12].item()),
            "p1_clinical_embedding_norm": float(torch.norm(p1_zc).item()),
            "p1_logit_0": float(p1_logits[0, 0].item()),
            "p1_logit_1": float(p1_logits[0, 1].item()),
            "p1_melanoma_prob": float(p1_mel_prob),
            "p1_prediction": p1_pred,
            "m6_clinical_embedding_norm": float(torch.norm(zc).item()),
            "m6_gate_c_mod_mean": float(c_mod.mean().item()),
            "m6_gate_c_mod_min": float(c_mod.min().item()),
            "m6_gate_c_mod_max": float(c_mod.max().item()),
            "m6_amplification_mean": float((1.0 + c_mod).mean().item()),
            "m6_logit_0": float(m6_logits[0, 0].item()),
            "m6_logit_1": float(m6_logits[0, 1].item()),
            "m6_melanoma_prob": float(m6_mel_prob),
            "m6_pred_f1_0_690": m6_pred_f1,
            "m6_pred_spec_0_625": m6_pred_spec,
        }
        records.append(record)

    df = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "reports" / "CEFM_STEP4_COUNTERFACTUAL_INPUT_ANALYSIS.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved counterfactual analysis to {out_csv}")
    print("\nResults Summary:")
    for r in records:
        print(f"--- {r['condition']} ---")
        print(f"  P1: logits=[{r['p1_logit_0']:.4f}, {r['p1_logit_1']:.4f}], mel_prob={r['p1_melanoma_prob']*100:.2f}%, pred={r['p1_prediction']}")
        print(f"  M6 c_norm: [11]={r['m6_index_11_normalized']:.1f}, [12]={r['m6_index_12_normalized']:.1f}")
        print(f"  M6 gate: c_mod_mean={r['m6_gate_c_mod_mean']:.4f}, amp_mean={r['m6_amplification_mean']:.4f}")
        print(f"  M6: logits=[{r['m6_logit_0']:.4f}, {r['m6_logit_1']:.4f}], mel_prob={r['m6_melanoma_prob']*100:.4f}%, pred_f1={r['m6_pred_f1_0_690']}, pred_spec={r['m6_pred_spec_0_625']}")

if __name__ == "__main__":
    run_counterfactual_analysis()
