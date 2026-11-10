"""
Audits ISIC_0012418 across all 7 trained models M1–M7.
"""
import json
import sys
import torch
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.multimodal.dual_scale_models import MODEL_REGISTRY
import app.services.classification as cl
from app.services.segmentation import _get_unet_model
from scripts.cache_dual_scale_features import segment_batch_unet, extract_vit_batch
from app.services.dual_scale import extract_dual_scale_views
from app.services.analysis import extract_abcde_features
from app.services.abcde.vector import build_abcde_vector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cl._load_torch_model()
processor = cl._processor
vit_model = cl._classifier.to(DEVICE)
unet = _get_unet_model().to(DEVICE)
unet.eval()

img_path = Path('D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg')
img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

mask = segment_batch_unet([img_rgb], unet, DEVICE)[0]
views = extract_dual_scale_views(img_rgb, mask)

g_emb = extract_vit_batch([views['global_image']], vit_model, processor, DEVICE)
patch_imgs = [p['image'] for p in views['local_patches']]
l_embs = extract_vit_batch(patch_imgs, vit_model, processor, DEVICE)
w = torch.ones(len(l_embs))
w[0] = 2.0
w = (w / w.sum()).unsqueeze(1)
l_emb = (l_embs * w).sum(dim=0, keepdim=True)

abcde = extract_abcde_features(img_rgb, mask)
clin = build_abcde_vector(abcde).to_tensor().unsqueeze(0)

train_cache = torch.load('data/dual_scale_cache/dual_scale_train.pt', map_location='cpu')
mean = train_cache['clinical'].mean(dim=0, keepdim=True)
std = train_cache['clinical'].std(dim=0, keepdim=True).clamp(min=1e-6)
clin_norm = (clin - mean) / std

with open('reports/phase2_m1_m7_comparative_benchmarks.json') as f:
    benchmarks = json.load(f)

print('=' * 95)
print('CASE EVALUATION: ISIC_0012418 (Audited Ground Truth: MELANOMA)')
print('=' * 95)

for m_id in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']:
    weights_path = Path(f'models/experiments/{m_id}/weights.pt')
    model = MODEL_REGISTRY[m_id]['class']().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        p = model.predict_proba(g_emb.to(DEVICE), l_emb.to(DEVICE), clin_norm.to(DEVICE))[0, 1].item()

    t_f1 = benchmarks[m_id]['calibrated_threshold_f1']
    t_spec = benchmarks[m_id]['calibrated_threshold_matched_spec']
    pred_f1 = 'MEL' if p >= t_f1 else 'NON_MEL'
    pred_spec = 'MEL' if p >= t_spec else 'NON_MEL'

    benchmarks[m_id]['isic_0012418_audit'] = {
        'image_id': 'ISIC_0012418',
        'ground_truth': 1,
        'melanoma_probability': round(p, 4),
        'calibrated_f1_threshold': t_f1,
        'calibrated_f1_prediction': pred_f1,
        'matched_spec_threshold': t_spec,
        'matched_spec_prediction': pred_spec,
    }

    m_name = benchmarks[m_id]['name']
    print(f"{m_id} ({m_name:<38}): Prob = {p*100:6.2f}% | Pred@F1({t_f1:.2f}) = {pred_f1:<7} | Pred@Spec({t_spec:.2f}) = {pred_spec}")

with open('reports/phase2_m1_m7_comparative_benchmarks.json', 'w') as f:
    json.dump(benchmarks, f, indent=2)

print('=' * 95)
print('Audit results saved to reports/phase2_m1_m7_comparative_benchmarks.json')
