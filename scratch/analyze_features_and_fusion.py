import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
import cv2

train_data = torch.load('data/multimodal_cache/train_features.pt', map_location='cpu', weights_only=False)
train_visual = train_data['visual']
train_clinical = train_data['clinical'].numpy()  # (6978, 16)
train_labels = train_data['labels'].numpy()

from app.services.classification import get_vit_embedding, classify
from app.services.segmentation import segment_lesion
from app.services.calibration import estimate_pixels_per_mm
from app.services.analysis import extract_abcde_features
from app.services.abcde.vector import build_abcde_vector
from app.ml.multimodal.fusion import CEFMMultimodalFusion

img_path = 'D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg'
img = cv2.imread(img_path)
cal = estimate_pixels_per_mm(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
exclude_poly = cal.get('exclude_polygon') or cal.get('corners')
mask = segment_lesion(img_path, exclude_polygon=exclude_poly)

ppm = cal.get('pixels_per_mm')
features = extract_abcde_features(img, mask, pixels_per_mm=ppm)

color = features['color']
diameter = features['diameter']
flat = {
    'asymmetry': float(features['asymmetry']),
    'border_irregularity': float(features['border_irregularity']),
    'color_h': float(color.get('h', 0.0)),
    'color_s': float(color.get('s', 0.0)),
    'color_v': float(color.get('v', 0.0)),
    'color_index': float(color.get('index', 0.0)),
    'diameter_pixels': float(features['diameter_pixels']),
    'diameter_mm': float(diameter.get('max_diameter_mm', 0.0) or 0.0),
    'diameter_mm_available': bool(diameter.get('available', False)),
}

vec = build_abcde_vector(flat)
vec_t = vec.to_tensor().numpy()

feature_names = [
    'asymmetry', 'border_irregularity',
    'color_h_mean', 'color_s_mean', 'color_v_mean', 'color_var_index',
    'diameter_norm', 'diameter_available',
    'evolution_available', 'evolution_change',
    'age_norm', 'sex_male', 'sex_female',
    'loc_head_neck', 'loc_trunk', 'loc_extremities'
]

print("=== 16 CLINICAL FEATURES: ISIC_0012418 vs HAM10000 TRAIN SET ===")
mel_mask = (train_labels == 1)
non_mel_mask = (train_labels == 0)

for i in range(16):
    val = vec_t[i]
    mel_mean = train_clinical[mel_mask, i].mean()
    mel_std = train_clinical[mel_mask, i].std()
    non_mean = train_clinical[non_mel_mask, i].mean()
    non_std = train_clinical[non_mel_mask, i].std()
    
    # Percentile in overall train
    perc = (train_clinical[:, i] < val).mean() * 100
    fname = feature_names[i] if i < len(feature_names) else f'feat_{i}'
    print(f"{i:2d}. {fname:22s}: val={val:.4f} | Non-Mel: {non_mean:.4f} +/- {non_std:.4f} | Mel: {mel_mean:.4f} +/- {mel_std:.4f} | Train Pct: {perc:5.1f}%")

# Load Multimodal Fusion Model
fusion_model = CEFMMultimodalFusion(visual_dim=768, clinical_dim=16, hidden_dim=128, num_classes=2)
fusion_model.load_state_dict(torch.load('models/multimodal/fusion_model.pt', map_location='cpu'))
fusion_model.eval()

target_emb = get_vit_embedding(img_path, mask=mask)
v_tensor = torch.tensor(target_emb, dtype=torch.float32).unsqueeze(0)
c_tensor = torch.tensor(vec_t, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    base_logits = fusion_model(v_tensor, c_tensor)
    base_prob = torch.softmax(base_logits, dim=-1)[0, 1].item()

print(f"\nBaseline Multimodal Melanoma Probability: {base_prob:.6f} ({base_prob*100:.4f}%)")

# Sensitivity Experiments:
# 1. What if clinical vector is replaced by average melanoma clinical vector?
c_mel_avg = torch.tensor(train_clinical[mel_mask].mean(axis=0), dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    mel_avg_logits = fusion_model(v_tensor, c_mel_avg)
    mel_avg_prob = torch.softmax(mel_avg_logits, dim=-1)[0, 1].item()
print(f"Prob with Average Melanoma Clinical Vector: {mel_avg_prob:.6f} ({mel_avg_prob*100:.4f}%)")

# 2. What if clinical vector is all zeros (neutral/missing clinical signals)?
c_zeros = torch.zeros((1, 16), dtype=torch.float32)
with torch.no_grad():
    zeros_logits = fusion_model(v_tensor, c_zeros)
    zeros_prob = torch.softmax(zeros_logits, dim=-1)[0, 1].item()
print(f"Prob with Neutral (Zeros) Clinical Vector:  {zeros_prob:.6f} ({zeros_prob*100:.4f}%)")

# 3. What if asymmetry, border, and color are set to extreme melanoma levels (1.0)?
c_extreme = c_tensor.clone()
c_extreme[0, 0] = 1.0 # asymmetry
c_extreme[0, 1] = 1.0 # border
c_extreme[0, 5] = 1.0 # color index
c_extreme[0, 6] = 1.0 # diameter
with torch.no_grad():
    ext_logits = fusion_model(v_tensor, c_extreme)
    ext_prob = torch.softmax(ext_logits, dim=-1)[0, 1].item()
print(f"Prob with Extreme ABCDE Values (1.0):       {ext_prob:.6f} ({ext_prob*100:.4f}%)")

# 4. Compare with an actual true melanoma sample's visual embedding
# Take the nearest melanoma sample ISIC_0026774
mel_idx = mel_indices = np.where(train_labels == 1)[0][0]
v_mel = train_visual[mel_idx].unsqueeze(0)
with torch.no_grad():
    mel_visual_with_our_clinical = torch.softmax(fusion_model(v_mel, c_tensor), dim=-1)[0, 1].item()
    mel_visual_with_mel_clinical = torch.softmax(fusion_model(v_mel, torch.tensor(train_clinical[mel_idx]).unsqueeze(0)), dim=-1)[0, 1].item()
print(f"True Melanoma Visual + ISIC_0012418 Clinical: {mel_visual_with_our_clinical:.6f}")
print(f"True Melanoma Visual + Its Own Clinical:      {mel_visual_with_mel_clinical:.6f}")
