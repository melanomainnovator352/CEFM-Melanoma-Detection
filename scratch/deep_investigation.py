import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# 1. Load train features
train_data = torch.load('data/multimodal_cache/train_features.pt', map_location='cpu', weights_only=False)
train_visual = train_data['visual']  # (6978, 768)
train_clinical = train_data['clinical']  # (6978, 16)
train_labels = train_data['labels'].numpy()  # (6978,)

train_df = pd.read_csv('data/HAM10000/metadata/train.csv')
print(f"Train samples: {len(train_df)}, visual shape: {train_visual.shape}")

# 2. Extract embedding for ISIC_0012418
from app.services.classification import get_vit_embedding, classify
from app.services.segmentation import segment_lesion
from app.services.calibration import estimate_pixels_per_mm

img_path = 'D:/datasets/ISIC2018/ISIC2018_Task1-2_Test_Input/ISIC_0012418.jpg'
img = cv2.imread(img_path)
cal = estimate_pixels_per_mm(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
exclude_poly = cal.get('exclude_polygon') or cal.get('corners')
mask = segment_lesion(img_path, exclude_polygon=exclude_poly)

target_emb = get_vit_embedding(img_path, mask=mask)
target_emb_t = torch.tensor(target_emb, dtype=torch.float32).unsqueeze(0)

# Cosine similarity with all 6978 training samples
sims = torch.nn.functional.cosine_similarity(target_emb_t, train_visual).numpy()

# Top 10 nearest neighbors
top10_idx = np.argsort(sims)[::-1][:10]

print("\n=== TOP 10 NEAREST TRAINING SAMPLES (ViT Embedding Space) ===")
for rank, idx in enumerate(top10_idx):
    row = train_df.iloc[idx]
    label_str = 'MELANOMA (1)' if train_labels[idx] == 1 else 'NON-MELANOMA (0)'
    print(f"{rank+1}. Image: {row['image_id']}, dx: {row['dx']}, Label: {label_str}, Cosine Sim: {sims[idx]:.4f}, Age: {row['age']}, Sex: {row['sex']}, Loc: {row['localization']}")

# Check melanoma neighbors specifically
mel_indices = np.where(train_labels == 1)[0]
mel_sims = sims[mel_indices]
top_mel_idx = mel_indices[np.argsort(mel_sims)[::-1][:5]]
print("\n=== TOP 5 NEAREST MELANOMA TRAINING SAMPLES ===")
for rank, idx in enumerate(top_mel_idx):
    row = train_df.iloc[idx]
    print(f"{rank+1}. Image: {row['image_id']}, dx: {row['dx']}, Cosine Sim: {sims[idx]:.4f}, Age: {row['age']}, Sex: {row['sex']}, Loc: {row['localization']}")

# Compare distribution of cosine similarities to mel vs non-mel
print("\n=== SIMILARITY DISTRIBUTION ===")
print(f"Mean similarity to Non-Melanomas (N={(1-train_labels).sum()}): {sims[train_labels == 0].mean():.4f} +/- {sims[train_labels == 0].std():.4f}")
print(f"Mean similarity to Melanomas     (N={train_labels.sum()}):     {sims[train_labels == 1].mean():.4f} +/- {sims[train_labels == 1].std():.4f}")
print(f"Max similarity to any Non-Melanoma: {sims[train_labels == 0].max():.4f}")
print(f"Max similarity to any Melanoma:     {sims[train_labels == 1].max():.4f}")
