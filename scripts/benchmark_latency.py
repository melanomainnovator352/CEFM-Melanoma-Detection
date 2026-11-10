"""
Benchmark latency across all stages of the CEFM Multimodal Assessment Pipeline:
1. Segmentation (UltraLight VM-UNet)
2. ABC Feature Extraction
3. ViT Visual Feature Extraction
4. Multimodal Fusion Head Inference
5. Total Inference Time
6. FastAPI Request Latency (End-to-End HTTP Request)
"""
import io
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features, analyze_image
from app.services.classification import get_vit_embedding, predict_multimodal_fusion, classify
TEST_IMAGE = REPO_ROOT / "data" / "uploads" / "debug_crop_mel.jpg"

if not TEST_IMAGE.exists():
    # Fallback to any jpg in uploads
    uploads = list((REPO_ROOT / "data" / "uploads").glob("*.jpg"))
    if uploads:
        TEST_IMAGE = uploads[0]


def run_benchmark(num_runs: int = 5):
    print("=" * 75)
    print("CEFM MULTIMODAL SYSTEM LATENCY BENCHMARK")
    print(f"Target test image: {TEST_IMAGE.name}")
    print(f"Number of measured iterations: {num_runs} (after warmup)")
    print("=" * 75)

    img_bgr = cv2.imread(str(TEST_IMAGE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Warm-up
    print("Warming up models...")
    mask = segment_lesion(str(TEST_IMAGE))
    abcde_dict = extract_abcde_features(img_rgb, mask)
    emb = get_vit_embedding(str(TEST_IMAGE), mask=mask)
    if emb is not None:
        predict_multimodal_fusion(emb, abcde_dict)

    # Measure Segmentation Time
    seg_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        m = segment_lesion(str(TEST_IMAGE))
        seg_times.append((time.perf_counter() - t0) * 1000.0)

    # Measure ABC Feature Extraction Time
    abc_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        f = extract_abcde_features(img_rgb, mask)
        abc_times.append((time.perf_counter() - t0) * 1000.0)

    # Measure ViT Feature Extraction Time
    vit_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        e = get_vit_embedding(str(TEST_IMAGE), mask=mask)
        vit_times.append((time.perf_counter() - t0) * 1000.0)

    # Measure Fusion-Head Inference Time
    fusion_times = []
    for _ in range(num_runs * 10):
        t0 = time.perf_counter()
        out = predict_multimodal_fusion(emb, abcde_dict)
        fusion_times.append((time.perf_counter() - t0) * 1000.0)

    # Measure Total Pipeline Inference Time (Single Call)
    total_pipeline_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        # Full analyze_image call
        res = analyze_image(str(TEST_IMAGE), TEST_IMAGE.name)
        total_pipeline_times.append((time.perf_counter() - t0) * 1000.0)

    # Measure FastAPI Request Latency (End-to-End via TestClient)
    client = TestClient(app)
    api_times = []
    with open(TEST_IMAGE, "rb") as f:
        img_bytes = f.read()

    for _ in range(num_runs):
        t0 = time.perf_counter()
        response = client.post(
            "/api/v1/analyze",
            files={"image": (TEST_IMAGE.name, io.BytesIO(img_bytes), "image/jpeg")},
            data={"patient_name": "Benchmark Patient", "case_label": "Benchmark Patient"},
        )
        api_times.append((time.perf_counter() - t0) * 1000.0)

    print("\nBENCHMARK RESULTS (Mean +/- Std):")
    print(f"  1. Segmentation Time (UltraLight VM-UNet):  {np.mean(seg_times):8.2f} ms  (std: {np.std(seg_times):.2f} ms)")
    print(f"  2. ABC Feature Extraction Time:            {np.mean(abc_times):8.2f} ms  (std: {np.std(abc_times):.2f} ms)")
    print(f"  3. ViT Visual Feature Extraction Time:     {np.mean(vit_times):8.2f} ms  (std: {np.std(vit_times):.2f} ms)")
    print(f"  4. Multimodal Fusion Head Inference Time:  {np.mean(fusion_times):8.3f} ms  (std: {np.std(fusion_times):.3f} ms)")
    print(f"  5. Total Pipeline Inference Time:          {np.mean(total_pipeline_times):8.2f} ms  (std: {np.std(total_pipeline_times):.2f} ms)")
    print(f"  6. FastAPI Request Latency (End-to-End):    {np.mean(api_times):8.2f} ms  (std: {np.std(api_times):.2f} ms)")
    print("=" * 75)
    return {
        "segmentation_ms": round(float(np.mean(seg_times)), 2),
        "abc_extraction_ms": round(float(np.mean(abc_times)), 2),
        "vit_extraction_ms": round(float(np.mean(vit_times)), 2),
        "fusion_head_ms": round(float(np.mean(fusion_times)), 3),
        "total_inference_ms": round(float(np.mean(total_pipeline_times)), 2),
        "fastapi_request_latency_ms": round(float(np.mean(api_times)), 2),
    }


if __name__ == "__main__":
    run_benchmark(num_runs=3)
