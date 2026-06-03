#!/usr/bin/env python3
"""
Rebuild embeddings.npy using the multi-output INT8 model's embedding output.

This ensures the reference bank uses the SAME preprocessing (BGR, no channel
swap) as the live multi-output inference path, eliminating the drift score
inflation caused by the BGR↔RGB mismatch with the old dual-model pipeline.
"""

import os
import sys
import time
import numpy as np
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fpa_agent.detection_model import DetectionModel


def main():
    classes_path = os.path.join(_PROJECT_ROOT, "class.txt")
    dataset_dir = os.path.join(_PROJECT_ROOT, "development_tools", "Training-Dataset")
    output_path = os.path.join(_PROJECT_ROOT, "embeddings.npy")
    backup_path = os.path.join(_PROJECT_ROOT, "embeddings_dual_model_backup.npy")

    # Back up old embeddings
    if os.path.exists(output_path):
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"[INFO] Backed up old embeddings.npy -> {os.path.basename(backup_path)}")

    # Load the multi-output INT8 model
    print("[INFO] Loading Multi-Output INT8 model...")
    model = DetectionModel(
        pth_path=os.path.join(_PROJECT_ROOT, "weights", "sakku_multi_output_int8.xml"),
        exp_path=os.path.join(_PROJECT_ROOT, "development_tools", "yolox_voc_s 3.py"),
        classes_path=classes_path,
        device="cpu",
        enable_tracking=False,
        backend="openvino",
        openvino_device="CPU",
    )

    if not model.can_encode_drift_embedding():
        print("[ERROR] Multi-output model cannot encode drift embeddings!")
        sys.exit(1)
    print(f"[INFO] Drift encoder: {model.drift_encoder_description()}")

    # Scan images
    print(f"[INFO] Scanning dataset: {dataset_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"[ERROR] No images found in {dataset_dir}")
        sys.exit(1)
    print(f"[INFO] Found {len(image_paths)} total images")

    # Process ALL images — no sampling. One-time cost for a production-quality
    # reference bank that covers the full training distribution.
    print(f"[INFO] Will process all {len(image_paths)} images")

    # Extract embeddings
    vectors = []
    t0 = time.time()
    for i, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            continue

        # Run detection first (populates cached embedding from multi-output)
        model.predict(frame)

        # Extract embedding (will use cached multi-output embedding)
        try:
            emb = model.encode_frame_embedding(frame)
            vectors.append(emb)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(path)}: {e}")

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  Encoded {i + 1}/{len(image_paths)} ({fps:.1f} img/s)")

    elapsed = time.time() - t0
    print(f"\n[INFO] Encoded {len(vectors)} embeddings in {elapsed:.1f}s")

    if not vectors:
        print("[ERROR] No embeddings produced!")
        sys.exit(1)

    # Save
    arr = np.stack(vectors, axis=0).astype(np.float32)
    np.save(output_path, arr)
    print(f"[INFO] Saved {arr.shape} -> {output_path}")

    # Quick sanity: compute self-cosine stats
    norms = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    centroid = norms.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    cos = norms @ centroid
    print(f"[INFO] Mean cosine to centroid: {cos.mean():.4f} (expect >0.85 for similar scenes)")
    print(f"[INFO] Min cosine: {cos.min():.4f}, Max cosine: {cos.max():.4f}")

    # Quick validation: load a frame from video and check drift
    video_path = os.path.join(_PROJECT_ROOT, "2 wheeler at Gate 2.mp4")
    if os.path.exists(video_path):
        from fpa_agent.drift_score import EmbeddingDriftScorer
        scorer = EmbeddingDriftScorer(output_path, device="cpu", encoder="yolox_standard")
        scorer.load()
        scorer.attach_yolox_model(model)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            model.predict(frame)
            drift = scorer.score_frame(frame, 1)
            print(f"\n[VALIDATION] Drift score on '2 wheeler at Gate 2.mp4' frame 1:")
            print(f"  Drift Score:       {drift['drift_score']:.2f}%")
            print(f"  Cosine Centroid:   {drift['cosine_centroid']:.4f}")
            print(f"  KNN Mean Sim:      {drift['knn_mean_sim']:.4f}")


if __name__ == "__main__":
    main()
