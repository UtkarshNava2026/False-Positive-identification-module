#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ONNX Workflow: Build Reference Embeddings Script
Extracts features/embeddings from a reference dataset directory using the standard ONNX embedding model.

Usage:
    python build_reference.py --model ../weights/sakku_embedding.onnx --images ../Training-Dataset --output ../embeddings.npy
"""

import argparse
import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

# Standard YOLOX preprocessing inline (eliminates repository dependencies)
def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r



def iter_images(path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for root, _, files in os.walk(path):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in valid_exts:
                yield os.path.join(root, file)


def main():
    ap = argparse.ArgumentParser(description="Extract reference embeddings from a dataset using an ONNX model")
    ap.add_argument("--model", required=True, help="Path to standard ONNX embedding model (.onnx)")
    ap.add_argument("--images", required=True, help="Directory containing reference/baseline images")
    ap.add_argument("--output", default="embeddings.npy", help="Output path for embeddings (.npy)")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="Model input size (H,W)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Target device (cpu or gpu)")
    ap.add_argument("--max-images", type=int, default=0, help="Maximum number of images to extract (0 for all)")
    args = ap.parse_args()
    input_size = tuple(args.input_size)

    print("=" * 60)
    print("ONNX Embedding Extraction")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Images:     {args.images}")
    print(f"  Output:     {args.output}")
    print(f"  Input size: {input_size}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"ONNX model file not found: {args.model}")
    if not os.path.isdir(args.images):
        raise FileNotFoundError(f"Images directory not found: {args.images}")

    # Initialize ONNX Runtime Session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "gpu" else ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(args.model, providers=providers)
    except Exception:
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    vectors = []
    image_paths = list(iter_images(args.images))
    total_images = len(image_paths)

    if total_images == 0:
        print(f"ERROR: No valid images found in: {args.images}")
        sys.exit(1)

    print(f"\nFound {total_images} images. Starting extraction...")

    for i, path in enumerate(image_paths):
        if args.max_images and i >= args.max_images:
            break

        img = cv2.imread(path)
        if img is None:
            continue

        # Preprocess using YOLOX standard letterbox
        img_pre, _ = preprocess(img, input_size)
        blob = img_pre[None, :, :, :].astype(np.float32)  # NCHW

        # Inference
        emb = session.run([output_name], {input_name: blob})[0]
        
        # Flatten and normalize embedding vector
        emb = emb.reshape(-1)[:512]
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
            
        vectors.append(emb)

        if (i + 1) % 500 == 0 or (i + 1) == total_images:
            print(f"  Processed {i + 1}/{total_images} images...")

    if not vectors:
        print("ERROR: No embeddings produced.")
        sys.exit(1)

    # Save to disk
    arr = np.stack(vectors, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, arr)
    print(f"\n--> Successfully saved {arr.shape} embeddings -> {args.output}")

    # Print mean similarity to centroid as quality check
    centroid = arr.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 1e-8:
        centroid = centroid / centroid_norm
    cos_sims = arr @ centroid
    print(f"  Quality Check (mean cosine similarity to centroid): {cos_sims.mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
