#!/usr/bin/env python3
"""
Rebuild embeddings.npy using the YOLOX Standard pipeline (team reference).

letterbox 640 → backbone → neck → GAP each scale → concat → L2 normalize

Example:
  python build_reference_embeddings.py \\
    --pth sakku-gate.pth \\
    --exp "yolox_voc_s 3.py" \\
    --images /path/to/JPEGImages \\
    --output embeddings.npy
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def iter_images(path: str):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            # Sort files in alphabetical order
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    yield os.path.join(root, file)
    elif os.path.isfile(path) and path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // 5000) if total > 0 else 1
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                yield frame
            i += 1
        cap.release()
    elif os.path.isfile(path):
        yield path


def main():
    ap = argparse.ArgumentParser(description="Build embeddings.npy (YOLOX Standard)")
    ap.add_argument("--pth", help="YOLOX checkpoint e.g. sakku-gate.pth")
    ap.add_argument("--exp", help="YOLOX exp .py")
    ap.add_argument("--images", required=True, help="Image folder, image path, or video")
    ap.add_argument("--output", default="embeddings.npy")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--openvino-embedding", help="Path to OpenVINO embedding model (.xml)")
    ap.add_argument("--onnx-embedding", help="Path to ONNX embedding model (.onnx)")
    args = ap.parse_args()

    if not args.openvino_embedding and not args.onnx_embedding and (not args.pth or not args.exp):
        ap.error("Either --openvino-embedding, --onnx-embedding, or both --pth and --exp must be provided.")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fpa_agent.detection_model import DetectionModel

    isize = (args.input_size[0], args.input_size[1])
    if args.openvino_embedding:
        model = DetectionModel(
            None,
            None,
            "",
            device=args.device,
            drift_input_size=isize,
            drift_encoder="yolox_standard",
            drift_openvino_embedding_path=args.openvino_embedding,
            backend="openvino"
        )
    elif args.onnx_embedding:
        model = DetectionModel(
            None,
            None,
            "",
            device=args.device,
            drift_input_size=isize,
            drift_encoder="yolox_standard",
            drift_onnx_embedding_path=args.onnx_embedding,
            backend="onnxruntime"
        )
    else:
        model = DetectionModel(
            args.pth,
            args.exp,
            "",
            device=args.device,
            drift_input_size=isize,
            drift_encoder="yolox_standard",
        )
    if not model.can_encode_drift_embedding():
        print("ERROR: YOLOX standard drift embedder not available (need .pth with backbone+neck).")
        sys.exit(1)

    vectors = []
    sources = list(iter_images(args.images))
    if not sources:
        print(f"No images under {args.images}")
        sys.exit(1)

    for i, item in enumerate(sources):
        if args.max and i >= args.max:
            break
        if isinstance(item, str):
            frame = cv2.imread(item)
            if frame is None:
                continue
        else:
            frame = item
        try:
            emb = model.encode_frame_embedding(frame)
            vectors.append(emb)
        except Exception as e:
            print(f"skip {item}: {e}")
        if (i + 1) % 500 == 0:
            print(f"encoded {i + 1} ...")

    if not vectors:
        print("No embeddings produced.")
        sys.exit(1)

    arr = np.stack(vectors, axis=0).astype(np.float32)
    np.save(args.output, arr)
    print(f"Saved {arr.shape} -> {args.output}")

    norms = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    cent = norms.mean(axis=0)
    cent = cent / (np.linalg.norm(cent) + 1e-8)
    cos = norms @ cent
    print(f"mean cos to centroid: {cos.mean():.4f} (expect high for similar scenes)")


if __name__ == "__main__":
    main()
