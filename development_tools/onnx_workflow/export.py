#!/usr/bin/env python3
"""
ONNX Workflow: Model Export Script
Converts a PyTorch YOLOX .pth checkpoint to ONNX format (detection and embedding models).

Usage:
    python export.py --pth ../weights/best_ckpt.pth --exp ../yolox_voc_s\ 3.py --output-dir ../weights
"""

import argparse
import os
import sys
import torch

# Add parent directory to sys.path to find export_to_openvino and yolox modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from export_to_openvino import YOLOXStandardEmbeddingWrapper, load_yolox_model


def main():
    ap = argparse.ArgumentParser(description="Export YOLOX models to ONNX format")
    ap.add_argument("--pth", default=os.path.join(_PROJECT_ROOT, "weights", "best_ckpt.pth"), help="Path to YOLOX checkpoint (.pth)")
    ap.add_argument("--exp", default=os.path.join(_PROJECT_ROOT, "yolox_voc_s 3.py"), help="Path to YOLOX experiment (.py)")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="Model input resolution (height width)")
    ap.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "weights"), help="Directory to save exported ONNX models")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.pth):
        print(f"ERROR: Checkpoint file not found: {args.pth}")
        sys.exit(1)
    if not os.path.exists(args.exp):
        print(f"ERROR: Experiment file not found: {args.exp}")
        sys.exit(1)

    print(f"Loading PyTorch model from {args.pth}...")
    model, exp = load_yolox_model(args.exp, args.pth)
    
    h, w = args.input_size
    dummy_input = torch.randn(1, 3, h, w)
    
    # 1. Export YOLOX Detection Model
    model.head.decode_in_inference = False  # Set to false to export raw model
    detection_onnx_path = os.path.join(args.output_dir, "sakku_best.onnx")
    print(f"\nConverting YOLOX Detection model to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            detection_onnx_path,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11,
        )
        print(f"--> Successfully exported Detection model to {detection_onnx_path}")
    except Exception as e:
        print(f"ERROR exporting Detection model: {e}")

    # 2. Export YOLOX Embedding Model (CSPDarknet + Neck + GAP + L2 Norm)
    embedding_onnx_path = os.path.join(args.output_dir, "sakku_embedding.onnx")
    print(f"\nConverting YOLOX Embedding model to ONNX...")
    try:
        wrapper = YOLOXStandardEmbeddingWrapper(model, pool_mode="last_scale")
        wrapper.eval()
        torch.onnx.export(
            wrapper,
            dummy_input,
            embedding_onnx_path,
            input_names=["images"],
            output_names=["embedding"],
            dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=11,
        )
        print(f"--> Successfully exported Embedding model to {embedding_onnx_path}")
    except Exception as e:
        print(f"ERROR exporting Embedding model: {e}")


if __name__ == "__main__":
    main()
