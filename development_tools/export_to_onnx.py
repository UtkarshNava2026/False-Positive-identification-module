#!/usr/bin/env python3
"""
Direct model conversion from PyTorch .pth to ONNX.

Supports converting:
1. The full YOLOX detection model
2. The standard drift embedding extraction model (backbone + neck + GAP + L2 normalize)
"""

import argparse
import os
import sys
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the wrapper and loader from export_to_openvino
from export_to_openvino import YOLOXStandardEmbeddingWrapper, load_yolox_model


def main():
    ap = argparse.ArgumentParser(description="Export YOLOX models directly from .pth to ONNX")
    ap.add_argument("--pth", default="weights/best_ckpt.pth", help="Path to YOLOX PyTorch .pth checkpoint")
    ap.add_argument("--exp", default="yolox_voc_s 3.py", help="Path to YOLOX experiment .py file")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="Model input resolution (height width)")
    ap.add_argument("--output-dir", default="weights", help="Directory to save exported ONNX models")
    args = ap.parse_args()

    # Create output dir if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load PyTorch model
    if not os.path.exists(args.pth):
        print(f"ERROR: Checkpoint file not found: {args.pth}")
        sys.exit(1)
    if not os.path.exists(args.exp):
        print(f"ERROR: Experiment file not found: {args.exp}")
        sys.exit(1)

    model, exp = load_yolox_model(args.exp, args.pth)
    
    h, w = args.input_size
    dummy_input = torch.randn(1, 3, h, w)
    
    # --- Part A: Export Detection Model ---
    # Disable decode_in_inference to export as a raw model (matches standard ONNX configs)
    model.head.decode_in_inference = False
    
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
        print(f"Successfully exported Detection model -> {detection_onnx_path}")
    except Exception as e:
        print(f"ERROR converting Detection model: {e}")
        import traceback
        traceback.print_exc()

    # --- Part B: Export Embedding Model ---
    embedding_onnx_path = os.path.join(args.output_dir, "sakku_embedding.onnx")
    print(f"\nWrapping and converting YOLOX Embedding model to ONNX...")
    try:
        pool_mode = "last_scale"
        wrapper = YOLOXStandardEmbeddingWrapper(model, pool_mode=pool_mode)
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
        print(f"Successfully exported Embedding model -> {embedding_onnx_path}")
    except Exception as e:
        print(f"ERROR converting Embedding model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
