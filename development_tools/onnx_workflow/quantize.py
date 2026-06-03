#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ONNX Workflow: Model Quantization Script using ONNX Runtime
Quantizes the FP32 ONNX model to INT8 ONNX using ONNX Runtime's native quantization engine.
Optimized for GPU execution (CUDA/TensorRT) using the QDQ format.

Usage:
    python quantize.py --model ../weights/sakku_best.onnx --input ../calib\ data --output ../weights/sakku_int8.onnx
"""

import argparse
import os
import random
import sys
import time
import cv2
import numpy as np
import onnx
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError:
    print("ERROR: onnxruntime or onnx package is not installed correctly.")
    print("Please install them on your device: pip install onnx==1.16.1 onnxruntime-gpu (or onnxruntime for CPU)")
    sys.exit(1)

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



class YOLOXCalibrationDataReader(CalibrationDataReader):
    """
    ONNX Runtime Calibration Data Reader for YOLOX.
    Iterates over calibration images, pre-processes them, and returns input tensors.
    """
    def __init__(self, image_paths, input_name, input_shape):
        super().__init__()
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.input_shape = input_shape
        self.index = 0
        self.count = len(self.image_paths)

    def get_next(self):
        if self.index >= self.count:
            return None
        
        path = self.image_paths[self.index]
        self.index += 1
        
        img = cv2.imread(path)
        if img is None:
            # Skip invalid images and get next
            return self.get_next()
            
        img_pre, _ = preprocess(img, self.input_shape)
        # Return dict mapping input tensor name to numpy array (NCHW float32)
        return {self.input_name: img_pre[None, :, :, :].astype(np.float32)}

    def rewind(self):
        self.index = 0


def main():
    ap = argparse.ArgumentParser("YOLOX ONNX Runtime Native Quantization")
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "weights", "sakku_best.onnx"),
        help="Path to FP32 ONNX model",
    )
    ap.add_argument(
        "-i",
        "--input",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "calib data"),
        help="Path to folder of calibration images",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "weights", "sakku_int8.onnx"),
        help="Path to save quantized ONNX model",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of calibration samples to use",
    )
    ap.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Model input shape (H,W)",
    )
    ap.add_argument(
        "--format",
        type=str,
        default="qdq",
        choices=["qdq", "qoperator"],
        help="Quantization format: 'qdq' (preferred for GPU/TensorRT) or 'qoperator' (preferred for CPU)",
    )
    args = ap.parse_args()
    input_shape = tuple(map(int, args.input_shape.split(",")))

    print("=" * 60)
    print("ONNX Runtime Native Static Quantization")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Calib dir:   {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Input shape: {input_shape}")
    print(f"  Format:      {args.format.upper()}")
    print("=" * 60)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Calibration folder not found: {args.input}")

    # 1. Collect calibration images
    image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
    images = []
    for root, _, files in os.walk(args.input):
        for name in files:
            full = os.path.join(root, name)
            if os.path.splitext(full)[1].lower() in image_ext:
                images.append(full)

    if len(images) == 0:
        raise ValueError(f"No calibration images found in: {args.input}")

    num_samples = min(args.num_samples, len(images))
    print(f"\nTotal available images: {len(images)}. Sampling {num_samples} for calibration...")
    random.seed(42)
    sampled_images = random.sample(images, num_samples)

    # 2. Get input name of the model
    onnx_model = onnx.load(args.model)
    input_name = onnx_model.graph.input[0].name
    print(f"Model input name: '{input_name}'")

    # 3. Create calibration data reader
    data_reader = YOLOXCalibrationDataReader(sampled_images, input_name, input_shape)

    # 4. Run quantization
    print(f"\nRunning ONNX Runtime static quantization ({args.format.upper()} format)...")
    print("This may take several minutes depending on model complexity...")
    
    t_start = time.time()
    
    # Selection of format
    q_format = QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator
    
    quantize_static(
        model_input=args.model,
        model_output=args.output,
        calibration_data_reader=data_reader,
        quant_format=q_format,
        per_channel=True,               # Better accuracy for YOLOX conv layers
        weight_type=QuantType.QInt8,    # INT8 for weights
        activation_type=QuantType.QInt8,# INT8 for activations (best for GPU Tensor Cores)
    )
    
    t_quant = time.time() - t_start
    print(f"Quantization completed in {t_quant:.1f}s")

    # Size comparison
    fp32_size = os.path.getsize(args.model) / (1024 * 1024)
    int8_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nFP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {int8_size:.2f} MB")
    print(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")
    print("=" * 60)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
