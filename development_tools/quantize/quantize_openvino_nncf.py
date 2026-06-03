#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Post-Training Quantization script using OpenVINO NNCF.
Quantizes FP32 ONNX models directly into optimized OpenVINO INT8 format (.xml/.bin).

Usage (from project root):
    python quantize/quantize_openvino_nncf.py
    python quantize/quantize_openvino_nncf.py -m weights/sakku_fp32.onnx -i "calib data" -o weights/sakku_int8.xml
"""

import os
import sys
import random
import argparse
import time
import cv2
import numpy as np
import openvino as ov
import nncf

# Ensure project root is on sys.path so YOLOX imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from yolox.data.data_augment import preproc as preprocess


def make_parser():
    parser = argparse.ArgumentParser("YOLOX OpenVINO NNCF Quantization script")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "weights", "sakku_fp32.onnx"),
        help="Path to FP32 ONNX model (default: weights/sakku_fp32.onnx)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "calib data"),
        help='Path to folder of calibration images (default: "calib data")',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "weights", "sakku_int8.xml"),
        help="Path to save quantized XML model (default: weights/sakku_int8.xml)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of calibration samples to use",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Model input shape (H,W)",
    )
    return parser


def main():
    args = make_parser().parse_args()
    input_shape = tuple(map(int, args.input_shape.split(",")))

    print("=" * 60)
    print("YOLOX OpenVINO NNCF Post-Training Quantization")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Calib dir:   {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Input shape: {input_shape}")
    print("=" * 60)

    # Validate inputs
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.isdir(args.input):
        raise FileNotFoundError(
            f"Calibration image directory not found: {args.input}\n"
            f"Please place 100-300 representative images in this folder."
        )

    # 1. Get image list
    image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
    images = []
    for root, _, files in os.walk(args.input):
        for name in files:
            full = os.path.join(root, name)
            if os.path.splitext(full)[1].lower() in image_ext:
                images.append(full)

    if len(images) == 0:
        raise ValueError(
            f"No calibration images found in: {args.input}\n"
            f"Supported formats: {image_ext}"
        )

    num_samples = min(args.num_samples, len(images))
    print(f"\nTotal available images: {len(images)}. Sampling {num_samples} for calibration...")
    random.seed(42)  # For reproducibility
    sampled_images = random.sample(images, num_samples)

    # 2. Read model
    print(f"\nReading FP32 model: {args.model}")
    t_read_start = time.time()
    core = ov.Core()
    model = core.read_model(args.model)
    t_read = time.time() - t_read_start
    input_name = model.inputs[0].any_name
    print(f"  Model input name: '{input_name}', shape: {model.inputs[0].shape}")
    print(f"  Model read time: {t_read:.2f}s")

    # 3. Create calibration dataset
    calibration_data = []
    print(f"\nPreprocessing {num_samples} calibration images...")
    t_preproc_start = time.time()
    skipped = 0
    for idx, image_path in enumerate(sampled_images):
        img = cv2.imread(image_path)
        if img is None:
            skipped += 1
            continue
        # Preprocess using YOLOX data preprocessing (letterbox + normalize)
        img_pre, _ = preprocess(img, input_shape)
        # Add batch dimension → NCHW
        tensor = img_pre[None, :, :, :]
        calibration_data.append(tensor)
        if (idx + 1) % 50 == 0:
            print(f"  Preprocessed {idx + 1}/{num_samples} images...")
    t_preproc = time.time() - t_preproc_start
    print(f"  Done. {len(calibration_data)} images preprocessed in {t_preproc:.2f}s ({skipped} skipped)")

    if len(calibration_data) == 0:
        raise ValueError("All calibration images failed to load. Check image files.")

    # Define dataset wrapper for NNCF
    def transform_fn(data_item):
        return {input_name: data_item}

    nncf_dataset = nncf.Dataset(calibration_data, transform_fn)

    # 4. Run quantization
    print(f"\nRunning NNCF Post-Training Quantization with {len(calibration_data)} samples...")
    print("  This may take several minutes depending on model size and CPU...")
    t_quant_start = time.time()
    quantized_model = nncf.quantize(model, nncf_dataset)
    t_quant = time.time() - t_quant_start
    print(f"  Quantization completed in {t_quant:.1f}s ({t_quant / 60:.1f} minutes)")

    # 5. Save quantized model
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving quantized model to: {args.output}")
    ov.save_model(quantized_model, args.output)

    # 6. Report size comparison
    fp32_size = os.path.getsize(args.model) / (1024 * 1024)
    xml_path = args.output
    bin_path = os.path.splitext(args.output)[0] + ".bin"
    int8_size = 0
    if os.path.exists(xml_path):
        int8_size += os.path.getsize(xml_path) / (1024 * 1024)
    if os.path.exists(bin_path):
        int8_size += os.path.getsize(bin_path) / (1024 * 1024)

    t_total = time.time() - t_read_start

    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"  FP32 model size:  {fp32_size:.2f} MB")
    print(f"  INT8 model size:  {int8_size:.2f} MB")
    if fp32_size > 0:
        print(f"  Size reduction:   {(1 - int8_size / fp32_size) * 100:.1f}%")
    print(f"  Model read time:  {t_read:.2f}s")
    print(f"  Preprocess time:  {t_preproc:.2f}s")
    print(f"  Quantize time:    {t_quant:.1f}s")
    print(f"  Total time:       {t_total:.1f}s")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  XML: {xml_path}")
    print(f"  BIN: {bin_path}")
    print(f"\nTo use in the app, update config.json:")
    print(f'  "model.path": "{os.path.relpath(xml_path, _PROJECT_ROOT)}"')
    print(f'  "model.backend": "openvino"')


if __name__ == "__main__":
    main()
