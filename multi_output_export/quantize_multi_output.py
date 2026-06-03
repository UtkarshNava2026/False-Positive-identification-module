#!/usr/bin/env python3
"""
Quantize YOLOX Multi-Output Model to True OpenVINO INT8.

Reads the FP32 ONNX/OpenVINO multi-output model and performs post-training
quantization using OpenVINO NNCF and calibration images.
"""

import os
import sys
import time
import random
import cv2
import numpy as np

# Add project root to sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def preprocess(img, input_size=(640, 640), swap=(2, 0, 1)):
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

def main():
    print("=" * 70)
    print("OpenVINO NNCF Multi-Output INT8 Quantization")
    print("=" * 70)

    try:
        import openvino as ov
        import nncf
        print(f"OpenVINO version: {ov.__version__}")
        print(f"NNCF version:     {nncf.__version__}")
    except ImportError:
        print("ERROR: openvino or nncf is not installed.")
        print("Please install them: pip install openvino nncf")
        return

    # Check for calibration images
    calib_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, "development_tools", "calib data"))
    if not os.path.exists(calib_dir):
        # Fallback to project root "calib data"
        calib_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, "calib data"))
    
    if not os.path.exists(calib_dir):
        print(f"ERROR: Calibration data directory not found. Checked: {calib_dir}")
        return

    # Collect images
    image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
    images = []
    for root, _, files in os.walk(calib_dir):
        for name in files:
            full = os.path.join(root, name)
            if os.path.splitext(full)[1].lower() in image_ext:
                images.append(full)

    if not images:
        print(f"ERROR: No calibration images found in: {calib_dir}")
        return

    # Load FP32 model
    fp32_onnx = os.path.normpath(os.path.join(_PROJECT_ROOT, "weights", "sakku_multi_output.onnx"))
    fp32_xml = os.path.normpath(os.path.join(_PROJECT_ROOT, "weights", "sakku_multi_output.xml"))
    
    input_model_path = None
    if os.path.exists(fp32_xml):
        input_model_path = fp32_xml
    elif os.path.exists(fp32_onnx):
        input_model_path = fp32_onnx
        
    if not input_model_path:
        print("ERROR: FP32 multi-output model not found in weights/.")
        print("Please run export_multi_output.py first.")
        return

    output_xml = os.path.normpath(os.path.join(_PROJECT_ROOT, "weights", "sakku_multi_output_int8.xml"))
    output_bin = os.path.normpath(os.path.join(_PROJECT_ROOT, "weights", "sakku_multi_output_int8.bin"))

    print(f"  Input Model:  {input_model_path}")
    print(f"  Output Model: {output_xml}")
    print(f"  Calib Folder: {calib_dir}")
    print(f"  Total Images: {len(images)}")

    # Sample images
    num_samples = min(150, len(images))
    random.seed(42)
    sampled = random.sample(images, num_samples)

    # Read model
    print("\n  Reading model into OpenVINO...")
    core = ov.Core()
    model = core.read_model(input_model_path)
    input_name = model.inputs[0].any_name
    
    # Reshape model to static shape for quantization
    model.reshape({input_name: [1, 3, 640, 640]})
    print(f"  Model input name: '{input_name}', shape reshaped to: {list(model.inputs[0].shape)}")

    # Preprocess images
    print(f"  Preprocessing {num_samples} calibration images...")
    calib_data = []
    for idx, path in enumerate(sampled):
        img = cv2.imread(path)
        if img is None:
            continue
        img_pre, _ = preprocess(img, (640, 640))
        calib_data.append(img_pre[None, :, :, :])

    def transform_fn(data_item):
        return {input_name: data_item}

    # NNCF Dataset
    nncf_dataset = nncf.Dataset(calib_data, transform_fn)

    # Quantize
    print("\n  Running NNCF Post-Training Quantization (FP32 -> INT8)...")
    print("  This may take 1-3 minutes. Please wait...")
    t0 = time.time()
    quantized_model = nncf.quantize(
        model=model,
        calibration_dataset=nncf_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE # Optimize for speed (INT8)
    )
    t_quant = time.time() - t0
    print(f"  Quantization finished in {t_quant:.1f}s")

    # Save
    print("  Saving INT8 model...")
    ov.save_model(quantized_model, output_xml)
    print(f"\nSUCCESS: True INT8 Multi-Output OpenVINO model saved to:")
    print(f"  - {output_xml}")
    print(f"  - {output_bin}")


if __name__ == "__main__":
    main()
