#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
One-click quantize and validate script for YOLOX models.

Reads settings from config.json (quantize section) and:
  1. Quantizes the FP32 ONNX model to INT8 OpenVINO IR (.xml/.bin)
  2. Validates the quantized model on calibration images
  3. Reports model size reduction and latency comparison (FP32 vs INT8)

After quantization, update config.json model.path to the INT8 .xml
and set model.backend to "openvino" to use the quantized model in the app.

Usage:
    python quantize_and_validate.py                        # Use config.json defaults
    python quantize_and_validate.py --skip-validate        # Quantize only, no inference
    python quantize_and_validate.py --validate-only        # Skip quantization, run inference on existing INT8
    python quantize_and_validate.py --compare              # Run both FP32 and INT8, compare
"""

import os
import sys
import json
import time
import random
import argparse
import cv2
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_config():
    """Load config.json from project root."""
    config_path = os.path.join(_PROJECT_ROOT, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)


def resolve_path(p):
    """Resolve relative path against project root."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_PROJECT_ROOT, p))


def get_image_list(folder):
    """Collect all image files from folder (recursive)."""
    image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
    images = []
    for root, _, files in os.walk(folder):
        for name in files:
            full = os.path.join(root, name)
            if os.path.splitext(full)[1].lower() in image_ext:
                images.append(full)
    return sorted(images)


def check_openvino():
    """Check that OpenVINO and NNCF are installed."""
    try:
        import openvino as ov
        ov_version = ov.__version__
    except ImportError:
        print("ERROR: openvino not installed.")
        print("Run: pip install -r requirements-openvino.txt")
        sys.exit(1)

    try:
        import nncf
        nncf_version = nncf.__version__
    except ImportError:
        print("ERROR: nncf not installed.")
        print("Run: pip install -r requirements-openvino.txt")
        sys.exit(1)

    return ov_version, nncf_version


def run_quantization(config, num_samples=None):
    """Run NNCF post-training quantization."""
    import openvino as ov
    import nncf
    from yolox.data.data_augment import preproc as preprocess

    q_cfg = config.get("quantize", {})
    model_path = resolve_path(q_cfg.get("input_model", "weights/sakku_fp32.onnx"))
    output_path = resolve_path(q_cfg.get("output_model", "weights/sakku_int8.xml"))
    calib_dir = resolve_path(q_cfg.get("calibration_data", "calib data"))
    max_samples = num_samples or int(q_cfg.get("num_samples", 200))
    input_shape_str = q_cfg.get("input_shape", "640,640")
    input_shape = tuple(map(int, input_shape_str.split(",")))

    # Validate
    if not os.path.isfile(model_path):
        print(f"ERROR: Input model not found: {model_path}")
        sys.exit(1)
    if not os.path.isdir(calib_dir):
        print(f"ERROR: Calibration data directory not found: {calib_dir}")
        print(f"Please place 100-300 representative images in: {calib_dir}")
        sys.exit(1)

    images = get_image_list(calib_dir)
    if not images:
        print(f"ERROR: No images found in {calib_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" STEP 1: POST-TRAINING QUANTIZATION (FP32 → INT8)")
    print("=" * 70)
    print(f"  Input model:     {model_path}")
    print(f"  Output model:    {output_path}")
    print(f"  Calib directory: {calib_dir}")
    print(f"  Total images:    {len(images)}")
    print(f"  Sample size:     {min(max_samples, len(images))}")
    print(f"  Input shape:     {input_shape}")

    # Sample images
    n = min(max_samples, len(images))
    random.seed(42)
    sampled = random.sample(images, n)

    # Read model
    print(f"\n  Reading FP32 ONNX model...")
    t_start = time.time()
    core = ov.Core()
    model = core.read_model(model_path)
    input_name = model.inputs[0].any_name
    print(f"  Input name: '{input_name}', shape: {model.inputs[0].shape}")

    # Preprocess calibration images
    print(f"  Preprocessing {n} calibration images...")
    calib_data = []
    t_preproc = time.time()
    for idx, path in enumerate(sampled):
        img = cv2.imread(path)
        if img is None:
            continue
        img_pre, _ = preprocess(img, input_shape)
        calib_data.append(img_pre[None, :, :, :])
        if (idx + 1) % 50 == 0:
            print(f"    Preprocessed {idx + 1}/{n}...")
    t_preproc = time.time() - t_preproc
    print(f"  Preprocessed {len(calib_data)} images in {t_preproc:.1f}s")

    if not calib_data:
        print("ERROR: All calibration images failed to load")
        sys.exit(1)

    # NNCF quantization
    def transform_fn(data_item):
        return {input_name: data_item}

    nncf_dataset = nncf.Dataset(calib_data, transform_fn)

    print(f"\n  Running NNCF quantization ({len(calib_data)} samples)...")
    print("  This may take several minutes...\n")
    t_quant = time.time()
    quantized = nncf.quantize(model, nncf_dataset)
    t_quant = time.time() - t_quant

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ov.save_model(quantized, output_path)
    t_total = time.time() - t_start

    # Size comparison
    fp32_size = os.path.getsize(model_path) / (1024 * 1024)
    bin_path = os.path.splitext(output_path)[0] + ".bin"
    int8_size = 0
    if os.path.exists(output_path):
        int8_size += os.path.getsize(output_path) / (1024 * 1024)
    if os.path.exists(bin_path):
        int8_size += os.path.getsize(bin_path) / (1024 * 1024)

    reduction = (1 - int8_size / fp32_size) * 100 if fp32_size > 0 else 0

    print("  " + "-" * 50)
    print("  QUANTIZATION RESULTS:")
    print("  " + "-" * 50)
    print(f"  FP32 model size:    {fp32_size:.2f} MB")
    print(f"  INT8 model size:    {int8_size:.2f} MB")
    print(f"  Size reduction:     {reduction:.1f}%")
    print(f"  Preprocess time:    {t_preproc:.1f}s")
    print(f"  Quantization time:  {t_quant:.1f}s ({t_quant/60:.1f} min)")
    print(f"  Total time:         {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"\n  Output: {output_path}")
    print(f"  Output: {bin_path}")

    return output_path, fp32_size, int8_size


def run_inference_benchmark(model_path, config, label="", max_images=50):
    """Run inference benchmark on calibration images using OpenVINO."""
    import openvino as ov
    from yolox.data.data_augment import preproc as preprocess

    q_cfg = config.get("quantize", {})
    calib_dir = resolve_path(q_cfg.get("calibration_data", "calib data"))
    input_shape_str = q_cfg.get("input_shape", "640,640")
    input_shape = tuple(map(int, input_shape_str.split(",")))
    ov_device = config.get("model", {}).get("openvino_device", "CPU")

    images = get_image_list(calib_dir)
    if not images:
        print(f"  WARNING: No images in {calib_dir}, skipping benchmark")
        return None

    images = images[:max_images]

    print(f"\n  Benchmarking {label}: {model_path}")
    print(f"  Device: {ov_device} | Images: {len(images)} | Shape: {input_shape}")

    core = ov.Core()
    t_load = time.time()
    ov_model = core.read_model(model_path)
    compiled = core.compile_model(ov_model, ov_device)
    infer_req = compiled.create_infer_request()
    output_key = compiled.output(0)
    t_load = time.time() - t_load

    pre_times = []
    model_times = []
    total_times = []
    total_detections = 0

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.time()
        img_pre, ratio = preprocess(img, input_shape)
        t_pre = time.time() - t0

        t_model = time.time()
        result = infer_req.infer({0: img_pre[None, :, :, :]})
        raw = result[output_key]
        t_model = time.time() - t_model

        t_total = time.time() - t0
        pre_times.append(t_pre)
        model_times.append(t_model)
        total_times.append(t_total)

        # Count output detections (raw shape)
        pred = np.asarray(raw)
        if pred.ndim == 3:
            pred = pred[0]
        total_detections += pred.shape[0] if pred.ndim >= 1 else 0

    if not total_times:
        return None

    avg_pre = sum(pre_times) / len(pre_times)
    avg_model = sum(model_times) / len(model_times)
    avg_total = sum(total_times) / len(total_times)

    sorted_total = sorted(total_times)
    n = len(sorted_total)
    p50 = sorted_total[int(n * 0.50)]
    p95 = sorted_total[min(int(n * 0.95), n - 1)]
    p99 = sorted_total[min(int(n * 0.99), n - 1)]
    fps = 1.0 / avg_total if avg_total > 0 else 0

    results = {
        "model_path": model_path,
        "label": label,
        "device": ov_device,
        "num_images": len(total_times),
        "load_time_s": round(t_load, 3),
        "avg_preprocess_ms": round(avg_pre * 1000, 2),
        "avg_model_ms": round(avg_model * 1000, 2),
        "avg_total_ms": round(avg_total * 1000, 2),
        "p50_ms": round(p50 * 1000, 2),
        "p95_ms": round(p95 * 1000, 2),
        "p99_ms": round(p99 * 1000, 2),
        "fps": round(fps, 1),
    }

    print(f"  " + "-" * 50)
    print(f"  {label} BENCHMARK:")
    print(f"  " + "-" * 50)
    print(f"  Model load + compile: {t_load:.2f}s")
    print(f"  Avg Preprocess:       {avg_pre * 1000:.2f} ms")
    print(f"  Avg Model Forward:    {avg_model * 1000:.2f} ms")
    print(f"  Avg Total Latency:    {avg_total * 1000:.2f} ms")
    print(f"  P50 Latency:          {p50 * 1000:.2f} ms")
    print(f"  P95 Latency:          {p95 * 1000:.2f} ms")
    print(f"  P99 Latency:          {p99 * 1000:.2f} ms")
    print(f"  Estimated FPS:        {fps:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="One-click quantize and validate YOLOX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize_and_validate.py                  # Full pipeline (quantize + validate)
  python quantize_and_validate.py --skip-validate  # Quantize only
  python quantize_and_validate.py --validate-only  # Benchmark existing INT8 model
  python quantize_and_validate.py --compare        # Compare FP32 vs INT8 latency
  python quantize_and_validate.py --num-samples 100 --max-bench-images 30
""",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the inference validation after quantization",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip quantization, only run validation on existing INT8 model",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FP32 and INT8 inference latency side-by-side",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of calibration samples (overrides config.json)",
    )
    parser.add_argument(
        "--max-bench-images",
        type=int,
        default=50,
        help="Max images for benchmark (default: 50)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available OpenVINO devices and exit",
    )
    args = parser.parse_args()

    # Check dependencies
    ov_ver, nncf_ver = check_openvino()

    config = load_config()

    print("\n" + "=" * 70)
    print(" YOLOX QUANTIZATION & VALIDATION PIPELINE")
    print("=" * 70)
    print(f"  OpenVINO: {ov_ver}")
    print(f"  NNCF:     {nncf_ver}")
    print(f"  Device:   {config.get('model', {}).get('openvino_device', 'CPU')}")

    # List devices if requested
    if args.list_devices:
        import openvino as ov
        core = ov.Core()
        print("\n  Available OpenVINO devices:")
        for dev in core.available_devices:
            try:
                name = core.get_property(dev, "FULL_DEVICE_NAME")
                print(f"    {dev}: {name}")
            except Exception:
                print(f"    {dev}: (name unavailable)")
        print()
        return

    q_cfg = config.get("quantize", {})
    fp32_path = resolve_path(q_cfg.get("input_model", "weights/sakku_fp32.onnx"))
    int8_path = resolve_path(q_cfg.get("output_model", "weights/sakku_int8.xml"))

    t_pipeline = time.time()

    # STEP 1: Quantize
    if not args.validate_only:
        run_quantization(config, num_samples=args.num_samples)
    else:
        if not os.path.exists(int8_path):
            print(f"\nERROR: INT8 model not found: {int8_path}")
            print("Run without --validate-only first to quantize the model.")
            sys.exit(1)
        print(f"\n  Skipping quantization. Using existing: {int8_path}")

    # STEP 2: Validate
    if not args.skip_validate:
        print("\n" + "=" * 70)
        print(" STEP 2: INFERENCE VALIDATION")
        print("=" * 70)

        int8_results = run_inference_benchmark(
            int8_path, config, label="INT8", max_images=args.max_bench_images
        )

        fp32_results = None
        if args.compare and os.path.exists(fp32_path):
            fp32_results = run_inference_benchmark(
                fp32_path, config, label="FP32", max_images=args.max_bench_images
            )

        # Print comparison
        if fp32_results and int8_results:
            print("\n" + "=" * 70)
            print(" COMPARISON: FP32 vs INT8")
            print("=" * 70)
            speedup = fp32_results["avg_model_ms"] / int8_results["avg_model_ms"] if int8_results["avg_model_ms"] > 0 else 0
            print(f"  {'Metric':<25} {'FP32':>12} {'INT8':>12} {'Speedup':>10}")
            print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
            print(f"  {'Avg Model Forward':<25} {fp32_results['avg_model_ms']:>10.2f}ms {int8_results['avg_model_ms']:>10.2f}ms {speedup:>9.2f}x")
            print(f"  {'Avg Total Latency':<25} {fp32_results['avg_total_ms']:>10.2f}ms {int8_results['avg_total_ms']:>10.2f}ms")
            print(f"  {'P95 Latency':<25} {fp32_results['p95_ms']:>10.2f}ms {int8_results['p95_ms']:>10.2f}ms")
            print(f"  {'Estimated FPS':<25} {fp32_results['fps']:>10.1f}   {int8_results['fps']:>10.1f}  ")

    t_pipeline = time.time() - t_pipeline

    # Final summary
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)
    print(f"  Total pipeline time: {t_pipeline:.1f}s ({t_pipeline/60:.1f} min)")
    print(f"\n  To use the quantized model in the app:")
    print(f"  1. Update config.json:")
    print(f'     "model.path": "{os.path.relpath(int8_path, _PROJECT_ROOT)}"')
    print(f'     "model.backend": "openvino"')
    print(f'     "model.openvino_device": "CPU"')
    print(f"  2. Run: python detection.py")
    print()


if __name__ == "__main__":
    main()
