#!/usr/bin/env python3
"""
Dataset Inference Processor & Exporter.

Takes 1500 images from Training-Dataset, runs baseline Dual-Model vs
Multi-Output INT8 model inferences, prints a final comparison, and exports
the images along with their YOLO annotations in a structured output folder.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fpa_agent.detection_model import DetectionModel


def load_classes(classes_path):
    with open(classes_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    print("=" * 80)
    print("Dataset Inference Processor & Performance Comparison")
    print("=" * 80)

    # 1. Search for images in Training-Dataset
    dataset_dir = os.path.normpath(
        os.path.join(_PROJECT_ROOT, "development_tools", "Training-Dataset")
    )
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found at: {dataset_dir}")
        return

    print(f"Scanning directory: {dataset_dir} ...")
    image_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith(image_ext):
                image_paths.append(os.path.join(root, name))

    print(f"Found {len(image_paths)} images.")
    if not image_paths:
        print("ERROR: No images found. Cannot proceed.")
        return

    # Limit to 1500 frames
    max_frames = 1500
    image_paths = image_paths[:max_frames]
    num_images = len(image_paths)
    print(f"Selected {num_images} images for processing.")

    classes_path = os.path.join(_PROJECT_ROOT, "class.txt")
    class_names = load_classes(classes_path)

    # Output directories
    output_dir = os.path.normpath(
        os.path.join(_PROJECT_ROOT, "multi_output_export", "dataset_export")
    )
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # ----------------------------------------------------
    # Load Models
    # ----------------------------------------------------
    print("\nLoading Dual-Model Pipeline (OpenVINO)...")
    try:
        t0 = time.time()
        dual_model = DetectionModel(
            pth_path=os.path.join(_PROJECT_ROOT, "weights", "sakku_int8.xml"),
            exp_path=os.path.join(
                _PROJECT_ROOT, "development_tools", "yolox_voc_s 3.py"
            ),
            classes_path=classes_path,
            device="cpu",
            enable_tracking=False,
            backend="openvino",
            openvino_device="CPU",
            drift_openvino_embedding_path=os.path.join(
                _PROJECT_ROOT, "weights", "sakku_embedding.xml"
            ),
        )
        print(f"Dual-Model loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Error loading Dual-Model: {e}")
        return

    print("\nLoading Multi-Output INT8 Model (OpenVINO)...")
    try:
        t0 = time.time()
        multi_model = DetectionModel(
            pth_path=os.path.join(
                _PROJECT_ROOT, "weights", "sakku_multi_output_int8.xml"
            ),
            exp_path=os.path.join(
                _PROJECT_ROOT, "development_tools", "yolox_voc_s 3.py"
            ),
            classes_path=classes_path,
            device="cpu",
            enable_tracking=False,
            backend="openvino",
            openvino_device="CPU",
            drift_openvino_embedding_path="",
        )
        print(f"Multi-Output Model loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Error loading Multi-Output model: {e}")
        return

    # ----------------------------------------------------
    # Phase 1: Benchmark Dual-Model
    # ----------------------------------------------------
    print(f"\nPhase 1: Benchmarking Dual-Model over {num_images} images...")
    dual_times = []

    # Warmup
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        dual_model.predict(dummy_img)
        dual_model.encode_frame_embedding(dummy_img)

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        t_start = time.time()
        # Sequential passes: detection + embedding extraction
        _ = dual_model.predict(img)
        _ = dual_model.encode_frame_embedding(img)
        t_elapsed = time.time() - t_start
        dual_times.append(t_elapsed)
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{num_images} images...")

    dual_total = sum(dual_times)
    dual_avg = (dual_total / len(dual_times)) * 1000
    dual_fps = 1.0 / np.mean(dual_times)
    print(
        f"Phase 1 complete! Total time: {dual_total:.2f}s | Avg: {dual_avg:.2f} ms | FPS: {dual_fps:.2f}"
    )

    # ----------------------------------------------------
    # Phase 2: Benchmark and Export Multi-Output INT8
    # ----------------------------------------------------
    print(
        f"\nPhase 2: Benchmarking & Exporting Multi-Output model over {num_images} images..."
    )
    multi_times = []
    exported_count = 0

    # Warmup
    for _ in range(5):
        multi_model.predict(dummy_img)
        multi_model.encode_frame_embedding(dummy_img)

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        t_start = time.time()
        # Single inference pass (predict does detection and caches embedding)
        detections = multi_model.predict(img)
        _ = multi_model.encode_frame_embedding(img)
        t_elapsed = time.time() - t_start

        multi_times.append(t_elapsed)

        # Export frame image
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_img_path = os.path.join(images_out, f"{base_name}.jpg")
        cv2.imwrite(out_img_path, img)

        # Export YOLO txt label
        img_h, img_w = img.shape[:2]
        label_path = os.path.join(labels_out, f"{base_name}.txt")
        with open(label_path, "w") as f:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                cls_id = det.get("cls_id", 0)
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        exported_count += 1
        if (idx + 1) % 100 == 0:
            print(f"  Processed & Exported {idx + 1}/{num_images} images...")

    multi_total = sum(multi_times)
    multi_avg = (multi_total / len(multi_times)) * 1000
    multi_fps = 1.0 / np.mean(multi_times)
    print(
        f"Phase 2 complete! Total time: {multi_total:.2f}s | Avg: {multi_avg:.2f} ms | FPS: {multi_fps:.2f}"
    )

    # ----------------------------------------------------
    # Final report
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print(" INFERENCE DATASET PROCESSOR SUMMARY")
    print("=" * 80)
    print(f"Total Frames Processed: {num_images}")
    print(f"Total Frames Exported:  {exported_count}")
    print(f"Export Folder:          {output_dir}")
    print("-" * 80)
    print(f"1. OpenVINO Dual-Model Pipeline (INT8 + FP32):")
    print(f"   - Total Time: {dual_total:.2f} seconds")
    print(f"   - Avg Time:   {dual_avg:.2f} ms/frame")
    print(f"   - Speed:      {dual_fps:.2f} FPS")
    print()
    print(f"2. OpenVINO Multi-Output Pipeline (INT8):")
    print(f"   - Total Time: {multi_total:.2f} seconds")
    print(f"   - Avg Time:   {multi_avg:.2f} ms/frame")
    print(f"   - Speed:      {multi_fps:.2f} FPS")

    speedup = dual_total / multi_total
    print("-" * 80)
    print(f"RESULT: Multi-Output INT8 is {speedup:.2f}x FASTER than Dual-Model!")
    print(
        f"Saves a total of {dual_total - multi_total:.2f} seconds of computation over {num_images} frames."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
