#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Native OpenVINO inference script for YOLOX models.
Supports reading .onnx or .xml directly and runs with maximum CPU optimization.
Outputs annotated images + JSON metadata with full timing breakdown.

Usage (from project root):
    python quantize/infer_openvino.py -m weights/sakku_int8.xml -i "calib data" -o openvino_results
    python quantize/infer_openvino.py -m weights/sakku_fp32.onnx -i "calib data" -o openvino_results
"""

import argparse
import os
import sys
import time
import json
import cv2
import numpy as np
import openvino as ov

# Ensure project root is on sys.path so YOLOX imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

# Default class file at project root
CLASS_FILE = os.path.join(_PROJECT_ROOT, "class.txt")

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_class_names(class_file=None):
    """Load class names from file. Falls back to project root class.txt."""
    paths_to_try = []
    if class_file:
        paths_to_try.append(class_file)
    paths_to_try.append(CLASS_FILE)

    for path in paths_to_try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                names = tuple(line.strip() for line in f if line.strip())
                if names:
                    return names

    raise FileNotFoundError(
        f"No class file found. Tried: {paths_to_try}\n"
        f"Create a class.txt with one class name per line."
    )


def get_image_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if os.path.isfile(path):
        return [path]
    names = []
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if os.path.splitext(full)[1].lower() in IMAGE_EXT:
                names.append(full)
    return sorted(names)


def cxcywh_to_xyxy(boxes):
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return boxes_xyxy


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    box1, box2: list or array of [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def stabilize_bboxes(detections, active_tracks, iou_threshold=0.45, alpha=0.4):
    """
    Stabilizes bounding boxes using an exponential moving average (EMA) over sequential frames.
    detections: list of dicts with keys: 'bbox' (list of 4 floats), 'class_id' (int), 'score' (float), 'class' (str)
    active_tracks: list of dicts representing the tracked bboxes in the previous frame
    """
    if not active_tracks:
        return detections, [dict(d) for d in detections]

    updated_tracks = []
    stabilized_detections = []

    matched_detection_indices = set()
    matched_track_indices = set()

    # Calculate all IoUs and filter by class and threshold
    matches = []
    for d_idx, det in enumerate(detections):
        for t_idx, track in enumerate(active_tracks):
            if det["class_id"] == track["class_id"]:
                iou = calculate_iou(det["bbox"], track["bbox"])
                if iou >= iou_threshold:
                    matches.append((iou, d_idx, t_idx))

    # Sort matches by IoU descending
    matches.sort(key=lambda x: x[0], reverse=True)

    for iou, d_idx, t_idx in matches:
        if d_idx in matched_detection_indices or t_idx in matched_track_indices:
            continue
        matched_detection_indices.add(d_idx)
        matched_track_indices.add(t_idx)

        det = detections[d_idx]
        track = active_tracks[t_idx]

        # Smooth coordinates
        smooth_bbox = []
        for c_curr, c_prev in zip(det["bbox"], track["bbox"]):
            smooth_bbox.append(round(alpha * c_curr + (1 - alpha) * c_prev, 2))

        smooth_score = round(alpha * det["score"] + (1 - alpha) * track["score"], 4)

        stabilized_det = {
            "class": det["class"],
            "class_id": det["class_id"],
            "score": smooth_score,
            "bbox": smooth_bbox
        }
        stabilized_detections.append(stabilized_det)
        updated_tracks.append(stabilized_det)

    # Unmatched detections are added as new tracks
    for d_idx, det in enumerate(detections):
        if d_idx not in matched_detection_indices:
            stabilized_detections.append(det)
            updated_tracks.append(dict(det))

    return stabilized_detections, updated_tracks


def predictions_from_output(output, input_shape, decoded):
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    
    # Auto-detect if decoded
    is_decoded = decoded or (float(np.max(pred[:, :4])) > 10.0)
    if is_decoded:
        return pred
    return demo_postprocess(pred[None, :, :], input_shape)[0]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Native OpenVINO inference script")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "weights", "sakku_int8.xml"),
        help="Path to input model file (.onnx, .xml) (default: weights/sakku_int8.xml)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input image or folder of images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="openvino_inference_results",
        help="Path to output directory for saving annotated images and JSON files",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.25,
        help="Score threshold to filter detections",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS threshold",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Input shape for YOLOX inference",
    )
    parser.add_argument(
        "--class-file",
        type=str,
        default=CLASS_FILE,
        help=f"Custom class names file path (default: {os.path.basename(CLASS_FILE)})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=True,
        help="Model has raw head outputs and needs decoding postprocess (default: True)",
    )
    parser.add_argument(
        "--decoded",
        action="store_true",
        help="Model was exported with --decode_in_inference (disable raw postprocessing)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Device to compile model for (CPU, GPU, AUTO)",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Enable temporal bounding box stabilization/smoothing for sequential frames",
    )
    parser.add_argument(
        "--stabilize_alpha",
        type=float,
        default=0.4,
        help="EMA smoothing factor (0.0 < alpha <= 1.0; lower is smoother, 1.0 is no smoothing)",
    )
    parser.add_argument(
        "--stabilize_iou",
        type=float,
        default=0.45,
        help="IoU threshold for matching boxes between consecutive frames",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available OpenVINO devices and exit",
    )
    return parser


def list_available_devices():
    """Print all available OpenVINO devices and their properties."""
    core = ov.Core()
    devices = core.available_devices
    print("\n" + "=" * 60)
    print("AVAILABLE OPENVINO DEVICES")
    print("=" * 60)
    for device in devices:
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"  {device}: {full_name}")
    print("=" * 60)
    print(f"\nUse -d/--device <DEVICE> to select (e.g. -d CPU, -d GPU, -d AUTO)")
    return devices


def main():
    args = make_parser().parse_args()

    if args.list_devices:
        list_available_devices()
        return

    decoded = args.decoded or not args.raw
    class_names = get_class_names(args.class_file)
    input_shape = tuple(map(int, args.input_shape.split(",")))

    print("=" * 60)
    print("YOLOX Native OpenVINO Inference")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Device:      {args.device}")
    print(f"  Score thr:   {args.score_thr}")
    print(f"  NMS thr:     {args.nms}")
    print(f"  Classes:     {len(class_names)} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
    print("=" * 60)

    # Validate model exists
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Get image list
    images = get_image_list(args.input)
    if not images:
        print(f"No images found in input path: {args.input}")
        return

    # Create output directory
    mkdir(args.output)

    # Initialize OpenVINO Core and Compile Model
    print(f"\nInitializing OpenVINO Core...")
    core = ov.Core()

    # Show available devices
    print(f"  Available devices: {core.available_devices}")

    print(f"Reading model: {args.model}")
    t_load_start = time.time()
    model = core.read_model(args.model)
    print(f"Compiling model for device: {args.device}...")
    compiled_model = core.compile_model(model, args.device)
    infer_request = compiled_model.create_infer_request()
    t_load = time.time() - t_load_start
    print(f"  Model load + compile time: {t_load:.2f}s")

    summary_results = {}

    pre_times = []
    model_times = []
    post_times = []
    total_times = []

    active_tracks = []
    for idx, image_path in enumerate(images):
        origin_img = cv2.imread(image_path)
        if origin_img is None:
            print(f"[{idx+1}/{len(images)}] Skipping unreadable image: {image_path}")
            continue

        t0 = time.time()
        img, ratio = preprocess(origin_img, input_shape)
        t_pre = time.time() - t0

        # Run OpenVINO inference session
        t_model_start = time.time()
        # OpenVINO takes inputs in NCHW format
        input_tensor = img[None, :, :, :]
        # Execute synchronous inference
        results = infer_request.infer({0: input_tensor})
        output_data = results[compiled_model.output(0)]  # safe: index 0 by port
        t_model = time.time() - t_model_start

        # Postprocess predictions
        t_post_start = time.time()
        predictions = predictions_from_output(output_data, input_shape, decoded=decoded)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = cxcywh_to_xyxy(boxes)
        boxes_xyxy /= ratio

        # Apply NMS
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=args.nms, score_thr=args.score_thr)
        t_post = time.time() - t_post_start

        t_total = time.time() - t0

        pre_times.append(t_pre)
        model_times.append(t_model)
        post_times.append(t_post)
        total_times.append(t_total)

        image_detections = []
        n_det = 0

        # Parse detections that meet score threshold
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = (
                dets[:, :4],
                dets[:, 4],
                dets[:, 5],
            )

            for box, score, cls_id in zip(final_boxes, final_scores, final_cls_inds):
                cls_id = int(cls_id)
                if score >= args.score_thr:
                    n_det += 1
                    bbox_list = [round(float(coord), 2) for coord in box]
                    image_detections.append({
                        "class": class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                        "class_id": cls_id,
                        "score": round(float(score), 4),
                        "bbox": bbox_list  # [xmin, ymin, xmax, ymax]
                    })

        # Apply temporal stabilization if enabled
        if args.stabilize:
            image_detections, active_tracks = stabilize_bboxes(
                image_detections, active_tracks, iou_threshold=args.stabilize_iou, alpha=args.stabilize_alpha
            )
            n_det = len(image_detections)

        # Annotate image
        annotated_img = origin_img.copy()
        if image_detections:
            vis_boxes = np.array([d["bbox"] for d in image_detections], dtype=np.float32)
            vis_scores = np.array([d["score"] for d in image_detections], dtype=np.float32)
            vis_cls_inds = np.array([d["class_id"] for d in image_detections], dtype=np.float32)

            annotated_img = vis(
                annotated_img,
                vis_boxes,
                vis_scores,
                vis_cls_inds,
                conf=args.score_thr,
                class_names=class_names,
            )

        print(
            f"[{idx+1}/{len(images)}] {os.path.basename(image_path)}: "
            f"Total {t_total:.4f}s (Pre: {t_pre:.4f}s | Model: {t_model:.4f}s | Post: {t_post:.4f}s) | "
            f"{n_det} detections (conf >= {args.score_thr})"
        )

        # Save annotated image
        output_img_name = os.path.basename(image_path)
        output_img_path = os.path.join(args.output, output_img_name)
        cv2.imwrite(output_img_path, annotated_img)

        # Save individual JSON file
        json_name = os.path.splitext(output_img_name)[0] + ".json"
        output_json_path = os.path.join(args.output, json_name)

        file_meta = {
            "image_path": os.path.abspath(image_path),
            "inference_time_seconds": round(t_total, 4),
            "breakdown": {
                "preprocess_seconds": round(t_pre, 4),
                "model_run_seconds": round(t_model, 4),
                "postprocess_seconds": round(t_post, 4)
            },
            "detections": image_detections
        }

        with open(output_json_path, "w", encoding="utf-8") as jf:
            json.dump(file_meta, jf, indent=2)

        # Store for consolidated summary
        summary_results[output_img_name] = file_meta

    if not total_times:
        print("No images were processed.")
        return

    # Save consolidated summary JSON file
    summary_path = os.path.join(args.output, "detections_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_results, sf, indent=2)

    avg_pre = sum(pre_times) / len(pre_times)
    avg_model = sum(model_times) / len(model_times)
    avg_post = sum(post_times) / len(post_times)
    avg_total = sum(total_times) / len(total_times)

    # Compute p50/p95/p99 latency
    sorted_total = sorted(total_times)
    n = len(sorted_total)
    p50 = sorted_total[int(n * 0.50)]
    p95 = sorted_total[min(int(n * 0.95), n - 1)]
    p99 = sorted_total[min(int(n * 0.99), n - 1)]
    fps_avg = 1.0 / avg_total if avg_total > 0 else 0

    print("\n" + "=" * 60)
    print("NATIVE OPENVINO SPEED BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Model:                  {args.model}")
    print(f"  Device:                 {args.device}")
    print(f"  Total Images Processed: {len(total_times)}")
    print(f"  Model Load + Compile:   {t_load:.2f}s")
    print(f"  ---")
    print(f"  Avg Preprocessing:      {avg_pre * 1000:.2f} ms")
    print(f"  Avg Model Forward:      {avg_model * 1000:.2f} ms")
    print(f"  Avg Postprocessing:     {avg_post * 1000:.2f} ms")
    print(f"  Avg Total Latency:      {avg_total * 1000:.2f} ms")
    print(f"  ---")
    print(f"  P50 Latency:            {p50 * 1000:.2f} ms")
    print(f"  P95 Latency:            {p95 * 1000:.2f} ms")
    print(f"  P99 Latency:            {p99 * 1000:.2f} ms")
    print(f"  ---")
    print(f"  Estimated FPS:          {fps_avg:.1f}")
    print("=" * 60 + "\n")

    print(f"All results saved to: {args.output}")
    print(f"Consolidated JSON summary written to: {summary_path}")


if __name__ == "__main__":
    main()
