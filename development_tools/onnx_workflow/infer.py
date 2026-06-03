#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ONNX Workflow: Standalone Inference & Drift Scoring Script
Runs detection with quantized ONNX and measures data drift against embeddings.npy.

Usage:
    python infer.py --detection ../weights/sakku_int8.onnx --embedding ../weights/sakku_embedding.onnx --reference ../embeddings.npy --image test_frame.jpg
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

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



def yolox_decode_and_nms(raw_preds, ratio, input_size, conf_threshold=0.4, nms_threshold=0.45):
    """
    Decode raw YOLOX predictions ([1, N, 85]) into xyxy boxes and run multi-class NMS.
    Handles both raw offsets and already-decoded absolute coordinates.
    """
    preds = raw_preds[0] if raw_preds.ndim == 3 else raw_preds
    if preds.ndim != 2 or preds.shape[1] < 6:
        return np.zeros((0, 6), dtype=np.float32)

    num_classes = preds.shape[1] - 5
    
    # Auto-detect if coordinates are already decoded (pixel scale) or raw logits
    is_decoded = float(np.max(preds[:, :4])) > 10.0
    
    if is_decoded:
        # Bounding boxes are already decoded: [cx, cy, w, h]
        boxes_xyxy = np.zeros((preds.shape[0], 4), dtype=np.float32)
        boxes_xyxy[:, 0] = preds[:, 0] - preds[:, 2] / 2.0
        boxes_xyxy[:, 1] = preds[:, 1] - preds[:, 3] / 2.0
        boxes_xyxy[:, 2] = preds[:, 0] + preds[:, 2] / 2.0
        boxes_xyxy[:, 3] = preds[:, 1] + preds[:, 3] / 2.0
    else:
        # Raw coordinates need grid decoding
        input_hw = input_size[0]
        strides = [8, 16, 32]
        hsizes = [input_hw // s for s in strides]
        wsizes = [input_hw // s for s in strides]

        grids = []
        expanded_strides = []
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))

        grids = np.concatenate(grids, axis=1).astype(np.float32)
        expanded_strides = np.concatenate(expanded_strides, axis=1)

        preds_xy = (preds[:, :2] + grids[0]) * expanded_strides[0]
        wh_log = np.clip(preds[:, 2:4], -20.0, 20.0)
        preds_wh = np.exp(wh_log) * expanded_strides[0]

        boxes_xyxy = np.zeros((preds.shape[0], 4), dtype=np.float32)
        boxes_xyxy[:, 0] = preds_xy[:, 0] - preds_wh[:, 0] / 2.0
        boxes_xyxy[:, 1] = preds_xy[:, 1] - preds_wh[:, 1] / 2.0
        boxes_xyxy[:, 2] = preds_xy[:, 0] + preds_wh[:, 0] / 2.0
        boxes_xyxy[:, 3] = preds_xy[:, 1] + preds_wh[:, 1] / 2.0

    # Scale back to original image dimensions
    boxes_xyxy /= ratio

    # Compute confidence score
    obj_conf = preds[:, 4:5]
    cls_scores = preds[:, 5:]
    scores = obj_conf * cls_scores

    final_detections = []
    for cls_ind in range(num_classes):
        cls_sc = scores[:, cls_ind]
        keep = cls_sc > conf_threshold
        if not np.any(keep):
            continue

        v_scores = cls_sc[keep]
        v_boxes = boxes_xyxy[keep]
        
        # OpenCV NMS expects xywh list format
        xywh = np.zeros_like(v_boxes)
        xywh[:, 0] = v_boxes[:, 0]
        xywh[:, 1] = v_boxes[:, 1]
        xywh[:, 2] = v_boxes[:, 2] - v_boxes[:, 0]
        xywh[:, 3] = v_boxes[:, 3] - v_boxes[:, 1]

        idxs = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            v_scores.tolist(),
            conf_threshold,
            nms_threshold,
        )
        if len(idxs) == 0:
            continue
        
        for idx in idxs.flatten():
            final_detections.append([
                v_boxes[idx, 0], v_boxes[idx, 1], v_boxes[idx, 2], v_boxes[idx, 3],
                v_scores[idx], cls_ind
            ])

    return np.array(final_detections, dtype=np.float32) if final_detections else np.zeros((0, 6), dtype=np.float32)


def calculate_drift_score(emb, reference_matrix, knn_sample_size=2048):
    """
    Calculate raw drift score by comparing the active frame's embedding vector
    to the reference embedding distribution.
    """
    # L2 Normalize input vector
    emb = emb.reshape(-1)[:512]
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm

    # L2 Normalize reference rows
    norms = np.linalg.norm(reference_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    ref_matrix_norm = reference_matrix / norms

    # Calculate centroid distance
    centroid = ref_matrix_norm.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 1e-8:
        centroid = centroid / centroid_norm
    cos_centroid = float(np.dot(emb, centroid))

    # Calculate KNN average similarity
    n = ref_matrix_norm.shape[0]
    k = min(knn_sample_size, n)
    if k < n:
        # Sample subset for fast KNN computation
        idx = np.random.choice(n, size=k, replace=False)
        sample = ref_matrix_norm[idx]
    else:
        sample = ref_matrix_norm

    sims = sample @ emb
    knn_mean_sim = float(np.mean(sims)) if sims.size else cos_centroid

    # Calculate drift score
    dist_centroid = max(0.0, 1.0 - cos_centroid)
    dist_knn = max(0.0, 1.0 - knn_mean_sim)
    
    # 60% Centroid distance + 40% Average KNN distance
    drift_raw = 0.6 * dist_centroid + 0.4 * dist_knn
    drift_score = float(min(100.0, drift_raw * 100.0))

    return drift_score, cos_centroid, knn_mean_sim


def main():
    ap = argparse.ArgumentParser("YOLOX standalone ONNX inference + drift scorer")
    ap.add_argument("--detection", required=True, help="Path to detection model (sakku_best.onnx / sakku_int8.onnx)")
    ap.add_argument("--embedding", required=True, help="Path to embedding extractor model (sakku_embedding.onnx)")
    ap.add_argument("--reference", required=True, help="Path to reference embeddings file (embeddings.npy)")
    ap.add_argument("--image", required=True, help="Path to test image file")
    ap.add_argument("--classes", default=os.path.join(_PROJECT_ROOT, "class.txt"), help="Path to class.txt file")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="Model input resolution (height width)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Target device (cpu or gpu)")
    args = ap.parse_args()
    input_size = tuple(args.input_size)

    # 1. Load configuration and files
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Test image not found: {args.image}")
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference embeddings not found: {args.reference}")
    
    # Read classes
    classes = ["object"]
    if os.path.exists(args.classes):
        with open(args.classes, "r") as f:
            classes = [line.strip() for line in f if line.strip()]

    # Load reference distribution
    ref_embeddings = np.load(args.reference)

    # Initialize ONNX Sessions
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "gpu" else ["CPUExecutionProvider"]
    try:
        det_session = ort.InferenceSession(args.detection, providers=providers)
        emb_session = ort.InferenceSession(args.embedding, providers=providers)
    except Exception:
        det_session = ort.InferenceSession(args.detection, providers=["CPUExecutionProvider"])
        emb_session = ort.InferenceSession(args.embedding, providers=["CPUExecutionProvider"])

    det_in_name = det_session.get_inputs()[0].name
    det_out_name = det_session.get_outputs()[0].name

    emb_in_name = emb_session.get_inputs()[0].name
    emb_out_name = emb_session.get_outputs()[0].name

    # 2. Preprocess Input Image
    img_bgr = cv2.imread(args.image)
    h_orig, w_orig = img_bgr.shape[:2]
    img_pre, ratio = preprocess(img_bgr, input_size)
    blob = img_pre[None, :, :, :].astype(np.float32)

    # 3. Perform Inference
    print(f"\nRunning detection on {args.image}...")
    t_start = time.time()
    det_raw = det_session.run([det_out_name], {det_in_name: blob})[0]
    t_det = time.time() - t_start

    print(f"Extracting embedding vector...")
    t_start = time.time()
    emb_raw = emb_session.run([emb_out_name], {emb_in_name: blob})[0]
    t_emb = time.time() - t_start

    # 4. Post-process detection coordinates
    detections = yolox_decode_and_nms(det_raw, ratio, input_size)
    
    # 5. Compute Drift Score
    drift_score, cos_centroid, knn_mean_sim = calculate_drift_score(emb_raw, ref_embeddings)

    print("\n" + "=" * 60)
    print(" INFERENCE RESULTS")
    print("=" * 60)
    print(f"  Detection latency:    {t_det*1000:.2f} ms")
    print(f"  Embedding latency:    {t_emb*1000:.2f} ms")
    print(f"  Total detections:     {len(detections)}")
    print(f"  Drift Score:          {drift_score:.2f} / 100.0")
    print(f"  Centroid Cosine Sim:  {cos_centroid:.4f}")
    print(f"  Mean KNN Similarity:  {knn_mean_sim:.4f}")
    print("=" * 60)

    # Print Detections and annotate image
    if len(detections) > 0:
        print("\nDetections:")
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, score, cls_id = det
            cls_name = classes[int(cls_id)] if int(cls_id) < len(classes) else "unknown"
            print(f"  [{idx+1}] {cls_name}: conf={score:.2f} @ [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
            
            # Draw on image
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                f"{cls_name} {score:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
    # Draw drift score in top left corner
    cv2.putText(
        img_bgr,
        f"Drift Score: {drift_score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if drift_score > 50 else (255, 0, 0),
        2
    )

    out_name = "annotated_frame.jpg"
    cv2.imwrite(out_name, img_bgr)
    print(f"\nAnnotated frame saved to: {out_name}\n")


if __name__ == "__main__":
    main()
