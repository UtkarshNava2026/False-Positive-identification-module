#!/usr/bin/env python3
"""
Dedicated OpenVINO YOLOX Inference Script
----------------------------------------
This script is self-contained and optimized for OpenVINO (.xml) models.
It only requires:
  - openvino
  - opencv-python (cv2)
  - numpy

It does NOT depend on PyTorch, ONNX Runtime, or any local repository modules.

Usage:
  # Run inference on an image
  python openvino_inference.py --model_path weights/sakku_multi_output_int8.xml --image_path input.jpg --output_path output.jpg

  # Run inference on a video file
  python openvino_inference.py --model_path weights/sakku_multi_output_int8.xml --video_path input_video.mp4 --output_path output_video.mp4

  # Run inference on a live webcam (channel 0)
  python openvino_inference.py --model_path weights/sakku_multi_output_int8.xml --video_path 0
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Default classes matches class.txt
DEFAULT_CLASSES = [
    "person", "bike", "car", "truck", "tractor", "auto", "disinfection",
    "security_guard", "bus", "van", "ambulance", "pickup_truck", "bicycle"
]

_COLORS = [
    (189, 114, 0), (25, 83, 217), (32, 177, 237), (142, 47, 126), (48, 172, 119),
    (238, 190, 77), (47, 20, 162), (77, 77, 77), (153, 153, 153), (0, 0, 255),
    (0, 128, 255), (0, 191, 191), (0, 255, 0), (255, 0, 0), (255, 0, 170)
]


def preproc(img, input_size, swap=(2, 0, 1)):
    """Letterbox image to input size with standard YOLOX padding (value 114)."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def _centered_letterbox_rgb(image_bgr, input_size=(640, 640)):
    """BGR→RGB, centered letterbox with pad=114, CHW float32.
    This matches the exact preprocessing used to build the reference bank."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    th, tw = int(input_size[0]), int(input_size[1])
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    pt = (th - nh) // 2
    pl = (tw - nw) // 2
    padded[pt:pt + nh, pl:pl + nw] = resized
    blob = padded.astype(np.float32).transpose(2, 0, 1)
    return blob[np.newaxis, :].astype(np.float32)


class StandaloneDriftScorer:
    def __init__(self, reference_path, knn_sample_size=2048):
        self.reference_path = reference_path
        if not reference_path or not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference path not found: {reference_path}")
        
        arr = np.load(reference_path, allow_pickle=False).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.matrix = arr / norms
        self.dim = self.matrix.shape[1]
        
        centroid = self.matrix.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        self.centroid = centroid / centroid_norm if centroid_norm > 1e-8 else centroid
        
        n = self.matrix.shape[0]
        k = min(knn_sample_size, n)
        if k < n:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=k, replace=False)
            self.sample = self.matrix[idx]
        else:
            self.sample = self.matrix

    def calculate_drift(self, emb):
        if emb is None:
            return None
        
        norm = np.linalg.norm(emb)
        emb_norm = emb / norm if norm > 1e-8 else emb
        
        cos_centroid = float(np.dot(emb_norm, self.centroid))
        sims = self.sample @ emb_norm
        knn_mean_sim = float(np.mean(sims)) if sims.size else cos_centroid
        
        dist_centroid = max(0.0, 1.0 - cos_centroid)
        dist_knn = max(0.0, 1.0 - knn_mean_sim)
        drift_raw = 0.6 * dist_centroid + 0.4 * dist_knn
        drift_score = float(min(100.0, drift_raw * 100.0))
        
        return {
            "drift_score": drift_score,
            "cos_centroid": cos_centroid,
            "knn_mean_sim": knn_mean_sim
        }


def decode_and_nms(preds, ratio, input_size, conf_thr, nms_thr, img_hw=None):
    """Decodes raw YOLOX bounding boxes and applies multiclass NMS."""
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[1] < 6:
        return np.zeros((0, 6), dtype=np.float32)

    num_classes = preds.shape[1] - 5

    # Check if predictions are already decoded (absolute pixel scale, e.g. up to 640)
    is_decoded = float(np.max(preds[:, :4])) > 10.0

    if is_decoded:
        boxes_xyxy = np.zeros((preds.shape[0], 4), dtype=np.float32)
        boxes_xyxy[:, 0] = preds[:, 0] - preds[:, 2] / 2.0
        boxes_xyxy[:, 1] = preds[:, 1] - preds[:, 3] / 2.0
        boxes_xyxy[:, 2] = preds[:, 0] + preds[:, 2] / 2.0
        boxes_xyxy[:, 3] = preds[:, 1] + preds[:, 3] / 2.0
    else:
        # Strides & grids
        strides = [8, 16, 32]
        hsizes = [input_size[0] // s for s in strides]
        wsizes = [input_size[1] // s for s in strides]

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

    boxes_xyxy /= ratio
    if img_hw is not None:
        ih, iw = img_hw
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, iw)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, ih)

    obj_scores = preds[:, 4:5]
    cls_scores = preds[:, 5:]
    scores = obj_scores * cls_scores

    final_detections = []
    for cls_ind in range(num_classes):
        cls_sc = scores[:, cls_ind]
        keep = cls_sc > conf_thr
        if not np.any(keep):
            continue
        v_scores = cls_sc[keep]
        v_boxes = boxes_xyxy[keep]

        # OpenCV NMS expects [x, y, w, h] format
        xywh = np.zeros_like(v_boxes)
        xywh[:, 0] = v_boxes[:, 0]
        xywh[:, 1] = v_boxes[:, 1]
        xywh[:, 2] = v_boxes[:, 2] - v_boxes[:, 0]
        xywh[:, 3] = v_boxes[:, 3] - v_boxes[:, 1]

        idxs = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            v_scores.tolist(),
            float(conf_thr),
            float(nms_thr)
        )
        if len(idxs) == 0:
            continue
        for idx in idxs.flatten():
            final_detections.append([
                float(v_boxes[idx, 0]),
                float(v_boxes[idx, 1]),
                float(v_boxes[idx, 2]),
                float(v_boxes[idx, 3]),
                float(v_scores[idx]),
                int(cls_ind)
            ])

    return final_detections


def draw_detections(frame, detections, classes, drift_info=None):
    """Draws elegant boxes, text labels, and drift overlay onto the visual frame."""
    disp = frame.copy()

    # Overlay drift score at the top left if available
    if drift_info is not None:
        drift_score = drift_info.get("drift_score", 0.0)
        cos_c = drift_info.get("cos_centroid", 0.0)
        knn_s = drift_info.get("knn_mean_sim", 0.0)
        
        overlay_text = f"Drift Score: {drift_score:.1f}%  |  Centroid Sim: {cos_c:.4f}  |  KNN Sim: {knn_s:.4f}"
        
        h, w = disp.shape[:2]
        # Draw translucent dark bar at the top
        bar_height = 35
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)
        
        cv2.putText(
            disp,
            overlay_text,
            (15, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        label_text = classes[cls_id] if cls_id < len(classes) else f"cls_{cls_id}"
        label_text += f" {conf:.2f}"

        # Get color
        color = _COLORS[cls_id % len(_COLORS)]

        # Draw box
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

        # Label box styling
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        txt_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

        # Dynamic positioning (inside box if close to frame top boundary)
        y_text_top = y1 - txt_size[1] - 6
        if y_text_top < 0:
            back_tl = (x1, y1)
            back_br = (x1 + txt_size[0] + 6, y1 + txt_size[1] + 6)
            text_pos = (x1 + 3, y1 + txt_size[1] + 3)
        else:
            back_tl = (x1, y1 - txt_size[1] - 6)
            back_br = (x1 + txt_size[0] + 6, y1)
            text_pos = (x1 + 3, y1 - 3)

        cv2.rectangle(disp, back_tl, back_br, color, -1)

        # Choose text color (black or white) dynamically based on label block brightness
        brightness = sum(color) / 3.0
        txt_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

        cv2.putText(
            disp,
            label_text,
            text_pos,
            font,
            font_scale,
            txt_color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )
    return disp


class OpenVINOYOLOX:
    def __init__(self, model_path, device="cpu", test_size=(640, 640), conf_thr=0.4, nms_thr=0.45):
        self.model_path = model_path
        self.device = device.lower()
        self.input_size = test_size
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr

        import openvino as ov

        print(f"[INFO] Initializing OpenVINO Core runtime...")
        core = ov.Core()
        
        print(f"[INFO] Loading OpenVINO Model: {model_path}")
        ov_model = core.read_model(model_path)
        ov_device = "CPU" if self.device == "cpu" else "GPU"

        # Auto-reshape input to static [1, 3, H, W] for CPU performance stability
        try:
            ov_model.reshape({ov_model.inputs[0]: [1, 3, self.input_size[0], self.input_size[1]]})
        except Exception:
            pass

        print(f"[INFO] Compiling model for target device: {ov_device}")
        self.compiled_model = core.compile_model(ov_model, ov_device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.output_key = self.compiled_model.output(0)

        # Probe for multi-output embedding head (e.g. sakku_multi_output_int8.xml)
        self.embedding_output_index = None
        outputs = self.compiled_model.outputs
        if len(outputs) >= 2:
            for i, meta in enumerate(outputs):
                if i == 0:
                    continue
                names = list(meta.get_names())
                name_str = "".join(names).lower() if names else ""
                if any(t in name_str for t in ("embed", "drift", "feature", "vector")):
                    self.embedding_output_index = i
                    break
                try:
                    shape = list(meta.get_shape())
                    if shape[-1] == 512:
                        self.embedding_output_index = i
                        break
                except Exception:
                    pass

        if self.embedding_output_index is not None:
            print(f"[INFO] Multi-output model detected! Sibling embedding head found at index: {self.embedding_output_index}")
        else:
            print("[INFO] Standard single-output YOLOX detection model loaded.")

    def predict(self, frame):
        h, w = frame.shape[:2]
        img_norm, ratio = preproc(frame, self.input_size)
        blob = img_norm[np.newaxis, :].astype(np.float32)

        # Run OpenVINO inference for detection
        results = self.infer_request.infer({0: blob})
        raw_out = results[self.output_key]

        # Extract embeddings if multi-output model using centered letterbox BGR->RGB
        embedding = None
        if self.embedding_output_index is not None:
            try:
                # Use centered letterbox + BGR→RGB to exactly match reference bank
                rgb_blob = _centered_letterbox_rgb(frame, self.input_size)
                rgb_results = self.infer_request.infer({0: rgb_blob})
                emb_tensor = rgb_results[self.compiled_model.output(self.embedding_output_index)]
                v = np.asarray(emb_tensor, dtype=np.float32).reshape(-1)[:512]
                norm = float(np.linalg.norm(v))
                embedding = v / norm if norm > 1e-8 else v
            except Exception as e:
                print(f"[WARNING] Failed to extract embedding output: {e}")

        # Decode predictions and apply NMS
        detections = decode_and_nms(raw_out, ratio, self.input_size, self.conf_thr, self.nms_thr, img_hw=(h, w))
        return detections, embedding


def main():
    parser = argparse.ArgumentParser(description="Standalone Dedicated OpenVINO YOLOX Inference with Drift Scoring")
    parser.add_argument("--model_path", type=str, required=True, help="Path to OpenVINO .xml model file")
    parser.add_argument("--reference_path", type=str, default=None, help="Path to reference embeddings .npy file for drift scoring")
    parser.add_argument("--image_path", type=str, default=None, help="Path to an input image to process")
    parser.add_argument("--video_path", type=str, default=None, help="Path to an input video file or webcam (e.g. 0)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to write the output image or video")
    parser.add_argument("--classes_path", type=str, default=None, help="Path to class labels txt file (one label per line)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device choice (cpu or gpu)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    args = parser.parse_args()

    # Load classes
    classes = DEFAULT_CLASSES
    if args.classes_path and os.path.exists(args.classes_path):
        with open(args.classes_path, "r") as f:
            classes = [line.strip() for line in f if line.strip()]

    # Validate inputs
    if not args.image_path and not args.video_path:
        print("[ERROR] Please provide either --image_path or --video_path.")
        parser.print_help()
        sys.exit(1)

    # Validate file format
    if not args.model_path.lower().endswith(".xml"):
        print("[ERROR] This script only supports OpenVINO .xml model paths.")
        sys.exit(1)

    # Initialize model
    try:
        model = OpenVINOYOLOX(
            model_path=args.model_path,
            device=args.device,
            conf_thr=args.conf,
            nms_thr=args.nms
        )
    except Exception as e:
        print(f"[ERROR] Failed to compile OpenVINO model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize Drift Scorer if reference path is specified
    drift_scorer = None
    if args.reference_path:
        try:
            print(f"[INFO] Loading reference embeddings for drift scoring from: {args.reference_path}")
            drift_scorer = StandaloneDriftScorer(args.reference_path)
            print(f"[INFO] Loaded {drift_scorer.matrix.shape[0]} reference embeddings.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize drift scorer: {e}")
            sys.exit(1)

    # Image mode execution
    if args.image_path:
        frame = cv2.imread(args.image_path)
        if frame is None:
            print(f"[ERROR] Cannot read input image at: {args.image_path}")
            sys.exit(1)

        t0 = time.time()
        detections, embedding = model.predict(frame)
        latency = (time.time() - t0) * 1000.0
        
        print(f"[INFO] Inference complete in {latency:.2f}ms. Found {len(detections)} objects.")
        
        drift_info = None
        if embedding is not None:
            print(f"[INFO] Successfully extracted drift embedding: Shape {embedding.shape}")
            if drift_scorer:
                drift_info = drift_scorer.calculate_drift(embedding)
                print(f"[DRIFT] Score: {drift_info['drift_score']:.2f}% (Centroid Sim: {drift_info['cos_centroid']:.4f}, KNN Sim: {drift_info['knn_mean_sim']:.4f})")
        elif drift_scorer:
            print("[WARNING] Drift scorer was requested, but model did not output embedding branch.")

        annotated = draw_detections(frame, detections, classes, drift_info=drift_info)

        if args.output_path:
            cv2.imwrite(args.output_path, annotated)
            print(f"[INFO] Saved output image to: {args.output_path}")
        else:
            cv2.imshow("Dedicated OpenVINO YOLOX", annotated)
            print("[INFO] Press any key on the display window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Video mode execution
    elif args.video_path:
        source = int(args.video_path) if args.video_path.isdigit() else args.video_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {args.video_path}")
            sys.exit(1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if args.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output_path, fourcc, fps if fps > 0 else 30.0, (w, h))
            print(f"[INFO] Output video writer enabled: {args.output_path}")

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                detections, embedding = model.predict(frame)
                latency = (time.time() - t0) * 1000.0

                drift_info = None
                drift_msg = ""
                if embedding is not None and drift_scorer:
                    drift_info = drift_scorer.calculate_drift(embedding)
                    drift_msg = f" | Drift Score: {drift_info['drift_score']:.1f}%"

                annotated = draw_detections(frame, detections, classes, drift_info=drift_info)
                frame_idx += 1

                emb_msg = " [Embed: OK]" if embedding is not None else ""
                print(f"[Inference] Frame {frame_idx}/{total_frames if total_frames > 0 else 'N/A'} -> {len(detections)} objects in {latency:.2f}ms{emb_msg}{drift_msg}", end="\r")

                if writer:
                    writer.write(annotated)
                else:
                    cv2.imshow("Dedicated OpenVINO YOLOX", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
                        break
        finally:
            print("\n[INFO] Releasing video resources...")
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
