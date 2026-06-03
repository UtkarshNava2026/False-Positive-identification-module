#!/usr/bin/env python3
"""
Standalone YOLOX Inference Script
---------------------------------
This script is self-contained and can be shared. It only requires:
  - opencv-python (cv2)
  - numpy
  - PyTorch (only if using PyTorch .pth models)
  - onnxruntime (only if using ONNX .onnx models)
  - openvino (only if using OpenVINO .xml models)

It does NOT depend on any local modules (such as fpa_agent).

Usage:
  # Run ONNX inference on an image
  python standalone_inference.py --model_path weights/sakku.onnx --image_path input.jpg --output_path output.jpg

  # Run OpenVINO INT8 inference on a video
  python standalone_inference.py --model_path weights/sakku_multi_output_int8.xml --video_path video.mp4 --output_path out.mp4
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
    """Letterbox image to input size with standard padding value 114."""
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


def decode_and_nms(preds, ratio, input_size, conf_thr, nms_thr, img_hw=None):
    """
    Decodes raw YOLOX network outputs ([1, N, 85] or [N, 85]) into bounding boxes
    and applies OpenCV multiclass non-maximum suppression (NMS).
    """
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[1] < 6:
        return np.zeros((0, 6), dtype=np.float32)

    num_classes = preds.shape[1] - 5

    # Check if predictions are already decoded (e.g. PyTorch eval mode coordinates scale)
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


def draw_detections(frame, detections, classes):
    """Draws elegant boxes and text labels onto the visual frame."""
    disp = frame.copy()
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


class StandaloneYOLOX:
    def __init__(self, model_path, exp_path=None, backend=None, device="cpu", test_size=(640, 640), conf_thr=0.4, nms_thr=0.45):
        self.model_path = model_path
        self.exp_path = exp_path
        self.backend = backend
        self.device = device.lower()
        self.input_size = test_size
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr

        if not self.backend:
            # Auto-detect backend from extension
            ext = os.path.splitext(model_path)[1].lower()
            if ext == ".xml":
                self.backend = "openvino"
            elif ext == ".onnx":
                self.backend = "onnx"
            elif ext in (".pth", ".pt"):
                self.backend = "pytorch"
            else:
                self.backend = "pytorch"

        print(f"[INFO] Initializing model: {os.path.basename(model_path)}")
        print(f"[INFO] Selected backend: {self.backend} (device: {self.device})")

        if self.backend == "openvino":
            self._init_openvino()
        elif self.backend == "onnx":
            self._init_onnx()
        elif self.backend == "pytorch":
            self._init_pytorch()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_openvino(self):
        import openvino as ov
        core = ov.Core()
        ov_model = core.read_model(self.model_path)
        ov_device = "CPU" if self.device == "cpu" else "GPU"
        
        # Dynamic shape to static shape conversion if needed for CPU compatibility
        try:
            ov_model.reshape({ov_model.inputs[0]: [1, 3, self.input_size[0], self.input_size[1]]})
        except Exception:
            pass

        self.compiled_model = core.compile_model(ov_model, ov_device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.output_key = self.compiled_model.output(0)

    def _init_onnx(self):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "gpu" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _init_pytorch(self):
        import torch
        import importlib.util
        if not self.exp_path or not os.path.exists(self.exp_path):
            raise ValueError("PyTorch backend (.pth) requires a valid YOLOX --exp_path file.")
        
        spec = importlib.util.spec_from_file_location("custom_exp", self.exp_path)
        exp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp_module)
        exp = exp_module.Exp()
        self.input_size = getattr(exp, "test_size", self.input_size)
        self.model = exp.get_model()

        device_obj = torch.device("cuda" if self.device == "gpu" else "cpu")
        ckpt = torch.load(self.model_path, map_location=device_obj)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(device_obj)
        self.model.eval()

    def predict(self, frame):
        h, w = frame.shape[:2]
        img_norm, ratio = preproc(frame, self.input_size)
        blob = img_norm[np.newaxis, :].astype(np.float32)

        if self.backend == "openvino":
            results = self.infer_request.infer({0: blob})
            raw_out = results[self.output_key]
        elif self.backend == "onnx":
            raw_out = self.session.run(None, {self.input_name: blob})[0]
        elif self.backend == "pytorch":
            import torch
            device_obj = torch.device("cuda" if self.device == "gpu" else "cpu")
            tensor = torch.from_numpy(blob).to(device_obj)
            with torch.no_grad():
                out = self.model(tensor)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                raw_out = out.cpu().numpy()

        # Decodes coordinates and executes NMS
        return decode_and_nms(raw_out, ratio, self.input_size, self.conf_thr, self.nms_thr, img_hw=(h, w))


def main():
    parser = argparse.ArgumentParser(description="Standalone YOLOX Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file (.xml, .onnx, or .pth)")
    parser.add_argument("--exp_path", type=str, default=None, help="Experiment file (.py) — only required for PyTorch (.pth)")
    parser.add_argument("--image_path", type=str, default=None, help="Path to an input image to process")
    parser.add_argument("--video_path", type=str, default=None, help="Path to an input video file or camera stream (e.g. RTSP/Webcam)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to write the output image or video")
    parser.add_argument("--classes_path", type=str, default=None, help="Path to class labels txt file (one label per line)")
    parser.add_argument("--backend", type=str, choices=["pytorch", "onnx", "openvino"], default=None, help="Force backend (otherwise auto-detected)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device choice (cpu or gpu)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    args = parser.parse_args()

    # Load classes
    classes = DEFAULT_CLASSES
    if args.classes_path and os.path.exists(args.classes_path):
        with open(args.classes_path, "r") as f:
            classes = [line.strip() for line in f if line.strip()]

    # Check input parameters
    if not args.image_path and not args.video_path:
        print("[ERROR] Please provide either --image_path or --video_path.")
        parser.print_help()
        sys.exit(1)

    # Initialize model loader
    try:
        model = StandaloneYOLOX(
            model_path=args.model_path,
            exp_path=args.exp_path,
            backend=args.backend,
            device=args.device,
            conf_thr=args.conf,
            nms_thr=args.nms
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Image mode execution
    if args.image_path:
        frame = cv2.imread(args.image_path)
        if frame is None:
            print(f"[ERROR] Cannot read input image at: {args.image_path}")
            sys.exit(1)

        t0 = time.time()
        detections = model.predict(frame)
        latency = (time.time() - t0) * 1000.0
        print(f"[INFO] Detections complete in {latency:.2f}ms. Found {len(detections)} objects.")

        annotated = draw_detections(frame, detections, classes)

        if args.output_path:
            cv2.imwrite(args.output_path, annotated)
            print(f"[INFO] Saved output visual representation to: {args.output_path}")
        else:
            cv2.imshow("Standalone YOLOX Inference", annotated)
            print("[INFO] Press any key on the display window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Video mode execution
    elif args.video_path:
        # Check if integer input for webcam
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
                detections = model.predict(frame)
                latency = (time.time() - t0) * 1000.0

                annotated = draw_detections(frame, detections, classes)
                frame_idx += 1

                print(f"[Inference] Frame {frame_idx}/{total_frames if total_frames > 0 else 'N/A'} -> {len(detections)} objects in {latency:.2f}ms", end="\r")

                if writer:
                    writer.write(annotated)
                else:
                    cv2.imshow("Standalone YOLOX Inference", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
                        break
        finally:
            print("\n[INFO] Releasing resources...")
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
