import os
import importlib.util
import torch
import cv2
import numpy as np
from .tracker import ByteTracker


class DetectionModel:
    """Loads YOLOX-based detection model and performs inference with object tracking."""

    def __init__(self, pth_path, exp_path, classes_path, device='cpu', enable_tracking=True):
        # NOTE: arg name kept as pth_path for backward compatibility; it may be a .pth or .onnx path.
        self.model_path = pth_path
        self.device_str = device or 'cpu'
        self.device = torch.device(self.device_str)
        self.classes = self._load_classes(classes_path)
        self.model = None
        self.ort_session = None
        self.ort_input_name = None
        self.exp = None
        self.input_size = (640, 640)
        self.test_conf = 0.4
        self.nms_thr = 0.45
        self.preproc = None
        
        # Tracking
        self.enable_tracking = enable_tracking
        self.tracker = ByteTracker(track_thresh=self.test_conf,
                                   match_thresh=0.3,
                                   max_time_lost=30) if enable_tracking else None
        self.frame_count = 0

        self._load_model(self.model_path, exp_path)

    def _load_model(self, model_path, exp_path):
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        self.postprocess = postprocess
        self.preproc = preproc

        if exp_path and os.path.exists(exp_path):
            spec = importlib.util.spec_from_file_location("custom_exp", exp_path)
            exp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exp_module)
            exp_class = getattr(exp_module, "Exp", None)
            if exp_class is None:
                raise RuntimeError("Experiment file does not define an Exp class.")
            self.exp = exp_class()
        else:
            from yolox.exp import get_exp
            self.exp = get_exp("yolox_s", None)

        self.input_size = self.exp.test_size if hasattr(self.exp, "test_size") else (640, 640)
        self.model = self.exp.get_model()
        model_path_lower = (model_path or "").lower()
        if model_path_lower.endswith(".onnx"):
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, pth_path):
        ckpt = torch.load(pth_path, map_location=self.device, weights_only=False)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.to(self.device)
        self.model.eval()

    def _load_onnx_model(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required to run .onnx models. "
                "Install it via `pip install onnxruntime` (or onnxruntime-gpu)."
            ) from e

        # Providers are driven by config.json `model.device`.
        # - cpu  -> CPUExecutionProvider only
        # - cuda -> CUDAExecutionProvider with CPU fallback (if some ops fall back)
        device_lower = (self.device_str or "cpu").lower()
        use_cuda = device_lower.startswith("cuda") or device_lower == "gpu"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

        try:
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            # Fallback to CPU if CUDA provider isn't available.
            self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        inputs = self.ort_session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")
        self.ort_input_name = inputs[0].name
        # Keep `self.model` as None to signal ONNX backend in predict().
        self.model = None

        # ONNX YOLOX exports often output raw predictions with shape [1, N, 5 + num_classes]
        # (e.g. [1, 8400, 18] => 13 classes). We'll infer this at runtime in predict().
        self.ort_num_classes = None

    def _yolox_decode_and_nms(self, raw_preds: np.ndarray, ratio: float):
        """
        Decode raw YOLOX predictions ([1, N, 5+C]) into xyxy boxes, run multiclass NMS,
        return detections as Nx6: x1,y1,x2,y2,score,cls.
        """
        # raw_preds: [1, N, 5+C] or [N, 5+C]
        preds = raw_preds[0] if raw_preds.ndim == 3 else raw_preds  # [N, 5+C]
        if preds.ndim != 2 or preds.shape[1] < 6:
            return np.zeros((0, 6), dtype=np.float32)

        num_classes = int(preds.shape[1] - 5)
        self.ort_num_classes = num_classes

        input_hw = self.input_size[0] if isinstance(self.input_size, (list, tuple)) else int(self.input_size)
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

        grids = np.concatenate(grids, axis=1).astype(np.float32)              # [1, N, 2]
        expanded_strides = np.concatenate(expanded_strides, axis=1)           # [1, N, 1]

        # Decode in-place like your working script.
        preds = preds.astype(np.float32, copy=False)
        preds_xy = (preds[:, :2] + grids[0]) * expanded_strides[0]            # [N, 2]
        preds_wh = np.exp(preds[:, 2:4]) * expanded_strides[0]                # [N, 2]
        obj = preds[:, 4:5]                                                   # [N, 1]
        cls_scores = preds[:, 5:]                                             # [N, C]

        # xyxy
        boxes_xyxy = np.zeros((preds.shape[0], 4), dtype=np.float32)
        boxes_xyxy[:, 0] = preds_xy[:, 0] - preds_wh[:, 0] / 2.0
        boxes_xyxy[:, 1] = preds_xy[:, 1] - preds_wh[:, 1] / 2.0
        boxes_xyxy[:, 2] = preds_xy[:, 0] + preds_wh[:, 0] / 2.0
        boxes_xyxy[:, 3] = preds_xy[:, 1] + preds_wh[:, 1] / 2.0

        # Scale back to original image coords
        boxes_xyxy /= float(ratio)

        scores = obj * cls_scores                                             # [N, C]

        # Multiclass NMS (same idea as your script)
        final = []
        conf_thres = float(self.test_conf)
        nms_thres = float(self.nms_thr)
        for cls_ind in range(num_classes):
            cls_sc = scores[:, cls_ind]
            keep = cls_sc > conf_thres
            if int(keep.sum()) == 0:
                continue
            v_scores = cls_sc[keep]
            v_boxes = boxes_xyxy[keep]

            # OpenCV expects xywh
            xywh = np.zeros_like(v_boxes)
            xywh[:, 0] = v_boxes[:, 0]
            xywh[:, 1] = v_boxes[:, 1]
            xywh[:, 2] = v_boxes[:, 2] - v_boxes[:, 0]
            xywh[:, 3] = v_boxes[:, 3] - v_boxes[:, 1]

            idxs = cv2.dnn.NMSBoxes(
                xywh.tolist(),
                v_scores.tolist(),
                conf_thres,
                nms_thres,
            )
            if len(idxs) == 0:
                continue
            for i in idxs.flatten():
                final.append([v_boxes[i, 0], v_boxes[i, 1], v_boxes[i, 2], v_boxes[i, 3], v_scores[i], float(cls_ind)])

        return np.asarray(final, dtype=np.float32) if final else np.zeros((0, 6), dtype=np.float32)

    def _load_classes(self, classes_path):
        if not classes_path or not os.path.exists(classes_path):
            return ['object']
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes if classes else ['object']

    def get_anomalies(self):
        """Get detected false positives and missed detections from tracking history."""
        if self.enable_tracking and self.tracker:
            return self.tracker.get_anomalies()
        return {
            'false_positives': [],
            'missed_detections': [],
            'total_tracks': 0,
            'active_tracks': 0
        }

    def get_track_summary(self, track_id):
        """Get detailed summary of a specific track."""
        if self.enable_tracking and self.tracker:
            return self.tracker.get_track_summary(track_id)
        return None

    def reset_tracker(self):
        """Reset tracker for new video/stream."""
        if self.enable_tracking and self.tracker:
            self.tracker.reset()
            self.frame_count = 0

    def predict(self, image_bgr):
        if self.model is None and self.ort_session is None:
            h, w, _ = image_bgr.shape
            detections = [{'bbox': [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                     'label': 'dummy', 'conf': 0.5}]
        else:
            image_norm, ratio = self.preproc(image_bgr, self.input_size)
            image_norm = image_norm[np.newaxis, :].astype(np.float32)
            if self.ort_session is not None:
                ort_outs = self.ort_session.run(None, {self.ort_input_name: image_norm})
                if not ort_outs:
                    outputs = None
                else:
                    # ONNX export returns raw preds (e.g. [1, 8400, 5+C]); decode + NMS here.
                    raw0 = ort_outs[0]
                    dets = self._yolox_decode_and_nms(raw0, ratio)
                    outputs = dets  # Nx6 float32: x1,y1,x2,y2,score,cls
            else:
                image_norm_t = torch.from_numpy(image_norm).to(self.device)
                with torch.no_grad():
                    outputs = self.model(image_norm_t)
                    outputs = self.postprocess(outputs, self.exp.num_classes, self.test_conf, self.nms_thr)[0]

            detections = []
            if outputs is not None:
                outputs = outputs.cpu().numpy() if hasattr(outputs, "cpu") else np.asarray(outputs)
                for det in outputs:
                    # Two possible formats:
                    # - PyTorch postprocess: (x1,y1,x2,y2,obj_conf,class_conf,class_id)
                    # - ONNX decoded:        (x1,y1,x2,y2,score,class_id)
                    if len(det) >= 7:
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        obj_conf = float(det[4])
                        class_conf = float(det[5])
                        cls = int(det[6])
                        score = obj_conf * class_conf
                        x1 /= ratio
                        y1 /= ratio
                        x2 /= ratio
                        y2 /= ratio
                    elif len(det) >= 6:
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        score = float(det[4])
                        cls = int(det[5])
                    else:
                        continue
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    label = self.classes[cls] if cls < len(self.classes) else f'class_{cls}'
                    detections.append({'bbox': [x1, y1, x2, y2],
                                       'label': label,
                                       'conf': float(score)})
        
        # Apply tracking if enabled
        if self.enable_tracking and self.tracker:
            self.frame_count += 1
            tracked_detections = self.tracker.update(detections, self.frame_count)
            return tracked_detections
        
        return detections
