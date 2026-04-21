import os
import importlib.util
import torch
import cv2
import numpy as np
from .tracker import ByteTracker


class DetectionModel:
    """Loads YOLOX-based detection model and performs inference with object tracking."""

    def __init__(self, pth_path, exp_path, classes_path, device='cpu', enable_tracking=True):
        self.device = torch.device(device)
        self.classes = self._load_classes(classes_path)
        self.model = None
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

        self._load_yolox_model(pth_path, exp_path)

    def _load_yolox_model(self, pth_path, exp_path):
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

        ckpt = torch.load(pth_path, map_location=self.device , weights_only=False)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.to(self.device)
        self.model.eval()

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
        if self.model is None:
            h, w, _ = image_bgr.shape
            detections = [{'bbox': [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                     'label': 'dummy', 'conf': 0.5}]
        else:
            image_norm, ratio = self.preproc(image_bgr, self.input_size)
            image_norm = image_norm[np.newaxis, :].astype(np.float32)
            image_norm = torch.from_numpy(image_norm).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_norm)
                outputs = self.postprocess(outputs, self.exp.num_classes, self.test_conf, self.nms_thr)[0]

            detections = []
            if outputs is not None:
                outputs = outputs.cpu().numpy()
                for det in outputs:
                    # YOLOX postprocess: (x1,y1,x2,y2, obj_conf, class_conf, class_id); do not use [:6] or class id is wrong.
                    if len(det) >= 7:
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        obj_conf = float(det[4])
                        class_conf = float(det[5])
                        cls = int(det[6])
                        score = obj_conf * class_conf
                    elif len(det) >= 6:
                        x1, y1, x2, y2, score, cls = det[0], det[1], det[2], det[3], det[4], int(det[5])
                    else:
                        continue
                    x1 /= ratio
                    y1 /= ratio
                    x2 /= ratio
                    y2 /= ratio
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
