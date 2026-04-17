import os
import importlib.util
import torch
import cv2
import numpy as np


class DetectionModel:
    """Loads YOLOX-based detection model and performs inference."""

    def __init__(self, pth_path, exp_path, classes_path, device='cpu'):
        self.device = torch.device(device)
        self.classes = self._load_classes(classes_path)
        self.model = None
        self.exp = None
        self.input_size = (640, 640)
        self.test_conf = 0.4
        self.nms_thr = 0.45
        self.preproc = None

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

        ckpt = torch.load(pth_path, map_location=self.device)
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

    def predict(self, image_bgr):
        if self.model is None:
            h, w, _ = image_bgr.shape
            return [{'bbox': [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                     'label': 'dummy', 'conf': 0.5}]

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
                if len(det) >= 6:
                    x1, y1, x2, y2, score, cls = det[:6]
                    x1 /= ratio
                    y1 /= ratio
                    x2 /= ratio
                    y2 /= ratio
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    label = self.classes[int(cls)] if int(cls) < len(self.classes) else f'class_{int(cls)}'
                    detections.append({'bbox': [x1, y1, x2, y2],
                                       'label': label,
                                       'conf': float(score)})
        return detections
