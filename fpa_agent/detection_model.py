import os
import importlib.util
import torch
import cv2
import numpy as np
from .tracker import ByteTracker


class DetectionModel:
    """Loads YOLOX-based detection model and performs inference with object tracking."""

    def __init__(
        self,
        pth_path,
        exp_path,
        classes_path,
        device='cpu',
        enable_tracking=True,
        drift_projection_path=None,
        drift_onnx_embedding_path=None,
        drift_input_size=None,
        drift_encoder="yolox_standard",
        drift_pool_mode="auto",
        drift_projection_type="linear_relu",
        drift_projection_weights=None,
    ):
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
        self._drift_projection_path = drift_projection_path
        self._drift_onnx_embedding_path = drift_onnx_embedding_path
        self._drift_input_size = drift_input_size
        self._drift_encoder = drift_encoder or "yolox_standard"
        self._drift_pool_mode = drift_pool_mode or "auto"
        self._drift_projection_type = drift_projection_type or "linear_relu"
        self._drift_projection_weights = drift_projection_weights
        self._drift_embedder = None
        self._drift_proj = None
        self.ort_embed_session = None
        self.ort_embed_input_name = None
        self._ort_embedding_output_index = None
        self._cached_frame_embedding = None

        # Tracking
        self.enable_tracking = enable_tracking
        self.tracker = ByteTracker(track_thresh=self.test_conf,
                                   match_thresh=0.3,
                                   max_time_lost=30) if enable_tracking else None
        self.frame_count = 0

        self._load_model(self.model_path, exp_path)

    def _load_model(self, model_path, exp_path):
        from yolox.data.data_augment import preproc
        # NOTE: we keep YOLOX postprocess as a fallback only.
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
        self._init_drift_embedder()

    def _init_drift_embedder(self):
        """YOLOX Standard: letterbox → backbone → neck → GAP concat → L2."""
        self._drift_embedder = None
        if self.model is None:
            return

        isize = self._drift_input_size
        if isize is None:
            isize = (640, 640)
        if isinstance(isize, (list, tuple)):
            isize = (int(isize[0]), int(isize[1]))
        else:
            isize = (int(isize), int(isize))

        enc = (self._drift_encoder or "yolox_standard").lower()
        legacy = enc in ("legacy", "linear_relu", "mlp_1024_512", "projection_head")

        try:
            from .embedding_extractor import create_drift_embedder

            kwargs = {}
            if legacy:
                weights = self._drift_projection_weights
                if not weights and self._drift_projection_path:
                    p = str(self._drift_projection_path)
                    if "projection" in p.lower() or "embed" in p.lower():
                        weights = p
                kwargs = {
                    "projection_type": self._drift_projection_type,
                    "projection_weights_path": weights,
                }
                enc = "legacy"

            if not legacy:
                kwargs["pool_mode"] = self._drift_pool_mode
            self._drift_embedder = create_drift_embedder(
                self.model,
                self.device,
                encoder=enc,
                input_size=isize,
                **kwargs,
            )
        except Exception as e:
            print(f"Drift embedder init failed: {e}")
            self._drift_embedder = None

    @staticmethod
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                return ckpt["model"]
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
        return ckpt

    def _try_load_drift_projection(self, model_path, projection_path=None):
        """Load Linear(backbone_dim -> 512) used when building embeddings.npy."""
        self._drift_proj = None
        paths = []
        if projection_path:
            paths.append(projection_path)
        if model_path and str(model_path).lower().endswith(".pth"):
            paths.append(model_path)

        for path in paths:
            if not path or not os.path.exists(path):
                continue
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                sd = self._extract_state_dict(ckpt)
                if not isinstance(sd, dict):
                    continue
                proj = self._find_linear_512(sd)
                if proj is not None:
                    in_dim, layer = proj
                    layer.eval()
                    layer.to(self.device)
                    self._drift_proj = layer
                    return
            except Exception:
                continue

    @staticmethod
    def _find_linear_512(state_dict):
        """Find nn.Linear with out_features=512 (embedding head) in a checkpoint."""
        import torch.nn as nn

        weight_keys = [
            k for k, v in state_dict.items()
            if k.endswith(".weight")
            and hasattr(v, "shape")
            and len(v.shape) == 2
            and int(v.shape[0]) == 512
        ]
        if not weight_keys:
            return None

        def score_key(k):
            kl = k.lower()
            priority = 0
            for token in ("embed", "drift", "proj", "neck", "head_fc"):
                if token in kl:
                    priority -= 10
            return (priority, len(k))

        weight_keys.sort(key=score_key)
        wkey = weight_keys[0]
        prefix = wkey[: -len(".weight")]
        bkey = f"{prefix}.bias"
        w = state_dict[wkey]
        b = state_dict.get(bkey)
        in_dim = int(w.shape[1])
        layer = nn.Linear(in_dim, 512, bias=b is not None)
        layer.weight.data.copy_(w.float())
        if b is not None:
            layer.bias.data.copy_(b.float())
        return in_dim, layer

    def can_encode_drift_embedding(self) -> bool:
        """YOLOX standard embedder (.pth) and/or ONNX embedding session."""
        if self._drift_embedder is not None:
            return True
        if self.model is not None and hasattr(self.model, "backbone") and self.ort_session is None:
            return True
        if self.ort_embed_session is not None:
            return True
        if self.ort_session is not None and self._ort_embedding_output_index is not None:
            return True
        return False

    def drift_encoder_description(self) -> str:
        if self._drift_embedder is not None:
            return self._drift_embedder.description()
        if self.model is not None and hasattr(self.model, "backbone"):
            return "YOLOX neck concat @ 640"
        if self.ort_embed_session is not None:
            return "YOLOX embedding ONNX"
        if self._ort_embedding_output_index is not None:
            return "YOLOX detection ONNX (embedding output)"
        return "not available"

    @staticmethod
    def _l2_normalize_numpy(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size > 512:
            v = v[:512]
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return v
        return (v / n).astype(np.float32)

    @staticmethod
    def _parse_embedding_array(arr: np.ndarray) -> np.ndarray:
        v = np.asarray(arr, dtype=np.float32).reshape(-1)
        if v.size < 512:
            raise RuntimeError(f"ONNX embedding output size {v.size}, expected 512")
        return DetectionModel._l2_normalize_numpy(v[:512])

    def encode_frame_embedding(self, image_bgr) -> np.ndarray:
        """
        YOLOX CSPDarknet → GAP → Linear(→512), L2-normalized.
        Works for .pth, dedicated embedding .onnx, or multi-output detection .onnx.
        """
        if not self.can_encode_drift_embedding():
            raise RuntimeError(
                "Drift embedding not available. For ONNX detection, set drift.onnx_embedding_path "
                "or export with: python export_embedding_onnx.py --pth ... --exp ... --output *_embed.onnx"
            )

        if self._drift_embedder is not None:
            return self._drift_embedder.extract_bgr(image_bgr)

        image_norm, _ratio = self.preproc(image_bgr, self.input_size)
        blob = image_norm[np.newaxis, :].astype(np.float32)

        if self._cached_frame_embedding is not None:
            emb = self._cached_frame_embedding.copy()
            self._cached_frame_embedding = None
            return emb

        if self.ort_embed_session is not None:
            out = self.ort_embed_session.run(
                None, {self.ort_embed_input_name: blob}
            )[0]
            return self._parse_embedding_array(out)

        if self.ort_session is not None and self._ort_embedding_output_index is not None:
            outs = self.ort_session.run(None, {self.ort_input_name: blob})
            return self._parse_embedding_array(outs[self._ort_embedding_output_index])

        x = torch.from_numpy(blob).to(self.device)
        with torch.no_grad():
            fpn_outs = self.model.backbone(x)
            feat = fpn_outs[-1] if isinstance(fpn_outs, (list, tuple)) else fpn_outs
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            if self._drift_proj is not None:
                emb = self._drift_proj(pooled)
            elif pooled.shape[1] == 512:
                emb = pooled
            else:
                raise RuntimeError(
                    f"Backbone pooled dim is {pooled.shape[1]}, expected 512. "
                    "Set drift.projection_path or export drift.onnx_embedding_path."
                )
            emb = torch.nn.functional.normalize(emb, dim=1)
            return emb.cpu().numpy()[0].astype(np.float32)

    @staticmethod
    def _onnx_providers(device_str: str):
        import onnxruntime as ort

        use_cuda = (device_str or "cpu").lower() in ("cuda", "gpu")
        if use_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _create_ort_session(self, onnx_path: str):
        import onnxruntime as ort

        providers = self._onnx_providers(self.device_str)
        try:
            return ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    @staticmethod
    def _resolve_embedding_onnx_path(path: str) -> str:
        """
        ONNX models use a .onnx protobuf file; large weights may live in a sibling
        .onnx.data file. Config must point at the .onnx file, not the .data blob.
        """
        if not path:
            return path
        path = path.strip()
        lower = path.lower()
        if lower.endswith(".onnx.data"):
            corrected = path[: -len(".data")]
            print(
                f"drift.onnx_embedding_path should be the .onnx file, not .data. "
                f"Using: {corrected}"
            )
            return corrected
        if lower.endswith(".data") and not lower.endswith(".onnx.data"):
            raise ValueError(
                f"Invalid drift ONNX path (expected .onnx): {path}"
            )
        return path

    def _embedding_onnx_candidates(self, detection_onnx_path: str):
        explicit = self._drift_onnx_embedding_path
        if explicit:
            yield self._resolve_embedding_onnx_path(explicit)
        if not detection_onnx_path:
            return
        base, _ = os.path.splitext(detection_onnx_path)
        folder = os.path.dirname(detection_onnx_path) or "."
        stem = os.path.basename(base)
        for name in (
            f"{base}_embed.onnx",
            f"{base}_embedding.onnx",
            f"{base}-embed.onnx",
            os.path.join(folder, f"{stem}_embed.onnx"),
            os.path.join(folder, "embedding.onnx"),
            os.path.join(folder, "embed.onnx"),
            os.path.join(folder, "drift_embedding.onnx"),
        ):
            yield name

    def _load_onnx_embedding_session(self, detection_onnx_path: str):
        self.ort_embed_session = None
        self.ort_embed_input_name = None
        seen = set()
        for path in self._embedding_onnx_candidates(detection_onnx_path):
            if not path or path in seen:
                continue
            seen.add(path)
            if not os.path.exists(path):
                continue
            session = self._create_ort_session(path)
            inputs = session.get_inputs()
            if not inputs:
                continue
            self.ort_embed_session = session
            self.ort_embed_input_name = inputs[0].name
            return path
        return None

    def _probe_embedding_output_index(self, detection_output_index: int = 0):
        """Run a dummy forward pass to find a 512-D embedding output (multi-output ONNX)."""
        outputs = self.ort_session.get_outputs()
        if len(outputs) < 2:
            return None

        for i, meta in enumerate(outputs):
            if i == detection_output_index:
                continue
            name = meta.name.lower()
            if any(t in name for t in ("embed", "drift", "feature", "vector")):
                return i

        h = w = int(self.input_size[0]) if isinstance(self.input_size, (list, tuple)) else 640
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        try:
            ort_outs = self.ort_session.run(None, {self.ort_input_name: dummy})
        except Exception:
            return None

        for i, arr in enumerate(ort_outs):
            if i == detection_output_index:
                continue
            flat = np.asarray(arr).reshape(-1)
            if flat.size == 512:
                return i
            sh = np.asarray(arr).shape
            if len(sh) >= 1 and sh[-1] == 512:
                return i
        return None

    def _load_onnx_model(self, onnx_path: str):
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required to run .onnx models. "
                "Install it via `pip install onnxruntime` (or onnxruntime-gpu)."
            ) from e

        self.ort_session = self._create_ort_session(onnx_path)

        inputs = self.ort_session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")
        self.ort_input_name = inputs[0].name
        self.model = None
        self.ort_num_classes = None

        embed_path = self._load_onnx_embedding_session(onnx_path)
        self._ort_embedding_output_index = None
        if embed_path:
            self._ort_embedding_source = embed_path
        else:
            self._ort_embedding_output_index = self._probe_embedding_output_index(0)
            self._ort_embedding_source = (
                "detection ONNX multi-output"
                if self._ort_embedding_output_index is not None
                else None
            )
        self._cached_frame_embedding = None

    @staticmethod
    def _safe_ratio(ratio: float) -> float:
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(r) or r <= 1e-6:
            return 1.0
        return r

    @staticmethod
    def _safe_bbox_int(x1, y1, x2, y2, img_w: int, img_h: int):
        """Clamp to image bounds; avoid OverflowError from inf/nan ONNX decode artifacts."""

        def _c(v, lo, hi):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return lo
            if not np.isfinite(fv):
                return lo
            return int(max(lo, min(hi, fv)))

        x1i = _c(x1, 0, max(0, img_w - 1))
        y1i = _c(y1, 0, max(0, img_h - 1))
        x2i = _c(x2, 0, img_w)
        y2i = _c(y2, 0, img_h)
        if x2i <= x1i or y2i <= y1i:
            return None
        return x1i, y1i, x2i, y2i

    def _yolox_decode_and_nms(self, raw_preds: np.ndarray, ratio: float, img_hw=None):
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
        ratio = self._safe_ratio(ratio)

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
        # Clip log-wh before exp — int8 ONNX can emit extreme logits on some frames.
        wh_log = np.clip(preds[:, 2:4], -20.0, 20.0)
        with np.errstate(over="ignore", invalid="ignore"):
            preds_wh = np.exp(wh_log) * expanded_strides[0]
        preds_wh = np.nan_to_num(preds_wh, nan=0.0, posinf=0.0, neginf=0.0)
        obj = preds[:, 4:5]                                                   # [N, 1]
        cls_scores = preds[:, 5:]                                             # [N, C]

        # xyxy
        boxes_xyxy = np.zeros((preds.shape[0], 4), dtype=np.float32)
        boxes_xyxy[:, 0] = preds_xy[:, 0] - preds_wh[:, 0] / 2.0
        boxes_xyxy[:, 1] = preds_xy[:, 1] - preds_wh[:, 1] / 2.0
        boxes_xyxy[:, 2] = preds_xy[:, 0] + preds_wh[:, 0] / 2.0
        boxes_xyxy[:, 3] = preds_xy[:, 1] + preds_wh[:, 1] / 2.0

        # Scale back to original image coords
        boxes_xyxy /= ratio
        boxes_xyxy = np.nan_to_num(boxes_xyxy, nan=0.0, posinf=0.0, neginf=0.0)

        if img_hw is not None:
            ih, iw = int(img_hw[0]), int(img_hw[1])
            max_coord = float(max(iw, ih) * 4)
            boxes_xyxy = np.clip(boxes_xyxy, -max_coord, max_coord)

        scores = obj * cls_scores                                             # [N, C]
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

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
            finite = np.isfinite(v_boxes).all(axis=1) & np.isfinite(v_scores)
            if not np.any(finite):
                continue
            v_scores = v_scores[finite]
            v_boxes = v_boxes[finite]

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

    def _head_decodes_in_inference(self) -> bool:
        """YOLOX .pth eval mode returns decoded cxcywh; do not run raw-grid decode again."""
        if self.model is None:
            return False
        head = getattr(self.model, "head", None)
        return bool(getattr(head, "decode_in_inference", False))

    def predict(self, image_bgr):
        img_h, img_w = image_bgr.shape[:2]
        if self.model is None and self.ort_session is None:
            detections = [{'bbox': [int(img_w * 0.2), int(img_h * 0.2), int(img_w * 0.8), int(img_h * 0.8)],
                     'label': 'dummy', 'conf': 0.5}]
        else:
            image_norm, ratio = self.preproc(image_bgr, self.input_size)
            image_norm = image_norm[np.newaxis, :].astype(np.float32)
            if self.ort_session is not None:
                ort_outs = self.ort_session.run(None, {self.ort_input_name: image_norm})
                self._cached_frame_embedding = None
                if (
                    ort_outs
                    and self._ort_embedding_output_index is not None
                    and self._ort_embedding_output_index < len(ort_outs)
                ):
                    try:
                        self._cached_frame_embedding = self._parse_embedding_array(
                            ort_outs[self._ort_embedding_output_index]
                        )
                    except Exception:
                        pass
                if not ort_outs:
                    outputs = None
                else:
                    raw0 = ort_outs[0]
                    dets = self._yolox_decode_and_nms(
                        raw0, ratio, img_hw=(img_h, img_w)
                    )
                    outputs = dets
            else:
                image_norm_t = torch.from_numpy(image_norm).to(self.device)
                with torch.no_grad():
                    raw_out = self.model(image_norm_t)

                if self._head_decodes_in_inference():
                    # Head already decoded to letterbox-space cxcywh + sigmoid conf/cls.
                    outputs = self.postprocess(
                        raw_out, self.exp.num_classes, self.test_conf, self.nms_thr
                    )[0]
                else:
                    raw_tensor = raw_out
                    if isinstance(raw_out, (tuple, list)) and len(raw_out) > 0:
                        raw_tensor = raw_out[0]

                    decoded = None
                    try:
                        raw_np = raw_tensor.detach().float().cpu().numpy()
                        if raw_np.ndim in (2, 3) and raw_np.shape[-1] >= 6:
                            decoded = self._yolox_decode_and_nms(
                                raw_np, ratio, img_hw=(img_h, img_w)
                            )
                    except Exception:
                        decoded = None

                    if decoded is not None and len(decoded) > 0:
                        outputs = decoded
                    else:
                        outputs = self.postprocess(
                            raw_out, self.exp.num_classes, self.test_conf, self.nms_thr
                        )[0]

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
                    box = self._safe_bbox_int(x1, y1, x2, y2, img_w, img_h)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    label = self.classes[cls] if cls < len(self.classes) else f'class_{cls}'
                    detections.append({'bbox': [x1, y1, x2, y2],
                                       'label': label,
                                       'cls_id': int(cls),
                                       'conf': float(score)})

        # Apply tracking if enabled
        if self.enable_tracking and self.tracker:
            self.frame_count += 1
            tracked_detections = self.tracker.update(detections, self.frame_count)
            return tracked_detections
        
        return detections
