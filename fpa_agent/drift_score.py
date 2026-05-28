"""
Embedding-based data drift vs embeddings.npy / embeddings.pkl (N × 512).

Preferred live encoder (matches your reference bank):
  YOLOX CSPDarknet → global avg pool → Linear(→512)  [same .pth as detection]

Optional fallback:
  ResNet-18 (ImageNet) — only if drift.encoder = "resnet" (not comparable to YOLOX bank).
"""

from __future__ import annotations

import os
import pickle
from typing import Callable, Optional, Tuple

import numpy as np


def load_reference_embeddings(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Reference embeddings not found: {path}")

    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    elif lower.endswith(".pkl"):
        with open(path, "rb") as f:
            arr = pickle.load(f)
    else:
        raise ValueError("Reference embeddings must be .npy or .pkl")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {arr.shape}")
    return arr


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def _l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < 1e-8:
        return x
    return x / n


class ReferenceEmbeddingStore:
    def __init__(self, matrix: np.ndarray, knn_sample_size: int = 2048, seed: int = 42):
        self.matrix = _l2_normalize_rows(np.asarray(matrix, dtype=np.float32))
        self.dim = int(self.matrix.shape[1])
        self.centroid = _l2_normalize_vec(self.matrix.mean(axis=0))

        n = self.matrix.shape[0]
        k = min(int(knn_sample_size), n) if knn_sample_size else n
        if k < n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=k, replace=False)
            self.sample = self.matrix[idx]
        else:
            self.sample = self.matrix

    @classmethod
    def from_path(cls, path: str, knn_sample_size: int = 2048) -> "ReferenceEmbeddingStore":
        return cls(load_reference_embeddings(path), knn_sample_size=knn_sample_size)


class ResNetFrameEmbedder:
    """Fallback only — different semantics from YOLOX-trained embeddings."""

    def __init__(self, device: str = "cpu"):
        self.device_str = device or "cpu"
        self._model = None
        self._transform = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        import cv2

        self._cv2 = cv2
        device = torch.device(self.device_str if self.device_str != "gpu" else "cuda")
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=True)
        backbone.fc = torch.nn.Identity()
        backbone.eval().to(device)
        self._model = backbone
        self._device = device
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch

        self._ensure_model()
        rgb = self._cv2.cvtColor(image_bgr, self._cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(tensor).flatten().cpu().numpy().astype(np.float32)
        return _l2_normalize_vec(feat)


class EmbeddingDriftScorer:
    def __init__(
        self,
        reference_path: str,
        device: str = "cpu",
        knn_sample_size: int = 2048,
        encoder: str = "yolox",
    ):
        self.reference_path = reference_path
        self._device = device
        self._knn_sample_size = knn_sample_size
        self.encoder = (encoder or "yolox").lower()
        self.ready = False
        self._store: Optional[ReferenceEmbeddingStore] = None
        self._encode_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._encoder_label = "not attached"
        self._last = {
            "drift_score": 0.0,
            "cosine_centroid": 1.0,
            "knn_mean_sim": 1.0,
            "ready": False,
            "encoder": self._encoder_label,
            "reference_path": reference_path,
            "reference_count": 0,
            "frame_index": 0,
        }

    def load(self) -> Tuple[bool, str]:
        """Load reference bank only; live encoder attached when detection model loads."""
        try:
            self._store = ReferenceEmbeddingStore.from_path(
                self.reference_path, knn_sample_size=self._knn_sample_size
            )
            self._last["reference_count"] = int(self._store.matrix.shape[0])
            if self.encoder in ("yolox", "yolox_standard", "standard", "neck_concat"):
                self.ready = bool(self._encode_fn is not None)
                self._last["ready"] = self.ready
                self._last["encoder"] = self._encoder_label
                return True, (
                    f"Reference: {self._store.matrix.shape[0]}×{self._store.dim} — "
                    "load sakku-gate.pth for YOLOX standard drift (neck concat @ 640)"
                )
            if self.encoder == "resnet":
                self._encode_fn = ResNetFrameEmbedder(device=self._device).encode_bgr
                self._encoder_label = "ResNet-18 (fallback)"
                self.ready = True
                self._last["ready"] = True
                self._last["encoder"] = self._encoder_label
                return True, (
                    f"Reference: {self._store.matrix.shape[0]}×{self._store.dim} "
                    f"(encoder: {self._encoder_label})"
                )
            self.ready = bool(self._encode_fn is not None)
            self._last["ready"] = self.ready
            self._last["encoder"] = self._encoder_label
            return True, (
                f"Reference: {self._store.matrix.shape[0]}×{self._store.dim} — "
                "load sakku-gate.pth for YOLOX standard drift (neck concat @ 640)"
            )
        except Exception as e:
            self.ready = False
            return False, str(e)

    def attach_yolox_model(self, detection_model) -> Tuple[bool, str]:
        """Wire live encoder to DetectionModel.encode_frame_embedding() (.pth or ONNX)."""
        if detection_model is None:
            return False, "No detection model"
        if not getattr(detection_model, "can_encode_drift_embedding", lambda: False)():
            return False, (
                "Drift needs PyTorch .pth (YOLOX backbone+neck @ 640). "
                "ONNX detection alone cannot run standard drift."
            )
        self._encode_fn = detection_model.encode_frame_embedding
        label = getattr(detection_model, "drift_encoder_description", lambda: "YOLOX")()
        self._encoder_label = label
        self.ready = self._store is not None
        self._last["ready"] = self.ready
        self._last["encoder"] = self._encoder_label
        return self.ready, self._encoder_label

    def score_frame(self, image_bgr: np.ndarray, frame_index: int = 0) -> dict:
        if self._store is None or self._encode_fn is None:
            out = dict(self._last)
            out["frame_index"] = frame_index
            out["ready"] = False
            out["message"] = (
                "Waiting for .pth model (YOLOX standard drift @ 640)"
                if self.encoder in ("yolox", "yolox_standard", "standard")
                else "Encoder not ready"
            )
            return out

        emb = self._encode_fn(image_bgr)
        emb = _l2_normalize_vec(np.asarray(emb, dtype=np.float32))

        cos_centroid = float(np.dot(emb, self._store.centroid))
        sims = self._store.sample @ emb
        knn_mean_sim = float(np.mean(sims)) if sims.size else cos_centroid

        dist_centroid = max(0.0, 1.0 - cos_centroid)
        dist_knn = max(0.0, 1.0 - knn_mean_sim)
        drift_raw = 0.6 * dist_centroid + 0.4 * dist_knn
        drift_score = float(min(100.0, drift_raw * 100.0))

        # Reference bank built with a different checkpoint/preproc → cos ≈ 0, drift stuck at 100.
        bank_mismatch = cos_centroid < 0.2 and knn_mean_sim < 0.2

        self._last = {
            "drift_score": drift_score,
            "cosine_centroid": cos_centroid,
            "knn_mean_sim": knn_mean_sim,
            "ready": True,
            "bank_mismatch": bank_mismatch,
            "encoder": self._encoder_label,
            "reference_path": self.reference_path,
            "reference_count": int(self._store.matrix.shape[0]),
            "frame_index": frame_index,
        }
        return dict(self._last)

    def get_last(self) -> dict:
        return dict(self._last)
