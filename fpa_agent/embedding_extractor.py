"""
Drift embeddings — YOLOX Standard pipeline (matches team reference script).

letterbox 640×640 → backbone → neck (PAFPN) → GAP per scale → concat → L2 normalize.
No learned projection head. Reference bank: embeddings.npy (N × 512).
"""

from __future__ import annotations

from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def letterbox_preprocess_bgr(
    image_bgr: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
) -> torch.Tensor:
    """
    YOLOX-style letterbox: preserve aspect, pad 114, float32 CHW, no /255.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    th, tw = int(input_size[0]), int(input_size[1])

    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    pt = (th - nh) // 2
    pl = (tw - nw) // 2
    padded[pt : pt + nh, pl : pl + nw] = resized

    tensor = padded.astype(np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, 0)
    return torch.from_numpy(tensor)


def _forward_backbone_and_neck(yolox_model: nn.Module, x: torch.Tensor):
    """
    Image → backbone → neck (PAFPN), matching the reference extraction script.

    - Custom loaders: model.backbone (CSPDarknet) then model.neck (PAFPN).
    - Stock YOLOX: model.backbone is YOLOPAFPN (CSPDarknet + PAFPN inside one module).
    """
    neck = getattr(yolox_model, "neck", None)
    backbone = getattr(yolox_model, "backbone", None)
    if backbone is None:
        raise RuntimeError("YOLOX model has no backbone")

    if neck is not None:
        backbone_feats = backbone(x)
        return neck(backbone_feats)

    # Stock YOLOX: YOLOPAFPN.forward = CSPDarknet then PAFPN neck
    return backbone(x)


class YOLOXStandardEmbedder:
    """
    Reference pipeline (embeddings.npy):
      Letterbox 640 → Backbone → Neck (PAFPN) → AdaptiveAvgPool2d → Flatten → L2.
    """

    def __init__(
        self,
        yolox_model: nn.Module,
        device: Union[str, torch.device],
        input_size: Tuple[int, int] = (640, 640),
        expected_dim: int = 512,
        pool_mode: str = "last_scale",
    ):
        self.model = yolox_model
        self.device = torch.device(device if device != "gpu" else "cuda")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.expected_dim = int(expected_dim)
        self.pool_mode = (pool_mode or "auto").lower()

        if not hasattr(self.model, "backbone"):
            raise RuntimeError("YOLOX model has no backbone")

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        # 1) Letterbox 640×640 (BGR→RGB, pad 114, float32 CHW, no /255)
        x = letterbox_preprocess_bgr(image_bgr, self.input_size).to(self.device)

        # 2) Backbone → 3) Neck (PAFPN)
        neck_feats = _forward_backbone_and_neck(self.model, x)
        if not isinstance(neck_feats, (list, tuple)):
            neck_feats = (neck_feats,)

        # 4) AdaptiveAvgPool2d(·, 1) → 5) Flatten (per scale)
        parts = [
            F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in neck_feats
        ]
        concat_dim = sum(p.shape[1] for p in parts)

        use_concat = self.pool_mode == "concat_all"
        if self.pool_mode == "auto":
            use_concat = concat_dim == self.expected_dim
        if use_concat:
            # All PAFPN scales: 128+256+512 = 896 for YOLOX-S width 0.5
            embedding = torch.cat(parts, dim=1).squeeze(0)
            mode_label = "concat_all"
        else:
            # 512-D bank (e.g. ~69k training images): finest PAFPN scale (512 ch)
            embedding = parts[-1].squeeze(0)
            mode_label = "last_scale"

        # 6) L2 normalize
        embedding = F.normalize(embedding, dim=0)
        out = embedding.cpu().numpy().astype(np.float32)

        if self.expected_dim and out.shape[0] != self.expected_dim:
            print(
                f"Warning: embedding dim {out.shape[0]} != expected {self.expected_dim} ({mode_label})"
            )
        self._last_pool_mode = mode_label
        return out

    def description(self) -> str:
        mode = getattr(self, "_last_pool_mode", self.pool_mode)
        return (
            f"YOLOX backbone+neck GAP {mode} @ "
            f"{self.input_size[0]}x{self.input_size[1]}"
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Legacy pipelines (hook + projection) — only if drift.encoder is set explicitly
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, emb_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class YOLOXLegacyDriftEmbedder:
    """Deprecated: hook + learned projection. Use YOLOXStandardEmbedder instead."""

    def __init__(self, yolox_model, device, input_size=(416, 416), projection_type="linear_relu", projection_weights_path=None):
        from yolox.data.data_augment import preproc
        import os

        self.model = yolox_model
        self.device = torch.device(device if device != "gpu" else "cuda")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.projection_type = projection_type
        self._preproc = preproc
        self.proj = None
        self._feat = None
        self._hook_handle = None

        self.model.to(self.device)
        self.model.eval()
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise RuntimeError("No backbone")
        self._hook_handle = backbone.register_forward_hook(self._hook_fn)
        if projection_weights_path and os.path.isfile(projection_weights_path):
            self._load_weights(projection_weights_path)

    def _hook_fn(self, module, inp, out):
        self._feat = out[-1] if isinstance(out, (list, tuple)) else out

    def _load_weights(self, path):
        import os
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt.get("projection", ckpt) if isinstance(ckpt, dict) else ckpt
        # minimal load — legacy only
        pass

    @torch.inference_mode()
    def extract_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        img, _ = self._preproc(image_bgr, self.input_size)
        t = torch.from_numpy(img).float()
        if t.ndim == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1)
        x = t.unsqueeze(0).to(self.device)
        self._feat = None
        _ = self.model(x)
        feat = F.adaptive_avg_pool2d(self._feat, 1).flatten(1)
        if self.proj is None:
            in_dim = int(feat.shape[1])
            if self.projection_type in ("mlp", "mlp_1024_512"):
                self.proj = ProjectionHead(in_dim, 512).to(self.device)
            else:
                self.proj = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU()).to(self.device)
        emb = F.normalize(self.proj(feat), p=2, dim=1)
        return emb.cpu().numpy()[0].astype(np.float32)

    def description(self) -> str:
        return f"YOLOX legacy projection @{self.input_size[0]}x{self.input_size[1]}"

    def close(self):
        if self._hook_handle:
            self._hook_handle.remove()


def create_drift_embedder(
    yolox_model: nn.Module,
    device: Union[str, torch.device],
    encoder: str = "yolox_standard",
    input_size: Tuple[int, int] = (640, 640),
    **legacy_kwargs,
):
    """Factory: default is YOLOXStandardEmbedder."""
    enc = (encoder or "yolox_standard").lower()
    if enc in ("yolox", "yolox_standard", "standard", "neck_concat"):
        pool_mode = legacy_kwargs.get("pool_mode", "last_scale")
        return YOLOXStandardEmbedder(
            yolox_model, device, input_size=input_size, pool_mode=pool_mode
        )
    return YOLOXLegacyDriftEmbedder(
        yolox_model,
        device,
        input_size=input_size,
        projection_type=legacy_kwargs.get("projection_type", "linear_relu"),
        projection_weights_path=legacy_kwargs.get("projection_weights_path"),
    )
