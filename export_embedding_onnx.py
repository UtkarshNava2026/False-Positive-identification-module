#!/usr/bin/env python3
"""
Export team-style drift embedding ONNX (hook + GAP + projection + L2).

Requires drift_projection.pth from build_reference_embeddings.py
(or your team's saved projection weights).

Example:
  python export_embedding_onnx.py \\
    --pth sakku-gate.pth \\
    --exp "yolox_voc_s 3.py" \\
    --projection-weights drift_projection.pth \\
    --input-size 416 416 \\
    --output gate_embed.onnx
"""

import argparse
import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftExportWrapper(nn.Module):
    """Exportable graph matching YOLOXDriftEmbedder (linear_relu or mlp)."""

    def __init__(self, yolox_model, proj: nn.Module):
        super().__init__()
        self.backbone = yolox_model.backbone
        self.proj = proj

    def forward(self, x):
        fpn = self.backbone(x)
        feat = fpn[-1] if isinstance(fpn, (list, tuple)) else fpn
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        emb = self.proj(feat)
        return F.normalize(emb, p=2, dim=1)


def load_exp(exp_path):
    spec = importlib.util.spec_from_file_location("custom_exp", exp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--projection-weights", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-size", nargs=2, type=int, default=[416, 416])
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fpa_agent.embedding_extractor import YOLOXDriftEmbedder

    exp = load_exp(args.exp)
    model = exp.get_model()
    ckpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    isize = (args.input_size[0], args.input_size[1])
    embedder = YOLOXDriftEmbedder(
        model,
        "cpu",
        input_size=isize,
        projection_weights_path=args.projection_weights,
    )
    if embedder.proj is None:
        raise RuntimeError(
            "Could not load projection from --projection-weights. "
            "Run build_reference_embeddings.py --save-projection drift_projection.pth first."
        )

    wrapper = DriftExportWrapper(model, embedder.proj)
    wrapper.eval()
    dummy = torch.randn(1, 3, isize[0], isize[1])
    out_path = args.output
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=12,
    )
    print(f"Wrote {out_path}")
    print("config.json → drift.onnx_embedding_path (use .onnx not .onnx.data)")


if __name__ == "__main__":
    main()
