#!/usr/bin/env python3
"""
Direct model conversion from PyTorch .pth to OpenVINO IR (.xml/.bin).

Supports converting:
1. The full YOLOX detection model
2. The standard drift embedding extraction model (backbone + neck + GAP + L2 normalize)
"""

import argparse
import importlib.util
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class YOLOXStandardEmbeddingWrapper(nn.Module):
    """
    Wrap YOLOX backbone + neck + multi-scale global average pooling + L2 normalization
    to extract standard 512-D embeddings.
    """
    def __init__(self, yolox_model, pool_mode="last_scale", expected_dim=512):
        super().__init__()
        self.backbone = yolox_model.backbone
        self.neck = getattr(yolox_model, "neck", None)
        self.pool_mode = pool_mode
        self.expected_dim = expected_dim

    def forward(self, x):
        backbone_feats = self.backbone(x)
        if self.neck is not None:
            neck_feats = self.neck(backbone_feats)
        else:
            neck_feats = backbone_feats

        if not isinstance(neck_feats, (list, tuple)):
            neck_feats = (neck_feats,)

        # Multi-scale global average pooling
        parts = []
        for feat in neck_feats:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            parts.append(pooled)

        # Concatenate or take last scale
        concat_dim = sum(p.shape[1] for p in parts)
        use_concat = self.pool_mode == "concat_all" or (
            self.pool_mode == "auto" and concat_dim == self.expected_dim
        )
        
        if use_concat:
            emb = torch.cat(parts, dim=1)
        else:
            emb = parts[-1]

        # L2 normalization along the feature dimension
        emb = F.normalize(emb, p=2.0, dim=1)
        return emb


def load_yolox_model(exp_path, pth_path):
    spec = importlib.util.spec_from_file_location("custom_exp", exp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = mod.Exp()
    
    model = exp.get_model()
    print(f"Loading weights from {pth_path}...")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, exp


def main():
    ap = argparse.ArgumentParser(description="Export YOLOX models directly from .pth to OpenVINO IR (.xml)")
    ap.add_argument("--pth", default="weights/best_ckpt.pth", help="Path to YOLOX PyTorch .pth checkpoint")
    ap.add_argument("--exp", default="yolox_voc_s 3.py", help="Path to YOLOX experiment .py file")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="Model input resolution (height width)")
    ap.add_argument("--output-dir", default="weights", help="Directory to save exported OpenVINO models")
    args = ap.parse_args()

    # Create output dir if needed
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import openvino as ov
    except ImportError:
        print("ERROR: openvino package not installed.")
        print("Please run: pip install -r requirements-openvino.txt")
        sys.exit(1)

    print(f"OpenVINO version: {ov.__version__}")
    
    # 1. Load PyTorch model
    if not os.path.exists(args.pth):
        print(f"ERROR: Checkpoint file not found: {args.pth}")
        sys.exit(1)
    if not os.path.exists(args.exp):
        print(f"ERROR: Experiment file not found: {args.exp}")
        sys.exit(1)

    model, exp = load_yolox_model(args.exp, args.pth)
    
    h, w = args.input_size
    dummy_input = torch.randn(1, 3, h, w)
    
    # --- Part A: Export Detection Model ---
    detection_xml_path = os.path.join(args.output_dir, "sakku_best.xml")
    print(f"\nConverting YOLOX Detection model to OpenVINO...")
    try:
        ov_det_model = ov.convert_model(model, example_input=dummy_input)
        # Set tensor names if possible
        try:
            ov_det_model.inputs[0].get_tensor().set_names({"images"})
            ov_det_model.outputs[0].get_tensor().set_names({"output"})
        except Exception as t_err:
            print(f"Note (tensor naming): {t_err}")
            
        ov.save_model(ov_det_model, detection_xml_path)
        print(f"Successfully exported Detection model -> {detection_xml_path}")
    except Exception as e:
        print(f"ERROR converting Detection model: {e}")
        import traceback
        traceback.print_exc()

    # --- Part B: Export Embedding Model ---
    embedding_xml_path = os.path.join(args.output_dir, "sakku_embedding.xml")
    print(f"\nWrapping and converting YOLOX Embedding model to OpenVINO...")
    try:
        # Standard YOLOX-S features concat yields 128+256+512 = 896. If 512 is expected, pool_mode="last_scale"
        pool_mode = "last_scale"
        wrapper = YOLOXStandardEmbeddingWrapper(model, pool_mode=pool_mode)
        wrapper.eval()
        
        ov_emb_model = ov.convert_model(wrapper, example_input=dummy_input)
        try:
            ov_emb_model.inputs[0].get_tensor().set_names({"images"})
            ov_emb_model.outputs[0].get_tensor().set_names({"embedding"})
        except Exception as t_err:
            print(f"Note (tensor naming): {t_err}")
            
        ov.save_model(ov_emb_model, embedding_xml_path)
        print(f"Successfully exported Embedding model -> {embedding_xml_path}")
    except Exception as e:
        print(f"ERROR converting Embedding model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
