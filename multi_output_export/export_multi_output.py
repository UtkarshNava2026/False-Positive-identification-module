#!/usr/bin/env python3
"""
YOLOX Multi-Output Export Script.

Wraps the YOLOX model to yield both detections and frame embeddings in a single
forward pass, then exports to ONNX and OpenVINO IR.
"""

import argparse
import os
import sys
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root and development_tools to sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "development_tools") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "development_tools"))


class YOLOXMultiOutputWrapper(nn.Module):
    """
    Wraps YOLOX to return both:
      1. Bounding box detections (from YOLOXHead)
      2. Frame-level embeddings (from PAFPN Neck features + GAP + L2 Norm)
    """
    def __init__(self, yolox_model, pool_mode="last_scale", expected_dim=512):
        super().__init__()
        self.yolox_model = yolox_model
        # Disable decode_in_inference to export raw outputs
        self.yolox_model.head.decode_in_inference = False
        self.pool_mode = pool_mode
        self.expected_dim = expected_dim

    def forward(self, x):
        # 1. Run backbone + neck (PAFPN) -> returns tuple of features (e.g., [dark3, dark4, dark5])
        fpn_outs = self.yolox_model.backbone(x)
        
        # 2. Run head to get detection outputs (shape: [batch, 8400, 85] or similar)
        det_out = self.yolox_model.head(fpn_outs)
        
        # 3. Apply Multi-scale GAP to PAFPN outputs for embeddings
        if not isinstance(fpn_outs, (list, tuple)):
            neck_feats = (fpn_outs,)
        else:
            neck_feats = fpn_outs

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

        # L2 normalize
        emb = F.normalize(emb, p=2.0, dim=1)
        
        return det_out, emb


def load_yolox_model(exp_path, pth_path):
    print(f"Loading experiment from: {exp_path}")
    spec = importlib.util.spec_from_file_location("custom_exp", exp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = mod.Exp()
    
    model = exp.get_model()
    print(f"Loading weights from: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, exp


def main():
    ap = argparse.ArgumentParser(description="Export YOLOX with multiple outputs (detections + embeddings)")
    ap.add_argument("--pth", default=os.path.join(_PROJECT_ROOT, "weights", "backups", "best_ckpt.pth"), help="Path to YOLOX checkpoint")
    ap.add_argument("--exp", default=os.path.join(_PROJECT_ROOT, "development_tools", "yolox_voc_s 3.py"), help="Path to YOLOX experiment .py")
    ap.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="Model input size (H W)")
    ap.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "weights"), help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    if not os.path.exists(args.pth):
        print(f"ERROR: Checkpoint file not found: {args.pth}")
        sys.exit(1)
    if not os.path.exists(args.exp):
        print(f"ERROR: Experiment file not found: {args.exp}")
        sys.exit(1)

    model, exp = load_yolox_model(args.exp, args.pth)
    
    # 2. Wrap model
    wrapper = YOLOXMultiOutputWrapper(model, pool_mode="last_scale")
    wrapper.eval()
    
    h, w = args.input_size
    dummy_input = torch.randn(1, 3, h, w)
    
    # 3. Export to ONNX
    onnx_path = os.path.join(args.output_dir, "sakku_multi_output.onnx")
    print(f"\nExporting wrapped model to ONNX: {onnx_path}")
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            input_names=["images"],
            output_names=["output", "embedding"],
            dynamic_axes={
                "images": {0: "batch"},
                "output": {0: "batch"},
                "embedding": {0: "batch"}
            },
            opset_version=11,
        )
        print(f"SUCCESS: Exported multi-output ONNX model -> {onnx_path}")
    except Exception as e:
        print(f"ERROR during ONNX export: {e}")
        import traceback
        traceback.print_exc()

    # 4. Export to OpenVINO
    xml_path = os.path.join(args.output_dir, "sakku_multi_output.xml")
    print(f"\nConverting to OpenVINO IR: {xml_path}")
    try:
        import openvino as ov
        print(f"OpenVINO version: {ov.__version__}")
        
        ov_model = ov.convert_model(wrapper, example_input=dummy_input)
        try:
            ov_model.inputs[0].get_tensor().set_names({"images"})
            ov_model.outputs[0].get_tensor().set_names({"output"})
            ov_model.outputs[1].get_tensor().set_names({"embedding"})
        except Exception as t_err:
            print(f"Note (tensor naming): {t_err}")
            
        ov.save_model(ov_model, xml_path)
        print(f"SUCCESS: Exported multi-output OpenVINO model -> {xml_path}")
    except ImportError:
        print("\nWARNING: openvino is not installed in the current environment.")
        print("To convert the exported ONNX model to OpenVINO IR, run:")
        print("  pip install openvino nncf")
        print("  python -c \"import openvino as ov; core = ov.Core(); m = core.read_model('" + onnx_path.replace("\\", "/") + "'); ov.save_model(m, '" + xml_path.replace("\\", "/") + "')\"")
    except Exception as e:
        print(f"ERROR during OpenVINO conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
