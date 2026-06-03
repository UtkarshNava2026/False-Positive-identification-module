# False Positive Identification Agent

PyQt5 desktop app for YOLOX detection (**PyTorch `.pth`**, **ONNX `.onnx`**, or **OpenVINO IR `.xml`**), live **embedding-based data drift** scoring, false-positive flagging, and annotation export. Supports video files, images, and RTSP streams.

## Install

```bash
pip install -r requirements.txt
# GPU ONNX (optional):
pip install -r requirements-gpu.txt
# OpenVINO quantization (optional):
pip install -r requirements-openvino.txt
```

If startup fails with **Qt platform plugin "xcb"** and `cv2/qt/plugins`, switch to headless OpenCV:

```bash
pip uninstall opencv-python -y && pip install opencv-python-headless
```

`detection.py` also forces PyQt5 plugin paths before OpenCV loads.

## Run

```bash
python detection.py
```

## Where to set model paths (`config.json`)

All paths are **absolute** or relative to the project folder (where `config.json` lives).

| Setting | Required for | Description |
|--------|----------------|-------------|
| **`model.path`** | **All** | **Main model file** — use a `.onnx`, `.pth`, or `.xml` path here. This is the only field you must set. |
| `model.pth_path` | Legacy | Same as `model.path`; kept for older configs. Prefer `model.path`. |
| `model.exp_path` | **`.pth` only** | YOLOX experiment Python file (defines `Exp` class). Not used for pure ONNX/OpenVINO. |
| `model.classes_path` | Recommended | Text file, one class name per line. |
| `model.device` | Both | `"cpu"` or `"cuda"` — PyTorch device and ONNX Runtime providers. |
| **`model.backend`** | Recommended | `"onnxruntime"` or `"openvino"` — inference backend. Auto-detected from file extension if not set. |
| **`model.openvino_device`** | OpenVINO | `"CPU"`, `"GPU"`, or `"AUTO"` — OpenVINO target device. |

### Example: ONNX (default)

```json
{
  "model": {
    "path": "weights/backups/sakku_fp32.onnx",
    "exp_path": "yolox_voc_s 3.py",
    "classes_path": "class.txt",
    "device": "cpu",
    "backend": "onnxruntime"
  }
}
```

### Example: OpenVINO INT8 (quantized)

```json
{
  "model": {
    "path": "weights/sakku_int8.xml",
    "exp_path": "yolox_voc_s 3.py",
    "classes_path": "class.txt",
    "device": "cpu",
    "backend": "openvino",
    "openvino_device": "CPU"
  }
}
```

### Example: PyTorch checkpoint

```json
{
  "model": {
    "path": "/path/to/weights.pth",
    "exp_path": "/path/to/yolox_exp.py",
    "classes_path": "/path/to/class.txt",
    "device": "cuda"
  }
}
```

You can also use **Load Model** in the UI; paths are saved back to `config.json`.

---

## OpenVINO INT8 Quantization

Quantize your FP32 ONNX model to INT8 OpenVINO IR for faster CPU inference with minimal accuracy loss.

### Prerequisites

```bash
pip install -r requirements-openvino.txt
```

### Check available OpenVINO devices

```bash
# Quick device check (Python one-liner):
python -c "import openvino as ov; core = ov.Core(); [print(f'  {d}: {core.get_property(d, \"FULL_DEVICE_NAME\")}') for d in core.available_devices]"

# Or use the built-in script:
python development_tools/quantize/infer_openvino.py --list-devices

# Or from the quantize_and_validate script:
python development_tools/quantize_and_validate.py --list-devices
```

**Typical output:**
```
  CPU: Intel(R) Core(TM) i7-12700H
  GPU: Intel(R) UHD Graphics 770
```

Use the device name (e.g. `CPU`, `GPU`, `AUTO`) in `config.json` → `model.openvino_device`.

### Step 1: Prepare calibration images

Place **100–300 representative images** from your deployment cameras in the `calib data/` folder:

```
calib data/
  ├── frame_001.jpg
  ├── frame_002.jpg
  ├── ...
  └── frame_200.jpg
```

These should be typical scenes from your deployment — the more representative, the better the INT8 calibration.

### Step 2: Quantize (one-time)

**Option A — One-click pipeline** (recommended):

```bash
# Full quantize + benchmark
python development_tools/quantize_and_validate.py

# Quantize + compare FP32 vs INT8 speed
python development_tools/quantize_and_validate.py --compare

# Quantize only, skip validation
python development_tools/quantize_and_validate.py --skip-validate

# Control calibration sample size
python development_tools/quantize_and_validate.py --num-samples 100
```

**Option B — Manual quantization:**

```bash
python development_tools/quantize/quantize_openvino_nncf.py \
  -m weights/backups/sakku_fp32.onnx \
  -i "development_tools/calib data" \
  -o weights/sakku_int8.xml \
  --num_samples 200
```

This produces `weights/sakku_int8.xml` + `weights/sakku_int8.bin`.

### Step 3: Use in the app

Update `config.json`:

```json
{
  "model": {
    "path": "weights/sakku_int8.xml",
    "backend": "openvino",
    "openvino_device": "CPU"
  }
}
```

Then run:

```bash
python detection.py
```

### Step 4: Validate / benchmark (optional)

```bash
# Benchmark INT8 model
python development_tools/quantize_and_validate.py --validate-only

# Compare FP32 vs INT8 side-by-side
python development_tools/quantize_and_validate.py --validate-only --compare

# Run standalone inference with annotated output
python development_tools/quantize/infer_openvino.py \
  -m weights/sakku_int8.xml \
  -i "development_tools/calib data" \
  -o openvino_results \
  --class-file class.txt \
  -s 0.25
```

The benchmark reports:
- Model load + compile time
- Avg / P50 / P95 / P99 latency (ms)
- Estimated FPS
- Size reduction (FP32 → INT8)

---

## Data drift (embeddings)

Uses team **YOLOX Standard** pipeline (same as `embeddings.npy`):

**letterbox 640×640 → backbone → neck (PAFPN) → GAP each scale → concat → L2 normalize**

| Setting | Description |
|--------|-------------|
| `drift.reference_path` | `embeddings.npy` — shape `(N, 512)` |
| `drift.encoder` | `yolox_standard` (default) |
| `drift.input_size` | `[640, 640]` — must match bank build |
| `model.path` | **`sakku-gate.pth`** required for drift (same checkpoint as bank) |

```json
"drift": {
  "reference_path": "embeddings.npy",
  "encoder": "yolox_standard",
  "input_size": [640, 640],
  "knn_sample_size": 2048
}
```

Rebuild bank (optional):

```bash
python development_tools/build_reference_embeddings.py \
  --pth sakku-gate.pth \
  --exp "development_tools/yolox_voc_s 3.py" \
  --images /path/to/images \
  --output embeddings.npy
```

## RTSP passwords

URLs with `#` in the password (e.g. `Nava#321`) are encoded automatically. Paste the URL as-is.

## Project layout

```
config.json                        # model paths, device, backend, quantize config
detection.py                       # entry point (PyQt5 app)
class.txt                          # class names (one per line)
embeddings.npy                     # reference embeddings (or .pkl)
requirements.txt                   # base dependencies
requirements-gpu.txt               # GPU ONNX Runtime
requirements-openvino.txt          # OpenVINO + NNCF
weights/
  sakku_fp32.onnx                  # FP32 ONNX model
  sakku_int8.xml + .bin            # INT8 OpenVINO IR (after quantization)
fpa_agent/
  main_window.py                   # UI
  threads.py                       # video / RTSP / model workers
  drift_score.py                   # embedding drift module
  detection_model.py               # YOLOX .pth / .onnx / .xml inference
development_tools/                 # Quantization, calibration, exporting, and embedding builders
  quantize_and_validate.py         # one-click quantize + benchmark
  build_reference_embeddings.py    # build reference embeddings
  export_to_onnx.py                # PyTorch to ONNX export script
  export_to_openvino.py            # ONNX to OpenVINO conversion script
  export_embedding_onnx.py         # export embedding-only ONNX
  yolox_voc_s 3.py                 # YOLOX VOC config file
  calib data/                      # calibration images for quantization
  quantize/                        # low-level quantization/inference scripts
  onnx_workflow/                   # onnx helpers
  Training-Dataset/                # training references
```

See `README_CONFIG.md` for the full config reference.
