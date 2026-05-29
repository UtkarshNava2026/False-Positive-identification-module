# False Positive Identification Agent

PyQt5 desktop app for YOLOX detection (**PyTorch `.pth`** or **ONNX `.onnx`**), live **embedding-based data drift** scoring, false-positive flagging, and annotation export. Supports video files, images, and RTSP streams.

## Install

```bash
pip install -r requirements.txt
# GPU ONNX (optional):
pip install -r requirements-gpu.txt
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
| **`model.path`** | **Both** | **Main model file** — use a `.onnx` or `.pth` path here. This is the only field you must set. |
| `model.pth_path` | Legacy | Same as `model.path`; kept for older configs. Prefer `model.path`. |
| `model.exp_path` | **`.pth` only** | YOLOX experiment Python file (defines `Exp` class). Not used for pure ONNX. |
| `model.classes_path` | Recommended | Text file, one class name per line. |
| `model.device` | Both | `"cpu"` or `"cuda"` — PyTorch device and ONNX Runtime providers. |

### Example: ONNX

```json
{
  "model": {
    "path": "/path/to/model.onnx",
    "exp_path": "/path/to/yolox_exp.py",
    "classes_path": "/path/to/class.txt",
    "device": "cpu"
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

## Data drift

Embedding-based drift scoring compares each frame to a local reference bank (`embeddings.npy`) using the **YOLOX Standard** pipeline and the same checkpoint as detection (`sakku-gate.pth`).

**Full documentation:** [`README_DRIFT.md`](README_DRIFT.md) — reference bank, live embedding steps, drift formula, score interpretation, rebuild, troubleshooting.

Quick config:

```json
"drift": {
  "reference_path": "embeddings.npy",
  "encoder": "yolox_standard",
  "input_size": [640, 640],
  "knn_sample_size": 2048
}
```

Place `embeddings.npy` and `sakku-gate.pth` locally (not tracked in Git).

## RTSP passwords

URLs with `#` in the password (e.g. `Nava#321`) are encoded automatically. Paste the URL as-is.

## Project layout

```
config.json          # model paths, device, drift reference, video FPS
detection.py         # entry point
embeddings.npy       # reference embeddings (or .pkl)
fpa_agent/
  main_window.py     # UI
  threads.py         # video / RTSP / model workers
  drift_score.py     # embedding drift module
  detection_model.py # YOLOX .pth / .onnx inference
```

See `README_CONFIG.md` for the full config reference.  
See `README_DRIFT.md` for the complete data drift module guide.
