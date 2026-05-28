## False Positive Identification Module (GUI)

PyQt5 desktop app for running YOLOX detection (PyTorch `.pth` or ONNX `.onnx`), reviewing frames, and exporting annotations. Supports **video files**, **images**, and **RTSP streams**.

## Install

Create/activate a venv, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU ONNX Runtime (CUDA), install:

```bash
pip install -r requirements-gpu.txt
```

## Run

```bash
python detection.py
```

The app will auto-load the model if `config.json` is filled.

## Configuration (`config.json`)

The key setting for CPU/GPU is:

- **`model.device`**: `"cpu"` or `"cuda"`
  - **PyTorch**: uses `torch.device(model.device)`
  - **ONNX**:
    - `"cpu"` → `CPUExecutionProvider`
    - `"cuda"` → `CUDAExecutionProvider` with CPU fallback

Example:

```json
{
  "model": {
    "pth_path": "/abs/path/to/model.onnx",
    "exp_path": "/abs/path/to/yolox_exp.py",
    "classes_path": "/abs/path/to/classes.txt",
    "device": "cuda"
  }
}
```

More details are in `README_CONFIG.md`.
