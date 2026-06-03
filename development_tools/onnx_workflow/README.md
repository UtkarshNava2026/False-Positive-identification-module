# Standalone YOLOX ONNX Workflow

This folder contains standalone scripts to convert, quantize, extract reference embeddings, and run inference using YOLOX models on the **ONNX** backend. Use this workflow on other devices that do not support OpenVINO.

---

## 📋 Prerequisites

Before running the scripts, install the necessary libraries on your device:

```bash
# If your target PC has an NVIDIA GPU:
pip install torch torchvision onnx onnxruntime-gpu onnxruntime-quantization opencv-python numpy

# OR if it is CPU only:
pip install torch torchvision onnx onnxruntime opencv-python numpy
```

---

## 🚀 Step-by-Step Workflow

### **Step 1: Export PyTorch `.pth` to FP32 ONNX**
Converts your trained PyTorch checkpoint (`best_ckpt.pth`) into standard FP32 ONNX models for detection and embedding extraction:
```bash
python export.py --pth ../weights/best_ckpt.pth --exp ../yolox_voc_s\ 3.py --output-dir ../weights
```
* **Outputs generated:**
  * `weights/sakku_best.onnx` (Detection model)
  * `weights/sakku_embedding.onnx` (Embedding extraction model)

---

### **Step 2: Quantize the Detection Model to INT8 ONNX**
Runs ONNX Runtime native post-training static quantization on the detection model using a folder containing representative calibration images (e.g., `calib data` with 100–300 images). 

By default, this script uses **QDQ format** which is highly optimized for GPU Tensor Cores (TensorRT/CUDA):
```bash
python quantize.py -m ../weights/sakku_best.onnx -i ../calib\ data -o ../weights/sakku_int8.onnx
```
* **Output generated:** 
  * `weights/sakku_int8.onnx` (INT8 Quantized QDQ Detection model)

---

### **Step 3: Extract Reference Embeddings (One-Time)**
Generates the baseline feature distribution array (`embeddings.npy`) from your training dataset using the FP32 ONNX embedding extractor:
```bash
python build_reference.py --model ../weights/sakku_embedding.onnx --images ../Training-Dataset --output ../embeddings.npy
```
* **Output generated:**
  * `embeddings.npy` (Baseline embedding bank)

---

### **Step 4: Standalone Inference & Drift Scoring**
Verify your models and measure data drift on a test image frame:
```bash
python infer.py --detection ../weights/sakku_int8.onnx --embedding ../weights/sakku_embedding.onnx --reference ../embeddings.npy --image test_frame.jpg
```
* **Output:** Prints latency times, detected objects, and calculates a drift score (0–100) comparing the test frame to the reference database. Saves an `annotated_frame.jpg` to visualize results.

---

## ⚙️ Configuration for the Main GUI App

To run these ONNX models in the main PyQt5 application, update your `config.json` in the project root:

```json
  "model": {
    "path": "weights/sakku_int8.onnx",
    "pth_path": "weights/sakku_int8.onnx",
    "exp_path": "yolox_voc_s 3.py",
    "classes_path": "class.txt",
    "device": "cpu",
    "backend": "onnxruntime"
  },
  "drift": {
    "reference_path": "embeddings.npy",
    "onnx_embedding_path": "weights/sakku_embedding.onnx",
    "input_size": [640, 640],
    "knn_sample_size": 2048,
    "auto_flag_threshold": 50.0
  }
```
Then, launch the application:
```bash
python detection.py
```
