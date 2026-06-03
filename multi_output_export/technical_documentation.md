# Technical Documentation: YOLOX Multi-Output Pipeline & Data Drift Scoring

This document describes the architectural design, implementation details, mathematical formulation, and benchmark results of the optimized single-model **Multi-Output INT8 OpenVINO pipeline** implemented for the False Positive Identification Agent.

---

## 1. How the Multi-Output Model Was Made

Historically, the application executed two separate inference passes per frame:
1. **Detection Pass:** Input BGR Image $\rightarrow$ YOLOX-S Backbone (CSPDarknet + PAFPN Neck) $\rightarrow$ YOLOXHead $\rightarrow$ Box Detections.
2. **Embedding Pass:** Input BGR Image $\rightarrow$ YOLOX-S Backbone (CSPDarknet + PAFPN Neck) $\rightarrow$ Global Average Pooling (GAP) $\rightarrow$ L2 Normalization $\rightarrow$ Embedding Vector ($512$-D).

Since the backbone represents **>90%** of the computational complexity (FLOPs) of the network, running it twice was highly redundant.

### The Wrapper Approach
We created a custom PyTorch model wrapper `YOLOXMultiOutputWrapper` (located in [export_multi_output.py](file:///c:/NAVA/False-Positive-identification-module/multi_output_export/export_multi_output.py)) that exposes the backbone features to both heads in a single execution graph:

```
                  [ Input Frame (640x640) ]
                              │
                    [ YOLOX-S Backbone ] (CSPDarknet)
                              │
                    [ YOLOX-S Neck FPN ] (PAFPN) -> Outputs multiscale features (dark3, dark4, dark5)
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
       [ YOLOX Head ]                 [ GAP & L2 Normalization ]
    (Detections Output)                  (Embedding Output)
     Shape: [1, 8400, 18]                  Shape: [1, 512]
```

1. **Feature Extraction:** The backbone and neck are run exactly once, producing multiscale feature maps (`dark3`, `dark4`, `dark5`).
2. **Detection Output:** The feature maps are passed to `model.head` (with `decode_in_inference = False` to export raw boxes matching standard optimized ONNX/OpenVINO shapes).
3. **Embedding Output:** The last scale feature map (`dark5`) is passed through Global Average Pooling (GAP) to collapse spatial dimensions $(1 \times 512 \times 20 \times 20 \rightarrow 1 \times 512 \times 1 \times 1)$ and flattened to $512$ channels, then L2-normalized along the channel axis.
4. **Result:** The model returns both output tensors (`output`, `embedding`) simultaneously in a single forward pass.

### Export and Quantization
* **FP32 Model:** The wrapped PyTorch graph was compiled directly to OpenVINO IR using `openvino.convert_model` to create `weights/sakku_multi_output.xml`.
* **INT8 Model:** We ran OpenVINO NNCF static post-training quantization (`quantize_multi_output.py`) on $150$ representative calibration images. This quantized all convolution weights and activations to INT8, creating `weights/sakku_multi_output_int8.xml` and `.bin`.

---

## 2. How the Data Drift Score is Calculated

The data drift score indicates how far the visual environment in the live feed has drifted compared to the baseline distribution stored in [embeddings.npy](file:///c:/NAVA/False-Positive-identification-module/embeddings.npy).

```
[ Live Frame ] -> [ Embedder ] -> Live Embedding (1x512)
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                      [ Centroid Sim ]          [ K-NN Sim ]
                        Dot product with       Average similarity
                        normalized mean        against all bank
                        reference vector       elements
                               │                     │
                               ▼                     ▼
                      Centroid Distance         K-NN Distance
                         (1.0 - Sim)             (1.0 - Sim)
                               │                     │
                               └──────────┬──────────┘
                                          ▼
                                   Weighted Average:
                              (60% Centroid + 40% K-NN)
                                          │
                                          ▼
                                   [ Drift Score ] (0% to 100%)
```

### Mathematical Formulation
1. **Live Frame Normalization:** The live frame's $512$-D embedding $\mathbf{e}_{\text{live}}$ is L2-normalized to sit on a unit hypersphere (length $= 1.0$):
   $$\mathbf{e}_{\text{live}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$
2. **Reference Centroid:** The average vector of all reference bank embeddings in `embeddings.npy` is computed and L2-normalized:
   $$\mathbf{c}_{\text{ref}} = \text{Normalize}\left( \frac{1}{N} \sum_{i=1}^N \mathbf{e}_{\text{ref}, i} \right)$$
3. **Centroid Similarity ($S_{\text{centroid}}$):** Cosine similarity between the live embedding and reference centroid:
   $$S_{\text{centroid}} = \mathbf{e}_{\text{live}} \cdot \mathbf{c}_{\text{ref}}$$
4. **K-NN Mean Similarity ($S_{\text{knn}}$):** Mean cosine similarity between the live embedding and $K$ random samples from the reference database (default $K = 2048$):
   $$S_{\text{knn}} = \frac{1}{K} \sum_{i=1}^K (\mathbf{e}_{\text{live}} \cdot \mathbf{e}_{\text{ref}, i})$$
5. **Cosine Distances:**
   $$D_{\text{centroid}} = \max(0.0, 1.0 - S_{\text{centroid}})$$
   $$D_{\text{knn}} = \max(0.0, 1.0 - S_{\text{knn}})$$
6. **Final Drift Score:** Weighted average of the two distances:
   $$Drift_{\text{raw}} = 0.6 \times D_{\text{centroid}} + 0.4 \times D_{\text{knn}}$$
   $$Drift_{\text{score}} = \min(100.0, Drift_{\text{raw}} \times 100.0)$$

---

## 3. Performance & FPS Comparisons

Inference latency and throughput benchmarks were performed directly on your local hardware (13th Gen Intel Core i7-1355U CPU).

### 1500-Frame Dataset Benchmark Results

| Model Configuration | Avg Latency / Frame | Throughput (FPS) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **1. OpenVINO Dual-Model (INT8 + FP32)** *(Original)* | **174.11 ms** | **5.74 FPS** | 1.00x (Baseline) |
| **2. OpenVINO Multi-Output (FP32)** | **107.68 ms** | **9.30 FPS** | 1.62x |
| **3. OpenVINO Multi-Output (INT8)** *(Active)* | **70.82 ms** | **14.12 FPS** | **2.46x FASTER** |

* **Time Saved:** Running the multi-output INT8 model saved **154.94 seconds** of CPU processing time over 1500 images.
* **Why it's faster:**
  - Shared backbone eliminates redundant convolutions (saving ~45% of total FLOPs).
  - Single-model inference halves OpenVINO compilation and graph execution scheduling overhead.
  - INT8 quantization leverages hardware-optimized low-precision convolutions on the Intel CPU.

---

## 4. Folder & Config Layout

### Weights Organization
Legacy sequential models have been isolated to clean up the workspace, leaving only active multi-output formats in the root:

```
weights/
├── sakku_multi_output_int8.xml   <-- Active quantized multi-output detection model
├── sakku_multi_output_int8.bin   <-- Active quantized weights
├── sakku_multi_output.onnx       <-- FP32 Multi-Output ONNX representation
├── sakku_multi_output.xml        <-- FP32 Multi-Output OpenVINO graph
├── sakku_multi_output.bin        <-- FP32 Multi-Output OpenVINO weights
├── backups/                      <-- Original checkpoint backups
└── legacy_dual_model/            <-- Isolated old dual-model files (sakku_int8 & sakku_embedding)
```

### Config.json Setting
The application **[config.json](file:///c:/NAVA/False-Positive-identification-module/config.json)** is configured to run the multi-output OpenVINO INT8 model automatically:

```json
{
  "model": {
    "path": "C:/NAVA/False-Positive-identification-module/weights/sakku_multi_output_int8.xml",
    "pth_path": "C:/NAVA/False-Positive-identification-module/weights/sakku_multi_output_int8.xml",
    "backend": "openvino",
    "openvino_device": "CPU"
  },
  "drift": {
    "reference_path": "embeddings.npy",
    "encoder": "yolox_standard",
    "openvino_embedding_path": ""
  }
}
```
*Note: Setting `"openvino_embedding_path": ""` is required to tell the drift scorer to fetch the cached embedding from output index 1 of the multi-output model, bypassing separate embedding passes.*
