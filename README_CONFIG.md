# Configuration reference (`config.json`)

For the full data drift guide (embeddings, formula, interpretation), see **`README_DRIFT.md`**.

## Model (`model`)

| Key | `.pth` | `.onnx` | Notes |
|-----|--------|---------|--------|
| **`path`** | ✓ | ✓ | **Primary model file.** Set this to your `.pth` or `.onnx`. |
| `pth_path` | ✓ | ✓ | Alias for `path` (backward compatible). |
| `exp_path` | ✓ | optional | YOLOX `Exp` class file; required for PyTorch weights. |
| `classes_path` | ✓ | ✓ | Class names `.txt` (one per line). |
| `device` | ✓ | ✓ | `cpu` or `cuda`. |

**Rule of thumb:** put **any** model file in `model.path`. Use `.onnx` extension for ONNX, `.pth` for PyTorch.

## Drift (`drift`) — YOLOX Standard

See **`README_DRIFT.md`** for pipeline details, drift formula, score bands, and troubleshooting.

Matches team reference script: **letterbox 640 → backbone → neck → GAP per scale → concat/last_scale → L2**.

| Key | Default | Description |
|-----|---------|-------------|
| `reference_path` | `embeddings.npy` | Reference bank `(N, 512)` from team extraction pipeline. |
| `encoder` | `yolox_standard` | Use `yolox_standard` (or `yolox`). Requires `.pth` with `backbone` + `neck`. |
| `input_size` | `[640, 640]` | Letterbox size — **must match** how `embeddings.npy` was built. |
| `knn_sample_size` | `2048` | Random subsample for kNN speed. |
| `pool_mode` | `auto` (optional) | `auto`: concat scales if dim=512, else last PAFPN scale (128+256+512→512 for YOLOX-S bank). `concat_all` / `last_scale` to force. |

**No** `projection_weights` — no learned projection head in the team script.

**Note:** Stock YOLOX has `YOLOPAFPN` as `.backbone` (no separate `.neck`). Multi-scale concat is 896-D; your `embeddings.npy` is 512-D, so `auto` uses **last scale** (512 channels) to match the bank.

### Interpreting drift score (0–100)

The score is `100 × (0.6 × (1 − cos_to_centroid) + 0.4 × (1 − kNN_mean_similarity))` after L2-normalized embeddings.

| Range | Typical meaning |
|-------|-----------------|
| **0–15** | In-distribution (training / gate footage) |
| **15–35** | Unseen but same domain (new camera angle, lighting, site) — **~22 is normal** |
| **35–70** | Noticeable shift (different scene or heavy weather) |
| **70–100** | Strong OOD or **bank mismatch** (wrong checkpoint / encoder; check cos &lt; 0.2 warning) |

Use the gauge’s **cosine** and **kNN** lines: in-distribution frames usually show cos &gt; 0.7 and kNN &gt; 0.65.

**Detection vs drift:** detection uses YOLOX `preproc`; drift uses **letterbox @ 640** (by design).

### Rebuild reference bank

```bash
python build_reference_embeddings.py \
  --pth sakku-gate.pth \
  --exp "yolox_voc_s 3.py" \
  --images /path/to/images \
  --output embeddings.npy \
  --input-size 640 640
```

### ONNX detection

Drift with `yolox_standard` requires **PyTorch** `sakku-gate.pth`. ONNX-only detection does not run standard drift until a dedicated embed ONNX is exported.

## Video (`video`)

| Key | Default | Description |
|-----|---------|-------------|
| `fps` | `0` | Max processing FPS (`0` = unlimited). |
| `frame_step` | `1` | Process every Nth frame (`1` = all frames). |

## UI (`ui`)

| Key | Default |
|-----|---------|
| `window_width` | `1280` |
| `window_height` | `800` |

## Export (`export`)

| Key | Default |
|-----|---------|
| `default_format` | `YOLO` |
