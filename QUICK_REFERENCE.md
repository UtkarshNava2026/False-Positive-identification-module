# Quick Reference: False Positive Frame Export

## Feature At A Glance

Mark false positive frames while reviewing video and export them instantly for retraining your detection model.

---

## Quick Start (3 Steps)

### 1️⃣ Load & Watch
```
Open Video → Play → Watch for false positives
```

### 2️⃣ Mark Frames
```
When FP detected → Click "🚩 Flag Frame"
(Frame auto-captured with detections)
```

### 3️⃣ Export for Training
```
Click "📦 Export FP Frames" → Select folder
→ Get images/ + labels/ + metadata.json
→ Ready for model retraining!
```

---

## UI Controls

| Control | Action |
|---------|--------|
| **▶️ Play** | Start/pause video playback |
| **🚩 Flag Frame** | Capture current frame + detections |
| **➕ Add** | Manually add frame number |
| **Right-click frame** | Remove from list |
| **🗑️ Clear** | Remove all marked frames |
| **📦 Export FP Frames** | Batch export all marked frames |
| **Frame Counter** | Shows "Frames: N" selected |

---

## Export Formats

### YOLO (Recommended)
```
class_id center_x center_y width height
0 0.512 0.456 0.234 0.567
```
✅ Best for YOLOX training
✅ Normalized coordinates

### VOC
```xml
<annotation>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```
✅ Standard XML format
✅ Pixel coordinates

### COCO
```json
{
  "images": [{...}],
  "annotations": [{...}],
  "categories": [{...}]
}
```
✅ Research standard
✅ Detailed structure

---

## Output Folder Structure

```
my_fp_export/
├── images/
│   ├── frame_000001.jpg  ← Your frames
│   ├── frame_000002.jpg
│   └── frame_000003.jpg
├── labels/
│   ├── frame_000001.txt  ← Annotations
│   ├── frame_000002.txt
│   └── frame_000003.txt
└── metadata.json          ← Summary info
```

---

## Metadata File Breakdown

```json
{
  "total_frames": 10,              # Total marked
  "exported_frames": 8,            # With image data
  "exported_detections": 42,       # Total boxes
  "format": "yolo",                # Export format
  "class_names": [...],            # Your classes
  "frames": [
    {
      "frame_id": 123,
      "image_file": "frame_000123.jpg",
      "num_detections": 3,
      "timestamp": "2026-04-20T14:30:45.123456",
      "classes": ["person", "truck"]
    }
  ]
}
```

---

## Common Tasks

### Flag 50 Hard Examples
1. Load video → Play
2. Click Flag whenever model is wrong
3. Continue to 50 frames
4. Export → Done

### Find Similar Errors
1. Export FP frames
2. Review metadata.json
3. Group by class in "classes" field
4. Use for targeted retraining

### Merge with Existing Dataset
```bash
cp my_fp_export/images/* dataset/images/train/
cp my_fp_export/labels/* dataset/labels/train/
```

### Check Annotation Quality
```bash
# YOLO format check
for frame in frame_*.txt; do
  echo "$(basename $frame): $(wc -l < $frame) detections"
done
```

---

## Tips & Tricks

### Getting Best Results
✅ Flag frames as you find them (don't rely on memory)
✅ Include context (frame before/after issue)
✅ Balance classes (not all errors of one type)
✅ Aim for 50-200 hard examples per training round

### Avoid Common Mistakes
❌ Don't manually add frame numbers without watching (no image data)
❌ Don't forget to clear list between sessions
❌ Don't export frames with low confidence (annotate manually)
❌ Don't mix multiple videos in one export (confusing metadata)

### Performance
- Flagging: Instant (O(1))
- Export: ~2-5 sec per 50 frames
- Metadata: Generated automatically

---

## Integration with Training

### For YOLOX
```bash
# 1. Export FP frames in YOLO format
# 2. Copy to training dataset
cp -r my_fp_export/images/* data/train/images/
cp -r my_fp_export/labels/* data/train/labels/

# 3. Retrain
python tools/train.py --exp_file exps/default/yolox_s.py
```

### For PyTorch DataLoader
```python
from pathlib import Path
import torch
from torchvision.datasets import CocoDetection

# Point to export folder
fp_folder = Path('my_fp_export')
images = list((fp_folder / 'images').glob('*.jpg'))
labels = list((fp_folder / 'labels').glob('*.txt'))

# Add to existing dataset
dataset.images.extend(images)
dataset.labels.extend(labels)
```

---

## Troubleshooting

### Export says "No frame data"
→ You manually added frame numbers without watching video
→ Solution: Flag frames while playing video instead

### Class mismatch after training
→ Check class.txt file order matches YOLO export
→ Solution: Verify metadata.json class_names list

### Frames look wrong
→ Check if VideoThread is capturing properly
→ Solution: Try with simpler test video first

### Export is slow
→ Normal for 100+ frames
→ Solution: Consider smaller batches

---

## Files Created by Feature

📁 **images/** - Frame screenshots (JPEG)
📁 **labels/** - Annotations in selected format
📄 **metadata.json** - Frame info + class mapping

---

## Keyboard/Mouse Actions (Current)

| Action | Result |
|--------|--------|
| Click "🚩 Flag Frame" | Mark current frame |
| Right-click frame | Delete from list |
| Double-click frame | Select frame |
| Click "📦 Export FP Frames" | Batch export dialog |

---

## Next Steps After Export

1. **Review** - Check images/ and labels/ folders
2. **Merge** - Add to training dataset
3. **Verify** - Check class IDs match
4. **Train** - Retrain model with new hard examples
5. **Evaluate** - Test improved model

---

## See Full Documentation

- **FP_EXPORT_GUIDE.md** - Complete user guide with examples
- **IMPLEMENTATION_SUMMARY.md** - Technical implementation details
- **config.json** - Configuration reference
- **CODEBASE_OVERVIEW.md** - Full system architecture

