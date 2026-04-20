# False Positive Frame Export Guide

## Overview
The False Positive Identification Agent now supports manual frame selection and batch export for model retraining. This guide explains how to use these features.

---

## Feature: Manual False Positive Frame Selection

### What It Does
Allows you to manually mark frames containing false positives while reviewing video, then export them in bulk with all detection data for retraining your model.

### How to Use

#### 1. **Flag Frames While Playing Video**

**Step-by-step:**
1. Open a video file using **📁 Open Video** button
2. Click **▶️ Play** to start playback
3. When you spot a false positive, click **🚩 Flag Frame**
   - The frame number and current detections are captured
   - Frame appears in the "Selected FP Frames" list below
4. Continue playing and flagging more frames as needed
5. View marked frames in the list below the playback controls

#### 2. **Manually Add Frame Numbers**

If you want to mark specific frames without watching the entire video:
1. Enter a frame number in the "Frame #" input field
2. Click **➕ Add**
3. Frame is added to the list (note: without captured image/detection data)

#### 3. **Manage Your Selection**

**View marked frames:**
- See count in "Frames: N" label
- Scroll through "Selected FP Frames" list

**Remove a frame:**
- Right-click on frame in the list
- Select "Yes" to remove it

**Clear all frames:**
- Click **🗑️ Clear** button
- Confirm deletion

---

## Feature: Batch Export for Retraining

### What It Does
Exports all marked false positive frames with their detection annotations in a format ready for model retraining.

### How to Use

#### Step 1: Prepare Your Frames
- Flag frames while playing video (see above)
- Ensure at least one frame has actual image data
- Use the count indicator to verify selection

#### Step 2: Export

1. Click **📦 Export FP Frames** button (enabled when frames selected)
2. Select destination folder
3. Choose export format:
   - **YOLO**: Normalized center-x, center-y, width, height format
   - **VOC**: XML annotation format
   - **COCO**: JSON annotation format
4. Click "Select Folder"

#### Step 3: Review Output

The export creates this structure:

```
export_folder/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── labels/
│   ├── frame_000001.txt      (YOLO format)
│   ├── frame_000001.xml      (VOC format)
│   ├── frame_000001.json     (COCO format)
│   └── ...
└── metadata.json
```

### Exported File Details

#### `images/frame_######.jpg`
- Actual frame image from video
- JPEG format, original resolution
- Preserves all visual information for retraining

#### `labels/frame_######.txt` (YOLO Format)
```
<class_id> <center_x> <center_y> <width> <height>
```
- All coordinates normalized to [0, 1]
- One detection per line
- Class IDs correspond to your class list order

Example:
```
0 0.512 0.456 0.234 0.567
1 0.234 0.789 0.123 0.456
```

#### `labels/frame_######.xml` (VOC Format)
```xml
<annotation>
  <filename>frame_000001.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>500</ymax>
    </bndbox>
  </object>
</annotation>
```

#### `labels/frame_######.json` (COCO Format)
```json
{
  "images": [{
    "id": 1,
    "file_name": "frame_000001.jpg",
    "width": 1920,
    "height": 1080
  }],
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 0,
    "bbox": [100, 200, 200, 300],
    "area": 60000,
    "iscrowd": 0
  }],
  "categories": [{
    "id": 0,
    "name": "person"
  }]
}
```

#### `metadata.json`
```json
{
  "total_frames": 10,
  "exported_frames": 8,
  "exported_detections": 42,
  "format": "yolo",
  "class_names": ["person", "dock_open", "truck", ...],
  "frames": [
    {
      "frame_id": 123,
      "image_file": "frame_000123.jpg",
      "num_detections": 3,
      "timestamp": "2026-04-20T14:30:45.123456",
      "classes": ["person", "truck"]
    },
    ...
  ]
}
```

---

## Use Cases

### Training Dataset Creation
1. Review video output with current model
2. Flag frames where detections are wrong
3. Export flagged frames
4. Use as hard-negative examples for retraining

### Quality Assurance
1. After training new model, review outputs
2. Flag uncertain detections
3. Export for manual review/annotation
4. Create gold-standard dataset

### Model Comparison
1. Run multiple model versions on same video
2. Flag FPs from each version separately
3. Compare which model has fewer FPs
4. Use comparison frames for targeted retraining

---

## Tips for Best Results

### Flagging Frames
- ✅ Flag frames as you find them (don't rely on memory)
- ✅ Flag all variations of the same error type
- ✅ Include context frames (frames just before/after issue)
- ✅ Be consistent: decide what counts as "false positive"

### Export for Retraining
- ✅ Export minimum ~50-100 hard examples
- ✅ Balance classes: avoid all FPs being one class
- ✅ Include frames with varying quality/lighting
- ✅ Annotate manually if confidence is borderline

### Format Selection
- **YOLO**: Best for YOLOv3/v4/v5/X training pipelines
- **VOC**: Compatible with older training frameworks
- **COCO**: Use for multi-modal training pipelines

---

## Troubleshooting

### "No frame data to export"
**Problem:** Only manually added frame numbers, no captured image data

**Solution:** 
- Flag frames while playing video instead
- Play video and capture frames naturally

### Export is slow
**Problem:** Large number of frames selected

**Solution:**
- This is normal (processing 100+ frames takes time)
- Use smaller batches if needed
- Consider reducing video quality/resolution

### Class mismatch in exported files
**Problem:** Class IDs in YOLO format don't match your class list

**Solution:**
- Check `class.txt` file in config
- Ensure it has classes in same order as training data
- Compare with metadata.json `class_names` list

---

## Integration with Training

### For YOLOX Retraining
1. Export frames in YOLO format
2. Split `images/` and `labels/` into train/val folders
3. Update your dataset config to include new hard examples
4. Train with existing model as pretrained weights

### Example Dataset Structure
```
dataset/
├── images/
│   ├── train/  (existing images + new FP frames)
│   └── val/
└── labels/
    ├── train/  (existing + new annotations)
    └── val/
```

### Python Code Example
```python
from pathlib import Path
from fpa_agent import DetectionModel

# Export FP frames from GUI (creates export_dir/)

# Prepare for retraining
fp_dir = Path('export_dir/images')
dataset_train = Path('dataset/images/train')

# Copy exported frames to training set
for img in fp_dir.glob('*.jpg'):
    shutil.copy(img, dataset_train / img.name)

# Copy corresponding labels
labels_src = Path('export_dir/labels')
labels_dst = Path('dataset/labels/train')
for lbl in labels_src.glob('*.txt'):
    shutil.copy(lbl, labels_dst / lbl.name)

# Retrain with updated dataset
# python YOLOX/tools/train.py --config config.py
```

---

## Data Privacy & Storage

- **Local storage only**: All frames stored locally in export folder
- **No uploading**: Application doesn't upload frames anywhere
- **No tracking**: Frame selections not logged or tracked
- **Metadata only**: Metadata.json contains only frame info, no raw data

---

## Keyboard Shortcuts (Future Enhancement)

Currently not implemented, but planned:
- `Shift + F`: Flag current frame (faster marking)
- `Delete`: Remove selected frame from list
- `Ctrl + E`: Export marked frames
- `Ctrl + L`: Clear list

---

## Limitations

1. **Manual frame entries**: Frames added via manual entry don't have image/detection data
2. **Real-time only**: Can only flag frames while video is playing
3. **Single video**: Export list is cleared when loading new video
4. **Memory**: Storing many frames in RAM (consider ~100MB per 100 frames)

---

## FAQs

**Q: Can I flag frames from images?**
A: No, currently only video playback supports flagging. You can manually enter frame numbers.

**Q: How many frames can I mark?**
A: No hard limit, but memory usage scales with frame count (~1-2MB per frame).

**Q: Can I export frames and continue flagging?**
A: No, you'll need to export in batches. Clear the list between exports.

**Q: Will flagged frames be lost if app crashes?**
A: Yes. Marked frames are stored in RAM only. Save to disk via export.

**Q: Can I edit detections before export?**
A: No, detected bboxes are exported as-is. Manually edit JSON/TXT if needed.

**Q: What format should I use?**
A: Use YOLO if training with YOLOX, otherwise choose what your framework prefers.

---

## See Also
- [README.md](README.md) - Project overview
- [TRACKING_README.md](TRACKING_README.md) - Tracking system details
- [config.json](config.json) - Configuration guide
