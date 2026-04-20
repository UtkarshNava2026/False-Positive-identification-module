# Object Tracking & False Positive Analysis

This document explains the object tracking system added to the False Positive Identification Agent.

## Overview

The tracking system analyzes detection patterns across frames to identify:
1. **False Positives**: Objects with inconsistent detection patterns (high confidence → low confidence → missed)
2. **Missed Detections**: Objects that disappear and reappear, indicating detection failures

## How It Works

### ByteTrack-style Tracking
The system uses a **ByteTrack-style tracker** to associate detections across frames:
- Matches detections and existing tracks using IoU
- Maintains track IDs across frames for the same object
- Handles temporary misses and track recovery

### Anomaly Detection
For each tracked object, the system analyzes:

| Metric | Indicates |
|--------|-----------|
| **Confidence Drop** | Sharp decrease in model confidence for the same object |
| **High Variance** | Inconsistent confidence scores across frames |
| **Missed Frames** | Object disappears for multiple frames then reappears |
| **Low Avg Confidence** | Object consistently detected with low confidence |

## Key Components

### 1. **tracker.py**
Core tracking module containing:

- `TrackedObject`: Represents a tracked object with history
  - `track_id`: Unique identifier for the object
  - `detections`: List of detections across frames
  - `confidence_history`: Confidence scores over time
  - `is_false_positive`: Flag indicating likely FP
  - `is_missed_detection`: Flag indicating likely missed detection

- `CentroidTracker`: Main tracking algorithm
  - `update()`: Process new detections
  - `get_anomalies()`: Retrieve detected anomalies
  - `get_track_summary()`: Get details of specific track

### 2. **detection_model.py** (Enhanced)
Updated to integrate tracking:
- `enable_tracking`: Toggle tracking on/off
- `tracker`: CentroidTracker instance
- `get_anomalies()`: Get detected false positives/missed detections
- `get_track_summary()`: Get specific track details
- `reset_tracker()`: Reset for new video

### 3. **threads.py** (Enhanced)
VideoThread now displays track IDs and emits anomaly signals:
- Track IDs shown as `ID:123` on bounding boxes
- `anomalies_signal`: Emitted every 30 frames with current anomalies

### 4. **analysis.py**
Analysis utility for generating reports:
- `TrackingAnalyzer`: Analyzes tracking results
- Generates recommendations based on false positive/missed detection rates
- Exports JSON reports

## Usage Examples

### Basic Usage with Detection Model
```python
from fpa_agent import DetectionModel, TrackingAnalyzer

# Load model with tracking enabled (default)
model = DetectionModel(
    pth_path="mondelich.pth",
    exp_path="exp.py",
    classes_path="class.txt",
    device="cpu",
    enable_tracking=True  # Enable tracking
)

# Process frames
for frame in video_frames:
    tracked_detections = model.predict(frame)
    for det in tracked_detections:
        print(f"Object ID: {det['track_id']}, Label: {det['label']}, Conf: {det['conf']:.2f}")

# Get anomaly analysis
anomalies = model.get_anomalies()
print(f"False Positives: {len(anomalies['false_positives'])}")
print(f"Missed Detections: {len(anomalies['missed_detections'])}")
```

### Generate Analysis Report
```python
from fpa_agent import TrackingAnalyzer

analyzer = TrackingAnalyzer()

# After processing video
anomalies = model.get_anomalies()
analysis = analyzer.analyze_detections(anomalies)

# Print human-readable report
analyzer.print_analysis(analysis)

# Export to JSON
analyzer.export_report("tracking_report.json")
```

### Get Specific Track Information
```python
# Get summary of specific track
track_summary = model.get_track_summary(track_id=5)
print(f"Track 5 Summary:")
print(f"  Detections: {track_summary['detections_count']}")
print(f"  Avg Confidence: {track_summary['avg_confidence']:.3f}")
print(f"  Status: {track_summary['status']}")
```

## Understanding Track Status

| Status | Meaning |
|--------|---------|
| `active` | Object currently being detected |
| `missed` | Object hasn't been detected for several frames |
| `recovered` | Object was missed but then detected again |
| `confirmed_fp` | High confidence this is a false positive |

## Configuration

### Tracking Parameters
Edit tracking parameters in `detection_model.py`:

```python
self.tracker = CentroidTracker(
    max_disappear=30,      # Max frames before removing track
    max_distance=50        # Max pixel distance for centroid matching
)
```

Adjust these based on your use case:
- **Slow-moving objects**: Increase `max_distance`
- **Fast-moving objects**: Decrease `max_distance` or `max_disappear`
- **High FPS videos**: Decrease `max_distance`

## Output Format

Each detected object now includes:
```python
{
    'bbox': [x1, y1, x2, y2],           # Bounding box coordinates
    'label': 'person',                   # Class label
    'conf': 0.85,                        # Confidence score
    'track_id': 5,                       # Unique track ID
    'centroid': [cx, cy]                # Center point
}
```

## Anomaly Report Structure

```json
{
  "summary": {
    "total_objects_tracked": 150,
    "false_positive_rate": 12.5,
    "missed_detection_rate": 8.3,
    "false_positives_found": 18,
    "missed_detections_found": 12
  },
  "false_positives": {
    "person": [
      {
        "track_id": 5,
        "avg_confidence": 0.45,
        "confidence_std": 0.32,
        "anomaly_score": 0.35
      }
    ]
  },
  "missed_detections": [...],
  "recommendations": [...]
}
```

## Tips for Better Tracking

1. **Adjust Confidence Threshold**
   - Lower threshold → catch more objects but more false positives
   - In config.json: `"test_conf": 0.4`

2. **Tune NMS Threshold**
   - Controls overlap filtering of boxes
   - In detection_model.py: `self.nms_thr = 0.45`

3. **Monitor Anomaly Rates**
   - False Positive Rate > 30% → threshold too low
   - Missed Detection Rate > 20% → threshold too high or model limitation

4. **Video FPS**
   - Higher FPS = better tracking continuity
   - Lower FPS = can use larger `max_distance`

## Disable Tracking

If you want to run without tracking:
```python
model = DetectionModel(
    pth_path="mondelich.pth",
    exp_path="exp.py",
    classes_path="class.txt",
    device="cpu",
    enable_tracking=False  # Disable tracking
)
```

## Performance Considerations

- Tracking adds minimal computational overhead
- Centroid matching is O(n²) but fast for typical object counts
- Memory usage increases with number of simultaneous tracks
- Track history can be exported to disk if memory is a concern

## Future Enhancements

Possible improvements:
- DeepSORT for better matching using appearance features
- ByteTrack for more robust tracking
- Kalman filtering for motion prediction
- Multi-class specific thresholds
- Real-time anomaly visualization dashboard
