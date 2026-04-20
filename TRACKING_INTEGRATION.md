# Object Tracking Integration - Summary

This document summarizes the object tracking system added to the False Positive Identification Agent.

## What Was Added

### New Files Created

1. **`fpa_agent/tracker.py`** (Main tracking module)
   - `TrackedObject`: Data class for individual tracked objects
   - `ByteTracker`: Main tracking algorithm using IoU and score-based matching
   - Analyzes confidence patterns to detect false positives
   - Tracks missed detections when objects disappear

2. **`fpa_agent/analysis.py`** (Analysis and reporting)
   - `TrackingAnalyzer`: Analyzes tracking results
   - Generates anomaly reports with statistics
   - Provides recommendations based on false positive/missed detection rates
   - Exports reports to JSON

3. **`TRACKING_README.md`** (Full documentation)
   - Detailed guide on how tracking works
   - Configuration options
   - Usage examples
   - Performance tips

4. **`example_tracking.py`** (Example script)
   - Standalone script to test tracking on videos
   - Demonstrates full workflow
   - Saves annotated videos with track IDs

### Modified Files

1. **`fpa_agent/detection_model.py`**
   - Added `tracker` instance variable
   - Added `enable_tracking` parameter (default: True)
   - Modified `predict()` to return tracked detections with track IDs
   - Added `get_anomalies()` method
   - Added `get_track_summary()` method
   - Added `reset_tracker()` method

2. **`fpa_agent/threads.py`**
   - VideoThread now displays track IDs on bounding boxes
   - Added `anomalies_signal` for emitting anomaly updates
   - Resets tracker when starting new video
   - Emits anomaly analysis every 30 frames

3. **`fpa_agent/__init__.py`**
   - Exported new tracker and analysis modules

## Quick Start

### 1. Basic Usage
```python
from fpa_agent import DetectionModel

# Load model with tracking (enabled by default)
model = DetectionModel(
    pth_path="mondelich.pth",
    exp_path="exp.py",
    classes_path="class.txt"
)

# Get tracked detections
detections = model.predict(frame)
# detections now include 'track_id' field
```

### 2. Analyze for Anomalies
```python
# After processing video
anomalies = model.get_anomalies()

from fpa_agent import TrackingAnalyzer
analyzer = TrackingAnalyzer()
analysis = analyzer.analyze_detections(anomalies)
analyzer.print_analysis(analysis)
```

### 3. Run Example Script
```bash
# From command line
python example_tracking.py <video_file> --model mondelich.pth --classes class.txt

# With output video
python example_tracking.py video.mp4 --output output_tracked.mp4
```

## Key Features

### ✅ Object Tracking
- Unique track IDs for each object across frames
- Centroid-based association algorithm
- Handles object appearance/disappearance

### ✅ False Positive Detection
Detects anomalies like:
- Sudden confidence drops for same object
- High variance in confidence scores
- Low average detection confidence

### ✅ Missed Detection Detection
Identifies when:
- Tracked object disappears for multiple frames
- Object reappears after being missed
- Pattern indicates detection failure

### ✅ Comprehensive Analysis
Generates reports with:
- False positive rate and missed detection rate
- Per-class breakdown of anomalies
- Automated recommendations for improvement

### ✅ Minimal API Changes
- Backward compatible
- Can disable tracking if needed: `enable_tracking=False`
- Seamless integration with existing UI

## Detection Output Format

Each detection now includes:
```python
{
    'bbox': [x1, y1, x2, y2],      # Original
    'label': 'person',              # Original
    'conf': 0.85,                   # Original
    'track_id': 5,                  # ← NEW: Unique track ID
    'centroid': [cx, cy]            # ← NEW: Center point
}
```

## Anomaly Detection Logic

### False Positive Indicators
```
- Confidence drops > 0.3 between frames
- Standard deviation of confidence > 0.25 + low average
- Low average confidence (< 0.5)
```

### Missed Detection Indicators
```
- Object disappeared for 5+ frames
- Object was tracked for 3+ frames before disappearing
```

## Configuration

### Tracking Parameters (in detection_model.py)
```python
self.tracker = CentroidTracker(
    max_disappear=30,       # Frames before deleting track
    max_distance=50         # Max pixel distance for matching
)
```

Adjust based on your videos:
- **Fast-moving objects**: Decrease max_distance, increase max_disappear
- **Slow-moving objects**: Increase max_distance
- **High FPS**: Decrease max_distance
- **Crowded scenes**: Decrease max_distance

## Performance

- ⚡ Negligible computational overhead
- 🎯 O(n²) matching but fast for typical object counts (< 100 objects)
- 💾 Memory increases with simultaneous tracks (usually < 1MB)

## Disabling Tracking

If you want to use the original code without tracking:
```python
model = DetectionModel(
    pth_path="mondelich.pth",
    exp_path="exp.py",
    classes_path="class.txt",
    enable_tracking=False  # Disable tracking
)
```

Detection output will not include track_id or centroid.

## Next Steps

1. **Read** `TRACKING_README.md` for detailed documentation
2. **Run** `example_tracking.py` on a test video
3. **Review** generated reports to understand anomalies
4. **Adjust** thresholds based on your use case
5. **Integrate** analysis into your workflow

## Troubleshooting

### Objects getting mixed up / Wrong track IDs
- Decrease `max_distance` if objects are close together
- Increase `max_distance` if tracking is losing objects

### Too many false positives detected
- Increase `test_conf` threshold in config
- Increase NMS threshold

### Too many missed detections detected
- Decrease `test_conf` threshold in config
- Check if objects are truly being missed or just tracked poorly

### Memory usage too high
- Decrease `max_disappear` to remove old tracks faster
- Export/archive track history to disk

## Technical Details

### Centroid Tracking Algorithm
1. Calculate centroid (center point) of each detection
2. For each existing track, find closest detection based on distance
3. If within `max_distance`, associate detection to track
4. Unmatched detections create new tracks
5. Tracks not updated for `max_disappear` frames are deleted

### Anomaly Analysis
Each track is analyzed for:
1. Confidence history variance
2. Sharp drops in confidence
3. Detection/non-detection patterns
4. Duration and coverage in video

## References

For more advanced tracking, consider:
- **DeepSORT**: Combines motion and appearance features
- **ByteTrack**: Focuses on high-recall tracking
- **Kalman Filters**: Motion prediction for fast objects

## Support

For issues or questions:
1. Check `TRACKING_README.md` for detailed guide
2. Review `example_tracking.py` for working example
3. Check tracker parameter documentation
4. Examine generated JSON reports for insights
