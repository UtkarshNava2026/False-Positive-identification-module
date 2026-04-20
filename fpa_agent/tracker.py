import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    track_id: int
    class_label: str
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    score: float = 0.0
    detections: List[Dict] = field(default_factory=list)  # List of {frame, bbox, conf, centroid}
    confidence_history: List[float] = field(default_factory=list)
    time_since_update: int = 0
    age: int = 0
    last_frame: int = 0
    start_frame: int = 0
    missed_frames: int = 0
    status: str = "ACTIVE"  # ACTIVE, LOST, REMOVED
    is_false_positive: bool = False
    is_missed_detection: bool = False
    anomaly_score: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    feature: np.ndarray = field(default_factory=lambda: np.zeros(128, dtype=np.float32))
    ema_alpha: float = 0.9

    def add_detection(self, frame_idx, bbox, conf, centroid=None):
        """Add a detection to the track history and update motion state."""
        if centroid is None:
            centroid = self._centroid(bbox)

        self._update_velocity(centroid)
        self._smooth_bbox(bbox)
        self.score = conf
        self.detections.append({
            'frame': frame_idx,
            'bbox': bbox,
            'conf': conf,
            'centroid': centroid
        })
        self.confidence_history.append(conf)
        self.last_frame = frame_idx
        self.time_since_update = 0
        self.missed_frames = 0
        self.age += 1
        self.status = "ACTIVE"

    def miss(self):
        """Increment missed frame counter and mark the track as LOST."""
        self.time_since_update += 1
        self.missed_frames = self.time_since_update
        self.status = "LOST"
        return self.missed_frames

    def recover(self, frame_idx):
        """Recover a missed track and restore ACTIVE status."""
        self.last_frame = frame_idx
        self.time_since_update = 0
        self.missed_frames = 0
        self.status = "ACTIVE"

    def predict_centroid(self) -> np.ndarray:
        """Predict next centroid based on motion history."""
        if self.age < 2 or np.linalg.norm(self.velocity) < 1e-3:
            return self._centroid(self.bbox)
        return self._centroid(self.bbox) + self.velocity

    def _update_velocity(self, centroid: np.ndarray):
        """Update motion vector based on the last tracked centroid."""
        if len(self.detections) >= 1:
            previous_centroid = np.array(self.detections[-1]['centroid'], dtype=np.float32)
            self.velocity = centroid - previous_centroid

    def _smooth_bbox(self, bbox: List[float]):
        """Smooth bbox using exponential moving average."""
        if self.age == 0:
            self.bbox = list(bbox)
            return
        self.bbox = [
            float(self.ema_alpha * old + (1.0 - self.ema_alpha) * new)
            for old, new in zip(self.bbox, bbox)
        ]

    @staticmethod
    def _centroid(bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def analyze_anomaly(self):
        """Analyze track for false positives and missed detections."""
        if len(self.confidence_history) < 2:
            return

        conf_array = np.array(self.confidence_history)
        
        # Check for confidence drops (false positive indicator)
        conf_diff = np.diff(conf_array)
        sharp_drop = np.any(conf_diff < -0.3)  # Sudden drop > 0.3
        high_variance = np.std(conf_array) > 0.25  # High confidence variance
        
        # Low average confidence
        low_avg_conf = np.mean(conf_array) < 0.5
        
        # Detection/non-detection pattern (likely missed detection or FP)
        if self.missed_frames > 5 and len(self.detections) > 3:
            self.is_missed_detection = True
            self.status = "confirmed_fp"
            
        if sharp_drop or (high_variance and low_avg_conf):
            self.is_false_positive = True
            self.anomaly_score = max(np.std(conf_array), max(np.abs(conf_diff)))

    def get_summary(self):
        """Get a summary of the tracked object."""
        return {
            'track_id': self.track_id,
            'label': self.class_label,
            'detections_count': len(self.detections),
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0,
            'confidence_std': np.std(self.confidence_history) if self.confidence_history else 0,
            'last_frame': self.last_frame,
            'missed_frames': self.missed_frames,
            'is_false_positive': self.is_false_positive,
            'is_missed_detection': self.is_missed_detection,
            'anomaly_score': self.anomaly_score,
            'status': self.status,
            'frame_range': (self.detections[0]['frame'], self.last_frame) if self.detections else (0, 0)
        }


class ByteTracker:
    """ByteTrack-style tracker optimized for stable IDs and recovery."""

    def __init__(self,
                 track_thresh=0.5,
                 match_thresh=0.45,
                 max_time_lost=30,
                 max_distance=80,
                 iou_weight=0.6,
                 dist_weight=0.4,
                 ema_alpha=0.9):
        """
        Initialize the ByteTracker.

        Args:
            track_thresh: Confidence threshold to treat a detection as high-confidence.
            match_thresh: Minimum combined similarity for a valid match.
            max_time_lost: Frames to keep a LOST track before removal.
            max_distance: Maximum centroid distance to allow matching.
            iou_weight: Weight for IoU in combined similarity.
            dist_weight: Weight for centroid distance in combined similarity.
            ema_alpha: Exponential moving average factor for bbox smoothing.
        """
        self.next_object_id = 0
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.lost_tracks: Dict[int, TrackedObject] = {}
        self.removed_tracks: Dict[int, TrackedObject] = {}
        self.track_history: List[TrackedObject] = []

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.max_distance = max_distance
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.ema_alpha = ema_alpha

        self.frame_count = 0

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Update tracker with detections and return stable tracked results.

        This implements ByteTrack-style two-stage association:
        1. Match high-confidence detections to ACTIVE tracks.
        2. Match remaining detections to LOST tracks.
        3. Create new tracks only from unmatched high-confidence detections.
        """
        self.frame_count = frame_idx

        detections = [dict(det) for det in detections]
        high_conf = [det for det in detections if det['conf'] >= self.track_thresh]
        low_conf = [det for det in detections if det['conf'] < self.track_thresh]

        # Stage 1: match active tracks with high-confidence detections.
        matched_active, unmatched_active_ids, unmatched_high_indices = self._match_tracks(
            self.active_tracks, high_conf
        )

        # Update matched ACTIVE tracks.
        for track_id, det_idx in matched_active:
            det = high_conf[det_idx]
            self._update_track(self.active_tracks[track_id], det, frame_idx)

        # Move unmatched active tracks to LOST state.
        for track_id in unmatched_active_ids:
            self._mark_track_lost(track_id)

        # Stage 2: attempt to recover LOST tracks with remaining detections.
        remaining_high = [high_conf[idx] for idx in unmatched_high_indices]
        second_stage_detections = remaining_high + low_conf
        matched_lost, unmatched_lost_ids, unmatched_second_indices = self._match_tracks(
            self.lost_tracks, second_stage_detections
        )

        # Recover matched LOST tracks.
        for track_id, det_idx in matched_lost:
            det = second_stage_detections[det_idx]
            self._recover_track(track_id, det, frame_idx)

        # Remove aged LOST tracks.
        for track_id in unmatched_lost_ids:
            track = self.lost_tracks.get(track_id)
            if track is None:
                continue
            track.time_since_update += 1
            track.missed_frames = track.time_since_update
            if track.time_since_update > self.max_time_lost:
                self._remove_track(track_id)

        # Create new tracks for remaining high-confidence detections only.
        num_recovered = len(remaining_high) - sum(1 for _, det_idx in matched_lost if det_idx < len(remaining_high))
        for idx in unmatched_second_indices:
            if idx >= len(remaining_high):
                continue
            det = second_stage_detections[idx]
            self._register_track(det, frame_idx)

        # Build output for all ACTIVE tracks updated this frame.
        tracked_detections = []
        for track_id, track in self.active_tracks.items():
            if track.last_frame == frame_idx:
                last_det = track.detections[-1]
                tracked_detections.append({
                    'bbox': last_det['bbox'],
                    'label': track.class_label,
                    'conf': last_det['conf'],
                    'track_id': track_id,
                    'centroid': last_det['centroid']
                })

        return tracked_detections

    def _match_tracks(self, tracks: Dict[int, TrackedObject], detections: List[Dict]):
        """Match a set of detections to existing tracks using combined similarity."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(tracks.keys()), list(range(len(detections)))

        track_ids = list(tracks.keys())
        det_boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
        det_centroids = np.array([self._centroid(det['bbox']) for det in detections], dtype=np.float32)

        track_boxes = np.array([
            self._predict_bbox(track) for track in tracks.values()
        ], dtype=np.float32)
        track_centroids = np.array([self._centroid(box) for box in track_boxes], dtype=np.float32)

        iou_matrix = self._iou_matrix(track_boxes, det_boxes)
        dist_matrix = np.linalg.norm(
            track_centroids[:, None, :] - det_centroids[None, :, :], axis=2
        )

        # Combined similarity: IoU + distance score.
        distance_score = np.clip(1.0 - dist_matrix / float(self.max_distance), 0.0, 1.0)
        similarity = self.iou_weight * iou_matrix + self.dist_weight * distance_score

        # Reject matches beyond max distance or label mismatch.
        invalid = dist_matrix > self.max_distance
        for track_idx, track_id in enumerate(track_ids):
            for det_idx, det in enumerate(detections):
                if invalid[track_idx, det_idx] or det['label'] != tracks[track_id].class_label:
                    similarity[track_idx, det_idx] = -1.0

        matched = []
        used_tracks = set()
        used_dets = set()

        # Greedy matching on combined similarity.
        while True:
            idx = np.argmax(similarity)
            max_score = similarity.flatten()[idx]
            if max_score < self.match_thresh:
                break
            track_idx, det_idx = divmod(idx, similarity.shape[1])
            track_id = track_ids[track_idx]
            if track_id in used_tracks or det_idx in used_dets:
                similarity[track_idx, det_idx] = -1.0
                continue

            matched.append((track_id, det_idx))
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            similarity[track_idx, :] = -1.0
            similarity[:, det_idx] = -1.0

        unmatched_tracks = [track_id for track_id in track_ids if track_id not in used_tracks]
        unmatched_dets = [det_idx for det_idx in range(len(detections)) if det_idx not in used_dets]
        return matched, unmatched_tracks, unmatched_dets

    def _register_track(self, det: Dict, frame_idx: int):
        """Create a new track from a detection."""
        track = TrackedObject(
            track_id=self.next_object_id,
            class_label=det['label'],
            bbox=list(det['bbox']),
            score=det['conf'],
            time_since_update=0,
            age=1,
            last_frame=frame_idx,
            start_frame=frame_idx,
            missed_frames=0,
            status='ACTIVE'
        )
        track.add_detection(frame_idx, det['bbox'], det['conf'], self._centroid(det['bbox']))
        self.active_tracks[self.next_object_id] = track
        self.next_object_id += 1

    def _update_track(self, track: TrackedObject, det: Dict, frame_idx: int):
        """Update an existing ACTIVE track with a matching detection."""
        track.add_detection(frame_idx, det['bbox'], det['conf'], self._centroid(det['bbox']))
        track.status = 'ACTIVE'
        track.time_since_update = 0
        track.missed_frames = 0
        track.age += 1

    def _recover_track(self, track_id: int, det: Dict, frame_idx: int):
        """Recover a LOST track if it matches a detection."""
        track = self.lost_tracks.pop(track_id)
        track.add_detection(frame_idx, det['bbox'], det['conf'], self._centroid(det['bbox']))
        track.status = 'ACTIVE'
        track.time_since_update = 0
        track.missed_frames = 0
        self.active_tracks[track_id] = track

    def _mark_track_lost(self, track_id: int):
        """Move an unmatched ACTIVE track to LOST state."""
        track = self.active_tracks.pop(track_id)
        track.status = 'LOST'
        track.time_since_update += 1
        track.missed_frames = track.time_since_update
        self.lost_tracks[track_id] = track

    def _remove_track(self, track_id: int):
        """Remove a track that has exceeded max_time_lost."""
        track = self.lost_tracks.pop(track_id)
        track.status = 'REMOVED'
        track.analyze_anomaly()
        self.removed_tracks[track_id] = track
        self.track_history.append(track)

    def _predict_bbox(self, track: TrackedObject) -> np.ndarray:
        """Predict the next bbox using motion consistency."""
        if track.age < 2 or np.linalg.norm(track.velocity) < 1e-3:
            return np.array(track.bbox, dtype=np.float32)

        centroid = track.predict_centroid()
        width = track.bbox[2] - track.bbox[0]
        height = track.bbox[3] - track.bbox[1]
        return np.array([
            centroid[0] - width / 2.0,
            centroid[1] - height / 2.0,
            centroid[0] + width / 2.0,
            centroid[1] + height / 2.0
        ], dtype=np.float32)

    @staticmethod
    def _centroid(bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    @staticmethod
    def _iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.maximum(rb - lt, 0.0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.maximum(union, 1e-6)

    def get_anomalies(self) -> Dict[str, List]:
        """Get detected anomalies from all tracks."""
        false_positives = []
        missed_detections = []

        all_tracks = list(self.active_tracks.values()) + list(self.lost_tracks.values()) + self.track_history

        for track in all_tracks:
            track.analyze_anomaly()
            summary = track.get_summary()
            if track.is_false_positive:
                false_positives.append(summary)
            elif track.is_missed_detection:
                missed_detections.append(summary)

        return {
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'total_tracks': len(all_tracks),
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks)
        }

    def get_track_summary(self, track_id: int) -> Dict:
        """Get detailed summary for a specific track."""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id].get_summary()
        if track_id in self.lost_tracks:
            return self.lost_tracks[track_id].get_summary()
        if track_id in self.removed_tracks:
            return self.removed_tracks[track_id].get_summary()
        for track in self.track_history:
            if track.track_id == track_id:
                return track.get_summary()
        return None

    def reset(self):
        """Reset tracker state for a new video or stream."""
        self.next_object_id = 0
        self.active_tracks = {}
        self.lost_tracks = {}
        self.removed_tracks = {}
        self.track_history = []
        self.frame_count = 0
