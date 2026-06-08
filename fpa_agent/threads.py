import os
import queue
import threading
import time
from urllib.parse import quote, unquote

from fpa_agent.qt_bootstrap import ensure_qt_bootstrap

ensure_qt_bootstrap()

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import cv2


class ModelLoaderThread(QThread):
    """Load detection model off the UI thread."""
    model_loaded_signal = pyqtSignal(object, bool, str)
    progress_signal = pyqtSignal(str)

    def __init__(
        self,
        model_path,
        exp_path,
        classes_path,
        device="cpu",
        drift_projection_path=None,
        drift_onnx_embedding_path=None,
        drift_input_size=None,
        drift_encoder="yolox_standard",
        drift_projection_type="linear_relu",
        drift_projection_weights=None,
        backend=None,
        openvino_device="CPU",
        drift_openvino_embedding_path=None,
    ):
        super().__init__()
        self.model_path = model_path
        self.exp_path = exp_path
        self.classes_path = classes_path
        self.device = device
        self.drift_projection_path = drift_projection_path
        self.drift_onnx_embedding_path = drift_onnx_embedding_path
        self.drift_input_size = drift_input_size
        self.drift_encoder = drift_encoder
        self.drift_projection_type = drift_projection_type
        self.drift_projection_weights = drift_projection_weights
        self.backend = backend
        self.openvino_device = openvino_device
        self.drift_openvino_embedding_path = drift_openvino_embedding_path

    def run(self):
        try:
            self.progress_signal.emit("Loading model...")
            from .detection_model import DetectionModel

            model = DetectionModel(
                self.model_path,
                self.exp_path,
                self.classes_path,
                self.device,
                drift_projection_path=self.drift_projection_path,
                drift_onnx_embedding_path=self.drift_onnx_embedding_path,
                drift_input_size=self.drift_input_size,
                drift_encoder=self.drift_encoder,
                drift_projection_type=self.drift_projection_type,
                drift_projection_weights=self.drift_projection_weights,
                backend=self.backend,
                openvino_device=self.openvino_device,
                drift_openvino_embedding_path=self.drift_openvino_embedding_path,
            )
            self.model_loaded_signal.emit(model, True, "Model loaded successfully!")
        except Exception as e:
            self.model_loaded_signal.emit(None, False, f"Failed to load model:\n{e}")


class DriftLoaderThread(QThread):
    """Load reference embeddings + embedder without blocking UI."""
    loaded_signal = pyqtSignal(object, bool, str)

    def __init__(self, reference_path, device="cpu", knn_sample_size=2048, encoder="yolox"):
        super().__init__()
        self.reference_path = reference_path
        self.device = device
        self.knn_sample_size = knn_sample_size
        self.encoder = encoder

    def run(self):
        try:
            from .drift_score import EmbeddingDriftScorer

            encoder = getattr(self, "encoder", "yolox")
            scorer = EmbeddingDriftScorer(
                self.reference_path,
                device=self.device,
                knn_sample_size=self.knn_sample_size,
                encoder=encoder,
            )
            ok, msg = scorer.load()
            self.loaded_signal.emit(scorer, ok, msg)
        except Exception as e:
            self.loaded_signal.emit(None, False, str(e))


def normalize_rtsp_url(url: str) -> str:
    """Percent-encode RTSP credentials so '#' in passwords does not break parsing."""
    url = (url or "").strip()
    if not url.lower().startswith("rtsp://"):
        return url

    rest = url[7:]
    if "@" not in rest:
        return url

    userinfo, hostpath = rest.rsplit("@", 1)
    if ":" not in userinfo:
        return url

    user, password = userinfo.split(":", 1)

    def _encode_userinfo_part(part: str) -> str:
        return quote(unquote(part), safe="")

    return f"rtsp://{_encode_userinfo_part(user)}:{_encode_userinfo_part(password)}@{hostpath}"


class RtspProbeThread(QThread):
    """Open RTSP capture on a worker thread (avoids UI freeze while connecting)."""
    result_signal = pyqtSignal(bool, str, object)

    def __init__(self, url: str):
        super().__init__()
        self.url = normalize_rtsp_url(url.strip())

    def _open(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
            "stimeout;5000000|max_delay;500000"
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        except Exception:
            pass
        return cap

    def run(self):
        cap = self._open()
        if cap.isOpened():
            self.result_signal.emit(True, self.url, cap)
        else:
            cap.release()
            self.result_signal.emit(False, self.url, None)


def open_video_capture(source: str):
    """Open file or RTSP source; returns (cap, opened_source)."""
    src = source.strip()
    if src.lower().startswith("rtsp://"):
        src = normalize_rtsp_url(src)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
            "stimeout;5000000|max_delay;500000"
        )
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        except Exception:
            pass
        return cap, src
    return cv2.VideoCapture(src), src


class ImageProcessThread(QThread):
    """Load and infer on a still image off the UI thread."""
    finished_signal = pyqtSignal(object, list, object, str)
    error_signal = pyqtSignal(str)

    def __init__(self, path, model, drift_scorer=None):
        super().__init__()
        self.path = path
        self.model = model
        self.drift_scorer = drift_scorer

    def run(self):
        frame = cv2.imread(self.path)
        if frame is None:
            self.error_signal.emit("Cannot read image.")
            return

        detections = self.model.predict(frame) if self.model else []
        drift = None
        if self.drift_scorer and getattr(self.drift_scorer, "ready", False):
            drift = self.drift_scorer.score_frame(frame, frame_index=0)

        disp = _draw_detections(frame, detections)
        pixmap = _bgr_to_pixmap(disp)
        self.finished_signal.emit(pixmap, detections, frame.copy(), self.path)
        if drift is not None:
            pass  # caller can read drift from scorer.get_last()


class VideoThread(QThread):
    """
    Video/RTSP processing with a reader thread + bounded queue so decode and
    inference do not block the Qt event loop.
    """
    change_pixmap_signal = pyqtSignal(QPixmap, list, int, object)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    drift_signal = pyqtSignal(dict)
    total_frames_signal = pyqtSignal(int)   # emitted once: total frames in the file
    progress_signal = pyqtSignal(int)       # emitted every frame: current frame index

    def __init__(
        self,
        source,
        model,
        drift_scorer=None,
        target_fps: int = 0,
        frame_step: int = 1,
        preopened_cap=None,
    ):
        super().__init__()
        self.source = source
        self.model = model
        self.drift_scorer = drift_scorer
        self.target_fps = int(target_fps) if target_fps else 0
        self.frame_step = max(1, int(frame_step) if frame_step else 1)
        self.preopened_cap = preopened_cap
        self.paused = False
        self.stop_flag = False
        self.current_frame = None
        self.frame_index = 0
        self.cap = None
        self._frame_queue = queue.Queue(maxsize=3)
        self._reader_stop = threading.Event()
        self._seek_to = None    # target frame set by seek(); consumed in reader loop

    def run(self):
        if self.preopened_cap is not None:
            self.cap = self.preopened_cap
            opened_source = self.source
        else:
            self.cap, opened_source = open_video_capture(self.source)

        if not self.cap or not self.cap.isOpened():
            self.error_signal.emit(f"Failed to open: {opened_source}")
            self.finished_signal.emit()
            return

        if self.model and hasattr(self.model, "reset_tracker"):
            self.model.reset_tracker()

        # Emit total frame count so the UI can enable the seek slider
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            self.total_frames_signal.emit(total)

        reader = threading.Thread(target=self._reader_loop, daemon=True)
        reader.start()

        last_emit_time = time.monotonic()
        try:
            while not self.stop_flag:
                if self.paused:
                    self.msleep(8)
                    continue

                try:
                    frame = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if self._reader_stop.is_set():
                        break
                    continue

                if frame is None:
                    break

                self.current_frame = frame
                self.frame_index += 1
                try:
                    detections = self.model.predict(frame) if self.model else []
                except Exception as exc:
                    print(f"Inference error (frame {self.frame_index}): {exc}")
                    detections = []

                disp = _draw_detections(frame, detections)
                pixmap = _bgr_to_pixmap(disp)
                self.change_pixmap_signal.emit(
                    pixmap, detections, self.frame_index, self.current_frame
                )
                self.progress_signal.emit(self.frame_index)

                if self.drift_scorer and getattr(self.drift_scorer, "ready", False):
                    try:
                        drift = self.drift_scorer.score_frame(frame, self.frame_index)
                        self.drift_signal.emit(drift)
                    except Exception as exc:
                        print(f"Drift error (frame {self.frame_index}): {exc}")

                if self.target_fps > 0:
                    now = time.monotonic()
                    period = 1.0 / float(self.target_fps)
                    sleep_s = period - (now - last_emit_time)
                    if sleep_s > 0:
                        self.msleep(int(sleep_s * 1000))
                    last_emit_time = time.monotonic()
        finally:
            self._reader_stop.set()
            reader.join(timeout=2.0)
            if self.cap:
                self.cap.release()
            self.finished_signal.emit()

    def _reader_loop(self):
        is_live = False
        if isinstance(self.source, str):
            src_str = self.source.strip().lower()
            if src_str.startswith("rtsp://") or src_str.startswith("rtmp://") or src_str.isdigit():
                is_live = True
        elif isinstance(self.source, int):
            is_live = True

        while not self.stop_flag and not self._reader_stop.is_set():
            # Handle a pending seek (must be executed on this thread, which owns cap)
            if self._seek_to is not None:
                target = self._seek_to
                self._seek_to = None
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                self.frame_index = max(0, target - 1)

            if self.paused:
                time.sleep(0.01)
                continue

            for _ in range(self.frame_step - 1):
                if not self.cap.grab():
                    self._enqueue_sentinel()
                    return

            ret, frame = self.cap.read()
            if not ret:
                self._enqueue_sentinel()
                return

            if is_live:
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            else:
                try:
                    self._frame_queue.put(frame, timeout=1.0)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put(frame, timeout=0.5)
                    except queue.Full:
                        pass

    def _enqueue_sentinel(self):
        self._reader_stop.set()
        try:
            self._frame_queue.put(None, timeout=0.5)
        except queue.Full:
            pass

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def seek(self, frame_number: int):
        """Thread-safe seek: clear queue then signal reader loop to call cap.set()."""
        # Drain stale frames so the display updates immediately after seek
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._seek_to = int(frame_number)

    def stop(self, wait_ms: int = 3000):
        self.stop_flag = True
        self._reader_stop.set()
        if self.isRunning():
            self.wait(wait_ms)


_BGR_COLORS = [
    (189, 114, 0),
    (25, 83, 217),
    (32, 177, 237),
    (142, 47, 126),
    (48, 172, 119),
    (238, 190, 77),
    (47, 20, 162),
    (77, 77, 77),
    (153, 153, 153),
    (0, 0, 255),
    (0, 128, 255),
    (0, 191, 191),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 170),
    (0, 85, 85),
    (0, 170, 85),
    (0, 255, 85),
    (0, 85, 170),
    (0, 170, 170),
]


def _get_label_color(label, cls_id=None):
    if cls_id is not None:
        try:
            return _BGR_COLORS[int(cls_id) % len(_BGR_COLORS)]
        except Exception:
            pass
    import hashlib
    h = int(hashlib.md5(label.encode('utf-8')).hexdigest(), 16)
    return _BGR_COLORS[h % len(_BGR_COLORS)]


def _draw_detections(frame, detections):
    disp = frame.copy()
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Get label and track_id if available
        label_text = det.get("label", "object")
        if "track_id" in det:
            label_text = f"{label_text} #{det['track_id']}"
            
        # Add confidence if available
        conf = det.get("conf")
        if conf is not None:
            label_text += f" {conf:.2f}"
            
        color = _get_label_color(det.get("label", "object"), det.get("cls_id"))
        
        # Draw bounding box
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        txt_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        
        # Position label text background
        # If the box is too close to the top of the frame, draw the label inside the box
        y_text_top = y1 - txt_size[1] - 6
        if y_text_top < 0:
            back_tl = (x1, y1)
            back_br = (x1 + txt_size[0] + 6, y1 + txt_size[1] + 6)
            text_pos = (x1 + 3, y1 + txt_size[1] + 3)
        else:
            back_tl = (x1, y1 - txt_size[1] - 6)
            back_br = (x1 + txt_size[0] + 6, y1)
            text_pos = (x1 + 3, y1 - 3)
            
        # Draw text background block (semi-solid color for clean overlay)
        cv2.rectangle(disp, back_tl, back_br, color, -1)
        
        # Choose text color (black or white) dynamically based on background color brightness
        brightness = sum(color) / 3.0
        txt_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        # Draw the label text
        cv2.putText(
            disp,
            label_text,
            text_pos,
            font,
            font_scale,
            txt_color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )
    return disp



def _bgr_to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img.copy())


class ImageFolderThread(QThread):
    """
    Iterate over a sorted list of image files, run inference + drift on each,
    and emit results so the UI can display them and apply auto-save logic.
    Supports pause/resume/stop just like VideoThread.
    """
    frame_signal = pyqtSignal(QPixmap, list, int, object, str)  # pixmap, dets, idx, raw, path
    progress_signal = pyqtSignal(int, int)    # current_index, total
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    drift_signal = pyqtSignal(dict)

    def __init__(self, image_paths, model, drift_scorer=None):
        super().__init__()
        self.image_paths = list(image_paths)
        self.model = model
        self.drift_scorer = drift_scorer
        self.paused = False
        self.stop_flag = False

    def run(self):
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            if self.stop_flag:
                break
            # Respect pause
            while self.paused and not self.stop_flag:
                self.msleep(50)
            if self.stop_flag:
                break

            frame = cv2.imread(path)
            if frame is None:
                self.progress_signal.emit(i + 1, total)
                continue

            try:
                detections = self.model.predict(frame) if self.model else []
            except Exception as exc:
                print(f"[ImageFolderThread] Inference error ({path}): {exc}")
                detections = []

            disp = _draw_detections(frame, detections)
            pixmap = _bgr_to_pixmap(disp)
            self.frame_signal.emit(pixmap, detections, i + 1, frame.copy(), path)
            self.progress_signal.emit(i + 1, total)

            if self.drift_scorer and getattr(self.drift_scorer, "ready", False):
                try:
                    drift = self.drift_scorer.score_frame(frame, frame_index=i + 1)
                    self.drift_signal.emit(drift)
                except Exception as exc:
                    print(f"[ImageFolderThread] Drift error ({path}): {exc}")

        self.finished_signal.emit()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self, wait_ms: int = 2000):
        self.stop_flag = True
        if self.isRunning():
            self.wait(wait_ms)


class VideoOfflineProcessThread(QThread):
    """
    Offline video processing thread to compute drift scores, detect classes,
    and save drift-flagged frames + X-AnyLabeling JSON annotations directly to disk.
    Runs as fast as possible without GUI update delays.
    """
    progress_signal = pyqtSignal(int, int, int)  # current_frame, total_frames, flagged_count
    finished_signal = pyqtSignal(int, str)       # flagged_count, save_dir
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, model, drift_scorer, threshold, save_dir, frame_step=1, target_offline_fps=0.0):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.drift_scorer = drift_scorer
        self.threshold = float(threshold) if threshold is not None else 0.0
        self.save_dir = save_dir
        self.frame_step = max(1, int(frame_step))
        self.target_offline_fps = float(target_offline_fps) if target_offline_fps is not None else 0.0
        self.stop_flag = False

    def run(self):
        import json
        if not os.path.exists(self.video_path):
            self.error_signal.emit(f"Video file not found: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap or not cap.isOpened():
            self.error_signal.emit(f"Failed to open video: {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # If target offline FPS is configured, calculate frame step dynamically
        if self.target_offline_fps > 0.0 and video_fps > 0.0:
            self.frame_step = max(1, round(video_fps / self.target_offline_fps))
            print(f"[OfflineProcess] Target offline FPS: {self.target_offline_fps}, Video FPS: {video_fps:.2f} -> Calculated frame_step: {self.frame_step}")

        # Create a video-specific folder under save_dir to avoid mixing files from different videos
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        video_save_dir = os.path.join(self.save_dir, video_name)

        # Setup paths for 'all_frames' and 'drifted_frames'
        all_dir = os.path.join(video_save_dir, "all_frames")
        drifted_dir = os.path.join(video_save_dir, "drifted_frames")
        os.makedirs(all_dir, exist_ok=True)
        os.makedirs(drifted_dir, exist_ok=True)

        flagged_count = 0
        frame_idx = 0

        if self.model and hasattr(self.model, "reset_tracker"):
            self.model.reset_tracker()

        try:
            while not self.stop_flag:
                if self.frame_step > 1:
                    # Grab frames to skip them quickly
                    for _ in range(self.frame_step - 1):
                        if not cap.grab():
                            break
                        frame_idx += 1

                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                drift_score = 0.0
                if self.drift_scorer and getattr(self.drift_scorer, "ready", False):
                    try:
                        drift = self.drift_scorer.score_frame(frame, frame_idx)
                        drift_score = drift.get("drift_score", 0.0)
                    except Exception as e:
                        print(f"[OfflineProcess] Drift score error at frame {frame_idx}: {e}")

                try:
                    detections = self.model.predict(frame) if self.model else []
                except Exception as e:
                    print(f"[OfflineProcess] Prediction error at frame {frame_idx}: {e}")
                    detections = []

                h, w = frame.shape[:2]
                shapes = []
                for det in detections:
                    bbox = det.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = bbox[:4]
                    shapes.append({
                        "kie_linking": [],
                        "label": det.get("label", "object"),
                        "score": float(det["conf"]) if det.get("conf") is not None else None,
                        "points": [
                            [round(float(x1), 6), round(float(y1), 6)],
                            [round(float(x2), 6), round(float(y1), 6)],
                            [round(float(x2), 6), round(float(y2), 6)],
                            [round(float(x1), 6), round(float(y2), 6)],
                        ],
                        "group_id": None,
                        "description": None,
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": None,
                        "attributes": {},
                    })

                # Base file naming stem
                stem = f"frame_{frame_idx:06d}_drift{drift_score:.1f}"

                # Save all annotations/frames to 'all_frames' subfolder
                all_img_path = os.path.join(all_dir, stem + ".jpg")
                all_json_path = os.path.join(all_dir, stem + ".json")
                cv2.imwrite(all_img_path, frame)

                all_payload = {
                    "version": "3.3.9",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": os.path.basename(all_img_path),
                    "imageData": None,
                    "imageHeight": int(h),
                    "imageWidth": int(w),
                    "description": "",
                }
                with open(all_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_payload, f, indent=2, ensure_ascii=False)

                # Save to 'drifted_frames' if drift score is more than or equal to threshold
                if drift_score >= self.threshold:
                    flagged_count += 1
                    drifted_img_path = os.path.join(drifted_dir, stem + ".jpg")
                    drifted_json_path = os.path.join(drifted_dir, stem + ".json")
                    cv2.imwrite(drifted_img_path, frame)

                    drifted_payload = {
                        "version": "3.3.9",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": os.path.basename(drifted_img_path),
                        "imageData": None,
                        "imageHeight": int(h),
                        "imageWidth": int(w),
                        "description": "",
                    }
                    with open(drifted_json_path, "w", encoding="utf-8") as f:
                        json.dump(drifted_payload, f, indent=2, ensure_ascii=False)

                self.progress_signal.emit(frame_idx, total_frames, flagged_count)

        except Exception as e:
            self.error_signal.emit(f"Error during video processing: {e}")
            return
        finally:
            cap.release()

        if not self.stop_flag:
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            video_save_dir = os.path.join(self.save_dir, video_name)
            self.finished_signal.emit(flagged_count, video_save_dir)

    def stop(self):
        self.stop_flag = True

