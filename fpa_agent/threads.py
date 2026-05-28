import os
import cv2
from urllib.parse import unquote, urlparse, urlunparse
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class ModelLoaderThread(QThread):
    """Load model in a separate thread to avoid freezing the UI."""
    model_loaded_signal = pyqtSignal(object, bool, str)
    progress_signal = pyqtSignal(str)

    def __init__(self, pth_path, exp_path, classes_path, device='cpu'):
        super().__init__()
        self.pth_path = pth_path
        self.exp_path = exp_path
        self.classes_path = classes_path
        self.device = device
        self.model = None

    def run(self):
        try:
            self.progress_signal.emit("Loading model...")
            from .detection_model import DetectionModel
            self.model = DetectionModel(self.pth_path, self.exp_path,
                                        self.classes_path, self.device)
            self.model_loaded_signal.emit(self.model, True,
                                         "Model loaded successfully!")
        except Exception as e:
            self.model_loaded_signal.emit(None, False,
                                         f"Failed to load model:\n{str(e)}")


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap, list, int, object)
    finished_signal = pyqtSignal()
    anomalies_signal = pyqtSignal(dict)

    def __init__(self, source, model, target_fps: int = 0, frame_step: int = 1):
        super().__init__()
        self.source = source
        self.model = model
        self.target_fps = int(target_fps) if target_fps else 0
        self.frame_step = max(1, int(frame_step) if frame_step else 1)
        self.paused = False
        self.stop_flag = False
        self.current_frame = None
        self.frame_index = 0
        self.cap = None

    def _open_capture(self, source: str):
        src = source.strip()

        # RTSP reliability: force FFMPEG backend + TCP transport + timeouts.
        if src.lower().startswith("rtsp://"):
            # Users often paste URL-encoded passwords (e.g. %23 for '#').
            # Properly decode only the username/password, not the entire URL,
            # to avoid issues with special characters like '#' in passwords.
            try:
                parsed = urlparse(src)
                if parsed.username or parsed.password:
                    # Decode username and password
                    username = unquote(parsed.username) if parsed.username else ""
                    password = unquote(parsed.password) if parsed.password else ""
                    
                    # Reconstruct netloc with decoded credentials
                    if password:
                        netloc = f"{username}:{password}@{parsed.hostname}"
                    else:
                        netloc = f"{username}@{parsed.hostname}" if username else parsed.hostname
                    
                    # Add port if present
                    if parsed.port:
                        netloc = f"{netloc}:{parsed.port}"
                    
                    # Reconstruct the URL
                    src = urlunparse((
                        parsed.scheme,
                        netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment
                    ))
            except Exception as e:
                print(f"Warning: Could not parse RTSP URL properly: {e}")
                # Fall back to using the original URL if parsing fails
                pass

            # These options are consumed by OpenCV's FFMPEG backend.
            # If they're unsupported in a given build, they are safely ignored.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
            )

            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
            except Exception:
                pass
            return cap, src

        return cv2.VideoCapture(src), src

    def run(self):
        self.cap, opened_source = self._open_capture(self.source)
        if not self.cap.isOpened():
            print(f"Failed to open source: {opened_source}")
            self.finished_signal.emit()
            return

        # Reset tracker when starting new video
        if self.model and hasattr(self.model, 'reset_tracker'):
            self.model.reset_tracker()

        last_emit_time = time.monotonic()
        while not self.stop_flag:
            if not self.paused:
                # Skip frames (fast validation): grab without decode.
                for _ in range(self.frame_step - 1):
                    grabbed = self.cap.grab()
                    if not grabbed:
                        self.stop_flag = True
                        break
                    self.frame_index += 1
                if self.stop_flag:
                    break

                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame = frame
                self.frame_index += 1
                detections = self.model.predict(frame) if self.model else []

                disp_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['conf']
                    label = det['label']
                    display_label = f"{label} {conf:.2f}"
                    color = (0, 255, 0)
                    
                    cv2.rectangle(disp_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(disp_frame, display_label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                rgb_image = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img)
                self.change_pixmap_signal.emit(pixmap, detections, self.frame_index, self.current_frame)

                # Periodically emit anomaly analysis
                if self.frame_index % 30 == 0 and self.model and hasattr(self.model, 'get_anomalies'):
                    anomalies = self.model.get_anomalies()
                    self.anomalies_signal.emit(anomalies)

                # Throttle processing/display FPS if configured.
                if self.target_fps and self.target_fps > 0:
                    now = time.monotonic()
                    frame_period = 1.0 / float(self.target_fps)
                    elapsed = now - last_emit_time
                    sleep_s = frame_period - elapsed
                    if sleep_s > 0:
                        self.msleep(int(sleep_s * 1000))
                    last_emit_time = time.monotonic()
            else:
                self.msleep(5)

        self.cap.release()
        self.finished_signal.emit()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stop_flag = True
        self.wait()
