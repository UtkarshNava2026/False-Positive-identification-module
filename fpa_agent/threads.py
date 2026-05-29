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
        drift_pool_mode="last_scale",
        drift_projection_type="linear_relu",
        drift_projection_weights=None,
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
        self.drift_pool_mode = drift_pool_mode
        self.drift_projection_type = drift_projection_type
        self.drift_projection_weights = drift_projection_weights

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
                drift_pool_mode=self.drift_pool_mode,
                drift_projection_type=self.drift_projection_type,
                drift_projection_weights=self.drift_projection_weights,
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
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
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
        while not self.stop_flag and not self._reader_stop.is_set():
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

    def stop(self, wait_ms: int = 3000):
        self.stop_flag = True
        self._reader_stop.set()
        if self.isRunning():
            self.wait(wait_ms)


def _draw_detections(frame, detections):
    disp = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['conf']:.2f}"
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            disp, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return disp


def _bgr_to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img.copy())
