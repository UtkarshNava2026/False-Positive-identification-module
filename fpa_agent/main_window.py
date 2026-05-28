import os

from fpa_agent.qt_bootstrap import ensure_qt_bootstrap

ensure_qt_bootstrap()

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QLineEdit, QComboBox,
    QMessageBox, QSlider, QGroupBox, QProgressBar,
    QListWidget, QStatusBar, QListWidgetItem, QSpinBox, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

import cv2
from datetime import datetime

from .config_manager import ConfigManager
from .style_manager import StyleSheetManager
from .threads import (
    ModelLoaderThread,
    VideoThread,
    RtspProbeThread,
    ImageProcessThread,
    DriftLoaderThread,
)
from .widgets import DriftGaugeWidget
from .export_utils import (
    export_yolo,
    export_voc,
    export_coco,
    export_false_positive_frames,
    detections_as_person_labels,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("False Positive Identification Agent")

        self.config = ConfigManager("config.json")
        w = self.config.get("ui.window_width", 1280)
        h = self.config.get("ui.window_height", 800)
        self.setMinimumSize(int(w), int(h))

        self.model = None
        self.model_loader_thread = None
        self.drift_scorer = None
        self.drift_loader_thread = None
        self.rtsp_probe_thread = None
        self.image_thread = None

        self.current_detections = []
        self.current_frame_pixmap = None
        self.current_raw_frame = None
        self.current_frame_index = 0
        self.false_positive_frames = []
        self.fp_frame_data = {}
        self.current_video_path = None
        self.is_video = False
        self.video_thread = None
        self._pending_rtsp_cap = None

        self.setStyleSheet(StyleSheetManager.get_stylesheet())
        self.init_ui()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.load_drift_reference_async()
        self.try_load_config_model()

    def load_drift_reference_async(self):
        ref_path = self.config.get("drift.reference_path", "embeddings.npy")
        if not ref_path:
            ref_path = "embeddings.npy"
        if not os.path.isabs(ref_path):
            base = os.path.dirname(os.path.abspath(self.config.config_path))
            ref_path = os.path.normpath(os.path.join(base, ref_path))
        if not os.path.exists(ref_path) and os.path.exists("embeddings.pkl"):
            ref_path = os.path.abspath("embeddings.pkl")
        elif not os.path.exists(ref_path) and os.path.exists("embeddings.npy"):
            ref_path = os.path.abspath("embeddings.npy")

        self.drift_gauge.update_drift({
            "ready": False,
            "loading": True,
            "message": "Loading reference embeddings…",
            "frame_index": 0,
        })

        device = self.config.get("model.device", "cpu")
        knn = int(self.config.get("drift.knn_sample_size", 2048) or 2048)
        enc = self.config.get("drift.encoder", "yolox")
        self.drift_loader_thread = DriftLoaderThread(
            ref_path, device=device, knn_sample_size=knn, encoder=enc
        )
        self.drift_loader_thread.loaded_signal.connect(self.on_drift_loaded)
        self.drift_loader_thread.start()

    def on_drift_loaded(self, scorer, success, message):
        if success and scorer is not None:
            self.drift_scorer = scorer
            self._attach_drift_to_model()
            last = scorer.get_last()
            self.drift_gauge.update_drift({
                "ready": last.get("ready", False),
                "drift_score": 0.0,
                "cosine_centroid": 1.0,
                "knn_mean_sim": 1.0,
                "reference_count": last.get("reference_count", 0),
                "frame_index": 0,
                "message": message,
                "encoder": last.get("encoder", ""),
            })
            self.status_bar.showMessage(f"Drift reference: {message}")
        else:
            self.drift_scorer = None
            self.drift_gauge.update_drift({
                "ready": False,
                "message": f"Drift unavailable: {message}",
                "frame_index": 0,
            })

    def _attach_drift_to_model(self):
        if not self.drift_scorer or not self.model:
            return
        enc = (self.config.get("drift.encoder", "yolox_standard") or "yolox_standard").lower()
        if enc in ("legacy", "linear_relu", "mlp_1024_512"):
            pass
        elif enc not in ("yolox", "yolox_standard", "standard", "neck_concat"):
            return
        ok, label = self.drift_scorer.attach_yolox_model(self.model)
        if ok:
            self.status_bar.showMessage(f"Drift encoder: {label}")
        elif self.model and str(self.config.get("model.path", "")).lower().endswith(".onnx"):
            embed = self.config.get("drift.onnx_embedding_path", "") or ""
            self.drift_gauge.update_drift({
                "ready": False,
                "message": (
                    "Export drift ONNX: python export_embedding_onnx.py "
                    f"→ set drift.onnx_embedding_path (now: {embed or 'empty'})"
                ),
                "frame_index": 0,
            })

    def try_load_config_model(self):
        model_path = self.config.get("model.path") or self.config.get("model.pth_path")
        if model_path and os.path.exists(model_path):
            self.load_model_async(
                model_path,
                self.config.get("model.exp_path", ""),
                self.config.get("model.classes_path", ""),
                self.config.get("model.device", "cpu"),
            )

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(14)
        root.setContentsMargins(14, 14, 14, 14)

        # —— Left: drift gauge + controls ——
        left_outer = QVBoxLayout()
        left_outer.setSpacing(12)

        self.drift_gauge = DriftGaugeWidget()
        left_outer.addWidget(self.drift_gauge)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setObjectName("leftScroll")

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self.build_input_group())
        controls_layout.addWidget(self.build_export_group())
        controls_layout.addStretch(1)
        scroll.setWidget(controls)
        left_outer.addWidget(scroll, 1)

        # —— Right: video + playback ——
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(self.build_display_group(), 1)
        right.addWidget(self.build_playback_group())

        root.addLayout(left_outer, 0)
        root.addLayout(right, 1)

    def build_input_group(self):
        group = QGroupBox("Input & Model")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        hint = QLabel(
            "Set model.path in config.json to a .pth or .onnx file.\n"
            "PyTorch (.pth) needs exp_path; ONNX runs without weights file."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_open_video = QPushButton("Open Video")
        self.btn_open_video.setMinimumHeight(36)
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.lbl_video_path = QLabel("No video / stream")
        self.lbl_video_path.setObjectName("pathLabel")
        layout.addWidget(self.btn_open_video)
        layout.addWidget(self.lbl_video_path)

        self.rtsp_edit = QLineEdit()
        self.rtsp_edit.setPlaceholderText("rtsp://user:pass@host:554/path")
        self.btn_load_rtsp = QPushButton("Connect RTSP")
        self.btn_load_rtsp.setMinimumHeight(36)
        self.btn_load_rtsp.clicked.connect(self.load_rtsp)
        layout.addWidget(QLabel("RTSP URL"))
        layout.addWidget(self.rtsp_edit)
        layout.addWidget(self.btn_load_rtsp)

        self.btn_open_image = QPushButton("Open Image")
        self.btn_open_image.setMinimumHeight(36)
        self.btn_open_image.clicked.connect(self.open_image_file)
        self.lbl_image_path = QLabel("No image")
        self.lbl_image_path.setObjectName("pathLabel")
        layout.addWidget(self.btn_open_image)
        layout.addWidget(self.lbl_image_path)

        model_row = QHBoxLayout()
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.setMinimumHeight(36)
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.lbl_model_status = QLabel("Model not loaded")
        self.lbl_model_status.setObjectName("modelStatusBad")
        model_row.addWidget(self.btn_load_model)
        model_row.addWidget(self.lbl_model_status, 1)
        layout.addLayout(model_row)

        self.model_progress = QProgressBar()
        self.model_progress.setVisible(False)
        self.model_progress.setMaximum(0)
        layout.addWidget(self.model_progress)

        group.setLayout(layout)
        return group

    def build_display_group(self):
        group = QGroupBox("Live View")
        layout = QVBoxLayout()
        self.display_label = QLabel("Load a video, RTSP stream, or image")
        self.display_label.setObjectName("videoDisplay")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(720, 405)
        layout.addWidget(self.display_label)
        group.setLayout(layout)
        return group

    def build_playback_group(self):
        group = QGroupBox("Playback & False Positives")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setMinimumHeight(34)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        row.addWidget(self.play_pause_btn)
        row.addWidget(self.slider)
        layout.addLayout(row)

        speed = QHBoxLayout()
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 60)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setToolTip("0 = unlimited")
        self.fps_spin.setValue(int(self.config.get("video.fps", 0) or 0))
        self.fps_spin.valueChanged.connect(self.on_target_fps_changed)
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 30)
        self.skip_spin.setPrefix("x")
        self.skip_spin.setValue(int(self.config.get("video.frame_step", 1) or 1))
        self.skip_spin.valueChanged.connect(self.on_frame_step_changed)
        speed.addWidget(QLabel("FPS cap:"))
        speed.addWidget(self.fps_spin)
        speed.addWidget(QLabel("Skip:"))
        speed.addWidget(self.skip_spin)
        speed.addStretch()
        layout.addLayout(speed)

        fp_row = QHBoxLayout()
        self.current_frame_label = QLabel("Frame: 0")
        self.btn_flag_fp = QPushButton("Flag FP")
        self.btn_flag_fp.setEnabled(False)
        self.btn_flag_fp.clicked.connect(self.flag_current_frame)
        self.fp_frame_input = QLineEdit()
        self.fp_frame_input.setPlaceholderText("#")
        self.fp_frame_input.setMaximumWidth(72)
        self.btn_add_frame = QPushButton("Add")
        self.btn_add_frame.clicked.connect(self.add_manual_frame)
        fp_row.addWidget(self.current_frame_label)
        fp_row.addWidget(self.btn_flag_fp)
        fp_row.addWidget(self.fp_frame_input)
        fp_row.addWidget(self.btn_add_frame)
        layout.addLayout(fp_row)

        self.fp_list = QListWidget()
        self.fp_list.setMinimumHeight(100)
        self.fp_list.itemDoubleClicked.connect(self.on_fp_list_double_click)
        layout.addWidget(self.fp_list)

        exp_row = QHBoxLayout()
        self.fp_count_label = QLabel("FP: 0")
        self.btn_export_fp_batch = QPushButton("Export FP Batch")
        self.btn_export_fp_batch.setEnabled(False)
        self.btn_export_fp_batch.clicked.connect(self.export_fp_frames_batch)
        self.btn_clear_fp = QPushButton("Clear")
        self.btn_clear_fp.clicked.connect(self.clear_fp_list)
        exp_row.addWidget(self.fp_count_label)
        exp_row.addWidget(self.btn_export_fp_batch, 1)
        exp_row.addWidget(self.btn_clear_fp)
        layout.addLayout(exp_row)

        group.setLayout(layout)
        return group

    def build_export_group(self):
        group = QGroupBox("Export Annotations")
        layout = QHBoxLayout()
        self.export_format = QComboBox()
        self.export_format.addItems(["YOLO", "VOC", "COCO"])
        fmt = self.config.get("export.default_format", "YOLO")
        idx = self.export_format.findText(fmt)
        if idx >= 0:
            self.export_format.setCurrentIndex(idx)
        self.btn_export = QPushButton("Export Current Frame")
        self.btn_export.clicked.connect(self.export_frame)
        layout.addWidget(QLabel("Format"))
        layout.addWidget(self.export_format)
        layout.addStretch()
        layout.addWidget(self.btn_export)
        group.setLayout(layout)
        return group

    def open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*)",
        )
        if path:
            if not self.model:
                QMessageBox.warning(
                    self, "Model",
                    "Load a model first (config.json loads on startup, or use Load Model).",
                )
                return
            self.lbl_video_path.setText(os.path.basename(path))
            self.start_video_source(path)

    def load_rtsp(self):
        url = self.rtsp_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "RTSP", "Enter an RTSP URL.")
            return
        if self.rtsp_probe_thread and self.rtsp_probe_thread.isRunning():
            return

        self.btn_load_rtsp.setEnabled(False)
        self.status_bar.showMessage("Connecting to RTSP…")
        self.rtsp_probe_thread = RtspProbeThread(url)
        self.rtsp_probe_thread.result_signal.connect(self.on_rtsp_probe_result)
        self.rtsp_probe_thread.start()

    def on_rtsp_probe_result(self, ok, url, cap):
        self.btn_load_rtsp.setEnabled(True)
        if ok and not self.model:
            cap.release()
            QMessageBox.warning(
                self, "Model",
                "Load a model first (config.json loads on startup, or use Load Model).",
            )
            self.status_bar.showMessage("RTSP: model not loaded")
            return
        if not ok:
            QMessageBox.critical(
                self, "RTSP failed",
                f"Could not open stream:\n{url}\n\n"
                "Check URL, credentials (# in password is auto-encoded), and network.",
            )
            self.status_bar.showMessage("RTSP connection failed")
            return
        self.lbl_video_path.setText(url[:80] + ("…" if len(url) > 80 else ""))
        self.start_video_source(url, preopened_cap=cap)

    def open_image_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All (*)",
        )
        if not path:
            return
        if not self.model:
            QMessageBox.warning(self, "Model", "Load a model first (config.json or Load Model).")
            return
        self.lbl_image_path.setText(os.path.basename(path))
        self._stop_video_async()
        self.is_video = False
        self.status_bar.showMessage("Processing image…")
        self.image_thread = ImageProcessThread(path, self.model, self.drift_scorer)
        self.image_thread.finished_signal.connect(self.on_image_processed)
        self.image_thread.error_signal.connect(self.on_image_error)
        self.image_thread.start()

    def on_image_processed(self, pixmap, detections, raw_frame, path):
        self.update_display(pixmap, detections, 0, raw_frame)
        self.play_pause_btn.setEnabled(False)
        self.current_frame_label.setText("Frame: N/A")
        if self.drift_scorer and self.drift_scorer.ready:
            drift = self.drift_scorer.get_last()
            drift["frame_index"] = 0
            self.on_drift_update(drift)
        self.status_bar.showMessage(f"Image: {os.path.basename(path)}")

    def on_image_error(self, msg):
        QMessageBox.warning(self, "Image", msg)
        self.status_bar.showMessage("Image load failed")

    def _stop_video_async(self):
        if self.video_thread:
            t = self.video_thread
            self.video_thread = None
            t.stop(wait_ms=1500)

    def start_video_source(self, source, preopened_cap=None):
        self._stop_video_async()
        self.is_video = True
        self.current_video_path = source
        self.false_positive_frames = []
        self.fp_frame_data = {}
        self.update_fp_list()

        target_fps = int(self.config.get("video.fps", 0) or 0)
        frame_step = int(self.config.get("video.frame_step", 1) or 1)
        self.video_thread = VideoThread(
            source,
            self.model,
            drift_scorer=self.drift_scorer,
            target_fps=target_fps,
            frame_step=frame_step,
            preopened_cap=preopened_cap,
        )
        self.video_thread.change_pixmap_signal.connect(self.update_display)
        self.video_thread.finished_signal.connect(self.on_video_finished)
        self.video_thread.error_signal.connect(self.on_video_error)
        self.video_thread.drift_signal.connect(self.on_drift_update)
        self.video_thread.start()
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("Pause")
        self.slider.setEnabled(False)
        self.status_bar.showMessage(f"Playing: {source[:60]}…" if len(source) > 60 else f"Playing: {source}")

    def on_video_error(self, msg):
        QMessageBox.critical(self, "Video", msg)
        self.status_bar.showMessage(msg)

    def on_drift_update(self, drift: dict):
        self.drift_gauge.update_drift(drift)

    def on_target_fps_changed(self, value):
        self.config.set("video.fps", int(value))
        self.config.save()
        if self.video_thread:
            self.video_thread.target_fps = int(value)

    def on_frame_step_changed(self, value):
        self.config.set("video.frame_step", max(1, int(value)))
        self.config.save()
        if self.video_thread:
            self.video_thread.frame_step = max(1, int(value))

    def update_display(self, pixmap, detections, frame_index=0, raw_frame=None):
        self.display_label.setPixmap(
            pixmap.scaled(
                self.display_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.current_detections = detections
        self.current_frame_pixmap = pixmap
        self.current_raw_frame = raw_frame
        self.current_frame_index = frame_index
        self.current_frame_label.setText(f"Frame: {frame_index}")
        self.btn_flag_fp.setEnabled(self.is_video and frame_index > 0)

    def flag_current_frame(self):
        if self.current_frame_index <= 0 or self.current_raw_frame is None:
            return
        if self.current_frame_index not in self.false_positive_frames:
            self.false_positive_frames.append(self.current_frame_index)
            self.false_positive_frames.sort()
            self.fp_frame_data[self.current_frame_index] = {
                "detections": [d.copy() for d in self.current_detections],
                "timestamp": datetime.now().isoformat(),
                "frame_image": self.current_raw_frame.copy(),
            }
            self.update_fp_list()
            self.status_bar.showMessage(f"Flagged frame {self.current_frame_index}")

    def add_manual_frame(self):
        text = self.fp_frame_input.text().strip()
        if not text.isdigit():
            QMessageBox.warning(self, "Frame", "Enter a valid frame number.")
            return
        n = int(text)
        if n <= 0:
            return
        if n not in self.false_positive_frames:
            self.false_positive_frames.append(n)
            self.false_positive_frames.sort()
            self.fp_frame_data[n] = {
                "detections": [],
                "timestamp": datetime.now().isoformat(),
                "frame_image": None,
                "manual_entry": True,
            }
            self.update_fp_list()
        self.fp_frame_input.clear()

    def update_fp_list(self):
        self.fp_list.clear()
        for f in self.false_positive_frames:
            self.fp_list.addItem(QListWidgetItem(f"Frame {f}"))
        self.fp_count_label.setText(f"FP: {len(self.false_positive_frames)}")
        self.btn_export_fp_batch.setEnabled(len(self.false_positive_frames) > 0)

    def on_fp_list_double_click(self, item):
        self.status_bar.showMessage(item.text())

    def clear_fp_list(self):
        if not self.false_positive_frames:
            return
        if QMessageBox.question(
            self, "Clear", f"Clear {len(self.false_positive_frames)} flagged frames?",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            self.false_positive_frames = []
            self.fp_frame_data = {}
            self.update_fp_list()

    def on_video_finished(self):
        self.play_pause_btn.setText("Play")
        self.play_pause_btn.setEnabled(False)
        self.status_bar.showMessage("Playback finished")

    def closeEvent(self, event):
        self._stop_video_async()
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait(2000)
        if self.drift_loader_thread and self.drift_loader_thread.isRunning():
            self.drift_loader_thread.quit()
            self.drift_loader_thread.wait(2000)
        event.accept()

    def toggle_play_pause(self):
        if not self.video_thread:
            return
        if self.video_thread.paused:
            self.video_thread.resume()
            self.play_pause_btn.setText("Pause")
        else:
            self.video_thread.pause()
            self.play_pause_btn.setText("Play")

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Model file (.pth or .onnx)", "",
            "Models (*.pth *.onnx);;PyTorch (*.pth);;ONNX (*.onnx)",
        )
        if not path:
            return
        exp_path, _ = QFileDialog.getOpenFileName(
            self, "YOLOX experiment (.py) — required for .pth", "", "Python (*.py);;All (*)",
        )
        classes_path, _ = QFileDialog.getOpenFileName(
            self, "Class names (.txt)", "", "Text (*.txt);;All (*)",
        )
        device = self.config.get("model.device", "cpu")
        self.load_model_async(path, exp_path or "", classes_path or "", device)

    def load_model_async(self, model_path, exp_path, classes_path, device="cpu"):
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait(1500)

        proj = self.config.get("drift.projection_path") or ""
        onnx_emb = self.config.get("drift.onnx_embedding_path") or ""
        drift_size = self.config.get("drift.input_size")
        drift_enc = self.config.get("drift.encoder", "yolox_standard")
        drift_ptype = self.config.get("drift.projection_type", "linear_relu")
        drift_pweights = self.config.get("drift.projection_weights") or ""
        self.model_loader_thread = ModelLoaderThread(
            model_path,
            exp_path,
            classes_path,
            device,
            drift_projection_path=proj or None,
            drift_onnx_embedding_path=onnx_emb or None,
            drift_input_size=drift_size,
            drift_encoder=drift_enc,
            drift_projection_type=drift_ptype,
            drift_projection_weights=drift_pweights or None,
        )
        self.model_loader_thread.model_loaded_signal.connect(self.on_model_loaded)
        self.model_loader_thread.progress_signal.connect(
            lambda m: self.lbl_model_status.setText(m)
        )
        self.model_progress.setVisible(True)
        self.btn_load_model.setEnabled(False)
        self.lbl_model_status.setText("Loading…")
        self.model_loader_thread.start()

    def on_model_loaded(self, model, success, message):
        self.model_progress.setVisible(False)
        self.btn_load_model.setEnabled(True)
        if success:
            self.model = model
            self._attach_drift_to_model()
            name = os.path.basename(self.model_loader_thread.model_path)
            self.lbl_model_status.setText(f"Loaded: {name}")
            self.lbl_model_status.setObjectName("modelStatusOk")
            self.lbl_model_status.style().unpolish(self.lbl_model_status)
            self.lbl_model_status.style().polish(self.lbl_model_status)
            ext = os.path.splitext(name)[1].lower()
            kind = "ONNX" if ext == ".onnx" else "PyTorch"
            self.status_bar.showMessage(f"{kind} model ready on {self.model_loader_thread.device}")
            self.config.set("model.path", self.model_loader_thread.model_path)
            self.config.set("model.pth_path", self.model_loader_thread.model_path)
            self.config.set("model.exp_path", self.model_loader_thread.exp_path)
            self.config.set("model.classes_path", self.model_loader_thread.classes_path)
            self.config.set("model.device", self.model_loader_thread.device)
            self.config.save()
        else:
            self.model = None
            self.lbl_model_status.setText("Not loaded")
            self.lbl_model_status.setObjectName("modelStatusBad")
            QMessageBox.critical(self, "Model", message)

    def export_fp_frames_batch(self):
        if not self.false_positive_frames:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export directory")
        if not out_dir:
            return
        frames_with_data = sum(
            1 for f in self.false_positive_frames
            if self.fp_frame_data.get(f, {}).get("frame_image") is not None
        )
        if frames_with_data == 0:
            QMessageBox.warning(
                self, "Export",
                "No frame images stored. Flag frames during playback.",
            )
            return
        try:
            result = export_false_positive_frames(
                fp_frame_data=self.fp_frame_data,
                output_dir=out_dir,
                class_names=self.model.classes if self.model else ["object"],
                format_type=self.export_format.currentText().lower(),
            )
            QMessageBox.information(
                self, "Export",
                f"Exported {result['exported_frames']} frames to:\n{out_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export", str(e))

    def export_frame(self):
        if self.current_raw_frame is None:
            QMessageBox.warning(self, "Export", "No frame loaded.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export directory")
        if not out_dir:
            return
        temp = os.path.join(out_dir, "exported_frame.jpg")
        cv2.imwrite(temp, self.current_raw_frame)
        fmt = self.export_format.currentText().lower()
        dets = detections_as_person_labels(self.current_detections)
        if fmt == "yolo":
            export_yolo(temp, dets, ["person"], out_dir)
        elif fmt == "voc":
            export_voc(temp, dets, ["person"], out_dir)
        else:
            export_coco(temp, dets, ["person"], out_dir)
        QMessageBox.information(self, "Export", f"Saved to {out_dir}")
