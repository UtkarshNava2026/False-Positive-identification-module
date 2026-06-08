import json
import os

from fpa_agent.qt_bootstrap import ensure_qt_bootstrap

ensure_qt_bootstrap()

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QLineEdit, QComboBox,
    QMessageBox, QSlider, QGroupBox, QProgressBar,
    QListWidget, QStatusBar, QListWidgetItem, QSpinBox, QScrollArea,
    QInputDialog,
)
from PyQt5.QtCore import Qt, QTimer
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
    ImageFolderThread,
    VideoOfflineProcessThread,
)
from .widgets import DriftGaugeWidget, AnnotationLabel
from .export_utils import (
    export_yolo,
    export_voc,
    export_coco,
    export_false_positive_frames,
    detections_as_person_labels,
)


# ── X-AnyLabeling JSON helpers ─────────────────────────────────────────────────

def _clamped_bbox(bbox, img_w, img_h):
    """Return (x1, y1, x2, y2) clamped to image bounds as floats."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x1 = max(0.0, min(x1, float(img_w)))
    y1 = max(0.0, min(y1, float(img_h)))
    x2 = max(0.0, min(x2, float(img_w)))
    y2 = max(0.0, min(y2, float(img_h)))
    return x1, y1, x2, y2


def _xanylabeling_shape(label, x1, y1, x2, y2, score=None):
    """
    Build one X-AnyLabeling 3.3.9 shape dict for a rectangle.
    Points are the four corners (TL, TR, BR, BL) in pixel coordinates.
    score=None for manually drawn boxes (no model confidence).
    """
    return {
        "kie_linking": [],
        "label": label,
        "score": float(score) if score is not None else None,
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
    }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("False Positive Identification Agent")

        self.config = ConfigManager("config.json")
        w = self.config.get("ui.window_width", 1366)
        h = self.config.get("ui.window_height", 850)
        self.setMinimumSize(int(w), int(h))

        self.model = None
        self.model_loader_thread = None
        self.drift_scorer = None
        self.drift_loader_thread = None
        self.rtsp_probe_thread = None
        self.image_thread = None
        self.current_image_path = None
        self.batch_thread = None

        self.current_detections = []
        self.current_frame_pixmap = None
        self.current_raw_frame = None
        self.current_frame_index = 0
        self.false_positive_frames = []
        self.fp_frame_data = {}
        import tempfile
        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="fpa_temp_")
        self.temp_dir_path = self._temp_dir_obj.name
        self.current_video_path = None
        self.is_video = False
        self.video_thread = None
        self._pending_rtsp_cap = None
        # Folder batch mode
        self.folder_thread = None
        self.is_folder = False
        self.folder_paths = []
        self.folder_index = 0
        self.folder_timer = QTimer(self)
        self.folder_timer.timeout.connect(self.next_folder_image)
        # Auto-save tracking
        self._auto_save_count = 0
        # Annotation state
        self._manual_annotations = []     # [{"label": str, "bbox": (x1,y1,x2,y2), "score": None}]
        self._annotation_mode = False

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
        controls_layout.addWidget(self.build_batch_group())
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
            "Set model.path in config.json to a .pth, .onnx, or .xml file.\n"
            "Set model.backend to 'onnxruntime' or 'openvino'.\n"
            "PyTorch (.pth) needs exp_path; ONNX/OpenVINO run without .pth."
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

        img_row = QHBoxLayout()
        self.btn_open_image = QPushButton("Open Image")
        self.btn_open_image.setMinimumHeight(36)
        self.btn_open_image.clicked.connect(self.open_image_file)
        self.btn_open_folder = QPushButton("Open Image Folder")
        self.btn_open_folder.setMinimumHeight(36)
        self.btn_open_folder.clicked.connect(self.open_image_folder)
        img_row.addWidget(self.btn_open_image)
        img_row.addWidget(self.btn_open_folder)
        self.lbl_image_path = QLabel("No image / folder")
        self.lbl_image_path.setObjectName("pathLabel")
        layout.addLayout(img_row)
        layout.addWidget(self.lbl_image_path)

        save_row = QHBoxLayout()
        self.btn_set_save_folder = QPushButton("📁 Flagged Save Folder")
        self.btn_set_save_folder.setMinimumHeight(32)
        self.btn_set_save_folder.setToolTip("Choose where drift-flagged frames are auto-saved")
        self.btn_set_save_folder.clicked.connect(self.set_flagged_folder)
        self.auto_save_label = QLabel("Auto-saved: 0")
        self.auto_save_label.setObjectName("pathLabel")
        save_row.addWidget(self.btn_set_save_folder)
        save_row.addWidget(self.auto_save_label, 1)
        layout.addLayout(save_row)

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

        self.display_label = AnnotationLabel("Load a video, RTSP stream, or image")
        self.display_label.setObjectName("videoDisplay")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(854, 480)
        self.display_label.box_drawn.connect(self.on_box_drawn)
        self.display_label.box_deleted.connect(self.on_box_deleted)
        self.display_label.box_selected.connect(self.on_box_selected_changed)
        self.display_label.boxes_changed.connect(self.on_boxes_changed)
        layout.addWidget(self.display_label)

        # ── Annotation toolbar ─────────────────────────────────────────────────
        ann_row = QHBoxLayout()
        self.btn_annotate = QPushButton("✏️ Annotate")
        self.btn_annotate.setMinimumHeight(32)
        self.btn_annotate.setCheckable(True)
        self.btn_annotate.setToolTip(
            "Toggle annotation mode: click and drag to draw a bounding box, "
            "then choose a class label."
        )
        self.btn_annotate.clicked.connect(self.toggle_annotation_mode)

        self.btn_save_annotation = QPushButton("💾 Save Annotations")
        self.btn_save_annotation.setMinimumHeight(32)
        self.btn_save_annotation.setEnabled(False)
        self.btn_save_annotation.setToolTip(
            "Save current frame + all annotations (model detections + manual boxes) "
            "as an X-AnyLabeling JSON file."
        )
        self.btn_save_annotation.clicked.connect(self.save_manual_annotations)

        self.btn_clear_annotation = QPushButton("🗑 Clear Boxes")
        self.btn_clear_annotation.setMinimumHeight(32)
        self.btn_clear_annotation.setEnabled(False)
        self.btn_clear_annotation.setToolTip("Remove all manually drawn boxes from this frame.")
        self.btn_clear_annotation.clicked.connect(self.clear_manual_annotations)

        self.ann_count_label = QLabel("Boxes: 0")
        self.ann_count_label.setObjectName("pathLabel")

        ann_row.addWidget(self.btn_annotate)
        ann_row.addWidget(self.btn_save_annotation)
        ann_row.addWidget(self.btn_clear_annotation)
        ann_row.addWidget(self.ann_count_label, 1)
        layout.addLayout(ann_row)

        group.setLayout(layout)
        return group

    def build_playback_group(self):
        group = QGroupBox("Playback & False Positives")
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(10, 8, 10, 8)

        # Row 1: Play/Pause button + Slider
        row1 = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setMinimumHeight(32)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.sliderMoved.connect(self.on_seek_requested)
        row1.addWidget(self.play_pause_btn)
        row1.addWidget(self.slider)
        layout.addLayout(row1)

        # Row 2: FPS cap + Skip spin + Current frame + Flag FP button
        row2 = QHBoxLayout()
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
        
        self.current_frame_label = QLabel("Frame: 0")
        self.current_frame_label.setObjectName("pathLabel")
        self.btn_flag_fp = QPushButton("🚩 Flag FP")
        self.btn_flag_fp.setEnabled(False)
        self.btn_flag_fp.clicked.connect(self.flag_current_frame)
        self.btn_flag_fp.setMinimumHeight(28)

        row2.addWidget(QLabel("FPS cap:"))
        row2.addWidget(self.fps_spin)
        row2.addWidget(QLabel("Skip:"))
        row2.addWidget(self.skip_spin)
        row2.addSpacing(15)
        row2.addWidget(self.current_frame_label)
        row2.addStretch(1)
        row2.addWidget(self.btn_flag_fp)
        layout.addLayout(row2)

        # Row 3: Flagged frames list widget (Compacted height)
        self.fp_list = QListWidget()
        self.fp_list.setMinimumHeight(60)
        self.fp_list.setMaximumHeight(80)  # Restrict height so it doesn't expand
        self.fp_list.itemDoubleClicked.connect(self.on_fp_list_double_click)
        layout.addWidget(self.fp_list)

        # Row 4: FP count + Manual entry (# + Add) + Export FP Batch + Clear
        row4 = QHBoxLayout()
        self.fp_count_label = QLabel("FP: 0")
        self.fp_frame_input = QLineEdit()
        self.fp_frame_input.setPlaceholderText("Frame #")
        self.fp_frame_input.setMaximumWidth(70)
        self.fp_frame_input.setMinimumHeight(26)
        self.btn_add_frame = QPushButton("Add")
        self.btn_add_frame.clicked.connect(self.add_manual_frame)
        self.btn_add_frame.setMinimumHeight(26)

        self.btn_export_fp_batch = QPushButton("Export FP Batch")
        self.btn_export_fp_batch.setEnabled(False)
        self.btn_export_fp_batch.clicked.connect(self.export_fp_frames_batch)
        self.btn_export_fp_batch.setMinimumHeight(28)
        self.btn_clear_fp = QPushButton("Clear")
        self.btn_clear_fp.clicked.connect(self.clear_fp_list)
        self.btn_clear_fp.setMinimumHeight(28)

        row4.addWidget(self.fp_count_label)
        row4.addWidget(self.fp_frame_input)
        row4.addWidget(self.btn_add_frame)
        row4.addSpacing(15)
        row4.addWidget(self.btn_export_fp_batch, 1)
        row4.addWidget(self.btn_clear_fp)
        layout.addLayout(row4)

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

    def build_batch_group(self):
        group = QGroupBox("Offline Video Batch Automation")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        hint = QLabel(
            "Upload a video to automatically process it in the background.\n"
            "This will extract and annotate all drifted frames directly to your flagged folder."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_batch_process = QPushButton("⚡ Upload & Process Video Offline")
        self.btn_batch_process.setMinimumHeight(38)
        self.btn_batch_process.clicked.connect(self.start_batch_process)
        layout.addWidget(self.btn_batch_process)

        self.lbl_batch_status = QLabel("Status: Idle")
        self.lbl_batch_status.setObjectName("pathLabel")
        layout.addWidget(self.lbl_batch_status)

        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)

        self.btn_cancel_batch = QPushButton("Cancel Processing")
        self.btn_cancel_batch.setMinimumHeight(28)
        self.btn_cancel_batch.setVisible(False)
        self.btn_cancel_batch.clicked.connect(self.cancel_batch_process)
        layout.addWidget(self.btn_cancel_batch)

        self.btn_open_batch_dir = QPushButton("📂 Open Output Folder")
        self.btn_open_batch_dir.setMinimumHeight(32)
        self.btn_open_batch_dir.setVisible(False)
        self.btn_open_batch_dir.clicked.connect(self.open_last_batch_dir)
        layout.addWidget(self.btn_open_batch_dir)

        group.setLayout(layout)
        return group

    def _set_batch_ui_enabled(self, enabled: bool):
        # Disable/enable input files selection buttons
        self.btn_open_video.setEnabled(enabled)
        self.btn_open_image.setEnabled(enabled)
        self.btn_open_folder.setEnabled(enabled)
        self.btn_load_rtsp.setEnabled(enabled)
        self.btn_load_model.setEnabled(enabled)
        self.btn_set_save_folder.setEnabled(enabled)
        self.play_pause_btn.setEnabled(enabled if (self.is_video or self.is_folder) else False)
        # Disallow starting another batch process
        self.btn_batch_process.setEnabled(enabled)

    def start_batch_process(self):
        if not self.model:
            QMessageBox.warning(self, "Model", "Load a model first (config.json or Load Model).")
            return
        if not self.drift_scorer or not getattr(self.drift_scorer, "ready", False):
            QMessageBox.warning(
                self, "Drift Scorer",
                "Drift reference is not loaded or ready. Load drift reference / model first."
            )
            return

        # Let the user pick a video
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video for Offline Processing", "",
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*)",
        )
        if not path:
            return

        # Prepare parameters
        threshold = self.config.get("drift.auto_flag_threshold", 15.0)
        save_dir = self.config.get("drift.auto_save_dir", "flagged_frames") or "flagged_frames"
        if not os.path.isabs(save_dir):
            base = os.path.dirname(os.path.abspath(self.config.config_path))
            save_dir = os.path.normpath(os.path.join(base, save_dir))
        
        # Stop any active online threads/timer
        self._stop_video_async()

        # Update UI state for processing
        self._set_batch_ui_enabled(False)
        self.btn_cancel_batch.setVisible(True)
        self.batch_progress.setVisible(True)
        self.batch_progress.setValue(0)
        self.btn_open_batch_dir.setVisible(False)
        self.lbl_batch_status.setText("Status: Initializing...")

        # Setup and start thread
        frame_step = max(1, int(self.config.get("video.frame_step", 1) or 1))
        target_offline_fps = float(self.config.get("video.offline_fps", 0.0) or 0.0)
        self.batch_thread = VideoOfflineProcessThread(
            video_path=path,
            model=self.model,
            drift_scorer=self.drift_scorer,
            threshold=threshold,
            save_dir=save_dir,
            frame_step=frame_step,
            target_offline_fps=target_offline_fps,
        )
        self.batch_thread.progress_signal.connect(self.on_batch_progress)
        self.batch_thread.finished_signal.connect(self.on_batch_finished)
        self.batch_thread.error_signal.connect(self.on_batch_error)
        self.batch_thread.start()

    def cancel_batch_process(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.lbl_batch_status.setText("Status: Cancelling...")
            self.batch_thread.stop()
            self.batch_thread.wait(2000)
            self.lbl_batch_status.setText("Status: Cancelled")
            
            # Reset UI
            self._set_batch_ui_enabled(True)
            self.btn_cancel_batch.setVisible(False)
            self.batch_progress.setVisible(False)
            self.btn_open_batch_dir.setVisible(False)
            QMessageBox.information(self, "Offline Processing", "Video batch processing was cancelled.")

    def open_last_batch_dir(self):
        if hasattr(self, "last_batch_save_dir") and self.last_batch_save_dir and os.path.exists(self.last_batch_save_dir):
            os.startfile(self.last_batch_save_dir)

    def on_batch_progress(self, current_frame, total_frames, flagged_count):
        pct = int(current_frame * 100 / total_frames)
        self.batch_progress.setValue(pct)
        self.lbl_batch_status.setText(
            f"Status: Processing {current_frame}/{total_frames} (Flagged: {flagged_count})"
        )

    def on_batch_finished(self, flagged_count, save_dir):
        # Reset UI
        self._set_batch_ui_enabled(True)
        self.btn_cancel_batch.setVisible(False)
        self.batch_progress.setVisible(False)
        self.lbl_batch_status.setText("Status: Idle")

        # Save last output directory and show button
        self.last_batch_save_dir = save_dir
        self.btn_open_batch_dir.setVisible(True)

        QMessageBox.information(
            self, "Offline Processing Finished",
            f"Successfully processed video!\n"
            f"Saved {flagged_count} drift-flagged frames with X-AnyLabeling annotations to:\n"
            f"{save_dir}"
        )

    def on_batch_error(self, message):
        # Reset UI
        self._set_batch_ui_enabled(True)
        self.btn_cancel_batch.setVisible(False)
        self.batch_progress.setVisible(False)
        self.btn_open_batch_dir.setVisible(False)
        self.lbl_batch_status.setText("Status: Error")

        QMessageBox.critical(self, "Offline Processing Error", message)



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
        self.is_folder = False
        self.status_bar.showMessage("Processing image…")
        self.image_thread = ImageProcessThread(path, self.model, self.drift_scorer)
        self.image_thread.finished_signal.connect(self.on_image_processed)
        self.image_thread.error_signal.connect(self.on_image_error)
        self.image_thread.start()

    def on_image_processed(self, pixmap, detections, raw_frame, path):
        self.current_image_path = path
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
        if hasattr(self, "folder_timer") and self.folder_timer.isActive():
            self.folder_timer.stop()
        if self.folder_thread:
            t = self.folder_thread
            self.folder_thread = None
            t.stop(wait_ms=1500)
        if self.video_thread:
            t = self.video_thread
            self.video_thread = None
            t.stop(wait_ms=1500)

    def start_video_source(self, source, preopened_cap=None):
        self._stop_video_async()
        self.is_video = True
        self.is_folder = False
        self.current_video_path = source
        self.false_positive_frames = []
        self.fp_frame_data = {}
        self._auto_save_count = 0
        self.auto_save_label.setText("Auto-saved: 0")
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
        self.video_thread.total_frames_signal.connect(self.on_total_frames)
        self.video_thread.progress_signal.connect(self.on_video_progress)
        self.video_thread.start()
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("Pause")
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)   # enabled once total_frames_signal fires
        self.status_bar.showMessage(f"Playing: {source[:60]}…" if len(source) > 60 else f"Playing: {source}")

    def on_video_error(self, msg):
        QMessageBox.critical(self, "Video", msg)
        self.status_bar.showMessage(msg)

    def on_drift_update(self, drift: dict):
        self.drift_gauge.update_drift(drift)

        score = drift.get("drift_score", 0.0)
        threshold = self.config.get("drift.auto_flag_threshold")

        if threshold is not None and float(threshold) > 0.0:
            if score >= float(threshold):
                if self.current_frame_index not in self.false_positive_frames:
                    self.flag_current_frame()
                    print(f"[Auto-Drift] Flagged frame {self.current_frame_index} (drift {score:.2f}% >= {threshold}%)")
                # Also auto-save the raw frame + YOLO labels to the configured folder
                if self.current_raw_frame is not None:
                    # Gather both model detections and manual annotations
                    if self._annotation_mode:
                        all_boxes = list(self.display_label._boxes)
                    else:
                        all_boxes = (
                            [{"bbox": d["bbox"], "label": d.get("label", "object"),
                              "conf": float(d.get("conf")) if d.get("conf") is not None else None, "source": "model"}
                             for d in self.current_detections if d.get("bbox")] +
                            [{"bbox": a["bbox"], "label": a["label"],
                              "conf": None, "source": "manual"}
                             for a in self._manual_annotations]
                        )
                    self._auto_save_frame(
                        self.current_raw_frame,
                        self.current_frame_index,
                        score,
                        detections=all_boxes,
                    )

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
        self.current_detections = detections
        self.current_frame_pixmap = pixmap
        self.current_raw_frame = raw_frame
        self.current_frame_index = frame_index
        self.current_frame_label.setText(f"Frame: {frame_index}")
        self.btn_flag_fp.setEnabled((self.is_video or self.is_folder) and frame_index > 0)

        if self._annotation_mode:
            # Annotation mode: show clean (no-overlay) pixmap; widget paints boxes
            if raw_frame is not None:
                from fpa_agent.threads import _bgr_to_pixmap
                clean_px = _bgr_to_pixmap(raw_frame)
                self.display_label.setPixmap(
                    clean_px.scaled(self.display_label.size(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        elif self._manual_annotations:
            self._redraw_with_annotations()
        else:
            self.display_label.setPixmap(
                pixmap.scaled(self.display_label.size(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def flag_current_frame(self):
        if self.current_frame_index <= 0 or self.current_raw_frame is None:
            return
        if self.current_frame_index not in self.false_positive_frames:
            self.false_positive_frames.append(self.current_frame_index)
            self.false_positive_frames.sort()
            temp_path = os.path.join(self.temp_dir_path, f"frame_{self.current_frame_index:06d}.jpg")
            cv2.imwrite(temp_path, self.current_raw_frame)
            self.fp_frame_data[self.current_frame_index] = {
                "detections": [d.copy() for d in self.current_detections],
                "timestamp": datetime.now().isoformat(),
                "frame_image_path": temp_path,
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
                "frame_image_path": None,
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
            for frame_id, data in self.fp_frame_data.items():
                path = data.get("frame_image_path")
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            self.false_positive_frames = []
            self.fp_frame_data = {}
            self.update_fp_list()

    def on_video_finished(self):
        # Guard: ignore signals from old threads that were stopped when a new source loaded
        if self.sender() is not self.video_thread:
            return
        self.play_pause_btn.setText("Play")
        self.play_pause_btn.setEnabled(False)
        self.slider.setEnabled(False)
        self.status_bar.showMessage("Playback finished")

    def closeEvent(self, event):
        self._stop_video_async()
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_thread.wait(2000)
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait(2000)
        if self.drift_loader_thread and self.drift_loader_thread.isRunning():
            self.drift_loader_thread.quit()
            self.drift_loader_thread.wait(2000)
        if hasattr(self, "_temp_dir_obj"):
            try:
                self._temp_dir_obj.cleanup()
            except Exception:
                pass
        event.accept()

    def toggle_play_pause(self):
        # Handle folder batch mode
        if self.is_folder:
            if self.folder_timer.isActive():
                self.folder_timer.stop()
                self.play_pause_btn.setText("Play")
            else:
                self.folder_timer.start(1000)
                self.play_pause_btn.setText("Pause")
            return
        # Handle video / RTSP mode
        if not self.video_thread:
            return
        if self.video_thread.paused:
            self.video_thread.resume()
            self.play_pause_btn.setText("Pause")
        else:
            self.video_thread.pause()
            self.play_pause_btn.setText("Play")

    def toggle_annotation_mode(self):
        self._annotation_mode = self.btn_annotate.isChecked()
        self.display_label.set_annotation_mode(self._annotation_mode)
        if self._annotation_mode:
            self.btn_annotate.setText("❌ Exit Annotate")
            # Auto-pause video so the frame stays still
            if self.video_thread and not self.video_thread.paused:
                self.video_thread.pause()
                self.play_pause_btn.setText("Play")
            # Load model detections + stored manual annotations into widget
            self._sync_annotation_boxes()
            # Show clean pixmap; widget paintEvent draws the boxes
            if self.current_raw_frame is not None:
                from fpa_agent.threads import _bgr_to_pixmap
                clean_px = _bgr_to_pixmap(self.current_raw_frame)
                self.display_label.setPixmap(
                    clean_px.scaled(self.display_label.size(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            self.status_bar.showMessage(
                "Annotation mode ON — click=select | drag handle=resize | drag box=move "
                "| draw on empty=new box | Del=delete | Right-click=delete"
            )
        else:
            self.btn_annotate.setText("✏️ Annotate")
            # Sync manual boxes back from widget
            self._manual_annotations = [
                b for b in self.display_label._boxes
                if b.get("source") == "manual"
            ]
            # Restore detection overlay pixmap
            if self.current_frame_pixmap:
                self.display_label.setPixmap(
                    self.current_frame_pixmap.scaled(
                        self.display_label.size(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            self.status_bar.showMessage("Annotation mode OFF")

    def _sync_annotation_boxes(self):
        """Push model detections + stored manual annotations into the widget."""
        if self.current_raw_frame is None:
            return
        h, w = self.current_raw_frame.shape[:2]
        combined = []
        for det in self.current_detections:
            bbox = det.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            combined.append({
                "bbox":   (float(x1), float(y1), float(x2), float(y2)),
                "label":  det.get("label", "object"),
                "score":  float(det.get("conf", 0)) if det.get("conf") is not None else None,
                "source": "model",
            })
        for ann in self._manual_annotations:
            combined.append({
                "bbox":   tuple(float(v) for v in ann["bbox"]),
                "label":  ann["label"],
                "score":  ann.get("score"),
                "source": "manual",
            })
        self.display_label.set_boxes(combined, w, h)
        self._update_ann_toolbar()

    def on_box_drawn(self, ix1, iy1, ix2, iy2):
        """Widget drew a new box (image coords). Ask for class label then add it."""
        classes = []
        if self.model and hasattr(self.model, "classes") and self.model.classes:
            classes = list(self.model.classes)
        if not classes:
            classes = ["object"]
        label, ok = QInputDialog.getItem(
            self, "Class Label", "Select class (or type a new one):",
            classes, 0, True,
        )
        if not ok or not label.strip():
            return
        self.display_label.add_box(ix1, iy1, ix2, iy2, label.strip(), source="manual")
        self._update_ann_toolbar()

    def on_box_deleted(self, idx):
        self._update_ann_toolbar()

    def on_box_selected_changed(self, idx):
        if 0 <= idx < len(self.display_label._boxes):
            b   = self.display_label._boxes[idx]
            x1, y1, x2, y2 = (int(v) for v in b["bbox"])
            src = b.get("source", "manual")
            self.status_bar.showMessage(
                f"[{b['label']}]  ({x1},{y1})→({x2},{y2})  source={src}  "
                f"| drag handle=resize | Del=delete"
            )
        else:
            self.status_bar.showMessage(
                "Annotation: draw on empty area | click box to select | Del to delete"
            )

    def on_boxes_changed(self):
        self._update_ann_toolbar()

    def _update_ann_toolbar(self):
        boxes   = self.display_label._boxes if self._annotation_mode else []
        n_man   = sum(1 for b in boxes if b.get("source") == "manual")
        n_model = sum(1 for b in boxes if b.get("source") == "model")
        if self._annotation_mode:
            self.ann_count_label.setText(f"🟢 {n_model} model  🟠 {n_man} manual")
        else:
            n_man = len(self._manual_annotations)
            self.ann_count_label.setText(f"Boxes: {n_man}")
        has_any = (n_man + n_model) > 0 if self._annotation_mode else len(self._manual_annotations) > 0
        self.btn_save_annotation.setEnabled(has_any)
        self.btn_clear_annotation.setEnabled(n_man > 0 if self._annotation_mode else len(self._manual_annotations) > 0)

    def _display_to_image_coords(self, dx, dy):
        """Map display-label pixel position to original image pixel position."""
        if self.current_raw_frame is None:
            return dx, dy
        img_h, img_w = self.current_raw_frame.shape[:2]
        lbl_w = self.display_label.width()
        lbl_h = self.display_label.height()
        if img_w == 0 or img_h == 0 or lbl_w == 0 or lbl_h == 0:
            return dx, dy
        scale = min(lbl_w / img_w, lbl_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        ox = (lbl_w - disp_w) / 2.0
        oy = (lbl_h - disp_h) / 2.0
        ix = int((dx - ox) / scale)
        iy = int((dy - oy) / scale)
        return max(0, min(ix, img_w - 1)), max(0, min(iy, img_h - 1))

    def _redraw_with_annotations(self):
        """Non-annotation-mode: draw model boxes + manual overlays via cv2."""
        if self.current_raw_frame is None:
            return
        from fpa_agent.threads import _draw_detections, _bgr_to_pixmap
        frame = _draw_detections(self.current_raw_frame.copy(), self.current_detections)
        for ann in self._manual_annotations:
            x1, y1, x2, y2 = (int(v) for v in ann["bbox"])
            lbl = ann["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, f"[{lbl}]", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
        self.display_label.setPixmap(
            _bgr_to_pixmap(frame).scaled(
                self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def clear_manual_annotations(self):
        if self._annotation_mode:
            # Keep model boxes, remove manual ones
            self.display_label._boxes = [
                b for b in self.display_label._boxes if b.get("source") == "model"
            ]
            self.display_label._sel = -1
            self.display_label.update()
        self._manual_annotations = []
        self._update_ann_toolbar()
        # Restore detection overlay if not in annotation mode
        if not self._annotation_mode and self.current_frame_pixmap:
            self.display_label.setPixmap(
                self.current_frame_pixmap.scaled(
                    self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def get_current_image_path(self):
        if self.is_folder and self.folder_paths and 0 <= self.folder_index < len(self.folder_paths):
            return self.folder_paths[self.folder_index]
        if not self.is_video and hasattr(self, "current_image_path") and self.current_image_path:
            return self.current_image_path
        return None

    def save_manual_annotations(self):
        """Save current raw frame + all annotations as an X-AnyLabeling JSON file."""
        if self.current_raw_frame is None:
            QMessageBox.warning(self, "Annotate", "No frame loaded.")
            return

        orig_img_path = self.get_current_image_path()
        if orig_img_path:
            orig_dir = os.path.dirname(orig_img_path)
            orig_name = os.path.basename(orig_img_path)
            stem = os.path.splitext(orig_name)[0]
            default = os.path.join(orig_dir, stem + ".jpg")
        else:
            auto_dir = self.config.get("drift.auto_save_dir", "flagged_frames") or "flagged_frames"
            if not os.path.isabs(auto_dir):
                base = os.path.dirname(os.path.abspath(self.config.config_path))
                auto_dir = os.path.normpath(os.path.join(base, auto_dir))
            os.makedirs(auto_dir, exist_ok=True)
            if self.is_video and self.current_video_path and not self.current_video_path.startswith("rtsp"):
                video_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
                default = os.path.join(auto_dir, f"{video_name}_frame_{self.current_frame_index:06d}.jpg")
            else:
                default = os.path.join(auto_dir, f"annotated_{self.current_frame_index:06d}.jpg")

        img_path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame + Annotation", default, "JPEG (*.jpg)"
        )
        if not img_path:
            return
        frame = self.current_raw_frame
        h, w  = frame.shape[:2]
        cv2.imwrite(img_path, frame)

        # Read live boxes from widget (annotation mode) or fall back to stored lists
        if self._annotation_mode:
            all_boxes = list(self.display_label._boxes)
        else:
            all_boxes = (
                [{"bbox": d["bbox"], "label": d.get("label", "object"),
                  "score": float(d.get("conf", 0)), "source": "model"}
                 for d in self.current_detections if d.get("bbox")] +
                [{"bbox": a["bbox"], "label": a["label"],
                  "score": None, "source": "manual"}
                 for a in self._manual_annotations]
            )

        all_shapes = [
            _xanylabeling_shape(b["label"], *b["bbox"], b.get("score"))
            for b in all_boxes if b.get("bbox")
        ]
        payload = {
            "version": "3.3.9",
            "flags": {},
            "shapes": all_shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w),
            "description": "",
        }
        json_path = os.path.splitext(img_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(all_shapes)} annotation(s):\n{json_path}"
        )

    # ── Auto-save ─────────────────────────────────────────────────────────────
    def _auto_save_frame(self, frame, frame_index, drift_score, detections=None):
        """Save raw frame + X-AnyLabeling JSON to the configured flagged-frames directory."""
        save_dir = self.config.get("drift.auto_save_dir", "flagged_frames") or "flagged_frames"
        if not os.path.isabs(save_dir):
            base = os.path.dirname(os.path.abspath(self.config.config_path))
            save_dir = os.path.normpath(os.path.join(base, save_dir))
        os.makedirs(save_dir, exist_ok=True)

        stem = f"frame_{frame_index:06d}_drift{drift_score:.1f}"
        img_path = os.path.join(save_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, frame)
            self._write_xanylabeling_json(img_path, frame, detections or [])
            self._auto_save_count += 1
            self.auto_save_label.setText(f"Auto-saved: {self._auto_save_count}")
            self.status_bar.showMessage(
                f"Auto-saved frame {frame_index} (drift {drift_score:.1f}%) → {os.path.basename(save_dir)}/"
            )

    def _write_xanylabeling_json(self, img_path, frame, detections):
        """
        Write an X-AnyLabeling 3.3.9 compatible JSON sidecar (auto-save version).
        Uses 4-corner points, score at shape root, kie_linking, difficult fields.
        """
        h, w = frame.shape[:2]
        shapes = [
            _xanylabeling_shape(
                det.get("label", "object"),
                *_clamped_bbox(det.get("bbox", [0, 0, 0, 0]), w, h),
                float(det["conf"]) if det.get("conf") is not None else (float(det["score"]) if det.get("score") is not None else None),
            )
            for det in detections
            if det.get("bbox")
        ]
        payload = {
            "version": "3.3.9",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w),
            "description": "",
        }
        json_path = os.path.splitext(img_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def set_flagged_folder(self):
        """Let the user pick a custom folder for auto-saved flagged frames."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for Flagged Frames")
        if folder:
            self.config.set("drift.auto_save_dir", folder)
            self.config.save()
            self._auto_save_count = 0
            self.auto_save_label.setText("Auto-saved: 0")
            self.status_bar.showMessage(f"Flagged frames → {folder}")

    # ── Video seek bar ─────────────────────────────────────────────────────────
    def on_total_frames(self, total: int):
        """Enable the seek slider once the video thread reports its frame count."""
        if total > 0:
            self.slider.setMaximum(total)
            self.slider.setEnabled(True)

    def on_video_progress(self, frame_index: int):
        """Track playback position in the slider without triggering a seek."""
        self.slider.blockSignals(True)
        self.slider.setValue(frame_index)
        self.slider.blockSignals(False)

    def on_seek_requested(self, value: int):
        """User dragged the slider handle — seek the active source."""
        if self.is_folder:
            self.load_folder_image(value)
        elif self.video_thread:
            self.video_thread.seek(value)

    # ── Image folder batch ─────────────────────────────────────────────────────
    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        if not self.model:
            QMessageBox.warning(self, "Model", "Load a model first (config.json or Load Model).")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not paths:
            QMessageBox.information(self, "Folder", "No images (.jpg/.png/.bmp/.webp) found in that folder.")
            return
        self.lbl_image_path.setText(f"{os.path.basename(folder)}/ ({len(paths)} images)")
        self._stop_video_async()
        self.is_video = False
        self.is_folder = True
        self._auto_save_count = 0
        self.auto_save_label.setText("Auto-saved: 0")

        self.folder_paths = paths
        self.folder_index = 0

        self.slider.setMinimum(0)
        self.slider.setMaximum(len(paths) - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)   # seek enabled for folder mode

        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("Play")
        self.status_bar.showMessage(
            f"Loaded folder: {os.path.basename(folder)} ({len(paths)} images). Press D (Next) / A (Prev) or Play."
        )
        self.load_folder_image(0)

    def load_folder_image(self, index):
        if not self.is_folder or not self.folder_paths:
            return
        index = max(0, min(index, len(self.folder_paths) - 1))
        self.folder_index = index
        path = self.folder_paths[self.folder_index]

        # Stop previous image process if running
        if self.image_thread and self.image_thread.isRunning():
            try:
                self.image_thread.finished_signal.disconnect()
            except TypeError:
                pass
            self.image_thread.terminate()
            self.image_thread.wait(500)

        self.image_thread = ImageProcessThread(path, self.model, self.drift_scorer)
        self.image_thread.finished_signal.connect(
            lambda pixmap, detections, raw_frame, p: self.on_folder_image_processed(pixmap, detections, raw_frame, p, index)
        )
        self.image_thread.error_signal.connect(self.on_image_error)
        self.image_thread.start()

    def on_folder_image_processed(self, pixmap, detections, raw_frame, path, index):
        if index != self.folder_index:
            return
        self.update_display(pixmap, detections, index + 1, raw_frame)
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.current_frame_label.setText(f"Image: {index + 1} / {len(self.folder_paths)}")

        if self.drift_scorer and self.drift_scorer.ready:
            drift = self.drift_scorer.get_last()
            if drift:
                drift = drift.copy()
                drift["frame_index"] = index + 1
                self.on_drift_update(drift)

        self.status_bar.showMessage(f"Image {index + 1}/{len(self.folder_paths)}: {os.path.basename(path)}")

    def next_folder_image(self):
        if not self.is_folder or not self.folder_paths:
            if self.folder_timer.isActive():
                self.folder_timer.stop()
            return
        if self.image_thread and self.image_thread.isRunning():
            return
        if self.folder_index < len(self.folder_paths) - 1:
            self.load_folder_image(self.folder_index + 1)
        else:
            if self.folder_timer.isActive():
                self.folder_timer.stop()
            self.play_pause_btn.setText("Play")
            self.status_bar.showMessage("Folder complete")

    def prev_folder_image(self):
        if not self.is_folder or not self.folder_paths:
            return
        if self.image_thread and self.image_thread.isRunning():
            return
        if self.folder_index > 0:
            self.load_folder_image(self.folder_index - 1)

    def keyPressEvent(self, event):
        # Ignore keys if focus is in a text input widget
        focused = self.focusWidget()
        from PyQt5.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit
        if focused and isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            super().keyPressEvent(event)
            return

        if self.is_folder:
            if event.key() == Qt.Key_A:
                self.prev_folder_image()
                event.accept()
                return
            elif event.key() == Qt.Key_D:
                self.next_folder_image()
                event.accept()
                return
        super().keyPressEvent(event)

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Model file (.pth, .onnx, or .xml)", "",
            "Models (*.pth *.onnx *.xml *.bin);;PyTorch (*.pth);;ONNX (*.onnx);;OpenVINO IR (*.xml)",
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
        backend = self.config.get("model.backend") or ""
        openvino_device = self.config.get("model.openvino_device", "CPU") or "CPU"
        ov_emb = self.config.get("drift.openvino_embedding_path") or ""
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
            backend=backend or None,
            openvino_device=openvino_device,
            drift_openvino_embedding_path=ov_emb or None,
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
            backend_map = {".onnx": "ONNX Runtime", ".xml": "OpenVINO", ".pth": "PyTorch"}
            kind = backend_map.get(ext, "PyTorch")
            device_label = self.model_loader_thread.device
            if ext == ".xml":
                device_label = self.model_loader_thread.openvino_device
            self.status_bar.showMessage(f"{kind} model ready on {device_label}")
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
            if self.fp_frame_data.get(f, {}).get("frame_image") is not None or self.fp_frame_data.get(f, {}).get("frame_image_path") is not None
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
