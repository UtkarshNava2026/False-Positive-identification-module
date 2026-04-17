import os
import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QLineEdit, QComboBox,
                             QMessageBox, QSlider, QGroupBox, QProgressBar,
                             QListWidget, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from .config_manager import ConfigManager
from .style_manager import StyleSheetManager
from .threads import ModelLoaderThread, VideoThread
from .export_utils import export_yolo, export_voc, export_coco


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("False Positive Identification Agent")

        self.config = ConfigManager("config.json")
        window_width = self.config.get("ui.window_width", 1100)
        window_height = self.config.get("ui.window_height", 720)
        self.setMinimumSize(window_width, window_height)

        self.model = None
        self.model_loader_thread = None
        self.current_detections = []
        self.current_frame_pixmap = None
        self.current_raw_frame = None
        self.current_frame_index = 0
        self.false_positive_frames = []
        self.is_video = False
        self.video_thread = None

        self.setStyleSheet(StyleSheetManager.get_stylesheet())
        self.init_ui()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.try_load_config_model()

    def try_load_config_model(self):
        pth_path = self.config.get("model.pth_path")
        if pth_path and os.path.exists(pth_path):
            exp_path = self.config.get("model.exp_path", "")
            classes_path = self.config.get("model.classes_path", "")
            device = self.config.get("model.device", "cpu")
            self.load_model_async(pth_path, exp_path, classes_path, device)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        self.input_group = self.build_input_group()
        self.fp_group = self.build_false_positive_group()
        self.export_group = self.build_export_group()

        left_panel.addWidget(self.input_group)
        left_panel.addWidget(self.fp_group)
        left_panel.addWidget(self.export_group)
        left_panel.addStretch(1)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        self.display_group = self.build_display_group()
        self.playback_group = self.build_playback_group()

        right_panel.addWidget(self.display_group, 1)
        right_panel.addWidget(self.playback_group)

        main_layout.addLayout(left_panel, 0)
        main_layout.addLayout(right_panel, 1)

    def build_input_group(self):
        group = QGroupBox("Input & Model")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        file_layout = QHBoxLayout()
        self.btn_open_video = QPushButton("📁 Open Video")
        self.btn_open_video.setMinimumHeight(34)
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.lbl_video_path = QLabel("No file selected")
        self.lbl_video_path.setStyleSheet("color: #a0a0a0; font-style: italic;")
        file_layout.addWidget(self.btn_open_video)
        file_layout.addWidget(self.lbl_video_path)

        rtsp_layout = QHBoxLayout()
        rtsp_label = QLabel("RTSP Stream:")
        self.rtsp_edit = QLineEdit()
        self.rtsp_edit.setPlaceholderText("rtsp://username:password@ip:port/stream")
        self.btn_load_rtsp = QPushButton("🎬 Load RTSP")
        self.btn_load_rtsp.setMinimumHeight(34)
        self.btn_load_rtsp.clicked.connect(self.load_rtsp)
        rtsp_layout.addWidget(rtsp_label)
        rtsp_layout.addWidget(self.rtsp_edit)
        rtsp_layout.addWidget(self.btn_load_rtsp)

        image_layout = QHBoxLayout()
        self.btn_open_image = QPushButton("🖼️ Open Image")
        self.btn_open_image.setMinimumHeight(34)
        self.btn_open_image.clicked.connect(self.open_image_file)
        self.lbl_image_path = QLabel("No image selected")
        self.lbl_image_path.setStyleSheet("color: #a0a0a0; font-style: italic;")
        image_layout.addWidget(self.btn_open_image)
        image_layout.addWidget(self.lbl_image_path)

        model_layout = QHBoxLayout()
        self.btn_load_model = QPushButton("🤖 Load Model")
        self.btn_load_model.setMinimumHeight(34)
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.lbl_model_status = QLabel("Model not loaded")
        self.lbl_model_status.setStyleSheet("color: #ff6b6b;")
        self.model_progress = QProgressBar()
        self.model_progress.setVisible(False)
        self.model_progress.setMaximum(0)
        self.model_progress.setMinimumHeight(20)
        model_layout.addWidget(self.btn_load_model)
        model_layout.addWidget(self.lbl_model_status, 1)
        model_layout.addWidget(self.model_progress, 1)

        layout.addLayout(file_layout)
        layout.addLayout(rtsp_layout)
        layout.addLayout(image_layout)
        layout.addLayout(model_layout)
        group.setLayout(layout)
        return group

    def build_display_group(self):
        group = QGroupBox("Video / Image Display")
        layout = QVBoxLayout()
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet(
            "border: 2px solid #1b3a52; background-color: #0e1825; border-radius: 6px;"
        )
        self.display_label.setMinimumSize(640, 360)
        layout.addWidget(self.display_label)
        group.setLayout(layout)
        return group

    def build_playback_group(self):
        group = QGroupBox("Playback & Actions")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        control_layout = QHBoxLayout()
        self.play_pause_btn = QPushButton("▶️ Play")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setMinimumHeight(34)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek_frame)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.slider)

        fp_layout = QHBoxLayout()
        self.current_frame_label = QLabel("Frame: 0")
        self.current_frame_label.setMinimumWidth(120)
        self.btn_flag_fp = QPushButton("🚩 Flag Frame")
        self.btn_flag_fp.setMinimumHeight(34)
        self.btn_flag_fp.setEnabled(False)
        self.btn_flag_fp.clicked.connect(self.flag_current_frame)
        self.fp_frame_input = QLineEdit()
        self.fp_frame_input.setPlaceholderText("Frame #")
        self.fp_frame_input.setMaximumWidth(100)
        self.btn_add_frame = QPushButton("➕ Add")
        self.btn_add_frame.setMinimumHeight(34)
        self.btn_add_frame.clicked.connect(self.add_manual_frame)
        fp_layout.addWidget(self.current_frame_label)
        fp_layout.addWidget(self.btn_flag_fp)
        fp_layout.addWidget(self.fp_frame_input)
        fp_layout.addWidget(self.btn_add_frame)

        self.fp_list = QListWidget()
        self.fp_list.setMinimumHeight(120)

        layout.addLayout(control_layout)
        layout.addLayout(fp_layout)
        layout.addWidget(self.fp_list)
        group.setLayout(layout)
        return group

    def build_false_positive_group(self):
        group = QGroupBox("False Positive Review")
        layout = QVBoxLayout()
        self.fp_review_label = QLabel("Flag frames with questionable detections.")
        self.fp_review_label.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(self.fp_review_label)
        group.setLayout(layout)
        return group

    def build_export_group(self):
        group = QGroupBox("Export")
        layout = QHBoxLayout()
        export_label = QLabel("Format:")
        self.export_format = QComboBox()
        self.export_format.addItems(["YOLO", "VOC", "COCO"])
        self.export_format.setMinimumWidth(100)
        self.btn_export = QPushButton("💾 Export Frame")
        self.btn_export.setMinimumHeight(34)
        self.btn_export.clicked.connect(self.export_frame)
        layout.addWidget(export_label)
        layout.addWidget(self.export_format)
        layout.addStretch()
        layout.addWidget(self.btn_export)
        group.setLayout(layout)
        return group

    def open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                               "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.lbl_video_path.setText(path)
            self.start_video_source(path)

    def load_rtsp(self):
        url = self.rtsp_edit.text().strip()
        if url:
            self.start_video_source(url)

    def open_image_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.lbl_image_path.setText(path)
            self.start_image_source(path)

    def start_video_source(self, source):
        if self.video_thread:
            self.video_thread.stop()
        self.is_video = True
        self.video_thread = VideoThread(source, self.model)
        self.video_thread.change_pixmap_signal.connect(self.update_display)
        self.video_thread.finished_signal.connect(self.on_video_finished)
        self.video_thread.start()
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸️ Pause")
        self.slider.setEnabled(False)
        self.status_bar.showMessage(f"▶️ Playing: {source}")

    def start_image_source(self, path):
        if self.video_thread:
            self.video_thread.stop()
        self.is_video = False
        self.current_frame_index = 0
        self.current_frame_label.setText("Frame: N/A")
        self.btn_flag_fp.setEnabled(False)
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Error", "Cannot read image.")
            return
        self.current_raw_frame = frame.copy()
        detections = self.model.predict(frame) if self.model else []
        disp_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['conf']:.2f}"
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rgb_image = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.update_display(pixmap, detections, 0, frame)
        self.play_pause_btn.setEnabled(False)
        self.status_bar.showMessage(f"🖼️ Image loaded: {path}")

    def update_display(self, pixmap, detections, frame_index=0, raw_frame=None):
        self.display_label.setPixmap(pixmap.scaled(self.display_label.size(),
                                                   Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))
        self.current_detections = detections
        self.current_frame_pixmap = pixmap
        self.current_raw_frame = raw_frame
        self.current_frame_index = frame_index
        self.current_frame_label.setText(f"Frame: {frame_index}")
        self.btn_flag_fp.setEnabled(self.is_video and frame_index > 0)

    def flag_current_frame(self):
        if self.current_frame_index <= 0:
            return
        if self.current_frame_index not in self.false_positive_frames:
            self.false_positive_frames.append(self.current_frame_index)
            self.false_positive_frames.sort()
            self.update_fp_list()
            self.status_bar.showMessage(f"🚩 Flagged frame {self.current_frame_index} as false positive")

    def add_manual_frame(self):
        text = self.fp_frame_input.text().strip()
        if not text.isdigit():
            QMessageBox.warning(self, "Invalid frame", "Enter a valid frame number.")
            return
        frame_number = int(text)
        if frame_number <= 0:
            QMessageBox.warning(self, "Invalid frame", "Frame number must be greater than zero.")
            return
        if frame_number not in self.false_positive_frames:
            self.false_positive_frames.append(frame_number)
            self.false_positive_frames.sort()
            self.update_fp_list()
            self.status_bar.showMessage(f"🚩 Added frame {frame_number} as false positive")
        self.fp_frame_input.clear()

    def update_fp_list(self):
        self.fp_list.clear()
        for frame in self.false_positive_frames:
            self.fp_list.addItem(f"Frame {frame}")

    def on_video_finished(self):
        self.status_bar.showMessage("⏹️ Video finished.")
        self.play_pause_btn.setText("▶️ Play")
        self.play_pause_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait()
        event.accept()

    def toggle_play_pause(self):
        if not self.is_video or not self.video_thread:
            return
        if self.video_thread.paused:
            self.video_thread.resume()
            self.play_pause_btn.setText("⏸️ Pause")
            self.status_bar.showMessage("▶️ Resumed")
        else:
            self.video_thread.pause()
            self.play_pause_btn.setText("▶️ Play")
            self.status_bar.showMessage("⏸️ Paused")

    def seek_frame(self, pos):
        pass

    def load_model_dialog(self):
        pth_path, _ = QFileDialog.getOpenFileName(self, "Select .pth model file", "", "PyTorch Model (*.pth)")
        if not pth_path:
            return
        exp_path, _ = QFileDialog.getOpenFileName(self, "Select experiment file", "", "All Files (*)")
        classes_path, _ = QFileDialog.getOpenFileName(self, "Select class names file", "", "Text Files (*.txt)")
        self.load_model_async(pth_path, exp_path, classes_path)

    def load_model_async(self, pth_path, exp_path, classes_path, device="cpu"):
        if self.model_loader_thread and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait()

        self.model_loader_thread = ModelLoaderThread(pth_path, exp_path, classes_path, device)
        self.model_loader_thread.model_loaded_signal.connect(self.on_model_loaded)
        self.model_loader_thread.progress_signal.connect(self.on_model_loading_progress)
        self.model_progress.setVisible(True)
        self.btn_load_model.setEnabled(False)
        self.lbl_model_status.setText("Loading model...")
        self.model_loader_thread.start()

    def on_model_loading_progress(self, message):
        self.lbl_model_status.setText(message)

    def on_model_loaded(self, model, success, message):
        self.model_progress.setVisible(False)
        self.btn_load_model.setEnabled(True)
        if success:
            self.model = model
            self.lbl_model_status.setText(f"✓ Model loaded: {os.path.basename(self.model_loader_thread.pth_path)}")
            self.lbl_model_status.setStyleSheet("color: #4ade80;")
            QMessageBox.information(self, "Success", message)
            self.config.set("model.pth_path", self.model_loader_thread.pth_path)
            self.config.set("model.exp_path", self.model_loader_thread.exp_path)
            self.config.set("model.classes_path", self.model_loader_thread.classes_path)
            self.config.save()
        else:
            self.model = None
            self.lbl_model_status.setText("✗ Model: Not loaded")
            self.lbl_model_status.setStyleSheet("color: #ff6b6b;")
            QMessageBox.critical(self, "Error", message)

    def export_frame(self):
        if self.current_raw_frame is None:
            QMessageBox.warning(self, "No frame", "No frame to export. Load a video/image first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select export directory")
        if not out_dir:
            return
        temp_img_path = os.path.join(out_dir, "exported_frame.jpg")
        cv2.imwrite(temp_img_path, self.current_raw_frame)
        fmt = self.export_format.currentText().lower()
        if fmt == "yolo":
            export_yolo(temp_img_path, self.current_detections,
                        self.model.classes if self.model else ['object'], out_dir)
        elif fmt == "voc":
            export_voc(temp_img_path, self.current_detections,
                       self.model.classes if self.model else ['object'], out_dir)
        elif fmt == "coco":
            export_coco(temp_img_path, self.current_detections,
                        self.model.classes if self.model else ['object'], out_dir)
        QMessageBox.information(self, "Export complete",
                                f"Frame and annotation saved in:\n{out_dir}")
