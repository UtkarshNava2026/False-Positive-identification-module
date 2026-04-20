import cv2
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

    def __init__(self, source, model):
        super().__init__()
        self.source = source
        self.model = model
        self.paused = False
        self.stop_flag = False
        self.current_frame = None
        self.frame_index = 0
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Failed to open source: {self.source}")
            self.finished_signal.emit()
            return

        # Reset tracker when starting new video
        if self.model and hasattr(self.model, 'reset_tracker'):
            self.model.reset_tracker()

        while not self.stop_flag:
            if not self.paused:
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
                    
                    # Include track ID if available
                    if 'track_id' in det and det['track_id'] is not None:
                        display_label = f"ID:{det['track_id']} {label} {conf:.2f}"
                        # Different colors based on anomaly status
                        color = (0, 255, 0)  # Default green
                    else:
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

            self.msleep(30)

        self.cap.release()
        self.finished_signal.emit()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stop_flag = True
        self.wait()
