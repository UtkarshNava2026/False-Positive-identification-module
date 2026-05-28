class StyleSheetManager:
    """Manage application styling."""

    @staticmethod
    def get_stylesheet():
        return """
            QMainWindow {
                background-color: #0d1b2a;
            }

            QWidget {
                background-color: #0d1b2a;
                color: #e0e0e0;
                font-family: "Inter", "Segoe UI", "Ubuntu", sans-serif;
            }

            QGroupBox {
                color: #e0e0e0;
                border: 2px solid #1b3a52;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }

            QPushButton {
                background-color: #00a8cc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }

            QPushButton:hover {
                background-color: #00d4ff;
            }

            QPushButton:pressed {
                background-color: #008ba3;
            }

            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }

            QLineEdit {
                background-color: #1a2a3a;
                color: #e0e0e0;
                border: 2px solid #1b3a52;
                border-radius: 4px;
                padding: 6px;
                font-size: 10px;
            }

            QLineEdit:focus {
                border: 2px solid #00a8cc;
            }

            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }

            QComboBox {
                background-color: #1a2a3a;
                color: #e0e0e0;
                border: 2px solid #1b3a52;
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
            }

            QComboBox:hover {
                border: 2px solid #00a8cc;
            }

            QComboBox::drop-down {
                border: none;
                background: transparent;
            }

            QComboBox QAbstractItemView {
                background-color: #1a2a3a;
                color: #e0e0e0;
                selection-background-color: #00a8cc;
                border: 2px solid #1b3a52;
            }

            QProgressBar {
                border: 2px solid #1b3a52;
                border-radius: 5px;
                background-color: #1a2a3a;
                text-align: center;
                font-size: 9px;
            }

            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                 stop:0 #00a8cc, stop:1 #00d4ff);
                border-radius: 3px;
            }

            QSlider::groove:horizontal {
                border: 1px solid #1b3a52;
                height: 8px;
                background-color: #1a2a3a;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background-color: #00a8cc;
                border: 2px solid #00d4ff;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background-color: #00d4ff;
            }

            QStatusBar {
                background-color: #1a2a3a;
                color: #e0e0e0;
                border-top: 1px solid #1b3a52;
            }

            QListWidget {
                background-color: #0e1825;
                border: 2px solid #1b3a52;
                border-radius: 6px;
                padding: 4px;
                font-size: 11px;
            }

            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
            }

            QListWidget::item:hover {
                background-color: rgba(0, 168, 204, 0.15);
            }

            QListWidget::item:selected {
                background-color: rgba(0, 212, 255, 0.22);
                border: 1px solid rgba(0, 212, 255, 0.35);
            }

            QScrollBar:vertical {
                background: #0e1825;
                width: 10px;
                margin: 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: rgba(0, 212, 255, 0.35);
                min-height: 20px;
                border-radius: 5px;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QMessageBox QLabel {
                color: #e0e0e0;
            }

            QMessageBox {
                background-color: #0d1b2a;
            }
        """
