class StyleSheetManager:
    @staticmethod
    def get_stylesheet():
        return """
            QMainWindow, QWidget {
                background-color: #0a1628;
                color: #e8eef5;
                font-family: "Segoe UI", "Ubuntu", sans-serif;
                font-size: 12px;
            }

            QGroupBox {
                border: 1px solid #243b55;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #7dd3fc;
            }

            QPushButton {
                background-color: #0e7490;
                color: #fff;
                border: none;
                padding: 10px 16px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #06b6d4; }
            QPushButton:pressed { background-color: #155e75; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }

            QLineEdit, QSpinBox, QComboBox {
                background-color: #132337;
                border: 1px solid #243b55;
                border-radius: 5px;
                padding: 6px 8px;
                color: #e8eef5;
            }
            QLineEdit:focus, QSpinBox:focus { border-color: #06b6d4; }

            QLabel#hintLabel, QLabel#pathLabel {
                color: #94a3b8;
                font-size: 11px;
            }
            QLabel#modelStatusBad { color: #f87171; }
            QLabel#modelStatusOk { color: #4ade80; font-weight: 600; }

            QLabel#videoDisplay {
                border: 2px solid #1e3a5f;
                background-color: #060d18;
                border-radius: 10px;
                color: #64748b;
                font-size: 14px;
            }

            QFrame#driftGauge {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0f2847, stop:1 #0a1628);
                border: 2px solid #1e4976;
                border-radius: 14px;
            }
            QLabel#driftScore {
                font-size: 48px;
                font-weight: 800;
                color: #f0f9ff;
            }
            QLabel#driftTitle {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 2px;
                color: #7dd3fc;
            }
            QLabel#driftDetail {
                font-size: 10px;
                color: #94a3b8;
            }
            QLabel#driftFrame {
                font-size: 11px;
                color: #64748b;
            }

            QScrollArea#leftScroll {
                border: none;
                background: transparent;
            }

            QListWidget {
                background-color: #060d18;
                border: 1px solid #243b55;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background-color: rgba(6, 182, 212, 0.25);
            }

            QProgressBar {
                border: 1px solid #243b55;
                border-radius: 4px;
                background: #132337;
            }
            QProgressBar::chunk {
                background: #06b6d4;
                border-radius: 3px;
            }

            QStatusBar {
                background: #0f172a;
                border-top: 1px solid #243b55;
            }
        """
