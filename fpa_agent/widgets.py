"""Reusable PyQt widgets for the FPA application."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget


class DriftGaugeWidget(QFrame):
    """Large left-panel drift indicator updated every frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("driftGauge")
        self.setMinimumWidth(200)
        self.setMinimumHeight(220)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 20, 16, 20)

        self.icon_label = QLabel("◎")
        self.icon_label.setObjectName("driftIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.score_label = QLabel("—")
        self.score_label.setObjectName("driftScore")
        self.score_label.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel("DATA DRIFT")
        self.title_label.setObjectName("driftTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.detail_label = QLabel("Load reference embeddings")
        self.detail_label.setObjectName("driftDetail")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setWordWrap(True)

        self.frame_label = QLabel("Frame: —")
        self.frame_label.setObjectName("driftFrame")
        self.frame_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)
        layout.addStretch(1)
        layout.addWidget(self.frame_label)

        self.set_level("idle")

    def set_level(self, level: str):
        """level: idle | calibrating | low | medium | high"""
        colors = {
            "idle": ("#6b7c93", "◎"),
            "calibrating": ("#00d4ff", "◌"),
            "low": ("#4ade80", "●"),
            "medium": ("#fbbf24", "●"),
            "high": ("#f87171", "●"),
        }
        color, icon = colors.get(level, colors["idle"])
        self.icon_label.setText(icon)
        self.icon_label.setStyleSheet(f"color: {color}; font-size: 56px; font-weight: bold;")

    def update_drift(self, drift: dict):
        ready = bool(drift.get("ready", False))
        frame_index = drift.get("frame_index", 0)
        self.frame_label.setText(f"Frame: {frame_index}")

        if not ready:
            self.set_level("calibrating" if drift.get("loading") else "idle")
            self.score_label.setText("…")
            self.detail_label.setText(drift.get("message", "Waiting for reference embeddings"))
            return

        score = float(drift.get("drift_score", 0.0))
        cos_c = float(drift.get("cosine_centroid", 0.0))
        knn = float(drift.get("knn_mean_sim", 0.0))
        ref_n = drift.get("reference_count", 0)
        mismatch = bool(drift.get("bank_mismatch", False))

        if mismatch:
            self.score_label.setText("!")
            self.set_level("high")
            enc = drift.get("encoder", "")
            self.detail_label.setText(
                f"Reference bank mismatch\n"
                f"(cos={cos_c:.3f})\n\n"
                f"Use yolox_standard @ 640\n"
                f"and sakku-gate.pth\n"
                f"(neck concat pipeline)"
            )
            return

        self.score_label.setText(f"{score:.1f}")

        if score < 25:
            self.set_level("low")
        elif score < 55:
            self.set_level("medium")
        else:
            self.set_level("high")

        enc = drift.get("encoder", "")
        enc_line = f"{enc}\n" if enc else ""
        self.detail_label.setText(
            f"{enc_line}"
            f"cos(ref): {cos_c:.3f}\n"
            f"kNN sim: {knn:.3f}\n"
            f"ref bank: {ref_n:,}"
        )
