"""Reusable PyQt widgets for the FPA application."""

from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QFont, QPainter, QPen, QBrush, QColor,
)
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget


# Half-size (px) of corner/edge drag handles in display space
_H = 7


class AnnotationLabel(QLabel):
    """
    QLabel that doubles as a full interactive bounding-box editor.

    Modes (controlled by set_annotation_mode()):
    ─────────────────────────────────────────────
    OFF  Normal display label.
    ON   Interactive annotation mode:
           • Drag on empty area  → draw a new box  (box_drawn signal)
           • Click a box         → select it        (box_selected signal)
           • Drag selected box   → move it
           • Drag corner handle  → resize corner
           • Drag edge  handle   → resize one side
           • Del / Backspace     → delete selected
           • Right-click a box   → delete it immediately

    All box coordinates are stored in IMAGE pixel space.
    The widget handles display ↔ image conversion automatically using the
    same KeepAspectRatio centred-pixmap geometry Qt uses.

    Signals
    ───────
    box_drawn(x1, y1, x2, y2)   – image-space new box; caller should ask for
                                   label then call add_box().
    box_deleted(idx)             – a box was deleted.
    box_selected(idx)            – selected index changed (-1 = none).
    boxes_changed()              – any structural change (add/move/resize/delete).
    """

    box_drawn    = pyqtSignal(int, int, int, int)
    box_deleted  = pyqtSignal(int)
    box_selected = pyqtSignal(int)
    boxes_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotation_mode = False

        # All editable boxes: list of dicts
        #   {"bbox": (x1,y1,x2,y2), "label": str, "score": float|None, "source": "model"|"manual"}
        self._boxes: list = []
        self._img_w: int = 1
        self._img_h: int = 1

        self._sel: int = -1           # selected box index
        self._hov: int = -1           # hovered box index

        # Drag / draw state
        self._mode   = None           # None | "draw" | "move" | handle str
        self._ds     = None           # drag start in display coords (dx, dy)
        self._orig   = None           # original bbox at drag start
        self._dcurr  = None           # current pos while drawing new box

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ─────────────────────── Public API ────────────────────────────────────────

    def set_annotation_mode(self, enabled: bool):
        self.annotation_mode = enabled
        self._sel  = -1
        self._mode = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def set_boxes(self, boxes, img_w: int, img_h: int):
        """Replace full box list. boxes = list of dicts (bbox, label, score, source)."""
        self._boxes = [dict(b) for b in boxes]
        self._img_w = max(1, int(img_w))
        self._img_h = max(1, int(img_h))
        self._sel  = -1
        self._mode = None
        self.update()

    def add_box(self, x1, y1, x2, y2, label="", source="manual", score=None):
        """Add a box (image-space coords) and auto-select it."""
        self._boxes.append({
            "bbox":   (float(x1), float(y1), float(x2), float(y2)),
            "label":  label,
            "score":  score,
            "source": source,
        })
        self._sel = len(self._boxes) - 1
        self.update()
        self.boxes_changed.emit()

    # ─────────────────────── Coordinate transform ───────────────────────────────

    def _soo(self):
        """Return (scale, offset_x, offset_y) for the centred KeepAspectRatio pixmap."""
        s  = min(self.width() / self._img_w, self.height() / self._img_h)
        ox = (self.width()  - self._img_w * s) / 2.0
        oy = (self.height() - self._img_h * s) / 2.0
        return s, ox, oy

    def _d(self, ix, iy):
        """Image → display coords."""
        s, ox, oy = self._soo()
        return ox + ix * s, oy + iy * s

    def _i(self, dx, dy):
        """Display → image coords (clamped)."""
        s, ox, oy = self._soo()
        if s == 0:
            return 0.0, 0.0
        return (
            max(0.0, min((dx - ox) / s, float(self._img_w))),
            max(0.0, min((dy - oy) / s, float(self._img_h))),
        )

    def _drect(self, idx):
        """Return (bx1, by1, bx2, by2) in display coords for box[idx]."""
        x1, y1, x2, y2 = self._boxes[idx]["bbox"]
        dx1, dy1 = self._d(x1, y1)
        dx2, dy2 = self._d(x2, y2)
        return min(dx1,dx2), min(dy1,dy2), max(dx1,dx2), max(dy1,dy2)

    # ─────────────────────── Hit testing ────────────────────────────────────────

    def _hit(self, dx, dy):
        """Return (box_idx, handle_str) or (-1, None).
        Handles: "tl","tr","bl","br","t","b","l","r"  →  resize
                 "move"                                →  translate
        """
        # Handles on the selected box have first priority
        if 0 <= self._sel < len(self._boxes):
            bx1, by1, bx2, by2 = self._drect(self._sel)
            mx, my = (bx1+bx2)/2, (by1+by2)/2
            for tag, hx, hy in [
                ("tl", bx1, by1), ("tr", bx2, by1),
                ("bl", bx1, by2), ("br", bx2, by2),
                ("t",  mx,  by1), ("b",  mx,  by2),
                ("l",  bx1, my ), ("r",  bx2, my ),
            ]:
                if abs(dx - hx) <= _H and abs(dy - hy) <= _H:
                    return self._sel, tag

        # Interior check — selected box first, then others
        order = ([self._sel] +
                 [i for i in range(len(self._boxes)) if i != self._sel])
        for idx in order:
            if idx < 0 or idx >= len(self._boxes):
                continue
            bx1, by1, bx2, by2 = self._drect(idx)
            if bx1 <= dx <= bx2 and by1 <= dy <= by2:
                return idx, "move"

        return -1, None

    # ─────────────────────── Mouse events ───────────────────────────────────────

    def mousePressEvent(self, event):
        if not self.annotation_mode:
            super().mousePressEvent(event)
            return
        self.setFocus()
        dx, dy = float(event.x()), float(event.y())

        if event.button() == Qt.LeftButton:
            idx, handle = self._hit(dx, dy)
            if idx >= 0:
                prev = self._sel
                self._sel   = idx
                self._mode  = handle
                self._ds    = (dx, dy)
                self._orig  = self._boxes[idx]["bbox"]
                if prev != idx:
                    self.box_selected.emit(idx)
            else:
                # Start drawing a new box
                self._sel   = -1
                self._mode  = "draw"
                self._ds    = (dx, dy)
                self._dcurr = (dx, dy)
                self.box_selected.emit(-1)
            self.update()

        elif event.button() == Qt.RightButton:
            idx, _ = self._hit(dx, dy)
            if idx >= 0:
                self._sel = idx
                self._delete_selected()

    def mouseMoveEvent(self, event):
        if not self.annotation_mode:
            super().mouseMoveEvent(event)
            return
        dx, dy = float(event.x()), float(event.y())

        # ── Active draw ──────────────────────────────────────
        if self._mode == "draw":
            self._dcurr = (dx, dy)
            self.update()
            return

        # ── Active drag (move / resize) ──────────────────────
        if self._mode in ("move","tl","tr","bl","br","t","b","l","r"):
            ox1, oy1, ox2, oy2 = self._orig
            ix, iy   = self._i(dx, dy)
            isx, isy = self._i(*self._ds)
            ddx, ddy = ix - isx, iy - isy
            W, H     = float(self._img_w), float(self._img_h)

            if self._mode == "move":
                bw, bh = ox2 - ox1, oy2 - oy1
                nx1 = max(0.0, min(ox1 + ddx, W - bw))
                ny1 = max(0.0, min(oy1 + ddy, H - bh))
                nx2, ny2 = nx1 + bw, ny1 + bh
            else:
                m = self._mode
                nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2
                cx, cy = ix, iy
                if "l" in m: nx1 = max(0.0,  min(cx, ox2 - 2))
                if "r" in m: nx2 = min(W,    max(cx, ox1 + 2))
                if "t" in m: ny1 = max(0.0,  min(cy, oy2 - 2))
                if "b" in m: ny2 = min(H,    max(cy, oy1 + 2))

            self._boxes[self._sel]["bbox"] = (nx1, ny1, nx2, ny2)
            self.update()
            return

        # ── Hover + cursor update ────────────────────────────
        idx, handle = self._hit(dx, dy)
        if idx != self._hov:
            self._hov = idx
            self.update()

        _cur = {
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            "t":  Qt.SizeVerCursor,   "b":  Qt.SizeVerCursor,
            "l":  Qt.SizeHorCursor,   "r":  Qt.SizeHorCursor,
            "move": Qt.SizeAllCursor,
        }
        self.setCursor(_cur.get(handle, Qt.CrossCursor))

    def mouseReleaseEvent(self, event):
        if not self.annotation_mode:
            super().mouseReleaseEvent(event)
            return

        if event.button() != Qt.LeftButton:
            return

        if self._mode == "draw":
            dx, dy = float(event.x()), float(event.y())
            sx, sy = self._ds
            x1d, y1d = min(sx, dx), min(sy, dy)
            x2d, y2d = max(sx, dx), max(sy, dy)
            ix1, iy1 = self._i(x1d, y1d)
            ix2, iy2 = self._i(x2d, y2d)
            self._mode  = None
            self._ds    = None
            self._dcurr = None
            self.update()
            if (ix2 - ix1) > 4 and (iy2 - iy1) > 4:
                self.box_drawn.emit(int(ix1), int(iy1), int(ix2), int(iy2))

        elif self._mode is not None:
            self._mode = None
            self._ds   = None
            self._orig = None
            self.boxes_changed.emit()
            self.update()

    # ─────────────────────── Keyboard ────────────────────────────────────────────

    def keyPressEvent(self, event):
        if self.annotation_mode and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
        else:
            super().keyPressEvent(event)

    def _delete_selected(self):
        if 0 <= self._sel < len(self._boxes):
            idx = self._sel
            del self._boxes[idx]
            self._sel = -1
            self.box_selected.emit(-1)
            self.box_deleted.emit(idx)
            self.boxes_changed.emit()
            self.update()

    # ─────────────────────── Painting ────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)          # draw the background pixmap

        if not self.annotation_mode:
            return
        if not self._boxes and self._mode != "draw":
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", 9, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()

        # ── Rubber-band for new box being drawn ──────────────
        if self._mode == "draw" and self._ds and self._dcurr:
            sx, sy = self._ds
            cx, cy = self._dcurr
            x1d = min(sx, cx); y1d = min(sy, cy)
            painter.setPen(QPen(QColor(255, 255, 50), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x1d, y1d, abs(cx-sx), abs(cy-sy)))

        # ── Each stored box ──────────────────────────────────
        for idx in range(len(self._boxes)):
            info     = self._boxes[idx]
            bx1, by1, bx2, by2 = self._drect(idx)
            is_sel   = (idx == self._sel)
            is_hov   = (idx == self._hov) and not is_sel
            src      = info.get("source", "manual")

            # Colour scheme
            if is_sel:
                col = QColor(0, 180, 255)        # blue — selected
            elif is_hov:
                col = QColor(255, 220, 50)        # yellow — hover
            elif src == "model":
                col = QColor(80, 255, 100)        # green — model detection
            else:
                col = QColor(255, 130, 50)        # orange — manual

            # Box outline
            painter.setPen(QPen(col, 3 if is_sel else 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(bx1, by1, bx2-bx1, by2-by1))

            # Label chip
            txt = info["label"]
            sc  = info.get("score")
            if sc is not None:
                txt += f"  {sc:.2f}"
            tw = fm.boundingRect(txt).width() + 6
            th = fm.height() + 4
            lx = bx1
            ly = by1 - th if by1 > th + 2 else by2
            painter.fillRect(QRectF(lx, ly, tw, th), QColor(0, 0, 0, 170))
            painter.setPen(QPen(col))
            painter.drawText(QPointF(lx + 3, ly + th - 4), txt)

            # Drag handles for the selected box only
            if is_sel:
                mx, my = (bx1+bx2)/2, (by1+by2)/2
                # Corners (filled squares)
                painter.setBrush(QBrush(col))
                painter.setPen(QPen(QColor(0,0,0,200), 1))
                for hx, hy in [(bx1,by1),(bx2,by1),(bx1,by2),(bx2,by2)]:
                    painter.drawRect(QRectF(hx-_H, hy-_H, _H*2, _H*2))
                # Edge midpoints (circles)
                painter.setBrush(QBrush(col.lighter(140)))
                r = _H * 0.8
                for hx, hy in [(mx,by1),(mx,by2),(bx1,my),(bx2,my)]:
                    painter.drawEllipse(QPointF(hx, hy), r, r)

        painter.end()


# ─────────────────────────────────────────────────────────────────────────────
# DriftGaugeWidget  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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
            "idle":        ("#6b7c93", "◎"),
            "calibrating": ("#00d4ff", "◌"),
            "low":         ("#4ade80", "●"),
            "medium":      ("#fbbf24", "●"),
            "high":        ("#f87171", "●"),
        }
        color, icon = colors.get(level, colors["idle"])
        self.icon_label.setText(icon)
        self.icon_label.setStyleSheet(
            f"color: {color}; font-size: 56px; font-weight: bold;"
        )

    def update_drift(self, drift: dict):
        ready       = bool(drift.get("ready", False))
        frame_index = drift.get("frame_index", 0)
        self.frame_label.setText(f"Frame: {frame_index}")

        if not ready:
            self.set_level("calibrating" if drift.get("loading") else "idle")
            self.score_label.setText("…")
            self.detail_label.setText(
                drift.get("message", "Waiting for reference embeddings")
            )
            return

        score  = float(drift.get("drift_score",    0.0))
        cos_c  = float(drift.get("cosine_centroid", 0.0))
        knn    = float(drift.get("knn_mean_sim",    0.0))
        ref_n  = drift.get("reference_count", 0)
        mismatch = bool(drift.get("bank_mismatch", False))

        if mismatch:
            self.score_label.setText("!")
            self.set_level("high")
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

        enc      = drift.get("encoder", "")
        enc_line = f"{enc}\n" if enc else ""
        self.detail_label.setText(
            f"{enc_line}"
            f"cos(ref): {cos_c:.3f}\n"
            f"kNN sim: {knn:.3f}\n"
            f"ref bank: {ref_n:,}"
        )
