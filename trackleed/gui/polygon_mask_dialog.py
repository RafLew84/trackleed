from __future__ import annotations

from typing import Iterable

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class PolygonImageLabel(QLabel):
    """Simple polygon editor: left click adds points, right click removes the last one."""

    def __init__(
        self,
        initial_points: Iterable[tuple[float, float]] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._base_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 12.0
        self._point_radius_px = 4.0
        self.points: list[QPointF] = []
        if initial_points:
            for x, y in initial_points:
                self.points.append(QPointF(float(x), float(y)))

    def set_base_pixmap(self, pixmap: QPixmap):
        self._base_pixmap = pixmap
        self._refresh_scaled_pixmap()

    def clear_points(self):
        self.points.clear()
        self.update()

    def _refresh_scaled_pixmap(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.clear()
            return
        scaled_w = max(1, int(round(self._base_pixmap.width() * self._zoom)))
        scaled_h = max(1, int(round(self._base_pixmap.height() * self._zoom)))
        scaled = self._base_pixmap.scaled(
            scaled_w,
            scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setFixedSize(scaled.size())
        self.update()

    def _to_image_pos(self, pos) -> QPointF:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return QPointF(0.0, 0.0)
        x = float(pos.x()) / self._zoom
        y = float(pos.y()) / self._zoom
        x = max(0.0, min(float(self._base_pixmap.width() - 1), x))
        y = max(0.0, min(float(self._base_pixmap.height() - 1), y))
        return QPointF(x, y)

    def _to_display_pos(self, point: QPointF) -> QPointF:
        return QPointF(point.x() * self._zoom, point.y() * self._zoom)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self._zoom = min(self._zoom_max, self._zoom * 1.15)
        else:
            self._zoom = max(self._zoom_min, self._zoom / 1.15)
        self._refresh_scaled_pixmap()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append(self._to_image_pos(event.pos()))
            self.update()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.points:
                self.points.pop()
                self.update()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points:
            return

        painter = QPainter(self)
        polygon_pen = QPen(QColor(255, 165, 0, 230))
        polygon_pen.setWidth(2)
        painter.setPen(polygon_pen)

        display_points = [self._to_display_pos(point) for point in self.points]
        qpolygon = QPolygonF(display_points)
        if len(display_points) >= 3:
            painter.setBrush(QColor(255, 165, 0, 60))
            painter.drawPolygon(qpolygon)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(qpolygon)

        point_pen = QPen(QColor(0, 220, 255, 230))
        point_pen.setWidth(2)
        painter.setPen(point_pen)
        painter.setBrush(QColor(0, 220, 255, 180))
        radius = self._point_radius_px
        for point in display_points:
            painter.drawEllipse(point, radius, radius)

    def get_polygon_points(self) -> list[tuple[float, float]]:
        if len(self.points) < 3:
            return []
        return [(float(point.x()), float(point.y())) for point in self.points]


class PolygonMaskDialog(QDialog):
    """Dialog to define a polygon seed mask for a single spot."""

    def __init__(
        self,
        pixmap: QPixmap,
        existing_points: Iterable[tuple[float, float]] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Set Polygon Mask (Wheel: Zoom)")
        self.setMinimumSize(760, 620)

        layout = QVBoxLayout(self)

        self.image_label = PolygonImageLabel(initial_points=existing_points)
        self.image_label.set_base_pixmap(pixmap)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        controls = QHBoxLayout()
        self.reset_button = QPushButton("Reset Polygon")
        self.reset_button.clicked.connect(self.image_label.clear_points)
        controls.addWidget(self.reset_button)
        controls.addStretch()
        layout.addLayout(controls)

        hint = QLabel(
            "Left click: add polygon vertices. Right click: undo the last vertex. "
            "The polygon closes automatically once you have at least 3 points."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_polygon_points(self) -> list[tuple[float, float]]:
        return self.image_label.get_polygon_points()
