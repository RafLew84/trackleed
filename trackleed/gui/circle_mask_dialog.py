from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QColor, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QDialogButtonBox, QWidget, QScrollArea
import math


class CircleImageLabel(QLabel):
    """Simple widget to pick a single circular region with click-drag."""

    def __init__(self, initial_center: QPoint | None = None, initial_radius: int = 0, parent: QWidget | None = None):
        super().__init__(parent)
        self.center = initial_center
        self.radius = initial_radius
        self._base_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 12.0
        self.setMouseTracking(True)

    def set_base_pixmap(self, pixmap: QPixmap):
        self._base_pixmap = pixmap
        self._refresh_scaled_pixmap()

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

    def _to_image_pos(self, pos: QPoint) -> QPoint:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return QPoint(0, 0)
        x = int(round(pos.x() / self._zoom))
        y = int(round(pos.y() / self._zoom))
        x = max(0, min(self._base_pixmap.width() - 1, x))
        y = max(0, min(self._base_pixmap.height() - 1, y))
        return QPoint(x, y)

    def _to_display_pos(self, pos: QPoint) -> QPoint:
        return QPoint(int(round(pos.x() * self._zoom)), int(round(pos.y() * self._zoom)))

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self._zoom = min(self._zoom_max, self._zoom * 1.15)
        else:
            self._zoom = max(self._zoom_min, self._zoom / 1.15)
        self._refresh_scaled_pixmap()
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.center and self.radius > 0:
            painter = QPainter(self)
            pen = QPen(QColor(0, 200, 255, 220))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            display_center = self._to_display_pos(self.center)
            display_radius = int(round(self.radius * self._zoom))
            painter.drawEllipse(display_center, display_radius, display_radius)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.center = self._to_image_pos(event.pos())
            self.radius = 0
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            # Clear
            self.center = None
            self.radius = 0
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton and self.center:
            current_pos = self._to_image_pos(event.pos())
            dx = current_pos.x() - self.center.x()
            dy = current_pos.y() - self.center.y()
            self.radius = int(math.hypot(dx, dy))
            self.update()

    def get_circle(self):
        if self.center and self.radius > 0:
            return self.center, self.radius
        return None, 0


class CircleMaskDialog(QDialog):
    """Dialog to pick a single analysis circle (LMB drag = set, RMB = clear)."""

    def __init__(self, pixmap: QPixmap, existing_center: QPoint | None, existing_radius: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Set Analysis Circle (LMB drag, RMB clear, Wheel: Zoom)")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)
        self.image_label = CircleImageLabel(initial_center=existing_center, initial_radius=existing_radius)
        self.image_label.set_base_pixmap(pixmap)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_circle(self):
        return self.image_label.get_circle()
