from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QColor, QPixmap
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QDialogButtonBox, QWidget
import math


class CircleImageLabel(QLabel):
    """Simple widget to pick a single circular region with click-drag."""

    def __init__(self, initial_center: QPoint | None = None, initial_radius: int = 0, parent: QWidget | None = None):
        super().__init__(parent)
        self.center = initial_center
        self.radius = initial_radius
        self.setMouseTracking(True)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.center and self.radius > 0:
            painter = QPainter(self)
            pen = QPen(QColor(0, 200, 255, 220))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.center, self.radius, self.radius)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.center = event.pos()
            self.radius = 0
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            # Clear
            self.center = None
            self.radius = 0
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton and self.center:
            dx = event.pos().x() - self.center.x()
            dy = event.pos().y() - self.center.y()
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
        self.setWindowTitle("Set Analysis Circle (LMB drag, RMB clear)")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)
        self.image_label = CircleImageLabel(initial_center=existing_center, initial_radius=existing_radius)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        layout.addWidget(self.image_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_circle(self):
        return self.image_label.get_circle()
