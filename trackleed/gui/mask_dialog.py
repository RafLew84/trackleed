from __future__ import annotations

# stal/gui/mask_dialog.py

from PyQt6.QtCore import QRect, QSize, Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QPixmap, QPainter, QColor, QBrush, QPen, QWheelEvent
from PyQt6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QDialogButtonBox,
                             QRubberBand, QWidget, QScrollArea)

class InteractiveImageLabel(QLabel):
    """
    A custom QLabel that allows selecting and deleting multiple rectangular regions.
    - Left-click and drag to add a new mask region.
    - Right-click on a region to delete it.
    """
    def __init__(self, initial_masks: list[QRect] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.mask_list = [QRect(r) for r in (initial_masks or [])]
        self.origin_point: QPoint | None = None
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._base_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 12.0
        
        # Set a semi-transparent red color for displaying existing masks
        self.mask_brush = QBrush(QColor(255, 0, 0, 100)) # RGBA
        self.mask_pen = QPen(QColor(255, 0, 0, 180))
        self.mask_pen.setWidth(2)

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

    def _scaled_rect(self, rect: QRect) -> QRect:
        x = int(round(rect.x() * self._zoom))
        y = int(round(rect.y() * self._zoom))
        w = max(1, int(round(rect.width() * self._zoom)))
        h = max(1, int(round(rect.height() * self._zoom)))
        return QRect(x, y, w, h)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self._zoom = min(self._zoom_max, self._zoom * 1.15)
        else:
            self._zoom = max(self._zoom_min, self._zoom / 1.15)
        self._refresh_scaled_pixmap()
        event.accept()

    def paintEvent(self, event):
        """Overrides paintEvent to draw existing masks on top of the image."""
        super().paintEvent(event) # First, draw the label and its pixmap
        painter = QPainter(self)
        painter.setPen(self.mask_pen)
        painter.setBrush(self.mask_brush)
        for rect in self.mask_list:
            painter.drawRect(self._scaled_rect(rect))

    def mousePressEvent(self, event: QMouseEvent):
        """Handles adding a new mask (left-click) or deleting one (right-click)."""
        # Right-click to delete
        if event.button() == Qt.MouseButton.RightButton:
            image_pos = self._to_image_pos(event.pos())
            for rect in reversed(self.mask_list): # Iterate backwards to find the top-most rect first
                if rect.contains(image_pos):
                    self.mask_list.remove(rect)
                    self.update() # Trigger a repaint to show the deletion
                    return
        
        # Left-click to start drawing
        elif event.button() == Qt.MouseButton.LeftButton:
            self.origin_point = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin_point, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Updates the rubber band geometry as the mouse is dragged."""
        if self.origin_point is not None:
            self.rubber_band.setGeometry(QRect(self.origin_point, event.pos()).normalized())

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Finalizes adding a new mask region."""
        if event.button() == Qt.MouseButton.LeftButton and self.origin_point is not None:
            self.rubber_band.hide()
            new_rect = self.rubber_band.geometry()
            if new_rect.width() > 0 and new_rect.height() > 0:
                image_x = int(round(new_rect.x() / self._zoom))
                image_y = int(round(new_rect.y() / self._zoom))
                image_w = max(1, int(round(new_rect.width() / self._zoom)))
                image_h = max(1, int(round(new_rect.height() / self._zoom)))
                self.mask_list.append(QRect(image_x, image_y, image_w, image_h))
                self.update() # Trigger a repaint to show the new rect
            self.origin_point = None

    def get_selected_rects(self) -> list[QRect]:
        """Returns the final list of mask rectangles."""
        return self.mask_list


class MaskDialog(QDialog):
    """ The dialog window for mask editing. """
    def __init__(self, pixmap: QPixmap, existing_masks: list[QRect], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Metadata Mask (LMB: Add, RMB: Delete, Wheel: Zoom)")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        self.image_label = InteractiveImageLabel(initial_masks=existing_masks)
        self.image_label.set_base_pixmap(pixmap)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_selected_rects(self) -> list[QRect]:
        """Gets the result from the dialog."""
        return self.image_label.get_selected_rects()
