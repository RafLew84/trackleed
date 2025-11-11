# stal/gui/mask_dialog.py

from PyQt6.QtCore import QRect, QSize, Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QPixmap, QPainter, QColor, QBrush, QPen
from PyQt6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QDialogButtonBox,
                             QRubberBand, QWidget)

class InteractiveImageLabel(QLabel):
    """
    A custom QLabel that allows selecting and deleting multiple rectangular regions.
    - Left-click and drag to add a new mask region.
    - Right-click on a region to delete it.
    """
    def __init__(self, initial_masks: list[QRect] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.mask_list = initial_masks or []
        self.origin_point: QPoint | None = None
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        
        # Set a semi-transparent red color for displaying existing masks
        self.mask_brush = QBrush(QColor(255, 0, 0, 100)) # RGBA
        self.mask_pen = QPen(QColor(255, 0, 0, 180))
        self.mask_pen.setWidth(2)

    def paintEvent(self, event):
        """Overrides paintEvent to draw existing masks on top of the image."""
        super().paintEvent(event) # First, draw the label and its pixmap
        painter = QPainter(self)
        painter.setPen(self.mask_pen)
        painter.setBrush(self.mask_brush)
        for rect in self.mask_list:
            painter.drawRect(rect)

    def mousePressEvent(self, event: QMouseEvent):
        """Handles adding a new mask (left-click) or deleting one (right-click)."""
        # Right-click to delete
        if event.button() == Qt.MouseButton.RightButton:
            for rect in reversed(self.mask_list): # Iterate backwards to find the top-most rect first
                if rect.contains(event.pos()):
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
                self.mask_list.append(new_rect)
                self.update() # Trigger a repaint to show the new rect
            self.origin_point = None

    def get_selected_rects(self) -> list[QRect]:
        """Returns the final list of mask rectangles."""
        return self.mask_list


class MaskDialog(QDialog):
    """ The dialog window for mask editing. """
    def __init__(self, pixmap: QPixmap, existing_masks: list[QRect], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Metadata Mask (LMB: Add, RMB: Delete)")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        self.image_label = InteractiveImageLabel(initial_masks=existing_masks)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        layout.addWidget(self.image_label)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_selected_rects(self) -> list[QRect]:
        """Gets the result from the dialog."""
        return self.image_label.get_selected_rects()