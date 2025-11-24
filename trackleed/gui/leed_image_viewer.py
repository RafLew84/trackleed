# stal/gui/leed_image_viewer.py

from typing import List, Dict
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QWheelEvent, QColor, QBrush, QPen, QMouseEvent, QPainterPath, QContextMenuEvent 
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QMenu,
                             QGraphicsRectItem, QGraphicsSimpleTextItem, QGraphicsPathItem, QGraphicsEllipseItem)

from trackleed.core.data_models import SpotROI, Track


class LeedImageViewer(QGraphicsView):
    """
    An advanced interactive viewer for LEED images.
    Supports modes for panning, adding, and deleting ROIs.
    Emits signals to notify the main window of user actions.
    """
    # --- Signals ---
    # Signal emitted when user left-clicks in 'ADD_ROI' mode
    roi_added_at = pyqtSignal(QPointF) 
    # Signal emitted when user right-clicks on an ROI in 'DELETE_ROI' mode
    roi_deleted = pyqtSignal(int)

    roi_moved = pyqtSignal(int, QRectF)
    new_track_started_at = pyqtSignal(QPointF)

    track_point_context_menu_requested = pyqtSignal(int, QPoint)
    point_for_track_added_at = pyqtSignal(QPointF)

    def __init__(self, parent: QWidget | None = None):
        """
        Initializes the LeedImageViewer.

        This constructor sets up the underlying QGraphicsScene, initializes
        attributes for managing all graphical items (image, masks, ROIs, paths),
        defines default drawing styles, and sets the initial interaction mode.

        Args:
            parent: The parent Qt widget for this viewer.
        """
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item: QGraphicsPixmapItem | None = None
        
        # Lists and dictionaries to manage all drawn overlay items.
        self._mask_items: List[QGraphicsRectItem] = []
        self._roi_items: Dict[int, QGraphicsRectItem] = {}
        self._roi_label_items: Dict[int, QGraphicsSimpleTextItem] = {}
        self._track_path_item: QGraphicsPathItem | None = None
        self._track_path_items = []
        self._pattern_center_item = None
        self._analysis_circle_item: QGraphicsEllipseItem | None = None

        # --- Define Drawing Styles ---
        # Default pen for drawing standard ROIs.
        self.roi_pen_default = QPen(Qt.GlobalColor.cyan)
        self.roi_pen_default.setWidth(2)
        self.roi_pen_highlight = QPen(Qt.GlobalColor.yellow)
        self.roi_pen_highlight.setWidth(3)
        
        self.mode = 'PAN' # Default mode
        self._item_to_move = None
        self.set_mode(self.mode) # Apply default mode settings

    def draw_pattern_center(self, center_point: QPointF):
        """
        Draws a marker on the scene for the diffraction pattern's center.

        This method first clears any previously drawn center marker, then creates
        and displays a new yellow circular marker at the specified coordinates.

        Args:
            center_point: A QPointF object containing the (x, y) coordinates
                        where the marker should be drawn.
        """
        # Ensure only one center marker is visible at a time by clearing the old one.
        self.clear_pattern_center() 
        
        # --- Marker Style ---
        # Define the radius of the circular marker in pixels.
        radius = 8
        # Create a yellow pen for high visibility.
        pen = QPen(Qt.GlobalColor.yellow)
        pen.setWidth(2)
        
        # --- Create and Draw the Marker ---
        # Create a QGraphicsEllipseItem. The coordinates are calculated to
        # ensure the ellipse is centered on the provided center_point.
        ellipse = QGraphicsEllipseItem(
            center_point.x() - radius, center_point.y() - radius,
            2 * radius, 2 * radius
        )
        ellipse.setPen(pen)
        self._scene.addItem(ellipse)
        self._pattern_center_item = ellipse

    def clear_pattern_center(self):
        """
        Removes the previously drawn pattern center marker from the scene.
        """
        # Check if a reference to the marker item currently exists.
        if self._pattern_center_item:
            self._scene.removeItem(self._pattern_center_item)
            self._pattern_center_item = None

    def set_mode(self, mode: str):
        """
        Sets the current interaction mode for the viewer.

        This method updates the internal mode state and changes the view's
        drag behavior accordingly. This controls whether mouse drags are used for
        panning the scene or for other custom actions.

        Args:
            mode: A string identifier for the desired mode (e.g., 'PAN', 'ADD_ROI').
        """
        # Store the new mode as an instance attribute for use in other event handlers.
        self.mode = mode

        # Set the drag mode based on the selected interaction mode. 
        if self.mode == 'PAN':
            # In 'PAN' mode, enable the built-in "hand tool" for easy navigation.
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            # For all other modes (like adding/deleting ROIs), disable dragging
            # to allow for precise single-click actions.
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        print(f"Viewer mode set to: {self.mode}")
    
    def draw_track_path(self, points: List[QPointF], color: QColor, width: int = 1):
        """
        Draws a path on the scene by connecting a given list of points.

        This method is used to visualize a single spot's trajectory.

        Args:
            points: A list of QPointF objects representing the vertices of the path.
            color: A QColor object specifying the color of the path.
            width: An integer specifying the width of the line in pixels.
        """
        # A path cannot be drawn with fewer than two points.
        if len(points) < 2:
            return
            
        # Create a QPainterPath, starting from the first point in the list.
        path = QPainterPath(points[0])

        # Iterate through the rest of the points, adding line segments to the path.
        for i in range(1, len(points)):
            path.lineTo(points[i])
            
        # Create a QPen with the specified color and width for drawing the path.
        pen = QPen(color, width)

        # Set the pen to be "cosmetic". This is a key feature that ensures the line
        # thickness remains constant on the screen, regardless of the zoom level.
        pen.setCosmetic(True) 
        
        path_item = self._scene.addPath(path, pen)
        self._track_path_items.append(path_item)

    def clear_paths(self):
        """
        Removes all previously drawn track paths from the scene.
        """
        # Iterate through the list of stored QGraphicsPathItem references.
        for item in self._track_path_items:
            self._scene.removeItem(item)
        self._track_path_items.clear()

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse clicks based on the current interaction mode."""
        if self._image_item is None:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if self.mode == 'ADD_NEW_TRACK' and event.button() == Qt.MouseButton.LeftButton:
            self.new_track_started_at.emit(scene_pos)
            return

        elif self.mode == 'MOVE_ROI' and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item in self._roi_items.values():
                self._item_to_move = item
                self._drag_start_pos = scene_pos
                return # Consume the event

        elif self.mode == 'ADD_ROI' and event.button() == Qt.MouseButton.LeftButton:
            self.roi_added_at.emit(scene_pos)
        
        elif self.mode == 'ADD_CORRECT_POINT' and event.button() == Qt.MouseButton.LeftButton:
            self.point_for_track_added_at.emit(scene_pos)
        
        elif self.mode == 'DELETE_ROI' and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item in self._roi_items.values():
                # Find the ID associated with the clicked QGraphicsRectItem
                spot_id_to_delete = None
                for id, rect_item in self._roi_items.items():
                    if item == rect_item:
                        spot_id_to_delete = id
                        break
                if spot_id_to_delete is not None:
                    self.roi_deleted.emit(spot_id_to_delete)
        else:
            # Fallback to default behavior (e.g., panning)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles moving an item during a drag operation."""
        if self.mode == 'MOVE_ROI' and self._item_to_move is not None:
            new_scene_pos = self.mapToScene(event.pos())
            delta = new_scene_pos - self._drag_start_pos
            
            # Move the ROI rectangle
            self._item_to_move.moveBy(delta.x(), delta.y())
            
            # Move the associated label
            spot_id = self._item_to_move.data(0)
            if spot_id in self._roi_label_items:
                self._roi_label_items[spot_id].moveBy(delta.x(), delta.y())
                
            self._drag_start_pos = new_scene_pos
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Finalizes the move operation."""
        if self.mode == 'MOVE_ROI' and self._item_to_move is not None:
            spot_id = self._item_to_move.data(0)
            new_rect = self._item_to_move.sceneBoundingRect()
            self.roi_moved.emit(spot_id, new_rect)
            # Reset dragging state
            self._item_to_move = None
            self._drag_start_pos = None
        else:
            super().mouseReleaseEvent(event)

    def clear_rois(self):
        """Helper to clear only ROI-related items."""
        for item in self._roi_items.values(): self._scene.removeItem(item)
        for item in self._roi_label_items.values(): self._scene.removeItem(item)
        self._roi_items.clear()
        self._roi_label_items.clear()

    def update_rois(self, spots: List[SpotROI]):
        """
        Updates all ROI markers and their ID labels on the scene.

        This method performs a full refresh of the displayed ROIs. It first
        removes all existing ROI rectangles and labels from the scene, then
        iterates through the provided list of SpotROI objects to draw a new set.

        Args:
            spots: A list of SpotROI objects to be displayed on the scene.
        """
        for item in self._roi_items.values(): self._scene.removeItem(item)
        for item in self._roi_label_items.values(): self._scene.removeItem(item)

        self._roi_items.clear()
        self._roi_label_items.clear()

        for spot in spots:
            rect_item = self._scene.addRect(QRectF(spot.roi_rect), self.roi_pen_default)
            rect_item.setData(0, spot.id)
            self._roi_items[spot.id] = rect_item
            label_item = QGraphicsSimpleTextItem(str(spot.id))
            label_item.setBrush(Qt.GlobalColor.cyan)
            label_item.setPos(QPointF(spot.roi_rect.topLeft()))
            self._scene.addItem(label_item)
            self._roi_label_items[spot.id] = label_item

    def highlight_roi(self, spot_id: int | None):
        """
        Highlights a specific ROI on the scene by changing its border color.

        This method first resets all ROIs to their default appearance, then applies
        a special highlight style to the ROI corresponding to the given spot_id.

        Args:
            spot_id: The ID of the spot whose ROI should be highlighted. If None,
                    all highlights are cleared.
        """
        # First, reset all currently displayed ROIs to the default pen style.
        # This ensures that any previously highlighted ROI is returned to normal.
        for rect_item in self._roi_items.values(): rect_item.setPen(self.roi_pen_default)
        if spot_id is not None and spot_id in self._roi_items: self._roi_items[spot_id].setPen(self.roi_pen_highlight)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Creates a context menu when right-clicking on an ROI in edit mode."""
        if self.mode != 'EDIT_TRACK':
            super().contextMenuEvent(event) # Standard behavior for other modes
            return

        # Find if an ROI item is under the cursor
        item = self.itemAt(event.pos())
        if item in self._roi_items.values():
            spot_id = item.data(0)
            print(f"[Viewer] Context menu requested for item with Spot ID: {spot_id}")
            # Emit a signal to the dialog, telling it which spot was clicked and where
            self.track_point_context_menu_requested.emit(spot_id, event.globalPos())

    def clear_overlays(self):
        """
        Removes all overlay graphics from the scene.

        This method provides a complete cleanup of all drawn items, including
        exclusion masks, ROI rectangles, and ROI labels. It is typically called
        before displaying a new image to ensure a clean view.
        """
        for item in self._mask_items + list(self._roi_items.values()) + list(self._roi_label_items.values()): 
            self._scene.removeItem(item)
        self._mask_items.clear()
        self._roi_items.clear()
        self._roi_label_items.clear()
        if self._analysis_circle_item:
            self._scene.removeItem(self._analysis_circle_item)
            self._analysis_circle_item = None

    def set_image(self, pixmap: QPixmap | None, fit: bool = False):
        """
        Clears the scene and displays a new image.
        Optionally fits the view to the image.
        """
        self.clear_overlays()
        if self._image_item:
            self._scene.removeItem(self._image_item)
        self._image_item = None

        if pixmap and not pixmap.isNull():
            self._image_item = self._scene.addPixmap(pixmap)
            # Only fit the view if explicitly requested
            if fit:
                self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def update_masks(self, masks: List[QRect]):
        """
        Updates the exclusion mask overlays on the scene.

        This method performs a full refresh of the mask items. It first removes
        all currently displayed mask rectangles and then draws a new set based
        on the provided list of QRects. These masks define areas of the image
        that should be ignored during analysis.

        Args:
            masks: A list of QRect objects defining the geometry of the
                exclusion masks.
        """
        # First, remove all existing mask items from the scene.
        for item in self._mask_items: self._scene.removeItem(item)
        self._mask_items.clear()
        pen = QPen(QColor(255, 0, 0, 180))
        pen.setWidth(2)
        brush = QBrush(QColor(255, 0, 0, 70))
        for rect in masks: 
            self._mask_items.append(self._scene.addRect(QRectF(rect), pen, brush))

    def update_analysis_circle(self, circle: tuple[int, int, int] | None):
        """Draw a single analysis circle (cx, cy, r) if provided."""
        if self._analysis_circle_item:
            self._scene.removeItem(self._analysis_circle_item)
            self._analysis_circle_item = None
        if circle is None:
            return
        cx, cy, r = circle
        pen = QPen(QColor(0, 200, 255, 200))
        pen.setWidth(2)
        pen.setCosmetic(True)
        ellipse = QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
        ellipse.setPen(pen)
        self._analysis_circle_item = ellipse
        self._scene.addItem(ellipse)

    def wheelEvent(self, event: QWheelEvent):
        """
        Handles the mouse wheel scroll event to implement view zooming.

        This method overrides the default wheel event. It checks the direction
        of the scroll and applies a scaling transformation to the view's scene,
        effectively zooming in or out on the content.

        Args:
            event: The QWheelEvent object containing details about the scroll,
                such as the angle and direction.
        """
        # Define the scaling factors for zooming in and out.
        # A value of 1.15 means each scroll step zooms in by 15%.
        zoom_in_factor, zoom_out_factor = 1.15, 1.0 / 1.15
        if event.angleDelta().y() > 0: self.scale(zoom_in_factor, zoom_in_factor)
        else: self.scale(zoom_out_factor, zoom_out_factor)
    
    def show_track_path(self, track: Track | None):
        """Draws a line connecting all spots in a given track."""
        # Remove old path if it exists
        if self._track_path_item:
            self._scene.removeItem(self._track_path_item)
            self._track_path_item = None
            
        if track is None:
            return
            
        # Create a new path
        path = QPainterPath()
        
        # Sort points by frame id to draw in correct order
        sorted_frame_ids = sorted(track.spots_by_image_id.keys())
        if not sorted_frame_ids:
            return
            
        first_spot = track.spots_by_image_id[sorted_frame_ids[0]]
        path.moveTo(QPointF(first_spot.center_x, first_spot.center_y))
        
        for frame_id in sorted_frame_ids[1:]:
            spot = track.spots_by_image_id[frame_id]
            path.lineTo(QPointF(spot.center_x, spot.center_y))
            
        pen = QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.SolidLine)
        self._track_path_item = self._scene.addPath(path, pen)
