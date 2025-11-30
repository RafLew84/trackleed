# stal/gui/main_window.py

from pathlib import Path
from dataclasses import dataclass

from typing import List
from threading import Event
import numpy as np
import torch
import shutil
from PyQt6.QtCore import Qt, QPointF, QRect, QRectF, QThread, QPoint
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QBrush, QColor, QActionGroup , QIcon
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGroupBox,
                             QLabel, QFileDialog, QListWidgetItem, QMessageBox, QApplication,
                             QDialog, QDockWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
                             QPushButton, QCheckBox, QToolBar, QComboBox, QProgressDialog,
                             QScrollArea, QStackedWidget, QSizePolicy, QInputDialog, QLineEdit)
import os
import sys

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.file_parser import parse_directory
from trackleed.io.image_reader import load_tiff
from trackleed.gui.mask_dialog import MaskDialog
from trackleed.analysis.spot_finding import find_spots
from trackleed.analysis.dl_spot_finder import detect_spots_with_model, DLSpotDetectionError
from trackleed.analysis.yolo_detector import detect_spots_with_yolo, YOLODetectionError, YOLOSettings
from trackleed.io.session import load_session_from_file, save_session_to_file
from trackleed.gui.circle_mask_dialog import CircleMaskDialog
from trackleed.io.gif_loader import extract_gif_frames
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.gui.tracking_dialog import TrackingDialog
from trackleed.logic.generic_worker import GenericWorker
from trackleed.analysis.tracking import (
    track_spots_optical_flow,
    track_spots_optical_flow_skip_anchor,
    track_spots_hungarian,
    track_spots_raft_fallback,
)

from skimage.util import img_as_float
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.morphology import white_tophat, disk


@dataclass
class DLSpotSettings:
    model_path: str | None = None
    confidence: float = 0.5
    min_area_px: int = 10
    max_area_px: int = 10000
    bbox_margin_px: int = 4
    dilate_radius_px: int = 0


@dataclass
class YOLOSpotSettings:
    model_path: str | None = None
    confidence: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    bbox_margin_px: int = 2


class MainWindow(QMainWindow):
    """
    The main window class for the STAL application.
    It orchestrates the UI and connects user actions to the backend logic.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEEDTrack - Spot Tracking")
        self.setGeometry(100, 100, 1200, 800)

        # --- Store data ---
        self.data_files: list[LeedFile] = []
        self.background_file: LeedFile | None = None
        self.metadata_mask_list: list[QRect] = []
        self.analysis_circle: tuple[int, int, int] | None = None  # (cx, cy, r)
        self.tracking_results: List[Track] = []
        self.dl_settings = DLSpotSettings()
        self.yolo_settings = YOLOSpotSettings()
        self.current_spot_backend = "Classic"
        self.segmentation_models_dir = Path("segmentation_models")
        self._dl_models_available = False
        self._yolo_models_available = False

        # --- Initialize UI ---
        self._init_ui()
        self._create_analysis_panel()
        self._create_toolbar()
        print("MainWindow has been created and initialized.")

    def _init_ui(self):
        """Initializes the main UI components."""
        self._create_menu()
        self._create_layout()

    def _create_menu(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Directory...", self)
        open_action.triggered.connect(self.open_directory_dialog)
        file_menu.addAction(open_action)

        import_gif_action = QAction("Import GIF (sample 0.1s)...", self)
        import_gif_action.triggered.connect(self.import_gif_dialog)
        file_menu.addAction(import_gif_action)

        load_session_action = QAction("Load Session...", self)
        load_session_action.triggered.connect(self.load_session_dialog)
        file_menu.addAction(load_session_action)

        save_session_action = QAction("Save Session...", self)
        save_session_action.triggered.connect(self.save_session_dialog)
        file_menu.addAction(save_session_action)

        load_bg_action = QAction("&Load Background File...", self)
        load_bg_action.triggered.connect(self.load_background_file_dialog)
        file_menu.addAction(load_bg_action)

        export_yolo_action = QAction("Export Spots to YOLO Labels...", self)
        export_yolo_action.triggered.connect(self.export_spots_to_yolo)
        file_menu.addAction(export_yolo_action)

        export_all_yolo_action = QAction("Export All (images + YOLO labels)...", self)
        export_all_yolo_action.triggered.connect(self.export_all_to_yolo)
        file_menu.addAction(export_all_yolo_action)

        # --- Mask Menu ---
        mask_menu = menu_bar.addMenu("&Mask")
        set_mask_action = QAction("&Set Metadata Mask...", self)
        set_mask_action.triggered.connect(self.open_mask_dialog)
        mask_menu.addAction(set_mask_action)
        clear_mask_action = QAction("Clear Current Mask", self)
        clear_mask_action.triggered.connect(self.clear_mask)
        mask_menu.addAction(clear_mask_action)
        set_analysis_circle_action = QAction("Set Analysis Circle...", self)
        set_analysis_circle_action.triggered.connect(self.open_analysis_circle_dialog)
        mask_menu.addAction(set_analysis_circle_action)
        clear_analysis_circle_action = QAction("Clear Analysis Circle", self)
        clear_analysis_circle_action.triggered.connect(self.clear_analysis_circle)
        mask_menu.addAction(clear_analysis_circle_action)

    def _create_toolbar(self):
        """Creates the main toolbar for interaction modes."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Action group makes the actions behave like radio buttons
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)

        # Pan/Select Mode (default)
        self.pan_mode_action = QAction("Pan/Select", self)
        self.pan_mode_action.setCheckable(True)
        self.pan_mode_action.setChecked(True)
        self.pan_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('PAN'))
        toolbar.addAction(self.pan_mode_action)
        mode_group.addAction(self.pan_mode_action)

        # Move ROI Mode
        self.move_roi_mode_action = QAction("Move ROI", self)
        self.move_roi_mode_action.setCheckable(True)
        self.move_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('MOVE_ROI'))
        toolbar.addAction(self.move_roi_mode_action)
        mode_group.addAction(self.move_roi_mode_action)

        # Add ROI Mode
        self.add_roi_mode_action = QAction("Add ROI", self)
        self.add_roi_mode_action.setCheckable(True)
        self.add_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_ROI'))
        toolbar.addAction(self.add_roi_mode_action)
        mode_group.addAction(self.add_roi_mode_action)

        # Delete ROI Mode
        self.delete_roi_mode_action = QAction("Delete ROI", self)
        self.delete_roi_mode_action.setCheckable(True)
        self.delete_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('DELETE_ROI'))
        toolbar.addAction(self.delete_roi_mode_action)
        mode_group.addAction(self.delete_roi_mode_action)

    def _create_layout(self):
        """
        Creates the main layout of the application's central widget.

        This method sets up a two-column horizontal layout.
        - The left column contains the list of data files and a preview panel
        for the selected background image.
        - The right column contains the main, interactive LeedImageViewer.
        It also initializes a status bar label for background file information.
        """
        # Create the central widget that will hold the main layout.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # The main layout is a horizontal box layout.
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: File List & Background Preview ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # File list setup
        left_panel_layout.addWidget(QLabel("LEED Files:"))
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.display_selected_image)
        left_panel_layout.addWidget(self.file_list_widget)

        # Background preview group box
        bg_preview_group = QGroupBox("Background Preview")
        bg_preview_layout = QVBoxLayout(bg_preview_group)
        bg_preview_group.setMaximumHeight(250) # Limit the panel's vertical size

        # Label for the image thumbnail
        self.bg_preview_label = QLabel("No background loaded")
        self.bg_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bg_preview_label.setMinimumSize(128, 128)
        self.bg_preview_label.setStyleSheet("background-color: #2E2E2E; color: white;")
        
        # Label for the filename
        self.bg_filename_label = QLabel("Filename: N/A")
        self.bg_filename_label.setWordWrap(True) # Wrap text for long filenames

        bg_preview_layout.addWidget(self.bg_preview_label)
        bg_preview_layout.addWidget(self.bg_filename_label)
        left_panel_layout.addWidget(bg_preview_group)
        
        # --- Right Panel: Main Image Viewer ---
        self.image_viewer = LeedImageViewer(self)
        self.image_viewer.roi_added_at.connect(self.handle_add_roi)
        self.image_viewer.roi_deleted.connect(self.handle_delete_roi)
        self.image_viewer.roi_moved.connect(self.handle_move_roi)
        
        # --- Assemble Main Layout ---
        # Add the left and right panels to the main horizontal layout.
        main_layout.addWidget(left_panel_widget)
        main_layout.addWidget(self.image_viewer)
        
        # Set the stretch factors to make the right panel (image viewer) larger.
        # The viewer will take up 3/4 of the space, and the left panel 1/4.
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

    def _update_bg_preview(self):
        """Loads the background image and updates the preview widgets."""
        if not self.background_file: return
        
        image_data = load_tiff(self.background_file.filepath)
        pixmap = self._convert_numpy_to_pixmap(image_data)
        
        if not pixmap.isNull():
            self.bg_preview_label.setPixmap(pixmap.scaled(
                self.bg_preview_label.width(), self.bg_preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.bg_preview_label.setText("Preview failed")
            
        self.bg_filename_label.setText(f"Filename: {self.background_file.filename}")

    def _clear_bg_preview(self):
        """Resets the background preview widgets to their default state."""
        self.bg_preview_label.setText("No background loaded")
        self.bg_filename_label.setText("Filename: N/A")

    # def _create_layout(self):
    #     """Creates the main two-column layout."""
    #     central_widget = QWidget()
    #     self.setCentralWidget(central_widget)
    #     layout = QHBoxLayout(central_widget)

    #     # --- Left Column: Image Viewer ---
    #     self.image_viewer = LeedImageViewer(self)

    #     self.image_viewer.roi_added_at.connect(self.handle_add_roi)
    #     self.image_viewer.roi_deleted.connect(self.handle_delete_roi)
    #     self.image_viewer.roi_moved.connect(self.handle_move_roi)

    #     # --- Right Column: File List ---
    #     self.file_list_widget = QListWidget()
    #     self.file_list_widget.currentItemChanged.connect(self.display_selected_image)

    #     layout.addWidget(self.image_viewer, 3)  # 3/4 of the space
    #     layout.addWidget(self.file_list_widget, 1)    # 1/4 of the space

    #     self.bg_status_label = QLabel("Background file: Not loaded")
    #     self.statusBar().addPermanentWidget(self.bg_status_label)

    def handle_move_roi(self, spot_id: int, new_rect_f: QRectF):
        """Slot to handle updating an ROI's position in the data model."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item: return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Find the spot to update
        for spot in leed_file.spots:
            if spot.id == spot_id:
                # Update its data
                spot.roi_rect = new_rect_f.toRect() # Convert float rect back to int rect
                spot.center_x = new_rect_f.center().x()
                spot.center_y = new_rect_f.center().y()
                break # Stop searching

        # Refresh the list widget to show updated coordinates (if displayed)
        self.populate_spot_list(leed_file.spots)
        # Highlight the moved spot
        self.on_spot_selection_changed(self.spot_list_widget.currentItem(), None)

    def handle_add_roi(self, scene_pos: QPointF):
        """Slot to handle the creation of a new manual ROI."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item: return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Get ROI size from the GUI
        roi_size = self.roi_size_spinbox.value()
        roi_half = roi_size // 2

        # Create a new QRect centered on the click position
        new_rect = QRect(int(scene_pos.x() - roi_half), int(scene_pos.y() - roi_half), roi_size, roi_size)

        # Find the next available ID
        next_id = max([s.id for s in leed_file.spots] + [0]) + 1

        # Create the new SpotROI object
        new_spot = SpotROI(
            id=next_id,
            roi_rect=new_rect,
            center_x=scene_pos.x(),
            center_y=scene_pos.y(),
            intensity=0, # Manual spot has no intensity initially
            source='manual'
        )
        leed_file.spots.append(new_spot)
        self.image_viewer.update_rois(leed_file.spots)
        self.populate_spot_list(leed_file.spots)

    def handle_delete_roi(self, spot_id: int):
        """Slot to handle the deletion of an ROI."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item: return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Find and remove the spot with the given ID
        leed_file.spots = [spot for spot in leed_file.spots if spot.id != spot_id]
        self.image_viewer.update_rois(leed_file.spots)
        self.populate_spot_list(leed_file.spots)

    def open_directory_dialog(self):
        """Opens a dialog to select a directory and processes the files within."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select LEED Data Directory")
        if not dir_path:
            return  # User cancelled the dialog

        directory = Path(dir_path)
        try:
            self.data_files = parse_directory(directory)
            # Reset background metadata whenever a new directory is loaded
            self.background_file = None
            self._clear_bg_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse directory:\n{e}")
            return

        if not self.data_files:
            QMessageBox.warning(self, "No Files Found", "No valid LEED data files were found in the selected directory.")

        self.populate_file_list()

    def import_gif_dialog(self):
        """Import a GIF, sample frames every 0.1s, and append them as images."""
        gif_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select GIF to import",
            "",
            "GIF Files (*.gif);;All Files (*)",
        )
        if not gif_path_str:
            return
        interval, ok = QInputDialog.getInt(
            self,
            "Sampling interval",
            "Sample every N milliseconds:",
            value=100,
            min=1,
            max=10000,
        )
        if not ok:
            return
        gif_path = Path(gif_path_str)
        try:
            out_dir, saved = extract_gif_frames(gif_path, dest_root=Path("data/gif_import") / gif_path.stem, sample_ms=interval)
            new_files = parse_directory(out_dir)
            if not new_files:
                QMessageBox.warning(self, "No Frames", "Could not extract any frames from the GIF.")
                return
            # Offset IDs to avoid collisions
            max_id = max([f.id for f in self.data_files if f.id is not None] + [0])
            for idx, lf in enumerate(new_files, start=1):
                lf.id = max_id + idx
            self.data_files.extend(new_files)
            self.populate_file_list()
            QMessageBox.information(
                self,
                "GIF Imported",
                f"Extracted {len(saved)} frame(s) every {interval} ms to {out_dir} and added to the list.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import GIF:\n{e}")

    def load_background_file_dialog(self):
        """Opens a dialog to manually select a background image file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "", # Default directory
            "Image Files (*.tif *.tiff *.png *.jpg)"
        )
        
        if not filepath: # User cancelled the dialog
            return

        path_obj = Path(filepath)
        
        # Create a lightweight LeedFile object for the background frame
        self.background_file = LeedFile(
            filepath=path_obj,
            filename=path_obj.name,
            is_background=True,
            # Other metadata can stay at defaults because the background is not analyzed
            element='', surface=0, mode='', temperature=0, energy=0, id=None, coverage=None
        )
        
        # Update the status bar label
        self._update_bg_preview()
        QMessageBox.information(self, "Success", "Background file loaded successfully.")
    
    def open_mask_dialog(self):
        """
        Opens the mask editor, passing the current list of masks.
        """
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list before setting a mask.")
            return

        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)
        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the original image data to create a mask.")
            return

        height, width = original_image_data.shape[:2]
        if original_image_data.ndim == 2:
            q_image = QImage(original_image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        else: # Assuming RGB for 3D arrays for simplicity
            q_image = QImage(original_image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        full_res_pixmap = QPixmap.fromImage(q_image)

        # Pass the CURRENT list of masks to the dialog
        dialog = MaskDialog(full_res_pixmap, self.metadata_mask_list, self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # Retrieve the NEW, potentially modified list of masks
            self.metadata_mask_list = dialog.get_selected_rects()
            print(f"Mask updated. Now contains {len(self.metadata_mask_list)} regions.")
            self.display_selected_image(current_list_item, None)

    def populate_file_list(self):
        """Populates the QListWidget with the found files."""
        self.file_list_widget.clear()

        if self.background_file:
            item = QListWidgetItem(f"BACKGROUND: {self.background_file.filename}")
            item.setData(Qt.ItemDataRole.UserRole, self.background_file)
            self.file_list_widget.addItem(item)

        for leed_file in self.data_files:
            # Create a user-friendly display string for each file
            display_text = f"ID: {leed_file.id}, T: {leed_file.temperature}°C, E: {leed_file.energy} eV, C: {leed_file.coverage}"
            item = QListWidgetItem(display_text)
            # Store the entire LeedFile object with the list item
            item.setData(Qt.ItemDataRole.UserRole, leed_file)
            self.file_list_widget.addItem(item)

    def clear_mask(self):
        """Clears all defined mask regions."""
        if not self.metadata_mask_list:
            return # Nothing to clear
        
        self.metadata_mask_list.clear()
        print("Mask has been cleared.")
        
        # Refresh the current view to show the image without the mask
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def open_analysis_circle_dialog(self):
        """Opens a dialog to set a global analysis circle (applied to all images)."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list before setting an analysis circle.")
            return

        leed_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the image to set an analysis circle.")
            return

        height, width = original_image_data.shape[:2]
        if original_image_data.ndim == 2:
            q_image = QImage(original_image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            q_image = QImage(original_image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        full_res_pixmap = QPixmap.fromImage(q_image)

        existing_center = None
        existing_radius = 0
        if self.analysis_circle:
            cx, cy, r = self.analysis_circle
            existing_center = QPoint(cx, cy)
            existing_radius = r

        dialog = CircleMaskDialog(full_res_pixmap, existing_center, existing_radius, self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            center, radius = dialog.get_circle()
            if center and radius > 0:
                self.analysis_circle = (int(center.x()), int(center.y()), int(radius))
                print(f"Analysis circle set: center=({center.x()}, {center.y()}), r={radius}")
            else:
                self.analysis_circle = None
                print("Analysis circle cleared (empty selection).")
            self.refresh_current_view()

    def clear_analysis_circle(self):
        """Clears the analysis circle overlay and mask."""
        if self.analysis_circle is None:
            return
        self.analysis_circle = None
        print("Analysis circle has been cleared.")
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def display_selected_image(self, current_item: QListWidgetItem | None, previous_item: QListWidgetItem | None):
        """
        Prepares the base image (original or processed) and tells the viewer
        to display it along with any masks and ROIs.
        """
        self.spot_list_widget.clear() # Clear spot details from previous image
        if current_item is None:
            self.image_viewer.set_image(None)
            return

        leed_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        image_data = load_tiff(leed_file.filepath)
        
        # Populate the spot list with the data for the newly selected file
        self.populate_spot_list(leed_file.spots)

        if image_data is None or image_data.size == 0:
            self.image_viewer.set_image(None)
            return

        # 1. Prepare the base image to be displayed (original or processed)
        base_image_numpy = self._get_base_image_for_display(leed_file, image_data)
        final_pixmap = self._convert_numpy_to_pixmap(base_image_numpy)
        
        # 2. Set the base image in the viewer
        self.image_viewer.set_image(final_pixmap)
        
        # 3. Tell the viewer to draw overlays
        self.image_viewer.update_masks(self.metadata_mask_list)
        self.image_viewer.update_analysis_circle(self.analysis_circle)
        self.image_viewer.update_rois(leed_file.spots)

    def _get_base_image_for_display(self, leed_file, original_image_data):
        """Determines if original or processed image should be shown."""
        if self.show_processed_checkbox.isChecked() and not leed_file.is_background:
            try:
                bg_img = load_tiff(self.background_file.filepath) if self.background_file else None
                params = {'blur_sigma': self.blur_sigma_spinbox.value(), 'wth_radius': self.wth_radius_spinbox.value()}
                main_float = img_as_float(original_image_data.copy())
                subtracted_image = main_float
                if bg_img is not None:
                    bg_float = rgb2gray(bg_img) if bg_img.ndim == 3 else img_as_float(bg_img)
                    main_masked_for_mean, bg_masked_for_mean = main_float.copy(), bg_float.copy()
                    for rect in self.metadata_mask_list:
                        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                        main_masked_for_mean[y:y+h, x:x+w] = 0
                        bg_masked_for_mean[y:y+h, x:x+w] = 0
                    main_mean = main_masked_for_mean[main_masked_for_mean > 0].mean()
                    bg_mean = bg_masked_for_mean[bg_masked_for_mean > 0].mean()
                    if bg_mean > 1e-9: subtracted_image = main_float - (bg_float * (main_mean / bg_mean))
                residual_bg = gaussian_filter(subtracted_image, sigma=params['blur_sigma'])
                processed_image = subtracted_image - residual_bg
                return white_tophat(processed_image, footprint=disk(params['wth_radius']))
            except Exception as e:
                print(f"Could not generate processed preview: {e}")
        return original_image_data
    
    def _convert_numpy_to_pixmap(self, image_data):
        """Converts a NumPy array to a QPixmap for display."""
        if image_data.dtype != np.uint8:
            min_val, max_val = image_data.min(), image_data.max()
            if max_val - min_val > 1e-9: image_data = 255 * (image_data - min_val) / (max_val - min_val)
            else: image_data = np.zeros_like(image_data)
            image_data = image_data.astype(np.uint8)
            
        height, width = image_data.shape[:2]
        if image_data.ndim == 3 and image_data.shape[2] == 3: # RGB
            q_image = QImage(image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        else: # Grayscale
            q_image = QImage(image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_image)

    def _apply_analysis_circle_mask(self, image_data: np.ndarray) -> np.ndarray:
        """Zero out pixels outside the analysis circle, if defined."""
        if self.analysis_circle is None or image_data is None or image_data.size == 0:
            return image_data
        cx, cy, r = self.analysis_circle
        h, w = image_data.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        if image_data.ndim == 2:
            out = image_data.copy()
            out[~mask] = 0
            return out
        else:
            out = image_data.copy()
            out[~mask, ...] = 0
            return out
    
    def _create_analysis_panel(self):
        """Creates a dockable widget with analysis controls."""
        self.analysis_dock = QDockWidget("Analysis Parameters", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        # Create a container widget and layout for the controls
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        
        # Use a form layout for neat parameter alignment
        find_spots_group = QGroupBox("Find Spots")
        find_spots_layout = QVBoxLayout(find_spots_group)

        backend_form = QFormLayout()
        self.spot_backend_combo = QComboBox()
        self.spot_backend_combo.addItems(["Classic", "Deep Learning", "YOLO"])
        self.spot_backend_combo.setToolTip(
            "Select the spot detection backend. Classic uses the existing WTH/peak pipeline; "
            "Deep Learning uses a segmentation model from the segmentation_models folder."
        )
        backend_form.addRow("Spot detector:", self.spot_backend_combo)
        find_spots_layout.addLayout(backend_form)

        self.spot_backend_stack = QStackedWidget()
        self.spot_backend_stack.addWidget(self._build_classic_find_spots_panel())
        self.spot_backend_stack.addWidget(self._build_dl_find_spots_panel())
        self.spot_backend_stack.addWidget(self._build_yolo_find_spots_panel())
        find_spots_layout.addWidget(self.spot_backend_stack)

        self.dl_device_combo = QComboBox()
        self.dl_device_combo.addItems(["Auto", "CPU", "GPU"])
        self.dl_device_combo.setToolTip("Device for DL/YOLO inference. Auto = GPU if available.")
        find_spots_layout.addWidget(self.dl_device_combo)

        self.det_subtract_bg_checkbox = QCheckBox("Subtract background before detection (DL/YOLO)")
        self.det_subtract_bg_checkbox.setToolTip("Applies only when backend is Deep Learning or YOLO and a background file is loaded.")
        find_spots_layout.addWidget(self.det_subtract_bg_checkbox)

        self.find_spots_button = QPushButton("Find Spots")
        self.find_spots_button.clicked.connect(self.run_spot_finding)
        find_spots_layout.addWidget(self.find_spots_button)
        self.find_all_spots_button = QPushButton("Find All Spots")
        self.find_all_spots_button.setToolTip("Run the selected detector on every frame in the list.")
        self.find_all_spots_button.clicked.connect(self.run_spot_finding_all)
        find_spots_layout.addWidget(self.find_all_spots_button)

        layout.addWidget(find_spots_group)
        self.spot_backend_combo.currentTextChanged.connect(self._on_spot_backend_changed)

        layout.addStretch() # Pushes everything up

        tracking_group = QGroupBox("Tracking")
        

        form_layout = QFormLayout(tracking_group)
        self.tracking_scenario_combo = QComboBox()
        self.tracking_scenario_combo.addItems([
            "Optical Flow Tracking",
            "Optical Flow (skip, anchor)",
            "Detection Linking (Hungarian)",
            "RAFT (with LK fallback)",
        ])
        # Backwards compatibility alias used by some dialogs/scripts.
        self.trackingAlgoCombo = self.tracking_scenario_combo
        form_layout.addRow("Tracking Algorithm:", self.tracking_scenario_combo)
        # layout.addWidget(tracking_group)

        # --- Group for Scenario 1 Parameters ---
        # layout.addWidget(s1_group)

        # --- Optical Flow (shared) Parameters ---
        self.optical_basic_group = QGroupBox("Optical Flow Parameters")
        optical_basic_layout = QFormLayout(self.optical_basic_group)

        self.lk_win_size_spinbox = QSpinBox()
        self.lk_win_size_spinbox.setRange(21, 71)
        self.lk_win_size_spinbox.setSingleStep(2)
        self.lk_win_size_spinbox.setValue(41)
        self.lk_win_size_spinbox.setToolTip(
            "Lucas–Kanade window size. Larger windows tolerate more motion/deformation but blur fine detail."
        )
        optical_basic_layout.addRow("Window size (px):", self.lk_win_size_spinbox)

        self.lk_max_level_spinbox = QSpinBox()
        self.lk_max_level_spinbox.setRange(0, 6)
        self.lk_max_level_spinbox.setValue(4)
        self.lk_max_level_spinbox.setToolTip(
            "Number of pyramid levels. Higher values handle larger displacements but slow processing."
        )
        optical_basic_layout.addRow("Pyramid levels:", self.lk_max_level_spinbox)

        self.lk_iterations_spinbox = QSpinBox()
        self.lk_iterations_spinbox.setRange(10, 100)
        self.lk_iterations_spinbox.setValue(50)
        self.lk_iterations_spinbox.setToolTip(
            "Lucas–Kanade convergence criterion: maximum number of iterations."
        )
        self.lk_epsilon_spinbox = QDoubleSpinBox()
        self.lk_epsilon_spinbox.setDecimals(5)
        self.lk_epsilon_spinbox.setRange(0.0001, 0.01)
        self.lk_epsilon_spinbox.setSingleStep(0.0001)
        self.lk_epsilon_spinbox.setValue(0.01)
        self.lk_epsilon_spinbox.setToolTip(
            "Lucas–Kanade epsilon threshold: acceptable delta between iterations."
        )
        iterations_widget = QWidget()
        iterations_layout = QHBoxLayout(iterations_widget)
        iterations_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(QLabel("Iter:"))
        iterations_layout.addWidget(self.lk_iterations_spinbox)
        iterations_layout.addSpacing(8)
        iterations_layout.addWidget(QLabel("Eps:"))
        iterations_layout.addWidget(self.lk_epsilon_spinbox)
        optical_basic_layout.addRow("Iterations / epsilon:", iterations_widget)

        self.lk_min_eig_spinbox = QDoubleSpinBox()
        self.lk_min_eig_spinbox.setDecimals(6)
        self.lk_min_eig_spinbox.setRange(0.000001, 0.001)
        self.lk_min_eig_spinbox.setSingleStep(0.000001)
        self.lk_min_eig_spinbox.setValue(0.00001)
        self.lk_min_eig_spinbox.setToolTip(
            "Texture-quality threshold inside the LK window (lower = accept weaker detail but risk noise)."
        )
        optical_basic_layout.addRow("Min eigenvalue:", self.lk_min_eig_spinbox)

        self.lk_subtract_bg_checkbox = QCheckBox("Subtract background before LK")
        self.lk_subtract_bg_checkbox.setToolTip(
            "Use the loaded background image and subtract it from every frame before LK (requires a background file)."
        )
        optical_basic_layout.addRow(self.lk_subtract_bg_checkbox)

        self.lk_use_centroid_checkbox = QCheckBox("Centroid-based recentering")
        self.lk_use_centroid_checkbox.setChecked(True)
        self.lk_use_centroid_checkbox.setToolTip(
            "After each successful match, compute a local centroid to recentre the ROI and avoid clipping the spot."
        )
        optical_basic_layout.addRow(self.lk_use_centroid_checkbox)

        self.lk_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.lk_centroid_multiplier_spinbox.setDecimals(2)
        self.lk_centroid_multiplier_spinbox.setRange(1.0, 3.0)
        self.lk_centroid_multiplier_spinbox.setSingleStep(0.1)
        self.lk_centroid_multiplier_spinbox.setValue(1.5)
        self.lk_centroid_multiplier_spinbox.setToolTip("Multiplier applied to the ROI radius when extracting the centroid patch (e.g., 1.5 ~= 1.5x larger).")
        optical_basic_layout.addRow("Centroid radius multiplier:", self.lk_centroid_multiplier_spinbox)

        self.lk_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.lk_centroid_rel_thr_spinbox.setDecimals(2)
        self.lk_centroid_rel_thr_spinbox.setRange(0.1, 0.9)
        self.lk_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.lk_centroid_rel_thr_spinbox.setValue(0.4)
        self.lk_centroid_rel_thr_spinbox.setToolTip("Relative intensity threshold (with respect to the peak) for the centroid mask.")
        optical_basic_layout.addRow("Centroid relative threshold:", self.lk_centroid_rel_thr_spinbox)

        self.lk_centroid_expand_spinbox = QSpinBox()
        self.lk_centroid_expand_spinbox.setRange(0, 3)
        self.lk_centroid_expand_spinbox.setValue(1)
        self.lk_centroid_expand_spinbox.setToolTip("How many times the centroid patch may expand when it sits near the edge.")
        optical_basic_layout.addRow("Max centroid expansions:", self.lk_centroid_expand_spinbox)

        form_layout.addRow(self.optical_basic_group)

        # --- Optical Flow (skip anchor) Extras ---
        self.optical_skip_group = QGroupBox("Optical Flow (skip anchor) safeguards")
        optical_skip_layout = QFormLayout(self.optical_skip_group)

        self.lk_error_threshold_spinbox = QSpinBox()
        self.lk_error_threshold_spinbox.setRange(10, 80)
        self.lk_error_threshold_spinbox.setValue(30)
        self.lk_error_threshold_spinbox.setToolTip(
            "Maximum LK fit error (lower = stricter). Used to discard questionable matches."
        )
        optical_skip_layout.addRow("Error threshold (LK):", self.lk_error_threshold_spinbox)
        self.lkErrSpin = self.lk_error_threshold_spinbox

        self.lk_fb_checkbox = QCheckBox("Forward-Backward check")
        self.lk_fb_checkbox.setChecked(True)
        self.lk_fb_checkbox.setToolTip(
            "Validate tracks by running LK forward and backward. Improves quality at additional cost."
        )
        optical_skip_layout.addRow(self.lk_fb_checkbox)

        self.lk_fb_threshold_spinbox = QDoubleSpinBox()
        self.lk_fb_threshold_spinbox.setDecimals(1)
        self.lk_fb_threshold_spinbox.setRange(2.0, 20.0)
        self.lk_fb_threshold_spinbox.setSingleStep(0.1)
        default_fb_threshold = max(2.0, round(self.lk_win_size_spinbox.value() * 0.2, 1))
        self.lk_fb_threshold_spinbox.setValue(default_fb_threshold)
        self.lk_fb_threshold_spinbox.setToolTip(
            "Maximum closure distance allowed for the forward-backward pass (higher = more tolerant)."
        )
        optical_skip_layout.addRow("FB threshold (px):", self.lk_fb_threshold_spinbox)
        self.lkFBSpin = self.lk_fb_threshold_spinbox

        self.lk_max_skip_frames_spinbox = QSpinBox()
        self.lk_max_skip_frames_spinbox.setRange(0, 1000)
        self.lk_max_skip_frames_spinbox.setValue(0)
        self.lk_max_skip_frames_spinbox.setToolTip(
            "How many consecutive frames a point may skip before we terminate its track (0 = unlimited)."
        )
        optical_skip_layout.addRow("Max skip frames:", self.lk_max_skip_frames_spinbox)

        self.oob_behavior_combo = QComboBox()
        self.oob_behavior_combo.addItem("Ignore (skip)", "skip")
        self.oob_behavior_combo.addItem("Mark as lost", "mark_lost")
        self.oob_behavior_combo.setCurrentIndex(0)
        self.oob_behavior_combo.setToolTip(
            "Choose how to handle spots that leave the image bounds."
        )
        optical_skip_layout.addRow("Out-of-bounds:", self.oob_behavior_combo)

        self.lk_use_ncc_checkbox = QCheckBox("Enable NCC test")
        self.lk_use_ncc_checkbox.setChecked(True)
        self.lk_use_ncc_checkbox.setToolTip(
            "Run an additional Normalized Cross-Correlation check in a local patch to suppress hallucinations."
        )
        optical_skip_layout.addRow(self.lk_use_ncc_checkbox)

        self.lk_ncc_threshold_spinbox = QDoubleSpinBox()
        self.lk_ncc_threshold_spinbox.setDecimals(2)
        self.lk_ncc_threshold_spinbox.setRange(0.3, 0.95)
        self.lk_ncc_threshold_spinbox.setSingleStep(0.05)
        self.lk_ncc_threshold_spinbox.setValue(0.65)
        self.lk_ncc_threshold_spinbox.setToolTip("Minimum NCC value required to accept a match.")
        optical_skip_layout.addRow("NCC threshold:", self.lk_ncc_threshold_spinbox)

        self.lk_ncc_patch_spinbox = QSpinBox()
        self.lk_ncc_patch_spinbox.setRange(3, 6)
        self.lk_ncc_patch_spinbox.setValue(4)
        self.lk_ncc_patch_spinbox.setToolTip("Patch radius for NCC (r=3 => 7x7, r=4 => 9x9).")
        optical_skip_layout.addRow("NCC patch radius:", self.lk_ncc_patch_spinbox)

        form_layout.addRow(self.optical_skip_group)

        # --- Detection linking (Hungarian) params ---
        self.hungarian_group = QGroupBox("Detection Linking (Hungarian)")
        hung_layout = QFormLayout(self.hungarian_group)
        self.hung_max_dist_spinbox = QDoubleSpinBox()
        self.hung_max_dist_spinbox.setRange(1.0, 200.0)
        self.hung_max_dist_spinbox.setValue(20.0)
        self.hung_max_dist_spinbox.setSingleStep(1.0)
        self.hung_max_dist_spinbox.setToolTip("Maximum distance (px) to associate detections between consecutive frames.")
        hung_layout.addRow("Max distance (px):", self.hung_max_dist_spinbox)
        self.hung_max_gap_spinbox = QSpinBox()
        self.hung_max_gap_spinbox.setRange(0, 9999)
        self.hung_max_gap_spinbox.setValue(2)
        self.hung_max_gap_spinbox.setToolTip("Allowed gap (frames) to keep a track alive when no detection is matched.")
        hung_layout.addRow("Max gap (frames):", self.hung_max_gap_spinbox)
        self.hung_min_len_spinbox = QSpinBox()
        self.hung_min_len_spinbox.setRange(1, 200)
        self.hung_min_len_spinbox.setValue(2)
        self.hung_min_len_spinbox.setToolTip("Discard tracks shorter than this many detections.")
        hung_layout.addRow("Min track length:", self.hung_min_len_spinbox)
        form_layout.addRow(self.hungarian_group)

        # --- RAFT (with LK fallback) params ---
        self.raft_group = QGroupBox("RAFT (with LK fallback)")
        raft_layout = QFormLayout(self.raft_group)
        self.raft_model_path_edit = QLineEdit(str(Path("flow_models/raft-small.pth")))
        browse_btn = QPushButton("Browse…")
        def browse_raft():
            path, _ = QFileDialog.getOpenFileName(self, "Select RAFT checkpoint", "", "PyTorch (*.pth *.pt);;All Files (*)")
            if path:
                self.raft_model_path_edit.setText(path)
        browse_btn.clicked.connect(browse_raft)
        path_widget = QWidget()
        path_layout = QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.raft_model_path_edit)
        path_layout.addWidget(browse_btn)
        raft_layout.addRow("Model path:", path_widget)

        self.raft_resize_spinbox = QSpinBox()
        self.raft_resize_spinbox.setRange(128, 2048)
        self.raft_resize_spinbox.setValue(512)
        self.raft_resize_spinbox.setSingleStep(32)
        raft_layout.addRow("Resize short side:", self.raft_resize_spinbox)

        self.raft_tau_spinbox = QDoubleSpinBox()
        self.raft_tau_spinbox.setRange(0.1, 10.0)
        self.raft_tau_spinbox.setDecimals(2)
        self.raft_tau_spinbox.setValue(1.5)
        raft_layout.addRow("FB tau:", self.raft_tau_spinbox)

        self.raft_max_step_spinbox = QDoubleSpinBox()
        self.raft_max_step_spinbox.setRange(0.0, 500.0)
        self.raft_max_step_spinbox.setDecimals(1)
        self.raft_max_step_spinbox.setValue(0.0)
        self.raft_max_step_spinbox.setToolTip("0 = brak limitu, >0 ogranicza długość wektora flow.")
        raft_layout.addRow("Max step (px):", self.raft_max_step_spinbox)

        form_layout.addRow(self.raft_group)

        self.tracking_scenario_combo.currentTextChanged.connect(self._update_tracking_params_visibility)
        self._update_tracking_params_visibility(self.tracking_scenario_combo.currentText())

        # --- Action Buttons ---
        self.track_button = QPushButton("Track Spots")
        self.track_button.clicked.connect(self.run_tracking)
        form_layout.addRow(self.track_button)

        # layout.addWidget(tracking_group)

        self.visualize_button = QPushButton("Visualize Tracking")
        self.visualize_button.setEnabled(False)
        self.visualize_button.clicked.connect(self.show_tracking_visualizer)
        # layout.addWidget(self.visualize_button)
        form_layout.addRow(self.visualize_button)

        layout.addWidget(tracking_group)

        # --- Preview Checkbox ---
        self.show_processed_checkbox = QCheckBox("Show Processed Image")
        layout.addWidget(self.show_processed_checkbox)

        layout.addWidget(QLabel("Found Spots:"))
        self.spot_list_widget = QListWidget()
        self.spot_list_widget.currentItemChanged.connect(self.on_spot_selection_changed)
        layout.addWidget(self.spot_list_widget)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls_widget)
        self.analysis_dock.setWidget(scroll_area)

        self._refresh_segmentation_models()
        self._on_spot_backend_changed(self.spot_backend_combo.currentText())

    def _build_classic_find_spots_panel(self) -> QWidget:
        widget = QWidget()
        form_layout = QFormLayout(widget)

        self.wth_radius_spinbox = QSpinBox()
        self.wth_radius_spinbox.setRange(1, 50)
        self.wth_radius_spinbox.setValue(8)
        form_layout.addRow("WTH Radius:", self.wth_radius_spinbox)

        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setRange(1.0, 100.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(20.0)
        form_layout.addRow("Blur Sigma:", self.blur_sigma_spinbox)

        self.min_distance_spinbox = QSpinBox()
        self.min_distance_spinbox.setRange(1, 100)
        self.min_distance_spinbox.setValue(10)
        form_layout.addRow("Min Peak Distance:", self.min_distance_spinbox)

        self.threshold_rel_spinbox = QDoubleSpinBox()
        self.threshold_rel_spinbox.setRange(0.01, 1.0)
        self.threshold_rel_spinbox.setSingleStep(0.01)
        self.threshold_rel_spinbox.setValue(0.1)
        form_layout.addRow("Relative Threshold:", self.threshold_rel_spinbox)

        self.roi_size_spinbox = QSpinBox()
        self.roi_size_spinbox.setRange(8, 128)
        self.roi_size_spinbox.setSingleStep(2)
        self.roi_size_spinbox.setValue(32)
        form_layout.addRow("ROI Size:", self.roi_size_spinbox)

        return widget

    def _build_dl_find_spots_panel(self) -> QWidget:
        widget = QWidget()
        form_layout = QFormLayout(widget)

        model_row_widget = QWidget()
        model_row_layout = QHBoxLayout(model_row_widget)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dl_model_combo = QComboBox()
        self.dl_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dl_model_combo.setToolTip(
            "Select a trained segmentation model (.pt) placed in the segmentation_models folder."
        )
        self.dl_model_combo.currentIndexChanged.connect(self._on_dl_model_selected)
        self.dl_refresh_models_button = QPushButton("Refresh")
        self.dl_refresh_models_button.setToolTip("Re-scan the segmentation_models directory for new checkpoints.")
        self.dl_refresh_models_button.clicked.connect(self._refresh_segmentation_models)
        model_row_layout.addWidget(self.dl_model_combo)
        model_row_layout.addWidget(self.dl_refresh_models_button)
        form_layout.addRow("Model:", model_row_widget)

        self.dl_confidence_spinbox = QDoubleSpinBox()
        self.dl_confidence_spinbox.setRange(0.1, 0.99)
        self.dl_confidence_spinbox.setSingleStep(0.01)
        self.dl_confidence_spinbox.setValue(self.dl_settings.confidence)
        self.dl_confidence_spinbox.setToolTip(
            "Cut-off applied to the U-Net probability map (higher = fewer, cleaner components)."
        )
        self.dl_confidence_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "confidence", float(v))
        )
        form_layout.addRow("Confidence:", self.dl_confidence_spinbox)

        self.dl_min_area_spinbox = QSpinBox()
        self.dl_min_area_spinbox.setRange(1, 10000)
        self.dl_min_area_spinbox.setValue(self.dl_settings.min_area_px)
        self.dl_min_area_spinbox.setToolTip("Discard connected components smaller than this pixel area to avoid noise blobs.")
        self.dl_min_area_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "min_area_px", int(v))
        )
        form_layout.addRow("Min area (px):", self.dl_min_area_spinbox)

        self.dl_max_area_spinbox = QSpinBox()
        self.dl_max_area_spinbox.setRange(10, 100000)
        self.dl_max_area_spinbox.setValue(self.dl_settings.max_area_px)
        self.dl_max_area_spinbox.setToolTip("Discard unusually large components (pixels). Increase if spots cover many pixels.")
        self.dl_max_area_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "max_area_px", int(v))
        )
        form_layout.addRow("Max area (px):", self.dl_max_area_spinbox)

        self.dl_bbox_margin_spinbox = QSpinBox()
        self.dl_bbox_margin_spinbox.setRange(0, 50)
        self.dl_bbox_margin_spinbox.setValue(self.dl_settings.bbox_margin_px)
        self.dl_bbox_margin_spinbox.setToolTip("Extra padding (pixels) around each connected component when creating SpotROI rectangles.")
        self.dl_bbox_margin_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "bbox_margin_px", int(v))
        )
        form_layout.addRow("BBox margin (px):", self.dl_bbox_margin_spinbox)

        self.dl_dilate_spinbox = QSpinBox()
        self.dl_dilate_spinbox.setRange(0, 15)
        self.dl_dilate_spinbox.setValue(self.dl_settings.dilate_radius_px)
        self.dl_dilate_spinbox.setToolTip(
            "Optional dilation radius in pixels applied to the binary mask before measuring components (helps close gaps)."
        )
        self.dl_dilate_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "dilate_radius_px", int(v))
        )
        form_layout.addRow("Dilate radius (px):", self.dl_dilate_spinbox)

        info_label = QLabel(
            "Drop your trained checkpoints into <code>segmentation_models/</code>.\n"
            "The model produces a probability mask that we threshold, clean up, and convert to ROIs based on the settings above."
        )
        info_label.setWordWrap(True)
        form_layout.addRow(info_label)

        self.dl_warning_label = QLabel()
        self.dl_warning_label.setStyleSheet("color: #d9534f;")
        self.dl_warning_label.setWordWrap(True)
        form_layout.addRow(self.dl_warning_label)

        return widget

    def _build_yolo_find_spots_panel(self) -> QWidget:
        widget = QWidget()
        form_layout = QFormLayout(widget)

        model_row_widget = QWidget()
        model_row_layout = QHBoxLayout(model_row_widget)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.yolo_model_combo.setToolTip(
            "Select a YOLO checkpoint (.pt) placed in the segmentation_models folder."
        )
        self.yolo_model_combo.currentIndexChanged.connect(self._on_yolo_model_selected)
        self.yolo_refresh_models_button = QPushButton("Refresh")
        self.yolo_refresh_models_button.setToolTip("Re-scan the segmentation_models directory for YOLO checkpoints.")
        self.yolo_refresh_models_button.clicked.connect(self._refresh_segmentation_models)
        model_row_layout.addWidget(self.yolo_model_combo)
        model_row_layout.addWidget(self.yolo_refresh_models_button)
        form_layout.addRow("YOLO model:", model_row_widget)

        self.yolo_conf_spinbox = QDoubleSpinBox()
        self.yolo_conf_spinbox.setRange(0.01, 0.99)
        self.yolo_conf_spinbox.setSingleStep(0.01)
        self.yolo_conf_spinbox.setValue(self.yolo_settings.confidence)
        self.yolo_conf_spinbox.setToolTip("Detection confidence threshold (higher = fewer boxes).")
        self.yolo_conf_spinbox.valueChanged.connect(lambda v: setattr(self.yolo_settings, "confidence", float(v)))
        form_layout.addRow("Confidence:", self.yolo_conf_spinbox)

        self.yolo_iou_spinbox = QDoubleSpinBox()
        self.yolo_iou_spinbox.setRange(0.05, 0.95)
        self.yolo_iou_spinbox.setSingleStep(0.01)
        self.yolo_iou_spinbox.setValue(self.yolo_settings.iou)
        self.yolo_iou_spinbox.setToolTip("NMS IoU threshold used by YOLO.")
        self.yolo_iou_spinbox.valueChanged.connect(lambda v: setattr(self.yolo_settings, "iou", float(v)))
        form_layout.addRow("NMS IoU:", self.yolo_iou_spinbox)

        self.yolo_max_det_spinbox = QSpinBox()
        self.yolo_max_det_spinbox.setRange(1, 9999)
        self.yolo_max_det_spinbox.setValue(self.yolo_settings.max_det)
        self.yolo_max_det_spinbox.setToolTip("Maximum number of detections to keep.")
        self.yolo_max_det_spinbox.valueChanged.connect(lambda v: setattr(self.yolo_settings, "max_det", int(v)))
        form_layout.addRow("Max detections:", self.yolo_max_det_spinbox)

        self.yolo_bbox_margin_spinbox = QSpinBox()
        self.yolo_bbox_margin_spinbox.setRange(0, 50)
        self.yolo_bbox_margin_spinbox.setValue(self.yolo_settings.bbox_margin_px)
        self.yolo_bbox_margin_spinbox.setToolTip("Extra padding (px) around YOLO boxes when creating SpotROI.")
        self.yolo_bbox_margin_spinbox.valueChanged.connect(
            lambda v: setattr(self.yolo_settings, "bbox_margin_px", int(v))
        )
        form_layout.addRow("BBox margin (px):", self.yolo_bbox_margin_spinbox)

        warn = QLabel(
            "YOLO backend requires ultralytics installed. Detected boxes are converted to SpotROI; adjust confidence/IoU to control clutter."
        )
        warn.setWordWrap(True)
        form_layout.addRow(warn)

        self.yolo_warning_label = QLabel()
        self.yolo_warning_label.setStyleSheet("color: #d9534f;")
        self.yolo_warning_label.setWordWrap(True)
        form_layout.addRow(self.yolo_warning_label)

        return widget

    def _refresh_segmentation_models(self):
        model_paths: list[Path] = []
        models_dir = self.segmentation_models_dir
        if models_dir.exists():
            model_paths = sorted(models_dir.glob("*.pt"))

        # DL models
        self.dl_model_combo.blockSignals(True)
        self.dl_model_combo.clear()
        if not model_paths:
            self.dl_model_combo.addItem("No models found", userData=None)
            self.dl_model_combo.setEnabled(False)
            self._dl_models_available = False
            self.dl_settings.model_path = None
            if hasattr(self, "dl_warning_label"):
                self.dl_warning_label.setText(
                    "No models detected. Copy .pt checkpoints into the segmentation_models folder and press Refresh."
                )
        else:
            self.dl_model_combo.setEnabled(True)
            for path in model_paths:
                self.dl_model_combo.addItem(path.name, userData=str(path.resolve()))
            self._dl_models_available = True
            if self.dl_settings.model_path:
                idx = self.dl_model_combo.findData(self.dl_settings.model_path)
                if idx >= 0:
                    self.dl_model_combo.setCurrentIndex(idx)
            if not self.dl_settings.model_path and self.dl_model_combo.count() > 0:
                self.dl_model_combo.setCurrentIndex(0)
                self.dl_settings.model_path = self.dl_model_combo.currentData()
            if hasattr(self, "dl_warning_label"):
                self.dl_warning_label.setText("")
        self.dl_model_combo.blockSignals(False)

        # YOLO models
        self.yolo_model_combo.blockSignals(True)
        self.yolo_model_combo.clear()
        if not model_paths:
            self.yolo_model_combo.addItem("No models found", userData=None)
            self.yolo_model_combo.setEnabled(False)
            self._yolo_models_available = False
            self.yolo_settings.model_path = None
            if hasattr(self, "yolo_warning_label"):
                self.yolo_warning_label.setText(
                    "No YOLO models detected. Copy .pt checkpoints into segmentation_models and press Refresh."
                )
        else:
            self.yolo_model_combo.setEnabled(True)
            for path in model_paths:
                self.yolo_model_combo.addItem(path.name, userData=str(path.resolve()))
            self._yolo_models_available = True
            if self.yolo_settings.model_path:
                idx = self.yolo_model_combo.findData(self.yolo_settings.model_path)
                if idx >= 0:
                    self.yolo_model_combo.setCurrentIndex(idx)
            if not self.yolo_settings.model_path and self.yolo_model_combo.count() > 0:
                self.yolo_model_combo.setCurrentIndex(0)
                self.yolo_settings.model_path = self.yolo_model_combo.currentData()
            if hasattr(self, "yolo_warning_label"):
                self.yolo_warning_label.setText("")
        self.yolo_model_combo.blockSignals(False)

        self._update_find_spots_button_state()

    def _on_spot_backend_changed(self, backend: str):
        self.current_spot_backend = backend
        index = 0 if backend == "Classic" else 1 if backend == "Deep Learning" else 2
        self.spot_backend_stack.setCurrentIndex(index)
        self._update_find_spots_button_state()

    def _on_dl_model_selected(self, index: int):
        model_path = self.dl_model_combo.itemData(index)
        self.dl_settings.model_path = model_path
        self._update_find_spots_button_state()

    def _on_yolo_model_selected(self, index: int):
        model_path = self.yolo_model_combo.itemData(index)
        self.yolo_settings.model_path = model_path
        self._update_find_spots_button_state()

    def _update_find_spots_button_state(self):
        if not hasattr(self, "find_spots_button"):
            return
        buttons = [self.find_spots_button, getattr(self, "find_all_spots_button", None)]
        if self.current_spot_backend == "Deep Learning":
            enabled = self._dl_models_available and bool(self.dl_settings.model_path)
            for btn in buttons:
                if btn:
                    btn.setEnabled(enabled)
                    if not enabled:
                        btn.setToolTip(
                            "Add a .pt model to segmentation_models/ and click Refresh, then pick it from the list."
                        )
                    else:
                        btn.setToolTip("")
        elif self.current_spot_backend == "YOLO":
            enabled = self._yolo_models_available and bool(self.yolo_settings.model_path)
            for btn in buttons:
                if btn:
                    btn.setEnabled(enabled)
                    if not enabled:
                        btn.setToolTip(
                            "Add a YOLO .pt checkpoint to segmentation_models and click Refresh."
                        )
                    else:
                        btn.setToolTip("")
        else:
            for btn in buttons:
                if btn:
                    btn.setEnabled(True)
                    btn.setToolTip("")

    def _update_tracking_params_visibility(self, scenario: str | None = None):
        """Shows parameter groups only for the currently selected tracking algorithm."""
        if scenario is None:
            scenario = self.tracking_scenario_combo.currentText()

        show_optical_basic = scenario.startswith("Optical Flow")
        show_optical_skip = scenario == "Optical Flow (skip, anchor)"
        show_hungarian = scenario == "Detection Linking (Hungarian)"
        show_raft = scenario == "RAFT (with LK fallback)"

        if hasattr(self, "optical_basic_group"):
            self.optical_basic_group.setVisible(show_optical_basic)
        if hasattr(self, "optical_skip_group"):
            self.optical_skip_group.setVisible(show_optical_skip)
        if hasattr(self, "hungarian_group"):
            self.hungarian_group.setVisible(show_hungarian)
        if hasattr(self, "raft_group"):
            self.raft_group.setVisible(show_raft)

    def on_spot_selection_changed(self, current_item: QListWidgetItem | None, previous_item: QListWidgetItem | None):
        """Handles highlighting the ROI on the image when a spot is selected in the list."""
        if not current_item:
            self.image_viewer.highlight_roi(None) # Clear highlight if nothing is selected
            return
            
        # Retrieve the stored SpotROI object
        spot: SpotROI = current_item.data(Qt.ItemDataRole.UserRole)
        if spot:
            # Tell the viewer which spot ID to highlight
            self.image_viewer.highlight_roi(spot.id)

    def run_spot_finding(self):
        """Gathers parameters and runs the spot finding algorithm."""
        # 1. Get the current image file
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            QMessageBox.warning(self, "No Image", "Please select an image to analyze.")
            return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        spot_settings = self._collect_spot_settings()
        try:
            found_spots = self._detect_spots_on_leed_file(leed_file, spot_settings)
            leed_file.spots = found_spots
        except (DLSpotDetectionError, YOLODetectionError) as exc:
            QMessageBox.critical(self, "Detector Error", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Detector Error",
                f"Unexpected error during detection:\n{exc}",
            )
            return
        
        # 5. Update the UI with results
        self.populate_spot_list(found_spots)
        self.refresh_current_view()

    def _detect_spots_on_leed_file(self, leed_file: LeedFile, spot_settings: dict):
        """Core spot detection logic shared by single-frame and batch runs."""
        backend = spot_settings["backend"]
        main_image = load_tiff(leed_file.filepath)
        background_image = load_tiff(self.background_file.filepath) if self.background_file else None
        if main_image is None or main_image.size == 0:
            raise ValueError(f"Could not load image: {leed_file.filepath}")
        main_image = self._apply_analysis_circle_mask(main_image)
        if background_image is not None and background_image.size > 0:
            background_image = self._apply_analysis_circle_mask(background_image)
        subtract_bg = False
        if backend in ("Deep Learning", "YOLO"):
            subtract_bg = bool(spot_settings.get(backend.lower(), {}).get("subtract_bg")) or bool(
                spot_settings.get("dl", {}).get("subtract_bg") if backend == "Deep Learning" else spot_settings.get("yolo", {}).get("subtract_bg")
            )
        if subtract_bg:
            if background_image is None or background_image.size == 0:
                raise ValueError("Background file not loaded. Load a background image before subtracting.")
            main_image = self._subtract_background_image(main_image, background_image)

        if backend == "Deep Learning":
            dl_cfg = spot_settings["dl"]
            if not dl_cfg.get("model_path"):
                raise ValueError("Select a deep-learning model before running detection.")
            device = dl_cfg.get("device", "Auto")
            return detect_spots_with_model(main_image, self.metadata_mask_list, dl_cfg)
        elif backend == "YOLO":
            y_cfg = spot_settings["yolo"]
            if not y_cfg.get("model_path"):
                raise ValueError("Select a YOLO model before running detection.")
            yolo_settings = YOLOSettings(
                model_path=y_cfg.get("model_path"),
                confidence=float(y_cfg.get("confidence", 0.25)),
                iou=float(y_cfg.get("iou", 0.45)),
                max_det=int(y_cfg.get("max_det", 300)),
                bbox_margin_px=int(y_cfg.get("bbox_margin_px", 2)),
                device=y_cfg.get("device", "auto"),
            )
            return detect_spots_with_yolo(main_image, self.metadata_mask_list, yolo_settings)
        else:
            params = spot_settings["classic"]
            return find_spots(main_image, background_image, self.metadata_mask_list, params)

    def _subtract_background_image(self, main_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """Simple background subtraction with mean scaling and respecting metadata masks."""
        try:
            main_float = img_as_float(main_image)
            bg_float = rgb2gray(background_image) if background_image.ndim == 3 else img_as_float(background_image)
            main_masked_for_mean = main_float.copy()
            bg_masked_for_mean = bg_float.copy()
            for rect in self.metadata_mask_list:
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                main_masked_for_mean[y:y+h, x:x+w] = 0
                bg_masked_for_mean[y:y+h, x:x+w] = 0
            main_mean = main_masked_for_mean[main_masked_for_mean > 0].mean() if (main_masked_for_mean > 0).any() else 0
            bg_mean = bg_masked_for_mean[bg_masked_for_mean > 0].mean() if (bg_masked_for_mean > 0).any() else 0
            scale = (main_mean / bg_mean) if bg_mean > 1e-9 else 1.0
            subtracted = main_float - (bg_float * scale)
            return subtracted
        except Exception as exc:
            print(f"Background subtraction failed, returning original image: {exc}")
            return main_image

    def run_spot_finding_all(self):
        """Run spot detection on all frames in the list using the current backend."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before running batch detection.")
            return
        spot_settings = self._collect_spot_settings()
        files = [f for f in self.data_files if not f.is_background]
        if not files:
            QMessageBox.information(self, "No Frames", "No frames available for detection.")
            return

        progress = QProgressDialog("Detecting spots on all frames...", "Cancel", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        processed = 0
        for idx, lf in enumerate(files, start=1):
            progress.setValue(idx - 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                lf.spots = self._detect_spots_on_leed_file(lf, spot_settings)
            except Exception as exc:
                progress.cancel()
                QMessageBox.critical(self, "Detector Error", f"Failed on {lf.filename}:\n{exc}")
                return
            processed += 1
        progress.setValue(len(files))
        self.populate_file_list()
        current_item = self.file_list_widget.currentItem()
        if current_item is None and self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)
        QMessageBox.information(self, "Find All Spots", f"Processed {processed} frame(s).")

    def _collect_spot_settings(self) -> dict:
        """Serialize the current spot-finding UI state for persistence or reuse."""
        classic_settings = {
            'wth_radius': self.wth_radius_spinbox.value(),
            'blur_sigma': self.blur_sigma_spinbox.value(),
            'min_distance': self.min_distance_spinbox.value(),
            'threshold_rel': self.threshold_rel_spinbox.value(),
            'roi_size': self.roi_size_spinbox.value(),
        }

        dl_settings = {
            'model_path': self.dl_settings.model_path,
            'confidence': self.dl_confidence_spinbox.value(),
            'min_area_px': self.dl_min_area_spinbox.value(),
            'max_area_px': self.dl_max_area_spinbox.value(),
            'bbox_margin_px': self.dl_bbox_margin_spinbox.value(),
            'dilate_radius_px': self.dl_dilate_spinbox.value(),
            'subtract_bg': self.det_subtract_bg_checkbox.isChecked(),
            'device': self.dl_device_combo.currentText(),
        }

        yolo_settings = {
            'model_path': self.yolo_settings.model_path,
            'confidence': self.yolo_conf_spinbox.value(),
            'iou': self.yolo_iou_spinbox.value(),
            'max_det': self.yolo_max_det_spinbox.value(),
            'bbox_margin_px': self.yolo_bbox_margin_spinbox.value(),
            'subtract_bg': self.det_subtract_bg_checkbox.isChecked(),
            'device': self.dl_device_combo.currentText(),
        }

        return {
            'backend': self.current_spot_backend,
            'classic': classic_settings,
            'dl': dl_settings,
            'yolo': yolo_settings,
        }

    def _apply_spot_settings(self, settings: dict) -> None:
        """Apply serialized spot-finding settings back to the UI widgets."""
        classic = settings.get('classic', {})
        if classic:
            self.wth_radius_spinbox.setValue(int(classic.get('wth_radius', self.wth_radius_spinbox.value())))
            self.blur_sigma_spinbox.setValue(float(classic.get('blur_sigma', self.blur_sigma_spinbox.value())))
            self.min_distance_spinbox.setValue(int(classic.get('min_distance', self.min_distance_spinbox.value())))
            self.threshold_rel_spinbox.setValue(float(classic.get('threshold_rel', self.threshold_rel_spinbox.value())))
            self.roi_size_spinbox.setValue(int(classic.get('roi_size', self.roi_size_spinbox.value())))

        dl = settings.get('dl', {})
        if dl:
            if 'confidence' in dl:
                self.dl_confidence_spinbox.setValue(float(dl['confidence']))
            if 'min_area_px' in dl:
                self.dl_min_area_spinbox.setValue(int(dl['min_area_px']))
            if 'max_area_px' in dl:
                self.dl_max_area_spinbox.setValue(int(dl['max_area_px']))
            if 'bbox_margin_px' in dl:
                self.dl_bbox_margin_spinbox.setValue(int(dl['bbox_margin_px']))
            if 'dilate_radius_px' in dl:
                self.dl_dilate_spinbox.setValue(int(dl['dilate_radius_px']))
            if 'subtract_bg' in dl:
                self.det_subtract_bg_checkbox.setChecked(bool(dl['subtract_bg']))
            if 'device' in dl:
                idx = self.dl_device_combo.findText(str(dl['device']))
                if idx >= 0:
                    self.dl_device_combo.setCurrentIndex(idx)
            if 'model_path' in dl and dl['model_path']:
                idx = self.dl_model_combo.findData(dl['model_path'])
                if idx >= 0:
                    self.dl_model_combo.setCurrentIndex(idx)

        yolo = settings.get('yolo', {})
        if yolo:
            if 'confidence' in yolo:
                self.yolo_conf_spinbox.setValue(float(yolo['confidence']))
            if 'iou' in yolo:
                self.yolo_iou_spinbox.setValue(float(yolo['iou']))
            if 'max_det' in yolo:
                self.yolo_max_det_spinbox.setValue(int(yolo['max_det']))
            if 'bbox_margin_px' in yolo:
                self.yolo_bbox_margin_spinbox.setValue(int(yolo['bbox_margin_px']))
            if 'subtract_bg' in yolo:
                self.det_subtract_bg_checkbox.setChecked(bool(yolo['subtract_bg']))
            if 'device' in yolo:
                idx = self.dl_device_combo.findText(str(yolo['device']))
                if idx >= 0:
                    self.dl_device_combo.setCurrentIndex(idx)
            if 'model_path' in yolo and yolo['model_path']:
                idx = self.yolo_model_combo.findData(yolo['model_path'])
                if idx >= 0:
                    self.yolo_model_combo.setCurrentIndex(idx)

        backend = settings.get('backend')
        if backend:
            idx = self.spot_backend_combo.findText(backend)
            if idx >= 0:
                self.spot_backend_combo.setCurrentIndex(idx)

    def _analysis_circle_to_dict(self):
        if not self.analysis_circle:
            return None
        cx, cy, r = self.analysis_circle
        return {"cx": int(cx), "cy": int(cy), "r": int(r)}

    def _analysis_circle_from_dict(self, data):
        if not data:
            return None
        try:
            return (int(data.get("cx", 0)), int(data.get("cy", 0)), int(data.get("r", 0)))
        except Exception:
            return None

    def run_tracking(self):
        """Dispatches a tracking task to a background worker based on selected scenario."""
        selected_scenario = self.tracking_scenario_combo.currentText()
        worker = None
        cancel_event = Event()
        if "Optical Flow" in selected_scenario:
            optical_params = {"roi_size": self.roi_size_spinbox.value()}
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Optical Flow requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            subtract_background = self.lk_subtract_bg_checkbox.isChecked()
            if subtract_background and not self.background_file:
                QMessageBox.warning(
                    self,
                    "No Background",
                    "Load a background file before enabling background subtraction for Optical Flow.",
                )
                return

            optical_params.update({
                "lk_win_size": self.lk_win_size_spinbox.value(),
                "lk_max_level": self.lk_max_level_spinbox.value(),
                "lk_max_iterations": self.lk_iterations_spinbox.value(),
                "lk_epsilon": self.lk_epsilon_spinbox.value(),
                "lk_min_eigen_threshold": self.lk_min_eig_spinbox.value(),
                "use_centroid_refinement": self.lk_use_centroid_checkbox.isChecked(),
                "centroid_patch_multiplier": self.lk_centroid_multiplier_spinbox.value(),
                "centroid_rel_threshold": self.lk_centroid_rel_thr_spinbox.value(),
                "centroid_max_expand": self.lk_centroid_expand_spinbox.value(),
                "subtract_background": subtract_background,
            })
            if subtract_background and self.background_file:
                optical_params["background_filepath"] = str(self.background_file.filepath)

            if selected_scenario == "Optical Flow (skip, anchor)":
                optical_params.update({
                    "lk_err_threshold": self.lk_error_threshold_spinbox.value(),
                    "lk_fb_threshold": self.lk_fb_threshold_spinbox.value(),
                    "use_forward_backward": self.lk_fb_checkbox.isChecked(),
                    "max_skip_frames": self.lk_max_skip_frames_spinbox.value(),
                    "oob_behavior": self.oob_behavior_combo.currentData(),
                    "use_ncc_check": self.lk_use_ncc_checkbox.isChecked(),
                    "ncc_threshold": self.lk_ncc_threshold_spinbox.value(),
                    "ncc_patch_radius": self.lk_ncc_patch_spinbox.value(),
                })

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())

            if selected_scenario == "Optical Flow (skip, anchor)":
                worker = GenericWorker(
                    track_spots_optical_flow_skip_anchor,
                    file_map,
                    sorted_ids,
                    ref_file.id,
                    optical_params,
                    cancel_event=cancel_event,
                )
            else:
                worker = GenericWorker(
                    track_spots_optical_flow,
                    file_map,
                    sorted_ids,
                    ref_file.id,
                    optical_params,
                    cancel_event=cancel_event,
                )

        elif selected_scenario == "Detection Linking (Hungarian)":
            params = {
                "max_distance_px": self.hung_max_dist_spinbox.value(),
                "max_gap": self.hung_max_gap_spinbox.value(),
                "min_track_length": self.hung_min_len_spinbox.value(),
            }
            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            if not sorted_ids:
                QMessageBox.warning(self, "No Frames", "Load frames with detections before linking.")
                return
            if all((not lf.spots) for lf in file_map.values()):
                QMessageBox.warning(self, "No Detections", "Run Find Spots / Find All before linking.")
                return
            worker = GenericWorker(
                track_spots_hungarian,
                file_map,
                sorted_ids,
                params,
                cancel_event=cancel_event,
            )
        elif selected_scenario == "RAFT (with LK fallback)":
            params = {
                "model_path": self.raft_model_path_edit.text().strip(),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "raft_resize_short_side": self.raft_resize_spinbox.value(),
                "fwd_bwd_tau": self.raft_tau_spinbox.value(),
                "max_step_px": self.raft_max_step_spinbox.value() if self.raft_max_step_spinbox.value() > 0 else None,
                "roi_size": self.roi_size_spinbox.value(),
                "lk_win_size": self.lk_win_size_spinbox.value(),
                "lk_max_level": self.lk_max_level_spinbox.value(),
                "lk_max_iterations": self.lk_iterations_spinbox.value(),
                "lk_epsilon": self.lk_epsilon_spinbox.value(),
                "lk_min_eigen_threshold": self.lk_min_eig_spinbox.value(),
            }
            if not params["model_path"]:
                QMessageBox.warning(self, "No RAFT model", "Select RAFT checkpoint path before running.")
                return
            if not self._ensure_raft_on_path():
                QMessageBox.critical(
                    self,
                    "RAFT not available",
                    "RAFT code not found on PYTHONPATH. Clone https://github.com/princeton-vl/RAFT and add its 'core' folder to PYTHONPATH.",
                )
                return
            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            if not sorted_ids:
                QMessageBox.warning(self, "No Frames", "Load frames with detections before linking.")
                return
            if all((not lf.spots) for lf in file_map.values()):
                QMessageBox.warning(self, "No Detections", "Run Find Spots / Find All before tracking with RAFT.")
                return
            worker = GenericWorker(
                track_spots_raft_fallback,
                file_map,
                sorted_ids,
                params,
                cancel_event=cancel_event,
            )

        else:
            QMessageBox.information(self, "Not Implemented", "This scenario is not implemented yet.")
            return

        if worker:
            self._launch_worker_with_progress(worker, "Running tracking...", cancel_event)

    def _launch_worker_with_progress(self, worker: GenericWorker, title: str, cancel_event: Event):
        """Helper method to run any worker with a progress dialog."""
        progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        self.thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.thread)

        progress_dialog.canceled.connect(lambda: self._handle_worker_cancel(cancel_event, progress_dialog))
        self.worker.progress.connect(progress_dialog.setValue)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.error.connect(self.on_tracking_error)

        # Cleanup
        self.worker.finished.connect(progress_dialog.accept)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: setattr(self, "thread", None))
        self.thread.finished.connect(lambda: setattr(self, "worker", None))

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _handle_worker_cancel(self, cancel_event: Event, dialog: QProgressDialog):
        """Signal handler invoked when the user clicks Cancel on a progress dialog."""
        if cancel_event:
            cancel_event.set()
        dialog.setLabelText("Cancelling, please wait...")
        dialog.setEnabled(False)
        if self.thread:
            try:
                self.thread.requestInterruption()
            except RuntimeError:
                # Thread already deleted, nothing to cancel.
                pass

    def _ensure_raft_on_path(self) -> bool:
        """Try to ensure RAFT code is importable by appending common paths."""
        import importlib
        if importlib.util.find_spec("raft") is not None:
            return True
        extra_paths = [
            os.environ.get("RAFT_CORE_PATH"),
            Path("third_party/RAFT/core"),
            Path("RAFT/core"),
            Path.cwd() / "third_party" / "RAFT" / "core",
            Path.cwd() / "RAFT" / "core",
        ]
        for p in extra_paths:
            if p and Path(p).exists() and str(p) not in sys.path:
                sys.path.append(str(p))
                if importlib.util.find_spec("raft") is not None:
                    return True
        return importlib.util.find_spec("raft") is not None

    def on_tracking_finished(self, tracks: List[Track] | None):
        """Slot to handle the successful completion of the tracking task."""
        if tracks is None:
            print("Tracking task was cancelled by the user.")
            QMessageBox.information(self, "Tracking Cancelled", "Tracking run was cancelled.")
            return
        print("Tracking task finished successfully.")
        self.tracking_results = tracks
        self.visualize_button.setEnabled(True)
        self._apply_tracks_to_files(tracks)
        current_item = self.file_list_widget.currentItem()
        if current_item:
            lf: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
            self.populate_spot_list(lf.spots)
            self.refresh_current_view()
        # Reset progress bar state in case it was cancelled
        QApplication.processEvents() # Process any pending events
        self.show_tracking_visualizer()

    def on_tracking_error(self, error_message: str):
        """Slot to handle errors from the tracking worker."""
        print(f"Tracking task failed: {error_message}")
        QMessageBox.critical(self, "Tracking Error", f"An error occurred during tracking:\n{error_message}")

    def show_tracking_visualizer(self):
        """
        Opens the tracking results visualization dialog and handles the returned data.
        """
        if not self.tracking_results:
            QMessageBox.information(self, "No Tracking Data", "Please run the tracking analysis first.")
            return
        
        current_roi_size = self.roi_size_spinbox.value()
        lk_win_size = self.lk_win_size_spinbox.value()
        lk_max_level = self.lk_max_level_spinbox.value()

        dialog = TrackingDialog(
            self.tracking_results, 
            self.data_files, 
            self.metadata_mask_list,
            roi_size=current_roi_size,
            lk_win_size=lk_win_size,  
            lk_max_level=lk_max_level,
            parent=self,
            save_session_callback=lambda tracks: self.save_session_dialog(tracks),
        )

        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.tracking_results = dialog.get_updated_tracks()
            print(f"Tracking data accepted. Final number of tracks: {len(self.tracking_results)}")

            QMessageBox.information(self, "Tracks Confirmed", "Corrected tracks have been saved.")
            self._apply_tracks_to_files(self.tracking_results)
            current_item = self.file_list_widget.currentItem()
            if current_item:
                lf: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
                self.populate_spot_list(lf.spots)
                self.refresh_current_view()
        else:
            print("Tracking dialog was closed without accepting changes.")

    def _apply_tracks_to_files(self, tracks: List[Track]):
        """Sync SpotROI instances from tracking results back into LeedFile objects for display."""
        spots_by_file: dict[int, list[SpotROI]] = {}
        for tr in tracks:
            for frame_id, spot in tr.spots_by_image_id.items():
                spots_by_file.setdefault(frame_id, []).append(spot)
        file_map = {f.id: f for f in self.data_files if f.id is not None}
        for fid, spots in spots_by_file.items():
            lf = file_map.get(fid)
            if lf:
                lf.spots = spots
    
    def populate_spot_list(self, spots: list[SpotROI]):
        """Populates the spot list widget with analysis results."""
        self.spot_list_widget.clear()
        for spot in spots:
            display_text = f"ID: {spot.id}, X: {spot.center_x:.1f}, Y: {spot.center_y:.1f}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, spot)
            self.spot_list_widget.addItem(item)
    
    def refresh_current_view(self):
        """Refreshes the image display for the currently selected item."""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def save_session_dialog(self, tracks_override: List[Track] | None = None):
        """Open dialog to save current session (data, masks, spots, tracks)."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory or session before saving.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            "",
            "LEEDTrack Session (*.lts.json);;JSON (*.json);;All Files (*)",
        )
        if not path_str:
            return
        try:
            self._save_session(Path(path_str), tracks_override)
            QMessageBox.information(self, "Session Saved", f"Session saved to:\n{path_str}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session:\n{e}")

    def _save_session(self, path: Path, tracks_override: List[Track] | None = None):
        tracks_list: List[Track] = []
        if isinstance(tracks_override, list):
            tracks_list = tracks_override
            self.tracking_results = tracks_override
            self._apply_tracks_to_files(tracks_override)
        elif self.tracking_results:
            tracks_list = self.tracking_results
        spot_settings = self._collect_spot_settings()
        save_session_to_file(
            path,
            self.data_files,
            self.metadata_mask_list,
            self.background_file,
            tracks_list,
            spot_settings,
            self._analysis_circle_to_dict(),
        )

    def load_session_dialog(self):
        """Open dialog to load a session from disk."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "LEEDTrack Session (*.lts.json *.json);;All Files (*)",
        )
        if not path_str:
            return
        try:
            self._load_session(Path(path_str))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{e}")

    def _load_session(self, path: Path):
        files, background, masks, tracks, spot_settings, analysis_circle = load_session_from_file(path)

        missing_files = [str(f.filepath) for f in files if not f.filepath.exists()]
        if missing_files:
            QMessageBox.warning(
                self,
                "Missing Files",
                "Some image files could not be found and were skipped:\n" + "\n".join(missing_files),
            )
        valid_files = [f for f in files if f.filepath.exists()]
        self.data_files = valid_files

        if background and background.filepath.exists():
            self.background_file = background
        else:
            self.background_file = None

        self.metadata_mask_list = masks
        self.analysis_circle = self._analysis_circle_from_dict(analysis_circle)

        valid_ids = {f.id for f in self.data_files if f.id is not None}
        filtered_tracks: List[Track] = []
        for tr in tracks:
            filtered_map = {fid: spot for fid, spot in tr.spots_by_image_id.items() if fid in valid_ids}
            if filtered_map:
                filtered_tracks.append(Track(track_id=tr.track_id, spots_by_image_id=filtered_map))
        self.tracking_results = filtered_tracks
        if filtered_tracks:
            self._apply_tracks_to_files(filtered_tracks)

        self.populate_file_list()
        if spot_settings:
            self._apply_spot_settings(spot_settings)

        if self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            self.display_selected_image(self.file_list_widget.item(0), None)
        QMessageBox.information(self, "Session Loaded", f"Loaded session from:\n{path}")

    def export_spots_to_yolo(self):
        """Export current spots to YOLO label files for all loaded frames."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory with LEED frames first.")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not target_dir:
            return
        target_path = Path(target_dir)
        labels_dir = target_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        for lf in self.data_files:
            if not lf.spots:
                skipped += 1
                continue
            img = load_tiff(lf.filepath)
            if img is None or img.size == 0:
                skipped += 1
                continue
            height, width = img.shape[:2]
            yolo_lines = []
            for spot in lf.spots:
                rect = spot.roi_rect
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                bw = w / width
                bh = h / height
                cx = float(min(max(cx, 0.0), 1.0))
                cy = float(min(max(cy, 0.0), 1.0))
                bw = float(min(max(bw, 1e-6), 1.0))
                bh = float(min(max(bh, 1e-6), 1.0))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_name = f"{lf.filename}.txt"
            (labels_dir / label_name).write_text("\n".join(yolo_lines))
            exported += 1

        QMessageBox.information(
            self,
            "YOLO Export",
            f"Exported labels for {exported} frames to {labels_dir}\nSkipped {skipped} frames without spots or unreadable images.",
        )

    def export_all_to_yolo(self):
        """Export all frames as YOLO labels + copy images (for easy val set augmentation)."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory with LEED frames first.")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not target_dir:
            return
        target_path = Path(target_dir)
        labels_dir = target_path / "labels"
        images_dir = target_path / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        for lf in self.data_files:
            if not lf.spots:
                skipped += 1
                continue
            img_path = lf.filepath
            if not img_path.exists():
                skipped += 1
                continue
            img = load_tiff(img_path)
            if img is None or img.size == 0:
                skipped += 1
                continue
            height, width = img.shape[:2]
            yolo_lines = []
            for spot in lf.spots:
                rect = spot.roi_rect
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                bw = w / width
                bh = h / height
                cx = float(min(max(cx, 0.0), 1.0))
                cy = float(min(max(cy, 0.0), 1.0))
                bw = float(min(max(bw, 1e-6), 1.0))
                bh = float(min(max(bh, 1e-6), 1.0))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            stem = Path(lf.filename).stem
            label_name = f"{stem}.txt"
            image_name = lf.filename
            (labels_dir / label_name).write_text("\n".join(yolo_lines))
            shutil.copy2(img_path, images_dir / image_name)
            exported += 1

        QMessageBox.information(
            self,
            "YOLO Export",
            f"Exported {exported} frames to {target_path} (images + labels). Skipped {skipped} frames without spots or unreadable images.",
        )
