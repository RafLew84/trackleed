# stal/gui/main_window.py

from pathlib import Path

from typing import List
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRect, QRectF, QThread
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QBrush, QColor, QActionGroup , QIcon
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGroupBox,
                             QLabel, QFileDialog, QListWidgetItem, QMessageBox, QApplication,
                             QDialog, QDockWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
                             QPushButton, QCheckBox, QToolBar, QComboBox, QProgressDialog,
                             QScrollArea)

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.file_parser import parse_directory
from trackleed.io.image_reader import load_tiff
from trackleed.gui.mask_dialog import MaskDialog
from trackleed.analysis.spot_finding import find_spots
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.gui.tracking_dialog import TrackingDialog
from trackleed.logic.generic_worker import GenericWorker
from trackleed.analysis.tracking import (
    track_spots_hungarian,
    track_spots_optical_flow,
    track_spots_optical_flow_skip_anchor,
)

from skimage.util import img_as_float
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.morphology import white_tophat, disk


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
        self.tracking_results: List[Track] = []

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

        load_bg_action = QAction("&Load Background File...", self)
        load_bg_action.triggered.connect(self.load_background_file_dialog)
        file_menu.addAction(load_bg_action)

        # --- Mask Menu ---
        mask_menu = menu_bar.addMenu("&Mask")
        set_mask_action = QAction("&Set Metadata Mask...", self)
        set_mask_action.triggered.connect(self.open_mask_dialog)
        mask_menu.addAction(set_mask_action)
        clear_mask_action = QAction("Clear Current Mask", self)
        clear_mask_action.triggered.connect(self.clear_mask)
        mask_menu.addAction(clear_mask_action)

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
    
    def _create_analysis_panel(self):
        """Creates a dockable widget with analysis controls."""
        self.analysis_dock = QDockWidget("Analysis Parameters", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        # Create a container widget and layout for the controls
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        
        # Use a form layout for neat parameter alignment
        find_spots_group = QGroupBox("Find Spots")
        form_layout = QFormLayout(find_spots_group)
        # --- Parameter Widgets ---
        self.wth_radius_spinbox = QSpinBox()
        self.wth_radius_spinbox.setRange(1, 50)
        self.wth_radius_spinbox.setValue(8) # Default value
        form_layout.addRow("WTH Radius:", self.wth_radius_spinbox)
        
        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setRange(1.0, 100.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(20.0) # Default value
        form_layout.addRow("Blur Sigma:", self.blur_sigma_spinbox)
        
        self.min_distance_spinbox = QSpinBox()
        self.min_distance_spinbox.setRange(1, 100)
        self.min_distance_spinbox.setValue(10) # Default value
        form_layout.addRow("Min Peak Distance:", self.min_distance_spinbox)

        self.threshold_rel_spinbox = QDoubleSpinBox()
        self.threshold_rel_spinbox.setRange(0.01, 1.0)
        self.threshold_rel_spinbox.setSingleStep(0.01)
        self.threshold_rel_spinbox.setValue(0.1) # Default value
        form_layout.addRow("Relative Threshold:", self.threshold_rel_spinbox)

        self.roi_size_spinbox = QSpinBox()
        self.roi_size_spinbox.setRange(8, 128)
        self.roi_size_spinbox.setSingleStep(2)
        self.roi_size_spinbox.setValue(32) # Default value
        form_layout.addRow("ROI Size:", self.roi_size_spinbox)

        self.max_dist_spinbox = QSpinBox()
        self.max_dist_spinbox.setRange(5, 200)
        self.max_dist_spinbox.setValue(20)
        form_layout.addRow("Max Track Distance:", self.max_dist_spinbox)

        self.find_spots_button = QPushButton("Find Spots")
        self.find_spots_button.clicked.connect(self.run_spot_finding)
        form_layout.addRow(self.find_spots_button)

        layout.addWidget(find_spots_group)
        layout.addStretch() # Pushes everything up

        tracking_group = QGroupBox("Tracking")
        

        form_layout = QFormLayout(tracking_group)
        self.tracking_scenario_combo = QComboBox()
        self.tracking_scenario_combo.addItems([
            "Optical Flow Tracking",
            "Hungarian Tracking",
            "Optical Flow (skip, anchor)",
        ])
        # Backwards compatibility alias used by some dialogs/scripts.
        self.trackingAlgoCombo = self.tracking_scenario_combo
        form_layout.addRow("Tracking Algorithm:", self.tracking_scenario_combo)
        # layout.addWidget(tracking_group)

        # --- Group for Scenario 1 Parameters ---
        self.hungarian_group = QGroupBox("Hungarian Tracking Parameters")
        s1_layout = QFormLayout(self.hungarian_group)
        self.max_dist_spinbox = QSpinBox()
        self.max_dist_spinbox.setRange(5, 200); self.max_dist_spinbox.setValue(20)
        s1_layout.addRow("Max Track Distance:", self.max_dist_spinbox)
        form_layout.addRow(self.hungarian_group)
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

    def _update_tracking_params_visibility(self, scenario: str | None = None):
        """Shows parameter groups only for the currently selected tracking algorithm."""
        if scenario is None:
            scenario = self.tracking_scenario_combo.currentText()

        show_hungarian = scenario == "Hungarian Tracking"
        show_optical_basic = scenario.startswith("Optical Flow")
        show_optical_skip = scenario == "Optical Flow (skip, anchor)"

        if hasattr(self, "hungarian_group"):
            self.hungarian_group.setVisible(show_hungarian)
        if hasattr(self, "optical_basic_group"):
            self.optical_basic_group.setVisible(show_optical_basic)
        if hasattr(self, "optical_skip_group"):
            self.optical_skip_group.setVisible(show_optical_skip)

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
        
        # 2. Gather data (main image, background, mask)
        main_image = load_tiff(leed_file.filepath)
        background_image = None
        if self.background_file:
            background_image = load_tiff(self.background_file.filepath)

        # 3. Gather parameters from the GUI
        params = {
            'wth_radius': self.wth_radius_spinbox.value(),
            'blur_sigma': self.blur_sigma_spinbox.value(),
            'min_distance': self.min_distance_spinbox.value(),
            'threshold_rel': self.threshold_rel_spinbox.value(),
            'roi_size': self.roi_size_spinbox.value(),
        }

        # 4. Run the analysis in the backend
        try:
            print(f"Running analysis on {leed_file.filename} with params: {params}")
            found_spots = find_spots(main_image, background_image, self.metadata_mask_list, params)
            # Store results in the data model
            leed_file.spots = found_spots
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during spot finding:\n{e}")
            return
        
        # 5. Update the UI with results
        self.populate_spot_list(found_spots)
        self.refresh_current_view()

    def run_tracking(self):
        """Dispatches a tracking task to a background worker based on selected scenario."""
        selected_scenario = self.tracking_scenario_combo.currentText()
        worker = None
        if "Hungarian" in selected_scenario:
            all_spots = {f.id: f.spots for f in self.data_files if f.id is not None and f.spots}
            if not all_spots:
                QMessageBox.warning(self, "No Spots", "Hungarian Tracking requires spots to be defined on frames.")
                return
            max_dist = self.max_dist_spinbox.value()
            worker = GenericWorker(track_spots_hungarian, all_spots, max_dist)

        elif "Optical Flow" in selected_scenario:
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
                )
            else:
                worker = GenericWorker(
                    track_spots_optical_flow,
                    file_map,
                    sorted_ids,
                    ref_file.id,
                    optical_params,
                )

        else:
            QMessageBox.information(self, "Not Implemented", "This scenario is not implemented yet.")
            return

        if worker:
            self._launch_worker_with_progress(worker, "Running tracking...")

    def _launch_worker_with_progress(self, worker: GenericWorker, title: str):
        """Helper method to run any worker with a progress dialog."""
        progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        self.thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.thread)

        progress_dialog.canceled.connect(self.thread.requestInterruption) # Allow cancellation
        self.worker.progress.connect(progress_dialog.setValue)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.error.connect(self.on_tracking_error)

        # Cleanup
        self.worker.finished.connect(progress_dialog.accept)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def on_tracking_finished(self, tracks: List[Track]):
        """Slot to handle the successful completion of the tracking task."""
        print("Tracking task finished successfully.")
        self.tracking_results = tracks
        self.visualize_button.setEnabled(True)
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
            parent=self
        )

        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.tracking_results = dialog.get_updated_tracks()
            print(f"Tracking data accepted. Final number of tracks: {len(self.tracking_results)}")

            QMessageBox.information(self, "Tracks Confirmed", "Corrected tracks have been saved.")
        else:
            print("Tracking dialog was closed without accepting changes.")
    
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
