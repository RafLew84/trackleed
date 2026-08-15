from __future__ import annotations

from pathlib import Path
from typing import Any, List
from threading import Event
import numpy as np
import shutil
from PyQt6.QtCore import Qt, QPointF, QRect, QRectF, QThread, QPoint
from PyQt6.QtGui import QAction, QPixmap, QImage, QActionGroup
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGroupBox,
                             QLabel, QFileDialog, QListWidgetItem, QMessageBox, QApplication,
                             QDialog, QPushButton, QCheckBox, QToolBar, QProgressDialog,
                             QInputDialog)

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.file_parser import parse_directory
from trackleed.io.image_reader import load_tiff
from trackleed.gui.mask_dialog import MaskDialog
from trackleed.gui.spot_detection_adapter import (
    SpotDetectionCompletion,
    SpotDetectionSettingsBinding,
    build_detection_request,
    publish_detection_result,
    run_spot_detection_job,
)
from trackleed.gui.spot_detection_dialog import (
    SpotDetectionDialog,
    SpotDetectionDialogContext,
    SpotDetectionScope,
)
from trackleed.gui.spots_menu import SpotsMenuActions, add_spots_menu
from trackleed.gui.analysis_menu import AnalysisMenuActions, add_analysis_menu
from trackleed.gui.tracking_publisher import (
    clone_tracks_for_review,
    publish_tracks,
)
from trackleed.gui.seeded_tracking_controller import SeededTrackingController
from trackleed.gui.session_controller import SessionController
from trackleed.gui.detection_linking_controller import DetectionLinkingController
from trackleed.gui.tracking_menu import TrackingMenuActions, add_tracking_menu
from trackleed.logic.workflows.spot_detection import DetectionRequest
from trackleed.gui.circle_mask_dialog import CircleMaskDialog
from trackleed.gui.ellipse_mask_dialog import EllipseMaskDialog
from trackleed.gui.polygon_mask_dialog import PolygonMaskDialog
from trackleed.io.gif_loader import extract_gif_frames
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.gui.tracking_dialog import TrackingDialog
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.active_analysis_state import ActiveAnalysisState


class MainWindow(QMainWindow):
    """
    The main window class for the STAL application.
    It orchestrates the UI and connects user actions to the backend logic.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEEDTrack - Spot Tracking")
        self.setGeometry(100, 100, 1200, 800)
        self.data_files: list[LeedFile] = []
        self.background_file: LeedFile | None = None
        self.metadata_mask_list: list[QRect] = []
        self.analysis_circle: tuple[int, int, int] | None = None  # (cx, cy, r)
        self.analysis_ellipse: tuple[int, int, int, int, float] | None = None  # (cx, cy, rx, ry, angle_deg)
        self.tracking_results: List[Track] = []
        self._active_analysis_state = ActiveAnalysisState()
        self.thread: QThread | None = None
        self.worker: GenericWorker | None = None
        self._tracking_progress_dialog: QProgressDialog | None = None
        self._active_tracking_cancel_requested = False
        self.spot_detection_binding = SpotDetectionSettingsBinding()
        self.seeded_tracking_controller = SeededTrackingController(
            self,
            roi_size=self.spot_detection_binding.roi_size,
        )
        self.detection_linking_controller = DetectionLinkingController(self)
        self.session_controller = SessionController(self)
        self._bbox_transparency_pct = 0.0
        self._init_ui()
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
        load_session_action.triggered.connect(self.session_controller.load_dialog)
        file_menu.addAction(load_session_action)

        relocate_sources_action = QAction("Relocate Analysis Sources...", self)
        relocate_sources_action.triggered.connect(
            self.session_controller.relocate_sources_dialog
        )
        file_menu.addAction(relocate_sources_action)

        save_session_action = QAction("Save Session...", self)
        save_session_action.triggered.connect(self.session_controller.save_dialog)
        file_menu.addAction(save_session_action)

        quick_save_action = QAction("Quick Save", self)
        quick_save_action.setShortcut("Ctrl+S")
        quick_save_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        quick_save_action.triggered.connect(self.session_controller.quick_save)
        file_menu.addAction(quick_save_action)

        quick_load_action = QAction("Quick Load", self)
        quick_load_action.setShortcut("Ctrl+L")
        quick_load_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        quick_load_action.triggered.connect(self.session_controller.quick_load)
        file_menu.addAction(quick_load_action)

        load_bg_action = QAction("&Load Background File...", self)
        load_bg_action.triggered.connect(self.load_background_file_dialog)
        file_menu.addAction(load_bg_action)

        export_yolo_action = QAction("Export Spots to YOLO Labels...", self)
        export_yolo_action.triggered.connect(self.export_spots_to_yolo)
        file_menu.addAction(export_yolo_action)

        export_all_yolo_action = QAction("Export All (images + YOLO labels)...", self)
        export_all_yolo_action.triggered.connect(self.export_all_to_yolo)
        file_menu.addAction(export_all_yolo_action)

        clear_list_action = QAction("Clear File List", self)
        clear_list_action.setShortcut("Ctrl+Shift+C")
        clear_list_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        clear_list_action.triggered.connect(self.clear_file_list)
        file_menu.addAction(clear_list_action)

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
        set_analysis_ellipse_action = QAction("Set Analysis Ellipse...", self)
        set_analysis_ellipse_action.triggered.connect(self.open_analysis_ellipse_dialog)
        mask_menu.addAction(set_analysis_ellipse_action)
        clear_analysis_ellipse_action = QAction("Clear Analysis Ellipse", self)
        clear_analysis_ellipse_action.triggered.connect(self.clear_analysis_ellipse)
        mask_menu.addAction(clear_analysis_ellipse_action)

        # --- View Menu ---
        view_menu = menu_bar.addMenu("&View")
        bbox_transparency_action = QAction("Set BBox Transparency...", self)
        bbox_transparency_action.triggered.connect(self.prompt_bbox_transparency)
        view_menu.addAction(bbox_transparency_action)

        bbox_transparency_50_action = QAction("BBox Transparency 50%", self)
        bbox_transparency_50_action.setShortcut("Ctrl+5")
        bbox_transparency_50_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_50_action.triggered.connect(lambda: self.set_bbox_transparency(50.0))
        view_menu.addAction(bbox_transparency_50_action)

        bbox_transparency_20_action = QAction("BBox Transparency 20%", self)
        bbox_transparency_20_action.setShortcut("Ctrl+2")
        bbox_transparency_20_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_20_action.triggered.connect(lambda: self.set_bbox_transparency(20.0))
        view_menu.addAction(bbox_transparency_20_action)

        bbox_transparency_80_action = QAction("BBox Transparency 80%", self)
        bbox_transparency_80_action.setShortcut("Ctrl+8")
        bbox_transparency_80_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_80_action.triggered.connect(lambda: self.set_bbox_transparency(80.0))
        view_menu.addAction(bbox_transparency_80_action)

        bbox_transparency_off_action = QAction("BBox Transparency Off", self)
        bbox_transparency_off_action.setShortcut("Ctrl+0")
        bbox_transparency_off_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_off_action.triggered.connect(lambda: self.set_bbox_transparency(0.0))
        view_menu.addAction(bbox_transparency_off_action)

        self._add_workflow_menus(menu_bar)

    def _add_workflow_menus(self, menu_bar) -> None:
        add_spots_menu(menu_bar, self, self._spots_menu_actions())
        add_tracking_menu(
            menu_bar,
            self,
            TrackingMenuActions(
                seeded=self.seeded_tracking_controller.show_dialog,
                hungarian=self.detection_linking_controller.show_hungarian_dialog,
                directional=self.detection_linking_controller.show_directional_dialog,
                review=self.show_tracking_visualizer,
            ),
        )
        add_analysis_menu(
            menu_bar,
            self,
            AnalysisMenuActions(
                tracks=self.show_analysis_dialog,
                positions=self.show_positions_analysis_dialog,
            ),
        )

    def _spots_menu_actions(self) -> SpotsMenuActions:
        return SpotsMenuActions(
            detect=self.show_spot_detection_dialog,
            copy_to_all=self.copy_spots_to_all_frames,
            normalize_all=self.normalize_rois_all_frames,
        )

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
        self.pan_mode_action.setShortcut("P")
        self.pan_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toolbar.addAction(self.pan_mode_action)
        mode_group.addAction(self.pan_mode_action)

        # Move ROI Mode
        self.move_roi_mode_action = QAction("Move ROI", self)
        self.move_roi_mode_action.setCheckable(True)
        self.move_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('MOVE_ROI'))
        self.move_roi_mode_action.setShortcut("M")
        self.move_roi_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toolbar.addAction(self.move_roi_mode_action)
        mode_group.addAction(self.move_roi_mode_action)

        # Add ROI Mode
        self.add_roi_mode_action = QAction("Add ROI", self)
        self.add_roi_mode_action.setCheckable(True)
        self.add_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_ROI'))
        self.add_roi_mode_action.setShortcut("A")
        self.add_roi_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.add_roi_mode_action.setToolTip(
            "Create a manual ROI using the current ROI Size."
        )
        toolbar.addAction(self.add_roi_mode_action)
        mode_group.addAction(self.add_roi_mode_action)

        # Add Point Mode
        self.add_point_mode_action = QAction("Add Point", self)
        self.add_point_mode_action.setCheckable(True)
        self.add_point_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_POINT'))
        self.add_point_mode_action.setShortcut("T")
        self.add_point_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.add_point_mode_action.setToolTip(
            "Click a point to create a tracking seed. A bbox with the current ROI Size is drawn automatically around the clicked point."
        )
        toolbar.addAction(self.add_point_mode_action)
        mode_group.addAction(self.add_point_mode_action)

        # Delete ROI Mode
        self.delete_roi_mode_action = QAction("Delete ROI", self)
        self.delete_roi_mode_action.setCheckable(True)
        self.delete_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('DELETE_ROI'))
        self.delete_roi_mode_action.setShortcut("D")
        self.delete_roi_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toolbar.addAction(self.delete_roi_mode_action)
        mode_group.addAction(self.delete_roi_mode_action)

        # Cycle Pan -> Add -> Delete
        self.cycle_mode_action = QAction("Cycle Pan/Add/Delete", self)
        self.cycle_mode_action.setShortcut("L")
        self.cycle_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.cycle_mode_action.triggered.connect(self._cycle_pan_add_delete_mode)
        self.addAction(self.cycle_mode_action)

        # Cycle bbox transparency
        self.cycle_transparency_action = QAction("Cycle BBox Transparency", self)
        self.cycle_transparency_action.setShortcut("O")
        self.cycle_transparency_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.cycle_transparency_action.triggered.connect(self._cycle_bbox_transparency)
        self.addAction(self.cycle_transparency_action)

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

        self.remove_file_button = QPushButton("Remove Selected File")
        self.remove_file_button.clicked.connect(self.remove_selected_file)
        left_panel_layout.addWidget(self.remove_file_button)

        self._add_manual_spot_tools(left_panel_layout)

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

    def _add_manual_spot_tools(self, layout: QVBoxLayout) -> None:
        self.show_processed_checkbox = QCheckBox("Show Processed Image")
        layout.addWidget(self.show_processed_checkbox)

        self.copy_spots_skip_existing_checkbox = QCheckBox(
            "Skip frames that already have spots when copying"
        )
        layout.addWidget(self.copy_spots_skip_existing_checkbox)

        layout.addWidget(QLabel("Found Spots:"))
        self.spot_list_widget = QListWidget()
        self.spot_list_widget.currentItemChanged.connect(
            self.on_spot_selection_changed
        )
        layout.addWidget(self.spot_list_widget)

        self.polygon_seed_group = QGroupBox("Polygon Seed")
        polygon_layout = QVBoxLayout(self.polygon_seed_group)
        self.polygon_seed_info_label = QLabel(
            "Draw or edit a polygon seed for the selected spot."
        )
        self.polygon_seed_info_label.setWordWrap(True)
        polygon_layout.addWidget(self.polygon_seed_info_label)

        self.polygon_seed_status_label = QLabel("")
        self.polygon_seed_status_label.setWordWrap(True)
        polygon_layout.addWidget(self.polygon_seed_status_label)

        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.spot_polygon_button = QPushButton("Polygon Mask...")
        self.spot_polygon_button.clicked.connect(self.open_spot_polygon_mask_dialog)
        buttons_layout.addWidget(self.spot_polygon_button)
        self.clear_spot_polygon_button = QPushButton("Clear Polygon")
        self.clear_spot_polygon_button.clicked.connect(
            self.clear_selected_spot_polygon_mask
        )
        buttons_layout.addWidget(self.clear_spot_polygon_button)
        polygon_layout.addWidget(buttons)
        layout.addWidget(self.polygon_seed_group)
        self._update_spot_polygon_buttons()

    def _update_bg_preview(self):
        """Loads the background image and updates the preview widgets."""
        if not self.background_file:
            return
        
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
        if not current_list_item:
            return
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
        """Create a new spot from a clicked center point using the current ROI size."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Get ROI size from the GUI
        roi_size = self.spot_detection_binding.roi_size
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
            source='manual_point' if self.image_viewer.mode == 'ADD_POINT' else 'manual'
        )
        leed_file.spots.append(new_spot)
        self.image_viewer.update_rois(leed_file.spots)
        self.populate_spot_list(leed_file.spots)

    def handle_delete_roi(self, spot_id: int):
        """Slot to handle the deletion of an ROI."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            return
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
            self._active_analysis_state = self._active_analysis_state.invalidate_source()
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
        self.background_file = LeedFile(
            filepath=path_obj,
            filename=path_obj.name,
            is_background=True,
            element='', surface=0, mode='', temperature=0, energy=0, id=None, coverage=None
        )
        
        self._active_analysis_state = self._active_analysis_state.invalidate_source()
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
                self.analysis_ellipse = None
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

    def open_analysis_ellipse_dialog(self):
        """Opens a dialog to set a global analysis ellipse (applied to all images)."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list before setting an analysis ellipse.")
            return

        leed_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the image to set an analysis ellipse.")
            return

        height, width = original_image_data.shape[:2]
        if original_image_data.ndim == 2:
            q_image = QImage(original_image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            q_image = QImage(original_image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        full_res_pixmap = QPixmap.fromImage(q_image)

        existing_center = None
        existing_radius_x = 0
        existing_radius_y = 0
        existing_angle_deg = 0.0
        existing_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        if existing_ellipse:
            cx, cy, rx, ry, angle_deg = existing_ellipse
            existing_center = QPoint(cx, cy)
            existing_radius_x = rx
            existing_radius_y = ry
            existing_angle_deg = angle_deg

        dialog = EllipseMaskDialog(
            full_res_pixmap,
            existing_center,
            existing_radius_x,
            existing_radius_y,
            existing_angle_deg,
            self,
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            center, radius_x, radius_y, angle_deg = dialog.get_ellipse()
            if center and radius_x > 0 and radius_y > 0:
                self.analysis_ellipse = (
                    int(center.x()),
                    int(center.y()),
                    int(radius_x),
                    int(radius_y),
                    float(angle_deg),
                )
                self.analysis_circle = None
                print(
                    "Analysis ellipse set: "
                    f"center=({center.x()}, {center.y()}), rx={radius_x}, ry={radius_y}, angle={angle_deg:.1f}deg"
                )
            else:
                self.analysis_ellipse = None
                print("Analysis ellipse cleared (empty selection).")
            self._active_analysis_state = self._active_analysis_state.invalidate_geometry()
            self.refresh_current_view()

    def clear_analysis_ellipse(self):
        """Clears the analysis ellipse overlay and mask."""
        if self.analysis_ellipse is None:
            return
        self.analysis_ellipse = None
        self._active_analysis_state = self._active_analysis_state.invalidate_geometry()
        print("Analysis ellipse has been cleared.")
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
        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        self.analysis_ellipse = normalized_ellipse
        self.image_viewer.update_analysis_ellipse(normalized_ellipse)
        self.image_viewer.update_rois(leed_file.spots)

    def _get_base_image_for_display(self, leed_file, original_image_data):
        """Determines if original or processed image should be shown."""
        if self.show_processed_checkbox.isChecked() and not leed_file.is_background:
            try:
                bg_img = load_tiff(self.background_file.filepath) if self.background_file else None
                return self.spot_detection_binding.processed_preview(
                    original_image_data,
                    bg_img,
                    self.metadata_mask_list,
                )
            except Exception as e:
                print(f"Could not generate processed preview: {e}")
        return original_image_data
    
    def _convert_numpy_to_pixmap(self, image_data):
        """Converts a NumPy array to a QPixmap for display."""
        if image_data.dtype != np.uint8:
            min_val, max_val = image_data.min(), image_data.max()
            if max_val - min_val > 1e-9:
                image_data = 255 * (image_data - min_val) / (max_val - min_val)
            else:
                image_data = np.zeros_like(image_data)
            image_data = image_data.astype(np.uint8)

        height, width = image_data.shape[:2]
        if image_data.ndim == 3 and image_data.shape[2] == 3: # RGB
            q_image = QImage(image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        else: # Grayscale
            q_image = QImage(image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_image)

    def _apply_analysis_region_mask(self, image_data: np.ndarray) -> np.ndarray:
        """Zero out pixels outside the configured analysis circle/ellipse region."""
        if image_data is None or image_data.size == 0:
            return image_data

        out = image_data.copy()
        h, w = out.shape[:2]
        yy, xx = np.ogrid[:h, :w]

        if self.analysis_circle is not None:
            cx, cy, r = self.analysis_circle
            circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            if out.ndim == 2:
                out[~circle_mask] = 0
            else:
                out[~circle_mask, ...] = 0

        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        if normalized_ellipse is not None:
            cx, cy, rx, ry, angle_deg = normalized_ellipse
            if rx > 0 and ry > 0:
                theta = np.deg2rad(float(angle_deg))
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                dx = xx - cx
                dy = yy - cy
                xr = cos_t * dx + sin_t * dy
                yr = -sin_t * dx + cos_t * dy
                ellipse_mask = (xr * xr) / (rx * rx) + (yr * yr) / (ry * ry) <= 1.0
                if out.ndim == 2:
                    out[~ellipse_mask] = 0
                else:
                    out[~ellipse_mask, ...] = 0

        return out

    def clear_file_list(self):
        """Clear the file list and associated state."""
        self.data_files = []
        self.tracking_results = []
        self._active_analysis_state = self._active_analysis_state.clear()
        self.file_list_widget.clear()
        self.refresh_current_view()

    def remove_selected_file(self):
        """Remove the currently selected file from the list."""
        row = self.file_list_widget.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a file to remove.")
            return
        item = self.file_list_widget.takeItem(row)
        if item is None:
            return
        lf = item.data(Qt.ItemDataRole.UserRole)
        if lf in self.data_files:
            self.data_files.remove(lf)
        self._active_analysis_state = self._active_analysis_state.invalidate_source()
        if lf and lf.id is not None and self.tracking_results:
            for track in self.tracking_results:
                track.spots_by_image_id.pop(lf.id, None)
            self.tracking_results = [t for t in self.tracking_results if t.spots_by_image_id]
        self.refresh_current_view()
    
    def on_spot_selection_changed(self, current_item: QListWidgetItem | None, previous_item: QListWidgetItem | None):
        """Handles highlighting the ROI on the image when a spot is selected in the list."""
        if not current_item:
            self.image_viewer.highlight_roi(None) # Clear highlight if nothing is selected
            self._update_spot_polygon_buttons()
            return

        # Retrieve the stored SpotROI object
        spot: SpotROI = current_item.data(Qt.ItemDataRole.UserRole)
        if spot:
            # Tell the viewer which spot ID to highlight
            self.image_viewer.highlight_roi(spot.id)
        self._update_spot_polygon_buttons()

    def _get_selected_spot_for_polygon_edit(self) -> tuple[LeedFile | None, SpotROI | None]:
        current_file_item = self.file_list_widget.currentItem()
        current_spot_item = self.spot_list_widget.currentItem()
        if not current_file_item or not current_spot_item:
            return None, None
        leed_file: LeedFile = current_file_item.data(Qt.ItemDataRole.UserRole)
        spot: SpotROI = current_spot_item.data(Qt.ItemDataRole.UserRole)
        if leed_file is None or spot is None:
            return None, None
        return leed_file, spot

    def _update_spot_polygon_buttons(self):
        if (
            not hasattr(self, "spot_polygon_button")
            or not hasattr(self, "clear_spot_polygon_button")
            or not hasattr(self, "polygon_seed_status_label")
        ):
            return
        _, spot = self._get_selected_spot_for_polygon_edit()
        has_spot = spot is not None
        self.spot_polygon_button.setEnabled(has_spot)
        self.clear_spot_polygon_button.setEnabled(
            bool(has_spot and spot.has_polygon_seed)
        )

        if not has_spot:
            self.polygon_seed_status_label.setStyleSheet("color: #6c757d;")
            self.polygon_seed_status_label.setText(
                "Select a spot to create or edit its polygon seed."
            )
            return
        if spot.has_polygon_seed:
            self.polygon_seed_status_label.setStyleSheet("color: #5cb85c;")
            self.polygon_seed_status_label.setText(
                f"Spot {spot.id}: polygon ready ({len(spot.polygon_seed_points)} vertices)."
            )
            return
        self.polygon_seed_status_label.setStyleSheet("color: #d9534f;")
        self.polygon_seed_status_label.setText(
            f"Spot {spot.id}: polygon missing. Cutie/XMem2 tracking will be blocked until you draw it."
        )

    def open_spot_polygon_mask_dialog(self):
        leed_file, spot = self._get_selected_spot_for_polygon_edit()
        if leed_file is None or spot is None:
            QMessageBox.warning(self, "No Spot Selected", "Select a spot before editing its polygon mask.")
            return

        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the image to edit the polygon mask.")
            return

        full_res_pixmap = self._convert_numpy_to_pixmap(original_image_data)
        dialog = PolygonMaskDialog(
            full_res_pixmap,
            existing_points=list(getattr(spot, "polygon_seed_points", [])),
            parent=self,
        )
        result = dialog.exec()
        if result != QDialog.DialogCode.Accepted:
            return

        spot.polygon_seed_points = dialog.get_polygon_points()
        self.populate_spot_list(leed_file.spots)
        self._reselect_spot_in_list(spot.id)
        self.refresh_current_view()

    def clear_selected_spot_polygon_mask(self):
        leed_file, spot = self._get_selected_spot_for_polygon_edit()
        if leed_file is None or spot is None:
            QMessageBox.warning(self, "No Spot Selected", "Select a spot before clearing its polygon mask.")
            return
        if not spot.has_polygon_seed:
            return
        spot.polygon_seed_points = []
        self.populate_spot_list(leed_file.spots)
        self._reselect_spot_in_list(spot.id)
        self.refresh_current_view()

    def _reselect_spot_in_list(self, spot_id: int):
        for row in range(self.spot_list_widget.count()):
            item = self.spot_list_widget.item(row)
            spot = item.data(Qt.ItemDataRole.UserRole)
            if spot is not None and int(getattr(spot, "id", -1)) == int(spot_id):
                self.spot_list_widget.setCurrentRow(row)
                return

    def run_spot_finding(self):
        """Gathers parameters and runs the spot finding algorithm."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            QMessageBox.warning(self, "No Image", "Please select an image to analyze.")
            return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)
        self._start_spot_detection(
            (leed_file,), is_batch=False, title="Detecting spots..."
        )

    def run_spot_finding_all(self):
        """Run spot detection on all frames in the list using the current backend."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before running batch detection.")
            return
        files = tuple(f for f in self.data_files if not f.is_background)
        if not files:
            QMessageBox.information(self, "No Frames", "No frames available for detection.")
            return
        self._start_spot_detection(
            files, is_batch=True, title="Detecting spots on all frames..."
        )

    def show_spot_detection_dialog(self):
        """Open the catalog-bound detector dialog and run its accepted request."""
        current_item = self.file_list_widget.currentItem()
        current_file = (
            current_item.data(Qt.ItemDataRole.UserRole) if current_item else None
        )
        files = tuple(leed_file for leed_file in self.data_files if not leed_file.is_background)
        dialog = SpotDetectionDialog(
            context=SpotDetectionDialogContext(
                current_file=current_file,
                series_files=files,
                background_file=self.background_file,
                metadata_masks=tuple(self.metadata_mask_list),
                analysis_circle=self.analysis_circle,
                analysis_ellipse=self._normalize_analysis_ellipse(self.analysis_ellipse),
            ),
            settings=self.spot_detection_binding.dialog_settings,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.decision is None:
            return
        decision = dialog.decision
        self.spot_detection_binding.accept(decision.settings)
        is_batch = decision.scope is SpotDetectionScope.ALL_FRAMES
        title = (
            "Detecting spots on all frames..." if is_batch else "Detecting spots..."
        )
        self._start_spot_detection(
            decision.files,
            is_batch=is_batch,
            title=title,
            request=decision.request,
        )

    def _start_spot_detection(
        self,
        files,
        *,
        is_batch: bool,
        title: str,
        request: DetectionRequest | None = None,
    ):
        if self._tracking_worker_is_active():
            QMessageBox.information(self, "Busy", "Wait for the active operation to finish.")
            return
        if request is None:
            try:
                request = build_detection_request(
                    files=files,
                    spot_settings=self._collect_spot_settings(),
                    background_file=self.background_file,
                    metadata_masks=self.metadata_mask_list,
                    analysis_circle=self.analysis_circle,
                    analysis_ellipse=self._normalize_analysis_ellipse(self.analysis_ellipse),
                )
            except (TypeError, ValueError) as exc:
                QMessageBox.critical(self, "Detector Error", str(exc))
                return
        cancel_event = Event()
        worker = GenericWorker(
            run_spot_detection_job,
            request,
            load_tiff,
            cancel_event,
        )
        self._pending_spot_detection_files = files
        self._pending_spot_detection_is_batch = is_batch
        self._launch_worker_with_progress(
            worker,
            title,
            cancel_event,
            on_finished=lambda result: self._on_spot_detection_completed(
                SpotDetectionCompletion.from_worker_result(result)
            ),
            on_error=lambda message: self._on_spot_detection_completed(
                SpotDetectionCompletion.from_worker_error(message)
            ),
            cancel_label="Detection",
        )

    def _on_spot_detection_completed(self, completion: SpotDetectionCompletion):
        files = tuple(getattr(self, "_pending_spot_detection_files", ()))
        is_batch = bool(getattr(self, "_pending_spot_detection_is_batch", False))
        self._pending_spot_detection_files = ()
        self._pending_spot_detection_is_batch = False
        if completion.failed:
            QMessageBox.critical(
                self,
                "Detector Error",
                f"An error occurred during detection:\n{completion.error_message}",
            )
            return
        if completion.cancelled:
            QMessageBox.information(
                self,
                "Find Spots",
                "Detection cancelled. No frames were changed.",
            )
            return
        result = completion.successful_result
        try:
            publish_detection_result(files, result)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Detector Error", str(exc))
            return
        if not is_batch:
            self.populate_spot_list(files[0].spots)
            self.refresh_current_view()
            return
        self.populate_file_list()
        current_item = self.file_list_widget.currentItem()
        if current_item is None and self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)
        QMessageBox.information(self, "Find All Spots", f"Processed {len(files)} frame(s).")

    def copy_spots_to_all_frames(self):
        """Copy spots from the currently selected frame to all other frames."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before copying spots.")
            return
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Frame Selected", "Select a frame to copy spots from.")
            return
        source_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        if not source_file or not source_file.spots:
            QMessageBox.warning(self, "No Spots", "The selected frame has no spots to copy.")
            return

        skip_existing = self.copy_spots_skip_existing_checkbox.isChecked()
        other_files_with_spots = [
            lf for lf in self.data_files
            if lf is not source_file and lf.spots
        ]
        if other_files_with_spots and not skip_existing:
            reply = QMessageBox.question(
                self,
                "Overwrite Spots",
                "Some frames already contain spots. Overwrite them with the current frame's spots?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        copied = 0
        skipped = 0
        for lf in self.data_files:
            if lf is source_file:
                continue
            if skip_existing and lf.spots:
                skipped += 1
                continue
            new_spots = []
            for spot in source_file.spots:
                rect = QRect(spot.roi_rect)
                new_spots.append(
                    SpotROI(
                        id=spot.id,
                        roi_rect=rect,
                        center_x=spot.center_x,
                        center_y=spot.center_y,
                        intensity=spot.intensity,
                        source="manual_copy",
                    )
                )
            lf.spots = new_spots
            copied += 1

        self.populate_spot_list(source_file.spots)
        self.refresh_current_view()
        msg = f"Copied spots to {copied} frame(s)."
        if skipped:
            msg += f" Skipped {skipped} frame(s) with existing spots."
        QMessageBox.information(self, "Copy Spots", msg)

    @staticmethod
    def _square_roi_from_center(
        center_x: float,
        center_y: float,
        target_size: int,
        image_w: int,
        image_h: int,
    ) -> QRect:
        """Create a square ROI centered at (center_x, center_y), clamped to image bounds."""
        if image_w <= 0 or image_h <= 0:
            return QRect(0, 0, 1, 1)

        size = int(max(1, target_size))
        size = min(size, image_w, image_h)
        half = size / 2.0

        x = int(round(float(center_x) - half))
        y = int(round(float(center_y) - half))
        x = max(0, min(max(0, image_w - size), x))
        y = max(0, min(max(0, image_h - size), y))
        return QRect(int(x), int(y), int(size), int(size))

    def normalize_rois_all_frames(self):
        """Normalize all spot ROIs to the current ROI Size while preserving per-spot centers."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before normalizing ROIs.")
            return

        target_size = self.spot_detection_binding.roi_size
        if target_size <= 0:
            QMessageBox.warning(self, "Invalid ROI Size", "ROI Size must be greater than 0.")
            return

        updated_frames = 0
        updated_spots = 0
        skipped_frames = 0

        for lf in self.data_files:
            if lf.is_background or not lf.spots:
                continue

            image_data = load_tiff(lf.filepath)
            if image_data is None or image_data.size == 0:
                skipped_frames += 1
                continue

            image_h, image_w = image_data.shape[:2]
            frame_changed = False

            for spot in lf.spots:
                rect_center = spot.roi_rect.center()
                try:
                    cx_raw = float(spot.center_x)
                except Exception:
                    cx_raw = float("nan")
                try:
                    cy_raw = float(spot.center_y)
                except Exception:
                    cy_raw = float("nan")
                cx = cx_raw if np.isfinite(cx_raw) else float(rect_center.x())
                cy = cy_raw if np.isfinite(cy_raw) else float(rect_center.y())

                new_rect = self._square_roi_from_center(cx, cy, target_size, image_w, image_h)
                if spot.roi_rect != new_rect:
                    frame_changed = True
                spot.roi_rect = new_rect
                updated_spots += 1

            if frame_changed:
                updated_frames += 1

        self.populate_file_list()
        current_item = self.file_list_widget.currentItem()
        if current_item is None and self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

        msg = (
            f"Normalized {updated_spots} spot ROI(s) to size {target_size}px "
            f"across {updated_frames} frame(s)."
        )
        if skipped_frames:
            msg += f" Skipped {skipped_frames} frame(s) due to unreadable images."
        QMessageBox.information(self, "Normalize ROIs", msg)

    def _collect_spot_settings(self) -> dict:
        """Serialize the current spot-finding UI state for persistence or reuse."""
        return self.spot_detection_binding.to_mapping()

    def _apply_spot_settings(self, settings: dict) -> None:
        """Restore accepted spot-detection settings from a session mapping."""
        self.spot_detection_binding.restore(settings)

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

    def _normalize_analysis_ellipse(
        self, data: Any
    ) -> tuple[int, int, int, int, float] | None:
        """Coerce ellipse payload to (cx, cy, rx, ry, angle_deg)."""
        if data is None:
            return None

        def _to_float(value: Any) -> float | None:
            try:
                out = float(value)
            except Exception:
                return None
            if not np.isfinite(out):
                return None
            return out

        def _xy_from_any(value: Any) -> tuple[Any, Any]:
            if value is None:
                return None, None
            if hasattr(value, "x") and hasattr(value, "y"):
                try:
                    return value.x(), value.y()
                except Exception:
                    pass
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                return value[0], value[1]
            return None, None

        cx_raw = cy_raw = rx_raw = ry_raw = angle_raw = None

        if isinstance(data, dict):
            cx_raw, cy_raw = _xy_from_any(data.get("center"))
            if cx_raw is None:
                cx_raw = data.get("cx", data.get("center_x"))
            if cy_raw is None:
                cy_raw = data.get("cy", data.get("center_y"))
            rx_raw = data.get("rx", data.get("radius_x", data.get("a")))
            ry_raw = data.get("ry", data.get("radius_y", data.get("b")))
            angle_raw = data.get(
                "angle_deg",
                data.get("angle", data.get("theta_deg", data.get("theta", 0.0))),
            )
        elif isinstance(data, np.ndarray):
            return self._normalize_analysis_ellipse(data.tolist())
        elif isinstance(data, (tuple, list)):
            if len(data) >= 5:
                cx_raw, cy_raw, rx_raw, ry_raw, angle_raw = data[:5]
            elif len(data) >= 4:
                cx_raw, cy_raw = _xy_from_any(data[0])
                rx_raw, ry_raw, angle_raw = data[1], data[2], data[3]
                if cx_raw is None or cy_raw is None:
                    return None
            else:
                return None
        else:
            return None

        cx = _to_float(cx_raw)
        cy = _to_float(cy_raw)
        rx = _to_float(rx_raw)
        ry = _to_float(ry_raw)
        angle = _to_float(angle_raw)

        if cx is None or cy is None or rx is None or ry is None:
            return None
        if rx <= 0 or ry <= 0:
            return None
        if angle is None:
            angle = 0.0

        return (
            int(round(cx)),
            int(round(cy)),
            int(round(rx)),
            int(round(ry)),
            float(angle),
        )

    def _analysis_ellipse_to_dict(self):
        ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        if not ellipse:
            return None
        cx, cy, rx, ry, angle_deg = ellipse
        return {
            "cx": int(cx),
            "cy": int(cy),
            "rx": int(rx),
            "ry": int(ry),
            "angle_deg": float(angle_deg),
        }

    def _analysis_ellipse_from_dict(self, data):
        return self._normalize_analysis_ellipse(data)

    def set_bbox_transparency(self, percent: float):
        percent = max(0.0, min(100.0, float(percent)))
        self._bbox_transparency_pct = percent
        if hasattr(self, "image_viewer") and self.image_viewer:
            self.image_viewer.set_roi_transparency_percent(percent)

    def prompt_bbox_transparency(self):
        value, ok = QInputDialog.getDouble(
            self,
            "BBox Transparency",
            "Transparency (%): 0 = opaque, 100 = fully transparent",
            float(self._bbox_transparency_pct),
            0.0,
            100.0,
            1,
        )
        if ok:
            self.set_bbox_transparency(value)

    def _cycle_pan_add_delete_mode(self):
        current = getattr(self.image_viewer, "mode", "PAN")
        if current == "PAN":
            self.add_point_mode_action.trigger()
        elif current in {"ADD_ROI", "ADD_POINT"}:
            self.delete_roi_mode_action.trigger()
        else:
            self.pan_mode_action.trigger()

    def _cycle_bbox_transparency(self):
        steps = [0.0, 20.0, 50.0, 80.0]
        current = float(self._bbox_transparency_pct)
        for idx, value in enumerate(steps):
            if abs(current - value) < 0.5:
                next_value = steps[(idx + 1) % len(steps)]
                self.set_bbox_transparency(next_value)
                return
        self.set_bbox_transparency(0.0)

    def show_analysis_dialog(self):
        if not self.tracking_results:
            QMessageBox.information(self, "No Tracking Data", "Please run tracking before analysis.")
            return
        try:
            from trackleed.gui.analysis_dialog import AnalysisResultsDialog
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Analysis Unavailable",
                "Failed to load the analysis dialog.\n"
                f"{exc}\n\n"
                "Install requirements: pip install pyqtgraph PyOpenGL",
            )
            return
        file_map = {f.id: f for f in self.data_files if f.id is not None}
        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        self.analysis_ellipse = normalized_ellipse
        dialog = AnalysisResultsDialog(
            self.tracking_results,
            file_map,
            self.background_file,
            analysis_ellipse=normalized_ellipse,
            analysis_state=self._active_analysis_state, parent=self,
            save_session_callback=self.session_controller.save_dialog,
        )
        result = dialog.exec()
        self._active_analysis_state = dialog.analysis_state
        if result == QDialog.DialogCode.Accepted:
            self.tracking_results = dialog.tracks

    def show_positions_analysis_dialog(self):
        if not self.tracking_results:
            QMessageBox.information(self, "No Tracking Data", "Please run tracking before analysis.")
            return
        try:
            from trackleed.gui.positions_analysis_dialog import PositionsAnalysisDialog
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Positions Analysis Unavailable",
                "Failed to load the positions analysis dialog.\n"
                f"{exc}",
            )
            return

        file_map = {f.id: f for f in self.data_files if f.id is not None}
        dialog = PositionsAnalysisDialog(
            self.tracking_results,
            file_map,
            parent=self,
        )
        dialog.exec()

    def _tracking_worker_is_active(self) -> bool:
        thread = getattr(self, "thread", None)
        if thread is None:
            return False
        is_running = getattr(thread, "isRunning", None)
        if callable(is_running):
            try:
                return bool(is_running())
            except RuntimeError:
                return False
        return True

    def _clear_active_tracking_worker(
        self,
        *,
        worker_thread=None,
        worker=None,
        progress_dialog=None,
    ) -> None:
        if worker_thread is None or getattr(self, "thread", None) is worker_thread:
            self.thread = None
        if worker is None or getattr(self, "worker", None) is worker:
            self.worker = None
        if progress_dialog is None or getattr(self, "_tracking_progress_dialog", None) is progress_dialog:
            self._tracking_progress_dialog = None
        if self.thread is None and self.worker is None:
            self._active_tracking_cancel_requested = False

    def _launch_worker_with_progress(
        self,
        worker: GenericWorker,
        title: str,
        cancel_event: Event,
        *,
        on_finished=None,
        on_error=None,
        cancel_label: str = "Tracking",
    ):
        """Helper method to run any worker with a progress dialog."""
        progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.show()

        thread = QThread()
        self.thread = thread
        self.worker = worker
        self._tracking_progress_dialog = progress_dialog
        self._active_tracking_cancel_requested = False
        self.worker.moveToThread(thread)

        progress_dialog.canceled.connect(
            lambda: self._handle_worker_cancel(
                cancel_event,
                progress_dialog,
                cancel_label,
            )
        )
        self.worker.progress.connect(progress_dialog.setValue)
        self.worker.finished.connect(on_finished or self.on_tracking_finished)
        self.worker.error.connect(on_error or self.on_tracking_error)

        # Cleanup
        self.worker.finished.connect(progress_dialog.accept)
        self.worker.finished.connect(thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(progress_dialog.accept)
        self.worker.error.connect(thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda worker_thread=thread, current_worker=self.worker, current_dialog=progress_dialog: (
                self._clear_active_tracking_worker(
                    worker_thread=worker_thread,
                    worker=current_worker,
                    progress_dialog=current_dialog,
                )
            )
        )

        thread.started.connect(self.worker.run)
        thread.start()

    def _set_tracking_cancel_progress_state(
        self,
        dialog: QProgressDialog,
        operation_label: str = "Tracking",
    ) -> None:
        """Convert the active progress dialog into an indeterminate cancel-wait state."""
        dialog.setWindowTitle(f"Cancelling {operation_label}")
        dialog.setLabelText(f"Cancelling {operation_label.lower()}, please wait...")
        dialog.setRange(0, 0)
        dialog.setValue(0)
        try:
            dialog.setCancelButton(None)
        except Exception:
            # Some test doubles and styles may not expose a cancel button API.
            pass
        dialog.setEnabled(True)
        QApplication.processEvents()

    def _handle_worker_cancel(
        self,
        cancel_event: Event,
        dialog: QProgressDialog,
        operation_label: str = "Tracking",
    ):
        """Signal handler invoked when the user clicks Cancel on a progress dialog."""
        if cancel_event:
            cancel_event.set()
        self._active_tracking_cancel_requested = True
        self._set_tracking_cancel_progress_state(dialog, operation_label)
        if self.thread:
            try:
                self.thread.requestInterruption()
            except RuntimeError:
                # Thread already deleted, nothing to cancel.
                pass

    def on_tracking_finished(self, tracks: List[Track] | None):
        """Slot to handle the successful completion of the tracking task."""
        if tracks is None:
            print("Tracking task was cancelled by the user.")
            QMessageBox.information(self, "Tracking Cancelled", "Tracking run was cancelled.")
            return
        print("Tracking task finished successfully.")
        try:
            publish_tracks(self, self.data_files, tracks)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Tracking Error", str(exc))
            return
        current_item = self.file_list_widget.currentItem()
        if current_item:
            lf: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
            self.populate_spot_list(lf.spots)
            self.refresh_current_view()
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
        
        review_defaults = self.seeded_tracking_controller.review_defaults
        tracking_dialog_params = dict(
            getattr(self.seeded_tracking_controller, "review_backend_params", {})
        )
        dialog = TrackingDialog(
            clone_tracks_for_review(self.tracking_results),
            self.data_files, 
            self.metadata_mask_list,
            roi_size=review_defaults.roi_size,
            lk_win_size=review_defaults.lk_win_size,
            lk_max_level=review_defaults.lk_max_level,
            **tracking_dialog_params,
            parent=self,
            save_session_callback=self.session_controller.save_dialog,
        )

        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            try:
                publish_tracks(
                    self,
                    self.data_files,
                    dialog.get_updated_tracks(),
                )
            except (TypeError, ValueError) as exc:
                QMessageBox.critical(self, "Tracking Error", str(exc))
                return
            print(f"Tracking data accepted. Final number of tracks: {len(self.tracking_results)}")
            QMessageBox.information(self, "Tracks Confirmed", "Corrected tracks have been saved.")
            current_item = self.file_list_widget.currentItem()
            if current_item:
                lf: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
                self.populate_spot_list(lf.spots)
                self.refresh_current_view()
        else:
            print("Tracking dialog was closed without accepting changes.")

    def populate_spot_list(self, spots: list[SpotROI]):
        """Populates the spot list widget with analysis results."""
        self.spot_list_widget.clear()
        for spot in spots:
            display_text = f"ID: {spot.id}, X: {spot.center_x:.1f}, Y: {spot.center_y:.1f}"
            if getattr(spot, "has_polygon_seed", False):
                display_text += f" | poly:{len(getattr(spot, 'polygon_seed_points', []))}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, spot)
            self.spot_list_widget.addItem(item)
        self._update_spot_polygon_buttons()
    
    def refresh_current_view(self):
        """Refreshes the image display for the currently selected item."""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

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
