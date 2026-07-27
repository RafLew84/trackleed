from __future__ import annotations

# stal/gui/analysis_dialog.py

from pathlib import Path

from typing import Any, List, Dict, Optional
from threading import Event
import traceback
import pyqtgraph.opengl as gl
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from PyQt6.QtCore import Qt, QThread, QPointF
from PyQt6.QtGui import QColor, QPixmap, QImage
from PyQt6.QtWidgets import (QDialog, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QListWidgetItem, QAbstractItemView,
                             QTableWidget, QHeaderView, QCheckBox, QSpinBox, QLabel, QGridLayout,
                             QTableWidgetItem, QPushButton, QDialogButtonBox, QTabWidget, QGroupBox,
                             QSlider, QComboBox, QMessageBox, QProgressDialog, QListWidget)

# Import data models and the image viewer
from trackleed.core.data_models import Track
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.analysis_coordinator import run_full_analysis
from trackleed.core.data_models import LeedFile
from trackleed.analysis.peak_analyzer import gaussian_2d
from trackleed.analysis.result_parameters import ANALYSIS_RESULT_PARAMETERS
from trackleed.io.image_reader import load_tiff

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtWidgets import QFileDialog
from trackleed.io.results_exporter import export_full_results


class MplCanvas(FigureCanvas):
    """
    A universal Matplotlib canvas widget for embedding in a PyQt application.

    This class creates a Matplotlib figure and a subplot (either 2D or 3D)
    that can be added to any Qt layout, acting as a bridge between Matplotlib's
    plotting capabilities and the PyQt6 GUI framework. It inherits from
    Matplotlib's FigureCanvasQTAgg.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100, projection='3d'):
        """
        Initializes the canvas and creates the plotting axes.

        Args:
            parent: The parent Qt widget for this canvas. Defaults to None.
            width: The width of the figure in inches. Defaults to 5.
            height: The height of the figure in inches. Defaults to 4.
            dpi: The resolution of the figure in dots per inch. Defaults to 100.
            projection: A string that determines the type of axes to create.
                        Can be '2d' or '3d'. Defaults to '3d'.
        """
        # Create a new Matplotlib figure with a specified size and resolution.
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        # Conditionally create either a 3D or a standard 2D subplot (axes).
        if projection == '3d':
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class AnalysisResultsDialog(QDialog):
    """
    An advanced dialog for the final, quantitative analysis of spot tracks.

    This dialog serves as the main analysis hub, integrating configuration panels,
    2D/3D visualizations, and action controls into a single, interactive window.
    It allows users to configure analysis parameters, run calculations in a
    background thread, and explore the results through various plots.
    """
    def __init__(
        self,
        tracks: List[Track],
        all_files_map: Dict[int, LeedFile],
        background_file: Optional[LeedFile],
        analysis_ellipse: Optional[tuple[int, int, int, int, float]] = None,
        parent=None,
        save_session_callback=None,
    ):
        """
        Initializes the dialog, sets up the UI, and connects signals.

        Args:
            tracks: A list of Track objects containing the curated tracking data.
            all_files_map: A dictionary mapping frame IDs to LeedFile objects,
                           needed to access metadata and image paths.
            background_file: An optional LeedFile object for the background image.
            parent: The parent Qt widget.
        """
        super().__init__(parent)

        # --- Basic Window Setup ---
        self.setWindowTitle("Final Analysis & Results")
        self.setMinimumSize(1600, 900) # Use a larger default dialog size

        # Add minimize and maximize buttons to the window's title bar.
        current_flags = self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        # --- Store Core Data ---
        self.tracks = tracks
        self.all_files_map = all_files_map # Keep a local copy of the file map
        self.background_file = background_file
        self.analysis_ellipse = self._normalize_analysis_ellipse(analysis_ellipse)
        self.save_session_callback = save_session_callback
        # Placeholders for the background worker and thread.
        self.worker_thread = None
        self.worker = None
        self.analysis_cancel_event: Optional[Event] = None
        self.param_checkboxes = {} 

        # --- Pre-load and Process Background Image ---
        # For efficiency, the background image is loaded and converted to grayscale
        # only once during initialization, preventing repeated disk access.
        self.bg_image_data_gray = None
        if self.background_file:
            temp_bg_data = load_tiff(self.background_file.filepath)
            if temp_bg_data is not None:
                if temp_bg_data.ndim == 3:
                    self.bg_image_data_gray = rgb2gray(temp_bg_data)
                else:
                    self.bg_image_data_gray = img_as_float(temp_bg_data)

        # --- Main Layout Setup ---
        main_layout = QVBoxLayout(self)
        
        # The main splitter divides the dialog into a left config panel and a right visualization area.
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Create the main UI panels using helper methods.
        config_panel = self._create_config_panel()
        visualization_panel = self._create_visualization_panel()
        
        main_splitter.addWidget(config_panel)
        main_splitter.addWidget(visualization_panel)
        main_splitter.setSizes([400, 2000]) # Initial proportions for the split panes

        # Create and add the bottom panel with action buttons (Perform, Export, OK).
        action_panel = self._create_action_panel()
        main_layout.addLayout(action_panel)
        
        # Fill the configuration table with data from the tracks.
        self.populate_config_table()

        # --- Signal and Slot Connections ---
        # This section connects all user interactions to the appropriate handler methods
        # to ensure the UI updates dynamically.

        # General selection changes (main table, frame slider) trigger a full refresh.
        self.config_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.frame_slider.valueChanged.connect(self.on_selection_changed)

        # Changes in the "Dependency Plots" tab also trigger a refresh.
        self.y_axis_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.x_axis_combo.currentIndexChanged.connect(self.on_selection_changed)

        # Changes in the "Position Plot" tab trigger its specific update logic.
        self.track_checklist.itemChanged.connect(self.on_position_plot_config_changed)
        self.show_center_checkbox.stateChanged.connect(self.on_position_plot_config_changed)
        self.show_trace_checkbox.stateChanged.connect(self.on_position_plot_config_changed)
        self.show_center_trace_checkbox.stateChanged.connect(self.on_position_plot_config_changed)

        # Changes in the "Comparative Plots" tab trigger a full refresh.
        self.compare_track_checklist.itemChanged.connect(self.on_selection_changed)
        self.compare_y_axis_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.compare_x_axis_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.center_x_axis_combo.currentIndexChanged.connect(self.on_selection_changed)

        self.frame_filter_combo.currentIndexChanged.connect(self.on_table_filter_changed)
        # self.param_checklist.itemChanged.connect(self.on_table_filter_changed)

        # The "Clear Highlight" button deselects rows in the main config table.
        # self.clear_highlight_button.clicked.connect(self.config_table.clearSelection)

        self.radius_x_axis_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.radius_track_checklist.itemChanged.connect(self.on_selection_changed)

        self.show_errors_checkbox.stateChanged.connect(self.on_selection_changed)

        self._update_dialog_bg_preview()

    def _on_save_session(self):
        """Trigger saving the current session from inside the dialog."""
        if self.save_session_callback:
            self.save_session_callback(self.tracks)
        else:
            QMessageBox.information(self, "Save Session", "Save session is not available in this context.")

    def _update_dialog_bg_preview(self):
        """Refresh the background preview and cached data inside the dialog."""
        if self.background_file:
            # Update the preview widgets
            image_data = load_tiff(self.background_file.filepath)
            pixmap = self._numpy_to_pixmap(image_data)
            self.dialog_bg_preview_label.setPixmap(pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio))
            self.dialog_bg_filename_label.setText(f"Filename: {self.background_file.filename}")
            
            # Reload data into memory for subsequent processing
            temp_bg_data = load_tiff(self.background_file.filepath)
            if temp_bg_data is not None:
                if temp_bg_data.ndim == 3:
                    self.bg_image_data_gray = rgb2gray(temp_bg_data)
                else:
                    self.bg_image_data_gray = img_as_float(temp_bg_data)
        else:
            self.dialog_bg_preview_label.setText("No background")
            self.dialog_bg_filename_label.setText("Filename: N/A")
            self.bg_image_data_gray = None

    @staticmethod
    def _normalize_analysis_ellipse(
        data: Any,
    ) -> Optional[tuple[int, int, int, int, float]]:
        """Normalize ellipse payload to (cx, cy, rx, ry, angle_deg)."""
        if data is None:
            return None

        def _to_float(value: Any) -> Optional[float]:
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
            return AnalysisResultsDialog._normalize_analysis_ellipse(data.tolist())
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

    def _create_config_panel(self) -> QWidget:
        """
        Creates the left-hand panel containing the analysis configuration table.

        This panel holds a QTableWidget that allows the user to configure the
        analysis for each individual track, such as enabling/disabling it or
        setting track-specific parameters.

        Returns:
            A QWidget containing the configuration panel's UI elements.
        """
        # Create a container widget and a vertical layout for the panel.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Analysis Configuration:"))

        self.sub_bg_all_checkbox = QCheckBox("Subtract Background for All")
        self.sub_bg_all_checkbox.setChecked(True)
        self.sub_bg_all_checkbox.toggled.connect(self.toggle_all_sub_bg)
        layout.addWidget(self.sub_bg_all_checkbox)

        center_group = QGroupBox("Pattern Center Mode")
        center_layout = QHBoxLayout(center_group)
        center_layout.addWidget(QLabel("Center:"))
        self.center_mode_combo = QComboBox()
        self.center_mode_combo.addItem("Per-frame (RANSAC)", "per_frame")
        self.center_mode_combo.addItem("Per-frame (circle fit)", "circle_fit")
        self.center_mode_combo.addItem("Per-frame (circle fit + bbox fallback)", "circle_fit_bbox_fallback")
        self.center_mode_combo.addItem("Per-frame (ellipse fit)", "ellipse_fit")
        self.center_mode_combo.addItem("Per-frame (circle fit + temporal smoothing)", "circle_fit_temporal")
        self.center_mode_combo.addItem("Global median (all frames)", "global_median")
        self.center_mode_combo.addItem("Manual ellipse (global center, corrected radius)", "manual_ellipse")
        center_layout.addWidget(self.center_mode_combo)
        layout.addWidget(center_group)

        # Create the main table for configuration.
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(3)
        self.config_table.setHorizontalHeaderLabels(["Track ID", "Sub. BG", "Status"])

           # --- Configure Column Resizing Behavior ---
        # Make the last column ("Status") stretch to fill remaining space.
        self.config_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.config_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Make the other columns resize to fit their content snugly.
        self.config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # ID
        self.config_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # ID
        self.config_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # Checkbox
        
        layout.addWidget(self.config_table)
        bg_group = QGroupBox("Background Used for Analysis")
        bg_layout = QVBoxLayout(bg_group)
        
        self.dialog_bg_preview_label = QLabel("BG Preview")
        self.dialog_bg_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dialog_bg_preview_label.setMinimumSize(128, 128)
        self.dialog_bg_preview_label.setStyleSheet("background-color: #2E2E2E; color: white;")
        
        self.dialog_bg_filename_label = QLabel("Filename: N/A")
        self.dialog_bg_filename_label.setWordWrap(True)
        
        self.change_bg_button = QPushButton("Change Background...")
        self.change_bg_button.clicked.connect(self.change_background_file)

        bg_layout.addWidget(self.dialog_bg_preview_label)
        bg_layout.addWidget(self.dialog_bg_filename_label)
        bg_layout.addWidget(self.change_bg_button)
        
        layout.addWidget(bg_group)
        return container
    
    def toggle_all_sub_bg(self, checked: bool):
        """
        Sets the state of all 'Sub. BG' checkboxes in the table.

        Args:
            checked: The new state for the checkboxes (True for checked).
        """
        # Iterate over each row in the table
        for row in range(self.config_table.rowCount()):
            # Retrieve the checkbox widget stored in column 1 ("Sub. BG")
            checkbox = self.config_table.cellWidget(row, 1)
            if isinstance(checkbox, QCheckBox):
                # Match the state of the "apply to all" checkbox
                checkbox.setChecked(checked)

    def change_background_file(self):
        """Open a file picker to change the background used by this dialog."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Select New Background", "", "Image Files (*.tif *.tiff)")
        if not filepath: return

        path_obj = Path(filepath)
        self.background_file = LeedFile(
            filepath=path_obj, filename=path_obj.name, is_background=True,
            element='N/A', surface=0, mode='N/A', temperature=0,
            energy=0, id=None, coverage=None
        )
        self._update_dialog_bg_preview()
        QMessageBox.information(self, "Success", "Background file for this analysis session has been changed.")

    def _create_visualization_panel(self) -> QWidget:
        """
        Creates the main visualization panel for the dialog.

        This panel is a complex layout containing the primary 2D image preview,
        a zoomed-in ROI preview, and a tabbed widget with various data plots
        (3D views, 2D dependency plots, etc.).

        Returns:
            A QWidget containing the entire visualization panel's UI.
        """
        # Create the main container widget for this entire panel.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # This vertical splitter divides the panel into a top area (with 2D image previews)
        # and a bottom area (with the plot tabs).
        vis_splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Top Area: 2D Context and ROI Preview ---
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # This horizontal splitter divides the top area into the main image view (left)
        # and the zoomed-in ROI preview (right).
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Main 2D image viewer for showing the full frame context.
        self.image_viewer_2d = LeedImageViewer()
        
        # Widget for the zoomed-in ROI preview.
        roi_preview_container = QWidget()
        roi_preview_layout = QVBoxLayout(roi_preview_container)
        roi_preview_layout.addWidget(QLabel("Analyzed ROI Preview:"))
        self.roi_preview_label = QLabel("Select a point to preview ROI")
        self.roi_preview_label.setMinimumSize(256, 256)
        self.roi_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_preview_label.setStyleSheet("background-color: black; color: white;")
        roi_preview_layout.addWidget(self.roi_preview_label)
        
        top_splitter.addWidget(self.image_viewer_2d)
        top_splitter.addWidget(roi_preview_container)
        top_splitter.setSizes([1000, 300]) # Set initial proportions.
        
        top_layout.addWidget(top_splitter)
        vis_splitter.addWidget(top_container)
        
        # --- Bottom Area: Frame Slider and Plot Tabs ---
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        
        # Slider and labels for frame navigation and pattern center display.
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("Frame: -/-")
        self.pattern_center_label = QLabel("Pattern Center (0,0): Not calculated")
        self.pattern_center_label.setStyleSheet("font-weight: bold;")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)
        slider_layout.addStretch()
        slider_layout.addWidget(self.pattern_center_label)
        bottom_layout.addLayout(slider_layout)
        
        # The main tab widget to hold all the different plot types.
        self.tab_widget = QTabWidget()
        
        # --- Tab 1: 3D ROI Preview ---
        tab_3d = QWidget()
        tab_3d_layout = QHBoxLayout(tab_3d)
        
        # Layout for the raw data 3D plot.
        raw_plot_layout = QVBoxLayout()
        self.plot3d_raw_canvas = MplCanvas(self)
        raw_toolbar = NavigationToolbar(self.plot3d_raw_canvas, self)
        raw_plot_layout.addWidget(raw_toolbar)
        raw_plot_layout.addWidget(self.plot3d_raw_canvas)
        
        # Layout for the fitted model 3D plot.
        fit_plot_layout = QVBoxLayout()
        self.plot3d_fit_canvas = MplCanvas(self)
        fit_toolbar = NavigationToolbar(self.plot3d_fit_canvas, self)
        fit_plot_layout.addWidget(fit_toolbar)
        fit_plot_layout.addWidget(self.plot3d_fit_canvas)

        tab_3d_layout.addLayout(raw_plot_layout)
        tab_3d_layout.addLayout(fit_plot_layout)
        self.tab_widget.addTab(tab_3d, "ROI 3D Preview")
        
        # --- Tab 2: Dependency Plots ---
        tab_deps = QWidget()
        tab_deps_layout = QVBoxLayout(tab_deps)
        
        # Controls for selecting the X and Y axes.
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Y-Axis:"))
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems(list(ANALYSIS_RESULT_PARAMETERS))
        controls_layout.addWidget(self.y_axis_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("X-Axis:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["Frame ID", "Temperature", "Energy", "Coverage"])
        controls_layout.addWidget(self.x_axis_combo)
        
        # Canvas for the 2D plot.
        plot_2d_layout = QVBoxLayout()
        self.plot_2d_canvas = MplCanvas(self, projection='2d')
        plot_2d_toolbar = NavigationToolbar(self.plot_2d_canvas, self)
        plot_2d_layout.addWidget(plot_2d_toolbar)
        plot_2d_layout.addWidget(self.plot_2d_canvas)
        
        tab_deps_layout.addLayout(controls_layout)
        tab_deps_layout.addLayout(plot_2d_layout, 1)
        self.tab_widget.addTab(tab_deps, "Dependency Plots")

        # --- Tab 3: Position Plot ---
        tab_positions = QWidget()
        tab_positions_layout = QHBoxLayout(tab_positions)
        
        # Control panel with track selection list and checkboxes.
        controls_container = QWidget()
        controls_container.setMaximumWidth(200)
        controls_layout = QVBoxLayout(controls_container)
        
        self.show_center_checkbox = QCheckBox("Show Pattern Center")
        self.show_center_checkbox.setChecked(True)

        self.show_trace_checkbox = QCheckBox("Show Trace")
        self.show_trace_checkbox.setChecked(True)

        self.show_center_trace_checkbox = QCheckBox("Show Center Trace")
        self.show_center_trace_checkbox.setChecked(True)

        self.track_checklist = QListWidget()
        
        controls_layout.addWidget(QLabel("Display Tracks:"))
        controls_layout.addWidget(self.track_checklist)
        controls_layout.addWidget(self.show_center_checkbox)
        controls_layout.addWidget(self.show_trace_checkbox)
        controls_layout.addWidget(self.show_center_trace_checkbox)
        
        # Canvas for the position plot.
        position_plot_layout = QVBoxLayout()
        self.position_plot_canvas = MplCanvas(self, projection='2d')
        position_toolbar = NavigationToolbar(self.position_plot_canvas, self)
        position_plot_layout.addWidget(position_toolbar)
        position_plot_layout.addWidget(self.position_plot_canvas)
        
        tab_positions_layout.addLayout(position_plot_layout, 1)
        tab_positions_layout.addWidget(controls_container)
        
        self.tab_widget.addTab(tab_positions, "Position Plot")

        # --- Tab 4: Pattern Center ---
        tab_center = self._create_pattern_center_tab()
        self.tab_widget.addTab(tab_center, "Pattern Center")

        # --- Tab 5: Comparative Plots ---
        tab_compare = self._create_comparison_plot_tab()
        self.tab_widget.addTab(tab_compare, "Comparative Plots")

        # --- Tab 6: Results table ---
        tab_table = QWidget()
        tab_table_layout = QVBoxLayout(tab_table)

        filter_layout = QHBoxLayout()
        
        # Filtr dla Frame ID
        filter_layout.addWidget(QLabel("Filter by Frame ID:"))
        self.frame_filter_combo = QComboBox()
        filter_layout.addWidget(self.frame_filter_combo)
        
        filter_layout.addStretch()
        
        # filter_layout.addWidget(QLabel("Filter by Parameter:"))
        # self.param_checklist = QListWidget()
        # self.param_checklist.setMaximumHeight(100)

        tab_table_layout.addWidget(QLabel("Filter by Parameter:"))
        self.param_grid_layout = QGridLayout()
        self.param_grid_layout.setContentsMargins(10, 0, 10, 0)
        tab_table_layout.addLayout(self.param_grid_layout)
        
        tab_table_layout.addLayout(filter_layout)
        # tab_table_layout.addWidget(self.param_checklist) 
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Frame ID", "Parameter", "Value", "Uncertainty (+/-)"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tab_table_layout.addWidget(self.results_table)
        self.tab_widget.addTab(tab_table, "Results Table")

        tab_radius = self._create_radius_plot_tab()
        self.tab_widget.addTab(tab_radius, "Radius Plot")

        tab_radius_data = QWidget()
        tab_radius_data_layout = QVBoxLayout(tab_radius_data)
        
        self.radius_data_table = QTableWidget()
        tab_radius_data_layout.addWidget(QLabel("Radius and Uncertainty (pixels)"))
        tab_radius_data_layout.addWidget(self.radius_data_table)
        
        self.tab_widget.addTab(tab_radius_data, "Radius Data")
        
        # Final assembly of the bottom panel.
        bottom_layout.addWidget(self.tab_widget)
        vis_splitter.addWidget(bottom_container)
        vis_splitter.setSizes([500, 400])
        layout.addWidget(vis_splitter)
        return container

    def _create_pattern_center_tab(self) -> QWidget:
        """Create a dedicated tab for pattern center trajectory analysis."""
        container = QWidget()
        layout = QHBoxLayout(container)

        controls_container = QWidget()
        controls_container.setMaximumWidth(240)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.addWidget(QLabel("X-Axis:"))
        self.center_x_axis_combo = QComboBox()
        self.center_x_axis_combo.addItems(["Frame ID", "Temperature", "Energy", "Coverage"])
        controls_layout.addWidget(self.center_x_axis_combo)

        self.center_current_label = QLabel("Current center: N/A")
        self.center_current_label.setWordWrap(True)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.center_current_label)
        controls_layout.addStretch()

        plots_container = QWidget()
        plots_layout = QVBoxLayout(plots_container)

        self.center_xy_over_time_canvas = MplCanvas(self, projection='2d')
        center_time_toolbar = NavigationToolbar(self.center_xy_over_time_canvas, self)
        plots_layout.addWidget(center_time_toolbar)
        plots_layout.addWidget(self.center_xy_over_time_canvas)

        self.center_path_canvas = MplCanvas(self, projection='2d')
        center_path_toolbar = NavigationToolbar(self.center_path_canvas, self)
        plots_layout.addWidget(center_path_toolbar)
        plots_layout.addWidget(self.center_path_canvas)

        layout.addWidget(plots_container, 1)
        layout.addWidget(controls_container)
        return container
    
    def _create_radius_plot_tab(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        
        plot_layout = QVBoxLayout()
        self.radius_plot_canvas = MplCanvas(self, projection='2d')
        radius_toolbar = NavigationToolbar(self.radius_plot_canvas, self)
        plot_layout.addWidget(radius_toolbar)
        plot_layout.addWidget(self.radius_plot_canvas)
        
        controls_container = QWidget()
        controls_container.setMaximumWidth(250)
        controls_layout = QVBoxLayout(controls_container)
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("X-Axis:"))
        self.radius_x_axis_combo = QComboBox()
        self.radius_x_axis_combo.addItems(["Frame ID", "Temperature", "Energy", "Coverage"])
        axis_layout.addWidget(self.radius_x_axis_combo)
        self.show_errors_checkbox = QCheckBox("Show Error Bars")
        self.show_errors_checkbox.setChecked(True)
        self.radius_track_checklist = QListWidget()
        controls_layout.addLayout(axis_layout)
        controls_layout.addWidget(QLabel("Display Tracks:"))
        controls_layout.addWidget(self.radius_track_checklist)
        controls_layout.addWidget(self.show_errors_checkbox)
        
        layout.addLayout(plot_layout, 1)
        layout.addWidget(controls_container)
        return container
    
    def _create_comparison_plot_tab(self) -> QWidget:
        """
        Creates the UI for the 'Comparative Plots' tab.

        This tab is designed for overlaying and comparing dependency plots
        (e.g., Volume vs. Temperature) for multiple tracks simultaneously. It
        consists of a main plot area and a side control panel for configuration.

        Returns:
            A QWidget containing all the UI elements for this tab.
        """
        # Main container widget for the tab.
        container = QWidget()
        layout = QHBoxLayout(container)

        # --- Right-hand Side: Control Panel ---
        controls_container = QWidget()
        controls_container.setMaximumWidth(250)
        controls_layout = QVBoxLayout(controls_container)
        
        # Dropdown menus (QComboBox) for selecting the data for X and Y axes.
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Y:"))
        self.compare_y_axis_combo = QComboBox()
        self.compare_y_axis_combo.addItems(list(ANALYSIS_RESULT_PARAMETERS))
        axis_layout.addWidget(self.compare_y_axis_combo)
        axis_layout.addWidget(QLabel("X:"))
        self.compare_x_axis_combo = QComboBox()
        self.compare_x_axis_combo.addItems(["Frame ID", "Temperature", "Energy", "Coverage"])
        axis_layout.addWidget(self.compare_x_axis_combo)
        
        # A checklist to toggle the visibility of each track on the plot.
        self.compare_track_checklist = QListWidget()
        # A button to clear the highlight from the main configuration table.
        # self.clear_highlight_button = QPushButton("Clear Highlight")
        
        # Assemble the control panel layout.
        controls_layout.addLayout(axis_layout)
        controls_layout.addWidget(QLabel("Display Tracks:"))
        controls_layout.addWidget(self.compare_track_checklist)
        # controls_layout.addWidget(self.clear_highlight_button)
        
        # --- Left-hand Side: Plot Area ---
        # A vertical layout to hold the Matplotlib canvas and its toolbar.
        plot_layout = QVBoxLayout()
        self.compare_plot_canvas = MplCanvas(self, projection='2d')
        compare_toolbar = NavigationToolbar(self.compare_plot_canvas, self)
        plot_layout.addWidget(compare_toolbar)
        plot_layout.addWidget(self.compare_plot_canvas)
        
        # Assemble the final tab layout.
        # The plot area (plot_layout) is given a stretch factor of 1, making it
        # take up more space than the control panel.
        layout.addLayout(plot_layout, 1)
        layout.addWidget(controls_container)
        
        return container
    
    def on_selection_changed(self):
        """
        Main slot that reacts to UI changes and refreshes all visualizations.

        This method is the central handler for most user interactions in the dialog,
        such as selecting a track in the table or moving the frame slider. It
        determines the current context (selected track and frame) and calls all
        the individual plot update methods to ensure the entire UI is synchronized.
        """
        # Get the currently selected row from the main configuration table.
        current_row = self.config_table.currentRow()
        # If no row is selected (e.g., after clearing a selection), do nothing.
        if current_row < 0: return

        try:
            # --- 1. Determine the Current Context ---

            # Get the track ID and find the corresponding Track object.
            track_id = int(self.config_table.item(current_row, 0).text())
            track = next(t for t in self.tracks if t.track_id == track_id)
            
            # Get the list of frame IDs available for the selected track.
            frame_ids = sorted(track.spots_by_image_id.keys())
            if not frame_ids: return
            
            # Dynamically adjust the frame slider's range to match the selected track.
            if self.frame_slider.maximum() != len(frame_ids) - 1:
                self.frame_slider.setRange(0, len(frame_ids) - 1)
            
            # Get the current frame ID based on the slider's position.
            slider_index = self.frame_slider.value()
            if not (0 <= slider_index < len(frame_ids)): return
            
            current_frame_id = frame_ids[slider_index]

            # --- 2. Update UI Labels ---

            # Update the label showing the current frame ID.
            self.frame_label.setText(f"Frame ID: {current_frame_id}")

            # Update the label showing the calculated pattern center for this frame.
            center_info = self.all_files_map[current_frame_id].pattern_center
            if center_info:
                text = (f"Pattern Center (X,Y): "
                        f"({center_info['center_x']:.2f} +/- {center_info['error_x']:.2f}, "
                        f"{center_info['center_y']:.2f} +/- {center_info['error_y']:.2f})")
                circle_radius = center_info.get("circle_radius")
                if circle_radius is not None:
                    try:
                        circle_radius_value = float(circle_radius)
                        if np.isfinite(circle_radius_value) and circle_radius_value > 0:
                            text += f" | Circle R: {circle_radius_value:.2f}"
                    except (TypeError, ValueError):
                        pass
                self.pattern_center_label.setText(text)
            else:
                self.pattern_center_label.setText("Pattern Center (0,0): Not calculated")
            self.update_radius_plot(current_frame_id)
            
            
            # --- 3. Refresh All Visualizations ---
            # Call all the individual update methods to redraw every plot and preview.
            self.update_2d_preview(track, current_frame_id)
            self.update_3d_plots(track, current_frame_id)
            self.update_2d_plot(track, current_frame_id) # Invoke the new helper
            self.update_position_plot(current_frame_id)
            self.update_pattern_center_plots(current_frame_id)
            self.update_comparison_plot(current_frame_id)
            self.update_results_table(track)

        except (AttributeError, StopIteration, ValueError, TypeError) as e:
            print(f"Could not update plots: {e}")
            traceback.print_exc()
    
    def update_radius_plot(self, current_frame_id: int):
        """Update the radius plot, optionally showing error bars."""
        ax = self.radius_plot_canvas.axes
        ax.clear()

        show_errors = self.show_errors_checkbox.isChecked()
        
        # track_ids_to_show = {self.radius_track_checklist.item(i).data(Qt.ItemDataRole.UserRole)
        #                     for i in range(self.radius_track_checklist.count())
        #                     if self.radius_track_checklist.item(i).checkState() == Qt.CheckState.Checked}

        track_ids_to_show = {self.radius_track_checklist.item(i).data(Qt.ItemDataRole.UserRole)
                         for i in range(self.radius_track_checklist.count())
                         if self.radius_track_checklist.item(i).checkState() == Qt.CheckState.Checked}
    
        highlighted_track_id = -1
        current_row = self.config_table.currentRow()
        if current_row >= 0:
            track_id_item = self.config_table.item(current_row, 0)
            if track_id_item:
                highlighted_track_id = int(track_id_item.text())
        # Read the combo-box selection safely
        x_key = self.radius_x_axis_combo.currentText()
        x_data_key = {
            "Frame ID": "id",
            "Temperature": "temperature",
            "Energy": "energy",
            "Coverage": "coverage",
        }.get(x_key)
        if not x_data_key: return

        colors = plt.get_cmap('tab10').colors
        # Plot individual tracks
        for i, track in enumerate(self.tracks):
            if track.track_id in track_ids_to_show and hasattr(track, 'analysis_results'):
                x_data, y_data, y_error = [], [], []
                for frame_id, res in sorted(track.analysis_results.items()):
                    if not res or 'radius' not in res: continue
                    x_val = frame_id if x_data_key == 'id' else getattr(self.all_files_map[frame_id], x_data_key, None)
                    if x_val is None: continue
                    x_data.append(x_val)
                    y_data.append(res['radius'])
                    y_error.append(res.get('radius_error', 0))
                
                linewidth = 3.0 if track.track_id == highlighted_track_id else 1.5
                alpha = 1.0 if track.track_id == highlighted_track_id else 0.7

                if show_errors:
                    # Plot with error bars
                    ax.errorbar(x_data, y_data, yerr=y_error, fmt='o', markersize=4, 
                                color=colors[i % len(colors)], label=f'Track {track.track_id}', 
                                capsize=3, linewidth=linewidth, alpha=alpha)
                else:
                    # Plot without error bars
                    ax.plot(x_data, y_data, linestyle='-',
                            color=colors[i % len(colors)], label=f'Track {track.track_id}',
                            linewidth=linewidth, alpha=alpha)
                
                # ax.errorbar(x_data, y_data, yerr=y_error, fmt='o', markersize=4, 
                #             color=colors[i % len(colors)], label=f'Track {track.track_id}', 
                #             capsize=3, linewidth=linewidth, alpha=alpha)

        # Build the average series using only selected tracks
        avg_x_data, avg_y_data, avg_y_error = [], [], []
        for frame_id in sorted(self.all_files_map.keys()):
            radii = []
            for track in self.tracks:
                if track.track_id not in track_ids_to_show:
                    continue
                res = track.analysis_results.get(frame_id)
                if not res or 'radius' not in res:
                    continue
                radii.append(res['radius'])

            if len(radii) < 2:
                continue

            x_val = frame_id if x_data_key == 'id' else getattr(self.all_files_map[frame_id], x_data_key, None)
            if x_val is None:
                continue

            avg_x_data.append(x_val)
            avg_y_data.append(float(np.mean(radii)))
            avg_y_error.append(float(np.std(radii) / np.sqrt(len(radii))))

        # Plot the average radius with the correct X axis
        if avg_x_data and avg_y_data:
            sorted_indices = np.argsort(avg_x_data)
            sorted_avg_x = np.array(avg_x_data)[sorted_indices]
            sorted_avg_y = np.array(avg_y_data)[sorted_indices]
            sorted_avg_y_err = np.array(avg_y_error)[sorted_indices]

            ax.plot(sorted_avg_x, sorted_avg_y, color='black', linewidth=2, label='Average Radius')
            ax.fill_between(sorted_avg_x, sorted_avg_y - sorted_avg_y_err, sorted_avg_y + sorted_avg_y_err, color='black', alpha=0.2)
        # --- End of additional logic ---
        
        ax.set_xlabel(x_key)
        ax.set_ylabel("Radius (pixels)")
        ax.set_title("Spot Radius vs. " + x_key)
        # ax.legend() # Intentionally disabled per earlier request
        ax.grid(True)
        self.radius_plot_canvas.draw()


    def update_results_table(self, track: Track):
        """
        Fills the results table with data and uncertainties for the selected track,
        respecting the current filter settings.
        """
        self.results_table.setRowCount(0) 
        if not hasattr(track, 'analysis_results'): return

        # --- Read filter settings from the UI ---
        selected_frame = self.frame_filter_combo.currentText()
        checked_params = {name for name, checkbox in self.param_checkboxes.items() 
                        if checkbox.isChecked()}

        # --- Updated data mapping tables ---
        param_map = {
            label: result_key
            for label, (result_key, _error_key) in ANALYSIS_RESULT_PARAMETERS.items()
        }
        error_map = {
            result_key: error_key
            for result_key, error_key in ANALYSIS_RESULT_PARAMETERS.values()
            if error_key is not None
        }
        
        # --- Populate the table (loop logic unchanged) ---
        for frame_id, results in sorted(track.analysis_results.items()):
            if selected_frame != "All Frames" and str(frame_id) != selected_frame:
                continue
            if not results:
                continue
            
            for param_name, data_key in param_map.items():
                if param_name not in checked_params:
                    continue
                
                value = results.get(data_key)
                if value is None: continue
                
                error_key = error_map.get(data_key)
                error_value = results.get(error_key) if error_key else None

                row_pos = self.results_table.rowCount()
                self.results_table.insertRow(row_pos)
                
                self.results_table.setItem(row_pos, 0, QTableWidgetItem(str(frame_id)))
                self.results_table.setItem(row_pos, 1, QTableWidgetItem(param_name))
                self.results_table.setItem(row_pos, 2, QTableWidgetItem(f"{value:.4f}"))
                
                error_str = f"{error_value:.4f}" if error_value is not None else "N/A"
                self.results_table.setItem(row_pos, 3, QTableWidgetItem(error_str))


    def update_comparison_plot(self, current_frame_id: int):
        """
        Updates the comparative 2D plot for multiple tracks.

        This function is responsible for drawing the "Comparative Plots" tab. It
        overlays dependency plots for all user-selected tracks, highlights the
        track currently selected in the main config table, and marks the specific
        data points that correspond to the frame selected on the main slider.

        Args:
            current_frame_id: The ID of the frame currently selected by the slider,
                            used to highlight the active data points.
        """
        # Get the Matplotlib axes object for this plot and clear any previous content.
        ax = self.compare_plot_canvas.axes
        ax.clear()

        # --- 1. Gather Configuration from the UI ---

        # Get the set of track IDs the user has checked in this tab's checklist.
        track_ids_to_show = {self.compare_track_checklist.item(i).data(Qt.ItemDataRole.UserRole) 
                             for i in range(self.compare_track_checklist.count()) 
                             if self.compare_track_checklist.item(i).checkState() == Qt.CheckState.Checked}
        
        # Get the ID of the track that is currently selected in the main config table for highlighting.
        current_selection = self.config_table.selectedItems()
        # Corrected: Get ID from column 1 of the selected row.
        # highlighted_track_id = int(current_selection[0].text()) if current_selection and current_selection[0].column() == 1 else -1
        highlighted_track_id = -1
        current_row = self.config_table.currentRow()
        if current_row >= 0:
            # Read the track ID from column 0 of the selected row
            track_id_item = self.config_table.item(current_row, 0)
            if track_id_item:
                highlighted_track_id = int(track_id_item.text())

        # Get the user's choice for the X and Y axes from the dropdowns.
        y_key = self.y_axis_combo.model().item(self.compare_y_axis_combo.currentIndex()).text()
        x_key = self.x_axis_combo.model().item(self.compare_x_axis_combo.currentIndex()).text()
        
        # Map user-friendly names to the actual data keys in our results dictionaries.
        y_axis_map = {
            label: result_key
            for label, (result_key, _error_key) in ANALYSIS_RESULT_PARAMETERS.items()
        }
        x_axis_map = {"Frame ID": "id", "Temperature": "temperature", "Energy": "energy", "Coverage": "coverage"}
        y_data_key = y_axis_map.get(y_key)
        x_data_key = x_axis_map.get(x_key)
        if not (y_data_key and x_data_key): return

        # --- 2. Plot Data for Each Selected Track ---
        colors = plt.get_cmap('tab10').colors
        for i, track in enumerate(self.tracks):
            # Process only the tracks that are checked in the checklist.
            if track.track_id in track_ids_to_show and hasattr(track, 'analysis_results'):
                
                x_data, y_data, active_point = [], [], None
                # Iterate through the analysis results to build the data series for this track.
                for frame_id, results in sorted(track.analysis_results.items()):
                    if not results: continue

                    # Get the X-axis value (either from the frame ID or LeedFile metadata).
                    x_val = frame_id if x_data_key == 'id' else getattr(self.all_files_map[frame_id], x_data_key, None)
                    # Get the Y-axis value from the analysis results.
                    y_val = results.get(y_data_key)
                    if x_val is not None and y_val is not None:
                        x_data.append(x_val); y_data.append(y_val)
                        # Identify the point that corresponds to the current slider position.
                        if frame_id == current_frame_id: active_point = (x_val, y_val)
                
                if not x_data: continue

                # Set the line style based on whether this track is the highlighted one.
                linewidth = 3 if track.track_id == highlighted_track_id else 1.5
                alpha = 1.0 if track.track_id == highlighted_track_id else 0.7
                
                # Plot the full data series for the track.
                ax.plot(x_data, y_data, color=colors[i % len(colors)], linewidth=linewidth, alpha=alpha, label=f'Track {track.track_id}')
                
                # Draw a prominent marker on the point corresponding to the current frame.
                if active_point:
                    ax.scatter(active_point[0], active_point[1], color=colors[i % len(colors)], s=50, zorder=5)

        # --- 3. Final Plot Adjustments ---
        
        # Draw a vertical line to indicate the current slider position on the x-axis.
        ax.axvline(x=self.all_files_map[current_frame_id].id if x_data_key == 'id' else getattr(self.all_files_map[current_frame_id], x_data_key, 0), 
                   color='r', linestyle='--', linewidth=1, alpha=0.7)
        
        # Set labels, title, grid, and legend.
        ax.set_xlabel(x_key); ax.set_ylabel(y_key)
        ax.set_title(f"Comparison: {y_key} vs. {x_key}")
        ax.legend(loc='upper right'); ax.grid(True)
        self.compare_plot_canvas.draw()

    def update_position_plot(self, current_frame_id: int):
        """
        Updates the spot position plot based on the slider and checkboxes.

        This function visualizes the spatial movement (X,Y coordinates) of the
        spots. It is highly interactive and responds to multiple controls:
        - It displays only the tracks selected in its dedicated checklist.
        - It highlights the current position of each spot for the frame selected
        on the main slider.
        - It can optionally draw the historical "trace" of each spot up to the
        current frame.
        - It can optionally display the calculated pattern center.

        Args:
            current_frame_id: The ID of the frame currently selected by the main
                            slider, which determines the highlighted positions.
        """
        # Get the Matplotlib axes object and clear any previous content.
        ax = self.position_plot_canvas.axes
        ax.clear()

        # --- 1. Gather Configuration from the UI ---
        # Get the set of track IDs to display from the checklist.
        track_ids_to_show = {self.track_checklist.item(i).data(Qt.ItemDataRole.UserRole) 
                             for i in range(self.track_checklist.count()) 
                             if self.track_checklist.item(i).checkState() == Qt.CheckState.Checked}
        
        # Check if the "Show Trace" option is enabled.
        show_trace = self.show_trace_checkbox.isChecked()
        show_center_trace = self.show_center_trace_checkbox.isChecked()

        # --- 2. Set Stable Axis Limits ---
        # To prevent the plot from resizing/jumping as the slider moves, we first
        # find the global min/max coordinates across all data to set fixed limits.
        all_x, all_y = [], []
        for track in self.tracks:
            if track.track_id in track_ids_to_show and hasattr(track, 'analysis_results'):
                all_x.extend([res['global_center_x'] for res in track.analysis_results.values() if res])
                all_y.extend([res['global_center_y'] for res in track.analysis_results.values() if res])
        
        if all_x and all_y:
            # Add a 10% margin to the plot boundaries for better visualization.
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        # --- 3. Plot Selected Tracks ---
        colors = plt.get_cmap('tab10').colors
        for i, track in enumerate(self.tracks):
            if track.track_id in track_ids_to_show and hasattr(track, 'analysis_results'):
                
                # Collect the full trajectory data for the track.
                sorted_frames = sorted(track.analysis_results.items())
                full_x_pos = [res['global_center_x'] for _, res in sorted_frames if res]
                full_y_pos = [res['global_center_y'] for _, res in sorted_frames if res]

                # Optionally, draw the "trace" line from the start to the current frame.
                if show_trace:
                    frame_indices = [fid for fid, res in sorted_frames if res]
                    try:
                        # Find the index corresponding to the current frame.
                        up_to_index = frame_indices.index(current_frame_id) + 1
                        # Plot the line segment for the historical path.
                        ax.plot(full_x_pos[:up_to_index], full_y_pos[:up_to_index], 
                                color=colors[i % len(colors)], linestyle='-', alpha=0.5, marker='')
                    except ValueError:
                        pass # This occurs if the track has no data point on the current_frame_id.

                # Draw the current position as a large, prominent marker.
                current_results = track.analysis_results.get(current_frame_id)
                if current_results:
                    ax.scatter(current_results['global_center_x'], current_results['global_center_y'], 
                               color=colors[i % len(colors)], s=100, label=f'Track {track.track_id}', zorder=5)

        # --- 4. Plot the Pattern Center ---
        # Optionally, plot the calculated pattern center for the current frame.
        if self.show_center_checkbox.isChecked():
            if show_center_trace:
                center_trace_points = []
                for frame_id in sorted(self.all_files_map.keys()):
                    center_data = getattr(self.all_files_map[frame_id], "pattern_center", None)
                    if not center_data:
                        continue
                    center_trace_points.append((center_data['center_x'], center_data['center_y']))
                if len(center_trace_points) > 1:
                    ax.plot(
                        [p[0] for p in center_trace_points],
                        [p[1] for p in center_trace_points],
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Center trace',
                        zorder=4,
                    )

            center_info = self.all_files_map[current_frame_id].pattern_center
            if center_info:
                ax.scatter(center_info['center_x'], center_info['center_y'], 
                           marker='+', color='black', s=150, label='Pattern Center', zorder=6)

        # --- 5. Final Plot Adjustments ---
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        ax.set_title("Spot Positions")
        # Place the legend outside the plot area to avoid obscuring data.
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        # Adjust layout to make room for the external legend.
        self.position_plot_canvas.fig.tight_layout(rect=[0, 0, 0.85, 1])

        ax.grid(True)
        # Invert the Y-axis to match image coordinates (origin at top-left).
        ax.invert_yaxis()
        # Ensure a 1:1 aspect ratio to prevent spatial distortion.
        ax.set_aspect('equal', adjustable='box')
        self.position_plot_canvas.draw()

    def update_pattern_center_plots(self, current_frame_id: int):
        """Update dedicated plots that visualize the estimated pattern center trajectory."""
        ax_time = self.center_xy_over_time_canvas.axes
        ax_path = self.center_path_canvas.axes
        ax_time.clear()
        ax_path.clear()

        x_key = self.center_x_axis_combo.currentText()
        x_axis_map = {"Frame ID": "id", "Temperature": "temperature", "Energy": "energy", "Coverage": "coverage"}
        x_data_key = x_axis_map.get(x_key)
        if not x_data_key:
            self.center_xy_over_time_canvas.draw()
            self.center_path_canvas.draw()
            return

        records = []
        for frame_id in sorted(self.all_files_map.keys()):
            lf = self.all_files_map[frame_id]
            center_info = getattr(lf, "pattern_center", None)
            if not center_info:
                continue
            x_val = frame_id if x_data_key == 'id' else getattr(lf, x_data_key, None)
            if x_val is None:
                continue
            records.append((frame_id, x_val, center_info['center_x'], center_info['center_y']))

        if not records:
            ax_time.text(0.5, 0.5, "No pattern center data", ha='center', va='center', transform=ax_time.transAxes)
            ax_path.text(0.5, 0.5, "No pattern center data", ha='center', va='center', transform=ax_path.transAxes)
            self.center_current_label.setText("Current center: N/A")
            self.center_xy_over_time_canvas.draw()
            self.center_path_canvas.draw()
            return

        # --- Plot 1: center_x and center_y versus selected x-axis ---
        records_sorted_x = sorted(records, key=lambda row: row[1])
        x_values = [row[1] for row in records_sorted_x]
        center_x_values = [row[2] for row in records_sorted_x]
        center_y_values = [row[3] for row in records_sorted_x]

        ax_time.plot(x_values, center_x_values, color='tab:blue', linewidth=1.8, label='Center X')
        ax_time.plot(x_values, center_y_values, color='tab:orange', linewidth=1.8, label='Center Y')

        current_center_info = getattr(self.all_files_map[current_frame_id], "pattern_center", None)
        if current_center_info:
            current_x_val = current_frame_id if x_data_key == 'id' else getattr(self.all_files_map[current_frame_id], x_data_key, None)
            if current_x_val is not None:
                ax_time.axvline(x=current_x_val, color='red', linestyle='--', linewidth=1, alpha=0.6)
                ax_time.scatter([current_x_val], [current_center_info['center_x']], color='tab:blue', s=60, zorder=5)
                ax_time.scatter([current_x_val], [current_center_info['center_y']], color='tab:orange', s=60, zorder=5)

        ax_time.set_xlabel(x_key)
        ax_time.set_ylabel("Center position (pixels)")
        ax_time.set_title("Pattern Center X/Y vs " + x_key)
        ax_time.grid(True)
        ax_time.legend(loc='best')

        # --- Plot 2: center path in image coordinates (X-Y) ---
        records_sorted_frame = sorted(records, key=lambda row: row[0])
        path_x = [row[2] for row in records_sorted_frame]
        path_y = [row[3] for row in records_sorted_frame]
        ax_path.plot(path_x, path_y, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Center path')
        ax_path.scatter(path_x, path_y, color='black', s=12, alpha=0.5)

        if current_center_info:
            cx = current_center_info['center_x']
            cy = current_center_info['center_y']
            ax_path.scatter([cx], [cy], color='red', marker='x', s=90, linewidths=2, zorder=6, label='Current frame')
            self.center_current_label.setText(f"Current center: X={cx:.2f}, Y={cy:.2f}")
        else:
            self.center_current_label.setText("Current center: N/A")

        ax_path.set_xlabel("X Position (pixels)")
        ax_path.set_ylabel("Y Position (pixels)")
        ax_path.set_title("Pattern Center Trajectory (X-Y)")
        ax_path.grid(True)
        ax_path.invert_yaxis()
        ax_path.set_aspect('equal', adjustable='box')
        ax_path.legend(loc='best')

        self.center_xy_over_time_canvas.draw()
        self.center_path_canvas.draw()

    def update_2d_preview(self, selected_track: Track, current_frame_id: int):
        """
        Updates the 2D preview panel, showing the image, tracks, and ROI preview.

        This comprehensive method handles the entire top visualization area. It:
        1. Loads and displays the full image for the current frame.
        2. Clears all previous overlays (paths, ROIs, markers).
        3. Updates the separate ROI preview panel, showing the data that will be
        analyzed (either raw or background-subtracted).
        4. Draws all visible track paths on the main image viewer, highlighting
        the currently selected track.
        5. Draws and highlights the specific ROI and pattern center for the
        current frame.

        Args:
            selected_track: The Track object that is currently selected in the UI.
            current_frame_id: The ID of the frame to be displayed, determined by
                            the frame slider.
        """
        # --- 1. Load and Display the Main Full-Frame Image ---
        image_path = self.all_files_map[current_frame_id].filepath
        full_image = load_tiff(image_path)
        
        # Convert the numpy array to a QPixmap and set it in the main viewer.
        pixmap = self._numpy_to_pixmap(full_image)
        self.image_viewer_2d.set_image(pixmap, fit=True)
        
        # --- 2. Clear All Previous Overlays ---
        # This ensures a clean slate for redrawing.
        self.image_viewer_2d.clear_rois()
        self.image_viewer_2d.clear_paths()
        self.image_viewer_2d.clear_pattern_center()
        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        self.analysis_ellipse = normalized_ellipse
        if normalized_ellipse:
            self.image_viewer_2d.update_analysis_ellipse(normalized_ellipse)
        
        # Get the specific spot object for the current frame from the selected track.
        active_spot = selected_track.spots_by_image_id.get(current_frame_id)
        if not active_spot:
            self.roi_preview_label.setText("Spot not found")
            return

        # --- 3. Update the Zoomed-In ROI Preview Panel ---
        # This section prepares and displays the data that is being fed into the
        # fitting algorithm.

        # Get the original ROI data from the full image.
        rect = active_spot.roi_rect
        original_roi_data = full_image[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
        
        # Decide which data to display based on the "Subtract Background" checkbox.
        track_config = self._get_track_config(selected_track.track_id)
        should_subtract_bg = track_config.get('subtract_bg', False)
        
        print(f"DEBUG: Updating ROI preview for Track {selected_track.track_id}. Should subtract BG: {should_subtract_bg}")

        if should_subtract_bg and self.bg_image_data_gray is not None:
            # If subtraction is enabled, calculate the subtracted ROI.
            bg_roi_data = self.bg_image_data_gray[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
            if bg_roi_data.shape == original_roi_data.shape:
                roi_for_preview = img_as_float(original_roi_data) - bg_roi_data
                roi_for_preview[roi_for_preview < 0] = 0 # Ensure no negative intensity.
            else:
                roi_for_preview = original_roi_data # Fallback
        else:
            # Use the original ROI data
            roi_for_preview = original_roi_data

        # Convert the final ROI data (raw or subtracted) to a pixmap and display it.
        roi_pixmap = self._numpy_to_pixmap(roi_for_preview)
        if not roi_pixmap.isNull():
            # Scale the small ROI pixmap to a larger, fixed size for visibility.
            self.roi_preview_label.setPixmap(roi_pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        else:
            self.roi_preview_label.setText("Invalid ROI data")
        
        # --- 4. Draw Overlays on the Main 2D Image Viewer ----

        # Draw all track paths.
        for track in self.tracks:
            points = [QPointF(spot.roi_rect.center()) for fid, spot in sorted(track.spots_by_image_id.items())]
            is_selected = track.track_id == selected_track.track_id
            color = QColor("cyan") if is_selected else QColor(128, 128, 128, 100)
            width = 2 if is_selected else 1
            self.image_viewer_2d.draw_track_path(points, color, width)

        if self.show_center_trace_checkbox.isChecked():
            center_points = []
            for frame_id in sorted(self.all_files_map.keys()):
                center_info = getattr(self.all_files_map[frame_id], "pattern_center", None)
                if center_info:
                    center_points.append(QPointF(center_info['center_x'], center_info['center_y']))
            if len(center_points) > 1:
                center_path_color = QColor(255, 255, 0, 180)
                self.image_viewer_2d.draw_track_path(center_points, center_path_color, 2)

        self.image_viewer_2d.update_rois([active_spot])
        self.image_viewer_2d.highlight_roi(active_spot.id)

        center_info = self.all_files_map[current_frame_id].pattern_center
        if center_info:
            center_point = QPointF(center_info['center_x'], center_info['center_y'])
            self.image_viewer_2d.draw_pattern_center(center_point)
            circle_radius = center_info.get("circle_radius")
            if circle_radius is not None:
                self.image_viewer_2d.draw_pattern_circle(center_point, circle_radius)

    def on_position_plot_config_changed(self):
        """
        Slot that reacts to configuration changes in the "Position Plot" tab.

        This slot is connected to the signals from the checkboxes in the
        "Position Plot" tab's control panel (e.g., the track checklist or the
        "Show Trace" checkbox). Its purpose is to trigger a full visualization
        refresh by calling the main on_selection_changed() method.
        """
        # Call the main update handler to redraw all plots with the new settings.
        self.on_selection_changed()
    
    def _numpy_to_pixmap(self, numpy_data: np.ndarray) -> QPixmap:
        """
        Converts a NumPy array representing an image into a QPixmap.

        This utility function handles different data types (e.g., float, uint8)
        and formats (grayscale, RGB). It normalizes data if necessary and ensures
        it is in a memory-contiguous format that is safe to pass to the
        QImage constructor.

        Args:
            numpy_data: A NumPy array containing the image data.

        Returns:
            A QPixmap object that can be displayed in the GUI, or an empty
            QPixmap if the input data is invalid.
        """
        # Return an empty pixmap if the input data is invalid.
        pixmap = QPixmap()
        if numpy_data is None or numpy_data.size == 0: return pixmap
        
        # --- Data Normalization ---
        # QImage expects 8-bit integer data (0-255). If the input is another
        # type (like float from calculations), it must be converted.
        if numpy_data.dtype != np.uint8:
            # Linearly scale the float data to the full 0-255 integer range.
            min_val, max_val = numpy_data.min(), numpy_data.max()
            if max_val > min_val:
                # Skalujemy dane float do zakresu 0-255
                numpy_data = 255 * (numpy_data - min_val) / (max_val - min_val)
            else:
                # If the image is flat (all pixels have the same value), make it black.
                numpy_data = np.zeros_like(numpy_data)
            # Convert the scaled float data to the required 8-bit integer type.
            numpy_data = numpy_data.astype(np.uint8)

        height, width = numpy_data.shape[:2]
        bytes_per_line = width
        
        # Ensure the numpy array's data is stored in a contiguous block of memory.
        # This is required by the QImage constructor to prevent memory errors.
        contiguous_data = np.ascontiguousarray(numpy_data)
        
        # --- QImage Creation ---
        # Create the QImage based on the array's dimensions (grayscale vs. RGB).
        if numpy_data.ndim == 3 and numpy_data.shape[2] == 3: # RGB
            bytes_per_line = 3 * width
            q_image = QImage(contiguous_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else: # Grayscale
            q_image = QImage(contiguous_data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        return QPixmap.fromImage(q_image)

    def _get_track_config(self, track_id: int) -> Dict:
        """
        Retrieves the current configuration for a specific track from the table widget.

        This helper method iterates through the rows of the configuration table
        to find the one corresponding to the given track_id. It then reads the
        state of the widgets in that row (e.g., checkboxes) to determine the
        analysis settings.

        Args:
            track_id: The ID of the track for which to get the configuration.

        Returns:
            A dictionary containing the configuration settings for the specified
            track (e.g., {'subtract_bg': True}). Returns an empty dictionary if
            the track_id is not found in the table.
        """
        # Iterate through all rows in the configuration table.
        for row in range(self.config_table.rowCount()):
            # Check the "Track ID" column (column 1) to find the matching row.
            if int(self.config_table.item(row, 0).text()) == track_id:
                # Once the correct row is found, read the state of the widgets.
                return {
                    # 'num_peaks': self.config_table.cellWidget(row, 2).value(),
                    'subtract_bg': self.config_table.cellWidget(row, 1).isChecked()
                }
        return {}

    def update_3d_plots(self, track: Track, frame_id: int):
        """
        Generates and displays the 3D Matplotlib plots for a specific spot.

        This method updates the "ROI 3D Preview" tab. It creates two plots for
        visual comparison:
        1. A surface plot of the pixel data that was used for analysis (either
        the raw ROI data or the background-subtracted data).
        2. A surface plot of the ideal mathematical 2D Gaussian model, generated
        from the parameters obtained during the fitting process.

        Args:
            track: The Track object containing the analysis results.
            frame_id: The specific frame ID within the track to visualize.
        """
        # Guard clauses: Ensure the analysis results for this frame exist.
        if not hasattr(track, 'analysis_results') or frame_id not in track.analysis_results: return

        fit_results = track.analysis_results[frame_id]
        if not fit_results: return

        # --- Data Preparation ---
        # Get the spot object and load the corresponding full image.
        spot = track.spots_by_image_id[frame_id]
        full_image = load_tiff(self.all_files_map[frame_id].filepath)

        # Extract the specific Region of Interest (ROI) from the full image.
        rect = spot.roi_rect
        roi_data = full_image[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
        if roi_data is None or roi_data.size == 0:
            return
        if roi_data.ndim >= 3:
            if roi_data.shape[-1] >= 3:
                roi_data_2d = rgb2gray(roi_data[..., :3])
            else:
                roi_data_2d = roi_data[..., 0]
        else:
            roi_data_2d = roi_data
        if roi_data_2d.ndim != 2:
            return

        height, width = roi_data_2d.shape
        
        x = np.arange(width)
        y = np.arange(height)

        # Create X and Y coordinate grids for the surface plots.
        X, Y = np.meshgrid(x, y)

        # --- 1. Prepare Data for the First Plot (Analysis Data) ---
        # This block determines if we should display the raw data or the
        # background-subtracted data, based on the user's selection in the UI.
        track_config = self._get_track_config(track.track_id)
        should_subtract_bg = track_config.get('subtract_bg', False)
        
        # By default, use the original (but normalized to float) ROI data.
        data_for_plot = img_as_float(roi_data_2d)
        plot_title = "Raw ROI Data"

        if should_subtract_bg and self.bg_image_data_gray is not None:
            bg_roi_data = self.bg_image_data_gray[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
            if bg_roi_data.shape == data_for_plot.shape:
                # If subtraction is enabled, calculate the new data and update the title.
                data_for_plot = data_for_plot - bg_roi_data
                data_for_plot[data_for_plot < 0] = 0
                plot_title = "ROI with BG Subtracted"

        # --- 2. Draw the First Plot (Analysis Data) ---
        ax_raw = self.plot3d_raw_canvas.axes
        ax_raw.clear()
        ax_raw.plot_surface(X, Y, data_for_plot, cmap='viridis', rstride=1, cstride=1, antialiased=True)
        ax_raw.set_title(plot_title) # Apply the dynamic title determined above
        ax_raw.set_xlabel("X (pixels)"); ax_raw.set_ylabel("Y (pixels)"); ax_raw.set_zlabel("Intensity")
        self.plot3d_raw_canvas.draw()
        
        # --- 3. Draw the Second Plot (Fitted Model) ---
        # Create a clean dictionary with only the parameters required by the model function.
        model_param_names = ['amplitude', 'center_x', 'center_y', 'sigma_x', 'sigma_y', 'theta', 'offset']
        model_params = {key: fit_results[key] for key in model_param_names if key in fit_results}
        if len(model_params) != len(model_param_names): return
        
        # Generate the Z values for the ideal Gaussian surface from the fitted parameters.
        fitted_z = gaussian_2d((X, Y), **model_params).reshape(height, width)
        
        ax_fit = self.plot3d_fit_canvas.axes
        ax_fit.clear()
        ax_fit.plot_surface(X, Y, fitted_z, cmap='plasma', rstride=1, cstride=1, antialiased=True)
        ax_fit.set_title("Fitted Gaussian Model")
        ax_fit.set_xlabel("X (pixels)"); ax_fit.set_ylabel("Y (pixels)"); ax_fit.set_zlabel("Intensity")
        self.plot3d_fit_canvas.draw()

    def update_2d_plot(self, track: Track, current_frame_id: int):
        """
        Generates and displays the 2D dependency plot for a single track.

        This method updates the "Dependency Plots" tab. It plots a user-selected
        property (Y-axis) against another user-selected variable (X-axis) for
        the currently selected track. It also highlights the specific data point
        that corresponds to the frame selected on the main slider.

        Args:
            track: The Track object currently selected in the main config table.
            current_frame_id: The ID of the frame to be highlighted on the plot.
        """
        # Get the Matplotlib axes object and clear any previous content.
        ax = self.plot_2d_canvas.axes
        ax.clear()

        # Guard clause: Exit if the track has no analysis results to plot.
        if not hasattr(track, 'analysis_results') or not track.analysis_results:
            self.plot_2d_canvas.draw()
            return
            
        # --- 1. Gather Configuration from UI ---
    
        # Map user-friendly names from the dropdowns to internal data keys.
        y_axis_map = {
            label: result_key
            for label, (result_key, _error_key) in ANALYSIS_RESULT_PARAMETERS.items()
        }
        x_axis_map = {"Frame ID": "id", "Temperature": "temperature", "Energy": "energy", "Coverage": "coverage"}
        
        # Get the current selections from the combo boxes.
        y_key = y_axis_map[self.y_axis_combo.currentText()]
        x_key = x_axis_map[self.x_axis_combo.currentText()]
        
         # --- 2. Extract Data Series ---

        x_data, y_data = [], []
        active_point = None # To store the (x, y) coordinates of the highlighted point.

        # Iterate through all analyzed frames for the selected track.
        for frame_id, results in sorted(track.analysis_results.items()):
            if not results: continue
            
            # Get the X-axis value (either the frame ID or from LeedFile metadata).
            if x_key == 'id': x_val = frame_id
            else: x_val = getattr(self.all_files_map[frame_id], x_key, None)
            
            # Get the Y-axis value from the analysis results dictionary.
            y_val = results.get(y_key)
            
            if x_val is not None and y_val is not None:
                x_data.append(x_val)
                y_data.append(y_val)
                # Check if this frame is the one currently selected by the slider.
                if frame_id == current_frame_id:
                    active_point = (x_val, y_val)

        # --- 3. Render the Plot ---

        if x_data and y_data:
            # Plot the main data series with small markers and a connecting line.
            ax.plot(x_data, y_data, marker='o', markersize=4, linestyle='-', label=f'Track {track.track_id}')
            
            # If a point corresponding to the current frame was found, highlight it.
            if active_point:
                # Draw a vertical dashed line at the x-position of the active point.
                ax.axvline(x=active_point[0], color='r', linestyle='--', linewidth=1, alpha=0.7)
                # Draw a larger, more prominent marker on the active point itself.
                ax.plot(active_point[0], active_point[1], marker='o', markersize=8, color='red', label=f'Frame {current_frame_id}')

            # Set the labels and title of the plot dynamically based on user selections.
            ax.set_xlabel(self.x_axis_combo.currentText())
            ax.set_ylabel(self.y_axis_combo.currentText())
            ax.set_title(f"{self.y_axis_combo.currentText()} vs. {self.x_axis_combo.currentText()}")
            ax.grid(True)
            ax.legend()
            
        self.plot_2d_canvas.draw()

    def _create_action_panel(self) -> QHBoxLayout:
        """
        Creates the bottom panel containing the main action buttons.

        This panel holds buttons that trigger major actions like starting the
        analysis and exporting results, as well as the standard dialog buttons
        (OK/Cancel).

        Returns:
            A QHBoxLayout containing the configured buttons.
        """
        # Create a horizontal layout for the action panel.
        layout = QHBoxLayout()
        
        # --- "Perform Analysis" Button ---
        # This button starts the main analysis pipeline in a background thread.
        self.perform_analysis_button = QPushButton("Perform Analysis")
        self.perform_analysis_button.clicked.connect(self.start_analysis_task) 
        
        # --- "Export Results" Button ---
        # This button triggers the saving of all calculated data.
        self.export_button = QPushButton("Export Results")
        # The button is disabled by default and becomes active only after a
        # successful analysis.
        self.export_button.clicked.connect(self.export_results) 
        self.export_button.setEnabled(False) 

        # --- "Save Session" Button ---
        self.save_session_button = QPushButton("Save Session")
        self.save_session_button.clicked.connect(self._on_save_session)
        
        # --- Standard "OK" and "Cancel" Buttons ---
        # QDialogButtonBox provides a standard way to present dialog buttons.
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout.addWidget(self.perform_analysis_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.save_session_button)
        layout.addStretch()
        layout.addWidget(self.button_box)
        
        return layout
    
    def export_results(self):
        """
        Handles the 'Export Results' button click.

        This slot first checks if any analysis data is available to export. If so,
        it opens a native system dialog for the user to select a destination
        directory. Upon selection, it calls the backend exporter function to save
        all analysis data to a CSV file and shows a confirmation or error message.
        """
        # Guard clause: Check if analysis has been performed and results exist.
        if not self.tracks or not hasattr(self.tracks[0], 'analysis_results'):
            QMessageBox.warning(self, "No Data", "No analysis data available to export. Please run the analysis first.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results")

        if directory:
            try:
                export_full_results(self.tracks, self.all_files_map, directory)
                QMessageBox.information(self, "Success", f"Results successfully saved to:\n{directory}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred while exporting data:\n{e}")
    
    def start_analysis_task(self):
        """
        Gathers configuration from the UI and starts the analysis in a background thread.

        This slot is connected to the "Perform Analysis" button. It reads the
        current state of the configuration table, prepares the necessary data,
        and dispatches the main analysis function (`run_full_analysis`) to a
        GenericWorker to be executed without freezing the user interface.
        """
        # --- 1. Gather Configuration from the Table ---
        tracks_to_analyze = self.tracks
        analysis_config = {}

        # Iterate through all rows of the configuration table.
        for row in range(self.config_table.rowCount()):
            track_id = int(self.config_table.item(row, 0).text())
            sub_bg_checkbox = self.config_table.cellWidget(row, 1)
            analysis_config[track_id] = {
                'subtract_bg': sub_bg_checkbox.isChecked()
            }
            self.config_table.item(row, 2).setText("Processing...")

        #  Guard clause: If no tracks are selected, show a warning and do nothing.
        if not tracks_to_analyze:
            QMessageBox.warning(self, "No Tracks Selected", "Please select at least one track to analyze.")
            return

        # --- 2. Launch the Background Worker ---
        # Disable the button to prevent the user from starting multiple analyses at once.
        self.perform_analysis_button.setEnabled(False)
        total_frames = len({fid for t in tracks_to_analyze for fid in t.spots_by_image_id})

        # Create a GenericWorker instance, passing the target function (`run_full_analysis`)
        # and all the necessary data as arguments.
        center_mode = self.center_mode_combo.currentData() if hasattr(self, "center_mode_combo") else "per_frame"
        manual_ellipse_payload = None
        if center_mode == "manual_ellipse":
            normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
            self.analysis_ellipse = normalized_ellipse
            if not normalized_ellipse:
                QMessageBox.warning(
                    self,
                    "Missing Analysis Ellipse",
                    "Center mode 'Manual ellipse' requires setting an analysis ellipse first "
                    "(Mask -> Set Analysis Ellipse... in the main window).",
                )
                self.perform_analysis_button.setEnabled(True)
                for row in range(self.config_table.rowCount()):
                    status_item = self.config_table.item(row, 2)
                    if status_item and status_item.text() == "Processing...":
                        status_item.setText("Pending")
                return
            cx, cy, rx, ry, angle_deg = normalized_ellipse
            manual_ellipse_payload = {
                "cx": float(cx),
                "cy": float(cy),
                "rx": float(rx),
                "ry": float(ry),
                "angle_deg": float(angle_deg),
            }
        print(
            f"[AnalysisDialog] Launching analysis | tracks={len(tracks_to_analyze)} | "
            f"frames={total_frames} | center_mode={center_mode}"
        )
        self.analysis_cancel_event = Event()
        worker = GenericWorker(
            run_full_analysis,
            tracks_to_analyze,
            analysis_config,
            self.all_files_map,
            self.background_file,
            center_mode=center_mode,
            manual_ellipse=manual_ellipse_payload,
            cancel_event=self.analysis_cancel_event,
        )
        # The launch_analysis_worker method will handle thread management and the progress dialog.
        self.launch_analysis_worker(worker)

    def launch_analysis_worker(self, worker: GenericWorker):
        """
        Launches a GenericWorker in a separate QThread with a progress dialog.

        This method handles the boilerplate for running a task in the background.
        It creates a QThread, moves the provided worker to it, displays a modal
        progress dialog, and connects all the necessary signals (progress,
        finished, error, cancellation) to their corresponding slots. It also
        ensures proper cleanup of the thread and worker objects.

        Args:
            worker: An initialized GenericWorker instance ready to be run.
        """
        # 1. Create and configure the progress dialog.
        # The dialog shows the progress of the background task.
        self.progress_dialog = QProgressDialog("Performing analysis...", "Cancel", 0, 100, self)
        # WindowModal prevents interaction with the parent dialog while the task is running.
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

        # 2. Set up the QThread and move the worker to it.
        self.worker_thread = QThread()
        self.worker = worker
        # This is the core of Qt's multithreading: the worker object now lives in the new thread.
        self.worker.moveToThread(self.worker_thread)

        # 3. Connect signals from the worker and dialog to the appropriate slots.
        # Allow the user to cancel the task via the progress dialog's cancel button.
        self.progress_dialog.canceled.connect(self._handle_analysis_cancel)
        # Update the progress bar value when the worker emits progress.
        self.worker.progress.connect(self.progress_dialog.setValue)
        # Handle the results when the worker successfully finishes.
        self.worker.finished.connect(self.on_analysis_finished)
        # Handle any exceptions that occur in the worker thread.
        self.worker.error.connect(self.on_analysis_error)
        
        # 4. Set up cleanup connections.
        # These ensure that the thread and worker are properly terminated and deleted
        # after the task is complete, preventing memory leaks.
        self.worker.finished.connect(self.progress_dialog.accept)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.progress_dialog.reject)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, "worker_thread", None))
        self.worker_thread.finished.connect(lambda: setattr(self, "worker", None))

        # 5. Start the worker.
        # Connect the thread's started signal to the worker's run method.
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def _handle_analysis_cancel(self):
        """Handle Cancel click from analysis progress dialog."""
        print("[AnalysisDialog] Cancel requested by user.")
        if self.analysis_cancel_event:
            self.analysis_cancel_event.set()
        if self.progress_dialog:
            self.progress_dialog.setLabelText("Cancelling, please wait...")
            self.progress_dialog.setEnabled(False)
        if self.worker_thread:
            try:
                self.worker_thread.requestInterruption()
            except RuntimeError:
                # Thread already deleted, nothing to cancel.
                pass

    def on_analysis_finished(self, updated_tracks: Optional[List[Track]]):
        """
        Handles the successful completion of the background analysis task.

        This slot is connected to the 'finished' signal of the GenericWorker.
        It receives the completed analysis data, updates the dialog's internal
        state, re-enables UI controls, and triggers a full refresh of all
        visualizations to display the new results.

        Args:
            updated_tracks: A list of Track objects, now populated with the
                            'analysis_results' dictionaries from the worker.
        """
        self.analysis_cancel_event = None

        if updated_tracks is None:
            print("Analysis cancelled by user.")
            self.perform_analysis_button.setEnabled(True)
            for row in range(self.config_table.rowCount()):
                status_item = self.config_table.item(row, 2)
                if status_item and status_item.text() == "Processing...":
                    status_item.setText("Cancelled")
            QMessageBox.information(self, "Analysis Cancelled", "Analysis was cancelled.")
            return

        print("Analysis finished successfully.")

        # 1. Update the dialog's internal data with the new results.
        self.tracks = updated_tracks

        # 2. Re-enable the UI buttons.
        self.perform_analysis_button.setEnabled(True)
        self.export_button.setEnabled(True)

        all_frame_ids = sorted(list({fid for t in self.tracks for fid in t.analysis_results}))
        self.frame_filter_combo.clear()
        self.frame_filter_combo.addItems(["All Frames"] + [str(fid) for fid in all_frame_ids])

        for i in reversed(range(self.param_grid_layout.count())): 
            self.param_grid_layout.itemAt(i).widget().setParent(None)
        self.param_checkboxes.clear()

        params_to_show = list(ANALYSIS_RESULT_PARAMETERS)
        num_columns = 3

        for i, param_name in enumerate(params_to_show):
            checkbox = QCheckBox(param_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.on_table_filter_changed)
            
            self.param_checkboxes[param_name] = checkbox
            
            row = i // num_columns
            col = i % num_columns
            self.param_grid_layout.addWidget(checkbox, row, col)

        
        # 3. Update the status column in the main configuration table.
        for row in range(self.config_table.rowCount()):
            self.config_table.item(row, 2).setText("Done")
                
        # 4. Populate the track selection checklists in the plot tabs.
        # Clear the lists first to handle cases where analysis is run multiple times.
        self.track_checklist.clear()
        self.compare_track_checklist.clear()

        for track in self.tracks:
            # Create an item for the "Position Plot" tab's checklist.
            item1 = QListWidgetItem(f"Track {track.track_id}")
            item1.setData(Qt.ItemDataRole.UserRole, track.track_id)
            item1.setFlags(item1.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item1.setCheckState(Qt.CheckState.Checked)
            self.track_checklist.addItem(item1)
            
            # Create an identical item for the "Comparative Plots" tab's checklist.
            item2 = QListWidgetItem(f"Track {track.track_id}")
            item2.setData(Qt.ItemDataRole.UserRole, track.track_id)
            item2.setFlags(item2.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item2.setCheckState(Qt.CheckState.Checked)
            self.compare_track_checklist.addItem(item2)
        
        self.radius_track_checklist.clear()
        for track in self.tracks:
            item = QListWidgetItem(f"Track {track.track_id}")
            item.setData(Qt.ItemDataRole.UserRole, track.track_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.radius_track_checklist.addItem(item)

        if self.config_table.rowCount() > 0:
            self.config_table.selectRow(0)

        self.on_selection_changed()
        self.update_radius_data_table()
        
        # TODO: Call any additional plot refresh helpers if further views are added
        # e.g. self.update_plots()

    def update_radius_data_table(self):
        """
        Populates the "Radius Data" table with radius and uncertainty values.
        Rows represent frames, columns represent tracks.
        """
        self.radius_data_table.clear() # Reset the entire table structure
        if not self.tracks or not hasattr(self.tracks[0], 'analysis_results'):
            return

        # 1. Prepare headers and overall shape
        all_frame_ids = sorted(list({fid for t in self.tracks for fid in t.analysis_results}))
        
        self.radius_data_table.setRowCount(len(all_frame_ids))
        self.radius_data_table.setColumnCount(len(self.tracks))
        
        self.radius_data_table.setVerticalHeaderLabels([str(fid) for fid in all_frame_ids])
        self.radius_data_table.setHorizontalHeaderLabels([f"Track {t.track_id}" for t in self.tracks])

        # 2. Populate the cells with data
        for col_idx, track in enumerate(self.tracks):
            if not hasattr(track, 'analysis_results'): continue

            for row_idx, frame_id in enumerate(all_frame_ids):
                results = track.analysis_results.get(frame_id)
                
                cell_text = "N/A" # Default text when no data is available
                if results and 'radius' in results:
                    radius = results['radius']
                    error = results.get('radius_error', 0)
                    cell_text = f"{radius:.2f} +/- {error:.2f}"
                
                item = QTableWidgetItem(cell_text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.radius_data_table.setItem(row_idx, col_idx, item)
        
        self.radius_data_table.resizeColumnsToContents()
    
    def on_table_filter_changed(self):
        self.on_selection_changed()

    def on_analysis_error(self, error_message: str):
        """
        Handles errors emitted from the background analysis worker.

        This slot is connected to the 'error' signal of the GenericWorker. It is
        triggered when an unhandled exception occurs in the background thread.
        It displays a critical error message to the user, re-enables the UI,
        and updates the status table to reflect the failure.

        Args:
            error_message: A string containing the exception message from the
                        worker thread.
        """
        self.analysis_cancel_event = None
        # Display a critical error dialog to inform the user of the problem.
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{error_message}")

        # Re-enable the "Perform Analysis" button so the user can try again.
        self.perform_analysis_button.setEnabled(True)
        
        # Update the status column in the configuration table for all tracks
        # that were part of the failed analysis run.
        # for row in range(self.config_table.rowCount()):
        #     if self.config_table.cellWidget(row, 0).isChecked():
        #          self.config_table.item(row, 2).setText("Error")

    def populate_config_table(self):
        """
        Populates the configuration table with data from the loaded tracks.

        This method iterates through the list of Track objects and creates a
        corresponding row in the configuration table for each one. It populates
        the cells with widgets (like QCheckBox) and read-only text items to
        allow the user to view and configure the analysis for each track.
        """
        # Set the number of rows in the table to match the number of tracks.
        self.config_table.setRowCount(len(self.tracks))
        
        for row, track in enumerate(self.tracks):
            # --- Checkbox "Analyze" ---
            # analyze_checkbox = QCheckBox()
            # analyze_checkbox.setChecked(True)
            # self.config_table.setCellWidget(row, 0, analyze_checkbox)
            
            # --- Track ID ---
            track_id_item = QTableWidgetItem(str(track.track_id))
            track_id_item.setFlags(track_id_item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Tylko do odczytu
            self.config_table.setItem(row, 0, track_id_item)
            
            # # --- SpinBox "Num Peaks" ---
            # peaks_spinbox = QSpinBox()
            # peaks_spinbox.setRange(1, 5)
            # peaks_spinbox.setValue(1)
            # self.config_table.setCellWidget(row, 2, peaks_spinbox)
            
            # --- Checkbox "Subtract Background" ---
            bg_checkbox = QCheckBox()
            bg_checkbox.setChecked(True)
            # self.config_table.setCellWidget(row, 3, bg_checkbox)
            self.config_table.setCellWidget(row, 1, bg_checkbox)

            # --- Status Label ---
            status_item = QTableWidgetItem("Pending")
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # self.config_table.setItem(row, 4, status_item)
            self.config_table.setItem(row, 2, status_item)
