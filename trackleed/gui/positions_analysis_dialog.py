from __future__ import annotations

# stal/gui/positions_analysis_dialog.py

import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Dict, List

import numpy as np
from PyQt6.QtCore import QRect, QRectF, Qt, QThread
from PyQt6.QtGui import QAction, QColor, QImage, QPixmap, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QComboBox,
    QSpinBox,
    QSlider,
    QSplitter,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian, gaussian_2d
from trackleed.io.image_reader import load_tiff
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.positions_analysis_coordinator import (
    DEFAULT_VALIDATION_CONFIG,
    evaluate_fit_quality,
    fit_track_position_on_frame,
    format_quality_flags,
    is_fit_valid,
    normalize_validation_config,
    prepare_roi_for_fit,
    run_positions_analysis,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont


POSITIONS_SESSION_TYPE = "trackleed_positions_analysis_session"
POSITIONS_SESSION_VERSION = 1


@dataclass
class PositionsAnalysisGroup:
    group_id: int
    name: str
    reference_track_id: int | None = None
    satellite_track_ids: list[int] = field(default_factory=list)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class GaussPreviewCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_roi = self.fig.add_subplot(131)
        self.axes_fit = self.fig.add_subplot(132)
        self.axes_residual = self.fig.add_subplot(133)
        super().__init__(self.fig)


class ValidationPropertiesDialog(QDialog):
    def __init__(self, validation_config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validation Properties")
        self.setModal(True)
        self.setMinimumWidth(420)

        config = normalize_validation_config(validation_config)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Adjust validation thresholds used to accept or reject 2D Gaussian fits. "
            "The new settings will be used by the next Run Positions Analysis and local recompute operations."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()

        self.min_r_squared_spin = QDoubleSpinBox()
        self.min_r_squared_spin.setRange(0.0, 1.0)
        self.min_r_squared_spin.setSingleStep(0.01)
        self.min_r_squared_spin.setDecimals(3)
        self.min_r_squared_spin.setValue(config["min_r_squared"])
        form.addRow("Min R²:", self.min_r_squared_spin)

        self.min_amplitude_spin = QDoubleSpinBox()
        self.min_amplitude_spin.setRange(0.0, 1000.0)
        self.min_amplitude_spin.setSingleStep(0.01)
        self.min_amplitude_spin.setDecimals(3)
        self.min_amplitude_spin.setValue(config["min_amplitude"])
        form.addRow("Min amplitude:", self.min_amplitude_spin)

        self.center_margin_spin = QDoubleSpinBox()
        self.center_margin_spin.setRange(0.0, 100.0)
        self.center_margin_spin.setSingleStep(0.5)
        self.center_margin_spin.setDecimals(2)
        self.center_margin_spin.setValue(config["center_margin_px"])
        form.addRow("Center margin (px):", self.center_margin_spin)

        self.min_sigma_spin = QDoubleSpinBox()
        self.min_sigma_spin.setRange(0.01, 100.0)
        self.min_sigma_spin.setSingleStep(0.1)
        self.min_sigma_spin.setDecimals(3)
        self.min_sigma_spin.setValue(config["min_sigma_px"])
        form.addRow("Min sigma (px):", self.min_sigma_spin)

        self.max_sigma_factor_spin = QDoubleSpinBox()
        self.max_sigma_factor_spin.setRange(0.01, 10.0)
        self.max_sigma_factor_spin.setSingleStep(0.05)
        self.max_sigma_factor_spin.setDecimals(3)
        self.max_sigma_factor_spin.setValue(config["max_sigma_factor"])
        form.addRow("Max sigma factor:", self.max_sigma_factor_spin)

        self.center_preference_factor_spin = QDoubleSpinBox()
        self.center_preference_factor_spin.setRange(0.0, 1.5)
        self.center_preference_factor_spin.setSingleStep(0.05)
        self.center_preference_factor_spin.setDecimals(3)
        self.center_preference_factor_spin.setValue(config["center_preference_factor"])
        self.center_preference_factor_spin.setToolTip(
            "If > 0, the fit prefers a peak near the ROI center and constrains center drift around the ROI center. "
            "Higher values allow more drift; 0 disables this behavior."
        )
        form.addRow("Center preference factor:", self.center_preference_factor_spin)

        layout.addLayout(form)

        self.reset_defaults_button = QPushButton("Reset Defaults")
        self.reset_defaults_button.clicked.connect(self._reset_defaults)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.reset_defaults_button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(button_box)
        layout.addLayout(buttons_layout)

    def _reset_defaults(self):
        defaults = normalize_validation_config(DEFAULT_VALIDATION_CONFIG)
        self.min_r_squared_spin.setValue(defaults["min_r_squared"])
        self.min_amplitude_spin.setValue(defaults["min_amplitude"])
        self.center_margin_spin.setValue(defaults["center_margin_px"])
        self.min_sigma_spin.setValue(defaults["min_sigma_px"])
        self.max_sigma_factor_spin.setValue(defaults["max_sigma_factor"])
        self.center_preference_factor_spin.setValue(defaults["center_preference_factor"])

    def get_validation_config(self) -> dict:
        return normalize_validation_config(
            {
                "min_r_squared": self.min_r_squared_spin.value(),
                "min_amplitude": self.min_amplitude_spin.value(),
                "center_margin_px": self.center_margin_spin.value(),
                "min_sigma_px": self.min_sigma_spin.value(),
                "max_sigma_factor": self.max_sigma_factor_spin.value(),
                "center_preference_factor": self.center_preference_factor_spin.value(),
            }
        )


class PositionsAnalysisDialog(QDialog):
    """
    Stage 3 skeleton dialog for the upcoming positions analysis workflow.

    Current scope:
    - show list of tracks,
    - define simple groups in memory,
    - choose one reference and many satellites per group,
    - run positions analysis based on 2D Gaussian fitting,
    - show basic image preview with tracked spots for the selected frame.
    """

    def __init__(
        self,
        tracks: List[Track],
        all_files_map: Dict[int, LeedFile],
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle("Positions Analysis")
        self.setMinimumSize(1500, 850)

        current_flags = self.windowFlags()
        self.setWindowFlags(
            current_flags
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self.tracks = tracks
        self.track_map = {track.track_id: track for track in self.tracks}
        self.all_files_map = all_files_map
        self.sorted_frame_ids = sorted(self.all_files_map.keys())
        self.groups: list[PositionsAnalysisGroup] = []
        self._group_counter = 0
        self.positions_analysis_results: dict | None = None
        self.validation_config: dict = normalize_validation_config(DEFAULT_VALIDATION_CONFIG)
        self.manual_bbox_corrections: dict[tuple[int, int], dict] = {}
        self.pending_manual_recompute: set[tuple[int, int]] = set()
        self.original_roi_rects: dict[tuple[int, int], QRect] = self._capture_original_roi_rects()
        self.positions_session_path: Path | None = None
        self._bbox_scale_reference: dict | None = None
        self.current_frame_id: int | None = None
        self.current_frame_image_data = None
        self.frame_image_shape_cache: dict[int, tuple[int, int]] = {}
        self.worker_thread = None
        self.worker = None
        self.positions_analysis_cancel_event: Event | None = None

        self._build_ui()
        self._populate_tracks_table()
        self._populate_groups_list()
        self._initialize_view()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        menu_bar = QMenuBar(self)
        settings_menu = menu_bar.addMenu("Settings")
        validation_action = QAction("Validation Properties", self)
        validation_action.triggered.connect(self._show_validation_properties_dialog)
        settings_menu.addAction(validation_action)
        main_layout.setMenuBar(menu_bar)

        title_label = QLabel("Positions Analysis")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        main_layout.addWidget(title_label)

        subtitle_label = QLabel(
            "Positions relative to reference spots with manual correction, overlay preview, and live Gaussian fit inspection."
        )
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_tracks_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_groups_panel())
        splitter.setSizes([340, 820, 320])

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        main_layout.addWidget(button_box)

    def _build_tracks_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        group_box = QGroupBox("Tracks")
        group_layout = QVBoxLayout(group_box)

        self.tracks_table = QTableWidget()
        self.tracks_table.setColumnCount(4)
        self.tracks_table.setHorizontalHeaderLabels(
            ["Track ID", "Start Frame", "End Frame", "Length"]
        )
        self.tracks_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tracks_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.tracks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tracks_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tracks_table.itemSelectionChanged.connect(self._on_track_selection_changed)

        self.track_stats_label = QLabel("")
        self.track_stats_label.setWordWrap(True)

        group_layout.addWidget(self.tracks_table)
        group_layout.addWidget(self.track_stats_label)
        layout.addWidget(group_box)

        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)

        group_box = QGroupBox("Frame Preview")
        group_layout = QVBoxLayout(group_box)

        self.image_viewer = LeedImageViewer()
        self.image_viewer.set_mode("PAN")
        self.image_viewer.roi_moved.connect(self._handle_roi_moved)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, len(self.sorted_frame_ids) - 1))
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)

        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: -/-")
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)

        self.frame_meta_label = QLabel("Temperature: - | Coverage: - | Energy: -")
        self.frame_meta_label.setWordWrap(True)
        self.show_gauss_preview_checkbox = QCheckBox("Show 2D Gauss Preview")
        self.show_gauss_preview_checkbox.setToolTip(
            "Show a live preview of the current ROI, fitted Gaussian model, and residuals."
        )
        self.show_gauss_preview_checkbox.toggled.connect(self._toggle_gauss_preview)

        group_layout.addWidget(self.image_viewer, 1)
        group_layout.addLayout(slider_layout)
        group_layout.addWidget(self.frame_meta_label)
        group_layout.addWidget(self.show_gauss_preview_checkbox)
        vertical_splitter.addWidget(group_box)

        lower_panel = QWidget()
        lower_layout = QHBoxLayout(lower_panel)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(8)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        controls_layout = QHBoxLayout()
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItem("Coverage", "coverage")
        self.x_axis_combo.addItem("Temperature", "temperature")
        self.x_axis_combo.currentIndexChanged.connect(self._refresh_results_views)
        controls_layout.addWidget(QLabel("X axis:"))
        controls_layout.addWidget(self.x_axis_combo)
        controls_layout.addWidget(QLabel("BBox transparency:"))
        self.bbox_transparency_combo = QComboBox()
        for pct in range(0, 101, 10):
            self.bbox_transparency_combo.addItem(f"{pct}%", pct)
        self.bbox_transparency_combo.currentIndexChanged.connect(
            self._on_bbox_transparency_changed
        )
        self.bbox_transparency_combo.setCurrentIndex(0)
        controls_layout.addWidget(self.bbox_transparency_combo)
        self.show_selected_pair_only_checkbox = QCheckBox("Only selected pair")
        self.show_selected_pair_only_checkbox.setToolTip(
            "Show only the active group's reference point, selected satellite, and their bboxes."
        )
        self.show_selected_pair_only_checkbox.toggled.connect(
            self._on_selected_pair_filter_toggled
        )
        controls_layout.addWidget(self.show_selected_pair_only_checkbox)
        controls_layout.addStretch(1)

        self.results_status_label = QLabel(
            "Run positions analysis to populate the plot and results table."
        )
        self.results_status_label.setWordWrap(True)

        self.results_plot = MplCanvas(self, width=5, height=3, dpi=100)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Frame ID",
                "Temp",
                "Coverage",
                "dx",
                "dy",
                "r",
                "Ref Status",
                "Sat Status",
                "Flags",
            ]
        )
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        results_layout.addLayout(controls_layout)
        results_layout.addWidget(self.results_status_label)
        results_layout.addWidget(self.results_plot, 1)
        results_layout.addWidget(self.results_table, 1)
        lower_layout.addWidget(results_group, 2)

        self.gauss_preview_group = QGroupBox("2D Gauss Preview")
        gauss_preview_layout = QVBoxLayout(self.gauss_preview_group)
        self.gauss_preview_status_label = QLabel(
            "Enable the live preview and select a track to inspect the current Gaussian fit."
        )
        self.gauss_preview_status_label.setWordWrap(True)
        self.gauss_preview_canvas = GaussPreviewCanvas(self, width=7, height=3, dpi=100)
        gauss_preview_layout.addWidget(self.gauss_preview_status_label)
        gauss_preview_layout.addWidget(self.gauss_preview_canvas, 1)
        self.gauss_preview_group.setVisible(False)
        lower_layout.addWidget(self.gauss_preview_group, 1)

        vertical_splitter.addWidget(lower_panel)
        vertical_splitter.setSizes([520, 300])

        layout.addWidget(vertical_splitter)

        return panel

    def _build_groups_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        group_box = QGroupBox("Groups")
        group_layout = QVBoxLayout(group_box)

        self.groups_list = QListWidget()
        self.groups_list.itemSelectionChanged.connect(self._on_group_selection_changed)
        self.groups_info_label = QLabel(
            "Create groups and assign one reference track and satellite tracks."
        )
        self.groups_info_label.setWordWrap(True)
        self.analysis_summary_label = QLabel("No positions analysis results yet.")
        self.analysis_summary_label.setWordWrap(True)

        buttons_layout = QHBoxLayout()
        self.add_group_button = QPushButton("Add Group")
        self.add_group_button.clicked.connect(self._add_group)
        self.remove_group_button = QPushButton("Remove Group")
        self.remove_group_button.clicked.connect(self._remove_selected_group)
        buttons_layout.addWidget(self.add_group_button)
        buttons_layout.addWidget(self.remove_group_button)

        analysis_buttons_layout = QHBoxLayout()
        self.run_analysis_button = QPushButton("Run Positions Analysis")
        self.run_analysis_button.clicked.connect(self.start_positions_analysis)
        self.export_results_button = QPushButton("Export Results")
        self.export_results_button.clicked.connect(self._export_positions_results)
        self.export_results_button.setEnabled(False)
        analysis_buttons_layout.addWidget(self.run_analysis_button)
        analysis_buttons_layout.addWidget(self.export_results_button)

        export_layout_group = QGroupBox("Export Overlay")
        export_layout_form = QFormLayout(export_layout_group)

        self.export_label_distance_spin = QSpinBox()
        self.export_label_distance_spin.setRange(10, 300)
        self.export_label_distance_spin.setSingleStep(5)
        self.export_label_distance_spin.setValue(46)
        self.export_label_distance_spin.setToolTip(
            "Base distance of export labels from the detected point."
        )
        export_layout_form.addRow("Label distance:", self.export_label_distance_spin)

        self.export_label_spread_spin = QSpinBox()
        self.export_label_spread_spin.setRange(0, 200)
        self.export_label_spread_spin.setSingleStep(5)
        self.export_label_spread_spin.setValue(18)
        self.export_label_spread_spin.setToolTip(
            "Base lateral spread used to separate export labels from each other."
        )
        export_layout_form.addRow("Label spread:", self.export_label_spread_spin)

        details_group = QGroupBox("Selected Group")
        details_layout = QVBoxLayout(details_group)

        self.selected_track_label = QLabel("Selected track: none")
        self.reference_label = QLabel("Reference: none")
        self.reference_label.setWordWrap(True)

        self.set_reference_button = QPushButton("Set Selected Track as Reference")
        self.set_reference_button.clicked.connect(self._set_selected_track_as_reference)

        self.clear_reference_button = QPushButton("Clear Reference")
        self.clear_reference_button.clicked.connect(self._clear_reference_from_group)

        self.add_satellite_button = QPushButton("Add Selected Track as Satellite")
        self.add_satellite_button.clicked.connect(self._add_selected_track_as_satellite)

        self.remove_satellite_button = QPushButton("Remove Selected Satellite")
        self.remove_satellite_button.clicked.connect(self._remove_selected_satellite)

        self.satellites_list = QListWidget()
        self.satellites_list.itemSelectionChanged.connect(self._refresh_results_views)

        details_layout.addWidget(self.selected_track_label)
        details_layout.addWidget(self.reference_label)
        details_layout.addWidget(self.set_reference_button)
        details_layout.addWidget(self.clear_reference_button)
        details_layout.addWidget(QLabel("Satellites:"))
        details_layout.addWidget(self.satellites_list)
        details_layout.addWidget(self.add_satellite_button)
        details_layout.addWidget(self.remove_satellite_button)

        correction_group = QGroupBox("Manual Correction")
        correction_layout = QVBoxLayout(correction_group)

        self.move_bbox_button = QPushButton("Move BBox Mode")
        self.move_bbox_button.setCheckable(True)
        self.move_bbox_button.toggled.connect(self._toggle_move_bbox_mode)
        self.move_bbox_button.setToolTip(
            "Enable ROI moving in the preview. Drag inside ROI to move it. Changes affect only the current frame."
        )

        self.change_bbox_size_button = QPushButton("Change BBox Size Mode")
        self.change_bbox_size_button.setCheckable(True)
        self.change_bbox_size_button.toggled.connect(self._toggle_change_bbox_size_mode)
        self.change_bbox_size_button.setToolTip(
            "Enable ROI size editing in the preview. Drag an edge or corner to resize. Shift + mouse wheel over ROI also resizes it. Changes affect only the current frame."
        )

        self.bbox_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.bbox_scale_slider.setRange(25, 300)
        self.bbox_scale_slider.setSingleStep(5)
        self.bbox_scale_slider.setPageStep(10)
        self.bbox_scale_slider.setValue(100)
        self.bbox_scale_slider.valueChanged.connect(self._on_bbox_scale_slider_changed)
        self.bbox_scale_slider.setEnabled(False)
        self.bbox_scale_slider.setToolTip(
            "Rescale the current bbox around its center. 100% means unchanged size."
        )
        self.bbox_scale_label = QLabel("BBox scale: 100%")

        bbox_scale_layout = QVBoxLayout()
        bbox_scale_layout.addWidget(self.bbox_scale_label)
        bbox_scale_layout.addWidget(self.bbox_scale_slider)

        self.recompute_fit_button = QPushButton("Recompute 2D Gauss")
        self.recompute_fit_button.clicked.connect(self._recompute_selected_fit)
        self.recompute_fit_button.setToolTip(
            "Recompute fit only for the selected track on the current frame."
        )

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Apply size until frame:"))
        self.apply_size_until_spinbox = QSpinBox()
        if self.sorted_frame_ids:
            self.apply_size_until_spinbox.setRange(
                self.sorted_frame_ids[0], self.sorted_frame_ids[-1]
            )
            self.apply_size_until_spinbox.setValue(self.sorted_frame_ids[-1])
        else:
            self.apply_size_until_spinbox.setRange(0, 0)
            self.apply_size_until_spinbox.setValue(0)
        self.apply_size_to_end_checkbox = QCheckBox("To end")
        self.apply_size_to_end_checkbox.setChecked(True)
        self.apply_size_to_end_checkbox.toggled.connect(self._sync_resize_range_controls)
        range_layout.addWidget(self.apply_size_until_spinbox, 1)
        range_layout.addWidget(self.apply_size_to_end_checkbox)

        self.apply_size_to_range_button = QPushButton("Apply BBox Size To Range")
        self.apply_size_to_range_button.clicked.connect(self._apply_current_bbox_size_to_range)
        self.apply_size_to_range_button.setToolTip(
            "Apply the current bbox width/height from the current frame to a selected range while keeping each spot centered."
        )

        self.recompute_range_button = QPushButton("Batch Recompute Range")
        self.recompute_range_button.clicked.connect(self._recompute_selected_fit_range)
        self.recompute_range_button.setToolTip(
            "Recompute 2D Gaussian fits for the selected track from the current frame to the chosen end frame or the end of the set."
        )

        self.manual_correction_label = QLabel(
            "Select a track and frame. Move or resize its bbox if needed, then recompute the local fit or batch-recompute the selected range."
        )
        self.manual_correction_label.setWordWrap(True)

        correction_layout.addWidget(self.move_bbox_button)
        correction_layout.addWidget(self.change_bbox_size_button)
        correction_layout.addLayout(bbox_scale_layout)
        correction_layout.addWidget(self.recompute_fit_button)
        correction_layout.addLayout(range_layout)
        correction_layout.addWidget(self.apply_size_to_range_button)
        correction_layout.addWidget(self.recompute_range_button)
        correction_layout.addWidget(self.manual_correction_label)

        group_layout.addWidget(self.groups_list)
        group_layout.addLayout(buttons_layout)
        group_layout.addWidget(self.groups_info_label)
        group_layout.addLayout(analysis_buttons_layout)
        group_layout.addWidget(export_layout_group)
        group_layout.addWidget(self.analysis_summary_label)
        group_layout.addWidget(details_group)
        group_layout.addWidget(correction_group)
        content_layout.addWidget(group_box)

        status_group = QGroupBox("Session")
        status_layout = QFormLayout(status_group)
        status_layout.addRow("Loaded tracks:", QLabel(str(len(self.tracks))))
        status_layout.addRow("Loaded frames:", QLabel(str(len(self.sorted_frame_ids))))
        self.positions_session_status_label = QLabel("No Analyze Positions session loaded.")
        self.positions_session_status_label.setWordWrap(True)
        status_layout.addRow("Session status:", self.positions_session_status_label)

        session_buttons_widget = QWidget()
        session_buttons_layout = QHBoxLayout(session_buttons_widget)
        session_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.save_positions_session_button = QPushButton("Save Session")
        self.save_positions_session_button.clicked.connect(self._save_positions_session)
        self.load_positions_session_button = QPushButton("Load Session")
        self.load_positions_session_button.clicked.connect(self._load_positions_session)
        session_buttons_layout.addWidget(self.save_positions_session_button)
        session_buttons_layout.addWidget(self.load_positions_session_button)
        status_layout.addRow(session_buttons_widget)
        content_layout.addWidget(status_group)
        content_layout.addStretch(1)

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        return panel

    def _initialize_view(self):
        if self.tracks_table.rowCount() > 0:
            self.tracks_table.selectRow(0)

        if self.sorted_frame_ids:
            self._on_slider_value_changed(0, fit=True)
            self.frame_slider.setValue(0)
        else:
            self.frame_label.setText("Frame: no frames")
            self.frame_meta_label.setText("No image data available.")
        self.image_viewer.set_roi_transparency_percent(
            self.bbox_transparency_combo.currentData()
        )
        self._sync_resize_range_controls()
        self._update_manual_correction_label()
        self._update_positions_session_status()
        self._refresh_gauss_preview()

    def _capture_original_roi_rects(self) -> dict[tuple[int, int], QRect]:
        original_rects: dict[tuple[int, int], QRect] = {}
        for track in self.tracks:
            for frame_id, spot in track.spots_by_image_id.items():
                original_rects[(frame_id, track.track_id)] = QRect(spot.roi_rect)
        return original_rects

    def _toggle_move_bbox_mode(self, enabled: bool):
        if enabled and self.change_bbox_size_button.isChecked():
            self.change_bbox_size_button.blockSignals(True)
            self.change_bbox_size_button.setChecked(False)
            self.change_bbox_size_button.blockSignals(False)
        self.move_bbox_button.setText("Move BBox Mode: ON" if enabled else "Move BBox Mode")
        self._apply_roi_edit_mode()

    def _toggle_change_bbox_size_mode(self, enabled: bool):
        if enabled and self.move_bbox_button.isChecked():
            self.move_bbox_button.blockSignals(True)
            self.move_bbox_button.setChecked(False)
            self.move_bbox_button.blockSignals(False)
        self.change_bbox_size_button.setText(
            "Change BBox Size Mode: ON" if enabled else "Change BBox Size Mode"
        )
        self._apply_roi_edit_mode()

    def _apply_roi_edit_mode(self):
        if self.move_bbox_button.isChecked():
            self.image_viewer.set_mode("MOVE_ROI")
        elif self.change_bbox_size_button.isChecked():
            self.image_viewer.set_mode("RESIZE_ROI")
        else:
            self.image_viewer.set_mode("PAN")
        self.bbox_scale_slider.setEnabled(self.change_bbox_size_button.isChecked())
        if self.change_bbox_size_button.isChecked():
            self._reset_bbox_scale_reference()
        else:
            self._bbox_scale_reference = None
            self._set_bbox_scale_slider_value(100)
        self._update_manual_correction_label()

    def _set_bbox_scale_slider_value(self, value: int):
        self.bbox_scale_slider.blockSignals(True)
        self.bbox_scale_slider.setValue(int(value))
        self.bbox_scale_slider.blockSignals(False)
        self.bbox_scale_label.setText(f"BBox scale: {int(value)}%")

    def _reset_bbox_scale_reference(self):
        track = self._get_selected_track()
        frame_id = self._get_current_frame_id()
        if track is None or frame_id is None:
            self._bbox_scale_reference = None
            self._set_bbox_scale_slider_value(100)
            return

        spot = track.spots_by_image_id.get(frame_id)
        if spot is None:
            self._bbox_scale_reference = None
            self._set_bbox_scale_slider_value(100)
            return

        rect = QRect(spot.roi_rect)
        self._bbox_scale_reference = {
            "frame_id": frame_id,
            "track_id": track.track_id,
            "rect": rect,
        }
        self._set_bbox_scale_slider_value(100)

    def _on_bbox_scale_slider_changed(self, value: int):
        self.bbox_scale_label.setText(f"BBox scale: {int(value)}%")
        if not self.change_bbox_size_button.isChecked():
            return

        track = self._get_selected_track()
        frame_id = self._get_current_frame_id()
        if track is None or frame_id is None:
            return

        spot = track.spots_by_image_id.get(frame_id)
        if spot is None:
            return

        ref = self._bbox_scale_reference
        if (
            ref is None
            or ref.get("frame_id") != frame_id
            or ref.get("track_id") != track.track_id
        ):
            self._reset_bbox_scale_reference()
            ref = self._bbox_scale_reference
            if ref is None:
                return

        base_rect = QRect(ref["rect"])
        scale = max(0.01, float(value) / 100.0)
        desired_width = max(1, int(round(base_rect.width() * scale)))
        desired_height = max(1, int(round(base_rect.height() * scale)))
        center_point = base_rect.center()
        new_rect = self._build_centered_rect_for_frame(
            frame_id,
            float(center_point.x()),
            float(center_point.y()),
            desired_width,
            desired_height,
        )
        self._apply_manual_bbox_rect(
            track,
            frame_id,
            spot,
            new_rect,
            refresh_scale_reference=False,
        )

    def _show_validation_properties_dialog(self):
        dialog = ValidationPropertiesDialog(self.validation_config, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.validation_config = dialog.get_validation_config()
        QMessageBox.information(
            self,
            "Validation Updated",
            (
                "Validation properties were updated.\n\n"
                "Use Run Positions Analysis to recalculate all frames with the new settings. "
                "Local recompute actions will also use the updated thresholds."
            ),
        )
        self._refresh_gauss_preview()
        self._update_track_overlay()
        self._update_manual_correction_label()

    def _save_positions_session(self):
        suggested_path = (
            str(self.positions_session_path)
            if self.positions_session_path is not None
            else "positions_analysis_session.positions.json"
        )
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analyze Positions Session",
            suggested_path,
            "Positions Session (*.positions.json *.json)",
        )
        if not filepath:
            return

        session_path = self._normalize_positions_session_path(filepath)
        session_payload = self._serialize_positions_session()
        try:
            with session_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    session_payload,
                    handle,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=True,
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Session Failed",
                f"Could not save Analyze Positions session:\n{exc}",
            )
            return

        self.positions_session_path = session_path
        self._update_positions_session_status()
        QMessageBox.information(
            self,
            "Session Saved",
            f"Analyze Positions session saved to:\n{session_path}",
        )

    def _load_positions_session(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Analyze Positions Session",
            str(self.positions_session_path) if self.positions_session_path is not None else "",
            "Positions Session (*.positions.json *.json)",
        )
        if not filepath:
            return

        session_path = Path(filepath)
        try:
            with session_path.open("r", encoding="utf-8") as handle:
                session_payload = json.load(handle)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Session Failed",
                f"Could not load Analyze Positions session:\n{exc}",
            )
            return

        try:
            load_summary = self._apply_loaded_positions_session(session_payload, session_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Session Failed",
                f"Could not apply Analyze Positions session:\n{exc}",
            )
            return

        summary_lines = [
            f"Groups restored: {load_summary['groups_loaded']}",
            f"Manual bbox corrections restored: {load_summary['corrections_loaded']}",
            f"Pending recomputes restored: {load_summary['pending_loaded']}",
            "Analysis results: loaded" if load_summary["results_loaded"] else "Analysis results: not present",
        ]
        if load_summary["skipped_corrections"] > 0:
            summary_lines.append(
                f"Skipped corrections: {load_summary['skipped_corrections']}"
            )
        if load_summary["skipped_pending"] > 0:
            summary_lines.append(
                f"Skipped pending recomputes: {load_summary['skipped_pending']}"
            )
        QMessageBox.information(
            self,
            "Session Loaded",
            "\n".join(summary_lines),
        )

    @staticmethod
    def _normalize_positions_session_path(filepath: str | Path) -> Path:
        path = Path(filepath)
        if path.name.lower().endswith(".positions.json"):
            return path
        if path.suffix.lower() == ".json":
            return path.with_name(f"{path.stem}.positions.json")
        return Path(f"{path}.positions.json")

    def _serialize_positions_session(self) -> dict:
        return {
            "type": POSITIONS_SESSION_TYPE,
            "version": POSITIONS_SESSION_VERSION,
            "groups": self._serialize_groups_for_analysis(),
            "validation_config": self._json_safe_value(self.validation_config),
            "manual_bbox_corrections": self._serialize_manual_bbox_corrections(),
            "pending_manual_recompute": self._serialize_pending_manual_recompute(),
            "positions_analysis_results": self._json_safe_value(self.positions_analysis_results),
            "view_state": self._serialize_positions_view_state(),
            "context": {
                "frame_ids": list(self.sorted_frame_ids),
                "track_ids": sorted(int(track_id) for track_id in self.track_map.keys()),
            },
        }

    def _serialize_manual_bbox_corrections(self) -> list[dict]:
        serialized: list[dict] = []
        for (frame_id, track_id), rect_payload in sorted(self.manual_bbox_corrections.items()):
            serialized.append(
                {
                    "frame_id": int(frame_id),
                    "track_id": int(track_id),
                    "x": int(rect_payload.get("x", 0)),
                    "y": int(rect_payload.get("y", 0)),
                    "width": int(rect_payload.get("width", 0)),
                    "height": int(rect_payload.get("height", 0)),
                }
            )
        return serialized

    def _serialize_pending_manual_recompute(self) -> list[dict]:
        serialized: list[dict] = []
        for frame_id, track_id in sorted(self.pending_manual_recompute):
            serialized.append(
                {
                    "frame_id": int(frame_id),
                    "track_id": int(track_id),
                }
            )
        return serialized

    def _serialize_positions_view_state(self) -> dict:
        return {
            "current_frame_id": self._get_current_frame_id(),
            "selected_track_id": self._get_selected_track().track_id if self._get_selected_track() else None,
            "selected_group_id": self._get_selected_group().group_id if self._get_selected_group() else None,
            "selected_satellite_track_id": self._get_selected_satellite_track_id(),
            "x_axis": self.x_axis_combo.currentData(),
            "bbox_transparency_percent": self.bbox_transparency_combo.currentData(),
            "show_selected_pair_only": bool(self.show_selected_pair_only_checkbox.isChecked()),
            "show_gauss_preview": bool(self.show_gauss_preview_checkbox.isChecked()),
            "apply_size_to_end": bool(self.apply_size_to_end_checkbox.isChecked()),
            "apply_size_until_frame": int(self.apply_size_until_spinbox.value()),
            "move_bbox_mode": bool(self.move_bbox_button.isChecked()),
            "change_bbox_size_mode": bool(self.change_bbox_size_button.isChecked()),
            "export_label_distance": int(self.export_label_distance_spin.value()),
            "export_label_spread": int(self.export_label_spread_spin.value()),
        }

    def _apply_loaded_positions_session(self, session_payload: dict, session_path: Path) -> dict:
        if not isinstance(session_payload, dict):
            raise ValueError("Session file does not contain a valid JSON object.")
        session_type = session_payload.get("type")
        if session_type not in (None, POSITIONS_SESSION_TYPE):
            raise ValueError(f"Unsupported session type: {session_type}")

        self.setUpdatesEnabled(False)
        try:
            self._restore_original_roi_rects()
            self.manual_bbox_corrections.clear()
            self.pending_manual_recompute.clear()
            self.positions_analysis_results = None
            self.positions_session_path = session_path

            self.validation_config = normalize_validation_config(
                session_payload.get("validation_config", DEFAULT_VALIDATION_CONFIG)
            )

            self.groups = self._deserialize_groups(
                session_payload.get("groups", [])
            )
            self._group_counter = max(
                (group.group_id for group in self.groups),
                default=0,
            )
            self._populate_groups_list()

            corrections_loaded, skipped_corrections = self._apply_manual_bbox_corrections_from_session(
                session_payload.get("manual_bbox_corrections", [])
            )
            pending_loaded, skipped_pending = self._restore_pending_manual_recompute(
                session_payload.get("pending_manual_recompute", [])
            )

            loaded_results = self._deserialize_positions_analysis_results(
                session_payload.get("positions_analysis_results")
            )
            self.positions_analysis_results = loaded_results
            if self.positions_analysis_results:
                self._refresh_positions_analysis_summary()
                self.analysis_summary_label.setText(
                    self._format_analysis_summary(self.positions_analysis_results)
                )
            else:
                self.analysis_summary_label.setText("No positions analysis results yet.")

            self._restore_positions_view_state(session_payload.get("view_state", {}))
            self._update_positions_session_status()
            self._refresh_group_details()
            self._refresh_results_views()
            self._update_track_overlay()
            self._update_manual_correction_label()
            self._refresh_gauss_preview()
        finally:
            self.setUpdatesEnabled(True)

        return {
            "groups_loaded": len(self.groups),
            "corrections_loaded": corrections_loaded,
            "pending_loaded": pending_loaded,
            "results_loaded": bool(self.positions_analysis_results),
            "skipped_corrections": skipped_corrections,
            "skipped_pending": skipped_pending,
        }

    def _deserialize_groups(self, raw_groups) -> list[PositionsAnalysisGroup]:
        deserialized: list[PositionsAnalysisGroup] = []
        if not isinstance(raw_groups, list):
            return deserialized

        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                continue
            try:
                group_id = int(raw_group.get("group_id"))
            except Exception:
                continue

            name = str(raw_group.get("name", f"Group {group_id}"))
            reference_track_id = raw_group.get("reference_track_id")
            try:
                reference_track_id = (
                    int(reference_track_id) if reference_track_id is not None else None
                )
            except Exception:
                reference_track_id = None

            if reference_track_id is not None and reference_track_id not in self.track_map:
                reference_track_id = None

            satellite_track_ids: list[int] = []
            for raw_track_id in raw_group.get("satellite_track_ids", []):
                try:
                    track_id = int(raw_track_id)
                except Exception:
                    continue
                if track_id not in self.track_map:
                    continue
                if reference_track_id is not None and track_id == reference_track_id:
                    continue
                if track_id not in satellite_track_ids:
                    satellite_track_ids.append(track_id)

            deserialized.append(
                PositionsAnalysisGroup(
                    group_id=group_id,
                    name=name,
                    reference_track_id=reference_track_id,
                    satellite_track_ids=satellite_track_ids,
                )
            )

        return deserialized

    def _restore_original_roi_rects(self):
        for (frame_id, track_id), rect in self.original_roi_rects.items():
            track = self.track_map.get(track_id)
            if track is None:
                continue
            spot = track.spots_by_image_id.get(frame_id)
            if spot is None:
                continue
            spot.roi_rect = QRect(rect)

    def _apply_manual_bbox_corrections_from_session(self, raw_entries) -> tuple[int, int]:
        loaded_count = 0
        skipped_count = 0
        if not isinstance(raw_entries, list):
            return loaded_count, skipped_count

        for entry in raw_entries:
            if not isinstance(entry, dict):
                skipped_count += 1
                continue
            try:
                frame_id = int(entry.get("frame_id"))
                track_id = int(entry.get("track_id"))
                x = int(entry.get("x"))
                y = int(entry.get("y"))
                width = int(entry.get("width"))
                height = int(entry.get("height"))
            except Exception:
                skipped_count += 1
                continue

            track = self.track_map.get(track_id)
            if track is None or frame_id not in track.spots_by_image_id:
                skipped_count += 1
                continue

            if width <= 0 or height <= 0:
                skipped_count += 1
                continue

            rect = QRect(x, y, width, height)
            spot = track.spots_by_image_id[frame_id]
            spot.roi_rect = QRect(rect)
            self.manual_bbox_corrections[(frame_id, track_id)] = {
                "x": int(rect.x()),
                "y": int(rect.y()),
                "width": int(rect.width()),
                "height": int(rect.height()),
            }
            loaded_count += 1

        return loaded_count, skipped_count

    def _restore_pending_manual_recompute(self, raw_entries) -> tuple[int, int]:
        loaded_count = 0
        skipped_count = 0
        pending: set[tuple[int, int]] = set()
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    skipped_count += 1
                    continue
                try:
                    frame_id = int(entry.get("frame_id"))
                    track_id = int(entry.get("track_id"))
                except Exception:
                    skipped_count += 1
                    continue
                if (frame_id, track_id) not in self.manual_bbox_corrections:
                    skipped_count += 1
                    continue
                pending.add((frame_id, track_id))
                loaded_count += 1
        self.pending_manual_recompute = pending
        return loaded_count, skipped_count

    def _deserialize_positions_analysis_results(self, raw_results) -> dict | None:
        if not isinstance(raw_results, dict):
            return None

        valid_frame_ids = set(self.sorted_frame_ids)
        valid_track_ids = set(self.track_map.keys())
        valid_group_ids = {group.group_id for group in self.groups}

        fit_points_by_frame: dict[int, dict[int, dict]] = {}
        raw_fit_points = raw_results.get("fit_points_by_frame", {})
        if isinstance(raw_fit_points, dict):
            for raw_frame_id, frame_payload in raw_fit_points.items():
                try:
                    frame_id = int(raw_frame_id)
                except Exception:
                    continue
                if frame_id not in valid_frame_ids or not isinstance(frame_payload, dict):
                    continue

                normalized_frame_payload: dict[int, dict] = {}
                for raw_track_id, fit_payload in frame_payload.items():
                    try:
                        track_id = int(raw_track_id)
                    except Exception:
                        continue
                    if track_id not in valid_track_ids or not isinstance(fit_payload, dict):
                        continue
                    normalized_frame_payload[track_id] = fit_payload

                if normalized_frame_payload:
                    fit_points_by_frame[frame_id] = normalized_frame_payload

        rows: list[dict] = []
        raw_rows = raw_results.get("rows", [])
        if isinstance(raw_rows, list):
            for raw_row in raw_rows:
                if not isinstance(raw_row, dict):
                    continue
                try:
                    frame_id = int(raw_row.get("frame_id"))
                    group_id = int(raw_row.get("group_id"))
                    reference_track_id = int(raw_row.get("reference_track_id"))
                    satellite_track_id = int(raw_row.get("satellite_track_id"))
                except Exception:
                    continue

                if frame_id not in valid_frame_ids:
                    continue
                if group_id not in valid_group_ids:
                    continue
                if reference_track_id not in valid_track_ids or satellite_track_id not in valid_track_ids:
                    continue
                rows.append(dict(raw_row))

        results = {
            "groups": self._serialize_groups_for_analysis(),
            "rows": rows,
            "fit_points_by_frame": fit_points_by_frame,
            "summary": dict(raw_results.get("summary", {})),
            "validation_config": normalize_validation_config(
                raw_results.get("validation_config", self.validation_config)
            ),
        }
        return results

    def _restore_positions_view_state(self, raw_view_state):
        if not isinstance(raw_view_state, dict):
            raw_view_state = {}

        transparency = raw_view_state.get(
            "bbox_transparency_percent",
            self.bbox_transparency_combo.currentData(),
        )
        transparency_index = self.bbox_transparency_combo.findData(transparency)
        if transparency_index >= 0:
            self.bbox_transparency_combo.setCurrentIndex(transparency_index)

        x_axis = raw_view_state.get("x_axis", self.x_axis_combo.currentData())
        x_axis_index = self.x_axis_combo.findData(x_axis)
        if x_axis_index >= 0:
            self.x_axis_combo.setCurrentIndex(x_axis_index)

        self.show_selected_pair_only_checkbox.setChecked(
            bool(raw_view_state.get("show_selected_pair_only", False))
        )
        self.show_gauss_preview_checkbox.setChecked(
            bool(raw_view_state.get("show_gauss_preview", False))
        )
        self.apply_size_to_end_checkbox.setChecked(
            bool(raw_view_state.get("apply_size_to_end", True))
        )

        current_frame_id = self._coerce_int(raw_view_state.get("current_frame_id"))
        if current_frame_id is not None:
            self._set_current_frame_by_id(current_frame_id)
        else:
            self._set_current_frame_by_id(self.sorted_frame_ids[0] if self.sorted_frame_ids else None)

        target_frame_id = self._coerce_int(raw_view_state.get("apply_size_until_frame"))
        if target_frame_id is not None and self.sorted_frame_ids:
            clamped_target = int(
                np.clip(target_frame_id, self.apply_size_until_spinbox.minimum(), self.apply_size_until_spinbox.maximum())
            )
            self.apply_size_until_spinbox.setValue(clamped_target)

        export_label_distance = self._coerce_int(raw_view_state.get("export_label_distance"))
        if export_label_distance is not None:
            clamped_distance = int(
                np.clip(
                    export_label_distance,
                    self.export_label_distance_spin.minimum(),
                    self.export_label_distance_spin.maximum(),
                )
            )
            self.export_label_distance_spin.setValue(clamped_distance)

        export_label_spread = self._coerce_int(raw_view_state.get("export_label_spread"))
        if export_label_spread is not None:
            clamped_spread = int(
                np.clip(
                    export_label_spread,
                    self.export_label_spread_spin.minimum(),
                    self.export_label_spread_spin.maximum(),
                )
            )
            self.export_label_spread_spin.setValue(clamped_spread)

        selected_track_id = self._coerce_int(raw_view_state.get("selected_track_id"))
        if selected_track_id is not None:
            self._select_track_by_id(selected_track_id)
        elif self.tracks_table.rowCount() > 0:
            self.tracks_table.selectRow(0)

        selected_group_id = self._coerce_int(raw_view_state.get("selected_group_id"))
        if selected_group_id is not None:
            self._select_group_by_id(selected_group_id)
        elif self.groups_list.count() > 0:
            self.groups_list.setCurrentRow(0)

        selected_satellite_track_id = self._coerce_int(
            raw_view_state.get("selected_satellite_track_id")
        )
        if selected_satellite_track_id is not None:
            self._select_satellite_by_track_id(selected_satellite_track_id)

        move_mode = bool(raw_view_state.get("move_bbox_mode", False))
        resize_mode = bool(raw_view_state.get("change_bbox_size_mode", False))
        if resize_mode:
            self.change_bbox_size_button.setChecked(True)
        elif move_mode:
            self.move_bbox_button.setChecked(True)
        else:
            self.move_bbox_button.setChecked(False)
            self.change_bbox_size_button.setChecked(False)

    def _update_positions_session_status(self):
        session_path_text = (
            str(self.positions_session_path)
            if self.positions_session_path is not None
            else "No Analyze Positions session loaded."
        )
        if self.positions_session_path is not None:
            status_text = (
                f"{self.positions_session_path.name}\n"
                f"Groups: {len(self.groups)} | "
                f"Corrections: {len(self.manual_bbox_corrections)} | "
                f"Results: {'yes' if self.positions_analysis_results else 'no'}"
            )
            self.positions_session_status_label.setToolTip(session_path_text)
        else:
            status_text = session_path_text
            self.positions_session_status_label.setToolTip("")
        self.positions_session_status_label.setText(status_text)

    @staticmethod
    def _json_safe_value(value):
        if isinstance(value, dict):
            return {str(key): PositionsAnalysisDialog._json_safe_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [PositionsAnalysisDialog._json_safe_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return PositionsAnalysisDialog._json_safe_value(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, QRect):
            return {
                "x": int(value.x()),
                "y": int(value.y()),
                "width": int(value.width()),
                "height": int(value.height()),
            }
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _coerce_int(value) -> int | None:
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    def _set_current_frame_by_id(self, frame_id: int | None):
        if frame_id is None or frame_id not in self.sorted_frame_ids:
            return
        frame_index = self.sorted_frame_ids.index(frame_id)
        self.frame_slider.setValue(frame_index)

    def _select_group_by_id(self, group_id: int):
        for row, group in enumerate(self.groups):
            if group.group_id == group_id:
                self.groups_list.setCurrentRow(row)
                return

    def _select_satellite_by_track_id(self, track_id: int):
        for row in range(self.satellites_list.count()):
            item = self.satellites_list.item(row)
            if item is None:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            try:
                if int(data) == int(track_id):
                    self.satellites_list.setCurrentRow(row)
                    return
            except Exception:
                continue

    def _sync_resize_range_controls(self, _checked=None):
        if not hasattr(self, "apply_size_until_spinbox"):
            return
        if not self.sorted_frame_ids:
            self.apply_size_until_spinbox.setEnabled(False)
            return

        current_frame_id = self._get_current_frame_id()
        if current_frame_id is None:
            current_frame_id = self.sorted_frame_ids[0]
        last_frame_id = self.sorted_frame_ids[-1]

        self.apply_size_until_spinbox.setRange(current_frame_id, last_frame_id)
        if self.apply_size_to_end_checkbox.isChecked():
            self.apply_size_until_spinbox.setValue(last_frame_id)
            self.apply_size_until_spinbox.setEnabled(False)
        else:
            if self.apply_size_until_spinbox.value() < current_frame_id:
                self.apply_size_until_spinbox.setValue(current_frame_id)
            self.apply_size_until_spinbox.setEnabled(True)

    def _toggle_gauss_preview(self, enabled: bool):
        self.gauss_preview_group.setVisible(bool(enabled))
        self._refresh_gauss_preview()

    def _on_bbox_transparency_changed(self, _index=None):
        self.image_viewer.set_roi_transparency_percent(
            self.bbox_transparency_combo.currentData()
        )

    def _on_selected_pair_filter_toggled(self, _checked=None):
        frame_id = self._get_current_frame_id()
        if frame_id is not None:
            self._update_frame_rois(frame_id)
        self._update_track_overlay()

    def _populate_tracks_table(self):
        self.tracks_table.setRowCount(len(self.tracks))
        for row, track in enumerate(self.tracks):
            frame_ids = sorted(track.spots_by_image_id.keys())
            start_frame = frame_ids[0] if frame_ids else -1
            end_frame = frame_ids[-1] if frame_ids else -1

            self.tracks_table.setItem(row, 0, QTableWidgetItem(str(track.track_id)))
            self.tracks_table.setItem(row, 1, QTableWidgetItem(str(start_frame)))
            self.tracks_table.setItem(row, 2, QTableWidgetItem(str(end_frame)))
            self.tracks_table.setItem(row, 3, QTableWidgetItem(str(len(frame_ids))))

        self.track_stats_label.setText(
            f"Loaded tracks: {len(self.tracks)}\n"
            "Selecting a track highlights its current ROI and trajectory.\n"
            "The selected track can be assigned as a reference or satellite."
        )

    def _populate_groups_list(self):
        self.groups_list.clear()
        for group in self.groups:
            reference = (
                f"T{group.reference_track_id}"
                if group.reference_track_id is not None
                else "none"
            )
            satellites = len(group.satellite_track_ids)
            self.groups_list.addItem(
                f"{group.name} | Ref: {reference} | Sat: {satellites}"
            )

    def _add_group(self):
        self._group_counter += 1
        self.groups.append(
            PositionsAnalysisGroup(
                group_id=self._group_counter,
                name=f"Group {self._group_counter}",
            )
        )
        self._populate_groups_list()
        self.groups_list.setCurrentRow(self.groups_list.count() - 1)
        self._refresh_group_details()

    def _remove_selected_group(self):
        row = self.groups_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a group to remove.")
            return
        del self.groups[row]
        self._populate_groups_list()
        if self.groups:
            self.groups_list.setCurrentRow(min(row, len(self.groups) - 1))
        self._refresh_group_details()

    def _get_selected_track(self) -> Track | None:
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            return None
        item = self.tracks_table.item(current_row, 0)
        if item is None:
            return None
        track_id = int(item.text())
        return next((t for t in self.tracks if t.track_id == track_id), None)

    def _get_selected_group(self) -> PositionsAnalysisGroup | None:
        row = self.groups_list.currentRow()
        if row < 0 or row >= len(self.groups):
            return None
        return self.groups[row]

    def _on_track_selection_changed(self):
        if self.change_bbox_size_button.isChecked():
            self._reset_bbox_scale_reference()
        self._refresh_selected_track_label()
        self._update_track_overlay()
        self._refresh_gauss_preview()

    def _on_slider_value_changed(self, index: int, fit: bool = False):
        if not self.sorted_frame_ids or not (0 <= index < len(self.sorted_frame_ids)):
            return
        frame_id = self.sorted_frame_ids[index]
        self._show_frame(frame_id, fit=fit)
        self.frame_label.setText(
            f"Frame {index + 1}/{len(self.sorted_frame_ids)} (ID: {frame_id})"
        )
        if self.change_bbox_size_button.isChecked():
            self._reset_bbox_scale_reference()
        self._sync_resize_range_controls()
        self._update_manual_correction_label()

    def _show_frame(self, frame_id: int, fit: bool = False):
        leed_file = self.all_files_map.get(frame_id)
        if leed_file is None:
            return

        image_data = load_tiff(leed_file.filepath)
        self.current_frame_id = frame_id
        self.current_frame_image_data = image_data
        pixmap = self._convert_numpy_to_pixmap(image_data)
        self.image_viewer.set_image(pixmap, fit=fit)
        self.image_viewer.set_roi_transparency_percent(
            self.bbox_transparency_combo.currentData()
        )
        self._update_frame_rois(frame_id)
        self.frame_meta_label.setText(
            f"Temperature: {leed_file.temperature} | "
            f"Coverage: {leed_file.coverage if leed_file.coverage is not None else '-'} | "
            f"Energy: {leed_file.energy}"
        )
        self._update_track_overlay()
        self._refresh_gauss_preview()

    def _update_frame_rois(self, frame_id: int):
        visible_track_ids = self._get_visible_track_ids_for_frame(frame_id)
        spots_in_frame: List[SpotROI] = []
        for track in self.tracks:
            if visible_track_ids is not None and track.track_id not in visible_track_ids:
                continue
            spot = track.spots_by_image_id.get(frame_id)
            if spot is not None:
                spots_in_frame.append(spot)
        self.image_viewer.update_rois(spots_in_frame)

    def _get_visible_track_ids_for_frame(self, frame_id: int) -> set[int] | None:
        if not self.show_selected_pair_only_checkbox.isChecked():
            return None

        group = self._get_selected_group()
        sat_track_id = self._get_selected_satellite_track_id()
        if group is None or group.reference_track_id is None or sat_track_id is None:
            return None

        visible_track_ids = {group.reference_track_id, sat_track_id}
        return visible_track_ids

    def _on_group_selection_changed(self):
        self._refresh_group_details()

    def _serialize_groups_for_analysis(self) -> list[dict]:
        return [
            {
                "group_id": group.group_id,
                "name": group.name,
                "reference_track_id": group.reference_track_id,
                "satellite_track_ids": list(group.satellite_track_ids),
            }
            for group in self.groups
        ]

    def start_positions_analysis(self):
        groups_config = self._serialize_groups_for_analysis()
        valid_groups = [
            group
            for group in groups_config
            if group.get("reference_track_id") is not None
            and group.get("satellite_track_ids")
        ]
        if not valid_groups:
            QMessageBox.information(
                self,
                "No Valid Groups",
                "Define at least one group with one reference track and at least one satellite track.",
            )
            return

        self.run_analysis_button.setEnabled(False)
        self.positions_analysis_cancel_event = Event()
        worker = GenericWorker(
            run_positions_analysis,
            self.tracks,
            valid_groups,
            self.all_files_map,
            validation_config=self.validation_config,
            cancel_event=self.positions_analysis_cancel_event,
        )
        self._launch_positions_analysis_worker(worker)

    def _launch_positions_analysis_worker(self, worker: GenericWorker):
        self.progress_dialog = QProgressDialog(
            "Performing positions analysis...",
            "Cancel",
            0,
            100,
            self,
        )
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

        self.worker_thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.worker_thread)

        self.progress_dialog.canceled.connect(self._handle_positions_analysis_cancel)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.on_positions_analysis_finished)
        self.worker.error.connect(self.on_positions_analysis_error)

        self.worker.finished.connect(self.progress_dialog.accept)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.progress_dialog.reject)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, "worker_thread", None))
        self.worker_thread.finished.connect(lambda: setattr(self, "worker", None))

        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def _handle_positions_analysis_cancel(self):
        if self.positions_analysis_cancel_event:
            self.positions_analysis_cancel_event.set()
        if getattr(self, "progress_dialog", None):
            self.progress_dialog.setLabelText("Cancelling, please wait...")
            self.progress_dialog.setEnabled(False)
        if self.worker_thread:
            try:
                self.worker_thread.requestInterruption()
            except RuntimeError:
                pass

    def on_positions_analysis_finished(self, results: dict | None):
        self.positions_analysis_cancel_event = None
        self.run_analysis_button.setEnabled(True)

        if results is None:
            self.analysis_summary_label.setText("Positions analysis was cancelled.")
            self._update_export_button_state()
            self._refresh_results_views()
            QMessageBox.information(
                self,
                "Analysis Cancelled",
                "Positions analysis was cancelled.",
            )
            return

        self.positions_analysis_results = results
        self.validation_config = normalize_validation_config(
            results.get("validation_config", self.validation_config)
        )
        self.pending_manual_recompute.clear()
        self.analysis_summary_label.setText(self._format_analysis_summary(results))
        self._update_export_button_state()
        self._refresh_results_views()
        self._update_track_overlay()
        QMessageBox.information(
            self,
            "Positions Analysis Finished",
            "Positions analysis completed successfully.",
        )

    def on_positions_analysis_error(self, error_message: str):
        self.positions_analysis_cancel_event = None
        self.run_analysis_button.setEnabled(True)
        self.analysis_summary_label.setText("Positions analysis failed.")
        self._update_export_button_state()
        self._refresh_results_views()
        self._update_track_overlay()
        self._update_manual_correction_label()
        QMessageBox.critical(
            self,
            "Positions Analysis Error",
            f"An error occurred:\n{error_message}",
        )

    def _update_export_button_state(self):
        if not hasattr(self, "export_results_button"):
            return
        has_results = bool(
            self.positions_analysis_results
            and self.positions_analysis_results.get("rows")
        )
        self.export_results_button.setEnabled(has_results)

    def _export_positions_results(self):
        if not self.positions_analysis_results or not self.positions_analysis_results.get("rows"):
            QMessageBox.information(
                self,
                "No Results",
                "Run Positions Analysis first or load a session with results before exporting.",
            )
            return

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
        )
        if not directory:
            return

        export_dir = Path(directory)
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            exported_group_dirs = self._export_positions_results_grouped(export_dir)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Could not export Analyze Positions results:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export Finished",
            (
                "Analyze Positions export completed.\n\n"
                f"Root: {export_dir}\n"
                f"Groups exported: {len(exported_group_dirs)}"
            ),
        )

    def _export_positions_results_grouped(self, export_dir: Path) -> list[Path]:
        if not self.positions_analysis_results:
            return []

        exported_dirs: list[Path] = []
        for group in self.positions_analysis_results.get("groups", []):
            group_id = int(group.get("group_id", 0))
            group_dir = export_dir / f"group_{group_id:02d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            self._export_positions_results_csv(group_dir, group_id=group_id)
            self._export_positions_overlay_images(group_dir, group_id=group_id)
            exported_dirs.append(group_dir)

        return exported_dirs

    def _export_positions_results_csv(self, export_dir: Path, *, group_id: int | None = None) -> Path:
        suffix = f"_group_{group_id:02d}" if group_id is not None else ""
        output_path = export_dir / f"positions_analysis_results{suffix}.csv"
        rows = self._build_export_point_rows(group_id_filter=group_id)
        fieldnames = [
            "frame_id",
            "group_id",
            "point_role",
            "track_id",
            "distance_from_reference",
            "x",
            "y",
            "position_source",
            "fit_ok",
            "quality_status",
            "quality_flags",
            "sigma_x",
            "sigma_y",
            "fwhm_x",
            "fwhm_y",
            "fwhm_x_error",
            "fwhm_y_error",
            "theta",
            "r_squared",
        ]

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(
                rows,
                key=lambda current: (
                    current.get("frame_id", -1),
                    current.get("group_id", -1),
                    str(current.get("point_role", "")),
                    current.get("track_id", -1),
                ),
            ):
                writer.writerow(
                    {
                        fieldname: self._csv_safe_value(row.get(fieldname))
                        for fieldname in fieldnames
                    }
                )

        return output_path

    def _export_positions_overlay_images(self, export_dir: Path, *, group_id: int | None = None) -> Path:
        images_dir = export_dir / "positions_overlay_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        frame_ids = list(self.sorted_frame_ids)
        saved_count = 0
        skipped_missing_meta = 0
        skipped_missing_image = 0
        progress_dialog = QProgressDialog(
            "Exporting positions overlay images...",
            "Cancel",
            0,
            len(frame_ids),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.processEvents()

        for index, frame_id in enumerate(frame_ids, start=1):
            if progress_dialog.wasCanceled():
                break

            leed_file = self.all_files_map.get(frame_id)
            if leed_file is None:
                skipped_missing_meta += 1
                progress_dialog.setValue(index)
                QApplication.processEvents()
                continue

            image_data = load_tiff(leed_file.filepath)
            if image_data is None or np.asarray(image_data).size == 0:
                skipped_missing_image += 1
                progress_dialog.setValue(index)
                QApplication.processEvents()
                continue

            overlay_image = self._build_positions_overlay_export_image(
                frame_id,
                image_data,
                group_id_filter=group_id,
            )
            output_name = f"positions_{frame_id:04d}_{Path(leed_file.filename).stem}.png"
            output_path = images_dir / output_name
            overlay_image.save(str(output_path), format="PNG")
            if not output_path.exists():
                raise RuntimeError(f"Image export failed for frame {frame_id}: {output_path}")
            saved_count += 1

            progress_dialog.setValue(index)
            QApplication.processEvents()

        progress_dialog.close()

        if saved_count == 0:
            raise RuntimeError(
                "No overlay images were exported. "
                f"Skipped missing metadata: {skipped_missing_meta}, "
                f"skipped missing images: {skipped_missing_image}, "
                f"cancelled: {'yes' if progress_dialog.wasCanceled() else 'no'}."
            )
        return images_dir

    def _build_positions_overlay_export_image(
        self,
        frame_id: int,
        image_data,
        *,
        group_id_filter: int | None = None,
    ) -> Image.Image:
        export_image = self._convert_numpy_to_export_image(image_data)
        draw = ImageDraw.Draw(export_image)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        overlay_groups = self._build_positions_overlay_payload(
            frame_id,
            respect_selection_filter=False,
            group_id_filter=group_id_filter,
        )
        self._draw_export_overlay_groups(draw, overlay_groups, font, export_image.size)
        return export_image

    def _format_analysis_summary(self, results: dict | None) -> str:
        if not results:
            return "No positions analysis results yet."
        summary = results.get("summary", {})
        return (
            "Results ready:\n"
            f"Groups: {summary.get('groups_configured', 0)}\n"
            f"Tracks in analysis: {summary.get('tracks_in_analysis', 0)}\n"
            f"Frames: {summary.get('frames_processed', 0)}\n"
            f"Rows: {summary.get('rows_generated', 0)}\n"
            f"Successful fits: {summary.get('successful_fits', 0)}\n"
            f"Failed fits: {summary.get('failed_fits', 0)}\n"
            f"Suspicious fits: {summary.get('suspicious_fits', 0)}\n"
            f"Suspicious rows: {summary.get('suspicious_rows', 0)}"
        )

    def _select_track_by_id(self, track_id: int):
        for row in range(self.tracks_table.rowCount()):
            item = self.tracks_table.item(row, 0)
            if item is None:
                continue
            try:
                current_track_id = int(item.text())
            except Exception:
                continue
            if current_track_id == track_id:
                self.tracks_table.selectRow(row)
                return

    def _resolve_track_for_spot(self, frame_id: int, spot_id: int) -> tuple[Track, SpotROI] | None:
        for track in self.tracks:
            spot = track.spots_by_image_id.get(frame_id)
            if spot is not None and spot.id == spot_id:
                return track, spot
        return None

    def _handle_roi_moved(self, spot_id: int, new_rect_f: QRectF):
        frame_id = self._get_current_frame_id()
        if frame_id is None:
            return

        resolved = self._resolve_track_for_spot(frame_id, spot_id)
        if resolved is None:
            return

        track, spot = resolved
        new_rect = new_rect_f.toRect()
        if new_rect.width() <= 0 or new_rect.height() <= 0:
            return

        self._apply_manual_bbox_rect(track, frame_id, spot, new_rect)
        self._select_track_by_id(track.track_id)

    def _apply_manual_bbox_rect(
        self,
        track: Track,
        frame_id: int,
        spot: SpotROI,
        new_rect: QRect,
        *,
        refresh_scale_reference: bool = True,
    ):
        if new_rect.width() <= 0 or new_rect.height() <= 0:
            return

        spot.roi_rect = QRect(new_rect)

        correction_key = (frame_id, track.track_id)
        self.manual_bbox_corrections[correction_key] = {
            "x": int(new_rect.x()),
            "y": int(new_rect.y()),
            "width": int(new_rect.width()),
            "height": int(new_rect.height()),
        }
        self.pending_manual_recompute.add(correction_key)

        if self.change_bbox_size_button.isChecked() and refresh_scale_reference:
            self._bbox_scale_reference = {
                "frame_id": frame_id,
                "track_id": track.track_id,
                "rect": QRect(new_rect),
            }
            self._set_bbox_scale_slider_value(100)

        self._update_frame_rois(frame_id)
        self._update_positions_session_status()
        self._update_manual_correction_label()
        self._update_track_overlay()
        self._refresh_gauss_preview()

    def _recompute_selected_fit(self):
        if not self.positions_analysis_results:
            QMessageBox.information(
                self,
                "No Analysis Results",
                "Run positions analysis first before recomputing a local fit.",
            )
            return

        frame_id = self._get_current_frame_id()
        track = self._get_selected_track()
        if frame_id is None or track is None:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a track and a valid frame first.",
            )
            return

        if frame_id not in track.spots_by_image_id:
            QMessageBox.information(
                self,
                "Track Missing On Frame",
                "The selected track does not exist on the current frame.",
            )
            return

        if not self._track_participates_in_positions_analysis(track.track_id):
            QMessageBox.information(
                self,
                "Track Not In Analysis",
                "The selected track is not part of any configured positions-analysis group.",
            )
            return

        leed_file = self.all_files_map.get(frame_id)
        if leed_file is None:
            QMessageBox.warning(
                self,
                "Missing Frame",
                "Could not resolve the current frame metadata.",
            )
            return

        image_data = load_tiff(leed_file.filepath)
        if image_data is None or np.asarray(image_data).size == 0:
            QMessageBox.warning(
                self,
                "Image Load Failed",
                "Could not load image data for the current frame.",
            )
            return

        fit_payload = fit_track_position_on_frame(
            track,
            frame_id,
            image_data,
            validation_config=self.validation_config,
        )
        fit_points_by_frame = self.positions_analysis_results.setdefault("fit_points_by_frame", {})
        frame_fits = fit_points_by_frame.setdefault(frame_id, {})
        frame_fits[track.track_id] = fit_payload

        self._refresh_rows_for_frame(frame_id)
        self._refresh_positions_analysis_summary()
        self.pending_manual_recompute.discard((frame_id, track.track_id))

        self.analysis_summary_label.setText(
            self._format_analysis_summary(self.positions_analysis_results)
        )
        self._refresh_results_views()
        self._update_track_overlay()
        self._update_manual_correction_label()
        self._refresh_gauss_preview()

    def _resolve_selected_range_target_frame_id(self, current_frame_id: int) -> int | None:
        if not self.sorted_frame_ids:
            return None

        if self.apply_size_to_end_checkbox.isChecked():
            return self.sorted_frame_ids[-1]

        target_frame_id = int(self.apply_size_until_spinbox.value())
        if target_frame_id < current_frame_id:
            QMessageBox.information(
                self,
                "Invalid Range",
                "The end frame must be greater than or equal to the current frame.",
            )
            return None
        return target_frame_id

    def _get_track_frame_ids_in_selected_range(
        self,
        track: Track,
        current_frame_id: int,
    ) -> list[int]:
        target_frame_id = self._resolve_selected_range_target_frame_id(current_frame_id)
        if target_frame_id is None:
            return []

        return [
            frame_id
            for frame_id in self.sorted_frame_ids
            if current_frame_id <= frame_id <= target_frame_id
            and frame_id in track.spots_by_image_id
        ]

    def _recompute_selected_fit_range(self):
        if not self.positions_analysis_results:
            QMessageBox.information(
                self,
                "No Analysis Results",
                "Run positions analysis first before batch recomputing fits.",
            )
            return

        frame_id = self._get_current_frame_id()
        track = self._get_selected_track()
        if frame_id is None or track is None:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a track and a valid frame first.",
            )
            return

        if frame_id not in track.spots_by_image_id:
            QMessageBox.information(
                self,
                "Track Missing On Frame",
                "The selected track does not exist on the current frame.",
            )
            return

        if not self._track_participates_in_positions_analysis(track.track_id):
            QMessageBox.information(
                self,
                "Track Not In Analysis",
                "The selected track is not part of any configured positions-analysis group.",
            )
            return

        affected_frame_ids = self._get_track_frame_ids_in_selected_range(track, frame_id)
        if not affected_frame_ids:
            QMessageBox.information(
                self,
                "No Matching Frames",
                "No track instances were found in the selected range.",
            )
            return

        fit_points_by_frame = self.positions_analysis_results.setdefault("fit_points_by_frame", {})
        processed_frame_ids: list[int] = []
        cancelled = False

        progress_dialog = QProgressDialog(
            f"Recomputing 2D Gauss for T{track.track_id}...",
            "Cancel",
            0,
            len(affected_frame_ids),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()

        for index, target_frame_id in enumerate(affected_frame_ids, start=1):
            if progress_dialog.wasCanceled():
                cancelled = True
                break

            leed_file = self.all_files_map.get(target_frame_id)
            if leed_file is None:
                progress_dialog.setValue(index)
                QApplication.processEvents()
                continue

            image_data = load_tiff(leed_file.filepath)
            if image_data is None or np.asarray(image_data).size == 0:
                progress_dialog.setValue(index)
                QApplication.processEvents()
                continue

            fit_payload = fit_track_position_on_frame(
                track,
                target_frame_id,
                image_data,
                validation_config=self.validation_config,
            )
            frame_fits = fit_points_by_frame.setdefault(target_frame_id, {})
            frame_fits[track.track_id] = fit_payload
            self.pending_manual_recompute.discard((target_frame_id, track.track_id))
            processed_frame_ids.append(target_frame_id)

            progress_dialog.setValue(index)
            QApplication.processEvents()

        progress_dialog.close()

        if processed_frame_ids:
            for processed_frame_id in processed_frame_ids:
                self._refresh_rows_for_frame(processed_frame_id)
            self._refresh_positions_analysis_summary()
            self.analysis_summary_label.setText(
                self._format_analysis_summary(self.positions_analysis_results)
            )
            self._refresh_results_views()
            self._update_track_overlay()
            self._update_manual_correction_label()
            self._refresh_gauss_preview()

        if cancelled:
            QMessageBox.information(
                self,
                "Batch Recompute Cancelled",
                (
                    f"Recomputed {len(processed_frame_ids)} frame(s) for track T{track.track_id} "
                    "before cancellation."
                ),
            )
            return

        QMessageBox.information(
            self,
            "Batch Recompute Finished",
            (
                f"Recomputed 2D Gauss for {len(processed_frame_ids)} frame(s) "
                f"for track T{track.track_id}, from frame {affected_frame_ids[0]} "
                f"to frame {affected_frame_ids[-1]}."
            ),
        )

    def _apply_current_bbox_size_to_range(self):
        track = self._get_selected_track()
        current_frame_id = self._get_current_frame_id()
        if track is None or current_frame_id is None:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a track and a current frame first.",
            )
            return

        current_spot = track.spots_by_image_id.get(current_frame_id)
        if current_spot is None:
            QMessageBox.information(
                self,
                "Track Missing On Frame",
                "The selected track is not present on the current frame.",
            )
            return

        desired_width = max(1, int(current_spot.roi_rect.width()))
        desired_height = max(1, int(current_spot.roi_rect.height()))

        affected_frame_ids = self._get_track_frame_ids_in_selected_range(track, current_frame_id)
        if not affected_frame_ids:
            QMessageBox.information(
                self,
                "No Matching Frames",
                "No track instances were found in the selected range.",
            )
            return

        updated_count = 0
        for frame_id in affected_frame_ids:
            spot = track.spots_by_image_id[frame_id]
            center_point = spot.roi_rect.center()
            center_x = float(center_point.x())
            center_y = float(center_point.y())

            new_rect = self._build_centered_rect_for_frame(
                frame_id,
                center_x,
                center_y,
                desired_width,
                desired_height,
            )
            spot.roi_rect = new_rect

            correction_key = (frame_id, track.track_id)
            self.manual_bbox_corrections[correction_key] = {
                "x": int(new_rect.x()),
                "y": int(new_rect.y()),
                "width": int(new_rect.width()),
                "height": int(new_rect.height()),
            }
            self.pending_manual_recompute.add(correction_key)
            updated_count += 1

        if current_frame_id in affected_frame_ids:
            self._update_frame_rois(current_frame_id)
            self._update_track_overlay()
            self._refresh_gauss_preview()
        self._update_positions_session_status()
        self._update_manual_correction_label()

        QMessageBox.information(
            self,
            "BBox Size Applied",
            (
                f"Applied bbox size {desired_width}x{desired_height} to {updated_count} frame(s) "
                f"for track T{track.track_id}, from frame {current_frame_id} "
                f"to frame {affected_frame_ids[-1]}.\n\n"
                "The center of each bbox was preserved. The affected frames are marked for recompute."
            ),
        )

    def _build_centered_rect_for_frame(
        self,
        frame_id: int,
        center_x: float,
        center_y: float,
        desired_width: int,
        desired_height: int,
    ) -> QRect:
        frame_shape = self._get_frame_image_shape(frame_id)
        width = max(1, int(desired_width))
        height = max(1, int(desired_height))

        if frame_shape is not None:
            image_height, image_width = frame_shape
            max_width = max(1, int(np.floor(2.0 * min(center_x, image_width - center_x))))
            max_height = max(1, int(np.floor(2.0 * min(center_y, image_height - center_y))))
            width = min(width, max_width if max_width > 0 else width)
            height = min(height, max_height if max_height > 0 else height)

        left = int(round(center_x - width / 2.0))
        top = int(round(center_y - height / 2.0))
        rect = QRect(left, top, width, height)

        if frame_shape is None:
            return rect

        image_height, image_width = frame_shape
        if rect.left() < 0:
            rect.moveLeft(0)
        if rect.top() < 0:
            rect.moveTop(0)
        if rect.right() >= image_width:
            rect.moveRight(image_width - 1)
        if rect.bottom() >= image_height:
            rect.moveBottom(image_height - 1)
        return rect

    def _get_frame_image_shape(self, frame_id: int) -> tuple[int, int] | None:
        cached = self.frame_image_shape_cache.get(frame_id)
        if cached is not None:
            return cached

        image_data = None
        if self.current_frame_id == frame_id and self.current_frame_image_data is not None:
            image_data = self.current_frame_image_data
        else:
            leed_file = self.all_files_map.get(frame_id)
            if leed_file is None:
                return None
            image_data = load_tiff(leed_file.filepath)
        if image_data is None or np.asarray(image_data).size == 0:
            return None

        arr = np.asarray(image_data)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        shape = (int(arr.shape[0]), int(arr.shape[1]))
        self.frame_image_shape_cache[frame_id] = shape
        return shape

    def _refresh_rows_for_frame(self, frame_id: int):
        if not self.positions_analysis_results:
            return

        rows = self.positions_analysis_results.get("rows", [])
        frame_fits = self.positions_analysis_results.get("fit_points_by_frame", {}).get(frame_id, {})
        for row in rows:
            if row.get("frame_id") != frame_id:
                continue
            ref_fit = frame_fits.get(row.get("reference_track_id"))
            sat_fit = frame_fits.get(row.get("satellite_track_id"))
            self._apply_fit_payloads_to_row(row, ref_fit, sat_fit)

    def _apply_fit_payloads_to_row(
        self,
        row: dict,
        ref_fit: dict | None,
        sat_fit: dict | None,
    ):
        reference_track_id = row.get("reference_track_id")
        satellite_track_id = row.get("satellite_track_id")
        frame_id = row.get("frame_id")

        row["reference_spot_present"] = bool(ref_fit and ref_fit.get("track_present"))
        row["satellite_spot_present"] = bool(sat_fit and sat_fit.get("track_present"))
        row["reference_fit_ok"] = bool(ref_fit and ref_fit.get("fit_ok"))
        row["satellite_fit_ok"] = bool(sat_fit and sat_fit.get("fit_ok"))
        row["reference_quality_status"] = (
            ref_fit.get("quality_status", "failed") if ref_fit else "failed"
        )
        row["satellite_quality_status"] = (
            sat_fit.get("quality_status", "failed") if sat_fit else "failed"
        )
        row["reference_quality_flags"] = list(ref_fit.get("quality_flags", [])) if ref_fit else ["fit_missing"]
        row["satellite_quality_flags"] = list(sat_fit.get("quality_flags", [])) if sat_fit else ["fit_missing"]
        row["reference_validation_failures"] = list(ref_fit.get("validation_failures", [])) if ref_fit else ["fit_missing"]
        row["satellite_validation_failures"] = list(sat_fit.get("validation_failures", [])) if sat_fit else ["fit_missing"]
        row["reference_warning_flags"] = list(ref_fit.get("warning_flags", [])) if ref_fit else []
        row["satellite_warning_flags"] = list(sat_fit.get("warning_flags", [])) if sat_fit else []
        row["reference_suspicious"] = bool(ref_fit and ref_fit.get("is_suspicious", False))
        row["satellite_suspicious"] = bool(sat_fit and sat_fit.get("is_suspicious", False))
        row["reference_spot_id"] = ref_fit.get("spot_id") if ref_fit else None
        row["satellite_spot_id"] = sat_fit.get("spot_id") if sat_fit else None
        row["x_fit_ref"] = self._payload_numeric(ref_fit, "global_center_x")
        row["y_fit_ref"] = self._payload_numeric(ref_fit, "global_center_y")
        row["x_fit_sat"] = self._payload_numeric(sat_fit, "global_center_x")
        row["y_fit_sat"] = self._payload_numeric(sat_fit, "global_center_y")
        row["reference_r_squared"] = self._payload_numeric(ref_fit, "r_squared")
        row["satellite_r_squared"] = self._payload_numeric(sat_fit, "r_squared")
        row["reference_manually_corrected"] = (frame_id, reference_track_id) in self.manual_bbox_corrections
        row["satellite_manually_corrected"] = (frame_id, satellite_track_id) in self.manual_bbox_corrections
        row["manually_corrected"] = bool(
            row["reference_manually_corrected"] or row["satellite_manually_corrected"]
        )

        if row["reference_fit_ok"] and row["satellite_fit_ok"]:
            dx = float(row["x_fit_sat"] - row["x_fit_ref"])
            dy = float(row["y_fit_sat"] - row["y_fit_ref"])
            row["dx"] = dx
            row["dy"] = dy
            row["r"] = float(np.hypot(dx, dy))
        else:
            row["dx"] = np.nan
            row["dy"] = np.nan
            row["r"] = np.nan
        row["suspicious"] = bool(
            row["reference_suspicious"] or row["satellite_suspicious"]
        )
        row["row_quality_status"] = self._combine_quality_statuses(
            row["reference_quality_status"],
            row["satellite_quality_status"],
        )

    @staticmethod
    def _payload_numeric(payload: dict | None, key: str):
        if not payload:
            return np.nan
        value = payload.get(key, np.nan)
        try:
            value = float(value)
        except Exception:
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _refresh_positions_analysis_summary(self):
        if not self.positions_analysis_results:
            return

        fit_points_by_frame = self.positions_analysis_results.get("fit_points_by_frame", {})
        successful_fits = 0
        failed_fits = 0
        suspicious_fits = 0
        tracks_in_analysis: set[int] = set()

        for frame_fits in fit_points_by_frame.values():
            for track_id, payload in frame_fits.items():
                if not payload.get("track_present"):
                    continue
                tracks_in_analysis.add(int(track_id))
                if payload.get("fit_ok"):
                    successful_fits += 1
                else:
                    failed_fits += 1
                if payload.get("is_suspicious"):
                    suspicious_fits += 1

        summary = self.positions_analysis_results.setdefault("summary", {})
        summary["groups_configured"] = len(self.positions_analysis_results.get("groups", []))
        summary["tracks_in_analysis"] = len(tracks_in_analysis)
        summary["frames_processed"] = len(fit_points_by_frame)
        summary["rows_generated"] = len(self.positions_analysis_results.get("rows", []))
        summary["successful_fits"] = successful_fits
        summary["failed_fits"] = failed_fits
        summary["suspicious_fits"] = suspicious_fits
        summary["suspicious_rows"] = int(
            sum(1 for row in self.positions_analysis_results.get("rows", []) if row.get("suspicious"))
        )

    def _track_participates_in_positions_analysis(self, track_id: int) -> bool:
        if not self.positions_analysis_results:
            return False
        for group in self.positions_analysis_results.get("groups", []):
            if group.get("reference_track_id") == track_id:
                return True
            if track_id in group.get("satellite_track_ids", []):
                return True
        return False

    def _update_manual_correction_label(self):
        track = self._get_selected_track()
        frame_id = self._get_current_frame_id()
        if track is None or frame_id is None:
            self.manual_correction_label.setText(
                "Select a track and frame. Use Move BBox Mode to shift ROI or Change BBox Size Mode to resize it with the slider, edge drag, or Shift + mouse wheel, then recompute the local fit or batch-recompute the selected range."
            )
            return

        correction_key = (frame_id, track.track_id)
        parts = [f"Frame {frame_id} | Track T{track.track_id}."]
        if correction_key in self.pending_manual_recompute:
            parts.append("BBox moved. Recompute 2D Gauss to update local results.")
        elif correction_key in self.manual_bbox_corrections:
            parts.append("Manual bbox correction saved for this frame.")

        if self.positions_analysis_results:
            fit_payload = (
                self.positions_analysis_results
                .get("fit_points_by_frame", {})
                .get(frame_id, {})
                .get(track.track_id)
            )
            if fit_payload and fit_payload.get("track_present"):
                if fit_payload.get("fit_ok"):
                    r_squared = fit_payload.get("r_squared")
                    quality_status = self._format_quality_status_text(
                        fit_payload.get("quality_status")
                    )
                    if self._safe_float(r_squared) is not None:
                        parts.append(
                            f"Current fit {quality_status} (R²={float(r_squared):.4f})."
                        )
                    else:
                        parts.append(f"Current fit {quality_status}.")
                else:
                    parts.append(
                        f"Current fit {self._format_quality_status_text(fit_payload.get('quality_status'))}."
                    )
                quality_flags_text = format_quality_flags(
                    fit_payload.get("quality_flags", [])
                )
                if quality_flags_text:
                    parts.append(f"Flags: {quality_flags_text}.")
            else:
                parts.append("No fit stored yet for this track/frame.")
        else:
            parts.append("Run positions analysis first to populate fit results.")

        if self.move_bbox_button.isChecked():
            parts.append("Move BBox Mode is ON. Drag inside ROI to move it.")
        if self.change_bbox_size_button.isChecked():
            parts.append("Change BBox Size Mode is ON. Use the BBox scale slider to rescale around the center, or drag an edge/corner; Shift + mouse wheel over ROI also resizes it.")

        parts.append("Use Batch Recompute Range to update the selected track over the chosen frame range.")

        self.manual_correction_label.setText(" ".join(parts))

    def _refresh_selected_track_label(self):
        selected_track = self._get_selected_track()
        if selected_track is None:
            self.selected_track_label.setText("Selected track: none")
            self._update_manual_correction_label()
            return
        self.selected_track_label.setText(
            f"Selected track: T{selected_track.track_id}"
        )
        self._update_manual_correction_label()

    def _refresh_group_details(self):
        self._refresh_selected_track_label()
        group = self._get_selected_group()
        self.satellites_list.clear()

        if group is None:
            self.reference_label.setText("Reference: none")
            self.groups_info_label.setText(
                "Select a group to assign the selected track as reference or satellite."
            )
            self._refresh_results_views()
            return

        if group.reference_track_id is None:
            self.reference_label.setText("Reference: none")
        else:
            self.reference_label.setText(f"Reference: T{group.reference_track_id}")

        for track_id in group.satellite_track_ids:
            self.satellites_list.addItem(f"T{track_id}")
            self.satellites_list.item(self.satellites_list.count() - 1).setData(
                Qt.ItemDataRole.UserRole, track_id
            )

        self.groups_info_label.setText(
            f"{group.name}: choose one reference track and add one or more satellites."
        )
        if self.satellites_list.count() > 0:
            self.satellites_list.setCurrentRow(0)
        self._refresh_results_views()
        self._update_manual_correction_label()

    def _set_selected_track_as_reference(self):
        group = self._get_selected_group()
        track = self._get_selected_track()
        if group is None:
            QMessageBox.information(self, "No Group", "Create or select a group first.")
            return
        if track is None:
            QMessageBox.information(self, "No Track", "Select a track first.")
            return

        group.reference_track_id = track.track_id
        group.satellite_track_ids = [
            tid for tid in group.satellite_track_ids if tid != track.track_id
        ]
        self._populate_groups_list()
        self.groups_list.setCurrentRow(self.groups.index(group))
        self._refresh_group_details()

    def _clear_reference_from_group(self):
        group = self._get_selected_group()
        if group is None:
            QMessageBox.information(self, "No Group", "Select a group first.")
            return
        group.reference_track_id = None
        self._populate_groups_list()
        self.groups_list.setCurrentRow(self.groups.index(group))
        self._refresh_group_details()

    def _add_selected_track_as_satellite(self):
        group = self._get_selected_group()
        track = self._get_selected_track()
        if group is None:
            QMessageBox.information(self, "No Group", "Create or select a group first.")
            return
        if track is None:
            QMessageBox.information(self, "No Track", "Select a track first.")
            return
        if group.reference_track_id == track.track_id:
            QMessageBox.information(
                self,
                "Track Already Reference",
                "The selected track is already the reference track for this group.",
            )
            return
        if track.track_id in group.satellite_track_ids:
            QMessageBox.information(
                self,
                "Already Added",
                "The selected track is already in the satellites list.",
            )
            return

        group.satellite_track_ids.append(track.track_id)
        self._populate_groups_list()
        self.groups_list.setCurrentRow(self.groups.index(group))
        self._refresh_group_details()

    def _remove_selected_satellite(self):
        group = self._get_selected_group()
        row = self.satellites_list.currentRow()
        if group is None:
            QMessageBox.information(self, "No Group", "Select a group first.")
            return
        if row < 0 or row >= len(group.satellite_track_ids):
            QMessageBox.information(
                self,
                "No Satellite",
                "Select a satellite from the list first.",
            )
            return

        del group.satellite_track_ids[row]
        self._populate_groups_list()
        self.groups_list.setCurrentRow(self.groups.index(group))
        self._refresh_group_details()

    def _update_track_overlay(self):
        selected_track = self._get_selected_track()
        if selected_track is None:
            self.image_viewer.highlight_roi(None)
            self.image_viewer.show_track_path(None)
            self._update_positions_overlay()
            return

        self.image_viewer.show_track_path(selected_track)

        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            self.image_viewer.highlight_roi(None)
            self._update_positions_overlay()
            return

        frame_id = self.sorted_frame_ids[slider_index]
        spot = selected_track.spots_by_image_id.get(frame_id)
        self.image_viewer.highlight_roi(spot.id if spot is not None else None)
        self._update_positions_overlay()

    def _get_selected_satellite_track_id(self) -> int | None:
        item = self.satellites_list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return None
        try:
            return int(data)
        except Exception:
            return None

    def _get_rows_for_current_selection(self) -> list[dict]:
        if not self.positions_analysis_results:
            return []

        group = self._get_selected_group()
        sat_track_id = self._get_selected_satellite_track_id()
        if group is None or sat_track_id is None:
            return []

        rows = self.positions_analysis_results.get("rows", [])
        filtered_rows = [
            row
            for row in rows
            if row.get("group_id") == group.group_id
            and row.get("satellite_track_id") == sat_track_id
        ]
        return sorted(filtered_rows, key=lambda row: row.get("frame_id", -1))

    def _refresh_results_views(self):
        rows = self._get_rows_for_current_selection()
        self._update_results_table(rows)
        self._update_results_plot(rows)
        self._update_export_button_state()
        frame_id = self._get_current_frame_id()
        if frame_id is not None:
            self._update_frame_rois(frame_id)
        self._update_positions_overlay()
        self._update_positions_session_status()
        self._update_manual_correction_label()
        self._refresh_gauss_preview()

    def _update_results_table(self, rows: list[dict]):
        self.results_table.setRowCount(len(rows))
        if not rows:
            self.results_status_label.setText(
                "No results to display for the current group/satellite selection."
                if self.positions_analysis_results
                else "Run positions analysis to populate the plot and results table."
            )
            return

        x_axis_name = self.x_axis_combo.currentText()
        suspicious_rows = sum(1 for row in rows if row.get("suspicious"))
        self.results_status_label.setText(
            f"Showing {len(rows)} row(s) for selected group/satellite. "
            f"X axis: {x_axis_name}. Suspicious rows: {suspicious_rows}."
        )

        for row_idx, row in enumerate(rows):
            flags_text = self._format_row_quality_flags(row)
            values = [
                row.get("frame_id"),
                row.get("temperature"),
                row.get("coverage"),
                row.get("dx"),
                row.get("dy"),
                row.get("r"),
                self._format_quality_status_text(row.get("reference_quality_status")),
                self._format_quality_status_text(row.get("satellite_quality_status")),
                flags_text,
            ]
            for col_idx, value in enumerate(values):
                if isinstance(value, (float, np.floating)):
                    text = "NaN" if np.isnan(value) else f"{value:.4f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col_idx == 8 and flags_text:
                    item.setToolTip(flags_text)
                self.results_table.setItem(row_idx, col_idx, item)
            self._style_results_table_row(row_idx, row)

    def _update_results_plot(self, rows: list[dict]):
        axes = self.results_plot.axes
        axes.clear()

        if not rows:
            axes.set_title("Distance vs X")
            axes.set_xlabel("Coverage / Temperature")
            axes.set_ylabel("Distance r")
            axes.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            self._finalize_results_plot_draw()
            return

        x_key = self.x_axis_combo.currentData()
        valid_rows = []
        for row in rows:
            x_val = row.get(x_key)
            y_val = row.get("r")
            if x_val is None or y_val is None:
                continue
            try:
                x_num = float(x_val)
                y_num = float(y_val)
            except Exception:
                continue
            if not np.isfinite(x_num) or not np.isfinite(y_num):
                continue
            valid_rows.append((row, x_num, y_num))

        if not valid_rows:
            axes.set_title("Distance vs X")
            axes.set_xlabel("Coverage" if x_key == "coverage" else "Temperature")
            axes.set_ylabel("Distance r")
            axes.text(
                0.5,
                0.5,
                "No valid fitted points",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            self._finalize_results_plot_draw()
            return

        valid_rows.sort(key=lambda item: (item[1], item[0].get("frame_id", -1)))
        x_values = [item[1] for item in valid_rows]
        y_values = [item[2] for item in valid_rows]
        suspicious_rows = [item for item in valid_rows if item[0].get("suspicious")]
        normal_rows = [item for item in valid_rows if not item[0].get("suspicious")]

        group = self._get_selected_group()
        sat_track_id = self._get_selected_satellite_track_id()
        ref_track_id = group.reference_track_id if group else None

        axes.plot(x_values, y_values, linewidth=1.3, color="#2f6db3", alpha=0.75)
        if normal_rows:
            axes.scatter(
                [item[1] for item in normal_rows],
                [item[2] for item in normal_rows],
                s=28,
                color="#2f6db3",
                label="OK",
                zorder=3,
            )
        if suspicious_rows:
            axes.scatter(
                [item[1] for item in suspicious_rows],
                [item[2] for item in suspicious_rows],
                s=42,
                color="#f59e0b",
                edgecolors="black",
                linewidths=0.6,
                label="Suspicious",
                zorder=4,
            )
        axes.set_xlabel("Coverage" if x_key == "coverage" else "Temperature")
        axes.set_ylabel("Distance r")
        axes.set_title(
            f"Group {group.group_id if group else '-'} | "
            f"Ref T{ref_track_id if ref_track_id is not None else '-'} -> "
            f"Sat T{sat_track_id if sat_track_id is not None else '-'}"
        )
        axes.grid(True, alpha=0.3)
        if suspicious_rows:
            axes.legend(loc="best")
        if len(valid_rows) > 1 and len({round(value, 9) for value in x_values}) == 1:
            axes.text(
                0.02,
                0.98,
                "Selected X axis is constant in this subset",
                transform=axes.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff4cc", "edgecolor": "#d4a72c", "alpha": 0.9},
            )
        self._finalize_results_plot_draw()

    def _finalize_results_plot_draw(self):
        canvas_width = self.results_plot.width()
        canvas_height = self.results_plot.height()
        if canvas_width > 1 and canvas_height > 1:
            try:
                self.results_plot.fig.tight_layout()
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
                print(f"[PositionsAnalysis] tight_layout skipped: {exc}")
                self.results_plot.fig.subplots_adjust(
                    left=0.12,
                    right=0.97,
                    bottom=0.16,
                    top=0.90,
                )
        self.results_plot.draw()

    def _refresh_gauss_preview(self):
        if not hasattr(self, "gauss_preview_group"):
            return
        if not self.show_gauss_preview_checkbox.isChecked():
            self._clear_gauss_preview(
                "Enable the live preview and select a track to inspect the current Gaussian fit."
            )
            return

        track = self._get_selected_track()
        frame_id = self._get_current_frame_id()
        if track is None or frame_id is None:
            self._clear_gauss_preview("Select a track and a valid frame to inspect the fit.")
            return

        if frame_id not in track.spots_by_image_id:
            self._clear_gauss_preview(
                f"Track T{track.track_id} is not present on frame {frame_id}."
            )
            return

        full_image = self.current_frame_image_data
        if full_image is None or self.current_frame_id != frame_id:
            leed_file = self.all_files_map.get(frame_id)
            if leed_file is None:
                self._clear_gauss_preview("Could not load image data for the current frame.")
                return
            full_image = load_tiff(leed_file.filepath)
        if full_image is None or np.asarray(full_image).size == 0:
            self._clear_gauss_preview("Current frame image is empty.")
            return

        spot = track.spots_by_image_id[frame_id]
        rect = spot.roi_rect
        image_arr = np.asarray(full_image)
        if image_arr.ndim == 3 and image_arr.shape[0] in (3, 4) and image_arr.shape[-1] not in (3, 4):
            image_arr = np.transpose(image_arr, (1, 2, 0))

        height_full = image_arr.shape[0]
        width_full = image_arr.shape[1]
        x0 = max(0, int(rect.x()))
        y0 = max(0, int(rect.y()))
        x1 = min(width_full, x0 + int(rect.width()))
        y1 = min(height_full, y0 + int(rect.height()))
        if x1 <= x0 or y1 <= y0:
            self._clear_gauss_preview("Current ROI is empty.")
            return

        roi_data = image_arr[y0:y1, x0:x1].copy()
        roi_to_fit = prepare_roi_for_fit(roi_data)
        if roi_to_fit is None or np.asarray(roi_to_fit).size == 0 or np.asarray(roi_to_fit).ndim != 2:
            self._clear_gauss_preview("Current ROI could not be converted to 2D fit data.")
            return

        fit_results = fit_spot_to_gaussian(
            roi_to_fit,
            center_preference_factor=self.validation_config.get("center_preference_factor", 0.0),
        )
        quality_info = evaluate_fit_quality(fit_results, roi_to_fit.shape, self.validation_config)
        self._draw_gauss_preview(track.track_id, frame_id, roi_to_fit, fit_results, quality_info)

    def _draw_gauss_preview(
        self,
        track_id: int,
        frame_id: int,
        roi_to_fit: np.ndarray,
        fit_results: dict | None,
        quality_info: dict,
    ):
        ax_roi = self.gauss_preview_canvas.axes_roi
        ax_fit = self.gauss_preview_canvas.axes_fit
        ax_residual = self.gauss_preview_canvas.axes_residual
        ax_roi.clear()
        ax_fit.clear()
        ax_residual.clear()

        ax_roi.imshow(roi_to_fit, cmap="viridis", origin="lower")
        ax_roi.set_title("ROI")
        ax_roi.set_xlabel("X")
        ax_roi.set_ylabel("Y")

        status_parts = [f"Track T{track_id} | Frame {frame_id}"]
        fit_ok = bool(quality_info.get("fit_ok"))

        if fit_results:
            height, width = roi_to_fit.shape
            x = np.arange(width)
            y = np.arange(height)
            X, Y = np.meshgrid(x, y)
            model_param_names = [
                "amplitude",
                "center_x",
                "center_y",
                "sigma_x",
                "sigma_y",
                "theta",
                "offset",
            ]
            model_params = {
                key: fit_results[key]
                for key in model_param_names
                if key in fit_results
            }
            if len(model_params) == len(model_param_names):
                fitted_z = gaussian_2d((X, Y), **model_params).reshape(height, width)
                residual = roi_to_fit - fitted_z

                center_x = float(fit_results["center_x"])
                center_y = float(fit_results["center_y"])
                ax_roi.scatter(
                    [center_x],
                    [center_y],
                    s=48,
                    c="white",
                    edgecolors="black",
                    linewidths=0.8,
                )

                ax_fit.imshow(fitted_z, cmap="plasma", origin="lower")
                ax_fit.scatter(
                    [center_x],
                    [center_y],
                    s=48,
                    c="white",
                    edgecolors="black",
                    linewidths=0.8,
                )
                ax_fit.set_title("Fitted Gaussian")
                ax_fit.set_xlabel("X")
                ax_fit.set_ylabel("Y")

                vmax = float(np.max(np.abs(residual))) if residual.size else 0.0
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                ax_residual.imshow(
                    residual,
                    cmap="coolwarm",
                    origin="lower",
                    vmin=-vmax,
                    vmax=vmax,
                )
                ax_residual.set_title("Residual")
                ax_residual.set_xlabel("X")
                ax_residual.set_ylabel("Y")

                r_squared = self._safe_float(fit_results.get("r_squared"))
                status_parts.append(
                    self._format_quality_status_text(quality_info.get("quality_status"))
                )
                if r_squared is not None:
                    status_parts.append(f"R²={r_squared:.4f}")
                status_parts.append(
                    f"center=({center_x:.2f}, {center_y:.2f})"
                )
            else:
                ax_fit.text(0.5, 0.5, "Incomplete fit", ha="center", va="center", transform=ax_fit.transAxes)
                ax_residual.text(0.5, 0.5, "No residual", ha="center", va="center", transform=ax_residual.transAxes)
                status_parts.append("fit parameters incomplete")
        else:
            ax_fit.text(
                0.5,
                0.5,
                "Fit failed",
                ha="center",
                va="center",
                transform=ax_fit.transAxes,
            )
            ax_residual.text(
                0.5,
                0.5,
                "No residual",
                ha="center",
                va="center",
                transform=ax_residual.transAxes,
            )
            ax_fit.set_title("Fitted Gaussian")
            ax_residual.set_title("Residual")
            status_parts.append("fit failed")

        validation_failures_text = format_quality_flags(
            quality_info.get("validation_failures", [])
        )
        warning_flags_text = format_quality_flags(
            quality_info.get("warning_flags", [])
        )
        if validation_failures_text:
            status_parts.append(f"failures: {validation_failures_text}")
        if warning_flags_text:
            status_parts.append(f"warnings: {warning_flags_text}")

        for axis in (ax_fit, ax_residual):
            if not axis.get_title():
                axis.set_xlabel("X")
                axis.set_ylabel("Y")

        correction_key = (frame_id, track_id)
        if correction_key in self.pending_manual_recompute:
            status_parts.append("preview uses current bbox; stored analysis result is outdated")
        elif correction_key in self.manual_bbox_corrections:
            status_parts.append("manual bbox correction saved")

        self.gauss_preview_status_label.setText(" | ".join(status_parts))
        self._finalize_gauss_preview_draw()

    def _clear_gauss_preview(self, message: str):
        if not hasattr(self, "gauss_preview_canvas"):
            return
        for axis, title in (
            (self.gauss_preview_canvas.axes_roi, "ROI"),
            (self.gauss_preview_canvas.axes_fit, "Fitted Gaussian"),
            (self.gauss_preview_canvas.axes_residual, "Residual"),
        ):
            axis.clear()
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        self.gauss_preview_status_label.setText(message)
        self._finalize_gauss_preview_draw()

    def _finalize_gauss_preview_draw(self):
        canvas_width = self.gauss_preview_canvas.width()
        canvas_height = self.gauss_preview_canvas.height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        try:
            self.gauss_preview_canvas.fig.tight_layout()
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            print(f"[PositionsAnalysis] gauss preview tight_layout skipped: {exc}")
            self.gauss_preview_canvas.fig.subplots_adjust(
                left=0.06,
                right=0.98,
                bottom=0.18,
                top=0.88,
                wspace=0.28,
            )

        try:
            self.gauss_preview_canvas.draw()
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            print(f"[PositionsAnalysis] gauss preview draw skipped: {exc}")
            try:
                self.gauss_preview_canvas.fig.subplots_adjust(
                    left=0.06,
                    right=0.98,
                    bottom=0.18,
                    top=0.88,
                    wspace=0.28,
                )
                self.gauss_preview_canvas.draw_idle()
            except (np.linalg.LinAlgError, RuntimeError, ValueError) as redraw_exc:
                print(f"[PositionsAnalysis] gauss preview redraw skipped: {redraw_exc}")

    def _update_positions_overlay(self):
        frame_id = self._get_current_frame_id()
        if frame_id is None:
            self.image_viewer.update_positions_overlay([], [])
            return

        overlay_groups = self._build_positions_overlay_payload(frame_id)
        focus_points = self._build_focus_points_overlay_payload(frame_id)
        self.image_viewer.update_positions_overlay(overlay_groups, focus_points)

    def _get_current_frame_id(self) -> int | None:
        if not self.sorted_frame_ids:
            return None
        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            return None
        return self.sorted_frame_ids[slider_index]

    def _build_positions_overlay_payload(
        self,
        frame_id: int,
        *,
        respect_selection_filter: bool = True,
        group_id_filter: int | None = None,
    ) -> list[dict]:
        if not self.positions_analysis_results:
            return []

        fit_points_by_frame = self.positions_analysis_results.get("fit_points_by_frame", {})
        frame_fits = fit_points_by_frame.get(frame_id, {})
        results_groups = self.positions_analysis_results.get("groups", [])
        if not results_groups:
            return []

        selected_group = self._get_selected_group()
        selected_group_id = selected_group.group_id if selected_group else None
        selected_satellite_track_id = self._get_selected_satellite_track_id()
        show_selected_pair_only = (
            self.show_selected_pair_only_checkbox.isChecked()
            if respect_selection_filter
            else False
        )

        if show_selected_pair_only and (
            selected_group_id is None or selected_satellite_track_id is None
        ):
            return []

        overlay_groups: list[dict] = []
        for group in results_groups:
            group_id = int(group.get("group_id"))
            if group_id_filter is not None and group_id != int(group_id_filter):
                continue
            if show_selected_pair_only and group_id != selected_group_id:
                continue
            emphasized = group_id == selected_group_id
            reference_track_id = group.get("reference_track_id")
            reference = self._build_overlay_point(
                frame_id,
                reference_track_id,
                frame_fits,
                label=f"G{group_id}-R:T{reference_track_id}",
                selected=emphasized,
            )
            satellites = []
            for index, track_id in enumerate(group.get("satellite_track_ids", []), start=1):
                if show_selected_pair_only and track_id != selected_satellite_track_id:
                    continue
                satellites.append(
                    self._build_overlay_point(
                        frame_id,
                        track_id,
                        frame_fits,
                        label=f"G{group_id}-S{index}:T{track_id}",
                        selected=emphasized and track_id == selected_satellite_track_id,
                    )
                )

            overlay_groups.append(
                {
                    "group_id": group_id,
                    "group_name": group.get("name", f"Group {group_id}"),
                    "color": self._group_color(group_id, emphasized=emphasized),
                    "emphasized": emphasized,
                    "reference": reference,
                    "satellites": satellites,
                }
            )

        return overlay_groups

    def _build_focus_points_overlay_payload(self, frame_id: int) -> list[dict]:
        if not self.positions_analysis_results:
            return []

        selected_satellite_track_id = self._get_selected_satellite_track_id()
        if selected_satellite_track_id is None:
            return []

        track = self.track_map.get(selected_satellite_track_id)
        if track is None:
            return []

        fit_points_by_frame = self.positions_analysis_results.get("fit_points_by_frame", {})
        frame_fits = fit_points_by_frame.get(frame_id, {})
        correction_key = (frame_id, track.track_id)
        fit_payload = (
            {}
            if correction_key in self.pending_manual_recompute
            else (frame_fits.get(track.track_id) or {})
        )

        fit_x = fit_payload.get("global_center_x")
        fit_y = fit_payload.get("global_center_y")
        if not self._is_finite_pair(fit_x, fit_y):
            return []

        fit_ok = bool(fit_payload.get("fit_ok", False))
        suspicious = bool(fit_payload.get("is_suspicious", False))
        if fit_ok:
            label = f"T{track.track_id}{'!' if suspicious else ''}"
        else:
            label = f"T{track.track_id}?"
        return [
            {
                "x": float(fit_x),
                "y": float(fit_y),
                "label": label,
            }
        ]

    def _build_overlay_point(
        self,
        frame_id: int,
        track_id: int | None,
        frame_fits: dict,
        *,
        label: str,
        selected: bool = False,
    ) -> dict | None:
        if track_id is None:
            return None

        correction_key = (frame_id, track_id)
        fit_payload = {} if correction_key in self.pending_manual_recompute else (frame_fits.get(track_id) or {})
        fit_x = fit_payload.get("global_center_x")
        fit_y = fit_payload.get("global_center_y")
        fit_ok = bool(fit_payload.get("fit_ok", False))
        if fit_ok and self._is_finite_pair(fit_x, fit_y):
            return {
                "track_id": track_id,
                "x": float(fit_x),
                "y": float(fit_y),
                "label": f"{label}{'!' if fit_payload.get('is_suspicious') else ''}",
                "from_fit": True,
                "selected": selected,
            }

        track = self.track_map.get(track_id)
        if track is None:
            return None
        spot = track.spots_by_image_id.get(frame_id)
        if spot is None:
            return None

        x = self._safe_float(spot.center_x)
        y = self._safe_float(spot.center_y)
        if x is None or y is None:
            return None

        return {
            "track_id": track_id,
            "x": x,
            "y": y,
            "label": f"{label}*",
            "from_fit": False,
            "selected": selected,
        }

    @staticmethod
    def _has_overlay_point(payload: dict | None) -> bool:
        if payload is None:
            return False
        try:
            x = float(payload.get("x"))
            y = float(payload.get("y"))
        except Exception:
            return False
        return np.isfinite(x) and np.isfinite(y)

    def _build_export_roi_payloads(self, frame_id: int) -> list[dict]:
        if not self.positions_analysis_results:
            return []

        payloads: list[dict] = []
        for group in self.positions_analysis_results.get("groups", []):
            group_id = int(group.get("group_id", 0))
            color = self._group_color(group_id, emphasized=False)
            reference_track_id = group.get("reference_track_id")

            if reference_track_id is not None:
                roi_payload = self._build_export_roi_payload(
                    frame_id,
                    int(reference_track_id),
                    color,
                    f"G{group_id}-R:T{reference_track_id}",
                )
                if roi_payload is not None:
                    payloads.append(roi_payload)

            for satellite_index, track_id in enumerate(group.get("satellite_track_ids", []), start=1):
                roi_payload = self._build_export_roi_payload(
                    frame_id,
                    int(track_id),
                    color,
                    f"G{group_id}-S{satellite_index}:T{track_id}",
                )
                if roi_payload is not None:
                    payloads.append(roi_payload)

        return payloads

    def _build_export_roi_payload(
        self,
        frame_id: int,
        track_id: int,
        color: QColor,
        label: str,
    ) -> dict | None:
        track = self.track_map.get(track_id)
        if track is None:
            return None
        spot = track.spots_by_image_id.get(frame_id)
        if spot is None:
            return None
        return {
            "rect": QRect(spot.roi_rect),
            "color": QColor(color),
            "label": label,
        }

    def _build_export_point_rows(self, *, group_id_filter: int | None = None) -> list[dict]:
        if not self.positions_analysis_results:
            return []

        export_rows: list[dict] = []
        seen_reference_keys: set[tuple[int, int, int]] = set()
        fit_points_by_frame = self.positions_analysis_results.get("fit_points_by_frame", {})
        groups = self.positions_analysis_results.get("groups", [])

        for frame_id in self.sorted_frame_ids:
            frame_fits = fit_points_by_frame.get(frame_id, {})
            for group in groups:
                group_id = int(group.get("group_id", 0))
                if group_id_filter is not None and group_id != int(group_id_filter):
                    continue
                reference_track_id = group.get("reference_track_id")
                if reference_track_id is None:
                    continue

                reference_point = self._build_overlay_point(
                    frame_id,
                    int(reference_track_id),
                    frame_fits,
                    label="",
                    selected=False,
                )
                if self._has_overlay_point(reference_point):
                    reference_key = (frame_id, group_id, int(reference_track_id))
                    if reference_key not in seen_reference_keys:
                        fit_payload = frame_fits.get(int(reference_track_id), {}) or {}
                        export_rows.append(
                            {
                                "frame_id": int(frame_id),
                                "group_id": group_id,
                                "point_role": "R",
                                "track_id": int(reference_track_id),
                                "distance_from_reference": 0.0,
                                "x": float(reference_point["x"]),
                                "y": float(reference_point["y"]),
                                **self._build_fit_export_fields(
                                    fit_payload,
                                    from_fit=bool(reference_point.get("from_fit")),
                                ),
                            }
                        )
                        seen_reference_keys.add(reference_key)

                for satellite_index, track_id in enumerate(
                    group.get("satellite_track_ids", []),
                    start=1,
                ):
                    satellite_point = self._build_overlay_point(
                        frame_id,
                        int(track_id),
                        frame_fits,
                        label="",
                        selected=False,
                    )
                    if not self._has_overlay_point(satellite_point):
                        continue

                    distance = np.nan
                    if self._has_overlay_point(reference_point):
                        distance = float(
                            np.hypot(
                                float(satellite_point["x"]) - float(reference_point["x"]),
                                float(satellite_point["y"]) - float(reference_point["y"]),
                            )
                        )

                    fit_payload = frame_fits.get(int(track_id), {}) or {}
                    export_rows.append(
                        {
                            "frame_id": int(frame_id),
                            "group_id": group_id,
                            "point_role": f"S{satellite_index}",
                            "track_id": int(track_id),
                            "distance_from_reference": distance,
                            "x": float(satellite_point["x"]),
                            "y": float(satellite_point["y"]),
                            **self._build_fit_export_fields(
                                fit_payload,
                                from_fit=bool(satellite_point.get("from_fit")),
                            ),
                        }
                    )

        return export_rows

    @classmethod
    def _build_fit_export_fields(cls, fit_payload: dict | None, *, from_fit: bool) -> dict:
        if not isinstance(fit_payload, dict):
            fit_payload = {}

        fit_ok = bool(fit_payload.get("fit_ok", False)) and from_fit
        raw_fit_results = fit_payload.get("fit_results")
        fit_results = raw_fit_results if isinstance(raw_fit_results, dict) else {}
        export_fit_results = fit_results if fit_ok else {}
        r_squared = cls._safe_float(
            fit_payload.get("r_squared", fit_results.get("r_squared"))
        )

        return {
            "position_source": "gaussian_fit" if from_fit else "tracked_center",
            "fit_ok": fit_ok,
            "quality_status": fit_payload.get("quality_status", "failed"),
            "quality_flags": fit_payload.get("quality_flags") or [],
            "sigma_x": export_fit_results.get("sigma_x"),
            "sigma_y": export_fit_results.get("sigma_y"),
            "fwhm_x": export_fit_results.get("fwhm_x"),
            "fwhm_y": export_fit_results.get("fwhm_y"),
            "fwhm_x_error": export_fit_results.get("fwhm_x_error"),
            "fwhm_y_error": export_fit_results.get("fwhm_y_error"),
            "theta": export_fit_results.get("theta"),
            "r_squared": r_squared,
        }

    def _draw_export_roi(self, draw: ImageDraw.ImageDraw, payload: dict, font):
        rect = payload.get("rect")
        if not isinstance(rect, QRect):
            return

        color = self._qcolor_to_rgb(payload.get("color"))
        left = int(rect.x())
        top = int(rect.y())
        right = int(rect.x() + rect.width())
        bottom = int(rect.y() + rect.height())
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        label_text = str(payload.get("label", "")).strip()
        if label_text:
            text_position = (left + 3, max(0, top - 16))
            draw.text(text_position, label_text, fill=color, font=font)

    def _draw_export_overlay_groups(
        self,
        draw: ImageDraw.ImageDraw,
        overlay_groups: list[dict],
        font,
        image_size: tuple[int, int],
    ):
        label_rank = 0
        for group in overlay_groups:
            color = self._qcolor_to_rgb(group.get("color"))
            reference = group.get("reference")
            satellites = group.get("satellites", [])

            if reference is None:
                continue

            self._draw_export_marker(
                draw,
                reference,
                color,
                font,
                image_size=image_size,
                is_reference=True,
                label_rank=label_rank,
            )
            label_rank += 1
            for satellite in satellites:
                self._draw_export_marker(
                    draw,
                    satellite,
                    color,
                    font,
                    image_size=image_size,
                    is_reference=False,
                    label_rank=label_rank,
                )
                label_rank += 1

    def _draw_export_connection(
        self,
        draw: ImageDraw.ImageDraw,
        reference: dict,
        satellite: dict,
        color: tuple[int, int, int],
    ):
        draw.line(
            [
                (float(reference["x"]), float(reference["y"])),
                (float(satellite["x"]), float(satellite["y"])),
            ],
            fill=color,
            width=2,
        )

    def _draw_export_marker(
        self,
        draw: ImageDraw.ImageDraw,
        payload: dict | None,
        color: tuple[int, int, int],
        font,
        *,
        image_size: tuple[int, int],
        is_reference: bool,
        label_rank: int = 0,
    ):
        if not self._has_overlay_point(payload):
            return

        x = float(payload["x"])
        y = float(payload["y"])
        radius = 3 if is_reference else 2
        fill_color = color
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(255, 255, 255),
            fill=fill_color,
            width=1,
        )

        label_text = str(payload.get("label", "")).strip()
        if label_text:
            self._draw_export_label_with_arrow(
                draw,
                x,
                y,
                label_text,
                color,
                font,
                image_size=image_size,
                label_rank=label_rank,
            )

    def _draw_export_label_with_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        point_x: float,
        point_y: float,
        label_text: str,
        color: tuple[int, int, int],
        font,
        *,
        image_size: tuple[int, int],
        label_rank: int,
    ):
        image_width, image_height = image_size
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        dx = float(point_x - center_x)
        dy = float(point_y - center_y)
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            direction = np.array([1.0, -1.0], dtype=np.float64)
        else:
            direction = np.array([dx / norm, dy / norm], dtype=np.float64)
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)
        side = -1.0 if (label_rank % 2) else 1.0
        base_distance = float(self.export_label_distance_spin.value()) if hasattr(self, "export_label_distance_spin") else 46.0
        base_spread = float(self.export_label_spread_spin.value()) if hasattr(self, "export_label_spread_spin") else 18.0
        distance = base_distance + (0.9 * base_spread) * (label_rank % 4)
        side_shift = (base_spread + 0.6 * base_spread * ((label_rank // 2) % 3)) * side

        anchor_x = float(point_x + direction[0] * distance + perpendicular[0] * side_shift)
        anchor_y = float(point_y + direction[1] * distance + perpendicular[1] * side_shift)

        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = float(text_bbox[2] - text_bbox[0])
        text_h = float(text_bbox[3] - text_bbox[1])
        text_x = min(max(4.0, anchor_x), max(4.0, image_width - text_w - 4.0))
        text_y = min(max(4.0, anchor_y - text_h / 2.0), max(4.0, image_height - text_h - 4.0))

        leader_end_x = text_x - 6.0 if text_x >= point_x else text_x + text_w + 6.0
        leader_end_y = text_y + text_h / 2.0
        self._draw_export_arrow(
            draw,
            float(leader_end_x),
            float(leader_end_y),
            float(point_x),
            float(point_y),
            color,
        )

        padding = 3
        draw.rounded_rectangle(
            [
                text_x - padding,
                text_y - padding,
                text_x + text_w + padding,
                text_y + text_h + padding,
            ],
            radius=4,
            fill=(255, 255, 255),
            outline=color,
            width=1,
        )
        draw.text((text_x, text_y), label_text, fill=color, font=font)

    @staticmethod
    def _draw_export_arrow(
        draw: ImageDraw.ImageDraw,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        color: tuple[int, int, int],
    ):
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=1)

        dx = float(end_x - start_x)
        dy = float(end_y - start_y)
        norm = float(np.hypot(dx, dy))
        if norm < 1e-6:
            return

        ux = dx / norm
        uy = dy / norm
        arrow_len = 8.0
        wing_len = 3.5
        base_x = end_x - ux * arrow_len
        base_y = end_y - uy * arrow_len
        perp_x = -uy
        perp_y = ux
        left = (base_x + perp_x * wing_len, base_y + perp_y * wing_len)
        right = (base_x - perp_x * wing_len, base_y - perp_y * wing_len)
        draw.polygon([(end_x, end_y), left, right], fill=color)

    @staticmethod
    def _convert_numpy_to_export_image(image_data) -> Image.Image:
        arr = np.asarray(image_data)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr_float = arr.astype(np.float32, copy=False)
            min_val = float(np.nanmin(arr_float))
            max_val = float(np.nanmax(arr_float))
            if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                arr = (
                    255.0 * (arr_float - min_val) / (max_val - min_val)
                ).clip(0, 255).astype(np.uint8)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint8)

        if arr.ndim == 2:
            arr = np.stack((arr,) * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] not in (3, 4):
            arr = np.stack((arr[:, :, 0],) * 3, axis=-1)

        return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")

    @staticmethod
    def _qcolor_to_rgb(color: QColor | None) -> tuple[int, int, int]:
        if not isinstance(color, QColor):
            color = QColor(0, 170, 255)
        return (int(color.red()), int(color.green()), int(color.blue()))

    @staticmethod
    def _csv_safe_value(value):
        if isinstance(value, (list, tuple)):
            return "; ".join(str(item) for item in value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            return numeric
        return value

    @staticmethod
    def _combine_quality_statuses(reference_status: str | None, satellite_status: str | None) -> str:
        statuses = {reference_status or "failed", satellite_status or "failed"}
        if "failed" in statuses:
            return "failed"
        if "warning" in statuses:
            return "warning"
        return "ok"

    @staticmethod
    def _format_quality_status_text(status: str | None) -> str:
        normalized = str(status or "failed").strip().lower()
        if normalized == "ok":
            return "OK"
        if normalized == "warning":
            return "Warn"
        return "Fail"

    def _format_row_quality_flags(self, row: dict) -> str:
        parts: list[str] = []
        ref_text = format_quality_flags(row.get("reference_quality_flags", []))
        sat_text = format_quality_flags(row.get("satellite_quality_flags", []))
        if ref_text:
            parts.append(f"Ref: {ref_text}")
        if sat_text:
            parts.append(f"Sat: {sat_text}")
        return " | ".join(parts)

    def _style_results_table_row(self, row_idx: int, row: dict):
        if row.get("row_quality_status") == "failed":
            background = QColor(255, 232, 232)
        elif row.get("suspicious"):
            background = QColor(255, 245, 224)
        else:
            background = QColor(240, 255, 240)

        for col_idx in range(self.results_table.columnCount()):
            item = self.results_table.item(row_idx, col_idx)
            if item is None:
                continue
            item.setBackground(QBrush(background))

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    @classmethod
    def _is_finite_pair(cls, x, y) -> bool:
        return cls._safe_float(x) is not None and cls._safe_float(y) is not None

    @staticmethod
    def _group_color(group_id: int, *, emphasized: bool = False) -> QColor:
        palette = [
            QColor(0, 170, 255),
            QColor(255, 140, 0),
            QColor(50, 205, 50),
            QColor(255, 64, 129),
            QColor(156, 39, 176),
            QColor(255, 193, 7),
            QColor(121, 85, 72),
            QColor(0, 188, 212),
        ]
        color = QColor(palette[(max(1, int(group_id)) - 1) % len(palette)])
        color.setAlpha(255 if emphasized else 205)
        return color

    @staticmethod
    def _convert_numpy_to_pixmap(image_data) -> QPixmap:
        if image_data is None or image_data.size == 0:
            return QPixmap()

        arr = np.asarray(image_data)

        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr_float = arr.astype(np.float32)
            min_val = float(np.nanmin(arr_float))
            max_val = float(np.nanmax(arr_float))
            if max_val > min_val:
                arr = (
                    255.0 * (arr_float - min_val) / (max_val - min_val)
                ).clip(0, 255).astype(np.uint8)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint8)

        if arr.ndim == 2:
            height, width = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(
                arr.data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(q_image)

        if arr.ndim == 3 and arr.shape[2] == 3:
            height, width, _ = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(
                arr.data,
                width,
                height,
                3 * width,
                QImage.Format.Format_RGB888,
            ).copy()
            return QPixmap.fromImage(q_image)

        if arr.ndim == 3 and arr.shape[2] == 4:
            height, width, _ = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(
                arr.data,
                width,
                height,
                4 * width,
                QImage.Format.Format_RGBA8888,
            ).copy()
            return QPixmap.fromImage(q_image)

        if arr.ndim == 3:
            arr = np.mean(arr, axis=2).astype(np.uint8)
            height, width = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(
                arr.data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(q_image)

        return QPixmap()
