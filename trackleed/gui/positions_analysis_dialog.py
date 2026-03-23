# stal/gui/positions_analysis_dialog.py

from dataclasses import dataclass, field
from threading import Event
from typing import Dict, List

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QThread
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QComboBox,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.io.image_reader import load_tiff
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.positions_analysis_coordinator import (
    fit_track_position_on_frame,
    run_positions_analysis,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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
        self.manual_bbox_corrections: dict[tuple[int, int], dict] = {}
        self.pending_manual_recompute: set[tuple[int, int]] = set()
        self.worker_thread = None
        self.worker = None
        self.positions_analysis_cancel_event: Event | None = None

        self._build_ui()
        self._populate_tracks_table()
        self._populate_groups_list()
        self._initialize_view()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        title_label = QLabel("Positions Analysis")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        main_layout.addWidget(title_label)

        subtitle_label = QLabel(
            "Stage 5: tracks list, group definition, positions analysis, results, and overlay preview."
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

        group_layout.addWidget(self.image_viewer, 1)
        group_layout.addLayout(slider_layout)
        group_layout.addWidget(self.frame_meta_label)
        vertical_splitter.addWidget(group_box)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        controls_layout = QHBoxLayout()
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItem("Coverage", "coverage")
        self.x_axis_combo.addItem("Temperature", "temperature")
        self.x_axis_combo.currentIndexChanged.connect(self._refresh_results_views)
        controls_layout.addWidget(QLabel("X axis:"))
        controls_layout.addWidget(self.x_axis_combo)
        controls_layout.addStretch(1)

        self.results_status_label = QLabel(
            "Run positions analysis to populate the plot and results table."
        )
        self.results_status_label.setWordWrap(True)

        self.results_plot = MplCanvas(self, width=5, height=3, dpi=100)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Frame ID",
                "Temp",
                "Coverage",
                "dx",
                "dy",
                "r",
                "Ref OK",
                "Sat OK",
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
        vertical_splitter.addWidget(results_group)
        vertical_splitter.setSizes([520, 300])

        layout.addWidget(vertical_splitter)

        return panel

    def _build_groups_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

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
        analysis_buttons_layout.addWidget(self.run_analysis_button)

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
            "Enable ROI dragging in the preview. Changes affect only the current frame."
        )

        self.recompute_fit_button = QPushButton("Recompute 2D Gauss")
        self.recompute_fit_button.clicked.connect(self._recompute_selected_fit)
        self.recompute_fit_button.setToolTip(
            "Recompute fit only for the selected track on the current frame."
        )

        self.manual_correction_label = QLabel(
            "Select a track and frame. Move its bbox if needed, then recompute the local fit."
        )
        self.manual_correction_label.setWordWrap(True)

        correction_layout.addWidget(self.move_bbox_button)
        correction_layout.addWidget(self.recompute_fit_button)
        correction_layout.addWidget(self.manual_correction_label)

        group_layout.addWidget(self.groups_list)
        group_layout.addLayout(buttons_layout)
        group_layout.addWidget(self.groups_info_label)
        group_layout.addLayout(analysis_buttons_layout)
        group_layout.addWidget(self.analysis_summary_label)
        group_layout.addWidget(details_group)
        group_layout.addWidget(correction_group)
        layout.addWidget(group_box)

        status_group = QGroupBox("Session")
        status_layout = QFormLayout(status_group)
        status_layout.addRow("Loaded tracks:", QLabel(str(len(self.tracks))))
        status_layout.addRow("Loaded frames:", QLabel(str(len(self.sorted_frame_ids))))
        layout.addWidget(status_group)
        layout.addStretch(1)

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
        self._update_manual_correction_label()

    def _toggle_move_bbox_mode(self, enabled: bool):
        self.image_viewer.set_mode("MOVE_ROI" if enabled else "PAN")
        self.move_bbox_button.setText("Move BBox Mode: ON" if enabled else "Move BBox Mode")
        self._update_manual_correction_label()

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
        self._refresh_selected_track_label()
        self._update_track_overlay()

    def _on_slider_value_changed(self, index: int, fit: bool = False):
        if not self.sorted_frame_ids or not (0 <= index < len(self.sorted_frame_ids)):
            return
        frame_id = self.sorted_frame_ids[index]
        self._show_frame(frame_id, fit=fit)
        self.frame_label.setText(
            f"Frame {index + 1}/{len(self.sorted_frame_ids)} (ID: {frame_id})"
        )
        self._update_manual_correction_label()

    def _show_frame(self, frame_id: int, fit: bool = False):
        leed_file = self.all_files_map.get(frame_id)
        if leed_file is None:
            return

        image_data = load_tiff(leed_file.filepath)
        pixmap = self._convert_numpy_to_pixmap(image_data)
        self.image_viewer.set_image(pixmap, fit=fit)

        spots_in_frame: List[SpotROI] = []
        for track in self.tracks:
            spot = track.spots_by_image_id.get(frame_id)
            if spot is not None:
                spots_in_frame.append(spot)

        self.image_viewer.update_rois(spots_in_frame)
        self.frame_meta_label.setText(
            f"Temperature: {leed_file.temperature} | "
            f"Coverage: {leed_file.coverage if leed_file.coverage is not None else '-'} | "
            f"Energy: {leed_file.energy}"
        )
        self._update_track_overlay()

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
            self._refresh_results_views()
            QMessageBox.information(
                self,
                "Analysis Cancelled",
                "Positions analysis was cancelled.",
            )
            return

        self.positions_analysis_results = results
        self.pending_manual_recompute.clear()
        self.analysis_summary_label.setText(self._format_analysis_summary(results))
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
        self._refresh_results_views()
        self._update_track_overlay()
        self._update_manual_correction_label()
        QMessageBox.critical(
            self,
            "Positions Analysis Error",
            f"An error occurred:\n{error_message}",
        )

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
            f"Failed fits: {summary.get('failed_fits', 0)}"
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

        spot.roi_rect = new_rect
        spot.center_x = float(new_rect.center().x())
        spot.center_y = float(new_rect.center().y())

        correction_key = (frame_id, track.track_id)
        self.manual_bbox_corrections[correction_key] = {
            "x": int(new_rect.x()),
            "y": int(new_rect.y()),
            "width": int(new_rect.width()),
            "height": int(new_rect.height()),
        }
        self.pending_manual_recompute.add(correction_key)

        self._select_track_by_id(track.track_id)
        self._update_manual_correction_label()
        self._update_track_overlay()

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

        fit_payload = fit_track_position_on_frame(track, frame_id, image_data)
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

        summary = self.positions_analysis_results.setdefault("summary", {})
        summary["groups_configured"] = len(self.positions_analysis_results.get("groups", []))
        summary["tracks_in_analysis"] = len(tracks_in_analysis)
        summary["frames_processed"] = len(fit_points_by_frame)
        summary["rows_generated"] = len(self.positions_analysis_results.get("rows", []))
        summary["successful_fits"] = successful_fits
        summary["failed_fits"] = failed_fits

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
                "Select a track and frame. Move its bbox if needed, then recompute the local fit."
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
                    if self._safe_float(r_squared) is not None:
                        parts.append(f"Current fit OK (R²={float(r_squared):.4f}).")
                    else:
                        parts.append("Current fit OK.")
                else:
                    parts.append("Current fit failed or is missing.")
            else:
                parts.append("No fit stored yet for this track/frame.")
        else:
            parts.append("Run positions analysis first to populate fit results.")

        if self.move_bbox_button.isChecked():
            parts.append("Move BBox Mode is ON.")

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
        self._update_positions_overlay()
        self._update_manual_correction_label()

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
        self.results_status_label.setText(
            f"Showing {len(rows)} row(s) for selected group/satellite. X axis: {x_axis_name}."
        )

        for row_idx, row in enumerate(rows):
            values = [
                row.get("frame_id"),
                row.get("temperature"),
                row.get("coverage"),
                row.get("dx"),
                row.get("dy"),
                row.get("r"),
                "Yes" if row.get("reference_fit_ok") else "No",
                "Yes" if row.get("satellite_fit_ok") else "No",
            ]
            for col_idx, value in enumerate(values):
                if isinstance(value, (float, np.floating)):
                    text = "NaN" if np.isnan(value) else f"{value:.4f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row_idx, col_idx, item)

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

        group = self._get_selected_group()
        sat_track_id = self._get_selected_satellite_track_id()
        ref_track_id = group.reference_track_id if group else None

        axes.plot(x_values, y_values, marker="o", linewidth=1.5)
        axes.set_xlabel("Coverage" if x_key == "coverage" else "Temperature")
        axes.set_ylabel("Distance r")
        axes.set_title(
            f"Group {group.group_id if group else '-'} | "
            f"Ref T{ref_track_id if ref_track_id is not None else '-'} -> "
            f"Sat T{sat_track_id if sat_track_id is not None else '-'}"
        )
        axes.grid(True, alpha=0.3)
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

    def _update_positions_overlay(self):
        frame_id = self._get_current_frame_id()
        if frame_id is None:
            self.image_viewer.update_positions_overlay([])
            return

        overlay_groups = self._build_positions_overlay_payload(frame_id)
        self.image_viewer.update_positions_overlay(overlay_groups)

    def _get_current_frame_id(self) -> int | None:
        if not self.sorted_frame_ids:
            return None
        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            return None
        return self.sorted_frame_ids[slider_index]

    def _build_positions_overlay_payload(self, frame_id: int) -> list[dict]:
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

        overlay_groups: list[dict] = []
        for group in results_groups:
            group_id = int(group.get("group_id"))
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
                "label": label,
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
