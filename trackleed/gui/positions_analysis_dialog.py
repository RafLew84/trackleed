# stal/gui/positions_analysis_dialog.py

from typing import Dict, List

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
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


class PositionsAnalysisDialog(QDialog):
    """
    Stage 1 skeleton dialog for the upcoming positions analysis workflow.

    Current scope:
    - show list of tracks,
    - show simple list of groups,
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
        self.all_files_map = all_files_map
        self.sorted_frame_ids = sorted(self.all_files_map.keys())
        self.groups: list[str] = []
        self._group_counter = 0

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
            "Stage 1: tracks list, groups list, and frame preview."
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

        group_box = QGroupBox("Frame Preview")
        group_layout = QVBoxLayout(group_box)

        self.image_viewer = LeedImageViewer()
        self.image_viewer.set_mode("PAN")

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
        layout.addWidget(group_box)

        return panel

    def _build_groups_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        group_box = QGroupBox("Groups")
        group_layout = QVBoxLayout(group_box)

        self.groups_list = QListWidget()
        self.groups_info_label = QLabel(
            "Stage 1 placeholder.\n\nGroups can be added and removed here.\n"
            "Reference/satellite assignment will be added in the next step."
        )
        self.groups_info_label.setWordWrap(True)

        buttons_layout = QHBoxLayout()
        self.add_group_button = QPushButton("Add Group")
        self.add_group_button.clicked.connect(self._add_group)
        self.remove_group_button = QPushButton("Remove Group")
        self.remove_group_button.clicked.connect(self._remove_selected_group)
        buttons_layout.addWidget(self.add_group_button)
        buttons_layout.addWidget(self.remove_group_button)

        group_layout.addWidget(self.groups_list)
        group_layout.addLayout(buttons_layout)
        group_layout.addWidget(self.groups_info_label)
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
            "Selecting a track highlights its current ROI and trajectory."
        )

    def _populate_groups_list(self):
        self.groups_list.clear()
        for group_name in self.groups:
            self.groups_list.addItem(group_name)

    def _add_group(self):
        self._group_counter += 1
        self.groups.append(f"Group {self._group_counter}")
        self._populate_groups_list()
        self.groups_list.setCurrentRow(self.groups_list.count() - 1)

    def _remove_selected_group(self):
        row = self.groups_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a group to remove.")
            return
        del self.groups[row]
        self._populate_groups_list()

    def _get_selected_track(self) -> Track | None:
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            return None
        item = self.tracks_table.item(current_row, 0)
        if item is None:
            return None
        track_id = int(item.text())
        return next((t for t in self.tracks if t.track_id == track_id), None)

    def _on_track_selection_changed(self):
        self._update_track_overlay()

    def _on_slider_value_changed(self, index: int, fit: bool = False):
        if not self.sorted_frame_ids or not (0 <= index < len(self.sorted_frame_ids)):
            return
        frame_id = self.sorted_frame_ids[index]
        self._show_frame(frame_id, fit=fit)
        self.frame_label.setText(
            f"Frame {index + 1}/{len(self.sorted_frame_ids)} (ID: {frame_id})"
        )

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

    def _update_track_overlay(self):
        selected_track = self._get_selected_track()
        if selected_track is None:
            self.image_viewer.highlight_roi(None)
            self.image_viewer.show_track_path(None)
            return

        self.image_viewer.show_track_path(selected_track)

        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            self.image_viewer.highlight_roi(None)
            return

        frame_id = self.sorted_frame_ids[slider_index]
        spot = selected_track.spots_by_image_id.get(frame_id)
        self.image_viewer.highlight_roi(spot.id if spot is not None else None)

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
