from __future__ import annotations

# stal/gui/tracking_dialog.py

from pathlib import Path
from typing import List
from threading import Event
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QThread
from PyQt6.QtGui import QImage, QPixmap, QActionGroup, QAction
from PyQt6.QtWidgets import (QDialog, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QSplitter,
                             QLabel, QTableWidget, QAbstractItemView, QTableWidgetItem, QDialogButtonBox,
                             QHeaderView, QPushButton, QMessageBox, QToolBar, QMenu, QMessageBox, QProgressDialog,
                             QComboBox)

from trackleed.core.data_models import LeedFile, Track, SpotROI
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.tapir_backend import (
    format_tapir_environment_report,
    probe_tapir_environment,
)
from trackleed.analysis.locotrack_backend import (
    format_locotrack_environment_report,
    probe_locotrack_environment,
)
from trackleed.analysis.cotracker_backend import (
    format_cotracker_environment_report,
    probe_cotracker_environment,
)
from trackleed.analysis.tapnext_backend import (
    format_tapnext_environment_report,
    probe_tapnext_environment,
)
from trackleed.analysis.trackonr_backend import (
    format_trackonr_environment_report,
    probe_trackonr_environment,
)
from trackleed.analysis.pipspp_backend import (
    format_pipspp_environment_report,
    probe_pipspp_environment,
)
from trackleed.analysis.mftiq_backend import (
    format_mftiq_environment_report,
    probe_mftiq_environment,
)
from trackleed.analysis.cutie_backend import (
    format_cutie_environment_report,
    probe_cutie_environment,
)
from trackleed.analysis.sam2_backend import (
    format_sam2_environment_report,
    probe_sam2_environment,
)
from trackleed.analysis.tracking import (
    retrack_single_spot_cotracker3_offline,
    retrack_single_spot_cutie_offline,
    retrack_single_spot_mftiq_offline,
    retrack_single_spot_optical_flow,
    retrack_single_spot_locotrack_offline,
    retrack_single_spot_pipspp_offline,
    retrack_single_spot_sam2_offline,
    retrack_single_spot_trackonr_offline,
    retrack_single_spot_tapnext_offline,
    retrack_single_spot_tapir_offline,
    retrack_single_spot_tapir_online,
)
from trackleed.logic.generic_worker import GenericWorker


class TrackingDialog(QDialog):
    """
    A dedicated dialog for visualizing, analyzing, and editing tracking results.
    """
    def __init__(
        self,
        tracking_data: List[Track],
        all_files: List[LeedFile],
        masks: List,
        roi_size: int,
        lk_win_size: int,
        lk_max_level: int,
        tapir_params: dict | None = None,
        locotrack_params: dict | None = None,
        cotracker_params: dict | None = None,
        tapnext_params: dict | None = None,
        trackonr_params: dict | None = None,
        pipspp_params: dict | None = None,
        mftiq_params: dict | None = None,
        cutie_params: dict | None = None,
        sam2_params: dict | None = None,
        parent=None,
        save_session_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Tracking Results Visualizer")
        self.setMinimumSize(1200, 800)

        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.worker_thread = None 
        self.worker = None 

        self.tracks = tracking_data
        self.file_map = {f.id: f for f in all_files if f.id is not None}
        self.masks = masks
        self.roi_size = roi_size
        self.lk_win_size = lk_win_size  
        self.lk_max_level = lk_max_level
        self.tapir_params = dict(tapir_params or {})
        self.locotrack_params = dict(locotrack_params or {})
        self.cotracker_params = dict(cotracker_params or {})
        self.tapnext_params = dict(tapnext_params or {})
        self.trackonr_params = dict(trackonr_params or {})
        self.pipspp_params = dict(pipspp_params or {})
        self.mftiq_params = dict(mftiq_params or {})
        self.cutie_params = dict(cutie_params or {})
        self.sam2_params = dict(sam2_params or {})
        self.save_session_callback = save_session_callback
        self._active_retrack_label = "Re-tracking"

        self.sorted_frame_ids = sorted(list(self.file_map.keys()))
        
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_viewer = LeedImageViewer()
        self.image_viewer.track_point_context_menu_requested.connect(self.on_track_point_context_menu)
        self.image_viewer.point_for_track_added_at.connect(self.handle_add_or_correct_point)
        
        self.controls_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("Frame: -/-")
        self.controls_layout.addWidget(self.frame_slider)
        self.controls_layout.addWidget(self.frame_label)
        
        left_layout.addWidget(self.image_viewer)
        left_layout.addLayout(self.controls_layout)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.tracks_table = QTableWidget()
        self.plot_widget = QLabel("Select a track to view details.")
        self.plot_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_widget.setWordWrap(True)
        self.plot_widget.setTextFormat(Qt.TextFormat.RichText)
        self.plot_widget.setMinimumHeight(160)
        self.roi_transparency_combo = QComboBox()
        self.show_selected_track_only_button = QPushButton("Hide Other BBoxes")
        self.show_selected_track_only_button.setCheckable(True)

        self.add_track_button = QPushButton("Add New Track")
        self.add_track_button.clicked.connect(self.enter_add_new_track_mode)

        self.delete_track_button = QPushButton("Delete Selected Track")
        self.delete_track_button.clicked.connect(self.delete_selected_track)
        self.merge_tracks_button = QPushButton("Merge Selected Tracks")
        self.merge_tracks_button.setToolTip("Select two tracks to merge them into one timeline.")
        self.merge_tracks_button.clicked.connect(self.merge_selected_tracks)
        self._init_visual_controls()
        view_controls_layout = QHBoxLayout()
        view_controls_layout.addWidget(QLabel("BBox transparency:"))
        view_controls_layout.addWidget(self.roi_transparency_combo)
        view_controls_layout.addStretch(1)
        right_layout.addLayout(view_controls_layout)
        right_layout.addWidget(QLabel("Found Tracks:"))
        right_layout.addWidget(self.tracks_table)
        right_layout.addWidget(self.show_selected_track_only_button)
        right_layout.addWidget(self.delete_track_button)
        right_layout.insertWidget(3, self.merge_tracks_button)
        right_layout.insertWidget(3, self.add_track_button)
        right_layout.addWidget(QLabel("Track Analysis:"))
        right_layout.addWidget(self.plot_widget)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([800, 400])

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.advance_frame)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_session_button = self.button_box.addButton(
            "Save Session", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.save_session_button.clicked.connect(self._on_save_session)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self._init_ui_elements()
        self.populate_tracks_table()
        
        if self.sorted_frame_ids:
            # Show the first frame by its index (0)
            self.on_slider_value_changed(0)
            self.show_frame(self.sorted_frame_ids[0], fit=True)
            self.frame_slider.setValue(0)
        
        toolbar = QToolBar("Interaction Modes")
        main_layout.addWidget(toolbar) # Add the toolbar as a regular widget to the layout

        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)

        pan_mode_action = QAction("Pan/View", self)
        pan_mode_action.setCheckable(True)
        pan_mode_action.setChecked(True)
        pan_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('PAN'))
        toolbar.addAction(pan_mode_action)
        mode_group.addAction(pan_mode_action)

        edit_mode_action = QAction("Edit Track", self)
        edit_mode_action.setCheckable(True)
        edit_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('EDIT_TRACK'))
        toolbar.addAction(edit_mode_action)
        mode_group.addAction(edit_mode_action)

        add_correct_mode_action = QAction("Add/Correct Point", self)
        add_correct_mode_action.setCheckable(True)
        add_correct_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_CORRECT_POINT'))
        toolbar.addAction(add_correct_mode_action)
        mode_group.addAction(add_correct_mode_action)

        add_new_track_mode_action = QAction("Add New Track", self)
        add_new_track_mode_action.setCheckable(True)
        add_new_track_mode_action.triggered.connect(self.enter_add_new_track_mode)
        toolbar.addAction(add_new_track_mode_action)
        mode_group.addAction(add_new_track_mode_action)

        self.image_viewer.new_track_started_at.connect(self.handle_add_new_track)

    def _on_save_session(self):
        """Trigger saving the current session from inside the dialog."""
        if self.save_session_callback:
            self.save_session_callback(self.tracks)
        else:
            QMessageBox.information(self, "Save Session", "Save session is not available in this context.")


    def _init_ui_elements(self):
        """Initializes the UI widgets like table and slider."""
        if self.sorted_frame_ids:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(len(self.sorted_frame_ids) - 1)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)

        self.tracks_table.setColumnCount(6)
        self.tracks_table.setHorizontalHeaderLabels(["Track ID", "Method", "Start Frame", "End Frame", "Length", "Gaps"])
        self.tracks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tracks_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tracks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tracks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tracks_table.itemSelectionChanged.connect(self.on_track_selected)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_animation)
        self.controls_layout.insertWidget(0, self.play_button) 

    def _init_visual_controls(self):
        """Initialize controls related to ROI visibility in the tracking viewer."""
        for percent in range(0, 101, 10):
            self.roi_transparency_combo.addItem(f"{percent}%", percent)
        self.roi_transparency_combo.setCurrentIndex(0)
        self.roi_transparency_combo.currentIndexChanged.connect(self.on_roi_transparency_changed)
        self.roi_transparency_combo.setToolTip(
            "Adjust ROI/bbox transparency in the tracking viewer."
        )

        self.show_selected_track_only_button.clicked.connect(self.toggle_selected_track_only)
        self.show_selected_track_only_button.setShortcut("H")
        self.show_selected_track_only_button.setToolTip(
            "When enabled, show only the ROI belonging to the currently selected track. Shortcut: H"
        )
        self._update_track_filter_button_text()

    def enter_add_new_track_mode(self):
        """Switches the viewer to the mode for adding a new track."""
        self.image_viewer.set_mode('ADD_NEW_TRACK')

    def on_roi_transparency_changed(self, index: int):
        """Apply ROI transparency selected in the dropdown."""
        percent = self.roi_transparency_combo.itemData(index)
        if percent is None:
            percent = 0
        self.image_viewer.set_roi_transparency_percent(float(percent))

    def toggle_selected_track_only(self, checked: bool):
        """Toggle ROI visibility between all tracks and only the selected track."""
        self._update_track_filter_button_text()
        self.refresh_current_frame()

    def _update_track_filter_button_text(self):
        if self.show_selected_track_only_button.isChecked():
            self.show_selected_track_only_button.setText("Show All BBoxes")
        else:
            self.show_selected_track_only_button.setText("Hide Other BBoxes")

    def refresh_current_frame(self):
        """Re-render the currently displayed frame using the active view settings."""
        if not self.sorted_frame_ids:
            return
        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            return
        self.show_frame(self.sorted_frame_ids[slider_index])

    def get_selected_track(self) -> Track | None:
        """Return the track associated with the current table row."""
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            return None

        id_item = self.tracks_table.item(current_row, 0)
        if not id_item:
            return None

        track_id = int(id_item.text())
        return next((t for t in self.tracks if t.track_id == track_id), None)

    def _classify_track_method(self, track: Track) -> str:
        """Return a short human-readable label for the dominant source in a track."""
        sources = {
            str(getattr(spot, "source", "")).strip().lower()
            for spot in track.spots_by_image_id.values()
            if getattr(spot, "source", None)
        }
        if not sources:
            return "Unknown"

        manual_sources = {"manual", "manual_new", "manual_corr", "manual_point"}
        if sources == {"tapir_offline"}:
            return "TAPIR"
        if sources == {"tapir_online"}:
            return "BootsTAPIR Online"
        if sources == {"locotrack_offline"}:
            return "LocoTrack"
        if sources == {"cotracker3_offline"}:
            return "CoTracker3"
        if sources == {"tapnext_offline"}:
            return "TAPNext"
        if sources == {"trackonr_offline"}:
            return "Track-On-R"
        if sources == {"pipspp_offline"}:
            return "PIPs++"
        if sources == {"mftiq_offline"}:
            return "MFTIQ"
        if sources == {"cutie_offline"}:
            return "Cutie"
        if sources == {"sam2_offline"}:
            return "SAM2"
        if sources.issubset(manual_sources):
            return "Manual"
        if "tapir_offline" in sources and sources.issubset({"tapir_offline"} | manual_sources):
            return "TAPIR+Manual"
        if "tapir_online" in sources and sources.issubset({"tapir_online"} | manual_sources):
            return "BootsTAPIR Online+Manual"
        if "locotrack_offline" in sources and sources.issubset({"locotrack_offline"} | manual_sources):
            return "LocoTrack+Manual"
        if "cotracker3_offline" in sources and sources.issubset({"cotracker3_offline"} | manual_sources):
            return "CoTracker3+Manual"
        if "tapnext_offline" in sources and sources.issubset({"tapnext_offline"} | manual_sources):
            return "TAPNext+Manual"
        if "trackonr_offline" in sources and sources.issubset({"trackonr_offline"} | manual_sources):
            return "Track-On-R+Manual"
        if "pipspp_offline" in sources and sources.issubset({"pipspp_offline"} | manual_sources):
            return "PIPs++ + Manual"
        if "mftiq_offline" in sources and sources.issubset({"mftiq_offline"} | manual_sources):
            return "MFTIQ + Manual"
        if "cutie_offline" in sources and sources.issubset({"cutie_offline"} | manual_sources):
            return "Cutie + Manual"
        if "sam2_offline" in sources and sources.issubset({"sam2_offline"} | manual_sources):
            return "SAM2 + Manual"
        if any("optical" in src or "flow" in src for src in sources):
            return "OptFlow"
        if any("yolo" in src for src in sources):
            return "YOLO"
        if sources == {"auto"}:
            return "Auto"
        return "Mixed"

    def _count_track_gaps(self, track: Track) -> int:
        """Count missing frames inside the track span."""
        frame_ids = sorted(track.spots_by_image_id.keys())
        if len(frame_ids) < 2:
            return 0
        frame_index_map = {frame_id: idx for idx, frame_id in enumerate(self.sorted_frame_ids)}
        indexed = [frame_index_map[fid] for fid in frame_ids if fid in frame_index_map]
        if len(indexed) < 2:
            return 0
        start_idx = min(indexed)
        end_idx = max(indexed)
        return max(0, (end_idx - start_idx + 1) - len(indexed))

    def _update_track_details(self):
        """Refresh the right-side summary panel for the selected track and frame."""
        selected_track = self.get_selected_track()
        if selected_track is None:
            self.plot_widget.setText("Select a track to view details.")
            return

        slider_index = self.frame_slider.value()
        current_frame_id = None
        if 0 <= slider_index < len(self.sorted_frame_ids):
            current_frame_id = self.sorted_frame_ids[slider_index]

        frame_ids = sorted(selected_track.spots_by_image_id.keys())
        start_frame = frame_ids[0] if frame_ids else "-"
        end_frame = frame_ids[-1] if frame_ids else "-"
        gaps = self._count_track_gaps(selected_track)
        method = self._classify_track_method(selected_track)

        lines = [
            f"<b>Track {selected_track.track_id}</b>",
            f"Method: {method}",
            f"Span: {start_frame} -> {end_frame}",
            f"Frames: {len(frame_ids)} | Gaps: {gaps}",
        ]

        if current_frame_id is None:
            self.plot_widget.setText("<br>".join(lines))
            return

        spot = selected_track.spots_by_image_id.get(current_frame_id)
        if spot is None:
            lines.extend(
                [
                    "",
                    f"<b>Frame {current_frame_id}</b>",
                    "No spot on this frame.",
                ]
            )
            self.plot_widget.setText("<br>".join(lines))
            return

        rect = spot.roi_rect
        lines.extend(
            [
                "",
                f"<b>Frame {current_frame_id}</b>",
                f"Source: {getattr(spot, 'source', 'unknown')}",
                f"Center: ({float(spot.center_x):.2f}, {float(spot.center_y):.2f})",
                f"ROI: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}",
            ]
        )

        source = str(getattr(spot, "source", "")).lower()
        if source in {"tapir_offline", "tapir_online"}:
            raw_x = getattr(spot, "tapir_raw_x", None)
            raw_y = getattr(spot, "tapir_raw_y", None)
            refined_x = getattr(spot, "tapir_refined_x", None)
            refined_y = getattr(spot, "tapir_refined_y", None)
            visibility = getattr(spot, "tapir_visibility", None)
            rejected_reason = getattr(spot, "tapir_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                label = "Raw BootsTAPIR Online" if source == "tapir_online" else "Raw TAPIR"
                lines.append(f"{label}: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "locotrack_offline":
            raw_x = getattr(spot, "locotrack_raw_x", None)
            raw_y = getattr(spot, "locotrack_raw_y", None)
            refined_x = getattr(spot, "locotrack_refined_x", None)
            refined_y = getattr(spot, "locotrack_refined_y", None)
            visibility = getattr(spot, "locotrack_visibility", None)
            rejected_reason = getattr(spot, "locotrack_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Raw LocoTrack: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "cotracker3_offline":
            raw_x = getattr(spot, "cotracker3_raw_x", None)
            raw_y = getattr(spot, "cotracker3_raw_y", None)
            refined_x = getattr(spot, "cotracker3_refined_x", None)
            refined_y = getattr(spot, "cotracker3_refined_y", None)
            visibility = getattr(spot, "cotracker3_visibility", None)
            rejected_reason = getattr(spot, "cotracker3_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Raw CoTracker3: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "tapnext_offline":
            raw_x = getattr(spot, "tapnext_raw_x", None)
            raw_y = getattr(spot, "tapnext_raw_y", None)
            refined_x = getattr(spot, "tapnext_refined_x", None)
            refined_y = getattr(spot, "tapnext_refined_y", None)
            visibility = getattr(spot, "tapnext_visibility", None)
            visibility_score = getattr(spot, "tapnext_visibility_score", None)
            certainty_score = getattr(spot, "tapnext_certainty_score", None)
            combined_score = getattr(spot, "tapnext_combined_score", None)
            rejected_reason = getattr(spot, "tapnext_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Raw TAPNext: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if visibility_score is not None:
                lines.append(f"Visibility score: {float(visibility_score):.3f}")
            if certainty_score is not None:
                lines.append(f"Certainty score: {float(certainty_score):.3f}")
            if combined_score is not None:
                lines.append(f"Combined score: {float(combined_score):.3f}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "trackonr_offline":
            raw_x = getattr(spot, "trackonr_raw_x", None)
            raw_y = getattr(spot, "trackonr_raw_y", None)
            refined_x = getattr(spot, "trackonr_refined_x", None)
            refined_y = getattr(spot, "trackonr_refined_y", None)
            visibility = getattr(spot, "trackonr_visibility", None)
            rejected_reason = getattr(spot, "trackonr_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Raw Track-On-R: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "pipspp_offline":
            raw_x = getattr(spot, "pipspp_raw_x", None)
            raw_y = getattr(spot, "pipspp_raw_y", None)
            refined_x = getattr(spot, "pipspp_refined_x", None)
            refined_y = getattr(spot, "pipspp_refined_y", None)
            visibility = getattr(spot, "pipspp_visibility", None)
            rejected_reason = getattr(spot, "pipspp_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Raw PIPs++: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "mftiq_offline":
            raw_x = getattr(spot, "mftiq_raw_x", None)
            raw_y = getattr(spot, "mftiq_raw_y", None)
            refined_x = getattr(spot, "mftiq_refined_x", None)
            refined_y = getattr(spot, "mftiq_refined_y", None)
            visibility = getattr(spot, "mftiq_visibility", None)
            occlusion_score = getattr(spot, "mftiq_occlusion_score", None)
            sigma_score = getattr(spot, "mftiq_sigma_score", None)
            rejected_reason = getattr(spot, "mftiq_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Raw MFTIQ: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if occlusion_score is not None:
                lines.append(f"Occlusion score: {float(occlusion_score):.3f}")
            if sigma_score is not None:
                lines.append(f"Sigma score: {float(sigma_score):.3f}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "cutie_offline":
            raw_x = getattr(spot, "cutie_raw_x", None)
            raw_y = getattr(spot, "cutie_raw_y", None)
            refined_x = getattr(spot, "cutie_refined_x", None)
            refined_y = getattr(spot, "cutie_refined_y", None)
            visibility = getattr(spot, "cutie_visibility", None)
            mask_area = getattr(spot, "cutie_mask_area", None)
            mask_score = getattr(spot, "cutie_mask_score", None)
            mask_bbox_xyxy = getattr(spot, "cutie_mask_bbox_xyxy", None)
            mask_bbox_aspect_ratio = getattr(spot, "cutie_mask_bbox_aspect_ratio", None)
            repr_mode = getattr(spot, "cutie_repr_mode", None)
            initial_mask_mode = getattr(spot, "cutie_initial_mask_mode", None)
            rejected_reason = getattr(spot, "cutie_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Representative point: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Stored point: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if repr_mode:
                lines.append(f"Representative mode: {repr_mode}")
            if initial_mask_mode:
                lines.append(f"Initial mask mode: {initial_mask_mode}")
            if mask_area is not None:
                lines.append(f"Mask area: {float(mask_area):.1f}")
            if mask_score is not None:
                lines.append(f"Mask score: {float(mask_score):.3f}")
            if mask_bbox_xyxy is not None:
                try:
                    x0, y0, x1, y1 = [float(v) for v in mask_bbox_xyxy]
                    lines.append(
                        f"Mask bbox: x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}"
                    )
                except Exception:
                    pass
            if mask_bbox_aspect_ratio is not None:
                lines.append(f"BBox aspect ratio: {float(mask_bbox_aspect_ratio):.3f}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")
        elif source == "sam2_offline":
            raw_x = getattr(spot, "sam2_raw_x", None)
            raw_y = getattr(spot, "sam2_raw_y", None)
            refined_x = getattr(spot, "sam2_refined_x", None)
            refined_y = getattr(spot, "sam2_refined_y", None)
            visibility = getattr(spot, "sam2_visibility", None)
            mask_area = getattr(spot, "sam2_mask_area", None)
            mask_score = getattr(spot, "sam2_mask_score", None)
            mask_component_count = getattr(spot, "sam2_mask_component_count", None)
            mask_bbox_xyxy = getattr(spot, "sam2_mask_bbox_xyxy", None)
            mask_bbox_aspect_ratio = getattr(spot, "sam2_mask_bbox_aspect_ratio", None)
            repr_mode = getattr(spot, "sam2_repr_mode", None)
            prompt_mode = getattr(spot, "sam2_prompt_mode", None)
            rejected_reason = getattr(spot, "sam2_filter_rejected_reason", None)
            if raw_x is not None and raw_y is not None:
                lines.append(f"Representative point: ({float(raw_x):.2f}, {float(raw_y):.2f})")
            if refined_x is not None and refined_y is not None:
                lines.append(f"Stored point: ({float(refined_x):.2f}, {float(refined_y):.2f})")
            if visibility is not None:
                lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
            if repr_mode:
                lines.append(f"Representative mode: {repr_mode}")
            if prompt_mode:
                lines.append(f"Prompt mode: {prompt_mode}")
            if mask_area is not None:
                lines.append(f"Mask area: {float(mask_area):.1f}")
            if mask_score is not None:
                lines.append(f"Mask score: {float(mask_score):.3f}")
            if mask_component_count is not None:
                lines.append(f"Mask components: {int(mask_component_count)}")
            if mask_bbox_xyxy is not None:
                try:
                    x0, y0, x1, y1 = [float(v) for v in mask_bbox_xyxy]
                    lines.append(
                        f"Mask bbox: x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}"
                    )
                except Exception:
                    pass
            if mask_bbox_aspect_ratio is not None:
                lines.append(f"BBox aspect ratio: {float(mask_bbox_aspect_ratio):.3f}")
            if rejected_reason:
                lines.append(f"Filter: rejected ({rejected_reason})")

        self.plot_widget.setText("<br>".join(lines))

    def _get_spots_to_display_for_frame(self, frame_id: int) -> List[SpotROI]:
        """Build the ROI list for the current frame, honoring the selected-track filter."""
        selected_track = self.get_selected_track()
        if self.show_selected_track_only_button.isChecked() and selected_track is not None:
            spot = selected_track.spots_by_image_id.get(frame_id)
            return [spot] if spot is not None else []

        spots_in_frame = []
        for track in self.tracks:
            if frame_id in track.spots_by_image_id:
                spots_in_frame.append(track.spots_by_image_id[frame_id])
        return spots_in_frame

    def _build_point_tracker_focus_points_for_frame(self, frame_id: int) -> list[dict]:
        """
        Build overlay payloads for point-tracker raw points on the current frame.

        We show the raw model point for the currently selected track so the user
        can compare the model's actual trajectory against the refined ROI center.
        """
        selected_track = self.get_selected_track()
        if selected_track is None:
            return []
        spot = selected_track.spots_by_image_id.get(frame_id)
        if spot is None:
            return []
        source = str(getattr(spot, "source", "")).lower()
        if source in {"tapir_offline", "tapir_online"}:
            raw_x = getattr(spot, "tapir_raw_x", None)
            raw_y = getattr(spot, "tapir_raw_y", None)
            label = (
                f"BOOTSTAPIR T{selected_track.track_id}"
                if source == "tapir_online"
                else f"TAPIR T{selected_track.track_id}"
            )
        elif source == "locotrack_offline":
            raw_x = getattr(spot, "locotrack_raw_x", None)
            raw_y = getattr(spot, "locotrack_raw_y", None)
            label = f"LOCOTRACK T{selected_track.track_id}"
        elif source == "cotracker3_offline":
            raw_x = getattr(spot, "cotracker3_raw_x", None)
            raw_y = getattr(spot, "cotracker3_raw_y", None)
            label = f"COTRACKER3 T{selected_track.track_id}"
        elif source == "tapnext_offline":
            raw_x = getattr(spot, "tapnext_raw_x", None)
            raw_y = getattr(spot, "tapnext_raw_y", None)
            label = f"TAPNEXT T{selected_track.track_id}"
        elif source == "trackonr_offline":
            raw_x = getattr(spot, "trackonr_raw_x", None)
            raw_y = getattr(spot, "trackonr_raw_y", None)
            label = f"TRACK-ON-R T{selected_track.track_id}"
        elif source == "pipspp_offline":
            raw_x = getattr(spot, "pipspp_raw_x", None)
            raw_y = getattr(spot, "pipspp_raw_y", None)
            label = f"PIPS++ T{selected_track.track_id}"
        elif source == "mftiq_offline":
            raw_x = getattr(spot, "mftiq_raw_x", None)
            raw_y = getattr(spot, "mftiq_raw_y", None)
            label = f"MFTIQ T{selected_track.track_id}"
        elif source == "cutie_offline":
            raw_x = getattr(spot, "cutie_raw_x", None)
            raw_y = getattr(spot, "cutie_raw_y", None)
            label = f"CUTIE T{selected_track.track_id}"
        elif source == "sam2_offline":
            raw_x = getattr(spot, "sam2_raw_x", None)
            raw_y = getattr(spot, "sam2_raw_y", None)
            label = f"SAM2 T{selected_track.track_id}"
        else:
            return []

        if raw_x is None or raw_y is None:
            raw_x = spot.center_x
            raw_y = spot.center_y

        if np.isfinite(float(raw_x)) and np.isfinite(float(raw_y)):
            return [{"x": float(raw_x), "y": float(raw_y), "label": label}]
        return []

    def handle_add_new_track(self, scene_pos: QPointF):
        """Creates a new track and initiates tracking from the user's click."""
        next_track_id = max([t.track_id for t in self.tracks] + [0]) + 1
        
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        
        roi_size = self.roi_size
        roi_half = roi_size // 2
        new_rect = QRect(int(scene_pos.x() - roi_half), int(scene_pos.y() - roi_half), roi_size, roi_size)
        
        new_spot = SpotROI(
            id=next_track_id,
            roi_rect=new_rect,
            center_x=scene_pos.x(),
            center_y=scene_pos.y(),
            intensity=0,
            source='manual_new'
        )
        
        new_track = Track(
            track_id=next_track_id, 
            spots_by_image_id={current_frame_id: new_spot}
        )
        
        self.tracks.append(new_track)
        
        self.populate_tracks_table()
        for row in range(self.tracks_table.rowCount()):
            if int(self.tracks_table.item(row, 0).text()) == next_track_id:
                self.tracks_table.selectRow(row)
                break
        self.show_frame(current_frame_id)
        
        print(f"Created new track {next_track_id} starting at frame {current_frame_id}.")


    def on_track_point_context_menu(self, spot_id: int, global_pos: QPoint):
        """Creates and shows the context menu for editing a track point."""
        current_row = self.tracks_table.currentRow()
        if current_row < 0: return

        menu = QMenu(self)
        
        delete_action = menu.addAction("Delete this point from track")
        
        menu.addSeparator()
        retrack_action = menu.addAction("Re-track from this point (Optical Flow)")
        retrack_tapir_action = menu.addAction("Re-track from this point (TAPIR)")
        retrack_locotrack_action = menu.addAction("Re-track from this point (LocoTrack)")
        retrack_cotracker_action = menu.addAction("Re-track from this point (CoTracker3)")
        retrack_tapnext_action = menu.addAction("Re-track from this point (TAPNext)")
        retrack_trackonr_action = menu.addAction("Re-track from this point (Track-On-R)")
        retrack_pipspp_action = menu.addAction("Re-track from this point (PIPs++)")
        retrack_mftiq_action = menu.addAction("Re-track from this point (MFTIQ)")
        retrack_cutie_action = menu.addAction("Re-track from this point (Cutie)")
        retrack_sam2_action = menu.addAction("Re-track from this point (SAM2)")
        
        action = menu.exec(global_pos)
        
        if action == delete_action:
            self.delete_point_from_track(spot_id)
        elif action == retrack_action:
            self.initiate_retrack_optical_flow(spot_id)
        elif action == retrack_tapir_action:
            self.initiate_retrack_tapir(spot_id)
        elif action == retrack_locotrack_action:
            self.initiate_retrack_locotrack(spot_id)
        elif action == retrack_cotracker_action:
            self.initiate_retrack_cotracker3(spot_id)
        elif action == retrack_tapnext_action:
            self.initiate_retrack_tapnext(spot_id)
        elif action == retrack_trackonr_action:
            self.initiate_retrack_trackonr(spot_id)
        elif action == retrack_pipspp_action:
            self.initiate_retrack_pipspp(spot_id)
        elif action == retrack_mftiq_action:
            self.initiate_retrack_mftiq(spot_id)
        elif action == retrack_cutie_action:
            self.initiate_retrack_cutie(spot_id)
        elif action == retrack_sam2_action:
            self.initiate_retrack_sam2(spot_id)

    def _get_retrack_target(self) -> tuple[Track, int] | None:
        """Return the currently selected track and frame for a local retrack."""
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            return None
        track_id_item = self.tracks_table.item(current_row, 0)
        if track_id_item is None:
            return None
        selected_track_id = int(track_id_item.text())
        track_to_retrack = next((t for t in self.tracks if t.track_id == selected_track_id), None)
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]

        if not track_to_retrack or current_frame_id not in track_to_retrack.spots_by_image_id:
            QMessageBox.warning(self, "Error", "Cannot start re-tracking. The selected track does not exist on the current frame.")
            return None
        return track_to_retrack, current_frame_id

    def initiate_retrack_optical_flow(self, spot_id: int):
        """Prepares and launches optical-flow re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        params = {
            'lk_win_size': self.lk_win_size,
            'lk_max_level': self.lk_max_level,
            'roi_size': self.roi_size 
        }

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_optical_flow,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with Optical Flow...",
            cancel_event=cancel_event,
            method_label="Optical Flow re-tracking",
        )

    def initiate_retrack_tapir(self, spot_id: int):
        """Prepares and launches TAPIR re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        current_spot = track_to_retrack.spots_by_image_id.get(current_frame_id)
        spot_source = str(getattr(current_spot, "source", "")).strip().lower() if current_spot else ""
        runtime_mode = str(self.tapir_params.get("runtime_mode", "offline")).strip().lower()
        use_online = spot_source == "tapir_online" or runtime_mode == "online"

        if not self.tapir_params:
            QMessageBox.warning(
                self,
                "TAPIR Settings Missing",
                "TAPIR parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_tapir_environment(
            models_dir=self.tapir_params.get("tapir_models_dir"),
            checkpoint_path=self.tapir_params.get("checkpoint_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "TAPIR Unavailable",
                format_tapir_environment_report(status),
            )
            return
        if use_online and not status.selected_checkpoint_is_online:
            QMessageBox.warning(
                self,
                "Causal TAPIR Required",
                "BootsTAPIR Online re-tracking requires a causal / online TAPIR checkpoint.",
            )
            return

        tapir_params = dict(self.tapir_params)
        tapir_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_tapir_online if use_online else retrack_single_spot_tapir_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            tapir_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title=(
                "Re-tracking spot with BootsTAPIR Online..."
                if use_online
                else "Re-tracking spot with TAPIR..."
            ),
            cancel_event=cancel_event,
            method_label=(
                "BootsTAPIR Online re-tracking" if use_online else "TAPIR re-tracking"
            ),
        )

    def initiate_retrack_locotrack(self, spot_id: int):
        """Prepares and launches LocoTrack re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.locotrack_params:
            QMessageBox.warning(
                self,
                "LocoTrack Settings Missing",
                "LocoTrack parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_locotrack_environment(
            models_dir=self.locotrack_params.get("locotrack_models_dir"),
            checkpoint_path=self.locotrack_params.get("checkpoint_path"),
            repo_path=self.locotrack_params.get("repo_path") or self.locotrack_params.get("locotrack_repo_path"),
        )
        backend_mode = str(self.locotrack_params.get("backend_mode", "subprocess")).strip().lower()
        python_executable = self.locotrack_params.get("python_executable")
        python_path_ok = bool(
            python_executable and Path(str(python_executable)).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )

        if backend_mode == "direct":
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "LocoTrack Unavailable",
                    format_locotrack_environment_report(status),
                )
                return
        else:
            if not structural_ready:
                QMessageBox.warning(
                    self,
                    "LocoTrack Unavailable",
                    format_locotrack_environment_report(status),
                )
                return
            if not (status.is_ready or python_path_ok):
                QMessageBox.warning(
                    self,
                    "LocoTrack Unavailable",
                    format_locotrack_environment_report(status)
                    + "\n\nSet the Python executable of the dedicated `locotrack` env "
                    "or run Trackleed from that environment.",
                )
                return

        locotrack_params = dict(self.locotrack_params)
        locotrack_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_locotrack_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            locotrack_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with LocoTrack...",
            cancel_event=cancel_event,
            method_label="LocoTrack re-tracking",
        )

    def initiate_retrack_cotracker3(self, spot_id: int):
        """Prepares and launches CoTracker3 re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.cotracker_params:
            QMessageBox.warning(
                self,
                "CoTracker3 Settings Missing",
                "CoTracker3 parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_cotracker_environment(
            models_dir=self.cotracker_params.get("cotracker3_models_dir"),
            checkpoint_path=self.cotracker_params.get("checkpoint_path"),
            repo_path=self.cotracker_params.get("repo_path")
            or self.cotracker_params.get("cotracker3_repo_path")
            or self.cotracker_params.get("cotracker_repo_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "CoTracker3 Unavailable",
                format_cotracker_environment_report(status),
            )
            return

        cotracker_params = dict(self.cotracker_params)
        cotracker_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_cotracker3_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            cotracker_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with CoTracker3...",
            cancel_event=cancel_event,
            method_label="CoTracker3 re-tracking",
        )

    def initiate_retrack_tapnext(self, spot_id: int):
        """Prepares and launches TAPNext re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.tapnext_params:
            QMessageBox.warning(
                self,
                "TAPNext Settings Missing",
                "TAPNext parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_tapnext_environment(
            models_dir=self.tapnext_params.get("tapnext_models_dir"),
            checkpoint_path=self.tapnext_params.get("checkpoint_path"),
            repo_path=self.tapnext_params.get("repo_path") or self.tapnext_params.get("tapnext_repo_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "TAPNext Unavailable",
                format_tapnext_environment_report(status),
            )
            return

        tapnext_params = dict(self.tapnext_params)
        tapnext_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_tapnext_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            tapnext_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with TAPNext...",
            cancel_event=cancel_event,
            method_label="TAPNext re-tracking",
        )

    def initiate_retrack_trackonr(self, spot_id: int):
        """Prepares and launches Track-On-R re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.trackonr_params:
            QMessageBox.warning(
                self,
                "Track-On-R Settings Missing",
                "Track-On-R parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_trackonr_environment(
            models_dir=self.trackonr_params.get("trackonr_models_dir"),
            checkpoint_path=self.trackonr_params.get("checkpoint_path"),
            repo_path=(
                self.trackonr_params.get("repo_path")
                or self.trackonr_params.get("trackonr_repo_path")
                or self.trackonr_params.get("track_on_repo_path")
            ),
        )
        backend_mode = str(self.trackonr_params.get("backend_mode", "subprocess")).strip().lower()
        python_executable = self.trackonr_params.get("python_executable")
        python_path_ok = bool(
            python_executable and Path(str(python_executable)).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )

        if backend_mode == "direct":
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "Track-On-R Unavailable",
                    format_trackonr_environment_report(status),
                )
                return
        else:
            if not structural_ready:
                QMessageBox.warning(
                    self,
                    "Track-On-R Unavailable",
                    format_trackonr_environment_report(status),
                )
                return
            if not (status.is_ready or python_path_ok):
                QMessageBox.warning(
                    self,
                    "Track-On-R Unavailable",
                    format_trackonr_environment_report(status)
                    + "\n\nSet the Python executable of the dedicated `trackonr` env "
                    "or run Trackleed from that environment.",
                )
                return

        trackonr_params = dict(self.trackonr_params)
        trackonr_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_trackonr_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            trackonr_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with Track-On-R...",
            cancel_event=cancel_event,
            method_label="Track-On-R re-tracking",
        )

    def initiate_retrack_pipspp(self, spot_id: int):
        """Prepares and launches PIPs++ re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.pipspp_params:
            QMessageBox.warning(
                self,
                "PIPs++ Settings Missing",
                "PIPs++ parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_pipspp_environment(
            models_dir=self.pipspp_params.get("pipspp_models_dir"),
            checkpoint_path=self.pipspp_params.get("checkpoint_path"),
            repo_path=self.pipspp_params.get("repo_path") or self.pipspp_params.get("pipspp_repo_path"),
        )
        backend_mode = str(self.pipspp_params.get("backend_mode", "subprocess")).strip().lower()
        python_executable = self.pipspp_params.get("python_executable")
        python_path_ok = bool(
            python_executable and Path(str(python_executable)).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.selected_checkpoint_file is not None
        )

        if backend_mode == "direct":
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "PIPs++ Unavailable",
                    format_pipspp_environment_report(status),
                )
                return
        else:
            if not structural_ready:
                QMessageBox.warning(
                    self,
                    "PIPs++ Unavailable",
                    format_pipspp_environment_report(status),
                )
                return
            if not (status.is_ready or python_path_ok):
                QMessageBox.warning(
                    self,
                    "PIPs++ Unavailable",
                    format_pipspp_environment_report(status)
                    + "\n\nSet the Python executable of the dedicated `pipspp` env "
                    "or run Trackleed from that environment.",
                )
                return

        pipspp_params = dict(self.pipspp_params)
        pipspp_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_pipspp_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            pipspp_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with PIPs++...",
            cancel_event=cancel_event,
            method_label="PIPs++ re-tracking",
        )

    def initiate_retrack_mftiq(self, spot_id: int):
        """Prepares and launches MFTIQ re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.mftiq_params:
            QMessageBox.warning(
                self,
                "MFTIQ Settings Missing",
                "MFTIQ parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_mftiq_environment(
            models_dir=self.mftiq_params.get("mftiq_models_dir"),
            checkpoint_path=self.mftiq_params.get("checkpoint_path"),
            repo_path=self.mftiq_params.get("repo_path") or self.mftiq_params.get("mftiq_repo_path"),
            config_path=self.mftiq_params.get("config_path"),
        )
        backend_mode = str(self.mftiq_params.get("backend_mode", "subprocess")).strip().lower()
        python_executable = self.mftiq_params.get("python_executable")
        python_path_ok = bool(
            python_executable and Path(str(python_executable)).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.config_exists
        )

        if backend_mode == "direct":
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "MFTIQ Unavailable",
                    format_mftiq_environment_report(status),
                )
                return
        else:
            if not structural_ready:
                QMessageBox.warning(
                    self,
                    "MFTIQ Unavailable",
                    format_mftiq_environment_report(status),
                )
                return
            if not (status.is_ready or python_path_ok):
                QMessageBox.warning(
                    self,
                    "MFTIQ Unavailable",
                    format_mftiq_environment_report(status)
                    + "\n\nSet the Python executable of the dedicated `mftiq` env "
                    "or run Trackleed from that environment.",
                )
                return

        mftiq_params = dict(self.mftiq_params)
        mftiq_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_mftiq_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            mftiq_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with MFTIQ...",
            cancel_event=cancel_event,
            method_label="MFTIQ re-tracking",
        )

    def initiate_retrack_cutie(self, spot_id: int):
        """Prepares and launches Cutie re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.cutie_params:
            QMessageBox.warning(
                self,
                "Cutie Settings Missing",
                "Cutie parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_cutie_environment(
            models_dir=self.cutie_params.get("cutie_models_dir"),
            checkpoint_path=self.cutie_params.get("checkpoint_path"),
            repo_path=self.cutie_params.get("repo_path") or self.cutie_params.get("cutie_repo_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "Cutie Unavailable",
                format_cutie_environment_report(status),
            )
            return

        cutie_params = dict(self.cutie_params)
        cutie_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_cutie_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            cutie_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with Cutie...",
            cancel_event=cancel_event,
            method_label="Cutie re-tracking",
        )

    def initiate_retrack_sam2(self, spot_id: int):
        """Prepares and launches SAM 2 re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target

        if not self.sam2_params:
            QMessageBox.warning(
                self,
                "SAM2 Settings Missing",
                "SAM2 parameters were not provided to the tracking visualizer.",
            )
            return

        status = probe_sam2_environment(
            models_dir=self.sam2_params.get("sam2_models_dir"),
            checkpoint_path=self.sam2_params.get("checkpoint_path"),
            repo_path=self.sam2_params.get("repo_path") or self.sam2_params.get("sam2_repo_path"),
            config_path=self.sam2_params.get("config_path"),
        )
        if not status.is_ready and self.sam2_params.get("config_path"):
            status = probe_sam2_environment(
                models_dir=self.sam2_params.get("sam2_models_dir"),
                checkpoint_path=self.sam2_params.get("checkpoint_path"),
                repo_path=self.sam2_params.get("repo_path") or self.sam2_params.get("sam2_repo_path"),
                config_path=None,
            )
        backend_mode = str(self.sam2_params.get("backend_mode") or "subprocess").strip().lower()
        python_executable = self.sam2_params.get("python_executable")
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.config_exists
            and status.config_identifier is not None
        )
        if backend_mode == "direct":
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "SAM2 Unavailable",
                    format_sam2_environment_report(status),
                )
                return
        else:
            if not structural_ready:
                QMessageBox.warning(
                    self,
                    "SAM2 Unavailable",
                    format_sam2_environment_report(status),
                )
                return
            if not (status.is_ready or python_path_ok):
                QMessageBox.warning(
                    self,
                    "SAM2 Unavailable",
                    format_sam2_environment_report(status)
                    + "\n\nSet the Python executable of the dedicated `sam2` env "
                    "or run Trackleed from that environment.",
                )
                return

        sam2_params = dict(self.sam2_params)
        sam2_params["roi_size"] = self.roi_size

        cancel_event = Event()
        worker = GenericWorker(
            retrack_single_spot_sam2_offline,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            sam2_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title="Re-tracking spot with SAM2...",
            cancel_event=cancel_event,
            method_label="SAM2 re-tracking",
        )

    def launch_retrack_worker(
        self,
        worker: GenericWorker,
        title: str,
        cancel_event: Event | None = None,
        method_label: str = "Re-tracking",
    ):
        """Runs the re-tracking worker with a progress dialog."""
        self._active_retrack_label = method_label
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

        self.worker_thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.worker_thread)

        if cancel_event is not None:
            self.progress_dialog.canceled.connect(cancel_event.set)
        self.progress_dialog.canceled.connect(self.worker_thread.requestInterruption)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.on_retrack_finished)
        self.worker.error.connect(self.on_retrack_error)
        self.worker.error.connect(self.progress_dialog.accept)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.worker.finished.connect(self.progress_dialog.accept)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def on_retrack_finished(self, updated_track: Track):
        """Handles the successful completion of the re-tracking task."""
        if updated_track is None:
            QMessageBox.information(self, "Cancelled", f"{self._active_retrack_label} was cancelled.")
            return

        print(f"Re-tracking for track {updated_track.track_id} finished.")
        
        for i, track in enumerate(self.tracks):
            if track.track_id == updated_track.track_id:
                self.tracks[i] = updated_track
                break
        
        self.populate_tracks_table()
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        self.show_frame(current_frame_id)
        
        QMessageBox.information(self, "Success", f"{self._active_retrack_label} completed successfully.")

    def on_retrack_error(self, error_message: str):
        """Handles errors from the re-tracking worker."""
        QMessageBox.critical(self, f"{self._active_retrack_label} Error", f"An error occurred:\n{error_message}")
    
    def handle_add_or_correct_point(self, scene_pos: QPointF):
        """Slot to handle adding a new point to or correcting an existing point in a track."""
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Track Selected", "Please select a track from the table first.")
            return
        track_id_item = self.tracks_table.item(current_row, 0)
        if not track_id_item: return
        
        selected_track_id = int(track_id_item.text())
        selected_track = next((t for t in self.tracks if t.track_id == selected_track_id), None)
        if not selected_track: return

        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]

        # TODO: Pass roi_size from MainWindow
        roi_size = self.roi_size 
        roi_half = roi_size // 2
        new_rect = QRect(int(scene_pos.x() - roi_half), int(scene_pos.y() - roi_half), roi_size, roi_size)
        
        all_spot_ids_on_frame = {s.id for t in self.tracks for s in t.spots_by_image_id.values() if s.id is not None}
        next_spot_id = max(all_spot_ids_on_frame | {0}) + 1

        new_spot = SpotROI(
            id=next_spot_id,
            roi_rect=new_rect,
            center_x=scene_pos.x(),
            center_y=scene_pos.y(),
            intensity=0,
            source='manual_corr'
        )
        
        selected_track.spots_by_image_id[current_frame_id] = new_spot
        
        print(f"Updated track {selected_track.track_id} on frame {current_frame_id} with new spot {new_spot.id}")
        
        self.populate_tracks_table()
        self.show_frame(current_frame_id)

    def delete_point_from_track(self, spot_id_to_delete: int):
        """Finds the track containing the spot and removes the spot from it."""
        print(f"[Dialog] Attempting to delete spot with ID: {spot_id_to_delete}")

        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            print("[Dialog] ERROR: Invalid slider index. Aborting deletion.")
            return
        current_frame_id = self.sorted_frame_ids[slider_index]
        print(f"[Dialog] Current Frame ID is: {current_frame_id}")
        
        track_to_modify = None
        
        # Find the track that contains this spot on this frame
        for track in self.tracks:
            # Check if the track exists on the current frame
            if current_frame_id in track.spots_by_image_id:
                # Check if the spot on that frame has the ID we're looking for
                if track.spots_by_image_id[current_frame_id].id == spot_id_to_delete:
                    track_to_modify = track
                    print(f"[Dialog] Found spot in Track ID: {track.track_id}")
                    break
        
        if track_to_modify:
            print(f"[Dialog] Proceeding with deletion from track {track_to_modify.track_id}...")
            # Remove the spot entry for the current frame from the track's dictionary
            del track_to_modify.spots_by_image_id[current_frame_id]
            
            # If the track is now empty, remove it completely
            if not track_to_modify.spots_by_image_id:
                print(f"[Dialog] Track {track_to_modify.track_id} is now empty. Removing track.")
                self.tracks.remove(track_to_modify)

            # Refresh all UI elements
            print("[Dialog] Refreshing UI...")
            self.populate_tracks_table()
            self.show_frame(current_frame_id)
            self._update_highlights()
            print("[Dialog] Deletion and UI refresh complete.")
        else:
            print(f"[Dialog] ERROR: Could not find a track on frame {current_frame_id} containing spot with ID {spot_id_to_delete}.")

    
    def toggle_animation(self, checked: bool):
        """Starts or stops the frame-by-frame animation."""
        if checked:
            self.play_button.setText("Pause")
            self.animation_timer.start(100) # Czas w ms, 100ms = 10 klatek/s
        else:
            self.play_button.setText("Play")
            self.animation_timer.stop()

    def advance_frame(self):
        """Moves the slider to the next frame, looping at the end."""
        current_val = self.frame_slider.value()
        max_val = self.frame_slider.maximum()
        next_val = current_val + 1
        if next_val > max_val:
            next_val = self.frame_slider.minimum()
        self.frame_slider.setValue(next_val)

    def delete_selected_track(self):
        """Deletes the currently selected track from the table and the data model."""
        selected_items = self.tracks_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a track in the table to delete.")
            return

        track_id_to_delete = int(self.tracks_table.item(selected_items[0].row(), 0).text())

        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to delete Track ID: {track_id_to_delete}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # Remove track from the main list
            self.tracks = [t for t in self.tracks if t.track_id != track_id_to_delete]
            # Refresh the UI
            self.populate_tracks_table()
            self.image_viewer.show_track_path(None) # Clear any drawn path

    def merge_selected_tracks(self):
        """Merge exactly two selected tracks into one timeline."""
        selected_rows = sorted({idx.row() for idx in self.tracks_table.selectedIndexes()})
        if len(selected_rows) != 2:
            QMessageBox.warning(self, "Select Two Tracks", "Select exactly two tracks to merge.")
            return

        row_a, row_b = selected_rows
        track_id_a = int(self.tracks_table.item(row_a, 0).text())
        track_id_b = int(self.tracks_table.item(row_b, 0).text())

        track_a = next((t for t in self.tracks if t.track_id == track_id_a), None)
        track_b = next((t for t in self.tracks if t.track_id == track_id_b), None)
        if not track_a or not track_b:
            QMessageBox.warning(self, "Tracks Missing", "Could not find the selected tracks.")
            return

        # Pick base track: earlier start frame, fallback to lower track_id
        start_a = min(track_a.spots_by_image_id.keys())
        start_b = min(track_b.spots_by_image_id.keys())
        base, other = (track_a, track_b) if (start_a, track_a.track_id) <= (start_b, track_b.track_id) else (track_b, track_a)

        merged_spots = dict(base.spots_by_image_id)
        conflicts = []
        added = 0
        for fid, spot in other.spots_by_image_id.items():
            if fid in merged_spots:
                conflicts.append(fid)
                continue
            merged_spots[fid] = spot
            added += 1

        merged_track = Track(track_id=base.track_id, spots_by_image_id=merged_spots)

        # Replace tracks in list
        self.tracks = [t for t in self.tracks if t.track_id not in {track_id_a, track_id_b}]
        self.tracks.append(merged_track)

        self.populate_tracks_table()
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        self.show_frame(current_frame_id)

        # Select merged track row
        for row in range(self.tracks_table.rowCount()):
            if int(self.tracks_table.item(row, 0).text()) == merged_track.track_id:
                self.tracks_table.selectRow(row)
                break

        msg = f"Merged tracks {track_id_a} and {track_id_b} into {merged_track.track_id}. Added {added} frames."
        if conflicts:
            msg += f" Skipped {len(conflicts)} overlapping frame(s): {', '.join(map(str, conflicts))}."
        QMessageBox.information(self, "Tracks Merged", msg)

    def populate_tracks_table(self):
        """Fills the table with data from the tracks."""
        self.tracks_table.setRowCount(len(self.tracks))
        for row, track in enumerate(self.tracks):
            frame_ids = track.spots_by_image_id.keys()
            start_frame = min(frame_ids) if frame_ids else -1
            end_frame = max(frame_ids) if frame_ids else -1
            method = self._classify_track_method(track)
            gaps = self._count_track_gaps(track)
            self.tracks_table.setItem(row, 0, QTableWidgetItem(str(track.track_id)))
            self.tracks_table.setItem(row, 1, QTableWidgetItem(method))
            self.tracks_table.setItem(row, 2, QTableWidgetItem(str(start_frame)))
            self.tracks_table.setItem(row, 3, QTableWidgetItem(str(end_frame)))
            self.tracks_table.setItem(row, 4, QTableWidgetItem(str(len(frame_ids))))
            self.tracks_table.setItem(row, 5, QTableWidgetItem(str(gaps)))

    def on_slider_value_changed(self, index: int):
        """
        NEW: Slot that translates a slider index into a frame ID and calls show_frame.
        """
        if not self.sorted_frame_ids or not (0 <= index < len(self.sorted_frame_ids)):
            return
        
        # Translate index to the actual frame ID (time value)
        frame_id = self.sorted_frame_ids[index]
        self.show_frame(frame_id)
        
        # Update the label to be more informative
        self.frame_label.setText(f"Frame {index + 1}/{len(self.sorted_frame_ids)} (ID: {frame_id})")

    def on_track_selected(self):
        """Handles the table selection change by updating all highlights."""
        if self.show_selected_track_only_button.isChecked():
            self.refresh_current_frame()
        self._update_highlights()

    def _update_highlights(self):
        """
        Helper method to apply highlights to the viewer based on the current table selection.
        This should be called whenever the view or selection changes.
        """
        selected_track = self.get_selected_track()
        if selected_track is None:
            self.image_viewer.highlight_roi(None)
            self.image_viewer.show_track_path(None)
            self._update_track_details()
            return
        self.image_viewer.show_track_path(selected_track)
        
        # 1. Get the current SLIDER INDEX
        slider_index = self.frame_slider.value()
        # 2. Translate the INDEX to the actual FRAME ID
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            return
        current_frame_id = self.sorted_frame_ids[slider_index]
        
        # 3. Now use the correct frame_id to find the spot
        if current_frame_id in selected_track.spots_by_image_id:
            spot_id_to_highlight = selected_track.spots_by_image_id[current_frame_id].id
            self.image_viewer.highlight_roi(spot_id_to_highlight)
        else:
            self.image_viewer.highlight_roi(None)
        self._update_track_details()

    def show_frame(self, frame_id: int, fit: bool = False):
        """Loads and displays the image and overlays for a given frame ID."""
        leed_file = self.file_map.get(frame_id)
        if not leed_file: return

        image_data = load_tiff(leed_file.filepath)
        pixmap = self._convert_numpy_to_pixmap(image_data)
        self.image_viewer.set_image(pixmap, fit=fit)

        self.image_viewer.update_masks(self.masks)
        self.image_viewer.update_rois(self._get_spots_to_display_for_frame(frame_id))
        self.image_viewer.update_positions_overlay(
            overlay_groups=[],
            focus_points=self._build_point_tracker_focus_points_for_frame(frame_id),
        )

        self._update_highlights()

    def _convert_numpy_to_pixmap(self, image_data):
        """Helper to convert a NumPy array to a QPixmap."""
        if image_data is None or image_data.size == 0:
            return QPixmap()

        arr = np.asarray(image_data)

        # Handle channel-first TIFFs (C, H, W) by converting to (H, W, C).
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        # Normalize to uint8 for display.
        if arr.dtype != np.uint8:
            arr_float = arr.astype(np.float32)
            min_val = float(np.nanmin(arr_float))
            max_val = float(np.nanmax(arr_float))
            if max_val > min_val:
                arr = (255.0 * (arr_float - min_val) / (max_val - min_val)).clip(0, 255).astype(np.uint8)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint8)

        if arr.ndim == 2:
            height, width = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
            return QPixmap.fromImage(q_image)

        if arr.ndim == 3 and arr.shape[2] == 3:
            height, width, _ = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(arr.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(q_image)

        if arr.ndim == 3 and arr.shape[2] == 4:
            height, width, _ = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(arr.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
            return QPixmap.fromImage(q_image)

        # Fallback for uncommon layouts: reduce to grayscale.
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2).astype(np.uint8)
            height, width = arr.shape
            arr = np.ascontiguousarray(arr)
            q_image = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
            return QPixmap.fromImage(q_image)

        return QPixmap()

    def get_updated_tracks(self) -> List[Track]:
        """Returns the final, possibly edited, list of tracks."""
        return self.tracks
