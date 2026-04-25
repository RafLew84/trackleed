from __future__ import annotations

# stal/gui/tracking_dialog.py

from pathlib import Path
from typing import List
from threading import Event
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QThread
from PyQt6.QtGui import QImage, QPixmap, QActionGroup, QAction, QColor
from PyQt6.QtWidgets import (QDialog, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QSplitter,
                             QLabel, QTableWidget, QAbstractItemView, QTableWidgetItem, QDialogButtonBox,
                             QHeaderView, QPushButton, QMessageBox, QToolBar, QMenu, QMessageBox, QProgressDialog,
                             QComboBox)

from trackleed.core.data_models import LeedFile, Track, SpotROI
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.gui.polygon_mask_dialog import PolygonMaskDialog
from trackleed.gui.tracker_registry import get_tracker_spec_for_source, require_tracker_spec
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.tomp_backend import (
    select_default_tamos_checkpoint_for_variant,
    select_default_tomp_checkpoint_for_variant,
)
from trackleed.analysis.tracking_retrack import (
    retrack_single_spot_optical_flow,
)
from trackleed.logic.generic_worker import GenericWorker


MANUAL_TRACK_SOURCES = {"manual", "manual_new", "manual_corr", "manual_point"}

TRACKER_METHOD_LABEL_BY_SOURCE = {
    "tapir_offline": "TAPIR",
    "tapir_online": "BootsTAPIR Online",
    "locotrack_offline": "LocoTrack",
    "cotracker3_offline": "CoTracker3",
    "tapnext_offline": "TAPNext",
    "trackonr_offline": "Track-On-R",
    "pipspp_offline": "PIPs++",
    "mftiq_offline": "MFTIQ",
    "cutie_offline": "Cutie",
    "xmem2_offline": "XMem2",
    "tomp_offline": "ToMP",
    "tamos_offline": "TaMOs",
    "sam2_offline": "SAM2",
    "deva_offline": "DEVA",
}

MANUAL_AUGMENTED_METHOD_LABEL_BY_SOURCE = {
    "tapir_offline": "TAPIR+Manual",
    "tapir_online": "BootsTAPIR Online+Manual",
    "locotrack_offline": "LocoTrack+Manual",
    "cotracker3_offline": "CoTracker3+Manual",
    "tapnext_offline": "TAPNext+Manual",
    "trackonr_offline": "Track-On-R+Manual",
    "pipspp_offline": "PIPs++ + Manual",
    "mftiq_offline": "MFTIQ + Manual",
    "cutie_offline": "Cutie + Manual",
    "xmem2_offline": "XMem2 + Manual",
    "tomp_offline": "ToMP + Manual",
    "tamos_offline": "TaMOs + Manual",
    "sam2_offline": "SAM2 + Manual",
    "deva_offline": "DEVA + Manual",
}

TRACKER_VISUALIZER_RETRACK_METHOD_NAME_BY_ID = {
    "tapir_offline": "initiate_retrack_tapir",
    "tapir_online": "initiate_retrack_tapir",
    "locotrack_offline": "initiate_retrack_locotrack",
    "cotracker3_offline": "initiate_retrack_cotracker3",
    "tapnext_offline": "initiate_retrack_tapnext",
    "trackonr_offline": "initiate_retrack_trackonr",
    "pipspp_offline": "initiate_retrack_pipspp",
    "mftiq_offline": "initiate_retrack_mftiq",
    "cutie_offline": "initiate_retrack_cutie",
    "xmem2_offline": "initiate_retrack_xmem2",
    "tomp_offline": "initiate_retrack_tomp",
    "sam2_offline": "initiate_retrack_sam2",
}

TRACKER_UNAVAILABLE_TITLE_BY_ID = {
    "tapir_offline": "TAPIR Unavailable",
    "tapir_online": "TAPIR Unavailable",
    "locotrack_offline": "LocoTrack Unavailable",
    "cotracker3_offline": "CoTracker3 Unavailable",
    "tapnext_offline": "TAPNext Unavailable",
    "trackonr_offline": "Track-On-R Unavailable",
    "pipspp_offline": "PIPs++ Unavailable",
    "mftiq_offline": "MFTIQ Unavailable",
    "cutie_offline": "Cutie Unavailable",
    "xmem2_offline": "XMem2 Unavailable",
    "tomp_offline": "ToMP / TaMOs Unavailable",
    "sam2_offline": "SAM2 Unavailable",
    "deva_offline": "DEVA Unavailable",
}

TRACKER_ENV_HINT_BY_ID = {
    "locotrack_offline": "locotrack-pytorch",
    "trackonr_offline": "trackonr2",
    "pipspp_offline": "pipspp",
    "mftiq_offline": "mftiq",
    "cutie_offline": "cutie",
    "xmem2_offline": "xmem2",
    "tomp_offline": "tomp",
    "sam2_offline": "sam2",
    "deva_offline": "deva",
}

POINT_TRACKER_DETAIL_SPECS = {
    "tapir_offline": {"prefix": "tapir", "raw_label": "Raw TAPIR", "focus_label": "TAPIR"},
    "tapir_online": {"prefix": "tapir", "raw_label": "Raw BootsTAPIR Online", "focus_label": "BOOTSTAPIR"},
    "locotrack_offline": {"prefix": "locotrack", "raw_label": "Raw LocoTrack", "focus_label": "LOCOTRACK"},
    "cotracker3_offline": {"prefix": "cotracker3", "raw_label": "Raw CoTracker3", "focus_label": "COTRACKER3"},
    "tapnext_offline": {
        "prefix": "tapnext",
        "raw_label": "Raw TAPNext",
        "focus_label": "TAPNEXT",
        "extra_fields": (
            ("Visibility score", "tapnext_visibility_score", "float3"),
            ("Certainty score", "tapnext_certainty_score", "float3"),
            ("Combined score", "tapnext_combined_score", "float3"),
        ),
    },
    "trackonr_offline": {"prefix": "trackonr", "raw_label": "Raw Track-On-R", "focus_label": "TRACK-ON-R"},
    "pipspp_offline": {"prefix": "pipspp", "raw_label": "Raw PIPs++", "focus_label": "PIPS++"},
    "mftiq_offline": {
        "prefix": "mftiq",
        "raw_label": "Raw MFTIQ",
        "focus_label": "MFTIQ",
        "extra_fields": (
            ("Occlusion score", "mftiq_occlusion_score", "float3"),
            ("Sigma score", "mftiq_sigma_score", "float3"),
        ),
    },
}

MASK_TRACKER_DETAIL_SPECS = {
    "cutie_offline": {
        "prefix": "cutie",
        "include_initial_mask_mode": True,
        "include_memory_anchor": True,
        "include_prompt_mode": False,
        "include_component_count": False,
    },
    "xmem2_offline": {
        "prefix": "xmem2",
        "include_initial_mask_mode": True,
        "include_memory_anchor": False,
        "include_prompt_mode": False,
        "include_component_count": False,
    },
    "sam2_offline": {
        "prefix": "sam2",
        "include_initial_mask_mode": False,
        "include_memory_anchor": False,
        "include_prompt_mode": True,
        "include_component_count": True,
    },
    "deva_offline": {
        "prefix": "deva",
        "include_initial_mask_mode": True,
        "include_memory_anchor": False,
        "include_prompt_mode": False,
        "include_component_count": False,
    },
}


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
        xmem2_params: dict | None = None,
        tomp_params: dict | None = None,
        sam2_params: dict | None = None,
        deva_params: dict | None = None,
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
        self.xmem2_params = dict(xmem2_params or {})
        self.tomp_params = dict(tomp_params or {})
        self.sam2_params = dict(sam2_params or {})
        self.deva_params = dict(deva_params or {})
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

    def _get_current_frame_id(self) -> int | None:
        """Return the currently displayed frame id, if any."""
        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            return None
        return self.sorted_frame_ids[slider_index]

    def _get_selected_spot_for_frame(self, frame_id: int | None) -> SpotROI | None:
        """Return the spot of the selected track on the requested frame."""
        if frame_id is None:
            return None
        selected_track = self.get_selected_track()
        if selected_track is None:
            return None
        return selected_track.spots_by_image_id.get(frame_id)

    def _normalized_source_name(self, source_name: str | None) -> str:
        return str(source_name or "").strip().lower()

    def _tracker_spec_for_source_name(self, source_name: str | None):
        normalized_source = self._normalized_source_name(source_name)
        if not normalized_source or normalized_source in MANUAL_TRACK_SOURCES:
            return None
        return get_tracker_spec_for_source(normalized_source)

    def _tracker_spec_for_track(self, track: Track | None):
        if track is None:
            return None
        specs_by_id = {}
        for spot in track.spots_by_image_id.values():
            spec = self._tracker_spec_for_source_name(getattr(spot, "source", None))
            if spec is not None:
                specs_by_id[spec.tracker_id] = spec
        if len(specs_by_id) == 1:
            return next(iter(specs_by_id.values()))
        return None

    def _method_label_for_source_name(self, source_name: str | None, *, include_manual: bool = False) -> str | None:
        normalized_source = self._normalized_source_name(source_name)
        if not normalized_source:
            return None
        if normalized_source in MANUAL_TRACK_SOURCES:
            return "Manual"
        if include_manual:
            return MANUAL_AUGMENTED_METHOD_LABEL_BY_SOURCE.get(normalized_source)
        return TRACKER_METHOD_LABEL_BY_SOURCE.get(normalized_source)

    def _base_method_label_for_track(self, track: Track | None) -> str | None:
        if track is None:
            return None
        method = self._classify_track_method(track)
        method = method.removesuffix(" + Manual").removesuffix("+Manual")
        if method in {"Manual", "Unknown", "Mixed", "OptFlow", "YOLO", "Auto"}:
            return None
        return method

    def _append_formatted_value_line(self, lines: list[str], label: str, value, fmt: str) -> None:
        if value is None:
            return
        if fmt == "float1":
            lines.append(f"{label}: {float(value):.1f}")
        elif fmt == "float2":
            lines.append(f"{label}: {float(value):.2f}")
        elif fmt == "float3":
            lines.append(f"{label}: {float(value):.3f}")
        elif fmt == "int":
            lines.append(f"{label}: {int(value)}")
        else:
            lines.append(f"{label}: {value}")

    def _append_point_tracker_detail_lines(self, lines: list[str], spot: SpotROI, *, source: str) -> bool:
        spec = POINT_TRACKER_DETAIL_SPECS.get(source)
        if spec is None:
            return False

        prefix = spec["prefix"]
        raw_x = getattr(spot, f"{prefix}_raw_x", None)
        raw_y = getattr(spot, f"{prefix}_raw_y", None)
        refined_x = getattr(spot, f"{prefix}_refined_x", None)
        refined_y = getattr(spot, f"{prefix}_refined_y", None)
        visibility = getattr(spot, f"{prefix}_visibility", None)
        rejected_reason = getattr(spot, f"{prefix}_filter_rejected_reason", None)

        if raw_x is not None and raw_y is not None:
            lines.append(f"{spec['raw_label']}: ({float(raw_x):.2f}, {float(raw_y):.2f})")
        if refined_x is not None and refined_y is not None:
            lines.append(f"Refined: ({float(refined_x):.2f}, {float(refined_y):.2f})")
        if visibility is not None:
            lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")

        for label, attr_name, fmt in spec.get("extra_fields", ()):
            self._append_formatted_value_line(lines, label, getattr(spot, attr_name, None), fmt)

        if rejected_reason:
            lines.append(f"Filter: rejected ({rejected_reason})")
        return True

    def _append_mask_tracker_detail_lines(self, lines: list[str], spot: SpotROI, *, source: str) -> bool:
        spec = MASK_TRACKER_DETAIL_SPECS.get(source)
        if spec is None:
            return False

        prefix = spec["prefix"]
        raw_x = getattr(spot, f"{prefix}_raw_x", None)
        raw_y = getattr(spot, f"{prefix}_raw_y", None)
        refined_x = getattr(spot, f"{prefix}_refined_x", None)
        refined_y = getattr(spot, f"{prefix}_refined_y", None)
        visibility = getattr(spot, f"{prefix}_visibility", None)
        repr_mode = getattr(spot, f"{prefix}_repr_mode", None)
        rejected_reason = getattr(spot, f"{prefix}_filter_rejected_reason", None)
        mask_area = getattr(spot, f"{prefix}_mask_area", None)
        mask_score = getattr(spot, f"{prefix}_mask_score", None)
        mask_bbox_xyxy = getattr(spot, f"{prefix}_mask_bbox_xyxy", None)
        mask_bbox_aspect_ratio = getattr(spot, f"{prefix}_mask_bbox_aspect_ratio", None)

        if raw_x is not None and raw_y is not None:
            lines.append(f"Representative point: ({float(raw_x):.2f}, {float(raw_y):.2f})")
        if refined_x is not None and refined_y is not None:
            lines.append(f"Stored point: ({float(refined_x):.2f}, {float(refined_y):.2f})")
        if visibility is not None:
            lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
        if repr_mode:
            lines.append(f"Representative mode: {repr_mode}")
        if spec["include_initial_mask_mode"]:
            initial_mask_mode = getattr(spot, f"{prefix}_initial_mask_mode", None)
            if initial_mask_mode:
                lines.append(f"Initial mask mode: {initial_mask_mode}")
        if spec["include_prompt_mode"]:
            prompt_mode = getattr(spot, f"{prefix}_prompt_mode", None)
            if prompt_mode:
                lines.append(f"Prompt mode: {prompt_mode}")
        if spec["include_memory_anchor"] and bool(getattr(spot, "xmem2_permanent_reference", False)):
            lines.append("Permanent memory anchor: yes")
        self._append_formatted_value_line(lines, "Mask area", mask_area, "float1")
        self._append_formatted_value_line(lines, "Mask score", mask_score, "float3")
        if spec["include_component_count"]:
            self._append_formatted_value_line(
                lines,
                "Mask components",
                getattr(spot, f"{prefix}_mask_component_count", None),
                "int",
            )
        if mask_bbox_xyxy is not None:
            try:
                x0, y0, x1, y1 = [float(v) for v in mask_bbox_xyxy]
                lines.append(f"Mask bbox: x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}")
            except Exception:
                pass
        self._append_formatted_value_line(lines, "BBox aspect ratio", mask_bbox_aspect_ratio, "float3")
        if rejected_reason:
            lines.append(f"Filter: rejected ({rejected_reason})")
        return True

    def _append_tomp_detail_lines(self, lines: list[str], spot: SpotROI, *, source: str) -> bool:
        if source not in {"tomp_offline", "tamos_offline"}:
            return False

        raw_x = getattr(spot, "tomp_raw_x", None)
        raw_y = getattr(spot, "tomp_raw_y", None)
        refined_x = getattr(spot, "tomp_refined_x", None)
        refined_y = getattr(spot, "tomp_refined_y", None)
        visibility = getattr(spot, "tomp_visibility", None)
        bbox_score = getattr(spot, "tomp_bbox_score", None)
        bbox_xywh = getattr(spot, "tomp_bbox_xywh", None)
        bbox_xyxy = getattr(spot, "tomp_bbox_xyxy", None)
        bbox_aspect_ratio = getattr(spot, "tomp_bbox_aspect_ratio", None)
        repr_mode = getattr(spot, "tomp_repr_mode", None)
        model_variant = getattr(spot, "tomp_model_variant", None)
        rejected_reason = getattr(spot, "tomp_filter_rejected_reason", None)

        if raw_x is not None and raw_y is not None:
            lines.append(f"Representative point: ({float(raw_x):.2f}, {float(raw_y):.2f})")
        if refined_x is not None and refined_y is not None:
            lines.append(f"Stored point: ({float(refined_x):.2f}, {float(refined_y):.2f})")
        if visibility is not None:
            lines.append(f"Visible: {'yes' if bool(visibility) else 'no'}")
        if model_variant:
            lines.append(f"Model variant: {model_variant}")
        if repr_mode:
            lines.append(f"Representative mode: {repr_mode}")
        self._append_formatted_value_line(lines, "BBox score", bbox_score, "float3")
        if bbox_xywh is not None:
            try:
                x, y, w, h = [float(v) for v in bbox_xywh]
                lines.append(f"BBox xywh: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
            except Exception:
                pass
        if bbox_xyxy is not None:
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
                lines.append(f"BBox xyxy: x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}")
            except Exception:
                pass
        self._append_formatted_value_line(lines, "BBox aspect ratio", bbox_aspect_ratio, "float3")
        if rejected_reason:
            lines.append(f"Filter: rejected ({rejected_reason})")
        return True

    def _append_spot_backend_detail_lines(self, lines: list[str], spot: SpotROI) -> None:
        source = self._normalized_source_name(getattr(spot, "source", None))
        if self._append_point_tracker_detail_lines(lines, spot, source=source):
            return
        if self._append_mask_tracker_detail_lines(lines, spot, source=source):
            return
        self._append_tomp_detail_lines(lines, spot, source=source)

    def _build_focus_point_payload_for_spot(self, track: Track, spot: SpotROI) -> list[dict]:
        source = self._normalized_source_name(getattr(spot, "source", None))
        spec = POINT_TRACKER_DETAIL_SPECS.get(source)
        if spec is not None:
            raw_x = getattr(spot, f"{spec['prefix']}_raw_x", None)
            raw_y = getattr(spot, f"{spec['prefix']}_raw_y", None)
            label = f"{spec['focus_label']} T{track.track_id}"
        elif source == "cutie_offline":
            raw_x = getattr(spot, "cutie_raw_x", None)
            raw_y = getattr(spot, "cutie_raw_y", None)
            label = f"CUTIE T{track.track_id}"
        elif source == "xmem2_offline":
            raw_x = getattr(spot, "xmem2_raw_x", None)
            raw_y = getattr(spot, "xmem2_raw_y", None)
            label = f"XMEM2 T{track.track_id}"
        elif source in {"tomp_offline", "tamos_offline"}:
            raw_x = getattr(spot, "tomp_raw_x", None)
            raw_y = getattr(spot, "tomp_raw_y", None)
            label = f"{'TAMOS' if source == 'tamos_offline' else 'TOMP'} T{track.track_id}"
        elif source == "sam2_offline":
            raw_x = getattr(spot, "sam2_raw_x", None)
            raw_y = getattr(spot, "sam2_raw_y", None)
            label = f"SAM2 T{track.track_id}"
        elif source == "deva_offline":
            raw_x = getattr(spot, "deva_raw_x", None)
            raw_y = getattr(spot, "deva_raw_y", None)
            label = f"DEVA T{track.track_id}"
        else:
            return []

        if raw_x is None or raw_y is None:
            raw_x = spot.center_x
            raw_y = spot.center_y

        if np.isfinite(float(raw_x)) and np.isfinite(float(raw_y)):
            return [{"x": float(raw_x), "y": float(raw_y), "label": label}]
        return []

    def _backend_retrack_action_text(self, track: Track | None) -> str | None:
        tracker_spec = self._tracker_spec_for_track(track)
        if tracker_spec is None or not tracker_spec.has_visualizer_retrack:
            return None
        method_label = self._base_method_label_for_track(track)
        if not method_label:
            return None
        return f"Re-track from this point ({method_label})"

    def _initiate_retrack_for_tracker_spec(self, tracker_spec, spot_id: int) -> None:
        tracker_id = str(getattr(tracker_spec, "tracker_id", "") or "").strip().lower()
        method_name = TRACKER_VISUALIZER_RETRACK_METHOD_NAME_BY_ID.get(tracker_id)
        if not method_name:
            raise ValueError(f"Unsupported tracker for visualizer retrack: {tracker_id}")
        retrack_method = getattr(self, method_name, None)
        if not callable(retrack_method):
            raise AttributeError(
                f"TrackingDialog does not expose retrack handler `{method_name}` for `{tracker_id}`."
            )
        retrack_method(spot_id)

    def _track_supports_polygon_seed(self, track: Track | None) -> bool:
        """Polygon seed workflow is currently available only for Cutie and XMem2."""
        tracker_spec = self._tracker_spec_for_track(track)
        return tracker_spec is not None and tracker_spec.tracker_id in {"cutie_offline", "xmem2_offline"}

    def _build_polygon_seed_overlay_for_frame(self, frame_id: int) -> list[dict]:
        """Build polygon overlay payload for the selected track on the current frame."""
        selected_track = self.get_selected_track()
        if selected_track is None or not self._track_supports_polygon_seed(selected_track):
            return []

        spot = selected_track.spots_by_image_id.get(frame_id)
        if spot is None or not getattr(spot, "has_polygon_seed", False):
            return []

        method_label = self._classify_track_method(selected_track)
        return [
            {
                "points": list(getattr(spot, "polygon_seed_points", [])),
                "color": QColor(255, 165, 0),
                "label": f"{method_label.upper()} POLY T{selected_track.track_id}",
                "emphasized": True,
            }
        ]

    def _edit_polygon_mask_for_selected_spot(self):
        """Open the polygon editor for the selected track on the current frame."""
        current_frame_id = self._get_current_frame_id()
        selected_track = self.get_selected_track()
        spot = self._get_selected_spot_for_frame(current_frame_id)
        if (
            current_frame_id is None
            or selected_track is None
            or spot is None
            or not self._track_supports_polygon_seed(selected_track)
        ):
            QMessageBox.information(
                self,
                "Polygon Seed",
                "Polygon seed editing is available only for Cutie and XMem2 tracks.",
            )
            return

        leed_file = self.file_map.get(current_frame_id)
        if leed_file is None:
            QMessageBox.warning(self, "Image Missing", "Could not find the current frame to edit its polygon mask.")
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
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        spot.polygon_seed_points = dialog.get_polygon_points()
        self.show_frame(current_frame_id)

    def _clear_polygon_mask_for_selected_spot(self):
        """Remove the polygon seed from the selected track on the current frame."""
        current_frame_id = self._get_current_frame_id()
        selected_track = self.get_selected_track()
        spot = self._get_selected_spot_for_frame(current_frame_id)
        if (
            current_frame_id is None
            or selected_track is None
            or spot is None
            or not self._track_supports_polygon_seed(selected_track)
        ):
            QMessageBox.information(
                self,
                "Polygon Seed",
                "Polygon seed editing is available only for Cutie and XMem2 tracks.",
            )
            return
        if not getattr(spot, "has_polygon_seed", False):
            return

        spot.polygon_seed_points = []
        self.show_frame(current_frame_id)

    def _classify_track_method(self, track: Track) -> str:
        """Return a short human-readable label for the dominant source in a track."""
        sources = {
            self._normalized_source_name(getattr(spot, "source", None))
            for spot in track.spots_by_image_id.values()
            if getattr(spot, "source", None)
        }
        if not sources:
            return "Unknown"

        if sources.issubset(MANUAL_TRACK_SOURCES):
            return "Manual"

        non_manual_sources = sources - MANUAL_TRACK_SOURCES
        if len(non_manual_sources) == 1:
            source_name = next(iter(non_manual_sources))
            if sources.issubset({source_name} | MANUAL_TRACK_SOURCES):
                if sources & MANUAL_TRACK_SOURCES:
                    manual_label = self._method_label_for_source_name(source_name, include_manual=True)
                    if manual_label is not None:
                        return manual_label
                method_label = self._method_label_for_source_name(source_name)
                if method_label is not None:
                    return method_label
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
        if bool(getattr(spot, "xmem2_permanent_reference", False)):
            lines.append("XMem2 permanent reference: yes")
        if self._track_supports_polygon_seed(selected_track) or getattr(spot, "has_polygon_seed", False):
            if getattr(spot, "has_polygon_seed", False):
                lines.append("Seed type: polygon")
                lines.append(f"Polygon vertices: {len(getattr(spot, 'polygon_seed_points', []))}")
            else:
                lines.append("Seed type: polygon missing")

        self._append_spot_backend_detail_lines(lines, spot)

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
        return self._build_focus_point_payload_for_spot(selected_track, spot)

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

        current_track = self.get_selected_track()
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        current_spot = (
            current_track.spots_by_image_id.get(current_frame_id)
            if current_track is not None
            else None
        )
        current_tracker_spec = self._tracker_spec_for_track(current_track)
        xmem2_reference_action = None
        if current_tracker_spec is not None and current_tracker_spec.tracker_id == "xmem2_offline":
            xmem2_reference_marked = bool(
                current_spot is not None and getattr(current_spot, "xmem2_permanent_reference", False)
            )
            xmem2_reference_action = menu.addAction(
                "Unmark as XMem2 permanent reference"
                if xmem2_reference_marked
                else "Mark as XMem2 permanent reference"
            )
        polygon_edit_action = None
        polygon_clear_action = None
        if current_track is not None and current_spot is not None and self._track_supports_polygon_seed(current_track):
            polygon_edit_action = menu.addAction("Edit polygon mask")
            polygon_clear_action = menu.addAction("Clear polygon mask")
            polygon_clear_action.setEnabled(bool(getattr(current_spot, "has_polygon_seed", False)))
        
        menu.addSeparator()
        retrack_action = menu.addAction("Re-track from this point (Optical Flow)")
        backend_retrack_action = None
        backend_retrack_text = self._backend_retrack_action_text(current_track)
        if backend_retrack_text:
            backend_retrack_action = menu.addAction(backend_retrack_text)
        
        action = menu.exec(global_pos)
        
        if action == delete_action:
            self.delete_point_from_track(spot_id)
        elif action == xmem2_reference_action:
            self.toggle_xmem2_permanent_reference(spot_id)
        elif action == polygon_edit_action:
            self._edit_polygon_mask_for_selected_spot()
        elif action == polygon_clear_action:
            self._clear_polygon_mask_for_selected_spot()
        elif action == retrack_action:
            self.initiate_retrack_optical_flow(spot_id)
        elif action == backend_retrack_action and current_tracker_spec is not None:
            self._initiate_retrack_for_tracker_spec(current_tracker_spec, spot_id)

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

    def _tracker_params_for_spec(self, tracker_spec) -> dict:
        params_attr = getattr(tracker_spec, "params_argument_name", None)
        if not params_attr:
            return {}
        params = getattr(self, params_attr, None)
        if not isinstance(params, dict):
            return {}
        return dict(params)

    def _probe_tracker_environment_for_spec(self, tracker_spec, **kwargs):
        probe_fn = getattr(tracker_spec, "probe_environment", None)
        if probe_fn is None:
            raise AttributeError(f"Tracker '{tracker_spec.tracker_id}' does not define probe_environment")
        return probe_fn(**kwargs)

    def _format_tracker_environment_report_for_spec(self, tracker_spec, status) -> str:
        format_fn = getattr(tracker_spec, "format_environment_report", None)
        if format_fn is None:
            raise AttributeError(f"Tracker '{tracker_spec.tracker_id}' does not define format_environment_report")
        return format_fn(status)

    def _tracker_unavailable_title_for_spec(self, tracker_spec, *, default: str = "Tracker Unavailable") -> str:
        tracker_id = str(getattr(tracker_spec, "tracker_id", "") or "").strip().lower()
        return TRACKER_UNAVAILABLE_TITLE_BY_ID.get(tracker_id, default)

    def _python_executable_path_ok(self, python_executable: str | None) -> bool:
        return bool(python_executable and Path(str(python_executable)).expanduser().exists())

    def _warn_tracker_unavailable(
        self,
        tracker_spec,
        status,
        *,
        title: str | None = None,
        extra_message: str = "",
    ) -> None:
        message = self._format_tracker_environment_report_for_spec(tracker_spec, status)
        if extra_message:
            message += extra_message
        QMessageBox.warning(
            self,
            title or self._tracker_unavailable_title_for_spec(tracker_spec),
            message,
        )

    def _ensure_registered_retrack_runtime_ready(
        self,
        tracker_spec,
        status,
        tracker_params: dict,
        *,
        ready_for_direct: bool,
        structural_ready: bool,
        title: str | None = None,
        env_name: str | None = None,
    ) -> bool:
        if not tracker_spec.supports_subprocess:
            if not ready_for_direct:
                self._warn_tracker_unavailable(tracker_spec, status, title=title)
                return False
            return True

        backend_mode = str(tracker_params.get("backend_mode") or "subprocess").strip().lower()
        if backend_mode == "direct":
            if not ready_for_direct:
                self._warn_tracker_unavailable(tracker_spec, status, title=title)
                return False
            return True

        if not structural_ready:
            self._warn_tracker_unavailable(tracker_spec, status, title=title)
            return False

        if ready_for_direct or self._python_executable_path_ok(tracker_params.get("python_executable")):
            return True

        extra_message = ""
        if env_name:
            extra_message = (
                f"\n\nSet the Python executable of the dedicated `{env_name}` env "
                "or run Trackleed from that environment."
            )
        self._warn_tracker_unavailable(
            tracker_spec,
            status,
            title=title,
            extra_message=extra_message,
        )
        return False

    def _launch_registered_retrack(
        self,
        tracker_spec,
        track_to_retrack: Track,
        current_frame_id: int,
        tracker_params: dict,
        *,
        title: str,
        method_label: str,
    ) -> None:
        cancel_event = Event()
        worker = GenericWorker(
            tracker_spec.run_retrack,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            tracker_params,
            cancel_event=cancel_event,
        )
        self.launch_retrack_worker(
            worker,
            title=title,
            cancel_event=cancel_event,
            method_label=method_label,
        )

    def toggle_xmem2_permanent_reference(self, spot_id: int):
        """Toggle the current spot as an extra XMem2 permanent reference."""
        target = self._get_retrack_target()
        if target is None:
            return
        track, current_frame_id = target
        spot = track.spots_by_image_id.get(current_frame_id)
        if spot is None:
            QMessageBox.warning(self, "Error", "No spot exists on the current frame.")
            return

        currently_marked = bool(getattr(spot, "xmem2_permanent_reference", False))
        setattr(spot, "xmem2_permanent_reference", not currently_marked)
        self._update_track_details()
        self.show_frame(current_frame_id)

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
        tracker_spec = require_tracker_spec("tapir_online" if use_online else "tapir_offline")

        tapir_params = self._tracker_params_for_spec(tracker_spec)
        if not tapir_params:
            QMessageBox.warning(
                self,
                "TAPIR Settings Missing",
                "TAPIR parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=tapir_params.get("tapir_models_dir"),
            checkpoint_path=tapir_params.get("checkpoint_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "TAPIR Unavailable",
                self._format_tracker_environment_report_for_spec(tracker_spec, status),
            )
            return
        if use_online and not status.selected_checkpoint_is_online:
            QMessageBox.warning(
                self,
                "Causal TAPIR Required",
                "BootsTAPIR Online re-tracking requires a causal / online TAPIR checkpoint.",
            )
            return

        tapir_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            tapir_params,
            title=(
                "Re-tracking spot with BootsTAPIR Online..."
                if use_online
                else "Re-tracking spot with TAPIR..."
            ),
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
        tracker_spec = require_tracker_spec("locotrack_offline")

        locotrack_params = self._tracker_params_for_spec(tracker_spec)
        if not locotrack_params:
            QMessageBox.warning(
                self,
                "LocoTrack Settings Missing",
                "LocoTrack parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=locotrack_params.get("locotrack_models_dir"),
            checkpoint_path=locotrack_params.get("checkpoint_path"),
            repo_path=locotrack_params.get("repo_path") or locotrack_params.get("locotrack_repo_path"),
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            locotrack_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        locotrack_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            locotrack_params,
            title="Re-tracking spot with LocoTrack...",
            method_label="LocoTrack re-tracking",
        )

    def initiate_retrack_cotracker3(self, spot_id: int):
        """Prepares and launches CoTracker3 re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("cotracker3_offline")

        cotracker_params = self._tracker_params_for_spec(tracker_spec)
        if not cotracker_params:
            QMessageBox.warning(
                self,
                "CoTracker3 Settings Missing",
                "CoTracker3 parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=cotracker_params.get("cotracker3_models_dir"),
            checkpoint_path=cotracker_params.get("checkpoint_path"),
            repo_path=cotracker_params.get("repo_path")
            or cotracker_params.get("cotracker3_repo_path")
            or cotracker_params.get("cotracker_repo_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "CoTracker3 Unavailable",
                self._format_tracker_environment_report_for_spec(tracker_spec, status),
            )
            return

        cotracker_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            cotracker_params,
            title="Re-tracking spot with CoTracker3...",
            method_label="CoTracker3 re-tracking",
        )

    def initiate_retrack_tapnext(self, spot_id: int):
        """Prepares and launches TAPNext re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("tapnext_offline")

        tapnext_params = self._tracker_params_for_spec(tracker_spec)
        if not tapnext_params:
            QMessageBox.warning(
                self,
                "TAPNext Settings Missing",
                "TAPNext parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=tapnext_params.get("tapnext_models_dir"),
            checkpoint_path=tapnext_params.get("checkpoint_path"),
            repo_path=tapnext_params.get("repo_path") or tapnext_params.get("tapnext_repo_path"),
        )
        if not status.is_ready:
            QMessageBox.warning(
                self,
                "TAPNext Unavailable",
                self._format_tracker_environment_report_for_spec(tracker_spec, status),
            )
            return

        tapnext_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            tapnext_params,
            title="Re-tracking spot with TAPNext...",
            method_label="TAPNext re-tracking",
        )

    def initiate_retrack_trackonr(self, spot_id: int):
        """Prepares and launches Track-On-R re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("trackonr_offline")

        trackonr_params = self._tracker_params_for_spec(tracker_spec)
        if not trackonr_params:
            QMessageBox.warning(
                self,
                "Track-On-R Settings Missing",
                "Track-On-R parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=trackonr_params.get("trackonr_models_dir"),
            checkpoint_path=trackonr_params.get("checkpoint_path"),
            repo_path=(
                trackonr_params.get("repo_path")
                or trackonr_params.get("trackonr_repo_path")
                or trackonr_params.get("track_on_repo_path")
            ),
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            trackonr_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        trackonr_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            trackonr_params,
            title="Re-tracking spot with Track-On-R...",
            method_label="Track-On-R re-tracking",
        )

    def initiate_retrack_pipspp(self, spot_id: int):
        """Prepares and launches PIPs++ re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("pipspp_offline")

        pipspp_params = self._tracker_params_for_spec(tracker_spec)
        if not pipspp_params:
            QMessageBox.warning(
                self,
                "PIPs++ Settings Missing",
                "PIPs++ parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=pipspp_params.get("pipspp_models_dir"),
            checkpoint_path=pipspp_params.get("checkpoint_path"),
            repo_path=pipspp_params.get("repo_path") or pipspp_params.get("pipspp_repo_path"),
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.selected_checkpoint_file is not None
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            pipspp_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        pipspp_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            pipspp_params,
            title="Re-tracking spot with PIPs++...",
            method_label="PIPs++ re-tracking",
        )

    def initiate_retrack_mftiq(self, spot_id: int):
        """Prepares and launches MFTIQ re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("mftiq_offline")

        mftiq_params = self._tracker_params_for_spec(tracker_spec)
        if not mftiq_params:
            QMessageBox.warning(
                self,
                "MFTIQ Settings Missing",
                "MFTIQ parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=mftiq_params.get("mftiq_models_dir"),
            checkpoint_path=mftiq_params.get("checkpoint_path"),
            repo_path=mftiq_params.get("repo_path") or mftiq_params.get("mftiq_repo_path"),
            config_path=mftiq_params.get("config_path"),
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.config_exists
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            mftiq_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        mftiq_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            mftiq_params,
            title="Re-tracking spot with MFTIQ...",
            method_label="MFTIQ re-tracking",
        )

    def initiate_retrack_cutie(self, spot_id: int):
        """Prepares and launches Cutie re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("cutie_offline")

        cutie_params = self._tracker_params_for_spec(tracker_spec)
        if not cutie_params:
            QMessageBox.warning(
                self,
                "Cutie Settings Missing",
                "Cutie parameters were not provided to the tracking visualizer.",
            )
            return
        start_spot = track_to_retrack.spots_by_image_id.get(current_frame_id)
        if start_spot is None or not getattr(start_spot, "has_polygon_seed", False):
            QMessageBox.warning(
                self,
                "Cutie Requires Polygon Mask",
                "Cutie re-tracking requires a polygon mask on the selected start spot.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=cutie_params.get("cutie_models_dir"),
            checkpoint_path=cutie_params.get("checkpoint_path"),
            repo_path=cutie_params.get("repo_path") or cutie_params.get("cutie_repo_path"),
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            cutie_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        cutie_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            cutie_params,
            title="Re-tracking spot with Cutie...",
            method_label="Cutie re-tracking",
        )

    def initiate_retrack_xmem2(self, spot_id: int):
        """Prepares and launches XMem2 re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("xmem2_offline")
        start_spot = track_to_retrack.spots_by_image_id.get(current_frame_id)
        if start_spot is None or not getattr(start_spot, "has_polygon_seed", False):
            QMessageBox.warning(
                self,
                "XMem2 Requires Polygon Mask",
                "XMem2 re-tracking requires a polygon mask on the selected start spot.",
            )
            return

        xmem2_params = self._tracker_params_for_spec(tracker_spec)
        if not xmem2_params:
            QMessageBox.warning(
                self,
                "XMem2 Settings Missing",
                "XMem2 parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=xmem2_params.get("xmem2_models_dir"),
            checkpoint_path=xmem2_params.get("checkpoint_path"),
            repo_path=xmem2_params.get("repo_path") or xmem2_params.get("xmem2_repo_path"),
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            xmem2_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        xmem2_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            xmem2_params,
            title="Re-tracking spot with XMem2...",
            method_label="XMem2 re-tracking",
        )

    def initiate_retrack_sam2(self, spot_id: int):
        """Prepares and launches SAM 2 re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("sam2_offline")

        sam2_params = self._tracker_params_for_spec(tracker_spec)
        if not sam2_params:
            QMessageBox.warning(
                self,
                "SAM2 Settings Missing",
                "SAM2 parameters were not provided to the tracking visualizer.",
            )
            return

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            models_dir=sam2_params.get("sam2_models_dir"),
            checkpoint_path=sam2_params.get("checkpoint_path"),
            repo_path=sam2_params.get("repo_path") or sam2_params.get("sam2_repo_path"),
            config_path=sam2_params.get("config_path"),
        )
        if not status.is_ready and sam2_params.get("config_path"):
            status = self._probe_tracker_environment_for_spec(
                tracker_spec,
                models_dir=sam2_params.get("sam2_models_dir"),
                checkpoint_path=sam2_params.get("checkpoint_path"),
                repo_path=sam2_params.get("repo_path") or sam2_params.get("sam2_repo_path"),
                config_path=None,
            )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.config_exists
            and status.config_identifier is not None
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            sam2_params,
            ready_for_direct=status.is_ready,
            structural_ready=structural_ready,
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        sam2_params["roi_size"] = self.roi_size

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            sam2_params,
            title="Re-tracking spot with SAM2...",
            method_label="SAM2 re-tracking",
        )

    def initiate_retrack_tomp(self, spot_id: int):
        """Prepares and launches ToMP / TaMOs re-tracking from the current point."""
        target = self._get_retrack_target()
        if target is None:
            return
        track_to_retrack, current_frame_id = target
        tracker_spec = require_tracker_spec("tomp_offline")

        tomp_params = self._tracker_params_for_spec(tracker_spec)
        if not tomp_params:
            QMessageBox.warning(
                self,
                "ToMP Settings Missing",
                "ToMP / TaMOs parameters were not provided to the tracking visualizer.",
            )
            return

        start_spot = track_to_retrack.spots_by_image_id.get(current_frame_id)
        if start_spot is None:
            return

        variant_value = (
            tomp_params.get("model_variant")
            or getattr(start_spot, "tomp_model_variant", None)
            or ("tamos_resnet50" if str(getattr(start_spot, "source", "")).strip().lower() == "tamos_offline" else "tomp50")
        )
        variant_key = str(variant_value or "tomp50").strip().lower()
        if variant_key in {"tamos_swin", "tamos_swin_base", "tamos-swin-base", "tamos_swinbase"}:
            model_variant = "tamos_swin_base"
            variant_label = "TaMOs"
        elif variant_key in {"tamos", "tamos_resnet50", "tamos-resnet50", "tamos_50"}:
            model_variant = "tamos_resnet50"
            variant_label = "TaMOs"
        elif variant_key in {"tomp101", "tomp_101"}:
            model_variant = "tomp101"
            variant_label = "ToMP"
        else:
            model_variant = "tomp50"
            variant_label = "ToMP"

        status = self._probe_tracker_environment_for_spec(
            tracker_spec,
            tomp_models_dir=tomp_params.get("tomp_models_dir"),
            tamos_models_dir=tomp_params.get("tamos_models_dir"),
            tomp_checkpoint_path=(
                tomp_params.get("checkpoint_path")
                if model_variant not in {"tamos_resnet50", "tamos_swin_base"}
                else None
            ),
            tamos_checkpoint_path=(
                tomp_params.get("checkpoint_path")
                if model_variant in {"tamos_resnet50", "tamos_swin_base"}
                else None
            ),
            repo_path=(
                tomp_params.get("repo_path")
                or tomp_params.get("tomp_repo_path")
                or tomp_params.get("pytracking_repo_path")
            ),
        )

        if model_variant in {"tamos_resnet50", "tamos_swin_base"}:
            variant_ready = status.tamos_ready
            selected_checkpoint_obj = select_default_tamos_checkpoint_for_variant(
                status.available_tamos_checkpoints,
                model_variant,
            )
        else:
            variant_ready = status.tomp_ready
            selected_checkpoint_obj = select_default_tomp_checkpoint_for_variant(
                status.available_tomp_checkpoints,
                model_variant,
            )
        structural_ready = bool(
            status.repo_path is not None
            and selected_checkpoint_obj is not None
            and selected_checkpoint_obj.exists()
        )
        if not self._ensure_registered_retrack_runtime_ready(
            tracker_spec,
            status,
            tomp_params,
            ready_for_direct=variant_ready,
            structural_ready=structural_ready,
            title=f"{variant_label} Unavailable",
            env_name=TRACKER_ENV_HINT_BY_ID.get(tracker_spec.tracker_id),
        ):
            return

        tomp_params["roi_size"] = self.roi_size
        tomp_params["model_variant"] = model_variant

        self._launch_registered_retrack(
            tracker_spec,
            track_to_retrack,
            current_frame_id,
            tomp_params,
            title=f"Re-tracking spot with {variant_label}...",
            method_label=f"{variant_label} re-tracking",
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
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
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
            self.image_viewer.update_polygon_overlay([])
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
        self.image_viewer.update_polygon_overlay(
            self._build_polygon_seed_overlay_for_frame(current_frame_id)
        )
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
        self.image_viewer.update_polygon_overlay(
            self._build_polygon_seed_overlay_for_frame(frame_id)
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
