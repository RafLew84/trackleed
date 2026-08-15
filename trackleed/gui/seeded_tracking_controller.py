"""GUI orchestration for the headless seeded-tracking workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QMessageBox

from trackleed.analysis.seeded_bbox_tracking_runtime import (
    bbox_tracking_definitions,
)
from trackleed.analysis.seeded_point_tracking_runtime import (
    point_tracking_definitions,
)
from trackleed.analysis.seeded_mask_tracking_runtime import (
    mask_tracking_definitions,
)
from trackleed.gui.seeded_bbox_tracking_compatibility import (
    bbox_tracker_legacy_binding,
    legacy_bbox_tracker_params,
)
from trackleed.gui.seeded_point_tracking_compatibility import (
    legacy_point_tracker_params,
    point_tracker_legacy_binding,
)
from trackleed.gui.seeded_mask_tracking_compatibility import (
    legacy_mask_tracker_params,
    mask_tracker_legacy_binding,
)
from trackleed.gui.seeded_tracking_adapter import (
    SeededTrackingDialogSettings,
    SeededTrackingSettingsBinding,
    publish_tracking_result,
    run_seeded_tracking_job,
)
from trackleed.gui.seeded_tracking_dialog import (
    SeededTrackingDialog,
    SeededTrackingDialogDecision,
)
from trackleed.io.image_reader import load_tiff
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.workflows.seeded_tracking import TrackingResult


class TrackingNotificationSink(Protocol):
    def information(self, title: str, message: str) -> None: ...

    def error(self, title: str, message: str) -> None: ...


class QtTrackingNotifications:
    def __init__(self, parent: Any) -> None:
        self._parent = parent

    def information(self, title: str, message: str) -> None:
        QMessageBox.information(self._parent, title, message)

    def error(self, title: str, message: str) -> None:
        QMessageBox.critical(self._parent, title, message)


class SeededTrackingCompletionKind(Enum):
    SUCCEEDED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class SeededTrackingCompletion:
    kind: SeededTrackingCompletionKind
    result: TrackingResult | None = None
    error_message: str | None = None

    @classmethod
    def from_worker_result(
        cls,
        result: TrackingResult | None,
    ) -> SeededTrackingCompletion:
        kind = (
            SeededTrackingCompletionKind.SUCCEEDED
            if result is not None
            else SeededTrackingCompletionKind.CANCELLED
        )
        return cls(kind, result=result)

    @classmethod
    def from_worker_error(cls, message: str) -> SeededTrackingCompletion:
        return cls(SeededTrackingCompletionKind.FAILED, error_message=message)


@dataclass(frozen=True)
class SeededTrackingReviewDefaults:
    roi_size: int
    lk_win_size: int
    lk_max_level: int


class SeededTrackingController:
    """Bind dialog, worker hosting, and atomic result publication."""

    def __init__(
        self,
        host: Any,
        *,
        roi_size: int,
        dialog_factory: Callable[..., Any] = SeededTrackingDialog,
        worker_factory: Callable[..., Any] = GenericWorker,
        image_reader: Callable[[Path], np.ndarray] = load_tiff,
        notifications: TrackingNotificationSink | None = None,
    ) -> None:
        self._host = host
        self._dialog_factory = dialog_factory
        self._worker_factory = worker_factory
        self._image_reader = image_reader
        self._notifications = notifications or QtTrackingNotifications(host)
        self.settings = SeededTrackingSettingsBinding(roi_size=roi_size)

    @property
    def review_defaults(self) -> SeededTrackingReviewDefaults:
        settings = self.settings.dialog_settings
        point_configs = dict(settings.point_configs)
        tracker_configs = (
            point_configs | dict(settings.bbox_configs) | dict(settings.mask_configs)
        )
        if settings.selected_method_id in tracker_configs:
            roi_size = tracker_configs[settings.selected_method_id].roi_size
            config = settings.optical_flow
        else:
            config = (
                settings.skip_anchor
                if settings.selected_method_id == "optical_flow_skip_anchor"
                else settings.optical_flow
            )
            roi_size = config.roi_size
        return SeededTrackingReviewDefaults(
            roi_size=roi_size,
            lk_win_size=config.win_size,
            lk_max_level=config.max_level,
        )

    @property
    def review_backend_params(self) -> dict[str, object]:
        settings = self.settings.dialog_settings
        point_configs = dict(settings.point_configs)
        bbox_configs = dict(settings.bbox_configs)
        mask_configs = dict(settings.mask_configs)
        params: dict[str, object] = {
            "seeded_tracking_configs": point_configs | bbox_configs | mask_configs,
        }
        point_config = point_configs.get(settings.selected_method_id)
        if point_config is not None:
            binding = point_tracker_legacy_binding(settings.selected_method_id)
            params[binding.review_argument_name] = legacy_point_tracker_params(
                settings.selected_method_id,
                point_config,
            )
            return params
        bbox_config = bbox_configs.get(settings.selected_method_id)
        if bbox_config is not None:
            bbox_binding = bbox_tracker_legacy_binding(settings.selected_method_id)
            params[bbox_binding.review_argument_name] = legacy_bbox_tracker_params(
                settings.selected_method_id,
                bbox_config,
            )
            return params
        mask_config = mask_configs.get(settings.selected_method_id)
        if mask_config is not None:
            mask_binding = mask_tracker_legacy_binding(settings.selected_method_id)
            params[mask_binding.review_argument_name] = legacy_mask_tracker_params(
                settings.selected_method_id,
                mask_config,
            )
        return params

    def show_dialog(self) -> None:
        self._show_dialog(self.settings.dialog_settings)

    def show_legacy_scenario(self, scenario: str) -> bool:
        """Route legacy seeded selectors through the canonical dialog."""
        method_id = {
            "Optical Flow Tracking": "optical_flow",
            "Optical Flow (skip, anchor)": "optical_flow_skip_anchor",
        }.get(scenario)
        if method_id is None:
            method_id = next(
                (
                    item.method_id
                    for item in point_tracking_definitions()
                    if item.display_name == scenario
                ),
                None,
            )
        if method_id is None:
            method_id = next(
                (
                    item.method_id
                    for item in mask_tracking_definitions()
                    if item.display_name == scenario
                ),
                None,
            )
        if method_id is None:
            method_id = next(
                (
                    item.method_id
                    for item in bbox_tracking_definitions()
                    if item.display_name == scenario
                ),
                None,
            )
        if method_id is None:
            return False
        self._show_dialog(
            replace(self.settings.dialog_settings, selected_method_id=method_id)
        )
        return True

    def _show_dialog(self, settings: SeededTrackingDialogSettings) -> None:
        if self._host._tracking_worker_is_active():
            self._notifications.information(
                "Busy",
                "Wait for the active operation to finish.",
            )
            return
        current_item = self._host.file_list_widget.currentItem()
        reference_file = (
            current_item.data(Qt.ItemDataRole.UserRole) if current_item else None
        )
        files = tuple(
            sorted(
                (file for file in self._host.data_files if not file.is_background),
                key=lambda file: (
                    file.id is None,
                    int(file.id) if file.id is not None else 0,
                ),
            )
        )
        dialog = self._dialog_factory(
            reference_file=reference_file,
            series_files=files,
            background_file=self._host.background_file,
            settings=settings,
            parent=self._host,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.decision is None:
            return
        self.settings.accept(dialog.decision.settings)
        self._start(dialog.decision)

    def _start(self, decision: SeededTrackingDialogDecision) -> None:
        if self._host._tracking_worker_is_active():
            self._notifications.information(
                "Busy",
                "Wait for the active operation to finish.",
            )
            return
        cancel_event = Event()
        worker = self._worker_factory(
            run_seeded_tracking_job,
            decision.request,
            self._image_reader,
            cancel_event,
        )
        self._host._launch_worker_with_progress(
            worker,
            "Running seeded tracking...",
            cancel_event,
            on_finished=lambda result: self._complete(
                SeededTrackingCompletion.from_worker_result(result),
                decision,
            ),
            on_error=lambda message: self._complete(
                SeededTrackingCompletion.from_worker_error(message),
                decision,
            ),
            cancel_label="Seeded Tracking",
        )

    def _complete(
        self,
        completion: SeededTrackingCompletion,
        decision: SeededTrackingDialogDecision,
    ) -> None:
        if completion.kind is SeededTrackingCompletionKind.FAILED:
            self._notifications.error(
                "Tracking Error",
                "An error occurred during seeded tracking:\n"
                f"{completion.error_message}",
            )
            return
        if completion.kind is SeededTrackingCompletionKind.CANCELLED:
            self._notifications.information(
                "Tracking Cancelled",
                "Seeded tracking was cancelled. Existing tracks were not changed.",
            )
            return
        if completion.result is None:
            raise RuntimeError("Successful seeded tracking has no result")
        try:
            tracks = publish_tracking_result(decision.files, completion.result)
        except (TypeError, ValueError) as exc:
            self._notifications.error("Tracking Error", str(exc))
            return
        self._host.tracking_results = tracks
        self._refresh_host_after_publication(decision)
        self._host.show_tracking_visualizer()

    def _refresh_host_after_publication(
        self,
        decision: SeededTrackingDialogDecision,
    ) -> None:
        for name in (
            "visualize_button",
            "analysis_button",
            "positions_analysis_button",
        ):
            button = getattr(self._host, name, None)
            if button is not None:
                button.setEnabled(True)
        current_item = self._host.file_list_widget.currentItem()
        if current_item is None:
            return
        current_file = current_item.data(Qt.ItemDataRole.UserRole)
        if current_file not in decision.files:
            return
        if hasattr(self._host, "populate_spot_list"):
            self._host.populate_spot_list(current_file.spots)
        if hasattr(self._host, "refresh_current_view"):
            self._host.refresh_current_view()
