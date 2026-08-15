"""GUI orchestration for the headless detection-linking workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Any, Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QMessageBox

from trackleed.gui.detection_linking_adapter import (
    DetectionLinkingSettingsBinding,
    publish_detection_linking_result,
    run_detection_linking_job,
)
from trackleed.gui.detection_linking_dialog import (
    DetectionLinkingDialog,
    DetectionLinkingDialogDecision,
    DirectionalDetectionLinkingDialog,
)
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.workflows.detection_linking import (
    default_detection_linking_catalog,
)
from trackleed.logic.workflows.seeded_tracking import TrackingResult


class DetectionLinkingNotificationSink(Protocol):
    def information(self, title: str, message: str) -> None: ...

    def error(self, title: str, message: str) -> None: ...


class QtDetectionLinkingNotifications:
    def __init__(self, parent: Any) -> None:
        self._parent = parent

    def information(self, title: str, message: str) -> None:
        QMessageBox.information(self._parent, title, message)

    def error(self, title: str, message: str) -> None:
        QMessageBox.critical(self._parent, title, message)


class DetectionLinkingCompletionKind(Enum):
    SUCCEEDED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class DetectionLinkingCompletion:
    kind: DetectionLinkingCompletionKind
    result: TrackingResult | None = None
    error_message: str | None = None

    @classmethod
    def from_worker_result(
        cls,
        result: TrackingResult | None,
    ) -> DetectionLinkingCompletion:
        kind = (
            DetectionLinkingCompletionKind.SUCCEEDED
            if result is not None
            else DetectionLinkingCompletionKind.CANCELLED
        )
        return cls(kind, result=result)

    @classmethod
    def from_worker_error(cls, message: str) -> DetectionLinkingCompletion:
        return cls(DetectionLinkingCompletionKind.FAILED, error_message=message)


class DetectionLinkingController:
    """Bind catalog linking dialogs, one worker host, and atomic publication."""

    def __init__(
        self,
        host: Any,
        *,
        dialog_factory: Callable[..., Any] = DetectionLinkingDialog,
        directional_dialog_factory: Callable[..., Any] = (
            DirectionalDetectionLinkingDialog
        ),
        worker_factory: Callable[..., Any] = GenericWorker,
        notifications: DetectionLinkingNotificationSink | None = None,
    ) -> None:
        self._host = host
        self._dialog_factory = dialog_factory
        self._directional_dialog_factory = directional_dialog_factory
        self._worker_factory = worker_factory
        self._notifications = notifications or QtDetectionLinkingNotifications(host)
        self.settings = DetectionLinkingSettingsBinding()

    def show_hungarian_dialog(self) -> None:
        self._show_dialog(self._dialog_factory)

    def show_directional_dialog(self) -> None:
        self._show_dialog(self._directional_dialog_factory)

    def _show_dialog(self, dialog_factory: Callable[..., Any]) -> None:
        if self._host._tracking_worker_is_active():
            self._notifications.information(
                "Busy",
                "Wait for the active operation to finish.",
            )
            return
        files = tuple(
            sorted(
                (file for file in self._host.data_files if not file.is_background),
                key=lambda file: (
                    file.id is None,
                    int(file.id) if file.id is not None else 0,
                ),
            )
        )
        dialog = dialog_factory(
            files=files,
            settings=self.settings.dialog_settings,
            parent=self._host,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.decision is None:
            return
        self.settings.accept(dialog.decision.settings)
        self._start(dialog.decision)

    def show_legacy_scenario(self, scenario: str) -> bool:
        if scenario == "Detection Linking (Hungarian)":
            self.show_hungarian_dialog()
            return True
        if scenario == "Detection Linking (Directional)":
            self.show_directional_dialog()
            return True
        return False

    def _start(self, decision: DetectionLinkingDialogDecision) -> None:
        if self._host._tracking_worker_is_active():
            self._notifications.information(
                "Busy",
                "Wait for the active operation to finish.",
            )
            return
        cancel_event = Event()
        worker = self._worker_factory(
            run_detection_linking_job,
            decision.request,
            cancel_event,
        )
        display_name = _method_display_name(decision)
        self._host._launch_worker_with_progress(
            worker,
            f"Running {display_name} linking...",
            cancel_event,
            on_finished=lambda result: self._complete(
                DetectionLinkingCompletion.from_worker_result(result),
                decision,
            ),
            on_error=lambda message: self._complete(
                DetectionLinkingCompletion.from_worker_error(message),
                decision,
            ),
            cancel_label="Detection Linking",
        )

    def _complete(
        self,
        completion: DetectionLinkingCompletion,
        decision: DetectionLinkingDialogDecision,
    ) -> None:
        if completion.kind is DetectionLinkingCompletionKind.FAILED:
            display_name = _method_display_name(decision)
            self._notifications.error(
                "Detection Linking Error",
                f"{display_name} linking failed:\n{completion.error_message}",
            )
            return
        if completion.kind is DetectionLinkingCompletionKind.CANCELLED:
            self._notifications.information(
                "Detection Linking Cancelled",
                "Existing tracks were not changed.",
            )
            return
        if completion.result is None:
            raise RuntimeError("Successful detection linking has no result")
        try:
            tracks = publish_detection_linking_result(
                decision.files,
                completion.result,
                previous_tracks=self._host.tracking_results,
            )
        except (TypeError, ValueError) as exc:
            self._notifications.error("Detection Linking Error", str(exc))
            return
        self._host.tracking_results = tracks
        self._refresh_host(decision)
        self._host.show_tracking_visualizer()

    def _refresh_host(self, decision: DetectionLinkingDialogDecision) -> None:
        for name in ("visualize_button", "analysis_button", "positions_analysis_button"):
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


def _method_display_name(decision: DetectionLinkingDialogDecision) -> str:
    return default_detection_linking_catalog().method(
        decision.request.method_id
    ).display_name
