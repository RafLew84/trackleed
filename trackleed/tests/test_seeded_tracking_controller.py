from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QDialog

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_tracking_adapter import SeededTrackingDialogSettings
from trackleed.gui.seeded_tracking_controller import SeededTrackingController
from trackleed.gui.seeded_tracking_dialog import SeededTrackingDialogDecision
from trackleed.logic.workflows.point_tracking import TapirPointConfig
from trackleed.logic.workflows.seeded_tracking import (
    OpticalFlowConfig,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingResult,
    TrackingSeed,
    TrackObservation,
)


def _frame(frame_id: int, *, seed: bool = False) -> LeedFile:
    spots = [SpotROI(7, QRect(12, 12, 8, 8), 16.0, 16.0, 1.0, "manual")] if seed else []
    return LeedFile(
        Path(f"frame-{frame_id}.tif"),
        f"frame-{frame_id}.tif",
        False,
        "Cu",
        111,
        "G",
        300.0,
        80.0,
        frame_id,
        None,
        spots,
    )


def _decision(files: tuple[LeedFile, ...]) -> SeededTrackingDialogDecision:
    settings = SeededTrackingDialogSettings.defaults(roi_size=18)
    request = SeededTrackingRequest(
        method_id="optical_flow",
        frames=tuple(
            SeededTrackingFrame(int(file.id), file.filepath) for file in files
        ),
        reference_frame_id=1,
        seeds=(TrackingSeed(7, 16.0, 16.0, SeedRect(12, 12, 8, 8)),),
        config=OpticalFlowConfig(roi_size=18),
    )
    return SeededTrackingDialogDecision(request, files, settings)


class _Dialog:
    decision: SeededTrackingDialogDecision
    result = QDialog.DialogCode.Accepted
    last_kwargs: dict[str, object]

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    def exec(self) -> QDialog.DialogCode:
        return self.result


class _Worker:
    def __init__(self, function: object, *args: object) -> None:
        self.function = function
        self.args = args


class _Notifications:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def information(self, title: str, message: str) -> None:
        self.messages.append((title, message))

    def error(self, title: str, message: str) -> None:
        self.messages.append((title, message))


class _Item:
    def __init__(self, frame: LeedFile) -> None:
        self._frame = frame

    def data(self, _role: object) -> LeedFile:
        return self._frame


class _Host:
    def __init__(self, files: tuple[LeedFile, ...]) -> None:
        self.data_files = list(files)
        self.background_file = None
        self.file_list_widget = SimpleNamespace(currentItem=lambda: _Item(files[0]))
        self.tracking_results = [Track(99, {})]
        self.review_calls = 0
        self.launch: dict[str, object] = {}

    def _tracking_worker_is_active(self) -> bool:
        return False

    def _launch_worker_with_progress(
        self, worker: object, title: str, cancel_event: object, **callbacks: object
    ) -> None:
        self.launch = {
            "worker": worker,
            "title": title,
            "cancel": cancel_event,
            **callbacks,
        }

    def show_tracking_visualizer(self) -> None:
        self.review_calls += 1


class SeededTrackingControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.files = (_frame(1, seed=True), _frame(2))
        self.host = _Host(self.files)
        _Dialog.decision = _decision(self.files)
        _Dialog.result = QDialog.DialogCode.Accepted
        self.controller = SeededTrackingController(
            self.host,
            roi_size=32,
            dialog_factory=_Dialog,
            worker_factory=_Worker,
            notifications=_Notifications(),
        )

    def test_accept_remembers_settings_and_starts_shared_worker_host(self) -> None:
        self.controller.show_dialog()

        self.assertEqual(
            self.controller.settings.dialog_settings.optical_flow.roi_size, 18
        )
        self.assertEqual(self.host.launch["title"], "Running seeded tracking...")
        worker = self.host.launch["worker"]
        self.assertEqual(worker.function.__name__, "run_seeded_tracking_job")
        self.assertEqual(self.host.tracking_results[0].track_id, 99)

    def test_error_and_cancel_keep_previous_complete_result(self) -> None:
        self.controller.show_dialog()
        self.host.launch["on_error"]("boom")
        self.assertEqual(self.host.tracking_results[0].track_id, 99)

        self.host.launch["on_finished"](None)
        self.assertEqual(self.host.tracking_results[0].track_id, 99)
        self.assertEqual(self.host.review_calls, 0)

    def test_success_publishes_atomically_then_opens_review(self) -> None:
        self.controller.show_dialog()
        result = TrackingResult(
            (
                TrackedSpotIdentity(
                    7,
                    (
                        TrackObservation(1, SeedRect(12, 12, 8, 8), 16.0, 16.0),
                        TrackObservation(2, SeedRect(14, 13, 8, 8), 18.0, 17.0),
                    ),
                ),
            )
        )

        self.host.launch["on_finished"](result)

        self.assertEqual(self.host.tracking_results[0].track_id, 7)
        self.assertEqual(self.files[1].spots[0].center_x, 18.0)
        self.assertEqual(self.host.review_calls, 1)

    def test_legacy_optical_choice_only_becomes_memory_after_accept(self) -> None:
        _Dialog.result = QDialog.DialogCode.Rejected

        handled = self.controller.show_legacy_scenario("Optical Flow (skip, anchor)")

        self.assertTrue(handled)
        shown_settings = _Dialog.last_kwargs["settings"]
        self.assertEqual(shown_settings.selected_method_id, "optical_flow_skip_anchor")
        self.assertEqual(
            self.controller.settings.dialog_settings.selected_method_id,
            "optical_flow",
        )
        self.assertFalse(self.controller.show_legacy_scenario("Offline TAPIR"))

    def test_review_defaults_come_from_last_accepted_method_config(self) -> None:
        self.controller.show_dialog()

        defaults = self.controller.review_defaults

        self.assertEqual(defaults.roi_size, 18)
        self.assertEqual(defaults.lk_win_size, 41)
        self.assertEqual(defaults.lk_max_level, 4)

    def test_dialog_series_preserves_legacy_frame_id_ordering(self) -> None:
        self.host.data_files.reverse()

        self.controller.show_dialog()

        shown_files = _Dialog.last_kwargs["series_files"]
        self.assertEqual([file.id for file in shown_files], [1, 2])

    def test_point_review_uses_the_last_accepted_point_configuration(self) -> None:
        settings = SeededTrackingDialogSettings.defaults(roi_size=18)
        settings = settings.with_point_config(
            "tapir_offline",
            TapirPointConfig(models_dir=Path("custom-models"), roi_size=26),
        )
        self.controller.settings.accept(
            replace(settings, selected_method_id="tapir_offline")
        )

        defaults = self.controller.review_defaults
        backend_params = self.controller.review_backend_params

        self.assertEqual(defaults.roi_size, 26)
        self.assertEqual(backend_params["tapir_params"]["roi_size"], 26)
        self.assertEqual(
            backend_params["tapir_params"]["tapir_models_dir"],
            "custom-models",
        )


if __name__ == "__main__":
    unittest.main()
