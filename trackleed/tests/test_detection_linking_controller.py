from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QDialog

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.detection_linking_adapter import DetectionLinkingDialogSettings
from trackleed.gui.detection_linking_controller import DetectionLinkingController
from trackleed.gui.detection_linking_dialog import DetectionLinkingDialogDecision
from trackleed.logic.workflows.detection_linking import (
    DirectionalLinkingConfig,
    HungarianLinkingConfig,
)
from trackleed.logic.workflows.seeded_tracking import (
    SeedRect,
    TrackObservation,
    TrackedSpotIdentity,
    TrackingResult,
)


def _frame(frame_id: int, x: float) -> LeedFile:
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
        [SpotROI(frame_id, QRect(round(x) - 4, 8, 8, 8), x, 12.0, 5.0, "auto")],
    )


class _Dialog:
    result = QDialog.DialogCode.Accepted
    decision: DetectionLinkingDialogDecision
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


class _Host:
    def __init__(self, files: tuple[LeedFile, ...]) -> None:
        self.data_files = list(files)
        self.tracking_results = [Track(99, {})]
        self.file_list_widget = SimpleNamespace(currentItem=lambda: None)
        self.launch: dict[str, object] = {}
        self.review_calls = 0

    def _tracking_worker_is_active(self) -> bool:
        return False

    def _launch_worker_with_progress(
        self,
        worker: object,
        title: str,
        cancel_event: object,
        **callbacks: object,
    ) -> None:
        self.launch = {
            "worker": worker,
            "title": title,
            "cancel": cancel_event,
            **callbacks,
        }

    def show_tracking_visualizer(self) -> None:
        self.review_calls += 1


class DetectionLinkingControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.files = (_frame(1, 10.0), _frame(2, 12.0))
        config = HungarianLinkingConfig(max_distance_px=8.0)
        from trackleed.gui.detection_linking_adapter import build_detection_linking_request

        _Dialog.decision = DetectionLinkingDialogDecision(
            build_detection_linking_request(self.files, config),
            self.files,
            DetectionLinkingDialogSettings(config),
        )
        _Dialog.result = QDialog.DialogCode.Accepted
        self.host = _Host(self.files)
        self.notifications = _Notifications()
        self.controller = DetectionLinkingController(
            self.host,
            dialog_factory=_Dialog,
            worker_factory=_Worker,
            notifications=self.notifications,
        )

    def test_accept_remembers_config_and_starts_shared_worker(self) -> None:
        self.controller.show_hungarian_dialog()

        self.assertEqual(
            self.controller.settings.dialog_settings.hungarian.max_distance_px,
            8.0,
        )
        self.assertEqual(self.host.launch["title"], "Running Hungarian linking...")
        worker = self.host.launch["worker"]
        self.assertEqual(worker.function.__name__, "run_detection_linking_job")
        self.assertEqual(self.host.tracking_results[0].track_id, 99)

    def test_directional_uses_same_controller_worker_and_settings_binding(self) -> None:
        from trackleed.gui.detection_linking_adapter import build_detection_linking_request

        config = DirectionalLinkingConfig(
            center_mode="manual",
            manual_center=(50.0, 50.0),
            direction_mode="outward",
        )
        _Dialog.decision = DetectionLinkingDialogDecision(
            build_detection_linking_request(
                self.files,
                config,
                method_id="directional",
            ),
            self.files,
            DetectionLinkingDialogSettings(directional=config),
        )
        controller = DetectionLinkingController(
            self.host,
            dialog_factory=_Dialog,
            directional_dialog_factory=_Dialog,
            worker_factory=_Worker,
            notifications=self.notifications,
        )

        controller.show_directional_dialog()

        self.assertEqual(
            controller.settings.dialog_settings.directional.direction_mode,
            "outward",
        )
        self.assertEqual(self.host.launch["title"], "Running Directional linking...")
        worker = self.host.launch["worker"]
        self.assertEqual(worker.function.__name__, "run_detection_linking_job")
        self.host.launch["on_finished"](
            TrackingResult(
                (
                    TrackedSpotIdentity(
                        1,
                        (
                            TrackObservation(1, SeedRect(6, 8, 8, 8), 10.0, 12.0),
                            TrackObservation(2, SeedRect(8, 8, 8, 8), 12.0, 12.0),
                        ),
                    ),
                )
            )
        )
        self.assertEqual(self.host.review_calls, 1)
        self.assertTrue(
            controller.show_legacy_scenario("Detection Linking (Directional)")
        )

    def test_directional_cancel_does_not_replace_accepted_settings(self) -> None:
        _Dialog.result = QDialog.DialogCode.Rejected
        controller = DetectionLinkingController(
            self.host,
            dialog_factory=_Dialog,
            directional_dialog_factory=_Dialog,
            worker_factory=_Worker,
            notifications=self.notifications,
        )

        controller.show_directional_dialog()

        self.assertEqual(
            controller.settings.dialog_settings.directional,
            DirectionalLinkingConfig(),
        )
        self.assertEqual(self.host.launch, {})

    def test_cancelled_dialog_does_not_change_remembered_config(self) -> None:
        _Dialog.result = QDialog.DialogCode.Rejected

        self.controller.show_hungarian_dialog()

        self.assertEqual(
            self.controller.settings.dialog_settings.hungarian.max_distance_px,
            20.0,
        )
        self.assertEqual(self.host.launch, {})

    def test_cancelled_dialog_does_not_repair_duplicate_frame_ids(self) -> None:
        _Dialog.result = QDialog.DialogCode.Rejected
        self.files[0].id = 0
        self.files[1].id = 0

        self.controller.show_hungarian_dialog()

        shown_files = _Dialog.last_kwargs["files"]
        self.assertEqual(tuple(file.id for file in shown_files), (0, 0))

    def test_failure_and_cancellation_keep_previous_complete_tracks(self) -> None:
        self.controller.show_hungarian_dialog()

        self.host.launch["on_error"]("boom")
        self.host.launch["on_finished"](None)

        self.assertEqual(self.host.tracking_results[0].track_id, 99)
        self.assertEqual(self.host.review_calls, 0)

    def test_success_publishes_complete_result_then_opens_review(self) -> None:
        self.controller.show_hungarian_dialog()
        result = TrackingResult(
            (
                TrackedSpotIdentity(
                    1,
                    (
                        TrackObservation(1, SeedRect(6, 8, 8, 8), 10.0, 12.0),
                        TrackObservation(2, SeedRect(8, 8, 8, 8), 12.0, 12.0),
                    ),
                ),
            )
        )

        self.host.launch["on_finished"](result)

        self.assertEqual(self.host.tracking_results[0].track_id, 1)
        self.assertIsNot(self.files[0].spots[0], self.host.tracking_results[0].spots_by_image_id[1])
        self.assertEqual(self.host.review_calls, 1)

    def test_success_preserves_fits_for_unchanged_observation_rois(self) -> None:
        previous = Track(
            1,
            {
                1: SpotROI(1, QRect(6, 8, 8, 8), 10.0, 12.0, 5.0, "tracking"),
                2: SpotROI(1, QRect(8, 8, 8, 8), 12.0, 12.0, 5.0, "tracking"),
            },
        )
        previous.analysis_results = {1: {"radius": 4.0}, 2: {"radius": 5.0}}
        self.host.tracking_results = [previous]
        self.controller.show_hungarian_dialog()
        result = TrackingResult(
            (
                TrackedSpotIdentity(
                    1,
                    (
                        TrackObservation(1, SeedRect(6, 8, 8, 8), 10.0, 12.0),
                        TrackObservation(2, SeedRect(8, 8, 8, 8), 12.0, 12.0),
                    ),
                ),
            )
        )

        self.host.launch["on_finished"](result)

        self.assertEqual(
            self.host.tracking_results[0].analysis_results,
            previous.analysis_results,
        )

    def test_invalid_result_keeps_previous_tracks_and_frame_detections(self) -> None:
        self.controller.show_hungarian_dialog()
        original_spots = tuple(file.spots[0] for file in self.files)
        invalid_result = TrackingResult(
            (
                TrackedSpotIdentity(
                    1,
                    (TrackObservation(999, SeedRect(6, 8, 8, 8), 10.0, 12.0),),
                ),
            )
        )

        self.host.launch["on_finished"](invalid_result)

        self.assertEqual(self.host.tracking_results[0].track_id, 99)
        self.assertEqual(tuple(file.spots[0] for file in self.files), original_spots)
        self.assertEqual(self.host.review_calls, 0)


if __name__ == "__main__":
    unittest.main()
