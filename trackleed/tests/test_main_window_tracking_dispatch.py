from __future__ import annotations

import types
import unittest
from types import SimpleNamespace

from trackleed.tests.test_main_window_registry_dispatch import (
    MainWindowImportMixin,
    _DummyCombo,
)


class MainWindowTrackingDispatchTests(MainWindowImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.registry, self.main_window = self.import_main_window()

    def test_run_tracking_dispatches_registered_tracker_via_registry_worker_builder(
        self,
    ) -> None:
        recorded: dict[str, object] = {}
        fake = self._tracking_host("Offline TAPIR")

        def _build_registered_tracking_worker(tracker_spec, cancel_event):
            recorded["tracker_spec"] = tracker_spec
            recorded["cancel_event"] = cancel_event
            return "worker-object"

        def _launch_worker_with_progress(worker, title, cancel_event):
            recorded["worker"] = worker
            recorded["title"] = title
            recorded["launch_cancel_event"] = cancel_event

        fake._build_registered_tracking_worker = _build_registered_tracking_worker
        fake._launch_worker_with_progress = _launch_worker_with_progress

        self.main_window.MainWindow.run_tracking(fake)

        self.assertIs(
            recorded["tracker_spec"],
            self.registry.require_tracker_spec("tapir_offline"),
        )
        self.assertEqual(recorded["worker"], "worker-object")
        self.assertEqual(recorded["title"], "Running tracking...")
        self.assertIs(recorded["cancel_event"], recorded["launch_cancel_event"])

    def test_run_tracking_routes_legacy_optical_choice_to_seeded_controller(
        self,
    ) -> None:
        calls: list[str] = []
        fake = self._tracking_host(
            "Optical Flow Tracking",
            show_legacy=lambda scenario: calls.append(scenario) or True,
        )

        self.main_window.MainWindow.run_tracking(fake)

        self.assertEqual(calls, ["Optical Flow Tracking"])

    def test_run_tracking_stops_when_registered_worker_builder_returns_none(
        self,
    ) -> None:
        recorded: dict[str, object] = {}
        fake = self._tracking_host("Offline TAPIR")
        fake._build_registered_tracking_worker = lambda tracker_spec, cancel_event: None
        fake._launch_worker_with_progress = lambda *args, **kwargs: recorded.update(
            {"launched": True}
        )

        self.main_window.MainWindow.run_tracking(fake)

        self.assertNotIn("launched", recorded)

    def test_detection_linking_repairs_duplicate_frame_ids_before_dispatch(
        self,
    ) -> None:
        recorded: dict[str, object] = {}
        original_generic_worker = self.main_window.GenericWorker
        original_repair_duplicate_frame_ids = (
            self.main_window.repair_duplicate_frame_ids
        )
        self.addCleanup(
            setattr, self.main_window, "GenericWorker", original_generic_worker
        )
        self.addCleanup(
            setattr,
            self.main_window,
            "repair_duplicate_frame_ids",
            original_repair_duplicate_frame_ids,
        )

        class _GenericWorker:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        def _repair_duplicate_frame_ids(files):
            files[0].id = 14
            files[1].id = 15
            return "temperature"

        self.main_window.GenericWorker = _GenericWorker
        self.main_window.repair_duplicate_frame_ids = _repair_duplicate_frame_ids
        fake = self._tracking_host("Detection Linking (Hungarian)")
        fake.data_files = [
            SimpleNamespace(
                id=0, filename="a_T0014_E1100_C0000.tiff", spots=[object()]
            ),
            SimpleNamespace(
                id=0, filename="b_T0015_E1100_C0000.tiff", spots=[object()]
            ),
        ]
        fake.hung_max_dist_spinbox = SimpleNamespace(value=lambda: 20.0)
        fake.hung_max_gap_spinbox = SimpleNamespace(value=lambda: 2)
        fake.hung_min_len_spinbox = SimpleNamespace(value=lambda: 2)
        fake._build_detection_linking_file_map = types.MethodType(
            self.main_window.MainWindow._build_detection_linking_file_map,
            fake,
        )
        fake._launch_worker_with_progress = (
            lambda worker, title, cancel_event: recorded.update(
                {"worker": worker, "title": title, "cancel_event": cancel_event}
            )
        )

        self.main_window.MainWindow.run_tracking(fake)

        worker = recorded["worker"]
        self.assertEqual(recorded["title"], "Running tracking...")
        self.assertEqual(worker.args[2], [14, 15])
        self.assertEqual(set(worker.args[1]), {14, 15})
        self.assertEqual([file.id for file in fake.data_files], [14, 15])

    def test_run_tracking_blocks_when_previous_tracking_is_still_active(self) -> None:
        recorded: dict[str, object] = {}
        original_information = self.main_window.QMessageBox.information
        self.addCleanup(
            setattr,
            self.main_window.QMessageBox,
            "information",
            original_information,
        )
        self.main_window.QMessageBox.information = lambda *_args: recorded.setdefault(
            "message", _args[2]
        )
        fake = SimpleNamespace(
            thread=SimpleNamespace(isRunning=lambda: True),
            worker=object(),
            _active_tracking_cancel_requested=True,
        )
        fake._tracking_worker_is_active = types.MethodType(
            self.main_window.MainWindow._tracking_worker_is_active,
            fake,
        )

        self.main_window.MainWindow.run_tracking(fake)

        self.assertIn("still cancelling", recorded["message"])

    def _tracking_host(self, scenario: str, *, show_legacy=None):
        fake = SimpleNamespace(
            tracking_scenario_combo=_DummyCombo(scenario),
            thread=None,
            seeded_tracking_controller=SimpleNamespace(
                show_legacy_scenario=show_legacy or (lambda _scenario: False)
            ),
        )
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )
        fake._tracking_worker_is_active = types.MethodType(
            self.main_window.MainWindow._tracking_worker_is_active,
            fake,
        )
        return fake


if __name__ == "__main__":
    unittest.main()
