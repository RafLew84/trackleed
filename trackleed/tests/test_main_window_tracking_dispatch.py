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

    def test_run_tracking_routes_legacy_point_choice_to_seeded_controller(
        self,
    ) -> None:
        calls: list[str] = []
        fake = self._tracking_host(
            "Offline TAPIR",
            show_legacy=lambda scenario: calls.append(scenario) or True,
        )

        self.main_window.MainWindow.run_tracking(fake)

        self.assertEqual(calls, ["Offline TAPIR"])

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
        fake = self._tracking_host("Offline ToMP")
        fake._build_registered_tracking_worker = lambda tracker_spec, cancel_event: None
        fake._launch_worker_with_progress = lambda *args, **kwargs: recorded.update(
            {"launched": True}
        )

        self.main_window.MainWindow.run_tracking(fake)

        self.assertNotIn("launched", recorded)

    def test_legacy_hungarian_choice_routes_to_detection_linking_controller(
        self,
    ) -> None:
        calls: list[str] = []
        fake = self._tracking_host("Detection Linking (Hungarian)")
        fake.detection_linking_controller = SimpleNamespace(
            show_legacy_scenario=lambda scenario: calls.append(scenario) or True
        )

        self.main_window.MainWindow.run_tracking(fake)

        self.assertEqual(calls, ["Detection Linking (Hungarian)"])

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
            detection_linking_controller=SimpleNamespace(
                show_legacy_scenario=lambda _scenario: False
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
