from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.gui.detection_linking_adapter import DetectionLinkingDialogSettings
from trackleed.gui.detection_linking_dialog import (
    DetectionLinkingDialog,
    DirectionalDetectionLinkingDialog,
)
from trackleed.logic.workflows.detection_linking import DirectionalLinkingConfig


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


def _empty_frame(frame_id: int) -> LeedFile:
    frame = _frame(frame_id, 10.0)
    frame.spots = []
    return frame


def _symmetric_frame(frame_id: int, radius: float) -> LeedFile:
    frame = _frame(frame_id, 50.0 + radius)
    frame.spots = [
        SpotROI(index, QRect(round(x) - 4, round(y) - 4, 8, 8), x, y, 5.0, "auto")
        for index, (x, y) in enumerate(
            (
                (50.0 + radius, 50.0),
                (50.0 - radius, 50.0),
                (50.0, 50.0 + radius),
                (50.0, 50.0 - radius),
            ),
            start=1,
        )
    ]
    return frame


class DetectionLinkingDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_run_builds_hungarian_request_and_accepted_settings(self) -> None:
        files = (_frame(1, 10.0), _frame(2, 12.0))
        dialog = DetectionLinkingDialog(
            files=files,
            settings=DetectionLinkingDialogSettings(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.max_distance_spinbox.setValue(8.5)
        dialog.max_gap_spinbox.setValue(3)
        dialog.min_track_length_spinbox.setValue(2)
        dialog.run_button.click()

        self.assertEqual(dialog.windowTitle(), "Detection Linking (Hungarian)")
        self.assertIsNotNone(dialog.decision)
        self.assertEqual(dialog.decision.request.method_id, "hungarian")
        self.assertEqual(dialog.decision.request.config.max_distance_px, 8.5)
        self.assertEqual(dialog.decision.settings.hungarian.max_gap, 3)
        self.assertTrue(dialog.preflight_report.ready)

    def test_directional_dialog_binds_all_visible_settings(self) -> None:
        files = (_frame(1, 10.0), _frame(2, 12.0))
        initial = DirectionalLinkingConfig(
            allow_direction_flip=False,
            min_tracks_for_flip=7,
            use_ring_penalty=False,
        )
        dialog = DirectionalDetectionLinkingDialog(
            files=files,
            settings=DetectionLinkingDialogSettings(directional=initial),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.center_mode_combo.setCurrentIndex(1)
        dialog.center_x_spinbox.setValue(50.0)
        dialog.center_y_spinbox.setValue(51.0)
        dialog.direction_mode_combo.setCurrentIndex(2)
        dialog.lateral_tolerance_spinbox.setValue(15.0)
        dialog.ring_bin_spinbox.setValue(22.0)
        dialog.base_distance_spinbox.setValue(18.0)
        dialog.dynamic_distance_checkbox.setChecked(False)
        dialog.dynamic_margin_spinbox.setValue(4.0)
        dialog.max_gap_spinbox.setValue(3)
        dialog.max_misses_spinbox.setValue(4)
        dialog.min_track_length_spinbox.setValue(2)
        dialog.run_button.click()

        self.assertEqual(dialog.windowTitle(), "Detection Linking (Directional)")
        self.assertIsNotNone(dialog.decision)
        config = dialog.decision.request.config
        self.assertEqual(dialog.decision.request.method_id, "directional")
        self.assertEqual(config.center_mode, "manual")
        self.assertEqual(config.manual_center, (50.0, 51.0))
        self.assertEqual(config.direction_mode, "outward")
        self.assertEqual(config.lateral_angle_tolerance_deg, 15.0)
        self.assertEqual(config.ring_bin_px, 22.0)
        self.assertEqual(config.base_max_distance_px, 18.0)
        self.assertFalse(config.dynamic_max_distance)
        self.assertEqual(config.dynamic_margin_px, 4.0)
        self.assertEqual(config.max_gap, 3)
        self.assertEqual(config.max_misses, 4)
        self.assertEqual(config.min_track_length, 2)
        self.assertFalse(config.allow_direction_flip)
        self.assertEqual(config.min_tracks_for_flip, 7)
        self.assertFalse(config.use_ring_penalty)
        self.assertEqual(dialog.decision.settings.directional, config)

    def test_directional_auto_center_refusal_stays_in_dialog(self) -> None:
        dialog = DirectionalDetectionLinkingDialog(
            files=(_frame(1, 10.0), _frame(2, 12.0)),
            settings=DetectionLinkingDialogSettings(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.run_button.click()

        self.assertIsNone(dialog.decision)
        self.assertIn("[pattern_center_unavailable]", dialog.preflight_label.text())

    def test_directional_auto_mode_preserves_dormant_manual_center(self) -> None:
        dialog = DirectionalDetectionLinkingDialog(
            files=(_symmetric_frame(1, 20.0), _symmetric_frame(2, 18.0)),
            settings=DetectionLinkingDialogSettings(),
        )
        self.addCleanup(dialog.deleteLater)
        dialog.center_x_spinbox.setValue(47.0)
        dialog.center_y_spinbox.setValue(53.0)

        dialog.run_button.click()

        self.assertIsNotNone(dialog.decision)
        self.assertEqual(dialog.decision.request.config.center_mode, "auto")
        self.assertEqual(dialog.decision.request.config.manual_center, (47.0, 53.0))

    def test_missing_detections_stays_open_with_structured_preflight(self) -> None:
        dialog = DetectionLinkingDialog(
            files=(_empty_frame(1), _empty_frame(2)),
            settings=DetectionLinkingDialogSettings(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.run_button.click()

        self.assertIsNone(dialog.decision)
        self.assertIn("[detections_required]", dialog.preflight_label.text())

    def test_manual_seeds_stay_open_with_structured_preflight(self) -> None:
        files = (_frame(1, 10.0), _frame(2, 12.0))
        files[0].spots[0].source = "manual_point"
        dialog = DetectionLinkingDialog(
            files=files,
            settings=DetectionLinkingDialogSettings(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.run_button.click()

        self.assertIsNone(dialog.decision)
        self.assertIn(
            "[independent_detections_required]",
            dialog.preflight_label.text(),
        )

    def test_duplicate_ids_stay_open_without_mutating_frames(self) -> None:
        files = (_frame(1, 10.0), _frame(1, 12.0))
        dialog = DetectionLinkingDialog(
            files=files,
            settings=DetectionLinkingDialogSettings(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.run_button.click()

        self.assertIsNone(dialog.decision)
        self.assertIn("[duplicate_frame_id]", dialog.preflight_label.text())
        self.assertEqual(tuple(file.id for file in files), (1, 1))


if __name__ == "__main__":
    unittest.main()
