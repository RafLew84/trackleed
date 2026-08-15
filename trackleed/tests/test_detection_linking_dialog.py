from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.gui.detection_linking_adapter import DetectionLinkingDialogSettings
from trackleed.gui.detection_linking_dialog import DetectionLinkingDialog


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
