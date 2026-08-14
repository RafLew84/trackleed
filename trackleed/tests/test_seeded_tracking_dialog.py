from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.gui.seeded_tracking_adapter import SeededTrackingDialogSettings
from trackleed.gui.seeded_tracking_dialog import SeededTrackingDialog


def _frame(frame_id: int, *, with_seed: bool = False) -> LeedFile:
    spots = (
        [SpotROI(7, QRect(16, 17, 8, 8), 20.0, 21.0, 12.0, "manual")]
        if with_seed
        else []
    )
    return LeedFile(
        filepath=Path(f"frame-{frame_id}.tif"),
        filename=f"frame-{frame_id}.tif",
        is_background=False,
        element="Cu",
        surface=111,
        mode="G",
        temperature=300.0,
        energy=80.0,
        id=frame_id,
        coverage=None,
        spots=spots,
    )


class SeededTrackingDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_skip_anchor_tab_builds_preflighted_request_from_reference_seeds(
        self,
    ) -> None:
        reference = _frame(1, with_seed=True)
        second = _frame(2)
        dialog = SeededTrackingDialog(
            reference_file=reference,
            series_files=(reference, second),
            background_file=None,
            settings=SeededTrackingDialogSettings.defaults(roi_size=18),
        )
        self.addCleanup(dialog.deleteLater)

        self.assertEqual(
            [dialog.method_tabs.tabText(index) for index in range(2)],
            ["Optical Flow Tracking", "Optical Flow (skip, anchor)"],
        )
        self.assertEqual(dialog.run_button.text(), "Run Seeded Tracking")
        self.assertEqual(dialog.cancel_button.text(), "Cancel")
        dialog.method_tabs.setCurrentIndex(1)
        dialog.skip_anchor_page.error_threshold_spinbox.setValue(42)
        dialog.run_button.click()

        self.assertIsNotNone(dialog.decision)
        self.assertEqual(dialog.decision.request.method_id, "optical_flow_skip_anchor")
        self.assertEqual(dialog.decision.request.reference_frame_id, 1)
        self.assertEqual(
            tuple(seed.seed_id for seed in dialog.decision.request.seeds), (7,)
        )
        self.assertEqual(dialog.decision.request.config.error_threshold, 42)
        self.assertEqual(dialog.decision.settings.optical_flow.roi_size, 18)
        self.assertTrue(dialog.preflight_report.ready)

    def test_missing_reference_is_a_structured_block_without_closing_dialog(
        self,
    ) -> None:
        first = _frame(1)
        second = _frame(2)
        dialog = SeededTrackingDialog(
            reference_file=None,
            series_files=(first, second),
            background_file=None,
            settings=SeededTrackingDialogSettings.defaults(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.run_button.click()

        self.assertIsNone(dialog.decision)
        self.assertIn("[invalid_request]", dialog.preflight_label.text())
        self.assertIn("Reference frame", dialog.preflight_label.text())


if __name__ == "__main__":
    unittest.main()
