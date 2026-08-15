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
from trackleed.logic.workflows.point_tracking import TapirPointConfig
from trackleed.logic.workflows.mask_tracking import CutieMaskConfig
from trackleed.logic.workflows.seeded_tracking import (
    SeededTrackingCapabilities,
    SeededTrackingCatalog,
    SeededTrackingMethod,
    SeedRepresentation,
    TrackingFamily,
    TrackingPreflightIssue,
)


class _ReadyRuntime:
    def __init__(self):
        self.preflight_calls = 0

    def validate_config(self, _config):
        return ()

    def preflight(self, _config, _context=None):
        self.preflight_calls += 1
        return (TrackingPreflightIssue("unexpected_probe", "must run in worker"),)

    def run(self, _request, _context):
        raise AssertionError("Dialog must not execute tracking")


def _frame(frame_id: int, *, with_seed: bool = False) -> LeedFile:
    spots = (
        [
            SpotROI(
                7,
                QRect(16, 17, 8, 8),
                20.0,
                21.0,
                12.0,
                "manual",
                [(16.0, 17.0), (24.0, 17.0), (20.0, 25.0)],
            )
        ]
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
            [
                dialog.method_tabs.tabText(index)
                for index in range(dialog.method_tabs.count())
            ],
            [
                "Optical Flow Tracking",
                "Optical Flow (skip, anchor)",
                "Offline TAPIR",
                "BootsTAPIR Online / Causal",
                "Offline LocoTrack",
                "Offline CoTracker3",
                "Offline TAPNext",
                "Offline Track-On-R",
                "Offline PIPs++",
                "Offline MFTIQ",
                "Offline ToMP / TaMOs",
                "Offline OSTrack",
                "Offline Cutie",
                "Offline XMem2",
                "Offline SAM2",
                "Offline DAM4SAM",
                "Offline SAMURAI",
                "Offline DEVA",
            ],
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

    def test_point_page_builds_typed_request_through_stable_catalog_id(self) -> None:
        reference = _frame(1, with_seed=True)
        second = _frame(2)
        runtime = _ReadyRuntime()
        method = SeededTrackingMethod(
            "tapir_offline",
            "Offline TAPIR",
            TrackingFamily.POINT,
            SeededTrackingCapabilities(
                SeedRepresentation.POINT,
                supports_background=False,
                preserves_gaps=True,
                supports_retrack=True,
            ),
            runtime,
        )
        dialog = SeededTrackingDialog(
            reference_file=reference,
            series_files=(reference, second),
            background_file=None,
            settings=SeededTrackingDialogSettings.defaults(roi_size=18),
            catalog=SeededTrackingCatalog((method,)),
        )
        self.addCleanup(dialog.deleteLater)

        page = dialog.point_pages["tapir_offline"]
        page.field_widgets["checkpoint_path"].setText("custom-tapir.pt")
        page.field_widgets["roi_size"].setValue(26)
        dialog.run_button.click()

        self.assertIsNotNone(dialog.decision)
        config = dialog.decision.request.config
        self.assertIsInstance(config, TapirPointConfig)
        self.assertEqual(config.checkpoint_path, Path("custom-tapir.pt"))
        self.assertEqual(config.roi_size, 26)
        self.assertEqual(dialog.decision.settings.point_config("tapir_offline"), config)
        self.assertEqual(runtime.preflight_calls, 0)

    def test_mask_page_builds_typed_request_through_stable_catalog_id(self) -> None:
        reference = _frame(1, with_seed=True)
        second = _frame(2)
        runtime = _ReadyRuntime()
        method = SeededTrackingMethod(
            "cutie_offline",
            "Offline Cutie",
            TrackingFamily.MASK,
            SeededTrackingCapabilities(
                SeedRepresentation.MASK,
                supports_background=False,
                preserves_gaps=True,
                supports_retrack=True,
            ),
            runtime,
        )
        dialog = SeededTrackingDialog(
            reference_file=reference,
            series_files=(reference, second),
            background_file=None,
            settings=SeededTrackingDialogSettings.defaults(roi_size=18),
            catalog=SeededTrackingCatalog((method,)),
        )
        self.addCleanup(dialog.deleteLater)

        page = dialog.tracker_pages["cutie_offline"]
        page.field_widgets["checkpoint_path"].setText("custom-cutie.pth")
        page.field_widgets["roi_size"].setValue(26)
        dialog.run_button.click()

        self.assertIsNotNone(dialog.decision)
        config = dialog.decision.request.config
        self.assertIsInstance(config, CutieMaskConfig)
        self.assertEqual(config.checkpoint_path, Path("custom-cutie.pth"))
        self.assertEqual(config.roi_size, 26)
        self.assertIs(dialog.decision.request.seed_representation, SeedRepresentation.MASK)
        self.assertEqual(dialog.decision.settings.mask_config("cutie_offline"), config)
        self.assertEqual(runtime.preflight_calls, 0)

    def test_default_mask_page_preserves_optional_numeric_values(self) -> None:
        reference = _frame(1, with_seed=True)
        second = _frame(2)
        dialog = SeededTrackingDialog(
            reference_file=reference,
            series_files=(reference, second),
            background_file=None,
            settings=SeededTrackingDialogSettings.defaults(roi_size=18),
        )
        self.addCleanup(dialog.deleteLater)
        dialog.method_tabs.setCurrentIndex(dialog._method_ids.index("cutie_offline"))

        dialog.run_button.click()

        self.assertIsNotNone(dialog.decision)
        config = dialog.decision.request.config
        self.assertIsInstance(config, CutieMaskConfig)
        self.assertIsNone(config.mem_every)
        self.assertTrue(dialog.preflight_report.ready)


if __name__ == "__main__":
    unittest.main()
