from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from trackleed.core.data_models import LeedFile
from trackleed.gui.spot_detection_dialog import (
    SpotDetectionDialog,
    SpotDetectionDialogContext,
    SpotDetectionDialogSettings,
    SpotDetectionScope,
)
from trackleed.logic.workflows.spot_detection import (
    DetectionCapabilities,
    DetectionMethod,
    PreflightIssue,
    SpotDetectionCatalog,
    SpotDetectionFamily,
)


class _ReadyRuntime:
    def preflight(self, _config):
        return ()

    def detect(self, _runtime_input, _config):
        return ()


def _catalog() -> SpotDetectionCatalog:
    runtime = _ReadyRuntime()
    return SpotDetectionCatalog(
        (
            DetectionMethod(
                "classic",
                "Classic",
                SpotDetectionFamily.CLASSICAL,
                DetectionCapabilities(True, True, False),
                runtime,
            ),
            DetectionMethod(
                "yolo",
                "YOLO",
                SpotDetectionFamily.NEURAL,
                DetectionCapabilities(True, True, True),
                runtime,
            ),
        )
    )


class _BlockedYoloRuntime(_ReadyRuntime):
    def preflight(self, _config):
        return (PreflightIssue("model_required", "Select a YOLO model."),)


def _catalog_with_blocked_yolo() -> SpotDetectionCatalog:
    catalog = _catalog()
    classic = catalog.method("classic")
    yolo = catalog.method("yolo")
    return SpotDetectionCatalog(
        (
            classic,
            DetectionMethod(
                yolo.method_id,
                yolo.display_name,
                yolo.family,
                yolo.capabilities,
                _BlockedYoloRuntime(),
            ),
        )
    )


def _leed_file(frame_id: int) -> LeedFile:
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
    )


class SpotDetectionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_classic_current_frame_builds_headless_request(self) -> None:
        current = _leed_file(7)
        dialog = SpotDetectionDialog(
            context=SpotDetectionDialogContext(
                current_file=current,
                series_files=(current, _leed_file(8)),
            ),
            settings=SpotDetectionDialogSettings.from_mapping({}),
            catalog=_catalog(),
        )
        self.addCleanup(dialog.deleteLater)

        self.assertEqual(
            [dialog.method_tabs.tabText(index) for index in range(2)],
            ["Classic", "YOLO"],
        )
        self.assertEqual(dialog.detect_current_button.text(), "Detect Current Frame")
        self.assertEqual(dialog.detect_all_button.text(), "Detect All Frames")
        self.assertEqual(dialog.cancel_button.text(), "Cancel")

        dialog.classic_page.roi_size_spinbox.setValue(44)
        dialog.detect_current_button.click()

        self.assertIsNotNone(dialog.decision)
        self.assertIs(dialog.decision.scope, SpotDetectionScope.CURRENT_FRAME)
        self.assertEqual(dialog.decision.request.method_id, "classic")
        self.assertEqual(dialog.decision.request.frames[0].frame_id, 7)
        self.assertEqual(dialog.decision.request.config.roi_size, 44)
        self.assertTrue(dialog.preflight_report.ready)
        self.assertIn("Ready", dialog.preflight_label.text())

    def test_yolo_all_frames_accepts_both_method_configurations(self) -> None:
        first = _leed_file(7)
        settings = SpotDetectionDialogSettings.from_mapping(
            {
                "backend": "Classic",
                "classic": {"roi_size": 40, "threshold_rel": 0.2},
                "yolo": {
                    "model_path": "model.pt",
                    "confidence": 0.3,
                    "iou": 0.4,
                    "max_det": 25,
                    "bbox_margin_px": 3,
                    "subtract_bg": False,
                    "device": "cpu",
                },
            }
        )
        dialog = SpotDetectionDialog(
            context=SpotDetectionDialogContext(
                current_file=first,
                series_files=(first, _leed_file(8)),
            ),
            settings=settings,
            catalog=_catalog(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.method_tabs.setCurrentIndex(1)
        dialog.yolo_page.model_path_edit.setText("updated.pt")
        dialog.yolo_page.confidence_spinbox.setValue(0.55)
        dialog.detect_all_button.click()

        self.assertEqual(dialog.decision.request.method_id, "yolo")
        self.assertEqual(
            tuple(frame.frame_id for frame in dialog.decision.request.frames),
            (7, 8),
        )
        self.assertEqual(dialog.decision.settings.classic.roi_size, 40)
        self.assertEqual(dialog.decision.settings.yolo.confidence, 0.55)
        self.assertEqual(dialog.decision.settings.yolo.model_path, Path("updated.pt"))
        self.assertEqual(dialog.decision.settings.yolo.iou, 0.4)
        self.assertEqual(dialog.decision.settings.yolo.max_detections, 25)
        self.assertEqual(dialog.decision.settings.yolo.bbox_margin_px, 3)
        self.assertEqual(dialog.decision.settings.yolo.device, "cpu")
        self.assertIs(dialog.decision.scope, SpotDetectionScope.ALL_FRAMES)

    def test_cancel_discards_unaccepted_edits(self) -> None:
        current = _leed_file(7)
        settings = SpotDetectionDialogSettings.from_mapping(
            {
                "backend": "Classic",
                "classic": {"roi_size": 40},
                "yolo": {"confidence": 0.3},
            }
        )
        dialog = SpotDetectionDialog(
            context=SpotDetectionDialogContext(current, (current,)),
            settings=settings,
            catalog=_catalog(),
        )
        self.addCleanup(dialog.deleteLater)
        dialog.classic_page.roi_size_spinbox.setValue(64)
        dialog.yolo_page.confidence_spinbox.setValue(0.75)

        dialog.cancel_button.click()

        self.assertIsNone(dialog.decision)
        self.assertEqual(settings.classic.roi_size, 40)
        self.assertEqual(settings.yolo.confidence, 0.3)

    def test_structural_preflight_keeps_unavailable_yolo_open(self) -> None:
        current = _leed_file(7)
        dialog = SpotDetectionDialog(
            context=SpotDetectionDialogContext(current, (current,)),
            settings=SpotDetectionDialogSettings.from_mapping({"backend": "YOLO"}),
            catalog=_catalog_with_blocked_yolo(),
        )
        self.addCleanup(dialog.deleteLater)

        dialog.detect_current_button.click()

        self.assertIsNone(dialog.decision)
        self.assertFalse(dialog.preflight_report.ready)
        self.assertEqual(dialog.preflight_report.issues[0].code, "model_required")
        self.assertIn("[model_required]", dialog.preflight_label.text())

    def test_legacy_hidden_backend_falls_back_without_losing_session_values(self) -> None:
        current = _leed_file(7)
        settings = SpotDetectionDialogSettings.from_mapping(
            {
                "backend": "Deep Learning",
                "dl": {"model_path": "legacy.pt", "confidence": 0.7},
            }
        )

        dialog = SpotDetectionDialog(
            context=SpotDetectionDialogContext(current, (current,)),
            settings=settings,
            catalog=_catalog(),
        )
        self.addCleanup(dialog.deleteLater)

        self.assertEqual(dialog.method_tabs.currentIndex(), 0)
        dialog.detect_current_button.click()
        serialized = dialog.decision.settings.to_mapping()
        self.assertEqual(dialog.decision.request.method_id, "classic")
        self.assertEqual(
            serialized["dl"],
            {"model_path": "legacy.pt", "confidence": 0.7},
        )


if __name__ == "__main__":
    unittest.main()
