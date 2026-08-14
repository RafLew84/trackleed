import unittest
from pathlib import Path
from types import SimpleNamespace

from trackleed.logic.workflows.spot_detection import (
    DetectionProgress,
    DetectionResult,
    FrameDetections,
)
from trackleed.tests.test_main_window_registry_dispatch import MainWindowImportMixin


def _classic_settings() -> dict[str, object]:
    return {
        "backend": "Classic",
        "classic": {
            "blur_sigma": 2.0,
            "wth_radius": 3,
            "min_distance": 4,
            "threshold_rel": 0.1,
            "roi_size": 16,
        },
    }


class MainWindowSpotDetectionActionTests(MainWindowImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        _registry, self.main_window = self.import_main_window()
        self.recorded: dict[str, object] = {}
        original_run = self.main_window.run_spot_detection
        self.addCleanup(setattr, self.main_window, "run_spot_detection", original_run)

        def run(request, context):
            self.recorded["request"] = request
            self.recorded["context"] = context
            for completed, frame in enumerate(request.frames, start=1):
                context.report_progress(
                    DetectionProgress(completed, len(request.frames), frame.frame_id)
                )
            return DetectionResult(
                tuple(FrameDetections(frame.frame_id, ()) for frame in request.frames)
            )

        self.main_window.run_spot_detection = run

    def test_current_frame_action_delegates_to_public_workflow(self) -> None:
        leed_file = SimpleNamespace(filepath=Path("one.tif"), spots=[object()])
        item = SimpleNamespace(data=lambda _role: leed_file)
        calls: list[object] = []
        fake = SimpleNamespace(
            file_list_widget=SimpleNamespace(currentItem=lambda: item),
            background_file=None,
            metadata_mask_list=[],
            analysis_circle=None,
            analysis_ellipse=None,
            _normalize_analysis_ellipse=lambda value: value,
            _collect_spot_settings=_classic_settings,
            populate_spot_list=lambda spots: calls.append(("spots", spots)),
            refresh_current_view=lambda: calls.append("refresh"),
        )

        self.main_window.MainWindow.run_spot_finding(fake)

        request = self.recorded["request"]
        self.assertEqual(request.method_id, "classic")
        self.assertEqual(tuple(frame.source_path for frame in request.frames), (Path("one.tif"),))
        self.assertEqual(leed_file.spots, [])
        self.assertIn("refresh", calls)

    def test_full_series_action_uses_same_workflow_and_progress_context(self) -> None:
        files = [
            SimpleNamespace(filepath=Path("one.tif"), spots=[object()], is_background=False),
            SimpleNamespace(filepath=Path("two.tif"), spots=[object()], is_background=False),
        ]
        progress_values: list[int] = []

        class ProgressDialog:
            def __init__(self, *_args):
                pass

            def setWindowModality(self, _value):
                pass

            def setValue(self, value):
                progress_values.append(value)

            def wasCanceled(self):
                return False

            def cancel(self):
                pass

        original_dialog = self.main_window.QProgressDialog
        self.addCleanup(setattr, self.main_window, "QProgressDialog", original_dialog)
        self.main_window.QProgressDialog = ProgressDialog
        self.main_window.QApplication.processEvents = lambda: None
        item = SimpleNamespace()
        fake = SimpleNamespace(
            data_files=files,
            background_file=None,
            metadata_mask_list=[],
            analysis_circle=None,
            analysis_ellipse=None,
            _normalize_analysis_ellipse=lambda value: value,
            _collect_spot_settings=_classic_settings,
            populate_file_list=lambda: None,
            file_list_widget=SimpleNamespace(
                currentItem=lambda: item,
                count=lambda: 1,
                setCurrentRow=lambda _row: None,
            ),
            display_selected_image=lambda *_args: None,
        )

        self.main_window.MainWindow.run_spot_finding_all(fake)

        request = self.recorded["request"]
        self.assertEqual(len(request.frames), 2)
        self.assertEqual(progress_values, [0, 1, 2, 2])
        self.assertEqual([file.spots for file in files], [[], []])
