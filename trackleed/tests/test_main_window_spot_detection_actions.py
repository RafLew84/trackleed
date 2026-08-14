import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from trackleed.gui.spot_detection_adapter import SpotDetectionCompletion
from trackleed.logic.workflows.spot_detection import DetectionResult, FrameDetections
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
        original_worker = self.main_window.GenericWorker
        self.addCleanup(setattr, self.main_window, "GenericWorker", original_worker)
        recorded = self.recorded

        class Worker:
            def __init__(self, function, *args, **kwargs):
                self.function = function
                self.args = args
                self.kwargs = kwargs
                recorded["worker"] = self

        self.main_window.GenericWorker = Worker

    def _bind_detection_handlers(self, fake: SimpleNamespace) -> None:
        fake._start_spot_detection = types.MethodType(
            self.main_window.MainWindow._start_spot_detection,
            fake,
        )
        fake._on_spot_detection_completed = types.MethodType(
            self.main_window.MainWindow._on_spot_detection_completed,
            fake,
        )

    def test_current_frame_action_delegates_to_public_workflow_worker(self) -> None:
        original_spots = [object()]
        leed_file = SimpleNamespace(id=11, filepath=Path("one.tif"), spots=original_spots)
        item = SimpleNamespace(data=lambda _role: leed_file)
        fake = SimpleNamespace(
            file_list_widget=SimpleNamespace(currentItem=lambda: item),
            background_file=None,
            metadata_mask_list=[],
            analysis_circle=None,
            analysis_ellipse=None,
            _normalize_analysis_ellipse=lambda value: value,
            _collect_spot_settings=_classic_settings,
            _tracking_worker_is_active=lambda: False,
            _launch_worker_with_progress=lambda *args, **kwargs: self.recorded.update(
                {"launch": (args, kwargs)}
            ),
        )
        self._bind_detection_handlers(fake)

        self.main_window.MainWindow.run_spot_finding(fake)

        worker = self.recorded["worker"]
        request = worker.args[0]
        self.assertIs(worker.function, self.main_window.run_spot_detection_job)
        self.assertEqual(request.method_id, "classic")
        self.assertEqual(request.frames[0].frame_id, 11)
        self.assertEqual(request.frames[0].source_path, Path("one.tif"))
        self.assertIs(leed_file.spots, original_spots)
        self.assertEqual(self.recorded["launch"][1]["cancel_label"], "Detection")

    def test_full_series_action_uses_the_same_public_workflow_worker(self) -> None:
        files = [
            SimpleNamespace(id=11, filepath=Path("one.tif"), spots=[object()], is_background=False),
            SimpleNamespace(id=12, filepath=Path("two.tif"), spots=[object()], is_background=False),
        ]
        original_spots = [file.spots for file in files]
        fake = SimpleNamespace(
            data_files=files,
            background_file=None,
            metadata_mask_list=[],
            analysis_circle=None,
            analysis_ellipse=None,
            _normalize_analysis_ellipse=lambda value: value,
            _collect_spot_settings=_classic_settings,
            _tracking_worker_is_active=lambda: False,
            _launch_worker_with_progress=lambda *args, **kwargs: self.recorded.update(
                {"launch": (args, kwargs)}
            ),
        )
        self._bind_detection_handlers(fake)

        self.main_window.MainWindow.run_spot_finding_all(fake)

        worker = self.recorded["worker"]
        request = worker.args[0]
        self.assertIs(worker.function, self.main_window.run_spot_detection_job)
        self.assertEqual(tuple(frame.frame_id for frame in request.frames), (11, 12))
        self.assertEqual([file.spots for file in files], original_spots)
        self.assertIn("all frames", self.recorded["launch"][0][1])

    def test_worker_result_is_published_only_on_the_gui_completion_handler(self) -> None:
        files = [
            SimpleNamespace(id=11, filepath=Path("one.tif"), spots=[object()]),
            SimpleNamespace(id=12, filepath=Path("two.tif"), spots=[object()]),
        ]
        calls: list[object] = []
        fake = SimpleNamespace(
            _pending_spot_detection_files=tuple(files),
            _pending_spot_detection_is_batch=True,
            populate_file_list=lambda: calls.append("files"),
            file_list_widget=SimpleNamespace(currentItem=lambda: None, count=lambda: 0),
        )
        result = DetectionResult(
            (FrameDetections(11, ()), FrameDetections(12, ()))
        )

        self.main_window.MainWindow._on_spot_detection_completed(
            fake,
            SpotDetectionCompletion.from_worker_result(result),
        )

        self.assertEqual([file.spots for file in files], [[], []])
        self.assertIn("files", calls)
