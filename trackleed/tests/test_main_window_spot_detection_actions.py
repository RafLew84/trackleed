import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from trackleed.gui.spot_detection_adapter import (
    SpotDetectionCompletion,
    SpotDetectionDialogSettings,
    SpotDetectionSettingsBinding,
)
from trackleed.gui.spot_detection_dialog import (
    SpotDetectionDialogDecision,
    SpotDetectionScope,
)
from trackleed.logic.workflows.spot_detection import (
    DetectionFrame,
    DetectionRequest,
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

    def test_import_uses_its_own_peak_analyzer_scipy_contract(self) -> None:
        scipy_module = sys.modules["scipy"]
        optimize_module = sys.modules["scipy.optimize"]
        peak_analyzer = sys.modules["trackleed.analysis.peak_analyzer"]

        self.assertIs(scipy_module.optimize, optimize_module)
        self.assertIs(peak_analyzer.curve_fit, optimize_module.curve_fit)
        self.assertEqual(
            optimize_module.curve_fit()["name"],
            "scipy.optimize.curve_fit",
        )

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
        launch_args, launch_kwargs = self.recorded["launch"]
        self.assertIs(launch_args[2], worker.args[2])
        self.assertEqual(launch_kwargs["cancel_label"], "Detection")
        worker.args[2].set()
        self.assertTrue(worker.args[2].is_set())

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

    def test_detect_spots_dialog_starts_accepted_request_with_shared_host(self) -> None:
        current = SimpleNamespace(
            id=11,
            filepath=Path("one.tif"),
            spots=[],
            is_background=False,
        )
        second = SimpleNamespace(
            id=12,
            filepath=Path("two.tif"),
            spots=[],
            is_background=False,
        )
        settings = SpotDetectionDialogSettings.from_mapping(
            {"backend": "Classic", "classic": {"roi_size": 40}}
        )
        request = DetectionRequest(
            "classic",
            (DetectionFrame(11, Path("one.tif")),),
            settings.classic,
        )
        decision = SpotDetectionDialogDecision(
            request,
            (current,),
            SpotDetectionScope.CURRENT_FRAME,
            settings,
        )
        captured: dict[str, object] = {}
        binding = SpotDetectionSettingsBinding()

        class Dialog:
            def __init__(self, **kwargs):
                captured["dialog"] = kwargs
                self.decision = decision

            def exec(self):
                return 1

        original_dialog = self.main_window.SpotDetectionDialog
        original_dialog_code = getattr(self.main_window.QDialog, "DialogCode", None)
        self.main_window.SpotDetectionDialog = Dialog
        self.main_window.QDialog.DialogCode = SimpleNamespace(Accepted=1)
        self.addCleanup(setattr, self.main_window, "SpotDetectionDialog", original_dialog)
        self.addCleanup(
            setattr,
            self.main_window.QDialog,
            "DialogCode",
            original_dialog_code,
        )
        item = SimpleNamespace(data=lambda _role: current)
        fake = SimpleNamespace(
            file_list_widget=SimpleNamespace(currentItem=lambda: item),
            data_files=[current, second],
            background_file=None,
            metadata_mask_list=[],
            analysis_circle=None,
            analysis_ellipse=None,
            spot_detection_binding=binding,
            _normalize_analysis_ellipse=lambda value: value,
            _start_spot_detection=lambda *args, **kwargs: captured.update(
                {"start": (args, kwargs)}
            ),
        )

        self.main_window.MainWindow.show_spot_detection_dialog(fake)

        context = captured["dialog"]["context"]
        self.assertIs(context.current_file, current)
        self.assertEqual(context.series_files, (current, second))
        self.assertEqual(binding.roi_size, 40)
        self.assertIs(captured["start"][1]["request"], request)
        self.assertFalse(captured["start"][1]["is_batch"])

    def test_session_settings_round_trip_without_detector_widgets(self) -> None:
        fake = SimpleNamespace(spot_detection_binding=SpotDetectionSettingsBinding())
        session_settings = {
            "backend": "YOLO",
            "classic": {"roi_size": 48, "threshold_rel": 0.15},
            "dl": {"confidence": 0.6, "model_path": "legacy.pt"},
            "yolo": {
                "model_path": "spots.pt",
                "confidence": 0.35,
                "iou": 0.5,
                "max_det": 120,
                "bbox_margin_px": 4,
                "subtract_bg": True,
                "device": "cpu",
            },
        }

        self.main_window.MainWindow._apply_spot_settings(fake, session_settings)
        serialized = self.main_window.MainWindow._collect_spot_settings(fake)

        self.assertEqual(
            fake.spot_detection_binding.dialog_settings.selected_method_id,
            "yolo",
        )
        self.assertEqual(serialized["classic"]["roi_size"], 48)
        self.assertEqual(serialized["dl"], session_settings["dl"])
        self.assertEqual(serialized["yolo"]["model_path"], "spots.pt")
        self.assertTrue(serialized["yolo"]["subtract_bg"])
