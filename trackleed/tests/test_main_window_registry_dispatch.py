from __future__ import annotations

import importlib
import sys
import types
import unittest
from types import SimpleNamespace


def _stub_callable(name: str):
    def _fn(*args, **kwargs):
        return {"name": name, "args": args, "kwargs": kwargs}

    _fn.__name__ = name.rsplit(".", 1)[-1]
    return _fn


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _DummyCombo:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:
        return self._text


class _Qt:
    class ItemDataRole:
        UserRole = 0

    class WindowModality:
        WindowModal = 0


class MainWindowImportMixin:
    tracker_registry_module_name = "trackleed.gui.tracker_registry"
    main_window_module_name = "trackleed.gui.main_window"

    def import_main_window(self):
        originals: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            "PyQt6.QtWidgets",
            "trackleed.io.file_parser",
            "trackleed.io.image_reader",
            "trackleed.gui.mask_dialog",
            "trackleed.analysis.spot_finding",
            "trackleed.analysis.dl_spot_finder",
            "trackleed.analysis.yolo_detector",
            "trackleed.analysis.tapir_backend",
            "trackleed.analysis.locotrack_backend",
            "trackleed.analysis.cotracker_backend",
            "trackleed.analysis.tapnext_backend",
            "trackleed.analysis.trackonr_backend",
            "trackleed.analysis.pipspp_backend",
            "trackleed.analysis.mftiq_backend",
            "trackleed.analysis.cutie_backend",
            "trackleed.analysis.xmem2_backend",
            "trackleed.analysis.tomp_backend",
            "trackleed.analysis.sam2_backend",
            "trackleed.analysis.tracking_bbox_backends",
            "trackleed.analysis.tracking_mask_backends",
            "trackleed.analysis.tracking_point_backends",
            "trackleed.analysis.tracking_retrack",
            "trackleed.io.session",
            "trackleed.gui.circle_mask_dialog",
            "trackleed.gui.ellipse_mask_dialog",
            "trackleed.gui.polygon_mask_dialog",
            "trackleed.io.gif_loader",
            "trackleed.gui.leed_image_viewer",
            "trackleed.gui.tracking_dialog",
            "trackleed.logic.generic_worker",
            "trackleed.analysis.tracking",
            "skimage",
            "skimage.util",
            "skimage.color",
            "skimage.feature",
            "skimage.morphology",
            "scipy",
            "scipy.ndimage",
            self.tracker_registry_module_name,
            self.main_window_module_name,
        ):
            originals[module_name] = sys.modules.get(module_name)
            sys.modules.pop(module_name, None)

        qtcore_module = types.ModuleType("PyQt6.QtCore")
        qtcore_module.Qt = _Qt
        qtcore_module.QPointF = _Dummy
        qtcore_module.QRect = _Dummy
        qtcore_module.QRectF = _Dummy
        qtcore_module.QThread = _Dummy
        qtcore_module.QPoint = _Dummy

        qtgui_module = types.ModuleType("PyQt6.QtGui")
        for attr_name in (
            "QAction",
            "QPixmap",
            "QImage",
            "QPainter",
            "QPen",
            "QBrush",
            "QColor",
            "QActionGroup",
            "QIcon",
        ):
            setattr(qtgui_module, attr_name, _Dummy)

        class _MessageBox:
            @staticmethod
            def warning(*args, **kwargs):
                return None

            @staticmethod
            def information(*args, **kwargs):
                return None

            @staticmethod
            def critical(*args, **kwargs):
                return None

        qtwidgets_module = types.ModuleType("PyQt6.QtWidgets")
        qtwidgets_module.QMessageBox = _MessageBox
        for attr_name in (
            "QMainWindow",
            "QWidget",
            "QHBoxLayout",
            "QVBoxLayout",
            "QListWidget",
            "QGroupBox",
            "QLabel",
            "QFileDialog",
            "QListWidgetItem",
            "QApplication",
            "QDialog",
            "QDockWidget",
            "QFormLayout",
            "QSpinBox",
            "QDoubleSpinBox",
            "QPushButton",
            "QCheckBox",
            "QToolBar",
            "QComboBox",
            "QProgressDialog",
            "QScrollArea",
            "QStackedWidget",
            "QSizePolicy",
            "QInputDialog",
            "QLineEdit",
        ):
            setattr(qtwidgets_module, attr_name, _Dummy)

        pyqt6_module = types.ModuleType("PyQt6")
        pyqt6_module.QtCore = qtcore_module
        pyqt6_module.QtGui = qtgui_module
        pyqt6_module.QtWidgets = qtwidgets_module

        sys.modules["PyQt6"] = pyqt6_module
        sys.modules["PyQt6.QtCore"] = qtcore_module
        sys.modules["PyQt6.QtGui"] = qtgui_module
        sys.modules["PyQt6.QtWidgets"] = qtwidgets_module

        self._install_simple_stub(
            originals,
            "trackleed.io.file_parser",
            parse_directory=_stub_callable("trackleed.io.file_parser.parse_directory"),
        )
        self._install_simple_stub(
            originals,
            "trackleed.io.image_reader",
            load_tiff=_stub_callable("trackleed.io.image_reader.load_tiff"),
        )
        self._install_simple_stub(
            originals,
            "trackleed.gui.mask_dialog",
            MaskDialog=_Dummy,
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.spot_finding",
            find_spots=_stub_callable("trackleed.analysis.spot_finding.find_spots"),
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.dl_spot_finder",
            detect_spots_with_model=_stub_callable(
                "trackleed.analysis.dl_spot_finder.detect_spots_with_model"
            ),
            DLSpotDetectionError=RuntimeError,
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.yolo_detector",
            detect_spots_with_yolo=_stub_callable(
                "trackleed.analysis.yolo_detector.detect_spots_with_yolo"
            ),
            YOLODetectionError=RuntimeError,
            YOLOSettings=_Dummy,
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tapir_backend",
            discover_online_tapir_checkpoints=_stub_callable(
                "trackleed.analysis.tapir_backend.discover_online_tapir_checkpoints"
            ),
            format_tapir_environment_report=lambda status: f"tapir:{status}",
            probe_tapir_environment=_stub_callable(
                "trackleed.analysis.tapir_backend.probe_tapir_environment"
            ),
        )
        for module_name, prefix in (
            ("trackleed.analysis.locotrack_backend", "locotrack"),
            ("trackleed.analysis.cotracker_backend", "cotracker"),
            ("trackleed.analysis.tapnext_backend", "tapnext"),
            ("trackleed.analysis.trackonr_backend", "trackonr"),
            ("trackleed.analysis.pipspp_backend", "pipspp"),
            ("trackleed.analysis.mftiq_backend", "mftiq"),
            ("trackleed.analysis.cutie_backend", "cutie"),
            ("trackleed.analysis.xmem2_backend", "xmem2"),
            ("trackleed.analysis.sam2_backend", "sam2"),
        ):
            self._install_simple_stub(
                originals,
                module_name,
                **{
                    f"format_{prefix}_environment_report": lambda status, p=prefix: f"{p}:{status}",
                    f"probe_{prefix}_environment": _stub_callable(
                        f"{module_name}.probe_{prefix}_environment"
                    ),
                },
            )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tomp_backend",
            expected_tomp_checkpoint_name_for_variant=lambda variant: f"{variant}.pth.tar",
            format_tomp_environment_report=lambda status: f"tomp:{status}",
            probe_tomp_environment=_stub_callable(
                "trackleed.analysis.tomp_backend.probe_tomp_environment"
            ),
            select_default_tamos_checkpoint_for_variant=lambda checkpoints, variant: None,
            select_default_tomp_checkpoint_for_variant=lambda checkpoints, variant: None,
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tracking_bbox_backends",
            track_spots_tomp_offline=_stub_callable(
                "trackleed.analysis.tracking_bbox_backends.track_spots_tomp_offline"
            ),
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tracking_mask_backends",
            track_spots_cutie_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_cutie_offline"
            ),
            track_spots_sam2_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_sam2_offline"
            ),
            track_spots_xmem2_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_xmem2_offline"
            ),
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tracking_point_backends",
            track_spots_cotracker3_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_cotracker3_offline"
            ),
            track_spots_locotrack_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_locotrack_offline"
            ),
            track_spots_mftiq_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_mftiq_offline"
            ),
            track_spots_pipspp_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_pipspp_offline"
            ),
            track_spots_tapir_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_tapir_offline"
            ),
            track_spots_tapir_online=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_tapir_online"
            ),
            track_spots_tapnext_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_tapnext_offline"
            ),
            track_spots_trackonr_offline=_stub_callable(
                "trackleed.analysis.tracking_point_backends.track_spots_trackonr_offline"
            ),
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tracking_retrack",
            retrack_single_spot_cotracker3_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_cotracker3_offline"
            ),
            retrack_single_spot_cutie_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_cutie_offline"
            ),
            retrack_single_spot_locotrack_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_locotrack_offline"
            ),
            retrack_single_spot_mftiq_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_mftiq_offline"
            ),
            retrack_single_spot_pipspp_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_pipspp_offline"
            ),
            retrack_single_spot_sam2_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_sam2_offline"
            ),
            retrack_single_spot_tapir_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_tapir_offline"
            ),
            retrack_single_spot_tapir_online=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_tapir_online"
            ),
            retrack_single_spot_tapnext_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_tapnext_offline"
            ),
            retrack_single_spot_tomp_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_tomp_offline"
            ),
            retrack_single_spot_trackonr_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_trackonr_offline"
            ),
            retrack_single_spot_xmem2_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_xmem2_offline"
            ),
        )
        self._install_simple_stub(
            originals,
            "trackleed.io.session",
            load_session_from_file=_stub_callable("trackleed.io.session.load_session_from_file"),
            save_session_to_file=_stub_callable("trackleed.io.session.save_session_to_file"),
        )
        for module_name, class_name in (
            ("trackleed.gui.circle_mask_dialog", "CircleMaskDialog"),
            ("trackleed.gui.ellipse_mask_dialog", "EllipseMaskDialog"),
            ("trackleed.gui.polygon_mask_dialog", "PolygonMaskDialog"),
            ("trackleed.gui.leed_image_viewer", "LeedImageViewer"),
            ("trackleed.gui.tracking_dialog", "TrackingDialog"),
        ):
            self._install_simple_stub(originals, module_name, **{class_name: _Dummy})
        self._install_simple_stub(
            originals,
            "trackleed.io.gif_loader",
            extract_gif_frames=_stub_callable("trackleed.io.gif_loader.extract_gif_frames"),
        )
        self._install_simple_stub(
            originals,
            "trackleed.logic.generic_worker",
            GenericWorker=_Dummy,
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.tracking",
            track_spots_optical_flow=_stub_callable(
                "trackleed.analysis.tracking.track_spots_optical_flow"
            ),
            track_spots_optical_flow_skip_anchor=_stub_callable(
                "trackleed.analysis.tracking.track_spots_optical_flow_skip_anchor"
            ),
            track_spots_hungarian=_stub_callable(
                "trackleed.analysis.tracking.track_spots_hungarian"
            ),
            track_spots_hungarian_directional=_stub_callable(
                "trackleed.analysis.tracking.track_spots_hungarian_directional"
            ),
        )

        skimage_module = types.ModuleType("skimage")
        skimage_util = types.ModuleType("skimage.util")
        skimage_util.img_as_float = _stub_callable("skimage.util.img_as_float")
        skimage_color = types.ModuleType("skimage.color")
        skimage_color.rgb2gray = _stub_callable("skimage.color.rgb2gray")
        skimage_feature = types.ModuleType("skimage.feature")
        skimage_feature.peak_local_max = _stub_callable("skimage.feature.peak_local_max")
        skimage_morph = types.ModuleType("skimage.morphology")
        skimage_morph.white_tophat = _stub_callable("skimage.morphology.white_tophat")
        skimage_morph.disk = _stub_callable("skimage.morphology.disk")
        scipy_module = types.ModuleType("scipy")
        scipy_ndimage = types.ModuleType("scipy.ndimage")
        scipy_ndimage.gaussian_filter = _stub_callable("scipy.ndimage.gaussian_filter")
        sys.modules["skimage"] = skimage_module
        sys.modules["skimage.util"] = skimage_util
        sys.modules["skimage.color"] = skimage_color
        sys.modules["skimage.feature"] = skimage_feature
        sys.modules["skimage.morphology"] = skimage_morph
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.ndimage"] = scipy_ndimage

        self.addCleanup(self._restore_modules, originals)

        tracker_registry = importlib.import_module(self.tracker_registry_module_name)
        main_window = importlib.import_module(self.main_window_module_name)
        self.addCleanup(sys.modules.pop, self.main_window_module_name, None)
        self.addCleanup(sys.modules.pop, self.tracker_registry_module_name, None)
        return tracker_registry, main_window

    def _install_simple_stub(self, originals, module_name: str, **attrs) -> None:
        stub_module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(stub_module, attr_name, value)
        sys.modules[module_name] = stub_module

    def _restore_modules(self, originals: dict[str, types.ModuleType | None]) -> None:
        for module_name, original_module in originals.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module


class MainWindowRegistryDispatchTests(MainWindowImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.registry, self.main_window = self.import_main_window()

    def test_tracker_spec_lookup_uses_registry_for_selected_scenario(self) -> None:
        fake = SimpleNamespace(tracking_scenario_combo=_DummyCombo("  Offline ToMP  "))
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )
        fake._current_tracker_spec = types.MethodType(
            self.main_window.MainWindow._current_tracker_spec,
            fake,
        )

        spec = fake._current_tracker_spec()

        self.assertIs(spec, self.registry.require_tracker_spec("tomp_offline"))

    def test_collect_tracker_params_for_spec_calls_registered_method(self) -> None:
        spec = self.registry.require_tracker_spec("tapir_offline")
        fake = SimpleNamespace(
            _collect_tapir_tracking_params=lambda: {"backend": "tapir", "mode": "offline"}
        )

        result = self.main_window.MainWindow._collect_tracker_params_for_spec(fake, spec)

        self.assertEqual(result, {"backend": "tapir", "mode": "offline"})

    def test_collect_tracker_params_for_spec_raises_for_missing_method(self) -> None:
        spec = self.registry.require_tracker_spec("tapir_offline")

        with self.assertRaises(AttributeError):
            self.main_window.MainWindow._collect_tracker_params_for_spec(SimpleNamespace(), spec)

    def test_collect_tracking_dialog_backend_params_deduplicates_argument_names(self) -> None:
        calls: list[str] = []

        def _collect(tracker_spec):
            calls.append(tracker_spec.tracker_id)
            return {"tracker_id": tracker_spec.tracker_id}

        fake = SimpleNamespace(_collect_tracker_params_for_spec=_collect)

        params = self.main_window.MainWindow._collect_tracking_dialog_backend_params(fake)

        expected_argument_names = []
        seen_argument_names = set()
        for spec in self.registry.iter_tracker_specs():
            argument_name = str(spec.params_argument_name or "").strip()
            if not argument_name or argument_name in seen_argument_names:
                continue
            seen_argument_names.add(argument_name)
            expected_argument_names.append(argument_name)

        self.assertEqual(set(params), set(expected_argument_names))
        self.assertEqual(len(calls), len(expected_argument_names))
        self.assertIn("tapir_offline", calls)
        self.assertNotIn("tapir_online", calls)

    def test_run_tracking_dispatches_registered_tracker_via_registry_worker_builder(self) -> None:
        recorded: dict[str, object] = {}
        fake = SimpleNamespace(
            tracking_scenario_combo=_DummyCombo("Offline TAPIR"),
        )
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )

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

    def test_run_tracking_stops_when_registered_worker_builder_returns_none(self) -> None:
        recorded: dict[str, object] = {}
        fake = SimpleNamespace(
            tracking_scenario_combo=_DummyCombo("Offline TAPIR"),
        )
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )
        fake._build_registered_tracking_worker = lambda tracker_spec, cancel_event: None
        fake._launch_worker_with_progress = lambda *args, **kwargs: recorded.update({"launched": True})

        self.main_window.MainWindow.run_tracking(fake)

        self.assertNotIn("launched", recorded)
