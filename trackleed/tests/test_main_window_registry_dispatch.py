from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
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
            "trackleed.analysis.ostrack_backend",
            "trackleed.analysis.dam4sam_backend",
            "trackleed.analysis.cutie_backend",
            "trackleed.analysis.xmem2_backend",
            "trackleed.analysis.tomp_backend",
            "trackleed.analysis.sam2_backend",
            "trackleed.analysis.samurai_backend",
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
            repair_duplicate_frame_ids=_stub_callable(
                "trackleed.io.file_parser.repair_duplicate_frame_ids"
            ),
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
            ("trackleed.analysis.deva_backend", "deva"),
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
            "trackleed.analysis.ostrack_backend",
            DEFAULT_OSTRACK_CONFIG_NAME="vitb_256_mae_ce_32x4_ep300",
            SUPPORTED_OSTRACK_CONFIG_NAMES=(
                "vitb_256_mae_ce_32x4_ep300",
                "vitb_384_mae_ce_32x4_ep300",
                "vitb_256_mae_32x4_ep300",
                "vitb_384_mae_32x4_ep300",
                "vitb_256_mae_ce_32x4_got10k_ep100",
                "vitb_384_mae_ce_32x4_got10k_ep100",
            ),
            format_ostrack_environment_report=lambda status: f"ostrack:{status}",
            probe_ostrack_environment=_stub_callable(
                "trackleed.analysis.ostrack_backend.probe_ostrack_environment"
            ),
            select_default_ostrack_checkpoint=lambda checkpoints, config_name=None: None,
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.dam4sam_backend",
            DAM4SAM_TRACKER_VARIANTS={
                "sam21pp-L": ("sam2.1_hiera_large.pt", "sam21pp_hiera_l.yaml"),
                "sam21pp-B": ("sam2.1_hiera_base_plus.pt", "sam21pp_hiera_b+.yaml"),
            },
            DEFAULT_DAM4SAM_TRACKER_NAME="sam21pp-L",
            format_dam4sam_environment_report=lambda status: f"dam4sam:{status}",
            probe_dam4sam_environment=_stub_callable(
                "trackleed.analysis.dam4sam_backend.probe_dam4sam_environment"
            ),
            select_default_dam4sam_checkpoint=lambda checkpoints, tracker_name=None, models_dir=None: (
                checkpoints[0] if checkpoints else None
            ),
        )
        self._install_simple_stub(
            originals,
            "trackleed.analysis.samurai_backend",
            SAMURAI_MODEL_VARIANTS={
                "sam2.1_hiera_large": (
                    "sam2.1_hiera_large.pt",
                    "sam2.1_hiera_l.yaml",
                    "configs/samurai/sam2.1_hiera_l.yaml",
                ),
                "sam2.1_hiera_base_plus": (
                    "sam2.1_hiera_base_plus.pt",
                    "sam2.1_hiera_b+.yaml",
                    "configs/samurai/sam2.1_hiera_b+.yaml",
                ),
                "sam2.1_hiera_small": (
                    "sam2.1_hiera_small.pt",
                    "sam2.1_hiera_s.yaml",
                    "configs/samurai/sam2.1_hiera_s.yaml",
                ),
                "sam2.1_hiera_tiny": (
                    "sam2.1_hiera_tiny.pt",
                    "sam2.1_hiera_t.yaml",
                    "configs/samurai/sam2.1_hiera_t.yaml",
                ),
            },
            DEFAULT_SAMURAI_MODEL_VARIANT="sam2.1_hiera_base_plus",
            format_samurai_environment_report=lambda status: f"samurai:{status}",
            probe_samurai_environment=_stub_callable(
                "trackleed.analysis.samurai_backend.probe_samurai_environment"
            ),
            select_default_samurai_checkpoint=lambda checkpoints, model_variant=None, models_dir=None: (
                checkpoints[0] if checkpoints else None
            ),
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
            track_spots_ostrack_offline=_stub_callable(
                "trackleed.analysis.tracking_bbox_backends.track_spots_ostrack_offline"
            ),
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
            track_spots_deva_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_deva_offline"
            ),
            track_spots_dam4sam_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_dam4sam_offline"
            ),
            track_spots_sam2_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_sam2_offline"
            ),
            track_spots_samurai_offline=_stub_callable(
                "trackleed.analysis.tracking_mask_backends.track_spots_samurai_offline"
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
            retrack_single_spot_dam4sam_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_dam4sam_offline"
            ),
            retrack_single_spot_deva_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_deva_offline"
            ),
            retrack_single_spot_locotrack_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_locotrack_offline"
            ),
            retrack_single_spot_mftiq_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_mftiq_offline"
            ),
            retrack_single_spot_ostrack_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_ostrack_offline"
            ),
            retrack_single_spot_pipspp_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_pipspp_offline"
            ),
            retrack_single_spot_sam2_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_sam2_offline"
            ),
            retrack_single_spot_samurai_offline=_stub_callable(
                "trackleed.analysis.tracking_retrack.retrack_single_spot_samurai_offline"
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
            thread=None,
        )
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )
        fake._tracking_worker_is_active = types.MethodType(
            self.main_window.MainWindow._tracking_worker_is_active,
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
            thread=None,
        )
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )
        fake._tracking_worker_is_active = types.MethodType(
            self.main_window.MainWindow._tracking_worker_is_active,
            fake,
        )
        fake._build_registered_tracking_worker = lambda tracker_spec, cancel_event: None
        fake._launch_worker_with_progress = lambda *args, **kwargs: recorded.update({"launched": True})

        self.main_window.MainWindow.run_tracking(fake)

        self.assertNotIn("launched", recorded)

    def test_detection_linking_repairs_duplicate_frame_ids_before_dispatch(self) -> None:
        recorded: dict[str, object] = {}
        original_generic_worker = self.main_window.GenericWorker
        original_repair_duplicate_frame_ids = self.main_window.repair_duplicate_frame_ids
        self.addCleanup(setattr, self.main_window, "GenericWorker", original_generic_worker)
        self.addCleanup(
            setattr,
            self.main_window,
            "repair_duplicate_frame_ids",
            original_repair_duplicate_frame_ids,
        )

        class _GenericWorker:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        def _repair_duplicate_frame_ids(files):
            files[0].id = 14
            files[1].id = 15
            return "temperature"

        self.main_window.GenericWorker = _GenericWorker
        self.main_window.repair_duplicate_frame_ids = _repair_duplicate_frame_ids

        fake = SimpleNamespace(
            tracking_scenario_combo=_DummyCombo("Detection Linking (Hungarian)"),
            thread=None,
            data_files=[
                SimpleNamespace(id=0, filename="a_T0014_E1100_C0000.tiff", spots=[object()]),
                SimpleNamespace(id=0, filename="b_T0015_E1100_C0000.tiff", spots=[object()]),
            ],
            hung_max_dist_spinbox=SimpleNamespace(value=lambda: 20.0),
            hung_max_gap_spinbox=SimpleNamespace(value=lambda: 2),
            hung_min_len_spinbox=SimpleNamespace(value=lambda: 2),
        )
        fake._tracker_spec_for_scenario = types.MethodType(
            self.main_window.MainWindow._tracker_spec_for_scenario,
            fake,
        )
        fake._tracking_worker_is_active = types.MethodType(
            self.main_window.MainWindow._tracking_worker_is_active,
            fake,
        )
        fake._build_detection_linking_file_map = types.MethodType(
            self.main_window.MainWindow._build_detection_linking_file_map,
            fake,
        )
        fake._launch_worker_with_progress = (
            lambda worker, title, cancel_event: recorded.update(
                {"worker": worker, "title": title, "cancel_event": cancel_event}
            )
        )

        self.main_window.MainWindow.run_tracking(fake)

        worker = recorded["worker"]
        self.assertEqual(recorded["title"], "Running tracking...")
        self.assertEqual(worker.args[2], [14, 15])
        self.assertEqual(set(worker.args[1]), {14, 15})
        self.assertEqual([file.id for file in fake.data_files], [14, 15])

    def test_run_tracking_blocks_when_previous_tracking_is_still_active(self) -> None:
        recorded: dict[str, object] = {}
        original_information = self.main_window.QMessageBox.information
        self.addCleanup(setattr, self.main_window.QMessageBox, "information", original_information)
        self.main_window.QMessageBox.information = (
            lambda *_args: recorded.setdefault("message", _args[2])
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

    def test_clear_active_tracking_worker_only_clears_matching_objects(self) -> None:
        fake = SimpleNamespace(
            thread="new-thread",
            worker="new-worker",
            _tracking_progress_dialog="new-dialog",
            _active_tracking_cancel_requested=True,
        )

        self.main_window.MainWindow._clear_active_tracking_worker(
            fake,
            worker_thread="old-thread",
            worker="old-worker",
            progress_dialog="old-dialog",
        )
        self.assertEqual(fake.thread, "new-thread")
        self.assertEqual(fake.worker, "new-worker")
        self.assertEqual(fake._tracking_progress_dialog, "new-dialog")
        self.assertTrue(fake._active_tracking_cancel_requested)

        self.main_window.MainWindow._clear_active_tracking_worker(
            fake,
            worker_thread="new-thread",
            worker="new-worker",
            progress_dialog="new-dialog",
        )
        self.assertIsNone(fake.thread)
        self.assertIsNone(fake.worker)
        self.assertIsNone(fake._tracking_progress_dialog)
        self.assertFalse(fake._active_tracking_cancel_requested)

    def test_handle_worker_cancel_switches_dialog_into_busy_cancelling_state(self) -> None:
        calls: list[tuple[str, object]] = []
        original_process_events = getattr(self.main_window.QApplication, "processEvents", None)
        self.addCleanup(
            setattr,
            self.main_window.QApplication,
            "processEvents",
            original_process_events if original_process_events is not None else _Dummy,
        )
        self.main_window.QApplication.processEvents = lambda: calls.append(("process_events", True))

        class _CancelEvent:
            def __init__(self) -> None:
                self.was_set = False

            def set(self) -> None:
                self.was_set = True

        class _Dialog:
            def setWindowTitle(self, value):
                calls.append(("title", value))

            def setLabelText(self, value):
                calls.append(("label", value))

            def setRange(self, minimum, maximum):
                calls.append(("range", (minimum, maximum)))

            def setValue(self, value):
                calls.append(("value", value))

            def setCancelButton(self, value):
                calls.append(("cancel_button", value))

            def setEnabled(self, value):
                calls.append(("enabled", value))

        fake = SimpleNamespace(
            thread=SimpleNamespace(requestInterruption=lambda: calls.append(("interrupt", True))),
            _active_tracking_cancel_requested=False,
        )
        fake._set_tracking_cancel_progress_state = types.MethodType(
            self.main_window.MainWindow._set_tracking_cancel_progress_state,
            fake,
        )

        cancel_event = _CancelEvent()
        dialog = _Dialog()

        self.main_window.MainWindow._handle_worker_cancel(fake, cancel_event, dialog)

        self.assertTrue(cancel_event.was_set)
        self.assertTrue(fake._active_tracking_cancel_requested)
        self.assertIn(("title", "Cancelling Tracking"), calls)
        self.assertIn(("label", "Cancelling tracking, please wait..."), calls)
        self.assertIn(("range", (0, 0)), calls)
        self.assertIn(("value", 0), calls)
        self.assertIn(("cancel_button", None), calls)
        self.assertIn(("enabled", True), calls)
        self.assertIn(("process_events", True), calls)
        self.assertIn(("interrupt", True), calls)

    def test_launch_worker_with_progress_disables_qprogressdialog_autoclose(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Signal:
            def __init__(self) -> None:
                self.callbacks = []

            def connect(self, callback):
                self.callbacks.append(callback)

        class _ProgressDialog:
            def __init__(self, title, cancel_text, minimum, maximum, parent):
                calls.append(("init", (title, cancel_text, minimum, maximum, parent)))
                self.canceled = _Signal()

            def setWindowModality(self, value):
                calls.append(("modality", value))

            def setAutoClose(self, value):
                calls.append(("auto_close", value))

            def setAutoReset(self, value):
                calls.append(("auto_reset", value))

            def show(self):
                calls.append(("show", True))

            def accept(self):
                calls.append(("accept", True))

            def setValue(self, value):
                calls.append(("value", value))

        class _Thread:
            def __init__(self) -> None:
                self.started = _Signal()
                self.finished = _Signal()

            def start(self):
                calls.append(("thread_start", True))

            def quit(self):
                calls.append(("thread_quit", True))

            def deleteLater(self):
                calls.append(("thread_delete_later", True))

        class _Worker:
            def __init__(self) -> None:
                self.progress = _Signal()
                self.finished = _Signal()
                self.error = _Signal()

            def moveToThread(self, thread):
                calls.append(("move_to_thread", thread))

            def deleteLater(self):
                calls.append(("worker_delete_later", True))

            def run(self):
                calls.append(("worker_run", True))

        original_progress_dialog = self.main_window.QProgressDialog
        original_qthread = self.main_window.QThread
        self.addCleanup(setattr, self.main_window, "QProgressDialog", original_progress_dialog)
        self.addCleanup(setattr, self.main_window, "QThread", original_qthread)
        self.main_window.QProgressDialog = _ProgressDialog
        self.main_window.QThread = _Thread

        fake = SimpleNamespace(
            thread=None,
            worker=None,
            _tracking_progress_dialog=None,
            _active_tracking_cancel_requested=False,
            on_tracking_finished=lambda *_args, **_kwargs: None,
            on_tracking_error=lambda *_args, **_kwargs: None,
        )
        fake._handle_worker_cancel = types.MethodType(
            self.main_window.MainWindow._handle_worker_cancel,
            fake,
        )
        fake._clear_active_tracking_worker = types.MethodType(
            self.main_window.MainWindow._clear_active_tracking_worker,
            fake,
        )

        worker = _Worker()
        cancel_event = SimpleNamespace(set=lambda: calls.append(("cancel_set", True)))

        self.main_window.MainWindow._launch_worker_with_progress(
            fake,
            worker,
            "Running tracking...",
            cancel_event,
        )

        self.assertIn(("auto_close", False), calls)
        self.assertIn(("auto_reset", False), calls)
        self.assertIn(("show", True), calls)
        self.assertIs(fake.worker, worker)
        self.assertIsNotNone(fake.thread)
        self.assertIsNotNone(fake._tracking_progress_dialog)

    def test_default_subprocess_python_executables_are_filled_from_conda_envs(self) -> None:
        self.assertEqual(self.main_window.CutieTrackingSettings().backend_mode, "subprocess")

        with tempfile.TemporaryDirectory() as temp_dir:
            conda_root = Path(temp_dir)
            locotrack_python = conda_root / "envs" / "locotrack-pytorch" / "python.exe"
            sam2_python = conda_root / "envs" / "sam2" / "python.exe"
            dam4sam_python = conda_root / "envs" / "dam4sam" / "python.exe"
            samurai_python = conda_root / "envs" / "samurai" / "python.exe"
            locotrack_python.parent.mkdir(parents=True)
            sam2_python.parent.mkdir(parents=True)
            dam4sam_python.parent.mkdir(parents=True)
            samurai_python.parent.mkdir(parents=True)
            locotrack_python.touch()
            sam2_python.touch()
            dam4sam_python.touch()
            samurai_python.touch()

            fake = SimpleNamespace(
                locotrack_settings=self.main_window.LocotrackTrackingSettings(),
                sam2_settings=self.main_window.Sam2TrackingSettings(
                    python_executable="C:/custom/sam2/python.exe"
                ),
                dam4sam_settings=self.main_window.Dam4SamTrackingSettings(),
                samurai_settings=self.main_window.SamuraiTrackingSettings(),
            )
            fake._tracker_settings_for_spec = types.MethodType(
                self.main_window.MainWindow._tracker_settings_for_spec,
                fake,
            )
            fake._candidate_conda_roots = lambda: [conda_root]
            fake._default_conda_python_executable = types.MethodType(
                self.main_window.MainWindow._default_conda_python_executable,
                fake,
            )

            self.main_window.MainWindow._apply_default_subprocess_python_executables(fake)

            self.assertEqual(
                fake.locotrack_settings.python_executable,
                str(locotrack_python.resolve()),
            )
            self.assertEqual(
                fake.sam2_settings.python_executable,
                "C:/custom/sam2/python.exe",
            )
            self.assertEqual(
                fake.dam4sam_settings.python_executable,
                str(dam4sam_python.resolve()),
            )
            self.assertEqual(
                fake.samurai_settings.python_executable,
                str(samurai_python.resolve()),
            )

    def test_ostrack_variant_options_and_presets_cover_supported_configs(self) -> None:
        supported = set(self.main_window.SUPPORTED_OSTRACK_CONFIG_NAMES)

        self.assertEqual(set(self.main_window.OSTRACK_MODEL_VARIANT_LABELS), supported)
        self.assertEqual(set(self.main_window.OSTRACK_VARIANT_PRESETS), supported)
        self.assertIn("dim_low_contrast", self.main_window.OSTRACK_SCENARIO_PRESETS)
        self.assertIn("crowded_spots", self.main_window.OSTRACK_SCENARIO_PRESETS)
        self.assertIn("low_vram", self.main_window.OSTRACK_SCENARIO_PRESETS)

        for variant_key, preset in self.main_window.OSTRACK_VARIANT_PRESETS.items():
            with self.subTest(variant_key=variant_key):
                self.assertIn(
                    preset["representative_point_mode"],
                    {"bbox_center", "peak_in_bbox", "gaussian_in_bbox"},
                )
                self.assertGreaterEqual(int(preset["window_size"]), 2)
                self.assertGreaterEqual(int(preset["window_overlap"]), 0)
                self.assertFalse(bool(preset["use_presence_score_gating"]))

    def test_samurai_variant_options_cover_supported_configs(self) -> None:
        supported = set(self.main_window.SAMURAI_MODEL_VARIANTS)

        self.assertEqual(set(self.main_window.SAMURAI_MODEL_VARIANT_LABELS), supported)
        self.assertEqual(
            self.main_window.SamuraiTrackingSettings().model_variant,
            self.main_window.DEFAULT_SAMURAI_MODEL_VARIANT,
        )
