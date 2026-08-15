from __future__ import annotations

import importlib
import sys
import types
import unittest
from types import SimpleNamespace


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


class _Qt:
    class WindowModality:
        WindowModal = 0


class _GenericWorkerStub:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.progress = _Dummy()
        self.finished = _Dummy()
        self.error = _Dummy()

    def moveToThread(self, *_args, **_kwargs):
        return None


class TrackingDialogImportMixin:
    module_name = "trackleed.gui.tracking_dialog"

    def import_tracking_dialog(self):
        originals: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            "PyQt6.QtWidgets",
            "trackleed.gui.leed_image_viewer",
            "trackleed.gui.polygon_mask_dialog",
            "trackleed.gui.tracker_registry",
            "trackleed.io.image_reader",
            "trackleed.analysis.tomp_backend",
            "trackleed.analysis.tracking_retrack",
            "trackleed.logic.generic_worker",
            self.module_name,
        ):
            originals[module_name] = sys.modules.get(module_name)
            sys.modules.pop(module_name, None)

        qtcore_module = types.ModuleType("PyQt6.QtCore")
        qtcore_module.Qt = _Qt
        qtcore_module.QTimer = _Dummy
        qtcore_module.QPoint = _Dummy
        qtcore_module.QPointF = _Dummy
        qtcore_module.QRect = _Dummy
        qtcore_module.QThread = _Dummy

        qtgui_module = types.ModuleType("PyQt6.QtGui")
        for attr_name in ("QImage", "QPixmap", "QActionGroup", "QAction", "QColor"):
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
            "QDialog",
            "QWidget",
            "QHBoxLayout",
            "QVBoxLayout",
            "QSlider",
            "QSplitter",
            "QLabel",
            "QTableWidget",
            "QAbstractItemView",
            "QTableWidgetItem",
            "QDialogButtonBox",
            "QHeaderView",
            "QPushButton",
            "QToolBar",
            "QMenu",
            "QProgressDialog",
            "QComboBox",
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

        tracker_specs = {
            "tapir_offline": SimpleNamespace(
                tracker_id="tapir_offline",
                params_argument_name="tapir_params",
                run_retrack=lambda *args, **kwargs: ("tapir_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "tapir_online": SimpleNamespace(
                tracker_id="tapir_online",
                params_argument_name="tapir_params",
                run_retrack=lambda *args, **kwargs: ("tapir_online", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "tomp_offline": SimpleNamespace(
                tracker_id="tomp_offline",
                params_argument_name="tomp_params",
                run_retrack=lambda *args, **kwargs: ("tomp_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "sam2_offline": SimpleNamespace(
                tracker_id="sam2_offline",
                params_argument_name="sam2_params",
                run_retrack=lambda *args, **kwargs: ("sam2_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "xmem2_offline": SimpleNamespace(
                tracker_id="xmem2_offline",
                params_argument_name="xmem2_params",
                run_retrack=lambda *args, **kwargs: ("xmem2_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "deva_offline": SimpleNamespace(
                tracker_id="deva_offline",
                params_argument_name="deva_params",
                run_retrack=lambda *args, **kwargs: ("deva_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "dam4sam_offline": SimpleNamespace(
                tracker_id="dam4sam_offline",
                params_argument_name="dam4sam_params",
                run_retrack=lambda *args, **kwargs: ("dam4sam_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "samurai_offline": SimpleNamespace(
                tracker_id="samurai_offline",
                params_argument_name="samurai_params",
                run_retrack=lambda *args, **kwargs: ("samurai_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
            "ostrack_offline": SimpleNamespace(
                tracker_id="ostrack_offline",
                params_argument_name="ostrack_params",
                run_retrack=lambda *args, **kwargs: ("ostrack_offline", args, kwargs),
                has_visualizer_retrack=True,
            ),
        }

        source_to_spec = {
            "tapir_offline": tracker_specs["tapir_offline"],
            "tapir_online": tracker_specs["tapir_online"],
            "tomp_offline": tracker_specs["tomp_offline"],
            "tamos_offline": tracker_specs["tomp_offline"],
            "sam2_offline": tracker_specs["sam2_offline"],
            "xmem2_offline": tracker_specs["xmem2_offline"],
            "deva_offline": tracker_specs["deva_offline"],
            "dam4sam_offline": tracker_specs["dam4sam_offline"],
            "samurai_offline": tracker_specs["samurai_offline"],
            "ostrack_offline": tracker_specs["ostrack_offline"],
        }

        tracker_registry_module = types.ModuleType("trackleed.gui.tracker_registry")
        tracker_registry_module.get_tracker_spec_for_source = (
            lambda source_name: source_to_spec.get(str(source_name or "").strip().lower())
        )
        tracker_registry_module.require_tracker_spec = (
            lambda tracker_id: tracker_specs[str(tracker_id).strip().lower()]
        )
        sys.modules["trackleed.gui.tracker_registry"] = tracker_registry_module

        viewer_module = types.ModuleType("trackleed.gui.leed_image_viewer")
        viewer_module.LeedImageViewer = _Dummy
        sys.modules["trackleed.gui.leed_image_viewer"] = viewer_module

        polygon_module = types.ModuleType("trackleed.gui.polygon_mask_dialog")
        polygon_module.PolygonMaskDialog = _Dummy
        sys.modules["trackleed.gui.polygon_mask_dialog"] = polygon_module

        image_reader_module = types.ModuleType("trackleed.io.image_reader")
        image_reader_module.load_tiff = lambda *_args, **_kwargs: None
        sys.modules["trackleed.io.image_reader"] = image_reader_module

        tomp_backend_module = types.ModuleType("trackleed.analysis.tomp_backend")
        tomp_backend_module.select_default_tamos_checkpoint_for_variant = (
            lambda checkpoints, variant: None
        )
        tomp_backend_module.select_default_tomp_checkpoint_for_variant = (
            lambda checkpoints, variant: None
        )
        sys.modules["trackleed.analysis.tomp_backend"] = tomp_backend_module

        retrack_module = types.ModuleType("trackleed.analysis.tracking_retrack")
        retrack_module.retrack_single_spot_optical_flow = lambda *args, **kwargs: None
        sys.modules["trackleed.analysis.tracking_retrack"] = retrack_module

        generic_worker_module = types.ModuleType("trackleed.logic.generic_worker")
        generic_worker_module.GenericWorker = _GenericWorkerStub
        sys.modules["trackleed.logic.generic_worker"] = generic_worker_module

        self.addCleanup(self._restore_modules, originals)
        self.addCleanup(sys.modules.pop, self.module_name, None)

        module = importlib.import_module(self.module_name)
        return module, tracker_specs

    def _restore_modules(self, originals: dict[str, types.ModuleType | None]) -> None:
        for module_name, original_module in originals.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module


class TrackingDialogRetrackDispatchTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _spot(self, source: str):
        return SimpleNamespace(source=source, id=1)

    def _track(self, *sources: str):
        return SimpleNamespace(
            track_id=7,
            spots_by_image_id={
                idx + 1: self._spot(source)
                for idx, source in enumerate(sources)
            },
        )

    def test_tracker_spec_lookup_for_source_and_track(self) -> None:
        dialog = SimpleNamespace()
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._tracker_spec_for_source_name = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_source_name,
            dialog,
        )
        dialog._tracker_spec_for_track = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_track,
            dialog,
        )

        self.assertIsNone(dialog._tracker_spec_for_source_name("manual_corr"))
        self.assertIs(
            dialog._tracker_spec_for_source_name("  tamos_offline  "),
            self.tracker_specs["tomp_offline"],
        )
        self.assertIs(
            dialog._tracker_spec_for_track(self._track("tomp_offline", "manual_corr")),
            self.tracker_specs["tomp_offline"],
        )
        self.assertIsNone(
            dialog._tracker_spec_for_track(self._track("tomp_offline", "sam2_offline"))
        )

    def test_backend_retrack_action_text_uses_base_method_label(self) -> None:
        dialog = SimpleNamespace()
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._tracker_spec_for_source_name = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_source_name,
            dialog,
        )
        dialog._tracker_spec_for_track = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_track,
            dialog,
        )
        dialog._method_label_for_source_name = types.MethodType(
            self.module.TrackingDialog._method_label_for_source_name,
            dialog,
        )
        dialog._classify_track_method = types.MethodType(
            self.module.TrackingDialog._classify_track_method,
            dialog,
        )
        dialog._base_method_label_for_track = types.MethodType(
            self.module.TrackingDialog._base_method_label_for_track,
            dialog,
        )
        dialog._backend_retrack_action_text = types.MethodType(
            self.module.TrackingDialog._backend_retrack_action_text,
            dialog,
        )

        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("tamos_offline", "manual_corr")),
            "Re-track from this point (TaMOs)",
        )
        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("ostrack_offline", "manual_corr")),
            "Re-track from this point (OSTrack)",
        )
        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("dam4sam_offline", "manual_corr")),
            "Re-track from this point (DAM4SAM)",
        )
        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("samurai_offline", "manual_corr")),
            "Re-track from this point (SAMURAI)",
        )
        self.assertIsNone(dialog._backend_retrack_action_text(self._track("manual_corr")))

    def test_tracker_params_for_spec_returns_dict_copy(self) -> None:
        dialog = SimpleNamespace(
            tomp_params={"backend_mode": "subprocess", "model_variant": "tomp50"}
        )

        params = self.module.TrackingDialog._tracker_params_for_spec(
            dialog,
            self.tracker_specs["tomp_offline"],
        )
        params["model_variant"] = "changed"

        self.assertEqual(dialog.tomp_params["model_variant"], "tomp50")
        self.assertEqual(params["backend_mode"], "subprocess")

    def test_launch_registered_retrack_builds_worker_and_delegates_launch(self) -> None:
        recorded: dict[str, object] = {}
        dialog = SimpleNamespace(
            file_map={1: "frame1"},
            sorted_frame_ids=[1, 2, 3],
        )

        def launch_retrack_worker(worker, *, title, cancel_event, method_label):
            recorded["worker"] = worker
            recorded["title"] = title
            recorded["cancel_event"] = cancel_event
            recorded["method_label"] = method_label

        dialog.launch_retrack_worker = launch_retrack_worker
        track = self._track("tomp_offline")

        self.module.TrackingDialog._launch_registered_retrack(
            dialog,
            self.tracker_specs["tomp_offline"],
            track,
            2,
            {"roi_size": 32},
            title="Re-track with ToMP",
            method_label="ToMP re-tracking",
        )

        worker = recorded["worker"]
        self.assertIsInstance(worker, _GenericWorkerStub)
        self.assertIs(worker.function, self.tracker_specs["tomp_offline"].run_retrack)
        self.assertEqual(worker.args[0], track)
        self.assertEqual(worker.args[1], 2)
        self.assertEqual(worker.args[2], {1: "frame1"})
        self.assertEqual(worker.args[3], [1, 2, 3])
        self.assertEqual(worker.args[4], {"roi_size": 32})
        self.assertEqual(recorded["title"], "Re-track with ToMP")
        self.assertEqual(recorded["method_label"], "ToMP re-tracking")
        self.assertIs(worker.kwargs["cancel_event"], recorded["cancel_event"])

    def test_initiate_retrack_for_tracker_spec_dispatches_to_backend_specific_method(self) -> None:
        calls: list[tuple[str, int]] = []
        self.module.launch_seeded_point_retrack = (
            lambda _dialog, method_id, spot_id: calls.append((method_id, spot_id))
        )
        dialog = SimpleNamespace(
            initiate_retrack_cutie=lambda spot_id: calls.append(("cutie", spot_id)),
            initiate_retrack_xmem2=lambda spot_id: calls.append(("xmem2", spot_id)),
            initiate_retrack_tomp=lambda spot_id: calls.append(("tomp", spot_id)),
            initiate_retrack_ostrack=lambda spot_id: calls.append(("ostrack", spot_id)),
            initiate_retrack_sam2=lambda spot_id: calls.append(("sam2", spot_id)),
            initiate_retrack_dam4sam=lambda spot_id: calls.append(("dam4sam", spot_id)),
            initiate_retrack_samurai=lambda spot_id: calls.append(("samurai", spot_id)),
            initiate_retrack_deva=lambda spot_id: calls.append(("deva", spot_id)),
        )

        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["tapir_online"],
            11,
        )
        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["tomp_offline"],
            22,
        )
        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["sam2_offline"],
            33,
        )
        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["deva_offline"],
            44,
        )
        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["ostrack_offline"],
            55,
        )
        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["dam4sam_offline"],
            66,
        )
        self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
            dialog,
            self.tracker_specs["samurai_offline"],
            77,
        )

        self.assertEqual(
            calls,
            [
                ("tapir_online", 11),
                ("tomp", 22),
                ("sam2", 33),
                ("deva", 44),
                ("ostrack", 55),
                ("dam4sam", 66),
                ("samurai", 77),
            ],
        )

    def test_initiate_retrack_for_tracker_spec_rejects_unknown_tracker(self) -> None:
        dialog = SimpleNamespace()

        with self.assertRaises(ValueError):
            self.module.TrackingDialog._initiate_retrack_for_tracker_spec(
                dialog,
                SimpleNamespace(tracker_id="unknown_tracker"),
                5,
            )

    def test_initiate_retrack_ostrack_launches_registered_worker_with_validated_checkpoint(self) -> None:
        recorded: dict[str, object] = {}
        probe_kwargs: dict[str, object] = {}

        class _Checkpoint:
            def exists(self) -> bool:
                return True

            def resolve(self) -> str:
                return "C:/models/OSTrack_ep0300.pth.tar"

        tracker_spec = SimpleNamespace(
            tracker_id="ostrack_offline",
            params_argument_name="ostrack_params",
            supports_subprocess=True,
            supports_direct=True,
            run_retrack=lambda *args, **kwargs: None,
        )
        status = SimpleNamespace(
            repo_path="C:/ostrack",
            config_exists=True,
            selected_checkpoint=_Checkpoint(),
            is_ready=True,
        )
        params = {
            "backend_mode": "subprocess",
            "model_variant": "vitb_256_mae_ce_32x4_ep300",
            "checkpoint_path": "tracking_models/ostrack/OSTrack_ep0300.pth.tar",
            "python_executable": "C:/Users/rlewa/anaconda3/envs/ostrack/python.exe",
            "ostrack_models_dir": "tracking_models/ostrack",
            "repo_path": "C:/ostrack",
        }
        track = SimpleNamespace(
            track_id=7,
            spots_by_image_id={
                2: SimpleNamespace(
                    source="ostrack_offline",
                    ostrack_model_variant="vitb_256_mae_ce_32x4_ep300",
                )
            },
        )

        def launch_retrack(tracker_spec_arg, track_arg, frame_id_arg, params_arg, *, title, method_label):
            recorded["tracker_spec"] = tracker_spec_arg
            recorded["track"] = track_arg
            recorded["frame_id"] = frame_id_arg
            recorded["params"] = params_arg
            recorded["title"] = title
            recorded["method_label"] = method_label

        dialog = SimpleNamespace(
            roi_size=32,
            _get_retrack_target=lambda: (track, 2),
            _tracker_params_for_spec=lambda spec: dict(params),
            _probe_tracker_environment_for_spec=lambda spec, **kwargs: (
                probe_kwargs.update(kwargs) or status
            ),
            _ensure_registered_retrack_runtime_ready=lambda *args, **kwargs: True,
            _launch_registered_retrack=launch_retrack,
        )
        original_require_tracker_spec = self.module.require_tracker_spec
        self.module.require_tracker_spec = lambda tracker_id: tracker_spec
        self.addCleanup(setattr, self.module, "require_tracker_spec", original_require_tracker_spec)

        self.module.TrackingDialog.initiate_retrack_ostrack(dialog, 1)

        self.assertIs(recorded["tracker_spec"], tracker_spec)
        self.assertIs(recorded["track"], track)
        self.assertEqual(recorded["frame_id"], 2)
        self.assertEqual(recorded["title"], "Re-tracking spot with OSTrack...")
        self.assertEqual(recorded["method_label"], "OSTrack re-tracking")
        self.assertEqual(probe_kwargs["config_name"], "vitb_256_mae_ce_32x4_ep300")
        launched_params = recorded["params"]
        self.assertEqual(launched_params["checkpoint_path"], "C:/models/OSTrack_ep0300.pth.tar")
        self.assertEqual(launched_params["roi_size"], 32)
        self.assertEqual(launched_params["model_variant"], "vitb_256_mae_ce_32x4_ep300")

    def test_initiate_retrack_dam4sam_launches_registered_worker_with_validated_checkpoint(self) -> None:
        recorded: dict[str, object] = {}
        probe_kwargs: dict[str, object] = {}

        class _Checkpoint:
            def exists(self) -> bool:
                return True

            def resolve(self) -> str:
                return "C:/models/sam2.1_hiera_large.pt"

        tracker_spec = SimpleNamespace(
            tracker_id="dam4sam_offline",
            params_argument_name="dam4sam_params",
            supports_subprocess=True,
            supports_direct=True,
            run_retrack=lambda *args, **kwargs: None,
        )
        status = SimpleNamespace(
            repo_path="C:/dam4sam",
            config_exists=True,
            selected_checkpoint=_Checkpoint(),
            is_ready=True,
        )
        params = {
            "backend_mode": "subprocess",
            "dam4sam_tracker_name": "sam21pp-L",
            "checkpoint_path": "tracking_models/dam4sam/sam2.1_hiera_large.pt",
            "python_executable": "C:/Users/rlewa/anaconda3/envs/dam4sam/python.exe",
            "dam4sam_models_dir": "tracking_models/dam4sam",
            "repo_path": "C:/dam4sam",
        }
        track = SimpleNamespace(
            track_id=7,
            spots_by_image_id={
                2: SimpleNamespace(
                    source="dam4sam_offline",
                    dam4sam_tracker_name="sam21pp-L",
                )
            },
        )

        def launch_retrack(tracker_spec_arg, track_arg, frame_id_arg, params_arg, *, title, method_label):
            recorded["tracker_spec"] = tracker_spec_arg
            recorded["track"] = track_arg
            recorded["frame_id"] = frame_id_arg
            recorded["params"] = params_arg
            recorded["title"] = title
            recorded["method_label"] = method_label

        dialog = SimpleNamespace(
            roi_size=32,
            _get_retrack_target=lambda: (track, 2),
            _tracker_params_for_spec=lambda spec: dict(params),
            _probe_tracker_environment_for_spec=lambda spec, **kwargs: (
                probe_kwargs.update(kwargs) or status
            ),
            _ensure_registered_retrack_runtime_ready=lambda *args, **kwargs: True,
            _launch_registered_retrack=launch_retrack,
        )
        original_require_tracker_spec = self.module.require_tracker_spec
        self.module.require_tracker_spec = lambda tracker_id: tracker_spec
        self.addCleanup(setattr, self.module, "require_tracker_spec", original_require_tracker_spec)

        self.module.TrackingDialog.initiate_retrack_dam4sam(dialog, 1)

        self.assertIs(recorded["tracker_spec"], tracker_spec)
        self.assertIs(recorded["track"], track)
        self.assertEqual(recorded["frame_id"], 2)
        self.assertEqual(recorded["title"], "Re-tracking spot with DAM4SAM...")
        self.assertEqual(recorded["method_label"], "DAM4SAM re-tracking")
        self.assertEqual(probe_kwargs["models_dir"], "tracking_models/dam4sam")
        self.assertEqual(probe_kwargs["checkpoint_path"], "tracking_models/dam4sam/sam2.1_hiera_large.pt")
        self.assertEqual(probe_kwargs["repo_path"], "C:/dam4sam")
        self.assertEqual(
            probe_kwargs["python_executable"],
            "C:/Users/rlewa/anaconda3/envs/dam4sam/python.exe",
        )
        self.assertEqual(probe_kwargs["tracker_name"], "sam21pp-L")
        launched_params = recorded["params"]
        self.assertEqual(launched_params["checkpoint_path"], "C:/models/sam2.1_hiera_large.pt")
        self.assertEqual(launched_params["roi_size"], 32)
        self.assertEqual(launched_params["tracker_name"], "sam21pp-L")
        self.assertEqual(launched_params["dam4sam_tracker_name"], "sam21pp-L")

    def test_initiate_retrack_samurai_launches_registered_worker_with_validated_checkpoint(self) -> None:
        recorded: dict[str, object] = {}
        probe_kwargs: dict[str, object] = {}

        class _Checkpoint:
            def exists(self) -> bool:
                return True

            def resolve(self) -> str:
                return "C:/models/sam2.1_hiera_base_plus.pt"

        tracker_spec = SimpleNamespace(
            tracker_id="samurai_offline",
            params_argument_name="samurai_params",
            supports_subprocess=True,
            supports_direct=True,
            run_retrack=lambda *args, **kwargs: None,
        )
        status = SimpleNamespace(
            repo_path="C:/samurai",
            config_exists=True,
            selected_checkpoint=_Checkpoint(),
            is_ready=True,
        )
        params = {
            "backend_mode": "subprocess",
            "model_variant": "sam2.1_hiera_base_plus",
            "checkpoint_path": "tracking_models/samurai/sam2.1_hiera_base_plus.pt",
            "python_executable": "C:/Users/rlewa/anaconda3/envs/samurai/python.exe",
            "samurai_models_dir": "tracking_models/samurai",
            "repo_path": "C:/samurai",
        }
        track = SimpleNamespace(
            track_id=7,
            spots_by_image_id={
                2: SimpleNamespace(
                    source="samurai_offline",
                    samurai_model_variant="sam2.1_hiera_base_plus",
                )
            },
        )

        def launch_retrack(tracker_spec_arg, track_arg, frame_id_arg, params_arg, *, title, method_label):
            recorded["tracker_spec"] = tracker_spec_arg
            recorded["track"] = track_arg
            recorded["frame_id"] = frame_id_arg
            recorded["params"] = params_arg
            recorded["title"] = title
            recorded["method_label"] = method_label

        dialog = SimpleNamespace(
            roi_size=32,
            _get_retrack_target=lambda: (track, 2),
            _tracker_params_for_spec=lambda spec: dict(params),
            _probe_tracker_environment_for_spec=lambda spec, **kwargs: (
                probe_kwargs.update(kwargs) or status
            ),
            _ensure_registered_retrack_runtime_ready=lambda *args, **kwargs: True,
            _launch_registered_retrack=launch_retrack,
        )
        original_require_tracker_spec = self.module.require_tracker_spec
        self.module.require_tracker_spec = lambda tracker_id: tracker_spec
        self.addCleanup(setattr, self.module, "require_tracker_spec", original_require_tracker_spec)

        self.module.TrackingDialog.initiate_retrack_samurai(dialog, 1)

        self.assertIs(recorded["tracker_spec"], tracker_spec)
        self.assertIs(recorded["track"], track)
        self.assertEqual(recorded["frame_id"], 2)
        self.assertEqual(recorded["title"], "Re-tracking spot with SAMURAI...")
        self.assertEqual(recorded["method_label"], "SAMURAI re-tracking")
        self.assertEqual(probe_kwargs["models_dir"], "tracking_models/samurai")
        self.assertEqual(probe_kwargs["checkpoint_path"], "tracking_models/samurai/sam2.1_hiera_base_plus.pt")
        self.assertEqual(probe_kwargs["repo_path"], "C:/samurai")
        self.assertEqual(
            probe_kwargs["python_executable"],
            "C:/Users/rlewa/anaconda3/envs/samurai/python.exe",
        )
        self.assertEqual(probe_kwargs["model_variant"], "sam2.1_hiera_base_plus")
        launched_params = recorded["params"]
        self.assertEqual(launched_params["checkpoint_path"], "C:/models/sam2.1_hiera_base_plus.pt")
        self.assertEqual(launched_params["roi_size"], 32)
        self.assertEqual(launched_params["model_variant"], "sam2.1_hiera_base_plus")
        self.assertEqual(launched_params["samurai_model_variant"], "sam2.1_hiera_base_plus")


class TrackingDialogDevaVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_mask_tracker_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_mask_tracker_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_deva_detail_lines_expose_mask_metadata_and_filter_reason(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="deva_offline",
            center_x=12.0,
            center_y=34.0,
            deva_raw_x=12.345,
            deva_raw_y=34.567,
            deva_refined_x=12.0,
            deva_refined_y=34.0,
            deva_visibility=True,
            deva_repr_mode="mask_centroid",
            deva_initial_mask_mode="bbox",
            deva_mask_area=123.4,
            deva_mask_score=1.0,
            deva_mask_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            deva_mask_bbox_aspect_ratio=0.8333333,
            deva_filter_rejected_reason="max_bbox_center_jump",
        )
        lines: list[str] = []

        handled = dialog._append_mask_tracker_detail_lines(
            lines,
            spot,
            source="deva_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Representative mode: mask_centroid", lines)
        self.assertIn("Initial mask mode: bbox", lines)
        self.assertIn("Mask area: 123.4", lines)
        self.assertIn("Mask score: 1.000", lines)
        self.assertIn("Mask bbox: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_bbox_center_jump)", lines)

    def test_deva_focus_point_uses_mask_tracker_spec(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="deva_offline",
            center_x=10.0,
            center_y=11.0,
            deva_raw_x=20.5,
            deva_raw_y=21.5,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(payload, [{"x": 20.5, "y": 21.5, "label": "DEVA T9"}])


class TrackingDialogDam4SamVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_mask_tracker_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_mask_tracker_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_dam4sam_detail_lines_expose_mask_metadata_without_polygon_fields(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="dam4sam_offline",
            center_x=12.0,
            center_y=34.0,
            dam4sam_raw_x=12.345,
            dam4sam_raw_y=34.567,
            dam4sam_refined_x=12.0,
            dam4sam_refined_y=34.0,
            dam4sam_visibility=True,
            dam4sam_repr_mode="mask_centroid",
            dam4sam_tracker_name="sam21pp-L",
            dam4sam_checkpoint_name="sam2.1_hiera_large.pt",
            dam4sam_config_name="sam21pp_hiera_l.yaml",
            dam4sam_mask_area=123.4,
            dam4sam_mask_score=1.0,
            dam4sam_mask_component_count=2,
            dam4sam_mask_bbox_xywh=(3.0, 4.0, 10.0, 12.0),
            dam4sam_mask_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            dam4sam_mask_bbox_aspect_ratio=0.8333333,
            dam4sam_filter_rejected_reason="max_mask_components",
        )
        lines: list[str] = []

        handled = dialog._append_mask_tracker_detail_lines(
            lines,
            spot,
            source="dam4sam_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Representative mode: mask_centroid", lines)
        self.assertIn("Tracker variant: sam21pp-L", lines)
        self.assertIn("Checkpoint: sam2.1_hiera_large.pt", lines)
        self.assertIn("Config: sam21pp_hiera_l.yaml", lines)
        self.assertIn("Mask area: 123.4", lines)
        self.assertIn("Mask score: 1.000", lines)
        self.assertIn("Mask components: 2", lines)
        self.assertIn("Mask bbox xywh: x=3.0, y=4.0, w=10.0, h=12.0", lines)
        self.assertIn("Mask bbox: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_mask_components)", lines)
        self.assertNotIn("Initial mask mode: bbox", lines)
        self.assertNotIn("Prompt mode: bbox", lines)

    def test_dam4sam_focus_point_uses_mask_tracker_spec(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="dam4sam_offline",
            center_x=10.0,
            center_y=11.0,
            dam4sam_raw_x=20.5,
            dam4sam_raw_y=21.5,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(payload, [{"x": 20.5, "y": 21.5, "label": "DAM4SAM T9"}])


class TrackingDialogSamuraiVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_mask_tracker_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_mask_tracker_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_samurai_detail_lines_expose_mask_metadata_without_polygon_fields(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="samurai_offline",
            center_x=12.0,
            center_y=34.0,
            samurai_raw_x=12.345,
            samurai_raw_y=34.567,
            samurai_refined_x=12.0,
            samurai_refined_y=34.0,
            samurai_visibility=True,
            samurai_repr_mode="gaussian_in_bbox",
            samurai_model_variant="sam2.1_hiera_base_plus",
            samurai_checkpoint_name="sam2.1_hiera_base_plus.pt",
            samurai_config_name="sam2.1_hiera_b+.yaml",
            samurai_mask_area=123.4,
            samurai_mask_score=1.0,
            samurai_mask_component_count=2,
            samurai_mask_bbox_xywh=(3.0, 4.0, 10.0, 12.0),
            samurai_mask_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            samurai_mask_bbox_aspect_ratio=0.8333333,
            samurai_filter_rejected_reason="max_mask_components",
        )
        lines: list[str] = []

        handled = dialog._append_mask_tracker_detail_lines(
            lines,
            spot,
            source="samurai_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Representative mode: gaussian_in_bbox", lines)
        self.assertIn("Model variant: sam2.1_hiera_base_plus", lines)
        self.assertIn("Checkpoint: sam2.1_hiera_base_plus.pt", lines)
        self.assertIn("Config: sam2.1_hiera_b+.yaml", lines)
        self.assertIn("Mask area: 123.4", lines)
        self.assertIn("Mask score: 1.000", lines)
        self.assertIn("Mask components: 2", lines)
        self.assertIn("Mask bbox xywh: x=3.0, y=4.0, w=10.0, h=12.0", lines)
        self.assertIn("Mask bbox: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_mask_components)", lines)
        self.assertNotIn("Initial mask mode: bbox", lines)
        self.assertNotIn("Prompt mode: bbox", lines)

    def test_samurai_focus_point_uses_mask_tracker_spec(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="samurai_offline",
            center_x=10.0,
            center_y=11.0,
            samurai_raw_x=20.5,
            samurai_raw_y=21.5,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(payload, [{"x": 20.5, "y": 21.5, "label": "SAMURAI T9"}])


class TrackingDialogOstrackVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_ostrack_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_ostrack_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_ostrack_detail_lines_expose_bbox_metadata_and_filter_reason(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="ostrack_offline",
            center_x=12.0,
            center_y=34.0,
            ostrack_raw_x=12.345,
            ostrack_raw_y=34.567,
            ostrack_refined_x=12.0,
            ostrack_refined_y=34.0,
            ostrack_visibility=True,
            ostrack_repr_mode="peak_in_bbox",
            ostrack_model_variant="vitb_256_mae_ce_32x4_ep300",
            ostrack_checkpoint_name="OSTrack_ep0300.pth.tar",
            ostrack_bbox_score=0.87654,
            ostrack_bbox_xywh=(3.0, 4.0, 10.0, 12.0),
            ostrack_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            ostrack_bbox_aspect_ratio=0.8333333,
            ostrack_filter_rejected_reason="max_bbox_center_jump",
        )
        lines: list[str] = []

        handled = dialog._append_ostrack_detail_lines(
            lines,
            spot,
            source="ostrack_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Model variant: vitb_256_mae_ce_32x4_ep300", lines)
        self.assertIn("Checkpoint: OSTrack_ep0300.pth.tar", lines)
        self.assertIn("Representative mode: peak_in_bbox", lines)
        self.assertIn("BBox score: 0.877", lines)
        self.assertIn("Raw bbox xywh: x=3.0, y=4.0, w=10.0, h=12.0", lines)
        self.assertIn("Raw bbox xyxy: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_bbox_center_jump)", lines)

    def test_ostrack_focus_payload_exposes_raw_and_stored_points(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="ostrack_offline",
            center_x=10.0,
            center_y=11.0,
            ostrack_raw_x=20.5,
            ostrack_raw_y=21.5,
            ostrack_refined_x=22.0,
            ostrack_refined_y=24.0,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(
            payload,
            [
                {"x": 20.5, "y": 21.5, "label": "OSTrack T9 raw"},
                {"x": 22.0, "y": 24.0, "label": "OSTrack T9 stored"},
            ],
        )
