from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np


class _StubQRect:
    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0) -> None:
        self._x = int(x)
        self._y = int(y)
        self._width = int(width)
        self._height = int(height)

    def left(self) -> int:
        return self._x

    def top(self) -> int:
        return self._y

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


def _stub_callable(name: str):
    def _fn(*args, **kwargs):
        return {"name": name, "args": args, "kwargs": kwargs}

    _fn.__name__ = name.rsplit(".", 1)[-1]
    return _fn


class _RuntimeSettings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ShimImportMixin:
    runtime_module_names = (
        "trackleed.core.data_models",
        "trackleed.analysis.tracking_result_builders",
        "trackleed.analysis.tracking_visibility_utils",
        "trackleed.analysis.tracking_geometry",
        "trackleed.analysis.tracking_filter_utils",
        "trackleed.analysis.tracking_windowing_utils",
        "trackleed.analysis.tracking_family_adapters",
        "trackleed.analysis.tracking_retrack_utils",
        "trackleed.analysis.tracking_point_backends",
        "trackleed.analysis.tracking_bbox_backends",
        "trackleed.analysis.tracking_mask_backends",
        "trackleed.analysis.tracking_retrack",
        "trackleed.analysis.tracking",
    )

    def import_shim_modules(self):
        originals: dict[str, types.ModuleType | None] = {}
        module_names_to_clear = (
            "PyQt6",
            "PyQt6.QtCore",
            "cv2",
            "scipy",
            "scipy.ndimage",
            "scipy.optimize",
            "skimage",
            "skimage.color",
            "skimage.morphology",
            "skimage.util",
            "trackleed.io.image_reader",
            "trackleed.analysis.peak_analyzer",
            "trackleed.analysis.tracking_config",
            "trackleed.analysis.polygon_mask",
            "trackleed.analysis.tapir_backend",
            "trackleed.analysis.locotrack_backend",
            "trackleed.analysis.cotracker_backend",
            "trackleed.analysis.tapnext_backend",
            "trackleed.analysis.trackonr_backend",
            "trackleed.analysis.pipspp_backend",
            "trackleed.analysis.mftiq_backend",
            "trackleed.analysis.sam2_backend",
            "trackleed.analysis.cutie_backend",
            "trackleed.analysis.deva_backend",
            "trackleed.analysis.dam4sam_backend",
            "trackleed.analysis.samurai_backend",
            "trackleed.analysis.xmem2_backend",
            "trackleed.analysis.tomp_backend",
            "trackleed.analysis.ostrack_backend",
            *self.runtime_module_names,
        )
        for module_name in module_names_to_clear:
            originals[module_name] = sys.modules.get(module_name)
            sys.modules.pop(module_name, None)

        pyqt_module = types.ModuleType("PyQt6")
        qtcore_module = types.ModuleType("PyQt6.QtCore")
        qtcore_module.QRect = _StubQRect
        pyqt_module.QtCore = qtcore_module
        sys.modules["PyQt6"] = pyqt_module
        sys.modules["PyQt6.QtCore"] = qtcore_module

        sys.modules["cv2"] = types.ModuleType("cv2")

        scipy_module = types.ModuleType("scipy")
        scipy_ndimage_module = types.ModuleType("scipy.ndimage")
        scipy_ndimage_module.gaussian_filter = lambda image, sigma=0, *args, **kwargs: image
        scipy_optimize_module = types.ModuleType("scipy.optimize")
        scipy_optimize_module.linear_sum_assignment = (
            lambda cost_matrix, *args, **kwargs: (np.array([], dtype=int), np.array([], dtype=int))
        )
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.ndimage"] = scipy_ndimage_module
        sys.modules["scipy.optimize"] = scipy_optimize_module

        skimage_module = types.ModuleType("skimage")
        skimage_color_module = types.ModuleType("skimage.color")
        skimage_color_module.rgb2gray = lambda image, *args, **kwargs: image
        skimage_morph_module = types.ModuleType("skimage.morphology")
        skimage_morph_module.white_tophat = lambda image, footprint=None, *args, **kwargs: image
        skimage_morph_module.disk = lambda radius, *args, **kwargs: radius
        skimage_util_module = types.ModuleType("skimage.util")
        skimage_util_module.img_as_ubyte = lambda image, *args, **kwargs: image
        sys.modules["skimage"] = skimage_module
        sys.modules["skimage.color"] = skimage_color_module
        sys.modules["skimage.morphology"] = skimage_morph_module
        sys.modules["skimage.util"] = skimage_util_module

        image_reader_module = types.ModuleType("trackleed.io.image_reader")
        image_reader_module.load_tiff = lambda filepath: np.zeros((8, 8), dtype=np.float32)
        sys.modules["trackleed.io.image_reader"] = image_reader_module

        peak_analyzer_module = types.ModuleType("trackleed.analysis.peak_analyzer")
        peak_analyzer_module.find_pattern_center = _stub_callable(
            "trackleed.analysis.peak_analyzer.find_pattern_center"
        )
        peak_analyzer_module.fit_spot_to_gaussian = _stub_callable(
            "trackleed.analysis.peak_analyzer.fit_spot_to_gaussian"
        )
        sys.modules["trackleed.analysis.peak_analyzer"] = peak_analyzer_module

        tracking_config_module = types.ModuleType("trackleed.analysis.tracking_config")
        tracking_config_module.default_directional_hybrid_params = (
            lambda: {"base_max_distance_px": 10.0}
        )
        sys.modules["trackleed.analysis.tracking_config"] = tracking_config_module

        polygon_mask_module = types.ModuleType("trackleed.analysis.polygon_mask")
        polygon_mask_module.normalize_polygon_points = (
            lambda points, *args, **kwargs: list(points or [])
        )
        polygon_mask_module.polygon_points_are_valid = lambda points, *args, **kwargs: bool(points)
        polygon_mask_module.polygon_points_to_mask = _stub_callable(
            "trackleed.analysis.polygon_mask.polygon_points_to_mask"
        )
        sys.modules["trackleed.analysis.polygon_mask"] = polygon_mask_module

        self._install_backend_stub(
            "trackleed.analysis.tapir_backend",
            default_models_dir="tracking_models/tapir",
            runtime_settings=("TapirRuntimeSettings", "TapirOnlineRuntimeSettings"),
            extra_attrs={
                "discover_online_tapir_checkpoints": lambda *args, **kwargs: [],
                "discover_tapir_checkpoints": lambda *args, **kwargs: [],
                "run_tapir_offline": _stub_callable(
                    "trackleed.analysis.tapir_backend.run_tapir_offline"
                ),
                "run_tapir_online": _stub_callable(
                    "trackleed.analysis.tapir_backend.run_tapir_online"
                ),
                "select_default_online_tapir_checkpoint": lambda *args, **kwargs: None,
                "select_default_tapir_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.locotrack_backend",
            default_models_dir="tracking_models/locotrack",
            runtime_settings=("LocotrackRuntimeSettings",),
            extra_attrs={
                "discover_locotrack_checkpoints": lambda *args, **kwargs: [],
                "run_locotrack_offline": _stub_callable(
                    "trackleed.analysis.locotrack_backend.run_locotrack_offline"
                ),
                "select_default_locotrack_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.cotracker_backend",
            default_models_dir="tracking_models/cotracker",
            runtime_settings=("CotrackerRuntimeSettings",),
            extra_attrs={
                "discover_cotracker_checkpoints": lambda *args, **kwargs: [],
                "run_cotracker3_offline": _stub_callable(
                    "trackleed.analysis.cotracker_backend.run_cotracker3_offline"
                ),
                "select_default_cotracker_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.tapnext_backend",
            default_models_dir="tracking_models/tapnext",
            runtime_settings=("TapnextRuntimeSettings",),
            extra_attrs={
                "discover_tapnext_checkpoints": lambda *args, **kwargs: [],
                "run_tapnext_offline": _stub_callable(
                    "trackleed.analysis.tapnext_backend.run_tapnext_offline"
                ),
                "select_default_tapnext_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.trackonr_backend",
            default_models_dir="tracking_models/trackonr",
            runtime_settings=("TrackonrRuntimeSettings",),
            extra_attrs={
                "discover_trackonr_checkpoints": lambda *args, **kwargs: [],
                "run_trackonr_offline": _stub_callable(
                    "trackleed.analysis.trackonr_backend.run_trackonr_offline"
                ),
                "select_default_trackonr_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.pipspp_backend",
            default_models_dir="tracking_models/pipspp",
            runtime_settings=("PipsppRuntimeSettings",),
            extra_attrs={
                "discover_pipspp_checkpoints": lambda *args, **kwargs: [],
                "run_pipspp_offline": _stub_callable(
                    "trackleed.analysis.pipspp_backend.run_pipspp_offline"
                ),
                "select_default_pipspp_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.mftiq_backend",
            default_models_dir="tracking_models/mftiq",
            runtime_settings=("MftiqRuntimeSettings",),
            extra_attrs={
                "discover_mftiq_checkpoints": lambda *args, **kwargs: [],
                "run_mftiq_offline": _stub_callable(
                    "trackleed.analysis.mftiq_backend.run_mftiq_offline"
                ),
                "select_default_mftiq_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.sam2_backend",
            default_models_dir="tracking_models/sam2",
            runtime_settings=("Sam2RuntimeSettings",),
            extra_attrs={
                "discover_sam2_checkpoints": lambda *args, **kwargs: [],
                "run_sam2_offline": _stub_callable(
                    "trackleed.analysis.sam2_backend.run_sam2_offline"
                ),
                "select_default_sam2_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.cutie_backend",
            default_models_dir="tracking_models/cutie",
            runtime_settings=("CutieRuntimeSettings",),
            extra_attrs={
                "discover_cutie_checkpoints": lambda *args, **kwargs: [],
                "run_cutie_offline": _stub_callable(
                    "trackleed.analysis.cutie_backend.run_cutie_offline"
                ),
                "select_default_cutie_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.deva_backend",
            default_models_dir="tracking_models/deva",
            runtime_settings=("DevaRuntimeSettings",),
            extra_attrs={
                "discover_deva_checkpoints": lambda *args, **kwargs: [],
                "run_deva_offline": _stub_callable(
                    "trackleed.analysis.deva_backend.run_deva_offline"
                ),
                "select_default_deva_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.dam4sam_backend",
            default_models_dir="tracking_models/dam4sam",
            runtime_settings=("Dam4SamRuntimeSettings",),
            extra_attrs={
                "DEFAULT_DAM4SAM_TRACKER_NAME": "sam21pp-L",
                "discover_dam4sam_checkpoints": lambda *args, **kwargs: [],
                "run_dam4sam_offline": _stub_callable(
                    "trackleed.analysis.dam4sam_backend.run_dam4sam_offline"
                ),
                "select_default_dam4sam_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.samurai_backend",
            default_models_dir="tracking_models/samurai",
            runtime_settings=("SamuraiRuntimeSettings",),
            extra_attrs={
                "DEFAULT_SAMURAI_MODEL_VARIANT": "sam2.1_hiera_base_plus",
                "discover_samurai_checkpoints": lambda *args, **kwargs: [],
                "run_samurai_offline": _stub_callable(
                    "trackleed.analysis.samurai_backend.run_samurai_offline"
                ),
                "select_default_samurai_checkpoint": lambda *args, **kwargs: None,
            },
        )
        self._install_backend_stub(
            "trackleed.analysis.xmem2_backend",
            default_models_dir="tracking_models/xmem2",
            runtime_settings=("XMem2RuntimeSettings",),
            extra_attrs={
                "discover_xmem2_checkpoints": lambda *args, **kwargs: [],
                "run_xmem2_offline": _stub_callable(
                    "trackleed.analysis.xmem2_backend.run_xmem2_offline"
                ),
                "select_default_xmem2_checkpoint": lambda *args, **kwargs: None,
            },
        )

        ostrack_module = types.ModuleType("trackleed.analysis.ostrack_backend")
        ostrack_module.DEFAULT_OSTRACK_CONFIG_NAME = "vitb_256_mae_ce_32x4_ep300"
        ostrack_module.DEFAULT_OSTRACK_MODELS_DIR = Path("tracking_models/ostrack")
        ostrack_module.SUPPORTED_OSTRACK_CONFIG_NAMES = (
            "vitb_256_mae_ce_32x4_ep300",
            "vitb_384_mae_ce_32x4_ep300",
            "vitb_256_mae_32x4_ep300",
            "vitb_384_mae_32x4_ep300",
            "vitb_256_mae_ce_32x4_got10k_ep100",
            "vitb_384_mae_ce_32x4_got10k_ep100",
        )
        ostrack_module.OstrackRuntimeSettings = _RuntimeSettings
        ostrack_module.discover_ostrack_checkpoints = lambda *args, **kwargs: []
        ostrack_module.run_ostrack_offline = _stub_callable(
            "trackleed.analysis.ostrack_backend.run_ostrack_offline"
        )
        ostrack_module.select_default_ostrack_checkpoint = lambda *args, **kwargs: None
        ostrack_module.format_ostrack_environment_report = lambda status: f"ostrack:{status}"
        ostrack_module.probe_ostrack_environment = _stub_callable(
            "trackleed.analysis.ostrack_backend.probe_ostrack_environment"
        )
        sys.modules["trackleed.analysis.ostrack_backend"] = ostrack_module

        tomp_module = types.ModuleType("trackleed.analysis.tomp_backend")
        tomp_module.DEFAULT_TAMOS_MODELS_DIR = Path("tracking_models/tamos")
        tomp_module.DEFAULT_TOMP_MODELS_DIR = Path("tracking_models/tomp")
        tomp_module.TompRuntimeSettings = _RuntimeSettings
        tomp_module.discover_tamos_checkpoints = lambda *args, **kwargs: []
        tomp_module.discover_tomp_checkpoints = lambda *args, **kwargs: []
        tomp_module.expected_tomp_checkpoint_name_for_variant = lambda variant: f"{variant}.pth.tar"
        tomp_module.run_tomp_offline = _stub_callable(
            "trackleed.analysis.tomp_backend.run_tomp_offline"
        )
        tomp_module.select_default_tamos_checkpoint_for_variant = lambda *args, **kwargs: None
        tomp_module.select_default_tomp_checkpoint_for_variant = lambda *args, **kwargs: None
        tomp_module.format_tomp_environment_report = lambda status: f"tomp:{status}"
        tomp_module.probe_tomp_environment = _stub_callable(
            "trackleed.analysis.tomp_backend.probe_tomp_environment"
        )
        sys.modules["trackleed.analysis.tomp_backend"] = tomp_module

        self.addCleanup(self._restore_modules, originals)

        modules = {
            "point_backends": importlib.import_module("trackleed.analysis.tracking_point_backends"),
            "bbox_backends": importlib.import_module("trackleed.analysis.tracking_bbox_backends"),
            "mask_backends": importlib.import_module("trackleed.analysis.tracking_mask_backends"),
            "tracking_retrack": importlib.import_module("trackleed.analysis.tracking_retrack"),
            "tracking": importlib.import_module("trackleed.analysis.tracking"),
        }
        self.addCleanup(self._cleanup_runtime_modules)
        return modules

    def _cleanup_runtime_modules(self) -> None:
        for module_name in self.runtime_module_names:
            sys.modules.pop(module_name, None)

    def _restore_modules(self, originals: dict[str, types.ModuleType | None]) -> None:
        for module_name, original_module in originals.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def _install_backend_stub(
        self,
        module_name: str,
        *,
        default_models_dir: str,
        runtime_settings: tuple[str, ...],
        extra_attrs: dict[str, object],
    ) -> None:
        module = types.ModuleType(module_name)
        suffix = module_name.rsplit(".", 1)[-1].replace("_backend", "")
        module.__dict__[f"DEFAULT_{suffix.upper()}_MODELS_DIR"] = Path(default_models_dir)
        for class_name in runtime_settings:
            module.__dict__[class_name] = _RuntimeSettings
        module.__dict__[f"format_{suffix}_environment_report"] = lambda status, s=suffix: f"{s}:{status}"
        module.__dict__[f"probe_{suffix}_environment"] = _stub_callable(
            f"{module_name}.probe_{suffix}_environment"
        )
        for attr_name, value in extra_attrs.items():
            module.__dict__[attr_name] = value
        sys.modules[module_name] = module


class TrackingShimCompatibilityTests(ShimImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.modules = self.import_shim_modules()
        self.point_backends = self.modules["point_backends"]
        self.bbox_backends = self.modules["bbox_backends"]
        self.mask_backends = self.modules["mask_backends"]
        self.tracking_retrack = self.modules["tracking_retrack"]
        self.tracking = self.modules["tracking"]

    def test_tracking_shim_reexports_family_tracking_entrypoints(self) -> None:
        self.assertTrue(callable(self.tracking.estimate_pattern_center_from_detections))
        self.assertIs(self.tracking.track_spots_tapir_offline, self.point_backends.track_spots_tapir_offline)
        self.assertIs(self.tracking.track_spots_tapir_online, self.point_backends.track_spots_tapir_online)
        self.assertIs(self.tracking.track_spots_locotrack_offline, self.point_backends.track_spots_locotrack_offline)
        self.assertIs(self.tracking.track_spots_cotracker3_offline, self.point_backends.track_spots_cotracker3_offline)
        self.assertIs(self.tracking.track_spots_tapnext_offline, self.point_backends.track_spots_tapnext_offline)
        self.assertIs(self.tracking.track_spots_trackonr_offline, self.point_backends.track_spots_trackonr_offline)
        self.assertIs(self.tracking.track_spots_pipspp_offline, self.point_backends.track_spots_pipspp_offline)
        self.assertIs(self.tracking.track_spots_mftiq_offline, self.point_backends.track_spots_mftiq_offline)
        self.assertIs(self.tracking.track_spots_tomp_offline, self.bbox_backends.track_spots_tomp_offline)
        self.assertIs(self.tracking.track_spots_ostrack_offline, self.bbox_backends.track_spots_ostrack_offline)
        self.assertIs(self.tracking.track_spots_cutie_offline, self.mask_backends.track_spots_cutie_offline)
        self.assertIs(self.tracking.track_spots_xmem2_offline, self.mask_backends.track_spots_xmem2_offline)
        self.assertIs(self.tracking.track_spots_sam2_offline, self.mask_backends.track_spots_sam2_offline)
        self.assertIs(self.tracking.track_spots_deva_offline, self.mask_backends.track_spots_deva_offline)
        self.assertIs(self.tracking.track_spots_dam4sam_offline, self.mask_backends.track_spots_dam4sam_offline)
        self.assertIs(self.tracking.track_spots_samurai_offline, self.mask_backends.track_spots_samurai_offline)

    def test_tracking_shim_reexports_retrack_entrypoints(self) -> None:
        self.assertIs(
            self.tracking.retrack_single_spot_tapir_offline,
            self.tracking_retrack.retrack_single_spot_tapir_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_tapir_online,
            self.tracking_retrack.retrack_single_spot_tapir_online,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_locotrack_offline,
            self.tracking_retrack.retrack_single_spot_locotrack_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_cotracker3_offline,
            self.tracking_retrack.retrack_single_spot_cotracker3_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_tapnext_offline,
            self.tracking_retrack.retrack_single_spot_tapnext_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_trackonr_offline,
            self.tracking_retrack.retrack_single_spot_trackonr_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_pipspp_offline,
            self.tracking_retrack.retrack_single_spot_pipspp_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_mftiq_offline,
            self.tracking_retrack.retrack_single_spot_mftiq_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_tomp_offline,
            self.tracking_retrack.retrack_single_spot_tomp_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_ostrack_offline,
            self.tracking_retrack.retrack_single_spot_ostrack_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_cutie_offline,
            self.tracking_retrack.retrack_single_spot_cutie_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_xmem2_offline,
            self.tracking_retrack.retrack_single_spot_xmem2_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_sam2_offline,
            self.tracking_retrack.retrack_single_spot_sam2_offline,
        )
        self.assertIs(
            self.tracking.retrack_single_spot_deva_offline,
            self.tracking_retrack.retrack_single_spot_deva_offline,
        )

    def test_point_wrapper_and_optical_flow_retrack_delegate_to_legacy_support(self) -> None:
        point_calls: list[str] = []

        class _PointSupport:
            def __getattr__(self, name: str):
                def _fn(*args, **kwargs):
                    point_calls.append(name)
                    return name

                return _fn

        original_point_import_module = self.point_backends.import_module
        original_retrack_import_module = self.tracking_retrack.import_module
        self.addCleanup(setattr, self.point_backends, "import_module", original_point_import_module)
        self.addCleanup(setattr, self.tracking_retrack, "import_module", original_retrack_import_module)

        self.point_backends.import_module = lambda module_name: _PointSupport()
        self.tracking_retrack.import_module = lambda module_name: _PointSupport()

        self.assertEqual(
            self.point_backends.track_spots_tapir_offline({}, [], 1, {}),
            "_legacy_track_spots_tapir_offline",
        )
        self.assertEqual(
            self.point_backends.retrack_single_spot_tapir_online(object(), 1, {}, [], {}),
            "_legacy_retrack_single_spot_tapir_online",
        )
        self.assertEqual(
            self.tracking_retrack.retrack_single_spot_optical_flow(object(), 1, {}, [], {}),
            "_legacy_retrack_single_spot_optical_flow",
        )
        self.assertEqual(
            point_calls,
            [
                "_legacy_track_spots_tapir_offline",
                "_legacy_retrack_single_spot_tapir_online",
                "_legacy_retrack_single_spot_optical_flow",
            ],
        )
