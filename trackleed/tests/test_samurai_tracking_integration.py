from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class _StubRuntimeSettings:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeTrackingSupport:
    def __init__(self) -> None:
        self.load_calls: list[list[int]] = []

    def estimate_pattern_center_from_detections(self, *args, **kwargs):
        return None

    def _load_frames_for_tapir(
        self,
        all_files_map,
        frame_ids,
        progress_callback=None,
        cancel_event=None,
    ):
        frame_ids = list(frame_ids)
        self.load_calls.append(frame_ids)
        frames = np.zeros((len(frame_ids), 20, 24, 3), dtype=np.uint8)
        for local_idx, frame_id in enumerate(frame_ids):
            frames[local_idx, 0, 0, 0] = int(frame_id)
        return frames, [(20, 24) for _frame_id in frame_ids]

    def _roi_rect_from_bbox_xyxy(
        self,
        bbox_xyxy,
        image_shape,
        *,
        fallback_center_x,
        fallback_center_y,
        fallback_roi_size,
    ):
        if bbox_xyxy is None:
            size = int(fallback_roi_size)
            return _StubQRect(
                int(round(float(fallback_center_x) - size / 2.0)),
                int(round(float(fallback_center_y) - size / 2.0)),
                size,
                size,
            )
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
            return self._roi_rect_from_bbox_xyxy(
                None,
                image_shape,
                fallback_center_x=fallback_center_x,
                fallback_center_y=fallback_center_y,
                fallback_roi_size=fallback_roi_size,
            )
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        return _StubQRect(
            int(round(x0)),
            int(round(y0)),
            int(round(max(1.0, x1 - x0))),
            int(round(max(1.0, y1 - y0))),
        )


class SamuraiTrackingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_modules: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "trackleed.io.image_reader",
            "trackleed.core.data_models",
            "trackleed.analysis.sam2_backend",
            "trackleed.analysis.cutie_backend",
            "trackleed.analysis.deva_backend",
            "trackleed.analysis.dam4sam_backend",
            "trackleed.analysis.samurai_backend",
            "trackleed.analysis.xmem2_backend",
            "trackleed.analysis.tracking_mask_backends",
        ):
            self.original_modules[module_name] = sys.modules.get(module_name)
            sys.modules.pop(module_name, None)

        pyqt6_module = types.ModuleType("PyQt6")
        qtcore_module = types.ModuleType("PyQt6.QtCore")
        qtcore_module.QRect = _StubQRect
        pyqt6_module.QtCore = qtcore_module
        sys.modules["PyQt6"] = pyqt6_module
        sys.modules["PyQt6.QtCore"] = qtcore_module

        image_reader_module = types.ModuleType("trackleed.io.image_reader")
        image_reader_module.load_tiff = lambda _filepath: np.zeros((20, 24), dtype=np.float32)
        sys.modules["trackleed.io.image_reader"] = image_reader_module
        self._install_backend_dependency_stubs()

        self.data_models = importlib.import_module("trackleed.core.data_models")
        self.mask_backends = importlib.import_module("trackleed.analysis.tracking_mask_backends")
        self.addCleanup(self._restore_modules)

    def _install_backend_dependency_stubs(self) -> None:
        backend_specs = {
            "trackleed.analysis.sam2_backend": {
                "models_dir": "tracking_models/sam2",
                "settings_name": "Sam2RuntimeSettings",
                "discover_name": "discover_sam2_checkpoints",
                "run_name": "run_sam2_offline",
                "select_name": "select_default_sam2_checkpoint",
                "default_name": "DEFAULT_SAM2_MODELS_DIR",
            },
            "trackleed.analysis.cutie_backend": {
                "models_dir": "tracking_models/cutie",
                "settings_name": "CutieRuntimeSettings",
                "discover_name": "discover_cutie_checkpoints",
                "run_name": "run_cutie_offline",
                "select_name": "select_default_cutie_checkpoint",
                "default_name": "DEFAULT_CUTIE_MODELS_DIR",
            },
            "trackleed.analysis.deva_backend": {
                "models_dir": "tracking_models/deva",
                "settings_name": "DevaRuntimeSettings",
                "discover_name": "discover_deva_checkpoints",
                "run_name": "run_deva_offline",
                "select_name": "select_default_deva_checkpoint",
                "default_name": "DEFAULT_DEVA_MODELS_DIR",
            },
            "trackleed.analysis.dam4sam_backend": {
                "models_dir": "tracking_models/dam4sam",
                "settings_name": "Dam4SamRuntimeSettings",
                "discover_name": "discover_dam4sam_checkpoints",
                "run_name": "run_dam4sam_offline",
                "select_name": "select_default_dam4sam_checkpoint",
                "default_name": "DEFAULT_DAM4SAM_MODELS_DIR",
            },
            "trackleed.analysis.xmem2_backend": {
                "models_dir": "tracking_models/xmem2",
                "settings_name": "XMem2RuntimeSettings",
                "discover_name": "discover_xmem2_checkpoints",
                "run_name": "run_xmem2_offline",
                "select_name": "select_default_xmem2_checkpoint",
                "default_name": "DEFAULT_XMEM2_MODELS_DIR",
            },
        }
        for module_name, spec in backend_specs.items():
            module = types.ModuleType(module_name)
            setattr(module, spec["default_name"], Path(spec["models_dir"]))
            setattr(module, spec["settings_name"], _StubRuntimeSettings)
            setattr(module, spec["discover_name"], lambda *args, **kwargs: [])
            setattr(module, spec["select_name"], lambda *args, **kwargs: None)
            setattr(
                module,
                spec["run_name"],
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("Unexpected non-SAMURAI backend call.")
                ),
            )
            if module_name == "trackleed.analysis.dam4sam_backend":
                module.DEFAULT_DAM4SAM_TRACKER_NAME = "sam21pp-L"
            sys.modules[module_name] = module

        samurai_module = types.ModuleType("trackleed.analysis.samurai_backend")
        samurai_module.DEFAULT_SAMURAI_MODELS_DIR = Path("tracking_models/samurai")
        samurai_module.DEFAULT_SAMURAI_MODEL_VARIANT = "sam2.1_hiera_base_plus"
        samurai_module.SamuraiRuntimeSettings = _StubRuntimeSettings
        samurai_module.discover_samurai_checkpoints = lambda *args, **kwargs: [
            Path("tracking_models/samurai/sam2.1_hiera_base_plus.pt")
        ]
        samurai_module.select_default_samurai_checkpoint = (
            lambda checkpoints, *args, **kwargs: checkpoints[0] if checkpoints else None
        )
        samurai_module.run_samurai_offline = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run_samurai_offline must be patched by the test.")
        )
        sys.modules["trackleed.analysis.samurai_backend"] = samurai_module

    def _restore_modules(self) -> None:
        for module_name, original_module in self.original_modules.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def _spot(self, spot_id: int, x: float, y: float):
        return self.data_models.SpotROI(
            id=int(spot_id),
            roi_rect=_StubQRect(int(x - 4), int(y - 5), 8, 10),
            center_x=float(x),
            center_y=float(y),
            intensity=1.0,
            source="manual",
        )

    def _file(self, frame_id: int, spots=None):
        return self.data_models.LeedFile(
            filepath=Path(f"frame_{frame_id:03d}.tif"),
            filename=f"frame_{frame_id:03d}.tif",
            is_background=False,
            element="X",
            surface=1,
            mode="G",
            temperature=300.0,
            energy=100.0,
            id=int(frame_id),
            coverage=None,
            spots=list(spots or []),
        )

    def _fake_result(self, *, frame_count: int, tracks_xy: np.ndarray):
        query_count = int(tracks_xy.shape[0])
        masks = np.zeros((query_count, frame_count, 20, 24), dtype=bool)
        bboxes_xyxy = np.zeros((query_count, frame_count, 4), dtype=np.float32)
        bboxes_xywh = np.zeros((query_count, frame_count, 4), dtype=np.float32)
        for query_idx in range(query_count):
            for local_idx in range(frame_count):
                x = float(tracks_xy[query_idx, local_idx, 0])
                y = float(tracks_xy[query_idx, local_idx, 1])
                x0 = max(0, int(round(x - 2.0)))
                y0 = max(0, int(round(y - 2.0)))
                x1 = min(24, x0 + 4)
                y1 = min(20, y0 + 4)
                masks[query_idx, local_idx, y0:y1, x0:x1] = True
                bboxes_xyxy[query_idx, local_idx] = [x0, y0, x1, y1]
                bboxes_xywh[query_idx, local_idx] = [x0, y0, x1 - x0, y1 - y0]
        return SimpleNamespace(
            masks=masks,
            tracks_xy=tracks_xy.astype(np.float32),
            visible_mask=np.ones((query_count, frame_count), dtype=bool),
            mask_areas=np.full((query_count, frame_count), 16.0, dtype=np.float32),
            mask_bboxes_xyxy=bboxes_xyxy,
            mask_bboxes_xywh=bboxes_xywh,
            mask_scores=np.full((query_count, frame_count), 1.0, dtype=np.float32),
            representative_point_mode="mask_centroid",
            model_variant="sam2.1_hiera_base_plus",
            checkpoint_name="sam2.1_hiera_base_plus.pt",
            config_name="sam2.1_hiera_b+.yaml",
        )

    def test_samurai_tracking_uses_roi_seed_and_writes_mask_metadata(self) -> None:
        all_files_map = {frame_id: self._file(frame_id) for frame_id in range(3)}
        all_files_map[0].spots = [self._spot(1, 8.0, 11.0)]
        fake_support = _FakeTrackingSupport()
        run_calls: list[dict] = []

        def fake_run_samurai_offline(*, frames, query_points_tyx, settings, query_boxes_xyxy=None):
            run_calls.append(
                {
                    "frame_ids": [int(value) for value in frames[:, 0, 0, 0].tolist()],
                    "query_points_tyx": np.asarray(query_points_tyx, dtype=np.float32).copy(),
                    "query_boxes_xyxy": np.asarray(query_boxes_xyxy, dtype=np.float32).copy(),
                    "settings": settings,
                }
            )
            tracks_xy = np.asarray([[[8.0, 11.0], [9.0, 12.0], [10.0, 13.0]]], dtype=np.float32)
            return self._fake_result(frame_count=int(frames.shape[0]), tracks_xy=tracks_xy)

        with (
            mock.patch.object(self.mask_backends, "_tracking_support", return_value=fake_support),
            mock.patch.object(
                self.mask_backends,
                "run_samurai_offline",
                side_effect=fake_run_samurai_offline,
            ),
        ):
            tracks = self.mask_backends.track_spots_samurai_offline(
                all_files_map,
                [0, 1, 2],
                0,
                {
                    "checkpoint_path": "sam2.1_hiera_base_plus.pt",
                    "samurai_repo_path": "repo",
                    "python_executable": "python.exe",
                    "backend_mode": "subprocess",
                    "model_variant": "base_plus",
                    "roi_size": 8,
                    "keep_invisible_points_as_gaps": False,
                    "max_mask_components": 4,
                },
            )

        self.assertIsNotNone(tracks)
        self.assertEqual(fake_support.load_calls, [[0, 1, 2]])
        self.assertEqual(run_calls[0]["frame_ids"], [0, 1, 2])
        np.testing.assert_allclose(run_calls[0]["query_points_tyx"][0], [0.0, 11.0, 8.0])
        np.testing.assert_allclose(run_calls[0]["query_boxes_xyxy"][0], [4.0, 6.0, 12.0, 16.0])
        self.assertEqual(run_calls[0]["settings"].model_variant, "base_plus")

        track = tracks[0]
        self.assertEqual(sorted(track.spots_by_image_id), [0, 1, 2])
        spot = track.spots_by_image_id[1]
        self.assertEqual(spot.source, "samurai_offline")
        self.assertTrue(spot.samurai_visibility)
        self.assertEqual(spot.samurai_mask_area, 16.0)
        self.assertEqual(spot.samurai_mask_component_count, 1)
        self.assertEqual(spot.samurai_model_variant, "sam2.1_hiera_base_plus")
        self.assertEqual(spot.samurai_checkpoint_name, "sam2.1_hiera_base_plus.pt")
        self.assertEqual(spot.samurai_config_name, "sam2.1_hiera_b+.yaml")
        self.assertEqual(spot.samurai_mask_bbox_xyxy, (7.0, 10.0, 11.0, 14.0))
        self.assertEqual(spot.samurai_mask_bbox_xywh, (7.0, 10.0, 4.0, 4.0))

    def test_samurai_retrack_trims_future_points_and_uses_current_roi_seed(self) -> None:
        all_files_map = {frame_id: self._file(frame_id) for frame_id in range(4)}
        start_spot = self._spot(1, 10.0, 12.0)
        original_first_spot = self._spot(1, 8.0, 11.0)
        stale_future_spot = self._spot(1, 99.0, 99.0)
        track = self.data_models.Track(
            track_id=1,
            spots_by_image_id={
                0: original_first_spot,
                1: start_spot,
                2: stale_future_spot,
            },
        )
        fake_support = _FakeTrackingSupport()
        run_calls: list[dict] = []

        def fake_run_samurai_offline(*, frames, query_points_tyx, settings, query_boxes_xyxy=None):
            run_calls.append(
                {
                    "frame_ids": [int(value) for value in frames[:, 0, 0, 0].tolist()],
                    "query_points_tyx": np.asarray(query_points_tyx, dtype=np.float32).copy(),
                    "query_boxes_xyxy": np.asarray(query_boxes_xyxy, dtype=np.float32).copy(),
                    "settings": settings,
                }
            )
            tracks_xy = np.asarray([[[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]]], dtype=np.float32)
            return self._fake_result(frame_count=int(frames.shape[0]), tracks_xy=tracks_xy)

        with (
            mock.patch.object(self.mask_backends, "_tracking_support", return_value=fake_support),
            mock.patch.object(
                self.mask_backends,
                "run_samurai_offline",
                side_effect=fake_run_samurai_offline,
            ),
        ):
            updated_track = self.mask_backends.retrack_single_spot_samurai_offline(
                track,
                1,
                all_files_map,
                [0, 1, 2, 3],
                {
                    "checkpoint_path": "sam2.1_hiera_base_plus.pt",
                    "samurai_repo_path": "repo",
                    "python_executable": "python.exe",
                    "backend_mode": "subprocess",
                    "model_variant": "base_plus",
                    "roi_size": 8,
                    "keep_invisible_points_as_gaps": False,
                    "max_mask_components": 4,
                },
            )

        self.assertIs(updated_track, track)
        self.assertEqual(fake_support.load_calls, [[1, 2, 3]])
        self.assertEqual(run_calls[0]["frame_ids"], [1, 2, 3])
        np.testing.assert_allclose(run_calls[0]["query_points_tyx"][0], [0.0, 12.0, 10.0])
        np.testing.assert_allclose(run_calls[0]["query_boxes_xyxy"][0], [6.0, 7.0, 14.0, 17.0])
        self.assertEqual(run_calls[0]["settings"].model_variant, "base_plus")

        self.assertEqual(sorted(track.spots_by_image_id), [0, 1, 2, 3])
        self.assertIs(track.spots_by_image_id[0], original_first_spot)
        self.assertIs(track.spots_by_image_id[1], start_spot)
        self.assertIsNot(track.spots_by_image_id[2], stale_future_spot)
        retracked_spot = track.spots_by_image_id[2]
        self.assertEqual(retracked_spot.source, "samurai_offline")
        self.assertTrue(retracked_spot.samurai_visibility)
        self.assertEqual(retracked_spot.samurai_model_variant, "sam2.1_hiera_base_plus")
        self.assertEqual(retracked_spot.samurai_checkpoint_name, "sam2.1_hiera_base_plus.pt")
        self.assertEqual(retracked_spot.samurai_config_name, "sam2.1_hiera_b+.yaml")


if __name__ == "__main__":
    unittest.main()
