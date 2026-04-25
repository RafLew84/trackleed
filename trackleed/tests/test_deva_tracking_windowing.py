from __future__ import annotations

import sys
import types
import unittest
import importlib
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

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y


class _FakeTrackingSupport:
    def __init__(self, data_models_module) -> None:
        self.data_models = data_models_module
        self.load_calls: list[list[int]] = []
        self.append_calls: list[dict] = []

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

    def _append_deva_sequence_to_track(
        self,
        track,
        state,
        reference_spot,
        frame_ids,
        frame_shapes,
        query_tracks,
        query_visible,
        query_mask_areas,
        query_mask_bboxes_xyxy,
        query_mask_scores,
        **kwargs,
    ) -> bool:
        self.append_calls.append(
            {
                "frame_ids": list(frame_ids),
                "start_local_idx": kwargs["start_local_idx"],
                "min_global_write_idx": kwargs["min_global_write_idx"],
                "global_start_idx": kwargs["global_start_idx"],
                "max_step_distance_px": kwargs["max_step_distance_px"],
                "max_reacquisition_gap": kwargs["max_reacquisition_gap"],
                "terminate_after_invisible_frames": kwargs[
                    "terminate_after_invisible_frames"
                ],
                "max_mask_area_ratio_change": kwargs["max_mask_area_ratio_change"],
                "max_bbox_aspect_ratio_change": kwargs["max_bbox_aspect_ratio_change"],
                "max_bbox_center_jump_px": kwargs["max_bbox_center_jump_px"],
                "direction_mode": kwargs["direction_mode"],
                "direction_center_xy": kwargs["direction_center_xy"],
            }
        )
        for local_idx in range(int(kwargs["start_local_idx"]), len(frame_ids)):
            global_idx = int(kwargs["global_start_idx"]) + local_idx
            min_global_write_idx = kwargs["min_global_write_idx"]
            if min_global_write_idx is not None and global_idx <= int(min_global_write_idx):
                continue
            if not bool(query_visible[local_idx]):
                continue
            x = float(query_tracks[local_idx, 0])
            y = float(query_tracks[local_idx, 1])
            spot = self.data_models.SpotROI(
                id=int(reference_spot.id),
                roi_rect=_StubQRect(int(round(x - 2)), int(round(y - 2)), 4, 4),
                center_x=x,
                center_y=y,
                intensity=1.0,
                source="deva_offline",
            )
            spot.deva_visibility = True
            spot.deva_mask_area = float(query_mask_areas[local_idx])
            spot.deva_mask_score = float(query_mask_scores[local_idx])
            track.spots_by_image_id[int(frame_ids[local_idx])] = spot
        return True


class _StubRuntimeSettings:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class DevaTrackingWindowingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_modules: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "trackleed.io.image_reader",
            "trackleed.core.data_models",
            "trackleed.analysis.tracking_result_builders",
            "trackleed.analysis.tracking_visibility_utils",
            "trackleed.analysis.tracking_filter_utils",
            "trackleed.analysis.tracking_windowing_utils",
            "trackleed.analysis.tracking_family_adapters",
            "trackleed.analysis.sam2_backend",
            "trackleed.analysis.cutie_backend",
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
        self._install_mask_backend_dependency_stubs()

        self.data_models = importlib.import_module("trackleed.core.data_models")
        self.mask_backends = importlib.import_module("trackleed.analysis.tracking_mask_backends")
        self.addCleanup(self._restore_modules)

    def _install_mask_backend_dependency_stubs(self) -> None:
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
            setattr(module, spec["select_name"], lambda checkpoints: None)
            setattr(
                module,
                spec["run_name"],
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("Unexpected non-DEVA backend call in DEVA windowing test.")
                ),
            )
            sys.modules[module_name] = module

    def _restore_modules(self) -> None:
        for module_name, original_module in self.original_modules.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def _spot(self, spot_id: int, x: float, y: float):
        return self.data_models.SpotROI(
            id=int(spot_id),
            roi_rect=_StubQRect(int(x - 2), int(y - 2), 4, 4),
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

    def test_deva_windowing_reseeds_each_window_from_local_zero(self) -> None:
        all_files_map = {frame_id: self._file(frame_id) for frame_id in range(5)}
        all_files_map[0].spots = [self._spot(1, 6.0, 8.0)]
        fake_support = _FakeTrackingSupport(self.data_models)
        run_calls: list[dict] = []

        def fake_run_deva_offline(
            *,
            frames,
            query_points_tyx,
            settings,
            query_boxes_xyxy=None,
            query_object_ids=None,
            query_initial_masks=None,
        ):
            query_points_tyx = np.asarray(query_points_tyx, dtype=np.float32)
            run_calls.append(
                {
                    "frame_count": int(frames.shape[0]),
                    "frame_ids": [int(value) for value in frames[:, 0, 0, 0].tolist()],
                    "query_points_tyx": query_points_tyx.copy(),
                    "query_boxes_xyxy": np.asarray(query_boxes_xyxy, dtype=np.float32).copy(),
                    "settings": settings,
                }
            )
            query_count = int(query_points_tyx.shape[0])
            frame_count = int(frames.shape[0])
            tracks_xy = np.zeros((query_count, frame_count, 2), dtype=np.float32)
            mask_bboxes = np.zeros((query_count, frame_count, 4), dtype=np.float32)
            for query_idx in range(query_count):
                seed_x = float(query_points_tyx[query_idx, 2])
                seed_y = float(query_points_tyx[query_idx, 1])
                for local_idx in range(frame_count):
                    x = seed_x + float(local_idx)
                    y = seed_y + float(local_idx)
                    tracks_xy[query_idx, local_idx] = [x, y]
                    mask_bboxes[query_idx, local_idx] = [x - 2.0, y - 2.0, x + 2.0, y + 2.0]
            return SimpleNamespace(
                tracks_xy=tracks_xy,
                visible_mask=np.ones((query_count, frame_count), dtype=bool),
                mask_areas=np.full((query_count, frame_count), 16.0, dtype=np.float32),
                mask_bboxes_xyxy=mask_bboxes,
                mask_scores=np.ones((query_count, frame_count), dtype=np.float32),
                representative_point_mode=settings.representative_point_mode,
                initial_mask_mode="bbox",
            )

        with (
            mock.patch.object(
                self.mask_backends,
                "_tracking_support",
                return_value=fake_support,
            ),
            mock.patch.object(
                self.mask_backends,
                "run_deva_offline",
                side_effect=fake_run_deva_offline,
            ),
        ):
            tracks = self.mask_backends.track_spots_deva_offline(
                all_files_map,
                [0, 1, 2, 3, 4],
                0,
                {
                    "checkpoint_path": "DEVA-propagation.pth",
                    "backend_mode": "subprocess",
                    "window_size": 3,
                    "window_overlap": 0,
                    "roi_size": 8,
                    "keep_invisible_points_as_gaps": False,
                    "max_step_distance_px": 7.0,
                    "max_reacquisition_gap": 2,
                    "terminate_after_invisible_frames": 3,
                    "max_mask_area_ratio_change": 1.5,
                    "max_bbox_aspect_ratio_change": 2.0,
                    "max_bbox_center_jump_px": 9.0,
                    "direction_mode": "outward",
                    "center_mode": "manual",
                    "manual_center": (0.0, 0.0),
                    "lateral_angle_tolerance_deg": 35.0,
                    "max_radial_jump_px": 5.0,
                    "min_directional_motion_px": 0.25,
                },
            )

        self.assertIsNotNone(tracks)
        self.assertEqual(fake_support.load_calls, [[0, 1, 2], [2, 3, 4]])
        self.assertEqual([call["frame_ids"] for call in run_calls], [[0, 1, 2], [2, 3, 4]])
        self.assertEqual([call["frame_count"] for call in run_calls], [3, 3])
        self.assertTrue(np.all(run_calls[0]["query_points_tyx"][:, 0] == 0.0))
        self.assertTrue(np.all(run_calls[1]["query_points_tyx"][:, 0] == 0.0))
        np.testing.assert_allclose(run_calls[1]["query_points_tyx"][0], [0.0, 10.0, 8.0])
        np.testing.assert_allclose(run_calls[1]["query_boxes_xyxy"][0], [6.0, 8.0, 10.0, 12.0])

        self.assertEqual(
            [
                (
                    call["frame_ids"],
                    call["start_local_idx"],
                    call["min_global_write_idx"],
                    call["global_start_idx"],
                )
                for call in fake_support.append_calls
            ],
            [
                ([0, 1, 2], 1, 0, 0),
                ([2, 3, 4], 1, 2, 2),
            ],
        )
        self.assertEqual(fake_support.append_calls[0]["max_step_distance_px"], 7.0)
        self.assertEqual(fake_support.append_calls[0]["max_reacquisition_gap"], 2)
        self.assertEqual(fake_support.append_calls[0]["terminate_after_invisible_frames"], 3)
        self.assertEqual(fake_support.append_calls[0]["max_mask_area_ratio_change"], 1.5)
        self.assertEqual(fake_support.append_calls[0]["max_bbox_aspect_ratio_change"], 2.0)
        self.assertEqual(fake_support.append_calls[0]["max_bbox_center_jump_px"], 9.0)
        self.assertEqual(fake_support.append_calls[0]["direction_mode"], "outward")
        self.assertEqual(fake_support.append_calls[0]["direction_center_xy"], (0.0, 0.0))
        self.assertEqual(sorted(tracks[0].spots_by_image_id), [0, 1, 2, 3, 4])

    def test_deva_retrack_trims_future_frames_and_reseeds_from_selected_frame(self) -> None:
        all_files_map = {frame_id: self._file(frame_id) for frame_id in range(7)}
        track = self.data_models.Track(
            track_id=1,
            spots_by_image_id={
                frame_id: self._spot(1, float(6 + frame_id), float(8 + frame_id))
                for frame_id in range(7)
            },
        )
        fake_support = _FakeTrackingSupport(self.data_models)
        run_calls: list[dict] = []

        def fake_run_deva_offline(
            *,
            frames,
            query_points_tyx,
            settings,
            query_boxes_xyxy=None,
            query_object_ids=None,
            query_initial_masks=None,
        ):
            query_points_tyx = np.asarray(query_points_tyx, dtype=np.float32)
            run_calls.append(
                {
                    "frame_count": int(frames.shape[0]),
                    "frame_ids": [int(value) for value in frames[:, 0, 0, 0].tolist()],
                    "query_points_tyx": query_points_tyx.copy(),
                    "query_boxes_xyxy": np.asarray(query_boxes_xyxy, dtype=np.float32).copy(),
                }
            )
            frame_count = int(frames.shape[0])
            seed_x = float(query_points_tyx[0, 2])
            seed_y = float(query_points_tyx[0, 1])
            tracks_xy = np.zeros((1, frame_count, 2), dtype=np.float32)
            mask_bboxes = np.zeros((1, frame_count, 4), dtype=np.float32)
            for local_idx in range(frame_count):
                x = seed_x + float(local_idx)
                y = seed_y + float(local_idx)
                tracks_xy[0, local_idx] = [x, y]
                mask_bboxes[0, local_idx] = [x - 2.0, y - 2.0, x + 2.0, y + 2.0]
            return SimpleNamespace(
                tracks_xy=tracks_xy,
                visible_mask=np.ones((1, frame_count), dtype=bool),
                mask_areas=np.full((1, frame_count), 16.0, dtype=np.float32),
                mask_bboxes_xyxy=mask_bboxes,
                mask_scores=np.ones((1, frame_count), dtype=np.float32),
                representative_point_mode=settings.representative_point_mode,
                initial_mask_mode="bbox",
            )

        with (
            mock.patch.object(
                self.mask_backends,
                "_tracking_support",
                return_value=fake_support,
            ),
            mock.patch.object(
                self.mask_backends,
                "run_deva_offline",
                side_effect=fake_run_deva_offline,
            ),
        ):
            result = self.mask_backends.retrack_single_spot_deva_offline(
                track,
                2,
                all_files_map,
                list(range(7)),
                {
                    "checkpoint_path": "DEVA-propagation.pth",
                    "backend_mode": "subprocess",
                    "window_size": 3,
                    "window_overlap": 0,
                    "roi_size": 8,
                    "max_step_distance_px": 0.0,
                },
            )

        self.assertIs(result, track)
        self.assertEqual(fake_support.load_calls, [[2, 3, 4], [4, 5, 6]])
        self.assertEqual([call["frame_ids"] for call in run_calls], [[2, 3, 4], [4, 5, 6]])
        self.assertTrue(np.all(run_calls[0]["query_points_tyx"][:, 0] == 0.0))
        self.assertTrue(np.all(run_calls[1]["query_points_tyx"][:, 0] == 0.0))
        np.testing.assert_allclose(run_calls[0]["query_points_tyx"][0], [0.0, 10.0, 8.0])
        np.testing.assert_allclose(run_calls[1]["query_points_tyx"][0], [0.0, 12.0, 10.0])
        self.assertEqual(
            [
                (
                    call["frame_ids"],
                    call["start_local_idx"],
                    call["min_global_write_idx"],
                    call["global_start_idx"],
                )
                for call in fake_support.append_calls
            ],
            [
                ([2, 3, 4], 1, 0, 0),
                ([4, 5, 6], 1, 2, 2),
            ],
        )
        self.assertEqual(sorted(track.spots_by_image_id), list(range(7)))
        self.assertEqual(track.spots_by_image_id[6].source, "deva_offline")


if __name__ == "__main__":
    unittest.main()
