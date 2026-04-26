from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
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
        fallback_center_x: float,
        fallback_center_y: float,
        fallback_roi_size: int,
    ):
        if bbox_xyxy is not None:
            bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
            if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
                x0, y0, x1, y1 = (float(v) for v in bbox[:4])
                return _StubQRect(
                    int(round(x0)),
                    int(round(y0)),
                    max(1, int(round(x1 - x0))),
                    max(1, int(round(y1 - y0))),
                )
        half = max(1, int(fallback_roi_size) // 2)
        return _StubQRect(
            int(round(float(fallback_center_x) - half)),
            int(round(float(fallback_center_y) - half)),
            max(1, int(fallback_roi_size)),
            max(1, int(fallback_roi_size)),
        )


class _StubRuntimeSettings:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class OstrackTrackingWindowingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_modules: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "trackleed.io.image_reader",
            "trackleed.core.data_models",
            "trackleed.analysis.ostrack_backend",
            "trackleed.analysis.tomp_backend",
            "trackleed.analysis.tracking_result_builders",
            "trackleed.analysis.tracking_visibility_utils",
            "trackleed.analysis.tracking_geometry",
            "trackleed.analysis.tracking_filter_utils",
            "trackleed.analysis.tracking_windowing_utils",
            "trackleed.analysis.tracking_family_adapters",
            "trackleed.analysis.tracking_retrack_utils",
            "trackleed.analysis.tracking_bbox_backends",
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
        self._install_bbox_backend_dependency_stubs()

        self.data_models = importlib.import_module("trackleed.core.data_models")
        self.bbox_backends = importlib.import_module("trackleed.analysis.tracking_bbox_backends")
        self.addCleanup(self._restore_modules)

    def _install_bbox_backend_dependency_stubs(self) -> None:
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
        ostrack_module.OstrackRuntimeSettings = _StubRuntimeSettings
        ostrack_module.discover_ostrack_checkpoints = lambda *args, **kwargs: []
        ostrack_module.select_default_ostrack_checkpoint = (
            lambda checkpoints, config_name=None: checkpoints[0] if checkpoints else None
        )
        ostrack_module.run_ostrack_offline = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Unexpected unpatched OSTrack backend call.")
        )
        sys.modules["trackleed.analysis.ostrack_backend"] = ostrack_module

        tomp_module = types.ModuleType("trackleed.analysis.tomp_backend")
        tomp_module.DEFAULT_TAMOS_MODELS_DIR = Path("tracking_models/tamos")
        tomp_module.DEFAULT_TOMP_MODELS_DIR = Path("tracking_models/tomp")
        tomp_module.TompRuntimeSettings = _StubRuntimeSettings
        tomp_module.discover_tamos_checkpoints = lambda *args, **kwargs: []
        tomp_module.discover_tomp_checkpoints = lambda *args, **kwargs: []
        tomp_module.expected_tomp_checkpoint_name_for_variant = lambda variant: f"{variant}.pth.tar"
        tomp_module.run_tomp_offline = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Unexpected ToMP backend call in OSTrack windowing test.")
        )
        tomp_module.select_default_tamos_checkpoint_for_variant = lambda *args, **kwargs: None
        tomp_module.select_default_tomp_checkpoint_for_variant = lambda *args, **kwargs: None
        sys.modules["trackleed.analysis.tomp_backend"] = tomp_module

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

    def test_ostrack_model_variant_normalizer_accepts_supported_upstream_configs(self) -> None:
        for variant_key in self.bbox_backends.SUPPORTED_OSTRACK_CONFIG_NAMES:
            with self.subTest(variant_key=variant_key):
                self.assertEqual(
                    self.bbox_backends._normalize_ostrack_model_variant_key(variant_key),
                    variant_key,
                )

    def test_ostrack_windowing_reseeds_each_window_from_local_zero(self) -> None:
        all_files_map = {frame_id: self._file(frame_id) for frame_id in range(5)}
        all_files_map[0].spots = [self._spot(1, 6.0, 8.0)]
        fake_support = _FakeTrackingSupport()
        run_calls: list[dict] = []

        def fake_run_ostrack_offline(
            *,
            frames,
            query_points_tyx,
            settings,
            query_boxes_xyxy=None,
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
            bboxes_xyxy = np.zeros((query_count, frame_count, 4), dtype=np.float32)
            bboxes_xywh = np.zeros((query_count, frame_count, 4), dtype=np.float32)
            for query_idx in range(query_count):
                seed_x = float(query_points_tyx[query_idx, 2])
                seed_y = float(query_points_tyx[query_idx, 1])
                for local_idx in range(frame_count):
                    x = seed_x + float(local_idx)
                    y = seed_y + float(local_idx)
                    tracks_xy[query_idx, local_idx] = [x, y]
                    bboxes_xyxy[query_idx, local_idx] = [x - 2.0, y - 2.0, x + 2.0, y + 2.0]
                    bboxes_xywh[query_idx, local_idx] = [x - 2.0, y - 2.0, 4.0, 4.0]
            return SimpleNamespace(
                tracks_xy=tracks_xy,
                visible_mask=np.ones((query_count, frame_count), dtype=bool),
                tracked_bboxes_xywh=bboxes_xywh,
                tracked_bboxes_xyxy=bboxes_xyxy,
                bbox_scores=np.ones((query_count, frame_count), dtype=np.float32),
                representative_point_mode=settings.representative_point_mode,
                model_variant=settings.model_variant,
                checkpoint_name="OSTrack_ep0300.pth.tar",
            )

        with (
            mock.patch.object(
                self.bbox_backends,
                "_tracking_support",
                return_value=fake_support,
            ),
            mock.patch.object(
                self.bbox_backends,
                "run_ostrack_offline",
                side_effect=fake_run_ostrack_offline,
            ),
        ):
            tracks = self.bbox_backends.track_spots_ostrack_offline(
                all_files_map,
                [0, 1, 2, 3, 4],
                0,
                {
                    "checkpoint_path": "OSTrack_ep0300.pth.tar",
                    "backend_mode": "subprocess",
                    "model_variant": "vitb_256_mae_ce_32x4_ep300",
                    "window_size": 3,
                    "window_overlap": 0,
                    "roi_size": 8,
                    "keep_invisible_points_as_gaps": False,
                    "max_step_distance_px": 7.0,
                    "max_reacquisition_gap": 2,
                    "terminate_after_invisible_frames": 3,
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
        self.assertTrue(np.all(run_calls[0]["query_points_tyx"][:, 0] == 0.0))
        self.assertTrue(np.all(run_calls[1]["query_points_tyx"][:, 0] == 0.0))
        np.testing.assert_allclose(run_calls[1]["query_points_tyx"][0], [0.0, 10.0, 8.0])
        np.testing.assert_allclose(run_calls[1]["query_boxes_xyxy"][0], [6.0, 8.0, 10.0, 12.0])
        self.assertEqual(sorted(tracks[0].spots_by_image_id), [0, 1, 2, 3, 4])
        self.assertEqual(tracks[0].spots_by_image_id[4].source, "ostrack_offline")
        self.assertEqual(
            tracks[0].spots_by_image_id[4].ostrack_checkpoint_name,
            "OSTrack_ep0300.pth.tar",
        )

    def test_ostrack_retrack_trims_future_points_and_uses_current_roi_seed(self) -> None:
        all_files_map = {frame_id: self._file(frame_id) for frame_id in range(5)}
        start_spot = self._spot(1, 10.0, 12.0)
        track = self.data_models.Track(
            track_id=1,
            spots_by_image_id={
                0: self._spot(1, 6.0, 8.0),
                1: self._spot(1, 8.0, 10.0),
                2: start_spot,
                3: self._spot(1, 99.0, 99.0),
                4: self._spot(1, 100.0, 100.0),
            },
        )
        fake_support = _FakeTrackingSupport()
        run_calls: list[dict] = []

        def fake_run_ostrack_offline(
            *,
            frames,
            query_points_tyx,
            settings,
            query_boxes_xyxy=None,
        ):
            query_points_tyx = np.asarray(query_points_tyx, dtype=np.float32)
            query_boxes_xyxy = np.asarray(query_boxes_xyxy, dtype=np.float32)
            run_calls.append(
                {
                    "frame_ids": [int(value) for value in frames[:, 0, 0, 0].tolist()],
                    "query_points_tyx": query_points_tyx.copy(),
                    "query_boxes_xyxy": query_boxes_xyxy.copy(),
                    "settings": settings,
                }
            )
            frame_count = int(frames.shape[0])
            seed_x = float(query_points_tyx[0, 2])
            seed_y = float(query_points_tyx[0, 1])
            tracks_xy = np.zeros((1, frame_count, 2), dtype=np.float32)
            bboxes_xyxy = np.zeros((1, frame_count, 4), dtype=np.float32)
            bboxes_xywh = np.zeros((1, frame_count, 4), dtype=np.float32)
            for local_idx in range(frame_count):
                x = seed_x + float(local_idx)
                y = seed_y + float(local_idx)
                tracks_xy[0, local_idx] = [x, y]
                bboxes_xyxy[0, local_idx] = [x - 2.0, y - 2.0, x + 2.0, y + 2.0]
                bboxes_xywh[0, local_idx] = [x - 2.0, y - 2.0, 4.0, 4.0]
            return SimpleNamespace(
                tracks_xy=tracks_xy,
                visible_mask=np.ones((1, frame_count), dtype=bool),
                tracked_bboxes_xywh=bboxes_xywh,
                tracked_bboxes_xyxy=bboxes_xyxy,
                bbox_scores=np.ones((1, frame_count), dtype=np.float32),
                representative_point_mode=settings.representative_point_mode,
                model_variant=settings.model_variant,
                checkpoint_name="OSTrack_ep0300.pth.tar",
            )

        with (
            mock.patch.object(
                self.bbox_backends,
                "_tracking_support",
                return_value=fake_support,
            ),
            mock.patch.object(
                self.bbox_backends,
                "run_ostrack_offline",
                side_effect=fake_run_ostrack_offline,
            ),
        ):
            updated_track = self.bbox_backends.retrack_single_spot_ostrack_offline(
                track,
                2,
                all_files_map,
                [0, 1, 2, 3, 4],
                {
                    "checkpoint_path": "OSTrack_ep0300.pth.tar",
                    "backend_mode": "subprocess",
                    "model_variant": "vitb_256_mae_ce_32x4_ep300",
                    "window_size": 99,
                    "window_overlap": 0,
                    "roi_size": 8,
                    "keep_invisible_points_as_gaps": False,
                    "max_step_distance_px": 0.0,
                    "max_reacquisition_gap": 0,
                    "terminate_after_invisible_frames": 0,
                    "max_bbox_aspect_ratio_change": 0.0,
                    "max_bbox_center_jump_px": 0.0,
                    "direction_mode": "none",
                },
            )

        self.assertIs(updated_track, track)
        self.assertEqual(fake_support.load_calls, [[2, 3, 4]])
        self.assertEqual([call["frame_ids"] for call in run_calls], [[2, 3, 4]])
        np.testing.assert_allclose(run_calls[0]["query_points_tyx"][0], [0.0, 12.0, 10.0])
        np.testing.assert_allclose(run_calls[0]["query_boxes_xyxy"][0], [8.0, 10.0, 12.0, 14.0])
        self.assertEqual(sorted(track.spots_by_image_id), [0, 1, 2, 3, 4])
        self.assertEqual(track.spots_by_image_id[3].source, "ostrack_offline")
        np.testing.assert_allclose(
            [track.spots_by_image_id[3].center_x, track.spots_by_image_id[3].center_y],
            [11.0, 13.0],
        )
        self.assertEqual(
            track.spots_by_image_id[4].ostrack_checkpoint_name,
            "OSTrack_ep0300.pth.tar",
        )

    def test_ostrack_sequence_terminates_before_reacquisition_after_gap(self) -> None:
        reference_spot = self._spot(1, 6.0, 8.0)
        track = self.data_models.Track(track_id=1, spots_by_image_id={0: reference_spot})
        state = self.bbox_backends.build_bbox_mask_motion_tracking_state(
            (6.0, 8.0),
            last_valid_mask_area=16.0,
            last_valid_bbox_aspect_ratio=1.0,
            last_valid_bbox_center_xy=(6.0, 8.0),
        )
        tracks_xy = np.asarray(
            [[6.0, 8.0], [6.0, 8.0], [6.0, 8.0], [9.0, 11.0]],
            dtype=np.float32,
        )
        bboxes_xywh = np.asarray(
            [[4.0, 6.0, 4.0, 4.0], [4.0, 6.0, 4.0, 4.0], [4.0, 6.0, 4.0, 4.0], [7.0, 9.0, 4.0, 4.0]],
            dtype=np.float32,
        )
        bboxes_xyxy = np.asarray(
            [[4.0, 6.0, 8.0, 10.0], [4.0, 6.0, 8.0, 10.0], [4.0, 6.0, 8.0, 10.0], [7.0, 9.0, 11.0, 13.0]],
            dtype=np.float32,
        )

        ok = self.bbox_backends._append_ostrack_sequence_to_track(
            track,
            state,
            reference_spot,
            [0, 1, 2, 3],
            [(20, 24)] * 4,
            tracks_xy,
            np.asarray([True, False, False, True], dtype=bool),
            bboxes_xywh,
            bboxes_xyxy,
            np.ones((4,), dtype=np.float32),
            roi_size=8,
            keep_invisible_as_gaps=True,
            representative_point_mode="bbox_center",
            model_variant="vitb_256_mae_ce_32x4_ep300",
            max_step_distance_px=0.0,
            max_reacquisition_gap=1,
            terminate_after_invisible_frames=0,
            max_bbox_aspect_ratio_change=0.0,
            max_bbox_center_jump_px=0.0,
            use_presence_score_gating=False,
            presence_threshold=0.0,
            direction_center_xy=None,
            direction_mode="none",
            lateral_angle_tolerance_deg=20.0,
            max_radial_jump_px=0.0,
            min_directional_motion_px=1.0,
            start_local_idx=1,
        )

        self.assertTrue(ok)
        self.assertTrue(state["terminated"])
        self.assertEqual(sorted(track.spots_by_image_id), [0])

    def test_ostrack_sequence_rejects_large_step(self) -> None:
        reference_spot = self._spot(1, 6.0, 8.0)
        track = self.data_models.Track(track_id=1, spots_by_image_id={0: reference_spot})
        state = self.bbox_backends.build_bbox_mask_motion_tracking_state(
            (6.0, 8.0),
            last_valid_mask_area=16.0,
            last_valid_bbox_aspect_ratio=1.0,
            last_valid_bbox_center_xy=(6.0, 8.0),
        )
        fake_support = _FakeTrackingSupport()

        with mock.patch.object(
            self.bbox_backends,
            "_tracking_support",
            return_value=fake_support,
        ):
            ok = self.bbox_backends._append_ostrack_sequence_to_track(
                track,
                state,
                reference_spot,
                [0, 1],
                [(200, 220), (200, 220)],
                np.asarray([[6.0, 8.0], [100.0, 100.0]], dtype=np.float32),
                np.asarray([True, True], dtype=bool),
                np.asarray([[4.0, 6.0, 4.0, 4.0], [98.0, 98.0, 4.0, 4.0]], dtype=np.float32),
                np.asarray([[4.0, 6.0, 8.0, 10.0], [98.0, 98.0, 102.0, 102.0]], dtype=np.float32),
                np.ones((2,), dtype=np.float32),
                roi_size=8,
                keep_invisible_as_gaps=False,
                representative_point_mode="bbox_center",
                model_variant="vitb_256_mae_ce_32x4_ep300",
                max_step_distance_px=5.0,
                max_reacquisition_gap=0,
                terminate_after_invisible_frames=0,
                max_bbox_aspect_ratio_change=0.0,
                max_bbox_center_jump_px=0.0,
                use_presence_score_gating=False,
                presence_threshold=0.0,
                direction_center_xy=None,
                direction_mode="none",
                lateral_angle_tolerance_deg=20.0,
                max_radial_jump_px=0.0,
                min_directional_motion_px=1.0,
                start_local_idx=1,
            )

        self.assertTrue(ok)
        self.assertIn(1, track.spots_by_image_id)
        rejected_spot = track.spots_by_image_id[1]
        self.assertFalse(getattr(rejected_spot, "ostrack_visibility"))
        self.assertTrue(getattr(rejected_spot, "ostrack_filter_rejected"))
        self.assertEqual(
            getattr(rejected_spot, "ostrack_filter_rejected_reason"),
            "max_step_distance",
        )
        np.testing.assert_allclose(state["last_valid_xy"], [6.0, 8.0])


if __name__ == "__main__":
    unittest.main()
