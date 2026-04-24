from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np


class _StubQRect:
    def __init__(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0) -> None:
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)


def _stub_load_tiff(_filepath):
    return np.zeros((8, 8), dtype=np.float32)


class RuntimeHelperImportMixin:
    runtime_module_names = (
        "trackleed.core.data_models",
        "trackleed.analysis.tracking_result_builders",
        "trackleed.analysis.tracking_visibility_utils",
        "trackleed.analysis.tracking_filter_utils",
        "trackleed.analysis.tracking_windowing_utils",
        "trackleed.analysis.tracking_family_adapters",
        "trackleed.analysis.tracking_retrack_utils",
    )

    def import_runtime_modules(self):
        originals: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "trackleed.io.image_reader",
            *self.runtime_module_names,
        ):
            originals[module_name] = sys.modules.get(module_name)
            sys.modules.pop(module_name, None)

        pyqt6_module = types.ModuleType("PyQt6")
        qtcore_module = types.ModuleType("PyQt6.QtCore")
        qtcore_module.QRect = _StubQRect
        pyqt6_module.QtCore = qtcore_module
        sys.modules["PyQt6"] = pyqt6_module
        sys.modules["PyQt6.QtCore"] = qtcore_module

        image_reader_module = types.ModuleType("trackleed.io.image_reader")
        image_reader_module.load_tiff = _stub_load_tiff
        sys.modules["trackleed.io.image_reader"] = image_reader_module

        self.addCleanup(self._restore_modules, originals)

        modules = {
            "data_models": importlib.import_module("trackleed.core.data_models"),
            "result_builders": importlib.import_module(
                "trackleed.analysis.tracking_result_builders"
            ),
            "visibility_utils": importlib.import_module(
                "trackleed.analysis.tracking_visibility_utils"
            ),
            "filter_utils": importlib.import_module(
                "trackleed.analysis.tracking_filter_utils"
            ),
            "windowing_utils": importlib.import_module(
                "trackleed.analysis.tracking_windowing_utils"
            ),
            "family_adapters": importlib.import_module(
                "trackleed.analysis.tracking_family_adapters"
            ),
            "retrack_utils": importlib.import_module(
                "trackleed.analysis.tracking_retrack_utils"
            ),
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


class TrackingWindowingUtilsTests(RuntimeHelperImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.modules = self.import_runtime_modules()
        self.windowing_utils = self.modules["windowing_utils"]

    def test_reseed_window_context_enforces_overlap_and_seed_frames(self) -> None:
        context = self.windowing_utils.build_reseed_tracking_window_context(
            list(range(10)),
            window_size=4,
            window_overlap=0,
        )

        self.assertTrue(context.require_reseed_overlap)
        self.assertEqual(context.overlap, 1)
        self.assertEqual([plan.start_idx for plan in context.plans], [0, 3, 6])
        self.assertEqual(context.plans[0].frame_ids, [0, 1, 2, 3])
        self.assertEqual(context.plans[0].seed_frame_ids, {0, 1})

    def test_plain_window_context_preserves_zero_overlap(self) -> None:
        context = self.windowing_utils.build_plain_tracking_window_context(
            list(range(10)),
            window_size=4,
            window_overlap=0,
        )

        self.assertFalse(context.require_reseed_overlap)
        self.assertEqual(context.overlap, 0)
        self.assertEqual(context.plans[0].seed_frame_ids, {0})
        self.assertEqual(context.plans[-1].end_idx, 10)
        self.assertEqual(self.windowing_utils.advance_window_coverage(2, 5), 4)


class TrackingFilterUtilsTests(RuntimeHelperImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.modules = self.import_runtime_modules()
        self.data_models = self.modules["data_models"]
        self.result_builders = self.modules["result_builders"]
        self.filter_utils = self.modules["filter_utils"]

    def _spot(self, *, x: float, y: float, source: str = "test"):
        return self.result_builders.build_spot_roi(
            spot_id=1,
            roi_rect=_StubQRect(0, 0, 8, 8),
            center_x=x,
            center_y=y,
            source=source,
            visible=True,
        )

    def test_motion_params_are_normalized(self) -> None:
        params = self.filter_utils.normalize_motion_filter_params(
            max_step_distance_px=-2.0,
            max_reacquisition_gap=-1,
            terminate_after_invisible_frames=-3,
            max_radial_jump_px=-4.0,
            lateral_angle_tolerance_deg=-5.0,
            min_directional_motion_px=-6.0,
            direction_mode="  OUTWARD  ",
        )

        self.assertEqual(params["max_step_distance_px"], 0.0)
        self.assertEqual(params["max_reacquisition_gap"], 0)
        self.assertEqual(params["terminate_after_invisible_frames"], 0)
        self.assertEqual(params["max_radial_jump_px"], 0.0)
        self.assertEqual(params["lateral_angle_tolerance_deg"], 0.0)
        self.assertEqual(params["min_directional_motion_px"], 0.0)
        self.assertEqual(params["direction_mode"], "outward")

    def test_gate_by_max_step_distance_marks_rejected_spot(self) -> None:
        spot = self._spot(x=10.0, y=0.0)
        accepted, reason = self.filter_utils.gate_by_max_step_distance(
            spot,
            last_valid_xy=np.array([0.0, 0.0], dtype=np.float32),
            max_step_distance_px=5.0,
            prefix="tapir",
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "max_step_distance")
        self.assertTrue(getattr(spot, "tapir_filter_rejected"))
        self.assertEqual(getattr(spot, "tapir_filter_rejected_reason"), "max_step_distance")

    def test_gate_by_direction_rejects_wrong_radial_direction(self) -> None:
        spot = self._spot(x=1.0, y=0.0)
        accepted, reason = self.filter_utils.gate_by_direction(
            spot,
            {},
            last_valid_xy=np.array([2.0, 0.0], dtype=np.float32),
            direction_center_xy=(0.0, 0.0),
            direction_mode="outward",
            lateral_angle_tolerance_deg=20.0,
            max_radial_jump_px=0.0,
            min_directional_motion_px=0.1,
            prefix="tapir",
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "direction_gating")

    def test_gate_bbox_quality_uses_presence_score_gating(self) -> None:
        spot = self._spot(x=2.0, y=2.0, source="tomp")
        context = self.filter_utils.build_bbox_quality_filter_context(
            max_bbox_aspect_ratio_change=0.0,
            max_bbox_center_jump_px=0.0,
            use_presence_score_gating=True,
            presence_threshold=0.5,
        )

        accepted, reason = self.filter_utils.gate_bbox_quality(
            spot,
            context=context,
            score=0.2,
            bbox_aspect_ratio=1.0,
            last_valid_bbox_aspect_ratio=1.0,
            bbox_center_xy=np.array([2.0, 2.0], dtype=np.float32),
            last_valid_bbox_center_xy=np.array([2.0, 2.0], dtype=np.float32),
            prefix="tomp",
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "presence_score")
        self.assertTrue(getattr(spot, "tomp_filter_rejected"))
        self.assertEqual(getattr(spot, "tomp_filter_rejected_reason"), "presence_score")


class TrackingFamilyAdaptersTests(RuntimeHelperImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.modules = self.import_runtime_modules()
        self.family_adapters = self.modules["family_adapters"]

    def test_point_family_adapter_context_uses_manual_center_and_recentering_defaults(self) -> None:
        context = self.family_adapters.build_point_family_adapter_context(
            {
                "keep_invisible_points_as_gaps": False,
                "roi_size": 24,
                "window_size": 48,
                "window_overlap": 6,
                "max_step_distance_px": 11.0,
                "direction_mode": "outward",
                "center_mode": "manual",
                "manual_center": (12.0, 15.0),
                "use_centroid_refinement": False,
            },
            all_files_map={},
            subset_frame_ids=[1, 2, 3],
            anchor_filepath=None,
            estimate_pattern_center_fn=lambda *args, **kwargs: {"center_x": 100.0, "center_y": 200.0},
        )

        self.assertFalse(context.keep_invisible_as_gaps)
        self.assertEqual(context.roi_size, 24)
        self.assertEqual(context.window_size, 48)
        self.assertEqual(context.window_overlap, 6)
        self.assertEqual(context.motion_filter_params["max_step_distance_px"], 11.0)
        self.assertEqual(context.direction_filter.direction_center_xy, (12.0, 15.0))
        self.assertEqual(context.recenter_mode, "none")

    def test_mask_and_bbox_family_contexts_collect_quality_params(self) -> None:
        mask_context = self.family_adapters.build_mask_family_adapter_context(
            {
                "max_mask_area_ratio_change": 1.7,
                "max_bbox_aspect_ratio_change": 2.5,
                "max_mask_components": 3,
                "max_bbox_center_jump_px": 9.0,
            },
            all_files_map={},
            subset_frame_ids=[1, 2],
            anchor_filepath=None,
            estimate_pattern_center_fn=lambda *args, **kwargs: None,
        )
        bbox_context = self.family_adapters.build_bbox_family_adapter_context(
            {
                "use_presence_score_gating": True,
                "presence_threshold": 0.42,
                "max_bbox_aspect_ratio_change": 2.0,
                "max_bbox_center_jump_px": 5.0,
            },
            all_files_map={},
            subset_frame_ids=[1, 2],
            anchor_filepath=None,
            estimate_pattern_center_fn=lambda *args, **kwargs: None,
        )

        self.assertEqual(mask_context.quality_filter_params["max_mask_components"], 3)
        self.assertEqual(mask_context.quality_filter_params["max_bbox_center_jump_px"], 9.0)
        self.assertTrue(bbox_context.quality_filter_params["use_presence_score_gating"])
        self.assertEqual(bbox_context.quality_filter_params["presence_threshold"], 0.42)


class TrackingResultBuildersTests(RuntimeHelperImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.modules = self.import_runtime_modules()
        self.result_builders = self.modules["result_builders"]

    def test_bbox_helpers_compute_center_and_metadata(self) -> None:
        center_xy, aspect_ratio = self.result_builders.bbox_center_and_aspect_ratio_from_xyxy(
            (2.0, 4.0, 10.0, 8.0)
        )
        self.assertTrue(np.allclose(center_xy, np.array([6.0, 6.0], dtype=np.float32)))
        self.assertEqual(aspect_ratio, 2.0)

        spot = self.result_builders.build_spot_roi(
            spot_id=9,
            roi_rect=_StubQRect(1, 2, 3, 4),
            center_x=6.0,
            center_y=6.0,
            source="tomp",
            visible=True,
        )
        self.result_builders.apply_bbox_xyxy_metadata(
            spot,
            bbox_xyxy=(2.0, 4.0, 10.0, 8.0),
            bbox_attribute_name="tomp_bbox_xyxy",
            aspect_ratio_attribute_name="tomp_bbox_aspect_ratio",
        )
        self.result_builders.apply_prefixed_point_metadata(
            spot,
            prefix="tomp",
            raw_x=6.1,
            raw_y=6.2,
            refined_x=6.0,
            refined_y=6.0,
            visible=True,
        )

        self.assertEqual(getattr(spot, "tomp_bbox_xyxy"), (2.0, 4.0, 10.0, 8.0))
        self.assertEqual(getattr(spot, "tomp_bbox_aspect_ratio"), 2.0)
        self.assertEqual(getattr(spot, "tomp_visibility"), True)


class TrackingRetrackUtilsTests(RuntimeHelperImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.modules = self.import_runtime_modules()
        self.result_builders = self.modules["result_builders"]
        self.retrack_utils = self.modules["retrack_utils"]

    def _spot(self, spot_id: int, x: float, y: float):
        return self.result_builders.build_spot_roi(
            spot_id=spot_id,
            roi_rect=_StubQRect(0, 0, 4, 4),
            center_x=x,
            center_y=y,
            source="manual",
            visible=True,
        )

    def test_prepare_retrack_start_context_trims_future_frames(self) -> None:
        track = self.result_builders.build_track(
            track_id=3,
            spots_by_image_id={
                1: self._spot(1, 1.0, 1.0),
                2: self._spot(2, 2.0, 2.0),
                3: self._spot(3, 3.0, 3.0),
            },
        )

        context = self.retrack_utils.prepare_retrack_start_context(
            track,
            start_frame_id=2,
            sorted_frame_ids=[1, 2, 3, 4],
            operation_label="re-track",
        )

        self.assertEqual(context.start_frame_index, 1)
        self.assertEqual(context.subset_frame_ids, [2, 3, 4])
        self.assertEqual(sorted(track.spots_by_image_id), [1, 2])
        self.assertIs(context.start_spot, track.spots_by_image_id[2])

    def test_finish_retrack_if_too_short_returns_track_and_reports_progress(self) -> None:
        track = self.result_builders.build_track(track_id=7)
        progress_values: list[int] = []

        result = self.retrack_utils.finish_retrack_if_too_short(
            track,
            subset_frame_ids=[5],
            progress_callback=progress_values.append,
        )

        self.assertIs(result, track)
        self.assertEqual(progress_values, [100])

