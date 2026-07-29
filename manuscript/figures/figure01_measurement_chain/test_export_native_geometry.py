#!/usr/bin/env python3
"""Tests for the targeted native-geometry re-export used by Figure 1."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trackleed.analysis import publication_native_geometry


class NativeGeometryExportTest(unittest.TestCase):
    def test_ostrack_bbox_summary_preserves_geometry_and_reduced_center(
        self,
    ) -> None:
        summary = publication_native_geometry.summarize_ostrack_bbox(
            np.asarray([10.0, 20.0, 8.0, 12.0], dtype=np.float32),
            expected_center_original_px=np.asarray(
                [54.6875, 101.5625],
                dtype=np.float64,
            ),
        )

        self.assertEqual(summary["geometry_type"], "bounding_box")
        self.assertEqual(summary["bbox_xywh_cache_px"], [10.0, 20.0, 8.0, 12.0])
        self.assertEqual(summary["bbox_xyxy_cache_px"], [10.0, 20.0, 18.0, 32.0])
        self.assertEqual(summary["bbox_xywh_original_px"], [39.0625, 78.125, 31.25, 46.875])
        self.assertEqual(summary["reduced_center_cache_px"], [14.0, 26.0])
        self.assertEqual(
            summary["reduced_center_original_px"],
            [54.6875, 101.5625],
        )
        self.assertEqual(summary["center_consistency_error_original_px"], 0.0)

    def test_dam4sam_mask_summary_preserves_pixels_bbox_and_centroid(
        self,
    ) -> None:
        mask = np.zeros((8, 9), dtype=bool)
        mask[2:5, 3:7] = True
        expected = np.asarray([4.5, 3.0], dtype=np.float64) / 0.256

        summary = publication_native_geometry.summarize_dam4sam_mask(
            mask,
            expected_center_original_px=expected,
        )

        self.assertEqual(summary["geometry_type"], "binary_mask")
        self.assertEqual(summary["mask_shape_cache_px"], [8, 9])
        self.assertEqual(summary["mask_area_cache_px"], 12)
        self.assertEqual(summary["mask_bbox_xyxy_cache_px"], [3.0, 2.0, 7.0, 5.0])
        self.assertEqual(summary["mask_bbox_xywh_cache_px"], [3.0, 2.0, 4.0, 3.0])
        self.assertEqual(summary["reduced_centroid_cache_px"], [4.5, 3.0])
        self.assertEqual(
            summary["reduced_centroid_original_px"],
            [17.578125, 11.71875],
        )
        self.assertEqual(summary["center_consistency_error_original_px"], 0.0)

    def test_empty_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            publication_native_geometry.summarize_dam4sam_mask(
                np.zeros((8, 9), dtype=bool),
                expected_center_original_px=np.asarray([1.0, 1.0]),
            )

    def test_reduced_center_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            publication_native_geometry.summarize_ostrack_bbox(
                np.asarray([10.0, 20.0, 8.0, 12.0], dtype=np.float32),
                expected_center_original_px=np.asarray([0.0, 0.0]),
            )

    def test_full_replay_comparison_recomputes_all_frame_metrics(self) -> None:
        expected_tracks = np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float64,
        )
        actual_tracks = expected_tracks.copy()
        actual_tracks[2, 0] += 1e-5
        available = np.asarray([True, False, True])

        summary = publication_native_geometry.compare_reduced_trajectories(
            actual_tracks,
            available,
            expected_tracks,
            available,
            method="example",
        )

        self.assertTrue(summary["availability_matches_all_frames"])
        self.assertTrue(
            summary["reduced_center_matches_all_available_frames"]
        )
        self.assertEqual(summary["available_frame_count"], 2)
        self.assertAlmostEqual(
            summary["maximum_reduced_center_error_cache_px"],
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
