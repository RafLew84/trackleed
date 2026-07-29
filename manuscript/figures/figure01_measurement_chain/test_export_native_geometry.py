#!/usr/bin/env python3
"""Tests for the targeted native-geometry re-export used by Figure 1."""

from __future__ import annotations

import unittest

import numpy as np

import export_native_geometry


class NativeGeometryExportTest(unittest.TestCase):
    def test_ostrack_bbox_summary_preserves_geometry_and_reduced_center(
        self,
    ) -> None:
        summary = export_native_geometry.summarize_ostrack_bbox(
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

        summary = export_native_geometry.summarize_dam4sam_mask(
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
            export_native_geometry.summarize_dam4sam_mask(
                np.zeros((8, 9), dtype=bool),
                expected_center_original_px=np.asarray([1.0, 1.0]),
            )

    def test_reduced_center_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            export_native_geometry.summarize_ostrack_bbox(
                np.asarray([10.0, 20.0, 8.0, 12.0], dtype=np.float32),
                expected_center_original_px=np.asarray([0.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
