from __future__ import annotations

import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from PyQt6.QtCore import QRect

from trackleed.analysis import tracking
from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.tests.test_seeded_tracking_workflow import _translated_frames


def _frame(frame_id: int, *, seed: bool = False) -> LeedFile:
    spots = (
        [SpotROI(7, QRect(21, 21, 16, 16), 29.0, 29.0, 1.0, "manual")] if seed else []
    )
    return LeedFile(
        Path("one.tif" if frame_id == 1 else "two.tif"),
        f"frame-{frame_id}.tif",
        False,
        "Cu",
        111,
        "G",
        300.0,
        80.0,
        frame_id,
        None,
        spots,
    )


class SeededTrackingLegacyAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = _translated_frames()
        self.files = {1: _frame(1, seed=True), 2: _frame(2)}

    def test_classic_entrypoint_returns_track_models_from_headless_result(self) -> None:
        progress: list[int] = []
        with patch.object(tracking, "load_tiff", self.images.__getitem__):
            tracks = tracking.track_spots_optical_flow(
                self.files,
                [1, 2],
                1,
                {
                    "roi_size": 32,
                    "lk_win_size": 41,
                    "lk_max_level": 4,
                    "lk_max_iterations": 50,
                    "lk_epsilon": 0.01,
                    "use_centroid_refinement": False,
                },
                progress_callback=progress.append,
            )

        self.assertIsNotNone(tracks)
        self.assertEqual(tuple(tracks[0].spots_by_image_id), (1, 2))
        tracked = tracks[0].spots_by_image_id[2]
        self.assertAlmostEqual(tracked.center_x, 31.0, delta=0.75)
        self.assertEqual(tracked.roi_rect.x(), int(tracked.center_x - 16))
        self.assertEqual(progress[-1], 100)

    def test_skip_anchor_entrypoint_maps_legacy_config(self) -> None:
        with patch.object(tracking, "load_tiff", self.images.__getitem__):
            tracks = tracking.track_spots_optical_flow_skip_anchor(
                self.files,
                [1, 2],
                1,
                {
                    "roi_size": 32,
                    "use_centroid_refinement": False,
                    "use_ncc_check": False,
                },
            )

        self.assertIsNotNone(tracks)
        self.assertEqual(tuple(tracks[0].spots_by_image_id), (1, 2))
        self.assertEqual(
            tracks[0].spots_by_image_id[2].source,
            "optical_flow_skip_anchor",
        )

    def test_cancelled_legacy_entrypoint_returns_none(self) -> None:
        cancelled = Event()
        cancelled.set()

        tracks = tracking.track_spots_optical_flow(
            self.files,
            [1, 2],
            1,
            {},
            cancel_event=cancelled,
        )

        self.assertIsNone(tracks)

    def test_legacy_entrypoints_keep_their_distinct_lk_defaults(self) -> None:
        classic = tracking._legacy_optical_flow_config({})
        skip_anchor = tracking._legacy_skip_anchor_config({})

        self.assertEqual(
            (classic.win_size, classic.max_level, classic.max_iterations),
            (31, 3, 10),
        )
        self.assertEqual(
            (skip_anchor.win_size, skip_anchor.max_level, skip_anchor.max_iterations),
            (41, 4, 50),
        )
        self.assertEqual(skip_anchor.forward_backward_threshold, 8.2)


if __name__ == "__main__":
    unittest.main()
