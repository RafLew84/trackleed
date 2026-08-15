import unittest
from pathlib import Path

import numpy as np

from trackleed.logic.workflows.seeded_tracking import (
    OpticalFlowConfig,
    RunContext,
    SeedRepresentation,
    SeedRect,
    SeededTrackingFrame,
    SeededTrackingCancelled,
    SeededTrackingExecutionError,
    SeededTrackingRequest,
    SkipAnchorOpticalFlowConfig,
    TrackingFamily,
    TrackingSeed,
    default_seeded_tracking_catalog,
    run_seeded_tracking,
)


def _translated_frames() -> dict[Path, np.ndarray]:
    reference = np.zeros((64, 64), dtype=np.uint8)
    reference[25:34, 24:35] = np.array(
        [
            [10, 30, 60, 90, 120, 150, 180, 210, 160, 80, 20],
            [20, 45, 80, 120, 160, 200, 240, 180, 120, 60, 15],
            [30, 60, 100, 150, 210, 255, 220, 160, 100, 45, 10],
            [25, 55, 95, 145, 205, 250, 215, 155, 95, 40, 8],
            [20, 45, 80, 120, 170, 210, 180, 130, 75, 35, 5],
            [15, 35, 60, 90, 125, 155, 135, 95, 55, 25, 3],
            [10, 25, 45, 65, 85, 105, 90, 65, 40, 18, 2],
            [5, 15, 30, 45, 60, 75, 65, 45, 25, 10, 1],
            [2, 8, 18, 28, 38, 48, 40, 28, 15, 6, 0],
        ],
        dtype=np.uint8,
    )
    translated = np.zeros_like(reference)
    translated[1:, 2:] = reference[:-1, :-2]
    return {Path("one.tif"): reference, Path("two.tif"): translated}


class SeededTrackingWorkflowTests(unittest.TestCase):
    def test_catalog_exposes_stable_optical_flow_ids_and_capabilities(self) -> None:
        catalog = default_seeded_tracking_catalog()

        self.assertEqual(
            tuple(method.method_id for method in catalog.methods),
            ("optical_flow", "optical_flow_skip_anchor"),
        )
        self.assertIs(
            catalog.method("optical_flow").capabilities.seed_representation,
            SeedRepresentation.POINT,
        )
        self.assertTrue(
            catalog.method("optical_flow_skip_anchor").capabilities.preserves_gaps
        )
        self.assertTrue(catalog.method("optical_flow").capabilities.supports_background)
        self.assertIs(
            catalog.method("optical_flow").family,
            TrackingFamily.CLASSICAL,
        )
        self.assertEqual(
            SkipAnchorOpticalFlowConfig().forward_backward_threshold,
            8.2,
        )

    def test_preflight_rejects_config_for_a_different_method(self) -> None:
        request = SeededTrackingRequest(
            method_id="optical_flow",
            frames=(
                SeededTrackingFrame(1, Path("one.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(1, 10.0, 10.0, SeedRect(6, 6, 8, 8)),),
            config=SkipAnchorOpticalFlowConfig(),
        )

        report = default_seeded_tracking_catalog().preflight(request)

        self.assertFalse(report.ready)
        self.assertIn("invalid_config", {issue.code for issue in report.issues})

    def test_structural_preflight_rejects_invalid_seed_geometry(self) -> None:
        request = SeededTrackingRequest(
            method_id="optical_flow",
            frames=(
                SeededTrackingFrame(1, Path("one.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(1, float("nan"), 10.0, SeedRect(6, 6, 0, 8)),),
            config=OpticalFlowConfig(),
        )

        report = default_seeded_tracking_catalog().preflight(request)

        self.assertIn("invalid_seed", {issue.code for issue in report.issues})

    def test_preflight_matches_requested_seed_representation_to_capabilities(
        self,
    ) -> None:
        request = SeededTrackingRequest(
            method_id="optical_flow",
            frames=(
                SeededTrackingFrame(1, Path("one.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(1, 10.0, 10.0, SeedRect(6, 6, 8, 8)),),
            config=OpticalFlowConfig(),
            seed_representation=SeedRepresentation.MASK,
        )

        report = default_seeded_tracking_catalog().preflight(request)

        self.assertIn(
            "unsupported_seed_representation",
            {issue.code for issue in report.issues},
        )

    def test_optical_flow_propagates_reference_seed_through_public_workflow(
        self,
    ) -> None:
        images = _translated_frames()
        progress: list[tuple[int, int]] = []
        request = SeededTrackingRequest(
            method_id="optical_flow",
            frames=(
                SeededTrackingFrame(1, Path("one.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(7, 29.0, 29.0, SeedRect(21, 21, 16, 16)),),
            config=OpticalFlowConfig(use_centroid_refinement=False),
        )

        result = run_seeded_tracking(
            request,
            RunContext(
                image_reader=images.__getitem__,
                report_progress=lambda value: progress.append(
                    (value.completed, value.total)
                ),
            ),
        )

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].track_id, 7)
        self.assertEqual(
            tuple(item.frame_id for item in result.tracks[0].observations),
            (1, 2),
        )
        tracked = result.tracks[0].observations[-1]
        self.assertAlmostEqual(tracked.center_x, 31.0, delta=0.75)
        self.assertAlmostEqual(tracked.center_y, 30.0, delta=0.75)
        self.assertEqual(
            tracked.roi,
            SeedRect(
                int(tracked.center_x - request.config.roi_size // 2),
                int(tracked.center_y - request.config.roi_size // 2),
                request.config.roi_size,
                request.config.roi_size,
            ),
        )
        self.assertEqual(progress, [(1, 1)])

    def test_skip_anchor_optical_flow_uses_its_stable_catalog_identity(self) -> None:
        images = _translated_frames()
        request = SeededTrackingRequest(
            method_id="optical_flow_skip_anchor",
            frames=(
                SeededTrackingFrame(1, Path("one.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(9, 29.0, 29.0, SeedRect(21, 21, 16, 16)),),
            config=SkipAnchorOpticalFlowConfig(
                use_centroid_refinement=False,
                use_ncc_check=False,
            ),
        )

        result = run_seeded_tracking(
            request,
            RunContext(image_reader=images.__getitem__),
        )

        self.assertEqual(result.tracks[0].track_id, 9)
        self.assertEqual(
            tuple(item.frame_id for item in result.tracks[0].observations),
            (1, 2),
        )
        tracked = result.tracks[0].observations[-1]
        self.assertAlmostEqual(tracked.center_x, 31.0, delta=0.75)
        self.assertAlmostEqual(tracked.center_y, 30.0, delta=0.75)
        self.assertEqual(tracked.source, "optical_flow_skip_anchor")

    def test_cancellation_after_progress_does_not_return_partial_tracks(self) -> None:
        images = _translated_frames()
        images[Path("three.tif")] = images[Path("two.tif")].copy()
        cancelled = False
        request = SeededTrackingRequest(
            method_id="optical_flow",
            frames=(
                SeededTrackingFrame(1, Path("one.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
                SeededTrackingFrame(3, Path("three.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(7, 29.0, 29.0, SeedRect(21, 21, 16, 16)),),
            config=OpticalFlowConfig(use_centroid_refinement=False),
        )

        def _record_progress(_progress: object) -> None:
            nonlocal cancelled
            cancelled = True

        with self.assertRaises(SeededTrackingCancelled):
            run_seeded_tracking(
                request,
                RunContext(
                    image_reader=images.__getitem__,
                    report_progress=_record_progress,
                    is_cancelled=lambda: cancelled,
                ),
            )

    def test_reader_failure_is_wrapped_as_execution_error(self) -> None:
        request = SeededTrackingRequest(
            method_id="optical_flow",
            frames=(
                SeededTrackingFrame(1, Path("missing.tif")),
                SeededTrackingFrame(2, Path("two.tif")),
            ),
            reference_frame_id=1,
            seeds=(TrackingSeed(7, 29.0, 29.0, SeedRect(21, 21, 16, 16)),),
            config=OpticalFlowConfig(),
        )

        with self.assertRaisesRegex(SeededTrackingExecutionError, "read failed"):
            run_seeded_tracking(
                request,
                RunContext(
                    image_reader=lambda _path: (_ for _ in ()).throw(
                        OSError("read failed")
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
