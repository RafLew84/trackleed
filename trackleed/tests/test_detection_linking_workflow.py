import inspect
import unittest

from trackleed.analysis import detection_linking_runtime, directional_linking_runtime, tracking
from trackleed.logic.workflows import detection_linking

from trackleed.logic.workflows.detection_linking import (
    DetectionLinkingCancelled,
    DetectionLinkingConfig,
    DetectionLinkingFamily,
    DetectionLinkingRequest,
    DetectionLinkingPreflightError,
    DirectionalLinkingConfig,
    RunContext,
    default_detection_linking_catalog,
    run_detection_linking,
)
from trackleed.logic.workflows.seeded_tracking import SeedRect
from trackleed.logic.workflows.spot_detection import (
    DetectedSpot,
    FrameDetections,
    RectSnapshot,
)


def _spot(
    spot_id: int,
    x: float,
    y: float,
    *,
    source: str = "auto",
) -> DetectedSpot:
    return DetectedSpot(
        spot_id,
        RectSnapshot(round(x) - 4, round(y) - 4, 8, 8),
        x,
        y,
        intensity=10.0,
        source=source,
    )


class DetectionLinkingWorkflowTests(unittest.TestCase):
    def test_headless_contract_and_runtime_have_no_gui_model_dependency(self) -> None:
        source = inspect.getsource(detection_linking) + inspect.getsource(
            detection_linking_runtime
        ) + inspect.getsource(
            directional_linking_runtime
        )
        for forbidden in ("PyQt", "trackleed.gui", "LeedFile", "SpotROI", "QRect"):
            self.assertNotIn(forbidden, source)
        shim_source = inspect.getsource(tracking.track_spots_hungarian)
        shim_source += inspect.getsource(tracking.track_spots_hungarian_directional)
        self.assertNotIn("trackleed.gui", shim_source)

    def test_hungarian_catalog_method_links_independent_detections(self) -> None:
        catalog = default_detection_linking_catalog()
        method = catalog.method("hungarian")
        request = DetectionLinkingRequest(
            method_id="hungarian",
            frames=(
                FrameDetections(10, (_spot(101, 10.0, 10.0), _spot(102, 30.0, 30.0))),
                FrameDetections(20, (_spot(201, 12.0, 11.0), _spot(202, 29.0, 32.0))),
            ),
            config=DetectionLinkingConfig(
                max_distance_px=5.0,
                max_gap=0,
                min_track_length=2,
            ),
        )

        result = run_detection_linking(request, RunContext(catalog=catalog))

        self.assertEqual(method.family, DetectionLinkingFamily.ASSIGNMENT)
        self.assertTrue(method.capabilities.consumes_detections)
        self.assertIsInstance(result.tracks[0].observations[0].roi, SeedRect)
        self.assertEqual(
            result.tracks[0].observations[0].source,
            "detection_linking:hungarian",
        )
        self.assertEqual(
            dict(result.tracks[0].observations[0].diagnostics)["detection_source"],
            "auto",
        )
        self.assertEqual(
            tuple(
                tuple(
                    (observation.frame_id, observation.center_x, observation.center_y)
                    for observation in track.observations
                )
                for track in result.tracks
            ),
            (
                ((10, 10.0, 10.0), (20, 12.0, 11.0)),
                ((10, 30.0, 30.0), (20, 29.0, 32.0)),
            ),
        )

    def test_directional_catalog_method_honors_manual_inward_direction(self) -> None:
        catalog = default_detection_linking_catalog()
        method = catalog.method("directional")
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(10, (_spot(1, 70.0, 50.0),)),
                FrameDetections(
                    20,
                    (
                        _spot(2, 75.0, 50.0),
                        _spot(3, 65.0, 50.0),
                    ),
                ),
            ),
            DirectionalLinkingConfig(
                center_mode="manual",
                manual_center=(50.0, 50.0),
                direction_mode="inward",
                base_max_distance_px=10.0,
                dynamic_max_distance=False,
                max_gap=0,
                max_misses=0,
                min_track_length=2,
            ),
        )

        result = run_detection_linking(request, RunContext(catalog=catalog))

        self.assertTrue(method.capabilities.requires_pattern_center)
        self.assertEqual(
            tuple(
                (item.frame_id, item.center_x)
                for item in result.tracks[0].observations
            ),
            ((10, 70.0), (20, 65.0)),
        )
        self.assertEqual(
            result.tracks[0].observations[0].source,
            "detection_linking:directional",
        )

    def test_directional_auto_center_links_symmetric_inward_motion(self) -> None:
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(
                    10,
                    (
                        _spot(1, 70.0, 50.0),
                        _spot(2, 30.0, 50.0),
                        _spot(3, 50.0, 70.0),
                        _spot(4, 50.0, 30.0),
                    ),
                ),
                FrameDetections(
                    20,
                    (
                        _spot(5, 65.0, 50.0),
                        _spot(6, 35.0, 50.0),
                        _spot(7, 50.0, 65.0),
                        _spot(8, 50.0, 35.0),
                    ),
                ),
            ),
            DirectionalLinkingConfig(
                center_mode="auto",
                direction_mode="inward",
                base_max_distance_px=10.0,
                dynamic_max_distance=False,
                min_track_length=2,
            ),
        )

        result = run_detection_linking(request, RunContext())

        self.assertEqual(len(result.tracks), 4)
        self.assertEqual(
            tuple(len(track.observations) for track in result.tracks),
            (2, 2, 2, 2),
        )

    def test_directional_auto_center_refuses_unusable_detections_in_preflight(self) -> None:
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(10, (_spot(1, 70.0, 50.0),)),
                FrameDetections(20, (_spot(2, 65.0, 50.0),)),
            ),
            DirectionalLinkingConfig(center_mode="auto"),
        )
        catalog = default_detection_linking_catalog()

        report = catalog.preflight(request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("pattern_center_unavailable",),
        )
        with self.assertRaises(DetectionLinkingPreflightError):
            run_detection_linking(request, RunContext(catalog=catalog))

    def test_directional_auto_direction_infers_outward_motion(self) -> None:
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(
                    10,
                    (_spot(1, 70.0, 50.0), _spot(2, 50.0, 70.0)),
                ),
                FrameDetections(
                    20,
                    (_spot(3, 75.0, 50.0), _spot(4, 50.0, 75.0)),
                ),
                FrameDetections(
                    30,
                    (
                        _spot(5, 70.0, 50.0),
                        _spot(6, 50.0, 70.0),
                        _spot(7, 80.0, 50.0),
                        _spot(8, 50.0, 80.0),
                    ),
                ),
            ),
            DirectionalLinkingConfig(
                center_mode="manual",
                manual_center=(50.0, 50.0),
                direction_mode="auto",
                base_max_distance_px=10.0,
                dynamic_max_distance=False,
                use_ring_penalty=False,
                min_track_length=3,
                min_tracks_for_flip=2,
                flip_hold_frames=1,
            ),
        )

        result = run_detection_linking(request, RunContext())

        self.assertEqual(
            tuple(
                track.observations[-1].center_x
                + track.observations[-1].center_y
                for track in result.tracks
            ),
            (130.0, 130.0),
        )

    def test_directional_manual_outward_prefers_increasing_radius(self) -> None:
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(10, (_spot(1, 70.0, 50.0),)),
                FrameDetections(
                    20,
                    (
                        _spot(2, 65.0, 50.0),
                        _spot(3, 75.0, 50.0),
                    ),
                ),
            ),
            DirectionalLinkingConfig(
                center_mode="manual",
                manual_center=(50.0, 50.0),
                direction_mode="outward",
                base_max_distance_px=10.0,
                dynamic_max_distance=False,
                use_ring_penalty=False,
                max_gap=0,
                max_misses=0,
                min_track_length=2,
            ),
        )

        result = run_detection_linking(request, RunContext())

        self.assertEqual(result.tracks[0].observations[-1].center_x, 75.0)

    def test_directional_lateral_tolerance_changes_assignment_preference(self) -> None:
        frames = (
            FrameDetections(10, (_spot(1, 70.0, 50.0),)),
            FrameDetections(
                20,
                (
                    _spot(2, 71.0, 53.0),
                    _spot(3, 75.0, 50.0),
                ),
            ),
        )

        strict = run_detection_linking(
            DetectionLinkingRequest(
                "directional",
                frames,
                DirectionalLinkingConfig(
                    center_mode="manual",
                    manual_center=(50.0, 50.0),
                    lateral_angle_tolerance_deg=0.0,
                    base_max_distance_px=10.0,
                    dynamic_max_distance=False,
                    use_ring_penalty=False,
                    min_track_length=2,
                ),
            ),
            RunContext(),
        )
        permissive = run_detection_linking(
            DetectionLinkingRequest(
                "directional",
                frames,
                DirectionalLinkingConfig(
                    center_mode="manual",
                    manual_center=(50.0, 50.0),
                    lateral_angle_tolerance_deg=90.0,
                    base_max_distance_px=10.0,
                    dynamic_max_distance=False,
                    use_ring_penalty=False,
                    min_track_length=2,
                ),
            ),
            RunContext(),
        )

        self.assertEqual(strict.tracks[0].observations[-1].center_x, 75.0)
        self.assertEqual(permissive.tracks[0].observations[-1].center_y, 53.0)

    def test_directional_ring_bins_penalize_cross_ring_assignment(self) -> None:
        frames = (
            FrameDetections(10, (_spot(1, 74.0, 50.0),)),
            FrameDetections(
                20,
                (
                    _spot(2, 76.0, 50.0),
                    _spot(3, 71.0, 50.0),
                ),
            ),
        )

        def linked_x(use_ring_penalty: bool) -> float:
            result = run_detection_linking(
                DetectionLinkingRequest(
                    "directional",
                    frames,
                    DirectionalLinkingConfig(
                        center_mode="manual",
                        manual_center=(50.0, 50.0),
                        ring_bin_px=25.0,
                        base_max_distance_px=10.0,
                        dynamic_max_distance=False,
                        use_ring_penalty=use_ring_penalty,
                        min_track_length=2,
                    ),
                ),
                RunContext(),
            )
            return result.tracks[0].observations[-1].center_x

        self.assertEqual(linked_x(use_ring_penalty=False), 76.0)
        self.assertEqual(linked_x(use_ring_penalty=True), 71.0)

    def test_directional_dynamic_distance_uses_recent_step_and_margin(self) -> None:
        frames = (
            FrameDetections(10, (_spot(1, 70.0, 50.0),)),
            FrameDetections(20, (_spot(2, 75.0, 50.0),)),
            FrameDetections(30, (_spot(3, 83.0, 50.0),)),
        )

        def track_lengths(dynamic: bool) -> tuple[int, ...]:
            result = run_detection_linking(
                DetectionLinkingRequest(
                    "directional",
                    frames,
                    DirectionalLinkingConfig(
                        center_mode="manual",
                        manual_center=(50.0, 50.0),
                        direction_mode="outward",
                        base_max_distance_px=10.0,
                        dynamic_max_distance=dynamic,
                        dynamic_margin_px=1.0,
                        use_ring_penalty=False,
                        min_track_length=2,
                    ),
                ),
                RunContext(),
            )
            return tuple(len(track.observations) for track in result.tracks)

        self.assertEqual(track_lengths(dynamic=False), (3,))
        self.assertEqual(track_lengths(dynamic=True), (2,))

    def test_directional_gap_and_miss_limits_control_retirement(self) -> None:
        frames = (
            FrameDetections(10, (_spot(1, 70.0, 50.0),)),
            FrameDetections(20, ()),
            FrameDetections(30, (_spot(2, 75.0, 50.0),)),
        )

        def linked_frames(max_gap: int, max_misses: int) -> tuple[int, ...]:
            result = run_detection_linking(
                DetectionLinkingRequest(
                    "directional",
                    frames,
                    DirectionalLinkingConfig(
                        center_mode="manual",
                        manual_center=(50.0, 50.0),
                        direction_mode="outward",
                        base_max_distance_px=10.0,
                        dynamic_max_distance=False,
                        use_ring_penalty=False,
                        max_gap=max_gap,
                        max_misses=max_misses,
                        min_track_length=2,
                    ),
                ),
                RunContext(),
            )
            if not result.tracks:
                return ()
            return tuple(
                observation.frame_id for observation in result.tracks[0].observations
            )

        self.assertEqual(linked_frames(max_gap=1, max_misses=1), (10, 30))
        self.assertEqual(linked_frames(max_gap=0, max_misses=5), ())
        self.assertEqual(linked_frames(max_gap=5, max_misses=0), ())

    def test_directional_invalid_manual_center_is_structured_preflight(self) -> None:
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(10, (_spot(1, 70.0, 50.0),)),
                FrameDetections(20, (_spot(2, 75.0, 50.0),)),
            ),
            DirectionalLinkingConfig(
                center_mode="manual",
                manual_center=None,
            ),
        )

        report = default_detection_linking_catalog().preflight(request)

        self.assertEqual(tuple(issue.code for issue in report.issues), ("invalid_config",))

    def test_directional_invalid_coordinates_are_structured_preflight(self) -> None:
        for value in ("not-a-number", "70.0"):
            with self.subTest(value=value):
                invalid = DetectedSpot(
                    1,
                    RectSnapshot(1, 1, 8, 8),
                    value,  # type: ignore[arg-type]
                    50.0,
                    10.0,
                    "auto",
                )
                request = DetectionLinkingRequest(
                    "directional",
                    (
                        FrameDetections(10, (invalid, invalid, invalid, invalid)),
                        FrameDetections(20, (_spot(2, 70.0, 50.0),)),
                    ),
                    DirectionalLinkingConfig(center_mode="auto"),
                )

                report = default_detection_linking_catalog().preflight(request)

                self.assertEqual(tuple(issue.code for issue in report.issues), ("invalid_detection",))

    def test_directional_cancellation_does_not_return_partial_tracks(self) -> None:
        request = DetectionLinkingRequest(
            "directional",
            (
                FrameDetections(10, (_spot(1, 70.0, 50.0),)),
                FrameDetections(20, (_spot(2, 75.0, 50.0),)),
                FrameDetections(30, (_spot(3, 80.0, 50.0),)),
            ),
            DirectionalLinkingConfig(
                center_mode="manual",
                manual_center=(50.0, 50.0),
                direction_mode="outward",
            ),
        )
        progress = []

        with self.assertRaises(DetectionLinkingCancelled):
            run_detection_linking(
                request,
                RunContext(
                    report_progress=progress.append,
                    is_cancelled=lambda: bool(progress),
                ),
            )

        self.assertEqual(tuple(item.frame_id for item in progress), (10,))

    def test_missing_detections_is_structured_preflight_failure(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (FrameDetections(10, ()), FrameDetections(20, ())),
            DetectionLinkingConfig(),
        )
        catalog = default_detection_linking_catalog()

        report = catalog.preflight(request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("detections_required",),
        )
        with self.assertRaises(DetectionLinkingPreflightError):
            run_detection_linking(request, RunContext(catalog=catalog))

    def test_catalog_adapters_reject_manual_seeds_as_detection_identity(self) -> None:
        configs = {
            "hungarian": DetectionLinkingConfig(),
            "directional": DirectionalLinkingConfig(
                center_mode="manual",
                manual_center=(50.0, 50.0),
            ),
        }
        for method_id, config in configs.items():
            with self.subTest(method_id=method_id):
                request = DetectionLinkingRequest(
                    method_id,
                    (
                        FrameDetections(
                            10,
                            (_spot(1, 10.0, 10.0, source="manual_point"),),
                        ),
                        FrameDetections(20, (_spot(2, 11.0, 11.0),)),
                    ),
                    config,
                )

                report = default_detection_linking_catalog().preflight(request)

                self.assertEqual(
                    tuple(issue.code for issue in report.issues),
                    ("independent_detections_required",),
                )

    def test_wrongly_typed_config_fails_closed(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0),)),
                FrameDetections(20, (_spot(2, 11.0, 11.0),)),
            ),
            DetectionLinkingConfig(max_distance_px="invalid"),  # type: ignore[arg-type]
        )

        report = default_detection_linking_catalog().preflight(request)

        self.assertEqual(
            tuple(issue.code for issue in report.issues),
            ("invalid_config",),
        )

    def test_cancellation_after_progress_does_not_return_partial_tracks(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0),)),
                FrameDetections(20, (_spot(2, 11.0, 11.0),)),
                FrameDetections(30, (_spot(3, 12.0, 12.0),)),
            ),
            DetectionLinkingConfig(),
        )
        progress = []
        context = RunContext(
            report_progress=progress.append,
            is_cancelled=lambda: bool(progress),
        )

        with self.assertRaises(DetectionLinkingCancelled):
            run_detection_linking(request, context)

        self.assertEqual(tuple(item.frame_id for item in progress), (10,))

    def test_max_gap_links_across_one_empty_detection_frame(self) -> None:
        request = DetectionLinkingRequest(
            "hungarian",
            (
                FrameDetections(10, (_spot(1, 10.0, 10.0),)),
                FrameDetections(20, ()),
                FrameDetections(30, (_spot(3, 12.0, 12.0),)),
            ),
            DetectionLinkingConfig(max_gap=1),
        )

        result = run_detection_linking(request, RunContext())

        self.assertEqual(
            tuple(item.frame_id for item in result.tracks[0].observations),
            (10, 30),
        )


if __name__ == "__main__":
    unittest.main()
