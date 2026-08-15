from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

import numpy as np

from trackleed.analysis.legacy_track_geometry import (
    GeometryObservationRoi,
    LegacyCenterMode,
    LegacyGeometryRequest,
    derive_legacy_geometry,
)
from trackleed.logic.workflows.track_analysis import (
    FitFingerprint,
    FitFailure,
    FitObservation,
    FitObservationInput,
    FitRequest,
    FitRunContext,
    FitSet,
    FitStatus,
    Rect,
    fit,
)
from trackleed.logic.workflows.track_geometry import (
    GeometryCancelled,
    GeometryMethod,
    GeometryParameters,
    GeometryPreflightError,
    GeometryRequest,
    GeometryRunContext,
    GeometryStatus,
    GeometryUnavailableReason,
    ManualEllipse,
    derive_geometry,
)


def _gaussian_frame() -> np.ndarray:
    yy, xx = np.indices((31, 31), dtype=np.float64)
    return 0.05 + np.exp(-((xx - 15.0) ** 2 + (yy - 14.0) ** 2) / 18.0)


def _seed_observation() -> FitObservation:
    source_path = Path("synthetic/frame.tif")
    fit_set = fit(
        FitRequest(
            observations=(FitObservationInput(1, 1, source_path, Rect(4, 3, 23, 23)),)
        ),
        FitRunContext(image_reader=lambda _path: _gaussian_frame()),
    )
    observation = fit_set.observation(1, 1)
    assert observation is not None and observation.result is not None
    return observation


def _fit_set(points: tuple[tuple[int, int, float, float], ...]) -> FitSet:
    seed = _seed_observation()
    assert seed.result is not None
    observations = []
    for track_id, frame_id, center_x, center_y in points:
        digest = sha256(
            f"{track_id}:{frame_id}:{center_x}:{center_y}".encode()
        ).hexdigest()
        observations.append(
            replace(
                seed,
                track_id=track_id,
                frame_id=frame_id,
                fingerprint=FitFingerprint(digest),
                status=FitStatus.FITTED,
                result=replace(
                    seed.result,
                    global_center_x=center_x,
                    global_center_y=center_y,
                ),
            )
        )
    return FitSet(tuple(observations))


def _failed_observation(
    track_id: int,
    frame_id: int,
    roi: Rect,
) -> FitObservation:
    seed = _seed_observation()
    digest = sha256(f"failed:{track_id}:{frame_id}:{roi}".encode()).hexdigest()
    return replace(
        seed,
        track_id=track_id,
        frame_id=frame_id,
        fingerprint=FitFingerprint(digest),
        status=FitStatus.FAILED,
        provenance=replace(seed.provenance, original_roi=roi),
        result=None,
        failure=FitFailure("fit_failed", "Synthetic controlled failure"),
    )


class TrackGeometryWorkflowTests(unittest.TestCase):
    def test_circle_fit_returns_immutable_center_radii_and_frame_summary(self) -> None:
        fit_set = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
                (3, 10, 5.0, 20.0),
                (4, 10, 10.0, 15.0),
            )
        )
        snapshot = fit_set

        geometry = derive_geometry(
            fit_set,
            GeometryRequest(
                method=GeometryMethod.CIRCLE_FIT,
                center_reference_track_ids=(1, 2, 3, 4),
            ),
        )

        self.assertEqual(geometry.status, GeometryStatus.AVAILABLE)
        self.assertAlmostEqual(geometry.frames[0].center.center_x, 10.0)
        self.assertAlmostEqual(geometry.frames[0].center.center_y, 20.0)
        self.assertEqual(
            tuple(round(item.radius, 12) for item in geometry.observations),
            (5.0, 5.0, 5.0, 5.0),
        )
        self.assertAlmostEqual(geometry.frames[0].average_radius, 5.0)
        self.assertIs(fit_set, snapshot)
        self.assertFalse(hasattr(fit_set.observations[0].result, "radius"))

    def test_circle_fit_returns_machine_readable_unavailability(self) -> None:
        cases = (
            (
                _fit_set(((1, 10, 15.0, 20.0),)),
                GeometryRequest(
                    method=GeometryMethod.CIRCLE_FIT,
                    center_reference_track_ids=(),
                ),
                GeometryUnavailableReason.NO_REFERENCE_TRACKS,
            ),
            (
                _fit_set(((1, 10, 15.0, 20.0), (2, 10, 10.0, 25.0))),
                GeometryRequest(
                    method=GeometryMethod.CIRCLE_FIT,
                    center_reference_track_ids=(1, 2),
                ),
                GeometryUnavailableReason.INSUFFICIENT_POINTS,
            ),
            (
                _fit_set(
                    (
                        (1, 10, 5.0, 20.0),
                        (2, 10, 10.0, 20.0),
                        (3, 10, 15.0, 20.0),
                    )
                ),
                GeometryRequest(
                    method=GeometryMethod.CIRCLE_FIT,
                    center_reference_track_ids=(1, 2, 3),
                ),
                GeometryUnavailableReason.COLLINEAR_POINTS,
            ),
            (
                _fit_set(
                    (
                        (1, 10, 15.0, 20.0),
                        (2, 10, 10.0, 25.0),
                        (3, 10, 5.0, 20.0),
                        (4, 10, 10.0, 15.0),
                    )
                ),
                GeometryRequest(
                    method=GeometryMethod.CIRCLE_FIT,
                    center_reference_track_ids=(1, 2, 3, 4),
                    parameters=GeometryParameters(max_condition_number=1.0),
                ),
                GeometryUnavailableReason.NUMERICALLY_UNRELIABLE,
            ),
            (
                _fit_set(((1, 10, 15.0, 20.0),)),
                GeometryRequest(
                    method=GeometryMethod.MANUAL_ELLIPSE,
                    center_reference_track_ids=(1,),
                ),
                GeometryUnavailableReason.MANUAL_ELLIPSE_REQUIRED,
            ),
            (
                _fit_set(
                    (
                        (1, 10, 15.0, 20.0),
                        (2, 10, 10.0, 25.0),
                        (3, 10, float("nan"), 20.0),
                    )
                ),
                GeometryRequest(
                    method=GeometryMethod.CIRCLE_FIT,
                    center_reference_track_ids=(1, 2, 3),
                ),
                GeometryUnavailableReason.NUMERICALLY_UNRELIABLE,
            ),
        )
        for fit_set, request, reason in cases:
            with self.subTest(reason=reason):
                geometry = derive_geometry(fit_set, request)
                self.assertEqual(
                    geometry.status,
                    GeometryStatus.GEOMETRY_UNAVAILABLE,
                )
                self.assertEqual(geometry.failure.reason, reason)
                self.assertEqual(geometry.frames[0].failure.reason, reason)

    def test_every_current_center_method_uses_existing_fits(self) -> None:
        fit_set = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
                (3, 10, 5.0, 20.0),
                (4, 10, 10.0, 15.0),
                (1, 20, 17.0, 22.0),
                (2, 20, 12.0, 27.0),
                (3, 20, 7.0, 22.0),
                (4, 20, 12.0, 17.0),
            )
        )
        references = (1, 2, 3, 4)
        expected_centers = {
            GeometryMethod.PER_FRAME: ((10.0, 20.0), (12.0, 22.0)),
            GeometryMethod.CIRCLE_FIT: ((10.0, 20.0), (12.0, 22.0)),
            GeometryMethod.ELLIPSE_FIT: ((10.0, 20.0), (12.0, 22.0)),
            GeometryMethod.CIRCLE_FIT_TEMPORAL: ((11.0, 21.0), (11.0, 21.0)),
            GeometryMethod.GLOBAL_MEDIAN: ((11.0, 21.0), (11.0, 21.0)),
            GeometryMethod.MANUAL_ELLIPSE: ((9.0, 19.0), (9.0, 19.0)),
        }

        for method, expected in expected_centers.items():
            with self.subTest(method=method):
                geometry = derive_geometry(
                    fit_set,
                    GeometryRequest(
                        method=method,
                        center_reference_track_ids=references,
                        manual_ellipse=(
                            ManualEllipse(9.0, 19.0, 10.0, 5.0)
                            if method is GeometryMethod.MANUAL_ELLIPSE
                            else None
                        ),
                    ),
                )
                self.assertEqual(geometry.status, GeometryStatus.AVAILABLE)
                self.assertEqual(
                    tuple(
                        (
                            round(frame.center.center_x, 9),
                            round(frame.center.center_y, 9),
                        )
                        for frame in geometry.frames
                    ),
                    expected,
                )
                if method is GeometryMethod.CIRCLE_FIT_TEMPORAL:
                    self.assertTrue(
                        all(
                            frame.center.circle_radius is not None
                            for frame in geometry.frames
                        )
                    )

    def test_explicit_bbox_fallback_records_the_variant_actually_used(self) -> None:
        fitted = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
            )
        )
        failed = _failed_observation(3, 10, Rect(3, 18, 4, 4))
        fit_set = FitSet((*fitted.observations, failed))

        geometry = derive_geometry(
            fit_set,
            GeometryRequest(
                method=GeometryMethod.CIRCLE_FIT_BBOX_FALLBACK,
                center_reference_track_ids=(1, 2, 3),
            ),
        )

        self.assertEqual(geometry.status, GeometryStatus.AVAILABLE)
        self.assertEqual(
            geometry.frames[0].center.method_used,
            GeometryMethod.CIRCLE_FIT_BBOX_FALLBACK,
        )
        self.assertAlmostEqual(geometry.frames[0].center.center_x, 10.0)
        self.assertAlmostEqual(geometry.frames[0].center.center_y, 20.0)

    def test_geometry_fingerprint_invalidates_only_geometry_inputs(self) -> None:
        fit_set = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
                (3, 10, 5.0, 20.0),
                (4, 10, 10.0, 15.0),
            )
        )
        request = GeometryRequest(
            method=GeometryMethod.CIRCLE_FIT,
            center_reference_track_ids=(1, 2, 3, 4),
        )
        baseline = derive_geometry(fit_set, request)
        cached = derive_geometry(fit_set, replace(request, previous=baseline))
        metadata_only = FitSet(
            tuple(
                replace(
                    item,
                    provenance=replace(
                        item.provenance,
                        energy=999.0,
                        temperature=123.0,
                    ),
                )
                for item in fit_set.observations
            )
        )

        self.assertIs(cached, baseline)
        self.assertIs(
            derive_geometry(
                metadata_only,
                replace(request, previous=baseline),
            ),
            baseline,
        )
        changed_requests = (
            replace(request, method=GeometryMethod.PER_FRAME),
            replace(
                request,
                parameters=replace(request.parameters, ransac_threshold=2.0),
            ),
            replace(request, center_reference_track_ids=(4, 3, 2, 1)),
            replace(
                request,
                manual_ellipse=ManualEllipse(10.0, 20.0, 5.0, 4.0),
            ),
        )
        for changed in changed_requests:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    derive_geometry(fit_set, changed).fingerprint,
                    baseline.fingerprint,
                )
        self.assertNotEqual(
            derive_geometry(
                FitSet(tuple(reversed(fit_set.observations))), request
            ).fingerprint,
            baseline.fingerprint,
        )
        first = fit_set.observations[0]
        assert first.result is not None
        changed_error = replace(
            first,
            result=replace(
                first.result,
                errors=replace(
                    first.result.errors,
                    center_x=first.result.errors.center_x + 1.0,
                ),
            ),
        )
        self.assertNotEqual(
            derive_geometry(
                FitSet((changed_error, *fit_set.observations[1:])),
                request,
            ).fingerprint,
            baseline.fingerprint,
        )

    def test_ransac_is_repeatable_and_stable_across_declared_seeds(self) -> None:
        fit_set = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
                (3, 10, 5.0, 20.0),
                (4, 10, 10.0, 15.0),
                (5, 10, 30.0, 40.0),
                (6, 10, -10.0, 0.0),
            )
        )
        centers = []
        for seed in (1, 7, 20260811, 999999):
            request = GeometryRequest(
                method=GeometryMethod.PER_FRAME,
                center_reference_track_ids=(1, 2, 3, 4, 5, 6),
                parameters=GeometryParameters(
                    random_seed=seed,
                    ransac_threshold=0.5,
                ),
            )
            first = derive_geometry(fit_set, request)
            repeated = derive_geometry(fit_set, request)
            self.assertEqual(first, repeated)
            centers.append(
                (
                    first.frames[0].center.center_x,
                    first.frames[0].center.center_y,
                )
            )

        np.testing.assert_allclose(
            centers,
            np.asarray([(10.0, 20.0)] * 4),
            rtol=1e-12,
            atol=1e-9,
        )

    def test_cancellation_never_exposes_partial_geometry(self) -> None:
        fit_set = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
                (3, 10, 5.0, 20.0),
                (4, 10, 10.0, 15.0),
                (1, 20, 17.0, 22.0),
                (2, 20, 12.0, 27.0),
                (3, 20, 7.0, 22.0),
                (4, 20, 12.0, 17.0),
            )
        )
        request = GeometryRequest(
            method=GeometryMethod.CIRCLE_FIT,
            center_reference_track_ids=(1, 2, 3, 4),
        )
        previous = derive_geometry(fit_set, request)
        completed = 0

        def report_progress(_progress) -> None:
            nonlocal completed
            completed += 1

        with self.assertRaises(GeometryCancelled):
            derive_geometry(
                fit_set,
                replace(request, method=GeometryMethod.PER_FRAME),
                GeometryRunContext(
                    report_progress=report_progress,
                    is_cancelled=lambda: completed == 1,
                ),
            )

        self.assertEqual(previous.status, GeometryStatus.AVAILABLE)
        self.assertEqual(len(previous.frames), 2)

    def test_geometry_matches_deterministic_full_run_tolerance(self) -> None:
        fit_set = _fit_set(
            (
                (1, 10, 15.0, 20.0),
                (2, 10, 10.0, 25.0),
                (3, 10, 5.0, 20.0),
                (4, 10, 10.0, 15.0),
            )
        )
        geometry = derive_geometry(
            fit_set,
            GeometryRequest(
                method=GeometryMethod.CIRCLE_FIT,
                center_reference_track_ids=(1, 2, 3, 4),
            ),
        )
        legacy = derive_legacy_geometry(
            fit_set,
            LegacyGeometryRequest(
                observation_rois=tuple(
                    GeometryObservationRoi(
                        item.track_id,
                        item.frame_id,
                        item.provenance.original_roi,
                    )
                    for item in fit_set.observations
                ),
                center_mode=LegacyCenterMode.CIRCLE_FIT,
            ),
        )

        np.testing.assert_allclose(
            [
                geometry.frames[0].center.center_x,
                geometry.frames[0].center.center_y,
                *(item.radius for item in geometry.observations),
            ],
            [
                dict(legacy.frames[0].pattern_center)["center_x"],
                dict(legacy.frames[0].pattern_center)["center_y"],
                *(item.radius for item in legacy.observations),
            ],
            rtol=1e-12,
            atol=1e-9,
        )

    def test_temporal_geometry_preserves_legacy_clamp_and_error_inflation(self) -> None:
        points = []
        for frame_id, center_x in ((10, 10.0), (20, 12.0), (30, 90.0)):
            points.extend(
                (
                    (1, frame_id, center_x + 5.0, 20.0),
                    (2, frame_id, center_x, 25.0),
                    (3, frame_id, center_x - 5.0, 20.0),
                    (4, frame_id, center_x, 15.0),
                )
            )
        fit_set = _fit_set(tuple(points))
        request = GeometryRequest(
            method=GeometryMethod.CIRCLE_FIT_TEMPORAL,
            center_reference_track_ids=(1, 2, 3, 4),
        )

        geometry = derive_geometry(fit_set, request)
        legacy = derive_legacy_geometry(
            fit_set,
            LegacyGeometryRequest(
                observation_rois=tuple(
                    GeometryObservationRoi(
                        item.track_id,
                        item.frame_id,
                        item.provenance.original_roi,
                    )
                    for item in fit_set.observations
                ),
                center_mode=LegacyCenterMode.CIRCLE_FIT_TEMPORAL,
            ),
        )

        actual = [
            value
            for frame in geometry.frames
            for value in (
                frame.center.center_x,
                frame.center.center_y,
                frame.center.error_x,
                frame.center.error_y,
            )
        ]
        expected = [
            value
            for frame in legacy.frames
            for value in (
                dict(frame.pattern_center)["center_x"],
                dict(frame.pattern_center)["center_y"],
                dict(frame.pattern_center)["error_x"],
                dict(frame.pattern_center)["error_y"],
            )
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(
            [item.radius for item in geometry.observations],
            [item.radius for item in legacy.observations],
            rtol=1e-12,
            atol=1e-9,
        )

    def test_invalid_geometry_request_fails_preflight(self) -> None:
        fit_set = _fit_set(((1, 10, 15.0, 20.0),))
        requests = (
            GeometryRequest(
                center_reference_track_ids=(1, 1),
            ),
            GeometryRequest(
                center_reference_track_ids=(1,),
                parameters=GeometryParameters(ransac_iterations=0),
            ),
            GeometryRequest(
                center_reference_track_ids=(1,),
                parameters=GeometryParameters(temporal_window=2),
            ),
            GeometryRequest(
                center_reference_track_ids=(1,),
                parameters=GeometryParameters(temporal_alpha=0.0),
            ),
        )

        for request in requests:
            with self.subTest(request=request):
                with self.assertRaises(GeometryPreflightError):
                    derive_geometry(fit_set, request)


if __name__ == "__main__":
    unittest.main()
