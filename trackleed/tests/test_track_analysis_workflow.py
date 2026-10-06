from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import numpy as np

from trackleed.analysis.legacy_track_geometry import (
    GeometryObservationRoi,
    LegacyCenterMode,
    LegacyGeometryRequest,
    ManualEllipse,
    derive_legacy_geometry,
)
from trackleed.logic.track_analysis_compatibility import project_fit_set
from trackleed.logic.workflows.track_analysis import (
    FitCancelled,
    FitObservationInput,
    FitPreflightError,
    FitRecoveryConfig,
    FitRequest,
    FitRunContext,
    FitStatus,
    FitValidation,
    GaussianFitterConfig,
    PreprocessingConfig,
    Rect,
    RecoveryAttemptStatus,
    RecoverySelectionReason,
    RecoveryStatus,
    fit,
)


_TEST_IMAGES: dict[Path, np.ndarray] = {}


def _gaussian_frame(
    *,
    center_x: float = 15.0,
    center_y: float = 14.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    yy, xx = np.indices((31, 31), dtype=np.float64)
    return 0.05 + amplitude * np.exp(
        -(
            ((xx - center_x) ** 2) / (2.0 * 3.0**2)
            + ((yy - center_y) ** 2) / (2.0 * 4.0**2)
        )
    )


def _isotropic_gaussian_frame(
    center_x: float,
    center_y: float,
    sigma: float = 1.0,
) -> np.ndarray:
    yy, xx = np.indices((31, 31), dtype=np.float64)
    return 0.05 + 0.8 * np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
    )


def _two_spot_frame() -> np.ndarray:
    yy, xx = np.indices((31, 31), dtype=np.float64)
    first = np.exp(-((xx - 8.2) ** 2 + (yy - 13.0) ** 2) / 8.0)
    second = np.exp(-((xx - 15.0) ** 2 + (yy - 13.0) ** 2) / 8.0)
    return 0.02 + first + second


def _observation(
    track_id: int = 1,
    frame_id: int = 10,
    *,
    pixels: np.ndarray | None = None,
    roi: Rect = Rect(4, 3, 23, 23),
    subtract_background: bool = False,
    background: np.ndarray | None = None,
    source_path: Path = Path("first-location/frame.tif"),
    background_path: Path = Path("first-location/background.tif"),
    energy: float = 80.0,
    temperature: float = 300.0,
) -> FitObservationInput:
    _TEST_IMAGES[source_path] = _gaussian_frame() if pixels is None else pixels
    if background is not None:
        _TEST_IMAGES[background_path] = background
    return FitObservationInput(
        track_id=track_id,
        frame_id=frame_id,
        source_path=source_path,
        roi=roi,
        subtract_background=subtract_background,
        background_path=background_path if background is not None else None,
        energy=energy,
        temperature=temperature,
    )


def _run(
    request: FitRequest,
    *,
    report_progress=lambda _progress: None,
    is_cancelled=lambda: False,
):
    context = FitRunContext(
        image_reader=lambda path: _TEST_IMAGES[path],
        report_progress=report_progress,
        is_cancelled=is_cancelled,
    )
    return fit(request, context)


class TrackAnalysisWorkflowTests(unittest.TestCase):
    def test_request_is_a_value_snapshot_and_runtime_dependencies_are_separate(
        self,
    ) -> None:
        observation_fields = set(FitObservationInput.__dataclass_fields__)
        request_fields = set(FitRequest.__dataclass_fields__)

        self.assertNotIn("frame_pixels", observation_fields)
        self.assertNotIn("background_pixels", observation_fields)
        self.assertNotIn("report_progress", request_fields)
        self.assertNotIn("is_cancelled", request_fields)

    def test_fit_returns_immutable_geometry_free_measurement(self) -> None:
        fit_set = _run(FitRequest(observations=(_observation(),)))
        observation = fit_set.observation(1, 10)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.status, FitStatus.FITTED)
        self.assertIsNone(observation.failure)
        self.assertIsNotNone(observation.result)
        assert observation.result is not None
        self.assertAlmostEqual(observation.result.global_center_x, 15.0, places=5)
        self.assertAlmostEqual(observation.result.global_center_y, 14.0, places=5)
        self.assertGreater(observation.result.intensity.volume, 0.0)
        self.assertGreater(observation.result.widths.fwhm_major, 0.0)
        self.assertGreater(observation.result.quality.r_squared, 0.99)
        for geometry_field in (
            "pattern_center",
            "radius",
            "radius_error",
            "avg_radius",
        ):
            self.assertFalse(hasattr(observation.result, geometry_field))
        with self.assertRaises(FrozenInstanceError):
            observation.result.global_center_x = 0.0  # type: ignore[misc]

    def test_fingerprint_is_content_based_and_ignores_source_metadata(self) -> None:
        pixels = _gaussian_frame()
        first = _run(FitRequest(observations=(_observation(pixels=pixels),)))
        relocated = _run(
            FitRequest(
                observations=(
                    _observation(
                        pixels=pixels.copy(),
                        source_path=Path("other/location/renamed.tif"),
                        energy=250.0,
                        temperature=950.0,
                    ),
                ),
                previous=first,
            )
        )

        self.assertEqual(
            first.observation(1, 10).fingerprint,
            relocated.observation(1, 10).fingerprint,
        )
        self.assertEqual(len(first.observation(1, 10).fingerprint.sha256), 64)
        self.assertIs(
            relocated.observation(1, 10).result,
            first.observation(1, 10).result,
        )
        self.assertEqual(relocated.observation(1, 10).provenance.energy, 250.0)
        self.assertEqual(relocated.observation(1, 10).provenance.temperature, 950.0)

    def test_fingerprint_selectively_tracks_pixels_roi_and_used_background(
        self,
    ) -> None:
        background = np.full((31, 31), 0.01, dtype=np.float64)
        inputs = (
            _observation(track_id=1, subtract_background=False, background=background),
            _observation(track_id=2, subtract_background=True, background=background),
        )
        baseline = _run(FitRequest(observations=inputs))
        changed_background = background.copy()
        changed_background[0, 0] = 0.02
        rerun = _run(
            FitRequest(
                observations=(
                    _observation(
                        track_id=1,
                        subtract_background=False,
                        background=changed_background,
                    ),
                    _observation(
                        track_id=2,
                        subtract_background=True,
                        background=changed_background,
                    ),
                ),
                previous=baseline,
            )
        )

        self.assertIs(rerun.observation(1, 10), baseline.observation(1, 10))
        self.assertNotEqual(
            rerun.observation(2, 10).fingerprint,
            baseline.observation(2, 10).fingerprint,
        )

        changed_pixels = _gaussian_frame()
        changed_pixels[0, 0] += 0.001
        pixel_rerun = _run(
            FitRequest(
                observations=(_observation(pixels=changed_pixels),),
                previous=baseline,
            )
        )
        roi_rerun = _run(
            FitRequest(
                observations=(_observation(roi=Rect(3, 3, 23, 23)),),
                previous=baseline,
            )
        )
        self.assertNotEqual(
            pixel_rerun.observation(1, 10).fingerprint,
            baseline.observation(1, 10).fingerprint,
        )
        self.assertNotEqual(
            roi_rerun.observation(1, 10).fingerprint,
            baseline.observation(1, 10).fingerprint,
        )

    def test_configuration_changes_invalidate_matching_fingerprints(self) -> None:
        baseline = _run(FitRequest(observations=(_observation(),)))
        variants = (
            FitRequest(
                observations=(_observation(),),
                preprocessing=PreprocessingConfig(clamp_negative=False),
                previous=baseline,
            ),
            FitRequest(
                observations=(_observation(),),
                fitter=GaussianFitterConfig(center_preference_factor=0.25),
                previous=baseline,
            ),
            FitRequest(
                observations=(_observation(),),
                validation=FitValidation(min_r_squared=0.90),
                previous=baseline,
            ),
            FitRequest(
                observations=(_observation(),),
                recovery=FitRecoveryConfig(policy_version="bounded-v2"),
                previous=baseline,
            ),
        )

        baseline_fingerprint = baseline.observation(1, 10).fingerprint
        for request in variants:
            with self.subTest(request=request):
                self.assertNotEqual(
                    _run(request).observation(1, 10).fingerprint,
                    baseline_fingerprint,
                )

    def test_unknown_preprocessing_and_fitter_fail_closed(self) -> None:
        invalid_requests = (
            FitRequest(
                observations=(_observation(),),
                preprocessing=PreprocessingConfig(grayscale_mode="unknown"),
            ),
            FitRequest(
                observations=(_observation(),),
                fitter=GaussianFitterConfig(fitter_id="unknown"),
            ),
        )

        for request in invalid_requests:
            with self.subTest(request=request), self.assertRaises(FitPreflightError):
                _run(request)

    def test_cache_hit_returns_the_exact_stored_observation(self) -> None:
        baseline = _run(FitRequest(observations=(_observation(),)))
        cached = _run(FitRequest(observations=(_observation(),), previous=baseline))

        self.assertIs(cached.observation(1, 10), baseline.observation(1, 10))
        self.assertEqual(cached, baseline)

    def test_controlled_failure_is_cached_and_distinct_from_missing_spot(self) -> None:
        flat_pixels = np.ones((31, 31), dtype=np.float64)
        request = FitRequest(observations=(_observation(pixels=flat_pixels),))
        failed = _run(request)
        cached = _run(FitRequest(observations=request.observations, previous=failed))

        observation = failed.observation(1, 10)
        self.assertEqual(observation.status, FitStatus.FAILED)
        self.assertIsNotNone(observation.failure)
        self.assertIsNone(observation.result)
        self.assertIsNone(failed.observation(99, 10))
        self.assertIs(cached.observation(1, 10), observation)

    def test_recovery_is_opt_in_and_disabled_run_has_only_baseline_attempt(
        self,
    ) -> None:
        fit_set = _run(
            FitRequest(
                observations=(_observation(pixels=np.ones((31, 31), dtype=np.float64)),)
            )
        )

        observation = fit_set.observation(1, 10)
        self.assertEqual(
            observation.provenance.recovery.status, RecoveryStatus.DISABLED
        )
        self.assertEqual(len(observation.provenance.recovery.attempts), 1)
        self.assertEqual(
            observation.provenance.recovery.attempts[0].effective_roi,
            Rect(4, 3, 23, 23),
        )

    def test_disabled_recovery_preserves_the_baseline_validation_margin(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(
                        pixels=_isotropic_gaussian_frame(17.5, 12.5),
                        roi=Rect(10, 10, 5, 5),
                    ),
                ),
                validation=FitValidation(center_margin=5.0),
            )
        )

        observation = fit_set.observation(1, 10)
        self.assertEqual(observation.status, FitStatus.FITTED)
        self.assertEqual(
            observation.provenance.recovery.status, RecoveryStatus.DISABLED
        )
        self.assertEqual(len(observation.provenance.recovery.attempts), 1)

    def test_enabled_recovery_stops_after_successful_baseline_fit(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(_observation(),),
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        observation = fit_set.observation(1, 10)
        self.assertEqual(observation.status, FitStatus.FITTED)
        self.assertEqual(
            observation.provenance.recovery.status,
            RecoveryStatus.NOT_NEEDED,
        )
        self.assertEqual(len(observation.provenance.recovery.attempts), 1)

    def test_opt_in_recovery_recovers_fit_without_changing_original_roi(self) -> None:
        original_roi = Rect(10, 10, 5, 5)
        fit_input = _observation(
            track_id=7,
            pixels=_isotropic_gaussian_frame(8.25, 8.25),
            roi=original_roi,
        )
        input_identity = (fit_input.track_id, fit_input.roi)
        # Require the fitted center inside the attempted ROI so baseline
        # rejection follows from geometry, not optimizer convergence.
        validation = FitValidation(center_margin=0.0)
        baseline = _run(
            FitRequest(observations=(fit_input,), validation=validation)
        ).observation(7, 10)
        self.assertEqual(baseline.status, FitStatus.FAILED)
        self.assertEqual(baseline.failure.code, "validation_failed")
        baseline_recovery = baseline.provenance.recovery
        self.assertEqual(baseline_recovery.status, RecoveryStatus.DISABLED)
        self.assertEqual(len(baseline_recovery.attempts), 1)
        self.assertEqual(
            baseline_recovery.attempts[0].rejection_reasons,
            ("center_x", "center_y"),
        )

        fit_set = _run(
            FitRequest(
                observations=(fit_input,),
                validation=validation,
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        observation = fit_set.observation(7, 10)
        self.assertEqual(observation.status, FitStatus.FITTED)
        self.assertEqual((fit_input.track_id, fit_input.roi), input_identity)
        self.assertIs(fit_input.roi, original_roi)
        self.assertEqual((observation.track_id, observation.frame_id), (7, 10))
        self.assertEqual(observation.provenance.original_roi, original_roi)
        recovery = observation.provenance.recovery
        self.assertEqual(recovery.status, RecoveryStatus.RECOVERED)
        self.assertEqual(recovery.original_roi, original_roi)
        self.assertEqual(recovery.attempts[0], baseline_recovery.attempts[0])
        self.assertGreater(len(recovery.attempts), 1)
        self.assertLessEqual(len(recovery.attempts), 11)
        self.assertNotEqual(recovery.effective_roi, original_roi)
        selected = next(
            attempt
            for attempt in recovery.attempts
            if attempt.effective_roi == recovery.effective_roi
        )
        self.assertEqual(selected.status, RecoveryAttemptStatus.ACCEPTED)
        self.assertEqual(
            recovery.selection_reason,
            RecoverySelectionReason.SMALLEST_TRANSFORM_HIGHEST_QUALITY,
        )
        self.assertAlmostEqual(observation.result.global_center_x, 8.25, places=3)
        self.assertAlmostEqual(observation.result.global_center_y, 8.25, places=3)

    def test_disagreeing_recovery_candidates_fail_as_ambiguous(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(
                        pixels=_two_spot_frame(),
                        roi=Rect(10, 10, 7, 7),
                    ),
                ),
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        observation = fit_set.observation(1, 10)
        self.assertEqual(observation.status, FitStatus.FAILED)
        self.assertEqual(observation.failure.code, "ambiguous_recovery")
        self.assertEqual(
            observation.provenance.recovery.status,
            RecoveryStatus.AMBIGUOUS,
        )
        self.assertIsNone(observation.provenance.recovery.effective_roi)

    def test_failed_recovery_has_bounded_deterministic_attempt_history(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(
                        pixels=np.ones((31, 31), dtype=np.float64),
                    ),
                ),
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        observation = fit_set.observation(1, 10)
        attempts = observation.provenance.recovery.attempts
        self.assertEqual(observation.failure.code, "recovery_failed")
        self.assertEqual(observation.provenance.recovery.status, RecoveryStatus.FAILED)
        self.assertEqual(len(attempts), 11)
        self.assertEqual(
            tuple((item.offset_x, item.offset_y, item.expand_by) for item in attempts),
            (
                (0, 0, 0),
                (-1, -1, 0),
                (0, -1, 0),
                (1, -1, 0),
                (-1, 0, 0),
                (1, 0, 0),
                (-1, 1, 0),
                (0, 1, 0),
                (1, 1, 0),
                (0, 0, 1),
                (0, 0, 2),
            ),
        )

    def test_recovery_rejects_positions_outside_original_roi_margin(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(
                        pixels=_isotropic_gaussian_frame(7.5, 12.5),
                        roi=Rect(10, 10, 5, 5),
                    ),
                ),
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        observation = fit_set.observation(1, 10)
        statuses = {item.status for item in observation.provenance.recovery.attempts}
        self.assertEqual(observation.status, FitStatus.FAILED)
        self.assertIn(RecoveryAttemptStatus.OUTSIDE_ORIGINAL_MARGIN, statuses)
        self.assertNotIn(RecoveryAttemptStatus.ACCEPTED, statuses)

    def test_recovery_tie_break_uses_highest_quality_small_transform(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(
                        pixels=_isotropic_gaussian_frame(17.5, 12.5),
                        roi=Rect(10, 10, 5, 5),
                    ),
                ),
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        recovery = fit_set.observation(1, 10).provenance.recovery
        self.assertEqual(recovery.status, RecoveryStatus.RECOVERED)
        self.assertEqual(recovery.effective_roi, Rect(10, 9, 5, 5))
        self.assertGreater(
            recovery.attempts[2].r_squared,
            recovery.attempts[3].r_squared,
        )
        self.assertEqual(
            recovery.selection_reason,
            RecoverySelectionReason.SMALLEST_TRANSFORM_HIGHEST_QUALITY,
        )

    def test_recovery_prefers_smaller_transform_over_higher_quality(self) -> None:
        pixels = _isotropic_gaussian_frame(8.25, 8.25)
        yy, xx = np.indices(pixels.shape)
        pixels = pixels + 0.002 * np.sin(xx * 0.7 + yy * 0.2)
        fit_set = _run(
            FitRequest(
                observations=(_observation(pixels=pixels, roi=Rect(10, 10, 5, 5)),),
                recovery=FitRecoveryConfig(enabled=True),
            )
        )

        recovery = fit_set.observation(1, 10).provenance.recovery
        selected = next(
            item
            for item in recovery.attempts
            if item.effective_roi == recovery.effective_roi
        )
        larger = next(item for item in recovery.attempts if item.expand_by == 2)
        self.assertEqual(recovery.status, RecoveryStatus.RECOVERED)
        self.assertEqual(selected.expand_by, 1)
        self.assertEqual(larger.status, RecoveryAttemptStatus.ACCEPTED)
        self.assertGreater(larger.r_squared, selected.r_squared)

    def test_cancellation_after_any_recovery_attempt_exposes_no_partial_fit_set(
        self,
    ) -> None:
        previous = _run(FitRequest(observations=(_observation(),)))
        previous_snapshot = previous
        fit_input = _observation(pixels=np.ones((31, 31), dtype=np.float64))
        input_identity = (fit_input.track_id, fit_input.roi)

        for attempt_index in range(11):
            checks = 0
            cancel_at = 3 + 2 * attempt_index

            def cancel_after_attempt() -> bool:
                nonlocal checks
                checks += 1
                return checks == cancel_at

            with self.subTest(attempt_index=attempt_index):
                with self.assertRaises(FitCancelled):
                    _run(
                        FitRequest(
                            observations=(fit_input,),
                            recovery=FitRecoveryConfig(enabled=True),
                            previous=previous,
                        ),
                        is_cancelled=cancel_after_attempt,
                    )
                self.assertIs(previous, previous_snapshot)
                self.assertEqual(len(previous.observations), 1)
                self.assertEqual(previous.observation(1, 10).status, FitStatus.FITTED)
                self.assertEqual((fit_input.track_id, fit_input.roi), input_identity)

    def test_recovery_policy_version_caches_and_invalidates_controlled_failure(
        self,
    ) -> None:
        fit_input = _observation(pixels=np.ones((31, 31), dtype=np.float64))
        first = _run(
            FitRequest(
                observations=(fit_input,),
                recovery=FitRecoveryConfig(enabled=True, policy_version="bounded-v1"),
            )
        )
        cached = _run(
            FitRequest(
                observations=(fit_input,),
                recovery=FitRecoveryConfig(enabled=True, policy_version="bounded-v1"),
                previous=first,
            )
        )
        invalidated = _run(
            FitRequest(
                observations=(fit_input,),
                recovery=FitRecoveryConfig(enabled=True, policy_version="bounded-v2"),
                previous=first,
            )
        )

        self.assertIs(cached.observation(1, 10), first.observation(1, 10))
        self.assertNotEqual(
            invalidated.observation(1, 10).fingerprint,
            first.observation(1, 10).fingerprint,
        )
        self.assertEqual(invalidated.observation(1, 10).status, FitStatus.FAILED)

    def test_track_set_change_reuses_only_identical_fingerprints(self) -> None:
        baseline = _run(
            FitRequest(
                observations=(
                    _observation(track_id=1),
                    _observation(
                        track_id=2,
                        pixels=_gaussian_frame(center_x=16.0),
                        source_path=Path("first-location/second-frame.tif"),
                    ),
                )
            )
        )
        changed = _gaussian_frame(center_x=17.0)
        rerun = _run(
            FitRequest(
                observations=(
                    _observation(track_id=1),
                    _observation(
                        track_id=3,
                        pixels=changed,
                        source_path=Path("first-location/third-frame.tif"),
                    ),
                ),
                previous=baseline,
            )
        )

        self.assertEqual(tuple(item.track_id for item in rerun.observations), (1, 3))
        self.assertIs(rerun.observation(1, 10), baseline.observation(1, 10))
        self.assertNotIn(2, {item.track_id for item in rerun.observations})

    def test_cancellation_never_returns_a_partial_fit_set(self) -> None:
        baseline = _run(FitRequest(observations=(_observation(),)))
        completed = 0

        def report_progress(_progress) -> None:
            nonlocal completed
            completed += 1

        with self.assertRaises(FitCancelled):
            _run(
                FitRequest(
                    observations=(
                        _observation(track_id=1),
                        _observation(track_id=2),
                    ),
                    previous=baseline,
                ),
                report_progress=report_progress,
                is_cancelled=lambda: completed == 1,
            )

        self.assertEqual(len(baseline.observations), 1)

    def test_compatibility_projection_preserves_legacy_fit_fields(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(track_id=1),
                    _observation(
                        track_id=2,
                        pixels=np.ones((31, 31), dtype=np.float64),
                        source_path=Path("first-location/flat-frame.tif"),
                    ),
                )
            )
        )
        projection = project_fit_set(fit_set)

        fitted = projection[1][10]
        self.assertIsNotNone(fitted)
        self.assertAlmostEqual(fitted["global_center_x"], 15.0, places=5)
        self.assertAlmostEqual(fitted["global_center_y"], 14.0, places=5)
        self.assertIn("covariance_matrix", fitted)
        self.assertIn("fwhm_effective", fitted)
        self.assertNotIn("radius", fitted)
        self.assertIsNone(projection[2][10])

    def test_legacy_geometry_consumes_fit_set_without_mutating_it(self) -> None:
        fit_set = _run(
            FitRequest(
                observations=(
                    _observation(track_id=1),
                    _observation(
                        track_id=2,
                        pixels=_gaussian_frame(center_x=20.0),
                        source_path=Path("first-location/geometry-frame.tif"),
                    ),
                )
            )
        )
        request = LegacyGeometryRequest(
            observation_rois=(
                GeometryObservationRoi(1, 10, Rect(4, 3, 23, 23)),
                GeometryObservationRoi(2, 10, Rect(4, 3, 23, 23)),
            ),
            center_mode=LegacyCenterMode.MANUAL_ELLIPSE,
            manual_ellipse=ManualEllipse(15.0, 14.0, 10.0, 5.0),
        )

        geometry = derive_legacy_geometry(fit_set, request)

        self.assertEqual(len(geometry.observations), 2)
        self.assertAlmostEqual(geometry.observations[0].radius, 0.0, places=5)
        self.assertGreater(geometry.observations[1].radius, 0.0)
        self.assertFalse(hasattr(fit_set.observations[0].result, "radius"))
        self.assertEqual(
            dict(geometry.frames[0].pattern_center),
            {
                "center_x": 15.0,
                "center_y": 14.0,
                "error_x": 0.0,
                "error_y": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
