from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import QRect

from trackleed.analysis.current_analysis_benchmark import (
    STAGE_NAMES,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    _select_evenly_spaced,
    _enforce_performance_budgets,
    _summarize_records,
    run_benchmark,
)
from trackleed.analysis.current_analysis_acceptance import (
    AnalysisSnapshot,
    assert_reference_compatibility,
    require_reference_frame_count,
)
from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.session import SessionContents, save_session_to_file
from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.workflows.track_analysis import (
    FitObservationInput,
    FitRequest,
    FitRunContext,
    FitSet,
    Rect,
    fit,
)
from trackleed.logic.workflows.track_geometry import GeometryRequest, derive_geometry


class CurrentAnalysisBenchmarkTests(unittest.TestCase):
    def test_reference_compatibility_rejects_sub_csv_precision_fit_change(self) -> None:
        yy, xx = np.mgrid[0:32, 0:32]
        pixels = np.exp(-((xx - 16.0) ** 2 + (yy - 16.0) ** 2) / 8.0)
        fit_set = fit(
            FitRequest(
                (
                    FitObservationInput(
                        1,
                        1,
                        Path("memory.png"),
                        Rect(0, 0, 32, 32),
                    ),
                )
            ),
            FitRunContext(image_reader=lambda _path: pixels),
        )
        observation = fit_set.observations[0]
        self.assertIsNotNone(observation.result)
        assert observation.result is not None
        changed_parameters = replace(
            observation.result.parameters,
            amplitude=observation.result.parameters.amplitude + 1.0e-10,
        )
        changed_observation = replace(
            observation,
            result=replace(observation.result, parameters=changed_parameters),
        )
        changed_fit_set = FitSet((changed_observation,))
        geometry = derive_geometry(fit_set, GeometryRequest())

        with self.assertRaisesRegex(RuntimeError, "exact fit"):
            assert_reference_compatibility(
                AnalysisSnapshot(changed_fit_set, geometry, [], {}),
                AnalysisSnapshot(fit_set, geometry, [], {}),
            )

    def test_diagnostic_benchmark_consumes_current_session_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.png"
            yy, xx = np.mgrid[0:32, 0:32]
            pixels = 255.0 * np.exp(
                -((xx - 16.0) ** 2 + (yy - 16.0) ** 2) / 8.0
            )
            Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(image_path)
            source = LeedFile(
                image_path,
                image_path.name,
                False,
                "X",
                1,
                "G",
                1.0,
                1.0,
                1,
                0.0,
            )
            spot = SpotROI(1, QRect(8, 8, 17, 17), 16.0, 16.0, 1.0, "manual")
            source.spots = [spot]
            session_path = root / "case.lts.json"
            save_session_to_file(
                session_path,
                SessionContents(
                    [source],
                    None,
                    [],
                    [Track(1, {1: spot})],
                    {},
                    None,
                    None,
                    ActiveAnalysisState(),
                ),
            )

            report = run_benchmark(
                BenchmarkConfig(
                    cases=(BenchmarkCase("one-track", session_path, 1),),
                    output_path=root / "report.json",
                    repeats=1,
                    frame_limit=1,
                    acceptance=False,
                )
            )

        self.assertEqual(report["acceptance_status"], "diagnostic_not_gated")
        self.assertEqual(report["cases"][0]["track_count"], 1)
        pools = report["environment"]["native_threadpools"]
        self.assertTrue(pools)
        self.assertTrue(
            all(int(pool["num_threads"]) == 1 for pool in pools),
            pools,
        )

    def test_invalid_cold_center_mode_fails_before_loading_a_case(self) -> None:
        config = BenchmarkConfig(
            cases=(BenchmarkCase("case", Path("missing-session.json"), 1),),
            output_path=Path("unused-report.json"),
            cold_center_mode="typo",
        )

        with self.assertRaisesRegex(ValueError, "cold_center_mode must be one of"):
            run_benchmark(config)

    def test_acceptance_rejects_noncanonical_measurement_before_loading(self) -> None:
        config = BenchmarkConfig(
            cases=(
                BenchmarkCase("4-track", Path("missing-four.json"), 4),
                BenchmarkCase("20-track", Path("missing-twenty.json"), 20),
            ),
            output_path=Path("unused-report.json"),
            repeats=1,
        )

        with self.assertRaisesRegex(ValueError, "acceptance benchmark requires"):
            run_benchmark(config)

    def test_acceptance_requires_120_actual_sampled_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "120 actual frames; loaded 10"):
            require_reference_frame_count(10)

    def test_even_sampling_keeps_endpoints_and_requested_count(self) -> None:
        selected = _select_evenly_spaced(list(range(100)), 5)

        self.assertEqual(selected, [0, 25, 50, 74, 99])

    def test_summary_reports_median_and_mad_for_each_stage(self) -> None:
        records = []
        for repeat, total in enumerate((10.0, 12.0, 50.0), start=1):
            stages = {name: 0.0 for name in STAGE_NAMES}
            stages["gaussian_fit"] = total - 2.0
            stages["image_io"] = 2.0
            records.append(
                BenchmarkRecord(
                    case="4-track",
                    run_kind="cold_full",
                    repeat=repeat,
                    center_mode="per_frame",
                    frame_count=10,
                    track_count=4,
                    observation_count=40,
                    valid_fit_count=38,
                    total_seconds=total,
                    stage_seconds=stages,
                    stage_calls={name: 0 for name in STAGE_NAMES},
                    image_cache_hits=0,
                    image_cache_misses=10,
                    numeric_output_bytes=100,
                )
            )

        summary = _summarize_records(records)[0]

        self.assertEqual(summary["total"]["median_seconds"], 12.0)
        self.assertEqual(summary["total"]["mad_seconds"], 2.0)
        self.assertAlmostEqual(
            summary["stages"]["gaussian_fit"]["median_share"],
            10.0 / 12.0,
        )

    def test_performance_budgets_are_blocking(self) -> None:
        summaries = [
            {
                "case": "4-track",
                "run_kind": "cold_full",
                "track_count": 4,
                "total": {"median_seconds": 8.91},
            },
            {
                "case": "4-track",
                "run_kind": "geometry_rerun",
                "track_count": 4,
                "total": {"median_seconds": 0.5},
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "cold_full.*8.900"):
            _enforce_performance_budgets(summaries)

        summaries[0]["total"]["median_seconds"] = 8.0
        summaries[1]["total"]["median_seconds"] = 1.99
        with self.assertRaisesRegex(RuntimeError, "geometry_rerun.*1.986"):
            _enforce_performance_budgets(summaries)


if __name__ == "__main__":
    unittest.main()
