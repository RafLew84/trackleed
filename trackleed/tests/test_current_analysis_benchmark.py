from __future__ import annotations

import unittest
from pathlib import Path

from trackleed.analysis.current_analysis_benchmark import (
    STAGE_NAMES,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    _select_evenly_spaced,
    _summarize_records,
    run_benchmark,
)


class CurrentAnalysisBenchmarkTests(unittest.TestCase):
    def test_invalid_cold_center_mode_fails_before_loading_a_case(self) -> None:
        config = BenchmarkConfig(
            cases=(BenchmarkCase("case", Path("missing-session.json"), 1),),
            output_path=Path("unused-report.json"),
            cold_center_mode="typo",
        )

        with self.assertRaisesRegex(ValueError, "cold_center_mode must be one of"):
            run_benchmark(config)

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


if __name__ == "__main__":
    unittest.main()
