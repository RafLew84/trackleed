"""Run the stage-level benchmark for the current TrackLEED analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trackleed.analysis.current_analysis_benchmark import (
    SUPPORTED_COLD_CENTER_MODES,
    BenchmarkCase,
    BenchmarkConfig,
    run_benchmark,
)

DEFAULT_SESSION_DIRECTORY = Path("data/26b32_ThExt_600CUD_122eV_Au04")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile current TrackLEED analysis for paired 4-track and 20-track "
            "saved sessions."
        )
    )
    parser.add_argument(
        "--four-track-session",
        type=Path,
        default=DEFAULT_SESSION_DIRECTORY / "26b32_ThExt_600CUD_122eV_Au04_4spots.lts.json",
    )
    parser.add_argument(
        "--twenty-track-session",
        type=Path,
        default=DEFAULT_SESSION_DIRECTORY / "26b32_ThExt_600CUD_122eV_Au04.lts.json",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=120,
        help="Deterministic evenly spaced frame count; use 0 for every frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".scratch/trackleed-current-workflow-hardening/artifacts/"
            "analysis-performance-baseline.json"
        ),
    )
    parser.add_argument(
        "--cold-center-mode",
        choices=sorted(SUPPORTED_COLD_CENTER_MODES),
        default="per_frame",
    )
    parser.add_argument(
        "--switched-center-mode",
        choices=("global_median",),
        default="global_median",
        help="Mode used by the current full rerun and geometry-only replay.",
    )
    parser.add_argument("--random-seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_limit = None if args.frame_limit == 0 else args.frame_limit
    config = BenchmarkConfig(
        cases=(
            BenchmarkCase("4-track", args.four_track_session, expected_track_count=4),
            BenchmarkCase("20-track", args.twenty_track_session, expected_track_count=20),
        ),
        output_path=args.output,
        repeats=args.repeats,
        frame_limit=frame_limit,
        cold_center_mode=args.cold_center_mode,
        switched_center_mode=args.switched_center_mode,
        random_seed=args.random_seed,
    )
    report = run_benchmark(config)
    print("\nMedian timings")
    for summary in report["summaries"]:
        print(
            f"  {summary['case']:>8} {summary['run_kind']:<18} "
            f"{summary['total']['median_seconds']:.3f} s"
        )
    print(f"\nJSON: {report['artifacts']['json']}")
    print(f"CSV:  {report['artifacts']['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
