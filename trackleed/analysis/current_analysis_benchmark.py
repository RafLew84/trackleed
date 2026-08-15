"""Stage-level benchmark for cold fitting and production geometry reruns."""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from unittest.mock import patch
# Pin native libraries so small least-squares repetitions remain comparable.
NATIVE_THREAD_LIMIT = int(os.environ.get("TRACKLEED_BENCHMARK_NATIVE_THREADS", "1"))
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = str(NATIVE_THREAD_LIMIT)

import numpy as np  # noqa: E402

from trackleed.core.data_models import LeedFile, Track  # noqa: E402
from trackleed.analysis.current_analysis_acceptance import (  # noqa: E402
    AnalysisSnapshot,
    assert_reference_compatibility,
    enforce_native_thread_limit,
    enforce_performance_budgets as _enforce_performance_budgets,
    environment_metadata,
    require_reference_frame_count,
    sha256_files,
)
from trackleed.io import image_reader, results_exporter  # noqa: E402
from trackleed.io.session import load_session_from_file  # noqa: E402
from trackleed.logic import track_analysis_compatibility  # noqa: E402
from trackleed.logic.workflows import gaussian_fit_recovery  # noqa: E402
from trackleed.logic.workflows.track_analysis import FitSet  # noqa: E402
from trackleed.logic.workflows.track_geometry import GeometryResult  # noqa: E402

STAGE_NAMES = (
    "image_io",
    "roi_preparation",
    "gaussian_fit",
    "center_estimation",
    "radius_calculation",
    "orchestration",
    "result_write",
)
SUPPORTED_COLD_CENTER_MODES = frozenset(
    {
        "per_frame",
        "circle_fit",
        "circle_fit_bbox_fallback",
        "ellipse_fit",
        "circle_fit_temporal",
        "global_median",
    }
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkCase:
    """One saved session used as a benchmark stratum."""

    name: str
    session_path: Path
    expected_track_count: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Small interface for a repeatable benchmark run."""

    cases: tuple[BenchmarkCase, ...]
    output_path: Path
    repeats: int = 3
    frame_limit: int | None = 120
    cold_center_mode: str = "per_frame"
    switched_center_mode: str = "global_median"
    random_seed: int = 20260811
    acceptance: bool = True


@dataclass
class BenchmarkRecord:
    case: str
    run_kind: str
    repeat: int
    center_mode: str
    frame_count: int
    track_count: int
    observation_count: int
    valid_fit_count: int
    total_seconds: float
    stage_seconds: dict[str, float]
    stage_calls: dict[str, int]
    image_cache_hits: int
    image_cache_misses: int
    numeric_output_bytes: int


@dataclass
class _PreparedCase:
    files: list[LeedFile]
    all_files_map: dict[int, LeedFile]
    background_file: LeedFile | None
    masks: list[Any]
    tracks: list[Track]
    spot_settings: dict[str, Any]
    analysis_circle: dict[str, Any] | None
    analysis_ellipse: dict[str, Any] | None


class _CallGapProfiler:
    """Attribute wall time around calls made by ``run_full_analysis``.

    Time inside wrapped functions belongs to the named stage.  The gap before a
    Gaussian fit is ROI preparation: track lookup, QRect access, image slicing,
    copying, float conversion and optional background subtraction.  Other gaps
    are orchestration.  This keeps the measurement on the current coordinator
    interface without adding a benchmark-only hook to production code.
    """

    def __init__(self) -> None:
        self.totals = {name: 0.0 for name in STAGE_NAMES}
        self.calls = {name: 0 for name in STAGE_NAMES}
        self._started_at: float | None = None
        self._last_event_end: float | None = None

    def start(self) -> None:
        now = perf_counter()
        self._started_at = now
        self._last_event_end = now

    def wrap(self, stage: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            self._attribute_gap(start, next_stage=stage)
            try:
                return function(*args, **kwargs)
            finally:
                end = perf_counter()
                self.totals[stage] += end - start
                self.calls[stage] += 1
                self._last_event_end = end

        return measured

    def finish(self) -> float:
        end = perf_counter()
        self._attribute_gap(end, next_stage=None)
        if self._started_at is None:
            raise RuntimeError("Profiler was not started")
        return end - self._started_at

    def _attribute_gap(self, now: float, next_stage: str | None) -> None:
        if self._last_event_end is None:
            self._last_event_end = now
            return
        gap = max(0.0, now - self._last_event_end)
        gap_stage = (
            "roi_preparation" if next_stage == "gaussian_fit" else "orchestration"
        )
        self.totals[gap_stage] += gap
        self._last_event_end = now


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run every configured stratum and write JSON plus a sibling CSV."""
    _validate_config(config)
    with enforce_native_thread_limit(NATIVE_THREAD_LIMIT):
        records: list[BenchmarkRecord] = []
        case_metadata: list[dict[str, Any]] = []
        for case in config.cases:
            metadata, case_records = _benchmark_case(config, case)
            case_metadata.append(metadata)
            records.extend(case_records)

        report = _build_report(config, case_metadata, records)
        if config.acceptance:
            _enforce_performance_budgets(report["summaries"])
            report["acceptance_status"] = "passed"
        else:
            report["acceptance_status"] = "diagnostic_not_gated"
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = config.output_path.with_suffix(".csv")
        _write_summary_csv(csv_path, report["summaries"])
        report["artifacts"] = {
            "json": str(config.output_path.resolve()),
            "csv": str(csv_path.resolve()),
        }
        config.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def _validate_config(config: BenchmarkConfig) -> None:
    if config.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if config.frame_limit is not None and config.frame_limit < 1:
        raise ValueError("frame_limit must be positive or None")
    if not config.cases:
        raise ValueError("at least one benchmark case is required")
    if config.cold_center_mode not in SUPPORTED_COLD_CENTER_MODES:
        supported = ", ".join(sorted(SUPPORTED_COLD_CENTER_MODES))
        raise ValueError(f"cold_center_mode must be one of: {supported}")
    if config.switched_center_mode != "global_median":
        raise ValueError(
            "geometry-only replay currently requires switched_center_mode='global_median'"
        )
    if config.acceptance:
        canonical = (
            config.repeats == 3
            and config.frame_limit == 120
            and config.random_seed == 20260811
            and NATIVE_THREAD_LIMIT == 1
            and sorted(case.expected_track_count for case in config.cases) == [4, 20]
        )
        if not canonical:
            raise ValueError(
                "acceptance benchmark requires 4/20 tracks, 120 frames, 3 repeats, "
                "seed 20260811 and one native thread; use diagnostic mode otherwise"
            )


def _benchmark_case(
    config: BenchmarkConfig,
    case: BenchmarkCase,
) -> tuple[dict[str, Any], list[BenchmarkRecord]]:
    initial = _load_case(case, config.frame_limit)
    frame_ids = sorted(initial.all_files_map)
    if config.acceptance:
        require_reference_frame_count(len(frame_ids))
    metadata = {
        "name": case.name,
        "session_path": str(case.session_path.resolve()),
        "source_frame_count": _source_frame_count(case.session_path),
        "sampled_frame_count": len(frame_ids),
        "sampled_frame_id_first": frame_ids[0] if frame_ids else None,
        "sampled_frame_id_last": frame_ids[-1] if frame_ids else None,
        "track_count": len(initial.tracks),
        "observation_count": _observation_count(initial.tracks),
        "input_sha256": sha256_files(
            (
                case.session_path,
                *(item.filepath for item in initial.all_files_map.values()),
                *(
                    (initial.background_file.filepath,)
                    if initial.background_file is not None
                    else ()
                ),
            )
        ),
        "sampling": (
            "all frames"
            if config.frame_limit is None
            else "deterministic evenly spaced frames"
        ),
    }
    records: list[BenchmarkRecord] = []
    for repeat_index in range(1, config.repeats + 1):
        records.extend(_benchmark_repetition(config, case, repeat_index))
    return metadata, records


def _benchmark_repetition(
    config: BenchmarkConfig,
    case: BenchmarkCase,
    repeat_index: int,
) -> tuple[BenchmarkRecord, BenchmarkRecord]:
    prepared = _load_case(case, config.frame_limit)
    label = f"[benchmark] {case.name} repeat {repeat_index}/{config.repeats}"
    print(f"{label}: application-cold full analysis")
    cold_record, fit_set, geometry = _run_full_pipeline(
        prepared,
        case_name=case.name,
        run_kind="cold_full",
        repeat=repeat_index,
        center_mode=config.cold_center_mode,
        clear_image_cache=True,
        random_seed=config.random_seed,
    )
    print(f"{label}: production geometry rerun from immutable fits")
    geometry_record, reused_fit_set, rerun_geometry = _run_full_pipeline(
        prepared,
        case_name=case.name,
        run_kind="geometry_rerun",
        repeat=repeat_index,
        center_mode=config.switched_center_mode,
        clear_image_cache=False,
        random_seed=config.random_seed,
        previous_fit_set=fit_set,
        previous_geometry_result=geometry,
    )
    if reused_fit_set is not fit_set:
        raise RuntimeError("Geometry benchmark did not preserve the exact FitSet")
    forbidden_calls = sum(
        geometry_record.stage_calls[name]
        for name in ("image_io", "roi_preparation", "gaussian_fit")
    )
    if forbidden_calls:
        raise RuntimeError(
            "Geometry benchmark performed image, ROI-preparation or Gaussian-fit work"
        )
    reference = _load_case(case, config.frame_limit)
    _record, reference_fit_set, reference_geometry = _run_full_pipeline(
        reference,
        case_name=case.name,
        run_kind="compatibility_reference",
        repeat=0,
        center_mode=config.switched_center_mode,
        clear_image_cache=True,
        random_seed=config.random_seed,
    )
    assert_reference_compatibility(
        AnalysisSnapshot(fit_set, rerun_geometry, prepared.tracks, prepared.all_files_map),
        AnalysisSnapshot(reference_fit_set, reference_geometry, reference.tracks, reference.all_files_map),
    )
    return cold_record, geometry_record


def _build_report(
    config: BenchmarkConfig,
    case_metadata: Sequence[dict[str, Any]],
    records: Sequence[BenchmarkRecord],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256_files(
            (
                *REPOSITORY_ROOT.joinpath("trackleed").rglob("*.py"),
                REPOSITORY_ROOT / "scripts" / "benchmark_current_analysis.py",
            ),
            REPOSITORY_ROOT,
        ),
        "environment": environment_metadata(
            NATIVE_THREAD_LIMIT,
            image_reader.CACHE_SIZE,
        ),
        "methodology": {
            "cold_definition": (
                "Freshly deserialized session state and cleared TrackLEED load_tiff LRU "
                "cache. The timed interval starts at run_full_analysis; session loading is "
                "excluded and the operating-system file cache is not flushed."
            ),
            "geometry_rerun": (
                "The production analysis adapter invokes derive_geometry with the immutable "
                "FitSet from the cold run; no image I/O, preprocessing or Gaussian fitting "
                "is permitted."
            ),
            "roi_preparation": (
                "Observed gaps immediately before Gaussian-fit calls, covering track/ROI "
                "lookup, image slicing and copy, float conversion and background subtraction."
            ),
            "result_write": (
                "Canonical numerical CSV outputs only: full per-spot data, pattern centers, "
                "average radii and track legend. Visualization PNG rendering is excluded."
            ),
            "cold_center_mode": config.cold_center_mode,
            "switched_center_mode": config.switched_center_mode,
            "repeats": config.repeats,
            "frame_limit": config.frame_limit,
            "random_seed": config.random_seed,
            "native_thread_limit": NATIVE_THREAD_LIMIT,
            "acceptance_mode": config.acceptance,
        },
        "cases": list(case_metadata),
        "records": [asdict(record) for record in records],
        "summaries": _summarize_records(records),
    }


def _run_full_pipeline(
    prepared: _PreparedCase,
    *,
    case_name: str,
    run_kind: str,
    repeat: int,
    center_mode: str,
    clear_image_cache: bool,
    random_seed: int,
    previous_fit_set: FitSet | None = None,
    previous_geometry_result: GeometryResult | None = None,
) -> tuple[BenchmarkRecord, FitSet, GeometryResult]:
    if clear_image_cache:
        image_reader.load_tiff.cache_clear()
    cache_before = image_reader.load_tiff.cache_info()
    profiler = _CallGapProfiler()
    _seed_random_generators(random_seed)
    with (
        patch.object(
            track_analysis_compatibility,
            "load_tiff",
            profiler.wrap("image_io", image_reader.load_tiff),
        ),
        patch.object(
            gaussian_fit_recovery,
            "fit_spot_to_gaussian",
            profiler.wrap(
                "gaussian_fit",
                gaussian_fit_recovery.fit_spot_to_gaussian,
            ),
        ),
        patch.object(
            track_analysis_compatibility,
            "derive_geometry",
            profiler.wrap(
                "center_estimation",
                track_analysis_compatibility.derive_geometry,
            ),
        ),
        open(os.devnull, "w", encoding="utf-8") as sink,
        redirect_stdout(sink),
    ):
        profiler.start()
        publication = _run_analysis_publication(
            prepared,
            center_mode,
            previous_fit_set,
            previous_geometry_result,
        )
        core_seconds = profiler.finish()
    if publication is None:
        raise RuntimeError(
            f"Analysis unexpectedly cancelled for {case_name}/{run_kind}"
        )
    track_analysis_compatibility.publish_analysis(
        publication,
        prepared.tracks,
        prepared.all_files_map,
    )
    cache_after = image_reader.load_tiff.cache_info()
    write_seconds, output_bytes = _time_numeric_result_write(prepared)
    profiler.totals["result_write"] = write_seconds
    profiler.calls["result_write"] = 1
    record = _benchmark_record(
        prepared,
        profiler,
        case_name=case_name,
        run_kind=run_kind,
        repeat=repeat,
        center_mode=center_mode,
        total_seconds=core_seconds + write_seconds,
        image_cache_hits=cache_after.hits - cache_before.hits,
        image_cache_misses=cache_after.misses - cache_before.misses,
        output_bytes=output_bytes,
    )
    return record, publication.fit_set, publication.geometry_result


def _run_analysis_publication(
    prepared: _PreparedCase,
    center_mode: str,
    previous_fit_set: FitSet | None,
    previous_geometry_result: GeometryResult | None,
) -> track_analysis_compatibility.AnalysisPublication | None:
    if previous_fit_set is None:
        return track_analysis_compatibility.run_full_analysis_publication(
            tracks_to_analyze=prepared.tracks,
            analysis_config={
                track.track_id: {"subtract_bg": False} for track in prepared.tracks
            },
            all_files_map=prepared.all_files_map,
            background_file=None,
            progress_callback=lambda _progress: None,
            center_mode=center_mode,
        )
    return track_analysis_compatibility.run_geometry_analysis_publication(
        tracks_to_analyze=prepared.tracks,
        all_files_map=prepared.all_files_map,
        fit_set=previous_fit_set,
        progress_callback=lambda _progress: None,
        center_mode=center_mode,
        previous_geometry_result=previous_geometry_result,
    )


def _benchmark_record(
    prepared: _PreparedCase,
    profiler: _CallGapProfiler,
    *,
    case_name: str,
    run_kind: str,
    repeat: int,
    center_mode: str,
    total_seconds: float,
    image_cache_hits: int,
    image_cache_misses: int,
    output_bytes: int,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        case=case_name,
        run_kind=run_kind,
        repeat=repeat,
        center_mode=center_mode,
        frame_count=len(prepared.all_files_map),
        track_count=len(prepared.tracks),
        observation_count=_observation_count(prepared.tracks),
        valid_fit_count=_valid_fit_count(prepared.tracks),
        total_seconds=total_seconds,
        stage_seconds={name: float(profiler.totals[name]) for name in STAGE_NAMES},
        stage_calls={name: int(profiler.calls[name]) for name in STAGE_NAMES},
        image_cache_hits=image_cache_hits,
        image_cache_misses=image_cache_misses,
        numeric_output_bytes=output_bytes,
    )


def _time_numeric_result_write(prepared: _PreparedCase) -> tuple[float, int]:
    with tempfile.TemporaryDirectory(
        prefix="trackleed-analysis-benchmark-"
    ) as temp_dir:
        started = perf_counter()
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            results_exporter._save_full_data_csv(
                prepared.tracks,
                prepared.all_files_map,
                temp_dir,
            )
            results_exporter._save_center_data_csv(prepared.all_files_map, temp_dir)
            results_exporter._save_average_radius_csv(prepared.all_files_map, temp_dir)
            results_exporter._save_track_legend_to_csv(prepared.tracks, temp_dir)
        elapsed = perf_counter() - started
        output_bytes = sum(
            path.stat().st_size for path in Path(temp_dir).rglob("*") if path.is_file()
        )
    return elapsed, int(output_bytes)


def _load_case(case: BenchmarkCase, frame_limit: int | None) -> _PreparedCase:
    contents = load_session_from_file(case.session_path)
    files = contents.files
    background = contents.background
    masks = contents.masks
    tracks = contents.tracks
    spot_settings = contents.spot_settings
    analysis_circle = contents.analysis_circle
    analysis_ellipse = contents.analysis_ellipse
    if len(tracks) != case.expected_track_count:
        raise ValueError(
            f"{case.name}: expected {case.expected_track_count} tracks, found {len(tracks)}"
        )
    _rebase_missing_image_paths(files, case.session_path.parent)
    if background is not None:
        _rebase_missing_image_paths([background], case.session_path.parent)

    all_files_map = {int(file.id): file for file in files if file.id is not None}
    all_frame_ids = sorted(
        frame_id
        for frame_id in {fid for track in tracks for fid in track.spots_by_image_id}
        if frame_id in all_files_map
    )
    selected_frame_ids = _select_evenly_spaced(all_frame_ids, frame_limit)
    selected_set = set(selected_frame_ids)
    files = [all_files_map[frame_id] for frame_id in selected_frame_ids]
    all_files_map = {int(file.id): file for file in files if file.id is not None}
    for track in tracks:
        track.spots_by_image_id = {
            frame_id: spot
            for frame_id, spot in track.spots_by_image_id.items()
            if frame_id in selected_set
        }
        if hasattr(track, "analysis_results"):
            track.analysis_results = {}

    missing = [str(file.filepath) for file in files if not file.filepath.exists()]
    if missing:
        example = missing[0]
        raise FileNotFoundError(
            f"{case.name}: {len(missing)} sampled image paths do not exist; first: {example}"
        )
    return _PreparedCase(
        files=files,
        all_files_map=all_files_map,
        background_file=background,
        masks=masks,
        tracks=tracks,
        spot_settings=spot_settings,
        analysis_circle=analysis_circle,
        analysis_ellipse=analysis_ellipse,
    )


def _rebase_missing_image_paths(
    files: Iterable[LeedFile], session_directory: Path
) -> None:
    for file in files:
        if file.filepath.exists():
            continue
        local_candidate = session_directory / file.filename
        if local_candidate.exists():
            file.filepath = local_candidate


def _select_evenly_spaced(frame_ids: Sequence[int], limit: int | None) -> list[int]:
    ordered = list(frame_ids)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = [
        round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)
    ]
    return [ordered[index] for index in indices]


def _seed_random_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _source_frame_count(session_path: Path) -> int:
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    return len(payload.get("data_files", []))


def _observation_count(tracks: Iterable[Track]) -> int:
    return sum(len(track.spots_by_image_id) for track in tracks)


def _valid_fit_count(tracks: Iterable[Track]) -> int:
    return sum(
        1
        for track in tracks
        for result in getattr(track, "analysis_results", {}).values()
        if result
    )


def _summarize_records(records: Sequence[BenchmarkRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault((record.case, record.run_kind), []).append(record)

    summaries = []
    for (case_name, run_kind), group in sorted(grouped.items()):
        total_values = [record.total_seconds for record in group]
        total_median = median(total_values)
        stage_summary = {}
        for stage in STAGE_NAMES:
            values = [record.stage_seconds[stage] for record in group]
            stage_median = median(values)
            stage_summary[stage] = {
                "median_seconds": float(stage_median),
                "min_seconds": float(min(values)),
                "max_seconds": float(max(values)),
                "mad_seconds": float(
                    median([abs(value - stage_median) for value in values])
                ),
                "median_share": float(stage_median / total_median)
                if total_median
                else 0.0,
            }
        summaries.append(
            {
                "case": case_name,
                "run_kind": run_kind,
                "repeats": len(group),
                "frame_count": group[0].frame_count,
                "track_count": group[0].track_count,
                "observation_count": group[0].observation_count,
                "valid_fit_count_median": int(
                    median([record.valid_fit_count for record in group])
                ),
                "total": {
                    "median_seconds": float(total_median),
                    "min_seconds": float(min(total_values)),
                    "max_seconds": float(max(total_values)),
                    "mad_seconds": float(
                        median([abs(value - total_median) for value in total_values])
                    ),
                },
                "stages": stage_summary,
                "image_cache_hits_median": int(
                    median([record.image_cache_hits for record in group])
                ),
                "image_cache_misses_median": int(
                    median([record.image_cache_misses for record in group])
                ),
                "numeric_output_bytes_median": int(
                    median([record.numeric_output_bytes for record in group])
                ),
            }
        )
    return summaries


def _write_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "run_kind",
        "repeats",
        "frame_count",
        "track_count",
        "observation_count",
        "valid_fit_count_median",
        "total_median_seconds",
        "total_min_seconds",
        "total_max_seconds",
        "total_mad_seconds",
        "stage",
        "stage_median_seconds",
        "stage_min_seconds",
        "stage_max_seconds",
        "stage_mad_seconds",
        "stage_median_share",
        "image_cache_hits_median",
        "image_cache_misses_median",
        "numeric_output_bytes_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for stage, stage_data in summary["stages"].items():
                writer.writerow(
                    {
                        "case": summary["case"],
                        "run_kind": summary["run_kind"],
                        "repeats": summary["repeats"],
                        "frame_count": summary["frame_count"],
                        "track_count": summary["track_count"],
                        "observation_count": summary["observation_count"],
                        "valid_fit_count_median": summary["valid_fit_count_median"],
                        "total_median_seconds": summary["total"]["median_seconds"],
                        "total_min_seconds": summary["total"]["min_seconds"],
                        "total_max_seconds": summary["total"]["max_seconds"],
                        "total_mad_seconds": summary["total"]["mad_seconds"],
                        "stage": stage,
                        "stage_median_seconds": stage_data["median_seconds"],
                        "stage_min_seconds": stage_data["min_seconds"],
                        "stage_max_seconds": stage_data["max_seconds"],
                        "stage_mad_seconds": stage_data["mad_seconds"],
                        "stage_median_share": stage_data["median_share"],
                        "image_cache_hits_median": summary["image_cache_hits_median"],
                        "image_cache_misses_median": summary[
                            "image_cache_misses_median"
                        ],
                        "numeric_output_bytes_median": summary[
                            "numeric_output_bytes_median"
                        ],
                    }
                )
