"""Blocking acceptance checks for the reference Track Analysis benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from trackleed.core.data_models import LeedFile, Track
from trackleed.io import results_exporter
from trackleed.logic.workflows.track_analysis import FitSet
from trackleed.logic.workflows.track_geometry import GeometryResult


@dataclass(frozen=True)
class PerformanceBudget:
    legacy_rerun_seconds: float
    geometry_seconds: float
    cold_seconds: float


@dataclass(frozen=True)
class AnalysisSnapshot:
    fit_set: FitSet
    geometry: GeometryResult
    tracks: list[Track]
    files: dict[int, LeedFile]


PERFORMANCE_BUDGETS = {
    4: PerformanceBudget(9.929, 2.0, 8.9),
    20: PerformanceBudget(26.708, 5.4, 37.1),
}
MAX_LEGACY_RERUN_RATIO = 0.20
REFERENCE_FRAME_COUNT = 120


def require_reference_frame_count(actual: int) -> None:
    """Reject a nominal 120-frame run that sampled fewer source frames."""
    if actual != REFERENCE_FRAME_COUNT:
        raise ValueError(
            f"acceptance benchmark requires {REFERENCE_FRAME_COUNT} actual frames; "
            f"loaded {actual}"
        )


def environment_metadata(
    native_thread_limit: int,
    load_tiff_cache_size: int,
) -> dict[str, Any]:
    """Describe the runtime that owns reference timing comparability."""
    metadata = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "pandas": version("pandas"),
        "scikit_image": version("scikit-image"),
        "tifffile": version("tifffile"),
        "load_tiff_cache_size": load_tiff_cache_size,
        "native_thread_limit": native_thread_limit,
    }
    try:
        metadata["native_threadpools"] = import_module(
            "threadpoolctl"
        ).threadpool_info()
    except ImportError:
        metadata["native_threadpools"] = "threadpoolctl unavailable"
    return metadata


def sha256_files(paths: Sequence[Path], root: Path | None = None) -> str:
    """Hash ordered file labels and contents, failing closed on missing inputs."""
    digest = sha256()
    resolved_root = None if root is None else root.resolve()
    resolved_paths = sorted({path.resolve() for path in paths}, key=str)
    if not resolved_paths:
        raise ValueError("At least one file is required for a reproducibility hash")
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot hash missing benchmark input: {path}")
        label = (
            path.relative_to(resolved_root).as_posix()
            if resolved_root is not None and path.is_relative_to(resolved_root)
            else path.name
        )
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def enforce_performance_budgets(
    summaries: Sequence[dict[str, Any]],
) -> None:
    """Fail when a reference stratum exceeds any PRD timing budget."""
    by_case = {(summary["case"], summary["run_kind"]): summary for summary in summaries}
    cases = {
        (summary["case"], int(summary["track_count"]))
        for summary in summaries
        if int(summary["track_count"]) in PERFORMANCE_BUDGETS
    }
    for case_name, track_count in cases:
        for run_kind in ("cold_full", "geometry_rerun"):
            if (case_name, run_kind) not in by_case:
                raise RuntimeError(
                    f"{case_name}: missing required {run_kind} benchmark summary"
                )
        budget = PERFORMANCE_BUDGETS[track_count]
        cold = float(by_case[(case_name, "cold_full")]["total"]["median_seconds"])
        geometry = float(
            by_case[(case_name, "geometry_rerun")]["total"]["median_seconds"]
        )
        geometry_limit = min(
            budget.geometry_seconds,
            MAX_LEGACY_RERUN_RATIO * budget.legacy_rerun_seconds,
        )
        if cold > budget.cold_seconds:
            raise RuntimeError(
                f"{case_name} cold_full median {cold:.3f}s exceeds "
                f"{budget.cold_seconds:.3f}s"
            )
        if geometry > geometry_limit:
            raise RuntimeError(
                f"{case_name} geometry_rerun median {geometry:.3f}s exceeds "
                f"{geometry_limit:.3f}s"
            )


def assert_reference_compatibility(
    actual: AnalysisSnapshot,
    reference: AnalysisSnapshot,
) -> None:
    """Require exact fits, tolerant geometry and byte-identical numeric CSVs."""
    fit_keys = tuple(
        (item.track_id, item.frame_id, item.fingerprint.sha256, item.status)
        for item in actual.fit_set.observations
    )
    reference_fit_keys = tuple(
        (item.track_id, item.frame_id, item.fingerprint.sha256, item.status)
        for item in reference.fit_set.observations
    )
    structure = (
        actual.geometry.status,
        actual.geometry.requested_method,
        actual.geometry.center_reference_track_ids,
        tuple((item.track_id, item.frame_id) for item in actual.geometry.observations),
        tuple(frame.frame_id for frame in actual.geometry.frames),
    )
    reference_structure = (
        reference.geometry.status,
        reference.geometry.requested_method,
        reference.geometry.center_reference_track_ids,
        tuple(
            (item.track_id, item.frame_id) for item in reference.geometry.observations
        ),
        tuple(frame.frame_id for frame in reference.geometry.frames),
    )
    if fit_keys != reference_fit_keys:
        raise RuntimeError("Geometry benchmark fit compatibility failed")
    if structure != reference_structure:
        raise RuntimeError(
            "Geometry benchmark structural compatibility failed: "
            f"{structure!r} != {reference_structure!r}"
        )
    actual_values: list[float] = []
    expected: list[float] = []
    for result, target in (
        (actual.geometry, actual_values),
        (reference.geometry, expected),
    ):
        for item in result.observations:
            target.extend(
                (
                    item.radius,
                    np.nan if item.radius_error is None else item.radius_error,
                )
            )
        for frame in result.frames:
            center = frame.center
            target.extend(
                (
                    np.nan if center is None else center.center_x,
                    np.nan if center is None else center.center_y,
                    np.nan if center is None else center.error_x,
                    np.nan if center is None else center.error_y,
                    np.nan if frame.average_radius is None else frame.average_radius,
                    np.nan
                    if frame.average_radius_error is None
                    else frame.average_radius_error,
                )
            )
    try:
        np.testing.assert_allclose(
            actual_values,
            expected,
            rtol=1e-12,
            atol=1e-9,
        )
    except AssertionError as cause:
        raise RuntimeError(
            "Geometry benchmark numerical compatibility failed"
        ) from cause
    if _numeric_csv_snapshot(actual.tracks, actual.files) != _numeric_csv_snapshot(
        reference.tracks, reference.files
    ):
        raise RuntimeError("Geometry benchmark CSV compatibility failed")


def _numeric_csv_snapshot(
    tracks: list[Track],
    files: dict[int, LeedFile],
) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="trackleed-analysis-csv-") as directory:
        results_exporter._save_full_data_csv(tracks, files, directory)
        results_exporter._save_center_data_csv(files, directory)
        results_exporter._save_average_radius_csv(files, directory)
        results_exporter._save_track_legend_to_csv(tracks, directory)
        return {path.name: path.read_bytes() for path in Path(directory).glob("*.csv")}
