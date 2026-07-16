"""Create non-destructive publication copies of stored analysis exports."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

from trackleed.io.publication_fwhm import (
    FwhmMetrics,
    FwhmSource,
    fit_key,
    is_true,
    iter_stored_fit_records,
    provenance_counts,
)


FWHM_EXPORT_COLUMNS = (
    "fwhm_x",
    "fwhm_x_error",
    "fwhm_y",
    "fwhm_y_error",
    "fwhm_mean",
    "fwhm_geometric",
    "fwhm_source",
)
EXPORT_DIRECTORY_NAMES = {"eksport", "export", "result", "results", "wew", "zew"}
EXPORT_SIDECAR_DIRECTORIES = {"positions_overlay_images", "track_visualizations"}


@dataclass(frozen=True)
class PublicationExportResult:
    """Summary returned after creating or reusing a publication export."""

    destination_root: Path
    reused: bool


@dataclass(frozen=True)
class StoredFitLookup:
    source_path: Path
    metrics_by_key: dict[tuple[int, int], FwhmMetrics]


@dataclass(frozen=True)
class CsvEnrichmentSummary:
    row_count: int
    provenance: dict[str, int]
    missing_fwhm_count: int
    per_frame_path: Path


@dataclass(frozen=True)
class FwhmRecordSummary:
    row_count: int
    provenance: dict[str, int]
    missing_fwhm_count: int


def create_publication_export(
    source_root: Path | str,
    destination_root: Path | str,
) -> PublicationExportResult:
    """Copy discovered exports and enrich copied per-spot CSV files with FWHM."""

    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source root does not exist or is not a directory: {source}")
    if _is_relative_to(destination, source):
        raise ValueError(
            "Destination must be outside the source tree to guarantee a non-destructive export: "
            f"{destination}"
        )

    per_spot_csv_paths = sorted(
        path for path in source.rglob("*.csv") if _is_per_spot_csv(path)
    )
    export_directories = sorted(
        {
            path
            for path in source.rglob("*")
            if path.is_dir() and _is_named_export_directory(path)
        }
    )
    standalone_export_directories = sorted(
        {
            csv_path.parent
            for csv_path in per_spot_csv_paths
            if not any(
                _is_relative_to(csv_path.parent, export_directory)
                for export_directory in export_directories
            )
        }
    )
    sidecar_directories = sorted(
        {
            path
            for path in source.rglob("*")
            if path.is_dir() and path.name.lower() in EXPORT_SIDECAR_DIRECTORIES
        }
    )
    session_paths = sorted(
        path
        for path in source.rglob("*.json")
        if _is_saved_analysis_path(path)
    )
    fit_lookups = [
        lookup
        for session_path in session_paths
        if (lookup := _load_stored_fit_lookup(session_path)) is not None
    ]
    source_files = _source_artifact_files(
        export_directories=export_directories,
        standalone_export_directories=standalone_export_directories,
        sidecar_directories=sidecar_directories,
        per_spot_csv_paths=per_spot_csv_paths,
        session_paths=session_paths,
    )
    input_fingerprint = _inventory_fingerprint(source, source_files)
    if destination.exists():
        return _reuse_existing_export(
            source=source,
            destination=destination,
            input_fingerprint=input_fingerprint,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )

    try:
        csv_summaries: dict[Path, CsvEnrichmentSummary] = {}
        session_summaries: dict[Path, FwhmRecordSummary] = {}
        for export_directory in export_directories:
            relative_directory = export_directory.relative_to(source)
            copied_directory = staging / relative_directory
            shutil.copytree(export_directory, copied_directory, dirs_exist_ok=True)
        for standalone_export_directory in standalone_export_directories:
            relative_directory = standalone_export_directory.relative_to(source)
            copied_directory = staging / relative_directory
            _copy_non_raw_export_tree(standalone_export_directory, copied_directory)
        for sidecar_directory in sidecar_directories:
            relative_directory = sidecar_directory.relative_to(source)
            copied_directory = staging / relative_directory
            shutil.copytree(sidecar_directory, copied_directory, dirs_exist_ok=True)
        for session_path in session_paths:
            copied_session_path = staging / session_path.relative_to(source)
            copied_session_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(session_path, copied_session_path)
            session_summaries[session_path] = _enrich_saved_analysis_json(
                copied_session_path
            )
        for csv_path in per_spot_csv_paths:
            copied_csv_path = staging / csv_path.relative_to(source)
            copied_csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not copied_csv_path.exists():
                shutil.copy2(csv_path, copied_csv_path)
            lookup = _select_fit_lookup(csv_path, fit_lookups)
            csv_summaries[csv_path] = _enrich_per_spot_csv(
                copied_csv_path,
                metrics_by_key=lookup.metrics_by_key if lookup else {},
            )
        manifest = _build_manifest(
            source=source,
            destination=destination,
            staging=staging,
            source_files=source_files,
            input_fingerprint=input_fingerprint,
            csv_summaries=csv_summaries,
            session_summaries=session_summaries,
        )
        (staging / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PublicationExportResult(destination_root=destination, reused=False)


def _reuse_existing_export(
    *,
    source: Path,
    destination: Path,
    input_fingerprint: str,
) -> PublicationExportResult:
    manifest_path = destination / "export_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        manifest = None
    if (
        isinstance(manifest, dict)
        and manifest.get("version") == 1
        and manifest.get("source_root") == str(source)
        and manifest.get("input_fingerprint") == input_fingerprint
        and _manifest_outputs_match(destination, manifest)
    ):
        return PublicationExportResult(destination_root=destination, reused=True)
    raise FileExistsError(
        f"Destination already exists and does not match the current inputs: {destination}"
    )


def _manifest_outputs_match(destination: Path, manifest: dict) -> bool:
    files = manifest.get("files")
    generated_hashes = manifest.get("generated_file_hashes")
    if not isinstance(files, list) or not isinstance(generated_hashes, dict):
        return False
    expected_hashes: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict):
            return False
        relative_path = entry.get("relative_output_path")
        output_hash = entry.get("output_sha256")
        if not isinstance(relative_path, str) or not isinstance(output_hash, str):
            return False
        expected_hashes[relative_path] = output_hash
    for relative_path, output_hash in generated_hashes.items():
        if not isinstance(relative_path, str) or not isinstance(output_hash, str):
            return False
        expected_hashes[relative_path] = output_hash
    return all(
        (path := destination / relative_path).is_file() and _sha256(path) == output_hash
        for relative_path, output_hash in expected_hashes.items()
    )


def _is_per_spot_csv(path: Path) -> bool:
    return path.name == "full_analysis_per_spot.csv" or (
        path.name.startswith("positions_analysis_results") and path.suffix.lower() == ".csv"
    )


def _is_named_export_directory(path: Path) -> bool:
    normalized_name = path.name.strip().lower()
    return normalized_name in EXPORT_DIRECTORY_NAMES or normalized_name.startswith(
        ("eksport_", "export_", "result_", "results_", "group_")
    )


def _is_saved_analysis_path(path: Path) -> bool:
    return path.name.endswith(".positions.json") or path.name.endswith(".lts.json")


def _copy_non_raw_export_tree(source: Path, destination: Path) -> None:
    for source_path in _iter_non_raw_export_paths(source):
        relative_path = source_path.relative_to(source)
        destination_path = destination / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def _iter_non_raw_export_paths(root: Path) -> Iterable[Path]:
    return (
        path
        for path in root.rglob("*")
        if path.is_dir() or (path.is_file() and not _is_raw_tiff(path))
    )


def _is_raw_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _enrich_per_spot_csv(
    path: Path,
    *,
    metrics_by_key: Mapping[tuple[int, int], FwhmMetrics],
) -> CsvEnrichmentSummary:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_columns = list(reader.fieldnames or [])
        rows = list(reader)

    output_columns = original_columns + [
        column for column in FWHM_EXPORT_COLUMNS if column not in original_columns
    ]
    for row in rows:
        metrics = FwhmMetrics.from_mapping(row)
        fit_allows_lookup = "fit_ok" not in row or is_true(row.get("fit_ok"))
        if metrics.source is FwhmSource.MISSING and fit_allows_lookup:
            key = _row_fit_key(row)
            if key is not None:
                metrics = metrics_by_key.get(key, metrics)
        row.update(metrics.as_csv_fields())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)

    per_frame_path = _available_per_frame_path(path)
    _write_fwhm_per_frame(per_frame_path, rows)
    provenance = provenance_counts(
        FwhmSource(row.get("fwhm_source", FwhmSource.MISSING.value))
        for row in rows
    )
    return CsvEnrichmentSummary(
        row_count=len(rows),
        provenance=provenance,
        missing_fwhm_count=provenance[FwhmSource.MISSING.value],
        per_frame_path=per_frame_path,
    )


def _available_per_frame_path(csv_path: Path) -> Path:
    default_path = csv_path.parent / "fwhm_per_frame.csv"
    if not default_path.exists():
        return default_path
    stem_path = csv_path.parent / f"fwhm_per_frame_{csv_path.stem}.csv"
    if not stem_path.exists():
        return stem_path
    suffix = 2
    while True:
        candidate = csv_path.parent / f"fwhm_per_frame_{csv_path.stem}_{suffix}.csv"
        if not candidate.exists():
            return candidate
        suffix += 1


def _write_fwhm_per_frame(path: Path, rows: list[dict[str, str]]) -> None:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row.get("frame_id", ""))].append(row)

    fieldnames = [
        "frame_id",
        "temperature",
        "energy",
        "coverage",
        "fwhm_x_mean",
        "fwhm_y_mean",
        "fwhm_mean",
        "fwhm_geometric_mean",
        "fwhm_count",
        "fwhm_missing_count",
    ]
    summary_rows: list[dict[str, object]] = []
    for frame_id in sorted(grouped_rows, key=_frame_sort_key):
        frame_rows = grouped_rows[frame_id]
        valid_metrics = [
            FwhmMetrics.from_mapping(row)
            for row in frame_rows
            if row.get("fwhm_source") != "missing"
        ]
        valid_metrics = [
            metrics
            for metrics in valid_metrics
            if metrics.fwhm_x is not None and metrics.fwhm_y is not None
        ]
        summary_rows.append(
            {
                "frame_id": frame_id,
                "temperature": _first_present(frame_rows, "temperature"),
                "energy": _first_present(frame_rows, "energy"),
                "coverage": _first_present(frame_rows, "coverage"),
                "fwhm_x_mean": _mean(metrics.fwhm_x for metrics in valid_metrics),
                "fwhm_y_mean": _mean(metrics.fwhm_y for metrics in valid_metrics),
                "fwhm_mean": _mean(metrics.mean for metrics in valid_metrics),
                "fwhm_geometric_mean": _mean(
                    metrics.geometric for metrics in valid_metrics
                ),
                "fwhm_count": len(valid_metrics),
                "fwhm_missing_count": len(frame_rows) - len(valid_metrics),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _source_artifact_files(
    *,
    export_directories: list[Path],
    standalone_export_directories: list[Path],
    sidecar_directories: list[Path],
    per_spot_csv_paths: list[Path],
    session_paths: list[Path],
) -> list[Path]:
    files = set(per_spot_csv_paths) | set(session_paths)
    for copied_directory in export_directories + sidecar_directories:
        files.update(path for path in copied_directory.rglob("*") if path.is_file())
    for copied_directory in standalone_export_directories:
        files.update(
            path
            for path in _iter_non_raw_export_paths(copied_directory)
            if path.is_file()
        )
    return sorted(files)


def _inventory_fingerprint(source: Path, source_files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in source_files:
        relative_path = path.relative_to(source).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_manifest(
    *,
    source: Path,
    destination: Path,
    staging: Path,
    source_files: list[Path],
    input_fingerprint: str,
    csv_summaries: Mapping[Path, CsvEnrichmentSummary],
    session_summaries: Mapping[Path, FwhmRecordSummary],
) -> dict:
    file_entries: list[dict[str, object]] = []
    generated_files: list[str] = []
    for source_path in source_files:
        relative_path = source_path.relative_to(source)
        output_path = staging / relative_path
        entry: dict[str, object] = {
            "relative_source_path": relative_path.as_posix(),
            "relative_output_path": relative_path.as_posix(),
            "source_sha256": _sha256(source_path),
            "output_sha256": _sha256(output_path),
            "kind": "copied",
        }
        csv_summary = csv_summaries.get(source_path)
        if csv_summary is not None:
            entry.update(
                {
                    "kind": "enriched_per_spot_csv",
                    "row_count": csv_summary.row_count,
                    "fwhm_provenance": csv_summary.provenance,
                    "missing_fwhm_count": csv_summary.missing_fwhm_count,
                }
            )
            generated_files.append(
                csv_summary.per_frame_path.relative_to(staging).as_posix()
            )
        elif _is_saved_analysis_path(source_path):
            entry["kind"] = "enriched_saved_analysis"
            session_summary = session_summaries.get(source_path)
            if session_summary is not None:
                entry.update(
                    {
                        "row_count": session_summary.row_count,
                        "fwhm_provenance": session_summary.provenance,
                        "missing_fwhm_count": session_summary.missing_fwhm_count,
                    }
                )
        file_entries.append(entry)

    generated_files = sorted(generated_files)
    return {
        "version": 1,
        "source_root": str(source),
        "destination_root": str(destination),
        "input_fingerprint": input_fingerprint,
        "files": file_entries,
        "generated_files": generated_files,
        "generated_file_hashes": {
            relative_path: _sha256(staging / relative_path)
            for relative_path in generated_files
        },
        "summary": {
            "source_file_count": len(source_files),
            "enriched_csv_count": len(csv_summaries),
            "saved_analysis_count": sum(
                1 for path in source_files if _is_saved_analysis_path(path)
            ),
            "row_count": (
                sum(summary.row_count for summary in csv_summaries.values())
                + sum(summary.row_count for summary in session_summaries.values())
            ),
            "missing_fwhm_count": sum(
                summary.missing_fwhm_count for summary in csv_summaries.values()
            )
            + sum(
                summary.missing_fwhm_count
                for summary in session_summaries.values()
            ),
        },
    }


def _load_stored_fit_lookup(path: Path) -> StoredFitLookup | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metrics_by_key = _stored_fit_metrics(payload)
    return StoredFitLookup(path, metrics_by_key) if metrics_by_key else None


def _stored_fit_metrics(payload: dict) -> dict[tuple[int, int], FwhmMetrics]:
    metrics_by_key: dict[tuple[int, int], FwhmMetrics] = {}
    for record in iter_stored_fit_records(payload):
        if record.accepted and record.key is not None and record.fit_results is not None:
            metrics_by_key[record.key] = FwhmMetrics.from_mapping(record.fit_results)
    return metrics_by_key


def _select_fit_lookup(
    csv_path: Path,
    lookups: list[StoredFitLookup],
) -> StoredFitLookup | None:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        keys = {
            key
            for row in csv.DictReader(handle)
            if (key := _row_fit_key(row)) is not None
        }
    eligible = [
        lookup
        for lookup in lookups
        if _is_relative_to(csv_path, lookup.source_path.parent)
        and keys.intersection(lookup.metrics_by_key)
    ]
    if not eligible:
        return None
    nearest_parent_depth = max(
        len(lookup.source_path.parent.parts) for lookup in eligible
    )
    nearest_matches = [
        lookup
        for lookup in eligible
        if len(lookup.source_path.parent.parts) == nearest_parent_depth
    ]
    if len(nearest_matches) > 1 and not _lookups_agree_for_keys(
        nearest_matches, keys
    ):
        return None
    return min(nearest_matches, key=lambda lookup: str(lookup.source_path))


def _lookups_agree_for_keys(
    lookups: list[StoredFitLookup],
    keys: set[tuple[int, int]],
) -> bool:
    first, *remaining = lookups
    reference = {key: first.metrics_by_key.get(key) for key in keys}
    return all(
        {key: lookup.metrics_by_key.get(key) for key in keys} == reference
        for lookup in remaining
    )


def _enrich_saved_analysis_json(path: Path) -> FwhmRecordSummary:
    all_metrics: list[FwhmMetrics] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _summarize_metrics(all_metrics)
    if not isinstance(payload, dict):
        return _summarize_metrics(all_metrics)

    for record in iter_stored_fit_records(payload):
        metrics = (
            FwhmMetrics.from_mapping(record.fit_results)
            if record.accepted and record.fit_results is not None
            else FwhmMetrics()
        )
        all_metrics.append(metrics)
        if record.fit_results is not None:
            record.fit_results.update(metrics.as_json_fields())

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _summarize_metrics(all_metrics)


def _summarize_metrics(metrics: Iterable[FwhmMetrics]) -> FwhmRecordSummary:
    metrics_list = list(metrics)
    provenance = provenance_counts(metric.source for metric in metrics_list)
    return FwhmRecordSummary(
        row_count=len(metrics_list),
        provenance=provenance,
        missing_fwhm_count=provenance[FwhmSource.MISSING.value],
    )


def _row_fit_key(row: Mapping[str, object]) -> tuple[int, int] | None:
    return fit_key(row.get("frame_id"), row.get("track_id"))


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _frame_sort_key(frame_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(frame_id)
    except ValueError:
        return 1, frame_id


def _first_present(rows: list[dict[str, str]], column: str) -> str:
    return next((row[column] for row in rows if str(row.get(column, "")).strip()), "")


def _mean(values: Iterable[float | None]) -> float | str:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
