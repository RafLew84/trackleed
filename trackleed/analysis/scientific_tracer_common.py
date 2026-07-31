"""Shared identity and filesystem primitives for the scientific tracer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
TRACER_ID = "scan_a_pair_1_locotrack_central"
BASELINE_PACKAGE = Path("manuscript/scientific_baseline")
TRACER_AUTHENTICATION_PATH = Path(
    "manuscript/scientific_tracer_authentication.json"
)
PROPOSAL_PATH = Path(
    "manuscript/scientific_baseline/frozen_inputs/proposals/"
    "scan_a_pair_1/locotrack.npz"
)
CASE_SUMMARY_PATH = Path(
    "manuscript/scientific_baseline/frozen_inputs/cases/"
    "scan_a_pair_1/summary.json"
)
CASE_MEASUREMENTS_PATH = Path(
    "manuscript/scientific_baseline/frozen_inputs/cases/"
    "scan_a_pair_1/scan_measurements.csv"
)
MEASUREMENT_CONTRACT_PATH = Path(
    "manuscript/scientific_baseline/measurement_contract.json"
)
OPPORTUNITIES_PATH = Path("manuscript/scientific_baseline/opportunities.csv")
PUBLISHED_CASE_RESULTS_PATH = Path(
    "manuscript/figures/figure03_independent_scane_stress/"
    "source_data/case_results.csv"
)
IMPLEMENTATION_SOURCE_PATHS = {
    "fwhm_metrics": Path("trackleed/analysis/fwhm_metrics.py"),
    "gaussian_fitter": Path("trackleed/analysis/peak_analyzer.py"),
    "tracer_contract": Path("trackleed/analysis/scientific_tracer_contract.py"),
    "tracer_evaluation": Path(
        "trackleed/analysis/scientific_tracer_evaluation.py"
    ),
    "tracer_records": Path("trackleed/analysis/scientific_tracer_records.py"),
    "tracer_reconstruction": Path(
        "trackleed/analysis/scientific_tracer_reconstruction.py"
    ),
    "tracer_validation": Path(
        "trackleed/analysis/scientific_tracer_validation.py"
    ),
    "tracer_orchestration": Path(
        "trackleed/publication/scientific_tracer.py"
    ),
}
BRANCHES = ("B1", "B2")


class TracerValidationError(ValueError):
    """The reconstruction tracer differs from frozen evidence or contract."""


class SchemaCompatibility(Enum):
    """How current typed records may relate to an authenticated frozen schema."""

    EXACT = "exact"
    RETAINED_FIELDS = "retained_fields"

    @property
    def requires_current_schema(self) -> bool:
        return self is SchemaCompatibility.EXACT

    @property
    def compares_retained_fields(self) -> bool:
        return self is SchemaCompatibility.RETAINED_FIELDS


@dataclass(frozen=True)
class TracerReport:
    candidate_count: int
    evaluation_count: int
    branch_consistent_count: int
    evaluated_gap_count: int
    recovered_gap_count: int
    matches_frozen_central_record: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    """Hash source text independently of a checkout's newline convention."""

    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def implementation_sources(workspace_root: Path) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": path.as_posix(),
            "sha256": source_sha256(workspace_root / path),
        }
        for name, path in IMPLEMENTATION_SOURCE_PATHS.items()
    }


def hash_manifest(
    files_sha256: Mapping[str, str],
    *,
    immutable: bool | None = None,
) -> dict[str, object]:
    """Build the shared identity envelope for package and output manifests."""

    manifest: dict[str, object] = {
        "files_sha256": dict(files_sha256),
        "schema_version": SCHEMA_VERSION,
        "tracer_id": TRACER_ID,
    }
    if immutable is not None:
        manifest["immutable"] = immutable
    return manifest


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TracerValidationError(f"Cannot read tracer JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TracerValidationError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def relative_file(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise TracerValidationError(f"Tracer path must be relative: {value}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise TracerValidationError(f"Tracer path escapes workspace: {value}")
    return resolved


__all__ = [
    "BASELINE_PACKAGE",
    "BRANCHES",
    "CASE_MEASUREMENTS_PATH",
    "CASE_SUMMARY_PATH",
    "IMPLEMENTATION_SOURCE_PATHS",
    "MEASUREMENT_CONTRACT_PATH",
    "OPPORTUNITIES_PATH",
    "PROPOSAL_PATH",
    "PUBLISHED_CASE_RESULTS_PATH",
    "SCHEMA_VERSION",
    "SchemaCompatibility",
    "TRACER_AUTHENTICATION_PATH",
    "TRACER_ID",
    "TracerReport",
    "TracerValidationError",
    "implementation_sources",
    "hash_manifest",
    "read_json",
    "relative_file",
    "sha256",
    "source_sha256",
    "write_json",
]
