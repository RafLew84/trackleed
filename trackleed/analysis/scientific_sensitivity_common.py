"""Identity, report and provenance primitives for Ag(111) sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trackleed.analysis.scientific_panel_common import (
    PANEL_ID,
    SCHEMA_VERSION,
    sha256,
)
from trackleed.analysis.scientific_tracer_common import source_sha256


SENSITIVITY_ID = f"{PANEL_ID}_measurement_sensitivity"
DATA_ARTIFACT_NAMES = (
    "candidate_records.csv",
    "evaluation_records.csv",
    "sensitivity_results.csv",
    "rejection_cause_counts.csv",
    "gate_margin_audit.csv",
    "summary.json",
    "provenance.json",
)
OUTPUT_FILE_NAMES = set(DATA_ARTIFACT_NAMES) | {
    "output_manifest.json",
    "validation_report.json",
}
IMPLEMENTATION_PATHS = {
    "panel_candidates": Path("trackleed/analysis/scientific_panel_candidates.py"),
    "panel_outcomes": Path("trackleed/analysis/scientific_panel_outcomes.py"),
    "panel_reporting": Path("trackleed/analysis/scientific_panel_reporting.py"),
    "sensitivity_analysis": Path("trackleed/analysis/scientific_sensitivity.py"),
    "sensitivity_audit": Path("trackleed/analysis/scientific_sensitivity_audit.py"),
    "sensitivity_records": Path("trackleed/analysis/scientific_sensitivity_records.py"),
    "sensitivity_reporting": Path(
        "trackleed/analysis/scientific_sensitivity_reporting.py"
    ),
    "sensitivity_validation": Path(
        "trackleed/analysis/scientific_sensitivity_validation.py"
    ),
    "threshold_evaluation": Path("trackleed/analysis/scientific_tracer_evaluation.py"),
    "sensitivity_orchestration": Path(
        "trackleed/publication/scientific_sensitivity.py"
    ),
}


@dataclass(frozen=True)
class SensitivityReport:
    variant_count: int
    unique_roi_candidate_set_count: int
    candidate_count: int
    evaluation_count: int
    width_sensitivity_triggered: bool
    width_contract_count: int
    central_variant_matches_frozen_record: bool


def implementation_sources(
    workspace_root: Path,
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": path.as_posix(),
            "sha256": source_sha256(workspace_root / path),
        }
        for name, path in IMPLEMENTATION_PATHS.items()
    }


def sensitivity_output_manifest(output_dir: Path) -> dict[str, object]:
    return {
        "files_sha256": {
            name: sha256(output_dir / name) for name in DATA_ARTIFACT_NAMES
        },
        "schema_version": SCHEMA_VERSION,
        "sensitivity_id": SENSITIVITY_ID,
    }


__all__ = [
    "DATA_ARTIFACT_NAMES",
    "OUTPUT_FILE_NAMES",
    "SENSITIVITY_ID",
    "SensitivityReport",
    "implementation_sources",
    "sensitivity_output_manifest",
]
