"""Identity, scope and filesystem primitives for the Ag(111) central panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trackleed.analysis.scientific_tracer_common import (
    SCHEMA_VERSION,
    TracerValidationError,
    hash_manifest,
    implementation_sources as tracer_implementation_sources,
    read_json,
    relative_file,
    sha256,
    source_sha256,
    write_json,
)


PANEL_ID = "ag111_five_method_four_case_central"
PANEL_AUTHENTICATION_PATH = Path(
    "manuscript/scientific_panel_authentication.json"
)
BASELINE_PACKAGE = Path("manuscript/scientific_baseline")
BASELINE_MANIFEST_PATH = BASELINE_PACKAGE / "package_manifest.json"
BASELINE_INPUT_MANIFEST_PATH = BASELINE_PACKAGE / "input_manifest.json"
MEASUREMENT_CONTRACT_PATH = BASELINE_PACKAGE / "measurement_contract.json"
OPPORTUNITIES_PATH = BASELINE_PACKAGE / "opportunities.csv"
GAPS_PATH = BASELINE_PACKAGE / "gaps.csv"
PUBLISHED_SOURCE_DATA = Path(
    "manuscript/figures/figure03_independent_scane_stress/source_data"
)
PUBLISHED_CASE_RESULTS_PATH = PUBLISHED_SOURCE_DATA / "case_results.csv"
PUBLISHED_AGGREGATE_RESULTS_PATH = (
    PUBLISHED_SOURCE_DATA / "aggregate_method_results.csv"
)
METHODS = ("easyleed", "locotrack", "ostrack", "dam4sam", "lk")
DISPLAY_METHODS = {
    "easyleed": "EasyLEED",
    "locotrack": "LocoTrack",
    "ostrack": "OSTrack",
    "dam4sam": "DAM4SAM",
    "lk": "Lucas-Kanade",
}
CASE_SPECS = (
    ("scan_a_pair_1", "scan_a", "pair_1", "26a21", "pair1", 35),
    ("scan_a_pair_2", "scan_a", "pair_2", "26a21", "pair2", 35),
    ("scan_b_pair_1", "scan_b", "pair_1", "26a21b", "pair1", 46),
    ("scan_b_pair_2", "scan_b", "pair_2", "26a21b", "pair2", 46),
)
IMPLEMENTATION_SOURCE_PATHS = {
    "panel_candidates": Path(
        "trackleed/analysis/scientific_panel_candidates.py"
    ),
    "panel_common": Path("trackleed/analysis/scientific_panel_common.py"),
    "panel_contract": Path("trackleed/analysis/scientific_panel_contract.py"),
    "panel_records": Path("trackleed/analysis/scientific_panel_records.py"),
    "panel_reconstruction": Path(
        "trackleed/analysis/scientific_panel_reconstruction.py"
    ),
    "panel_validation": Path(
        "trackleed/analysis/scientific_panel_validation.py"
    ),
    "panel_orchestration": Path("trackleed/publication/scientific_panel.py"),
    "panel_outcomes": Path("trackleed/analysis/scientific_panel_outcomes.py"),
    "panel_reporting": Path("trackleed/analysis/scientific_panel_reporting.py"),
}


@dataclass(frozen=True)
class PanelReport:
    candidate_count: int
    evaluation_count: int
    case_count: int
    method_count: int
    opportunity_reference_count: int
    gap_reference_count: int
    matches_frozen_central_record: bool


def implementation_sources(workspace_root: Path) -> dict[str, dict[str, str]]:
    shared = tracer_implementation_sources(workspace_root)
    selected = {
        name: shared[name]
        for name in (
            "gaussian_fitter",
            "tracer_evaluation",
            "tracer_records",
        )
    }
    selected.update(
        {
            name: {
                "path": path.as_posix(),
                "sha256": source_sha256(workspace_root / path),
            }
            for name, path in IMPLEMENTATION_SOURCE_PATHS.items()
        }
    )
    return selected


def panel_hash_manifest(files_sha256: dict[str, str], *, immutable: bool = False):
    manifest = hash_manifest(files_sha256, immutable=immutable)
    manifest["tracer_id"] = PANEL_ID
    return manifest


__all__ = [
    "BASELINE_INPUT_MANIFEST_PATH",
    "BASELINE_MANIFEST_PATH",
    "BASELINE_PACKAGE",
    "CASE_SPECS",
    "DISPLAY_METHODS",
    "GAPS_PATH",
    "MEASUREMENT_CONTRACT_PATH",
    "METHODS",
    "PANEL_AUTHENTICATION_PATH",
    "PANEL_ID",
    "PUBLISHED_AGGREGATE_RESULTS_PATH",
    "PUBLISHED_CASE_RESULTS_PATH",
    "PanelReport",
    "SCHEMA_VERSION",
    "TracerValidationError",
    "implementation_sources",
    "panel_hash_manifest",
    "read_json",
    "relative_file",
    "sha256",
    "write_json",
]
