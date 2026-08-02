"""Reader-facing evidence for measurement-window and capture-contract sensitivity."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from trackleed.analysis.scientific_panel_common import METHODS
from trackleed.analysis.scientific_sensitivity_records import SensitivityResultRecord
from trackleed.analysis.scientific_tracer_records import read_csv, write_csv
from trackleed.publication.scientific_sensitivity_sources import (
    sha256_file,
    validate_sensitivity_source_data,
)


VARIANT_COUNT = 15
CASE_RESULT_COUNT = 300
METHOD_RESULT_COUNT = 75
RESULT_COUNT = CASE_RESULT_COUNT + METHOD_RESULT_COUNT
METHOD_DISPLAY_NAMES = {
    "easyleed": "EasyLEED",
    "locotrack": "LocoTrack",
    "ostrack": "OSTrack",
    "dam4sam": "DAM4SAM",
    "lk": "Lucas-Kanade",
}
IMPLEMENTATION_PATHS = (
    Path("trackleed/publication/scientific_sensitivity_evidence.py"),
    Path("trackleed/publication/scientific_sensitivity_sources.py"),
)
TABLE_S6_BEGIN = "<!-- BEGIN TABLE S6 -->"
TABLE_S6_END = "<!-- END TABLE S6 -->"


@dataclass(frozen=True)
class TableS6Report:
    result_count: int
    method_result_count: int
    case_result_count: int
    variant_count: int


def _load_sensitivity_results(
    source_directory: Path,
) -> list[SensitivityResultRecord]:
    validate_sensitivity_source_data(source_directory)
    _, rows = read_csv(
        source_directory / "sensitivity_results.csv",
        SensitivityResultRecord,
    )
    scopes = Counter(row["scope"] for row in rows)
    variants = {row["variant_id"] for row in rows}
    method_keys = {
        (row["variant_id"], row["method"])
        for row in rows
        if row["scope"] == "method"
    }
    if (
        len(rows) != RESULT_COUNT
        or scopes != {"case": CASE_RESULT_COUNT, "method": METHOD_RESULT_COUNT}
        or len(variants) != VARIANT_COUNT
        or method_keys != {(variant, method) for variant in variants for method in METHODS}
    ):
        raise ValueError("Table S6 sensitivity result matrix changed")
    for row in rows:
        if (
            (
                row["scope"] == "method"
                and (
                    row["physics_baseline_visible_opportunities"] != 221
                    or row["evaluated_gap_count"] != 24
                )
            )
            or row["geometry_inconsistency_denominator"]
            != row["accepted_measurements_all_frames"]
            or row["pair_residual_denominator"]
            != row["jointly_consistent_pair_frames"]
        ):
            raise ValueError("Table S6 denominator contract changed")
    return rows


def _render_table_s6(rows: list[SensitivityResultRecord]) -> str:
    method_rows = [row for row in rows if row["scope"] == "method"]
    lines = [
        (
            "**Table S6. Measurement-window and capture-contract "
            "sensitivity.** For the independent Ag(111) stress test, coverage "
            "uses the fixed "
            "denominator 221 and post-gap branch-consistent recovery uses the "
            "fixed denominator 24. Geometry inconsistency is conditional on all "
            "accepted fits, while the pair-distance residual retains its method- "
            "and variant-specific conditional n. Values below retain full source "
            "precision. No comparative rank or automatic configuration selection "
            "is implied."
        ),
        "",
        (
            "| Variant | Family | Method | ROI [px] | Proposal-to-fit [px] | "
            "Branch tolerance [px] | R² minimum | Amplitude minimum | "
            "Coverage | Recovery | Geometry inconsistency | "
            "Conditional median / P95 residual [px] (n) |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for row in method_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["variant_id"],
                    row["family"],
                    METHOD_DISPLAY_NAMES[row["method"]],
                    str(row["roi_side_px"]),
                    str(row["proposal_to_fit_max_px"]),
                    str(row["branch_tolerance_px"]),
                    str(row["r_squared_min"]),
                    str(row["amplitude_threshold_min"]),
                    (
                        f"{row['branch_consistent_measurements']}/"
                        f"{row['physics_baseline_visible_opportunities']}"
                    ),
                    (
                        f"{row['branch_consistent_recovered_gap_count']}/"
                        f"{row['evaluated_gap_count']}"
                    ),
                    (
                        f"{row['geometry_inconsistent_accepted_fits']}/"
                        f"{row['geometry_inconsistency_denominator']}"
                    ),
                    (
                        f"{row['median_absolute_pair_distance_residual_px']} / "
                        f"{row['p95_absolute_pair_distance_residual_px']} "
                        f"(n={row['pair_residual_denominator']})"
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_table_s6(
    source_directory: Path,
    output_directory: Path,
) -> TableS6Report:
    """Build the full-precision source table and reader-facing Markdown."""

    rows = _load_sensitivity_results(source_directory.resolve())
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_directory / "table_s6_measurement_sensitivity.csv",
        SensitivityResultRecord,
        rows,
    )
    (output_directory / "table_s6_measurement_sensitivity.md").write_text(
        _render_table_s6(rows),
        encoding="utf-8",
        newline="\n",
    )
    return TableS6Report(
        result_count=len(rows),
        method_result_count=sum(row["scope"] == "method" for row in rows),
        case_result_count=sum(row["scope"] == "case" for row in rows),
        variant_count=len({row["variant_id"] for row in rows}),
    )


def build_table_s6_artifact(
    workspace_root: Path,
    source_directory: Path,
    output_directory: Path,
    manifest_path: Path,
) -> TableS6Report:
    """Build Table S6 and bind sources, implementation, and outputs by hash."""

    workspace_root = workspace_root.resolve()
    source_directory = source_directory.resolve()
    output_directory = output_directory.resolve()
    report = build_table_s6(source_directory, output_directory)
    manifest = {
        "schema_version": 1,
        "table": "Table S6",
        "title": "Measurement-window and capture-contract sensitivity",
        "status": "completed",
        "result_count": report.result_count,
        "method_result_count": report.method_result_count,
        "case_result_count": report.case_result_count,
        "variant_count": report.variant_count,
        "source_data_sha256": {
            path.name: sha256_file(path)
            for path in sorted(source_directory.iterdir())
            if path.is_file()
        },
        "implementation_sha256": {
            path.as_posix(): sha256_file(workspace_root / path)
            for path in IMPLEMENTATION_PATHS
        },
        "outputs_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_directory.iterdir())
            if path.is_file()
        },
        "denominator_policy": (
            "fixed reference opportunity sets with explicit variant-specific "
            "denominators for conditional outcomes"
        ),
        "automatic_configuration_selection": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def synchronize_table_s6(
    table_markdown_path: Path,
    supplementary_material_path: Path,
) -> None:
    """Insert or replace the generated Table S6 block in the supplement."""

    table_markdown = table_markdown_path.read_text(encoding="utf-8").rstrip()
    supplement = supplementary_material_path.read_text(encoding="utf-8")
    block = (
        f"{TABLE_S6_BEGIN}\n"
        f"{table_markdown}\n"
        f"{TABLE_S6_END}"
    )
    begin_count = supplement.count(TABLE_S6_BEGIN)
    end_count = supplement.count(TABLE_S6_END)
    if (begin_count, end_count) == (0, 0):
        supplement = (
            supplement.rstrip()
            + "\n\n## Table S6 — Measurement-window and capture-contract sensitivity\n\n"
            + block
            + "\n"
        )
    elif (begin_count, end_count) == (1, 1):
        before, remainder = supplement.split(TABLE_S6_BEGIN, 1)
        _, after = remainder.split(TABLE_S6_END, 1)
        supplement = before + block + after
    else:
        raise ValueError("Table S6 supplement markers changed")
    supplementary_material_path.write_text(
        supplement,
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "TableS6Report",
    "build_table_s6",
    "build_table_s6_artifact",
    "synchronize_table_s6",
]
