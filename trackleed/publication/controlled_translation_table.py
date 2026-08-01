"""Supplementary Table S5 for controlled-translation source trials."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from trackleed.analysis.controlled_translation_evidence import (
    TrialOutcome,
    build_trial_outcomes,
)
from trackleed.publication.controlled_translation_sources import (
    sha256_file,
    validate_controlled_translation_source_data,
)


IMPLEMENTATION_PATHS = (
    Path("trackleed/analysis/controlled_translation_evidence.py"),
    Path("trackleed/publication/controlled_translation_sources.py"),
    Path("trackleed/publication/controlled_translation_table.py"),
)
TABLE_S5_BEGIN = "<!-- BEGIN TABLE S5 -->"
TABLE_S5_END = "<!-- END TABLE S5 -->"


@dataclass(frozen=True)
class TableS5Report:
    row_count: int
    trial_count: int
    method_count: int


def _write_csv(path: Path, rows: list[TrialOutcome]) -> None:
    if not rows:
        raise ValueError("Table S5 cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].as_record()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(row.as_record() for row in rows)


def render_table_s5(rows: list[TrialOutcome]) -> str:
    """Render the reader-facing trial-stratified controlled-translation table."""

    lines = [
        (
            "**Table S5. Controlled-translation outcomes by independently "
            "initialized source trial.** Each deterministic six-image trial "
            "contains one seed and five evaluation steps on the same closed "
            "path, providing 20 spot-step and 10 pair-step fixed opportunities "
            "per method. Displacement summaries are conditional on accepted "
            "Gaussian measurements; pair invariance is conditional on jointly "
            "valid pairs, and both conditional n values are shown. The four "
            "source trials are descriptive strata drawn from one Cu(111) "
            "series, not four independent acquisitions. The oracle ROI is a "
            "measurement control, not a tracker. Known translations assess "
            "measurement-chain equivariance and common-estimator behavior, "
            "not absolute tracking accuracy and not identity ground truth."
        ),
        "",
        (
            "| Trial | T [deg C] | Method | Proposal / constructed-ROI coverage "
            "| Gaussian measurement coverage | Conditional median / P95 / max "
            "displacement error [px] (n) | Pair coverage | Max absolute "
            "Delta D/D [pp] (n) |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {temperature_c} | {method} | {proposal} | "
            "{measurement} | {median:.4f} / {p95:.4f} / {maximum:.4f} "
            "(n={measurement_n}) | {pair} | {pair_max:.6f} "
            "(n={pair_n}) |".format(
                trial=row.trial,
                temperature_c=row.temperature_c,
                method=row.display_method,
                proposal=row.proposal_coverage.fraction,
                measurement=row.measurement_coverage.fraction,
                median=row.measurement_error_median_px,
                p95=row.measurement_error_p95_px,
                maximum=row.measurement_error_max_px,
                measurement_n=row.conditional_measurement_error_n,
                pair=row.pair_coverage.fraction,
                pair_max=row.maximum_abs_delta_d_over_d_pp,
                pair_n=row.conditional_pair_invariance_n,
            )
        )
    return "\n".join(lines) + "\n"


def build_table_s5_artifact(
    *,
    workspace_root: Path,
    source_directory: Path,
    output_directory: Path,
    manifest_path: Path,
) -> TableS5Report:
    """Build Table S5 and bind its source, implementation, and outputs."""

    workspace_root = workspace_root.resolve()
    source_directory = source_directory.resolve()
    output_directory = output_directory.resolve()
    validate_controlled_translation_source_data(source_directory)
    rows = build_trial_outcomes(source_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "table_s5_controlled_translation.csv"
    markdown_path = output_directory / "table_s5_controlled_translation.md"
    _write_csv(csv_path, rows)
    markdown_path.write_text(
        render_table_s5(rows),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "table": "Table S5",
        "title": "Controlled-translation outcomes by source trial",
        "row_count": len(rows),
        "trial_count": len({row.source_frame_order for row in rows}),
        "method_count": len({row.method for row in rows}),
        "source_data_path": (
            source_directory.relative_to(workspace_root).as_posix()
        ),
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
            for path in (csv_path, markdown_path)
        },
        "fixed_denominators": {
            "spot_step_opportunities_per_trial_method": 20,
            "pair_step_opportunities_per_trial_method": 10,
        },
        "conditional_denominators": [
            "accepted Gaussian measurements",
            "jointly valid pair measurements",
        ],
        "claim_limit": (
            "Known translations assess measurement-chain equivariance and "
            "common-estimator behavior, not absolute tracking accuracy or "
            "identity ground truth."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    trial_count = len({row.source_frame_order for row in rows})
    method_count = len({row.method for row in rows})
    return TableS5Report(
        row_count=len(rows),
        trial_count=trial_count,
        method_count=method_count,
    )


def synchronize_table_s5(
    table_markdown_path: Path,
    supplementary_material_path: Path,
) -> None:
    """Insert or replace the generated Table S5 block in the supplement."""

    markdown = table_markdown_path.read_text(encoding="utf-8").rstrip()
    supplement = supplementary_material_path.read_text(encoding="utf-8")
    block = f"{TABLE_S5_BEGIN}\n{markdown}\n{TABLE_S5_END}"
    begin_count = supplement.count(TABLE_S5_BEGIN)
    end_count = supplement.count(TABLE_S5_END)
    if (begin_count, end_count) == (1, 1):
        before, remainder = supplement.split(TABLE_S5_BEGIN, 1)
        _, after = remainder.split(TABLE_S5_END, 1)
        updated = before + block + after
    elif (begin_count, end_count) == (0, 0):
        updated = (
            supplement.rstrip()
            + "\n\n## Table S5 - controlled translations by source trial\n\n"
            + block
            + "\n"
        )
    else:
        raise ValueError("Table S5 supplement markers changed")
    supplementary_material_path.write_text(
        updated,
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "TABLE_S5_BEGIN",
    "TABLE_S5_END",
    "TableS5Report",
    "build_table_s5_artifact",
    "render_table_s5",
    "synchronize_table_s5",
]
