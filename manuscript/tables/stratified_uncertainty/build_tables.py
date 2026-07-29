#!/usr/bin/env python3
"""Build stratified descriptive tables without treating images as replicates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"

SCAN_LABELS = {
    "26a21": "Scan A",
    "26a21b": "Scan B",
}
TRIAL_LABELS = {
    0: "initial low-temperature state",
    372: "800 °C plateau onset",
    378: "post-plateau contraction minimum",
    700: "final cooled state",
}
DISPLAY_METHOD_ORDER = (
    "LocoTrack",
    "OSTrack",
    "DAM4SAM",
    "Lucas-Kanade",
    "EasyLEED",
    "Oracle ROI control",
)


@dataclass(frozen=True)
class CountOutcome:
    """One count-based outcome with a single fraction/rate source."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            self.denominator <= 0
            or self.numerator < 0
            or self.numerator > self.denominator
        ):
            raise ValueError(
                "Count outcome requires 0 <= numerator <= denominator."
            )

    @property
    def fraction(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def rate(self) -> float:
        return self.numerator / self.denominator


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_data() -> None:
    provenance_path = SOURCE_DATA_DIRECTORY / "source_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for source in provenance["sources"]:
        frozen_path = TABLE_ROOT / str(source["frozen_path"])
        if sha256_file(frozen_path) != source["frozen_sha256"]:
            raise ValueError(
                f"Frozen source hash changed: {frozen_path.name}"
            )


def linear_percentile(values: list[float], percentile: float) -> float:
    """Match the standard linear percentile for a nonempty value list."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def build_energy_scan_case_table() -> list[dict[str, object]]:
    """Return one descriptive row per frozen scan-pair-method case."""

    rows = read_csv(SOURCE_DATA_DIRECTORY / "energy_scan_case_results.csv")
    output: list[dict[str, object]] = []
    for row in rows:
        measurement_coverage = CountOutcome(
            int(row["branch_consistent_measurements"]),
            int(row["visible_opportunities"]),
        )
        gap_recovery = CountOutcome(
            int(row["recovered_gaps"]),
            int(row["evaluated_gaps"]),
        )
        geometry_inconsistency = CountOutcome(
            int(row["geometry_inconsistent_accepted_fits"]),
            int(row["accepted_measurements_all_frames"]),
        )
        output.append(
            {
                "scan": SCAN_LABELS[row["series"]],
                "pair": row["pair"].replace("pair", "pair "),
                "method": row["display_method"],
                "measurement_coverage": measurement_coverage.fraction,
                "gap_recovery": gap_recovery.fraction,
                "geometry_inconsistency": geometry_inconsistency.fraction,
                "conditional_pair_frames": int(
                    row["jointly_consistent_pair_frames"]
                ),
                "conditional_median_abs_distance_residual_px": float(
                    row["case_median_abs_distance_residual_px"]
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            tuple(SCAN_LABELS.values()).index(str(row["scan"])),
            str(row["pair"]),
            DISPLAY_METHOD_ORDER.index(str(row["method"])),
        ),
    )


def build_leave_one_case_out_table() -> list[dict[str, object]]:
    """Recompute count outcomes after omitting each frozen scan-pair case."""

    rows = read_csv(SOURCE_DATA_DIRECTORY / "energy_scan_case_results.csv")
    cases = list(dict.fromkeys((row["series"], row["pair"]) for row in rows))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    display_names = {
        row["method"]: row["display_method"]
        for row in rows
    }
    output: list[dict[str, object]] = []
    for omitted_series, omitted_pair in cases:
        for method in methods:
            included = [
                row
                for row in rows
                if row["method"] == method
                and (row["series"], row["pair"])
                != (omitted_series, omitted_pair)
            ]
            visible = sum(
                int(row["visible_opportunities"]) for row in included
            )
            consistent = sum(
                int(row["branch_consistent_measurements"])
                for row in included
            )
            gaps = sum(int(row["evaluated_gaps"]) for row in included)
            recovered = sum(int(row["recovered_gaps"]) for row in included)
            accepted = sum(
                int(row["accepted_measurements_all_frames"])
                for row in included
            )
            inconsistent = sum(
                int(row["geometry_inconsistent_accepted_fits"])
                for row in included
            )
            measurement_coverage = CountOutcome(consistent, visible)
            gap_recovery = CountOutcome(recovered, gaps)
            geometry_inconsistency = CountOutcome(inconsistent, accepted)
            output.append(
                {
                    "omitted_scan": SCAN_LABELS[omitted_series],
                    "omitted_pair": omitted_pair.replace("pair", "pair "),
                    "method": display_names[method],
                    "included_case_count": len(included),
                    "measurement_coverage": measurement_coverage.fraction,
                    "measurement_coverage_rate": measurement_coverage.rate,
                    "gap_recovery": gap_recovery.fraction,
                    "gap_recovery_rate": gap_recovery.rate,
                    "geometry_inconsistency": (
                        geometry_inconsistency.fraction
                    ),
                    "geometry_inconsistency_rate": (
                        geometry_inconsistency.rate
                    ),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            tuple(SCAN_LABELS.values()).index(str(row["omitted_scan"])),
            str(row["omitted_pair"]),
            DISPLAY_METHOD_ORDER.index(str(row["method"])),
        ),
    )


def build_controlled_translation_trial_table() -> list[dict[str, object]]:
    """Summarize deterministic translation outcomes within each source trial."""

    spot_rows = read_csv(
        SOURCE_DATA_DIRECTORY / "controlled_translation_spot_results.csv"
    )
    pair_rows = read_csv(
        SOURCE_DATA_DIRECTORY / "controlled_translation_pair_results.csv"
    )
    methods = list(dict.fromkeys(row["method"] for row in spot_rows))
    trials = list(
        dict.fromkeys(int(row["source_frame_order"]) for row in spot_rows)
    )
    display_names = {
        row["method"]: row["display_method"]
        for row in spot_rows
    }
    roles = {row["method"]: row["role"] for row in spot_rows}
    output: list[dict[str, object]] = []
    for trial in trials:
        for method in methods:
            trial_spots = [
                row
                for row in spot_rows
                if row["method"] == method
                and int(row["source_frame_order"]) == trial
                and as_bool(row["is_evaluation_step"])
            ]
            valid_errors = [
                float(row["measurement_displacement_error_px"])
                for row in trial_spots
                if as_bool(row["fit_valid"])
            ]
            trial_pairs = [
                row
                for row in pair_rows
                if row["method"] == method
                and int(row["source_frame_order"]) == trial
                and as_bool(row["is_evaluation_step"])
            ]
            valid_pair_changes = [
                abs(float(row["delta_d_over_d_percent"]))
                for row in trial_pairs
                if as_bool(row["pair_valid"])
            ]
            proposal_count = sum(
                as_bool(row["proposal_available"]) for row in trial_spots
            )
            fit_count = sum(
                as_bool(row["fit_valid"]) for row in trial_spots
            )
            valid_pair_count = sum(
                as_bool(row["pair_valid"]) for row in trial_pairs
            )
            proposal_coverage = CountOutcome(
                proposal_count,
                len(trial_spots),
            )
            measurement_coverage = CountOutcome(
                fit_count,
                len(trial_spots),
            )
            pair_coverage = CountOutcome(
                valid_pair_count,
                len(trial_pairs),
            )
            output.append(
                {
                    "trial": TRIAL_LABELS[trial],
                    "temperature_c": int(
                        float(trial_spots[0]["temperature_c"])
                    ),
                    "method": display_names[method],
                    "role": roles[method],
                    "proposal_coverage": proposal_coverage.fraction,
                    "measurement_coverage": measurement_coverage.fraction,
                    "measurement_error_median_px": statistics.median(
                        valid_errors
                    ),
                    "measurement_error_p95_px": linear_percentile(
                        valid_errors,
                        95.0,
                    ),
                    "measurement_error_max_px": max(valid_errors),
                    "pair_coverage": pair_coverage.fraction,
                    "maximum_abs_delta_d_over_d_pp": max(
                        valid_pair_changes
                    ),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            tuple(TRIAL_LABELS.values()).index(str(row["trial"])),
            DISPLAY_METHOD_ORDER.index(str(row["method"])),
        ),
    )


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_percentage_range(
    values: list[float],
) -> str:
    return f"{100.0 * min(values):.1f}–{100.0 * max(values):.1f}%"


def render_energy_scan_markdown(
    case_rows: list[dict[str, object]],
    leave_one_out_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Table S3. Stratified Ag(111) energy-scan outcomes",
        "",
        (
            "Each row is one frozen scan-pair-method case. Conditional "
            "residuals are shown only with their surviving pair-frame count."
        ),
        "",
        (
            "| Scan | Pair | Method | Coverage | Gap recovery | Geometry "
            "inconsistency | Conditional median residual [px] | Conditional n |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in case_rows:
        lines.append(
            "| {scan} | {pair} | {method} | {measurement_coverage} | "
            "{gap_recovery} | {geometry_inconsistency} | {residual:.3f} | "
            "{conditional_pair_frames} |".format(
                **row,
                residual=float(
                    row["conditional_median_abs_distance_residual_px"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Leave-one-case-out sensitivity",
            "",
            (
                "One frozen scan-pair case is omitted at a time. The analysis "
                "recomputes count-based outcomes only; it is a sensitivity "
                "check, not a confidence interval. Case medians are not "
                "reaggregated as pooled residuals."
            ),
            "",
            (
                "| Method | Coverage range | Gap-recovery range | Geometry-"
                "inconsistency range |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for method in DISPLAY_METHOD_ORDER[:-1]:
        method_rows = [
            row
            for row in leave_one_out_rows
            if row["method"] == method
        ]
        lines.append(
            "| {method} | {coverage} | {recovery} | {geometry} |".format(
                method=method,
                coverage=format_percentage_range(
                    [
                        float(row["measurement_coverage_rate"])
                        for row in method_rows
                    ]
                ),
                recovery=format_percentage_range(
                    [
                        float(row["gap_recovery_rate"])
                        for row in method_rows
                    ]
                ),
                geometry=format_percentage_range(
                    [
                        float(row["geometry_inconsistency_rate"])
                        for row in method_rows
                    ]
                ),
            )
        )
    return "\n".join(lines) + "\n"


def render_translation_markdown(
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Table S5. Controlled translations by source trial",
        "",
        (
            "Errors are descriptive within each deterministic six-image "
            "trial. The oracle ROI is a measurement control, not a tracker."
        ),
        "",
        (
            "| Trial | T [°C] | Method | Proposal coverage | Measurement "
            "coverage | Median / P95 / max error [px] | Pair coverage | "
            "Maximum absolute Delta D/D [pp] |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {temperature_c} | {method} | {proposal_coverage} | "
            "{measurement_coverage} | {median:.4f} / {p95:.4f} / {maximum:.4f} "
            "| {pair_coverage} | {pair_max:.6f} |".format(
                **row,
                median=float(row["measurement_error_median_px"]),
                p95=float(row["measurement_error_p95_px"]),
                maximum=float(row["measurement_error_max_px"]),
                pair_max=float(row["maximum_abs_delta_d_over_d_pp"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_artifact(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    """Write deterministic supplementary tables and their manifest."""

    validate_source_data()
    output_directory.mkdir(parents=True, exist_ok=True)
    case_rows = build_energy_scan_case_table()
    leave_one_out_rows = build_leave_one_case_out_table()
    trial_rows = build_controlled_translation_trial_table()

    output_paths = {
        "controlled_translation_by_trial.csv": (
            output_directory / "controlled_translation_by_trial.csv"
        ),
        "energy_scan_by_case.csv": (
            output_directory / "energy_scan_by_case.csv"
        ),
        "energy_scan_leave_one_case_out.csv": (
            output_directory / "energy_scan_leave_one_case_out.csv"
        ),
        "table_s3_energy_scan_by_case.md": (
            output_directory / "table_s3_energy_scan_by_case.md"
        ),
        "table_s5_controlled_translation_by_trial.md": (
            output_directory / "table_s5_controlled_translation_by_trial.md"
        ),
    }
    write_csv_rows(
        output_paths["controlled_translation_by_trial.csv"],
        trial_rows,
    )
    write_csv_rows(output_paths["energy_scan_by_case.csv"], case_rows)
    write_csv_rows(
        output_paths["energy_scan_leave_one_case_out.csv"],
        leave_one_out_rows,
    )
    output_paths["table_s3_energy_scan_by_case.md"].write_text(
        render_energy_scan_markdown(case_rows, leave_one_out_rows),
        encoding="utf-8",
    )
    output_paths["table_s5_controlled_translation_by_trial.md"].write_text(
        render_translation_markdown(trial_rows),
        encoding="utf-8",
    )

    source_paths = sorted(
        path
        for path in SOURCE_DATA_DIRECTORY.iterdir()
        if path.suffix in {".csv", ".json"}
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "uncertainty_policy": {
            "inference": "descriptive; no confidence intervals or p-values",
            "replication_units": {
                "controlled_translation": (
                    "source trial (n=4; one Cu(111) series)"
                ),
                "independent_energy_scan": (
                    "series (n=2), stratified by frozen pair (four cases)"
                ),
                "controlled_sam2_family": (
                    "single series, stratified by frozen pair (n=2)"
                ),
                "cu111_positive_control": (
                    "single series, stratified by frozen pair (n=2)"
                ),
            },
            "sensitivity_analysis": (
                "leave one frozen scan-pair case out; recompute count-based "
                "outcomes only"
            ),
            "gaussian_fit_uncertainty": (
                "per-measurement diagnostic; not sampling uncertainty for "
                "tracker performance"
            ),
            "prohibited": [
                "image-level bootstrap",
                (
                    "binomial confidence intervals over correlated frames "
                    "or gaps"
                ),
                "reaggregation of case medians as pooled residuals",
            ],
        },
        "source_data": {
            path.relative_to(TABLE_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
        "outputs": {
            name: sha256_file(path)
            for name, path in output_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    build_artifact()
