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
REPOSITORY_ROOT = TABLE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"

SUPPLEMENT_BEGIN_MARKER = "<!-- BEGIN TABLE S3 -->"
SUPPLEMENT_END_MARKER = "<!-- END TABLE S3 -->"
SUPPLEMENT_HEADING = (
    "## Table S3 — stratified Ag(111) energy-scan outcomes"
)

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


def _validate_sha256(value: object, *, field: str) -> None:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Malformed SHA-256 in {field}: {digest}")


def validate_source_data() -> dict[str, object]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {"schema_version", "freeze_policy", "sources"}
    ):
        raise ValueError("Malformed source provenance.")
    sources = provenance["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Source provenance must list three frozen tables.")
    required_keys = {
        "frozen_path",
        "frozen_sha256",
        "original_path",
        "original_sha256",
    }
    optional_keys = {"original_required_at_build"}
    optional_original_count = 0
    for source in sources:
        if (
            not isinstance(source, dict)
            or not required_keys.issubset(source)
            or not set(source).issubset(required_keys | optional_keys)
        ):
            raise ValueError("Malformed source-provenance entry.")
        if (
            "original_required_at_build" in source
            and not isinstance(source["original_required_at_build"], bool)
        ):
            raise ValueError("original_required_at_build must be Boolean.")
        for text_key in ("frozen_path", "original_path"):
            if (
                not isinstance(source[text_key], str)
                or not source[text_key].strip()
            ):
                raise ValueError(f"Malformed provenance field: {text_key}")
        _validate_sha256(source["frozen_sha256"], field="frozen_sha256")
        _validate_sha256(source["original_sha256"], field="original_sha256")
        if not source.get("original_required_at_build", True):
            original_parts = Path(str(source["original_path"])).parts
            if not original_parts or original_parts[0] != ".scratch":
                raise ValueError(
                    "Only .scratch run-workspace originals may be optional."
                )
            optional_original_count += 1

        frozen_path = (
            TABLE_ROOT / str(source["frozen_path"])
        ).resolve()
        if TABLE_ROOT not in frozen_path.parents:
            raise ValueError("Frozen provenance path escapes the artifact.")
        if not frozen_path.is_file():
            raise ValueError(f"Missing frozen source: {frozen_path.name}")
        if sha256_file(frozen_path) != source["frozen_sha256"]:
            raise ValueError(
                f"Frozen source hash changed: {frozen_path.name}"
            )
        original_path = (
            REPOSITORY_ROOT / str(source["original_path"])
        ).resolve()
        if (
            original_path != REPOSITORY_ROOT
            and REPOSITORY_ROOT not in original_path.parents
        ):
            raise ValueError("Original provenance path escapes the repository.")
        if not original_path.is_file():
            if source.get("original_required_at_build", True):
                raise ValueError(f"Missing original source: {original_path.name}")
            continue
        if sha256_file(original_path) != source["original_sha256"]:
            raise ValueError(
                f"Original source hash changed: {original_path.name}"
            )
    if optional_original_count != 2:
        raise ValueError("Exactly two run-workspace originals must be optional.")
    return provenance


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
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
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
        (
            "**Table S3. Stratified outcomes for the independent Ag(111) "
            "energy-scan stress test.** Coverage is branch-consistent "
            "measurements / baseline-visible opportunities; post-gap identity "
            "recovery is recovered / evaluable natural I(V) gaps; geometry "
            "inconsistency is accepted Gaussian fits more than 18 px from the "
            "expected branch / all accepted Gaussian fits across all "
            "visibility regimes. The conditional residual is the median "
            "absolute pair-distance residual and is evaluated only when both "
            "branches are valid and branch-consistent, with its surviving "
            "pair-frame n shown separately. The four scan–pair cases are "
            "descriptive strata from two independently acquired scans, not "
            "four independent acquisitions."
        ),
        "",
        "**A. Outcomes by frozen scan–pair case**",
        "",
        (
            "| Scan | Pair | Method | Branch-consistent coverage | Post-gap "
            "identity recovery | Geometry-inconsistent accepted fits | "
            "Conditional median absolute pair-distance residual [px] | "
            "Conditional n |"
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
            "**B. Leave-one-case-out sensitivity**",
            "",
            (
                "Each of the four frozen scan–pair cases is omitted once. The "
                "remaining three cases are pooled and only the count-based "
                "coverage, post-gap identity recovery, and geometry-"
                "inconsistency rates are recomputed. The displayed extrema "
                "across the four omissions are a sensitivity check, not a "
                "confidence interval. Conditional case-median residuals are "
                "not reaggregated."
            ),
            "",
            (
                "| Method | Coverage range | Post-gap identity recovery range "
                "| Geometry-inconsistency range |"
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
        newline="\n",
    )
    output_paths["table_s5_controlled_translation_by_trial.md"].write_text(
        render_translation_markdown(trial_rows),
        encoding="utf-8",
        newline="\n",
    )

    source_paths = sorted(
        path
        for path in SOURCE_DATA_DIRECTORY.iterdir()
        if path.suffix in {".csv", ".json"}
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "builder_sha256": sha256_file(Path(__file__).resolve()),
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
        newline="\n",
    )
    return manifest


def _supplement_marker_bounds(
    supplement: str,
    *,
    allow_absent: bool,
) -> tuple[int, int] | None:
    begin_count = supplement.count(SUPPLEMENT_BEGIN_MARKER)
    end_count = supplement.count(SUPPLEMENT_END_MARKER)
    if begin_count == 0 and end_count == 0:
        if allow_absent:
            return None
        raise ValueError("Table S3 markers are missing from the supplement.")
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            "Supplement must contain exactly one ordered Table S3 marker pair."
        )
    begin = supplement.index(SUPPLEMENT_BEGIN_MARKER)
    end = supplement.index(SUPPLEMENT_END_MARKER)
    if end < begin:
        raise ValueError("Table S3 end marker precedes its begin marker.")
    return begin, end


def extract_supplement_block(supplement: str) -> str:
    bounds = _supplement_marker_bounds(supplement, allow_absent=False)
    assert bounds is not None
    begin, end = bounds
    content_start = begin + len(SUPPLEMENT_BEGIN_MARKER)
    return supplement[content_start:end].strip() + "\n"


def sync_supplement(markdown: str) -> None:
    supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
    bounds = _supplement_marker_bounds(supplement, allow_absent=True)
    block = (
        f"{SUPPLEMENT_BEGIN_MARKER}\n"
        f"{markdown.rstrip()}\n"
        f"{SUPPLEMENT_END_MARKER}"
    )
    if bounds is not None:
        begin, end = bounds
        before = supplement[:begin].rstrip()
        after = supplement[
            end + len(SUPPLEMENT_END_MARKER):
        ].lstrip()
        updated = f"{before}\n\n{block}\n"
        if after:
            updated += f"\n{after}"
    else:
        if SUPPLEMENT_HEADING in supplement:
            raise ValueError("Table S3 heading exists without generated markers.")
        updated = (
            supplement.rstrip()
            + f"\n\n{SUPPLEMENT_HEADING}\n\n"
            + block
            + "\n"
        )
    SUPPLEMENT_PATH.write_text(updated, encoding="utf-8", newline="\n")


def main() -> None:
    build_artifact()
    table_s3_path = OUTPUT_DIRECTORY / "table_s3_energy_scan_by_case.md"
    sync_supplement(table_s3_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
