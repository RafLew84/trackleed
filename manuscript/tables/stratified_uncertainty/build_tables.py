#!/usr/bin/env python3
"""Build stratified descriptive tables without treating images as replicates."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TABLE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"

SUPPLEMENT_S3_BEGIN_MARKER = "<!-- BEGIN TABLE S3 -->"
SUPPLEMENT_S3_END_MARKER = "<!-- END TABLE S3 -->"
SUPPLEMENT_S3_HEADING = (
    "## Table S3 — stratified Ag(111) energy-scan outcomes"
)

SCAN_LABELS = {
    "26a21": "Scan A",
    "26a21b": "Scan B",
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
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("Source provenance must list one frozen table.")
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
    if optional_original_count != 0:
        raise ValueError("The committed source original must be required.")
    return provenance


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


def build_artifact(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    """Write deterministic supplementary tables and their manifest."""

    validate_source_data()
    output_directory.mkdir(parents=True, exist_ok=True)
    case_rows = build_energy_scan_case_table()
    leave_one_out_rows = build_leave_one_case_out_table()

    output_paths = {
        "energy_scan_by_case.csv": (
            output_directory / "energy_scan_by_case.csv"
        ),
        "energy_scan_leave_one_case_out.csv": (
            output_directory / "energy_scan_leave_one_case_out.csv"
        ),
        "table_s3_energy_scan_by_case.md": (
            output_directory / "table_s3_energy_scan_by_case.md"
        ),
    }
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

    source_paths = sorted(
        path
        for path in SOURCE_DATA_DIRECTORY.iterdir()
        if path.suffix in {".csv", ".json"}
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "table_status": {
            "Table S3": "finalized",
        },
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "uncertainty_policy": {
            "inference": "descriptive; no confidence intervals or p-values",
            "replication_units": {
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


def _marker_bounds(
    supplement: str,
    *,
    begin_marker: str,
    end_marker: str,
    table_label: str,
    allow_absent: bool,
) -> tuple[int, int] | None:
    begin_count = supplement.count(begin_marker)
    end_count = supplement.count(end_marker)
    if begin_count == 0 and end_count == 0:
        if allow_absent:
            return None
        raise ValueError(f"{table_label} markers are missing from the supplement.")
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            f"Supplement must contain exactly one ordered {table_label} "
            "marker pair."
        )
    begin = supplement.index(begin_marker)
    end = supplement.index(end_marker)
    if end < begin:
        raise ValueError(f"{table_label} end marker precedes its begin marker.")
    return begin, end


def _extract_supplement_table(
    supplement: str,
    *,
    begin_marker: str,
    end_marker: str,
    table_label: str,
) -> str:
    bounds = _marker_bounds(
        supplement,
        begin_marker=begin_marker,
        end_marker=end_marker,
        table_label=table_label,
        allow_absent=False,
    )
    assert bounds is not None
    begin, end = bounds
    content_start = begin + len(begin_marker)
    return supplement[content_start:end].strip() + "\n"


def extract_supplement_s3_block(supplement: str) -> str:
    return _extract_supplement_table(
        supplement,
        begin_marker=SUPPLEMENT_S3_BEGIN_MARKER,
        end_marker=SUPPLEMENT_S3_END_MARKER,
        table_label="Table S3",
    )


def _sync_supplement_table(
    markdown: str,
    *,
    begin_marker: str,
    end_marker: str,
    heading: str,
    table_label: str,
) -> None:
    supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
    bounds = _marker_bounds(
        supplement,
        begin_marker=begin_marker,
        end_marker=end_marker,
        table_label=table_label,
        allow_absent=True,
    )
    block = (
        f"{begin_marker}\n"
        f"{markdown.rstrip()}\n"
        f"{end_marker}"
    )
    if bounds is not None:
        begin, end = bounds
        before = supplement[:begin].rstrip()
        after = supplement[
            end + len(end_marker):
        ].lstrip()
        updated = f"{before}\n\n{block}\n"
        if after:
            updated += f"\n{after}"
    else:
        if heading in supplement:
            raise ValueError(
                f"{table_label} heading exists without generated markers."
            )
        updated = (
            supplement.rstrip()
            + f"\n\n{heading}\n\n"
            + block
            + "\n"
        )
    SUPPLEMENT_PATH.write_text(updated, encoding="utf-8", newline="\n")


def sync_supplement_s3(markdown: str) -> None:
    _sync_supplement_table(
        markdown,
        begin_marker=SUPPLEMENT_S3_BEGIN_MARKER,
        end_marker=SUPPLEMENT_S3_END_MARKER,
        heading=SUPPLEMENT_S3_HEADING,
        table_label="Table S3",
    )


def main() -> None:
    build_artifact()
    table_s3_path = OUTPUT_DIRECTORY / "table_s3_energy_scan_by_case.md"
    sync_supplement_s3(table_s3_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
