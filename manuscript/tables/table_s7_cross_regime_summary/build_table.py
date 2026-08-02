#!/usr/bin/env python3
"""Build supplementary Table S7 without collapsing evidence roles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
SOURCE_MATRIX_PATH = SOURCE_DATA_DIRECTORY / "method_evidence_matrix.csv"
SOURCE_PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"

SUPPLEMENT_BEGIN_MARKER = "<!-- BEGIN TABLE S7 -->"
SUPPLEMENT_END_MARKER = "<!-- END TABLE S7 -->"

EXPECTED_METHODS = (
    "locotrack",
    "ostrack",
    "dam4sam",
    "lk",
    "easyleed",
)
REQUIRED_SOURCE_COLUMNS = {
    "method",
    "display_method",
    "cu_expected_measurements",
    "cu_valid_gaussian_fits",
    "cu_pair_1_6_valid",
    "cu_pair_3_5_valid",
    "cu_median_abs_endpoint_deviation_from_framewise_consensus_pp",
    "cu_max_abs_endpoint_deviation_from_framewise_consensus_pp",
    "translation_evaluation_spot_opportunities",
    "translation_native_proposals",
    "translation_valid_gaussian_measurements",
    "translation_measurement_error_median_px",
    "translation_measurement_error_p95_px",
    "translation_pair_opportunities",
    "translation_valid_pair_measurements",
    "translation_max_abs_delta_d_over_d_pp",
    "scane_visible_opportunities",
    "scane_branch_consistent_measurements",
    "scane_evaluated_gaps",
    "scane_recovered_gaps",
    "scane_accepted_measurements_all_frames",
    "scane_geometry_inconsistent_accepted_fits",
    "scane_jointly_consistent_pair_frames",
    "scane_conditional_median_abs_distance_residual_px",
    "interpretation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _count_with_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("Count outcome requires 0 <= numerator <= denominator.")
    return (
        f"{numerator}/{denominator} "
        f"({100.0 * numerator / denominator:.2f}%)"
    )


def load_and_validate_source(
    source_path: Path = SOURCE_MATRIX_PATH,
) -> list[dict[str, str]]:
    """Load the frozen five-method matrix and validate shared denominators."""

    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_SOURCE_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                "Evidence matrix is missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if tuple(row["method"] for row in rows) != EXPECTED_METHODS:
        raise ValueError("Evidence matrix method order changed.")
    if any(not row["display_method"].strip() for row in rows):
        raise ValueError("Every method requires a reader-facing label.")
    if any(_as_int(row, "cu_expected_measurements") != 2804 for row in rows):
        raise ValueError("Cu(111) measurement denominator changed.")
    if any(
        _as_int(row, "translation_evaluation_spot_opportunities") != 80
        for row in rows
    ):
        raise ValueError("Controlled-translation denominator changed.")
    if any(
        _as_int(row, "translation_pair_opportunities") != 40
        for row in rows
    ):
        raise ValueError("Controlled-translation pair denominator changed.")
    if any(
        _as_int(row, "scane_visible_opportunities") != 221
        for row in rows
    ):
        raise ValueError("Energy-scan visible-opportunity denominator changed.")
    if any(_as_int(row, "scane_evaluated_gaps") != 24 for row in rows):
        raise ValueError("Energy-scan gap denominator changed.")
    return rows


def build_blocks(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Format three regime-specific blocks from the common source rows."""

    controlled_translation: list[dict[str, str]] = []
    cu111_positive_control: list[dict[str, str]] = []
    energy_scan_stress_test: list[dict[str, str]] = []

    for row in rows:
        translation_opportunities = _as_int(
            row,
            "translation_evaluation_spot_opportunities",
        )
        pair_opportunities = _as_int(
            row,
            "translation_pair_opportunities",
        )
        controlled_translation.append(
            {
                "method": row["display_method"],
                "proposal_coverage": _count_with_rate(
                    _as_int(row, "translation_native_proposals"),
                    translation_opportunities,
                ),
                "measurement_coverage": _count_with_rate(
                    _as_int(row, "translation_valid_gaussian_measurements"),
                    translation_opportunities,
                ),
                "median_p95_displacement_error_px": (
                    f"{_as_float(row, 'translation_measurement_error_median_px'):.4f}"
                    " / "
                    f"{_as_float(row, 'translation_measurement_error_p95_px'):.4f}"
                ),
                "pair_coverage": _count_with_rate(
                    _as_int(row, "translation_valid_pair_measurements"),
                    pair_opportunities,
                ),
                "maximum_abs_delta_d_over_d_pp": (
                    f"{_as_float(row, 'translation_max_abs_delta_d_over_d_pp'):.6f}"
                ),
            }
        )

        cu_expected = _as_int(row, "cu_expected_measurements")
        cu_median_deviation = _as_float(
            row,
            "cu_median_abs_endpoint_deviation_from_framewise_consensus_pp",
        )
        cu_maximum_deviation = _as_float(
            row,
            "cu_max_abs_endpoint_deviation_from_framewise_consensus_pp",
        )
        cu111_positive_control.append(
            {
                "method": row["display_method"],
                "gaussian_measurement_coverage": _count_with_rate(
                    _as_int(row, "cu_valid_gaussian_fits"),
                    cu_expected,
                ),
                "valid_pair_endpoints": (
                    f"{_as_int(row, 'cu_pair_1_6_valid')}/701; "
                    f"{_as_int(row, 'cu_pair_3_5_valid')}/701"
                ),
                "median_max_abs_endpoint_deviation_pp": (
                    f"{cu_median_deviation:.6f} / "
                    f"{cu_maximum_deviation:.6f}"
                ),
            }
        )

        visible_opportunities = _as_int(
            row,
            "scane_visible_opportunities",
        )
        evaluated_gaps = _as_int(row, "scane_evaluated_gaps")
        accepted_measurements = _as_int(
            row,
            "scane_accepted_measurements_all_frames",
        )
        conditional_n = _as_int(
            row,
            "scane_jointly_consistent_pair_frames",
        )
        conditional_residual = _as_float(
            row,
            "scane_conditional_median_abs_distance_residual_px",
        )
        energy_scan_stress_test.append(
            {
                "method": row["display_method"],
                "measurement_coverage": _count_with_rate(
                    _as_int(row, "scane_branch_consistent_measurements"),
                    visible_opportunities,
                ),
                "branch_consistent_recovery": _count_with_rate(
                    _as_int(row, "scane_recovered_gaps"),
                    evaluated_gaps,
                ),
                "geometry_inconsistent_continuation": _count_with_rate(
                    _as_int(
                        row,
                        "scane_geometry_inconsistent_accepted_fits",
                    ),
                    accepted_measurements,
                ),
                "conditional_median_pair_distance_residual_px": (
                    f"{conditional_residual:.3f} (n={conditional_n})"
                ),
            }
        )

    return {
        "controlled_translation": controlled_translation,
        "cu111_positive_control": cu111_positive_control,
        "energy_scan_stress_test": energy_scan_stress_test,
    }


def _markdown_cell(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ").strip()


def _render_block(
    title: str,
    headers: tuple[tuple[str, str], ...],
    rows: list[dict[str, str]],
) -> list[str]:
    lines = [
        f"**{title}**",
        "",
        "| " + " | ".join(label for _, label in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row[source_name])
                for source_name, _ in headers
            )
            + " |"
        )
    return lines


def render_markdown(
    blocks: dict[str, list[dict[str, str]]],
) -> str:
    """Render one numbered table as three vertically stacked metric blocks."""

    lines = [
        (
            "**Table S7. Cross-regime numerical summary.** "
            "Each block answers a distinct validation question and retains "
            "its original denominators. The translation block uses 80 "
            "displaced-spot and 40 displaced-pair evaluation opportunities "
            "per method; the four initialization states reported in Table 1 "
            "are not displacement evaluations. Translation errors are "
            "conditional on accepted Gaussian measurements, with coverage "
            "shown first. The energy-scan residual is conditional on jointly "
            "branch-consistent pair frames and is shown with its n. Cu endpoint "
            "deviations are agreement with the framewise five-method median, "
            "not absolute accuracy. DAM4SAM used sam21pp-L for Cu(111) and "
            "controlled translations, and sam21pp-B for the energy scans. No "
            "composite score or rank is calculated."
        ),
        "",
    ]
    lines.extend(
        _render_block(
            "Controlled translation",
            (
                ("method", "Method"),
                ("proposal_coverage", "Proposal coverage"),
                ("measurement_coverage", "Measurement coverage"),
                (
                    "median_p95_displacement_error_px",
                    "Median / P95 displacement error [px]",
                ),
                ("pair_coverage", "Pair coverage"),
                (
                    "maximum_abs_delta_d_over_d_pp",
                    "Maximum absolute Delta D/D [pp]",
                ),
            ),
            blocks["controlled_translation"],
        )
    )
    lines.extend([""])
    lines.extend(
        _render_block(
            "Cu(111) positive control",
            (
                ("method", "Method"),
                (
                    "gaussian_measurement_coverage",
                    "Gaussian measurement coverage",
                ),
                (
                    "valid_pair_endpoints",
                    "Valid endpoints (pair 1-6; pair 3-5)",
                ),
                (
                    "median_max_abs_endpoint_deviation_pp",
                    "Median / maximum endpoint deviation [pp]",
                ),
            ),
            blocks["cu111_positive_control"],
        )
    )
    lines.extend([""])
    lines.extend(
        _render_block(
            "Independent Ag(111) energy-scan stress test",
            (
                ("method", "Method"),
                (
                    "measurement_coverage",
                    "Branch-consistent measurement coverage",
                ),
                (
                    "branch_consistent_recovery",
                    "Post-gap branch-consistent recovery",
                ),
                (
                    "geometry_inconsistent_continuation",
                    "Geometry-inconsistent continuation",
                ),
                (
                    "conditional_median_pair_distance_residual_px",
                    "Conditional median pair-distance residual [px] (n)",
                ),
            ),
            blocks["energy_scan_stress_test"],
        )
    )
    return "\n".join(lines) + "\n"


def extract_supplement_table(supplement: str) -> str:
    """Return the normalized generated block between Table S7 markers."""

    if supplement.count(SUPPLEMENT_BEGIN_MARKER) != 1:
        raise ValueError("Supplement must contain one Table S7 begin marker.")
    if supplement.count(SUPPLEMENT_END_MARKER) != 1:
        raise ValueError("Supplement must contain one Table S7 end marker.")
    start = supplement.index(SUPPLEMENT_BEGIN_MARKER) + len(
        SUPPLEMENT_BEGIN_MARKER
    )
    end = supplement.index(SUPPLEMENT_END_MARKER, start)
    return supplement[start:end].strip("\n") + "\n"


def validate_provenance() -> dict[str, object]:
    """Validate the committed matrix against its frozen SHA-256."""

    provenance = json.loads(
        SOURCE_PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    sources = provenance.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("Source provenance must list one frozen matrix.")
    source = sources[0]
    if set(source) != {
        "frozen_path",
        "frozen_sha256",
        "origin_path",
        "origin_sha256",
        "role",
    }:
        raise ValueError("Malformed source-provenance entry.")
    for digest_key in ("frozen_sha256", "origin_sha256"):
        digest = str(source[digest_key])
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("Malformed SHA-256 in source provenance.")
    frozen_path = (TABLE_ROOT / str(source["frozen_path"])).resolve()
    if TABLE_ROOT not in frozen_path.parents:
        raise ValueError("Frozen provenance path escapes the table artifact.")
    if not frozen_path.is_file():
        raise ValueError(f"Missing frozen source: {frozen_path.name}")
    if sha256_file(frozen_path) != source["frozen_sha256"]:
        raise ValueError(f"Frozen source hash changed: {frozen_path.name}")
    return provenance


def flatten_blocks(
    blocks: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Return one machine-readable row per method with regime prefixes."""

    output: list[dict[str, str]] = []
    for translation, cu, energy_scan in zip(
        blocks["controlled_translation"],
        blocks["cu111_positive_control"],
        blocks["energy_scan_stress_test"],
        strict=True,
    ):
        if not (
            translation["method"] == cu["method"] == energy_scan["method"]
        ):
            raise ValueError("Block method order is inconsistent.")
        output.append(
            {
                "Method": translation["method"],
                "Translation proposal coverage": (
                    translation["proposal_coverage"]
                ),
                "Translation measurement coverage": (
                    translation["measurement_coverage"]
                ),
                "Translation median / P95 displacement error [px]": (
                    translation["median_p95_displacement_error_px"]
                ),
                "Translation pair coverage": translation["pair_coverage"],
                "Translation maximum absolute Delta D/D [pp]": (
                    translation["maximum_abs_delta_d_over_d_pp"]
                ),
                "Cu Gaussian measurement coverage": (
                    cu["gaussian_measurement_coverage"]
                ),
                "Cu valid endpoints (pair 1-6; pair 3-5)": (
                    cu["valid_pair_endpoints"]
                ),
                "Cu median / maximum endpoint deviation [pp]": (
                    cu["median_max_abs_endpoint_deviation_pp"]
                ),
                "Energy-scan branch-consistent measurement coverage": (
                    energy_scan["measurement_coverage"]
                ),
                "Energy-scan post-gap branch-consistent recovery": (
                    energy_scan["branch_consistent_recovery"]
                ),
                "Energy-scan geometry-inconsistent continuation": (
                    energy_scan["geometry_inconsistent_continuation"]
                ),
                "Energy-scan conditional median residual [px] (n)": (
                    energy_scan[
                        "conditional_median_pair_distance_residual_px"
                    ]
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty Table S7.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_artifact(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    """Write deterministic CSV/Markdown outputs and their manifest."""

    provenance = validate_provenance()
    rows = load_and_validate_source()
    blocks = build_blocks(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "table_s7_cross_regime_summary.csv": (
            output_directory / "table_s7_cross_regime_summary.csv"
        ),
        "table_s7_cross_regime_summary.md": (
            output_directory / "table_s7_cross_regime_summary.md"
        ),
    }
    _write_csv(
        output_paths["table_s7_cross_regime_summary.csv"],
        flatten_blocks(blocks),
    )
    output_paths["table_s7_cross_regime_summary.md"].write_text(
        render_markdown(blocks),
        encoding="utf-8",
        newline="\n",
    )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "table_contract": {
            "placement": "supplementary",
            "table": "Table S7",
            "method_count": 5,
            "block_count": 3,
            "translation_displaced_spot_opportunities_per_method": 80,
            "translation_displaced_pair_opportunities_per_method": 40,
            "translation_initialization_states_are_not_evaluations": True,
            "ranking_policy": (
                "no composite score, rank, or rank-like color scale"
            ),
            "conditional_residual_keeps_pair_frame_n": True,
            "dam4sam_cross_regime_variant_change_disclosed": True,
        },
        "source_data": {
            path.relative_to(TABLE_ROOT).as_posix(): sha256_file(path)
            for path in (SOURCE_MATRIX_PATH, SOURCE_PROVENANCE_PATH)
        },
        "upstream_sources": provenance["sources"],
        "builder_sha256": sha256_file(Path(__file__)),
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


if __name__ == "__main__":
    build_artifact()
