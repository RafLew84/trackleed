#!/usr/bin/env python3
"""Build supplementary Table S2 from frozen gate and denominator records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TABLE_ROOT.parents[2]
SOURCE_DIRECTORY = TABLE_ROOT / "source_data"
GATES_PATH = SOURCE_DIRECTORY / "gate_records.csv"
DENOMINATORS_PATH = SOURCE_DIRECTORY / "denominator_records.csv"
PROVENANCE_PATH = SOURCE_DIRECTORY / "source_provenance.json"
UPSTREAM_DIRECTORY = SOURCE_DIRECTORY / "upstream_contracts"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"

SUPPLEMENT_BEGIN_MARKER = "<!-- BEGIN TABLE S2 -->"
SUPPLEMENT_END_MARKER = "<!-- END TABLE S2 -->"
SUPPLEMENT_HEADING = (
    "## Table S2 — Gaussian gates, fixed opportunity sets and conditional "
    "denominators"
)

GATE_COLUMNS = (
    "order",
    "regime",
    "stage",
    "frozen_criterion",
    "boundary_and_role",
    "failure_handling",
)
DENOMINATOR_COLUMNS = (
    "order",
    "regime",
    "metric",
    "denominator_type",
    "denominator_per_method",
    "unit",
    "construction",
    "replication_or_stratification",
    "method_specific_n",
    "scope_note",
)
FIXED_DENOMINATOR_TYPES = {
    "Fixed opportunity",
    "Fixed eligibility window",
    "Separate diagnostic",
    "Audit sample",
}
CONDITIONAL_DENOMINATOR_TYPES = {
    "Conditional available proposals",
    "Conditional accepted measurements",
    "Conditional jointly valid pairs",
    "Conditional accepted fits",
    "Conditional branch-consistent pairs",
    "Conditional paired native outputs",
    "Conditional jointly accepted fits",
}

MAIN_METHOD_ORDER = (
    "LocoTrack",
    "OSTrack",
    "DAM4SAM",
    "Lucas–Kanade",
    "EasyLEED",
)
FAMILY_METHOD_ORDER = ("SAM2", "SAMURAI", "DAM4SAM")

GATE_MARKDOWN_COLUMNS = (
    ("regime", "Regime"),
    ("stage", "Stage"),
    ("frozen_criterion", "Frozen criterion"),
    ("boundary_and_role", "Boundary and role"),
    ("failure_handling", "Failure handling"),
)
FIXED_MARKDOWN_COLUMNS = (
    ("regime", "Regime"),
    ("metric", "Metric"),
    ("denominator_type", "Denominator class"),
    ("denominator_per_method", "Denominator per method"),
    ("unit", "Unit"),
    ("construction", "Construction"),
    ("replication_or_stratification", "Replication/stratification"),
    ("scope_note", "Scope note"),
)
CONDITIONAL_MARKDOWN_COLUMNS = (
    ("regime", "Regime"),
    ("metric", "Metric"),
    ("denominator_type", "Denominator class"),
    ("unit", "Unit"),
    ("method_specific_n", "Method-specific or paired n"),
    ("construction", "Conditioning basis"),
    ("scope_note", "Conditioning rule"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf_normalized_text(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _validate_sha256(value: object, *, field: str) -> None:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Malformed SHA-256 in {field}: {digest}")


def _validate_git_commit(value: object) -> None:
    commit = str(value)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"Malformed Git commit: {commit}")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_contract_csv(
    path: Path,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"Unexpected columns in {path.name}.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No records in {path.name}.")
    if [row["order"] for row in rows] != [
        str(index) for index in range(1, len(rows) + 1)
    ]:
        raise ValueError(f"Nonsequential order in {path.name}.")
    for row in rows:
        if any(value == "" for key, value in row.items() if key != "method_specific_n"):
            raise ValueError(f"Incomplete record {row['order']} in {path.name}.")
    return rows


def load_and_validate_gates() -> list[dict[str, str]]:
    rows = _read_contract_csv(GATES_PATH, GATE_COLUMNS)
    if len(rows) != 10:
        raise ValueError("Table S2 must contain ten gate-contract rows.")
    if {row["regime"] for row in rows} != {
        "All experiments",
        "Cu(111) and controlled translations",
        "Energy scans",
        "Cu(111) primary pairs",
    }:
        raise ValueError("Unexpected gate regimes.")
    if sum(row["stage"] == "Branch consistency" for row in rows) != 1:
        raise ValueError("Branch consistency must be listed exactly once.")
    return rows


def load_and_validate_denominators() -> list[dict[str, str]]:
    rows = _read_contract_csv(DENOMINATORS_PATH, DENOMINATOR_COLUMNS)
    if len(rows) != 22:
        raise ValueError("Table S2 must contain 22 denominator records.")
    types = {row["denominator_type"] for row in rows}
    if types != FIXED_DENOMINATOR_TYPES | CONDITIONAL_DENOMINATOR_TYPES:
        raise ValueError("Unexpected denominator class.")
    fixed_rows, conditional_rows = partition_denominators(rows)
    if len(fixed_rows) != 13 or len(conditional_rows) != 9:
        raise ValueError("Expected 13 fixed/audit and nine conditional rows.")
    if any(row["method_specific_n"] for row in fixed_rows):
        raise ValueError("Fixed denominator rows must not contain conditional n.")
    if any(not row["method_specific_n"] for row in conditional_rows):
        raise ValueError("Conditional rows must record method-specific n.")
    return rows


def partition_denominators(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fixed_rows = [
        row
        for row in rows
        if row["denominator_type"] in FIXED_DENOMINATOR_TYPES
    ]
    conditional_rows = [
        row
        for row in rows
        if row["denominator_type"] in CONDITIONAL_DENOMINATOR_TYPES
    ]
    return fixed_rows, conditional_rows


def validate_provenance() -> dict[str, object]:
    loaded = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or set(loaded) != {"sources"}:
        raise ValueError("Malformed source provenance.")
    sources = loaded["sources"]
    if not isinstance(sources, list) or len(sources) != 10:
        raise ValueError("Source provenance must list ten records.")
    required_keys = {
        "frozen_path",
        "frozen_sha256",
        "origin_path",
        "origin_sha256",
        "role",
    }
    optional_keys = {
        "origin_commit",
        "origin_required_at_build",
        "origin_lf_sha256",
    }
    optional_run_workspace_count = 0
    historical_origin_count = 0
    for source in sources:
        if (
            not isinstance(source, dict)
            or not required_keys.issubset(source)
            or not set(source).issubset(required_keys | optional_keys)
        ):
            raise ValueError("Malformed provenance source.")
        if (
            "origin_required_at_build" in source
            and not isinstance(source["origin_required_at_build"], bool)
        ):
            raise ValueError("origin_required_at_build must be Boolean.")
        for text_key in ("frozen_path", "origin_path", "role"):
            if (
                not isinstance(source[text_key], str)
                or not source[text_key].strip()
            ):
                raise ValueError(f"Malformed provenance field: {text_key}")
        origin_required = source.get("origin_required_at_build", True)
        if not origin_required:
            origin_parts = Path(str(source["origin_path"])).parts
            if origin_parts and origin_parts[0] == ".scratch":
                if "origin_commit" in source:
                    raise ValueError(
                        "Run-workspace origins cannot declare origin_commit."
                    )
                optional_run_workspace_count += 1
            else:
                origin_commit = source.get("origin_commit")
                if not isinstance(origin_commit, str):
                    raise ValueError(
                        "Historical committed origins require origin_commit."
                    )
                _validate_git_commit(origin_commit)
                if source.get("origin_lf_sha256") != source["frozen_sha256"]:
                    raise ValueError(
                        "Historical origin must match the frozen LF snapshot."
                    )
                historical_origin_count += 1
        elif "origin_commit" in source:
            raise ValueError(
                "origin_commit is only valid for historical optional origins."
            )
        _validate_sha256(source["frozen_sha256"], field="frozen_sha256")
        _validate_sha256(source["origin_sha256"], field="origin_sha256")
        if "origin_lf_sha256" in source:
            _validate_sha256(
                source["origin_lf_sha256"],
                field="origin_lf_sha256",
            )
        frozen_relative = str(source["frozen_path"])
        if ".scratch" in frozen_relative:
            raise ValueError("Frozen evidence must not depend on .scratch.")
        frozen_path = (TABLE_ROOT / frozen_relative).resolve()
        if TABLE_ROOT not in frozen_path.parents:
            raise ValueError("Frozen provenance path escapes the artifact.")
        if not frozen_path.is_file():
            raise ValueError(f"Missing frozen source: {frozen_relative}")
        if sha256_file(frozen_path) != source["frozen_sha256"]:
            raise ValueError(f"Frozen-source hash mismatch: {frozen_relative}")
        origin_path = (
            REPOSITORY_ROOT / str(source["origin_path"])
        ).resolve()
        if (
            origin_path != REPOSITORY_ROOT
            and REPOSITORY_ROOT not in origin_path.parents
        ):
            raise ValueError("Origin provenance path escapes the repository.")
        if not origin_path.is_file():
            if source.get("origin_required_at_build", True):
                raise ValueError(
                    f"Missing source origin: {source['origin_path']}"
                )
            continue
        if "origin_lf_sha256" in source:
            origin_digest = sha256_lf_normalized_text(origin_path)
            expected_origin_digest = source["origin_lf_sha256"]
        else:
            origin_digest = sha256_file(origin_path)
            expected_origin_digest = source["origin_sha256"]
        if origin_digest != expected_origin_digest and "origin_commit" not in source:
            raise ValueError(f"Origin hash mismatch: {source['origin_path']}")
    if optional_run_workspace_count != 2:
        raise ValueError("Exactly two run-workspace origins must be optional.")
    if historical_origin_count != 1:
        raise ValueError("Exactly one committed origin must be history-pinned.")
    return loaded


def _denominator_by_metric(
    rows: list[dict[str, str]],
    regime: str,
    metric: str,
) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["regime"] == regime and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one denominator record for {regime}: {metric}")
    return matches[0]


def _format_method_counts(
    rows: list[dict[str, str]],
    column: str,
) -> str:
    return "; ".join(
        f"{row['display_method']}={row[column]}" for row in rows
    )


def validate_upstream_semantics(
    denominators: list[dict[str, str]],
) -> None:
    cu_source = (
        UPSTREAM_DIRECTORY / "executed_sources" / "cu_measurement_runner.py"
    ).read_text(encoding="utf-8")
    for token in (
        "ROI_SIZE = 48",
        'if diagnostics["r_squared"] < 0.85:',
        'if diagnostics["amplitude"] < 0.05:',
        "if not (-2.0 < local_x < ROI_SIZE + 2.0):",
        'if not (0.5 < diagnostics["sigma_x"] < ROI_SIZE):',
    ):
        if token not in cu_source:
            raise ValueError(f"Cu gate token missing from executed source: {token}")

    energy_source = (
        UPSTREAM_DIRECTORY / "executed_sources" / "energy_measurement_runner.py"
    ).read_text(encoding="utf-8")
    for token in (
        "ROI_SIZE = 48",
        "R2_MIN = 0.85",
        "AMPLITUDE_MIN = 0.012",
        "SIGMA_MIN = 0.6",
        "SIGMA_MAX = 13.0",
        "CENTER_TOLERANCE_PX = 15.0",
        "BRANCH_TOLERANCE_PX = 18.0",
        "center_preference_factor=0.34",
    ):
        if token not in energy_source:
            raise ValueError(
                f"Energy gate token missing from executed source: {token}"
            )

    peak_source = (
        UPSTREAM_DIRECTORY / "executed_sources" / "peak_analyzer.py"
    ).read_text(encoding="utf-8")
    if "radius_px = max(1.0, float(center_preference_factor)" not in peak_source:
        raise ValueError("Preferred-center radius implementation is missing.")

    main = read_csv_dicts(UPSTREAM_DIRECTORY / "method_evidence_matrix.csv")
    if tuple(row["display_method"] for row in main) != MAIN_METHOD_ORDER:
        raise ValueError("Unexpected five-method order.")
    if {row["scane_visible_opportunities"] for row in main} != {"221"}:
        raise ValueError("Independent energy coverage denominator changed.")
    if {row["scane_evaluated_gaps"] for row in main} != {"24"}:
        raise ValueError("Independent energy gap denominator changed.")

    translations = read_csv_dicts(
        UPSTREAM_DIRECTORY / "controlled_translation_method_metrics.csv"
    )
    if translations[-1]["display_method"] != "Oracle ROI":
        raise ValueError("Controlled-translation oracle record is missing.")
    if (
        translations[-1]["valid_gaussian_measurements"],
        translations[-1]["valid_pair_measurements"],
    ) != ("80", "40"):
        raise ValueError("Controlled-translation oracle counts changed.")

    family = read_csv_dicts(UPSTREAM_DIRECTORY / "family_aggregate.csv")
    if tuple(row["display_method"] for row in family) != FAMILY_METHOD_ORDER:
        raise ValueError("Unexpected controlled-family method order.")
    if {row["visible_opportunities"] for row in family} != {"182"}:
        raise ValueError("Controlled-family coverage denominator changed.")
    if {row["evaluated_gaps"] for row in family} != {"13"}:
        raise ValueError("Controlled-family gap denominator changed.")
    family_by_pair = read_csv_dicts(
        UPSTREAM_DIRECTORY / "family_by_pair.csv"
    )
    pair_counts = {
        (row["display_method"], row["pair"]): row[
            "jointly_consistent_pair_frames"
        ]
        for row in family_by_pair
    }
    expected_family_pair_keys = {
        (method, pair)
        for method in FAMILY_METHOD_ORDER
        for pair in ("pair 1", "pair 2")
    }
    if set(pair_counts) != expected_family_pair_keys:
        raise ValueError("Controlled-family method/pair contract changed.")
    family_totals = {
        row["display_method"]: row["jointly_consistent_pair_frames"]
        for row in family
    }
    if any(
        int(pair_counts[(method, "pair 1")])
        + int(pair_counts[(method, "pair 2")])
        != int(family_totals[method])
        for method in FAMILY_METHOD_ORDER
    ):
        raise ValueError("Controlled-family pair counts do not sum to totals.")

    external = json.loads(
        (UPSTREAM_DIRECTORY / "external_reference_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if external["branch_opportunities_per_method"] != 60:
        raise ValueError("External-reference opportunity denominator changed.")
    if external["valid_common_fits_per_method"] != 38:
        raise ValueError("External-reference common-fit count changed.")
    if external["branch_gap_checks"] != "4/4":
        raise ValueError("External-reference gap-check count changed.")

    cu_contract = json.loads(
        (UPSTREAM_DIRECTORY / "five_method_run_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        cu_contract["frame_count"],
        len(cu_contract["spot_ids"]),
        tuple(cu_contract["pairs"]),
    ) != (701, 4, ("1-6", "3-5")):
        raise ValueError("Cu(111) run denominator contract changed.")
    if set(cu_contract["valid_gaussian_fits_per_method"].values()) != {2804}:
        raise ValueError("Cu(111) accepted-fit count changed.")

    diagnostic = read_csv_dicts(
        UPSTREAM_DIRECTORY / "pair_2_4_missingness_diagnostic.csv"
    )
    rejected = [row for row in diagnostic if row["refit_accepted"] == "false"]
    if len(rejected) != 24 or not all(
        row["tracking_roi_present"] == "true"
        and row["r_squared_gate"] == "true"
        and row["amplitude_gate"] == "false"
        and row["center_gate"] == "true"
        and row["sigma_gate"] == "true"
        for row in rejected
    ):
        raise ValueError("Pair 2–4 amplitude-only diagnostic changed.")

    oracle = translations[-1]
    family_pair_mapping = "; ".join(
        (
            f"{method}: pair 1={pair_counts[(method, 'pair 1')]}, "
            f"pair 2={pair_counts[(method, 'pair 2')]}"
        )
        for method in FAMILY_METHOD_ORDER
    )
    expected_contract_rows = {
        (
            "Controlled translations",
            "Native-proposal displacement error",
        ): _format_method_counts(
            main,
            "translation_native_proposals",
        ),
        (
            "Controlled translations",
            "Gaussian-refined displacement error",
        ): (
            _format_method_counts(
                main,
                "translation_valid_gaussian_measurements",
            )
            + f"; Oracle ROI={oracle['valid_gaussian_measurements']}"
        ),
        (
            "Controlled translations",
            "Pair-invariance error",
        ): (
            _format_method_counts(
                main,
                "translation_valid_pair_measurements",
            )
            + f"; Oracle ROI={oracle['valid_pair_measurements']}"
        ),
        (
            "Independent energy scans",
            "Geometry-inconsistent continuation",
        ): _format_method_counts(
            main,
            "scane_accepted_measurements_all_frames",
        ),
        (
            "Independent energy scans",
            "Absolute pair-distance residual",
        ): _format_method_counts(
            main,
            "scane_jointly_consistent_pair_frames",
        ),
        (
            "Controlled SAM2 family",
            "Geometry-inconsistent continuation",
        ): _format_method_counts(
            family,
            "accepted_measurements_all_frames",
        ),
        (
            "Controlled SAM2 family",
            "Pair-stratified absolute pair-distance residual",
        ): family_pair_mapping,
        (
            "Restricted external reference",
            "Native cross-method coordinate agreement",
        ): (
            "EasyLEED–ViPErLEED="
            f"{external['cross_method_contract_agreement_px']['native_proposal_pairs']} "
            "paired correspondences"
        ),
        (
            "Restricted external reference",
            "Common-Gaussian center agreement",
        ): (
            "EasyLEED–ViPErLEED="
            f"{external['cross_method_contract_agreement_px']['jointly_valid_common_centers']} "
            "jointly accepted centers"
        ),
    }
    for key, expected in expected_contract_rows.items():
        row = _denominator_by_metric(denominators, *key)
        if row["method_specific_n"] != expected:
            raise ValueError(f"Normalized denominator does not match source: {key}")


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _render_table(
    rows: list[dict[str, str]],
    columns: tuple[tuple[str, str], ...],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| "
        + " | ".join(_markdown_cell(row[key]) for key, _ in columns)
        + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def render_markdown(
    gates: list[dict[str, str]],
    denominators: list[dict[str, str]],
) -> str:
    fixed, conditional = partition_denominators(denominators)
    caption = (
        "**Table S2. Gaussian gates, fixed opportunity sets and conditional "
        "denominators.** Block A lists the run-frozen acceptance contract and "
        "separates Gaussian acceptance from the subsequent energy-branch "
        "consistency classification. Block B retains every predefined "
        "opportunity in its fixed denominator, including unavailable proposals "
        "and rejected fits. Block C gives the method-dependent n used only for "
        "statistics conditional on available native outputs, accepted Gaussian "
        "measurements, or jointly branch-consistent pairs. Counts pool repeated "
        "observations of propagated trajectories; images, spot opportunities, "
        "pair images, and gaps are not statistically independent replicates."
    )
    return "\n\n".join(
        (
            caption,
            "**A. Gaussian measurement and evaluation gates**\n\n"
            + _render_table(gates, GATE_MARKDOWN_COLUMNS),
            "**B. Fixed opportunity denominators**\n\n"
            + _render_table(fixed, FIXED_MARKDOWN_COLUMNS),
            "**C. Method-dependent conditional denominators**\n\n"
            + _render_table(conditional, CONDITIONAL_MARKDOWN_COLUMNS),
        )
    ) + "\n"


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def _source_hashes(provenance: dict[str, object]) -> dict[str, str]:
    sources = {
        str(GATES_PATH.relative_to(TABLE_ROOT)): sha256_file(GATES_PATH),
        str(DENOMINATORS_PATH.relative_to(TABLE_ROOT)): sha256_file(
            DENOMINATORS_PATH
        ),
        str(PROVENANCE_PATH.relative_to(TABLE_ROOT)): sha256_file(
            PROVENANCE_PATH
        ),
        str(Path(__file__).resolve().relative_to(TABLE_ROOT)): sha256_file(
            Path(__file__).resolve()
        ),
    }
    for source in provenance["sources"]:
        frozen_path = TABLE_ROOT / str(source["frozen_path"])
        sources[str(frozen_path.relative_to(TABLE_ROOT))] = sha256_file(
            frozen_path
        )
    return dict(sorted(sources.items()))


def build_artifact(
    *,
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, str]:
    gates = load_and_validate_gates()
    denominators = load_and_validate_denominators()
    provenance = validate_provenance()
    validate_upstream_semantics(denominators)

    output_directory.mkdir(parents=True, exist_ok=True)
    fixed, conditional = partition_denominators(denominators)
    paths = {
        "table_s2_gaussian_gates.csv": output_directory
        / "table_s2_gaussian_gates.csv",
        "table_s2_fixed_denominators.csv": output_directory
        / "table_s2_fixed_denominators.csv",
        "table_s2_conditional_denominators.csv": output_directory
        / "table_s2_conditional_denominators.csv",
        "table_s2_gaussian_gates_validation_denominators.md": output_directory
        / "table_s2_gaussian_gates_validation_denominators.md",
    }
    _write_csv(paths["table_s2_gaussian_gates.csv"], gates, GATE_COLUMNS)
    _write_csv(
        paths["table_s2_fixed_denominators.csv"],
        fixed,
        DENOMINATOR_COLUMNS,
    )
    _write_csv(
        paths["table_s2_conditional_denominators.csv"],
        conditional,
        DENOMINATOR_COLUMNS,
    )
    paths["table_s2_gaussian_gates_validation_denominators.md"].write_text(
        render_markdown(gates, denominators),
        encoding="utf-8",
        newline="\n",
    )

    output_hashes = {
        name: sha256_file(path) for name, path in sorted(paths.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact": "Supplementary Table S2",
        "title": (
            "Gaussian gates, fixed opportunity sets and conditional "
            "denominators"
        ),
        "sources": _source_hashes(provenance),
        "outputs": output_hashes,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_hashes


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
        raise ValueError("Table S2 markers are missing from the supplement.")
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            "Supplement must contain exactly one ordered Table S2 marker pair."
        )
    begin = supplement.index(SUPPLEMENT_BEGIN_MARKER)
    end = supplement.index(SUPPLEMENT_END_MARKER)
    if end < begin:
        raise ValueError("Table S2 end marker precedes its begin marker.")
    return begin, end


def extract_supplement_block(supplement: str) -> str:
    bounds = _supplement_marker_bounds(supplement, allow_absent=False)
    assert bounds is not None
    begin, end = bounds
    content_start = begin + len(SUPPLEMENT_BEGIN_MARKER)
    content = supplement[content_start:end]
    return content.strip() + "\n"


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
            raise ValueError("Table S2 heading exists without generated markers.")
        updated = (
            supplement.rstrip()
            + f"\n\n{SUPPLEMENT_HEADING}\n\n"
            + block
            + "\n"
        )
    SUPPLEMENT_PATH.write_text(updated, encoding="utf-8", newline="\n")


def main() -> None:
    build_artifact()
    markdown_path = (
        OUTPUT_DIRECTORY
        / "table_s2_gaussian_gates_validation_denominators.md"
    )
    sync_supplement(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
