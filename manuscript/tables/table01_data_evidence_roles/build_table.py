#!/usr/bin/env python3
"""Build main Table 1 from a frozen, reader-facing data-regime contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
SOURCE_TABLE_PATH = SOURCE_DATA_DIRECTORY / "data_regimes.csv"
SOURCE_PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
FROZEN_SOURCE_DIRECTORY = SOURCE_DATA_DIRECTORY / "upstream_contracts"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"

MANUSCRIPT_BEGIN_MARKER = "<!-- BEGIN TABLE 1 -->"
MANUSCRIPT_END_MARKER = "<!-- END TABLE 1 -->"

SOURCE_COLUMNS = (
    "order",
    "row_type",
    "scientific_series_label",
    "control_variable_and_range",
    "image_count",
    "frozen_spots_or_pairs",
    "opportunity_set",
    "evidence_role",
    "primary_observable",
    "independence_and_reporting_unit",
)
DISPLAY_COLUMNS = (
    ("scientific_series_label", "Scientific series label"),
    ("control_variable_and_range", "Control variable/range"),
    ("image_count", "Image count"),
    ("frozen_spots_or_pairs", "Frozen spots/pairs"),
    ("opportunity_set", "Opportunity set"),
    ("evidence_role", "Evidence role"),
    ("primary_observable", "Primary observable"),
    (
        "independence_and_reporting_unit",
        "Independence/replication unit",
    ),
)
EXPECTED_ROW_TYPES = (
    "primary_regime",
    "derived_regime",
    "primary_regime",
    "primary_regime",
    "subset",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return loaded


def _expected_opportunity_sets() -> tuple[str, ...]:
    """Derive reader-facing opportunity sets from frozen upstream contracts."""

    cu = _load_json(
        FROZEN_SOURCE_DIRECTORY / "cu111_positive_control_manifest.json"
    )
    workload = cu.get("workload")
    methods = cu.get("main_methods")
    if not isinstance(workload, dict) or not isinstance(methods, list):
        raise ValueError("Cu opportunity contract is missing.")
    method_count = len(methods)
    spot_rows = int(workload["automatic_gaussian_measurement_rows"])
    pair_rows = int(workload["automatic_pair_endpoint_rows"])

    translation = _load_json(
        FROZEN_SOURCE_DIRECTORY / "controlled_translation_contract.json"
    )
    trials = translation.get("trials")
    states = translation.get("translations_original_px")
    spots = translation.get("spots")
    pairs = translation.get("pairs")
    if not all(
        isinstance(value, list) for value in (trials, states, spots, pairs)
    ):
        raise ValueError("Controlled-translation opportunity contract is missing.")
    trial_count = len(trials)
    state_count = len(states)
    translation_spots = trial_count * state_count * len(spots)
    translation_pairs = trial_count * state_count * len(pairs)

    energy = _load_json(
        FROZEN_SOURCE_DIRECTORY / "independent_energy_scan_summary.json"
    )
    aggregate_methods = energy.get("aggregate_methods")
    if not isinstance(aggregate_methods, dict):
        raise ValueError("Energy-scan opportunity contract is missing.")
    energy_counts = {
        (int(record["visible_opportunities"]), int(record["evaluated_gaps"]))
        for record in aggregate_methods.values()
        if isinstance(record, dict)
    }
    if len(energy_counts) != 1:
        raise ValueError("Energy-scan methods do not share one opportunity set.")
    energy_opportunities, energy_gaps = next(iter(energy_counts))

    family = _load_json(
        FROZEN_SOURCE_DIRECTORY / "sam2_family_contract.json"
    ).get("controlled_family")
    if not isinstance(family, dict):
        raise ValueError("SAM2-family opportunity contract is missing.")

    external = _load_json(
        FROZEN_SOURCE_DIRECTORY / "external_reference_summary.json"
    )
    gap_checks = str(external["branch_gap_checks"]).split("/", maxsplit=1)[0]

    return (
        (
            f"{spot_rows // method_count:,} spot and "
            f"{pair_rows // method_count:,} pair opportunities per method "
            f"({spot_rows:,} spot-method rows total)"
        ),
        (
            f"{translation_spots} spot and {translation_pairs} pair "
            f"evaluations per method ({trial_count} trials x "
            f"{state_count} states)"
        ),
        (
            f"{energy_opportunities} baseline-visible branch opportunities "
            f"and {energy_gaps} evaluable gaps per method"
        ),
        (
            f"{int(family['visible_opportunities_per_method'])} "
            "baseline-visible branch opportunities and "
            f"{int(family['evaluated_gaps_per_method'])} "
            "evaluable gaps per method"
        ),
        (
            f"{int(external['branch_opportunities_per_method'])} frozen branch "
            f"opportunities per method; {gap_checks} gap checks"
        ),
    )


def load_and_validate_rows(
    source_path: Path = SOURCE_TABLE_PATH,
) -> list[dict[str, str]]:
    """Load the frozen rows and enforce their evidence-role semantics."""

    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("Unexpected source-table columns.")
        rows = list(reader)

    if len(rows) != 5:
        raise ValueError("Table 1 must contain exactly five evidence roles.")
    if [int(row["order"]) for row in rows] != list(range(1, 6)):
        raise ValueError("Table 1 rows must retain the frozen order 1-5.")
    if tuple(row["row_type"] for row in rows) != EXPECTED_ROW_TYPES:
        raise ValueError("Table 1 row types do not match the frozen contract.")
    if (
        tuple(row["opportunity_set"] for row in rows)
        != _expected_opportunity_sets()
    ):
        raise ValueError("Table 1 opportunity sets changed from frozen contracts.")
    if rows[1]["scientific_series_label"] != "Controlled translations":
        raise ValueError("The derived translation regime is missing.")
    if "not four acquisitions" not in rows[1][
        "independence_and_reporting_unit"
    ]:
        raise ValueError("Translation trials must not imply four acquisitions.")
    if "not an independent replicate" not in rows[4][
        "independence_and_reporting_unit"
    ]:
        raise ValueError("The external reference must remain a subset.")

    reader_facing_text = "\n".join(
        value
        for row in rows
        for key, value in row.items()
        if key not in {"order", "row_type"}
    )
    prohibited_archive_terms = ("ScanE", "26a21", "26a24")
    if any(term in reader_facing_text for term in prohibited_archive_terms):
        raise ValueError(
            "Reader-facing Table 1 contains an internal archive identifier."
        )
    return rows


def _markdown_cell(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ").strip()


def display_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove internal contract columns from the publication table."""

    return [
        {
            display_name: row[source_name]
            for source_name, display_name in DISPLAY_COLUMNS
        }
        for row in rows
    ]


def render_markdown(rows: list[dict[str, str]]) -> str:
    """Render the exact Markdown block copied into the manuscript."""

    displayed = display_rows(rows)
    headers = [display_name for _, display_name in DISPLAY_COLUMNS]
    lines = [
        (
            "**Table 1. Data and evidence roles.** Rows describe evidence "
            "roles, not five independent datasets. Controlled translations "
            "are derived from the Cu(111) sequence, and the external-reference "
            "row is a subset of the memory-analysis scan and outside the "
            "common five-method matrix. Outcomes remain separate rather than "
            "being reduced to a composite tracker ranking."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in displayed:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(str(row[header])) for header in headers)
            + " |"
        )
    return "\n".join(lines) + "\n"


def extract_manuscript_table(manuscript: str) -> str:
    """Return the normalized generated block between Table 1 markers."""

    if manuscript.count(MANUSCRIPT_BEGIN_MARKER) != 1:
        raise ValueError("Manuscript must contain one Table 1 begin marker.")
    if manuscript.count(MANUSCRIPT_END_MARKER) != 1:
        raise ValueError("Manuscript must contain one Table 1 end marker.")
    start = manuscript.index(MANUSCRIPT_BEGIN_MARKER) + len(
        MANUSCRIPT_BEGIN_MARKER
    )
    end = manuscript.index(MANUSCRIPT_END_MARKER, start)
    return manuscript[start:end].strip("\n") + "\n"


def validate_provenance() -> dict[str, object]:
    """Validate the hashes of the committed upstream contract snapshots."""

    provenance = json.loads(
        SOURCE_PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    sources = provenance.get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        raise ValueError("Source provenance must list six upstream contracts.")
    for source in sources:
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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[display_name for _, display_name in DISPLAY_COLUMNS],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(display_rows(rows))


def build_artifact(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    """Write deterministic CSV/Markdown outputs and their manifest."""

    rows = load_and_validate_rows()
    provenance = validate_provenance()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "table01_data_evidence_roles.csv": (
            output_directory / "table01_data_evidence_roles.csv"
        ),
        "table01_data_evidence_roles.md": (
            output_directory / "table01_data_evidence_roles.md"
        ),
    }
    _write_csv(output_paths["table01_data_evidence_roles.csv"], rows)
    output_paths["table01_data_evidence_roles.md"].write_text(
        render_markdown(rows),
        encoding="utf-8",
    )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "table_contract": {
            "row_count": 5,
            "data_regime_count": 4,
            "controlled_translation_is_derived": True,
            "external_reference_is_subset": True,
            "ranking_policy": "no composite tracker ranking",
        },
        "source_data": {
            path.relative_to(TABLE_ROOT).as_posix(): sha256_file(path)
            for path in (
                SOURCE_TABLE_PATH,
                SOURCE_PROVENANCE_PATH,
                *sorted(FROZEN_SOURCE_DIRECTORY.glob("*.json")),
            )
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
    )
    return manifest


if __name__ == "__main__":
    build_artifact()
