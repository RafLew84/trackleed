#!/usr/bin/env python3
"""Build main Table 2 from a frozen tracking-method contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
SOURCE_TABLE_PATH = SOURCE_DATA_DIRECTORY / "methods.csv"
SOURCE_PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
FROZEN_SOURCE_DIRECTORY = SOURCE_DATA_DIRECTORY / "upstream_contracts"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"

MANUSCRIPT_BEGIN_MARKER = "<!-- BEGIN TABLE 2 -->"
MANUSCRIPT_END_MARKER = "<!-- END TABLE 2 -->"

SOURCE_COLUMNS = (
    "order",
    "panel",
    "method",
    "family_representation",
    "configuration",
    "initialization",
    "native_output",
    "proposal_coordinate",
    "reset_policy",
    "study_role",
    "upstream_revision",
    "software_version",
    "checkpoint_identifier",
    "checkpoint_sha256",
    "supporting_artifact_sha256",
    "executed_configuration_sha256",
)
DISPLAY_COLUMNS = (
    ("method", "Method"),
    ("reader_native_output", "Native output"),
    ("proposal_coordinate", "Proposal adapter"),
    ("reader_native_availability", "Native availability"),
    ("shared_downstream_measurement", "Shared downstream measurement"),
    ("reset_policy", "Reset policy"),
    ("study_role", "Study role"),
)
EXPECTED_PANELS = (
    ("Main", 5),
    ("Controlled SAM2 family", 3),
    ("External reference", 1),
)
EXPECTED_METHODS = (
    "LocoTrack",
    "OSTrack",
    "DAM4SAM",
    "Lucas–Kanade",
    "EasyLEED",
    "SAM2.1",
    "SAMURAI",
    "DAM4SAM",
    "ViPErLEED",
)
SHARED_SAM2_CHECKPOINT = (
    "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)
READER_NATIVE_OUTPUTS = (
    "Point plus occlusion and expected-distance diagnostics",
    "Bounding box plus native score",
    "Binary mask plus native diagnostics",
    "Optical-flow point plus residual and closure diagnostics",
    "Kalman x,y plus native intensity and tracking radius",
    "Binary mask",
    "Binary mask plus motion-aware selection",
    "Binary mask plus distractor-aware diagnostics",
    "Whole-pattern beam coordinate",
)
READER_NATIVE_AVAILABILITY = (
    "Finite point; energy scans also require the frozen native visibility gate",
    "Finite positive-area box",
    "Finite nonempty mask",
    "Match passing the frozen native flow gates",
    "Finite exported Kalman x,y",
    "Finite nonempty mask",
    "Finite nonempty mask",
    "Finite nonempty mask",
    "Finite pattern-associated coordinate for the corresponding beam",
)
SHARED_DOWNSTREAM_MEASUREMENT = (
    "48 × 48 raw-image ROI → common elliptical-Gaussian position"
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


def _panel_sequence() -> tuple[str, ...]:
    return tuple(
        panel
        for panel, count in EXPECTED_PANELS
        for _ in range(count)
    )


def _validate_against_frozen_contracts(
    rows: list[dict[str, str]],
) -> None:
    """Check audit fields against the committed upstream snapshots."""

    translation_contract = _load_json(
        FROZEN_SOURCE_DIRECTORY / "controlled_translation_contract.json"
    )
    run_manifest = _load_json(
        FROZEN_SOURCE_DIRECTORY / "five_method_run_manifest.json"
    )
    family_contract = _load_json(
        FROZEN_SOURCE_DIRECTORY / "sam2_family_contract.json"
    )
    external_contract = _load_json(
        FROZEN_SOURCE_DIRECTORY / "external_reference_contract.json"
    )
    domain_record = (
        FROZEN_SOURCE_DIRECTORY / "domain_baselines_installation.md"
    ).read_text(encoding="utf-8")

    method_keys = (
        "locotrack_base_offline",
        "ostrack_vitb_256_mae_ce",
        "dam4sam_sam21pp_l",
        "lucas_kanade_skip_anchor",
        "easyleed_kalman",
    )
    run_hashes = run_manifest.get("method_configuration_sha256")
    translation_hashes = translation_contract.get(
        "method_configuration_sha256"
    )
    if not isinstance(run_hashes, dict) or not isinstance(
        translation_hashes,
        dict,
    ):
        raise ValueError("Main-method configuration hashes are missing.")
    for row, method_key in zip(rows[:5], method_keys, strict=True):
        expected_hash = str(run_hashes[method_key])
        if expected_hash != str(translation_hashes[method_key]):
            raise ValueError(
                f"Configuration hash disagreement for {method_key}."
            )
        if expected_hash != row["executed_configuration_sha256"]:
            raise ValueError(
                f"Source row lost the executed configuration for {method_key}."
            )

    translation_configurations = translation_contract.get(
        "method_configurations"
    )
    translation_resources = translation_contract.get("method_resources")
    if not isinstance(translation_configurations, dict) or not isinstance(
        translation_resources,
        dict,
    ):
        raise ValueError("Main-method revisions or resources are missing.")
    learned_contract_keys = (
        ("locotrack_base_offline", "locotrack"),
        ("ostrack_vitb_256_mae_ce", "ostrack"),
        ("dam4sam_sam21pp_l", "dam4sam"),
    )
    for row, (configuration_key, resource_key) in zip(
        rows[:3],
        learned_contract_keys,
        strict=True,
    ):
        configuration = translation_configurations[configuration_key]
        resource = translation_resources[resource_key]
        if not isinstance(configuration, dict) or not isinstance(resource, dict):
            raise ValueError("Malformed learned-method contract.")
        if row["upstream_revision"] != configuration["upstream_revision"]:
            raise ValueError(
                f"Upstream revision changed for {row['method']}."
            )
        if str(resource["sha256"]) not in row[
            "checkpoint_sha256"
        ].split("; "):
            raise ValueError(
                f"Checkpoint hash changed for {row['method']}."
            )

    controlled_family = family_contract.get("controlled_family")
    if not isinstance(controlled_family, dict):
        raise ValueError("Controlled SAM2-family contract is missing.")
    family_rows = rows[5:8]
    if controlled_family.get("methods") != ["sam2", "samurai", "dam4sam"]:
        raise ValueError("Controlled SAM2-family method order changed.")
    if controlled_family.get("identical_checkpoint_weights") is not True:
        raise ValueError("Controlled family no longer has identical weights.")
    if controlled_family.get("checkpoint_sha256") != SHARED_SAM2_CHECKPOINT:
        raise ValueError("Controlled-family checkpoint hash changed.")
    if {
        row["checkpoint_sha256"] for row in family_rows
    } != {SHARED_SAM2_CHECKPOINT}:
        raise ValueError("Controlled-family source rows do not share weights.")
    if {
        row["executed_configuration_sha256"] for row in family_rows
    } != {"not recorded separately"}:
        raise ValueError(
            "Do not label the family-contract hash as a method configuration."
        )

    external = external_contract.get("external_reference")
    if not isinstance(external, dict):
        raise ValueError("External-reference contract is missing.")
    if external.get("part_of_main_five_method_matrix") is not False:
        raise ValueError("ViPErLEED must remain outside the main matrix.")
    if external.get("methods") != ["EasyLEED", "ViPErLEED"]:
        raise ValueError("Unexpected external-reference methods.")

    domain_anchors = (
        rows[4]["upstream_revision"],
        rows[4]["software_version"],
        rows[8]["upstream_revision"],
        rows[8]["software_version"],
        rows[8]["supporting_artifact_sha256"],
    )
    if any(anchor not in domain_record for anchor in domain_anchors):
        raise ValueError("Domain-baseline audit fields changed.")


def load_and_validate_rows(
    source_path: Path = SOURCE_TABLE_PATH,
) -> list[dict[str, str]]:
    """Load and validate the nine frozen method-contract rows."""

    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("Unexpected source-table columns.")
        rows = list(reader)

    if len(rows) != 9:
        raise ValueError("Table 2 must contain exactly nine rows.")
    if [int(row["order"]) for row in rows] != list(range(1, 10)):
        raise ValueError("Table 2 rows must retain frozen order 1-9.")
    if tuple(row["panel"] for row in rows) != _panel_sequence():
        raise ValueError("Table 2 panel grouping changed.")
    if tuple(row["method"] for row in rows) != EXPECTED_METHODS:
        raise ValueError("Table 2 method order changed.")
    if any("Prohibited" not in row["reset_policy"] for row in rows):
        raise ValueError("Every evaluated path must prohibit manual resets.")

    reader_facing_text = "\n".join(
        value for row in display_rows(rows) for value in row.values()
    )
    if any(
        term in reader_facing_text
        for term in ("ScanE", "26a21", "26a24")
    ):
        raise ValueError(
            "Reader-facing Table 2 contains an internal archive identifier."
        )

    _validate_against_frozen_contracts(rows)
    return rows


def _markdown_cell(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ").strip()


def display_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove audit-only fields from the publication table."""

    return [
        {
            "Method": row["method"],
            "Native output": native_output,
            "Proposal adapter": row["proposal_coordinate"],
            "Native availability": native_availability,
            "Shared downstream measurement": SHARED_DOWNSTREAM_MEASUREMENT,
            "Reset policy": row["reset_policy"],
            "Study role": row["study_role"],
        }
        for row, native_output, native_availability in zip(
            rows,
            READER_NATIVE_OUTPUTS,
            READER_NATIVE_AVAILABILITY,
            strict=True,
        )
    ]


def render_markdown(rows: list[dict[str, str]]) -> str:
    """Render the exact Markdown block copied into the manuscript."""

    displayed = display_rows(rows)
    headers = [display_name for _, display_name in DISPLAY_COLUMNS]
    lines = [
        (
            "**Table 2. Methods mapped to the common measurement "
            "contract.** The first five rows form the common "
            "strategy matrix. The controlled SAM2-family rows used matched "
            "checkpoint weights and a common prompt. Every available proposal "
            "entered the same raw-image ROI and two-dimensional "
            "elliptical-Gaussian position estimator. "
            "ViPErLEED is a separate whole-pattern external reference, not a "
            "sixth row in the common five-method matrix. Exact versions, "
            "revisions, configurations, thresholds, hashes, preprocessing, "
            "and environments are reported in Table S1 and its accompanying "
            "supplementary native-method contract."
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
    """Return the normalized generated block between Table 2 markers."""

    if manuscript.count(MANUSCRIPT_BEGIN_MARKER) != 1:
        raise ValueError("Manuscript must contain one Table 2 begin marker.")
    if manuscript.count(MANUSCRIPT_END_MARKER) != 1:
        raise ValueError("Manuscript must contain one Table 2 end marker.")
    start = manuscript.index(MANUSCRIPT_BEGIN_MARKER) + len(
        MANUSCRIPT_BEGIN_MARKER
    )
    end = manuscript.index(MANUSCRIPT_END_MARKER, start)
    return manuscript[start:end].strip("\n") + "\n"


def validate_provenance() -> dict[str, object]:
    """Validate hashes of committed upstream contract snapshots."""

    provenance = _load_json(SOURCE_PROVENANCE_PATH)
    sources = provenance.get("sources")
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("Source provenance must list five upstream records.")
    required_keys = {
        "frozen_path",
        "frozen_sha256",
        "origin_path",
        "origin_sha256",
        "role",
    }
    for source in sources:
        if not isinstance(source, dict) or set(source) != required_keys:
            raise ValueError("Malformed source-provenance entry.")
        for digest_key in ("frozen_sha256", "origin_sha256"):
            digest = str(source[digest_key])
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("Malformed SHA-256 in source provenance.")
        frozen_path = (TABLE_ROOT / str(source["frozen_path"])).resolve()
        if TABLE_ROOT not in frozen_path.parents:
            raise ValueError("Frozen provenance path escapes the artifact.")
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
        "table02_tracking_methods_contract.csv": (
            output_directory / "table02_tracking_methods_contract.csv"
        ),
        "table02_tracking_methods_contract.md": (
            output_directory / "table02_tracking_methods_contract.md"
        ),
    }
    _write_csv(output_paths["table02_tracking_methods_contract.csv"], rows)
    output_paths["table02_tracking_methods_contract.md"].write_text(
        render_markdown(rows),
        encoding="utf-8",
    )

    source_paths = (
        SOURCE_TABLE_PATH,
        SOURCE_PROVENANCE_PATH,
        *sorted(
            path
            for path in FROZEN_SOURCE_DIRECTORY.iterdir()
            if path.is_file()
        ),
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "completed",
        "table_contract": {
            "row_count": 9,
            "main_method_count": 5,
            "controlled_family_count": 3,
            "external_reference_count": 1,
            "external_reference_is_separate": True,
            "controlled_family_weights_identical": True,
            "controlled_family_checkpoint_sha256": SHARED_SAM2_CHECKPOINT,
            "manual_reset_or_correction": "prohibited for every row",
        },
        "source_data": {
            path.relative_to(TABLE_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
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
