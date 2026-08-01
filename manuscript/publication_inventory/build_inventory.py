#!/usr/bin/env python3
"""Validate and render the final twenty-item publication inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


INVENTORY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = INVENTORY_ROOT.parents[1]
INVENTORY_PATH = INVENTORY_ROOT / "source_data" / "display_inventory.json"
OUTPUT_PATH = INVENTORY_ROOT.parent / "figures-and-tables.md"
MANIFEST_PATH = INVENTORY_ROOT / "manifest.json"

EXPECTED_IDS = (
    "Figure 1",
    "Figure 2",
    "Figure 3",
    "Figure 4",
    "Figure 5",
    "Table 1",
    "Table 2",
    "Figure S1",
    "Figure S2",
    "Figure S3",
    "Figure S4",
    "Figure S5",
    "Figure S6",
    "Table S1",
    "Table S2",
    "Table S3",
    "Table S4",
    "Table S5",
    "Table S6",
    "Table S7",
)
APPROVED_INVENTORY_SHA256 = (
    "6be02ac86d17fb5591bbcb6bb737c57254283fe8b61bb8e162dafe15127528e5"
)
EXPECTED_PLACEMENT_COUNTS = {
    "main_figure": 5,
    "main_table": 2,
    "supplementary_figure": 6,
    "supplementary_table": 7,
}
EXPECTED_ROUNDING_RULE_IDS = {
    "ag_residual_px_3dp",
    "diagram_labels",
    "endpoint_pp_6dp",
    "exact_counts",
    "full_precision_si",
    "rates_2dp",
    "translation_px_4dp",
}
EXPECTED_METHOD_PALETTE = {
    "dam4sam": "#009E73",
    "easyleed": "#D55E00",
    "locotrack": "#0072B2",
    "lucas_kanade": "#CC79A7",
    "ostrack": "#E69F00",
}
RECOVERY_DISPLAY_IDS = {
    "Figure 4",
    "Figure 5",
    "Table S2",
    "Table S3",
    "Table S6",
    "Table S7",
}
RECOVERY_TERM = "post-gap branch-consistent recovery"
SENSITIVITY_TEST_TARGETS = (
    (
        "trackleed.tests.test_scientific_sensitivity_evidence."
        "ScientificSensitivityEvidenceTests."
        "test_table_s6_retains_full_precision_and_conditional_denominators"
    ),
    (
        "trackleed.tests.test_scientific_sensitivity_evidence."
        "ScientificSensitivityEvidenceTests."
        "test_table_s6_is_deterministic_and_rejects_changed_source_data"
    ),
    (
        "trackleed.tests.test_scientific_sensitivity_evidence."
        "ScientificSensitivityEvidenceTests."
        "test_acceptance_validation_requires_the_authenticated_origin"
    ),
    (
        "trackleed.tests.test_scientific_sensitivity_evidence."
        "ScientificSensitivityEvidenceTests."
        "test_table_s6_manifest_covers_sources_code_and_outputs"
    ),
    (
        "trackleed.tests.test_scientific_sensitivity_evidence."
        "ScientificSensitivityEvidenceTests."
        "test_figure_s2_covers_matrix_ofat_outcomes_and_gate_audit"
    ),
    (
        "trackleed.tests.test_scientific_sensitivity_evidence."
        "ScientificSensitivityEvidenceTests."
        "test_figure_s2_is_deterministic_and_rejects_changed_gate_audit"
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: Path = INVENTORY_PATH) -> dict[str, Any]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict):
        raise ValueError("Publication inventory must be a JSON object.")
    return inventory


def approved_inventory_sha256(items: list[dict[str, Any]]) -> str:
    """Hash every field in the reviewed per-display contract."""

    canonical = json.dumps(
        items,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Inventory path escapes the repository: {relative_path}")
    return path


def _files_under(relative_roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative_root in relative_roots:
        root = _resolve_repo_path(relative_root)
        if not root.is_dir():
            raise ValueError(f"Missing source-data root: {relative_root}")
        root_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
        if not root_files:
            raise ValueError(f"Empty source-data root: {relative_root}")
        files.extend(root_files)
    return sorted(set(files))


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _manifest_digest_records(
    value: object,
) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if isinstance(value, dict):
        paired_fields = (
            ("path", "sha256"),
            ("origin_path", "origin_sha256"),
            ("frozen_path", "frozen_sha256"),
            ("frozen_path", "frozen_source_sha256"),
        )
        for path_key, digest_key in paired_fields:
            path_value = value.get(path_key)
            digest = value.get(digest_key)
            if isinstance(path_value, str) and _is_sha256(digest):
                records.append((path_value, str(digest).lower()))
        output_files = value.get("output_files")
        output_hashes = value.get("outputs_sha256")
        if isinstance(output_files, dict) and isinstance(output_hashes, dict):
            for label, output_path in output_files.items():
                digest = output_hashes.get(label)
                if isinstance(output_path, str) and _is_sha256(digest):
                    records.append((output_path, str(digest).lower()))
        for key, child in value.items():
            if _is_sha256(child):
                reference = key.removesuffix("_sha256")
                records.append((reference, str(child).lower()))
            records.extend(_manifest_digest_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_manifest_digest_records(child))
    return records


def _manifest_evidence(
    item: dict[str, Any],
) -> tuple[str, list[Path], list[tuple[str, str]]]:
    manifest_paths = [
        _resolve_repo_path(path) for path in item["manifest_paths"]
    ]
    texts: list[str] = []
    digest_records: list[tuple[str, str]] = []
    for path in manifest_paths:
        if not path.is_file():
            raise ValueError(f"Missing manifest: {path.relative_to(REPO_ROOT)}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "schema_version" not in manifest:
            raise ValueError(f"Manifest has no schema version: {path.name}")
        texts.append(json.dumps(manifest, sort_keys=True))
        digest_records.extend(_manifest_digest_records(manifest))
    return "\n".join(texts).lower(), manifest_paths, digest_records


def _validate_hash_membership(
    files: list[Path],
    digest_records: list[tuple[str, str]],
    *,
    artifact_path: Path,
    label: str,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in files:
        digest = sha256_file(path)
        repo_relative = path.relative_to(REPO_ROOT).as_posix().lower()
        try:
            artifact_relative = path.relative_to(artifact_path).as_posix().lower()
        except ValueError:
            artifact_relative = repo_relative
        candidates = {
            repo_relative,
            artifact_relative,
            path.name.lower(),
            path.stem.lower(),
        }
        matched = any(
            record_digest == digest
            and (
                reference.replace("\\", "/").lower() in candidates
                or any(
                    reference.replace("\\", "/").lower().endswith(candidate)
                    for candidate in candidates
                )
            )
            for reference, record_digest in digest_records
        )
        if not matched:
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            raise ValueError(
                f"{label} path/hash pair is absent from the bundle manifest: "
                f"{relative_path}"
            )
        hashes[path.relative_to(REPO_ROOT).as_posix()] = digest
    return hashes


def _output_files(artifact_path: Path) -> list[Path]:
    output_root = artifact_path / "output"
    if not output_root.is_dir():
        raise ValueError(
            f"Missing output directory: {output_root.relative_to(REPO_ROOT)}"
        )
    outputs = sorted(path for path in output_root.rglob("*") if path.is_file())
    if not outputs:
        raise ValueError(
            f"Empty output directory: {output_root.relative_to(REPO_ROOT)}"
        )
    return outputs


def _bundle_fingerprint(records: dict[str, str]) -> str:
    canonical = "\n".join(
        f"{path}:{digest}" for path, digest in sorted(records.items())
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bundle_test_commands(
    inventory: dict[str, Any],
) -> list[tuple[Path, tuple[str, ...]]]:
    """Return one unittest command per bundle test directory."""

    grouped: dict[Path, set[str]] = {}
    sensitivity_path = (
        "trackleed/tests/test_scientific_sensitivity_evidence.py"
    )
    sensitivity_required = False
    for item in inventory["items"]:
        for relative_path in item["test_paths"]:
            if relative_path == sensitivity_path:
                sensitivity_required = True
                continue
            path = _resolve_repo_path(relative_path)
            grouped.setdefault(path.parent, set()).add(path.name)
    commands = [
        (cwd, tuple(sorted(targets)))
        for cwd, targets in sorted(grouped.items(), key=lambda pair: str(pair[0]))
    ]
    if sensitivity_required:
        commands.append((REPO_ROOT, SENSITIVITY_TEST_TARGETS))
    return commands


def run_bundle_contract_tests(inventory: dict[str, Any]) -> None:
    """Execute every final bundle's declared contract tests, failing closed."""

    environment = os.environ.copy()
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPO_ROOT), current_python_path)
        if part
    )
    for cwd, targets in bundle_test_commands(inventory):
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", *targets],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            details = (completed.stdout + completed.stderr).strip()
            raise ValueError(
                f"Bundle contract tests failed in {cwd.relative_to(REPO_ROOT)}:\n"
                f"{details}"
            )


def validate_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Validate exact membership and every bundle's source/output hash chain."""

    if inventory.get("schema_version") != "1.0":
        raise ValueError("Unsupported publication-inventory schema.")
    items = inventory.get("items")
    if not isinstance(items, list):
        raise ValueError("Publication inventory must contain an item list.")

    actual_ids = tuple(item.get("id") for item in items)
    if actual_ids != EXPECTED_IDS:
        raise ValueError("Items do not match the approved inventory.")
    alternate_caption_items = [
        item.get("id", "unknown")
        for item in items
        if "caption_title" in item
    ]
    if alternate_caption_items:
        raise ValueError(
            "Alternate caption title is not allowed for "
            f"{alternate_caption_items[0]}."
        )
    if approved_inventory_sha256(items) != APPROVED_INVENTORY_SHA256:
        raise ValueError("Items do not match the approved inventory.")

    placement_counts = Counter(item.get("placement") for item in items)
    if dict(placement_counts) != EXPECTED_PLACEMENT_COUNTS:
        raise ValueError("Publication placement counts changed.")
    rounding_rules = inventory.get("rounding_rules")
    if not isinstance(rounding_rules, dict) or set(rounding_rules) != (
        EXPECTED_ROUNDING_RULE_IDS
    ):
        raise ValueError("Frozen reader-facing rounding rules changed.")
    if inventory.get("method_palette") != EXPECTED_METHOD_PALETTE:
        raise ValueError("Frozen method palette changed.")
    missing_policy = inventory.get("missing_value_rendering")
    if not isinstance(missing_policy, str) or not missing_policy.strip():
        raise ValueError("Global missing-value rendering policy is required.")

    source_count = 0
    output_count = 0
    bundle_fingerprints: dict[str, str] = {}
    all_builders_present = True
    all_tests_present = True
    all_rounding_rules_frozen = True
    all_denominator_policies_explicit = True

    for item in items:
        artifact_path = _resolve_repo_path(item["artifact_path"])
        if not artifact_path.is_dir():
            raise ValueError(f"Missing artifact: {item['artifact_path']}")
        builder_path = _resolve_repo_path(item["builder_path"])
        all_builders_present &= builder_path.is_file()
        if not builder_path.is_file():
            raise ValueError(f"Missing builder: {item['builder_path']}")
        test_paths = [_resolve_repo_path(path) for path in item["test_paths"]]
        all_tests_present &= bool(test_paths) and all(
            path.is_file() for path in test_paths
        )
        if not all(path.is_file() for path in test_paths):
            raise ValueError(f"Missing contract test for {item['id']}.")

        caption_path = _resolve_repo_path(item["caption_path"])
        if not caption_path.is_file():
            raise ValueError(f"Missing caption contract for {item['id']}.")
        caption_text = caption_path.read_text(encoding="utf-8")
        if item["title"] not in caption_text:
            raise ValueError(f"Approved title is absent from {item['id']}.")
        if "identity recovery" in caption_text.casefold():
            raise ValueError(
                f"Superseded identity-recovery wording remains in {item['id']}."
            )
        if (
            item["id"] in RECOVERY_DISPLAY_IDS
            and RECOVERY_TERM not in caption_text.casefold()
        ):
            raise ValueError(
                f"Exact recovery term is absent from {item['id']}."
            )

        rule_ids = item.get("rounding_rule_ids")
        valid_rules = (
            isinstance(rule_ids, list)
            and bool(rule_ids)
            and set(rule_ids) <= EXPECTED_ROUNDING_RULE_IDS
        )
        all_rounding_rules_frozen &= valid_rules
        if not valid_rules:
            raise ValueError(f"Invalid rounding contract for {item['id']}.")
        denominator_policy = item.get("denominator_policy")
        denominator_is_explicit = (
            isinstance(denominator_policy, str)
            and bool(denominator_policy.strip())
        )
        all_denominator_policies_explicit &= denominator_is_explicit
        if not denominator_is_explicit:
            raise ValueError(f"Missing denominator policy for {item['id']}.")
        for field in ("role", "palette_policy", "missing_value_policy"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {field} for {item['id']}.")

        manifest_text, manifest_paths, digest_records = _manifest_evidence(item)
        if item["palette_policy"] == (
            "Uses the frozen five-method colorblind-safe palette."
        ):
            missing_colors = {
                color
                for color in EXPECTED_METHOD_PALETTE.values()
                if color.lower() not in manifest_text
            }
            if missing_colors:
                raise ValueError(
                    f"Method palette is not manifested for {item['id']}."
                )
        if item["placement"].endswith("figure"):
            missing_evidence = manifest_text + "\n" + caption_text.casefold()
            if not any(
                term in missing_evidence
                for term in (
                    "missing",
                    "gap",
                    "unavailable",
                    "rejected",
                    "invalid",
                    "accepted",
                    "coverage",
                )
            ):
                raise ValueError(
                    f"Missing-value rendering is not evidenced for {item['id']}."
                )
        source_files = _files_under(item["source_roots"])
        outputs = _output_files(artifact_path)
        source_hashes = _validate_hash_membership(
            source_files,
            digest_records,
            artifact_path=artifact_path,
            label="Source-data",
        )
        output_hashes = _validate_hash_membership(
            outputs,
            digest_records,
            artifact_path=artifact_path,
            label="Output",
        )
        manifest_hashes = {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in manifest_paths
        }
        contract_hashes = {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in [builder_path, caption_path, *test_paths]
        }
        records = (
            source_hashes
            | output_hashes
            | manifest_hashes
            | contract_hashes
        )
        bundle_fingerprints[item["id"]] = _bundle_fingerprint(records)
        source_count += len(source_files)
        output_count += len(outputs)

    return {
        "ids": list(EXPECTED_IDS),
        "placement_counts": dict(placement_counts),
        "bundle_count": len(items),
        "source_file_count": source_count,
        "hashed_source_file_count": source_count,
        "output_file_count": output_count,
        "hashed_output_file_count": output_count,
        "all_builders_present": all_builders_present,
        "all_tests_present": all_tests_present,
        "all_rounding_rules_frozen": all_rounding_rules_frozen,
        "all_denominator_policies_explicit": (
            all_denominator_policies_explicit
        ),
        "bundle_fingerprints": bundle_fingerprints,
    }


def _relative_link(path: str) -> str:
    prefix = "manuscript/"
    return path[len(prefix) :] if path.startswith(prefix) else f"../{path}"


def render_inventory(inventory: dict[str, Any]) -> str:
    sections = (
        ("main_figure", "Main figures"),
        ("main_table", "Main tables"),
        ("supplementary_figure", "Supplementary figures"),
        ("supplementary_table", "Supplementary tables"),
    )
    rules = inventory["rounding_rules"]
    lines = [
        "# Final publication display inventory",
        "",
        (
            "This file is generated from the validated cross-regime inventory. "
            "It contains exactly five main figures, two main tables, Figures "
            "S1-S6, and Tables S1-S7. Historical artifact directories are not "
            "final display placements."
        ),
        "",
        "## Shared rendering contract",
        "",
        f"- Missing values: {inventory['missing_value_rendering']}",
        "- Method palette: "
        + ", ".join(
            f"{method} `{color}`"
            for method, color in inventory["method_palette"].items()
        )
        + ".",
        "- Rounding rules: "
        + " ".join(rules[rule_id] for rule_id in sorted(rules)),
        "",
    ]
    for placement, heading in sections:
        lines.extend([f"## {heading}", ""])
        for item in inventory["items"]:
            if item["placement"] != placement:
                continue
            artifact_link = _relative_link(item["artifact_path"])
            caption_link = _relative_link(item["caption_path"])
            manifest_links = ", ".join(
                f"[manifest]({_relative_link(path)})"
                for path in item["manifest_paths"]
            )
            lines.extend(
                [
                    f"### {item['id']} — {item['title']}",
                    "",
                    item["role"],
                    "",
                    f"- Bundle: [{artifact_link}]({artifact_link}/README.md)",
                    f"- Caption/source display: [open]({caption_link})",
                    f"- Provenance: {manifest_links}",
                    f"- Denominators: {item['denominator_policy']}",
                    "- Rounding: "
                    + " ".join(rules[rule] for rule in item["rounding_rule_ids"]),
                    f"- Missing values: {item['missing_value_policy']}",
                    f"- Palette: {item['palette_policy']}",
                    "",
                ]
            )
    return "\n".join(lines)


def build_inventory_artifact(
    inventory: dict[str, Any],
    *,
    output_path: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    report = validate_inventory(inventory)
    rendered = render_inventory(inventory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "completed",
        "inventory_contract": {
            "ids": list(EXPECTED_IDS),
            "placement_counts": report["placement_counts"],
            "obsolete_main_table_present": False,
            "superseded_placements_present": False,
            "default_cli_runs_bundle_contract_tests": True,
            "bundle_test_command_count": len(bundle_test_commands(inventory)),
        },
        "source_inventory_sha256": sha256_file(INVENTORY_PATH),
        "builder_sha256": sha256_file(Path(__file__)),
        "bundle_fingerprints": report["bundle_fingerprints"],
        "output_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-contract-tests",
        action="store_true",
        help="Rebuild the index after contract tests were run separately.",
    )
    arguments = parser.parse_args()
    inventory = load_inventory()
    if not arguments.skip_contract_tests:
        run_bundle_contract_tests(inventory)
    build_inventory_artifact(inventory)


if __name__ == "__main__":
    main()
