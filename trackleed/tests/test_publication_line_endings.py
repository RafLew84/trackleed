from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from trackleed.publication.publication_line_endings import (
    BYTE_PRESERVED_PREFIXES as MIGRATION_BYTE_PRESERVED_PREFIXES,
    TEXT_SUFFIXES as MIGRATION_TEXT_SUFFIXES,
    migrate_line_ending_hashes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".md", ".py", ".tex", ".tsv", ".txt", ".yaml", ".yml"}
)
BYTE_PRESERVED_PREFIXES = (
    "manuscript/figures/figure_s4_easyleed_viperleed_compatibility/source_data/",
    "manuscript/scientific_baseline/frozen_inputs/",
    "manuscript/source_data/controlled_translations/",
    "manuscript/tables/stratified_uncertainty/source_data/",
    "manuscript/tables/table01_data_evidence_roles/source_data/",
    "manuscript/tables/table02_tracking_methods_contract/source_data/",
    "manuscript/tables/table_s1_software_environments/source_data/",
    "manuscript/tables/table_s7_cross_regime_summary/source_data/",
)


def _collect_path_keys(value: object) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and (REPO_ROOT / key).is_file():
                paths.add(key)
            paths.update(_collect_path_keys(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_collect_path_keys(child))
    return paths


def _hash_bound_text_paths() -> tuple[str, ...]:
    contract = json.loads(
        (REPO_ROOT / "manuscript/scientific_core/fingerprint_contract.json").read_text(
            encoding="utf-8"
        )
    )
    semantic_manifest = json.loads(
        (REPO_ROOT / "manuscript/publication_inventory/semantic_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    paths = {
        component["path"]
        for component in contract["components"]
        if Path(component["path"]).suffix.lower() in TEXT_SUFFIXES
    }
    paths.update(
        path
        for path in _collect_path_keys(semantic_manifest)
        if Path(path).suffix.lower() in TEXT_SUFFIXES
    )
    return tuple(sorted(paths))


def _git_attributes(paths: tuple[str, ...]) -> dict[str, dict[str, str]]:
    completed = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "text", "eol"],
        cwd=REPO_ROOT,
        input=("\0".join(paths) + "\0").encode(),
        capture_output=True,
        check=True,
    )
    fields = completed.stdout.decode().split("\0")
    if fields and not fields[-1]:
        fields.pop()
    if len(fields) % 3:
        raise AssertionError("git check-attr returned an incomplete record")
    attributes = {path: {} for path in paths}
    for index in range(0, len(fields), 3):
        path, attribute, value = fields[index : index + 3]
        attributes[path][attribute] = value
    return attributes


def _approved_builder_paths() -> tuple[Path, ...]:
    inventory = json.loads(
        (
            REPO_ROOT
            / "manuscript/publication_inventory/source_data/display_inventory.json"
        ).read_text(encoding="utf-8")
    )
    paths = {
        REPO_ROOT / item["builder_path"]
        for item in inventory["items"]
    }
    paths.update((REPO_ROOT / "trackleed/publication").rglob("*.py"))
    paths.update((REPO_ROOT / "manuscript/publication_inventory").rglob("*.py"))
    paths.add(REPO_ROOT / "scripts/reconstruct_publication.py")
    return tuple(sorted(paths))


class PublicationLineEndingContractTests(unittest.TestCase):
    def test_fail_closed_policy_catalog_matches_migrator(self) -> None:
        self.assertEqual(TEXT_SUFFIXES, MIGRATION_TEXT_SUFFIXES)
        self.assertEqual(
            BYTE_PRESERVED_PREFIXES,
            MIGRATION_BYTE_PRESERVED_PREFIXES,
        )

    def test_hash_bound_text_has_explicit_checkout_policy(self) -> None:
        paths = _hash_bound_text_paths()
        attributes = _git_attributes(paths)

        for path in paths:
            policy = attributes[path]
            if policy["text"] == "unset":
                self.assertTrue(
                    path.startswith(BYTE_PRESERVED_PREFIXES),
                    f"Undeclared byte-preserved text path: {path}",
                )
            else:
                self.assertEqual(policy["eol"], "lf", path)

    def test_reconstruction_writers_declare_lf_newlines(self) -> None:
        unsafe_calls: list[str] = []
        for path in _approved_builder_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text"
                    and not any(keyword.arg == "newline" for keyword in node.keywords)
                ):
                    unsafe_calls.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}"
                    )

        self.assertEqual(unsafe_calls, [])

    def test_hash_migration_propagates_transitively_and_preserves_fingerprint(
        self,
    ) -> None:
        child_lf = b"child\n"
        child_crlf = b"child\r\n"
        child_legacy_hash = hashlib.sha256(child_crlf).hexdigest()
        parent_lf = f'{{"child_sha256":"{child_legacy_hash}"}}\n'.encode()
        parent_crlf = parent_lf.replace(b"\n", b"\r\n")
        parent_legacy_hash = hashlib.sha256(parent_crlf).hexdigest()
        fingerprint = f'{{"parent_sha256":"{parent_legacy_hash}"}}\n'.encode()
        documents = {
            "manuscript/example/child.txt": child_crlf,
            "manuscript/example/parent.json": parent_crlf,
            "manuscript/scientific_core/fingerprint.json": fingerprint,
        }

        migration = migrate_line_ending_hashes(documents)

        child_hash = hashlib.sha256(child_lf).hexdigest()
        self.assertNotIn(b"\r", migration.documents["manuscript/example/child.txt"])
        self.assertIn(
            child_hash.encode(),
            migration.documents["manuscript/example/parent.json"],
        )
        self.assertEqual(
            migration.documents["manuscript/scientific_core/fingerprint.json"],
            fingerprint,
        )

    def test_hash_migration_preserves_authenticated_source_digest_collision(
        self,
    ) -> None:
        canonical = b"authenticated source\n"
        authenticated = canonical.replace(b"\n", b"\r\n")
        authenticated_hash = hashlib.sha256(authenticated).hexdigest()
        source_reference = (
            "{\n"
            '  "path": ".scratch/trackleed-physics-method-paper/source.json",\n'
            f'  "sha256": "{authenticated_hash}"\n'
            "}\n"
        ).encode()
        documents = {
            "manuscript/example/generated.txt": authenticated,
            "manuscript/example/source_reference.json": source_reference,
            (
                "manuscript/tables/table01_data_evidence_roles/source_data/"
                "upstream_contract.json"
            ): source_reference.replace(b"\n", b"\r\n"),
        }

        migration = migrate_line_ending_hashes(documents)

        self.assertEqual(
            migration.documents["manuscript/example/generated.txt"], canonical
        )
        self.assertIn(
            authenticated_hash.encode(),
            migration.documents["manuscript/example/source_reference.json"],
        )
        byte_preserved_path = (
            "manuscript/tables/table01_data_evidence_roles/source_data/"
            "upstream_contract.json"
        )
        self.assertEqual(
            migration.documents[byte_preserved_path],
            documents[byte_preserved_path],
        )

    def test_transitive_hash_migration_does_not_reintroduce_protected_digest(
        self,
    ) -> None:
        child_lf = b"child\n"
        child_crlf = child_lf.replace(b"\n", b"\r\n")
        child_legacy_hash = hashlib.sha256(child_crlf).hexdigest()
        contract = f'{{"child_sha256":"{child_legacy_hash}"}}\n'.encode()
        contract_hash = hashlib.sha256(contract).hexdigest()
        protected_path = (
            "manuscript/source_data/controlled_translations/contract.json"
        )
        reference = (
            "{\n"
            f'  "path": "{protected_path}",\n'
            f'  "sha256": "{contract_hash}"\n'
            "}\n"
        ).encode()
        documents = {
            "manuscript/example/child.txt": child_crlf,
            protected_path: contract,
            "manuscript/example/contract-copy.json": contract,
            "manuscript/example/reference.json": reference,
        }

        migration = migrate_line_ending_hashes(documents)

        self.assertEqual(migration.documents[protected_path], contract)
        self.assertNotEqual(
            migration.documents["manuscript/example/contract-copy.json"], contract
        )
        self.assertIn(
            contract_hash.encode(),
            migration.documents["manuscript/example/reference.json"],
        )

    def test_hash_migration_preserves_frozen_hash_constants_in_table_s1(self) -> None:
        child_crlf = b"child\r\n"
        legacy_hash = hashlib.sha256(child_crlf).hexdigest()
        builder_crlf = f'FROZEN_SHA256 = "{legacy_hash}"\r\n'.encode()
        documents = {
            "manuscript/example/child.txt": child_crlf,
            (
                "manuscript/tables/table_s1_software_environments/"
                "build_table.py"
            ): builder_crlf,
        }

        migration = migrate_line_ending_hashes(documents)
        builder = migration.documents[
            "manuscript/tables/table_s1_software_environments/build_table.py"
        ]

        self.assertNotIn(b"\r", builder)
        self.assertIn(legacy_hash.encode(), builder)


if __name__ == "__main__":
    unittest.main()
