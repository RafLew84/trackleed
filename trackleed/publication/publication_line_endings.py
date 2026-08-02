"""Deterministic LF normalization for publication hash metadata."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


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
IMMUTABLE_PATHS = frozenset({"manuscript/scientific_core/fingerprint.json"})
IMMUTABLE_PREFIXES = ("manuscript/scientific_baseline/frozen_inputs/",)
AUTHENTICATED_REFERENCE_PREFIXES = (
    ".scratch/trackleed-physics-method-paper/",
    "data/",
) + BYTE_PRESERVED_PREFIXES
HASH_VALUE_IMMUTABLE_PATHS = frozenset(
    {"manuscript/tables/table_s1_software_environments/build_table.py"}
)


@dataclass(frozen=True)
class LineEndingMigration:
    documents: dict[str, bytes]
    changed_paths: tuple[str, ...]
    iterations: int


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_lf(value: bytes) -> bytes:
    """Return UTF-8-compatible text bytes with LF-only line endings."""

    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _is_immutable(path: str) -> bool:
    return path in IMMUTABLE_PATHS or path.startswith(IMMUTABLE_PREFIXES)


def _is_byte_preserved(path: str) -> bool:
    return path.startswith(BYTE_PRESERVED_PREFIXES)


def _walk_json(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _protected_digests(documents: Mapping[str, bytes]) -> set[str]:
    protected: set[str] = set()
    for path, value in documents.items():
        if not _is_byte_preserved(path):
            continue
        protected.add(_sha256(value))
    for path, value in documents.items():
        if not path.endswith(".json"):
            continue
        try:
            payload = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for item in _walk_json(payload):
            referenced_path = item.get("path")
            digest = item.get("sha256")
            if (
                isinstance(referenced_path, str)
                and referenced_path.replace("\\", "/").startswith(
                    AUTHENTICATED_REFERENCE_PREFIXES
                )
                and isinstance(digest, str)
                and len(digest) == 64
            ):
                protected.add(digest)
    return protected


def migrate_line_ending_hashes(
    documents: Mapping[str, bytes],
    *,
    maximum_iterations: int = 32,
) -> LineEndingMigration:
    """Normalize repository text and propagate legacy CRLF hashes to a fixed point."""

    migrated = dict(documents)
    original = dict(documents)
    replacements: dict[str, str] = {}
    protected_digests = _protected_digests(documents)

    for path, value in tuple(migrated.items()):
        if _is_immutable(path) or _is_byte_preserved(path):
            continue
        canonical = normalize_lf(value)
        legacy = canonical.replace(b"\n", b"\r\n")
        migrated[path] = canonical
        if legacy != canonical:
            legacy_digest = _sha256(legacy)
            if legacy_digest not in protected_digests:
                replacements[legacy_digest] = _sha256(canonical)

    for iteration in range(maximum_iterations):
        before = dict(migrated)
        changed_this_iteration: list[str] = []
        for path, value in before.items():
            if (
                _is_immutable(path)
                or _is_byte_preserved(path)
                or path in HASH_VALUE_IMMUTABLE_PATHS
            ):
                continue
            text = value.decode("utf-8")
            for legacy_digest, canonical_digest in replacements.items():
                text = text.replace(legacy_digest, canonical_digest)
            updated = text.encode("utf-8")
            if updated != value:
                migrated[path] = updated
                changed_this_iteration.append(path)
        if not changed_this_iteration:
            return LineEndingMigration(
                documents=migrated,
                changed_paths=tuple(
                    sorted(path for path in migrated if migrated[path] != original[path])
                ),
                iterations=iteration,
            )
        for path in changed_this_iteration:
            before_digest = _sha256(before[path])
            if before_digest not in protected_digests:
                replacements[before_digest] = _sha256(migrated[path])

    raise ValueError("Publication line-ending hash migration did not reach a fixed point")


def load_utf8_documents(
    workspace_root: Path,
    relative_paths: list[str],
) -> dict[str, bytes]:
    """Load tracked UTF-8 text-shaped files for migration."""

    documents: dict[str, bytes] = {}
    for relative_path in relative_paths:
        path = workspace_root / relative_path
        if not path.is_file() or (
            path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitattributes"
        ):
            continue
        value = path.read_bytes()
        try:
            value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        documents[relative_path] = value
    return documents


__all__ = [
    "AUTHENTICATED_REFERENCE_PREFIXES",
    "BYTE_PRESERVED_PREFIXES",
    "HASH_VALUE_IMMUTABLE_PATHS",
    "IMMUTABLE_PATHS",
    "IMMUTABLE_PREFIXES",
    "LineEndingMigration",
    "TEXT_SUFFIXES",
    "load_utf8_documents",
    "migrate_line_ending_hashes",
    "normalize_lf",
]
