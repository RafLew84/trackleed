"""Create the initial source-linked metadata record for author review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.experimental_metadata_record import (  # noqa: E402
    initial_metadata_record,
    metadata_claim_constraints,
)


DEFAULT_PACKAGE = Path("manuscript/experimental_metadata")
README_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "trackleed"
    / "analysis"
    / "experimental_metadata_README.md"
)
GENERATED_FILES = (
    "README.md",
    "experimental_metadata.json",
    "metadata_claim_constraints.json",
    "author_checkpoint.json",
    "package_manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    package = args.package
    if not package.is_absolute():
        package = workspace_root / package
    package.mkdir(parents=True, exist_ok=True)
    existing = [name for name in GENERATED_FILES if (package / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to replace experimental-metadata record: "
            + ", ".join(existing)
        )
    tiff_inventory = package / "tiff_metadata_inventory.json"
    if not tiff_inventory.is_file():
        raise FileNotFoundError(
            "Build the TIFF metadata inventory before freezing the record"
        )

    readme_path = package / "README.md"
    record_path = package / "experimental_metadata.json"
    constraints_path = package / "metadata_claim_constraints.json"
    checkpoint_path = package / "author_checkpoint.json"
    readme_path.write_text(
        README_TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_json(record_path, initial_metadata_record(workspace_root))
    _write_json(constraints_path, metadata_claim_constraints())
    _write_json(
        checkpoint_path,
        {
            "schema_version": 1,
            "status": "pending",
            "record_sha256": _sha256(record_path),
            "constraints_sha256": _sha256(constraints_path),
            "confirmed_by": None,
            "confirmed_date": None,
            "confirmation_statement": None,
            "attestations": {
                "explicit_gaps_reviewed": False,
                "no_unsupported_inference": False,
                "recovered_values_reviewed": False,
            },
            "confirmed_not_recorded_field_ids": [],
        },
    )
    package_files = (
        readme_path,
        record_path,
        constraints_path,
        checkpoint_path,
        tiff_inventory,
    )
    _write_json(
        package / "package_manifest.json",
        {
            "schema_version": 1,
            "files_sha256": {
                path.name: _sha256(path) for path in package_files
            },
        },
    )
    print(f"Frozen experimental metadata for author review: {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
