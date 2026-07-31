"""Freeze the author-confirmed experimental-metadata HITL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.experimental_metadata import (  # noqa: E402
    validate_metadata_package,
)


DEFAULT_PACKAGE = REPO_ROOT / "manuscript" / "experimental_metadata"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--confirmed-date", default="2026-07-31")
    args = parser.parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    package = args.package.resolve()

    validate_metadata_package(workspace_root, package)
    record_path = package / "experimental_metadata.json"
    constraints_path = package / "metadata_claim_constraints.json"
    checkpoint_path = package / "author_checkpoint.json"
    manifest_path = package / "package_manifest.json"
    record = _read_json(record_path)
    checkpoint = _read_json(checkpoint_path)
    if (
        record.get("record_status") != "awaiting_author_confirmation"
        or checkpoint.get("status") != "pending"
    ):
        raise ValueError("Metadata checkpoint is not pending")

    record["record_status"] = "author_confirmed"
    _write_json(record_path, record)
    checkpoint.update(
        {
            "status": "confirmed",
            "record_sha256": _sha256(record_path),
            "constraints_sha256": _sha256(constraints_path),
            "confirmed_by": "publication author via repository HITL checkpoint",
            "confirmed_date": args.confirmed_date,
            "confirmation_statement": (
                "The recovered values, explicit gaps, absence of unsupported "
                "inference and metadata-derived claim constraints are confirmed."
            ),
            "attestations": {
                "explicit_gaps_reviewed": True,
                "no_unsupported_inference": True,
                "recovered_values_reviewed": True,
            },
            "confirmed_not_recorded_field_ids": [],
        }
    )
    _write_json(checkpoint_path, checkpoint)
    manifest = _read_json(manifest_path)
    files_sha256 = manifest.get("files_sha256")
    if not isinstance(files_sha256, dict):
        raise ValueError("Package manifest has no file hash map")
    files_sha256[record_path.name] = _sha256(record_path)
    files_sha256[checkpoint_path.name] = _sha256(checkpoint_path)
    _write_json(manifest_path, manifest)
    validate_metadata_package(
        workspace_root,
        package,
        require_author_confirmation=True,
    )
    print(f"Confirmed experimental-metadata checkpoint: {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
