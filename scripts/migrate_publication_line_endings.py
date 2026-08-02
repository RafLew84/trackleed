"""Migrate publication text and embedded CRLF hashes to the LF contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

from trackleed.publication.publication_line_endings import (
    load_utf8_documents,
    migrate_line_ending_hashes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--tracked-files", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def _tracked_paths(workspace_root: Path, tracked_files: Path | None) -> list[str]:
    if tracked_files is not None:
        return [
            line
            for line in tracked_files.read_text(encoding="utf-8").splitlines()
            if line
        ]
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=workspace_root,
        capture_output=True,
        check=True,
    )
    return [path for path in completed.stdout.decode().split("\0") if path]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace_root = arguments.workspace_root.resolve()
    paths = _tracked_paths(workspace_root, arguments.tracked_files)
    original = load_utf8_documents(workspace_root, paths)
    migration = migrate_line_ending_hashes(original)
    if arguments.apply:
        for relative_path in migration.changed_paths:
            (workspace_root / relative_path).write_bytes(
                migration.documents[relative_path]
            )
    report = {
        "schema_version": 1,
        "applied": arguments.apply,
        "changed_path_count": len(migration.changed_paths),
        "iterations": migration.iterations,
        "changed_paths": [
            {
                "path": path,
                "before_sha256": hashlib.sha256(original[path]).hexdigest(),
                "after_sha256": hashlib.sha256(migration.documents[path]).hexdigest(),
            }
            for path in migration.changed_paths
        ],
        "fingerprint_replaced": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
