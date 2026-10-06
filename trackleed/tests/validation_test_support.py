from __future__ import annotations

import json
from pathlib import Path
import subprocess


def initialize_repository(root: Path, files: dict[str, str]) -> None:
    run_git(root, "init")
    run_git(root, "config", "user.email", "validation@example.invalid")
    run_git(root, "config", "user.name", "Validation Test")
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    run_git(root, "add", ".")
    run_git(root, "commit", "-m", "baseline")


def write_manifest(
    root: Path,
    fixed_point: str,
    files: dict[str, object],
) -> Path:
    path = root / "code_size_baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixed_point": fixed_point,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return path


def run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def initialize_workflow_repository(root: Path, files: dict[str, str]) -> None:
    """Create the required generator scope in small public-validator fixtures."""
    initialize_repository(root, {
        "mypy-generator.ini": "[mypy]\nfollow_imports = silent\npython_version = 3.12\n",
        "trackleed/synthetic/__init__.py": "",
        "scripts/generate_tracking_benchmark.py": "",
        "scripts/export_segmentation_to_png.py": "",
        **files,
    })
