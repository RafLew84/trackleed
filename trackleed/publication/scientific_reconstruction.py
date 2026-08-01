"""Headless reconstruction helpers for the frozen scientific core."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from manuscript.publication_inventory import build_inventory


@dataclass(frozen=True)
class ReaderFacingArtifactReport:
    bundle_count: int
    source_file_count: int
    output_file_count: int


class ScientificReconstructionError(ValueError):
    """Raised when a declared scientific-core reconstruction step fails."""


def _require_workspace(workspace_root: Path) -> Path:
    workspace_root = workspace_root.resolve()
    if workspace_root != build_inventory.REPO_ROOT.resolve():
        raise ValueError(
            "Reader-facing reconstruction must run from its own repository root"
        )
    return workspace_root


def _display_builder_paths(workspace_root: Path) -> tuple[Path, ...]:
    inventory = build_inventory.load_inventory()
    items = inventory.get("items")
    if (
        inventory.get("schema_version") != "1.0"
        or not isinstance(items, list)
        or tuple(item.get("id") for item in items) != build_inventory.EXPECTED_IDS
        or build_inventory.approved_inventory_sha256(items)
        != build_inventory.APPROVED_INVENTORY_SHA256
    ):
        raise ScientificReconstructionError(
            "Display-builder inventory differs from the approved contract"
        )
    allowed_roots = (
        workspace_root / "manuscript" / "figures",
        workspace_root / "manuscript" / "tables",
    )
    builders: list[Path] = []
    for item in items:
        relative_path = item.get("builder_path")
        if not isinstance(relative_path, str):
            raise ScientificReconstructionError("Display builder path is missing")
        path = (workspace_root / relative_path).resolve()
        if (
            not path.is_file()
            or not path.is_relative_to(workspace_root)
            or not any(path.is_relative_to(root) for root in allowed_roots)
        ):
            raise ScientificReconstructionError(
                f"Display builder is outside the approved roots: {relative_path}"
            )
        builders.append(path)
    if len(builders) != 20 or len(set(builders)) != len(builders):
        raise ScientificReconstructionError(
            "Display-builder inventory must contain twenty unique paths"
        )
    return tuple(builders)


def _headless_environment(workspace_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MPLBACKEND": "Agg",
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(workspace_root), current_python_path) if part
    )
    return environment


def _run_builder(
    path: Path,
    workspace_root: Path,
    environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=workspace_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        details = (completed.stdout + completed.stderr).strip()
        raise ScientificReconstructionError(
            f"Scientific-core builder failed: {path}\n{details}"
        )


def _artifact_snapshot(
    workspace_root: Path,
) -> tuple[tuple[Path, ...], dict[Path, bytes]]:
    inventory = build_inventory.load_inventory()
    roots = tuple(
        sorted(
            {
                (workspace_root / str(item["artifact_path"])).resolve()
                for item in inventory["items"]
            },
            key=str,
        )
    )
    standalone = (
        workspace_root / "manuscript" / "figures-and-tables.md",
        workspace_root / "manuscript" / "manuscript.md",
        workspace_root / "manuscript" / "publication_inventory" / "manifest.json",
        workspace_root
        / "manuscript"
        / "publication_inventory"
        / "semantic_manifest.json",
        workspace_root / "manuscript" / "supplementary-material.md",
    )
    paths = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    paths.update(path.resolve() for path in standalone)
    return roots, {path: path.read_bytes() for path in paths}


def _restore_artifact_snapshot(
    roots: tuple[Path, ...],
    snapshot: dict[Path, bytes],
) -> None:
    current_paths = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    for path in current_paths - set(snapshot):
        path.unlink()
    for path, content in snapshot.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def rebuild_reader_facing_artifacts(
    workspace_root: Path,
) -> ReaderFacingArtifactReport:
    """Transactionally rebuild every approved main/SI display and manifest."""

    workspace_root = _require_workspace(workspace_root)
    validate_reader_facing_artifacts(workspace_root)
    roots, snapshot = _artifact_snapshot(workspace_root)
    environment = _headless_environment(workspace_root)
    try:
        for builder_path in _display_builder_paths(workspace_root):
            _run_builder(builder_path, workspace_root, environment)
        _run_builder(
            workspace_root
            / "manuscript"
            / "publication_inventory"
            / "build_inventory.py",
            workspace_root,
            environment,
        )
        return validate_reader_facing_artifacts(workspace_root)
    finally:
        _restore_artifact_snapshot(roots, snapshot)


def validate_reader_facing_artifacts(
    workspace_root: Path,
) -> ReaderFacingArtifactReport:
    """Validate all final displays and their committed integration artifact."""

    workspace_root = _require_workspace(workspace_root)
    inventory = build_inventory.load_inventory()
    with TemporaryDirectory() as temporary:
        generated_root = Path(temporary)
        generated_index = generated_root / "figures-and-tables.md"
        generated_manifest = generated_root / "manifest.json"
        generated_semantic_manifest = generated_root / "semantic_manifest.json"
        report = build_inventory.validate_inventory(inventory)
        build_inventory.build_inventory_artifact(
            inventory,
            output_path=generated_index,
            manifest_path=generated_manifest,
            semantic_manifest_path=generated_semantic_manifest,
        )
        committed_index = workspace_root / "manuscript" / "figures-and-tables.md"
        committed_manifest = (
            workspace_root / "manuscript" / "publication_inventory" / "manifest.json"
        )
        committed_semantic_manifest = (
            workspace_root
            / "manuscript"
            / "publication_inventory"
            / "semantic_manifest.json"
        )
        if generated_index.read_bytes() != committed_index.read_bytes():
            raise ValueError("Final publication display index changed")
        if generated_manifest.read_bytes() != committed_manifest.read_bytes():
            raise ValueError("Final publication inventory manifest changed")
        if (
            generated_semantic_manifest.read_bytes()
            != committed_semantic_manifest.read_bytes()
        ):
            raise ValueError("Final publication semantic inventory changed")
    return ReaderFacingArtifactReport(
        bundle_count=int(report["bundle_count"]),
        source_file_count=int(report["source_file_count"]),
        output_file_count=int(report["output_file_count"]),
    )


__all__ = [
    "ReaderFacingArtifactReport",
    "ScientificReconstructionError",
    "rebuild_reader_facing_artifacts",
    "validate_reader_facing_artifacts",
]
