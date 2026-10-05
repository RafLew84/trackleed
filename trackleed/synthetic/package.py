"""Generate and read complete, versioned static H6 packages."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.config import BenchmarkConfig, load_config, parse_config, read_json
from trackleed.synthetic.scene import RenderedObservation, render_observation
from trackleed.synthetic.storage import (
    MARKER, PACKAGE_TYPE, STAGING_PREFIX, file_sha256, generator_provenance,
    pixel_descriptor, publish_directory, validate_provenance, write_json,
)
from trackleed.synthetic.tables import FRAME_FIELDS, REFERENCE_FIELDS, SCHEMA, frame_identity, reference_rows


@dataclass(frozen=True)
class PackageReport:
    path: Path
    frames: int
    reference_records: int


def generate_benchmark(config_path: Path, output: Path) -> PackageReport:
    config = load_config(config_path)
    output = output.absolute()
    check_output(output)
    observation = render_observation(config, config.schedule.times_ns[0])
    stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=output.parent))
    try:
        _write_package(stage, config_path, config, observation)
        _verify(stage, allow_staging=True)
        publish_directory(stage, output)
    except BaseException as exc:
        if stage.is_dir():
            try:
                write_json(stage / "failure.json", {"status": "incomplete", "error": str(exc)})
            except OSError:
                pass  # Preserve the original failure, including when the disk is full.
        raise
    return verify_package(output)


def check_output(output: Path) -> None:
    """Check a prospective target without creating any files."""
    output = output.absolute()
    if any(p.name.startswith(STAGING_PREFIX) for p in (output, *output.parents)):
        raise ValueError("Output must not use the reserved staging path")
    if os.path.lexists(output):
        raise ValueError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent must exist: {output.parent}")


def _write_package(
    root: Path, config_path: Path, config: BenchmarkConfig, observation: RenderedObservation,
) -> None:
    (root / MARKER).write_text(PACKAGE_TYPE + "\n", encoding="utf-8")
    (root / "images").mkdir()
    (root / "rendered").mkdir()
    shutil.copyfile(config_path, root / "input.json")
    write_json(root / "scene.json", config.expanded())
    write_json(root / "schema.json", SCHEMA)
    files: dict[str, Any] = {}
    frames, references = [], []
    for index, time_ns in enumerate(config.schedule.times_ns):
        paths = _write_images(root, observation, index, files)
        frames.append(_frame_row(config, observation, index, time_ns, paths, files))
        references.extend(reference_rows(config, observation, index))
    _write_csv(root / "frames.csv", FRAME_FIELDS, frames)
    _write_csv(root / "reference.csv", REFERENCE_FIELDS, references)
    for name in (MARKER, "input.json", "scene.json", "schema.json", "frames.csv", "reference.csv"):
        files[name] = {"sha256": file_sha256(root / name)}
    write_json(root / "manifest.json", {
        "package_type": PACKAGE_TYPE, "schema_version": 1, "status": "complete",
        "scene_id": config.scene.id, "condition_id": config.condition.id,
        "realization_id": config.realization_id, "schedule_id": config.schedule.id,
        "plan": config.plan(), "generator": generator_provenance(), "files": files,
    })


def _write_images(
    root: Path, observation: RenderedObservation, index: int, files: dict[str, Any],
) -> tuple[str, str]:
    png_path, tiff_path = _image_paths(index)
    Image.fromarray(observation.png).save(root / png_path)
    tifffile.imwrite(root / tiff_path, observation.image, metadata=None)
    for name, array in ((png_path, observation.png), (tiff_path, observation.image)):
        files[name] = {"sha256": file_sha256(root / name), **pixel_descriptor(array)}
    return png_path, tiff_path


def _frame_row(
    config: BenchmarkConfig, observation: RenderedObservation, index: int, time_ns: int,
    paths: tuple[str, str], files: dict[str, Any],
) -> dict[str, Any]:
    png, tiff = paths
    return {
        **frame_identity(config, index), "time_ns": time_ns, "energy_ev": config.scene.energy_ev,
        "dt_ns": time_ns - config.schedule.times_ns[index - 1] if index else None,
        "de_ev": 0.0 if index and config.scene.energy_ev is not None else None,
        "png_path": png, "tiff_path": tiff,
        "png_sha256": files[png]["sha256"], "tiff_sha256": files[tiff]["sha256"],
        "png_pixel_sha256": files[png]["pixel_sha256"],
        "tiff_pixel_sha256": files[tiff]["pixel_sha256"], **observation.photometry,
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, bool)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _write_csv(path: Path, fields: dict[str, str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def verify_package(root: Path) -> PackageReport:
    """Read actual files, check fingerprints, pixels and every frame-by-ID record."""
    return _verify(root.absolute(), allow_staging=False)


def _verify(root: Path, *, allow_staging: bool) -> PackageReport:
    if not allow_staging and any(p.name.startswith(STAGING_PREFIX) for p in (root, *root.parents)):
        raise ValueError("Staging is not a published benchmark package")
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Package symlinks are unsupported")
    if not allow_staging and any(p.name.startswith(STAGING_PREFIX) for p in root.resolve().parents):
        raise ValueError("Staging is not a published benchmark package")
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("package_type") != PACKAGE_TYPE:
        raise ValueError("Wrong benchmark package_type")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported benchmark schema_version")
    if manifest.get("status") != "complete":
        raise ValueError("Benchmark package is incomplete")
    required = {"package_type", "schema_version", "status", "scene_id", "condition_id",
                "realization_id", "schedule_id", "plan", "generator", "files"}
    if set(manifest) != required:
        raise ValueError("Invalid manifest fields")
    validate_provenance(manifest["generator"])
    config = parse_config(read_json(root / "scene.json"))
    _verify_identity(manifest, config)
    if (root / MARKER).read_text(encoding="utf-8") != PACKAGE_TYPE + "\n":
        raise ValueError("Invalid benchmark marker")
    if load_config(root / "input.json") != config:
        raise ValueError("Input configuration disagrees with scene")
    if json.dumps(read_json(root / "schema.json"), sort_keys=True) != json.dumps(SCHEMA, sort_keys=True):
        raise ValueError("Benchmark schema does not match version 1")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _expected_files(config):
        raise ValueError("Manifest file inventory disagrees with scene")
    _verify_files(root, files)
    observation = render_observation(config, config.schedule.times_ns[0])
    expected_frames, expected_references = [], []
    for index, time_ns in enumerate(config.schedule.times_ns):
        paths = _image_paths(index)
        _verify_images(root, paths, files, observation)
        expected_frames.append(_frame_row(config, observation, index, time_ns, paths, files))
        expected_references.extend(reference_rows(config, observation, index))
    _verify_csv(root / "frames.csv", FRAME_FIELDS, expected_frames)
    _verify_csv(root / "reference.csv", REFERENCE_FIELDS, expected_references)
    if manifest.get("plan") != config.plan():
        raise ValueError("Manifest plan does not match scene")
    return PackageReport(root, len(expected_frames), len(expected_references))


def _verify_files(root: Path, files: dict[str, Any]) -> None:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Manifest file inventory does not match package")
    for name, descriptor in files.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError(f"Unsafe package path: {name}")
        if not isinstance(descriptor, dict) or descriptor.get("sha256") != file_sha256(path):
            raise ValueError(f"File checksum mismatch: {name}")


def _verify_images(
    root: Path, paths: tuple[str, str], files: dict[str, Any], observation: RenderedObservation,
) -> None:
    png_path, tiff_path = paths
    with Image.open(root / png_path) as image:
        if image.mode != "L":
            raise ValueError(f"PNG must be uint8 grayscale: {png_path}")
        png = np.array(image)
    tiff = tifffile.imread(root / tiff_path)
    for path, actual, expected in ((png_path, png, observation.png),
                                   (tiff_path, tiff, observation.image)):
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"Image does not match scene: {path}")
        if pixel_descriptor(actual) != {k: v for k, v in files[path].items() if k != "sha256"}:
            raise ValueError(f"Pixel checksum mismatch: {path}")


def _verify_csv(path: Path, fields: dict[str, str], expected: list[dict[str, Any]]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        actual = list(reader)
        if reader.fieldnames != list(fields):
            raise ValueError(f"CSV schema mismatch: {path.name}")
    encoded = [{key: _csv_value(value) for key, value in row.items()} for row in expected]
    if actual != encoded:
        raise ValueError(f"CSV records disagree with scene: {path.name}")


def _expected_files(config: BenchmarkConfig) -> set[str]:
    names = {MARKER, "input.json", "scene.json", "schema.json", "frames.csv", "reference.csv"}
    for index in range(len(config.schedule.times_ns)):
        names.update(_image_paths(index))
    return names


def _verify_identity(manifest: dict[str, Any], config: BenchmarkConfig) -> None:
    expected = {"scene_id": config.scene.id, "condition_id": config.condition.id,
                "realization_id": config.realization_id, "schedule_id": config.schedule.id}
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"Manifest {key} disagrees with scene")
    if json.dumps(manifest["plan"], sort_keys=True) != json.dumps(config.plan(), sort_keys=True):
        raise ValueError("Manifest plan disagrees with scene")


def _image_paths(index: int) -> tuple[str, str]:
    return f"images/frame_{index:06d}.png", f"rendered/frame_{index:06d}.tiff"
