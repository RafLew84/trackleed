"""Generate and read complete, versioned synthetic benchmark packages."""
from __future__ import annotations

from collections.abc import Iterator
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

from trackleed.synthetic.apparatus import prepare_apparatus
from trackleed.synthetic.config import BenchmarkConfig, Schedule, load_config, parse_config, read_json
from trackleed.synthetic.scene import RenderedObservation, render_observation
from trackleed.synthetic.instances import mask_paths
from trackleed.synthetic.coverage import build_coverage
from trackleed.synthetic.storage import (
    MARKER, PACKAGE_TYPE, STAGING_PREFIX, file_sha256, generator_provenance,
    pixel_descriptor, publish_directory, validate_provenance, write_json,
)
from trackleed.synthetic.tables import FRAME_FIELDS, frame_identity, reference_fields, reference_rows, schema_for


@dataclass(frozen=True)
class PackageReport:
    path: Path
    frames: int
    reference_records: int


def generate_benchmark(config_path: Path, output: Path) -> PackageReport:
    config = load_config(config_path)
    output = output.absolute()
    check_output(output)
    stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=output.parent))
    try:
        _write_package(stage, config_path, config)
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
    root: Path, config_path: Path, config: BenchmarkConfig,
) -> None:
    (root / MARKER).write_text(PACKAGE_TYPE + "\n", encoding="utf-8")
    shutil.copyfile(config_path, root / "input.json")
    write_json(root / "scene.json", config.expanded())
    write_json(root / "schema.json", schema_for(config.schema_version))
    files: dict[str, Any] = {}
    _write_binary_images(root, _apparatus_images(config), files)
    frames, references = [], []
    for schedule, index in _scheduled_frames(config):
        observation = render_observation(config, **schedule.coordinates(index))
        paths = _write_images(root, observation, _image_paths(config, schedule, index), files)
        _write_masks(root, observation, schedule, index, files)
        frames.append(_frame_row(config, observation, schedule, index, paths, files))
        references.extend(reference_rows(config, observation, index, schedule))
    _write_csv(root / "frames.csv", FRAME_FIELDS, frames)
    _write_csv(root / "reference.csv", reference_fields(config.schema_version), references)
    if config.schema_version >= 4:
        write_json(root / "coverage.json", build_coverage(config, references))
        files["coverage.json"] = {"sha256": file_sha256(root / "coverage.json")}
    for name in (MARKER, "input.json", "scene.json", "schema.json", "frames.csv", "reference.csv"):
        files[name] = {"sha256": file_sha256(root / name)}
    write_json(root / "manifest.json", {
        "package_type": PACKAGE_TYPE, "schema_version": config.schema_version, "status": "complete",
        **_identity(config),
        "plan": config.plan(), "generator": generator_provenance(), "files": files,
    })


def _write_images(
    root: Path, observation: RenderedObservation, paths: tuple[str, str], files: dict[str, Any],
) -> tuple[str, str]:
    png_path, tiff_path = paths
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation.png).save(root / png_path)
    tifffile.imwrite(root / tiff_path, observation.image, metadata=None)
    for name, array in ((png_path, observation.png), (tiff_path, observation.image)):
        files[name] = {"sha256": file_sha256(root / name), **pixel_descriptor(array)}
    return png_path, tiff_path


def _frame_row(
    config: BenchmarkConfig, observation: RenderedObservation, schedule: Schedule, index: int,
    paths: tuple[str, str], files: dict[str, Any],
) -> dict[str, Any]:
    png, tiff = paths
    coordinates = schedule.coordinates(index)
    time_ns = coordinates["time_ns"]
    energy = config.energy_at(**coordinates)
    previous_energy = config.energy_at(**schedule.coordinates(index - 1)) if index else None
    return {
        **frame_identity(schedule, index), "time_ns": time_ns, "energy_ev": energy,
        "dt_ns": time_ns - schedule.times_ns[index - 1] if index and time_ns is not None else None,
        "de_ev": ((0.0 if config.schema_version == 1 else energy - previous_energy)
                  if energy is not None and previous_energy is not None else None),
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
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (1, 2, 3, 4, 5):
        raise ValueError("Unsupported benchmark schema_version")
    if manifest.get("status") != "complete":
        raise ValueError("Benchmark package is incomplete")
    required = {"package_type", "schema_version", "status", "scene_id", "condition_id",
                "realization_id", "plan", "generator", "files"}
    required.add("schedule_id" if manifest["schema_version"] == 1 else "schedule_ids")
    if set(manifest) != required:
        raise ValueError("Invalid manifest fields")
    validate_provenance(manifest["generator"])
    config = parse_config(read_json(root / "scene.json"))
    _verify_identity(manifest, config)
    if (root / MARKER).read_text(encoding="utf-8") != PACKAGE_TYPE + "\n":
        raise ValueError("Invalid benchmark marker")
    if load_config(root / "input.json") != config:
        raise ValueError("Input configuration disagrees with scene")
    if json.dumps(read_json(root / "schema.json"), sort_keys=True) != json.dumps(schema_for(config.schema_version), sort_keys=True):
        raise ValueError("Benchmark schema does not match its version")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _expected_files(config):
        raise ValueError("Manifest file inventory disagrees with scene")
    _verify_files(root, files)
    _verify_binary_images(root, _apparatus_images(config), files)
    expected_frames, expected_references = [], []
    for schedule, index in _scheduled_frames(config):
        observation = render_observation(config, **schedule.coordinates(index))
        paths = _image_paths(config, schedule, index)
        _verify_images(root, paths, files, observation)
        _verify_masks(root, observation, schedule, index, files)
        expected_frames.append(_frame_row(config, observation, schedule, index, paths, files))
        expected_references.extend(reference_rows(config, observation, index, schedule))
    _verify_csv(root / "frames.csv", FRAME_FIELDS, expected_frames)
    _verify_csv(root / "reference.csv", reference_fields(config.schema_version), expected_references)
    if config.schema_version >= 4:
        expected_coverage = build_coverage(config, expected_references)
        if json.dumps(read_json(root / "coverage.json"), sort_keys=True) != json.dumps(expected_coverage, sort_keys=True):
            raise ValueError("Event coverage disagrees with scene and observations")
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
    if config.schema_version >= 4:
        names.add("coverage.json")
    if config.schema_version >= 5:
        names.update({"apparatus/screen.png", "apparatus/occluders.png"})
    for schedule, index in _scheduled_frames(config):
        names.update(_image_paths(config, schedule, index))
        if config.export_masks:
            for site in config.scene.sites(config.scene.lattice_scale_px):
                names.update(mask_paths(schedule.id, index, site.reflection_id))
    return names


def _verify_identity(manifest: dict[str, Any], config: BenchmarkConfig) -> None:
    if manifest["schema_version"] != config.schema_version:
        raise ValueError("Manifest schema_version disagrees with scene")
    for key, value in _identity(config).items():
        if manifest[key] != value:
            raise ValueError(f"Manifest {key} disagrees with scene")
    if json.dumps(manifest["plan"], sort_keys=True) != json.dumps(config.plan(), sort_keys=True):
        raise ValueError("Manifest plan disagrees with scene")


def _identity(config: BenchmarkConfig) -> dict[str, Any]:
    schedules = ({"schedule_id": config.schedule.id} if config.schema_version == 1
                 else {"schedule_ids": [schedule.id for schedule in config.schedules]})
    return {"scene_id": config.scene.id, "condition_id": config.condition.id,
            "realization_id": config.realization_id, **schedules}


def _scheduled_frames(config: BenchmarkConfig) -> Iterator[tuple[Schedule, int]]:
    for schedule in config.schedules:
        for index in range(schedule.count):
            yield schedule, index


def _image_paths(config: BenchmarkConfig, schedule: Schedule, index: int) -> tuple[str, str]:
    prefix = "" if config.schema_version == 1 else f"sequences/{schedule.id}/"
    return f"{prefix}images/frame_{index:06d}.png", f"{prefix}rendered/frame_{index:06d}.tiff"


def _mask_images(observation: RenderedObservation, schedule: Schedule, index: int) -> Iterator[tuple[str, np.ndarray]]:
    for instance in observation.instances:
        paths = mask_paths(schedule.id, index, instance.reflection_id)
        yield from zip(paths, (instance.geometry, instance.signal))


def _write_masks(root: Path, observation: RenderedObservation, schedule: Schedule,
                 index: int, files: dict[str, Any]) -> None:
    _write_binary_images(root, _mask_images(observation, schedule, index), files)


def _write_binary_images(root: Path, images: Iterator[tuple[str, np.ndarray]], files: dict[str, Any]) -> None:
    for path, array in images:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(root / path)
        files[path] = {"sha256": file_sha256(root / path), **pixel_descriptor(array)}


def _verify_masks(root: Path, observation: RenderedObservation, schedule: Schedule,
                  index: int, files: dict[str, Any]) -> None:
    _verify_binary_images(root, _mask_images(observation, schedule, index), files)


def _verify_binary_images(root: Path, images: Iterator[tuple[str, np.ndarray]], files: dict[str, Any]) -> None:
    for path, expected in images:
        with Image.open(root / path) as image:
            if image.mode != "L":
                raise ValueError(f"Mask must be uint8 grayscale: {path}")
            actual = np.array(image)
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"Mask does not match scene: {path}")
        if pixel_descriptor(actual) != {k: v for k, v in files[path].items() if k != "sha256"}:
            raise ValueError(f"Mask pixel checksum mismatch: {path}")


def _apparatus_images(config: BenchmarkConfig) -> Iterator[tuple[str, np.ndarray]]:
    if config.schema_version >= 5:
        apparatus = prepare_apparatus(config.condition.apparatus, config.scene.image_size,
                                      config.condition.background)
        yield "apparatus/screen.png", apparatus.screen.astype(np.uint8) * 255
        yield "apparatus/occluders.png", apparatus.occluders.astype(np.uint8) * 255
