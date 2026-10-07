"""Publish and verify the exact static inputs used by a benchmark package."""
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from trackleed.synthetic.backgrounds import ImportedBackground, PreparedBackground
from trackleed.synthetic.config import BenchmarkConfig, read_json
from trackleed.synthetic.storage import file_sha256, pixel_descriptor, write_json


def packaged_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """Keep original paths in input.json and bind scene.json to retained sources."""
    model = config.condition.background_model
    if not isinstance(model, ImportedBackground):
        return config
    model = replace(model, path="inputs/background.source",
                    screen_mask="inputs/screen_mask.source" if model.screen_mask else None,
                    occluder_mask="inputs/occluder_mask.source" if model.occluder_mask else None)
    return replace(config, condition=replace(config.condition, background_model=model))


def background_files(config: BenchmarkConfig) -> set[str]:
    if config.schema_version < 6:
        return set()
    names = {"background/prepared.tiff", "background.json"}
    model = config.condition.background_model
    if isinstance(model, ImportedBackground):
        names.add("inputs/background.source")
        if model.screen_mask:
            names.add("inputs/screen_mask.source")
        if model.occluder_mask:
            names.add("inputs/occluder_mask.source")
    return names


def write_background(root: Path, prepared: PreparedBackground, files: dict[str, Any]) -> None:
    for source in prepared.sources:
        path = root / source.package_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.content)
        files[source.package_path] = {"sha256": file_sha256(path), **pixel_descriptor(source.pixels)}
    name = "background/prepared.tiff"
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(root / name, prepared.apparatus.background, metadata=None)
    files[name] = {"sha256": file_sha256(root / name), **pixel_descriptor(prepared.apparatus.background)}
    write_json(root / "background.json", prepared.metadata)
    files["background.json"] = {"sha256": file_sha256(root / "background.json")}


def verify_background(root: Path, prepared: PreparedBackground, files: dict[str, Any]) -> None:
    if json.dumps(read_json(root / "background.json"), sort_keys=True) != json.dumps(prepared.metadata, sort_keys=True):
        raise ValueError("Background provenance disagrees with prepared inputs")
    name = "background/prepared.tiff"
    actual = tifffile.imread(root / name)
    expected = prepared.apparatus.background
    if actual.dtype != np.float64 or not np.array_equal(actual, expected):
        raise ValueError("Prepared background disagrees with scene and sources")
    if files[name] != {"sha256": file_sha256(root / name), **pixel_descriptor(actual)}:
        raise ValueError("Prepared background pixel checksum mismatch")
    for source in prepared.sources:
        expected_descriptor = {"sha256": file_sha256(root / source.package_path), **pixel_descriptor(source.pixels)}
        if files[source.package_path] != expected_descriptor:
            raise ValueError(f"Source pixel checksum mismatch: {source.role}")
