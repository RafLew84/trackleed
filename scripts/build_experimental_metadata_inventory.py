"""Freeze embedded TIFF metadata for the publication acquisition series."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


SERIES = (
    ("cu111_thermal_cycle", "data/25w28_ThExten_56eV_UD", 701),
    ("ag111_scan_a", "data/26a21_ScanE", 35),
    ("ag111_scan_b", "data/26a21b_ScanE", 46),
    ("ag111_memory_analysis", "data/26a24_ScanE", 69),
)
DEFAULT_OUTPUT = Path(
    "manuscript/experimental_metadata/tiff_metadata_inventory.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: object) -> object:
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def build_inventory(workspace_root: Path) -> dict[str, object]:
    workspace_root = workspace_root.resolve()
    series_records = []
    for series_id, relative_directory, expected_count in SERIES:
        directory = workspace_root / relative_directory
        paths = sorted((*directory.glob("*.tif"), *directory.glob("*.tiff")))
        if len(paths) != expected_count:
            raise ValueError(
                f"{series_id}: expected {expected_count} TIFF files, found {len(paths)}"
            )
        file_records = []
        for path in paths:
            with Image.open(path) as image:
                file_records.append(
                    {
                        "source_path": path.relative_to(workspace_root).as_posix(),
                        "source_sha256": _sha256(path),
                        "image_width": image.width,
                        "image_length": image.height,
                        "mode": image.mode,
                        "bits_per_sample": _scalar(
                            image.tag_v2.get(
                                258,
                                8 if image.mode == "L" else None,
                            )
                        ),
                        "samples_per_pixel": _scalar(image.tag_v2.get(277, 1)),
                        "compression": _scalar(image.tag_v2.get(259, 1)),
                        "software": _scalar(image.tag_v2.get(305)),
                    }
                )
        series_records.append(
            {
                "series_id": series_id,
                "source_directory": relative_directory,
                "expected_count": expected_count,
                "files": file_records,
            }
        )
    return {
        "schema_version": 1,
        "scope": "publication_acquisition_tiff_embedded_metadata",
        "series": series_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = workspace_root / output
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen inventory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_inventory(workspace_root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Built TIFF metadata inventory: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
