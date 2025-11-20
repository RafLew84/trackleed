#!/usr/bin/env python3
"""
Convert the synthetic segmentation dataset (TIFF) into PNG format while preserving
the train/val/test split structure.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert segmentation TIFF dataset to PNG.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/segmentation"),
        help="Root directory with train/val/test splits in TIFF format.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/segmentation_png"),
        help="Destination root where PNG files will be written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append/update destination instead of requiring it to be empty.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Subset of dataset splits to convert (default: all).",
    )
    return parser.parse_args()


def ensure_destination(dest: Path, resume: bool) -> None:
    if dest.exists():
        if resume:
            return
        if any(dest.iterdir()):
            raise SystemExit(f"Destination '{dest}' already exists and is not empty. Pass --resume to append.")
    dest.mkdir(parents=True, exist_ok=True)


def iter_tiffs(dir_path: Path) -> Iterable[Path]:
    return sorted(dir_path.glob("*.tiff"))


def convert_image(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint16 = np.round(arr * 65535.0).astype(np.uint16)
    return Image.fromarray(arr_uint16, mode="I;16")


def convert_mask(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def ensure_subdirs(dest: Path, split: str) -> dict[str, Path]:
    split_root = dest / split
    paths = {
        "images": split_root / "images",
        "masks": split_root / "masks",
        "metadata": split_root / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_metadata(source_meta: Path, dest_meta: Path) -> None:
    for json_file in source_meta.glob("*.json"):
        target = dest_meta / json_file.name
        if target.exists():
            continue
        shutil.copy2(json_file, target)


def convert_split(source_root: Path, dest_root: Path, split: str) -> None:
    src_images = source_root / split / "images"
    src_masks = source_root / split / "masks"
    src_meta = source_root / split / "metadata"
    paths = ensure_subdirs(dest_root, split)
    dest_images = paths["images"]
    dest_masks = paths["masks"]

    image_files = list(iter_tiffs(src_images))
    mask_files = list(iter_tiffs(src_masks))
    print(f"[{split}] converting {len(image_files)} images and {len(mask_files)} masks")

    for path in image_files:
        dest_file = dest_images / (path.stem + ".png")
        if dest_file.exists():
            continue
        array = tifffile.imread(path)
        img = convert_image(array)
        img.save(dest_file)

    for path in mask_files:
        dest_file = dest_masks / (path.stem + ".png")
        if dest_file.exists():
            continue
        array = tifffile.imread(path)
        img = convert_mask(array)
        img.save(dest_file)

    if src_meta.exists():
        copy_metadata(src_meta, paths["metadata"])


def main() -> None:
    args = parse_args()
    ensure_destination(args.dest, resume=args.resume)
    for split in args.splits:
        convert_split(args.source, args.dest, split)
    print("Conversion finished.")


if __name__ == "__main__":
    main()
