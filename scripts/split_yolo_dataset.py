#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a flat YOLO dataset (images/ + labels/) into train/val/test "
            "and generate data.yaml."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/train_data_dimmed_mixed_dedup"),
        help="Source dataset directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/train_data_dimmed_mixed_dedup_split"),
        help="Destination split dataset root.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before split.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="spot",
        help="Single class name for YOLO data.yaml (class id 0).",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="data.yaml",
        help="YAML filename written under destination root.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Split ratios must sum to 1.0, got {total:.6f}")
    for name, value in (
        ("train-ratio", train_ratio),
        ("val-ratio", val_ratio),
        ("test-ratio", test_ratio),
    ):
        if value < 0.0:
            raise SystemExit(f"{name} must be non-negative, got {value}")


def ensure_clean_dest(dest: Path, resume: bool) -> None:
    if dest.exists():
        if not resume and any(dest.iterdir()):
            raise SystemExit(
                f"Destination '{dest}' exists and is not empty. "
                f"Use --resume or remove it."
            )
    for split in ("train", "val", "test"):
        (dest / split / "images").mkdir(parents=True, exist_ok=True)
        (dest / split / "labels").mkdir(parents=True, exist_ok=True)


def collect_pairs(source: Path) -> List[Tuple[Path, Path]]:
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Source '{source}' must contain images/ and labels/ directories.")

    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        pairs.append((image_path, label_path))

    if not pairs:
        raise SystemExit("No valid image-label pairs found.")
    return pairs


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Safety correction for rounding edge cases
    if n_test < 0:
        n_test = 0
        overflow = n_train + n_val - n
        if n_val >= overflow:
            n_val -= overflow
        else:
            n_train -= (overflow - n_val)
            n_val = 0

    if n_train == 0 and n > 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    return n_train, n_val, n_test


def copy_split(dest: Path, split: str, items: List[Tuple[Path, Path]]) -> None:
    images_out = dest / split / "images"
    labels_out = dest / split / "labels"
    for image_path, label_path in items:
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)


def write_yaml(dest: Path, yaml_name: str, class_name: str) -> Path:
    yaml_path = dest / yaml_name
    yaml_path.write_text(
        (
            f"path: {dest.resolve()}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"names:\n"
            f"  0: {class_name}\n"
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    ensure_clean_dest(args.dest, resume=args.resume)

    pairs = collect_pairs(args.source)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train, n_val, n_test = split_indices(n, args.train_ratio, args.val_ratio, args.test_ratio)

    train_items = pairs[:n_train]
    val_items = pairs[n_train : n_train + n_val]
    test_items = pairs[n_train + n_val : n_train + n_val + n_test]

    copy_split(args.dest, "train", train_items)
    copy_split(args.dest, "val", val_items)
    copy_split(args.dest, "test", test_items)

    yaml_path = write_yaml(args.dest, args.yaml_name, args.class_name)

    print(f"Source pairs: {n}")
    print(f"Train pairs:  {len(train_items)}")
    print(f"Val pairs:    {len(val_items)}")
    print(f"Test pairs:   {len(test_items)}")
    print(f"YAML:         {yaml_path}")


if __name__ == "__main__":
    main()
