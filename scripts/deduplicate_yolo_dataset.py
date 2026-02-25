#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate a YOLO dataset by image content hash and keep one sample per hash group. "
            "The kept sample is chosen by the largest number of labels; ties are broken by filename."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/train_data_dimmed_mixed"),
        help="Source dataset directory with images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/train_data_dimmed_mixed_dedup"),
        help="Destination directory for deduplicated images/labels.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV path. Default: <dest>/dedup_manifest.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_non_empty_lines(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def ensure_destination(dest: Path, resume: bool) -> None:
    if dest.exists():
        if not resume and any(dest.iterdir()):
            raise SystemExit(
                f"Destination '{dest}' exists and is not empty. "
                f"Use --resume or choose a different destination."
            )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)


def collect_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_best_in_group(group: List[Path], labels_dir: Path) -> Tuple[Path, int]:
    # Primary key: label count descending.
    # Tie-break: lexical filename for deterministic output.
    candidates: List[Tuple[int, str, Path]] = []
    for image_path in group:
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_count = count_non_empty_lines(label_path)
        candidates.append((label_count, image_path.name, image_path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_count, _, best_path = candidates[0]
    return best_path, best_count


def deduplicate(source: Path, dest: Path, manifest_path: Path) -> None:
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Source '{source}' must contain both images/ and labels/ directories.")

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise SystemExit(f"No images found in '{images_dir}'.")

    groups: Dict[str, List[Path]] = defaultdict(list)
    for image_path in image_paths:
        groups[file_md5(image_path)].append(image_path)

    kept_rows: List[Dict[str, str | int]] = []
    total_labels = 0
    for image_hash, group in groups.items():
        best_image, best_label_count = choose_best_in_group(group, labels_dir)
        best_label = labels_dir / f"{best_image.stem}.txt"
        if not best_label.exists():
            # Skip groups without labels; caller can inspect manifest.
            continue

        dest_image = dest / "images" / best_image.name
        dest_label = dest / "labels" / best_label.name
        shutil.copy2(best_image, dest_image)
        shutil.copy2(best_label, dest_label)
        total_labels += max(best_label_count, 0)

        kept_rows.append(
            {
                "hash": image_hash,
                "group_size": len(group),
                "kept_image": best_image.name,
                "kept_label": best_label.name,
                "kept_label_count": best_label_count,
                "group_members": "|".join(sorted(p.name for p in group)),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hash",
                "group_size",
                "kept_image",
                "kept_label",
                "kept_label_count",
                "group_members",
            ],
        )
        writer.writeheader()
        writer.writerows(kept_rows)

    unique_count = len(kept_rows)
    original_count = len(image_paths)
    removed = original_count - unique_count
    avg_labels = total_labels / unique_count if unique_count else 0.0

    print(f"Source images:         {original_count}")
    print(f"Kept after dedup:      {unique_count}")
    print(f"Removed duplicates:    {removed}")
    print(f"Total labels kept:     {total_labels}")
    print(f"Avg labels / image:    {avg_labels:.2f}")
    print(f"Manifest:              {manifest_path}")


def main() -> None:
    args = parse_args()
    ensure_destination(args.dest, resume=args.resume)
    manifest = args.manifest or (args.dest / "dedup_manifest.csv")
    deduplicate(args.source, args.dest, manifest)


if __name__ == "__main__":
    main()
