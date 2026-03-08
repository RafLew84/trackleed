#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a GIF from an ordered image sequence. "
            "The selected frames are stretched over the requested total duration."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/segmentation_png/test/images"),
        help="Directory with input images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based inclusive start index in sorted file list.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="1-based inclusive end index in sorted file list. Default: last file.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Take every N-th frame in the selected range.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Total GIF duration in seconds.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count (0 = infinite loop).",
    )
    parser.add_argument(
        "--resize-to-first",
        action="store_true",
        help="Resize all frames to the first frame size if dimensions differ.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable GIF optimization in Pillow.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--force-rgb",
        action="store_true",
        help="Always process frames as RGB (disable grayscale-preserving pipeline).",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def collect_images(source: Path) -> List[Path]:
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    images = [
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=natural_sort_key)
    if not images:
        raise SystemExit(f"No supported images found in: {source}")
    return images


def select_range(images: List[Path], start: int, end: int | None, step: int) -> List[Path]:
    if start < 1:
        raise SystemExit("--start must be >= 1")
    if step < 1:
        raise SystemExit("--step must be >= 1")

    last_index = len(images)
    end_index = last_index if end is None else end
    if end_index < start:
        raise SystemExit("--end must be >= --start")
    if end_index > last_index:
        raise SystemExit(
            f"--end is out of range ({end_index}), but only {last_index} files were found."
        )

    selected = images[start - 1 : end_index : step]
    if not selected:
        raise SystemExit("Selected frame range is empty.")
    return selected


def build_frame_durations(total_seconds: float, frame_count: int) -> List[int]:
    if total_seconds <= 0:
        raise SystemExit("--duration must be > 0")

    total_ms = int(round(total_seconds * 1000))
    if total_ms < frame_count:
        total_ms = frame_count

    base = total_ms // frame_count
    remainder = total_ms - base * frame_count
    return [base + 1 if i < remainder else base for i in range(frame_count)]


def to_uint8_gray(arr: np.ndarray) -> np.ndarray:
    """Convert grayscale array to uint8 while preserving smooth gradients."""
    if arr.dtype == np.uint8:
        return arr

    if arr.dtype == np.uint16:
        # Preserve absolute intensity mapping for 16-bit microscopy-like images.
        return (arr >> 8).astype(np.uint8)

    arr_f = arr.astype(np.float32, copy=False)
    min_val = float(np.nanmin(arr_f))
    max_val = float(np.nanmax(arr_f))
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr_f - min_val) / (max_val - min_val)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def make_frame(path: Path, force_rgb: bool) -> Image.Image:
    with Image.open(path) as im:
        arr = np.array(im)

    if not force_rgb:
        if arr.ndim == 2:
            return Image.fromarray(to_uint8_gray(arr), mode="L")
        if arr.ndim == 3 and arr.shape[2] == 1:
            gray = to_uint8_gray(arr[:, :, 0])
            return Image.fromarray(gray, mode="L")

    return Image.fromarray(arr).convert("RGB")


def load_frames(paths: List[Path], resize_to_first: bool, force_rgb: bool) -> List[Image.Image]:
    frames: List[Image.Image] = []
    first_size = None
    first_mode = None

    for path in paths:
        frame = make_frame(path, force_rgb=force_rgb)

        if first_size is None:
            first_size = frame.size
            first_mode = frame.mode
        elif frame.size != first_size:
            if resize_to_first:
                frame = frame.resize(first_size, Image.Resampling.LANCZOS)
            else:
                raise SystemExit(
                    f"Frame size mismatch for '{path.name}': {frame.size} != {first_size}. "
                    "Use --resize-to-first to force a single size."
                )

        if first_mode is not None and frame.mode != first_mode:
            if first_mode == "L":
                frame = frame.convert("L")
            else:
                frame = frame.convert("RGB")

        frames.append(frame)

    return frames


def grayscale_palette() -> List[int]:
    palette: List[int] = []
    for i in range(256):
        palette.extend((i, i, i))
    return palette


def l_frame_to_palette(frame: Image.Image, palette: List[int]) -> Image.Image:
    arr = np.array(frame, dtype=np.uint8)
    p_frame = Image.fromarray(arr, mode="P")
    p_frame.putpalette(palette)
    return p_frame


def main() -> None:
    args = parse_args()
    images = collect_images(args.source)
    selected = select_range(images, args.start, args.end, args.step)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file exists: {args.output}. Use --overwrite to replace it."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames = load_frames(
        selected,
        resize_to_first=args.resize_to_first,
        force_rgb=args.force_rgb,
    )
    durations = build_frame_durations(args.duration, len(frames))

    if frames[0].mode == "L":
        palette = grayscale_palette()
        first = l_frame_to_palette(frames[0], palette)
        rest = [l_frame_to_palette(frame, palette) for frame in frames[1:]]
    else:
        first = frames[0]
        rest = frames[1:]

    first.save(
        args.output,
        save_all=True,
        append_images=rest,
        duration=durations,
        loop=args.loop,
        optimize=args.optimize,
        disposal=2,
    )

    total_ms = sum(durations)
    effective_seconds = total_ms / 1000.0
    fps = len(frames) / effective_seconds
    print(f"Saved GIF: {args.output.resolve()}")
    print(f"Frames: {len(frames)}")
    print(f"Range: {selected[0].name} -> {selected[-1].name}")
    print(f"Duration: {effective_seconds:.3f}s")
    print(f"Effective FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
