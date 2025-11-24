from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageSequence


def extract_gif_frames(gif_path: Path, dest_root: Path | None = None, sample_ms: int = 100) -> Tuple[Path, List[Path]]:
    """
    Extract frames from a GIF at a fixed time interval and save as PNGs.

    Args:
        gif_path: input GIF file.
        dest_root: destination directory to place extracted PNGs (created if needed).
        sample_ms: sampling interval in milliseconds (default 100 ms).

    Returns:
        (output_dir, list_of_saved_paths)
    """
    gif_path = Path(gif_path)
    if dest_root is None:
        dest_root = gif_path.parent / f"{gif_path.stem}_frames"
    dest_root.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    with Image.open(gif_path) as im:
        elapsed = 0
        next_sample = 0
        frame_out_idx = 0
        for frame in ImageSequence.Iterator(im):
            duration = frame.info.get("duration", 0) or 0  # ms
            if elapsed >= next_sample:
                gray = frame.convert("L")
                out_name = f"{gif_path.stem}_{frame_out_idx:05d}.png"
                out_path = dest_root / out_name
                gray.save(out_path)
                saved.append(out_path)
                frame_out_idx += 1
                next_sample += sample_ms
            # Guard: if duration is zero, step by sample_ms to avoid infinite loop
            elapsed += duration if duration > 0 else sample_ms
    return dest_root, saved
