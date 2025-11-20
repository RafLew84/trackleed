#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from trackleed.ai.synth import SynthConfig, generate_sequence


PROFILE_NAMES = ("moire_static", "moire_blinking", "global_drift", "inhomogeneous_bg")


@dataclass
class ProfileSpec:
    name: str
    apply: Callable[[SynthConfig, np.random.Generator], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate segmentation dataset with varied synthetic LEED sequences.")
    parser.add_argument("--output", type=Path, default=Path("data/segmentation"), help="Root output directory (train/val/test subfolders will be created).")
    parser.add_argument("--frames", type=int, default=500, help="Frames per generated sequence.")
    parser.add_argument("--frame-size", type=int, default=512, help="Spatial resolution of generated frames.")
    parser.add_argument("--train-seqs", type=int, default=60, help="Number of unique sequences to generate for the train split in this run.")
    parser.add_argument("--val-seqs", type=int, default=6, help="Number of unique sequences to generate for the validation split in this run.")
    parser.add_argument("--test-seqs", type=int, default=6, help="Number of unique sequences to generate for the test split in this run.")
    parser.add_argument("--seed", type=int, default=202401, help="Seed for sampling generator parameters.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing dataset (do not require an empty output directory, continue sequence numbering).",
    )
    parser.add_argument("--image-format", choices=("tiff", "png"), default="tiff", help="Image format for generated frames.")
    parser.add_argument("--no-masks", action="store_true", help="Skip saving binary masks.")
    parser.add_argument("--yolo-labels", action="store_true", help="Save YOLO annotation files (labels).")
    parser.add_argument("--yolo-box-scale", type=float, default=3.0, help="Bounding-box size multiplier relative to spot sigma (default 3.0).")
    return parser.parse_args()


def ensure_output_root(path: Path, allow_existing: bool) -> None:
    if path.exists():
        if not allow_existing and any(path.iterdir()):
            raise SystemExit(f"Output directory '{path}' is not empty; please back it up or remove it before regenerating.")
    else:
        path.mkdir(parents=True, exist_ok=True)


def prepare_split_dirs(root: Path, save_masks: bool, save_yolo: bool) -> Dict[str, Dict[str, Path]]:
    dirs: Dict[str, Dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        images = root / split / "images"
        metadata = root / split / "metadata"
        images.mkdir(parents=True, exist_ok=True)
        metadata.mkdir(parents=True, exist_ok=True)
        split_entry: Dict[str, Path] = {"images": images, "metadata": metadata}
        if save_masks:
            masks = root / split / "masks"
            masks.mkdir(parents=True, exist_ok=True)
            split_entry["masks"] = masks
        if save_yolo:
            labels = root / split / "labels"
            labels.mkdir(parents=True, exist_ok=True)
            split_entry["labels"] = labels
        dirs[split] = split_entry
    return dirs


def detect_existing_counts(split_dirs: Dict[str, Dict[str, Path]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split, dirs in split_dirs.items():
        metadata_dir = dirs["metadata"]
        counts[split] = sum(1 for _ in metadata_dir.glob("*.json")) if metadata_dir.exists() else 0
    return counts


def allocate_profile_counts(total_sequences: int, num_profiles: int) -> List[int]:
    base = total_sequences // num_profiles
    remainder = total_sequences % num_profiles
    counts = [base] * num_profiles
    for idx in range(remainder):
        counts[idx] += 1
    return counts


def base_config(args: argparse.Namespace, rng: np.random.Generator) -> SynthConfig:
    num_rings = int(rng.integers(1, 4))
    spots_per_ring = int(rng.integers(8, 20))
    drift = float(rng.uniform(0.1, 0.8))
    radial_expansion = float(rng.uniform(0.0, 0.25))
    noise = float(rng.uniform(0.015, 0.05))
    bg_base = float(rng.uniform(0.03, 0.07))
    wave_period = int(rng.integers(140, 360))
    config = SynthConfig(
        frames=args.frames,
        frame_size=args.frame_size,
        num_rings=num_rings,
        spots_per_ring=spots_per_ring,
        ring_radius=float(rng.uniform(60.0, 110.0)),
        drift_per_frame=drift,
        radial_expansion_per_frame=radial_expansion,
        noise_level=noise,
        bg_base_level=bg_base,
        bg_wave_period=wave_period,
    )
    min_intensity = float(rng.uniform(0.6, 0.9))
    config.intensity_range = (min_intensity, min_intensity + float(rng.uniform(0.4, 0.9)))
    sigma_x_min = float(rng.uniform(2.0, 3.0))
    config.sigma_x_range = (sigma_x_min, sigma_x_min + float(rng.uniform(1.0, 2.0)))
    sigma_y_min = float(rng.uniform(1.5, 2.5))
    config.sigma_y_range = (sigma_y_min, sigma_y_min + float(rng.uniform(1.0, 2.0)))
    config.theta_range = (-0.4, 0.4)
    config.satellite_probability = float(rng.uniform(0.15, 0.55))
    config.satellite_intensity_ratio_range = (0.18, 0.55)
    config.satellite_radius_offset = (5.0, 18.0)
    config.satellite_angle_offset = (-0.4, 0.4)
    config.bg_grad_amplitude = float(rng.uniform(0.0, 0.05))
    config.bg_quad_amplitude = float(rng.uniform(0.0, 0.03))
    config.bg_spot_amplitude = 0.0
    config.bg_spot_count = 0
    config.grid_mode = "none"
    config.grid_amplitude = 0.0
    config.blink_duty_cycle = float(rng.uniform(0.35, 0.75))
    config.intensity_wave = "flat"
    config.intensity_wave_period = float(rng.uniform(80.0, 200.0))
    return config


def apply_moire_static(config: SynthConfig, rng: np.random.Generator) -> None:
    config.grid_mode = rng.choice(["checker", "stripe"])
    config.grid_amplitude = float(rng.uniform(0.03, 0.08))
    config.grid_period_x = float(rng.uniform(12.0, 22.0))
    config.grid_period_y = float(rng.uniform(12.0, 22.0))
    if config.grid_mode == "stripe":
        config.grid_orientation = float(rng.uniform(0.0, 2 * np.pi))
    config.bg_wave_mode = "none"
    config.bg_wave_amplitude = 0.0
    config.intensity_wave = rng.choice(["flat", "sin"])
    if config.intensity_wave != "flat":
        config.intensity_wave_period = float(rng.uniform(90.0, 180.0))
    config.blink_period = None


def apply_moire_blinking(config: SynthConfig, rng: np.random.Generator) -> None:
    config.grid_mode = rng.choice(["checker", "stripe"])
    config.grid_amplitude = float(rng.uniform(0.04, 0.1))
    config.grid_period_x = float(rng.uniform(10.0, 18.0))
    config.grid_period_y = float(rng.uniform(10.0, 18.0))
    if config.grid_mode == "stripe":
        config.grid_orientation = float(rng.uniform(0.0, 2 * np.pi))
    config.bg_wave_mode = "sin"
    config.bg_wave_amplitude = float(rng.uniform(0.02, 0.05))
    config.bg_wave_period = int(rng.integers(100, 220))
    config.intensity_wave = rng.choice(["sin", "triangle"])
    config.intensity_wave_period = float(rng.uniform(30.0, 90.0))
    config.blink_period = int(rng.integers(15, 60))


def apply_global_drift(config: SynthConfig, rng: np.random.Generator) -> None:
    config.grid_mode = "none"
    config.grid_amplitude = 0.0
    config.bg_wave_mode = "sin"
    config.bg_wave_amplitude = float(rng.uniform(0.06, 0.12))
    config.bg_wave_period = int(rng.integers(180, 420))
    config.bg_grad_amplitude = float(rng.uniform(0.0, 0.04))
    config.bg_quad_amplitude = float(rng.uniform(0.0, 0.025))
    config.bg_spot_amplitude = float(rng.uniform(0.0, 0.03))
    config.bg_spot_count = int(rng.integers(0, 3))
    config.intensity_wave = rng.choice(["flat", "sin"])
    config.intensity_wave_period = float(rng.uniform(80.0, 220.0))
    config.blink_period = None


def apply_inhomogeneous_bg(config: SynthConfig, rng: np.random.Generator) -> None:
    config.grid_mode = rng.choice(["none", "checker"])
    if config.grid_mode == "none":
        config.grid_amplitude = 0.0
    else:
        config.grid_amplitude = float(rng.uniform(0.02, 0.06))
        config.grid_period_x = float(rng.uniform(12.0, 24.0))
        config.grid_period_y = float(rng.uniform(12.0, 24.0))
    config.bg_grad_amplitude = float(rng.uniform(0.05, 0.12))
    config.bg_quad_amplitude = float(rng.uniform(0.015, 0.04))
    config.bg_spot_amplitude = float(rng.uniform(0.05, 0.12))
    config.bg_spot_count = int(rng.integers(3, 9))
    config.bg_wave_mode = "sin"
    config.bg_wave_amplitude = float(rng.uniform(0.02, 0.05))
    config.bg_wave_period = int(rng.integers(120, 240))
    config.intensity_wave = rng.choice(["flat", "sin"])
    config.intensity_wave_period = float(rng.uniform(60.0, 140.0))
    if rng.random() < 0.3:
        config.blink_period = int(rng.integers(20, 90))
    else:
        config.blink_period = None


PROFILES: List[ProfileSpec] = [
    ProfileSpec("moire_static", apply_moire_static),
    ProfileSpec("moire_blinking", apply_moire_blinking),
    ProfileSpec("global_drift", apply_global_drift),
    ProfileSpec("inhomogeneous_bg", apply_inhomogeneous_bg),
]


def iter_sequence_plan(total_sequences: int, rng: np.random.Generator) -> List[ProfileSpec]:
    counts = allocate_profile_counts(total_sequences, len(PROFILES))
    plan: List[ProfileSpec] = []
    for spec, count in zip(PROFILES, counts):
        plan.extend([spec] * count)
    rng.shuffle(plan)
    return plan


def build_split_labels(args: argparse.Namespace, rng: np.random.Generator) -> List[str]:
    labels = ["train"] * args.train_seqs + ["val"] * args.val_seqs + ["test"] * args.test_seqs
    rng.shuffle(labels)
    return labels


def transfer_outputs(
    tmp_dir: Path,
    seq_stem: str,
    split_dirs: Dict[str, Path],
    image_ext: str,
    save_masks: bool,
    save_yolo: bool,
) -> None:
    images_dir: Path = split_dirs["images"]
    masks_dir: Path | None = split_dirs.get("masks")
    labels_dir: Path | None = split_dirs.get("labels")
    metadata_dir: Path = split_dirs["metadata"]

    for path in sorted(tmp_dir.iterdir()):
        if not path.name.startswith("synthetic_"):
            continue
        if path.suffix.lower() == ".json":
            continue
        stem = path.stem
        suffix = path.suffix.lower()
        if suffix == ".txt":
            if save_yolo and labels_dir is not None:
                frame_id = stem.replace("synthetic_", "")
                dest = labels_dir / f"{seq_stem}_{frame_id}.txt"
                shutil.move(str(path), dest)
            else:
                path.unlink(missing_ok=True)
            continue
        is_mask = stem.endswith("_mask")
        if is_mask:
            if not save_masks or masks_dir is None:
                path.unlink(missing_ok=True)
                continue
            frame_id = stem.replace("synthetic_", "").replace("_mask", "")
            dest = masks_dir / f"{seq_stem}_{frame_id}_mask{image_ext}"
        else:
            if suffix != image_ext:
                continue
            frame_id = stem.replace("synthetic_", "")
            dest = images_dir / f"{seq_stem}_{frame_id}{image_ext}"
        shutil.move(str(path), dest)

    metadata_src = tmp_dir / "synthetic_tracks.json"
    if metadata_src.exists():
        shutil.move(str(metadata_src), metadata_dir / f"{seq_stem}.json")
    shutil.rmtree(tmp_dir)


def count_generated_files(images_dir: Path, image_ext: str) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for _ in images_dir.glob(f"*{image_ext}"))


def main() -> None:
    args = parse_args()
    total_sequences = args.train_seqs + args.val_seqs + args.test_seqs
    rng = np.random.default_rng(args.seed)
    ensure_output_root(args.output, allow_existing=args.resume)
    save_mask = not args.no_masks
    split_dirs = prepare_split_dirs(args.output, save_mask, args.yolo_labels)
    tmp_root = args.output / "_tmp_sequences"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    sequence_plan = iter_sequence_plan(total_sequences, rng)
    split_labels = build_split_labels(args, rng)
    base_counts = detect_existing_counts(split_dirs) if args.resume else {k: 0 for k in split_dirs.keys()}
    per_split_counts = dict(base_counts)

    for idx, (spec, split) in enumerate(zip(sequence_plan, split_labels), start=1):
        per_split_counts[split] += 1
        seq_tag = f"{split}_{spec.name}_{per_split_counts[split]:02d}"
        seq_dir = tmp_root / seq_tag
        cfg = base_config(args, rng)
        cfg.seed = int(rng.integers(0, 2_000_000_000))
        spec.apply(cfg, rng)
        print(f"[{idx}/{total_sequences}] Generating '{seq_tag}' ({cfg.num_rings} rings, {cfg.spots_per_ring} spots/ring)")
        generate_sequence(
            cfg,
            seq_dir,
            save_mask=save_mask,
            save_yolo=args.yolo_labels,
            image_format=args.image_format,
            yolo_box_scale=args.yolo_box_scale,
        )
        transfer_outputs(
            seq_dir,
            seq_tag,
            split_dirs[split],
            image_ext=f".{args.image_format}",
            save_masks=save_mask,
            save_yolo=args.yolo_labels,
        )

    shutil.rmtree(tmp_root)

    for split, dirs in split_dirs.items():
        images_count = count_generated_files(dirs["images"], image_ext=f".{args.image_format}")
        masks_count = count_generated_files(dirs["masks"], image_ext=f".{args.image_format}") if save_mask else 0
        existing_base = base_counts.get(split, 0)
        new_sequences = per_split_counts[split] - existing_base
        print(
            f"{split}: {images_count} images, {masks_count} masks "
            f"(existing sequences: {existing_base}, added: {new_sequences})"
        )


if __name__ == "__main__":
    main()
