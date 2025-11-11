"""Synthetic LEED sequence generator for training/evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile


@dataclass
class SynthConfig:
    frames: int = 30
    frame_size: int = 256
    ring_radius: float = 80.0
    spots_per_ring: int = 8
    num_rings: int = 1
    drift_per_frame: float = 0.5
    radial_expansion_per_frame: float = 0.2
    noise_level: float = 0.02
    intensity_range: Tuple[float, float] = (0.8, 1.2)
    sigma_x_range: Tuple[float, float] = (2.5, 4.5)
    sigma_y_range: Tuple[float, float] = (1.5, 3.0)
    theta_range: Tuple[float, float] = (-0.35, 0.35)
    seed: int | None = 42
    grid_mode: str = "none"
    grid_amplitude: float = 0.05
    grid_period_x: float = 16.0
    grid_period_y: float = 16.0
    grid_orientation: float | None = None
    intensity_wave: str = "flat"
    intensity_wave_period: float = 30.0
    intensity_wave_phase: float | None = None
    blink_period: int | None = None
    blink_duty_cycle: float = 0.5
    satellite_probability: float = 0.3
    satellite_intensity_ratio_range: Tuple[float, float] = (0.2, 0.5)
    satellite_radius_offset: Tuple[float, float] = (5.0, 15.0)
    satellite_angle_offset: Tuple[float, float] = (-0.3, 0.3)


@dataclass
class SpotState:
    track_id: int
    angle: float
    ring_index: int
    radius: float
    intensity: float
    sigma_x: float
    sigma_y: float
    theta: float
    phase: float
    is_active: bool = True
    is_satellite: bool = False
    parent_track_id: int | None = None


def _generate_initial_spots(config: SynthConfig, rng: np.random.Generator) -> List[SpotState]:
    spots: List[SpotState] = []
    track_id = 0
    for ring in range(config.num_rings):
        radius = config.ring_radius * (ring + 1)
        for i in range(config.spots_per_ring):
            angle = 2 * np.pi * i / config.spots_per_ring
            intensity = rng.uniform(*config.intensity_range)
            sigma_x = rng.uniform(*config.sigma_x_range)
            sigma_y = rng.uniform(*config.sigma_y_range)
            theta = rng.uniform(*config.theta_range)
            phase = rng.uniform(0, 2 * np.pi)
            parent_spot = SpotState(
                track_id=track_id,
                angle=angle,
                ring_index=ring,
                radius=radius,
                intensity=intensity,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                theta=theta,
                phase=phase,
            )
            spots.append(parent_spot)
            track_id += 1

            if rng.random() < max(0.0, min(1.0, config.satellite_probability)):
                sat_radius_offset = rng.uniform(*config.satellite_radius_offset)
                sat_angle_offset = rng.uniform(*config.satellite_angle_offset)
                sat_intensity_ratio = rng.uniform(*config.satellite_intensity_ratio_range)
                sat_intensity = np.clip(
                    parent_spot.intensity * sat_intensity_ratio,
                    0.0,
                    config.intensity_range[1],
                )
                satellite_spot = SpotState(
                    track_id=track_id,
                    angle=angle + sat_angle_offset,
                    ring_index=ring,
                    radius=radius + sat_radius_offset,
                    intensity=sat_intensity,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    theta=theta,
                    phase=phase,
                    is_satellite=True,
                    parent_track_id=parent_spot.track_id,
                )
                spots.append(satellite_spot)
                track_id += 1
    return spots


def _render_gaussian_blob(
    image: np.ndarray, x: float, y: float, sigma_x: float, sigma_y: float, theta: float, amplitude: float
):
    support = max(int(3 * max(sigma_x, sigma_y)), 1)
    xs = np.arange(int(x) - support, int(x) + support + 1)
    ys = np.arange(int(y) - support, int(y) + support + 1)
    xs = xs[(0 <= xs) & (xs < image.shape[1])]
    ys = ys[(0 <= ys) & (ys < image.shape[0])]
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = grid_x - x
    dy = grid_y - y
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy
    gaussian = amplitude * np.exp(
        -((x_rot**2) / (2 * sigma_x**2) + (y_rot**2) / (2 * sigma_y**2))
    )
    image[np.ix_(ys, xs)] += gaussian


def _clamp_position(pos: np.ndarray, frame_size: int, sigma_x: float, sigma_y: float) -> np.ndarray:
    margin_x = max(4, int(3 * sigma_x) + 1)
    margin_y = max(4, int(3 * sigma_y) + 1)
    min_x, max_x = margin_x, frame_size - margin_x - 1
    min_y, max_y = margin_y, frame_size - margin_y - 1
    return np.array(
        [
            float(np.clip(pos[0], min_x, max_x)),
            float(np.clip(pos[1], min_y, max_y)),
        ],
        dtype=np.float32,
    )


def _apply_grid_artifact(image: np.ndarray, config: SynthConfig, rng: np.random.Generator):
    mode = (config.grid_mode or "none").lower()
    amplitude = max(0.0, config.grid_amplitude)
    if mode == "none" or amplitude <= 1e-6:
        return

    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]

    if mode == "checker":
        period_x = max(2.0, config.grid_period_x)
        period_y = max(2.0, config.grid_period_y)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        grid = 1.0 + amplitude * (
            np.sin((2 * np.pi * xx / period_x) + phase_x)
            * np.sin((2 * np.pi * yy / period_y) + phase_y)
        )
    elif mode == "stripe":
        period = max(2.0, config.grid_period_x)
        orientation = (
            config.grid_orientation if config.grid_orientation is not None else rng.uniform(0, 2 * np.pi)
        )
        phase = rng.uniform(0, 2 * np.pi)
        projection = (np.cos(orientation) * xx + np.sin(orientation) * yy) * (2 * np.pi / period)
        grid = 1.0 + amplitude * np.sin(projection + phase)
    else:
        return

    image *= grid
    np.clip(image, 0.0, None, out=image)


def _update_spot_state(
    spot: SpotState, config: SynthConfig, rng: np.random.Generator
) -> SpotState:
    return SpotState(
        track_id=spot.track_id,
        angle=spot.angle + rng.normal(0, 0.01),
        ring_index=spot.ring_index,
        radius=spot.radius + config.radial_expansion_per_frame + rng.normal(0, 0.05),
        intensity=max(
            config.intensity_range[0],
            min(config.intensity_range[1], spot.intensity + rng.normal(0, 0.01)),
        ),
        sigma_x=max(
            config.sigma_x_range[0],
            min(config.sigma_x_range[1], spot.sigma_x + rng.normal(0, 0.05)),
        ),
        sigma_y=max(
            config.sigma_y_range[0],
            min(config.sigma_y_range[1], spot.sigma_y + rng.normal(0, 0.05)),
        ),
        theta=np.clip(spot.theta + rng.normal(0, 0.01), *config.theta_range),
        phase=spot.phase,
        is_active=spot.is_active,
    )


def generate_sequence(config: SynthConfig, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    spots = _generate_initial_spots(config, rng)

    track_records = {spot.track_id: [] for spot in spots}
    center = np.array([config.frame_size / 2, config.frame_size / 2], dtype=np.float32)
    drift_direction = rng.uniform(0, 2 * np.pi)
    drift_vector = np.array(
        [np.cos(drift_direction), np.sin(drift_direction)], dtype=np.float32
    ) * config.drift_per_frame

    for frame_idx in range(config.frames):
        image = np.full((config.frame_size, config.frame_size), 0.05, dtype=np.float32)
        frame_center = center + drift_vector * frame_idx

        for spot in spots:
            pos = frame_center + spot.radius * np.array([np.cos(spot.angle), np.sin(spot.angle)])
            pos = _clamp_position(pos, config.frame_size, spot.sigma_x, spot.sigma_y)
            current_intensity = spot.intensity
            angle = (2 * np.pi * frame_idx / max(config.intensity_wave_period, 1e-3)) + spot.phase
            wave_mode = (config.intensity_wave or "flat").lower()
            if wave_mode == "sin":
                mod = 0.5 * (1 + np.sin(angle))
                current_intensity *= mod
            elif wave_mode == "triangle":
                mod = 2 * abs((frame_idx / max(config.intensity_wave_period, 1e-3)) % 1 - 0.5)
                current_intensity *= mod
            if config.blink_period:
                duty = np.clip(config.blink_duty_cycle, 0.05, 0.95)
                adjusted_phase = (spot.phase / (2 * np.pi)) * config.blink_period
                phase_frame = (frame_idx + adjusted_phase) % config.blink_period
                spot.is_active = phase_frame < duty * config.blink_period
            else:
                spot.is_active = True

            if not spot.is_active:
                continue

            _render_gaussian_blob(
                image,
                pos[0],
                pos[1],
                spot.sigma_x,
                spot.sigma_y,
                spot.theta,
                current_intensity,
            )
            track_records[spot.track_id].append(
                {
                    "frame": frame_idx,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "radius": float(spot.radius),
                    "sigma_x": float(spot.sigma_x),
                    "sigma_y": float(spot.sigma_y),
                    "theta": float(spot.theta),
                    "intensity": float(current_intensity),
                    "is_active": bool(spot.is_active),
                    "is_satellite": bool(spot.is_satellite),
                    "parent_track_id": spot.parent_track_id if spot.parent_track_id is not None else None,
                }
            )

        _apply_grid_artifact(image, config, rng)
        image += rng.normal(0, config.noise_level, image.shape).astype(np.float32)
        image_path = output_dir / f"synthetic_{frame_idx:04d}.tiff"
        tifffile.imwrite(image_path, image.astype(np.float32))

        spots = [_update_spot_state(spot, config, rng) for spot in spots]

    metadata = {
        "config": asdict(config),
        "tracks": track_records,
        "frames": config.frames,
        "frame_size": config.frame_size,
    }
    with open(output_dir / "synthetic_tracks.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _apply_elongation_preset(config: SynthConfig, preset: str):
    preset = preset.lower()
    if preset == "horizontal":
        config.sigma_x_range = (4.5, 6.0)
        config.sigma_y_range = (1.2, 2.2)
        config.theta_range = (-0.2, 0.2)
    elif preset == "vertical":
        config.sigma_x_range = (1.2, 2.2)
        config.sigma_y_range = (4.5, 6.0)
        # Orientations around 90 degrees
        config.theta_range = (np.pi / 2 - 0.2, np.pi / 2 + 0.2)
    elif preset == "isotropic":
        mean_sigma = np.mean(config.sigma_x_range)
        config.sigma_x_range = (mean_sigma, mean_sigma)
        config.sigma_y_range = (mean_sigma, mean_sigma)
        config.theta_range = (0.0, 0.0)
    else:  # mixed default, leave ranges as-is
        return


def main_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic LEED sequences.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for TIFFs/JSON.")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--spots-per-ring", type=int, default=8)
    parser.add_argument("--rings", type=int, default=1)
    parser.add_argument("--ring-radius", type=float, default=80.0)
    parser.add_argument("--drift", type=float, default=0.5, help="Drift per frame (pixels).")
    parser.add_argument("--expansion", type=float, default=0.2, help="Radial expansion per frame.")
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument(
        "--spot-sigma",
        type=float,
        default=None,
        help="Optional isotropic sigma; if provided overrides sigma_x_range/sigma_y_range.",
    )
    parser.add_argument(
        "--elongation",
        type=str,
        choices=["mixed", "horizontal", "vertical", "isotropic"],
        default="mixed",
        help="Preset controlling sigma_x/sigma_y ranges and orientation sampling.",
    )
    parser.add_argument(
        "--grid-mode",
        type=str,
        choices=["none", "stripe", "checker"],
        default="none",
        help="Apply multiplicative grid/moire artifact.",
    )
    parser.add_argument("--grid-amplitude", type=float, default=None, help="Strength of grid artifact (0-1).")
    parser.add_argument("--grid-period-x", type=float, default=None, help="Grid period along X axis.")
    parser.add_argument("--grid-period-y", type=float, default=None, help="Grid period along Y axis.")
    parser.add_argument("--grid-orientation", type=float, default=None, help="Stripe orientation in radians.")
    parser.add_argument(
        "--intensity-wave",
        type=str,
        choices=["flat", "sin", "triangle"],
        default="flat",
        help="Deterministic intensity modulation per track.",
    )
    parser.add_argument("--intensity-wave-period", type=float, default=None, help="Period (frames) of the intensity wave.")
    parser.add_argument("--blink-period", type=int, default=None, help="Frames between mandatory off windows.")
    parser.add_argument("--blink-duty", type=float, default=None, help="Fraction (0-1) of time track stays visible when blinking.")
    parser.add_argument("--satellite-probability", type=float, default=None, help="Probability [0-1] of spawning satellites per primary spot.")
    parser.add_argument(
        "--satellite-intensity-ratio-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Range for multiplying parent intensity when creating satellites.",
    )
    parser.add_argument(
        "--satellite-radius-offset",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Range of radial offsets (pixels) for satellites relative to parent radius.",
    )
    parser.add_argument(
        "--satellite-angle-offset",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Range of angular offsets (radians) for satellites relative to parent angle.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = SynthConfig(
        frames=args.frames,
        frame_size=args.frame_size,
        spots_per_ring=args.spots_per_ring,
        num_rings=args.rings,
        ring_radius=args.ring_radius,
        drift_per_frame=args.drift,
        radial_expansion_per_frame=args.expansion,
        noise_level=args.noise,
        seed=args.seed,
    )
    if args.spot_sigma is not None:
        config.sigma_x_range = (args.spot_sigma, args.spot_sigma)
        config.sigma_y_range = (args.spot_sigma, args.spot_sigma)
        config.theta_range = (0.0, 0.0)
    else:
        _apply_elongation_preset(config, args.elongation)
    config.grid_mode = args.grid_mode
    if args.grid_amplitude is not None:
        config.grid_amplitude = args.grid_amplitude
    if args.grid_period_x is not None:
        config.grid_period_x = args.grid_period_x
    if args.grid_period_y is not None:
        config.grid_period_y = args.grid_period_y
    if args.grid_orientation is not None:
        config.grid_orientation = args.grid_orientation
    config.intensity_wave = args.intensity_wave
    if args.intensity_wave_period:
        config.intensity_wave_period = args.intensity_wave_period
    if args.blink_period:
        config.blink_period = args.blink_period
    if args.blink_duty:
        config.blink_duty_cycle = args.blink_duty
    if args.satellite_probability is not None:
        config.satellite_probability = args.satellite_probability
    if args.satellite_intensity_ratio_range is not None:
        config.satellite_intensity_ratio_range = tuple(args.satellite_intensity_ratio_range)
    if args.satellite_radius_offset is not None:
        config.satellite_radius_offset = tuple(args.satellite_radius_offset)
    if args.satellite_angle_offset is not None:
        config.satellite_angle_offset = tuple(args.satellite_angle_offset)
    generate_sequence(config, args.output)
    print(f"Synthetic sequence saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
