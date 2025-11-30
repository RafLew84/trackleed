"""Synthetic LEED sequence generator for training/evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
import imageio.v3 as iio


@dataclass
class SynthConfig:
    frames: int = 30
    frame_size: int = 256
    # Background baseline and temporal drift
    bg_base_level: float = 0.05
    bg_wave_amplitude: float = 0.0
    bg_wave_period: int = 200
    bg_wave_mode: str = "none"  # "none" | "sin"
    # Spatial background inhomogeneity (gradients / smooth patches)
    bg_grad_amplitude: float = 0.0     # linear gradient component
    bg_quad_amplitude: float = 0.0     # radial quadratic component
    bg_spot_amplitude: float = 0.0     # amplitude for smooth local bright/dark blobs
    bg_spot_count: int = 0             # number of blobs to place
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


def _build_flow_from_spots(frame_a: list[dict], frame_b: list[dict], frame_size: int) -> np.ndarray:
    """
    Build dense flow map (H,W,2) from spot positions between consecutive frames.
    Flow is filled with per-spot displacement inside Gaussian support; overlaps are averaged.
    """
    flow = np.zeros((frame_size, frame_size, 2), dtype=np.float32)
    weight = np.zeros((frame_size, frame_size), dtype=np.float32)
    # Map track_id -> spot meta in frame B
    next_map = {s["track_id"]: s for s in frame_b}
    for spot in frame_a:
        tid = spot["track_id"]
        if tid not in next_map:
            continue
        nxt = next_map[tid]
        dx = nxt["x"] - spot["x"]
        dy = nxt["y"] - spot["y"]
        # Draw Gaussian weight and accumulate flow
        support = max(int(3 * max(spot["sigma_x"], spot["sigma_y"])), 1)
        xs = np.arange(int(spot["x"]) - support, int(spot["x"]) + support + 1)
        ys = np.arange(int(spot["y"]) - support, int(spot["y"]) + support + 1)
        xs = xs[(0 <= xs) & (xs < frame_size)]
        ys = ys[(0 <= ys) & (ys < frame_size)]
        if xs.size == 0 or ys.size == 0:
            continue
        grid_x, grid_y = np.meshgrid(xs, ys)
        dx_local = grid_x - spot["x"]
        dy_local = grid_y - spot["y"]
        cos_t = np.cos(spot["theta"])
        sin_t = np.sin(spot["theta"])
        x_rot = cos_t * dx_local + sin_t * dy_local
        y_rot = -sin_t * dx_local + cos_t * dy_local
        gaussian = np.exp(-((x_rot**2) / (2 * spot["sigma_x"] ** 2) + (y_rot**2) / (2 * spot["sigma_y"] ** 2)))
        flow[np.ix_(ys, xs, [0])] += gaussian[..., None] * dx
        flow[np.ix_(ys, xs, [1])] += gaussian[..., None] * dy
        weight[np.ix_(ys, xs)] += gaussian
    mask = weight > 1e-6
    flow[mask] /= weight[mask, None]
    return flow


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


def _generate_spatial_background(config: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a smooth, spatially varying background field with zero mean,
    combining a global gradient, radial component, and a few broad blobs.
    """
    h = config.frame_size
    w = config.frame_size
    yy, xx = np.mgrid[0:h, 0:w]
    xnorm = (xx / max(w - 1, 1)) - 0.5
    ynorm = (yy / max(h - 1, 1)) - 0.5

    field = np.zeros((h, w), dtype=np.float32)

    # Linear gradient along a random direction
    if config.bg_grad_amplitude != 0.0:
        theta = rng.uniform(0, 2 * np.pi)
        grad_dir = np.cos(theta) * xnorm + np.sin(theta) * ynorm
        field += config.bg_grad_amplitude * grad_dir.astype(np.float32)

    # Radial quadratic term (brighter/darker center vs edges)
    if config.bg_quad_amplitude != 0.0:
        r2 = xnorm**2 + ynorm**2
        r2 = r2 - r2.mean()
        field += config.bg_quad_amplitude * r2.astype(np.float32)

    # A few smooth local blobs (additive bright/dark patches)
    if config.bg_spot_amplitude != 0.0 and config.bg_spot_count > 0:
        blobs = np.zeros_like(field)
        for _ in range(config.bg_spot_count):
            cx = rng.uniform(-0.25, 0.25)
            cy = rng.uniform(-0.25, 0.25)
            sx = rng.uniform(0.15, 0.35)
            sy = rng.uniform(0.15, 0.35)
            sign = rng.choice([-1.0, 1.0])
            dx = xnorm - cx
            dy = ynorm - cy
            blob = np.exp(-((dx**2) / (2 * sx**2) + (dy**2) / (2 * sy**2)))
            blobs += sign * blob.astype(np.float32)
        # Normalize blobs to roughly [-1,1]
        maxabs = float(np.max(np.abs(blobs))) if blobs.size else 0.0
        if maxabs > 1e-6:
            blobs = blobs / maxabs
        field += config.bg_spot_amplitude * blobs

    # Normalize global field to zero mean and clamp to a reasonable range
    field = field - float(field.mean())
    maxabs = float(np.max(np.abs(field))) if field.size else 0.0
    if maxabs > 0:
        field = field / maxabs  # now roughly in [-1, 1]
    return field.astype(np.float32)


def _spot_to_yolo_label(
    pos: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    frame_size: int,
    box_scale: float = 3.0,
    class_id: int = 0,
) -> str:
    width = max(1.0, box_scale * sigma_x * 2.0)
    height = max(1.0, box_scale * sigma_y * 2.0)
    half_w = width / 2.0
    half_h = height / 2.0
    x0 = max(0.0, pos[0] - half_w)
    y0 = max(0.0, pos[1] - half_h)
    x1 = min(frame_size - 1.0, pos[0] + half_w)
    y1 = min(frame_size - 1.0, pos[1] + half_h)
    center_x = ((x0 + x1) / 2.0) / frame_size
    center_y = ((y0 + y1) / 2.0) / frame_size
    box_w = max(1e-6, (x1 - x0) / frame_size)
    box_h = max(1e-6, (y1 - y0) / frame_size)
    center_x = float(np.clip(center_x, 0.0, 1.0))
    center_y = float(np.clip(center_y, 0.0, 1.0))
    box_w = float(np.clip(box_w, 1e-6, 1.0))
    box_h = float(np.clip(box_h, 1e-6, 1.0))
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"


def _write_image(path: Path, image: np.ndarray, fmt: str, is_mask: bool = False) -> None:
    fmt = fmt.lower()
    if fmt == "tiff":
        if is_mask:
            tifffile.imwrite(path, image.astype(np.uint8))
        else:
            tifffile.imwrite(path, image.astype(np.float32))
        return

    if fmt == "png":
        if is_mask:
            arr = (image > 0).astype(np.uint8) * 255
        else:
            arr = image.astype(np.float32)
            min_v = float(arr.min())
            max_v = float(arr.max())
            if max_v > min_v:
                arr = (arr - min_v) / (max_v - min_v)
            else:
                arr = np.zeros_like(arr)
            arr = np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
        iio.imwrite(path, arr)
        return

    raise ValueError(f"Unsupported image format: {fmt}")


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


def generate_sequence(
    config: SynthConfig,
    output_dir: Path,
    save_mask: bool = False,
    save_yolo: bool = False,
    save_flow: bool = False,
    image_format: str = "tiff",
    yolo_box_scale: float = 3.0,
    yolo_class_id: int = 0,
):
    """
    Generate a synthetic LEED sequence.

    If save_mask is True, in addition to the intensity image a binary mask
    of all spots is written for each frame (useful for segmentation training).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    spots = _generate_initial_spots(config, rng)
    spatial_bg = _generate_spatial_background(config, rng)

    track_records = {spot.track_id: [] for spot in spots}
    center = np.array([config.frame_size / 2, config.frame_size / 2], dtype=np.float32)
    drift_direction = rng.uniform(0, 2 * np.pi)
    drift_vector = np.array(
        [np.cos(drift_direction), np.sin(drift_direction)], dtype=np.float32
    ) * config.drift_per_frame

    image_format = image_format.lower()
    if image_format not in {"tiff", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")

    # For flow export, keep images and per-frame spot metadata
    images_cache = []
    spots_per_frame: list[list[dict]] = []

    for frame_idx in range(config.frames):
        # Time-varying background: global level can brighten/darken over the sequence.
        bg_level = config.bg_base_level
        if (
            config.bg_wave_mode.lower() == "sin"
            and config.bg_wave_amplitude != 0.0
            and config.bg_wave_period > 0
        ):
            phase = 2 * np.pi * frame_idx / float(config.bg_wave_period)
            bg_level = config.bg_base_level + config.bg_wave_amplitude * np.sin(phase)
        # Start from global level plus spatial inhomogeneity
        base_background = np.full((config.frame_size, config.frame_size), float(bg_level), dtype=np.float32)
        if spatial_bg is not None:
            base_background = base_background + spatial_bg
        image = base_background.copy()
        spot_layer = np.zeros_like(image, dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.float32) if save_mask else None
        yolo_labels = [] if save_yolo else None
        frame_center = center + drift_vector * frame_idx

        frame_spots_meta = []
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
                spot_layer,
                pos[0],
                pos[1],
                spot.sigma_x,
                spot.sigma_y,
                spot.theta,
                current_intensity,
            )
            if mask is not None:
                # Draw a unit-amplitude Gaussian into the mask; will be thresholded later.
                _render_gaussian_blob(
                    mask,
                    pos[0],
                    pos[1],
                    spot.sigma_x,
                    spot.sigma_y,
                    spot.theta,
                    1.0,
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
            frame_spots_meta.append(
                {
                    "track_id": spot.track_id,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "sigma_x": float(spot.sigma_x),
                    "sigma_y": float(spot.sigma_y),
                    "theta": float(spot.theta),
                    "active": bool(spot.is_active),
                }
            )

            if save_yolo and yolo_labels is not None:
                yolo_labels.append(
                    _spot_to_yolo_label(
                        pos,
                        spot.sigma_x,
                        spot.sigma_y,
                        config.frame_size,
                        box_scale=yolo_box_scale,
                        class_id=yolo_class_id,
                    )
                )

        _apply_grid_artifact(spot_layer, config, rng)
        image = base_background + spot_layer
        image += rng.normal(0, config.noise_level, image.shape).astype(np.float32)
        image_path = output_dir / f"synthetic_{frame_idx:04d}.{image_format}"
        _write_image(image_path, image, image_format)
        if save_flow:
            images_cache.append(image.copy())
            spots_per_frame.append(frame_spots_meta)

        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
            mask_path = output_dir / f"synthetic_{frame_idx:04d}_mask.{image_format}"
            _write_image(mask_path, mask_bin.astype(np.float32), image_format, is_mask=True)

        if save_yolo and yolo_labels is not None:
            label_path = output_dir / f"synthetic_{frame_idx:04d}.txt"
            label_path.write_text("\n".join(yolo_labels))

        spots = [_update_spot_state(spot, config, rng) for spot in spots]

    if save_flow and len(images_cache) >= 2:
        flows_dir = output_dir / "flows"
        flows_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(images_cache) - 1):
            flow = _build_flow_from_spots(
                spots_per_frame[i],
                spots_per_frame[i + 1],
                config.frame_size,
            )
            np.save(flows_dir / f"synthetic_{i:04d}.npy", flow)

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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequence", "segmentation"],
        default="sequence",
        help="If 'segmentation', also save per-frame binary masks for spots.",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Explicitly request saving segmentation masks (equivalent to --mode segmentation).",
    )
    parser.add_argument("--image-format", choices=["tiff", "png"], default="tiff", help="Image format for outputs.")
    parser.add_argument("--save-yolo", action="store_true", help="Also save YOLO label files per frame.")
    parser.add_argument("--yolo-box-scale", type=float, default=3.0, help="Bounding-box multiplier relative to spot sigma (default 3.0).")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--spots-per-ring", type=int, default=8)
    parser.add_argument("--rings", type=int, default=1)
    parser.add_argument("--ring-radius", type=float, default=80.0)
    parser.add_argument("--drift", type=float, default=0.5, help="Drift per frame (pixels).")
    parser.add_argument("--expansion", type=float, default=0.2, help="Radial expansion per frame.")
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--bg-base", type=float, default=0.05, help="Base background level.")
    parser.add_argument(
        "--bg-wave-amplitude",
        type=float,
        default=0.0,
        help="Amplitude of global background variation over time.",
    )
    parser.add_argument(
        "--bg-wave-period",
        type=int,
        default=200,
        help="Period (in frames) of the background variation wave.",
    )
    parser.add_argument(
        "--bg-wave-mode",
        type=str,
        choices=["none", "sin"],
        default="none",
        help="Background variation mode over time.",
    )
    parser.add_argument(
        "--bg-grad-amplitude",
        type=float,
        default=0.0,
        help="Amplitude of a smooth linear gradient component in the background.",
    )
    parser.add_argument(
        "--bg-quad-amplitude",
        type=float,
        default=0.0,
        help="Amplitude of a smooth radial quadratic component in the background.",
    )
    parser.add_argument(
        "--bg-spot-amplitude",
        type=float,
        default=0.0,
        help="Amplitude of smooth local bright/dark background blobs.",
    )
    parser.add_argument(
        "--bg-spot-count",
        type=int,
        default=0,
        help="Number of smooth local background blobs to place.",
    )
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
    parser.add_argument("--save-flow", action="store_true", help="Also save dense flow .npy between consecutive frames.")
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
        bg_base_level=args.bg_base,
        bg_wave_amplitude=args.bg_wave_amplitude,
        bg_wave_period=args.bg_wave_period,
        bg_wave_mode=args.bg_wave_mode,
        bg_grad_amplitude=args.bg_grad_amplitude,
        bg_quad_amplitude=args.bg_quad_amplitude,
        bg_spot_amplitude=args.bg_spot_amplitude,
        bg_spot_count=args.bg_spot_count,
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

    save_mask = args.save_mask or (args.mode == "segmentation")
    generate_sequence(
        config,
        args.output,
        save_mask=save_mask,
        save_yolo=args.save_yolo,
        save_flow=args.save_flow,
        image_format=args.image_format,
        yolo_box_scale=args.yolo_box_scale,
    )
    print(f"Synthetic sequence saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
