"""Ideal H6 positions and point-sampled, finite Gaussian rendering."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trackleed.synthetic.config import BenchmarkConfig


@dataclass(frozen=True)
class Reflection:
    reflection_id: str
    h: int
    k: int
    x_px: float
    y_px: float
    amplitude: float
    sigma_px: float


@dataclass(frozen=True)
class RenderedObservation:
    image: NDArray[np.float32]
    png: NDArray[np.uint8]
    reflections: tuple[Reflection, ...]
    reference: tuple[dict[str, Any], ...]
    photometry: dict[str, int]


def evaluate_scene(
    config: BenchmarkConfig, time_ns: int | None = None, *, observation_index: int | None = None,
) -> tuple[Reflection, ...]:
    """Return all latent states at a time or an ordered energy observation."""
    scene = config.scene
    cx, cy = scene.center_px
    energy = config.energy_at(time_ns, observation_index=observation_index)
    radius = scene.lattice_radius(energy)
    if time_ns is not None:
        cx += scene.drift_px_per_s[0] * (time_ns / 1e9)
        cy += scene.drift_px_per_s[1] * (time_ns / 1e9)
    # Arbitrary-time evaluation must not bypass the current no-clipping contract.
    extents = scene.support_extents(radius)
    for center, extent, size in zip((cx, cy), extents, scene.image_size):
        if not math.isfinite(center + extent) or center - extent < 0 or center + extent > size - 1:
            raise ValueError("scene support must fit inside the frame in this stage")
    indices = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
    return tuple(
        Reflection(f"h{h:+d}_k{k:+d}", h, k,
                   cx + radius * (h + k / 2),
                   cy + radius * math.sqrt(3) * k / 2,
                   scene.amplitude, scene.sigma_px)
        for h, k in indices
    )


def render_observation(
    config: BenchmarkConfig, time_ns: int | None = None, *, observation_index: int | None = None,
) -> RenderedObservation:
    reflections = evaluate_scene(config, time_ns, observation_index=observation_index)
    profiles = tuple(_profile(reflection) for reflection in reflections)
    raw = _compose(config, reflections, profiles)
    with np.errstate(over="ignore"):
        image = raw.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Rendered image cannot be represented as finite float32")
    lower, upper = config.condition.png_lower, config.condition.png_upper
    png = encode_png(image, lower, upper)
    reference = tuple(
        _reference(config, reflections, profiles, image, png, i)
        for i in range(len(reflections))
    )
    photometry = {
        "below_lower_pixels": int(np.count_nonzero(image < lower)),
        "above_upper_pixels": int(np.count_nonzero(image > upper)),
        "png_zero_pixels": int(np.count_nonzero(png == 0)),
        "png_255_pixels": int(np.count_nonzero(png == 255)),
    }
    return RenderedObservation(image, png, reflections, reference, photometry)


def _compose(
    config: BenchmarkConfig, reflections: tuple[Reflection, ...],
    profiles: tuple[tuple[NDArray, NDArray, NDArray], ...], omit: int | None = None,
) -> NDArray[np.float64]:
    width, height = config.scene.image_size
    raw = np.full((height, width), config.condition.background, dtype=np.float64)
    for index, (reflection, (rows, columns, profile)) in enumerate(zip(reflections, profiles)):
        if index != omit:
            raw[np.ix_(rows, columns)] += reflection.amplitude * profile
    return raw


def _reference(
    config: BenchmarkConfig, reflections: tuple[Reflection, ...],
    profiles: tuple[tuple[NDArray, NDArray, NDArray], ...],
    image: NDArray, png: NDArray, index: int,
) -> dict[str, Any]:
    reflection = reflections[index]
    profile = profiles[index][2]
    without = _compose(config, reflections, profiles, omit=index).astype(np.float32)
    without_png = encode_png(without, config.condition.png_lower, config.condition.png_upper)
    weight = float(profile.sum(dtype=np.float64))
    signal = reflection.amplitude * profile
    return {
        "reflection_id": reflection.reflection_id, "parent_id": None, "role": "primary",
        "h": reflection.h, "k": reflection.k,
        "x_px": reflection.x_px, "y_px": reflection.y_px,
        "amplitude": reflection.amplitude,
        "sigma_u_px": reflection.sigma_px, "sigma_v_px": reflection.sigma_px,
        "theta_rad": 0.0, "extinguished": reflection.amplitude == 0,
        "center_in_frame": True, "w0": weight, "w1": weight, "w2": weight, "w3": weight,
        "frame_loss_fraction": 0.0, "screen_loss_fraction": 0.0,
        "occluder_loss_fraction": 0.0, "retained_fraction": 1.0,
        "prequant_signal_present": reflection.amplitude > 0,
        "float64_signal_pixels": int(np.count_nonzero(signal)),
        "float32_signal_pixels": int(np.count_nonzero(signal.astype(np.float32))),
        "float32_changed_pixels": int(np.count_nonzero(image != without)),
        "encoded_changed_pixels": int(np.count_nonzero(png != without_png)),
        "overlap_contour_ids": _overlaps(reflections, index, 3),
        "overlap_support_ids": _overlaps(reflections, index, 4),
    }


def _overlaps(reflections: tuple[Reflection, ...], index: int, extent: int) -> list[str]:
    target = reflections[index]
    return [
        other.reflection_id for i, other in enumerate(reflections) if i != index
        and math.hypot(target.x_px - other.x_px, target.y_px - other.y_px)
        <= extent * (target.sigma_px + other.sigma_px)
    ]


def encode_png(image: NDArray, lower: float, upper: float) -> NDArray[np.uint8]:
    """Map intensities with fixed bounds and round-half-up quantization."""
    scaled = np.clip((image.astype(np.float64) - lower) / (upper - lower), 0, 1)
    return np.floor(255 * scaled + 0.5).astype(np.uint8)


def _profile(reflection: Reflection) -> tuple[NDArray, NDArray, NDArray]:
    x, y, sigma = reflection.x_px, reflection.y_px, reflection.sigma_px
    columns = np.arange(math.ceil(x - 4 * sigma), math.floor(x + 4 * sigma) + 1)
    rows = np.arange(math.ceil(y - 4 * sigma), math.floor(y + 4 * sigma) + 1)
    dx, dy = np.meshgrid((columns - x) / sigma, (rows - y) / sigma)
    with np.errstate(over="ignore"):
        q2 = dx * dx + dy * dy
    profile = np.where(q2 <= 16, np.exp(-q2 / 2), 0)
    if not profile.sum() > 0:
        raise ValueError("Unsupported spatial sampling: W0 = 0")
    return rows, columns, profile
