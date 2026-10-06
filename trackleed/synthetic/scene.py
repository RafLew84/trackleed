"""Ideal H6 positions and point-sampled, finite Gaussian rendering."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trackleed.synthetic.config import BenchmarkConfig
from trackleed.synthetic.attenuation import amplitude_at
from trackleed.synthetic.profiles import GaussianShape, sample_profile
from trackleed.synthetic.instances import InstanceMasks, make_instance_masks


@dataclass(frozen=True)
class Reflection:
    reflection_id: str
    h: int
    k: int
    x_px: float
    y_px: float
    amplitude: float
    sigma_px: float
    parent_id: str | None = None
    role: str = "primary"
    shape: GaussianShape | None = None

    @property
    def profile_shape(self) -> GaussianShape:
        return self.shape or GaussianShape(self.sigma_px, self.sigma_px)


@dataclass(frozen=True)
class RenderedObservation:
    image: NDArray[np.float32]
    png: NDArray[np.uint8]
    reflections: tuple[Reflection, ...]
    reference: tuple[dict[str, Any], ...]
    photometry: dict[str, int]
    instances: tuple[InstanceMasks, ...] = ()


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
    return tuple(
        Reflection(site.reflection_id, site.h, site.k, cx + site.x, cy + site.y,
                   amplitude_at(scene.attenuations, site.reflection_id,
                                scene.amplitude if site.parent_id is None else scene.satellite_amplitude,
                                time_ns, energy),
                   scene.sigma_px, site.parent_id, site.role, scene.shape)
        for site in scene.sites(radius)
    )


def render_observation(
    config: BenchmarkConfig, time_ns: int | None = None, *, observation_index: int | None = None,
) -> RenderedObservation:
    reflections = evaluate_scene(config, time_ns, observation_index=observation_index)
    profiles = tuple(sample_profile(r.x_px, r.y_px, r.profile_shape) for r in reflections)
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
    instances = tuple(
        make_instance_masks(r.reflection_id, r.x_px, r.y_px, r.amplitude,
                            r.profile_shape, config.scene.image_size)
        for r in reflections
    ) if config.export_masks else ()
    return RenderedObservation(image, png, reflections, reference, photometry, instances)


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
    shape = reflection.profile_shape
    profile = profiles[index][2]
    without = _compose(config, reflections, profiles, omit=index).astype(np.float32)
    without_png = encode_png(without, config.condition.png_lower, config.condition.png_upper)
    weight = float(profile.sum(dtype=np.float64))
    signal = reflection.amplitude * profile
    return {
        "reflection_id": reflection.reflection_id, "parent_id": reflection.parent_id, "role": reflection.role,
        "h": reflection.h, "k": reflection.k,
        "x_px": reflection.x_px, "y_px": reflection.y_px,
        "amplitude": reflection.amplitude,
        "sigma_u_px": shape.sigma_u_px, "sigma_v_px": shape.sigma_v_px,
        "theta_rad": shape.theta_rad, "extinguished": reflection.amplitude == 0,
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
        and target.profile_shape.overlaps(target.x_px - other.x_px, target.y_px - other.y_px, extent)
    ]


def encode_png(image: NDArray, lower: float, upper: float) -> NDArray[np.uint8]:
    """Map intensities with fixed bounds and round-half-up quantization."""
    scaled = np.clip((image.astype(np.float64) - lower) / (upper - lower), 0, 1)
    return np.floor(255 * scaled + 0.5).astype(np.uint8)
