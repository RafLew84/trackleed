"""Versioned configuration for the first, static H6 benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class Scene:
    id: str = "static-h6"
    pattern: str = "H6"
    image_size: tuple[int, int] = (1000, 1000)
    center_px: tuple[float, float] = (499.5, 499.5)
    lattice_scale_px: float = 180.0
    sigma_px: float = 4.0
    amplitude: float = 0.5
    energy_ev: float | None = 100.0


@dataclass(frozen=True)
class Condition:
    id: str = "constant"
    background: float = 0.1
    png_lower: float = 0.0
    png_upper: float = 1.0


@dataclass(frozen=True)
class Schedule:
    id: str = "static"
    times_ns: tuple[int, ...] = (0, 50000000, 100000000)


@dataclass(frozen=True)
class BenchmarkConfig:
    scene: Scene
    condition: Condition
    schedule: Schedule
    realization_id: str = "deterministic"
    schema_version: int = 1

    def plan(self) -> dict[str, int]:
        frames = len(self.schedule.times_ns)
        return {"sequences": 1, "frames": frames, "reflections": 6,
                "reference_records": frames * 6}

    def expanded(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: Path) -> Any:
    """Read strict JSON, rejecting duplicate keys and non-finite constants."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_unique_keys, parse_constant=_bad_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_config(path: Path) -> BenchmarkConfig:
    return parse_config(read_json(path))


def parse_config(raw: Any) -> BenchmarkConfig:
    root = _object(raw, {f.name for f in fields(BenchmarkConfig)}, "config")
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    scene = _object(root.get("scene"), {f.name for f in fields(Scene)}, "scene")
    condition = _object(root.get("condition"), {f.name for f in fields(Condition)},
                        "condition")
    schedule = _object(root.get("schedule"), {f.name for f in fields(Schedule)},
                       "schedule")
    for name in ("image_size", "center_px"):
        if name in scene:
            value = scene[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"scene.{name} must contain two numbers")
            scene[name] = tuple(value)
    if "times_ns" in schedule:
        value = schedule["times_ns"]
        if not isinstance(value, (list, tuple)):
            raise ValueError("schedule.times_ns must be an array")
        schedule["times_ns"] = tuple(value)
    config = BenchmarkConfig(Scene(**scene), Condition(**condition), Schedule(**schedule),
                             root.get("realization_id", "deterministic"))
    _validate(config)
    return config


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label}: unsupported fields {sorted(unknown)}")
    return dict(value)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number: {value}")


def _number(value: Any, label: str, minimum: float | None = None) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return float(value)


def _identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is None:
        raise ValueError(f"{label} must be a portable identifier")


def _validate(config: BenchmarkConfig) -> None:
    scene, condition, schedule = config.scene, config.condition, config.schedule
    for label, value in (("scene.id", scene.id), ("condition.id", condition.id),
                         ("schedule.id", schedule.id),
                         ("realization_id", config.realization_id)):
        _identifier(value, label)
    if scene.pattern != "H6":
        raise ValueError("scene.pattern supports only H6 in this stage")
    if any(type(v) is not int or v <= 0 for v in scene.image_size):
        raise ValueError("scene.image_size must contain positive integers [width, height]")
    for coordinate in scene.center_px:
        _number(coordinate, "scene.center_px")
    if _number(scene.sigma_px, "scene.sigma_px", 0) == 0:
        raise ValueError("scene.sigma_px must be positive")
    if _number(scene.lattice_scale_px, "scene.lattice_scale_px", 0) == 0:
        raise ValueError("scene.lattice_scale_px must be positive")
    _number(scene.amplitude, "scene.amplitude", 0)
    if scene.energy_ev is not None and _number(scene.energy_ev, "scene.energy_ev", 0) == 0:
        raise ValueError("scene.energy_ev must be positive or null")
    _number(condition.background, "condition.background", 0)
    lower = _number(condition.png_lower, "condition.png_lower")
    upper = _number(condition.png_upper, "condition.png_upper")
    if upper <= lower or not math.isfinite(upper - lower):
        raise ValueError("condition.png_upper must exceed png_lower by a finite range")
    times = schedule.times_ns
    if not times or any(type(t) is not int or not 0 <= t <= 2**63 - 1 for t in times):
        raise ValueError("schedule.times_ns needs nonnegative int64 nanoseconds")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("schedule.times_ns must be strictly increasing")
    _validate_support(scene)


def _validate_support(scene: Scene) -> None:
    # Stage 1 explicitly rejects clipping; apparatus restrictions arrive in stage 5.
    radius, sigma = scene.lattice_scale_px, scene.sigma_px
    extents = (radius + 4 * sigma, math.sqrt(3) * radius / 2 + 4 * sigma)
    for center, extent, size in zip(scene.center_px, extents, scene.image_size):
        if not math.isfinite(extent) or center - extent < 0 or center + extent > size - 1:
            raise ValueError("scene support must fit inside the frame in this stage")
