"""Versioned benchmark configuration and exact observation coordinates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re
from typing import Any

from trackleed.synthetic.apparatus import Apparatus, parse_apparatus
from trackleed.synthetic.attenuation import Attenuation, parse_attenuations, validate_attenuations
from trackleed.synthetic.geometry import LatticeSite, lattice_sites
from trackleed.synthetic.profiles import GaussianShape


@dataclass(frozen=True)
class EnergyScan:
    model: str
    reference_energy_ev: float
    energies_ev: tuple[float, ...]
    times_ns: tuple[int, int] | None
    valid_energy_range_ev: tuple[float, float]
    projection_radius_px: float
    domain_justification: str

    def energy_at(self, time_ns: int) -> float:
        if self.model != "linear_time" or self.times_ns is None:
            raise ValueError("This energy scan is not parameterized by time")
        start, end = self.times_ns
        if not start <= time_ns <= end:
            raise ValueError("time_ns lies outside the declared energy scan")
        fraction = (time_ns - start) / (end - start)
        first, last = self.energies_ev
        if first <= last:
            return first + fraction * (last - first)
        # Interpolate from the lower endpoint to avoid cancellation at low energy.
        return last + ((end - time_ns) / (end - start)) * (first - last)


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
    energy_scan: EnergyScan | None = None
    drift_px_per_s: tuple[float, float] = (0.0, 0.0)
    satellite_offset_px: float = 18.0
    satellite_amplitude: float = 0.15
    shape: GaussianShape | None = None
    attenuations: tuple[Attenuation, ...] = ()

    def lattice_radius(self, energy_ev: float | None) -> float:
        if self.energy_scan is None:
            return self.lattice_scale_px
        if energy_ev is None:
            raise ValueError("energy_scan requires a known energy")
        return self.lattice_scale_px * math.sqrt(self.energy_scan.reference_energy_ev / energy_ev)

    def support_extents(self, radius: float) -> tuple[float, float]:
        sites = self.sites(radius)
        ex, ey = self.profile_shape.extents()
        return max(abs(s.x) for s in sites) + ex, max(abs(s.y) for s in sites) + ey

    @property
    def profile_shape(self) -> GaussianShape:
        return self.shape or GaussianShape(self.sigma_px, self.sigma_px)

    def sites(self, radius: float) -> tuple[LatticeSite, ...]:
        return lattice_sites(self.pattern, radius, self.satellite_offset_px * (radius / self.lattice_scale_px))

    def maximum_radius(self, energy_ev: float | None) -> float:
        return max(math.hypot(s.x, s.y) for s in self.sites(self.lattice_radius(energy_ev)))


@dataclass(frozen=True)
class Condition:
    id: str = "constant"
    background: float = 0.1
    png_lower: float = 0.0
    png_upper: float = 1.0
    apparatus: Apparatus | None = None


@dataclass(frozen=True)
class Sampling:
    start_ns: int
    end_ns: int
    interval_ns: int
    phase_ns: int

    def times(self) -> tuple[int, ...]:
        return tuple(range(self.start_ns + self.phase_ns, self.end_ns + 1, self.interval_ns))


@dataclass(frozen=True)
class Schedule:
    id: str = "static"
    times_ns: tuple[int, ...] = (0, 50000000, 100000000)
    observation_indices: tuple[int, ...] = ()
    sampling: Sampling | None = None

    @property
    def count(self) -> int:
        return len(self.observation_indices or self.times_ns)

    def coordinates(self, index: int) -> dict[str, int | None]:
        if self.observation_indices:
            return {"time_ns": None, "observation_index": self.observation_indices[index]}
        return {"time_ns": self.times_ns[index], "observation_index": None}

    def observation_id(self, index: int) -> str:
        return (f"e{self.observation_indices[index]}" if self.observation_indices
                else f"t{self.times_ns[index]}")


@dataclass(frozen=True)
class BenchmarkConfig:
    scene: Scene
    condition: Condition
    schedules: tuple[Schedule, ...]
    realization_id: str = "deterministic"
    schema_version: int = 1
    export_masks: bool = False

    @property
    def schedule(self) -> Schedule:
        if len(self.schedules) != 1:
            raise ValueError("Select a schedule from this multi-schedule configuration")
        return self.schedules[0]

    def energy_at(self, time_ns: int | None = None, *, observation_index: int | None = None) -> float | None:
        scan = self.scene.energy_scan
        if scan is not None and scan.model == "energy_list":
            if time_ns is not None:
                raise ValueError("energy_list observations do not have time_ns")
            if type(observation_index) is not int or not 0 <= observation_index < len(scan.energies_ev):
                raise ValueError("observation_index must select an energy_list observation")
            return scan.energies_ev[observation_index]
        if observation_index is not None:
            raise ValueError("observation_index requires energy_list")
        if type(time_ns) is not int or not 0 <= time_ns <= 2**63 - 1:
            raise ValueError("time_ns must be a nonnegative int64")
        return scan.energy_at(time_ns) if scan is not None else self.scene.energy_ev

    def plan(self) -> dict[str, int]:
        frames = sum(schedule.count for schedule in self.schedules)
        count = {"H6": 6, "H18": 18, "H54": 54}[self.scene.pattern]
        return {"sequences": len(self.schedules), "frames": frames, "reflections": count,
                "reference_records": frames * count}

    def expanded(self) -> dict[str, Any]:
        result = asdict(self)
        if self.schema_version < 5:
            result["condition"].pop("apparatus")
        if self.schema_version < 4:
            result["scene"].pop("attenuations")
        if self.schema_version < 3:
            result.pop("export_masks")
            result["scene"].pop("satellite_offset_px")
            result["scene"].pop("satellite_amplitude")
            result["scene"].pop("shape")
        elif self.scene.shape is not None:
            result["scene"].pop("sigma_px")
        if self.schema_version == 1:
            result["schedule"] = result.pop("schedules")[0]
            result["scene"].pop("energy_scan")
            result["scene"].pop("drift_px_per_s")
            result["schedule"].pop("observation_indices")
            result["schedule"].pop("sampling")
        return result


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
    root = _object(raw, {"scene", "condition", "schedule", "schedules",
                         "realization_id", "schema_version", "export_masks"}, "config")
    version = root.get("schema_version")
    if type(version) is not int or version not in (1, 2, 3, 4, 5):
        raise ValueError("schema_version must be integer 1, 2, 3, 4 or 5")
    if "export_masks" in root and version < 3:
        raise ValueError("export_masks requires schema_version 3")
    if type(root.get("export_masks", False)) is not bool:
        raise ValueError("export_masks must be boolean")
    scene_fields = {f.name for f in fields(Scene)}
    if version < 4:
        scene_fields.remove("attenuations")
    if version < 3:
        scene_fields.difference_update({"satellite_offset_px", "satellite_amplitude", "shape"})
    if version == 1:
        scene_fields.difference_update({"energy_scan", "drift_px_per_s"})
    scene = _object(root.get("scene"), scene_fields, "scene")
    if "attenuations" in scene:
        scene["attenuations"] = parse_attenuations(scene["attenuations"])
    if scene.get("shape") is not None:
        if "sigma_px" in scene:
            raise ValueError("scene.shape and scene.sigma_px cannot both be specified")
        shape_values = _object(scene["shape"], {f.name for f in fields(GaussianShape)}, "scene.shape")
        shape = GaussianShape(**shape_values)
        for name in ("sigma_u_px", "sigma_v_px"):
            if _number(getattr(shape, name), f"scene.shape.{name}", 0) == 0:
                raise ValueError(f"scene.shape.{name} must be positive")
        _number(shape.theta_rad, "scene.shape.theta_rad")
        scene["shape"] = shape
    condition = _object(root.get("condition"), {f.name for f in fields(Condition)},
                        "condition")
    if "apparatus" in condition:
        if version < 5:
            raise ValueError("condition.apparatus requires schema_version 5")
        if condition["apparatus"] is not None:
            condition["apparatus"] = parse_apparatus(condition["apparatus"])
    for name in ("image_size", "center_px", "drift_px_per_s"):
        if name in scene:
            scene[name] = _array(scene[name], f"scene.{name}", length=2)
    if scene.get("energy_scan") is not None:
        scene["energy_scan"] = _parse_scan(scene["energy_scan"])
        if "energy_ev" in scene and scene["energy_ev"] is not None:
            raise ValueError("scene.energy_ev conflicts with energy_scan")
        scene["energy_ev"] = None
    schedules: tuple[Schedule, ...]
    if version == 1:
        if "schedules" in root:
            raise ValueError("schedules requires schema_version 2")
        schedules = (_parse_schedule(root.get("schedule"), version),)
    else:
        if "schedule" in root:
            raise ValueError("schema_version 2 and later use schedules")
        values = _array(root.get("schedules"), "schedules")
        if not values:
            raise ValueError("schedules must not be empty")
        schedules = tuple(_parse_schedule(value, version) for value in values)
    config = BenchmarkConfig(Scene(**scene), Condition(**condition), schedules,
                             root.get("realization_id", "deterministic"), version,
                             root.get("export_masks", False))
    _validate(config)
    return config


def _array(value: Any, label: str, length: int | None = None) -> tuple:
    if not isinstance(value, (list, tuple)) or (length is not None and len(value) != length):
        raise ValueError(f"{label} must be an array" + (f" of length {length}" if length else ""))
    return tuple(value)


def _parse_schedule(value: Any, version: int) -> Schedule:
    names = {f.name for f in fields(Schedule)}
    if version == 1:
        names.difference_update({"observation_indices", "sampling"})
    values = _object(value, names, "schedule")
    if "times_ns" in values:
        values["times_ns"] = _array(values["times_ns"], "schedule.times_ns")
    if "observation_indices" in values:
        values["observation_indices"] = _array(values["observation_indices"], "schedule.observation_indices")
        if values["observation_indices"]:
            if values.get("times_ns"):
                raise ValueError("schedule must select time or energy observations, not both")
            values["times_ns"] = ()
    if values.get("sampling") is not None:
        sampling = _parse_sampling(values["sampling"])
        times = sampling.times()
        if values.get("observation_indices") or ("times_ns" in values and values["times_ns"] != times):
            raise ValueError("schedule.sampling disagrees with explicit observations")
        values.update(sampling=sampling, times_ns=times)
    if version >= 2 and not any(values.get(key) for key in ("times_ns", "observation_indices")):
        raise ValueError("schedule requires explicit times_ns, sampling or observation_indices")
    return Schedule(**values)


def _parse_sampling(value: Any) -> Sampling:
    names = {f.name for f in fields(Sampling)}
    values = _object(value, names, "schedule.sampling")
    if set(values) != names or any(type(v) is not int or not 0 <= v <= 2**63 - 1
                                  for v in values.values()):
        raise ValueError("schedule.sampling requires nonnegative int64 start/end/interval/phase")
    start, end = values["start_ns"], values["end_ns"]
    interval, phase = values["interval_ns"], values["phase_ns"]
    if interval == 0 or phase >= interval or start + phase > end:
        raise ValueError("schedule.sampling requires interval > phase >= 0 and a nonempty range")
    if (end - start - phase) // interval + 1 > 1000000:
        raise ValueError("schedule.sampling exceeds the 1000000-observation limit")
    return Sampling(**values)


def _parse_scan(value: Any) -> EnergyScan:
    names = {f.name for f in fields(EnergyScan)}
    values = _object(value, names, "scene.energy_scan")
    if set(values) != names:
        raise ValueError(f"scene.energy_scan requires {sorted(names - set(values))}")
    model = values["model"]
    if model not in ("linear_time", "energy_list"):
        raise ValueError("energy_scan.model supports linear_time or energy_list")
    values["valid_energy_range_ev"] = _array(values["valid_energy_range_ev"],
                                             "energy_scan.valid_energy_range_ev", length=2)
    values["energies_ev"] = _array(values["energies_ev"], "energy_scan.energies_ev",
                                   length=2 if model == "linear_time" else None)
    if not values["energies_ev"]:
        raise ValueError("energy_scan.energies_ev must not be empty")
    if model == "linear_time":
        values["times_ns"] = _array(values["times_ns"], "energy_scan.times_ns", length=2)
        _validate_times(values["times_ns"], "energy_scan.times_ns")
    elif values["times_ns"] is not None:
        raise ValueError("energy_list times_ns must be null")
    for name in ("reference_energy_ev", "projection_radius_px"):
        if _number(values[name], f"energy_scan.{name}", 0) == 0:
            raise ValueError(f"energy_scan.{name} must be positive")
    energies, domain = values["energies_ev"], values["valid_energy_range_ev"]
    for energy in (*energies, *domain):
        if _number(energy, "energy_scan energy", 0) == 0:
            raise ValueError("energy_scan energies must be positive")
    if domain[0] > domain[1]:
        raise ValueError("valid_energy_range_ev must be ordered")
    if any(not domain[0] <= energy <= domain[1]
           for energy in (*energies, values["reference_energy_ev"])):
        raise ValueError("energy_scan exceeds valid_energy_range_ev")
    if not isinstance(values["domain_justification"], str) or not values["domain_justification"].strip():
        raise ValueError("energy_scan.domain_justification is required")
    return EnergyScan(**values)


def _validate_times(times: tuple[int, ...], label: str) -> None:
    if not times or any(type(t) is not int or not 0 <= t <= 2**63 - 1 for t in times):
        raise ValueError(f"{label} needs nonnegative int64 nanoseconds")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{label} must be strictly increasing")


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
    scene, condition = config.scene, config.condition
    for label, value in (("scene.id", scene.id), ("condition.id", condition.id),
                         ("realization_id", config.realization_id)):
        _identifier(value, label)
    allowed_patterns = ("H6", "H18", "H54") if config.schema_version >= 3 else ("H6",)
    if scene.pattern not in allowed_patterns:
        raise ValueError(f"scene.pattern supports {allowed_patterns}")
    if _number(scene.satellite_offset_px, "scene.satellite_offset_px", 0) == 0:
        raise ValueError("scene.satellite_offset_px must be positive")
    _number(scene.satellite_amplitude, "scene.satellite_amplitude", 0)
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
    identifiers = set()
    for schedule in config.schedules:
        _identifier(schedule.id, "schedule.id")
        if config.schema_version >= 2 and re.fullmatch(
                r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", schedule.id, re.IGNORECASE):
            raise ValueError("schedule.id is a reserved Windows device name")
        if schedule.id.casefold() in identifiers:
            raise ValueError("schedule.id must be unique, ignoring case")
        identifiers.add(schedule.id.casefold())
        scan = scene.energy_scan
        if scan is not None and scan.model == "energy_list":
            indices = schedule.observation_indices
            if (schedule.times_ns or not indices
                    or any(type(i) is not int or not 0 <= i < len(scan.energies_ev) for i in indices)
                    or any(b <= a for a, b in zip(indices, indices[1:]))):
                raise ValueError("schedule.observation_indices must select ordered energy observations")
        else:
            if schedule.observation_indices:
                raise ValueError("schedule.observation_indices requires energy_list")
            _validate_times(schedule.times_ns, "schedule.times_ns")
        for index in range(schedule.count):
            config.energy_at(**schedule.coordinates(index))
    for speed in scene.drift_px_per_s:
        _number(speed, "scene.drift_px_per_s")
    scan = scene.energy_scan
    if scan is not None and scan.model == "energy_list" and any(scene.drift_px_per_s):
        raise ValueError("drift_px_per_s requires known observation times")
    if scan is not None:
        if scene.maximum_radius(scan.valid_energy_range_ev[0]) / scan.projection_radius_px > 1:
            raise ValueError("energy_scan declared domain violates beam propagation")
    _validate_support(config)
    validate_attenuations(config.scene.attenuations,
                          {site.reflection_id: (config.scene.amplitude if site.parent_id is None
                                                else config.scene.satellite_amplitude)
                           for site in config.scene.sites(config.scene.lattice_scale_px)},
                          has_time=not bool(config.schedules[0].observation_indices),
                          has_energy=config.scene.energy_scan is not None or config.scene.energy_ev is not None)


def _validate_support(config: BenchmarkConfig) -> None:
    scene = config.scene
    # Historical versions require full support; v5 permits finite off-frame positions.
    scan = scene.energy_scan
    energy = min(scan.energies_ev) if scan is not None else scene.energy_ev
    extents = scene.support_extents(scene.lattice_radius(energy))
    times = [time for schedule in config.schedules for time in schedule.times_ns]
    if scene.energy_scan is not None and scene.energy_scan.times_ns is not None:
        times.extend(scene.energy_scan.times_ns)
    first, last = min(times, default=0) / 1e9, max(times, default=0) / 1e9
    for center, extent, size, speed in zip(scene.center_px, extents, scene.image_size, scene.drift_px_per_s):
        low, high = sorted((center + first * speed, center + last * speed))
        if (not all(math.isfinite(value) for value in (extent, low - extent, high + extent))
                or (config.schema_version < 5 and (low - extent < 0 or high + extent > size - 1))):
            raise ValueError("scene support must fit inside the frame in this stage")
