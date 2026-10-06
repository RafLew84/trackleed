"""Report sampled attenuation independently of tracker detection or recovery."""
from dataclasses import asdict, dataclass
from typing import Any

from trackleed.synthetic.attenuation import Attenuation
from trackleed.synthetic.config import BenchmarkConfig, Schedule


COVERAGE_SCHEMA = {
    "file": "coverage.json", "report_version": 1,
    "record": "one event and schedule; all counts use actual ordered observations",
    "sampling_status": "captured iff minimum_sample_count>0; otherwise missed",
    "captured_zero_gap": "sampled declared minimum with amplitude==0; never a tracker result",
    "zero_signal_sample_count": "attenuated samples with zero float64 contribution, separate from amplitude==0",
    "context": "nearest baseline observations outside the entire attenuation, in acquisition order",
    "nonmonotonic": "global before/after and truncation are null; inspect sampled_runs in acquisition order",
    "sampled_runs": "contiguous attenuated observations, not an interpolated energy or time trajectory",
    "truncated_flags": "missing baseline context; outside_observations is separately identified",
    "encoded_loss": "positive float32 contribution lost in PNG; may include quantization or saturation",
}


@dataclass(frozen=True)
class CoverageSample:
    observation_id: str
    coordinate: int | float
    phase: str
    reference: dict[str, Any]


def build_coverage(config: BenchmarkConfig, references: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in references:
        grouped.setdefault((row["schedule_id"], row["reflection_id"]), []).append(row)
    events = []
    for schedule in config.schedules:
        for event in config.scene.attenuations:
            samples = []
            for index, reference in enumerate(grouped[schedule.id, event.reflection_id]):
                coordinates = schedule.coordinates(index)
                point = event.coordinate(coordinates["time_ns"], config.energy_at(**coordinates))
                samples.append(CoverageSample(schedule.observation_id(index), point, event.phase(point), reference))
            events.append(_event_report(schedule, event, samples))
    return {"report_version": 1, "scene_id": config.scene.id,
            "condition_id": config.condition.id, "realization_id": config.realization_id,
            "events": events}


def _event_report(schedule: Schedule, event: Attenuation, samples: list[CoverageSample]) -> dict[str, Any]:
    active = [s for s in samples if s.phase in ("fade", "minimum", "return")]
    minimum = [s for s in samples if s.phase == "minimum"]
    zero = [s for s in active if s.reference["amplitude"] == 0]
    signal_zero = [s for s in active if s.reference["float64_signal_pixels"] == 0]
    float32_loss = [s for s in active if s.reference["amplitude"] > 0
                    and s.reference["float64_signal_pixels"] > 0 and s.reference["float32_changed_pixels"] == 0]
    encoded_loss = [s for s in active if s.reference["amplitude"] > 0
                    and s.reference["float32_changed_pixels"] > 0 and s.reference["encoded_changed_pixels"] == 0]
    return {
        "event_id": event.id, "reflection_id": event.reflection_id, "schedule_id": schedule.id,
        "definition": asdict(event), "observation_count": len(samples),
        "minimum_sample_count": len(minimum), "attenuated_sample_count": len(active),
        "zero_amplitude_sample_count": len(zero), "zero_signal_sample_count": len(signal_zero),
        "minimum_observation_ids": [s.observation_id for s in minimum],
        "zero_amplitude_observation_ids": [s.observation_id for s in zero],
        "positive_signal_lost_in_float32_count": len(float32_loss),
        "positive_float32_signal_lost_in_png_count": len(encoded_loss),
        "sampling_status": "captured" if minimum else "missed",
        "attenuation_observed": bool(active), "captured_zero_gap": bool(zero),
        "sampled_runs": _sampled_runs(samples),
        **_context(samples),
    }


def _context(samples: list[CoverageSample]) -> dict[str, Any]:
    increasing = any(b.coordinate > a.coordinate for a, b in zip(samples, samples[1:]))
    decreasing = any(b.coordinate < a.coordinate for a, b in zip(samples, samples[1:]))
    if increasing and decreasing:
        return {"context_status": "nonmonotonic", "before_observation_id": None,
                "after_observation_id": None, "truncated_before": None, "truncated_after": None}
    descending = samples[0].coordinate > samples[-1].coordinate
    before_phase, after_phase = ("after", "before") if descending else ("before", "after")
    before = [s.observation_id for s in samples if s.phase == before_phase]
    after = [s.observation_id for s in samples if s.phase == after_phase]
    active = any(s.phase in ("fade", "minimum", "return") for s in samples)
    status = "bracketed" if before and after else ("truncated" if active else "outside_observations")
    return {"context_status": status, "before_observation_id": before[-1] if before else None,
            "after_observation_id": after[0] if after else None,
            "truncated_before": not bool(before), "truncated_after": not bool(after)}


def _sampled_runs(samples: list[CoverageSample]) -> list[dict[str, Any]]:
    runs = []
    index = 0
    while index < len(samples):
        if samples[index].phase in ("before", "after"):
            index += 1
            continue
        start = index
        while index < len(samples) and samples[index].phase not in ("before", "after"):
            index += 1
        selected = samples[start:index]
        runs.append({
            "observation_ids": [s.observation_id for s in selected],
            "minimum_observation_ids": [s.observation_id for s in selected if s.phase == "minimum"],
            "before_observation_id": samples[start - 1].observation_id if start else None,
            "after_observation_id": samples[index].observation_id if index < len(samples) else None,
            "truncated_before": start == 0, "truncated_after": index == len(samples),
        })
    return runs
