"""Version 1 tables: units, missing values and canonical CSV encoding."""
from __future__ import annotations

from typing import Any

from trackleed.synthetic.config import BenchmarkConfig, Schedule
from trackleed.synthetic.scene import RenderedObservation, evaluate_scene

FRAME_FIELDS = {
    "schedule_id": "string", "frame_index": "int64", "observation_id": "string",
    "time_ns": "int64:ns", "energy_ev": "nullable:float64:eV",
    "previous_observation_id": "nullable:string", "dt_ns": "nullable:int64:ns",
    "de_ev": "nullable:float64:eV", "png_path": "string", "tiff_path": "string",
    "png_sha256": "sha256", "tiff_sha256": "sha256",
    "png_pixel_sha256": "sha256", "tiff_pixel_sha256": "sha256",
    "below_lower_pixels": "int64", "above_upper_pixels": "int64",
    "png_zero_pixels": "int64", "png_255_pixels": "int64",
}
REFERENCE_FIELDS = {
    "schedule_id": "string", "frame_index": "int64", "observation_id": "string",
    "previous_observation_id": "nullable:string",
    "dx_px": "nullable:float64:px", "dy_px": "nullable:float64:px",
    "reflection_id": "string", "parent_id": "nullable:string", "role": "string",
    "h": "int64", "k": "int64", "x_px": "float64:px", "y_px": "float64:px",
    "amplitude": "float64:image_units", "sigma_u_px": "float64:px",
    "sigma_v_px": "float64:px", "theta_rad": "float64:rad",
    "extinguished": "boolean", "center_in_frame": "boolean",
    "w0": "float64", "w1": "float64", "w2": "float64", "w3": "float64",
    "frame_loss_fraction": "float64", "screen_loss_fraction": "float64",
    "occluder_loss_fraction": "float64", "retained_fraction": "float64",
    "prequant_signal_present": "boolean", "float64_signal_pixels": "int64",
    "float32_signal_pixels": "int64", "float32_changed_pixels": "int64",
    "encoded_changed_pixels": "int64", "overlap_contour_ids": "json:string[]",
    "overlap_support_ids": "json:string[]",
}
SCHEMA = {
    "schema_version": 1, "frames": FRAME_FIELDS, "reference": REFERENCE_FIELDS,
    "csv": {"encoding": "UTF-8", "separator": ",", "decimal": ".", "missing": ""},
    "coordinates": {"x": "right", "y": "down", "top_left_pixel_center": [0, 0]},
    "observation_type": "instantaneous", "exposure_ns": None,
    "rendering": {"profile": "circular_gaussian", "sampling": "pixel_centers",
                  "support_sigma": 4, "support_inclusive": True,
                  "omitted_continuous_mass": 0.00033546262790251185,
                  "nominal_contour_sigma": 3, "overlap": "analytic_closed_disks"},
    "encoding": {"tiff": "float32_unclipped", "png": "uint8",
                 "quantization": "floor(255*clip((I-L)/(U-L),0,1)+0.5)",
                 "counterfactual": "recompose_without_reflection_then_float32"},
    "stage": {"pattern": "H6", "background": "constant", "noise": "none",
              "screen": "full_frame", "occluders": "none", "instance_masks": False},
}


def schema_for(version: int) -> dict[str, Any]:
    if version == 1:
        return SCHEMA
    return {**SCHEMA, "schema_version": 2,
            "frames": {**FRAME_FIELDS, "time_ns": "nullable:int64:ns"}}


def frame_identity(schedule: Schedule, index: int) -> dict[str, Any]:
    return {
        "schedule_id": schedule.id, "frame_index": index,
        "observation_id": schedule.observation_id(index),
        "previous_observation_id": schedule.observation_id(index - 1) if index else None,
    }


def reference_rows(
    config: BenchmarkConfig, observation: RenderedObservation, index: int, schedule: Schedule,
) -> list[dict[str, Any]]:
    previous = ({r.reflection_id: r for r in evaluate_scene(config, **schedule.coordinates(index - 1))}
                if index else {})
    return [{**frame_identity(schedule, index),
             "dx_px": state["x_px"] - previous[state["reflection_id"]].x_px if index else None,
             "dy_px": state["y_px"] - previous[state["reflection_id"]].y_px if index else None,
             **state} for state in observation.reference]
