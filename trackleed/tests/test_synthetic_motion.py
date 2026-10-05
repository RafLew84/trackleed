"""Energy-dependent reference motion through the public scene interface."""
import math
import unittest

import numpy as np

from trackleed.synthetic.config import parse_config
from trackleed.synthetic.scene import evaluate_scene, render_observation


def scan_config():
    return {
        "schema_version": 2,
        "scene": {
            "id": "energy-h6", "image_size": [65, 65], "center_px": [32, 32],
            "lattice_scale_px": 12, "sigma_px": 1,
            "energy_scan": {
                "model": "linear_time", "reference_energy_ev": 100,
                "energies_ev": [80, 200], "times_ns": [0, 2000000000],
                "valid_energy_range_ev": [80, 200], "projection_radius_px": 40,
                "domain_justification": "Ideal projection: all included beams propagate.",
            },
        },
        "condition": {},
        "schedules": [{"id": "sampled", "times_ns": [0, 100000000, 2000000000]}],
    }


class SyntheticMotionTests(unittest.TestCase):
    def test_linear_energy_scan_matches_independent_positions_at_arbitrary_times(self):
        config = parse_config(scan_config())
        # Explicit reference hexagon; the model under test does not supply expected positions.
        offsets = np.array([[-12, 0], [-6, math.sqrt(108)], [-6, -math.sqrt(108)],
                            [6, math.sqrt(108)], [6, -math.sqrt(108)], [12, 0]])
        states = {}
        for time_ns, energy in ((2000000000, 200), (0, 80), (123456789, 87.40740734)):
            reflections = evaluate_scene(config, time_ns)
            actual = np.array([(r.x_px, r.y_px) for r in reflections])
            expected = 32 + offsets / math.sqrt(energy / 100)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
            self.assertTrue(all(r.amplitude == 0.5 and r.sigma_px == 1 for r in reflections))
            states[time_ns] = reflections
        for time_ns in reversed(states):
            self.assertEqual(evaluate_scene(config, time_ns), states[time_ns])
        self.assertEqual([r.reflection_id for r in states[0]],
                         [r.reflection_id for r in states[2000000000]])
        self.assertFalse(np.array_equal(render_observation(config, 0).png,
                                        render_observation(config, 2000000000).png))


    def test_constant_energy_and_common_drift_preserve_shape_and_background(self):
        raw = scan_config()
        raw["scene"].pop("energy_scan")
        raw["scene"]["energy_ev"] = 100
        stationary = parse_config(raw)
        np.testing.assert_array_equal(render_observation(stationary, 0).image,
                                      render_observation(stationary, 2000000000).image)
        raw["scene"]["drift_px_per_s"] = [6, -3]
        moving = parse_config(raw)
        initial = evaluate_scene(moving, 0)
        final = evaluate_scene(moving, 2000000000)
        for before, after in zip(initial, final):
            self.assertEqual((after.x_px - before.x_px, after.y_px - before.y_px), (12, -6))
            self.assertEqual(after.amplitude, before.amplitude)
            self.assertEqual(after.sigma_px, before.sigma_px)
        rendered = render_observation(moving, 2000000000)
        self.assertEqual(rendered.image[0, 0], np.float32(0.1))
        self.assertEqual(rendered.image[26, 56], np.float32(0.6))
        # Time remains valid without an energy label or a reference energy.
        raw["scene"]["energy_ev"] = None
        self.assertEqual(evaluate_scene(parse_config(raw), 2000000000), final)


    def test_regular_sampling_expands_exact_times_with_declared_phase(self):
        raw = scan_config()
        raw["schedules"] = [
            {"id": f"every-{step}ms", "sampling": {
                "start_ns": 0, "end_ns": 2000000000, "interval_ns": step * 1000000, "phase_ns": 0,
            }} for step in (20, 50, 100)
        ]
        config = parse_config(raw)
        self.assertEqual([schedule.count for schedule in config.schedules], [101, 41, 21])
        self.assertEqual(config.plan()["reference_records"], 978)
        raw["schedules"] = [{"id": "offset", "sampling": {
            "start_ns": 0, "end_ns": 200000000, "interval_ns": 50000000, "phase_ns": 10000000,
        }}]
        offset = parse_config(raw)
        self.assertEqual(offset.schedule.times_ns, (10000000, 60000000, 110000000, 160000000))
        self.assertEqual(parse_config(offset.expanded()), offset)


    def test_invalid_motion_and_sampling_contracts_are_rejected(self):
        from copy import deepcopy
        cases = []
        for field, value in (
            ("reference_energy_ev", None), ("valid_energy_range_ev", [90, 200]),
            ("projection_radius_px", 5), ("domain_justification", ""),
            ("times_ns", [0, True]), ("energies_ev", [200, 0]),
        ):
            raw = scan_config()
            raw["scene"]["energy_scan"][field] = value
            cases.append(raw)
        missing = scan_config()
        del missing["scene"]["energy_scan"]["reference_energy_ev"]
        cases.append(missing)
        duplicate = scan_config()
        duplicate["schedules"].append(deepcopy(duplicate["schedules"][0]))
        cases.append(duplicate)
        reserved = scan_config()
        reserved["schedules"][0]["id"] = "CON"
        cases.append(reserved)
        out_of_time = scan_config()
        out_of_time["schedules"][0]["times_ns"] = [0, 2000000001]
        cases.append(out_of_time)
        for bad_time in (None, False, 0, "invalid"):
            raw = scan_config()
            raw["scene"]["energy_scan"].update(model="energy_list", times_ns=None)
            raw["schedules"] = [{"id": "list", "observation_indices": [0, 1], "times_ns": bad_time}]
            cases.append(raw)
        drift = scan_config()
        drift["scene"]["energy_scan"].update(model="energy_list", times_ns=None)
        drift["scene"]["drift_px_per_s"] = [1, 0]
        drift["schedules"] = [{"id": "list", "observation_indices": [0, 1]}]
        cases.append(drift)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_config(raw)


    def test_declared_scan_endpoints_are_exact_even_for_large_energy_ranges(self):
        raw = scan_config()
        raw["scene"]["energy_scan"].update(
            reference_energy_ev=1, energies_ev=[1e18, 1], valid_energy_range_ev=[1, 1e18])
        config = parse_config(raw)
        self.assertEqual(config.energy_at(0), 1e18)
        self.assertEqual(config.energy_at(2000000000), 1)
        last = evaluate_scene(config, 2000000000)[-1]
        self.assertEqual((last.x_px, last.y_px), (44, 32))
