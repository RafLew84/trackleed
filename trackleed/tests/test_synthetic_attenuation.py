"""Attenuation changes signal while every reflection keeps its latent state."""
import copy
import math
import unittest

import numpy as np

from trackleed.synthetic.config import parse_config
from trackleed.synthetic.scene import evaluate_scene, render_observation


def attenuation_config(**event_values):
    event = {"id": "gap", "reflection_id": "h+1_k+0", "axis": "time_ns",
             "knots": [700000000, 700000000, 1100000000, 1100000000],
             "minimum_fraction": 0}
    event.update(event_values)
    return {"schema_version": 4, "scene": {
        "pattern": "H18", "image_size": [65, 65], "center_px": [32, 32],
        "lattice_scale_px": 10, "sigma_px": 1, "attenuations": [event]},
        "condition": {}, "schedules": [{"id": "sampled", "times_ns": [0, 1400000000]}]}


class SyntheticAttenuationTests(unittest.TestCase):
    def test_step_extinction_keeps_all_states_and_exact_half_open_boundaries(self):
        config = parse_config(attenuation_config())
        original = {r.reflection_id: r for r in evaluate_scene(config, 0)}
        for time, expected in ((699999999, .5), (700000000, 0),
                               (1099999999, 0), (1100000000, .5)):
            with self.subTest(time=time):
                states = {r.reflection_id: r for r in evaluate_scene(config, time)}
                self.assertEqual(set(states), set(original))
                self.assertEqual(len(states), 18)
                for key, state in states.items():
                    self.assertEqual((state.x_px, state.y_px, state.parent_id),
                                     (original[key].x_px, original[key].y_px, original[key].parent_id))
                    self.assertEqual(state.amplitude, expected if key == "h+1_k+0" else .5)
        self.assertEqual(parse_config(config.expanded()), config)

    def test_linear_fade_and_return_preserve_exact_minimum_and_large_time_keys(self):
        for origin in (0, 2**53):
            for minimum, midpoint in ((0, .25), (.05, .2625)):
                raw = attenuation_config(knots=[origin + t * 1000000 for t in (500, 700, 1100, 1300)],
                                         minimum_fraction=minimum)
                config = parse_config(raw)
                values = ((500, .5), (600, midpoint), (700, .5 * minimum),
                          (1100, .5 * minimum), (1200, midpoint), (1300, .5))
                for ms, expected in values:
                    states = evaluate_scene(config, origin + ms * 1000000)
                    target = next(r for r in states if r.reflection_id == "h+1_k+0")
                    self.assertAlmostEqual(target.amplitude, expected, places=15)
                if minimum == 0:
                    for delta in (699999999, 1100000001):
                        target = next(r for r in evaluate_scene(config, origin + delta)
                                      if r.reflection_id == "h+1_k+0")
                        self.assertGreater(target.amplitude, 0)
                        self.assertLess(target.amplitude, 1e-8)

    def test_zero_five_percent_float32_loss_and_png_loss_remain_distinct(self):
        for minimum, background, signal32, changed32, changed_png in (
            (0, .1, False, False, False), (.05, .1, True, True, True),
            (2e-50, .1, False, False, False), (2e-6, .1, True, True, False),
            (.05, 2, True, True, False),
        ):
            with self.subTest(minimum=minimum, background=background):
                raw = attenuation_config(minimum_fraction=minimum)
                raw["condition"]["background"] = background
                raw["export_masks"] = True
                config = parse_config(raw)
                observation = render_observation(config, 800000000)
                state = next(r for r in observation.reference if r["reflection_id"] == "h+1_k+0")
                instance = next(r for r in observation.instances if r.reflection_id == "h+1_k+0")
                self.assertEqual(state["amplitude"], .5 * minimum)
                self.assertEqual(state["extinguished"], minimum == 0)
                self.assertEqual(state["prequant_signal_present"], minimum > 0)
                self.assertEqual(state["float32_signal_pixels"] > 0, signal32)
                self.assertEqual(state["float32_changed_pixels"] > 0, changed32)
                self.assertEqual(state["encoded_changed_pixels"] > 0, changed_png)
                self.assertTrue(instance.geometry.any())
                self.assertEqual(bool(instance.signal.any()), minimum > 0)
                if minimum > 0:
                    np.testing.assert_array_equal(instance.signal, instance.geometry)
                if background == 2:
                    self.assertEqual(observation.photometry["above_upper_pixels"], 65 * 65)

    def test_satellite_gap_preserves_parent_and_common_energy_motion(self):
        raw = attenuation_config(reflection_id="h+1_k+0_s0")
        raw["scene"].update(pattern="H54", image_size=[129, 129], center_px=[64, 64],
                             lattice_scale_px=20, sigma_px=2, satellite_offset_px=9)
        raw["scene"]["energy_scan"] = {"model": "linear_time", "reference_energy_ev": 100,
            "energies_ev": [80, 200], "times_ns": [0, 1400000000],
            "valid_energy_range_ev": [80, 200], "projection_radius_px": 100,
            "domain_justification": "Ideal H54 test with all beams propagating."}
        raw["export_masks"] = True
        config = parse_config(raw)
        for time, energy, amplitude in ((0, 80, .15), (700000000, 140, 0), (1400000000, 200, .15)):
            observation = render_observation(config, time)
            states = {r["reflection_id"]: r for r in observation.reference}
            self.assertEqual(len(states), 54)
            parent, satellite = states["h+1_k+0"], states["h+1_k+0_s0"]
            self.assertEqual(parent["amplitude"], .5)
            self.assertEqual(satellite["amplitude"], amplitude)
            self.assertEqual(satellite["parent_id"], "h+1_k+0")
            np.testing.assert_allclose([parent["x_px"], satellite["x_px"]],
                                       [64 + 20 * math.sqrt(100 / energy), 64 + 29 * math.sqrt(100 / energy)],
                                       atol=1e-9, rtol=1e-12)
            if amplitude == 0:
                self.assertEqual(satellite["encoded_changed_pixels"], 0)
                self.assertIn("h+1_k+0", satellite["overlap_contour_ids"])
                masks = {m.reflection_id: m for m in observation.instances}
                self.assertFalse(masks["h+1_k+0_s0"].signal.any())
                self.assertTrue(masks["h+1_k+0"].signal.any())

    def test_energy_ramp_is_a_function_of_energy_in_either_scan_direction(self):
        for energies in ([80, 200], [200, 80]):
            raw = attenuation_config(axis="energy_ev", knots=[100, 120, 140, 160], minimum_fraction=.05)
            raw["scene"]["energy_scan"] = {"model": "linear_time", "reference_energy_ev": 100,
                "energies_ev": energies, "times_ns": [0, 1200000000],
                "valid_energy_range_ev": [80, 200], "projection_radius_px": 100,
                "domain_justification": "Controlled energy profile test."}
            raw["schedules"][0]["times_ns"] = [0, 1200000000]
            config = parse_config(raw)
            for energy, expected in ((100, .5), (110, .2625), (120, .025), (140, .025), (150, .2625), (160, .5)):
                time = abs(energy - energies[0]) * 10000000
                target = next(r for r in evaluate_scene(config, time) if r.reflection_id == "h+1_k+0")
                self.assertAlmostEqual(target.amplitude, expected, places=15)

    def test_invalid_profiles_fail_at_configuration_boundary_without_mutating_input(self):
        cases = [("axis", "frames"), ("reflection_id", "missing"), ("knots", [0, 1, 1, 2]),
                 ("knots", [-1, 0, 1, 2]), ("knots", [0, 0, 1, 2**63]),
                 ("knots", [0, False, 1, 2]), ("knots", [0, 0, 1]),
                 ("minimum_fraction", 1), ("minimum_fraction", -.1),
                 ("minimum_fraction", float("nan")), ("minimum_fraction", True)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_config(attenuation_config(**{key: value}))
        raw = attenuation_config()
        original = copy.deepcopy(raw)
        parse_config(raw)
        self.assertEqual(raw, original)
        raw["schema_version"] = 3
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            parse_config(raw)
        raw = attenuation_config()
        raw["scene"]["attenuations"] *= 2
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_config(raw)
        raw["scene"]["attenuations"][1] = {**raw["scene"]["attenuations"][1], "id": "other"}
        with self.assertRaisesRegex(ValueError, "one attenuation"):
            parse_config(raw)
        raw = attenuation_config(axis="energy_ev", knots=[1, 1, 2, 2])
        raw["scene"]["energy_ev"] = None
        with self.assertRaisesRegex(ValueError, "known energy_ev"):
            parse_config(raw)
        raw = attenuation_config()
        raw["scene"]["amplitude"] = 0
        with self.assertRaisesRegex(ValueError, "positive base"):
            parse_config(raw)
