"""Physical limits affect the signal without moving the latent reference."""
import copy
import math
import unittest

import numpy as np

from trackleed.synthetic.config import parse_config
from trackleed.synthetic.scene import evaluate_scene, render_observation


def apparatus_config(**scene):
    values = {"image_size": [65, 65], "center_px": [18, 32],
              "lattice_scale_px": 32, "sigma_px": 4, "amplitude": .5}
    values.update(scene)
    return {"schema_version": 5, "scene": values, "condition": {},
            "schedules": [{"id": "preview", "times_ns": [0, 100000000]}],
            "export_masks": True}


def target(observation, reflection_id="h-1_k+0"):
    return next(row for row in observation.reference if row["reflection_id"] == reflection_id)


class SyntheticApparatusTests(unittest.TestCase):
    def test_outside_center_with_empty_nominal_mask_retains_its_rendered_tail(self):
        raw = apparatus_config()
        config = parse_config(raw)
        observation = render_observation(config, 0)
        state = target(observation)
        self.assertEqual((state["x_px"], state["y_px"]), (-14, 32))
        self.assertFalse(state["center_in_frame"])
        self.assertFalse(state["extinguished"])
        self.assertTrue(state["prequant_signal_present"])
        self.assertGreater(state["w1"], 0)
        self.assertLess(state["w1"], state["w0"])
        self.assertEqual(state["w1"], state["w2"])
        self.assertEqual(state["w2"], state["w3"])
        self.assertAlmostEqual(state["frame_loss_fraction"] + state["retained_fraction"], 1)
        instance = next(i for i in observation.instances if i.reflection_id == "h-1_k+0")
        self.assertFalse(instance.geometry.any())
        self.assertFalse(instance.signal.any())
        self.assertIsNone(instance.bbox_geometry)
        self.assertIsNone(instance.bbox_signal)
        self.assertEqual(observation.image[32, 0], np.float32(.1 + .5 * math.exp(-6.125)))
        self.assertEqual(observation.image[32, 2], np.float32(.1 + .5 * math.exp(-8)))
        self.assertEqual(observation.image[32, 3], np.float32(.1))
        self.assertEqual(len(evaluate_scene(config, 0)), 6)
        raw["schema_version"] = 4
        with self.assertRaisesRegex(ValueError, "support"):
            parse_config(raw)

    def test_screen_and_occluder_losses_are_disjoint_and_do_not_erase_background(self):
        raw = apparatus_config(center_px=[-6, 16], lattice_scale_px=6, sigma_px=1)
        raw["condition"]["apparatus"] = {
            "screen": {"kind": "circle", "center_px": [0, 16], "radius_px": 2},
            "occluders": [{"kind": "vertical_band", "x_range_px": [0, 0]}],
            "outside_screen_level": .03, "occluder_level": .07}
        # Exact multiplicities of the integer lattice shells in a radius-four disk.
        w0 = sum(n * math.exp(-q2 / 2) for q2, n in
                 ((0, 1), (1, 4), (2, 4), (4, 4), (5, 8), (8, 4),
                  (9, 4), (10, 8), (13, 8), (16, 4)))
        axis4 = 1 + 2 * sum(math.exp(-q2 / 2) for q2 in (1, 4, 9, 16))
        axis2 = 1 + 2 * math.exp(-.5) + 2 * math.exp(-2)
        disk2 = 1 + 4 * math.exp(-.5) + 4 * math.exp(-1) + 4 * math.exp(-2)
        expected = [w0, (w0 + axis4) / 2, (disk2 + axis2) / 2, (disk2 - axis2) / 2]
        for amplitude in (.5, 0):
            raw["scene"]["amplitude"] = amplitude
            config = parse_config(raw)
            self.assertEqual(parse_config(config.expanded()), config)
            observation = render_observation(config, 0)
            state = target(observation, "h+1_k+0")
            np.testing.assert_allclose([state[f"w{i}"] for i in range(4)], expected,
                                       rtol=0, atol=1e-12)
            losses = [state[k] for k in ("frame_loss_fraction", "screen_loss_fraction",
                                         "occluder_loss_fraction", "retained_fraction")]
            self.assertTrue(all(0 < x < 1 for x in losses))
            self.assertAlmostEqual(sum(losses), 1, places=12)
            self.assertEqual(observation.image[16, 0], np.float32(.07))
            self.assertEqual(observation.image[16, 1], np.float32(.1 + amplitude * math.exp(-.5)))
            self.assertEqual(observation.image[16, 2], np.float32(.1 + amplitude * math.exp(-2)))
            self.assertEqual(observation.image[16, 3], np.float32(.03))
            self.assertEqual(observation.image[0, 0], np.float32(.03))
            instance = next(i for i in observation.instances if i.reflection_id == "h+1_k+0")
            self.assertEqual(instance.bbox_geometry, [0, 13, 4, 20])
            self.assertEqual(instance.bbox_signal, [1, 15, 3, 18] if amplitude else None)
            self.assertEqual(state["prequant_signal_present"], bool(amplitude))

    def test_fixed_band_gives_partial_then_total_occlusion_and_return_with_same_id(self):
        raw = apparatus_config(center_px=[28, 32], lattice_scale_px=10, sigma_px=1,
                               drift_px_per_s=[-4, 0])
        raw["condition"]["apparatus"] = {"occluders": [
            {"kind": "vertical_band", "x_range_px": [26, 34]}]}
        raw["schedules"][0]["times_ns"] = [0, 2000000000, 5000000000]
        config = parse_config(raw)
        states = [target(render_observation(config, t), "h+1_k+0") for t in config.schedule.times_ns]
        self.assertEqual([r["x_px"] for r in states], [38, 30, 18])
        self.assertTrue(all(r["reflection_id"] == "h+1_k+0" and not r["extinguished"] for r in states))
        self.assertGreater(states[0]["occluder_loss_fraction"], 0)
        self.assertLess(states[0]["occluder_loss_fraction"], 1)
        self.assertEqual(states[1]["occluder_loss_fraction"], 1)
        for key in ("float64_signal_pixels", "float32_signal_pixels", "float32_changed_pixels", "encoded_changed_pixels"):
            self.assertEqual(states[1][key], 0)
        self.assertFalse(states[1]["prequant_signal_present"])
        self.assertEqual(states[2]["retained_fraction"], 1)
        self.assertTrue(states[2]["prequant_signal_present"])

    def test_circle_and_duplicate_occluders_keep_closed_boundaries_and_union_losses(self):
        raw = apparatus_config(center_px=[22, 32], lattice_scale_px=10, sigma_px=1)
        circle = {"kind": "circle", "center_px": [32, 32], "radius_px": 1}
        raw["condition"]["apparatus"] = {"occluders": [circle]}
        single = render_observation(parse_config(raw), 0)
        state = target(single, "h+1_k+0")
        self.assertAlmostEqual(state["w2"] - state["w3"], 1 + 4 * math.exp(-.5), places=12)
        self.assertEqual(single.image[32, 33], np.float32(.04))
        self.assertEqual(single.image[32, 34], np.float32(.1 + .5 * math.exp(-2)))
        raw["condition"]["apparatus"]["occluders"].append(copy.deepcopy(circle))
        duplicate = render_observation(parse_config(raw), 0)
        np.testing.assert_array_equal(single.image, duplicate.image)
        self.assertEqual(single.reference, duplicate.reference)

    def test_overlap_of_elliptical_h54_profiles_is_not_an_occlusion(self):
        raw = apparatus_config(pattern="H54", center_px=[32, 32], lattice_scale_px=10,
                               satellite_offset_px=2, shape={"sigma_u_px": 2, "sigma_v_px": 1,
                                                             "theta_rad": math.pi / 6})
        raw["scene"].pop("sigma_px")
        observation = render_observation(parse_config(raw), 0)
        self.assertEqual(len(observation.reference), 54)
        self.assertTrue(any(r["overlap_contour_ids"] for r in observation.reference))
        for state in observation.reference:
            self.assertEqual(state["retained_fraction"], 1)
            self.assertEqual(state["occluder_loss_fraction"], 0)
        raw["condition"]["apparatus"] = {"occluders": [
            {"kind": "vertical_band", "x_range_px": [42, 42]}]}
        limited = render_observation(parse_config(raw), 0)
        self.assertEqual(observation.reflections, limited.reflections)
        for before, after in zip(observation.instances, limited.instances):
            np.testing.assert_array_equal(before.geometry, after.geometry)
            self.assertFalse(after.signal[:, 42].any())
            np.testing.assert_array_equal(before.signal[:, :42], after.signal[:, :42])

    def test_zero_sample_support_fails_even_when_entire_reflection_is_outside(self):
        raw = apparatus_config(center_px=[-32.5, -32.5], lattice_scale_px=10, sigma_px=.01)
        with self.assertRaisesRegex(ValueError, "W0"):
            render_observation(parse_config(raw), 0)

    def test_invalid_apparatus_and_historical_schema_options_are_rejected(self):
        invalid = [True, {"moving": True}, {"screen": None},
                   {"screen": {"kind": "circle", "center_px": [0, 0], "radius_px": 0}},
                   {"screen": {"kind": "circle", "center_px": [0], "radius_px": 2}},
                   {"screen": {"kind": "circle", "center_px": [True, 0], "radius_px": 2}},
                   {"screen": {"kind": "vertical_band", "x_range_px": [0, 2]}},
                   {"occluders": [{"kind": "full_frame"}]}, {"occluders": "none"},
                   {"occluders": [{"kind": "vertical_band", "x_range_px": [3, 2]}]},
                   {"occluders": [{"kind": "circle", "center_px": [0, 0], "radius_px": float("inf")}]},
                   {"occluder_level": -1}, {"outside_screen_level": float("nan")}]
        for apparatus in invalid:
            with self.subTest(apparatus=apparatus):
                raw = apparatus_config()
                raw["condition"]["apparatus"] = apparatus
                with self.assertRaisesRegex(ValueError, "apparatus"):
                    parse_config(raw)
        for version in (1, 2, 3, 4):
            raw = {"schema_version": version, "scene": {}, "condition": {"apparatus": {}}}
            raw.update({"schedule": {}} if version == 1 else {"schedules": [{"times_ns": [0]}]})
            with self.assertRaisesRegex(ValueError, "apparatus"):
                parse_config(raw)

    def test_off_frame_support_still_requires_finite_reference_coordinates(self):
        raw = apparatus_config(center_px=[1.7e308, 32], lattice_scale_px=1e308, sigma_px=1)
        with self.assertRaisesRegex(ValueError, "support"):
            parse_config(raw)
