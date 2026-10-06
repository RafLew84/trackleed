"""Ideal lattice geometry observed through configuration and scene interfaces."""
import math
import unittest

import numpy as np

from trackleed.synthetic.config import parse_config
from trackleed.synthetic.scene import evaluate_scene, render_observation


def pattern_config(pattern="H54", **scene_values):
    scene = {"pattern": pattern, "image_size": [129, 129], "center_px": [64, 64],
             "lattice_scale_px": 20, "sigma_px": 1, "satellite_offset_px": 9}
    scene.update(scene_values)
    return {"schema_version": 3, "scene": scene, "condition": {},
            "schedules": [{"id": "static", "times_ns": [0, 50000000]}]}


class SyntheticPatternTests(unittest.TestCase):
    def test_three_patterns_have_exact_lattice_sites_and_stable_satellite_parents(self):
        # Hexagonal shells, specified as an independent literal index inventory.
        first = {(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)}
        second = {(-2, 0), (-2, 1), (-2, 2), (-1, -1), (-1, 2), (0, -2),
                  (0, 2), (1, -2), (1, 1), (2, -2), (2, -1), (2, 0)}
        for pattern, count in (("H6", 6), ("H18", 18), ("H54", 54)):
            for offset in (9, 18):
                config = parse_config(pattern_config(pattern, satellite_offset_px=offset))
                states = evaluate_scene(config, 0)
                self.assertEqual(states, evaluate_scene(config, 123456))
                self.assertEqual(len(states), count)
                self.assertEqual(config.plan()["reference_records"], count * 2)
                by_id = {state.reflection_id: state for state in states}
                self.assertEqual(len(by_id), count)
                primary = [state for state in states if state.parent_id is None]
                self.assertEqual({(s.h, s.k) for s in primary},
                                 first if pattern == "H6" else first | second)
                right = by_id["h+2_k+0"] if pattern != "H6" else by_id["h+1_k+0"]
                self.assertEqual((right.x_px, right.y_px), (104 if count > 6 else 84, 64))
                for parent in primary:
                    children = [s for s in states if s.parent_id == parent.reflection_id]
                    self.assertEqual(len(children), 6 if count == 54 and (parent.h, parent.k) in first else 0)
                    if children:
                        expected = np.array([[offset, 0], [offset/2, offset*math.sqrt(3)/2],
                                             [-offset/2, offset*math.sqrt(3)/2], [-offset, 0],
                                             [-offset/2, -offset*math.sqrt(3)/2], [offset/2, -offset*math.sqrt(3)/2]])
                        actual = [(s.x_px-parent.x_px, s.y_px-parent.y_px) for s in children]
                        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-12)
                        self.assertTrue(all(s.role == "satellite" and s.amplitude == 0.15 for s in children))
                self.assertEqual(parse_config(config.expanded()), config)

    def test_rotated_profile_has_declared_axes_and_independent_analytic_pixels(self):
        raw = pattern_config("H6")
        raw["scene"].pop("sigma_px")
        raw["scene"]["shape"] = {"sigma_u_px": 2, "sigma_v_px": 1, "theta_rad": math.pi/6}
        config = parse_config(raw)
        observation = render_observation(config, 0)
        state = next(r for r in observation.reference if r["reflection_id"] == "h+1_k+0")
        self.assertEqual((state["sigma_u_px"], state["sigma_v_px"], state["theta_rad"]), (2, 1, math.pi/6))
        # At offset (2,0), u=sqrt(3), v=-1; q^2=7/4.
        self.assertAlmostEqual(float(observation.image[64, 86]), 0.1 + 0.5*math.exp(-7/8), places=7)
        # At offset (0,2), u=1, v=sqrt(3); q^2=13/4.
        self.assertAlmostEqual(float(observation.image[66, 84]), 0.1 + 0.5*math.exp(-13/8), places=7)
        self.assertEqual(parse_config(config.expanded()), config)

    def test_full_overlap_retains_all_six_satellite_identities_and_adds_signals(self):
        raw = pattern_config(satellite_offset_px=20)
        raw["export_masks"] = True
        observation = render_observation(parse_config(raw), 0)
        at_center = [r for r in observation.reference
                     if math.hypot(r["x_px"]-64, r["y_px"]-64) < 1e-9]
        self.assertEqual(len(at_center), 6)
        self.assertEqual(len({r["reflection_id"] for r in at_center}), 6)
        self.assertAlmostEqual(float(observation.image[64, 64]), 1.0, places=7)
        by_id = {instance.reflection_id: instance for instance in observation.instances}
        for state in at_center:
            self.assertEqual(by_id[state["reflection_id"]].signal[64, 64], 255)

    def test_shape_contour_support_and_extinction_remain_distinct(self):
        for amplitude in (0, 1e-50, 0.5):
            raw = pattern_config("H6", amplitude=amplitude)
            raw["scene"].pop("sigma_px")
            raw["scene"]["shape"] = {"sigma_u_px": 2, "sigma_v_px": 1, "theta_rad": math.pi/2}
            raw["export_masks"] = True
            observation = render_observation(parse_config(raw), 0)
            masks = next(m for m in observation.instances if m.reflection_id == "h+1_k+0")
            self.assertEqual(masks.bbox_geometry, [81, 58, 88, 71])
            self.assertEqual(masks.geometry[70, 84], 255)  # exactly 3 sigma along u
            self.assertEqual(masks.geometry[71, 84], 0)
            if amplitude == 0:
                self.assertIsNone(masks.bbox_signal)
                self.assertFalse(masks.signal.any())
            else:
                np.testing.assert_array_equal(masks.geometry, masks.signal)
            if amplitude == 0.5:
                self.assertEqual(observation.image[72, 84], np.float32(0.1 + 0.5*math.exp(-8)))
                self.assertEqual(observation.image[73, 84], np.float32(0.1))
            else:
                self.assertTrue(all(r["encoded_changed_pixels"] == 0 for r in observation.reference))

    def test_empty_nominal_raster_keeps_ellipse_parameters_and_rendered_tail(self):
        raw = pattern_config("H6", center_px=[64.5, 64.5], sigma_px=0.2)
        raw["export_masks"] = True
        observation = render_observation(parse_config(raw), 0)
        masks = next(m for m in observation.instances if m.reflection_id == "h+1_k+0")
        state = next(r for r in observation.reference if r["reflection_id"] == masks.reflection_id)
        self.assertIsNone(masks.bbox_geometry)
        self.assertIsNone(masks.bbox_signal)
        self.assertEqual((state["x_px"], state["y_px"], state["sigma_u_px"]), (84.5, 64.5, 0.2))
        self.assertTrue(state["prequant_signal_present"])
        self.assertGreater(state["float32_changed_pixels"], 0)

    def test_elliptical_overlap_uses_axes_instead_of_bounding_circles(self):
        raw = pattern_config(satellite_offset_px=9)
        raw["scene"].pop("sigma_px")
        raw["scene"]["shape"] = {"sigma_u_px": 2, "sigma_v_px": 0.5, "theta_rad": 0}
        observation = render_observation(parse_config(raw), 0)
        parent = next(r for r in observation.reference if r["reflection_id"] == "h+1_k+0")
        self.assertIn("h+1_k+0_s0", parent["overlap_contour_ids"])
        self.assertNotIn("h+1_k+0_s1", parent["overlap_contour_ids"])

    def test_v3_rejects_invalid_shapes_versions_and_out_of_frame_patterns(self):
        from copy import deepcopy
        cases = []
        for field, value in (("satellite_offset_px", 0), ("satellite_offset_px", True),
                             ("satellite_amplitude", -1), ("pattern", "H99"),
                             ("satellite_offset_px", 100), ("lattice_scale_px", 40)):
            raw = pattern_config(**{field: value})
            cases.append(raw)
        for shape in ({"sigma_u_px": 0}, {"sigma_v_px": float("inf")},
                      {"theta_rad": True}, {"width": 2}):
            raw = pattern_config()
            raw["scene"].pop("sigma_px")
            raw["scene"]["shape"] = shape
            cases.append(raw)
        ambiguous = pattern_config()
        ambiguous["scene"]["shape"] = {}
        cases.append(ambiguous)
        for field, value in (("schema_version", 2), ("export_masks", 1)):
            raw = pattern_config()
            raw[field] = value
            cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_config(deepcopy(raw))

    def test_scan_domain_and_motion_cover_outer_beams_and_satellites(self):
        raw = pattern_config(satellite_offset_px=9)
        raw["scene"]["energy_scan"] = {
            "model": "linear_time", "reference_energy_ev": 100, "energies_ev": [100, 400],
            "times_ns": [0, 1000000000], "valid_energy_range_ev": [100, 400],
            "projection_radius_px": 50, "domain_justification": "Declared ideal geometry."}
        config = parse_config(raw)
        final = {r.reflection_id: r for r in evaluate_scene(config, 1000000000)}
        self.assertEqual((final["h+2_k+0"].x_px, final["h+2_k+0"].y_px), (84, 64))
        self.assertEqual((final["h+1_k+0_s0"].x_px, final["h+1_k+0_s0"].y_px), (78.5, 64))
        raw["scene"]["energy_scan"]["projection_radius_px"] = 30
        with self.assertRaisesRegex(ValueError, "propagation"):
            parse_config(raw)
