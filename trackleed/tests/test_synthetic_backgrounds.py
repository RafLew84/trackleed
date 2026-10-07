"""Static backgrounds and camera-arm occlusion through public rendering."""
import copy
import unittest

import numpy as np

from trackleed.synthetic.config import parse_config
from trackleed.synthetic.scene import evaluate_scene, render_observation


def background_config(**scene):
    values = {"image_size": [65, 65], "center_px": [28, 32],
              "lattice_scale_px": 10, "sigma_px": 1, "amplitude": .5}
    values.update(scene)
    return {"schema_version": 6, "scene": values, "condition": {},
            "schedules": [{"id": "preview", "times_ns": [0, 2000000000, 5000000000]}],
            "export_masks": True}


class SyntheticBackgroundTests(unittest.TestCase):
    def test_camera_arm_occludes_and_releases_same_reference_without_erasing_background(self):
        raw = background_config(drift_px_per_s=[-4, 0])
        vertices = [[8, 0], [20, 0], [52, 64], [40, 64]]
        raw["condition"]["apparatus"] = {"occluders": [
            {"kind": "polygon", "vertices_px": vertices},
            {"kind": "circle", "center_px": [48, 60], "radius_px": 4}],
            "occluder_level": .07}
        config = parse_config(raw)
        observations = [render_observation(config, t) for t in config.schedule.times_ns]
        states = [next(r for r in obs.reference if r["reflection_id"] == "h+1_k+0")
                  for obs in observations]
        self.assertEqual([r["x_px"] for r in states], [38, 30, 18])
        self.assertTrue(all(r["y_px"] == 32 and r["amplitude"] == .5 for r in states))
        self.assertGreater(states[0]["occluder_loss_fraction"], 0)
        self.assertLess(states[0]["occluder_loss_fraction"], 1)
        self.assertEqual(states[1]["occluder_loss_fraction"], 1)
        self.assertEqual(states[1]["encoded_changed_pixels"], 0)
        self.assertFalse(states[1]["prequant_signal_present"])
        self.assertEqual(states[2]["retained_fraction"], 1)
        for obs in observations:
            self.assertEqual(obs.image[32, 30], np.float32(.07))
            self.assertEqual(obs.image[60, 48], np.float32(.07))
            self.assertEqual(obs.image[0, 8], np.float32(.07))
        instance = next(i for i in observations[1].instances if i.reflection_id == "h+1_k+0")
        self.assertTrue(instance.geometry.any())
        self.assertFalse(instance.signal.any())
        self.assertIsNone(instance.bbox_signal)
        self.assertEqual(parse_config(config.expanded()), config)
        raw["condition"]["apparatus"]["occluders"][0]["vertices_px"] = vertices[::-1]
        np.testing.assert_array_equal(render_observation(parse_config(raw), 0).image, observations[0].image)
        raw["schema_version"] = 5
        with self.assertRaisesRegex(ValueError, "polygon.*6|6.*polygon"):
            parse_config(raw)

    def test_static_gradient_glow_and_seeded_texture_do_not_change_latent_scene(self):
        raw = background_config()
        raw["condition"]["background"] = .3
        baseline = parse_config(raw)
        raw["scene"]["amplitude"] = 0
        raw["condition"]["background_model"] = {
            "kind": "procedural", "gradient": [.2, -.1],
            "glow": {"center_px": [32, 32], "sigma_px": [16, 16], "amplitude": .2},
            "texture_amplitude": 0, "texture_seed": 123}
        config = parse_config(raw)
        observation = render_observation(config, 0)
        # Worked background values: opposite corners have glow .2*exp(-4).
        np.testing.assert_allclose(observation.image[[0, 0, 64, 32], [0, 64, 0, 32]],
                                   [.25366312777774685, .45366312777774686,
                                    .15366312777774684, .5], rtol=0, atol=3e-8)
        raw["scene"]["amplitude"] = .5
        raw["condition"]["background_model"]["texture_amplitude"] = .03
        textured = parse_config(raw)
        first = render_observation(textured, 0)
        for t in textured.schedule.times_ns:
            np.testing.assert_array_equal(first.image, render_observation(textured, t).image)
            self.assertEqual(evaluate_scene(textured, t), evaluate_scene(baseline, t))
        raw["condition"]["background_model"]["texture_seed"] = 124
        different = render_observation(parse_config(raw), 0)
        self.assertFalse(np.array_equal(first.image, different.image))
        self.assertEqual(first.reflections, different.reflections)
        self.assertEqual(parse_config(textured.expanded()), textured)

    def test_invalid_backgrounds_and_nonconvex_arms_fail_configuration(self):
        bad = [
            {"kind": "procedural", "gradient": [1, float("nan")]},
            {"kind": "procedural", "texture_seed": True},
            {"kind": "procedural", "texture_amplitude": -.1},
            {"kind": "procedural", "glow": {"center_px": [0, 0], "sigma_px": [0, 1], "amplitude": 1}},
            {"kind": "procedural", "extra": 1},
            {"kind": "import", "path": "x.png", "gain": 0},
            {"kind": "import", "path": "x.png", "gain": 1, "transform": {"scale": [2, 1]}},
            {"kind": "import", "path": "x.png", "gain": 1, "transform": {"crop_xywh": [0, 0, 0, 2]}},
        ]
        for model in bad:
            with self.subTest(model=model), self.assertRaises(ValueError):
                raw = background_config()
                raw["condition"]["background_model"] = model
                parse_config(raw)
        for points in ([[0, 0], [4, 4], [0, 4], [4, 0]],
                       [[0, 0], [4, 0], [1, 1], [0, 4]],
                       [[0, 0], [0, 0], [1, 1]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                raw = background_config()
                raw["condition"]["apparatus"] = {"occluders": [{"kind": "polygon", "vertices_px": points}]}
                parse_config(raw)
        raw = background_config()
        raw["schema_version"] = 5
        raw["condition"]["background_model"] = {"kind": "procedural"}
        with self.assertRaisesRegex(ValueError, "schema_version 6"):
            parse_config(raw)
