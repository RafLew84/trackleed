"""Analytic scene and rendering checks independent of image fitting."""
import math
import unittest

import numpy as np

from trackleed.synthetic.config import parse_config
from trackleed.synthetic.scene import evaluate_scene, render_observation


def small_config(**scene_values):
    scene = {"image_size": [65, 65], "center_px": [32, 32],
             "lattice_scale_px": 12, "sigma_px": 1, "amplitude": 0.5}
    scene.update(scene_values)
    return parse_config({"schema_version": 1, "scene": scene,
                         "condition": {"background": 0.1}, "schedule": {}})


class SyntheticSceneTests(unittest.TestCase):
    def test_static_h6_has_analytic_centers_and_identical_observations(self):
        config = small_config()
        reflections = evaluate_scene(config, 0)
        self.assertEqual(reflections, evaluate_scene(config, 123456789))
        actual = sorted((s.x_px, s.y_px) for s in reflections)
        expected = [(20, 32), (26, 21.607695154586736), (26, 42.392304845413264),
                    (38, 21.607695154586736), (38, 42.392304845413264), (44, 32)]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        self.assertEqual(len({s.reflection_id for s in reflections}), 6)
        observation = render_observation(config, 0)
        np.testing.assert_array_equal(observation.image, render_observation(config, 9).image)
        self.assertEqual(observation.image.dtype, np.float32)
        self.assertEqual(observation.png.dtype, np.uint8)
        self.assertEqual(observation.image[32, 44], np.float32(0.6))
        self.assertEqual(observation.image[32, 48],
                         np.float32(0.1 + 0.5 * math.exp(-8)))
        self.assertEqual(observation.image[32, 49], np.float32(0.1))
        self.assertEqual(observation.png[32, 44], 153)

    def test_subpixel_peak_is_a_parameter_not_the_maximum_pixel(self):
        observation = render_observation(small_config(center_px=[32.5, 32.5]), 0)
        self.assertAlmostEqual(float(observation.image[32, 44]),
                               0.1 + 0.5 * math.exp(-0.25), places=7)
        self.assertLess(float(observation.image.max()), 0.6)


    def test_zero_weak_and_saturated_signals_keep_distinct_reference_states(self):
        for amplitude, background, generated, encoded in (
            (0, 0.1, False, False), (1e-50, 0.1, True, False),
            (1e-9, 0.1, True, False), (0.5, 2, True, False),
            (0.5, 0.1, True, True),
        ):
            with self.subTest(amplitude=amplitude, background=background):
                raw = small_config(amplitude=amplitude).expanded()
                raw["condition"]["background"] = background
                observation = render_observation(parse_config(raw), 0)
                self.assertEqual(len(observation.reference), 6)
                for state in observation.reference:
                    self.assertEqual(state["extinguished"], amplitude == 0)
                    self.assertEqual(state["prequant_signal_present"], generated)
                    self.assertEqual(state["encoded_changed_pixels"] > 0, encoded)
                    self.assertEqual(state["retained_fraction"], 1)
                    self.assertEqual(state["frame_loss_fraction"], 0)
                    self.assertEqual(state["w0"], state["w3"])
                if amplitude == 1e-50:
                    self.assertEqual(observation.reference[0]["float32_signal_pixels"], 0)
                if background == 2:
                    self.assertEqual(observation.photometry["above_upper_pixels"], 65 * 65)

    def test_round_half_up_clipping_and_gaussian_sampling_failure(self):
        from trackleed.synthetic.scene import encode_png
        values = np.array([-1, 0, 0.49/255, 0.5/255, 0.51/255, 254.5/255, 1, 2])
        np.testing.assert_array_equal(encode_png(values, 0, 1), [0, 0, 0, 1, 1, 255, 255, 255])
        with self.assertRaisesRegex(ValueError, "W0"):
            render_observation(small_config(center_px=[32.5, 32.5], sigma_px=0.01), 0)
