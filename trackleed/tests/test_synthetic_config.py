"""User configuration and ideal-scene contracts for the H6 benchmark."""
import json
from pathlib import Path
import tempfile
import unittest

from trackleed.synthetic.config import load_config


class SyntheticConfigTests(unittest.TestCase):
    def test_versioned_json_expands_one_static_h6_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "scene": {},
                "condition": {},
                "schedule": {"times_ns": [0, 50000000, 100000000]},
            }))
            config = load_config(path)
        self.assertEqual(config.scene.pattern, "H6")
        self.assertEqual(config.scene.center_px, (499.5, 499.5))
        self.assertEqual(config.scene.image_size, (1000, 1000))
        self.assertEqual(config.schedule.times_ns, (0, 50000000, 100000000))
        self.assertEqual(config.plan(), {"sequences": 1, "frames": 3, "reflections": 6,
                                         "reference_records": 18})


if __name__ == "__main__":
    unittest.main()


class InvalidConfigurationTests(unittest.TestCase):
    def test_invalid_or_future_options_are_rejected_with_field_errors(self):
        from trackleed.synthetic.config import parse_config
        cases = [
            ({"schema_version": True}, "schema_version"),
            ({"schema_version": 3}, "schema_version"),
            ({"noise": 0.01}, "noise"),
            ({"scene": {"pattern": "H18"}}, "pattern"),
            ({"scene": {"sigma_px": 0}}, "sigma_px"),
            ({"scene": {"sigma_px": float("nan")}}, "sigma_px"),
            ({"scene": {"amplitude": -1}}, "amplitude"),
            ({"scene": {"center_px": [1, 1]}}, "support"),
            ({"scene": {"image_size": [True, 1000]}}, "image_size"),
            ({"condition": {"png_upper": 0}}, "png_upper"),
            ({"schedule": {"times_ns": [0, 0]}}, "times_ns"),
            ({"schedule": {"times_ns": [0, 0.5]}}, "times_ns"),
            ({"schedule": {"times_ns": [0, True]}}, "times_ns"),
            ({"realization_id": "../other"}, "realization_id"),
        ]
        for update, field in cases:
            with self.subTest(update=update):
                raw = {"schema_version": 1, "scene": {}, "condition": {}, "schedule": {}}
                raw.update(update)
                with self.assertRaisesRegex(ValueError, field):
                    parse_config(raw)

    def test_json_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"schema_version":1,"schema_version":1}',
                         '{"scene":{"amplitude":NaN}}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)
