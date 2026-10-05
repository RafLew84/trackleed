"""Recorded schedules sample the same scene and preserve their own deltas."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.package import generate_benchmark, verify_package
from trackleed.tests.test_synthetic_motion import scan_config


def read_rows(root, name):
    with (root / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class SyntheticSamplingTests(unittest.TestCase):
    def test_three_recorded_schedules_share_pixels_but_use_their_own_predecessors(self):
        raw = scan_config()
        raw["schedules"] = [
            {"id": f"every-{step}ms", "times_ns": list(range(0, 2000000001, step * 1000000))}
            for step in (20, 50, 100)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.json"
            source.write_text(json.dumps(raw))
            target = root / "result"
            report = generate_benchmark(source, target)
            self.assertEqual(report, verify_package(target))
            self.assertEqual((report.frames, report.reference_records), (163, 978))
            rows = read_rows(target, "frames.csv")
            references = read_rows(target, "reference.csv")
            grouped = {step: [row for row in rows if row["schedule_id"] == f"every-{step}ms"]
                       for step in (20, 50, 100)}
            self.assertEqual([len(grouped[step]) for step in grouped], [101, 41, 21])
            for index in range(21):
                same = [grouped[step][index * 100 // step] for step in grouped]
                self.assertEqual(len({row["observation_id"] for row in same}), 1)
                self.assertEqual(len({row["tiff_pixel_sha256"] for row in same}), 1)
                self.assertEqual(len({row["png_pixel_sha256"] for row in same}), 1)
                arrays = [tifffile.imread(target / row["tiff_path"]) for row in same]
                for array in arrays[1:]:
                    np.testing.assert_array_equal(array, arrays[0])
                pngs = []
                for row in same:
                    with Image.open(target / row["png_path"]) as image:
                        pngs.append(np.array(image))
                for png in pngs[1:]:
                    np.testing.assert_array_equal(png, pngs[0])
            for step in grouped:
                frames = grouped[step]
                self.assertEqual(frames[0]["dt_ns"], "")
                self.assertEqual(frames[0]["de_ev"], "")
                self.assertEqual(frames[0]["previous_observation_id"], "")
                self.assertEqual(frames[1]["dt_ns"], str(step * 1000000))
                self.assertAlmostEqual(float(frames[1]["de_ev"]), step * 0.06)
                ref = [r for r in references if r["schedule_id"] == f"every-{step}ms"
                       and r["reflection_id"] == "h+1_k+0"]
                self.assertEqual(ref[0]["dx_px"], "")
                for previous, current in zip(ref, ref[1:]):
                    self.assertAlmostEqual(float(current["dx_px"]),
                                           float(current["x_px"]) - float(previous["x_px"]))
                    self.assertEqual(current["previous_observation_id"], previous["observation_id"])
            self.assertNotEqual(grouped[20][5]["previous_observation_id"],
                                grouped[100][1]["previous_observation_id"])
            self.assertEqual(json.loads((target / "manifest.json").read_text())["plan"]["sequences"], 3)


    def test_energy_list_keeps_order_distinct_observations_and_no_invented_time(self):
        raw = scan_config()
        scan = raw["scene"]["energy_scan"]
        scan.update(model="energy_list", energies_ev=[200, 120, 120, 80], times_ns=None)
        raw["schedules"] = [{"id": "energy-order", "observation_indices": [0, 1, 2, 3]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "list.json"
            source.write_text(json.dumps(raw))
            target = root / "result"
            generate_benchmark(source, target)
            frames = read_rows(target, "frames.csv")
            self.assertEqual([float(row["energy_ev"]) for row in frames], [200, 120, 120, 80])
            self.assertEqual(len({row["observation_id"] for row in frames}), 4)
            self.assertTrue(all(row["time_ns"] == row["dt_ns"] == "" for row in frames))
            self.assertEqual([row["de_ev"] for row in frames], ["", "-80", "0", "-40"])
            references = read_rows(target, "reference.csv")
            pair = [r for r in references if r["reflection_id"] == "h+1_k+0"]
            self.assertEqual(pair[1]["x_px"], pair[2]["x_px"])
            self.assertEqual(pair[2]["dx_px"], "0.0")
            self.assertEqual(frames[1]["png_pixel_sha256"], frames[2]["png_pixel_sha256"])
            # A new directory does not become a new physical realization.
            second = root / "repeat"
            generate_benchmark(source, second)
            for name in ("frames.csv", "reference.csv"):
                self.assertEqual((target / name).read_bytes(), (second / name).read_bytes())


    def test_descending_scan_preserves_nanosecond_keys_above_float_integer_precision(self):
        raw = scan_config()
        start = 2**53
        raw["scene"]["energy_scan"].update(energies_ev=[200, 80], times_ns=[start, start + 2])
        raw["schedules"] = [{"id": "reverse", "times_ns": [start, start + 1, start + 2]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reverse.json"
            source.write_text(json.dumps(raw))
            target = root / "result"
            generate_benchmark(source, target)
            rows = read_rows(target, "frames.csv")
            self.assertEqual([int(row["time_ns"]) for row in rows], [start, start + 1, start + 2])
            self.assertEqual([row["dt_ns"] for row in rows], ["", "1", "1"])
            self.assertEqual([float(row["energy_ev"]) for row in rows], [200, 140, 80])
            self.assertEqual([float(row["de_ev"]) for row in rows[1:]], [-60, -60])
            self.assertEqual(len({row["observation_id"] for row in rows}), 3)
            reference = [r for r in read_rows(target, "reference.csv") if r["reflection_id"] == "h+1_k+0"]
            self.assertTrue(all(float(r["dx_px"]) > 0 for r in reference[1:]))

    def test_failure_in_second_schedule_keeps_whole_package_unpublished(self):
        from unittest.mock import patch
        raw = scan_config()
        raw["schedules"] = [{"id": "first", "times_ns": [0]},
                            {"id": "second", "times_ns": [100000000]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.json"
            source.write_text(json.dumps(raw))
            target = root / "result"
            original_write = tifffile.imwrite

            def interrupted_write(path, *args, **kwargs):
                if "second" in Path(path).parts:
                    raise OSError("second schedule disk failure")
                return original_write(path, *args, **kwargs)

            with patch("tifffile.imwrite", side_effect=interrupted_write):
                with self.assertRaisesRegex(OSError, "second schedule"):
                    generate_benchmark(source, target)
            self.assertFalse(target.exists())
            stages = list(root.glob(".trackleed-incomplete-*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "sequences/first/images/frame_000000.png").is_file())
            with self.assertRaisesRegex(ValueError, "Staging"):
                verify_package(stages[0])

    def test_cli_roundtrip_and_legacy_guard_cover_nested_schedule_images(self):
        import subprocess
        import sys
        from trackleed.io.file_parser import parse_directory
        from trackleed.io.image_reader import load_tiff
        repository = Path(__file__).resolve().parents[2]
        cli = repository / "scripts/generate_tracking_benchmark.py"
        raw = scan_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.json"
            source.write_text(json.dumps(raw))
            target = root / "result"
            result = subprocess.run([sys.executable, str(cli), "--config", str(source),
                                     "--output", str(target)], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["frames"], 3)
            verified = subprocess.run([sys.executable, str(cli), "--verify", str(target)],
                                      cwd=root, capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            images = target / "sequences/sampled/images"
            parsed = parse_directory(images)
            self.assertEqual([frame.id for frame in parsed], [0, 1, 2])
            for frame in parsed:
                self.assertEqual(load_tiff(frame.filepath).shape, (65, 65))
            destination = root / "legacy"
            legacy = repository / "scripts/export_segmentation_to_png.py"
            for folder in (target, images.parent, images):
                refused = subprocess.run([sys.executable, str(legacy), "--source", str(folder),
                                          "--dest", str(destination)], cwd=root,
                                         capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("benchmark", refused.stderr.lower())
                self.assertIn(str(images), refused.stderr)
                self.assertFalse(destination.exists())

    def test_v1_integer_energy_keeps_original_delta_representation(self):
        from trackleed.tests.test_synthetic_package import write_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root, energy_ev=100)
            target = root / "v1"
            generate_benchmark(source, target)
            rows = read_rows(target, "frames.csv")
            self.assertEqual(rows[0]["energy_ev"], "100")
            self.assertEqual([row["de_ev"] for row in rows], ["", "0.0", "0.0"])
