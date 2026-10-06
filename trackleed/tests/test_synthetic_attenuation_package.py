"""Recorded attenuation packages preserve signal, identity and event coverage."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.package import generate_benchmark, verify_package
from trackleed.tests.test_synthetic_attenuation import attenuation_config


def write_source(root, raw):
    source = root / "config.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    return source


def rows(root, name):
    with (root / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class AttenuationPackageTests(unittest.TestCase):
    def test_recorded_extinction_returns_with_same_id_geometry_and_fixed_photometry(self):
        raw = attenuation_config()
        raw["export_masks"] = True
        raw["schedules"][0]["times_ns"] = [0, 700000000, 1100000000]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "package"
            report = generate_benchmark(write_source(root, raw), target)
            self.assertEqual(report, verify_package(target))
            self.assertEqual((report.frames, report.reference_records), (3, 54))
            references = rows(target, "reference.csv")
            states = [r for r in references if r["reflection_id"] == "h+1_k+0"]
            self.assertEqual([r["extinguished"] for r in states], ["false", "true", "false"])
            self.assertEqual([r["dx_px"] for r in states], ["", "0.0", "0.0"])
            self.assertEqual([r["x_px"] for r in states], ["42.0"] * 3)
            geometry = []
            for i, state in enumerate(states):
                with Image.open(target / state["geometry_mask_path"]) as image:
                    geometry.append(np.array(image))
                with Image.open(target / state["signal_mask_path"]) as image:
                    self.assertEqual(bool(np.array(image).any()), i != 1)
                self.assertEqual(state["bbox_signal"] == "", i == 1)
            for image in geometry:
                np.testing.assert_array_equal(image, geometry[0])
                self.assertTrue(image.any())
            for frame, intensity, code in zip(rows(target, "frames.csv"), (.6, .1, .6), (153, 26, 153)):
                self.assertEqual(tifffile.imread(target / frame["tiff_path"])[32, 42], np.float32(intensity))
                with Image.open(target / frame["png_path"]) as image:
                    self.assertEqual(image.getpixel((42, 32)), code)

    def test_report_counts_actual_minimum_samples_for_each_schedule_and_keeps_shared_pixels(self):
        raw = attenuation_config()
        cases = [("long", "h+1_k+0", [700, 700, 1100, 1100], 0, [20, 8, 4]),
                 ("missed", "h+0_k+1", [730, 730, 770, 770], 0, [2, 1, 0]),
                 ("captured", "h-1_k+0", [790, 790, 830, 830], 0, [2, 1, 1]),
                 ("ramp", "h+0_k-1", [500, 700, 1100, 1300], 0, [21, 9, 5]),
                 ("weak", "h+1_k-1", [500, 700, 1100, 1300], .05, [21, 9, 5])]
        raw["scene"]["attenuations"] = [
            {"id": key, "reflection_id": target, "axis": "time_ns",
             "knots": [t * 1000000 for t in knots], "minimum_fraction": minimum}
            for key, target, knots, minimum, _ in cases]
        raw["schedules"] = [{"id": f"every-{dt}", "times_ns": list(range(0, 1400000001, dt * 1000000))}
                            for dt in (20, 50, 100)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "package"
            generate_benchmark(write_source(root, raw), target)
            report = json.loads((target / "coverage.json").read_text())
            records = {(r["event_id"], r["schedule_id"]): r for r in report["events"]}
            for key, _, _, minimum, expected in cases:
                for dt, count in zip((20, 50, 100), expected):
                    row = records[key, f"every-{dt}"]
                    self.assertEqual(row["minimum_sample_count"], count)
                    self.assertEqual(row["zero_signal_sample_count"], 0 if minimum else count)
                    self.assertEqual(row["captured_zero_gap"], bool(count) and minimum == 0)
                    self.assertEqual(row["sampling_status"], "captured" if count else "missed")
                    self.assertEqual(row["context_status"], "bracketed")
                    self.assertFalse(row["truncated_before"] or row["truncated_after"])
            self.assertEqual(records["missed", "every-100"]["before_observation_id"], "t700000000")
            self.assertEqual(records["missed", "every-100"]["after_observation_id"], "t800000000")
            self.assertEqual(records["ramp", "every-100"]["after_observation_id"], "t1300000000")
            shared = {}
            for frame in rows(target, "frames.csv"):
                key = frame["observation_id"]
                hashes = frame["tiff_pixel_sha256"], frame["png_pixel_sha256"]
                self.assertEqual(hashes, shared.setdefault(key, hashes))

    def test_energy_report_preserves_repeated_observations_and_separate_nonmonotonic_runs(self):
        raw = attenuation_config(axis="energy_ev", knots=[110, 110, 130, 130], minimum_fraction=.05)
        raw["scene"]["energy_scan"] = {
            "model": "energy_list", "reference_energy_ev": 100, "energies_ev": [200, 120, 120, 80, 120, 200],
            "times_ns": None, "valid_energy_range_ev": [80, 200], "projection_radius_px": 100,
            "domain_justification": "Small ideal scene for ordered energy observations."}
        raw["schedules"] = [{"id": "repeat", "observation_indices": [0, 1, 2, 3, 4, 5]},
                            {"id": "descending", "observation_indices": [0, 1, 2, 3]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "package"
            generate_benchmark(write_source(root, raw), target)
            records = json.loads((target / "coverage.json").read_text())["events"]
            full, descending = records
            self.assertEqual(full["minimum_observation_ids"], ["e1", "e2", "e4"])
            self.assertEqual(full["context_status"], "nonmonotonic")
            self.assertIsNone(full["before_observation_id"])
            self.assertIsNone(full["truncated_after"])
            runs = full["sampled_runs"]
            self.assertEqual([r["observation_ids"] for r in runs], [["e1", "e2"], ["e4"]])
            self.assertEqual([(r["before_observation_id"], r["after_observation_id"]) for r in runs],
                             [("e0", "e3"), ("e3", "e5")])
            self.assertEqual(descending["context_status"], "bracketed")
            self.assertEqual((descending["before_observation_id"], descending["after_observation_id"]),
                             ("e0", "e3"))
            self.assertEqual(full["zero_signal_sample_count"], 0)
            frames = rows(target, "frames.csv")
            self.assertTrue(all(r["time_ns"] == r["dt_ns"] == "" for r in frames))
            same = [r for r in frames if r["observation_id"] in ("e1", "e2", "e4")]
            self.assertEqual(len({r["png_pixel_sha256"] for r in same}), 1)
            targets = [r for r in rows(target, "reference.csv") if r["reflection_id"] == "h+1_k+0"]
            self.assertEqual([float(r["amplitude"]) for r in targets[:6]], [.5, .025, .025, .5, .025, .5])

    def test_truncated_and_ramp_only_observations_do_not_claim_complete_gap_coverage(self):
        cases = [([700, 800, 1100], [700, 700, 1100, 1100], "captured", "truncated", True, False),
                 ([0, 700, 900], [700, 700, 1100, 1100], "captured", "truncated", False, True),
                 ([700, 900], [700, 700, 1100, 1100], "captured", "truncated", True, True),
                 ([0, 600, 1200, 1400], [500, 700, 1100, 1300], "missed", "bracketed", False, False),
                 ([0, 100], [700, 700, 1100, 1100], "missed", "outside_observations", False, True)]
        for times, knots, status, context, before, after in cases:
            with self.subTest(times=times), tempfile.TemporaryDirectory() as directory:
                raw = attenuation_config(knots=[v * 1000000 for v in knots])
                raw["schedules"][0]["times_ns"] = [v * 1000000 for v in times]
                root = Path(directory)
                generate_benchmark(write_source(root, raw), root / "result")
                row = json.loads((root / "result/coverage.json").read_text())["events"][0]
                self.assertEqual((row["sampling_status"], row["context_status"]), (status, context))
                self.assertEqual((row["truncated_before"], row["truncated_after"]), (before, after))
                self.assertEqual(row["captured_zero_gap"], status == "captured")
                if times == [0, 600, 1200, 1400]:
                    self.assertTrue(row["attenuation_observed"])
                    self.assertEqual(row["minimum_sample_count"], 0)

    def test_semantic_report_corruption_is_rejected_even_with_a_matching_hash(self):
        from hashlib import sha256
        raw = attenuation_config()
        raw["schedules"][0]["times_ns"] = [0, 700000000, 1100000000]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result"
            generate_benchmark(write_source(root, raw), target)
            path = target / "coverage.json"
            report = json.loads(path.read_text())
            self.assertEqual(report["events"][0]["minimum_sample_count"], 1)
            report["events"][0]["minimum_sample_count"] = True
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_package(target)
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["coverage.json"]["sha256"] = sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "coverage disagrees"):
                verify_package(target)

    def test_report_write_failure_leaves_no_published_package(self):
        from unittest.mock import patch
        original_write = Path.write_text

        def fail_report(path, *args, **kwargs):
            if path.name == "coverage.json":
                raise OSError("coverage disk failure")
            return original_write(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root, attenuation_config())
            with patch.object(Path, "write_text", fail_report):
                with self.assertRaisesRegex(OSError, "coverage disk failure"):
                    generate_benchmark(source, root / "result")
            self.assertFalse((root / "result").exists())
            stages = list(root.glob(".trackleed-incomplete-*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "failure.json").exists())
            with self.assertRaisesRegex(ValueError, "Staging"):
                verify_package(stages[0])

    def test_v4_cli_and_real_trackleed_reader_work_without_gui_or_model_imports(self):
        import subprocess
        import sys
        from trackleed.io.file_parser import parse_directory
        from trackleed.io.image_reader import load_tiff
        repository = Path(__file__).resolve().parents[2]
        program = ("import sys, runpy; sys.modules.update({k: None for k in ('PyQt6', 'torch', 'ultralytics')}); "
                   "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
        command = [sys.executable, "-c", program, str(repository / "scripts/generate_tracking_benchmark.py")]
        raw = attenuation_config(reflection_id="h+1_k+0_s0")
        raw["scene"].update(pattern="H54", satellite_offset_px=3)
        raw["schedules"][0]["times_ns"] = [0, 700000000, 1100000000]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root, raw)
            target = root / "result"
            for args, status in ((["--config", str(source), "--validate-only"], "valid"),
                                 (["--config", str(source), "--output", str(target)], "complete"),
                                 (["--verify", str(target)], "verified")):
                result = subprocess.run(command + args, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], status)
            images = target / "sequences/sampled/images"
            parsed = parse_directory(images)
            self.assertEqual([frame.id for frame in parsed], [0, 1, 2])
            for frame in parsed:
                self.assertEqual(load_tiff(frame.filepath).shape, (65, 65))
            for folder in (target, images):
                refused = subprocess.run([sys.executable, str(repository / "scripts/export_segmentation_to_png.py"),
                                          "--source", str(folder), "--dest", str(root / "legacy")],
                                         cwd=root, capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("benchmark", refused.stderr.lower())
                self.assertFalse((root / "legacy").exists())
