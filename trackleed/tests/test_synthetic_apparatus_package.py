"""Apparatus packages persist and verify real masks, pixels and full references."""
import copy
import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.storage import pixel_descriptor
from trackleed.synthetic.package import generate_benchmark, verify_package
from trackleed.tests.test_synthetic_apparatus import apparatus_config


def write_source(root, raw):
    source = root / "config.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    return source


def rows(root, name):
    with (root / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ApparatusPackageTests(unittest.TestCase):
    def test_package_records_static_apparatus_and_latent_positions_after_real_readback(self):
        raw = apparatus_config(center_px=[-6, 16], lattice_scale_px=6, sigma_px=1)
        raw["condition"]["apparatus"] = {
            "screen": {"kind": "circle", "center_px": [0, 16], "radius_px": 2},
            "occluders": [{"kind": "vertical_band", "x_range_px": [0, 0]}],
            "outside_screen_level": .03, "occluder_level": .07}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            report = generate_benchmark(write_source(root, raw), package)
            self.assertEqual(report, verify_package(package))
            self.assertEqual((report.frames, report.reference_records), (2, 12))
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 5)
            for name in ("apparatus/screen.png", "apparatus/occluders.png"):
                self.assertIn(name, manifest["files"])
                with Image.open(package / name) as image:
                    self.assertEqual((image.size, image.mode), ((65, 65), "L"))
                    self.assertEqual(set(np.unique(image)), {0, 255})
            with Image.open(package / "apparatus/screen.png") as image:
                self.assertEqual(np.count_nonzero(image), 9)
                self.assertEqual(image.getpixel((2, 16)), 255)
                self.assertEqual(image.getpixel((3, 16)), 0)
            with Image.open(package / "apparatus/occluders.png") as image:
                self.assertEqual(np.count_nonzero(image), 65)
            references = rows(package, "reference.csv")
            self.assertEqual(len({r["reflection_id"] for r in references}), 6)
            left = [r for r in references if r["reflection_id"] == "h-1_k+0"]
            self.assertEqual([float(r["x_px"]) for r in left], [-12, -12])
            self.assertTrue(all(r["prequant_signal_present"] == "false" for r in left))
            self.assertTrue(all(float(r["frame_loss_fraction"]) == 1 for r in left))
            frames = rows(package, "frames.csv")
            self.assertEqual(frames[0]["png_pixel_sha256"], frames[1]["png_pixel_sha256"])
            for frame in frames:
                self.assertEqual(tifffile.imread(package / frame["tiff_path"])[16, 0], np.float32(.07))
                with Image.open(package / frame["png_path"]) as image:
                    self.assertEqual(image.getpixel((0, 16)), 18)
            images = package / "sequences/preview/images"
            self.assertEqual(sorted(p.suffix for p in images.iterdir()), [".png", ".png"])

    def test_rehashed_apparatus_mask_and_reference_tampering_are_still_rejected(self):
        for changed in ("apparatus/screen.png", "apparatus/occluders.png", "reference.csv"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = root / "package"
                generate_benchmark(write_source(root, apparatus_config()), package)
                path = package / changed
                manifest = json.loads((package / "manifest.json").read_text())
                if changed.endswith("png"):
                    with Image.open(path) as image:
                        pixels = np.array(image)
                    pixels[0, 0] = 255 - pixels[0, 0]
                    Image.fromarray(pixels).save(path)
                    manifest["files"][changed].update(pixel_descriptor(pixels))
                else:
                    records = rows(package, changed)
                    records[0]["w3"] = "999.0"
                    with path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                        writer.writeheader()
                        writer.writerows(records)
                manifest["files"][changed]["sha256"] = sha256(path.read_bytes()).hexdigest()
                (package / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "match scene|disagree with scene"):
                    verify_package(package)

    def test_optional_instance_masks_and_schedule_do_not_change_pixels_or_reference_state(self):
        raw = apparatus_config(center_px=[28, 32], lattice_scale_px=10, sigma_px=1,
                               drift_px_per_s=[-4, 0])
        raw["condition"]["apparatus"] = {"occluders": [
            {"kind": "vertical_band", "x_range_px": [26, 34]}]}
        raw["schedules"] = [{"id": "dense", "times_ns": [0, 1000000000, 2000000000]},
                            {"id": "sparse", "times_ns": [0, 2000000000]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = []
            for enabled in (False, True):
                raw["export_masks"] = enabled
                package = root / str(enabled)
                generate_benchmark(write_source(root, raw), package)
                self.assertEqual(verify_package(package).reference_records, 30)
                outputs.append(package)
            frames = rows(outputs[0], "frames.csv")
            self.assertEqual(frames[0]["png_pixel_sha256"], frames[3]["png_pixel_sha256"])
            self.assertEqual(frames[2]["tiff_pixel_sha256"], frames[4]["tiff_pixel_sha256"])
            self.assertEqual(frames, rows(outputs[1], "frames.csv"))
            optional = {"geometry_mask_path", "signal_mask_path", "bbox_geometry", "bbox_signal"}
            for a, b in zip(rows(outputs[0], "reference.csv"), rows(outputs[1], "reference.csv")):
                self.assertEqual({k: v for k, v in a.items() if k not in optional},
                                 {k: v for k, v in b.items() if k not in optional})
            for output in outputs:
                self.assertTrue((output / "apparatus/screen.png").is_file())

    def test_apparatus_write_failure_keeps_output_unpublished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result"
            source = write_source(root, apparatus_config())
            original = Image.Image.save
            def fail_apparatus(image, path, *args, **kwargs):
                if Path(path).name == "occluders.png":
                    raise OSError("apparatus disk failure")
                return original(image, path, *args, **kwargs)
            with patch.object(Image.Image, "save", fail_apparatus):
                with self.assertRaisesRegex(OSError, "apparatus disk failure"):
                    generate_benchmark(source, output)
            self.assertFalse(output.exists())
            stages = list(root.glob(".trackleed-incomplete-*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "failure.json").is_file())
            with self.assertRaisesRegex(ValueError, "Staging"):
                verify_package(stages[0])

    def test_v5_cli_and_legacy_refusal_work_outside_repo_without_models(self):
        repo = Path(__file__).resolve().parents[2]
        cli = repo / "scripts/generate_tracking_benchmark.py"
        program = """
import importlib.abc, runpy, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PyQt6', 'torch', 'ultralytics'):
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root, apparatus_config())
            output = root / "result"
            base = [sys.executable, "-X", "utf8", "-c", program, str(cli)]
            commands = [["--config", str(source), "--output", str(output), "--validate-only"],
                        ["--config", str(source), "--output", str(output)], ["--verify", str(output)]]
            for index, args in enumerate(commands):
                result = subprocess.run(base + args, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if index == 0:
                    self.assertFalse(output.exists())
            legacy = repo / "scripts/export_segmentation_to_png.py"
            for path in (output, output / "apparatus", output / "sequences/preview/images"):
                destination = root / "legacy"
                result = subprocess.run([sys.executable, str(legacy), "--source", str(path),
                                         "--dest", str(destination)], cwd=root, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("benchmark", result.stderr.lower())
                self.assertFalse(destination.exists())
