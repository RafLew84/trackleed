"""Instance references survive real package export and TrackLEED image loading."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.package import generate_benchmark, verify_package
from trackleed.tests.test_synthetic_patterns import pattern_config


def table(root):
    with (root / "reference.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class SyntheticInstanceTests(unittest.TestCase):
    def test_h54_package_keeps_overlapping_instances_separate_and_images_unchanged(self):
        from trackleed.io.file_parser import parse_directory
        from trackleed.io.image_reader import load_tiff
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = pattern_config(sigma_px=2)
            packages = []
            for export in (False, True):
                raw["export_masks"] = export
                source = root / f"config-{export}.json"
                source.write_text(json.dumps(raw), encoding="utf-8")
                output = root / f"package-{export}"
                report = generate_benchmark(source, output)
                self.assertEqual((report.frames, report.reference_records), (2, 108))
                self.assertEqual(verify_package(output), report)
                packages.append(output)
            plain, masked = packages
            folder = "sequences/static/images"
            frames = parse_directory(masked / folder)
            self.assertEqual([frame.id for frame in frames], [0, 1])
            self.assertEqual(load_tiff(frames[0].filepath).dtype, np.uint8)
            for frame in frames:
                self.assertEqual(frame.filepath.read_bytes(), (plain / folder / frame.filepath.name).read_bytes())
            old_rows, rows = table(plain), table(masked)
            optional = {"geometry_mask_path", "signal_mask_path", "bbox_geometry", "bbox_signal"}
            self.assertEqual([{k: v for k, v in row.items() if k not in optional} for row in old_rows],
                             [{k: v for k, v in row.items() if k not in optional} for row in rows])
            self.assertTrue(all(not row["geometry_mask_path"] for row in old_rows))
            by_id = {row["reflection_id"]: row for row in rows if row["frame_index"] == "0"}
            parent, satellite = by_id["h+1_k+0"], by_id["h+1_k+0_s0"]
            self.assertEqual(satellite["parent_id"], parent["reflection_id"])
            self.assertEqual(json.loads(parent["bbox_geometry"]), [78, 58, 91, 71])
            self.assertEqual(json.loads(satellite["bbox_signal"]), [87, 58, 100, 71])
            for row in (parent, satellite):
                with Image.open(masked / row["geometry_mask_path"]) as mask:
                    self.assertEqual((mask.mode, mask.size), ("L", (129, 129)))
                    self.assertEqual(mask.getpixel((88, 64)), 255)
                    self.assertLessEqual(set(np.unique(mask)), {0, 255})
                self.assertIn(satellite["reflection_id"] if row is parent else parent["reflection_id"],
                              json.loads(row["overlap_contour_ids"]))
            image = tifffile.imread(masked / "sequences/static/rendered/frame_000000.tiff")
            np.testing.assert_array_equal(image, tifffile.imread(plain / "sequences/static/rendered/frame_000000.tiff"))
            self.assertEqual(len(list((masked / folder).iterdir())), 2)

    def test_zero_and_elliptical_instances_roundtrip_with_null_boxes_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pattern in ("H6", "H18", "H54"):
                raw = pattern_config(pattern, amplitude=0, satellite_amplitude=1e-50)
                raw["scene"].pop("sigma_px")
                raw["scene"]["shape"] = {"sigma_u_px": 2, "sigma_v_px": 1, "theta_rad": 0.5235987755982988}
                raw["export_masks"] = True
                source = root / f"{pattern}.json"
                source.write_text(json.dumps(raw), encoding="utf-8")
                output = root / pattern
                report = generate_benchmark(source, output)
                count = {"H6": 6, "H18": 18, "H54": 54}[pattern]
                self.assertEqual(report.reference_records, count * 2)
                rows = table(output)
                self.assertTrue(all(r["sigma_u_px"] == "2" and r["sigma_v_px"] == "1" for r in rows))
                for row in rows:
                    self.assertTrue(row["bbox_geometry"])
                    self.assertEqual(row["bbox_signal"] == "", row["role"] == "primary")
                    with Image.open(output / row["signal_mask_path"]) as mask:
                        self.assertEqual(bool(np.any(mask)), row["role"] == "satellite")
                    self.assertEqual(row["encoded_changed_pixels"], "0")

    def test_semantically_corrupted_masks_are_rejected_even_with_updated_hashes(self):
        from trackleed.synthetic.storage import file_sha256, pixel_descriptor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = pattern_config("H6")
            raw["export_masks"] = True
            source = root / "input.json"
            source.write_text(json.dumps(raw), encoding="utf-8")
            output = root / "result"
            generate_benchmark(source, output)
            path = table(output)[0]["geometry_mask_path"]
            array = np.zeros((129, 129), dtype=np.uint8)
            Image.fromarray(array).save(output / path)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][path] = {"sha256": file_sha256(output / path), **pixel_descriptor(array)}
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Mask does not match"):
                verify_package(output)

    def test_mask_write_failure_does_not_publish_a_partial_package(self):
        from unittest.mock import patch
        original_save = Image.Image.save

        def disk_failure(image, fp, *args, **kwargs):
            if "masks" in Path(fp).parts:
                raise OSError("mask disk failure")
            return original_save(image, fp, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = pattern_config("H6")
            raw["export_masks"] = True
            source = root / "input.json"
            source.write_text(json.dumps(raw))
            with patch.object(Image.Image, "save", disk_failure):
                with self.assertRaisesRegex(OSError, "mask disk failure"):
                    generate_benchmark(source, root / "result")
            self.assertFalse((root / "result").exists())
            stages = list(root.glob(".trackleed-incomplete-*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "failure.json").exists())

    def test_v3_cli_works_without_models_and_legacy_converter_refuses_masks(self):
        import subprocess
        import sys
        repository = Path(__file__).resolve().parents[2]
        cli = repository / "scripts/generate_tracking_benchmark.py"
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
            raw = pattern_config("H18")
            raw["export_masks"] = True
            source = root / "input.json"
            source.write_text(json.dumps(raw))
            output = root / "result"
            command = [sys.executable, "-c", program, str(cli)]
            validate = subprocess.run(command + ["--config", str(source), "--validate-only"],
                                      cwd=root, text=True, capture_output=True)
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertEqual(json.loads(validate.stdout)["plan"]["reference_records"], 36)
            self.assertEqual(set(p.name for p in root.iterdir()), {"input.json"})
            for arguments in (["--config", str(source), "--output", str(output)],
                              ["--verify", str(output)]):
                result = subprocess.run(command + arguments, cwd=root, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            legacy = repository / "scripts/export_segmentation_to_png.py"
            for folder in (output, output / "sequences/static/images", output / "sequences/static/masks"):
                result = subprocess.run([sys.executable, str(legacy), "--source", str(folder),
                                         "--dest", str(root / "legacy")], cwd=root,
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("benchmark", result.stderr.lower())
                self.assertFalse((root / "legacy").exists())
