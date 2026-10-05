"""Real file round trips and publication behavior for benchmark packages."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.package import generate_benchmark, verify_package


def write_config(directory, **scene_values):
    scene = {"image_size": [65, 65], "center_px": [32, 32],
             "lattice_scale_px": 12, "sigma_px": 1}
    scene.update(scene_values)
    path = directory / "input.json"
    path.write_text(json.dumps({"schema_version": 1, "scene": scene,
                               "condition": {}, "schedule": {}}))
    return path


class SyntheticPackageTests(unittest.TestCase):
    def test_complete_h6_package_round_trips_through_real_trackleed_reader(self):
        from trackleed.io.file_parser import parse_directory
        from trackleed.io.image_reader import load_tiff
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root)
            output = root / "result"
            generated = generate_benchmark(source, output)
            report = verify_package(output)
            self.assertEqual((report.frames, report.reference_records), (3, 18))
            self.assertEqual(generated, report)
            frames = parse_directory(output / "images")
            self.assertEqual([f.id for f in frames], [0, 1, 2])
            arrays = [load_tiff(f.filepath) for f in frames]
            for array in arrays[1:]:
                np.testing.assert_array_equal(array, arrays[0])
            self.assertEqual(arrays[0].dtype, np.uint8)
            self.assertEqual(arrays[0][32, 44], 153)
            image = tifffile.imread(output / "rendered" / "frame_000000.tiff")
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual(image[32, 44], np.float32(0.6))
            with (output / "reference.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertEqual(len({r["reflection_id"] for r in rows}), 6)
            self.assertEqual(rows[0]["dx_px"], "")
            self.assertEqual(rows[6]["dx_px"], "0.0")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["package_type"], "trackleed.synthetic-benchmark")
            self.assertTrue(manifest["generator"]["source_sha256"])
            self.assertIsInstance(manifest["generator"]["git_dirty"], bool)
            self.assertEqual(len(manifest["files"]), 12)
            self.assertEqual(set(p.suffix for p in (output / "images").iterdir()), {".png"})
            with Image.open(frames[0].filepath) as preview:
                self.assertEqual(preview.mode, "L")


    def test_existing_targets_and_interrupted_writes_never_publish(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_config(root)
            target = root / "existing"
            target.mkdir()
            with self.assertRaisesRegex(ValueError, "exists"):
                generate_benchmark(config, target)
            target = root / "broken"
            with patch("tifffile.imwrite", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    generate_benchmark(config, target)
            self.assertFalse(target.exists())
            stages = list(root.glob(".trackleed-incomplete-*"))
            self.assertEqual(len(stages), 1)
            with self.assertRaisesRegex(ValueError, "Staging"):
                verify_package(stages[0])

    def test_late_empty_directory_conflict_is_not_overwritten(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root)
            target = root / "race"
            original_write = tifffile.imwrite

            def competing_writer(*args, **kwargs):
                target.mkdir(exist_ok=True)
                return original_write(*args, **kwargs)

            with patch("tifffile.imwrite", side_effect=competing_writer):
                with self.assertRaises((ValueError, FileExistsError)):
                    generate_benchmark(source, target)
            self.assertEqual(list(target.iterdir()), [])

    def test_staging_with_complete_manifest_cannot_be_verified_via_an_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "done"
            generate_benchmark(write_config(root), target)
            stage = root / ".trackleed-incomplete-paused"
            target.rename(stage)
            with self.assertRaisesRegex(ValueError, "Staging"):
                verify_package(stage)
            alias = root / "alias"
            try:
                alias.symlink_to(stage, target_is_directory=True)
            except OSError:
                return  # Symlink creation may require a Windows privilege.
            with self.assertRaisesRegex(ValueError, "Staging|symlink"):
                verify_package(alias)

    def test_tampered_data_and_missing_reference_are_detected_after_read(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "done"
            generate_benchmark(write_config(root), target)
            references = target / "reference.csv"
            content = references.read_text()
            references.write_text(content.replace("h-1_k+0", "impostor", 1))
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_package(target)
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["reference.csv"]["sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reference.csv"):
                verify_package(target)

    def test_invalid_configuration_leaves_no_output_or_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root, pattern="H18")
            with self.assertRaisesRegex(ValueError, "pattern"):
                generate_benchmark(source, root / "result")
            self.assertEqual(set(p.name for p in root.iterdir()), {"input.json"})


    def test_manifest_identity_provenance_and_inventory_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "done"
            generate_benchmark(write_config(root), target)
            path = target / "manifest.json"
            original = json.loads(path.read_text())
            for field, value in (("scene_id", "wrong-scene"), ("generator", {}),
                                 ("schema_version", True), ("plan", {})):
                payload = dict(original)
                payload[field] = value
                path.write_text(json.dumps(payload))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_package(target)
            path.write_text(json.dumps(original))
            (target / "images" / "unexpected.png").write_bytes(b"extra frame")
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_package(target)

    def test_repeated_output_is_identical_and_portable(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root)
            first, second = root / "a", root / "b"
            generate_benchmark(source, first)
            generate_benchmark(source, second)
            for name in ("frames.csv", "reference.csv", "images/frame_000000.png",
                         "rendered/frame_000000.tiff"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            copy = root / "relocated"
            shutil.copytree(first, copy)
            self.assertEqual(verify_package(copy).reference_records, 18)

    def test_publication_and_late_conflict_on_repository_filesystem(self):
        from unittest.mock import patch
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix=".synthetic-test-", dir=repository) as directory:
            root = Path(directory)
            source = write_config(root)
            target = root / "successful"
            self.assertEqual(generate_benchmark(source, target).frames, 3)
            target = root / "competing"
            original_write = tifffile.imwrite

            def create_competing_directory(*args, **kwargs):
                target.mkdir(exist_ok=True)
                return original_write(*args, **kwargs)

            with patch("tifffile.imwrite", side_effect=create_competing_directory):
                with self.assertRaises((ValueError, FileExistsError)):
                    generate_benchmark(source, target)
            self.assertEqual(list(target.iterdir()), [])
