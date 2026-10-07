"""Prepared inputs remain usable after publication and relocation."""
import copy
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.config import load_config
from trackleed.synthetic.package import generate_benchmark, verify_package
from trackleed.synthetic.scene import render_observation
from trackleed.synthetic.storage import pixel_descriptor
from trackleed.tests.test_synthetic_backgrounds import background_config


def write_config(root, raw):
    path = root / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class SyntheticBackgroundPackageTests(unittest.TestCase):
    def test_rgb_import_preserves_background_under_masks_and_publishes_portable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            rgb = np.full((33, 33, 3), [10, 20, 30], dtype=np.uint8)
            rgb[16, 23, 0] = 50
            Image.fromarray(rgb).save(sources / "background.png")
            screen = np.full((33, 33), 255, dtype=np.uint8)
            screen[:, 0] = 0
            occluders = np.zeros((33, 33), dtype=np.uint8)
            occluders[16, 22] = 255
            Image.fromarray(screen).save(sources / "screen.png")
            Image.fromarray(occluders).save(sources / "occluders.png")
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6)
            raw["condition"]["background_model"] = {
                "kind": "import", "path": "background.png", "gain": .01, "offset": .1,
                "screen_mask": "screen.png", "occluder_mask": "occluders.png"}
            path = write_config(sources, raw)
            original_bytes = (sources / "background.png").read_bytes()
            package = root / "package"
            report = generate_benchmark(path, package)
            self.assertEqual((report.frames, report.reference_records), (3, 18))
            with (package / "frames.csv").open() as stream:
                frames = list(csv.DictReader(stream))
            image = tifffile.imread(package / frames[0]["tiff_path"])
            self.assertEqual(image[16, 22], np.float32(.2815))
            self.assertEqual(image[16, 0], np.float32(.2815))
            self.assertAlmostEqual(float(image[16, 23]), .4011 + .5 * math.exp(-.5), places=7)
            self.assertEqual((package / "inputs/background.source").read_bytes(), original_bytes)
            metadata = json.loads((package / "background.json").read_text())
            self.assertEqual(metadata["rgb_to_gray"], [.299, .587, .114])
            self.assertEqual(metadata["sources"]["background"]["sha256"], sha256(original_bytes).hexdigest())
            self.assertEqual(metadata["sources"]["background"]["shape"], [33, 33, 3])
            prepared = tifffile.imread(package / "background/prepared.tiff")
            self.assertEqual(prepared.dtype, np.float64)
            self.assertAlmostEqual(float(prepared[16, 23]), .4011, places=12)
            with Image.open(package / "apparatus/occluders.png") as mask:
                np.testing.assert_array_equal(np.array(mask), occluders)
            for frame in frames[1:]:
                np.testing.assert_array_equal(tifffile.imread(package / frame["tiff_path"]), image)
            moved = root / "relocated"
            package.rename(moved)
            shutil.rmtree(sources)
            self.assertEqual(verify_package(moved).frames, 3)
            np.testing.assert_array_equal(render_observation(load_config(moved / "scene.json"), 0).image, image)
            self.assertEqual({p.suffix for p in (moved / "sequences/preview/images").iterdir()}, {".png"})

    def test_crop_uniform_scale_and_translation_share_binary_mask_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray = np.zeros((4, 4), dtype=np.uint16)
            gray[1:3, 1:3] = [[10, 20], [30, 40]]
            tifffile.imwrite(root / "gray.tiff", gray)
            screen = np.full((4, 4), 255, dtype=np.uint8)
            screen[1, 1] = 0
            occluders = np.zeros((4, 4), dtype=np.uint8)
            occluders[2, 2] = 255
            Image.fromarray(screen).save(root / "screen.png")
            Image.fromarray(occluders).save(root / "occluders.png")
            raw = background_config(image_size=[7, 7], center_px=[3, 3], lattice_scale_px=1, amplitude=0)
            raw["condition"]["background_model"] = {
                "kind": "import", "path": "gray.tiff", "gain": .01,
                "screen_mask": "screen.png", "occluder_mask": "occluders.png",
                "transform": {"crop_xywh": [1, 1, 2, 2], "scale": 2, "offset_px": [1, 1], "fill": .9}}
            package = root / "package"
            generate_benchmark(write_config(root, raw), package)
            expected = np.full((7, 7), .9, dtype=np.float32)
            expected[1:5, 1:5] = [[.1, .125, .175, .2], [.15, .175, .225, .25],
                                      [.25, .275, .325, .35], [.3, .325, .375, .4]]
            actual = tifffile.imread(package / "sequences/preview/rendered/frame_000000.tiff")
            np.testing.assert_array_equal(actual, expected)
            expected_screen = np.zeros((7, 7), dtype=np.uint8)
            expected_screen[1:5, 1:5] = 255
            expected_screen[1:3, 1:3] = 0
            expected_occluders = np.zeros((7, 7), dtype=np.uint8)
            expected_occluders[3:5, 3:5] = 255
            with Image.open(package / "apparatus/screen.png") as image:
                np.testing.assert_array_equal(np.array(image), expected_screen)
            with Image.open(package / "apparatus/occluders.png") as image:
                np.testing.assert_array_equal(np.array(image), expected_occluders)
            metadata = json.loads((package / "background.json").read_text())
            self.assertEqual(metadata["interpolation"], {"background": "bilinear_edge_clamp", "masks": "nearest"})
            self.assertEqual(metadata["transform"]["crop_xywh"], [1, 1, 2, 2])
            self.assertEqual(metadata["sources"]["background"]["dtype"], "<u2")
            self.assertEqual(verify_package(package).reference_records, 18)

    def test_unrepresentable_import_is_rejected_before_creating_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tifffile.imwrite(root / "huge.tiff", np.full((33, 33), 1e40, dtype=np.float64))
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6)
            raw["condition"]["background_model"] = {"kind": "import", "path": "huge.tiff", "gain": 1}
            path = write_config(root, raw)
            with self.assertRaisesRegex(ValueError, "float32"):
                generate_benchmark(path, root / "rejected")
            self.assertFalse((root / "rejected").exists())
            self.assertEqual(list(root.glob(".trackleed-incomplete-*")), [])

    def test_animated_png_is_not_silently_reduced_to_one_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = Image.fromarray(np.full((33, 33), 10, dtype=np.uint8))
            second = Image.fromarray(np.full((33, 33), 20, dtype=np.uint8))
            first.save(root / "animated.png", save_all=True, append_images=[second], duration=100, loop=0)
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6)
            raw["condition"]["background_model"] = {"kind": "import", "path": "animated.png", "gain": .01}
            with self.assertRaisesRegex(ValueError, "one image"):
                generate_benchmark(write_config(root, raw), root / "rejected")
            self.assertEqual(list(root.glob(".trackleed-incomplete-*")), [])

    def test_cli_resolves_relative_inputs_from_config_directory_and_preflight_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, other = root / "sources", root / "other"
            sources.mkdir()
            other.mkdir()
            Image.fromarray(np.full((33, 33), 12, dtype=np.uint8)).save(sources / "background.png")
            Image.fromarray(np.full((33, 33), 250, dtype=np.uint8)).save(other / "background.png")
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6, amplitude=0)
            raw["condition"]["background_model"] = {"kind": "import", "path": "background.png", "gain": .01}
            path = write_config(sources, raw)
            cli = Path(__file__).resolve().parents[2] / "scripts/generate_tracking_benchmark.py"
            command = [sys.executable, "-X", "utf8", str(cli), "--config", str(path)]
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            checked = subprocess.run([*command, "--validate-only"], cwd=other, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(checked.stdout)["status"], "valid")
            self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob("*")), before)
            made = subprocess.run([*command, "--output", str(root / "package")], cwd=other, capture_output=True, text=True)
            self.assertEqual(made.returncode, 0, made.stderr)
            image = tifffile.imread(root / "package/sequences/preview/rendered/frame_000000.tiff")
            np.testing.assert_array_equal(image, np.full((33, 33), .12, dtype=np.float32))
            (sources / "background.png").unlink()
            rejected = subprocess.run([*command, "--validate-only"], cwd=other, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("background.png", rejected.stderr)

    def test_malformed_sources_and_masks_are_not_silently_adapted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((33, 33), 12, dtype=np.uint8)).save(root / "gray.png")
            Image.fromarray(np.full((32, 33), 255, dtype=np.uint8)).save(root / "wrong-size.png")
            Image.fromarray(np.full((33, 33), 127, dtype=np.uint8)).save(root / "nonbinary.png")
            tifffile.imwrite(root / "nonfinite.tiff", np.full((33, 33), np.nan, dtype=np.float32))
            Image.fromarray(np.zeros((33, 33, 4), dtype=np.uint8)).save(root / "rgba.png")
            variants = [
                {"path": "wrong-size.png"}, {"path": "nonfinite.tiff"}, {"path": "rgba.png"},
                {"screen_mask": "wrong-size.png"}, {"occluder_mask": "nonbinary.png"},
                {"transform": {"crop_xywh": [0, 0, 34, 33]}},
            ]
            for variant in variants:
                with self.subTest(variant=variant):
                    raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6)
                    raw["condition"]["background_model"] = {"kind": "import", "path": "gray.png", "gain": 1, **variant}
                    with self.assertRaises(ValueError):
                        generate_benchmark(write_config(root, raw), root / "rejected")
                    self.assertFalse((root / "rejected").exists())
                    self.assertEqual(list(root.glob(".trackleed-incomplete-*")), [])

    def test_procedural_packages_share_pixels_at_common_times_with_declared_texture_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6,
                                    drift_px_per_s=[1, 0])
            raw["condition"]["background_model"] = {"kind": "procedural", "gradient": [.1, .02],
                                                       "texture_amplitude": .02, "texture_seed": 72}
            raw["schedules"] = [{"id": "dense", "times_ns": [0, 1000000000, 2000000000]},
                                  {"id": "sparse", "times_ns": [0, 2000000000]}]
            path = write_config(root, raw)
            one, two = root / "one", root / "two"
            generate_benchmark(path, one)
            generate_benchmark(path, two)
            for dense, sparse in ((0, 0), (2, 1)):
                for package in (one, two):
                    left = tifffile.imread(package / f"sequences/dense/rendered/frame_{dense:06d}.tiff")
                    right = tifffile.imread(package / f"sequences/sparse/rendered/frame_{sparse:06d}.tiff")
                    np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(left, tifffile.imread(one / f"sequences/dense/rendered/frame_{dense:06d}.tiff"))
            metadata = json.loads((one / "background.json").read_text())
            self.assertEqual(metadata["texture"]["seed"], 72)
            self.assertEqual(metadata["additional_noise"], "none")
            schema = json.loads((one / "schema.json").read_text())
            self.assertEqual(schema["stage"]["background"], ["constant", "procedural", "import"])
            self.assertIn("polygon", schema["stage"]["occluders"])

    def test_rehashed_prepared_background_and_metadata_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6)
            raw["condition"]["background_model"] = {"kind": "procedural", "gradient": [.1, .02]}
            original = root / "original"
            generate_benchmark(write_config(root, raw), original)
            for name in ("background/prepared.tiff", "background.json"):
                with self.subTest(name=name):
                    target = root / ("changed-image" if name.endswith("tiff") else "changed-metadata")
                    shutil.copytree(original, target)
                    path = target / name
                    if name.endswith("tiff"):
                        values = tifffile.imread(path)
                        values[0, 0] += .1
                        tifffile.imwrite(path, values, metadata=None)
                        descriptor = pixel_descriptor(values)
                    else:
                        data = json.loads(path.read_text())
                        data["additional_noise"] = "hidden"
                        path.write_text(json.dumps(data))
                        descriptor = {}
                    manifest_path = target / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["files"][name] = {"sha256": sha256(path.read_bytes()).hexdigest(), **descriptor}
                    manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "[Bb]ackground"):
                        verify_package(target)

    def test_unsupported_tiff_photometric_is_not_interpreted_as_gray_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            palette = np.zeros((3, 256), dtype=np.uint16)
            palette[0, :] = 65535
            indices = np.arange(33 * 33, dtype=np.uint16).reshape(33, 33).astype(np.uint8)
            tifffile.imwrite(root / "palette.tiff", indices, photometric="palette", colormap=palette)
            tifffile.imwrite(root / "white.tiff", indices, photometric="miniswhite")
            tifffile.imwrite(root / "volume.tiff", np.zeros((33, 33, 3), dtype=np.uint16),
                             photometric="minisblack", volumetric=True)
            for name in ("palette.tiff", "white.tiff", "volume.tiff"):
                with self.subTest(name=name):
                    raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6)
                    raw["condition"]["background_model"] = {"kind": "import", "path": name, "gain": 1 / 255}
                    with self.assertRaisesRegex(ValueError, "TIFF.*(grayscale|RGB|photometric)"):
                        generate_benchmark(write_config(root, raw), root / ("rejected-" + name))
                    self.assertEqual(list(root.glob(".trackleed-incomplete-*")), [])

    def test_rgb_tiff_keeps_channel_units_before_explicit_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.zeros((33, 33, 3), dtype=np.uint16)
            rgb[:, :, 0] = 65535
            tifffile.imwrite(root / "rgb.tiff", rgb, photometric="rgb")
            raw = background_config(image_size=[33, 33], center_px=[16, 16], lattice_scale_px=6, amplitude=0)
            raw["condition"]["background_model"] = {"kind": "import", "path": "rgb.tiff", "gain": 1 / 65535}
            observation = render_observation(load_config(write_config(root, raw)), 0)
            np.testing.assert_array_equal(observation.image, np.full((33, 33), .299, dtype=np.float32))
