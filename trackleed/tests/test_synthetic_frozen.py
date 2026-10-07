"""Read immutable historical packages with the current public reader."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from PIL import Image
import tifffile

from trackleed.synthetic.package import verify_package


FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"
# Deliberately pinned; never regenerate historical packages during a test.
ARCHIVES = {
    "benchmark-v1.zip": "262397deadd4cb8fe3528ffde084f5df335da9f5639983d85e4ac8591194423d",
    "benchmark-v2.zip": "5ef1b63494d93428ed796eb6cefb88679ac85ff23d4682209090f04223e415f5",
}


def read_rows(root, filename):
    with (root / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class FrozenSyntheticPackageTests(unittest.TestCase):
    def test_historical_packages_are_unchanged_and_readable(self):
        for name, digest in ARCHIVES.items():
            with self.subTest(archive=name), tempfile.TemporaryDirectory() as directory:
                archive = FIXTURES / name
                self.assertTrue(archive.is_file(), f"Missing frozen benchmark fixture: {archive}")
                self.assertEqual(
                    sha256(archive.read_bytes()).hexdigest(), digest,
                    f"Frozen fixture changed: {name}; preserve historical bytes.",
                )
                root = Path(directory)
                with ZipFile(archive) as package:
                    package.extractall(root)
                report = verify_package(root)
                version = 1 if "v1" in name else 2
                self.assertEqual(
                    (report.frames, report.reference_records),
                    (2, 12) if version == 1 else (5, 30),
                )
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema_version"], version)
                frames = read_rows(root, "frames.csv")
                references = read_rows(root, "reference.csv")
                self.assertEqual(len({row["reflection_id"] for row in references}), 6)
                for row in frames:
                    with Image.open(root / row["png_path"]) as image:
                        self.assertEqual((image.size, image.mode), ((33, 33), "L"))
                    rendered = tifffile.imread(root / row["tiff_path"])
                    self.assertEqual(rendered.dtype, np.float32)
                if version == 1:
                    self.assertEqual([row["time_ns"] for row in frames], ["0", "100000000"])
                    with Image.open(root / frames[0]["png_path"]) as image:
                        self.assertEqual(image.getpixel((22, 16)), 153)
                    right = [r for r in references if r["reflection_id"] == "h+1_k+0"]
                    self.assertEqual([float(r["x_px"]) for r in right], [22.0, 22.0])
                else:
                    self.assertEqual(
                        [float(row["energy_ev"]) for row in frames],
                        [80, 140, 200, 80, 200],
                    )
                    self.assertEqual(frames[0]["png_pixel_sha256"], frames[3]["png_pixel_sha256"])
                    self.assertEqual(frames[2]["png_pixel_sha256"], frames[4]["png_pixel_sha256"])
                    self.assertEqual(frames[1]["dt_ns"], "1000000000")
                    self.assertEqual(frames[4]["dt_ns"], "2000000000")

    def test_current_reader_rejects_corrupted_copies_of_historical_packages(self):
        for name in ARCHIVES:
            with self.subTest(archive=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with ZipFile(FIXTURES / name) as package:
                    package.extractall(root)
                path = root / "reference.csv"
                path.write_bytes(path.read_bytes() + b"corrupted\n")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_package(root)

    def test_frozen_v3_keeps_h54_ellipses_and_independent_instance_masks(self):
        archive = FIXTURES / "benchmark-v3.zip"
        self.assertEqual(sha256(archive.read_bytes()).hexdigest(),
                         "21ff2bc6daf5c54f0188cfa649d508329fe70307dcf209207ff5fa0b6fc551c5")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ZipFile(archive) as package:
                package.extractall(root)
            report = verify_package(root)
            self.assertEqual((report.frames, report.reference_records), (2, 108))
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 3)
            references = read_rows(root, "reference.csv")
            self.assertEqual(len({r["reflection_id"] for r in references}), 54)
            satellite = [r for r in references if r["reflection_id"] == "h+1_k+0_s0"]
            self.assertTrue(all(r["parent_id"] == "h+1_k+0" for r in satellite))
            self.assertAlmostEqual(float(satellite[0]["x_px"]), 22.70820393249937)
            for state in references:
                self.assertEqual(float(state["theta_rad"]), .5)
                with Image.open(root / state["geometry_mask_path"]) as image:
                    geometry = np.array(image)
                with Image.open(root / state["signal_mask_path"]) as image:
                    np.testing.assert_array_equal(np.array(image), geometry)

    def test_frozen_v4_keeps_attenuation_coverage_and_masks(self):
        archive = FIXTURES / "benchmark-v4.zip"
        self.assertEqual(sha256(archive.read_bytes()).hexdigest(),
                         "b754a30ead38cf98950b75ccfea15a89dea89b41290db2ddb593d3351de008e0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ZipFile(archive) as package:
                package.extractall(root)
            report = verify_package(root)
            self.assertEqual((report.frames, report.reference_records), (3, 18))
            references = read_rows(root, "reference.csv")
            target = [r for r in references if r["reflection_id"] == "h+1_k+0"]
            self.assertEqual([r["amplitude"] for r in target], ["0.5", "0.0", "0.5"])
            self.assertEqual([float(r["x_px"]) for r in target], [22, 22, 22])
            for row, code in zip(target, (153, 26, 153)):
                with Image.open(root / row["signal_mask_path"]) as image:
                    self.assertEqual(bool(np.array(image).any()), code != 26)
            for frame, code in zip(read_rows(root, "frames.csv"), (153, 26, 153)):
                with Image.open(root / frame["png_path"]) as image:
                    self.assertEqual(image.getpixel((22, 16)), code)
            event = json.loads((root / "coverage.json").read_text())["events"][0]
            self.assertEqual(event["minimum_sample_count"], 1)
            self.assertTrue(event["captured_zero_gap"])
            self.assertEqual(event["context_status"], "bracketed")
