"""CLI use outside the repository and legacy-conversion compatibility."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from trackleed.tests.test_synthetic_package import write_config
from trackleed.synthetic.package import generate_benchmark

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "generate_tracking_benchmark.py"
LEGACY = ROOT / "scripts" / "export_segmentation_to_png.py"


class SyntheticCliTests(unittest.TestCase):
    def test_cli_validates_without_writing_or_importing_gui_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_config(root)
            output = root / "result"
            program = """
import importlib.abc, runpy, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PyQt6', 'torch', 'ultralytics'):
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
sys.argv = [sys.argv[1], '--config', sys.argv[2], '--output', sys.argv[3], '--validate-only']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
            result = subprocess.run([sys.executable, "-c", program, str(CLI),
                                     str(source), str(output)], cwd=root,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["plan"]["reference_records"], 18)
            self.assertEqual(set(p.name for p in root.iterdir()), {"input.json"})

    def test_legacy_refuses_new_package_and_subfolders_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "benchmark"
            generate_benchmark(write_config(root), package)
            for source in (package, package / "images", package / "rendered"):
                output = root / "legacy-output"
                result = subprocess.run([sys.executable, str(LEGACY), "--source",
                                         str(source), "--dest", str(output)],
                                        cwd=root, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("benchmark", result.stderr.lower())
                self.assertFalse(output.exists())
            (package / "manifest.json").write_text("{corrupt")
            result = subprocess.run([sys.executable, str(LEGACY), "--source",
                                     str(package / "images"), "--dest", str(output)],
                                    cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_legacy_valid_dataset_keeps_png16_and_mask_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            for folder in ("images", "masks", "metadata"):
                (source / "train" / folder).mkdir(parents=True)
            tifffile.imwrite(source / "train/images/old.tiff",
                             np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
            tifffile.imwrite(source / "train/masks/old.tiff", np.array([[0, 1, 1]]))
            (source / "train/metadata/old.json").write_text('{"legacy": true}')
            output = root / "converted"
            result = subprocess.run([sys.executable, str(LEGACY), "--source", str(source),
                                     "--dest", str(output), "--splits", "train"],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with Image.open(output / "train/images/old.png") as image:
                np.testing.assert_array_equal(np.asarray(image), [[0, 32768, 65535]])
            with Image.open(output / "train/masks/old.png") as image:
                np.testing.assert_array_equal(np.asarray(image), [[0, 255, 255]])
            self.assertEqual((output / "train/metadata/old.json").read_text(), '{"legacy": true}')
