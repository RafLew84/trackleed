from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from trackleed.analysis import dam4sam_backend


class Dam4SamBackendTests(unittest.TestCase):
    def test_checkpoint_selection_prefers_tracker_checkpoint_in_models_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_dir = root / "tracking_models" / "dam4sam"
            repo_path = root / "dam4sam"
            models_dir.mkdir(parents=True)
            (repo_path / "checkpoints").mkdir(parents=True)
            (repo_path / "dam4sam_tracker.py").write_text("# tracker marker\n")
            (repo_path / "sam2").mkdir()
            (repo_path / "sam2" / "build_sam.py").write_text("# build marker\n")

            repo_checkpoint = repo_path / "checkpoints" / "sam2.1_hiera_large.pt"
            models_checkpoint = models_dir / "sam2.1_hiera_large.pt"
            repo_checkpoint.write_bytes(b"repo")
            models_checkpoint.write_bytes(b"models")

            discovered = dam4sam_backend.discover_dam4sam_checkpoints(
                models_dir=models_dir,
                repo_path=repo_path,
            )
            selected = dam4sam_backend.select_default_dam4sam_checkpoint(
                discovered,
                "sam21pp-L",
                models_dir=models_dir,
            )

        self.assertEqual(selected, models_checkpoint.resolve())

    def test_parse_prefixed_json_ignores_noisy_stdout(self) -> None:
        stdout = (
            "warning line\n"
            + dam4sam_backend._IMPORT_PROBE_PREFIX
            + '{"torch_importable": true, "numpy_version": "1.26.4"}\n'
        )

        payload = dam4sam_backend._parse_prefixed_json_stdout(
            stdout,
            dam4sam_backend._IMPORT_PROBE_PREFIX,
        )

        self.assertTrue(payload["torch_importable"])
        self.assertEqual(payload["numpy_version"], "1.26.4")

    def test_probe_environment_reports_ready_contract_with_mocked_import_probe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = root / "dam4sam"
            models_dir = root / "tracking_models" / "dam4sam"
            sam2_dir = repo_path / "sam2"
            repo_path.mkdir()
            sam2_dir.mkdir()
            models_dir.mkdir(parents=True)
            (repo_path / "dam4sam_tracker.py").write_text("# tracker marker\n")
            (repo_path / "vot_wrapper_dam4sam.py").write_text("# wrapper marker\n")
            (sam2_dir / "build_sam.py").write_text("# build marker\n")
            (sam2_dir / "sam21pp_hiera_l.yaml").write_text("model: {}\n")
            checkpoint = models_dir / "sam2.1_hiera_large.pt"
            checkpoint.write_bytes(b"checkpoint")

            probe_payload = {
                "torch_importable": True,
                "torch_version": "2.1.0+cu121",
                "torch_cuda_version": "12.1",
                "cuda_available": True,
                "torchvision_importable": True,
                "cv2_importable": True,
                "numpy_importable": True,
                "numpy_version": "1.26.4",
                "yaml_importable": True,
                "hydra_importable": True,
                "omegaconf_importable": True,
                "pillow_importable": True,
                "vot_importable": True,
                "sam2_importable": True,
                "sam2_module_path": str(sam2_dir / "__init__.py"),
                "sam2_build_sam_importable": True,
                "dam4sam_tracker_importable": True,
                "dam4sam_tracker_module_path": str(repo_path / "dam4sam_tracker.py"),
                "vot_wrapper_dam4sam_importable": True,
                "vot_wrapper_dam4sam_module_path": str(repo_path / "vot_wrapper_dam4sam.py"),
                "sam2_cuda_extension_importable": False,
                "sam2_cuda_extension_error": "ModuleNotFoundError: No module named 'sam2._C'",
            }
            probe_result = dam4sam_backend.Dam4SamImportProbeResult(
                command=[sys.executable, "-c", "probe"],
                returncode=0,
                stdout="",
                stderr="",
                payload=probe_payload,
            )

            with mock.patch.object(
                dam4sam_backend,
                "_run_dam4sam_import_probe",
                return_value=probe_result,
            ):
                status = dam4sam_backend.probe_dam4sam_environment(
                    models_dir=models_dir,
                    repo_path=repo_path,
                    python_executable=sys.executable,
                    tracker_name="sam21pp-L",
                )

        self.assertTrue(status.is_ready)
        self.assertEqual(status.selected_checkpoint, checkpoint.resolve())
        report = dam4sam_backend.format_dam4sam_environment_report(status)
        self.assertIn("DAM4SAM environment diagnostics", report)
        self.assertIn("Ready: yes", report)
        self.assertIn("SAM2 CUDA extension: missing", report)
        self.assertNotIn("Errors:", report)

    def test_available_checkpoint_report_groups_duplicate_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "models" / "sam2.1_hiera_large.pt"
            second = root / "repo" / "checkpoints" / "sam2.1_hiera_large.pt"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            text = dam4sam_backend._format_available_checkpoints(
                [first.resolve(), second.resolve()]
            )

        self.assertEqual(text, "sam2.1_hiera_large.pt (2 paths)")

    def test_smoke_test_parses_subprocess_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            python_executable = Path(sys.executable)
            fake_completed = mock.Mock(
                returncode=0,
                stdout=(
                    "loading model\n"
                    + dam4sam_backend._SMOKE_TEST_PREFIX
                    + '{"ok": true, "tracked_mask_area": 42}\n'
                ),
                stderr="",
            )

            with mock.patch.object(
                dam4sam_backend.subprocess,
                "run",
                return_value=fake_completed,
            ) as run_mock:
                result = dam4sam_backend.run_dam4sam_smoke_test(
                    repo_path=repo_path,
                    python_executable=python_executable,
                    tracker_name="sam21pp-L",
                )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.payload["tracked_mask_area"], 42)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], str(python_executable.resolve()))

    def test_direct_offline_tracks_with_fake_dam4sam_tracker(self) -> None:
        frames = np.zeros((2, 12, 16, 3), dtype=np.uint8)
        frames[:, 3:7, 4:8, :] = 200
        query_points = np.asarray([[0.0, 5.0, 6.0]], dtype=np.float32)
        query_boxes = np.asarray([[4.0, 3.0, 8.0, 7.0]], dtype=np.float32)

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

        class FakeTorch:
            cuda = FakeCuda()

        class FakeTracker:
            def __init__(self, tracker_name: str) -> None:
                self.tracker_name = tracker_name
                self.checkpoint, self.model_cfg = fake_module.determine_tracker(tracker_name)
                self.frame_index = 0

            def initialize(self, image, init_mask, bbox=None):
                mask = np.zeros((12, 16), dtype=np.uint8)
                mask[3:7, 4:8] = 1
                return {"pred_mask": mask}

            def track(self, image):
                mask = np.zeros((12, 16), dtype=np.uint8)
                mask[3:7, 5:9] = 1
                return {"pred_mask": mask}

        fake_module = mock.Mock()
        fake_module.DAM4SAMTracker = FakeTracker
        fake_module.determine_tracker = lambda tracker_name: ("default.pt", "default.yaml")

        real_import_module = importlib.import_module

        def fake_import_module(name: str):
            if name == "torch":
                return FakeTorch
            if name == "dam4sam_tracker":
                return fake_module
            if name == "PIL.Image":
                from PIL import Image

                return Image
            return real_import_module(name)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = root / "dam4sam"
            sam2_dir = repo_path / "sam2"
            repo_path.mkdir()
            sam2_dir.mkdir()
            (repo_path / "dam4sam_tracker.py").write_text("# tracker marker\n")
            (sam2_dir / "build_sam.py").write_text("# build marker\n")
            checkpoint = root / "sam2.1_hiera_large.pt"
            checkpoint.write_bytes(b"checkpoint")

            with mock.patch.object(dam4sam_backend.importlib, "import_module", side_effect=fake_import_module):
                result = dam4sam_backend.run_dam4sam_offline(
                    frames,
                    query_points,
                    dam4sam_backend.Dam4SamRuntimeSettings(
                        checkpoint_path=checkpoint,
                        repo_path=repo_path,
                        backend_mode="direct",
                        tracker_name="sam21pp-L",
                        representative_point_mode="bbox_center",
                    ),
                    query_boxes_xyxy=query_boxes,
                )

        self.assertEqual(result.masks.shape, (1, 2, 12, 16))
        self.assertTrue(result.visible_mask[0, 0])
        self.assertTrue(result.visible_mask[0, 1])
        self.assertEqual(result.checkpoint_name, "sam2.1_hiera_large.pt")
        self.assertEqual(result.config_name, "sam21pp_hiera_l.yaml")
        self.assertEqual(result.backend_mode, "direct")
        self.assertEqual(result.mask_areas[0, 1], 16.0)
        np.testing.assert_allclose(result.tracks_xy[0, 1], [7.0, 5.0])

    def test_subprocess_offline_loads_output_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = root / "dam4sam"
            sam2_dir = repo_path / "sam2"
            repo_path.mkdir()
            sam2_dir.mkdir()
            (repo_path / "dam4sam_tracker.py").write_text("# tracker marker\n")
            (sam2_dir / "build_sam.py").write_text("# build marker\n")
            checkpoint = root / "sam2.1_hiera_large.pt"
            checkpoint.write_bytes(b"checkpoint")
            python_executable = root / "python.exe"
            python_executable.write_text("fake")

            def fake_run(command, **kwargs):
                output_path = Path(command[command.index("--output-npz") + 1])
                np.savez_compressed(
                    output_path,
                    masks=np.zeros((1, 2, 8, 10), dtype=bool),
                    tracks_xy=np.zeros((1, 2, 2), dtype=np.float32),
                    visible_mask=np.ones((1, 2), dtype=bool),
                    mask_areas=np.asarray([[4.0, 5.0]], dtype=np.float32),
                    mask_bboxes_xyxy=np.zeros((1, 2, 4), dtype=np.float32),
                    mask_bboxes_xywh=np.zeros((1, 2, 4), dtype=np.float32),
                    mask_scores=np.full((1, 2), np.nan, dtype=np.float32),
                    inference_resolution=np.asarray([8, 10], dtype=np.int32),
                    original_resolution=np.asarray([8, 10], dtype=np.int32),
                    checkpoint_name=np.asarray("sam2.1_hiera_large.pt"),
                    tracker_name=np.asarray("sam21pp-L"),
                    config_name=np.asarray("sam21pp_hiera_l.yaml"),
                    representative_point_mode=np.asarray("mask_centroid"),
                    backend_mode=np.asarray("subprocess"),
                )
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(dam4sam_backend.subprocess, "run", side_effect=fake_run):
                result = dam4sam_backend.run_dam4sam_offline(
                    np.zeros((2, 8, 10), dtype=np.uint8),
                    np.asarray([[0.0, 4.0, 5.0]], dtype=np.float32),
                    dam4sam_backend.Dam4SamRuntimeSettings(
                        checkpoint_path=checkpoint,
                        repo_path=repo_path,
                        python_executable=python_executable,
                        backend_mode="subprocess",
                    ),
                    query_boxes_xyxy=np.asarray([[3.0, 3.0, 7.0, 7.0]], dtype=np.float32),
                )

        self.assertEqual(result.backend_mode, "subprocess")
        self.assertEqual(result.visible_mask.shape, (1, 2))
        self.assertEqual(result.mask_areas[0, 1], 5.0)

    def test_representative_point_mode_aliases(self) -> None:
        self.assertEqual(
            dam4sam_backend._normalize_representative_point_mode("bboxcenter"),
            "bbox_center",
        )
        self.assertEqual(
            dam4sam_backend._normalize_representative_point_mode("peak"),
            "peak_in_mask",
        )

    def test_check_dam4sam_env_script_is_importable(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_dam4sam_env.py"
        spec = importlib.util.spec_from_file_location(
            "_trackleed_check_dam4sam_env_import_test",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
        finally:
            sys.modules.pop("_trackleed_check_dam4sam_env_import_test", None)

    def test_run_dam4sam_subprocess_script_is_importable(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_dam4sam_subprocess.py"
        spec = importlib.util.spec_from_file_location(
            "_trackleed_run_dam4sam_subprocess_import_test",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
        finally:
            sys.modules.pop("_trackleed_run_dam4sam_subprocess_import_test", None)


if __name__ == "__main__":
    unittest.main()
