from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
