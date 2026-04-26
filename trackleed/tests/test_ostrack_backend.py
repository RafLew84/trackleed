from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from trackleed.analysis import ostrack_backend


class OstrackBackendTests(unittest.TestCase):
    def test_checkpoint_discovery_prefers_variant_named_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            generic_checkpoint = models_dir / "OSTrack_ep0300.pth.tar"
            preferred_checkpoint = models_dir / "vitb_256_mae_ce_32x4_ep300.pth.tar"
            generic_checkpoint.write_bytes(b"generic")
            preferred_checkpoint.write_bytes(b"preferred")

            discovered = ostrack_backend.discover_ostrack_checkpoints(models_dir)
            selected = ostrack_backend.select_default_ostrack_checkpoint(
                discovered,
                "vitb_256_mae_ce_32x4_ep300",
            )

        self.assertEqual(selected, preferred_checkpoint.resolve())

    def test_import_probe_parser_ignores_noisy_stdout(self) -> None:
        stdout = (
            "WARNING: You are using tensorboardX instead\n"
            + ostrack_backend._IMPORT_PROBE_PREFIX
            + '{"torch_importable": true, "torch_version": "1.13.1+cu117"}\n'
        )

        payload = ostrack_backend._parse_import_probe_stdout(stdout)

        self.assertTrue(payload["torch_importable"])
        self.assertEqual(payload["torch_version"], "1.13.1+cu117")

    def test_probe_environment_reports_ready_contract_with_mocked_import_probe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = root / "ostrack"
            config_dir = repo_path / "experiments" / "ostrack"
            tracker_dir = repo_path / "lib" / "test" / "tracker"
            upstream_checkpoint_dir = (
                repo_path
                / "output"
                / "checkpoints"
                / "train"
                / "ostrack"
                / "vitb_256_mae_ce_32x4_ep300"
            )
            models_dir = root / "tracking_models" / "ostrack"
            config_dir.mkdir(parents=True)
            tracker_dir.mkdir(parents=True)
            upstream_checkpoint_dir.mkdir(parents=True)
            models_dir.mkdir(parents=True)
            (config_dir / "vitb_256_mae_ce_32x4_ep300.yaml").write_text("TEST: {}\n")
            (tracker_dir / "ostrack.py").write_text("# tracker marker\n")
            upstream_checkpoint = upstream_checkpoint_dir / "OSTrack_ep0300.pth.tar"
            upstream_checkpoint.write_bytes(b"checkpoint")

            probe_payload = {
                "torch_importable": True,
                "torch_version": "1.13.1+cu117",
                "torch_cuda_version": "11.7",
                "cuda_available": True,
                "torchvision_importable": True,
                "cv2_importable": True,
                "numpy_importable": True,
                "yaml_importable": True,
                "easydict_importable": True,
                "timm_importable": True,
                "lmdb_importable": True,
                "jpeg4py_importable": True,
                "ostrack_parameter_importable": True,
                "ostrack_tracker_importable": True,
                "ostrack_parameter_module_path": str(repo_path / "lib" / "test" / "parameter" / "ostrack.py"),
                "ostrack_tracker_module_path": str(tracker_dir / "ostrack.py"),
            }
            probe_result = ostrack_backend.OstrackImportProbeResult(
                command=[sys.executable, "-c", "probe"],
                returncode=0,
                stdout="",
                stderr="",
                payload=probe_payload,
            )

            with mock.patch.object(
                ostrack_backend,
                "_run_ostrack_import_probe",
                return_value=probe_result,
            ):
                status = ostrack_backend.probe_ostrack_environment(
                    models_dir=models_dir,
                    repo_path=repo_path,
                    python_executable=sys.executable,
                )

        self.assertTrue(status.is_ready)
        self.assertEqual(status.selected_checkpoint, upstream_checkpoint.resolve())
        report = ostrack_backend.format_ostrack_environment_report(status)
        self.assertIn("OSTrack environment diagnostics", report)
        self.assertIn("Ready: yes", report)

    def test_profile_smoke_test_parses_upstream_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            profile_script = repo_path / "tracking" / "profile_model.py"
            profile_script.parent.mkdir(parents=True)
            profile_script.write_text("# profile marker\n")

            fake_completed = mock.Mock(
                returncode=0,
                stdout=(
                    "overall macs is  21.499G\n"
                    "overall params is  92.082M\n"
                    "testing speed ...\n"
                    "The average overall latency is 12.83 ms\n"
                    "FPS is 77.97 fps\n"
                ),
                stderr="",
            )

            with mock.patch.object(
                ostrack_backend.subprocess,
                "run",
                return_value=fake_completed,
            ) as run_mock:
                result = ostrack_backend.run_ostrack_profile_smoke_test(
                    repo_path=repo_path,
                    python_executable=sys.executable,
                    config_name="vitb_256_mae_ce_32x4_ep300",
                )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.macs, "21.499G")
        self.assertEqual(result.params, "92.082M")
        self.assertEqual(result.latency_ms, 12.83)
        self.assertEqual(result.fps, 77.97)
        command = run_mock.call_args.args[0]
        self.assertIn(str(profile_script), command)
        self.assertIn("vitb_256_mae_ce_32x4_ep300", command)

    def test_check_ostrack_env_script_is_importable(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_ostrack_env.py"
        spec = importlib.util.spec_from_file_location(
            "_trackleed_check_ostrack_env_import_test",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
        finally:
            sys.modules.pop("_trackleed_check_ostrack_env_import_test", None)


if __name__ == "__main__":
    unittest.main()
