from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from trackleed.analysis import samurai_backend


class SamuraiBackendTests(unittest.TestCase):
    def _make_repo(self, root: Path) -> Path:
        repo_path = root / "samurai"
        (repo_path / "scripts").mkdir(parents=True)
        (repo_path / "scripts" / "demo.py").write_text("# demo marker\n", encoding="utf-8")
        (repo_path / "sam2" / "sam2").mkdir(parents=True)
        (repo_path / "sam2" / "sam2" / "build_sam.py").write_text(
            "# build marker\n",
            encoding="utf-8",
        )
        (repo_path / "sam2" / "sam2" / "configs" / "samurai").mkdir(parents=True)
        (repo_path / "sam2" / "sam2" / "configs" / "samurai" / "sam2.1_hiera_b+.yaml").write_text(
            "model: {}\n",
            encoding="utf-8",
        )
        return repo_path

    def test_checkpoint_selection_prefers_variant_checkpoint_in_models_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_dir = root / "tracking_models" / "samurai"
            repo_path = self._make_repo(root)
            (repo_path / "sam2" / "checkpoints").mkdir(parents=True)
            models_dir.mkdir(parents=True)

            repo_checkpoint = repo_path / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
            models_checkpoint = models_dir / "sam2.1_hiera_base_plus.pt"
            repo_checkpoint.write_bytes(b"repo")
            models_checkpoint.write_bytes(b"models")

            discovered = samurai_backend.discover_samurai_checkpoints(
                models_dir=models_dir,
                repo_path=repo_path,
            )
            selected = samurai_backend.select_default_samurai_checkpoint(
                discovered,
                "base_plus",
                models_dir=models_dir,
            )

        self.assertEqual(selected, models_checkpoint.resolve())

    def test_parse_prefixed_json_ignores_noisy_stdout(self) -> None:
        stdout = (
            "warning line\n"
            + samurai_backend._IMPORT_PROBE_PREFIX
            + '{"torch_importable": true, "numpy_version": "1.26.4"}\n'
        )

        payload = samurai_backend._parse_prefixed_json_stdout(
            stdout,
            samurai_backend._IMPORT_PROBE_PREFIX,
        )

        self.assertTrue(payload["torch_importable"])
        self.assertEqual(payload["numpy_version"], "1.26.4")

    def test_probe_environment_reports_ready_contract_with_mocked_import_probe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = self._make_repo(root)
            models_dir = root / "tracking_models" / "samurai"
            models_dir.mkdir(parents=True)
            checkpoint = models_dir / "sam2.1_hiera_base_plus.pt"
            checkpoint.write_bytes(b"checkpoint")

            probe_payload = {
                "torch_importable": True,
                "torch_version": "2.5.1+cu121",
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
                "jpeg4py_importable": True,
                "lmdb_importable": True,
                "loguru_importable": True,
                "sam2_importable": True,
                "sam2_module_path": str(repo_path / "sam2" / "sam2" / "__init__.py"),
                "sam2_build_sam_importable": True,
                "sam2_cuda_extension_importable": False,
                "sam2_cuda_extension_error": "ModuleNotFoundError: No module named 'sam2._C'",
            }
            probe_result = samurai_backend.SamuraiImportProbeResult(
                command=[sys.executable, "-c", "probe"],
                returncode=0,
                stdout="",
                stderr="",
                payload=probe_payload,
            )

            with mock.patch.object(
                samurai_backend,
                "_run_samurai_import_probe",
                return_value=probe_result,
            ):
                status = samurai_backend.probe_samurai_environment(
                    models_dir=models_dir,
                    repo_path=repo_path,
                    python_executable=sys.executable,
                    model_variant="base_plus",
                )

        self.assertTrue(status.is_ready)
        self.assertEqual(status.selected_checkpoint, checkpoint.resolve())
        self.assertEqual(status.config_identifier, "configs/samurai/sam2.1_hiera_b+.yaml")
        report = samurai_backend.format_samurai_environment_report(status)
        self.assertIn("SAMURAI environment diagnostics", report)
        self.assertIn("Ready: yes", report)
        self.assertIn("SAM2 CUDA extension: missing", report)
        self.assertNotIn("Errors:", report)

    def test_available_checkpoint_report_groups_duplicate_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "models" / "sam2.1_hiera_base_plus.pt"
            second = root / "repo" / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            text = samurai_backend._format_available_checkpoints(
                [first.resolve(), second.resolve()]
            )

        self.assertEqual(text, "sam2.1_hiera_base_plus.pt (2 paths)")

    def test_smoke_test_parses_subprocess_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = self._make_repo(root)
            checkpoint = root / "sam2.1_hiera_base_plus.pt"
            checkpoint.write_bytes(b"checkpoint")
            python_executable = Path(sys.executable)
            fake_completed = mock.Mock(
                returncode=0,
                stdout=(
                    "loading model\n"
                    + samurai_backend._SMOKE_TEST_PREFIX
                    + '{"ok": true, "propagated_frame_count": 2, "last_mask_area": 42}\n'
                ),
                stderr="",
            )

            with mock.patch.object(
                samurai_backend.subprocess,
                "run",
                return_value=fake_completed,
            ) as run_mock:
                result = samurai_backend.run_samurai_smoke_test(
                    repo_path=repo_path,
                    checkpoint_path=checkpoint,
                    python_executable=python_executable,
                    model_variant="base_plus",
                )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.payload["last_mask_area"], 42)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], str(python_executable.resolve()))

    def test_check_samurai_env_script_is_importable(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_samurai_env.py"
        spec = importlib.util.spec_from_file_location(
            "_trackleed_check_samurai_env_import_test",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
        finally:
            sys.modules.pop("_trackleed_check_samurai_env_import_test", None)

    def test_run_samurai_offline_subprocess_reads_npz_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = self._make_repo(root)
            models_dir = root / "tracking_models" / "samurai"
            models_dir.mkdir(parents=True)
            checkpoint = models_dir / "sam2.1_hiera_base_plus.pt"
            checkpoint.write_bytes(b"checkpoint")
            frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
            query_points = np.asarray([[0.0, 4.0, 5.0]], dtype=np.float32)
            query_boxes_xywh = np.asarray([[3.0, 2.0, 4.0, 4.0]], dtype=np.float32)

            def fake_subprocess_run(command, **kwargs):
                input_path = Path(command[command.index("--input-npz") + 1])
                with np.load(input_path, allow_pickle=False) as input_data:
                    self.assertFalse(bool(input_data["has_query_boxes"].tolist()))
                    self.assertTrue(bool(input_data["has_query_boxes_xywh"].tolist()))
                    self.assertEqual(
                        np.asarray(input_data["query_boxes_xywh"], dtype=np.float32).tolist(),
                        query_boxes_xywh.tolist(),
                    )
                output_path = Path(command[command.index("--output-npz") + 1])
                np.savez_compressed(
                    output_path,
                    masks=np.ones((1, 2, 8, 10), dtype=bool),
                    tracks_xy=np.asarray([[[5.0, 4.0], [6.0, 4.0]]], dtype=np.float32),
                    visible_mask=np.ones((1, 2), dtype=bool),
                    mask_areas=np.asarray([[12.0, 13.0]], dtype=np.float32),
                    mask_bboxes_xyxy=np.asarray(
                        [[[3.0, 2.0, 7.0, 6.0], [4.0, 2.0, 8.0, 6.0]]],
                        dtype=np.float32,
                    ),
                    mask_bboxes_xywh=np.asarray(
                        [[[3.0, 2.0, 4.0, 4.0], [4.0, 2.0, 4.0, 4.0]]],
                        dtype=np.float32,
                    ),
                    mask_scores=np.ones((1, 2), dtype=np.float32),
                    inference_resolution=np.asarray([8, 10], dtype=np.int32),
                    original_resolution=np.asarray([8, 10], dtype=np.int32),
                    checkpoint_name=np.asarray("sam2.1_hiera_base_plus.pt"),
                    model_variant=np.asarray("sam2.1_hiera_base_plus"),
                    config_name=np.asarray("sam2.1_hiera_b+.yaml"),
                    representative_point_mode=np.asarray("mask_centroid"),
                    backend_mode=np.asarray("subprocess"),
                )
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                samurai_backend.subprocess,
                "run",
                side_effect=fake_subprocess_run,
            ) as run_mock:
                result = samurai_backend.run_samurai_offline(
                    frames,
                    query_points,
                    samurai_backend.SamuraiRuntimeSettings(
                        checkpoint_path=checkpoint,
                        repo_path=repo_path,
                        python_executable=sys.executable,
                        backend_mode="subprocess",
                        model_variant="base_plus",
                    ),
                    query_boxes_xywh=query_boxes_xywh,
                )

        self.assertEqual(result.backend_mode, "subprocess")
        self.assertEqual(result.model_variant, "sam2.1_hiera_base_plus")
        self.assertEqual(result.masks.shape, (1, 2, 8, 10))
        np.testing.assert_allclose(result.tracks_xy[0, 1], [6.0, 4.0])
        command = run_mock.call_args.args[0]
        self.assertIn("--model-variant", command)
        self.assertIn("sam2.1_hiera_base_plus", command)

    def test_run_samurai_subprocess_script_is_importable(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_samurai_subprocess.py"
        spec = importlib.util.spec_from_file_location(
            "_trackleed_run_samurai_subprocess_import_test",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
        finally:
            sys.modules.pop("_trackleed_run_samurai_subprocess_import_test", None)


if __name__ == "__main__":
    unittest.main()
