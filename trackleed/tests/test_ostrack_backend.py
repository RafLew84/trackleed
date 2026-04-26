from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

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

    def test_initial_bbox_probe_evaluates_padding_candidates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_path = root / "ostrack"
            repo_path.mkdir()
            checkpoint = root / "OSTrack_ep0300.pth.tar"
            checkpoint.write_bytes(b"checkpoint")

            def fake_probe_subprocess(**kwargs):
                samples = kwargs["samples"]
                paddings = kwargs["candidate_paddings_px"]
                frame_count = int(samples[0].frames.shape[0])
                predicted = np.full(
                    (len(samples), len(paddings), frame_count, 4),
                    np.nan,
                    dtype=np.float32,
                )
                for sample_index, sample in enumerate(samples):
                    for padding_index, padding_px in enumerate(paddings):
                        for frame_index in range(frame_count):
                            predicted[sample_index, padding_index, frame_index] = (
                                ostrack_backend._expand_bbox_xywh(
                                    sample.gt_bboxes_xywh[frame_index],
                                    padding_px=int(padding_px),
                                    image_shape=sample.frames[frame_index].shape[:2],
                                )
                            )
                return predicted

            with mock.patch.object(
                ostrack_backend,
                "_run_ostrack_bbox_probe_subprocess",
                side_effect=fake_probe_subprocess,
            ):
                result = ostrack_backend.probe_ostrack_initial_bbox_padding(
                    repo_path=repo_path,
                    checkpoint_path=checkpoint,
                    python_executable=sys.executable,
                    candidate_paddings_px=[0, 4, 8],
                )

        self.assertEqual(result.recommended_padding_px, 0)
        self.assertEqual(result.candidate_paddings_px, [0, 4, 8])
        self.assertGreaterEqual(len(result.samples), 5)
        self.assertGreater(result.aggregate_mean_scores[0], result.aggregate_mean_scores[4])
        report = ostrack_backend.format_ostrack_initial_bbox_probe_report(result)
        self.assertIn("OSTrack initial bbox padding probe", report)
        self.assertIn("Recommended padding px: 0", report)
        self.assertIn("low_contrast", report)

    def test_run_ostrack_offline_subprocess_reads_output_contract(self) -> None:
        frames = np.zeros((3, 16, 20, 3), dtype=np.uint8)
        query_points = np.asarray([[0.0, 8.0, 10.0]], dtype=np.float32)
        query_boxes = np.asarray([[6.0, 5.0, 14.0, 11.0]], dtype=np.float32)

        def fake_subprocess_run(command, **kwargs):
            output_path = Path(command[command.index("--output-npz") + 1])
            np.savez_compressed(
                output_path,
                tracks_xy=np.asarray([[[10.0, 8.0], [11.0, 8.5], [12.0, 9.0]]], dtype=np.float32),
                visible_mask=np.asarray([[True, True, True]], dtype=bool),
                tracked_bboxes_xywh=np.asarray(
                    [[[6.0, 5.0, 8.0, 6.0], [7.0, 5.5, 8.0, 6.0], [8.0, 6.0, 8.0, 6.0]]],
                    dtype=np.float32,
                ),
                tracked_bboxes_xyxy=np.asarray(
                    [[[6.0, 5.0, 14.0, 11.0], [7.0, 5.5, 15.0, 11.5], [8.0, 6.0, 16.0, 12.0]]],
                    dtype=np.float32,
                ),
                bbox_scores=np.ones((1, 3), dtype=np.float32),
                inference_resolution=np.asarray([16, 20], dtype=np.int32),
                original_resolution=np.asarray([16, 20], dtype=np.int32),
                checkpoint_name=np.asarray("OSTrack_ep0300.pth.tar"),
                model_variant=np.asarray("vitb_256_mae_ce_32x4_ep300"),
                representative_point_mode=np.asarray("bbox_center"),
                initial_bbox_padding_px=np.asarray(0.0, dtype=np.float32),
            )
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(
            ostrack_backend.subprocess,
            "run",
            side_effect=fake_subprocess_run,
        ) as run_mock:
            result = ostrack_backend.run_ostrack_offline(
                frames,
                query_points,
                ostrack_backend.OstrackRuntimeSettings(
                    checkpoint_path="OSTrack_ep0300.pth.tar",
                    repo_path="C:/ostrack",
                    python_executable=sys.executable,
                    backend_mode="subprocess",
                    model_variant="vitb_256_mae_ce_32x4_ep300",
                    representative_point_mode="bbox_center",
                    initial_bbox_padding_px=0.0,
                    subprocess_timeout_sec=30.0,
                ),
                query_boxes_xyxy=query_boxes,
            )

        self.assertEqual(result.tracks_xy.shape, (1, 3, 2))
        self.assertEqual(result.tracked_bboxes_xyxy.shape, (1, 3, 4))
        self.assertTrue(np.all(result.visible_mask))
        self.assertEqual(result.checkpoint_name, "OSTrack_ep0300.pth.tar")
        command = run_mock.call_args.args[0]
        run_kwargs = run_mock.call_args.kwargs
        self.assertIn("run_ostrack_subprocess.py", " ".join(command))
        self.assertIn("--model-variant", command)
        self.assertIn("vitb_256_mae_ce_32x4_ep300", command)
        self.assertIn("env", run_kwargs)
        self.assertIn("timeout", run_kwargs)
        self.assertEqual(run_kwargs["timeout"], 30.0)

    def test_run_ostrack_subprocess_reports_runner_failure_context(self) -> None:
        frames = np.zeros((2, 16, 20, 3), dtype=np.uint8)
        query_points = np.asarray([[0.0, 8.0, 10.0]], dtype=np.float32)
        settings = ostrack_backend.OstrackRuntimeSettings(
            checkpoint_path="OSTrack_ep0300.pth.tar",
            repo_path="C:/repo/ostrack",
            python_executable=sys.executable,
            backend_mode="subprocess",
            model_variant="vitb_256_mae_ce_32x4_ep300",
        )

        def fake_run(command, **kwargs):
            return mock.Mock(
                returncode=7,
                stdout="runner stdout",
                stderr="runner stderr",
            )

        with mock.patch.object(ostrack_backend.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as error_context:
                ostrack_backend.run_ostrack_offline(frames, query_points, settings)

        message = str(error_context.exception)
        self.assertIn("OSTrack subprocess inference failed", message)
        self.assertIn("run_ostrack_subprocess.py", message)
        self.assertIn('"returncode": 7', message)
        self.assertIn("runner stdout", message)
        self.assertIn("runner stderr", message)

    def test_run_ostrack_subprocess_reports_timeout_context(self) -> None:
        frames = np.zeros((2, 16, 20, 3), dtype=np.uint8)
        query_points = np.asarray([[0.0, 8.0, 10.0]], dtype=np.float32)
        settings = ostrack_backend.OstrackRuntimeSettings(
            checkpoint_path="OSTrack_ep0300.pth.tar",
            repo_path="C:/repo/ostrack",
            python_executable=sys.executable,
            backend_mode="subprocess",
            model_variant="vitb_256_mae_ce_32x4_ep300",
            subprocess_timeout_sec=12.5,
        )

        def fake_run(command, **kwargs):
            raise ostrack_backend.subprocess.TimeoutExpired(
                command,
                timeout=kwargs["timeout"],
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

        with mock.patch.object(ostrack_backend.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as error_context:
                ostrack_backend.run_ostrack_offline(frames, query_points, settings)

        message = str(error_context.exception)
        self.assertIn("OSTrack subprocess inference timed out", message)
        self.assertIn("run_ostrack_subprocess.py", message)
        self.assertIn('"timeout_sec": 12.5', message)
        self.assertIn("partial stdout", message)
        self.assertIn("partial stderr", message)

    def test_run_ostrack_subprocess_reports_missing_output_context(self) -> None:
        frames = np.zeros((2, 16, 20, 3), dtype=np.uint8)
        query_points = np.asarray([[0.0, 8.0, 10.0]], dtype=np.float32)
        settings = ostrack_backend.OstrackRuntimeSettings(
            checkpoint_path="OSTrack_ep0300.pth.tar",
            repo_path="C:/repo/ostrack",
            python_executable=sys.executable,
            backend_mode="subprocess",
            model_variant="vitb_256_mae_ce_32x4_ep300",
        )

        def fake_run(command, **kwargs):
            return mock.Mock(returncode=0, stdout="ok stdout", stderr="ok stderr")

        with mock.patch.object(ostrack_backend.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as error_context:
                ostrack_backend.run_ostrack_offline(frames, query_points, settings)

        message = str(error_context.exception)
        self.assertIn("OSTrack subprocess did not create output", message)
        self.assertIn("run_ostrack_subprocess.py", message)
        self.assertIn('"expected_output":', message)
        self.assertIn("ok stdout", message)
        self.assertIn("ok stderr", message)

    def test_backend_smoke_test_formats_result_with_mocked_backend(self) -> None:
        fake_result = ostrack_backend.OstrackOfflineResult(
            tracks_xy=np.asarray([[[10.0, 8.0], [11.0, 9.0]]], dtype=np.float32),
            visible_mask=np.asarray([[True, True]], dtype=bool),
            tracked_bboxes_xywh=np.asarray(
                [[[6.0, 5.0, 8.0, 6.0], [7.0, 6.0, 8.0, 6.0]]],
                dtype=np.float32,
            ),
            tracked_bboxes_xyxy=np.asarray(
                [[[6.0, 5.0, 14.0, 11.0], [7.0, 6.0, 15.0, 12.0]]],
                dtype=np.float32,
            ),
            bbox_scores=np.ones((1, 2), dtype=np.float32),
            inference_resolution=(16, 20),
            original_resolution=(16, 20),
            checkpoint_name="OSTrack_ep0300.pth.tar",
            model_variant="vitb_256_mae_ce_32x4_ep300",
            representative_point_mode="bbox_center",
            initial_bbox_padding_px=0.0,
        )

        with mock.patch.object(
            ostrack_backend,
            "run_ostrack_offline",
            return_value=fake_result,
        ):
            result = ostrack_backend.run_ostrack_backend_smoke_test(
                repo_path="C:/ostrack",
                checkpoint_path="C:/models/OSTrack_ep0300.pth.tar",
                python_executable=sys.executable,
            )

        self.assertEqual(result.visible_count, 2)
        report = ostrack_backend.format_ostrack_backend_smoke_test_report(result)
        self.assertIn("OSTrack backend smoke test", report)
        self.assertIn("visible count: 2", report)

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

    def test_run_ostrack_subprocess_script_imports_without_pyqt(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_ostrack_subprocess.py"
        original_pyqt_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in ("PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets")
        }
        try:
            for module_name in original_pyqt_modules:
                sys.modules.pop(module_name, None)

            spec = importlib.util.spec_from_file_location(
                "_trackleed_run_ostrack_subprocess_import_test",
                script_path,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            self.assertTrue(callable(module.main))
            self.assertTrue(callable(module._bootstrap_windows_conda_dlls))
            self.assertNotIn("PyQt6", sys.modules)
        finally:
            sys.modules.pop("_trackleed_run_ostrack_subprocess_import_test", None)
            for module_name, original_module in original_pyqt_modules.items():
                if original_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = original_module


if __name__ == "__main__":
    unittest.main()
