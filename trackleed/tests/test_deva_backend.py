from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from trackleed.analysis import deva_backend


class DevaBackendTests(unittest.TestCase):
    def test_initial_label_mask_prefers_explicit_masks_and_first_label_wins(self) -> None:
        masks = np.zeros((2, 6, 7), dtype=bool)
        masks[0, 1:4, 1:4] = True
        masks[1, 2:5, 2:6] = True

        label_mask, mode = deva_backend._build_deva_initial_label_mask(
            frame_shape=(6, 7),
            query_points_tyx=None,
            query_boxes_xyxy=None,
            query_initial_masks=masks,
            object_ids=np.asarray([11, 22], dtype=np.int32),
            initial_bbox_padding_px=0.0,
            initial_point_radius_px=3.0,
        )

        self.assertEqual(mode, "initial_masks")
        self.assertEqual(label_mask[1, 1], 1)
        self.assertEqual(label_mask[2, 2], 1)
        self.assertEqual(label_mask[4, 5], 2)
        self.assertEqual(label_mask[0, 0], 0)

    def test_initial_label_mask_from_boxes_uses_padding_and_clipping(self) -> None:
        boxes = np.asarray(
            [
                [1.2, 1.2, 3.1, 3.1],
                [5.0, 4.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )

        label_mask, mode = deva_backend._build_deva_initial_label_mask(
            frame_shape=(6, 7),
            query_points_tyx=None,
            query_boxes_xyxy=boxes,
            query_initial_masks=None,
            object_ids=np.asarray([1, 2], dtype=np.int32),
            initial_bbox_padding_px=1.0,
            initial_point_radius_px=3.0,
        )

        self.assertEqual(mode, "bbox")
        self.assertEqual(label_mask[0, 0], 1)
        self.assertEqual(label_mask[3, 3], 1)
        self.assertEqual(label_mask[5, 6], 2)

    def test_initial_label_mask_from_points_rejects_nonzero_seed_frames(self) -> None:
        query_points = np.asarray([[1.0, 3.0, 3.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "local frame 0"):
            deva_backend._build_deva_initial_label_mask(
                frame_shape=(8, 8),
                query_points_tyx=query_points,
                query_boxes_xyxy=None,
                query_initial_masks=None,
                object_ids=np.asarray([1], dtype=np.int32),
                initial_bbox_padding_px=0.0,
                initial_point_radius_px=2.0,
            )

    def test_result_from_class_masks_exposes_masks_points_and_scores(self) -> None:
        frames = np.zeros((2, 8, 9, 3), dtype=np.uint8)
        frames[1, 4, 6] = [255, 255, 255]
        class_masks = np.zeros((2, 8, 9), dtype=np.uint8)
        class_masks[0, 1:3, 2:4] = 1
        class_masks[1, 4:6, 5:8] = 1

        result = deva_backend._build_deva_result_from_class_masks(
            class_masks,
            frames,
            checkpoint_name="DEVA-propagation.pth",
            representative_point_mode="peak_in_mask",
            initial_mask_mode="bbox",
            inference_size=480,
            backend_mode="direct",
        )

        self.assertEqual(result.masks.shape, (1, 2, 8, 9))
        self.assertTrue(result.visible_mask[0, 0])
        self.assertTrue(result.visible_mask[0, 1])
        self.assertEqual(result.mask_areas[0, 0], 4.0)
        np.testing.assert_allclose(result.mask_bboxes_xyxy[0, 1], [5.0, 4.0, 8.0, 6.0])
        np.testing.assert_allclose(result.tracks_xy[0, 1], [6.0, 4.0])
        self.assertEqual(result.mask_scores[0, 1], 1.0)

    def test_run_deva_subprocess_reads_runner_output_contract(self) -> None:
        frames = np.zeros((2, 5, 6, 3), dtype=np.uint8)
        query_points = np.asarray([[0.0, 2.0, 3.0]], dtype=np.float32)
        settings = deva_backend.DevaRuntimeSettings(
            checkpoint_path="DEVA-propagation.pth",
            repo_path="C:/repo/deva",
            python_executable=sys.executable,
            backend_mode="subprocess",
            inference_size=320,
            chunk_size=2,
        )

        def fake_run(command, **kwargs):
            output_path = Path(command[command.index("--output-npz") + 1])
            np.savez_compressed(
                output_path,
                masks=np.zeros((1, 2, 5, 6), dtype=bool),
                tracks_xy=np.zeros((1, 2, 2), dtype=np.float32),
                visible_mask=np.ones((1, 2), dtype=bool),
                mask_areas=np.ones((1, 2), dtype=np.float32),
                mask_bboxes_xyxy=np.zeros((1, 2, 4), dtype=np.float32),
                mask_scores=np.ones((1, 2), dtype=np.float32),
                inference_size=np.asarray(320, dtype=np.int32),
                original_resolution=np.asarray([5, 6], dtype=np.int32),
                checkpoint_name=np.asarray("DEVA-propagation.pth"),
                representative_point_mode=np.asarray("mask_centroid"),
                initial_mask_mode=np.asarray("point_disk"),
                backend_mode=np.asarray("subprocess"),
            )
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(deva_backend.subprocess, "run", side_effect=fake_run) as run_mock:
            result = deva_backend.run_deva_offline(frames, query_points, settings)

        self.assertEqual(result.backend_mode, "subprocess")
        self.assertEqual(result.inference_size, 320)
        self.assertEqual(result.initial_mask_mode, "point_disk")
        command = run_mock.call_args.args[0]
        run_kwargs = run_mock.call_args.kwargs
        self.assertIn("--inference-size", command)
        self.assertIn("320", command)
        self.assertIn("env", run_kwargs)
        self.assertIn("timeout", run_kwargs)
        self.assertEqual(run_kwargs["timeout"], settings.subprocess_timeout_sec)

    def test_run_deva_subprocess_reports_runner_failure_context(self) -> None:
        frames = np.zeros((2, 5, 6, 3), dtype=np.uint8)
        query_points = np.asarray([[0.0, 2.0, 3.0]], dtype=np.float32)
        settings = deva_backend.DevaRuntimeSettings(
            checkpoint_path="DEVA-propagation.pth",
            repo_path="C:/repo/deva",
            python_executable=sys.executable,
            backend_mode="subprocess",
        )

        def fake_run(command, **kwargs):
            return mock.Mock(
                returncode=3,
                stdout="runner stdout",
                stderr="runner stderr",
            )

        with mock.patch.object(deva_backend.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as error_context:
                deva_backend.run_deva_offline(frames, query_points, settings)

        message = str(error_context.exception)
        self.assertIn("DEVA subprocess inference failed", message)
        self.assertIn("run_deva_subprocess.py", message)
        self.assertIn('"returncode": 3', message)
        self.assertIn("runner stdout", message)
        self.assertIn("runner stderr", message)

    def test_subprocess_env_builder_returns_process_environment(self) -> None:
        env = deva_backend._build_windows_conda_subprocess_env(Path(sys.executable))

        self.assertIsInstance(env, dict)
        self.assertIn("PATH", env)

    def test_checkpoint_discovery_uses_tracking_models_deva_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_dir = root / "tracking_models" / "deva"
            repo_saves_dir = root / "deva" / "saves"
            models_dir.mkdir(parents=True)
            repo_saves_dir.mkdir(parents=True)
            preferred_checkpoint = models_dir / "DEVA-propagation.pth"
            ignored_repo_checkpoint = repo_saves_dir / "DEVA-propagation.pth"
            preferred_checkpoint.write_bytes(b"models")
            ignored_repo_checkpoint.write_bytes(b"repo")

            discovered = deva_backend.discover_deva_checkpoints(
                models_dir=models_dir,
                repo_path=root / "deva",
            )
            resolved = deva_backend._resolve_checkpoint_path(
                "DEVA-propagation.pth",
                discovered,
                models_dir,
                root / "deva",
            )

        self.assertEqual(discovered, [preferred_checkpoint.resolve()])
        self.assertEqual(resolved, preferred_checkpoint.resolve())

    def test_run_deva_subprocess_imports_without_pyqt(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_deva_subprocess.py"
        original_pyqt_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in ("PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets")
        }
        try:
            for module_name in original_pyqt_modules:
                sys.modules.pop(module_name, None)

            spec = importlib.util.spec_from_file_location(
                "_trackleed_run_deva_subprocess_import_test",
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
            sys.modules.pop("_trackleed_run_deva_subprocess_import_test", None)
            for module_name, original_module in original_pyqt_modules.items():
                if original_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = original_module

    def test_check_deva_env_bootstraps_trackleed_root_when_run_as_script(self) -> None:
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_deva_env.py"
        repo_root = script_path.parents[1]
        original_sys_path = list(sys.path)
        try:
            sys.path[:] = [path for path in sys.path if path != str(repo_root)]
            spec = importlib.util.spec_from_file_location(
                "_trackleed_check_deva_env_path_test",
                script_path,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            self.assertIn(str(repo_root), sys.path)
        finally:
            sys.modules.pop("_trackleed_check_deva_env_path_test", None)
            sys.path[:] = original_sys_path

    def test_environment_probe_report_is_available_for_registry(self) -> None:
        status = deva_backend.probe_deva_environment(
            models_dir=Path("missing-deva-models"),
            repo_path=Path("missing-deva-repo"),
        )
        report = deva_backend.format_deva_environment_report(status)

        self.assertIn("DEVA environment diagnostics", report)
        self.assertFalse(status.is_ready)
        self.assertTrue(callable(deva_backend.probe_deva_environment))
        self.assertTrue(callable(deva_backend.format_deva_environment_report))


if __name__ == "__main__":
    unittest.main()
