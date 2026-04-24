from __future__ import annotations

import ast
from pathlib import Path
import unittest

from trackleed.tests.test_tracker_registry import TrackerRegistryImportMixin


TRACKLEED_ROOT = Path(__file__).resolve().parents[1]
MAIN_WINDOW_PATH = TRACKLEED_ROOT / "gui" / "main_window.py"
TRACKING_DIALOG_PATH = TRACKLEED_ROOT / "gui" / "tracking_dialog.py"


def _load_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _extract_module_level_literal_dict(path: Path, variable_name: str) -> dict[str, str]:
    module = _load_module_ast(path)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise TypeError(f"{variable_name} in {path} is not a dict literal.")
                    return value
    raise KeyError(f"Could not find dict literal `{variable_name}` in {path}.")


def _extract_class_method_names(path: Path, class_name: str) -> set[str]:
    module = _load_module_ast(path)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise KeyError(f"Could not find class `{class_name}` in {path}.")


class TrackerRefactorReadinessTests(TrackerRegistryImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.registry = self.import_tracker_registry()

    def test_main_window_exposes_collect_methods_for_all_registered_trackers(self) -> None:
        main_window_methods = _extract_class_method_names(MAIN_WINDOW_PATH, "MainWindow")
        self.assertIn("_build_registered_tracking_worker", main_window_methods)
        self.assertIn("_ensure_registered_tracker_runtime_ready", main_window_methods)
        self.assertIn("_sync_registered_tracker_settings_from_widgets", main_window_methods)

        for spec in self.registry.iter_tracker_specs():
            self.assertIn(
                spec.collect_params_method_name,
                main_window_methods,
                msg=(
                    f"Tracker `{spec.tracker_id}` still requires manual collect params wiring "
                    f"outside MainWindow method `{spec.collect_params_method_name}`."
                ),
            )

    def test_tracking_dialog_label_maps_cover_all_registered_sources(self) -> None:
        method_labels = _extract_module_level_literal_dict(
            TRACKING_DIALOG_PATH,
            "TRACKER_METHOD_LABEL_BY_SOURCE",
        )
        manual_method_labels = _extract_module_level_literal_dict(
            TRACKING_DIALOG_PATH,
            "MANUAL_AUGMENTED_METHOD_LABEL_BY_SOURCE",
        )
        registered_sources = {
            source_name
            for spec in self.registry.iter_tracker_specs()
            for source_name in spec.source_names
        }

        self.assertTrue(
            registered_sources.issubset(method_labels.keys()),
            msg=(
                "TrackingDialog still needs explicit source label wiring for some registry sources: "
                f"{sorted(registered_sources - set(method_labels))}"
            ),
        )
        self.assertTrue(
            registered_sources.issubset(manual_method_labels.keys()),
            msg=(
                "TrackingDialog still needs explicit manual label wiring for some registry sources: "
                f"{sorted(registered_sources - set(manual_method_labels))}"
            ),
        )

    def test_visualizer_and_unavailable_maps_cover_registered_trackers(self) -> None:
        tracker_ids = {spec.tracker_id for spec in self.registry.iter_tracker_specs()}
        subprocess_tracker_ids = {
            spec.tracker_id for spec in self.registry.iter_tracker_specs() if spec.supports_subprocess
        }
        visualizer_retrack_ids = {
            spec.tracker_id for spec in self.registry.iter_tracker_specs() if spec.has_visualizer_retrack
        }

        main_window_titles = _extract_module_level_literal_dict(
            MAIN_WINDOW_PATH,
            "TRACKER_UNAVAILABLE_TITLE_BY_ID",
        )
        tracking_dialog_titles = _extract_module_level_literal_dict(
            TRACKING_DIALOG_PATH,
            "TRACKER_UNAVAILABLE_TITLE_BY_ID",
        )
        main_window_env_hints = _extract_module_level_literal_dict(
            MAIN_WINDOW_PATH,
            "TRACKER_ENV_HINT_BY_ID",
        )
        tracking_dialog_env_hints = _extract_module_level_literal_dict(
            TRACKING_DIALOG_PATH,
            "TRACKER_ENV_HINT_BY_ID",
        )
        visualizer_retrack_map = _extract_module_level_literal_dict(
            TRACKING_DIALOG_PATH,
            "TRACKER_VISUALIZER_RETRACK_METHOD_NAME_BY_ID",
        )
        tracking_dialog_methods = _extract_class_method_names(
            TRACKING_DIALOG_PATH,
            "TrackingDialog",
        )

        self.assertTrue(tracker_ids.issubset(main_window_titles.keys()))
        self.assertTrue(tracker_ids.issubset(tracking_dialog_titles.keys()))
        self.assertTrue(subprocess_tracker_ids.issubset(main_window_env_hints.keys()))
        self.assertTrue(subprocess_tracker_ids.issubset(tracking_dialog_env_hints.keys()))
        self.assertTrue(visualizer_retrack_ids.issubset(visualizer_retrack_map.keys()))

        for tracker_id in visualizer_retrack_ids:
            method_name = visualizer_retrack_map[tracker_id]
            self.assertIn(
                method_name,
                tracking_dialog_methods,
                msg=(
                    f"Tracker `{tracker_id}` is registered for visualizer retrack, "
                    f"but TrackingDialog is missing `{method_name}`."
                ),
            )


if __name__ == "__main__":
    unittest.main()
