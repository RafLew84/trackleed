from __future__ import annotations

import ast
import os
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDockWidget, QToolBar

from trackleed.gui.main_window import MainWindow
from trackleed.io.session import SessionContents
from trackleed.logic.active_analysis_state import ActiveAnalysisState


MAIN_WINDOW_PATH = Path(__file__).resolve().parents[1] / "gui" / "main_window.py"


class MainWindowCutoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)

    def test_exact_workflow_menus_have_one_review_entry(self) -> None:
        menus = {
            action.menu().title().replace("&", ""): action.menu()
            for action in self.window.menuBar().actions()
            if action.menu() is not None
        }

        self.assertEqual(
            [
                action.text()
                for action in menus["Spots"].actions()
                if not action.isSeparator()
            ],
            [
                "Detect Spots...",
                "Copy Spots to All Frames",
                "Normalize ROIs (All Frames)",
            ],
        )
        self.assertEqual(
            [
                action.text()
                for action in menus["Tracking"].actions()
                if not action.isSeparator()
            ],
            [
                "Seeded Spot Tracking...",
                "Detection Linking (Hungarian)...",
                "Detection Linking (Directional)...",
                "Review / Edit Tracks",
            ],
        )
        self.assertEqual(
            [
                action.text()
                for action in menus["Analysis"].actions()
                if not action.isSeparator()
            ],
            ["Track Analysis...", "Position Analysis..."],
        )
        review_locations = [
            menu_name
            for menu_name, menu in menus.items()
            for action in menu.actions()
            if action.text() == "Review / Edit Tracks"
        ]
        self.assertEqual(review_locations, ["Tracking"])

    def test_main_workspace_keeps_manual_spot_tools_without_a_dock(self) -> None:
        self.assertEqual(self.window.findChildren(QDockWidget), [])
        self.assertIsNotNone(self.window.file_list_widget)
        self.assertIsNotNone(self.window.image_viewer)
        self.assertIsNotNone(self.window.spot_list_widget)
        self.assertIsNotNone(self.window.copy_spots_skip_existing_checkbox)
        self.assertIsNotNone(self.window.spot_polygon_button)
        self.assertIsNotNone(self.window.clear_spot_polygon_button)
        toolbar_labels = {
            action.text()
            for toolbar in self.window.findChildren(QToolBar)
            for action in toolbar.actions()
        }
        self.assertTrue({"Pan/Select", "Add ROI", "Add Point"}.issubset(toolbar_labels))

    def test_main_window_has_no_legacy_algorithm_dispatch_surface(self) -> None:
        module = ast.parse(MAIN_WINDOW_PATH.read_text(encoding="utf-8"))
        main_window = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
        )
        method_names = {
            node.name
            for node in main_window.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        imported_modules = {
            node.module for node in module.body if isinstance(node, ast.ImportFrom)
        }

        self.assertNotIn("_create_analysis_panel", method_names)
        self.assertNotIn("run_tracking", method_names)
        self.assertNotIn("_build_registered_tracking_worker", method_names)
        self.assertNotIn("trackleed.gui.tracker_registry", imported_modules)

    def test_track_analysis_fit_cache_survives_dialog_reopen_and_rejection(
        self,
    ) -> None:
        seen = {}

        class FakeAnalysisDialog:
            def __init__(self, tracks, _files, _background, **kwargs):
                seen["analysis_state"] = kwargs["analysis_state"]
                self.analysis_state = ActiveAnalysisState(
                    "next-fit-set", "next-geometry-result"
                )
                self.tracks = tracks

            def exec(self):
                return 0

        module_name = "trackleed.gui.analysis_dialog"
        original_module = sys.modules.get(module_name)
        fake_module = types.ModuleType(module_name)
        fake_module.AnalysisResultsDialog = FakeAnalysisDialog
        sys.modules[module_name] = fake_module
        self.window.tracking_results = [object()]
        self.window._active_analysis_state = ActiveAnalysisState(
            "previous-fit-set", "previous-geometry-result", True
        )
        try:
            self.window.show_analysis_dialog()
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        previous = seen["analysis_state"]
        self.assertEqual(previous.fit_set, "previous-fit-set")
        self.assertTrue(previous.requires_fit_validation)
        self.assertEqual(self.window._active_analysis_state.fit_set, "next-fit-set")
        self.assertEqual(
            previous.geometry_result,
            "previous-geometry-result",
        )
        self.assertEqual(
            self.window._active_analysis_state.geometry_result,
            "next-geometry-result",
        )

    def test_analysis_cache_is_bound_to_source_background_and_session_lifecycle(
        self,
    ) -> None:
        self.window._active_analysis_state = ActiveAnalysisState(
            "stale-fit", "stale-geometry"
        )
        with (
            patch(
                "trackleed.gui.main_window.QFileDialog.getExistingDirectory",
                return_value="new-source",
            ),
            patch(
                "trackleed.gui.main_window.parse_directory",
                return_value=[object()],
            ),
            patch.object(self.window, "populate_file_list"),
            patch("trackleed.gui.main_window.QMessageBox.information"),
        ):
            self.window.open_directory_dialog()
        self.assertEqual(self.window._active_analysis_state.fit_set, "stale-fit")
        self.assertIsNone(self.window._active_analysis_state.geometry_result)
        self.assertTrue(self.window._active_analysis_state.requires_fit_validation)

        self.window._active_analysis_state = ActiveAnalysisState(
            "stale-fit", "stale-geometry"
        )
        with (
            patch(
                "trackleed.gui.main_window.QFileDialog.getOpenFileName",
                return_value=("new-background.tif", ""),
            ),
            patch.object(self.window, "_update_bg_preview"),
            patch("trackleed.gui.main_window.QMessageBox.information"),
        ):
            self.window.load_background_file_dialog()
        self.assertEqual(self.window._active_analysis_state.fit_set, "stale-fit")
        self.assertIsNone(self.window._active_analysis_state.geometry_result)
        self.assertTrue(self.window._active_analysis_state.requires_fit_validation)

        self.window._active_analysis_state = ActiveAnalysisState(
            "stale-fit", "stale-geometry"
        )
        with (
            patch(
                "trackleed.gui.session_controller.load_session_from_file",
                return_value=SessionContents(
                    [], None, [], [], {}, None, None, ActiveAnalysisState()
                ),
            ),
            patch.object(self.window, "populate_file_list"),
            patch("trackleed.gui.main_window.QMessageBox.information"),
        ):
            self.window.session_controller.load(Path("replacement.lts.json"))
        self.assertEqual(self.window._active_analysis_state, ActiveAnalysisState())

        self.window._active_analysis_state = ActiveAnalysisState(
            "stale-fit", "stale-geometry"
        )
        self.window.clear_file_list()
        self.assertEqual(self.window._active_analysis_state, ActiveAnalysisState())

        self.window._active_analysis_state = ActiveAnalysisState(
            "compatible-fit", "stale-geometry"
        )
        with (
            patch.object(self.window, "_apply_tracks_to_files"),
            patch.object(self.window, "show_tracking_visualizer"),
        ):
            self.window.on_tracking_finished([])
        self.assertEqual(self.window._active_analysis_state.fit_set, "compatible-fit")
        self.assertIsNone(self.window._active_analysis_state.geometry_result)

        self.window.analysis_ellipse = (10, 20, 30, 15, 5.0)
        self.window._active_analysis_state = ActiveAnalysisState(
            "compatible-fit", "manual-ellipse-geometry"
        )
        self.window.clear_analysis_ellipse()
        self.assertEqual(self.window._active_analysis_state.fit_set, "compatible-fit")
        self.assertIsNone(self.window._active_analysis_state.geometry_result)


if __name__ == "__main__":
    unittest.main()
