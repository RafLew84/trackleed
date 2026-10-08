from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QListWidgetItem

from trackleed.analysis.circle_fit import fit_circle_center
from trackleed.core.data_models import LeedFile, Track
from trackleed.gui.analysis_dialog import AnalysisResultsDialog
from trackleed.gui.radius_results import draw_radius_plot
from trackleed.io.analysis_session import decode_analysis_state, encode_analysis_state
from trackleed.io.results_exporter import export_full_results
from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.track_analysis_compatibility import run_geometry_analysis_publication
from trackleed.logic.workflows.track_geometry import (
    GeometryFingerprint, GeometryMethod, GeometryRequest, derive_geometry,
)
from trackleed.tests.test_track_geometry_workflow import _fit_set


def _positions(points: np.ndarray) -> list[dict[str, float]]:
    return [{"center_x": float(x), "center_y": float(y)} for x, y in points]


def _publication():
    angles = np.arange(6) * np.pi / 3
    radii = np.array([21.0, 19.0, 20.5, 21.0, 19.0, 20.5])
    points = np.column_stack((80 + radii * np.cos(angles), 80 + radii * np.sin(angles)))
    fit_set = _fit_set(tuple((i + 1, 1, x, y) for i, (x, y) in enumerate(points)))
    tracks = [Track(i + 1, {}) for i in range(6)]
    frame = LeedFile(Path("frame.tif"), "frame.tif", False, "Cu", 111, "G", 300, 80, 1, None)
    return run_geometry_analysis_publication(
        tracks, {1: frame}, fit_set, lambda _value: None, center_mode="circle_fit",
    )


class CircleRadiusNumericalTests(unittest.TestCase):
    def test_exact_circle_and_three_point_uncertainty(self) -> None:
        angles = np.arange(8) * np.pi / 4
        points = np.column_stack((100 + 20 * np.cos(angles), 200 + 20 * np.sin(angles)))
        result = fit_circle_center(_positions(points))
        self.assertAlmostEqual(result["circle_radius"], 20, places=9)
        self.assertLess(result["circle_radius_error"], 1e-9)
        three = fit_circle_center(_positions(points[[0, 2, 4]]))
        self.assertAlmostEqual(three["circle_radius"], 20, places=9)
        self.assertNotIn("circle_radius_error", three)
        self.assertIsNone(fit_circle_center(_positions(np.array([[1, 1], [2, 2], [3, 3]]))))

    def test_uncertainty_is_translation_invariant_and_scales_with_image(self) -> None:
        points = np.array([[10, 0], [5, 9], [-5, 8], [-11, 0], [-5, -8], [5, -9]])
        original = fit_circle_center(_positions(points))
        moved = fit_circle_center(_positions(points * 3 + [500, 700]))
        self.assertGreater(original["circle_radius_error"], 0)
        self.assertAlmostEqual(moved["circle_radius"], 3 * original["circle_radius"], places=7)
        self.assertAlmostEqual(moved["circle_radius_error"], 3 * original["circle_radius_error"], places=7)

    def test_reported_uncertainty_agrees_with_repeated_noisy_circle_fits(self) -> None:
        angles = np.arange(16) * 2 * np.pi / 16
        truth = np.column_stack((50 * np.cos(angles), 50 * np.sin(angles)))
        generator = np.random.default_rng(402)
        results = [fit_circle_center(_positions(truth + generator.normal(0, 0.2, truth.shape)))
                   for _ in range(400)]
        empirical = np.std([result["circle_radius"] for result in results], ddof=1)
        predicted = np.mean([result["circle_radius_error"] for result in results])
        self.assertGreater(predicted / empirical, 0.8)
        self.assertLess(predicted / empirical, 1.2)

    def test_geometry_session_and_csv_preserve_optional_uncertainty(self) -> None:
        publication = _publication()
        state = ActiveAnalysisState().publish(publication.fit_set, publication.geometry_result)
        center = state.geometry_result.frames[0].center
        self.assertGreater(center.circle_radius_error, 0)
        encoded = encode_analysis_state(state)
        restored = decode_analysis_state(json.loads(json.dumps(encoded)))
        self.assertEqual(restored.geometry_result, state.geometry_result)
        self.assertEqual(json.dumps(encode_analysis_state(restored), sort_keys=True),
                         json.dumps(encoded, sort_keys=True))
        with TemporaryDirectory() as folder, patch(
            "trackleed.io.results_exporter._save_visualization_images"
        ):
            export_full_results(list(publication.tracks), dict(publication.files), folder)
            with (Path(folder) / "pattern_center_data.csv").open(newline="") as source:
                row = next(csv.DictReader(source))
            self.assertAlmostEqual(float(row["circle_radius"]), center.circle_radius, delta=0.000051)
            self.assertAlmostEqual(float(row["circle_radius_error"]), center.circle_radius_error, delta=0.000051)

    def test_old_geometry_is_readable_and_recomputed_from_existing_fits(self) -> None:
        publication = _publication()
        state = ActiveAnalysisState().publish(publication.fit_set, publication.geometry_result)
        payload = encode_analysis_state(state)
        def remove_new_field(value):
            if isinstance(value, dict):
                value.pop("circle_radius_error", None)
                for child in value.values():
                    remove_new_field(child)
            elif isinstance(value, list):
                for child in value:
                    remove_new_field(child)
        remove_new_field(payload)
        old = decode_analysis_state(payload)
        self.assertIsNone(old.geometry_result.frames[0].center.circle_radius_error)
        previous = replace(old.geometry_result, fingerprint=GeometryFingerprint("0" * 64))
        with patch("trackleed.analysis.peak_analyzer.fit_spot_to_gaussian", side_effect=AssertionError):
            updated = derive_geometry(old.fit_set, GeometryRequest(
                method=GeometryMethod.CIRCLE_FIT,
                center_reference_track_ids=previous.center_reference_track_ids, previous=previous,
            ))
        self.assertGreater(updated.frames[0].center.circle_radius_error, 0)
        self.assertEqual(updated.observations, previous.observations)

    def test_temporal_smoothing_does_not_claim_raw_radius_uncertainty(self) -> None:
        publication = _publication()
        result = derive_geometry(publication.fit_set, GeometryRequest(
            method=GeometryMethod.CIRCLE_FIT_TEMPORAL,
            center_reference_track_ids=publication.geometry_result.center_reference_track_ids,
        ))
        self.assertIsNotNone(result.frames[0].center.circle_radius)
        self.assertIsNone(result.frames[0].center.circle_radius_error)


class CircleRadiusViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_ui_switches_summary_without_recomputing_or_changing_individual_radii(self) -> None:
        publication = _publication()
        state = ActiveAnalysisState().publish(publication.fit_set, publication.geometry_result)
        dialog = AnalysisResultsDialog(list(publication.tracks), dict(publication.files), None,
                                       analysis_state=state)
        self.addCleanup(dialog.close)
        self.addCleanup(dialog.deleteLater)
        for track in publication.tracks:
            item = QListWidgetItem(f"Track {track.track_id}")
            item.setData(Qt.ItemDataRole.UserRole, track.track_id)
            item.setCheckState(Qt.CheckState.Checked)
            dialog.radius_track_checklist.addItem(item)
        self.assertEqual(dialog.radius_summary_combo.currentData(), "average")
        dialog.update_radius_plot(1)
        dialog.update_radius_data_table()
        self.assertEqual(dialog.radius_data_table.columnCount(), 6)
        axes = dialog.radius_plot_canvas.axes
        average = next(line for line in axes.lines if line.get_label() == "Average Radius")
        expected_mean = np.mean([item.radius for item in state.geometry_result.observations])
        self.assertAlmostEqual(average.get_ydata()[0], expected_mean)
        with patch.object(dialog, "start_analysis_task", side_effect=AssertionError):
            dialog.radius_summary_combo.setCurrentIndex(1)
            circle = next(line for line in axes.lines if line.get_label() == "Circle-fit radius")
            self.assertAlmostEqual(circle.get_ydata()[0], state.geometry_result.frames[0].center.circle_radius)
            self.assertNotAlmostEqual(circle.get_ydata()[0], expected_mean)
            self.assertEqual(dialog.radius_data_table.columnCount(), 7)
            self.assertIn("+/-", dialog.radius_data_table.item(0, 6).text())
            dialog.radius_summary_combo.setCurrentIndex(0)
        self.assertIs(dialog.analysis_state, state)
        self.assertEqual(dialog.radius_data_table.columnCount(), 6)

    def test_circle_series_does_not_depend_on_display_selection_or_fake_missing_errors(self) -> None:
        publication = _publication()
        files = dict(publication.files)
        center = files[1].pattern_center
        center.pop("circle_radius_error")
        axes = Figure().subplots()
        note = draw_radius_plot(axes, publication.tracks, files, selected_track_ids=set(),
                                x_axis="Energy", summary_mode="circle_fit", show_errors=True)
        self.assertEqual(len(axes.lines), 1)
        self.assertAlmostEqual(axes.lines[0].get_ydata()[0], center["circle_radius"])
        self.assertIn("Uncertainty unavailable", note)
        self.assertFalse(any(collection.get_paths() for collection in axes.collections))
        files[1].pattern_center = {"center_x": 80.0, "center_y": 80.0}
        note = draw_radius_plot(axes, publication.tracks, files, selected_track_ids=set(),
                                x_axis="Energy", summary_mode="circle_fit", show_errors=True)
        self.assertEqual(len(axes.lines), 0)
        self.assertIn("No fitted-circle radius", note)
