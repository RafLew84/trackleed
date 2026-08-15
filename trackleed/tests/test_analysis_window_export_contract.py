from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.analysis_dialog import AnalysisResultsDialog
from trackleed.gui.positions_analysis_dialog import (
    PositionsAnalysisDialog,
    PositionsAnalysisGroup,
)
from trackleed.io.session import deserialize_session
from trackleed.tests.test_analysis_export_compatibility import (
    _flat_export_fixture,
    _legacy_session_payload,
)


def _positions_fixture(root: Path) -> tuple[list[Track], dict[int, LeedFile]]:
    image_path = root / "positions.png"
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)).save(
        image_path
    )
    spots = [
        SpotROI(1, QRect(12, 14, 10, 10), 17.0, 19.0, 1.0, "tracking"),
        SpotROI(2, QRect(34, 36, 10, 10), 39.0, 41.0, 1.0, "tracking"),
    ]
    tracks = [Track(spot.id, {1: spot}) for spot in spots]
    frame = LeedFile(
        image_path,
        image_path.name,
        False,
        "Cu",
        111,
        "G",
        301.0,
        81.0,
        1,
        0.1,
        spots,
    )
    return tracks, {1: frame}


def _fit_point(track_id: int, x: float, y: float) -> dict[str, object]:
    return {
        "track_present": True,
        "spot_id": track_id,
        "fit_ok": True,
        "global_center_x": x,
        "global_center_y": y,
        "quality_status": "ok",
        "quality_flags": [],
        "validation_failures": [],
        "warning_flags": [],
        "is_suspicious": False,
        "r_squared": 0.99,
        "fit_results": {"fwhm_x": 4.5, "fwhm_y": 5.5},
    }


def _positions_results() -> dict[str, object]:
    return {
        "groups": [
            {
                "group_id": 1,
                "name": "Group 1",
                "reference_track_id": 1,
                "satellite_track_ids": [2],
            }
        ],
        "fit_points_by_frame": {
            1: {1: _fit_point(1, 17.0, 19.0), 2: _fit_point(2, 39.0, 41.0)}
        },
        "rows": [
            {
                "frame_id": 1,
                "group_id": 1,
                "reference_track_id": 1,
                "satellite_track_id": 2,
                "temperature": 301.0,
                "coverage": 0.1,
                "dx": 22.0,
                "dy": 22.0,
                "r": float(np.hypot(22.0, 22.0)),
                "reference_quality_status": "ok",
                "satellite_quality_status": "ok",
                "reference_quality_flags": [],
                "satellite_quality_flags": [],
                "suspicious": False,
            }
        ],
        "summary": {
            "groups_configured": 1,
            "tracks_in_analysis": 2,
            "frames_processed": 1,
            "rows_generated": 1,
            "successful_fits": 2,
            "failed_fits": 0,
            "suspicious_fits": 0,
            "suspicious_rows": 0,
        },
    }


class AnalysisWindowExportContractTests(unittest.TestCase):
    app: QApplication

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_track_analysis_window_keeps_controls_views_selection_and_export(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks, files = _flat_export_fixture(root)
            tracks.insert(0, Track(99, {}))
            dialog = AnalysisResultsDialog(tracks, files, None)
            self.addCleanup(dialog.deleteLater)

            self.assertEqual(dialog.config_table.rowCount(), 2)
            self.assertEqual(dialog.center_mode_combo.count(), 7)
            self.assertIsNotNone(dialog.position_plot_canvas.axes)
            self.assertIsNotNone(dialog.results_table)
            self.assertTrue(dialog.export_button.isEnabled())
            dialog.config_table.selectRow(1)
            QApplication.processEvents()
            self.assertEqual(dialog.config_table.currentRow(), 1)
            self.assertEqual(dialog.results_table.columnCount(), 4)
            self.assertEqual(dialog.frame_label.text(), "Frame ID: 1")
            self.assertEqual(dialog.position_plot_canvas.axes.get_title(), "Spot Positions")

            output_dir = root / "current-export"
            output_dir.mkdir()
            with (
                patch(
                    "trackleed.gui.analysis_dialog.QFileDialog.getExistingDirectory",
                    return_value=str(output_dir),
                ),
                patch("trackleed.gui.analysis_dialog.QMessageBox.warning") as warning,
                patch("trackleed.gui.analysis_dialog.QMessageBox.information"),
            ):
                dialog.export_button.click()
            warning.assert_not_called()
            self.assertTrue((output_dir / "full_analysis_per_spot.csv").is_file())

    def test_legacy_unavailable_track_window_exports_saved_results(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = deserialize_session(
                _legacy_session_payload(root / "missing.tif"),
                root / "legacy.lts.json",
            )
            files = {frame.id: frame for frame in loaded.files if frame.id is not None}
            dialog = AnalysisResultsDialog(
                loaded.tracks,
                files,
                loaded.background,
                analysis_state=loaded.analysis_state,
            )
            self.addCleanup(dialog.deleteLater)
            self.assertFalse(dialog.perform_analysis_button.isEnabled())
            self.assertTrue(dialog.export_button.isEnabled())

            output_dir = root / "legacy-export"
            output_dir.mkdir()
            with (
                patch(
                    "trackleed.gui.analysis_dialog.QFileDialog.getExistingDirectory",
                    return_value=str(output_dir),
                ),
                patch("trackleed.gui.analysis_dialog.QMessageBox.information"),
            ):
                dialog.export_button.click()
            self.assertIn(
                "12.5000",
                (output_dir / "full_analysis_per_spot.csv").read_text(
                    encoding="utf-8"
                ),
            )

    def test_positions_window_keeps_controls_views_selection_and_export(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracks, files = _positions_fixture(root)
            dialog = PositionsAnalysisDialog(tracks, files)
            self.addCleanup(dialog.deleteLater)
            dialog.groups = [PositionsAnalysisGroup(1, "Group 1", 1, [2])]
            dialog.positions_analysis_results = _positions_results()
            dialog._populate_groups_list()
            dialog.groups_list.setCurrentRow(0)
            dialog._refresh_group_details()
            QApplication.processEvents()

            self.assertEqual(dialog.tracks_table.rowCount(), 2)
            self.assertEqual(dialog.groups_list.currentRow(), 0)
            self.assertEqual(dialog.satellites_list.currentRow(), 0)
            self.assertEqual(dialog.x_axis_combo.count(), 2)
            self.assertEqual(dialog.results_table.rowCount(), 1)
            self.assertTrue(dialog.results_plot.axes.lines)
            self.assertTrue(dialog.export_results_button.isEnabled())

            output_dir = root / "positions-export"
            with (
                patch(
                    "trackleed.gui.positions_analysis_dialog.QFileDialog.getExistingDirectory",
                    return_value=str(output_dir),
                ),
                patch("trackleed.gui.positions_analysis_dialog.QMessageBox.information"),
            ):
                dialog.export_results_button.click()
            group_dir = output_dir / "group_01"
            self.assertTrue(
                (group_dir / "positions_analysis_results_group_01.csv").is_file()
            )
            self.assertEqual(
                [path.name for path in (group_dir / "positions_overlay_images").iterdir()],
                ["positions_0001_positions.png"],
            )


if __name__ == "__main__":
    unittest.main()
