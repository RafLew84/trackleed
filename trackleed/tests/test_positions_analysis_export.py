from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
from types import SimpleNamespace
import unittest


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class PositionsAnalysisDialogImportMixin:
    module_name = "trackleed.gui.positions_analysis_dialog"

    def import_positions_analysis_dialog(self):
        stubbed_module_names = (
            "PyQt6",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            "PyQt6.QtWidgets",
            "matplotlib",
            "matplotlib.backends",
            "matplotlib.backends.backend_qt5agg",
            "matplotlib.figure",
            "trackleed.core.data_models",
            "trackleed.gui.leed_image_viewer",
            "trackleed.analysis.peak_analyzer",
            "trackleed.io.image_reader",
            "trackleed.logic.generic_worker",
            "trackleed.logic.positions_analysis_coordinator",
            self.module_name,
        )
        originals = {name: sys.modules.get(name) for name in stubbed_module_names}
        for name in stubbed_module_names:
            sys.modules.pop(name, None)

        qtcore_module = types.ModuleType("PyQt6.QtCore")
        for name in ("QRect", "QRectF", "QThread"):
            setattr(qtcore_module, name, _Dummy)
        qtcore_module.Qt = _Dummy()

        qtgui_module = types.ModuleType("PyQt6.QtGui")
        for name in ("QAction", "QColor", "QImage", "QPixmap", "QBrush"):
            setattr(qtgui_module, name, _Dummy)

        qtwidgets_module = types.ModuleType("PyQt6.QtWidgets")
        for name in (
            "QAbstractItemView",
            "QApplication",
            "QCheckBox",
            "QDialog",
            "QDialogButtonBox",
            "QDoubleSpinBox",
            "QFileDialog",
            "QFormLayout",
            "QGroupBox",
            "QHBoxLayout",
            "QLabel",
            "QListWidget",
            "QMenuBar",
            "QMessageBox",
            "QPushButton",
            "QProgressDialog",
            "QComboBox",
            "QSpinBox",
            "QSlider",
            "QSplitter",
            "QScrollArea",
            "QTableWidget",
            "QTableWidgetItem",
            "QVBoxLayout",
            "QWidget",
            "QHeaderView",
        ):
            setattr(qtwidgets_module, name, _Dummy)

        pyqt6_module = types.ModuleType("PyQt6")
        pyqt6_module.QtCore = qtcore_module
        pyqt6_module.QtGui = qtgui_module
        pyqt6_module.QtWidgets = qtwidgets_module

        matplotlib_module = types.ModuleType("matplotlib")
        matplotlib_module.use = lambda *_args, **_kwargs: None
        matplotlib_backends_module = types.ModuleType("matplotlib.backends")
        matplotlib_backend_module = types.ModuleType("matplotlib.backends.backend_qt5agg")
        matplotlib_backend_module.FigureCanvasQTAgg = _Dummy
        matplotlib_figure_module = types.ModuleType("matplotlib.figure")
        matplotlib_figure_module.Figure = _Dummy

        data_models_module = types.ModuleType("trackleed.core.data_models")
        for name in ("LeedFile", "SpotROI", "Track"):
            setattr(data_models_module, name, _Dummy)

        viewer_module = types.ModuleType("trackleed.gui.leed_image_viewer")
        viewer_module.LeedImageViewer = _Dummy

        peak_analyzer_module = types.ModuleType("trackleed.analysis.peak_analyzer")
        peak_analyzer_module.fit_spot_to_gaussian = lambda *_args, **_kwargs: None
        peak_analyzer_module.gaussian_2d = lambda *_args, **_kwargs: None

        image_reader_module = types.ModuleType("trackleed.io.image_reader")
        image_reader_module.load_tiff = lambda *_args, **_kwargs: None

        worker_module = types.ModuleType("trackleed.logic.generic_worker")
        worker_module.GenericWorker = _Dummy

        coordinator_module = types.ModuleType("trackleed.logic.positions_analysis_coordinator")
        coordinator_module.DEFAULT_VALIDATION_CONFIG = {}
        for name in (
            "evaluate_fit_quality",
            "fit_track_position_on_frame",
            "format_quality_flags",
            "is_fit_valid",
            "normalize_validation_config",
            "prepare_roi_for_fit",
            "run_positions_analysis",
        ):
            setattr(coordinator_module, name, lambda *_args, **_kwargs: None)

        sys.modules.update(
            {
                "PyQt6": pyqt6_module,
                "PyQt6.QtCore": qtcore_module,
                "PyQt6.QtGui": qtgui_module,
                "PyQt6.QtWidgets": qtwidgets_module,
                "matplotlib": matplotlib_module,
                "matplotlib.backends": matplotlib_backends_module,
                "matplotlib.backends.backend_qt5agg": matplotlib_backend_module,
                "matplotlib.figure": matplotlib_figure_module,
                "trackleed.core.data_models": data_models_module,
                "trackleed.gui.leed_image_viewer": viewer_module,
                "trackleed.analysis.peak_analyzer": peak_analyzer_module,
                "trackleed.io.image_reader": image_reader_module,
                "trackleed.logic.generic_worker": worker_module,
                "trackleed.logic.positions_analysis_coordinator": coordinator_module,
            }
        )

        module = importlib.import_module(self.module_name)

        def restore_modules() -> None:
            for name, original in originals.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self.addCleanup(restore_modules)
        return module


class PositionsAnalysisExportTests(PositionsAnalysisDialogImportMixin, unittest.TestCase):
    @staticmethod
    def _fit_payload(*, quality_status: str = "ok", quality_flags=None) -> dict:
        return {
            "fit_ok": True,
            "global_center_x": 374.4647799545324,
            "global_center_y": 398.9224448564599,
            "quality_status": quality_status,
            "quality_flags": list(quality_flags or []),
            "r_squared": 0.9143446543807014,
            "fit_results": {
                "sigma_x": 7.9127698711916,
                "sigma_y": 8.788337570122774,
                "fwhm_x": 18.633149104398942,
                "fwhm_y": 20.694953472623695,
                "fwhm_x_error": 0.7090809330934775,
                "fwhm_y_error": 0.7548464654868482,
                "fwhm_profile_x": 18.9,
                "fwhm_profile_x_error": 0.71,
                "fwhm_profile_y": 20.45,
                "fwhm_profile_y_error": 0.75,
                "fwhm_minor": 18.633149104398942,
                "fwhm_minor_error": 0.7090809330934775,
                "fwhm_major": 20.694953472623695,
                "fwhm_major_error": 0.7548464654868482,
                "fwhm_effective": 19.636,
                "fwhm_effective_error": 0.52,
                "fwhm_anisotropy": 1.1107,
                "fwhm_anisotropy_error": 0.031,
                "theta": 2.771682403608941,
                "theta_error": 0.02,
                "theta_major": 1.2008860768140445,
                "theta_major_error": 0.04,
                "major_axis_angle_deg": 68.805,
                "major_axis_angle_deg_error": 2.2918,
                "r_squared": 0.9143446543807014,
            },
        }

    @staticmethod
    def _single_reference_dialog(module, fit_payload: dict):
        dialog = module.PositionsAnalysisDialog.__new__(module.PositionsAnalysisDialog)
        dialog.sorted_frame_ids = [19]
        dialog.pending_manual_recompute = set()
        dialog.track_map = {}
        dialog.positions_analysis_results = {
            "groups": [
                {
                    "group_id": 1,
                    "reference_track_id": 1,
                    "satellite_track_ids": [],
                }
            ],
            "fit_points_by_frame": {
                19: {1: fit_payload}
            },
        }
        return dialog

    @staticmethod
    def _export_rows(dialog, *, group_id: int = 1) -> list[dict[str, str]]:
        with TemporaryDirectory() as temp_dir:
            output_path = dialog._export_positions_results_csv(
                Path(temp_dir),
                group_id=group_id,
            )
            with output_path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))

    def test_valid_stored_fit_exports_fwhm_values(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = self._single_reference_dialog(module, self._fit_payload())

        rows = self._export_rows(dialog)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            list(rows[0])[:7],
            [
                "frame_id",
                "group_id",
                "point_role",
                "track_id",
                "distance_from_reference",
                "x",
                "y",
            ],
        )
        self.assertEqual(rows[0]["fwhm_x"], "18.633149104398942")
        self.assertEqual(rows[0]["fwhm_y"], "20.694953472623695")

    def test_warning_fit_exports_metrics_and_quality_metadata(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = self._single_reference_dialog(
            module,
            self._fit_payload(
                quality_status="warning",
                quality_flags=["center_far_from_roi_center"],
            ),
        )

        row = self._export_rows(dialog)[0]

        self.assertEqual(row["position_source"], "gaussian_fit")
        self.assertEqual(row["fit_ok"], "True")
        self.assertEqual(row["quality_status"], "warning")
        self.assertEqual(row["quality_flags"], "center_far_from_roi_center")
        self.assertEqual(row["sigma_x"], "7.9127698711916")
        self.assertEqual(row["sigma_y"], "8.788337570122774")
        self.assertEqual(row["fwhm_x_error"], "0.7090809330934775")
        self.assertEqual(row["fwhm_y_error"], "0.7548464654868482")
        self.assertEqual(row["fwhm_profile_x"], "18.9")
        self.assertEqual(row["fwhm_profile_y"], "20.45")
        self.assertEqual(row["fwhm_minor"], "18.633149104398942")
        self.assertEqual(row["fwhm_major"], "20.694953472623695")
        self.assertEqual(row["fwhm_effective"], "19.636")
        self.assertEqual(row["fwhm_anisotropy"], "1.1107")
        self.assertEqual(row["fwhm_anisotropy_error"], "0.031")
        self.assertEqual(row["theta"], "2.771682403608941")
        self.assertEqual(row["theta_error"], "0.02")
        self.assertAlmostEqual(float(row["theta_major"]), 1.2008860768140445)
        self.assertEqual(row["major_axis_angle_deg"], "68.805")
        self.assertEqual(row["major_axis_angle_deg_error"], "2.2918")
        self.assertEqual(row["r_squared"], "0.9143446543807014")

    def test_failed_fit_uses_tracked_position_and_preserves_raw_fit(self) -> None:
        module = self.import_positions_analysis_dialog()
        fit_payload = self._fit_payload(
            quality_status="failed",
            quality_flags=["low_r_squared"],
        )
        fit_payload["fit_ok"] = False
        fit_payload["r_squared"] = 0.41
        dialog = self._single_reference_dialog(module, fit_payload)
        dialog.track_map = {
            1: SimpleNamespace(
                spots_by_image_id={
                    19: SimpleNamespace(center_x=375.0, center_y=399.0),
                }
            )
        }

        row = self._export_rows(dialog)[0]

        self.assertEqual(row["position_source"], "tracked_center")
        self.assertEqual(row["fit_ok"], "False")
        self.assertEqual(row["quality_status"], "failed")
        self.assertEqual(row["quality_flags"], "low_r_squared")
        self.assertEqual(row["x"], "375.0")
        self.assertEqual(row["y"], "399.0")
        self.assertEqual(row["sigma_x"], "7.9127698711916")
        self.assertEqual(row["sigma_y"], "8.788337570122774")
        self.assertEqual(row["fwhm_x"], "18.633149104398942")
        self.assertEqual(row["fwhm_y"], "20.694953472623695")
        self.assertEqual(row["fwhm_profile_x"], "18.9")
        self.assertEqual(row["fwhm_profile_y"], "20.45")
        self.assertEqual(row["fwhm_effective"], "19.636")
        self.assertEqual(row["r_squared"], "0.41")

    def test_missing_spot_is_omitted_from_export(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = self._single_reference_dialog(
            module,
            {
                "fit_ok": False,
                "global_center_x": float("nan"),
                "global_center_y": float("nan"),
                "quality_status": "failed",
                "quality_flags": [],
                "r_squared": float("nan"),
                "fit_results": None,
            },
        )

        self.assertEqual(self._export_rows(dialog), [])

    def test_pending_manual_recompute_does_not_export_stale_widths(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = self._single_reference_dialog(module, self._fit_payload())
        dialog.pending_manual_recompute = {(19, 1)}
        dialog.track_map = {
            1: SimpleNamespace(
                spots_by_image_id={
                    19: SimpleNamespace(center_x=375.0, center_y=399.0),
                }
            )
        }

        row = self._export_rows(dialog)[0]

        self.assertEqual(row["position_source"], "tracked_center")
        self.assertEqual(row["fit_ok"], "False")
        self.assertEqual(row["fwhm_x"], "")
        self.assertEqual(row["fwhm_y"], "")

    def test_csv_export_uses_stored_results_without_loading_tiff(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = self._single_reference_dialog(module, self._fit_payload())

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("CSV export must not load TIFF files")

        module.load_tiff = fail_if_called

        rows = self._export_rows(dialog)

        self.assertEqual(rows[0]["fwhm_x"], "18.633149104398942")

    def test_group_filter_preserves_reference_and_satellite_fit_metadata(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = module.PositionsAnalysisDialog.__new__(module.PositionsAnalysisDialog)
        dialog.sorted_frame_ids = [19]
        dialog.pending_manual_recompute = set()
        dialog.track_map = {}
        dialog.positions_analysis_results = {
            "groups": [
                {
                    "group_id": 1,
                    "reference_track_id": 1,
                    "satellite_track_ids": [],
                },
                {
                    "group_id": 2,
                    "reference_track_id": 2,
                    "satellite_track_ids": [3],
                },
            ],
            "fit_points_by_frame": {
                19: {
                    1: self._fit_payload(),
                    2: self._fit_payload(),
                    3: self._fit_payload(),
                }
            },
        }

        rows = self._export_rows(dialog, group_id=2)

        self.assertEqual([row["track_id"] for row in rows], ["2", "3"])
        self.assertEqual([row["point_role"] for row in rows], ["R", "S1"])
        self.assertTrue(all(row["group_id"] == "2" for row in rows))
        self.assertTrue(all(row["position_source"] == "gaussian_fit" for row in rows))
        self.assertTrue(all(row["fwhm_x"] == "18.633149104398942" for row in rows))

    def test_missing_fit_exports_empty_metrics_instead_of_nan(self) -> None:
        module = self.import_positions_analysis_dialog()
        dialog = self._single_reference_dialog(
            module,
            {
                "fit_ok": False,
                "global_center_x": float("nan"),
                "global_center_y": float("nan"),
                "quality_status": "failed",
                "quality_flags": [],
                "r_squared": float("nan"),
                "fit_results": None,
            },
        )
        dialog.track_map = {
            1: SimpleNamespace(
                spots_by_image_id={
                    19: SimpleNamespace(center_x=375.0, center_y=399.0),
                }
            )
        }

        row = self._export_rows(dialog)[0]

        self.assertEqual(row["position_source"], "tracked_center")
        for fieldname in (
            "sigma_x",
            "sigma_y",
            "fwhm_x",
            "fwhm_y",
            "fwhm_x_error",
            "fwhm_y_error",
            "fwhm_profile_x",
            "fwhm_profile_y",
            "fwhm_profile_x_error",
            "fwhm_profile_y_error",
            "fwhm_minor",
            "fwhm_minor_error",
            "fwhm_major",
            "fwhm_major_error",
            "fwhm_effective",
            "fwhm_effective_error",
            "fwhm_anisotropy",
            "fwhm_anisotropy_error",
            "theta",
            "theta_error",
            "theta_major",
            "theta_major_error",
            "major_axis_angle_deg",
            "major_axis_angle_deg_error",
            "r_squared",
        ):
            self.assertEqual(row[fieldname], "")


if __name__ == "__main__":
    unittest.main()
