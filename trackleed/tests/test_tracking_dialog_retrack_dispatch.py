from __future__ import annotations

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from typing import Protocol, cast


class _CapturedLaunchRequest(Protocol):
    method_id: str
    files: tuple[object, ...]
    track: object
    start_frame_id: int
    config: object


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


class _Qt:
    class WindowModality:
        WindowModal = 0


class _GenericWorkerStub:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.progress = _Dummy()
        self.finished = _Dummy()
        self.error = _Dummy()

    def moveToThread(self, *_args, **_kwargs):
        return None


class TrackingDialogImportMixin:
    module_name = "trackleed.gui.tracking_dialog"

    def import_tracking_dialog(self):
        originals: dict[str, types.ModuleType | None] = {}
        for module_name in (
            "PyQt6",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            "PyQt6.QtWidgets",
            "trackleed.gui.leed_image_viewer",
            "trackleed.gui.polygon_mask_dialog",
            "trackleed.io.image_reader",
            "trackleed.analysis.tomp_backend",
            "trackleed.analysis.tracking_retrack",
            "trackleed.logic.generic_worker",
            self.module_name,
        ):
            originals[module_name] = sys.modules.get(module_name)
            sys.modules.pop(module_name, None)

        qtcore_module = types.ModuleType("PyQt6.QtCore")
        qtcore_module.Qt = _Qt
        qtcore_module.QTimer = _Dummy
        qtcore_module.QPoint = _Dummy
        qtcore_module.QPointF = _Dummy
        qtcore_module.QRect = _Dummy
        qtcore_module.QThread = _Dummy

        qtgui_module = types.ModuleType("PyQt6.QtGui")
        for attr_name in ("QImage", "QPixmap", "QActionGroup", "QAction", "QColor"):
            setattr(qtgui_module, attr_name, _Dummy)

        class _MessageBox:
            @staticmethod
            def warning(*args, **kwargs):
                return None

            @staticmethod
            def information(*args, **kwargs):
                return None

            @staticmethod
            def critical(*args, **kwargs):
                return None

        qtwidgets_module = types.ModuleType("PyQt6.QtWidgets")
        qtwidgets_module.QMessageBox = _MessageBox
        for attr_name in (
            "QDialog",
            "QWidget",
            "QHBoxLayout",
            "QVBoxLayout",
            "QSlider",
            "QSplitter",
            "QLabel",
            "QTableWidget",
            "QAbstractItemView",
            "QTableWidgetItem",
            "QDialogButtonBox",
            "QHeaderView",
            "QPushButton",
            "QToolBar",
            "QMenu",
            "QProgressDialog",
            "QComboBox",
        ):
            setattr(qtwidgets_module, attr_name, _Dummy)

        pyqt6_module = types.ModuleType("PyQt6")
        pyqt6_module.QtCore = qtcore_module
        pyqt6_module.QtGui = qtgui_module
        pyqt6_module.QtWidgets = qtwidgets_module
        sys.modules["PyQt6"] = pyqt6_module
        sys.modules["PyQt6.QtCore"] = qtcore_module
        sys.modules["PyQt6.QtGui"] = qtgui_module
        sys.modules["PyQt6.QtWidgets"] = qtwidgets_module

        viewer_module = types.ModuleType("trackleed.gui.leed_image_viewer")
        viewer_module.LeedImageViewer = _Dummy
        sys.modules["trackleed.gui.leed_image_viewer"] = viewer_module

        polygon_module = types.ModuleType("trackleed.gui.polygon_mask_dialog")
        polygon_module.PolygonMaskDialog = _Dummy
        sys.modules["trackleed.gui.polygon_mask_dialog"] = polygon_module

        image_reader_module = types.ModuleType("trackleed.io.image_reader")
        image_reader_module.load_tiff = lambda *_args, **_kwargs: None
        sys.modules["trackleed.io.image_reader"] = image_reader_module

        tomp_backend_module = types.ModuleType("trackleed.analysis.tomp_backend")
        tomp_backend_module.select_default_tamos_checkpoint_for_variant = (
            lambda checkpoints, variant: None
        )
        tomp_backend_module.select_default_tomp_checkpoint_for_variant = (
            lambda checkpoints, variant: None
        )
        sys.modules["trackleed.analysis.tomp_backend"] = tomp_backend_module

        retrack_module = types.ModuleType("trackleed.analysis.tracking_retrack")
        retrack_module.retrack_single_spot_optical_flow = lambda *args, **kwargs: None
        sys.modules["trackleed.analysis.tracking_retrack"] = retrack_module

        generic_worker_module = types.ModuleType("trackleed.logic.generic_worker")
        generic_worker_module.GenericWorker = _GenericWorkerStub
        sys.modules["trackleed.logic.generic_worker"] = generic_worker_module

        self.addCleanup(self._restore_modules, originals)
        self.addCleanup(sys.modules.pop, self.module_name, None)

        module = importlib.import_module(self.module_name)
        return module, {}

    def _restore_modules(self, originals: dict[str, types.ModuleType | None]) -> None:
        for module_name, original_module in originals.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module


class TrackingDialogRetrackDispatchTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _spot(self, source: str):
        return SimpleNamespace(source=source, id=1)

    def _track(self, *sources: str):
        return SimpleNamespace(
            track_id=7,
            spots_by_image_id={
                idx + 1: self._spot(source)
                for idx, source in enumerate(sources)
            },
        )

    def test_tracker_spec_lookup_for_source_and_track(self) -> None:
        dialog = SimpleNamespace()
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._tracker_spec_for_source_name = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_source_name,
            dialog,
        )
        dialog._tracker_spec_for_track = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_track,
            dialog,
        )
        self.assertIsNone(dialog._tracker_spec_for_source_name("manual_corr"))
        self.assertEqual(
            dialog._tracker_spec_for_source_name("  tamos_offline  ").tracker_id,
            "tomp_offline",
        )
        self.assertEqual(
            dialog._tracker_spec_for_track(
                self._track("tomp_offline", "manual_corr")
            ).tracker_id,
            "tomp_offline",
        )
        self.assertIsNone(
            dialog._tracker_spec_for_track(self._track("tomp_offline", "sam2_offline"))
        )

    def test_backend_retrack_action_text_uses_base_method_label(self) -> None:
        dialog = SimpleNamespace()
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._tracker_spec_for_source_name = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_source_name,
            dialog,
        )
        dialog._tracker_spec_for_track = types.MethodType(
            self.module.TrackingDialog._tracker_spec_for_track,
            dialog,
        )
        dialog._method_label_for_source_name = types.MethodType(
            self.module.TrackingDialog._method_label_for_source_name,
            dialog,
        )
        dialog._classify_track_method = types.MethodType(
            self.module.TrackingDialog._classify_track_method,
            dialog,
        )
        dialog._base_method_label_for_track = types.MethodType(
            self.module.TrackingDialog._base_method_label_for_track,
            dialog,
        )
        dialog._backend_retrack_action_text = types.MethodType(
            self.module.TrackingDialog._backend_retrack_action_text,
            dialog,
        )

        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("tamos_offline", "manual_corr")),
            "Re-track from this point (TaMOs)",
        )
        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("ostrack_offline", "manual_corr")),
            "Re-track from this point (OSTrack)",
        )
        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("dam4sam_offline", "manual_corr")),
            "Re-track from this point (DAM4SAM)",
        )
        self.assertEqual(
            dialog._backend_retrack_action_text(self._track("samurai_offline", "manual_corr")),
            "Re-track from this point (SAMURAI)",
        )
        self.assertIsNone(dialog._backend_retrack_action_text(self._track("manual_corr")))

    def test_optical_flow_retrack_uses_public_seeded_workflow(self) -> None:
        track = self._track("optical_flow")
        config = object()
        files = {1: object(), 2: object()}
        calls: list[tuple[object, object]] = []
        self.module.launch_seeded_retrack = lambda host, request: calls.append(
            (host, request)
        )
        dialog = SimpleNamespace(
            _get_retrack_target=lambda: (track, 1),
            seeded_tracking_configs={"optical_flow": config},
            file_map=files,
            sorted_frame_ids=[1, 2],
        )
        dialog._initiate_seeded_retrack = types.MethodType(
            self.module.TrackingDialog._initiate_seeded_retrack,
            dialog,
        )

        self.module.TrackingDialog.initiate_retrack_optical_flow(
            dialog,
            17,
        )

        host, captured_request = calls[0]
        request = cast(_CapturedLaunchRequest, captured_request)
        self.assertIs(host, dialog)
        self.assertEqual(request.method_id, "optical_flow")
        self.assertEqual(request.files, (files[1], files[2]))
        self.assertIs(request.track, track)
        self.assertEqual(request.start_frame_id, 1)
        self.assertIs(request.config, config)

    def test_cancelled_retrack_keeps_the_existing_review_copy(self) -> None:
        track = self._track("tapir_offline")
        dialog = SimpleNamespace(
            _active_retrack_label="Offline TAPIR",
            tracks=[track],
            populate_tracks_table=lambda: self.fail(
                "Cancelled re-track must not refresh changed tracks"
            ),
            show_frame=lambda _frame_id: self.fail(
                "Cancelled re-track must not redraw changed tracks"
            ),
        )

        self.module.TrackingDialog.on_retrack_finished(dialog, None)

        self.assertEqual(dialog.tracks, [track])

    def test_failed_retrack_keeps_the_existing_review_copy(self) -> None:
        track = self._track("tapir_offline")
        dialog = SimpleNamespace(
            _active_retrack_label="Offline TAPIR",
            tracks=[track],
        )

        self.module.TrackingDialog.on_retrack_error(dialog, "boom")

        self.assertEqual(dialog.tracks, [track])

    def test_successful_retrack_updates_the_review_copy(self) -> None:
        original = self._track("tapir_offline")
        updated = self._track("tapir_offline", "tapir_offline")
        calls: list[object] = []
        dialog = SimpleNamespace(
            _active_retrack_label="Offline TAPIR",
            tracks=[original],
            sorted_frame_ids=[1, 2],
            frame_slider=SimpleNamespace(value=lambda: 0),
            populate_tracks_table=lambda: calls.append("table"),
            show_frame=lambda frame_id: calls.append(("frame", frame_id)),
        )

        self.module.TrackingDialog.on_retrack_finished(dialog, updated)

        self.assertIs(dialog.tracks[0], updated)
        self.assertEqual(calls, ["table", ("frame", 1)])

class TrackingDialogDevaVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_mask_tracker_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_mask_tracker_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_deva_detail_lines_expose_mask_metadata_and_filter_reason(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="deva_offline",
            center_x=12.0,
            center_y=34.0,
            deva_raw_x=12.345,
            deva_raw_y=34.567,
            deva_refined_x=12.0,
            deva_refined_y=34.0,
            deva_visibility=True,
            deva_repr_mode="mask_centroid",
            deva_initial_mask_mode="bbox",
            deva_mask_area=123.4,
            deva_mask_score=1.0,
            deva_mask_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            deva_mask_bbox_aspect_ratio=0.8333333,
            deva_filter_rejected_reason="max_bbox_center_jump",
        )
        lines: list[str] = []

        handled = dialog._append_mask_tracker_detail_lines(
            lines,
            spot,
            source="deva_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Representative mode: mask_centroid", lines)
        self.assertIn("Initial mask mode: bbox", lines)
        self.assertIn("Mask area: 123.4", lines)
        self.assertIn("Mask score: 1.000", lines)
        self.assertIn("Mask bbox: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_bbox_center_jump)", lines)

    def test_deva_focus_point_uses_mask_tracker_spec(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="deva_offline",
            center_x=10.0,
            center_y=11.0,
            deva_raw_x=20.5,
            deva_raw_y=21.5,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(payload, [{"x": 20.5, "y": 21.5, "label": "DEVA T9"}])


class TrackingDialogDam4SamVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_mask_tracker_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_mask_tracker_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_dam4sam_detail_lines_expose_mask_metadata_without_polygon_fields(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="dam4sam_offline",
            center_x=12.0,
            center_y=34.0,
            dam4sam_raw_x=12.345,
            dam4sam_raw_y=34.567,
            dam4sam_refined_x=12.0,
            dam4sam_refined_y=34.0,
            dam4sam_visibility=True,
            dam4sam_repr_mode="mask_centroid",
            dam4sam_tracker_name="sam21pp-L",
            dam4sam_checkpoint_name="sam2.1_hiera_large.pt",
            dam4sam_config_name="sam21pp_hiera_l.yaml",
            dam4sam_mask_area=123.4,
            dam4sam_mask_score=1.0,
            dam4sam_mask_component_count=2,
            dam4sam_mask_bbox_xywh=(3.0, 4.0, 10.0, 12.0),
            dam4sam_mask_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            dam4sam_mask_bbox_aspect_ratio=0.8333333,
            dam4sam_filter_rejected_reason="max_mask_components",
        )
        lines: list[str] = []

        handled = dialog._append_mask_tracker_detail_lines(
            lines,
            spot,
            source="dam4sam_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Representative mode: mask_centroid", lines)
        self.assertIn("Tracker variant: sam21pp-L", lines)
        self.assertIn("Checkpoint: sam2.1_hiera_large.pt", lines)
        self.assertIn("Config: sam21pp_hiera_l.yaml", lines)
        self.assertIn("Mask area: 123.4", lines)
        self.assertIn("Mask score: 1.000", lines)
        self.assertIn("Mask components: 2", lines)
        self.assertIn("Mask bbox xywh: x=3.0, y=4.0, w=10.0, h=12.0", lines)
        self.assertIn("Mask bbox: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_mask_components)", lines)
        self.assertNotIn("Initial mask mode: bbox", lines)
        self.assertNotIn("Prompt mode: bbox", lines)

    def test_dam4sam_focus_point_uses_mask_tracker_spec(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="dam4sam_offline",
            center_x=10.0,
            center_y=11.0,
            dam4sam_raw_x=20.5,
            dam4sam_raw_y=21.5,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(payload, [{"x": 20.5, "y": 21.5, "label": "DAM4SAM T9"}])


class TrackingDialogSamuraiVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_mask_tracker_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_mask_tracker_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_samurai_detail_lines_expose_mask_metadata_without_polygon_fields(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="samurai_offline",
            center_x=12.0,
            center_y=34.0,
            samurai_raw_x=12.345,
            samurai_raw_y=34.567,
            samurai_refined_x=12.0,
            samurai_refined_y=34.0,
            samurai_visibility=True,
            samurai_repr_mode="gaussian_in_bbox",
            samurai_model_variant="sam2.1_hiera_base_plus",
            samurai_checkpoint_name="sam2.1_hiera_base_plus.pt",
            samurai_config_name="sam2.1_hiera_b+.yaml",
            samurai_mask_area=123.4,
            samurai_mask_score=1.0,
            samurai_mask_component_count=2,
            samurai_mask_bbox_xywh=(3.0, 4.0, 10.0, 12.0),
            samurai_mask_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            samurai_mask_bbox_aspect_ratio=0.8333333,
            samurai_filter_rejected_reason="max_mask_components",
        )
        lines: list[str] = []

        handled = dialog._append_mask_tracker_detail_lines(
            lines,
            spot,
            source="samurai_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Representative mode: gaussian_in_bbox", lines)
        self.assertIn("Model variant: sam2.1_hiera_base_plus", lines)
        self.assertIn("Checkpoint: sam2.1_hiera_base_plus.pt", lines)
        self.assertIn("Config: sam2.1_hiera_b+.yaml", lines)
        self.assertIn("Mask area: 123.4", lines)
        self.assertIn("Mask score: 1.000", lines)
        self.assertIn("Mask components: 2", lines)
        self.assertIn("Mask bbox xywh: x=3.0, y=4.0, w=10.0, h=12.0", lines)
        self.assertIn("Mask bbox: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_mask_components)", lines)
        self.assertNotIn("Initial mask mode: bbox", lines)
        self.assertNotIn("Prompt mode: bbox", lines)

    def test_samurai_focus_point_uses_mask_tracker_spec(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="samurai_offline",
            center_x=10.0,
            center_y=11.0,
            samurai_raw_x=20.5,
            samurai_raw_y=21.5,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(payload, [{"x": 20.5, "y": 21.5, "label": "SAMURAI T9"}])


class TrackingDialogOstrackVisualizerTests(TrackingDialogImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.tracker_specs = self.import_tracking_dialog()

    def _dialog_with_detail_helpers(self):
        dialog = SimpleNamespace()
        dialog._append_formatted_value_line = types.MethodType(
            self.module.TrackingDialog._append_formatted_value_line,
            dialog,
        )
        dialog._append_ostrack_detail_lines = types.MethodType(
            self.module.TrackingDialog._append_ostrack_detail_lines,
            dialog,
        )
        dialog._normalized_source_name = types.MethodType(
            self.module.TrackingDialog._normalized_source_name,
            dialog,
        )
        dialog._build_focus_point_payload_for_spot = types.MethodType(
            self.module.TrackingDialog._build_focus_point_payload_for_spot,
            dialog,
        )
        return dialog

    def test_ostrack_detail_lines_expose_bbox_metadata_and_filter_reason(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        spot = SimpleNamespace(
            source="ostrack_offline",
            center_x=12.0,
            center_y=34.0,
            ostrack_raw_x=12.345,
            ostrack_raw_y=34.567,
            ostrack_refined_x=12.0,
            ostrack_refined_y=34.0,
            ostrack_visibility=True,
            ostrack_repr_mode="peak_in_bbox",
            ostrack_model_variant="vitb_256_mae_ce_32x4_ep300",
            ostrack_checkpoint_name="OSTrack_ep0300.pth.tar",
            ostrack_bbox_score=0.87654,
            ostrack_bbox_xywh=(3.0, 4.0, 10.0, 12.0),
            ostrack_bbox_xyxy=(3.0, 4.0, 13.0, 16.0),
            ostrack_bbox_aspect_ratio=0.8333333,
            ostrack_filter_rejected_reason="max_bbox_center_jump",
        )
        lines: list[str] = []

        handled = dialog._append_ostrack_detail_lines(
            lines,
            spot,
            source="ostrack_offline",
        )

        self.assertTrue(handled)
        self.assertIn("Representative point: (12.35, 34.57)", lines)
        self.assertIn("Stored point: (12.00, 34.00)", lines)
        self.assertIn("Visible: yes", lines)
        self.assertIn("Model variant: vitb_256_mae_ce_32x4_ep300", lines)
        self.assertIn("Checkpoint: OSTrack_ep0300.pth.tar", lines)
        self.assertIn("Representative mode: peak_in_bbox", lines)
        self.assertIn("BBox score: 0.877", lines)
        self.assertIn("Raw bbox xywh: x=3.0, y=4.0, w=10.0, h=12.0", lines)
        self.assertIn("Raw bbox xyxy: x0=3.0, y0=4.0, x1=13.0, y1=16.0", lines)
        self.assertIn("BBox aspect ratio: 0.833", lines)
        self.assertIn("Filter: rejected (max_bbox_center_jump)", lines)

    def test_ostrack_focus_payload_exposes_raw_and_stored_points(self) -> None:
        dialog = self._dialog_with_detail_helpers()
        track = SimpleNamespace(track_id=9)
        spot = SimpleNamespace(
            source="ostrack_offline",
            center_x=10.0,
            center_y=11.0,
            ostrack_raw_x=20.5,
            ostrack_raw_y=21.5,
            ostrack_refined_x=22.0,
            ostrack_refined_y=24.0,
        )

        payload = dialog._build_focus_point_payload_for_spot(track, spot)

        self.assertEqual(
            payload,
            [
                {"x": 20.5, "y": 21.5, "label": "OSTrack T9 raw"},
                {"x": 22.0, "y": 24.0, "label": "OSTrack T9 stored"},
            ],
        )
