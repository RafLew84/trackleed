from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui import main_window
from trackleed.gui import session_controller
from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.workflows.seeded_tracking import OpticalFlowConfig


def _spot(x: float) -> SpotROI:
    return SpotROI(7, QRect(int(x) - 4, 12, 8, 8), x, 16.0, 1.0, "tracking")


def _frame() -> LeedFile:
    return LeedFile(
        Path("frame-1.tif"),
        "frame-1.tif",
        False,
        "Cu",
        111,
        "G",
        300.0,
        80.0,
        1,
        None,
        [_spot(16.0)],
    )


class _ReviewDialog:
    result = QDialog.DialogCode.Rejected
    last_kwargs: dict[str, object]

    def __init__(
        self, tracking_data: list[Track], *_args: object, **kwargs: object
    ) -> None:
        type(self).last_kwargs = kwargs
        self.tracks = tracking_data
        self.tracks[0].spots_by_image_id[1].center_x = 44.0

    def exec(self) -> QDialog.DialogCode:
        return self.result

    def get_updated_tracks(self) -> list[Track]:
        return self.tracks


def _host(frame: LeedFile) -> SimpleNamespace:
    optical_flow_config = OpticalFlowConfig(roi_size=18)
    return SimpleNamespace(
        tracking_results=[Track(7, {1: _spot(16.0)})],
        data_files=[frame],
        metadata_mask_list=[],
        seeded_tracking_controller=SimpleNamespace(
            review_defaults=SimpleNamespace(
                roi_size=18,
                lk_win_size=41,
                lk_max_level=4,
            ),
            review_tracking_configs={"optical_flow": optical_flow_config},
        ),
        file_list_widget=SimpleNamespace(currentItem=lambda: None),
        session_controller=SimpleNamespace(save_dialog=lambda _tracks: None),
        _active_analysis_state=ActiveAnalysisState(),
    )


class SeededTrackingReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.original_dialog = main_window.TrackingDialog
        self.original_information = main_window.QMessageBox.information
        main_window.TrackingDialog = _ReviewDialog
        main_window.QMessageBox.information = lambda *_args: None
        self.addCleanup(setattr, main_window, "TrackingDialog", self.original_dialog)
        self.addCleanup(
            setattr,
            main_window.QMessageBox,
            "information",
            self.original_information,
        )

    def test_closing_review_leaves_raw_result_and_frame_projection_unchanged(
        self,
    ) -> None:
        frame = _frame()
        host = _host(frame)
        _ReviewDialog.result = QDialog.DialogCode.Rejected

        main_window.MainWindow.show_tracking_visualizer(host)

        self.assertEqual(host.tracking_results[0].spots_by_image_id[1].center_x, 16.0)
        self.assertEqual(frame.spots[0].center_x, 16.0)

    def test_accept_tracks_atomically_publishes_edited_copy(self) -> None:
        frame = _frame()
        host = _host(frame)
        _ReviewDialog.result = QDialog.DialogCode.Accepted

        main_window.MainWindow.show_tracking_visualizer(host)

        active_spot = host.tracking_results[0].spots_by_image_id[1]
        self.assertEqual(active_spot.center_x, 44.0)
        self.assertEqual(frame.spots[0].center_x, 44.0)
        self.assertIsNot(active_spot, frame.spots[0])
        self.assertEqual(host._active_analysis_state, ActiveAnalysisState())

    def test_review_dialog_uses_approved_title_and_accept_label(self) -> None:
        dialog = self.original_dialog([], [], [], 18, 41, 4)
        self.addCleanup(dialog.deleteLater)

        self.assertEqual(dialog.windowTitle(), "Review / Edit Tracks")
        accept_button = dialog.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.assertEqual(accept_button.text(), "Accept Tracks")

    def test_review_dialog_receives_only_typed_seeded_tracking_configs(self) -> None:
        frame = _frame()
        host = _host(frame)

        main_window.MainWindow.show_tracking_visualizer(host)

        kwargs = _ReviewDialog.last_kwargs
        review_configs = cast(
            dict[str, OpticalFlowConfig],
            kwargs["seeded_tracking_configs"],
        )
        self.assertIs(
            review_configs["optical_flow"],
            host.seeded_tracking_controller.review_tracking_configs["optical_flow"],
        )
        self.assertFalse(any(name.endswith("_params") for name in kwargs))

    def test_save_session_from_review_does_not_publish_working_copy(self) -> None:
        frame = _frame()
        raw_tracks = [Track(7, {1: _spot(16.0)})]
        edited_tracks = [Track(7, {1: _spot(44.0)})]
        saved: dict[str, object] = {}
        host = SimpleNamespace(
            data_files=[frame],
            metadata_mask_list=[],
            background_file=None,
            tracking_results=raw_tracks,
            _collect_spot_settings=lambda: {},
            _analysis_circle_to_dict=lambda: None,
            _analysis_ellipse_to_dict=lambda: None,
            _active_analysis_state=ActiveAnalysisState(),
        )
        original_save = session_controller.save_session_to_file
        session_controller.save_session_to_file = lambda _path, contents: saved.update(
            tracks=contents.tracks
        )
        self.addCleanup(
            setattr,
            session_controller,
            "save_session_to_file",
            original_save,
        )

        session_controller.SessionController(host).save(
            Path("ignored.lts.json"),
            edited_tracks,
        )

        self.assertIs(host.tracking_results, raw_tracks)
        self.assertEqual(frame.spots[0].center_x, 16.0)
        self.assertEqual(saved["tracks"][0].spots_by_image_id[1].center_x, 44.0)


if __name__ == "__main__":
    unittest.main()
