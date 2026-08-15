from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRect
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.main_window import MainWindow
from trackleed.io.session import (
    SessionContents,
    load_session_from_file,
    save_session_to_file,
)
from trackleed.io.results_exporter import export_full_results
from trackleed.logic import track_analysis_compatibility
from trackleed.logic.active_analysis_state import (
    ActiveAnalysisState,
    AnalysisResultStatus,
    AnalysisSourceState,
    AnalysisSourceStatus,
)
from trackleed.logic.workflows.gaussian_fit_recovery import (
    FitValidation,
    GaussianFitterConfig,
    RecoveryAttempt,
    RecoveryAttemptStatus,
    RecoveryRecord,
    RecoverySelectionReason,
    RecoveryStatus,
    Rect,
)
from trackleed.logic.workflows.session_sources import (
    SourceVerificationContext,
    verify_analysis_sources,
)
from trackleed.logic.workflows.track_analysis import (
    FitFingerprint,
    FitObservation,
    FitProvenance,
    FitSet,
    FitStatus,
    GaussianErrors,
    GaussianFitResult,
    GaussianIntensity,
    GaussianParameters,
    GaussianWidths,
    FitQuality,
    PreprocessingConfig,
)
from trackleed.logic.workflows.track_geometry import (
    FrameGeometry,
    GeometryFingerprint,
    GeometryMethod,
    GeometryResult,
    GeometryStatus,
    ObservationGeometry,
    PatternCenter,
)


class SessionLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_v1_missing_source_preserves_legacy_analysis_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "legacy.lts.json"
            missing_path = Path(temp_dir) / "missing-frame.tif"
            payload = _v1_payload(missing_path)
            original_bytes = json.dumps(payload, indent=2).encode("utf-8")
            session_path.write_bytes(original_bytes)

            loaded = load_session_from_file(session_path)

            self.assertEqual(session_path.read_bytes(), original_bytes)
            self.assertEqual([item.id for item in loaded.files], [1])
            self.assertEqual(
                loaded.tracks[0].analysis_results[1]["radius"],
                12.5,
            )
            self.assertEqual(
                loaded.analysis_state.result_status.value,
                "legacy_unverified",
            )
            self.assertEqual(
                loaded.analysis_state.sources[0].status.value,
                "source_unavailable",
            )
            migrated_path = Path(temp_dir) / "migrated.lts.json"
            save_session_to_file(migrated_path, loaded)
            migrated = load_session_from_file(migrated_path)
            self.assertEqual(
                json.loads(migrated_path.read_text(encoding="utf-8"))["version"],
                2,
            )
            self.assertIs(
                migrated.analysis_state.result_status,
                AnalysisResultStatus.LEGACY_UNVERIFIED,
            )

    def test_load_action_keeps_missing_frame_and_flat_results_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "legacy.lts.json"
            session_path.write_text(
                json.dumps(_v1_payload(root / "missing-frame.tif")),
                encoding="utf-8",
            )
            window = MainWindow()
            self.addCleanup(window.deleteLater)

            with (
                patch(
                    "trackleed.gui.session_controller.QFileDialog.getOpenFileName",
                    return_value=(str(session_path), ""),
                ),
                patch("trackleed.gui.session_controller.QMessageBox.information"),
                patch("trackleed.gui.session_controller.QMessageBox.warning"),
                patch("trackleed.gui.session_controller.QMessageBox.critical"),
            ):
                window.session_controller.load_dialog()

            self.assertEqual([item.id for item in window.data_files], [1])
            self.assertEqual(window.data_files[0].pattern_center["center_x"], 20.0)
            self.assertEqual(
                window.tracking_results[0].analysis_results[1]["radius"],
                12.5,
            )
            self.assertEqual(
                window._active_analysis_state.result_status,
                AnalysisResultStatus.LEGACY_UNVERIFIED,
            )

    def test_missing_source_keeps_export_but_blocks_new_analysis(self) -> None:
        from trackleed.gui.analysis_dialog import AnalysisResultsDialog

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_path = root / "legacy.lts.json"
            session_path.write_text(
                json.dumps(_v1_payload(root / "missing-frame.tif")),
                encoding="utf-8",
            )
            loaded = load_session_from_file(session_path)
            dialog = AnalysisResultsDialog(
                loaded.tracks,
                {item.id: item for item in loaded.files if item.id is not None},
                loaded.background,
                analysis_state=loaded.analysis_state,
            )
            self.addCleanup(dialog.deleteLater)

            self.assertFalse(dialog.perform_analysis_button.isEnabled())
            self.assertTrue(dialog.export_button.isEnabled())
            self.assertEqual(
                dialog.tracks[0].analysis_results[1]["radius"],
                12.5,
            )
            export_directory = root / "export"
            export_directory.mkdir()
            export_full_results(
                loaded.tracks,
                {item.id: item for item in loaded.files if item.id is not None},
                str(export_directory),
            )
            self.assertTrue((export_directory / "full_analysis_per_spot.csv").is_file())
            self.assertTrue((export_directory / "pattern_center_data.csv").is_file())

    def test_current_session_round_trips_analysis_and_marks_source_unverified(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame-1.tif"
            source_path.write_bytes(b"decoded later in the worker")
            output_path = root / "current.lts.json"
            frame, track = _analyzed_models(source_path)
            track.analysis_results[1]["covariance_matrix"] = np.eye(2)
            state = _current_analysis_state(source_path)

            save_session_to_file(
                output_path,
                _session_contents(frame, track, state),
            )
            loaded = load_session_from_file(output_path)

            saved_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["version"], 2)
            self.assertEqual(loaded.analysis_state.fit_set, state.fit_set)
            self.assertEqual(
                loaded.analysis_state.geometry_result,
                state.geometry_result,
            )
            self.assertEqual(
                loaded.analysis_state.result_status,
                AnalysisResultStatus.CURRENT,
            )
            self.assertIs(
                loaded.analysis_state.result_status,
                AnalysisResultStatus.CURRENT,
            )
            assert loaded.analysis_state.geometry_result is not None
            self.assertIs(
                loaded.analysis_state.geometry_result.requested_method,
                GeometryMethod.CIRCLE_FIT,
            )
            self.assertTrue(loaded.analysis_state.requires_fit_validation)
            self.assertEqual(
                loaded.analysis_state.sources[0].status,
                AnalysisSourceStatus.UNVERIFIED,
            )
            self.assertEqual(
                loaded.tracks[0].analysis_results[1]["radius"],
                track.analysis_results[1]["radius"],
            )
            self.assertEqual(
                set(loaded.tracks[0].analysis_results[1]),
                set(track.analysis_results[1]),
            )
            np.testing.assert_array_equal(
                loaded.tracks[0].analysis_results[1]["covariance_matrix"],
                track.analysis_results[1]["covariance_matrix"],
            )

    def test_current_save_records_source_path_relative_to_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "images" / "frame-1.tif"
            source_path.parent.mkdir()
            source_path.write_bytes(b"decoded later in the worker")
            session_path = root / "current.lts.json"
            frame, track = _analyzed_models(source_path)

            save_session_to_file(
                session_path,
                _session_contents(
                    frame,
                    track,
                    _current_analysis_state(source_path),
                ),
            )
            loaded = load_session_from_file(session_path)

            self.assertEqual(
                loaded.analysis_state.sources[0].relative_path,
                Path("images") / "frame-1.tif",
            )

    def test_save_action_writes_the_active_current_analysis_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame-1.tif"
            source_path.write_bytes(b"decoded later in the worker")
            output_path = root / "saved.lts.json"
            frame, track = _analyzed_models(source_path)
            state = _current_analysis_state(source_path)
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [track]
            window._active_analysis_state = state

            with (
                patch(
                    "trackleed.gui.session_controller.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ),
                patch("trackleed.gui.session_controller.QMessageBox.information"),
                patch("trackleed.gui.session_controller.QMessageBox.critical"),
            ):
                window.session_controller.save_dialog()

            loaded = load_session_from_file(output_path)
            self.assertEqual(loaded.analysis_state.fit_set, state.fit_set)
            self.assertEqual(
                loaded.analysis_state.geometry_result,
                state.geometry_result,
            )

    def test_review_save_drops_inactive_fit_entries_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame-1.tif"
            source_path.write_bytes(b"decoded later in the worker")
            output_path = root / "reviewed.lts.json"
            frame, active_track = _analyzed_models(source_path)
            state = _current_analysis_state(source_path)
            assert state.fit_set is not None
            first_fit = state.fit_set.observations[0]
            inactive_fit = replace(first_fit, frame_id=2)
            state = replace(
                state,
                fit_set=FitSet((first_fit, inactive_fit)),
            )
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [active_track]
            window._active_analysis_state = state

            with (
                patch(
                    "trackleed.gui.session_controller.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ),
                patch("trackleed.gui.session_controller.QMessageBox.information"),
                patch("trackleed.gui.session_controller.QMessageBox.critical"),
            ):
                window.session_controller.save_dialog([active_track])

            loaded = load_session_from_file(output_path)
            assert loaded.analysis_state.fit_set is not None
            self.assertEqual(
                [(item.track_id, item.frame_id) for item in loaded.analysis_state.fit_set.observations],
                [(7, 1)],
            )
            self.assertIsNone(loaded.analysis_state.geometry_result)
            self.assertNotIn("radius", loaded.tracks[0].analysis_results[1])
            self.assertEqual(
                loaded.tracks[0].analysis_results[1]["center_x"],
                13.0,
            )
            self.assertIsNone(loaded.files[0].pattern_center)
            self.assertNotIn(
                "average_radius",
                loaded.files[0].post_analysis_results,
            )

    def test_review_save_round_trips_working_spots_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "frame-1.tif"
            output_path = Path(temp_dir) / "working-review.lts.json"
            frame, active_track = _analyzed_models(source_path)
            working_spot = SpotROI(
                7,
                QRect(28, 9, 10, 11),
                33.0,
                14.0,
                2.0,
                "tracking",
            )
            working_track = Track(7, {1: working_spot})
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [active_track]
            window._active_analysis_state = _current_analysis_state(source_path)

            window.session_controller.save(output_path, [working_track])

            loaded = load_session_from_file(output_path)
            self.assertEqual(
                loaded.tracks[0].spots_by_image_id[1].roi_rect.x(),
                28,
            )
            self.assertEqual(loaded.files[0].spots[0].roi_rect.x(), 28)
            self.assertEqual(frame.spots[0].roi_rect.x(), 8)
            self.assertEqual(active_track.spots_by_image_id[1].roi_rect.x(), 8)

    def test_save_after_accepted_review_prunes_the_active_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame-1.tif"
            source_path.write_bytes(b"decoded later in the worker")
            output_path = root / "accepted-review.lts.json"
            frame, active_track = _analyzed_models(source_path)
            state = _current_analysis_state(source_path)
            assert state.fit_set is not None
            first_fit = state.fit_set.observations[0]
            state = replace(
                state,
                fit_set=FitSet((first_fit, replace(first_fit, frame_id=2))),
                geometry_result=None,
            )
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [active_track]
            window._active_analysis_state = state

            with (
                patch(
                    "trackleed.gui.session_controller.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ),
                patch("trackleed.gui.session_controller.QMessageBox.information"),
                patch("trackleed.gui.session_controller.QMessageBox.critical"),
            ):
                window.session_controller.save_dialog()

            loaded = load_session_from_file(output_path)
            assert loaded.analysis_state.fit_set is not None
            self.assertEqual(len(loaded.analysis_state.fit_set.observations), 1)
            self.assertNotIn("radius", loaded.tracks[0].analysis_results[1])
            self.assertIsNone(loaded.files[0].pattern_center)

    def test_save_drops_same_key_fit_after_roi_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame-1.tif"
            source_path.write_bytes(b"decoded later in the worker")
            output_path = root / "roi-edited.lts.json"
            frame, active_track = _analyzed_models(source_path)
            active_track.spots_by_image_id[1].roi_rect = QRect(9, 9, 10, 11)
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [active_track]
            window._active_analysis_state = _current_analysis_state(source_path)

            with (
                patch(
                    "trackleed.gui.session_controller.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ),
                patch("trackleed.gui.session_controller.QMessageBox.information"),
                patch("trackleed.gui.session_controller.QMessageBox.critical"),
            ):
                window.session_controller.save_dialog()

            loaded = load_session_from_file(output_path)
            assert loaded.analysis_state.fit_set is not None
            self.assertEqual(loaded.analysis_state.fit_set.observations, ())
            self.assertEqual(loaded.tracks[0].analysis_results, {})
            self.assertIsNone(loaded.analysis_state.geometry_result)

    def test_save_invalidates_geometry_when_review_adds_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "frame-1.tif"
            source_path.write_bytes(b"decoded later in the worker")
            output_path = root / "observation-added.lts.json"
            frame, active_track = _analyzed_models(source_path)
            added_spot = SpotROI(
                8,
                QRect(18, 19, 10, 11),
                23.0,
                24.0,
                2.0,
                "tracking",
            )
            added_track = Track(8, {1: added_spot})
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [active_track, added_track]
            window._active_analysis_state = _current_analysis_state(source_path)

            window.session_controller.save(output_path)

            loaded = load_session_from_file(output_path)
            assert loaded.analysis_state.fit_set is not None
            self.assertEqual(len(loaded.analysis_state.fit_set.observations), 1)
            self.assertIsNone(loaded.analysis_state.geometry_result)
            self.assertNotIn("radius", loaded.tracks[0].analysis_results[1])
            self.assertEqual(loaded.tracks[1].analysis_results, {})

    def test_raw_tracking_publication_drops_fit_after_roi_change(self) -> None:
        frame, active_track = _analyzed_models(Path("frame-1.tif"))
        active_track.spots_by_image_id[1].roi_rect = QRect(9, 9, 10, 11)
        window = MainWindow()
        self.addCleanup(window.deleteLater)
        window.data_files = [frame]
        window._active_analysis_state = _current_analysis_state(frame.filepath)

        with patch.object(window, "show_tracking_visualizer"):
            window.on_tracking_finished([active_track])

        assert window._active_analysis_state.fit_set is not None
        self.assertEqual(window._active_analysis_state.fit_set.observations, ())
        self.assertIsNone(window._active_analysis_state.geometry_result)

    def test_relocated_source_is_confirmed_from_decoded_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            relocated_path = root / "relocated" / "frame-1.tif"
            relocated_path.parent.mkdir()
            relocated_path.write_bytes(b"reader boundary owns decoding")
            pixels = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
            expected_hash = sha256(pixels.tobytes(order="C")).hexdigest()
            state = _current_analysis_state(Path("original") / "frame-1.tif")
            source = replace(
                state.sources[0],
                relative_path=Path("relocated") / "frame-1.tif",
                pixel_sha256=expected_hash,
                status=AnalysisSourceStatus.UNVERIFIED,
                pixel_dtype="uint16",
                pixel_shape=(32, 32),
            )
            state = replace(state, requires_fit_validation=True, sources=(source,))

            verified = verify_analysis_sources(
                state,
                SourceVerificationContext(
                    image_reader=lambda path: (
                        pixels if path == relocated_path else None
                    ),
                    relocation_root=root,
                ),
            )

            self.assertEqual(verified.failures, ())
            self.assertEqual(verified.state.sources[0].path, relocated_path)
            self.assertEqual(
                verified.state.sources[0].status,
                AnalysisSourceStatus.AVAILABLE,
            )
            self.assertEqual(
                state.sources[0].status,
                AnalysisSourceStatus.UNVERIFIED,
            )

    def test_wrong_relocated_pixels_block_the_full_workflow_before_refit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            relocated_path = Path(temp_dir) / "frame-1.tif"
            tifffile.imwrite(relocated_path, np.zeros((32, 32), dtype=np.uint16))
            frame, track = _analyzed_models(relocated_path)
            state = _current_analysis_state(Path("original") / "frame-1.tif")
            source = replace(
                state.sources[0],
                path=relocated_path,
                status=AnalysisSourceStatus.UNVERIFIED,
                pixel_dtype="uint16",
                pixel_shape=(32, 32),
            )
            state = replace(state, requires_fit_validation=True, sources=(source,))

            with self.assertRaisesRegex(
                RuntimeError,
                "pixel_fingerprint_mismatch",
            ):
                track_analysis_compatibility.run_full_analysis_publication(
                    [track],
                    {},
                    {1: frame},
                    None,
                    lambda _progress: None,
                    analysis_state=state,
                )

    def test_relocate_action_verifies_and_atomically_publishes_source_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            relocated_path = root / "frame-1.tif"
            pixels = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
            tifffile.imwrite(relocated_path, pixels)
            frame, track = _analyzed_models(Path("missing") / "frame-1.tif")
            state = _current_analysis_state(Path("original") / "frame-1.tif")
            source = replace(
                state.sources[0],
                status=AnalysisSourceStatus.SOURCE_UNAVAILABLE,
                pixel_sha256=sha256(pixels.tobytes(order="C")).hexdigest(),
                pixel_dtype="uint16",
                pixel_shape=(32, 32),
            )
            state = replace(state, requires_fit_validation=True, sources=(source,))
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            window.data_files = [frame]
            window.tracking_results = [track]
            window._active_analysis_state = state
            file_menu = next(
                action.menu()
                for action in window.menuBar().actions()
                if action.menu() is not None
                and action.menu().title().replace("&", "") == "File"
            )
            relocate_action = next(
                (
                    action
                    for action in file_menu.actions()
                    if action.text() == "Relocate Analysis Sources..."
                ),
                None,
            )

            self.assertIsNotNone(relocate_action)
            with (
                patch(
                    "trackleed.gui.session_controller.QFileDialog.getExistingDirectory",
                    return_value=str(root),
                ),
                patch("trackleed.gui.session_controller.QMessageBox.information"),
                patch("trackleed.gui.session_controller.QMessageBox.warning"),
                patch("trackleed.gui.session_controller.QMessageBox.critical"),
            ):
                relocate_action.trigger()
                deadline = time.monotonic() + 5.0
                while window.thread is not None and time.monotonic() < deadline:
                    QTest.qWait(20)

            self.assertIsNone(window.thread)
            self.assertEqual(frame.filepath, relocated_path)
            self.assertEqual(
                window._active_analysis_state.sources[0].status,
                AnalysisSourceStatus.AVAILABLE,
            )


def _analyzed_models(source_path: Path) -> tuple[LeedFile, Track]:
    spot = SpotROI(7, QRect(8, 9, 10, 11), 13.0, 14.0, 2.0, "tracking")
    frame = LeedFile(
        source_path,
        source_path.name,
        False,
        "Cu",
        111,
        "G",
        300.0,
        80.0,
        1,
        None,
        [spot],
        {"center_x": 20.0, "center_y": 21.0},
    )
    frame.post_analysis_results = {"average_radius": 12.5}
    track = Track(7, {1: spot})
    track.analysis_results = {1: {"radius": 12.5, "center_x": 13.0}}
    return frame, track


def _session_contents(
    frame: LeedFile,
    track: Track,
    state: ActiveAnalysisState,
) -> SessionContents:
    return SessionContents([frame], None, [], [track], {}, None, None, state)


def _current_analysis_state(source_path: Path) -> ActiveAnalysisState:
    roi = Rect(8, 9, 10, 11)
    provenance = _fit_provenance(source_path, roi)
    fit_set = FitSet(
        (
            FitObservation(
                7,
                1,
                FitFingerprint("c" * 64),
                FitStatus.FITTED,
                provenance,
                result=_gaussian_result(),
            ),
        )
    )
    source = AnalysisSourceState(
        1,
        source_path,
        Path(source_path.name),
        "a" * 64,
        AnalysisSourceStatus.AVAILABLE,
        pixel_dtype="uint16",
        pixel_shape=(32, 32),
    )
    return ActiveAnalysisState(
        fit_set,
        _geometry_result(),
        False,
        AnalysisResultStatus.CURRENT,
        (source,),
    )


def _fit_provenance(source_path: Path, roi: Rect) -> FitProvenance:
    attempt = RecoveryAttempt(
        0,
        0,
        0,
        0,
        roi,
        RecoveryAttemptStatus.ACCEPTED,
        r_squared=0.99,
        global_center_x=13.0,
        global_center_y=14.0,
    )
    recovery = RecoveryRecord(
        RecoveryStatus.NOT_NEEDED,
        "bounded-v1",
        roi,
        roi,
        (attempt,),
        RecoverySelectionReason.BASELINE_FIT_ACCEPTED,
    )
    return FitProvenance(
        source_path,
        80.0,
        300.0,
        "uint16",
        (32, 32),
        "a" * 64,
        roi,
        False,
        None,
        None,
        PreprocessingConfig(),
        GaussianFitterConfig(),
        FitValidation(),
        recovery,
    )


def _gaussian_result() -> GaussianFitResult:
    return GaussianFitResult(
        parameters=GaussianParameters(2.0, 5.0, 5.0, 1.2, 1.4, 0.1, 0.2),
        errors=GaussianErrors(0.1, 0.2, 0.2, 0.1, 0.1, 0.01, 0.02),
        intensity=GaussianIntensity(10.0, 0.5),
        widths=GaussianWidths(
            2.0,
            2.2,
            0.1,
            0.1,
            2.0,
            2.2,
            0.1,
            0.1,
            2.0,
            2.2,
            0.1,
            0.1,
            2.1,
            0.1,
            1.1,
            0.1,
            0.1,
            0.01,
            5.7,
            0.5,
            0.05,
        ),
        quality=FitQuality(0.99, 1.1),
        covariance_matrix=((1.0, 0.0), (0.0, 1.0)),
        global_center_x=13.0,
        global_center_y=14.0,
    )


def _geometry_result() -> GeometryResult:
    center = PatternCenter(20.0, 21.0, 0.2, 0.3, GeometryMethod.CIRCLE_FIT)
    return GeometryResult(
        GeometryFingerprint("d" * 64),
        GeometryStatus.AVAILABLE,
        GeometryMethod.CIRCLE_FIT,
        (7,),
        (ObservationGeometry(7, 1, 12.5, 0.4),),
        (FrameGeometry(1, GeometryStatus.AVAILABLE, center, 12.5, 0.4),),
    )


def _v1_payload(source_path: Path) -> dict[str, object]:
    spot = {
        "id": 7,
        "roi": {"x": 8, "y": 9, "w": 10, "h": 11},
        "center_x": 13.0,
        "center_y": 14.0,
        "intensity": 2.0,
        "source": "tracking",
    }
    return {
        "version": 1,
        "data_files": [
            {
                "filepath": str(source_path),
                "filename": source_path.name,
                "is_background": False,
                "element": "Cu",
                "surface": 111,
                "mode": "G",
                "temperature": 300.0,
                "energy": 80.0,
                "id": 1,
                "coverage": None,
                "pattern_center": {"center_x": 20.0, "center_y": 21.0},
                "post_analysis_results": {"average_radius": 12.5},
                "spots": [spot],
            }
        ],
        "background_file": None,
        "metadata_masks": [],
        "tracks": [
            {
                "track_id": 7,
                "spots_by_image_id": {"1": spot},
                "analysis_results": {"1": {"radius": 12.5}},
            }
        ],
        "spot_settings": {},
        "analysis_circle": None,
        "analysis_ellipse": None,
    }


if __name__ == "__main__":
    unittest.main()
