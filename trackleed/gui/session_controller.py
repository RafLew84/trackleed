"""Qt adapter for versioned session load, save, and source relocation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from trackleed.core.data_models import LeedFile, Track
from trackleed.gui.tracking_publisher import clone_tracks_for_review
from trackleed.io.image_reader import load_tiff
from trackleed.io.session import (
    SessionContents,
    load_session_from_file,
    save_session_to_file,
)
from trackleed.logic.active_analysis_state import (
    ActiveAnalysisState,
    AnalysisSourceState,
)
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.session_analysis_projection import project_analysis_session
from trackleed.logic.workflows.session_sources import (
    SourceVerificationContext,
    SourceVerificationResult,
    verify_analysis_sources,
)


def run_source_verification_job(
    analysis_state: ActiveAnalysisState,
    relocation_root: Path,
    cancel_event: Event,
    progress_callback: Callable[[int], None],
) -> SourceVerificationResult:
    """Run decoded-pixel verification in the shared worker host."""
    progress_callback(5)
    result = verify_analysis_sources(
        analysis_state,
        SourceVerificationContext(
            load_tiff,
            relocation_root=relocation_root,
            is_cancelled=cancel_event.is_set,
        ),
    )
    progress_callback(100)
    return result


class SessionController:
    def __init__(self, host: Any) -> None:
        self.host = host
        self.quick_save_path = Path("save") / "quick_save.lts.json"

    def save_dialog(
        self,
        tracks_override: list[Track] | None = None,
        analysis_state_override: ActiveAnalysisState | None = None,
    ) -> None:
        if not self.host.data_files:
            QMessageBox.information(
                self.host,
                "No Data",
                "Load a directory or session before saving.",
            )
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self.host,
            "Save Session",
            "",
            "LEEDTrack Session (*.lts.json);;JSON (*.json);;All Files (*)",
        )
        if not path_str:
            return
        self._save_and_report(
            Path(path_str),
            "Session Saved",
            "Save Error",
            "save session",
            tracks_override,
            analysis_state_override,
        )

    def quick_save(self) -> None:
        if not self.host.data_files:
            QMessageBox.information(
                self.host,
                "No Data",
                "Load a directory or session before quick-saving.",
            )
            return
        self._save_and_report(
            self.quick_save_path,
            "Quick Save",
            "Quick Save Error",
            "quick-save session",
        )

    def save(
        self,
        path: Path,
        tracks_override: list[Track] | None = None,
        analysis_state_override: ActiveAnalysisState | None = None,
    ) -> None:
        source_tracks = (
            tracks_override if isinstance(tracks_override, list)
            else self.host.tracking_results
        )
        tracks = clone_tracks_for_review(source_tracks)
        analysis_state = (
            analysis_state_override
            if analysis_state_override is not None
            else self.host._active_analysis_state
        )
        projection = project_analysis_session(
            analysis_state,
            tracks,
            self.host.data_files,
            synchronize_track_spots=isinstance(tracks_override, list),
        )
        save_session_to_file(
            path,
            SessionContents(
                list(projection.files),
                self.host.background_file,
                self.host.metadata_mask_list,
                list(projection.tracks),
                self.host._collect_spot_settings(),
                self.host._analysis_circle_to_dict(),
                self.host._analysis_ellipse_to_dict(),
                projection.state,
            ),
        )

    def _save_and_report(
        self,
        path: Path,
        success_title: str,
        error_title: str,
        action: str,
        tracks_override: list[Track] | None = None,
        analysis_state_override: ActiveAnalysisState | None = None,
    ) -> None:
        try:
            self.save(path, tracks_override, analysis_state_override)
            QMessageBox.information(
                self.host,
                success_title,
                f"Session saved to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self.host,
                error_title,
                f"Failed to {action}:\n{exc}",
            )

    def load_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self.host,
            "Load Session",
            "",
            "LEEDTrack Session (*.lts.json *.json);;All Files (*)",
        )
        if not path_str:
            return
        self._load_and_report(Path(path_str), "Load Error", "load session")

    def quick_load(self) -> None:
        if not self.quick_save_path.exists():
            QMessageBox.information(
                self.host,
                "Quick Load",
                "No quick-save file found.",
            )
            return
        self._load_and_report(
            self.quick_save_path,
            "Quick Load Error",
            "quick-load session",
        )

    def _load_and_report(
        self,
        path: Path,
        error_title: str,
        action: str,
    ) -> None:
        try:
            self.load(path)
        except Exception as exc:
            QMessageBox.critical(
                self.host,
                error_title,
                f"Failed to {action}:\n{exc}",
            )

    def load(self, path: Path) -> None:
        loaded = load_session_from_file(path)
        self.host._active_analysis_state = loaded.analysis_state
        self.host.data_files = loaded.files
        self.host.background_file = loaded.background
        self.host.metadata_mask_list = loaded.masks
        self.host.analysis_circle = self.host._analysis_circle_from_dict(
            loaded.analysis_circle
        )
        self.host.analysis_ellipse = self.host._analysis_ellipse_from_dict(
            loaded.analysis_ellipse
        )
        self.host.tracking_results = loaded.tracks
        session_sources = list(loaded.files)
        if loaded.background is not None:
            session_sources.append(loaded.background)
        missing_files = [
            str(item.filepath)
            for item in session_sources
            if not item.filepath.is_file()
        ]
        if missing_files:
            QMessageBox.warning(
                self.host,
                "Missing Files",
                "Some image sources are unavailable. Their frames and saved "
                "analysis remain available for inspection and export:\n"
                + "\n".join(missing_files),
            )
        self.host.populate_file_list()
        if loaded.spot_settings:
            self.host._apply_spot_settings(loaded.spot_settings)
        first_available = next(
            (index for index, item in enumerate(loaded.files) if item.filepath.is_file()),
            None,
        )
        if first_available is not None:
            self.host.file_list_widget.setCurrentRow(first_available)
            self.host.display_selected_image(
                self.host.file_list_widget.item(first_available),
                None,
            )
        QMessageBox.information(
            self.host,
            "Session Loaded",
            f"Loaded session from:\n{path}",
        )

    def relocate_sources_dialog(self) -> None:
        state = self.host._active_analysis_state
        if not state.sources:
            QMessageBox.information(
                self.host,
                "No Analysis Sources",
                "The active session has no persisted analysis sources.",
            )
            return
        if self.host._tracking_worker_is_active():
            QMessageBox.warning(
                self.host,
                "Task Already Running",
                "Wait for the current background task to finish.",
            )
            return
        root = QFileDialog.getExistingDirectory(
            self.host,
            "Select Relocated Analysis Source Directory",
        )
        if not root:
            return
        cancel_event = Event()
        worker = GenericWorker(
            run_source_verification_job,
            state,
            Path(root),
            cancel_event,
        )
        self.host._launch_worker_with_progress(
            worker,
            "Verifying analysis sources...",
            cancel_event,
            on_finished=self._on_sources_verified,
            on_error=self._on_source_error,
            cancel_label="Source verification",
        )

    def _on_sources_verified(self, result: SourceVerificationResult) -> None:
        if result.failures:
            reasons = ", ".join(item.reason.value for item in result.failures)
            QMessageBox.warning(
                self.host,
                "Analysis Sources Not Verified",
                f"Saved results were retained. Verification failed: {reasons}",
            )
            return
        targets = [self._source_target(source) for source in result.state.sources]
        if any(target is None for target in targets):
            QMessageBox.warning(
                self.host,
                "Analysis Sources Not Verified",
                "Saved results were retained because a source has no active frame.",
            )
            return
        for target, source in zip(targets, result.state.sources, strict=True):
            assert target is not None
            target.filepath = source.path
        self.host._active_analysis_state = result.state
        self.host.populate_file_list()
        QMessageBox.information(
            self.host,
            "Analysis Sources Verified",
            "Relocated sources match the persisted decoded-pixel fingerprints.",
        )

    def _source_target(self, source: AnalysisSourceState) -> LeedFile | None:
        if source.is_background:
            return self.host.background_file
        return next(
            (item for item in self.host.data_files if item.id == source.frame_id),
            None,
        )

    def _on_source_error(self, message: str) -> None:
        QMessageBox.critical(
            self.host,
            "Analysis Source Error",
            f"Source verification failed:\n{message}",
        )
