"""Catalog-bound configuration dialog for detection linking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from trackleed.core.data_models import LeedFile
from trackleed.gui.detection_linking_adapter import (
    DetectionLinkingDialogSettings,
    build_detection_linking_request,
)
from trackleed.logic.workflows.detection_linking import (
    DetectionLinkingCatalog,
    DetectionLinkingPreflightReport,
    DetectionLinkingRequest,
    HungarianLinkingConfig,
    default_detection_linking_catalog,
)


@dataclass(frozen=True)
class DetectionLinkingDialogDecision:
    request: DetectionLinkingRequest
    files: tuple[LeedFile, ...]
    settings: DetectionLinkingDialogSettings


class DetectionLinkingDialog(QDialog):
    def __init__(
        self,
        *,
        files: tuple[LeedFile, ...],
        settings: DetectionLinkingDialogSettings,
        catalog: DetectionLinkingCatalog | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._files = files
        self._catalog = catalog or default_detection_linking_catalog()
        self.decision: DetectionLinkingDialogDecision | None = None
        self.preflight_report = DetectionLinkingPreflightReport("hungarian")
        self.setWindowTitle("Detection Linking (Hungarian)")
        self._build_ui(settings.hungarian)

    def _build_ui(self, config: HungarianLinkingConfig) -> None:
        layout = QVBoxLayout(self)
        description = QLabel(
            "Associate independent frame detections into complete tracks."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.max_distance_spinbox = QDoubleSpinBox()
        self.max_distance_spinbox.setRange(0.1, 10000.0)
        self.max_distance_spinbox.setDecimals(1)
        self.max_distance_spinbox.setValue(config.max_distance_px)
        form.addRow("Max distance (px):", self.max_distance_spinbox)

        self.max_gap_spinbox = QSpinBox()
        self.max_gap_spinbox.setRange(0, 9999)
        self.max_gap_spinbox.setValue(config.max_gap)
        form.addRow("Max gap (frames):", self.max_gap_spinbox)

        self.min_track_length_spinbox = QSpinBox()
        self.min_track_length_spinbox.setRange(1, 9999)
        self.min_track_length_spinbox.setValue(config.min_track_length)
        form.addRow("Min track length:", self.min_track_length_spinbox)
        layout.addLayout(form)

        self.preflight_label = QLabel()
        self.preflight_label.setWordWrap(True)
        layout.addWidget(self.preflight_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.run_button = QPushButton("Run Detection Linking")
        self.cancel_button = QPushButton("Cancel")
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.run_button.clicked.connect(self._accept_request)
        self.cancel_button.clicked.connect(self.reject)

    def _accept_request(self) -> None:
        config = HungarianLinkingConfig(
            max_distance_px=self.max_distance_spinbox.value(),
            max_gap=self.max_gap_spinbox.value(),
            min_track_length=self.min_track_length_spinbox.value(),
        )
        try:
            request = build_detection_linking_request(self._files, config)
            report = self._catalog.preflight(request)
        except (TypeError, ValueError) as exc:
            self.preflight_label.setText(f"[invalid_request] {exc}")
            return
        self.preflight_report = report
        if not report.ready:
            self.preflight_label.setText(
                "\n".join(f"[{issue.code}] {issue.message}" for issue in report.issues)
            )
            return
        self.preflight_label.setText("Ready")
        settings = DetectionLinkingDialogSettings(config)
        self.decision = DetectionLinkingDialogDecision(request, self._files, settings)
        self.accept()
