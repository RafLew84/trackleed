"""Catalog-bound configuration dialog for detection linking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from PyQt6.QtWidgets import (
    QDialog,
    QCheckBox,
    QComboBox,
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
    DirectionalLinkingConfig,
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
        self._settings = settings
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
        settings = replace(self._settings, hungarian=config)
        self.decision = DetectionLinkingDialogDecision(request, self._files, settings)
        self.accept()


class DirectionalDetectionLinkingDialog(QDialog):
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
        self._settings = settings
        self._base_config = settings.directional
        self._catalog = catalog or default_detection_linking_catalog()
        self.decision: DetectionLinkingDialogDecision | None = None
        self.preflight_report = DetectionLinkingPreflightReport("directional")
        self.setWindowTitle("Detection Linking (Directional)")
        self._build_ui(settings.directional)

    def _build_ui(self, config: DirectionalLinkingConfig) -> None:
        layout = QVBoxLayout(self)
        description = QLabel(
            "Associate independent detections using radial LEED motion constraints."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self._add_center_controls(form, config)
        self._add_assignment_controls(form, config)
        self._add_lifecycle_controls(form, config)
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

    def _add_center_controls(
        self,
        form: QFormLayout,
        config: DirectionalLinkingConfig,
    ) -> None:
        self.center_mode_combo = QComboBox()
        self.center_mode_combo.addItem("Auto (RANSAC)", "auto")
        self.center_mode_combo.addItem("Manual", "manual")
        self.center_mode_combo.setCurrentIndex(
            1 if config.center_mode == "manual" else 0
        )
        form.addRow("Center mode:", self.center_mode_combo)
        center = config.manual_center or (0.0, 0.0)
        self.center_x_spinbox = QDoubleSpinBox()
        self.center_x_spinbox.setRange(0.0, 10000.0)
        self.center_x_spinbox.setValue(center[0])
        self.center_y_spinbox = QDoubleSpinBox()
        self.center_y_spinbox.setRange(0.0, 10000.0)
        self.center_y_spinbox.setValue(center[1])
        row = QHBoxLayout()
        row.addWidget(self.center_x_spinbox)
        row.addWidget(self.center_y_spinbox)
        form.addRow("Manual center Cx/Cy:", row)
        self.center_mode_combo.currentIndexChanged.connect(
            self._update_center_controls
        )
        self._update_center_controls()

    def _add_assignment_controls(
        self,
        form: QFormLayout,
        config: DirectionalLinkingConfig,
    ) -> None:
        self.direction_mode_combo = QComboBox()
        for label, value in (
            ("Auto", "auto"),
            ("Inward", "inward"),
            ("Outward", "outward"),
        ):
            self.direction_mode_combo.addItem(label, value)
        direction_index = {"auto": 0, "inward": 1, "outward": 2}
        self.direction_mode_combo.setCurrentIndex(
            direction_index.get(config.direction_mode, 0)
        )
        form.addRow("Direction:", self.direction_mode_combo)
        self.lateral_tolerance_spinbox = _double_spinbox(
            config.lateral_angle_tolerance_deg,
            0.0,
            90.0,
        )
        form.addRow("Lateral tolerance (deg):", self.lateral_tolerance_spinbox)
        self.ring_bin_spinbox = _double_spinbox(config.ring_bin_px, 0.1, 500.0)
        form.addRow("Ring bin (px):", self.ring_bin_spinbox)
        self.base_distance_spinbox = _double_spinbox(
            config.base_max_distance_px,
            0.1,
            500.0,
        )
        form.addRow("Base max distance (px):", self.base_distance_spinbox)
        self.dynamic_distance_checkbox = QCheckBox("Dynamic max distance")
        self.dynamic_distance_checkbox.setChecked(config.dynamic_max_distance)
        form.addRow(self.dynamic_distance_checkbox)
        self.dynamic_margin_spinbox = _double_spinbox(
            config.dynamic_margin_px,
            0.0,
            200.0,
        )
        form.addRow("Dynamic margin (px):", self.dynamic_margin_spinbox)

    def _add_lifecycle_controls(
        self,
        form: QFormLayout,
        config: DirectionalLinkingConfig,
    ) -> None:
        self.max_gap_spinbox = _integer_spinbox(config.max_gap, 0, 9999)
        form.addRow("Max gap (frames):", self.max_gap_spinbox)
        self.max_misses_spinbox = _integer_spinbox(config.max_misses, 0, 9999)
        form.addRow("Max misses:", self.max_misses_spinbox)
        self.min_track_length_spinbox = _integer_spinbox(
            config.min_track_length,
            1,
            9999,
        )
        form.addRow("Min track length:", self.min_track_length_spinbox)

    def _update_center_controls(self) -> None:
        manual = self.center_mode_combo.currentData() == "manual"
        self.center_x_spinbox.setEnabled(manual)
        self.center_y_spinbox.setEnabled(manual)

    def _accept_request(self) -> None:
        config = self._config_from_controls()
        try:
            request = build_detection_linking_request(
                self._files,
                config,
                method_id="directional",
            )
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
        settings = replace(self._settings, directional=config)
        self.decision = DetectionLinkingDialogDecision(request, self._files, settings)
        self.accept()

    def _config_from_controls(self) -> DirectionalLinkingConfig:
        center_mode = str(self.center_mode_combo.currentData())
        manual_center = (
            self.center_x_spinbox.value(),
            self.center_y_spinbox.value(),
        )
        return replace(
            self._base_config,
            center_mode=center_mode,
            manual_center=manual_center,
            direction_mode=str(self.direction_mode_combo.currentData()),
            lateral_angle_tolerance_deg=self.lateral_tolerance_spinbox.value(),
            ring_bin_px=self.ring_bin_spinbox.value(),
            base_max_distance_px=self.base_distance_spinbox.value(),
            dynamic_max_distance=self.dynamic_distance_checkbox.isChecked(),
            dynamic_margin_px=self.dynamic_margin_spinbox.value(),
            max_gap=self.max_gap_spinbox.value(),
            max_misses=self.max_misses_spinbox.value(),
            min_track_length=self.min_track_length_spinbox.value(),
        )


def _double_spinbox(value: float, minimum: float, maximum: float) -> QDoubleSpinBox:
    spinbox = QDoubleSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setDecimals(1)
    spinbox.setValue(value)
    return spinbox


def _integer_spinbox(value: int, minimum: int, maximum: int) -> QSpinBox:
    spinbox = QSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setValue(value)
    return spinbox
