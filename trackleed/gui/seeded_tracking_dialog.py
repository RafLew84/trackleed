"""Catalog-bound configuration dialog for seeded spot tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import cast

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trackleed.core.data_models import LeedFile
from trackleed.gui.seeded_tracking_adapter import (
    SeededTrackingDialogSettings,
    build_seeded_tracking_request,
)
from trackleed.logic.workflows.seeded_tracking import (
    OpticalFlowConfig,
    OutOfBoundsPolicy,
    SeededTrackingCatalog,
    SeededTrackingRequest,
    SkipAnchorOpticalFlowConfig,
    TrackingPreflightReport,
    default_seeded_tracking_catalog,
)


@dataclass(frozen=True)
class SeededTrackingDialogDecision:
    request: SeededTrackingRequest
    files: tuple[LeedFile, ...]
    settings: SeededTrackingDialogSettings


class OpticalFlowPage(QWidget):
    def __init__(self, config: OpticalFlowConfig) -> None:
        super().__init__()
        self._skip_anchor = isinstance(config, SkipAnchorOpticalFlowConfig)
        layout = QFormLayout(self)
        self.roi_size_spinbox = _integer_spinbox(8, 128, config.roi_size, step=2)
        layout.addRow("ROI size:", self.roi_size_spinbox)
        self.win_size_spinbox = _integer_spinbox(21, 71, config.win_size, step=2)
        layout.addRow("Window size (px):", self.win_size_spinbox)
        self.max_level_spinbox = _integer_spinbox(0, 6, config.max_level)
        layout.addRow("Pyramid levels:", self.max_level_spinbox)
        self.iterations_spinbox = _integer_spinbox(10, 100, config.max_iterations)
        layout.addRow("Maximum iterations:", self.iterations_spinbox)
        self.epsilon_spinbox = _double_spinbox(0.0001, 0.01, config.epsilon, 5)
        layout.addRow("Epsilon:", self.epsilon_spinbox)
        self.min_eigen_spinbox = _double_spinbox(
            0.000001,
            0.001,
            config.min_eigen_threshold,
            6,
        )
        layout.addRow("Minimum eigenvalue:", self.min_eigen_spinbox)
        self.subtract_background_checkbox = QCheckBox("Subtract background before LK")
        self.subtract_background_checkbox.setChecked(config.subtract_background)
        layout.addRow(self.subtract_background_checkbox)
        self.centroid_checkbox = QCheckBox("Centroid-based recentering")
        self.centroid_checkbox.setChecked(config.use_centroid_refinement)
        layout.addRow(self.centroid_checkbox)
        self.centroid_multiplier_spinbox = _double_spinbox(
            1.0,
            3.0,
            config.centroid_patch_multiplier,
            2,
        )
        layout.addRow("Centroid radius multiplier:", self.centroid_multiplier_spinbox)
        self.centroid_threshold_spinbox = _double_spinbox(
            0.1,
            0.9,
            config.centroid_rel_threshold,
            2,
        )
        layout.addRow("Centroid relative threshold:", self.centroid_threshold_spinbox)
        self.centroid_expand_spinbox = _integer_spinbox(
            0,
            3,
            config.centroid_max_expand,
        )
        layout.addRow("Maximum centroid expansions:", self.centroid_expand_spinbox)
        if isinstance(config, SkipAnchorOpticalFlowConfig):
            self._add_skip_anchor_fields(layout, config)

    def _add_skip_anchor_fields(
        self,
        layout: QFormLayout,
        config: SkipAnchorOpticalFlowConfig,
    ) -> None:
        self.error_threshold_spinbox = _integer_spinbox(
            10,
            80,
            round(config.error_threshold),
        )
        layout.addRow("Error threshold (LK):", self.error_threshold_spinbox)
        self.forward_backward_checkbox = QCheckBox("Forward-backward check")
        self.forward_backward_checkbox.setChecked(config.use_forward_backward)
        layout.addRow(self.forward_backward_checkbox)
        self.forward_backward_spinbox = _double_spinbox(
            2.0,
            20.0,
            config.forward_backward_threshold,
            1,
        )
        layout.addRow("FB threshold (px):", self.forward_backward_spinbox)
        self.max_skip_spinbox = _integer_spinbox(0, 1000, config.max_skip_frames)
        layout.addRow("Maximum skipped frames:", self.max_skip_spinbox)
        self.out_of_bounds_combo = QComboBox()
        self.out_of_bounds_combo.addItem("Ignore (skip)", OutOfBoundsPolicy.SKIP)
        self.out_of_bounds_combo.addItem("Mark as lost", OutOfBoundsPolicy.MARK_LOST)
        self.out_of_bounds_combo.setCurrentIndex(
            max(self.out_of_bounds_combo.findData(config.out_of_bounds), 0)
        )
        layout.addRow("Out-of-bounds:", self.out_of_bounds_combo)
        self.ncc_checkbox = QCheckBox("Enable NCC test")
        self.ncc_checkbox.setChecked(config.use_ncc_check)
        layout.addRow(self.ncc_checkbox)
        self.ncc_threshold_spinbox = _double_spinbox(
            0.3,
            0.95,
            config.ncc_threshold,
            2,
        )
        layout.addRow("NCC threshold:", self.ncc_threshold_spinbox)
        self.ncc_patch_spinbox = _integer_spinbox(3, 6, config.ncc_patch_radius)
        layout.addRow("NCC patch radius:", self.ncc_patch_spinbox)

    def config_snapshot(self) -> OpticalFlowConfig | SkipAnchorOpticalFlowConfig:
        common = OpticalFlowConfig(
            roi_size=self.roi_size_spinbox.value(),
            win_size=self.win_size_spinbox.value(),
            max_level=self.max_level_spinbox.value(),
            max_iterations=self.iterations_spinbox.value(),
            epsilon=self.epsilon_spinbox.value(),
            min_eigen_threshold=self.min_eigen_spinbox.value(),
            use_centroid_refinement=self.centroid_checkbox.isChecked(),
            centroid_patch_multiplier=self.centroid_multiplier_spinbox.value(),
            centroid_rel_threshold=self.centroid_threshold_spinbox.value(),
            centroid_max_expand=self.centroid_expand_spinbox.value(),
            subtract_background=self.subtract_background_checkbox.isChecked(),
        )
        if not self._skip_anchor:
            return common
        return SkipAnchorOpticalFlowConfig(
            **asdict(common),
            error_threshold=self.error_threshold_spinbox.value(),
            forward_backward_threshold=self.forward_backward_spinbox.value(),
            use_forward_backward=self.forward_backward_checkbox.isChecked(),
            max_skip_frames=self.max_skip_spinbox.value(),
            out_of_bounds=self.out_of_bounds_combo.currentData(),
            use_ncc_check=self.ncc_checkbox.isChecked(),
            ncc_threshold=self.ncc_threshold_spinbox.value(),
            ncc_patch_radius=self.ncc_patch_spinbox.value(),
        )


class SeededTrackingDialog(QDialog):
    def __init__(
        self,
        *,
        reference_file: LeedFile | None,
        series_files: tuple[LeedFile, ...],
        background_file: LeedFile | None,
        settings: SeededTrackingDialogSettings,
        catalog: SeededTrackingCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Seeded Spot Tracking")
        self._reference_file = reference_file
        self._files = series_files
        self._background_file = background_file
        self._settings = settings
        self._catalog = catalog or default_seeded_tracking_catalog()
        self._method_ids: list[str] = []
        self.decision: SeededTrackingDialogDecision | None = None
        self.preflight_report: TrackingPreflightReport | None = None
        layout = QVBoxLayout(self)
        self.method_tabs = QTabWidget()
        self.optical_flow_page = OpticalFlowPage(settings.optical_flow)
        self.skip_anchor_page = OpticalFlowPage(settings.skip_anchor)
        pages = {
            "optical_flow": self.optical_flow_page,
            "optical_flow_skip_anchor": self.skip_anchor_page,
        }
        for method in self._catalog.methods:
            page = pages.get(method.method_id)
            if page is None:
                continue
            self._method_ids.append(method.method_id)
            self.method_tabs.addTab(page, method.display_name)
        selected_index = (
            self._method_ids.index(settings.selected_method_id)
            if settings.selected_method_id in self._method_ids
            else 0
        )
        self.method_tabs.setCurrentIndex(selected_index)
        layout.addWidget(self.method_tabs)
        self.preflight_label = QLabel("Choose a tracking method and reference seeds.")
        self.preflight_label.setWordWrap(True)
        layout.addWidget(self.preflight_label)
        actions = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.run_button = QPushButton("Run Seeded Tracking")
        self.cancel_button.clicked.connect(self.reject)
        self.run_button.clicked.connect(self._preflight_and_accept)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

    def _preflight_and_accept(self) -> None:
        method_id = self._method_ids[self.method_tabs.currentIndex()]
        settings = replace(
            self._settings,
            selected_method_id=method_id,
            optical_flow=cast(
                OpticalFlowConfig,
                self.optical_flow_page.config_snapshot(),
            ),
            skip_anchor=cast(
                SkipAnchorOpticalFlowConfig,
                self.skip_anchor_page.config_snapshot(),
            ),
        )
        config = (
            settings.optical_flow
            if method_id == "optical_flow"
            else settings.skip_anchor
        )
        try:
            request = build_seeded_tracking_request(
                files=self._files,
                reference_file=self._reference_file,
                method_id=method_id,
                config=config,
                background_file=self._background_file,
            )
        except (TypeError, ValueError) as exc:
            self.preflight_label.setText(f"Preflight blocked:\n[invalid_request] {exc}")
            return
        report = self._catalog.preflight(request)
        self.preflight_report = report
        if not report.ready:
            self.preflight_label.setText(
                "Preflight blocked:\n"
                + "\n".join(
                    f"[{issue.code}] {issue.message}" for issue in report.issues
                )
            )
            return
        self.preflight_label.setText(
            f"Ready - {self._catalog.method(method_id).display_name} can run."
        )
        self.decision = SeededTrackingDialogDecision(request, self._files, settings)
        self.accept()


def _integer_spinbox(
    minimum: int,
    maximum: int,
    value: int,
    *,
    step: int = 1,
) -> QSpinBox:
    spinbox = QSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    return spinbox


def _double_spinbox(
    minimum: float,
    maximum: float,
    value: float,
    decimals: int,
) -> QDoubleSpinBox:
    spinbox = QDoubleSpinBox()
    spinbox.setDecimals(decimals)
    spinbox.setRange(minimum, maximum)
    spinbox.setValue(value)
    return spinbox
