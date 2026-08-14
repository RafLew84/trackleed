"""Configuration dialog bound to the headless spot-detection catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trackleed.core.data_models import LeedFile
from trackleed.gui.spot_detection_adapter import (
    SpotDetectionDialogSettings,
    build_detection_request,
)
from trackleed.logic.workflows.spot_detection import (
    ClassicDetectionConfig,
    DetectionRequest,
    PreflightReport,
    SpotDetectionCatalog,
    YoloDetectionConfig,
    default_spot_detection_catalog,
)


class SpotDetectionScope(Enum):
    CURRENT_FRAME = "current_frame"
    ALL_FRAMES = "all_frames"


@dataclass(frozen=True)
class SpotDetectionDialogContext:
    current_file: LeedFile | None
    series_files: tuple[LeedFile, ...]
    background_file: LeedFile | None = None
    metadata_masks: tuple[QRect, ...] = ()
    analysis_circle: tuple[int, int, int] | None = None
    analysis_ellipse: tuple[int, int, int, int, float] | None = None


@dataclass(frozen=True)
class SpotDetectionDialogDecision:
    request: DetectionRequest
    files: tuple[LeedFile, ...]
    scope: SpotDetectionScope
    settings: SpotDetectionDialogSettings


class ClassicDetectionPage(QWidget):
    def __init__(self, config: ClassicDetectionConfig) -> None:
        super().__init__()
        layout = QFormLayout(self)
        self.wth_radius_spinbox = QSpinBox()
        self.wth_radius_spinbox.setRange(1, 50)
        self.wth_radius_spinbox.setValue(config.wth_radius)
        layout.addRow("White top-hat radius:", self.wth_radius_spinbox)
        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setRange(1.0, 100.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(config.blur_sigma)
        layout.addRow("Blur sigma:", self.blur_sigma_spinbox)
        self.min_distance_spinbox = QSpinBox()
        self.min_distance_spinbox.setRange(1, 100)
        self.min_distance_spinbox.setValue(config.min_distance)
        layout.addRow("Minimum peak distance:", self.min_distance_spinbox)
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.01, 1.0)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(config.threshold_rel)
        layout.addRow("Relative threshold:", self.threshold_spinbox)
        self.roi_size_spinbox = QSpinBox()
        self.roi_size_spinbox.setRange(8, 128)
        self.roi_size_spinbox.setSingleStep(2)
        self.roi_size_spinbox.setValue(config.roi_size)
        layout.addRow("ROI size:", self.roi_size_spinbox)

    def config_snapshot(self) -> ClassicDetectionConfig:
        return ClassicDetectionConfig(
            blur_sigma=self.blur_sigma_spinbox.value(),
            wth_radius=self.wth_radius_spinbox.value(),
            min_distance=self.min_distance_spinbox.value(),
            threshold_rel=self.threshold_spinbox.value(),
            roi_size=self.roi_size_spinbox.value(),
        )


class YoloDetectionPage(QWidget):
    def __init__(self, config: YoloDetectionConfig) -> None:
        super().__init__()
        layout = QFormLayout(self)
        self.model_path_edit = QLineEdit(
            str(config.model_path) if config.model_path else ""
        )
        layout.addRow("Model:", self.model_path_edit)
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.01, 0.99)
        self.confidence_spinbox.setSingleStep(0.01)
        self.confidence_spinbox.setValue(config.confidence)
        layout.addRow("Confidence:", self.confidence_spinbox)
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.05, 0.95)
        self.iou_spinbox.setSingleStep(0.01)
        self.iou_spinbox.setValue(config.iou)
        layout.addRow("NMS IoU:", self.iou_spinbox)
        self.max_detections_spinbox = QSpinBox()
        self.max_detections_spinbox.setRange(1, 9999)
        self.max_detections_spinbox.setValue(config.max_detections)
        layout.addRow("Max detections:", self.max_detections_spinbox)
        self.bbox_margin_spinbox = QSpinBox()
        self.bbox_margin_spinbox.setRange(-50, 50)
        self.bbox_margin_spinbox.setValue(config.bbox_margin_px)
        layout.addRow("BBox margin (px):", self.bbox_margin_spinbox)
        self.subtract_background_checkbox = QCheckBox("Subtract background")
        self.subtract_background_checkbox.setChecked(config.subtract_background)
        layout.addRow(self.subtract_background_checkbox)
        self.device_combo = QComboBox()
        for label, device in (("Auto", "auto"), ("CPU", "cpu"), ("GPU", "gpu")):
            self.device_combo.addItem(label, device)
        device_id = {"cuda": "gpu"}.get(config.device.lower(), config.device.lower())
        device_index = self.device_combo.findData(device_id)
        self.device_combo.setCurrentIndex(max(device_index, 0))
        layout.addRow("Device:", self.device_combo)

    def config_snapshot(self) -> YoloDetectionConfig:
        model_text = self.model_path_edit.text().strip()
        return YoloDetectionConfig(
            model_path=Path(model_text) if model_text else None,
            confidence=self.confidence_spinbox.value(),
            iou=self.iou_spinbox.value(),
            max_detections=self.max_detections_spinbox.value(),
            bbox_margin_px=self.bbox_margin_spinbox.value(),
            subtract_background=self.subtract_background_checkbox.isChecked(),
            device=str(self.device_combo.currentData()),
        )


class SpotDetectionDialog(QDialog):
    def __init__(
        self,
        *,
        context: SpotDetectionDialogContext,
        settings: SpotDetectionDialogSettings,
        catalog: SpotDetectionCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spot Detection")
        self._context = context
        self._settings = settings
        self._catalog = catalog or default_spot_detection_catalog()
        self._method_ids: list[str] = []
        self.decision: SpotDetectionDialogDecision | None = None
        self.preflight_report: PreflightReport | None = None

        layout = QVBoxLayout(self)
        self.method_tabs = QTabWidget()
        self.classic_page = ClassicDetectionPage(settings.classic)
        self.yolo_page = YoloDetectionPage(settings.yolo)
        pages_by_method_id = {
            "classic": self.classic_page,
            "yolo": self.yolo_page,
        }
        for method in self._catalog.methods:
            page = pages_by_method_id.get(method.method_id)
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

        self.preflight_label = QLabel("Choose a method and run scope.")
        self.preflight_label.setWordWrap(True)
        layout.addWidget(self.preflight_label)

        action_layout = QHBoxLayout()
        self.detect_current_button = QPushButton("Detect Current Frame")
        self.detect_all_button = QPushButton("Detect All Frames")
        self.cancel_button = QPushButton("Cancel")
        self.detect_current_button.clicked.connect(
            lambda: self._preflight_and_accept(SpotDetectionScope.CURRENT_FRAME)
        )
        self.detect_all_button.clicked.connect(
            lambda: self._preflight_and_accept(SpotDetectionScope.ALL_FRAMES)
        )
        self.cancel_button.clicked.connect(self.reject)
        action_layout.addWidget(self.cancel_button)
        action_layout.addWidget(self.detect_current_button)
        action_layout.addWidget(self.detect_all_button)
        layout.addLayout(action_layout)

    def _preflight_and_accept(self, scope: SpotDetectionScope) -> None:
        method_id = self._method_ids[self.method_tabs.currentIndex()]
        settings = replace(
            self._settings,
            selected_method_id=method_id,
            classic=self.classic_page.config_snapshot(),
            yolo=self.yolo_page.config_snapshot(),
        )
        files = (
            (self._context.current_file,)
            if scope is SpotDetectionScope.CURRENT_FRAME
            and self._context.current_file is not None
            else self._context.series_files
            if scope is SpotDetectionScope.ALL_FRAMES
            else ()
        )
        request = build_detection_request(
            files=files,
            spot_settings=settings.to_mapping(),
            background_file=self._context.background_file,
            metadata_masks=self._context.metadata_masks,
            analysis_circle=self._context.analysis_circle,
            analysis_ellipse=self._context.analysis_ellipse,
        )
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
        self.decision = SpotDetectionDialogDecision(request, files, scope, settings)
        self.accept()
