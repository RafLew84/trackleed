from __future__ import annotations

# stal/gui/main_window.py

from pathlib import Path
from dataclasses import dataclass

from typing import Any, List
from threading import Event
import numpy as np
import shutil
from PyQt6.QtCore import Qt, QPointF, QRect, QRectF, QThread, QPoint
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QPen, QBrush, QColor, QActionGroup , QIcon
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QGroupBox,
                             QLabel, QFileDialog, QListWidgetItem, QMessageBox, QApplication,
                             QDialog, QDockWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
                             QPushButton, QCheckBox, QToolBar, QComboBox, QProgressDialog,
                             QScrollArea, QStackedWidget, QSizePolicy, QInputDialog, QLineEdit)

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.file_parser import parse_directory
from trackleed.io.image_reader import load_tiff
from trackleed.gui.mask_dialog import MaskDialog
from trackleed.analysis.spot_finding import find_spots
from trackleed.analysis.dl_spot_finder import detect_spots_with_model, DLSpotDetectionError
from trackleed.analysis.yolo_detector import detect_spots_with_yolo, YOLODetectionError, YOLOSettings
from trackleed.analysis.tapir_backend import (
    discover_online_tapir_checkpoints,
    format_tapir_environment_report,
    probe_tapir_environment,
)
from trackleed.analysis.locotrack_backend import (
    format_locotrack_environment_report,
    probe_locotrack_environment,
)
from trackleed.analysis.cotracker_backend import (
    format_cotracker_environment_report,
    probe_cotracker_environment,
)
from trackleed.analysis.tapnext_backend import (
    format_tapnext_environment_report,
    probe_tapnext_environment,
)
from trackleed.analysis.trackonr_backend import (
    format_trackonr_environment_report,
    probe_trackonr_environment,
)
from trackleed.analysis.pipspp_backend import (
    format_pipspp_environment_report,
    probe_pipspp_environment,
)
from trackleed.analysis.mftiq_backend import (
    format_mftiq_environment_report,
    probe_mftiq_environment,
)
from trackleed.analysis.cutie_backend import (
    format_cutie_environment_report,
    probe_cutie_environment,
)
from trackleed.analysis.xmem2_backend import (
    format_xmem2_environment_report,
    probe_xmem2_environment,
)
from trackleed.analysis.tomp_backend import (
    expected_tomp_checkpoint_name_for_variant,
    format_tomp_environment_report,
    probe_tomp_environment,
    select_default_tamos_checkpoint_for_variant,
    select_default_tomp_checkpoint_for_variant,
)
from trackleed.analysis.sam2_backend import (
    format_sam2_environment_report,
    probe_sam2_environment,
)
from trackleed.io.session import load_session_from_file, save_session_to_file
from trackleed.gui.circle_mask_dialog import CircleMaskDialog
from trackleed.gui.ellipse_mask_dialog import EllipseMaskDialog
from trackleed.gui.polygon_mask_dialog import PolygonMaskDialog
from trackleed.io.gif_loader import extract_gif_frames
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.gui.tracking_dialog import TrackingDialog
from trackleed.logic.generic_worker import GenericWorker
from trackleed.analysis.tracking import (
    track_spots_optical_flow,
    track_spots_optical_flow_skip_anchor,
    track_spots_hungarian,
    track_spots_hungarian_directional,
    track_spots_tapir_offline,
    track_spots_tapir_online,
    track_spots_locotrack_offline,
    track_spots_cotracker3_offline,
    track_spots_tapnext_offline,
    track_spots_trackonr_offline,
    track_spots_pipspp_offline,
    track_spots_mftiq_offline,
    track_spots_tomp_offline,
    track_spots_cutie_offline,
    track_spots_xmem2_offline,
    track_spots_sam2_offline,
)

from skimage.util import img_as_float
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.morphology import white_tophat, disk


@dataclass
class DLSpotSettings:
    model_path: str | None = None
    confidence: float = 0.5
    min_area_px: int = 10
    max_area_px: int = 10000
    bbox_margin_px: int = 4
    dilate_radius_px: int = 0


@dataclass
class YOLOSpotSettings:
    model_path: str | None = None
    confidence: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    bbox_margin_px: int = 2


@dataclass
class TapirTrackingSettings:
    checkpoint_path: str | None = None
    runtime_mode: str = "offline"
    device: str = "auto"
    inference_resolution: int = 256
    query_chunk_size: int = 64
    window_size: int = 64
    window_overlap: int = 8
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_uncertainty_gating: bool = True
    expected_dist_threshold: float = 0.5
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class LocotrackTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    backend_mode: str = "subprocess"
    model_size: str = "auto"
    device: str = "auto"
    inference_resolution: int = 256
    query_chunk_size: int = 64
    subprocess_timeout_sec: float = 1800.0
    window_size: int = 64
    window_overlap: int = 8
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_uncertainty_gating: bool = True
    expected_dist_threshold: float = 0.5
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class CotrackerTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    device: str = "auto"
    inference_resolution: int = 256
    window_len: int = 60
    window_size: int = 64
    window_overlap: int = 8
    use_visibility_gating: bool = True
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class TapnextTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    device: str = "auto"
    inference_resolution: int = 256
    autocast: bool = False
    window_size: int = 64
    window_overlap: int = 8
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.5
    use_certainty_gating: bool = False
    certainty_threshold: float = 0.5
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class TrackonrTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    backend_mode: str = "subprocess"
    device: str = "auto"
    inference_resolution: tuple[int, int] = (384, 512)
    support_grid_size: int = 20
    subprocess_timeout_sec: float = 1800.0
    window_size: int = 64
    window_overlap: int = 8
    use_visibility_gating: bool = True
    visibility_threshold: float = 0.8
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class PipsppTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    backend_mode: str = "subprocess"
    device: str = "auto"
    inference_resolution: tuple[int, int] = (256, 384)
    iters: int = 4
    beautify: bool = True
    use_inbounds_visibility: bool = True
    subprocess_timeout_sec: float = 1800.0
    window_size: int = 64
    window_overlap: int = 8
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class MftiqTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    config_path: str | None = None
    device: str = "auto"
    backend_mode: str = "subprocess"
    inference_resolution: tuple[int, int] | None = None
    occlusion_threshold: float = 0.5
    subprocess_timeout_sec: float = 1800.0
    window_size: int = 64
    window_overlap: int = 8
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0
    recenter_mode: str = "centroid"
    recenter_patch_multiplier: float = 1.25
    centroid_rel_threshold: float = 0.35
    centroid_max_shift_ratio: float = 0.45


@dataclass
class Sam2TrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    config_path: str | None = None
    device: str = "auto"
    backend_mode: str = "subprocess"
    prompt_mode: str = "point+bbox"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | None = None
    window_size: int = 64
    window_overlap: int = 8
    initial_bbox_padding_px: float = 12.0
    mask_threshold: float = 0.5
    min_mask_area: float = 1.0
    max_mask_area: float = 0.0
    use_vos_optimized: bool = False
    apply_postprocessing: bool = True
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False
    subprocess_timeout_sec: float = 1800.0
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    max_mask_area_ratio_change: float = 0.0
    max_bbox_aspect_ratio_change: float = 0.0
    max_mask_components: int = 0
    max_bbox_center_jump_px: float = 0.0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0


@dataclass
class CutieTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    device: str = "auto"
    backend_mode: str = "direct"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | None = None
    window_size: int = 64
    window_overlap: int = 8
    initial_bbox_padding_px: float = 12.0
    initial_mask_mode: str = "threshold"
    initial_mask_threshold_rel: float = 0.7
    initial_mask_disk_radius_ratio: float = 0.35
    mem_every: int | None = None
    subprocess_timeout_sec: float = 1800.0
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    max_mask_area_ratio_change: float = 0.0
    max_bbox_aspect_ratio_change: float = 0.0
    max_bbox_center_jump_px: float = 0.0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0


@dataclass
class XMem2TrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    device: str = "auto"
    backend_mode: str = "subprocess"
    representative_point_mode: str = "mask_centroid"
    inference_resolution: int | None = None
    window_size: int = 64
    window_overlap: int = 8
    initial_bbox_padding_px: float = 12.0
    initial_mask_mode: str = "threshold"
    initial_mask_threshold_rel: float = 0.7
    initial_mask_disk_radius_ratio: float = 0.35
    mem_every: int | None = None
    subprocess_timeout_sec: float = 1800.0
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    max_mask_area_ratio_change: float = 0.0
    max_bbox_aspect_ratio_change: float = 0.0
    max_bbox_center_jump_px: float = 0.0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0


@dataclass
class TompTrackingSettings:
    repo_path: str | None = None
    checkpoint_path: str | None = None
    python_executable: str | None = None
    backend_mode: str = "subprocess"
    model_variant: str = "tomp50"
    device: str = "auto"
    representative_point_mode: str = "bbox_center"
    inference_resolution: int | None = None
    window_size: int = 64
    window_overlap: int = 8
    initial_bbox_padding_px: float = 0.0
    subprocess_timeout_sec: float = 1800.0
    keep_invisible_points_as_gaps: bool = True
    max_step_distance_px: float = 0.0
    max_reacquisition_gap: int = 0
    terminate_after_invisible_frames: int = 0
    center_mode: str = "auto"
    manual_center_x: float = 0.0
    manual_center_y: float = 0.0
    direction_mode: str = "none"
    lateral_angle_tolerance_deg: float = 20.0
    max_radial_jump_px: float = 0.0


class MainWindow(QMainWindow):
    """
    The main window class for the STAL application.
    It orchestrates the UI and connects user actions to the backend logic.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEEDTrack - Spot Tracking")
        self.setGeometry(100, 100, 1200, 800)

        # --- Store data ---
        self.data_files: list[LeedFile] = []
        self.background_file: LeedFile | None = None
        self.metadata_mask_list: list[QRect] = []
        self.analysis_circle: tuple[int, int, int] | None = None  # (cx, cy, r)
        self.analysis_ellipse: tuple[int, int, int, int, float] | None = None  # (cx, cy, rx, ry, angle_deg)
        self.tracking_results: List[Track] = []
        self.dl_settings = DLSpotSettings()
        self.yolo_settings = YOLOSpotSettings()
        self.tapir_settings = TapirTrackingSettings()
        self.locotrack_settings = LocotrackTrackingSettings()
        self.cotracker_settings = CotrackerTrackingSettings()
        self.tapnext_settings = TapnextTrackingSettings()
        self.trackonr_settings = TrackonrTrackingSettings()
        self.pipspp_settings = PipsppTrackingSettings()
        self.mftiq_settings = MftiqTrackingSettings()
        self.tomp_settings = TompTrackingSettings()
        self.sam2_settings = Sam2TrackingSettings()
        self.cutie_settings = CutieTrackingSettings()
        self.xmem2_settings = XMem2TrackingSettings()
        self.current_spot_backend = "Classic"
        self.tracking_models_dir = Path("tracking_models")
        self.dl_models_dir = self.tracking_models_dir
        self.yolo_models_dir = self.tracking_models_dir / "yolo"
        self.tapir_models_dir = self.tracking_models_dir / "tapir"
        self.locotrack_models_dir = self.tracking_models_dir / "locotrack"
        self.cotracker_models_dir = self.tracking_models_dir / "cotracker3"
        self.tapnext_models_dir = self.tracking_models_dir / "tapnext"
        self.trackonr_models_dir = self.tracking_models_dir / "trackonr"
        self.pipspp_models_dir = self.tracking_models_dir / "pipspp"
        self.mftiq_models_dir = self.tracking_models_dir / "mftiq"
        self.tomp_models_dir = self.tracking_models_dir / "tomp"
        self.tamos_models_dir = self.tracking_models_dir / "tamos"
        self.cutie_models_dir = self.tracking_models_dir / "cutie"
        self.xmem2_models_dir = self.tracking_models_dir / "xmem2"
        self.sam2_models_dir = self.tracking_models_dir / "sam2"
        default_locotrack_repo = Path(__file__).resolve().parents[3] / "locotrack"
        if default_locotrack_repo.exists():
            self.locotrack_settings.repo_path = str(default_locotrack_repo.resolve())
        default_cotracker_repo = Path(__file__).resolve().parents[3] / "co-tracker"
        if default_cotracker_repo.exists():
            self.cotracker_settings.repo_path = str(default_cotracker_repo.resolve())
        default_tapnet_repo = Path(__file__).resolve().parents[3] / "tapnet"
        if default_tapnet_repo.exists():
            self.tapnext_settings.repo_path = str(default_tapnet_repo.resolve())
        default_trackon_repo = Path(__file__).resolve().parents[3] / "track_on"
        if default_trackon_repo.exists():
            self.trackonr_settings.repo_path = str(default_trackon_repo.resolve())
        default_pipspp_repo = Path(__file__).resolve().parents[3] / "pipspp"
        if default_pipspp_repo.exists():
            self.pipspp_settings.repo_path = str(default_pipspp_repo.resolve())
        default_mftiq_repo = Path(__file__).resolve().parents[3] / "mftiq"
        if default_mftiq_repo.exists():
            self.mftiq_settings.repo_path = str(default_mftiq_repo.resolve())
        default_pytracking_repo = Path(__file__).resolve().parents[3] / "pytracking"
        if default_pytracking_repo.exists():
            self.tomp_settings.repo_path = str(default_pytracking_repo.resolve())
        default_cutie_repo = Path(__file__).resolve().parents[3] / "cutie"
        if default_cutie_repo.exists():
            self.cutie_settings.repo_path = str(default_cutie_repo.resolve())
        default_xmem2_repo = Path(__file__).resolve().parents[3] / "xmem2"
        if default_xmem2_repo.exists():
            self.xmem2_settings.repo_path = str(default_xmem2_repo.resolve())
        default_sam2_repo = Path(__file__).resolve().parents[3] / "sam2"
        if default_sam2_repo.exists():
            self.sam2_settings.repo_path = str(default_sam2_repo.resolve())
        self.legacy_segmentation_models_dir = Path("segmentation_models")
        self._dl_models_available = False
        self._yolo_models_available = False
        self._tapir_checkpoint_available = False
        self._locotrack_checkpoint_available = False
        self._cotracker_checkpoint_available = False
        self._tapnext_checkpoint_available = False
        self._trackonr_checkpoint_available = False
        self._pipspp_checkpoint_available = False
        self._mftiq_checkpoint_available = False
        self._tomp_checkpoint_available = False
        self._cutie_checkpoint_available = False
        self._xmem2_checkpoint_available = False
        self._sam2_checkpoint_available = False
        self._bbox_transparency_pct = 0.0
        self.quick_save_dir = Path("save")
        self.quick_save_path = self.quick_save_dir / "quick_save.lts.json"

        # --- Initialize UI ---
        self._init_ui()
        self._create_analysis_panel()
        self._create_toolbar()
        print("MainWindow has been created and initialized.")

    def _init_ui(self):
        """Initializes the main UI components."""
        self._create_menu()
        self._create_layout()

    def _create_menu(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Directory...", self)
        open_action.triggered.connect(self.open_directory_dialog)
        file_menu.addAction(open_action)

        import_gif_action = QAction("Import GIF (sample 0.1s)...", self)
        import_gif_action.triggered.connect(self.import_gif_dialog)
        file_menu.addAction(import_gif_action)

        load_session_action = QAction("Load Session...", self)
        load_session_action.triggered.connect(self.load_session_dialog)
        file_menu.addAction(load_session_action)

        save_session_action = QAction("Save Session...", self)
        save_session_action.triggered.connect(self.save_session_dialog)
        file_menu.addAction(save_session_action)

        quick_save_action = QAction("Quick Save", self)
        quick_save_action.setShortcut("Ctrl+S")
        quick_save_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        quick_save_action.triggered.connect(self.quick_save_session)
        file_menu.addAction(quick_save_action)

        quick_load_action = QAction("Quick Load", self)
        quick_load_action.setShortcut("Ctrl+L")
        quick_load_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        quick_load_action.triggered.connect(self.quick_load_session)
        file_menu.addAction(quick_load_action)

        load_bg_action = QAction("&Load Background File...", self)
        load_bg_action.triggered.connect(self.load_background_file_dialog)
        file_menu.addAction(load_bg_action)

        export_yolo_action = QAction("Export Spots to YOLO Labels...", self)
        export_yolo_action.triggered.connect(self.export_spots_to_yolo)
        file_menu.addAction(export_yolo_action)

        export_all_yolo_action = QAction("Export All (images + YOLO labels)...", self)
        export_all_yolo_action.triggered.connect(self.export_all_to_yolo)
        file_menu.addAction(export_all_yolo_action)

        clear_list_action = QAction("Clear File List", self)
        clear_list_action.setShortcut("Ctrl+Shift+C")
        clear_list_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        clear_list_action.triggered.connect(self.clear_file_list)
        file_menu.addAction(clear_list_action)

        # --- Mask Menu ---
        mask_menu = menu_bar.addMenu("&Mask")
        set_mask_action = QAction("&Set Metadata Mask...", self)
        set_mask_action.triggered.connect(self.open_mask_dialog)
        mask_menu.addAction(set_mask_action)
        clear_mask_action = QAction("Clear Current Mask", self)
        clear_mask_action.triggered.connect(self.clear_mask)
        mask_menu.addAction(clear_mask_action)
        set_analysis_circle_action = QAction("Set Analysis Circle...", self)
        set_analysis_circle_action.triggered.connect(self.open_analysis_circle_dialog)
        mask_menu.addAction(set_analysis_circle_action)
        clear_analysis_circle_action = QAction("Clear Analysis Circle", self)
        clear_analysis_circle_action.triggered.connect(self.clear_analysis_circle)
        mask_menu.addAction(clear_analysis_circle_action)
        set_analysis_ellipse_action = QAction("Set Analysis Ellipse...", self)
        set_analysis_ellipse_action.triggered.connect(self.open_analysis_ellipse_dialog)
        mask_menu.addAction(set_analysis_ellipse_action)
        clear_analysis_ellipse_action = QAction("Clear Analysis Ellipse", self)
        clear_analysis_ellipse_action.triggered.connect(self.clear_analysis_ellipse)
        mask_menu.addAction(clear_analysis_ellipse_action)

        # --- View Menu ---
        view_menu = menu_bar.addMenu("&View")
        bbox_transparency_action = QAction("Set BBox Transparency...", self)
        bbox_transparency_action.triggered.connect(self.prompt_bbox_transparency)
        view_menu.addAction(bbox_transparency_action)

        bbox_transparency_50_action = QAction("BBox Transparency 50%", self)
        bbox_transparency_50_action.setShortcut("Ctrl+5")
        bbox_transparency_50_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_50_action.triggered.connect(lambda: self.set_bbox_transparency(50.0))
        view_menu.addAction(bbox_transparency_50_action)

        bbox_transparency_20_action = QAction("BBox Transparency 20%", self)
        bbox_transparency_20_action.setShortcut("Ctrl+2")
        bbox_transparency_20_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_20_action.triggered.connect(lambda: self.set_bbox_transparency(20.0))
        view_menu.addAction(bbox_transparency_20_action)

        bbox_transparency_80_action = QAction("BBox Transparency 80%", self)
        bbox_transparency_80_action.setShortcut("Ctrl+8")
        bbox_transparency_80_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_80_action.triggered.connect(lambda: self.set_bbox_transparency(80.0))
        view_menu.addAction(bbox_transparency_80_action)

        bbox_transparency_off_action = QAction("BBox Transparency Off", self)
        bbox_transparency_off_action.setShortcut("Ctrl+0")
        bbox_transparency_off_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        bbox_transparency_off_action.triggered.connect(lambda: self.set_bbox_transparency(0.0))
        view_menu.addAction(bbox_transparency_off_action)

        # --- Analysis Menu ---
        analysis_menu = menu_bar.addMenu("&Analysis")
        analysis_action = QAction("Final Analysis...", self)
        analysis_action.setShortcut("Ctrl+Shift+A")
        analysis_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        analysis_action.triggered.connect(self.show_analysis_dialog)
        analysis_menu.addAction(analysis_action)

    def _create_toolbar(self):
        """Creates the main toolbar for interaction modes."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Action group makes the actions behave like radio buttons
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)

        # Pan/Select Mode (default)
        self.pan_mode_action = QAction("Pan/Select", self)
        self.pan_mode_action.setCheckable(True)
        self.pan_mode_action.setChecked(True)
        self.pan_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('PAN'))
        self.pan_mode_action.setShortcut("P")
        self.pan_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toolbar.addAction(self.pan_mode_action)
        mode_group.addAction(self.pan_mode_action)

        # Move ROI Mode
        self.move_roi_mode_action = QAction("Move ROI", self)
        self.move_roi_mode_action.setCheckable(True)
        self.move_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('MOVE_ROI'))
        self.move_roi_mode_action.setShortcut("M")
        self.move_roi_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toolbar.addAction(self.move_roi_mode_action)
        mode_group.addAction(self.move_roi_mode_action)

        # Add ROI Mode
        self.add_roi_mode_action = QAction("Add ROI", self)
        self.add_roi_mode_action.setCheckable(True)
        self.add_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_ROI'))
        self.add_roi_mode_action.setShortcut("A")
        self.add_roi_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.add_roi_mode_action.setToolTip(
            "Create a manual ROI using the current ROI Size."
        )
        toolbar.addAction(self.add_roi_mode_action)
        mode_group.addAction(self.add_roi_mode_action)

        # Add Point Mode
        self.add_point_mode_action = QAction("Add Point", self)
        self.add_point_mode_action.setCheckable(True)
        self.add_point_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_POINT'))
        self.add_point_mode_action.setShortcut("T")
        self.add_point_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.add_point_mode_action.setToolTip(
            "Click a point to create a tracking seed. A bbox with the current ROI Size is drawn automatically around the clicked point."
        )
        toolbar.addAction(self.add_point_mode_action)
        mode_group.addAction(self.add_point_mode_action)

        # Delete ROI Mode
        self.delete_roi_mode_action = QAction("Delete ROI", self)
        self.delete_roi_mode_action.setCheckable(True)
        self.delete_roi_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('DELETE_ROI'))
        self.delete_roi_mode_action.setShortcut("D")
        self.delete_roi_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        toolbar.addAction(self.delete_roi_mode_action)
        mode_group.addAction(self.delete_roi_mode_action)

        # Cycle Pan -> Add -> Delete
        self.cycle_mode_action = QAction("Cycle Pan/Add/Delete", self)
        self.cycle_mode_action.setShortcut("L")
        self.cycle_mode_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.cycle_mode_action.triggered.connect(self._cycle_pan_add_delete_mode)
        self.addAction(self.cycle_mode_action)

        # Cycle bbox transparency
        self.cycle_transparency_action = QAction("Cycle BBox Transparency", self)
        self.cycle_transparency_action.setShortcut("O")
        self.cycle_transparency_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.cycle_transparency_action.triggered.connect(self._cycle_bbox_transparency)
        self.addAction(self.cycle_transparency_action)

    def _create_layout(self):
        """
        Creates the main layout of the application's central widget.

        This method sets up a two-column horizontal layout.
        - The left column contains the list of data files and a preview panel
        for the selected background image.
        - The right column contains the main, interactive LeedImageViewer.
        It also initializes a status bar label for background file information.
        """
        # Create the central widget that will hold the main layout.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # The main layout is a horizontal box layout.
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: File List & Background Preview ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # File list setup
        left_panel_layout.addWidget(QLabel("LEED Files:"))
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.display_selected_image)
        left_panel_layout.addWidget(self.file_list_widget)

        self.remove_file_button = QPushButton("Remove Selected File")
        self.remove_file_button.clicked.connect(self.remove_selected_file)
        left_panel_layout.addWidget(self.remove_file_button)

        # Background preview group box
        bg_preview_group = QGroupBox("Background Preview")
        bg_preview_layout = QVBoxLayout(bg_preview_group)
        bg_preview_group.setMaximumHeight(250) # Limit the panel's vertical size

        # Label for the image thumbnail
        self.bg_preview_label = QLabel("No background loaded")
        self.bg_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bg_preview_label.setMinimumSize(128, 128)
        self.bg_preview_label.setStyleSheet("background-color: #2E2E2E; color: white;")
        
        # Label for the filename
        self.bg_filename_label = QLabel("Filename: N/A")
        self.bg_filename_label.setWordWrap(True) # Wrap text for long filenames

        bg_preview_layout.addWidget(self.bg_preview_label)
        bg_preview_layout.addWidget(self.bg_filename_label)
        left_panel_layout.addWidget(bg_preview_group)
        
        # --- Right Panel: Main Image Viewer ---
        self.image_viewer = LeedImageViewer(self)
        self.image_viewer.roi_added_at.connect(self.handle_add_roi)
        self.image_viewer.roi_deleted.connect(self.handle_delete_roi)
        self.image_viewer.roi_moved.connect(self.handle_move_roi)
        
        # --- Assemble Main Layout ---
        # Add the left and right panels to the main horizontal layout.
        main_layout.addWidget(left_panel_widget)
        main_layout.addWidget(self.image_viewer)
        
        # Set the stretch factors to make the right panel (image viewer) larger.
        # The viewer will take up 3/4 of the space, and the left panel 1/4.
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

    def _update_bg_preview(self):
        """Loads the background image and updates the preview widgets."""
        if not self.background_file: return
        
        image_data = load_tiff(self.background_file.filepath)
        pixmap = self._convert_numpy_to_pixmap(image_data)
        
        if not pixmap.isNull():
            self.bg_preview_label.setPixmap(pixmap.scaled(
                self.bg_preview_label.width(), self.bg_preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.bg_preview_label.setText("Preview failed")
            
        self.bg_filename_label.setText(f"Filename: {self.background_file.filename}")

    def _clear_bg_preview(self):
        """Resets the background preview widgets to their default state."""
        self.bg_preview_label.setText("No background loaded")
        self.bg_filename_label.setText("Filename: N/A")

    # def _create_layout(self):
    #     """Creates the main two-column layout."""
    #     central_widget = QWidget()
    #     self.setCentralWidget(central_widget)
    #     layout = QHBoxLayout(central_widget)

    #     # --- Left Column: Image Viewer ---
    #     self.image_viewer = LeedImageViewer(self)

    #     self.image_viewer.roi_added_at.connect(self.handle_add_roi)
    #     self.image_viewer.roi_deleted.connect(self.handle_delete_roi)
    #     self.image_viewer.roi_moved.connect(self.handle_move_roi)

    #     # --- Right Column: File List ---
    #     self.file_list_widget = QListWidget()
    #     self.file_list_widget.currentItemChanged.connect(self.display_selected_image)

    #     layout.addWidget(self.image_viewer, 3)  # 3/4 of the space
    #     layout.addWidget(self.file_list_widget, 1)    # 1/4 of the space

    #     self.bg_status_label = QLabel("Background file: Not loaded")
    #     self.statusBar().addPermanentWidget(self.bg_status_label)

    def handle_move_roi(self, spot_id: int, new_rect_f: QRectF):
        """Slot to handle updating an ROI's position in the data model."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item: return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Find the spot to update
        for spot in leed_file.spots:
            if spot.id == spot_id:
                # Update its data
                spot.roi_rect = new_rect_f.toRect() # Convert float rect back to int rect
                spot.center_x = new_rect_f.center().x()
                spot.center_y = new_rect_f.center().y()
                break # Stop searching

        # Refresh the list widget to show updated coordinates (if displayed)
        self.populate_spot_list(leed_file.spots)
        # Highlight the moved spot
        self.on_spot_selection_changed(self.spot_list_widget.currentItem(), None)

    def handle_add_roi(self, scene_pos: QPointF):
        """Create a new spot from a clicked center point using the current ROI size."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item: return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Get ROI size from the GUI
        roi_size = self.roi_size_spinbox.value()
        roi_half = roi_size // 2

        # Create a new QRect centered on the click position
        new_rect = QRect(int(scene_pos.x() - roi_half), int(scene_pos.y() - roi_half), roi_size, roi_size)

        # Find the next available ID
        next_id = max([s.id for s in leed_file.spots] + [0]) + 1

        # Create the new SpotROI object
        new_spot = SpotROI(
            id=next_id,
            roi_rect=new_rect,
            center_x=scene_pos.x(),
            center_y=scene_pos.y(),
            intensity=0, # Manual spot has no intensity initially
            source='manual_point' if self.image_viewer.mode == 'ADD_POINT' else 'manual'
        )
        leed_file.spots.append(new_spot)
        self.image_viewer.update_rois(leed_file.spots)
        self.populate_spot_list(leed_file.spots)

    def handle_delete_roi(self, spot_id: int):
        """Slot to handle the deletion of an ROI."""
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item: return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        # Find and remove the spot with the given ID
        leed_file.spots = [spot for spot in leed_file.spots if spot.id != spot_id]
        self.image_viewer.update_rois(leed_file.spots)
        self.populate_spot_list(leed_file.spots)

    def open_directory_dialog(self):
        """Opens a dialog to select a directory and processes the files within."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select LEED Data Directory")
        if not dir_path:
            return  # User cancelled the dialog

        directory = Path(dir_path)
        try:
            self.data_files = parse_directory(directory)
            # Reset background metadata whenever a new directory is loaded
            self.background_file = None
            self._clear_bg_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse directory:\n{e}")
            return

        if not self.data_files:
            QMessageBox.warning(self, "No Files Found", "No valid LEED data files were found in the selected directory.")

        self.populate_file_list()

    def import_gif_dialog(self):
        """Import a GIF, sample frames every 0.1s, and append them as images."""
        gif_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select GIF to import",
            "",
            "GIF Files (*.gif);;All Files (*)",
        )
        if not gif_path_str:
            return
        interval, ok = QInputDialog.getInt(
            self,
            "Sampling interval",
            "Sample every N milliseconds:",
            value=100,
            min=1,
            max=10000,
        )
        if not ok:
            return
        gif_path = Path(gif_path_str)
        try:
            out_dir, saved = extract_gif_frames(gif_path, dest_root=Path("data/gif_import") / gif_path.stem, sample_ms=interval)
            new_files = parse_directory(out_dir)
            if not new_files:
                QMessageBox.warning(self, "No Frames", "Could not extract any frames from the GIF.")
                return
            # Offset IDs to avoid collisions
            max_id = max([f.id for f in self.data_files if f.id is not None] + [0])
            for idx, lf in enumerate(new_files, start=1):
                lf.id = max_id + idx
            self.data_files.extend(new_files)
            self.populate_file_list()
            QMessageBox.information(
                self,
                "GIF Imported",
                f"Extracted {len(saved)} frame(s) every {interval} ms to {out_dir} and added to the list.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import GIF:\n{e}")

    def load_background_file_dialog(self):
        """Opens a dialog to manually select a background image file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "", # Default directory
            "Image Files (*.tif *.tiff *.png *.jpg)"
        )
        
        if not filepath: # User cancelled the dialog
            return

        path_obj = Path(filepath)
        
        # Create a lightweight LeedFile object for the background frame
        self.background_file = LeedFile(
            filepath=path_obj,
            filename=path_obj.name,
            is_background=True,
            # Other metadata can stay at defaults because the background is not analyzed
            element='', surface=0, mode='', temperature=0, energy=0, id=None, coverage=None
        )
        
        # Update the status bar label
        self._update_bg_preview()
        QMessageBox.information(self, "Success", "Background file loaded successfully.")
    
    def open_mask_dialog(self):
        """
        Opens the mask editor, passing the current list of masks.
        """
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list before setting a mask.")
            return

        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)
        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the original image data to create a mask.")
            return

        height, width = original_image_data.shape[:2]
        if original_image_data.ndim == 2:
            q_image = QImage(original_image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        else: # Assuming RGB for 3D arrays for simplicity
            q_image = QImage(original_image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        full_res_pixmap = QPixmap.fromImage(q_image)

        # Pass the CURRENT list of masks to the dialog
        dialog = MaskDialog(full_res_pixmap, self.metadata_mask_list, self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # Retrieve the NEW, potentially modified list of masks
            self.metadata_mask_list = dialog.get_selected_rects()
            print(f"Mask updated. Now contains {len(self.metadata_mask_list)} regions.")
            self.display_selected_image(current_list_item, None)

    def populate_file_list(self):
        """Populates the QListWidget with the found files."""
        self.file_list_widget.clear()

        if self.background_file:
            item = QListWidgetItem(f"BACKGROUND: {self.background_file.filename}")
            item.setData(Qt.ItemDataRole.UserRole, self.background_file)
            self.file_list_widget.addItem(item)

        for leed_file in self.data_files:
            # Create a user-friendly display string for each file
            display_text = f"ID: {leed_file.id}, T: {leed_file.temperature}°C, E: {leed_file.energy} eV, C: {leed_file.coverage}"
            item = QListWidgetItem(display_text)
            # Store the entire LeedFile object with the list item
            item.setData(Qt.ItemDataRole.UserRole, leed_file)
            self.file_list_widget.addItem(item)

    def clear_mask(self):
        """Clears all defined mask regions."""
        if not self.metadata_mask_list:
            return # Nothing to clear
        
        self.metadata_mask_list.clear()
        print("Mask has been cleared.")
        
        # Refresh the current view to show the image without the mask
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def open_analysis_circle_dialog(self):
        """Opens a dialog to set a global analysis circle (applied to all images)."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list before setting an analysis circle.")
            return

        leed_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the image to set an analysis circle.")
            return

        height, width = original_image_data.shape[:2]
        if original_image_data.ndim == 2:
            q_image = QImage(original_image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            q_image = QImage(original_image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        full_res_pixmap = QPixmap.fromImage(q_image)

        existing_center = None
        existing_radius = 0
        if self.analysis_circle:
            cx, cy, r = self.analysis_circle
            existing_center = QPoint(cx, cy)
            existing_radius = r

        dialog = CircleMaskDialog(full_res_pixmap, existing_center, existing_radius, self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            center, radius = dialog.get_circle()
            if center and radius > 0:
                self.analysis_circle = (int(center.x()), int(center.y()), int(radius))
                self.analysis_ellipse = None
                print(f"Analysis circle set: center=({center.x()}, {center.y()}), r={radius}")
            else:
                self.analysis_circle = None
                print("Analysis circle cleared (empty selection).")
            self.refresh_current_view()

    def clear_analysis_circle(self):
        """Clears the analysis circle overlay and mask."""
        if self.analysis_circle is None:
            return
        self.analysis_circle = None
        print("Analysis circle has been cleared.")
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def open_analysis_ellipse_dialog(self):
        """Opens a dialog to set a global analysis ellipse (applied to all images)."""
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list before setting an analysis ellipse.")
            return

        leed_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the image to set an analysis ellipse.")
            return

        height, width = original_image_data.shape[:2]
        if original_image_data.ndim == 2:
            q_image = QImage(original_image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            q_image = QImage(original_image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        full_res_pixmap = QPixmap.fromImage(q_image)

        existing_center = None
        existing_radius_x = 0
        existing_radius_y = 0
        existing_angle_deg = 0.0
        existing_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        if existing_ellipse:
            cx, cy, rx, ry, angle_deg = existing_ellipse
            existing_center = QPoint(cx, cy)
            existing_radius_x = rx
            existing_radius_y = ry
            existing_angle_deg = angle_deg

        dialog = EllipseMaskDialog(
            full_res_pixmap,
            existing_center,
            existing_radius_x,
            existing_radius_y,
            existing_angle_deg,
            self,
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            center, radius_x, radius_y, angle_deg = dialog.get_ellipse()
            if center and radius_x > 0 and radius_y > 0:
                self.analysis_ellipse = (
                    int(center.x()),
                    int(center.y()),
                    int(radius_x),
                    int(radius_y),
                    float(angle_deg),
                )
                self.analysis_circle = None
                print(
                    "Analysis ellipse set: "
                    f"center=({center.x()}, {center.y()}), rx={radius_x}, ry={radius_y}, angle={angle_deg:.1f}deg"
                )
            else:
                self.analysis_ellipse = None
                print("Analysis ellipse cleared (empty selection).")
            self.refresh_current_view()

    def clear_analysis_ellipse(self):
        """Clears the analysis ellipse overlay and mask."""
        if self.analysis_ellipse is None:
            return
        self.analysis_ellipse = None
        print("Analysis ellipse has been cleared.")
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def display_selected_image(self, current_item: QListWidgetItem | None, previous_item: QListWidgetItem | None):
        """
        Prepares the base image (original or processed) and tells the viewer
        to display it along with any masks and ROIs.
        """
        self.spot_list_widget.clear() # Clear spot details from previous image
        if current_item is None:
            self.image_viewer.set_image(None)
            return

        leed_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        image_data = load_tiff(leed_file.filepath)
        
        # Populate the spot list with the data for the newly selected file
        self.populate_spot_list(leed_file.spots)

        if image_data is None or image_data.size == 0:
            self.image_viewer.set_image(None)
            return

        # 1. Prepare the base image to be displayed (original or processed)
        base_image_numpy = self._get_base_image_for_display(leed_file, image_data)
        final_pixmap = self._convert_numpy_to_pixmap(base_image_numpy)
        
        # 2. Set the base image in the viewer
        self.image_viewer.set_image(final_pixmap)
        
        # 3. Tell the viewer to draw overlays
        self.image_viewer.update_masks(self.metadata_mask_list)
        self.image_viewer.update_analysis_circle(self.analysis_circle)
        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        self.analysis_ellipse = normalized_ellipse
        self.image_viewer.update_analysis_ellipse(normalized_ellipse)
        self.image_viewer.update_rois(leed_file.spots)

    def _get_base_image_for_display(self, leed_file, original_image_data):
        """Determines if original or processed image should be shown."""
        if self.show_processed_checkbox.isChecked() and not leed_file.is_background:
            try:
                bg_img = load_tiff(self.background_file.filepath) if self.background_file else None
                params = {'blur_sigma': self.blur_sigma_spinbox.value(), 'wth_radius': self.wth_radius_spinbox.value()}
                main_float = img_as_float(original_image_data.copy())
                subtracted_image = main_float
                if bg_img is not None:
                    bg_float = rgb2gray(bg_img) if bg_img.ndim == 3 else img_as_float(bg_img)
                    main_masked_for_mean, bg_masked_for_mean = main_float.copy(), bg_float.copy()
                    for rect in self.metadata_mask_list:
                        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                        main_masked_for_mean[y:y+h, x:x+w] = 0
                        bg_masked_for_mean[y:y+h, x:x+w] = 0
                    main_mean = main_masked_for_mean[main_masked_for_mean > 0].mean()
                    bg_mean = bg_masked_for_mean[bg_masked_for_mean > 0].mean()
                    if bg_mean > 1e-9: subtracted_image = main_float - (bg_float * (main_mean / bg_mean))
                residual_bg = gaussian_filter(subtracted_image, sigma=params['blur_sigma'])
                processed_image = subtracted_image - residual_bg
                return white_tophat(processed_image, footprint=disk(params['wth_radius']))
            except Exception as e:
                print(f"Could not generate processed preview: {e}")
        return original_image_data
    
    def _convert_numpy_to_pixmap(self, image_data):
        """Converts a NumPy array to a QPixmap for display."""
        if image_data.dtype != np.uint8:
            min_val, max_val = image_data.min(), image_data.max()
            if max_val - min_val > 1e-9: image_data = 255 * (image_data - min_val) / (max_val - min_val)
            else: image_data = np.zeros_like(image_data)
            image_data = image_data.astype(np.uint8)
            
        height, width = image_data.shape[:2]
        if image_data.ndim == 3 and image_data.shape[2] == 3: # RGB
            q_image = QImage(image_data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        else: # Grayscale
            q_image = QImage(image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_image)

    def _apply_analysis_region_mask(self, image_data: np.ndarray) -> np.ndarray:
        """Zero out pixels outside the configured analysis circle/ellipse region."""
        if image_data is None or image_data.size == 0:
            return image_data

        out = image_data.copy()
        h, w = out.shape[:2]
        yy, xx = np.ogrid[:h, :w]

        if self.analysis_circle is not None:
            cx, cy, r = self.analysis_circle
            circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            if out.ndim == 2:
                out[~circle_mask] = 0
            else:
                out[~circle_mask, ...] = 0

        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        if normalized_ellipse is not None:
            cx, cy, rx, ry, angle_deg = normalized_ellipse
            if rx > 0 and ry > 0:
                theta = np.deg2rad(float(angle_deg))
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                dx = xx - cx
                dy = yy - cy
                xr = cos_t * dx + sin_t * dy
                yr = -sin_t * dx + cos_t * dy
                ellipse_mask = (xr * xr) / (rx * rx) + (yr * yr) / (ry * ry) <= 1.0
                if out.ndim == 2:
                    out[~ellipse_mask] = 0
                else:
                    out[~ellipse_mask, ...] = 0

        return out

    def clear_file_list(self):
        """Clear the file list and associated state."""
        self.data_files = []
        self.tracking_results = []
        self.file_list_widget.clear()
        if hasattr(self, "analysis_button"):
            self.analysis_button.setEnabled(False)
        if hasattr(self, "positions_analysis_button"):
            self.positions_analysis_button.setEnabled(False)
        if hasattr(self, "visualize_button"):
            self.visualize_button.setEnabled(False)
        self.refresh_current_view()

    def remove_selected_file(self):
        """Remove the currently selected file from the list."""
        row = self.file_list_widget.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a file to remove.")
            return
        item = self.file_list_widget.takeItem(row)
        if item is None:
            return
        lf = item.data(Qt.ItemDataRole.UserRole)
        if lf in self.data_files:
            self.data_files.remove(lf)
        if lf and lf.id is not None and self.tracking_results:
            for track in self.tracking_results:
                track.spots_by_image_id.pop(lf.id, None)
            self.tracking_results = [t for t in self.tracking_results if t.spots_by_image_id]
        if not self.tracking_results:
            if hasattr(self, "analysis_button"):
                self.analysis_button.setEnabled(False)
            if hasattr(self, "positions_analysis_button"):
                self.positions_analysis_button.setEnabled(False)
            if hasattr(self, "visualize_button"):
                self.visualize_button.setEnabled(False)
        self.refresh_current_view()
    
    def _create_analysis_panel(self):
        """Creates a dockable widget with analysis controls."""
        self.analysis_dock = QDockWidget("Analysis Parameters", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)

        # Create a container widget and layout for the controls
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        
        # Use a form layout for neat parameter alignment
        find_spots_group = QGroupBox("Find Spots")
        find_spots_layout = QVBoxLayout(find_spots_group)

        backend_form = QFormLayout()
        self.spot_backend_combo = QComboBox()
        # To re-enable U-Net/segmentation, add "Deep Learning" back to this list.
        self.spot_backend_combo.addItems(["Classic", "YOLO"])
        self.spot_backend_combo.setToolTip(
            "Select the spot detection backend. Classic uses the existing WTH/peak pipeline; "
            "YOLO uses checkpoints from tracking_models/yolo "
            "(legacy fallback: segmentation_models/)."
        )
        backend_form.addRow("Spot detector:", self.spot_backend_combo)
        find_spots_layout.addLayout(backend_form)

        self.spot_backend_stack = QStackedWidget()
        self.spot_backend_stack.addWidget(self._build_classic_find_spots_panel())
        self.spot_backend_stack.addWidget(self._build_dl_find_spots_panel())
        self.spot_backend_stack.addWidget(self._build_yolo_find_spots_panel())
        find_spots_layout.addWidget(self.spot_backend_stack)

        self.dl_device_combo = QComboBox()
        self.dl_device_combo.addItems(["Auto", "CPU", "GPU"])
        self.dl_device_combo.setToolTip("Device for YOLO inference. Auto = GPU if available.")
        find_spots_layout.addWidget(self.dl_device_combo)

        self.det_subtract_bg_checkbox = QCheckBox("Subtract background before detection (YOLO)")
        self.det_subtract_bg_checkbox.setToolTip("Applies only when backend is YOLO and a background file is loaded.")
        find_spots_layout.addWidget(self.det_subtract_bg_checkbox)

        self.find_spots_button = QPushButton("Find Spots")
        self.find_spots_button.clicked.connect(self.run_spot_finding)
        find_spots_layout.addWidget(self.find_spots_button)
        self.find_all_spots_button = QPushButton("Find All Spots")
        self.find_all_spots_button.setToolTip("Run the selected detector on every frame in the list.")
        self.find_all_spots_button.clicked.connect(self.run_spot_finding_all)
        find_spots_layout.addWidget(self.find_all_spots_button)

        self.copy_spots_button = QPushButton("Copy Spots to All Frames")
        self.copy_spots_button.setToolTip("Duplicate spots from the current frame to all other frames.")
        self.copy_spots_button.clicked.connect(self.copy_spots_to_all_frames)
        find_spots_layout.addWidget(self.copy_spots_button)

        self.normalize_rois_button = QPushButton("Normalize ROIs (All Frames)")
        self.normalize_rois_button.setToolTip(
            "Set every spot ROI to the current ROI Size while keeping each spot center."
        )
        self.normalize_rois_button.clicked.connect(self.normalize_rois_all_frames)
        find_spots_layout.addWidget(self.normalize_rois_button)

        self.copy_spots_skip_existing_checkbox = QCheckBox("Nie nadpisuj istniejacych")
        self.copy_spots_skip_existing_checkbox.setToolTip("If enabled, frames that already have spots will be skipped.")
        find_spots_layout.addWidget(self.copy_spots_skip_existing_checkbox)

        layout.addWidget(find_spots_group)
        self.spot_backend_combo.currentTextChanged.connect(self._on_spot_backend_changed)

        layout.addStretch() # Pushes everything up

        tracking_group = QGroupBox("Tracking")
        

        form_layout = QFormLayout(tracking_group)
        self.tracking_scenario_combo = QComboBox()
        self.tracking_scenario_combo.addItems([
            "Optical Flow Tracking",
            "Optical Flow (skip, anchor)",
            "Offline TAPIR",
            "BootsTAPIR Online / Causal",
            "Offline LocoTrack",
            "Offline CoTracker3",
            "Offline TAPNext",
            "Offline Track-On-R",
            "Offline PIPs++",
            "Offline MFTIQ",
            "Offline ToMP",
            "Offline Cutie",
            "Offline XMem2",
            "Offline SAM2",
            "Detection Linking (Hungarian)",
            "Detection Linking (Directional)",
        ])
        # Backwards compatibility alias used by some dialogs/scripts.
        self.trackingAlgoCombo = self.tracking_scenario_combo
        form_layout.addRow("Tracking Algorithm:", self.tracking_scenario_combo)
        # layout.addWidget(tracking_group)

        # --- Group for Scenario 1 Parameters ---
        # layout.addWidget(s1_group)

        # --- Optical Flow (shared) Parameters ---
        self.optical_basic_group = QGroupBox("Optical Flow Parameters")
        optical_basic_layout = QFormLayout(self.optical_basic_group)

        self.lk_win_size_spinbox = QSpinBox()
        self.lk_win_size_spinbox.setRange(21, 71)
        self.lk_win_size_spinbox.setSingleStep(2)
        self.lk_win_size_spinbox.setValue(41)
        self.lk_win_size_spinbox.setToolTip(
            "Lucas–Kanade window size. Larger windows tolerate more motion/deformation but blur fine detail."
        )
        optical_basic_layout.addRow("Window size (px):", self.lk_win_size_spinbox)

        self.lk_max_level_spinbox = QSpinBox()
        self.lk_max_level_spinbox.setRange(0, 6)
        self.lk_max_level_spinbox.setValue(4)
        self.lk_max_level_spinbox.setToolTip(
            "Number of pyramid levels. Higher values handle larger displacements but slow processing."
        )
        optical_basic_layout.addRow("Pyramid levels:", self.lk_max_level_spinbox)

        self.lk_iterations_spinbox = QSpinBox()
        self.lk_iterations_spinbox.setRange(10, 100)
        self.lk_iterations_spinbox.setValue(50)
        self.lk_iterations_spinbox.setToolTip(
            "Lucas–Kanade convergence criterion: maximum number of iterations."
        )
        self.lk_epsilon_spinbox = QDoubleSpinBox()
        self.lk_epsilon_spinbox.setDecimals(5)
        self.lk_epsilon_spinbox.setRange(0.0001, 0.01)
        self.lk_epsilon_spinbox.setSingleStep(0.0001)
        self.lk_epsilon_spinbox.setValue(0.01)
        self.lk_epsilon_spinbox.setToolTip(
            "Lucas–Kanade epsilon threshold: acceptable delta between iterations."
        )
        iterations_widget = QWidget()
        iterations_layout = QHBoxLayout(iterations_widget)
        iterations_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(QLabel("Iter:"))
        iterations_layout.addWidget(self.lk_iterations_spinbox)
        iterations_layout.addSpacing(8)
        iterations_layout.addWidget(QLabel("Eps:"))
        iterations_layout.addWidget(self.lk_epsilon_spinbox)
        optical_basic_layout.addRow("Iterations / epsilon:", iterations_widget)

        self.lk_min_eig_spinbox = QDoubleSpinBox()
        self.lk_min_eig_spinbox.setDecimals(6)
        self.lk_min_eig_spinbox.setRange(0.000001, 0.001)
        self.lk_min_eig_spinbox.setSingleStep(0.000001)
        self.lk_min_eig_spinbox.setValue(0.00001)
        self.lk_min_eig_spinbox.setToolTip(
            "Texture-quality threshold inside the LK window (lower = accept weaker detail but risk noise)."
        )
        optical_basic_layout.addRow("Min eigenvalue:", self.lk_min_eig_spinbox)

        self.lk_subtract_bg_checkbox = QCheckBox("Subtract background before LK")
        self.lk_subtract_bg_checkbox.setToolTip(
            "Use the loaded background image and subtract it from every frame before LK (requires a background file)."
        )
        optical_basic_layout.addRow(self.lk_subtract_bg_checkbox)

        self.lk_use_centroid_checkbox = QCheckBox("Centroid-based recentering")
        self.lk_use_centroid_checkbox.setChecked(True)
        self.lk_use_centroid_checkbox.setToolTip(
            "After each successful match, compute a local centroid to recentre the ROI and avoid clipping the spot."
        )
        optical_basic_layout.addRow(self.lk_use_centroid_checkbox)

        self.lk_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.lk_centroid_multiplier_spinbox.setDecimals(2)
        self.lk_centroid_multiplier_spinbox.setRange(1.0, 3.0)
        self.lk_centroid_multiplier_spinbox.setSingleStep(0.1)
        self.lk_centroid_multiplier_spinbox.setValue(1.5)
        self.lk_centroid_multiplier_spinbox.setToolTip("Multiplier applied to the ROI radius when extracting the centroid patch (e.g., 1.5 ~= 1.5x larger).")
        optical_basic_layout.addRow("Centroid radius multiplier:", self.lk_centroid_multiplier_spinbox)

        self.lk_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.lk_centroid_rel_thr_spinbox.setDecimals(2)
        self.lk_centroid_rel_thr_spinbox.setRange(0.1, 0.9)
        self.lk_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.lk_centroid_rel_thr_spinbox.setValue(0.4)
        self.lk_centroid_rel_thr_spinbox.setToolTip("Relative intensity threshold (with respect to the peak) for the centroid mask.")
        optical_basic_layout.addRow("Centroid relative threshold:", self.lk_centroid_rel_thr_spinbox)

        self.lk_centroid_expand_spinbox = QSpinBox()
        self.lk_centroid_expand_spinbox.setRange(0, 3)
        self.lk_centroid_expand_spinbox.setValue(1)
        self.lk_centroid_expand_spinbox.setToolTip("How many times the centroid patch may expand when it sits near the edge.")
        optical_basic_layout.addRow("Max centroid expansions:", self.lk_centroid_expand_spinbox)

        form_layout.addRow(self.optical_basic_group)

        # --- Optical Flow (skip anchor) Extras ---
        self.optical_skip_group = QGroupBox("Optical Flow (skip anchor) safeguards")
        optical_skip_layout = QFormLayout(self.optical_skip_group)

        self.lk_error_threshold_spinbox = QSpinBox()
        self.lk_error_threshold_spinbox.setRange(10, 80)
        self.lk_error_threshold_spinbox.setValue(30)
        self.lk_error_threshold_spinbox.setToolTip(
            "Maximum LK fit error (lower = stricter). Used to discard questionable matches."
        )
        optical_skip_layout.addRow("Error threshold (LK):", self.lk_error_threshold_spinbox)
        self.lkErrSpin = self.lk_error_threshold_spinbox

        self.lk_fb_checkbox = QCheckBox("Forward-Backward check")
        self.lk_fb_checkbox.setChecked(True)
        self.lk_fb_checkbox.setToolTip(
            "Validate tracks by running LK forward and backward. Improves quality at additional cost."
        )
        optical_skip_layout.addRow(self.lk_fb_checkbox)

        self.lk_fb_threshold_spinbox = QDoubleSpinBox()
        self.lk_fb_threshold_spinbox.setDecimals(1)
        self.lk_fb_threshold_spinbox.setRange(2.0, 20.0)
        self.lk_fb_threshold_spinbox.setSingleStep(0.1)
        default_fb_threshold = max(2.0, round(self.lk_win_size_spinbox.value() * 0.2, 1))
        self.lk_fb_threshold_spinbox.setValue(default_fb_threshold)
        self.lk_fb_threshold_spinbox.setToolTip(
            "Maximum closure distance allowed for the forward-backward pass (higher = more tolerant)."
        )
        optical_skip_layout.addRow("FB threshold (px):", self.lk_fb_threshold_spinbox)
        self.lkFBSpin = self.lk_fb_threshold_spinbox

        self.lk_max_skip_frames_spinbox = QSpinBox()
        self.lk_max_skip_frames_spinbox.setRange(0, 1000)
        self.lk_max_skip_frames_spinbox.setValue(0)
        self.lk_max_skip_frames_spinbox.setToolTip(
            "How many consecutive frames a point may skip before we terminate its track (0 = unlimited)."
        )
        optical_skip_layout.addRow("Max skip frames:", self.lk_max_skip_frames_spinbox)

        self.oob_behavior_combo = QComboBox()
        self.oob_behavior_combo.addItem("Ignore (skip)", "skip")
        self.oob_behavior_combo.addItem("Mark as lost", "mark_lost")
        self.oob_behavior_combo.setCurrentIndex(0)
        self.oob_behavior_combo.setToolTip(
            "Choose how to handle spots that leave the image bounds."
        )
        optical_skip_layout.addRow("Out-of-bounds:", self.oob_behavior_combo)

        self.lk_use_ncc_checkbox = QCheckBox("Enable NCC test")
        self.lk_use_ncc_checkbox.setChecked(True)
        self.lk_use_ncc_checkbox.setToolTip(
            "Run an additional Normalized Cross-Correlation check in a local patch to suppress hallucinations."
        )
        optical_skip_layout.addRow(self.lk_use_ncc_checkbox)

        self.lk_ncc_threshold_spinbox = QDoubleSpinBox()
        self.lk_ncc_threshold_spinbox.setDecimals(2)
        self.lk_ncc_threshold_spinbox.setRange(0.3, 0.95)
        self.lk_ncc_threshold_spinbox.setSingleStep(0.05)
        self.lk_ncc_threshold_spinbox.setValue(0.65)
        self.lk_ncc_threshold_spinbox.setToolTip("Minimum NCC value required to accept a match.")
        optical_skip_layout.addRow("NCC threshold:", self.lk_ncc_threshold_spinbox)

        self.lk_ncc_patch_spinbox = QSpinBox()
        self.lk_ncc_patch_spinbox.setRange(3, 6)
        self.lk_ncc_patch_spinbox.setValue(4)
        self.lk_ncc_patch_spinbox.setToolTip("Patch radius for NCC (r=3 => 7x7, r=4 => 9x9).")
        optical_skip_layout.addRow("NCC patch radius:", self.lk_ncc_patch_spinbox)

        form_layout.addRow(self.optical_skip_group)

        # --- Offline TAPIR parameters ---
        self.tapir_group = QGroupBox("TAPIR")
        tapir_layout = QFormLayout(self.tapir_group)

        tapir_model_row_widget = QWidget()
        tapir_model_row_layout = QHBoxLayout(tapir_model_row_widget)
        tapir_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.tapir_checkpoint_combo = QComboBox()
        self.tapir_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tapir_checkpoint_combo.setToolTip(
            "Select a TAPIR checkpoint (.pt) from tracking_models/tapir."
        )
        self.tapir_checkpoint_combo.currentIndexChanged.connect(self._on_tapir_checkpoint_selected)
        self.tapir_refresh_button = QPushButton("Refresh")
        self.tapir_refresh_button.setToolTip(
            "Re-scan tracking_models/tapir and re-run TAPIR environment diagnostics."
        )
        self.tapir_refresh_button.clicked.connect(self._refresh_tapir_checkpoints)
        tapir_model_row_layout.addWidget(self.tapir_checkpoint_combo)
        tapir_model_row_layout.addWidget(self.tapir_refresh_button)
        tapir_layout.addRow("Checkpoint:", tapir_model_row_widget)

        self.tapir_device_combo = QComboBox()
        self.tapir_device_combo.addItem("Auto", "auto")
        self.tapir_device_combo.addItem("GPU", "gpu")
        self.tapir_device_combo.addItem("CPU", "cpu")
        self.tapir_device_combo.setCurrentIndex(0)
        self.tapir_device_combo.setToolTip("Device used for TAPIR inference.")
        tapir_layout.addRow("Device:", self.tapir_device_combo)

        self.tapir_inference_resolution_combo = QComboBox()
        for value in (256, 384, 512):
            self.tapir_inference_resolution_combo.addItem(f"{value}", value)
        self.tapir_inference_resolution_combo.setCurrentIndex(0)
        self.tapir_inference_resolution_combo.setToolTip(
            "Square resolution used before TAPIR inference. Higher values may help on LEED but cost more VRAM."
        )
        tapir_layout.addRow("Inference resolution:", self.tapir_inference_resolution_combo)

        self.tapir_query_chunk_spinbox = QSpinBox()
        self.tapir_query_chunk_spinbox.setRange(1, 512)
        self.tapir_query_chunk_spinbox.setValue(self.tapir_settings.query_chunk_size)
        self.tapir_query_chunk_spinbox.setToolTip(
            "Split queries into chunks to reduce memory usage."
        )
        tapir_layout.addRow("Query chunk size:", self.tapir_query_chunk_spinbox)

        self.tapir_window_size_spinbox = QSpinBox()
        self.tapir_window_size_spinbox.setRange(2, 4096)
        self.tapir_window_size_spinbox.setValue(self.tapir_settings.window_size)
        self.tapir_window_size_spinbox.setToolTip(
            "Number of frames processed by one offline TAPIR window. Smaller values reduce VRAM and can reduce long-range drift."
        )
        tapir_layout.addRow("Window size:", self.tapir_window_size_spinbox)

        self.tapir_window_overlap_spinbox = QSpinBox()
        self.tapir_window_overlap_spinbox.setRange(0, 1024)
        self.tapir_window_overlap_spinbox.setValue(self.tapir_settings.window_overlap)
        self.tapir_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive TAPIR windows."
        )
        tapir_layout.addRow("Window overlap:", self.tapir_window_overlap_spinbox)

        self.tapir_visibility_checkbox = QCheckBox("Use visibility gating")
        self.tapir_visibility_checkbox.setChecked(self.tapir_settings.use_visibility_gating)
        self.tapir_visibility_checkbox.setToolTip(
            "Drop points whose TAPIR visibility score falls below the threshold."
        )
        tapir_layout.addRow(self.tapir_visibility_checkbox)

        self.tapir_visibility_threshold_spinbox = QDoubleSpinBox()
        self.tapir_visibility_threshold_spinbox.setDecimals(2)
        self.tapir_visibility_threshold_spinbox.setRange(0.01, 0.99)
        self.tapir_visibility_threshold_spinbox.setSingleStep(0.05)
        self.tapir_visibility_threshold_spinbox.setValue(self.tapir_settings.visibility_threshold)
        self.tapir_visibility_threshold_spinbox.setToolTip("Minimum visibility score to keep a point.")
        tapir_layout.addRow("Visibility threshold:", self.tapir_visibility_threshold_spinbox)

        self.tapir_uncertainty_checkbox = QCheckBox("Use uncertainty gating")
        self.tapir_uncertainty_checkbox.setChecked(self.tapir_settings.use_uncertainty_gating)
        self.tapir_uncertainty_checkbox.setToolTip(
            "Use TAPIR expected-distance logits to suppress uncertain predictions."
        )
        tapir_layout.addRow(self.tapir_uncertainty_checkbox)

        self.tapir_expected_dist_threshold_spinbox = QDoubleSpinBox()
        self.tapir_expected_dist_threshold_spinbox.setDecimals(2)
        self.tapir_expected_dist_threshold_spinbox.setRange(0.01, 0.99)
        self.tapir_expected_dist_threshold_spinbox.setSingleStep(0.05)
        self.tapir_expected_dist_threshold_spinbox.setValue(self.tapir_settings.expected_dist_threshold)
        self.tapir_expected_dist_threshold_spinbox.setToolTip(
            "Minimum uncertainty confidence to keep a point."
        )
        tapir_layout.addRow("Uncertainty threshold:", self.tapir_expected_dist_threshold_spinbox)

        self.tapir_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.tapir_keep_gaps_checkbox.setChecked(self.tapir_settings.keep_invisible_points_as_gaps)
        self.tapir_keep_gaps_checkbox.setToolTip(
            "If enabled, invisible TAPIR predictions create gaps instead of forced points."
        )
        tapir_layout.addRow(self.tapir_keep_gaps_checkbox)

        self.tapir_max_step_spinbox = QDoubleSpinBox()
        self.tapir_max_step_spinbox.setDecimals(1)
        self.tapir_max_step_spinbox.setRange(0.0, 500.0)
        self.tapir_max_step_spinbox.setSingleStep(1.0)
        self.tapir_max_step_spinbox.setValue(self.tapir_settings.max_step_distance_px)
        self.tapir_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        tapir_layout.addRow("Max jump (px):", self.tapir_max_step_spinbox)

        self.tapir_max_reacquisition_gap_spinbox = QSpinBox()
        self.tapir_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.tapir_max_reacquisition_gap_spinbox.setValue(self.tapir_settings.max_reacquisition_gap)
        self.tapir_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        tapir_layout.addRow("Max reacquisition gap:", self.tapir_max_reacquisition_gap_spinbox)

        self.tapir_terminate_after_invisible_spinbox = QSpinBox()
        self.tapir_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.tapir_terminate_after_invisible_spinbox.setValue(
            self.tapir_settings.terminate_after_invisible_frames
        )
        self.tapir_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        tapir_layout.addRow("Terminate after gap:", self.tapir_terminate_after_invisible_spinbox)

        self.tapir_center_mode_combo = QComboBox()
        self.tapir_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.tapir_center_mode_combo.addItem("Manual", "manual")
        tapir_center_mode_idx = self.tapir_center_mode_combo.findData(self.tapir_settings.center_mode)
        self.tapir_center_mode_combo.setCurrentIndex(tapir_center_mode_idx if tapir_center_mode_idx >= 0 else 0)
        self.tapir_center_mode_combo.setToolTip(
            "Center used for radial TAPIR direction gating. Auto tries detections/pattern center and falls back to image center."
        )
        self.tapir_center_mode_combo.currentIndexChanged.connect(self._on_tapir_center_mode_changed)
        tapir_layout.addRow("Direction center:", self.tapir_center_mode_combo)

        tapir_center_row_widget = QWidget()
        tapir_center_row_layout = QHBoxLayout(tapir_center_row_widget)
        tapir_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.tapir_center_x_spinbox = QDoubleSpinBox()
        self.tapir_center_x_spinbox.setRange(0.0, 10000.0)
        self.tapir_center_x_spinbox.setDecimals(1)
        self.tapir_center_x_spinbox.setValue(self.tapir_settings.manual_center_x)
        self.tapir_center_y_spinbox = QDoubleSpinBox()
        self.tapir_center_y_spinbox.setRange(0.0, 10000.0)
        self.tapir_center_y_spinbox.setDecimals(1)
        self.tapir_center_y_spinbox.setValue(self.tapir_settings.manual_center_y)
        tapir_center_row_layout.addWidget(self.tapir_center_x_spinbox)
        tapir_center_row_layout.addWidget(QLabel(","))
        tapir_center_row_layout.addWidget(self.tapir_center_y_spinbox)
        tapir_layout.addRow("Manual center (x, y):", tapir_center_row_widget)

        self.tapir_direction_mode_combo = QComboBox()
        self.tapir_direction_mode_combo.addItem("None", "none")
        self.tapir_direction_mode_combo.addItem("Auto", "auto")
        self.tapir_direction_mode_combo.addItem("Inward", "inward")
        self.tapir_direction_mode_combo.addItem("Outward", "outward")
        tapir_direction_idx = self.tapir_direction_mode_combo.findData(self.tapir_settings.direction_mode)
        self.tapir_direction_mode_combo.setCurrentIndex(tapir_direction_idx if tapir_direction_idx >= 0 else 0)
        self.tapir_direction_mode_combo.setToolTip(
            "Apply radial direction constraints to TAPIR trajectories."
        )
        tapir_layout.addRow("Direction mode:", self.tapir_direction_mode_combo)

        self.tapir_lateral_tol_spinbox = QDoubleSpinBox()
        self.tapir_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.tapir_lateral_tol_spinbox.setSingleStep(1.0)
        self.tapir_lateral_tol_spinbox.setValue(self.tapir_settings.lateral_angle_tolerance_deg)
        self.tapir_lateral_tol_spinbox.setToolTip(
            "Allowed angular deviation from the expected radial direction."
        )
        tapir_layout.addRow("Angular tol (deg):", self.tapir_lateral_tol_spinbox)

        self.tapir_max_radial_jump_spinbox = QDoubleSpinBox()
        self.tapir_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.tapir_max_radial_jump_spinbox.setSingleStep(1.0)
        self.tapir_max_radial_jump_spinbox.setValue(self.tapir_settings.max_radial_jump_px)
        self.tapir_max_radial_jump_spinbox.setToolTip(
            "Maximum accepted radial change between two accepted points. Set 0 to disable."
        )
        tapir_layout.addRow("Max radial jump (px):", self.tapir_max_radial_jump_spinbox)

        self.tapir_recenter_mode_combo = QComboBox()
        self.tapir_recenter_mode_combo.addItem("None", "none")
        self.tapir_recenter_mode_combo.addItem("Peak", "peak")
        self.tapir_recenter_mode_combo.addItem("Centroid", "centroid")
        self.tapir_recenter_mode_combo.addItem("Gaussian", "gaussian")
        recenter_mode_idx = self.tapir_recenter_mode_combo.findData(self.tapir_settings.recenter_mode)
        self.tapir_recenter_mode_combo.setCurrentIndex(recenter_mode_idx if recenter_mode_idx >= 0 else 2)
        self.tapir_recenter_mode_combo.setToolTip(
            "Local refinement applied after each TAPIR prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        tapir_layout.addRow("Recenter mode:", self.tapir_recenter_mode_combo)

        self.tapir_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.tapir_centroid_multiplier_spinbox.setDecimals(2)
        self.tapir_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.tapir_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.tapir_centroid_multiplier_spinbox.setValue(self.tapir_settings.recenter_patch_multiplier)
        self.tapir_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        tapir_layout.addRow("Recenter radius x ROI:", self.tapir_centroid_multiplier_spinbox)

        self.tapir_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.tapir_centroid_rel_thr_spinbox.setDecimals(2)
        self.tapir_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.tapir_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.tapir_centroid_rel_thr_spinbox.setValue(self.tapir_settings.centroid_rel_threshold)
        self.tapir_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        tapir_layout.addRow("Recenter threshold:", self.tapir_centroid_rel_thr_spinbox)

        self.tapir_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.tapir_centroid_max_shift_spinbox.setDecimals(2)
        self.tapir_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.tapir_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.tapir_centroid_max_shift_spinbox.setValue(self.tapir_settings.centroid_max_shift_ratio)
        self.tapir_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        tapir_layout.addRow("Max recenter shift:", self.tapir_centroid_max_shift_spinbox)

        self.tapir_warning_label = QLabel()
        self.tapir_warning_label.setStyleSheet("color: #d9534f;")
        self.tapir_warning_label.setWordWrap(True)
        tapir_layout.addRow(self.tapir_warning_label)
        self._on_tapir_center_mode_changed(self.tapir_center_mode_combo.currentIndex())

        form_layout.addRow(self.tapir_group)

        # --- Offline LocoTrack parameters ---
        self.locotrack_group = QGroupBox("Offline LocoTrack")
        locotrack_layout = QFormLayout(self.locotrack_group)

        self.locotrack_repo_path_edit = QLineEdit(self.locotrack_settings.repo_path or "")
        self.locotrack_repo_path_edit.setPlaceholderText("Path to local locotrack repo root")
        self.locotrack_repo_path_edit.setToolTip(
            "Local clone of the LocoTrack repository. Must contain locotrack_pytorch/models/locotrack_model.py."
        )
        self.locotrack_repo_path_edit.editingFinished.connect(self._refresh_locotrack_checkpoints)
        locotrack_layout.addRow("Repo path:", self.locotrack_repo_path_edit)

        self.locotrack_python_executable_edit = QLineEdit(self.locotrack_settings.python_executable or "")
        self.locotrack_python_executable_edit.setPlaceholderText(
            "Optional python.exe from dedicated locotrack env"
        )
        self.locotrack_python_executable_edit.setToolTip(
            "Used only in subprocess mode. Point this to the interpreter of the dedicated LocoTrack environment."
        )
        self.locotrack_python_executable_edit.editingFinished.connect(self._refresh_locotrack_checkpoints)
        locotrack_layout.addRow("Python executable:", self.locotrack_python_executable_edit)

        self.locotrack_backend_mode_combo = QComboBox()
        self.locotrack_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.locotrack_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        locotrack_backend_mode_idx = self.locotrack_backend_mode_combo.findData(self.locotrack_settings.backend_mode)
        self.locotrack_backend_mode_combo.setCurrentIndex(locotrack_backend_mode_idx if locotrack_backend_mode_idx >= 0 else 0)
        self.locotrack_backend_mode_combo.setToolTip(
            "Subprocess lets Trackleed stay in the main env while LocoTrack runs in its own dedicated env."
        )
        self.locotrack_backend_mode_combo.currentIndexChanged.connect(self._on_locotrack_backend_mode_changed)
        locotrack_layout.addRow("Backend mode:", self.locotrack_backend_mode_combo)

        locotrack_model_row_widget = QWidget()
        locotrack_model_row_layout = QHBoxLayout(locotrack_model_row_widget)
        locotrack_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.locotrack_checkpoint_combo = QComboBox()
        self.locotrack_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.locotrack_checkpoint_combo.setToolTip(
            "Select a LocoTrack checkpoint (.ckpt) from tracking_models/locotrack."
        )
        self.locotrack_checkpoint_combo.currentIndexChanged.connect(self._on_locotrack_checkpoint_selected)
        self.locotrack_refresh_button = QPushButton("Refresh")
        self.locotrack_refresh_button.setToolTip(
            "Re-scan tracking_models/locotrack and re-run LocoTrack environment diagnostics."
        )
        self.locotrack_refresh_button.clicked.connect(self._refresh_locotrack_checkpoints)
        locotrack_model_row_layout.addWidget(self.locotrack_checkpoint_combo)
        locotrack_model_row_layout.addWidget(self.locotrack_refresh_button)
        locotrack_layout.addRow("Checkpoint:", locotrack_model_row_widget)

        self.locotrack_model_size_combo = QComboBox()
        self.locotrack_model_size_combo.addItem("Auto (from checkpoint)", "auto")
        self.locotrack_model_size_combo.addItem("Small", "small")
        self.locotrack_model_size_combo.addItem("Base", "base")
        locotrack_size_idx = self.locotrack_model_size_combo.findData(self.locotrack_settings.model_size)
        self.locotrack_model_size_combo.setCurrentIndex(locotrack_size_idx if locotrack_size_idx >= 0 else 0)
        self.locotrack_model_size_combo.setToolTip(
            "Architecture used to instantiate the model. Auto infers small/base from the checkpoint name."
        )
        locotrack_layout.addRow("Model size:", self.locotrack_model_size_combo)

        self.locotrack_device_combo = QComboBox()
        self.locotrack_device_combo.addItem("Auto", "auto")
        self.locotrack_device_combo.addItem("GPU", "gpu")
        self.locotrack_device_combo.addItem("CPU", "cpu")
        self.locotrack_device_combo.setCurrentIndex(0)
        self.locotrack_device_combo.setToolTip("Device used for LocoTrack inference.")
        locotrack_layout.addRow("Device:", self.locotrack_device_combo)

        self.locotrack_inference_resolution_combo = QComboBox()
        for value in (256, 384, 512):
            self.locotrack_inference_resolution_combo.addItem(f"{value}", value)
        self.locotrack_inference_resolution_combo.setCurrentIndex(0)
        self.locotrack_inference_resolution_combo.setToolTip(
            "Square resolution used before LocoTrack inference. Higher values may help on LEED but cost more VRAM."
        )
        locotrack_layout.addRow("Inference resolution:", self.locotrack_inference_resolution_combo)

        self.locotrack_query_chunk_spinbox = QSpinBox()
        self.locotrack_query_chunk_spinbox.setRange(1, 512)
        self.locotrack_query_chunk_spinbox.setValue(self.locotrack_settings.query_chunk_size)
        self.locotrack_query_chunk_spinbox.setToolTip("Split queries into chunks to reduce memory usage.")
        locotrack_layout.addRow("Query chunk size:", self.locotrack_query_chunk_spinbox)

        self.locotrack_window_size_spinbox = QSpinBox()
        self.locotrack_window_size_spinbox.setRange(2, 4096)
        self.locotrack_window_size_spinbox.setValue(self.locotrack_settings.window_size)
        self.locotrack_window_size_spinbox.setToolTip(
            "Number of frames processed by one offline LocoTrack window. Smaller values reduce VRAM and can reduce long-range drift."
        )
        locotrack_layout.addRow("Window size:", self.locotrack_window_size_spinbox)

        self.locotrack_window_overlap_spinbox = QSpinBox()
        self.locotrack_window_overlap_spinbox.setRange(0, 1024)
        self.locotrack_window_overlap_spinbox.setValue(self.locotrack_settings.window_overlap)
        self.locotrack_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive LocoTrack windows."
        )
        locotrack_layout.addRow("Window overlap:", self.locotrack_window_overlap_spinbox)

        self.locotrack_visibility_checkbox = QCheckBox("Use visibility gating")
        self.locotrack_visibility_checkbox.setChecked(self.locotrack_settings.use_visibility_gating)
        self.locotrack_visibility_checkbox.setToolTip(
            "Drop points whose LocoTrack visibility score falls below the threshold."
        )
        locotrack_layout.addRow(self.locotrack_visibility_checkbox)

        self.locotrack_visibility_threshold_spinbox = QDoubleSpinBox()
        self.locotrack_visibility_threshold_spinbox.setDecimals(2)
        self.locotrack_visibility_threshold_spinbox.setRange(0.01, 0.99)
        self.locotrack_visibility_threshold_spinbox.setSingleStep(0.05)
        self.locotrack_visibility_threshold_spinbox.setValue(self.locotrack_settings.visibility_threshold)
        self.locotrack_visibility_threshold_spinbox.setToolTip("Minimum visibility score to keep a point.")
        locotrack_layout.addRow("Visibility threshold:", self.locotrack_visibility_threshold_spinbox)

        self.locotrack_uncertainty_checkbox = QCheckBox("Use uncertainty gating")
        self.locotrack_uncertainty_checkbox.setChecked(self.locotrack_settings.use_uncertainty_gating)
        self.locotrack_uncertainty_checkbox.setToolTip(
            "Use LocoTrack expected-distance logits to suppress uncertain predictions."
        )
        locotrack_layout.addRow(self.locotrack_uncertainty_checkbox)

        self.locotrack_expected_dist_threshold_spinbox = QDoubleSpinBox()
        self.locotrack_expected_dist_threshold_spinbox.setDecimals(2)
        self.locotrack_expected_dist_threshold_spinbox.setRange(0.01, 0.99)
        self.locotrack_expected_dist_threshold_spinbox.setSingleStep(0.05)
        self.locotrack_expected_dist_threshold_spinbox.setValue(self.locotrack_settings.expected_dist_threshold)
        self.locotrack_expected_dist_threshold_spinbox.setToolTip(
            "Minimum uncertainty confidence to keep a point."
        )
        locotrack_layout.addRow("Uncertainty threshold:", self.locotrack_expected_dist_threshold_spinbox)

        self.locotrack_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.locotrack_keep_gaps_checkbox.setChecked(self.locotrack_settings.keep_invisible_points_as_gaps)
        self.locotrack_keep_gaps_checkbox.setToolTip(
            "If enabled, invisible LocoTrack predictions create gaps instead of forced points."
        )
        locotrack_layout.addRow(self.locotrack_keep_gaps_checkbox)

        self.locotrack_max_step_spinbox = QDoubleSpinBox()
        self.locotrack_max_step_spinbox.setDecimals(1)
        self.locotrack_max_step_spinbox.setRange(0.0, 500.0)
        self.locotrack_max_step_spinbox.setSingleStep(1.0)
        self.locotrack_max_step_spinbox.setValue(self.locotrack_settings.max_step_distance_px)
        self.locotrack_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        locotrack_layout.addRow("Max jump (px):", self.locotrack_max_step_spinbox)

        self.locotrack_max_reacquisition_gap_spinbox = QSpinBox()
        self.locotrack_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.locotrack_max_reacquisition_gap_spinbox.setValue(self.locotrack_settings.max_reacquisition_gap)
        self.locotrack_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        locotrack_layout.addRow("Max reacquisition gap:", self.locotrack_max_reacquisition_gap_spinbox)

        self.locotrack_terminate_after_invisible_spinbox = QSpinBox()
        self.locotrack_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.locotrack_terminate_after_invisible_spinbox.setValue(
            self.locotrack_settings.terminate_after_invisible_frames
        )
        self.locotrack_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        locotrack_layout.addRow("Terminate after gap:", self.locotrack_terminate_after_invisible_spinbox)

        self.locotrack_center_mode_combo = QComboBox()
        self.locotrack_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.locotrack_center_mode_combo.addItem("Manual", "manual")
        locotrack_center_mode_idx = self.locotrack_center_mode_combo.findData(self.locotrack_settings.center_mode)
        self.locotrack_center_mode_combo.setCurrentIndex(locotrack_center_mode_idx if locotrack_center_mode_idx >= 0 else 0)
        self.locotrack_center_mode_combo.setToolTip(
            "Center used for radial LocoTrack direction gating. Auto tries detections/pattern center and falls back to image center."
        )
        self.locotrack_center_mode_combo.currentIndexChanged.connect(self._on_locotrack_center_mode_changed)
        locotrack_layout.addRow("Direction center:", self.locotrack_center_mode_combo)

        locotrack_center_row_widget = QWidget()
        locotrack_center_row_layout = QHBoxLayout(locotrack_center_row_widget)
        locotrack_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.locotrack_center_x_spinbox = QDoubleSpinBox()
        self.locotrack_center_x_spinbox.setRange(0.0, 10000.0)
        self.locotrack_center_x_spinbox.setDecimals(1)
        self.locotrack_center_x_spinbox.setValue(self.locotrack_settings.manual_center_x)
        self.locotrack_center_y_spinbox = QDoubleSpinBox()
        self.locotrack_center_y_spinbox.setRange(0.0, 10000.0)
        self.locotrack_center_y_spinbox.setDecimals(1)
        self.locotrack_center_y_spinbox.setValue(self.locotrack_settings.manual_center_y)
        locotrack_center_row_layout.addWidget(self.locotrack_center_x_spinbox)
        locotrack_center_row_layout.addWidget(QLabel(","))
        locotrack_center_row_layout.addWidget(self.locotrack_center_y_spinbox)
        locotrack_layout.addRow("Manual center (x, y):", locotrack_center_row_widget)

        self.locotrack_direction_mode_combo = QComboBox()
        self.locotrack_direction_mode_combo.addItem("None", "none")
        self.locotrack_direction_mode_combo.addItem("Auto", "auto")
        self.locotrack_direction_mode_combo.addItem("Inward", "inward")
        self.locotrack_direction_mode_combo.addItem("Outward", "outward")
        locotrack_direction_idx = self.locotrack_direction_mode_combo.findData(self.locotrack_settings.direction_mode)
        self.locotrack_direction_mode_combo.setCurrentIndex(locotrack_direction_idx if locotrack_direction_idx >= 0 else 0)
        self.locotrack_direction_mode_combo.setToolTip(
            "Apply radial direction constraints to LocoTrack trajectories."
        )
        locotrack_layout.addRow("Direction mode:", self.locotrack_direction_mode_combo)

        self.locotrack_lateral_tol_spinbox = QDoubleSpinBox()
        self.locotrack_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.locotrack_lateral_tol_spinbox.setSingleStep(1.0)
        self.locotrack_lateral_tol_spinbox.setValue(self.locotrack_settings.lateral_angle_tolerance_deg)
        self.locotrack_lateral_tol_spinbox.setToolTip(
            "Allowed angular deviation from the expected radial direction."
        )
        locotrack_layout.addRow("Angular tol (deg):", self.locotrack_lateral_tol_spinbox)

        self.locotrack_max_radial_jump_spinbox = QDoubleSpinBox()
        self.locotrack_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.locotrack_max_radial_jump_spinbox.setSingleStep(1.0)
        self.locotrack_max_radial_jump_spinbox.setValue(self.locotrack_settings.max_radial_jump_px)
        self.locotrack_max_radial_jump_spinbox.setToolTip(
            "Maximum accepted radial change between two accepted points. Set 0 to disable."
        )
        locotrack_layout.addRow("Max radial jump (px):", self.locotrack_max_radial_jump_spinbox)

        self.locotrack_recenter_mode_combo = QComboBox()
        self.locotrack_recenter_mode_combo.addItem("None", "none")
        self.locotrack_recenter_mode_combo.addItem("Peak", "peak")
        self.locotrack_recenter_mode_combo.addItem("Centroid", "centroid")
        self.locotrack_recenter_mode_combo.addItem("Gaussian", "gaussian")
        locotrack_recenter_mode_idx = self.locotrack_recenter_mode_combo.findData(self.locotrack_settings.recenter_mode)
        self.locotrack_recenter_mode_combo.setCurrentIndex(locotrack_recenter_mode_idx if locotrack_recenter_mode_idx >= 0 else 2)
        self.locotrack_recenter_mode_combo.setToolTip(
            "Local refinement applied after each LocoTrack prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        locotrack_layout.addRow("Recenter mode:", self.locotrack_recenter_mode_combo)

        self.locotrack_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.locotrack_centroid_multiplier_spinbox.setDecimals(2)
        self.locotrack_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.locotrack_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.locotrack_centroid_multiplier_spinbox.setValue(self.locotrack_settings.recenter_patch_multiplier)
        self.locotrack_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        locotrack_layout.addRow("Recenter radius x ROI:", self.locotrack_centroid_multiplier_spinbox)

        self.locotrack_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.locotrack_centroid_rel_thr_spinbox.setDecimals(2)
        self.locotrack_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.locotrack_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.locotrack_centroid_rel_thr_spinbox.setValue(self.locotrack_settings.centroid_rel_threshold)
        self.locotrack_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        locotrack_layout.addRow("Recenter threshold:", self.locotrack_centroid_rel_thr_spinbox)

        self.locotrack_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.locotrack_centroid_max_shift_spinbox.setDecimals(2)
        self.locotrack_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.locotrack_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.locotrack_centroid_max_shift_spinbox.setValue(self.locotrack_settings.centroid_max_shift_ratio)
        self.locotrack_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        locotrack_layout.addRow("Max recenter shift:", self.locotrack_centroid_max_shift_spinbox)

        self.locotrack_warning_label = QLabel()
        self.locotrack_warning_label.setStyleSheet("color: #d9534f;")
        self.locotrack_warning_label.setWordWrap(True)
        locotrack_layout.addRow(self.locotrack_warning_label)
        self._on_locotrack_backend_mode_changed(self.locotrack_backend_mode_combo.currentIndex())
        self._on_locotrack_center_mode_changed(self.locotrack_center_mode_combo.currentIndex())

        form_layout.addRow(self.locotrack_group)

        # --- Offline CoTracker3 parameters ---
        self.cotracker_group = QGroupBox("Offline CoTracker3")
        cotracker_layout = QFormLayout(self.cotracker_group)

        self.cotracker_repo_path_edit = QLineEdit(self.cotracker_settings.repo_path or "")
        self.cotracker_repo_path_edit.setPlaceholderText("Path to local co-tracker repo root")
        self.cotracker_repo_path_edit.setToolTip(
            "Local clone of the CoTracker repository. Must contain cotracker/predictor.py."
        )
        self.cotracker_repo_path_edit.editingFinished.connect(self._refresh_cotracker_checkpoints)
        cotracker_layout.addRow("Repo path:", self.cotracker_repo_path_edit)

        cotracker_model_row_widget = QWidget()
        cotracker_model_row_layout = QHBoxLayout(cotracker_model_row_widget)
        cotracker_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.cotracker_checkpoint_combo = QComboBox()
        self.cotracker_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cotracker_checkpoint_combo.setToolTip(
            "Select a CoTracker3 checkpoint (.pth) from tracking_models/cotracker3."
        )
        self.cotracker_checkpoint_combo.currentIndexChanged.connect(self._on_cotracker_checkpoint_selected)
        self.cotracker_refresh_button = QPushButton("Refresh")
        self.cotracker_refresh_button.setToolTip(
            "Re-scan tracking_models/cotracker3 and re-run CoTracker3 environment diagnostics."
        )
        self.cotracker_refresh_button.clicked.connect(self._refresh_cotracker_checkpoints)
        cotracker_model_row_layout.addWidget(self.cotracker_checkpoint_combo)
        cotracker_model_row_layout.addWidget(self.cotracker_refresh_button)
        cotracker_layout.addRow("Checkpoint:", cotracker_model_row_widget)

        self.cotracker_device_combo = QComboBox()
        self.cotracker_device_combo.addItem("Auto", "auto")
        self.cotracker_device_combo.addItem("GPU", "gpu")
        self.cotracker_device_combo.addItem("CPU", "cpu")
        self.cotracker_device_combo.setCurrentIndex(0)
        self.cotracker_device_combo.setToolTip("Device used for CoTracker3 inference.")
        cotracker_layout.addRow("Device:", self.cotracker_device_combo)

        self.cotracker_inference_resolution_combo = QComboBox()
        for value in (256, 384, 512):
            self.cotracker_inference_resolution_combo.addItem(f"{value}", value)
        self.cotracker_inference_resolution_combo.setCurrentIndex(0)
        self.cotracker_inference_resolution_combo.setToolTip(
            "Square pre-resize applied before CoTracker3 inference. The model still applies its own internal resize."
        )
        cotracker_layout.addRow("Inference resolution:", self.cotracker_inference_resolution_combo)

        self.cotracker_window_len_spinbox = QSpinBox()
        self.cotracker_window_len_spinbox.setRange(4, 256)
        self.cotracker_window_len_spinbox.setValue(self.cotracker_settings.window_len)
        self.cotracker_window_len_spinbox.setToolTip(
            "Native CoTracker3 offline window length passed to the predictor."
        )
        cotracker_layout.addRow("Model window len:", self.cotracker_window_len_spinbox)

        self.cotracker_window_size_spinbox = QSpinBox()
        self.cotracker_window_size_spinbox.setRange(2, 4096)
        self.cotracker_window_size_spinbox.setValue(self.cotracker_settings.window_size)
        self.cotracker_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed CoTracker3 window. Smaller values reduce VRAM and can reduce long-range drift."
        )
        cotracker_layout.addRow("Window size:", self.cotracker_window_size_spinbox)

        self.cotracker_window_overlap_spinbox = QSpinBox()
        self.cotracker_window_overlap_spinbox.setRange(0, 1024)
        self.cotracker_window_overlap_spinbox.setValue(self.cotracker_settings.window_overlap)
        self.cotracker_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive CoTracker3 windows."
        )
        cotracker_layout.addRow("Window overlap:", self.cotracker_window_overlap_spinbox)

        self.cotracker_visibility_checkbox = QCheckBox("Use visibility gating")
        self.cotracker_visibility_checkbox.setChecked(self.cotracker_settings.use_visibility_gating)
        self.cotracker_visibility_checkbox.setToolTip(
            "Keep only points marked visible by CoTracker3."
        )
        cotracker_layout.addRow(self.cotracker_visibility_checkbox)

        self.cotracker_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.cotracker_keep_gaps_checkbox.setChecked(self.cotracker_settings.keep_invisible_points_as_gaps)
        self.cotracker_keep_gaps_checkbox.setToolTip(
            "If enabled, invisible CoTracker3 predictions create gaps instead of forced points."
        )
        cotracker_layout.addRow(self.cotracker_keep_gaps_checkbox)

        self.cotracker_max_step_spinbox = QDoubleSpinBox()
        self.cotracker_max_step_spinbox.setDecimals(1)
        self.cotracker_max_step_spinbox.setRange(0.0, 500.0)
        self.cotracker_max_step_spinbox.setSingleStep(1.0)
        self.cotracker_max_step_spinbox.setValue(self.cotracker_settings.max_step_distance_px)
        self.cotracker_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        cotracker_layout.addRow("Max jump (px):", self.cotracker_max_step_spinbox)

        self.cotracker_max_reacquisition_gap_spinbox = QSpinBox()
        self.cotracker_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.cotracker_max_reacquisition_gap_spinbox.setValue(self.cotracker_settings.max_reacquisition_gap)
        self.cotracker_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        cotracker_layout.addRow("Max reacquisition gap:", self.cotracker_max_reacquisition_gap_spinbox)

        self.cotracker_terminate_after_invisible_spinbox = QSpinBox()
        self.cotracker_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.cotracker_terminate_after_invisible_spinbox.setValue(
            self.cotracker_settings.terminate_after_invisible_frames
        )
        self.cotracker_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        cotracker_layout.addRow("Terminate after gap:", self.cotracker_terminate_after_invisible_spinbox)

        self.cotracker_center_mode_combo = QComboBox()
        self.cotracker_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.cotracker_center_mode_combo.addItem("Manual", "manual")
        cotracker_center_mode_idx = self.cotracker_center_mode_combo.findData(self.cotracker_settings.center_mode)
        self.cotracker_center_mode_combo.setCurrentIndex(cotracker_center_mode_idx if cotracker_center_mode_idx >= 0 else 0)
        self.cotracker_center_mode_combo.setToolTip(
            "Center used for radial CoTracker3 direction gating. Auto tries detections/pattern center and falls back to image center."
        )
        self.cotracker_center_mode_combo.currentIndexChanged.connect(self._on_cotracker_center_mode_changed)
        cotracker_layout.addRow("Direction center:", self.cotracker_center_mode_combo)

        cotracker_center_row_widget = QWidget()
        cotracker_center_row_layout = QHBoxLayout(cotracker_center_row_widget)
        cotracker_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.cotracker_center_x_spinbox = QDoubleSpinBox()
        self.cotracker_center_x_spinbox.setRange(0.0, 10000.0)
        self.cotracker_center_x_spinbox.setDecimals(1)
        self.cotracker_center_x_spinbox.setValue(self.cotracker_settings.manual_center_x)
        self.cotracker_center_y_spinbox = QDoubleSpinBox()
        self.cotracker_center_y_spinbox.setRange(0.0, 10000.0)
        self.cotracker_center_y_spinbox.setDecimals(1)
        self.cotracker_center_y_spinbox.setValue(self.cotracker_settings.manual_center_y)
        cotracker_center_row_layout.addWidget(self.cotracker_center_x_spinbox)
        cotracker_center_row_layout.addWidget(QLabel(","))
        cotracker_center_row_layout.addWidget(self.cotracker_center_y_spinbox)
        cotracker_layout.addRow("Manual center (x, y):", cotracker_center_row_widget)

        self.cotracker_direction_mode_combo = QComboBox()
        self.cotracker_direction_mode_combo.addItem("None", "none")
        self.cotracker_direction_mode_combo.addItem("Auto", "auto")
        self.cotracker_direction_mode_combo.addItem("Inward", "inward")
        self.cotracker_direction_mode_combo.addItem("Outward", "outward")
        cotracker_direction_idx = self.cotracker_direction_mode_combo.findData(self.cotracker_settings.direction_mode)
        self.cotracker_direction_mode_combo.setCurrentIndex(cotracker_direction_idx if cotracker_direction_idx >= 0 else 0)
        self.cotracker_direction_mode_combo.setToolTip(
            "Apply radial direction constraints to CoTracker3 trajectories."
        )
        cotracker_layout.addRow("Direction mode:", self.cotracker_direction_mode_combo)

        self.cotracker_lateral_tol_spinbox = QDoubleSpinBox()
        self.cotracker_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.cotracker_lateral_tol_spinbox.setSingleStep(1.0)
        self.cotracker_lateral_tol_spinbox.setValue(self.cotracker_settings.lateral_angle_tolerance_deg)
        self.cotracker_lateral_tol_spinbox.setToolTip(
            "Allowed angular deviation from the expected radial direction."
        )
        cotracker_layout.addRow("Angular tol (deg):", self.cotracker_lateral_tol_spinbox)

        self.cotracker_max_radial_jump_spinbox = QDoubleSpinBox()
        self.cotracker_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.cotracker_max_radial_jump_spinbox.setSingleStep(1.0)
        self.cotracker_max_radial_jump_spinbox.setValue(self.cotracker_settings.max_radial_jump_px)
        self.cotracker_max_radial_jump_spinbox.setToolTip(
            "Maximum accepted radial change between two accepted points. Set 0 to disable."
        )
        cotracker_layout.addRow("Max radial jump (px):", self.cotracker_max_radial_jump_spinbox)

        self.cotracker_recenter_mode_combo = QComboBox()
        self.cotracker_recenter_mode_combo.addItem("None", "none")
        self.cotracker_recenter_mode_combo.addItem("Peak", "peak")
        self.cotracker_recenter_mode_combo.addItem("Centroid", "centroid")
        self.cotracker_recenter_mode_combo.addItem("Gaussian", "gaussian")
        cotracker_recenter_mode_idx = self.cotracker_recenter_mode_combo.findData(self.cotracker_settings.recenter_mode)
        self.cotracker_recenter_mode_combo.setCurrentIndex(cotracker_recenter_mode_idx if cotracker_recenter_mode_idx >= 0 else 2)
        self.cotracker_recenter_mode_combo.setToolTip(
            "Local refinement applied after each CoTracker3 prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        cotracker_layout.addRow("Recenter mode:", self.cotracker_recenter_mode_combo)

        self.cotracker_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.cotracker_centroid_multiplier_spinbox.setDecimals(2)
        self.cotracker_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.cotracker_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.cotracker_centroid_multiplier_spinbox.setValue(self.cotracker_settings.recenter_patch_multiplier)
        self.cotracker_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        cotracker_layout.addRow("Recenter radius x ROI:", self.cotracker_centroid_multiplier_spinbox)

        self.cotracker_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.cotracker_centroid_rel_thr_spinbox.setDecimals(2)
        self.cotracker_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.cotracker_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.cotracker_centroid_rel_thr_spinbox.setValue(self.cotracker_settings.centroid_rel_threshold)
        self.cotracker_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        cotracker_layout.addRow("Recenter threshold:", self.cotracker_centroid_rel_thr_spinbox)

        self.cotracker_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.cotracker_centroid_max_shift_spinbox.setDecimals(2)
        self.cotracker_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.cotracker_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.cotracker_centroid_max_shift_spinbox.setValue(self.cotracker_settings.centroid_max_shift_ratio)
        self.cotracker_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        cotracker_layout.addRow("Max recenter shift:", self.cotracker_centroid_max_shift_spinbox)

        self.cotracker_warning_label = QLabel()
        self.cotracker_warning_label.setStyleSheet("color: #d9534f;")
        self.cotracker_warning_label.setWordWrap(True)
        cotracker_layout.addRow(self.cotracker_warning_label)
        self._on_cotracker_center_mode_changed(self.cotracker_center_mode_combo.currentIndex())

        form_layout.addRow(self.cotracker_group)

        # --- Offline TAPNext parameters ---
        self.tapnext_group = QGroupBox("Offline TAPNext")
        tapnext_layout = QFormLayout(self.tapnext_group)

        self.tapnext_repo_path_edit = QLineEdit(self.tapnext_settings.repo_path or "")
        self.tapnext_repo_path_edit.setPlaceholderText("Path to local tapnet repo root")
        self.tapnext_repo_path_edit.setToolTip(
            "Local clone of the tapnet repository. Must contain tapnet/tapnext/tapnext_torch.py."
        )
        self.tapnext_repo_path_edit.editingFinished.connect(self._refresh_tapnext_checkpoints)
        tapnext_layout.addRow("Repo path:", self.tapnext_repo_path_edit)

        tapnext_model_row_widget = QWidget()
        tapnext_model_row_layout = QHBoxLayout(tapnext_model_row_widget)
        tapnext_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.tapnext_checkpoint_combo = QComboBox()
        self.tapnext_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tapnext_checkpoint_combo.setToolTip(
            "Select a TAPNext checkpoint (.pt or .npz) from tracking_models/tapnext."
        )
        self.tapnext_checkpoint_combo.currentIndexChanged.connect(self._on_tapnext_checkpoint_selected)
        self.tapnext_refresh_button = QPushButton("Refresh")
        self.tapnext_refresh_button.setToolTip(
            "Re-scan tracking_models/tapnext and re-run TAPNext environment diagnostics."
        )
        self.tapnext_refresh_button.clicked.connect(self._refresh_tapnext_checkpoints)
        tapnext_model_row_layout.addWidget(self.tapnext_checkpoint_combo)
        tapnext_model_row_layout.addWidget(self.tapnext_refresh_button)
        tapnext_layout.addRow("Checkpoint:", tapnext_model_row_widget)

        self.tapnext_device_combo = QComboBox()
        self.tapnext_device_combo.addItem("Auto", "auto")
        self.tapnext_device_combo.addItem("GPU", "gpu")
        self.tapnext_device_combo.addItem("CPU", "cpu")
        self.tapnext_device_combo.setCurrentIndex(0)
        self.tapnext_device_combo.setToolTip("Device used for TAPNext inference.")
        tapnext_layout.addRow("Device:", self.tapnext_device_combo)

        self.tapnext_inference_resolution_combo = QComboBox()
        for value in (256, 384, 512):
            self.tapnext_inference_resolution_combo.addItem(f"{value}", value)
        self.tapnext_inference_resolution_combo.setCurrentIndex(0)
        self.tapnext_inference_resolution_combo.setToolTip(
            "Square resolution used before TAPNext inference. Higher values may help on LEED but cost more VRAM."
        )
        tapnext_layout.addRow("Inference resolution:", self.tapnext_inference_resolution_combo)

        self.tapnext_autocast_checkbox = QCheckBox("Use autocast")
        self.tapnext_autocast_checkbox.setChecked(self.tapnext_settings.autocast)
        self.tapnext_autocast_checkbox.setToolTip(
            "Enable torch autocast during sequential TAPNext inference to reduce VRAM usage on GPU."
        )
        tapnext_layout.addRow(self.tapnext_autocast_checkbox)

        self.tapnext_window_size_spinbox = QSpinBox()
        self.tapnext_window_size_spinbox.setRange(2, 4096)
        self.tapnext_window_size_spinbox.setValue(self.tapnext_settings.window_size)
        self.tapnext_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed TAPNext window. Smaller values reduce VRAM and can reduce long-range drift."
        )
        tapnext_layout.addRow("Window size:", self.tapnext_window_size_spinbox)

        self.tapnext_window_overlap_spinbox = QSpinBox()
        self.tapnext_window_overlap_spinbox.setRange(0, 1024)
        self.tapnext_window_overlap_spinbox.setValue(self.tapnext_settings.window_overlap)
        self.tapnext_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive TAPNext windows. "
            "In multi-window mode TAPNext is reseeded from the overlap, so the backend clamps this to at least 1 frame."
        )
        tapnext_layout.addRow("Window overlap:", self.tapnext_window_overlap_spinbox)

        self.tapnext_visibility_checkbox = QCheckBox("Use visibility gating")
        self.tapnext_visibility_checkbox.setChecked(self.tapnext_settings.use_visibility_gating)
        self.tapnext_visibility_checkbox.setToolTip(
            "Drop points whose TAPNext visibility score falls below the threshold."
        )
        tapnext_layout.addRow(self.tapnext_visibility_checkbox)

        self.tapnext_visibility_threshold_spinbox = QDoubleSpinBox()
        self.tapnext_visibility_threshold_spinbox.setDecimals(2)
        self.tapnext_visibility_threshold_spinbox.setRange(0.01, 0.99)
        self.tapnext_visibility_threshold_spinbox.setSingleStep(0.05)
        self.tapnext_visibility_threshold_spinbox.setValue(self.tapnext_settings.visibility_threshold)
        self.tapnext_visibility_threshold_spinbox.setToolTip("Minimum visibility score to keep a point.")
        tapnext_layout.addRow("Visibility threshold:", self.tapnext_visibility_threshold_spinbox)

        self.tapnext_certainty_checkbox = QCheckBox("Use certainty gating")
        self.tapnext_certainty_checkbox.setChecked(self.tapnext_settings.use_certainty_gating)
        self.tapnext_certainty_checkbox.setToolTip(
            "Use TAPNext tracker certainty to suppress unstable predictions."
        )
        tapnext_layout.addRow(self.tapnext_certainty_checkbox)

        self.tapnext_certainty_threshold_spinbox = QDoubleSpinBox()
        self.tapnext_certainty_threshold_spinbox.setDecimals(2)
        self.tapnext_certainty_threshold_spinbox.setRange(0.01, 0.99)
        self.tapnext_certainty_threshold_spinbox.setSingleStep(0.05)
        self.tapnext_certainty_threshold_spinbox.setValue(self.tapnext_settings.certainty_threshold)
        self.tapnext_certainty_threshold_spinbox.setToolTip("Minimum certainty score to keep a point.")
        tapnext_layout.addRow("Certainty threshold:", self.tapnext_certainty_threshold_spinbox)

        self.tapnext_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.tapnext_keep_gaps_checkbox.setChecked(self.tapnext_settings.keep_invisible_points_as_gaps)
        self.tapnext_keep_gaps_checkbox.setToolTip(
            "If enabled, invisible TAPNext predictions create gaps instead of forced points."
        )
        tapnext_layout.addRow(self.tapnext_keep_gaps_checkbox)

        self.tapnext_max_step_spinbox = QDoubleSpinBox()
        self.tapnext_max_step_spinbox.setDecimals(1)
        self.tapnext_max_step_spinbox.setRange(0.0, 500.0)
        self.tapnext_max_step_spinbox.setSingleStep(1.0)
        self.tapnext_max_step_spinbox.setValue(self.tapnext_settings.max_step_distance_px)
        self.tapnext_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        tapnext_layout.addRow("Max jump (px):", self.tapnext_max_step_spinbox)

        self.tapnext_max_reacquisition_gap_spinbox = QSpinBox()
        self.tapnext_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.tapnext_max_reacquisition_gap_spinbox.setValue(self.tapnext_settings.max_reacquisition_gap)
        self.tapnext_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        tapnext_layout.addRow("Max reacquisition gap:", self.tapnext_max_reacquisition_gap_spinbox)

        self.tapnext_terminate_after_invisible_spinbox = QSpinBox()
        self.tapnext_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.tapnext_terminate_after_invisible_spinbox.setValue(
            self.tapnext_settings.terminate_after_invisible_frames
        )
        self.tapnext_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        tapnext_layout.addRow("Terminate after gap:", self.tapnext_terminate_after_invisible_spinbox)

        self.tapnext_center_mode_combo = QComboBox()
        self.tapnext_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.tapnext_center_mode_combo.addItem("Manual", "manual")
        tapnext_center_mode_idx = self.tapnext_center_mode_combo.findData(self.tapnext_settings.center_mode)
        self.tapnext_center_mode_combo.setCurrentIndex(
            tapnext_center_mode_idx if tapnext_center_mode_idx >= 0 else 0
        )
        self.tapnext_center_mode_combo.setToolTip(
            "Center used for radial TAPNext direction gating. Auto tries detections/pattern center and falls back to image center."
        )
        self.tapnext_center_mode_combo.currentIndexChanged.connect(self._on_tapnext_center_mode_changed)
        tapnext_layout.addRow("Direction center:", self.tapnext_center_mode_combo)

        tapnext_center_row_widget = QWidget()
        tapnext_center_row_layout = QHBoxLayout(tapnext_center_row_widget)
        tapnext_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.tapnext_center_x_spinbox = QDoubleSpinBox()
        self.tapnext_center_x_spinbox.setRange(0.0, 10000.0)
        self.tapnext_center_x_spinbox.setDecimals(1)
        self.tapnext_center_x_spinbox.setValue(self.tapnext_settings.manual_center_x)
        self.tapnext_center_y_spinbox = QDoubleSpinBox()
        self.tapnext_center_y_spinbox.setRange(0.0, 10000.0)
        self.tapnext_center_y_spinbox.setDecimals(1)
        self.tapnext_center_y_spinbox.setValue(self.tapnext_settings.manual_center_y)
        tapnext_center_row_layout.addWidget(self.tapnext_center_x_spinbox)
        tapnext_center_row_layout.addWidget(QLabel(","))
        tapnext_center_row_layout.addWidget(self.tapnext_center_y_spinbox)
        tapnext_layout.addRow("Manual center (x, y):", tapnext_center_row_widget)

        self.tapnext_direction_mode_combo = QComboBox()
        self.tapnext_direction_mode_combo.addItem("None", "none")
        self.tapnext_direction_mode_combo.addItem("Auto", "auto")
        self.tapnext_direction_mode_combo.addItem("Inward", "inward")
        self.tapnext_direction_mode_combo.addItem("Outward", "outward")
        tapnext_direction_idx = self.tapnext_direction_mode_combo.findData(self.tapnext_settings.direction_mode)
        self.tapnext_direction_mode_combo.setCurrentIndex(
            tapnext_direction_idx if tapnext_direction_idx >= 0 else 0
        )
        self.tapnext_direction_mode_combo.setToolTip(
            "Apply radial direction constraints to TAPNext trajectories."
        )
        tapnext_layout.addRow("Direction mode:", self.tapnext_direction_mode_combo)

        self.tapnext_lateral_tol_spinbox = QDoubleSpinBox()
        self.tapnext_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.tapnext_lateral_tol_spinbox.setSingleStep(1.0)
        self.tapnext_lateral_tol_spinbox.setValue(self.tapnext_settings.lateral_angle_tolerance_deg)
        self.tapnext_lateral_tol_spinbox.setToolTip(
            "Allowed angular deviation from the expected radial direction."
        )
        tapnext_layout.addRow("Angular tol (deg):", self.tapnext_lateral_tol_spinbox)

        self.tapnext_max_radial_jump_spinbox = QDoubleSpinBox()
        self.tapnext_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.tapnext_max_radial_jump_spinbox.setSingleStep(1.0)
        self.tapnext_max_radial_jump_spinbox.setValue(self.tapnext_settings.max_radial_jump_px)
        self.tapnext_max_radial_jump_spinbox.setToolTip(
            "Maximum accepted radial change between two accepted points. Set 0 to disable."
        )
        tapnext_layout.addRow("Max radial jump (px):", self.tapnext_max_radial_jump_spinbox)

        self.tapnext_recenter_mode_combo = QComboBox()
        self.tapnext_recenter_mode_combo.addItem("None", "none")
        self.tapnext_recenter_mode_combo.addItem("Peak", "peak")
        self.tapnext_recenter_mode_combo.addItem("Centroid", "centroid")
        self.tapnext_recenter_mode_combo.addItem("Gaussian", "gaussian")
        tapnext_recenter_mode_idx = self.tapnext_recenter_mode_combo.findData(self.tapnext_settings.recenter_mode)
        self.tapnext_recenter_mode_combo.setCurrentIndex(
            tapnext_recenter_mode_idx if tapnext_recenter_mode_idx >= 0 else 2
        )
        self.tapnext_recenter_mode_combo.setToolTip(
            "Local refinement applied after each TAPNext prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        tapnext_layout.addRow("Recenter mode:", self.tapnext_recenter_mode_combo)

        self.tapnext_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.tapnext_centroid_multiplier_spinbox.setDecimals(2)
        self.tapnext_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.tapnext_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.tapnext_centroid_multiplier_spinbox.setValue(self.tapnext_settings.recenter_patch_multiplier)
        self.tapnext_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        tapnext_layout.addRow("Recenter radius x ROI:", self.tapnext_centroid_multiplier_spinbox)

        self.tapnext_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.tapnext_centroid_rel_thr_spinbox.setDecimals(2)
        self.tapnext_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.tapnext_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.tapnext_centroid_rel_thr_spinbox.setValue(self.tapnext_settings.centroid_rel_threshold)
        self.tapnext_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        tapnext_layout.addRow("Recenter threshold:", self.tapnext_centroid_rel_thr_spinbox)

        self.tapnext_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.tapnext_centroid_max_shift_spinbox.setDecimals(2)
        self.tapnext_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.tapnext_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.tapnext_centroid_max_shift_spinbox.setValue(self.tapnext_settings.centroid_max_shift_ratio)
        self.tapnext_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        tapnext_layout.addRow("Max recenter shift:", self.tapnext_centroid_max_shift_spinbox)

        self.tapnext_warning_label = QLabel()
        self.tapnext_warning_label.setStyleSheet("color: #d9534f;")
        self.tapnext_warning_label.setWordWrap(True)
        tapnext_layout.addRow(self.tapnext_warning_label)
        self._on_tapnext_center_mode_changed(self.tapnext_center_mode_combo.currentIndex())

        form_layout.addRow(self.tapnext_group)

        # --- Offline Track-On-R parameters ---
        self.trackonr_group = QGroupBox("Offline Track-On-R")
        trackonr_layout = QFormLayout(self.trackonr_group)

        self.trackonr_repo_path_edit = QLineEdit(self.trackonr_settings.repo_path or "")
        self.trackonr_repo_path_edit.setPlaceholderText("Path to local track_on repo root")
        self.trackonr_repo_path_edit.setToolTip(
            "Local clone of the track_on repository. Must contain model/trackon_predictor.py."
        )
        self.trackonr_repo_path_edit.editingFinished.connect(self._refresh_trackonr_checkpoints)
        trackonr_layout.addRow("Repo path:", self.trackonr_repo_path_edit)

        self.trackonr_python_executable_edit = QLineEdit(self.trackonr_settings.python_executable or "")
        self.trackonr_python_executable_edit.setPlaceholderText(
            "Optional python.exe from dedicated trackonr env"
        )
        self.trackonr_python_executable_edit.setToolTip(
            "Used only in subprocess mode. Point this to the interpreter of the dedicated Track-On-R environment."
        )
        self.trackonr_python_executable_edit.editingFinished.connect(self._refresh_trackonr_checkpoints)
        trackonr_layout.addRow("Python executable:", self.trackonr_python_executable_edit)

        self.trackonr_backend_mode_combo = QComboBox()
        self.trackonr_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.trackonr_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        trackonr_backend_mode_idx = self.trackonr_backend_mode_combo.findData(self.trackonr_settings.backend_mode)
        self.trackonr_backend_mode_combo.setCurrentIndex(trackonr_backend_mode_idx if trackonr_backend_mode_idx >= 0 else 0)
        self.trackonr_backend_mode_combo.setToolTip(
            "Subprocess lets Trackleed stay in the main env while Track-On-R runs in its own dedicated env."
        )
        self.trackonr_backend_mode_combo.currentIndexChanged.connect(self._on_trackonr_backend_mode_changed)
        trackonr_layout.addRow("Backend mode:", self.trackonr_backend_mode_combo)

        trackonr_model_row_widget = QWidget()
        trackonr_model_row_layout = QHBoxLayout(trackonr_model_row_widget)
        trackonr_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.trackonr_checkpoint_combo = QComboBox()
        self.trackonr_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.trackonr_checkpoint_combo.setToolTip(
            "Select a Track-On-R checkpoint (.pt or .pth) from tracking_models/trackonr."
        )
        self.trackonr_checkpoint_combo.currentIndexChanged.connect(self._on_trackonr_checkpoint_selected)
        self.trackonr_refresh_button = QPushButton("Refresh")
        self.trackonr_refresh_button.setToolTip(
            "Re-scan tracking_models/trackonr and re-run Track-On-R environment diagnostics."
        )
        self.trackonr_refresh_button.clicked.connect(self._refresh_trackonr_checkpoints)
        trackonr_model_row_layout.addWidget(self.trackonr_checkpoint_combo)
        trackonr_model_row_layout.addWidget(self.trackonr_refresh_button)
        trackonr_layout.addRow("Checkpoint:", trackonr_model_row_widget)

        self.trackonr_device_combo = QComboBox()
        self.trackonr_device_combo.addItem("Auto", "auto")
        self.trackonr_device_combo.addItem("GPU", "gpu")
        self.trackonr_device_combo.addItem("CPU", "cpu")
        trackonr_device_idx = self.trackonr_device_combo.findData(self.trackonr_settings.device)
        self.trackonr_device_combo.setCurrentIndex(trackonr_device_idx if trackonr_device_idx >= 0 else 0)
        self.trackonr_device_combo.setToolTip("Device used for Track-On-R inference.")
        trackonr_layout.addRow("Device:", self.trackonr_device_combo)

        self.trackonr_inference_resolution_combo = QComboBox()
        for label, value in (
            ("256 x 256", (256, 256)),
            ("384 x 512", (384, 512)),
            ("512 x 512", (512, 512)),
        ):
            self.trackonr_inference_resolution_combo.addItem(label, value)
        trackonr_res_idx = self.trackonr_inference_resolution_combo.findData(
            self.trackonr_settings.inference_resolution
        )
        self.trackonr_inference_resolution_combo.setCurrentIndex(trackonr_res_idx if trackonr_res_idx >= 0 else 1)
        self.trackonr_inference_resolution_combo.setToolTip(
            "Input resolution passed to Track-On-R. Higher values may help with tiny LEED spots but cost more VRAM."
        )
        trackonr_layout.addRow("Inference resolution:", self.trackonr_inference_resolution_combo)

        self.trackonr_support_grid_spinbox = QSpinBox()
        self.trackonr_support_grid_spinbox.setRange(0, 256)
        self.trackonr_support_grid_spinbox.setValue(self.trackonr_settings.support_grid_size)
        self.trackonr_support_grid_spinbox.setToolTip(
            "Additional support-grid queries used by Track-On-R around the user points. Higher values may help stability at extra cost."
        )
        trackonr_layout.addRow("Support grid size:", self.trackonr_support_grid_spinbox)

        self.trackonr_window_size_spinbox = QSpinBox()
        self.trackonr_window_size_spinbox.setRange(2, 4096)
        self.trackonr_window_size_spinbox.setValue(self.trackonr_settings.window_size)
        self.trackonr_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed Track-On-R window. Smaller values reduce VRAM and reset long temporal context more often."
        )
        trackonr_layout.addRow("Window size:", self.trackonr_window_size_spinbox)

        self.trackonr_window_overlap_spinbox = QSpinBox()
        self.trackonr_window_overlap_spinbox.setRange(0, 1024)
        self.trackonr_window_overlap_spinbox.setValue(self.trackonr_settings.window_overlap)
        self.trackonr_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive Track-On-R windows. "
            "In multi-window mode Track-On-R is reset and reseeded from the overlap, "
            "so the backend clamps this to at least 1 shared frame."
        )
        trackonr_layout.addRow("Window overlap:", self.trackonr_window_overlap_spinbox)

        self.trackonr_visibility_checkbox = QCheckBox("Use visibility gating")
        self.trackonr_visibility_checkbox.setChecked(self.trackonr_settings.use_visibility_gating)
        self.trackonr_visibility_checkbox.setToolTip(
            "Drop points whose Track-On-R visibility falls below the threshold."
        )
        trackonr_layout.addRow(self.trackonr_visibility_checkbox)

        self.trackonr_visibility_threshold_spinbox = QDoubleSpinBox()
        self.trackonr_visibility_threshold_spinbox.setDecimals(2)
        self.trackonr_visibility_threshold_spinbox.setRange(0.01, 0.99)
        self.trackonr_visibility_threshold_spinbox.setSingleStep(0.05)
        self.trackonr_visibility_threshold_spinbox.setValue(self.trackonr_settings.visibility_threshold)
        self.trackonr_visibility_threshold_spinbox.setToolTip("Minimum visibility score to keep a point.")
        trackonr_layout.addRow("Visibility threshold:", self.trackonr_visibility_threshold_spinbox)

        self.trackonr_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.trackonr_keep_gaps_checkbox.setChecked(self.trackonr_settings.keep_invisible_points_as_gaps)
        self.trackonr_keep_gaps_checkbox.setToolTip(
            "If enabled, invisible Track-On-R predictions create gaps instead of forced points."
        )
        trackonr_layout.addRow(self.trackonr_keep_gaps_checkbox)

        self.trackonr_max_step_spinbox = QDoubleSpinBox()
        self.trackonr_max_step_spinbox.setDecimals(1)
        self.trackonr_max_step_spinbox.setRange(0.0, 500.0)
        self.trackonr_max_step_spinbox.setSingleStep(1.0)
        self.trackonr_max_step_spinbox.setValue(self.trackonr_settings.max_step_distance_px)
        self.trackonr_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        trackonr_layout.addRow("Max jump (px):", self.trackonr_max_step_spinbox)

        self.trackonr_max_reacquisition_gap_spinbox = QSpinBox()
        self.trackonr_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.trackonr_max_reacquisition_gap_spinbox.setValue(self.trackonr_settings.max_reacquisition_gap)
        self.trackonr_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        trackonr_layout.addRow("Max reacquisition gap:", self.trackonr_max_reacquisition_gap_spinbox)

        self.trackonr_terminate_after_invisible_spinbox = QSpinBox()
        self.trackonr_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.trackonr_terminate_after_invisible_spinbox.setValue(
            self.trackonr_settings.terminate_after_invisible_frames
        )
        self.trackonr_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        trackonr_layout.addRow("Terminate after gap:", self.trackonr_terminate_after_invisible_spinbox)

        self.trackonr_center_mode_combo = QComboBox()
        self.trackonr_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.trackonr_center_mode_combo.addItem("Manual", "manual")
        trackonr_center_mode_idx = self.trackonr_center_mode_combo.findData(self.trackonr_settings.center_mode)
        self.trackonr_center_mode_combo.setCurrentIndex(
            trackonr_center_mode_idx if trackonr_center_mode_idx >= 0 else 0
        )
        self.trackonr_center_mode_combo.setToolTip(
            "Reference center used by directional gating. Auto estimates the LEED pattern center from detections."
        )
        self.trackonr_center_mode_combo.currentIndexChanged.connect(self._on_trackonr_center_mode_changed)
        trackonr_layout.addRow("Direction center:", self.trackonr_center_mode_combo)

        trackonr_center_row_widget = QWidget()
        trackonr_center_row_layout = QHBoxLayout(trackonr_center_row_widget)
        trackonr_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.trackonr_center_x_spinbox = QDoubleSpinBox()
        self.trackonr_center_x_spinbox.setRange(0.0, 10000.0)
        self.trackonr_center_x_spinbox.setDecimals(1)
        self.trackonr_center_x_spinbox.setValue(self.trackonr_settings.manual_center_x)
        self.trackonr_center_y_spinbox = QDoubleSpinBox()
        self.trackonr_center_y_spinbox.setRange(0.0, 10000.0)
        self.trackonr_center_y_spinbox.setDecimals(1)
        self.trackonr_center_y_spinbox.setValue(self.trackonr_settings.manual_center_y)
        trackonr_center_row_layout.addWidget(self.trackonr_center_x_spinbox)
        trackonr_center_row_layout.addWidget(QLabel(","))
        trackonr_center_row_layout.addWidget(self.trackonr_center_y_spinbox)
        trackonr_layout.addRow("Manual center (x, y):", trackonr_center_row_widget)

        self.trackonr_direction_mode_combo = QComboBox()
        self.trackonr_direction_mode_combo.addItem("None", "none")
        self.trackonr_direction_mode_combo.addItem("Auto", "auto")
        self.trackonr_direction_mode_combo.addItem("Inward", "inward")
        self.trackonr_direction_mode_combo.addItem("Outward", "outward")
        trackonr_direction_idx = self.trackonr_direction_mode_combo.findData(self.trackonr_settings.direction_mode)
        self.trackonr_direction_mode_combo.setCurrentIndex(
            trackonr_direction_idx if trackonr_direction_idx >= 0 else 0
        )
        self.trackonr_direction_mode_combo.setToolTip(
            "Optional directional prior for radial motion relative to the LEED center."
        )
        trackonr_layout.addRow("Direction mode:", self.trackonr_direction_mode_combo)

        self.trackonr_lateral_tol_spinbox = QDoubleSpinBox()
        self.trackonr_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.trackonr_lateral_tol_spinbox.setSingleStep(1.0)
        self.trackonr_lateral_tol_spinbox.setValue(self.trackonr_settings.lateral_angle_tolerance_deg)
        self.trackonr_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        trackonr_layout.addRow("Angular tol (deg):", self.trackonr_lateral_tol_spinbox)

        self.trackonr_max_radial_jump_spinbox = QDoubleSpinBox()
        self.trackonr_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.trackonr_max_radial_jump_spinbox.setSingleStep(1.0)
        self.trackonr_max_radial_jump_spinbox.setValue(self.trackonr_settings.max_radial_jump_px)
        self.trackonr_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        trackonr_layout.addRow("Max radial jump (px):", self.trackonr_max_radial_jump_spinbox)

        self.trackonr_recenter_mode_combo = QComboBox()
        self.trackonr_recenter_mode_combo.addItem("None", "none")
        self.trackonr_recenter_mode_combo.addItem("Peak", "peak")
        self.trackonr_recenter_mode_combo.addItem("Centroid", "centroid")
        self.trackonr_recenter_mode_combo.addItem("Gaussian", "gaussian")
        trackonr_recenter_mode_idx = self.trackonr_recenter_mode_combo.findData(self.trackonr_settings.recenter_mode)
        self.trackonr_recenter_mode_combo.setCurrentIndex(
            trackonr_recenter_mode_idx if trackonr_recenter_mode_idx >= 0 else 2
        )
        self.trackonr_recenter_mode_combo.setToolTip(
            "Local refinement applied after each Track-On-R prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        trackonr_layout.addRow("Recenter mode:", self.trackonr_recenter_mode_combo)

        self.trackonr_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.trackonr_centroid_multiplier_spinbox.setDecimals(2)
        self.trackonr_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.trackonr_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.trackonr_centroid_multiplier_spinbox.setValue(self.trackonr_settings.recenter_patch_multiplier)
        self.trackonr_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        trackonr_layout.addRow("Recenter radius x ROI:", self.trackonr_centroid_multiplier_spinbox)

        self.trackonr_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.trackonr_centroid_rel_thr_spinbox.setDecimals(2)
        self.trackonr_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.trackonr_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.trackonr_centroid_rel_thr_spinbox.setValue(self.trackonr_settings.centroid_rel_threshold)
        self.trackonr_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        trackonr_layout.addRow("Recenter threshold:", self.trackonr_centroid_rel_thr_spinbox)

        self.trackonr_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.trackonr_centroid_max_shift_spinbox.setDecimals(2)
        self.trackonr_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.trackonr_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.trackonr_centroid_max_shift_spinbox.setValue(self.trackonr_settings.centroid_max_shift_ratio)
        self.trackonr_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        trackonr_layout.addRow("Max recenter shift:", self.trackonr_centroid_max_shift_spinbox)

        self.trackonr_warning_label = QLabel()
        self.trackonr_warning_label.setStyleSheet("color: #d9534f;")
        self.trackonr_warning_label.setWordWrap(True)
        trackonr_layout.addRow(self.trackonr_warning_label)
        self._on_trackonr_backend_mode_changed(self.trackonr_backend_mode_combo.currentIndex())
        self._on_trackonr_center_mode_changed(self.trackonr_center_mode_combo.currentIndex())

        form_layout.addRow(self.trackonr_group)

        # --- Offline PIPs++ parameters ---
        self.pipspp_group = QGroupBox("Offline PIPs++")
        pipspp_layout = QFormLayout(self.pipspp_group)

        self.pipspp_repo_path_edit = QLineEdit(self.pipspp_settings.repo_path or "")
        self.pipspp_repo_path_edit.setPlaceholderText("Path to local pipspp repo root")
        self.pipspp_repo_path_edit.setToolTip(
            "Local clone of the PIPs++ repository. Must contain nets/pips2.py and saverloader.py."
        )
        self.pipspp_repo_path_edit.editingFinished.connect(self._refresh_pipspp_checkpoints)
        pipspp_layout.addRow("Repo path:", self.pipspp_repo_path_edit)

        self.pipspp_python_executable_edit = QLineEdit(self.pipspp_settings.python_executable or "")
        self.pipspp_python_executable_edit.setPlaceholderText(
            "Optional python.exe from dedicated pipspp env"
        )
        self.pipspp_python_executable_edit.setToolTip(
            "Used only in subprocess mode. Point this to the interpreter of the dedicated PIPs++ environment."
        )
        self.pipspp_python_executable_edit.editingFinished.connect(self._refresh_pipspp_checkpoints)
        pipspp_layout.addRow("Python executable:", self.pipspp_python_executable_edit)

        self.pipspp_backend_mode_combo = QComboBox()
        self.pipspp_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.pipspp_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        pipspp_backend_mode_idx = self.pipspp_backend_mode_combo.findData(self.pipspp_settings.backend_mode)
        self.pipspp_backend_mode_combo.setCurrentIndex(pipspp_backend_mode_idx if pipspp_backend_mode_idx >= 0 else 0)
        self.pipspp_backend_mode_combo.setToolTip(
            "Subprocess lets Trackleed stay in the main env while PIPs++ runs in its own dedicated env."
        )
        self.pipspp_backend_mode_combo.currentIndexChanged.connect(self._on_pipspp_backend_mode_changed)
        pipspp_layout.addRow("Backend mode:", self.pipspp_backend_mode_combo)

        pipspp_model_row_widget = QWidget()
        pipspp_model_row_layout = QHBoxLayout(pipspp_model_row_widget)
        pipspp_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.pipspp_checkpoint_combo = QComboBox()
        self.pipspp_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.pipspp_checkpoint_combo.setToolTip(
            "Select a PIPs++ checkpoint reference from tracking_models/pipspp or repo/reference_model."
        )
        self.pipspp_checkpoint_combo.currentIndexChanged.connect(self._on_pipspp_checkpoint_selected)
        self.pipspp_refresh_button = QPushButton("Refresh")
        self.pipspp_refresh_button.setToolTip(
            "Re-scan tracking_models/pipspp and re-run PIPs++ environment diagnostics."
        )
        self.pipspp_refresh_button.clicked.connect(self._refresh_pipspp_checkpoints)
        pipspp_model_row_layout.addWidget(self.pipspp_checkpoint_combo)
        pipspp_model_row_layout.addWidget(self.pipspp_refresh_button)
        pipspp_layout.addRow("Checkpoint:", pipspp_model_row_widget)

        self.pipspp_device_combo = QComboBox()
        self.pipspp_device_combo.addItem("Auto", "auto")
        self.pipspp_device_combo.addItem("GPU", "gpu")
        self.pipspp_device_combo.addItem("CPU", "cpu")
        pipspp_device_idx = self.pipspp_device_combo.findData(self.pipspp_settings.device)
        self.pipspp_device_combo.setCurrentIndex(pipspp_device_idx if pipspp_device_idx >= 0 else 0)
        self.pipspp_device_combo.setToolTip("Device used for PIPs++ inference. Upstream runtime currently expects CUDA.")
        pipspp_layout.addRow("Device:", self.pipspp_device_combo)

        self.pipspp_inference_resolution_combo = QComboBox()
        for label, value in (
            ("256 x 384", (256, 384)),
            ("384 x 512", (384, 512)),
            ("512 x 896", (512, 896)),
        ):
            self.pipspp_inference_resolution_combo.addItem(label, value)
        pipspp_res_idx = self.pipspp_inference_resolution_combo.findData(
            self.pipspp_settings.inference_resolution
        )
        self.pipspp_inference_resolution_combo.setCurrentIndex(pipspp_res_idx if pipspp_res_idx >= 0 else 0)
        self.pipspp_inference_resolution_combo.setToolTip(
            "Input resolution passed to PIPs++. Values should remain divisible by 32."
        )
        pipspp_layout.addRow("Inference resolution:", self.pipspp_inference_resolution_combo)

        self.pipspp_iters_spinbox = QSpinBox()
        self.pipspp_iters_spinbox.setRange(1, 32)
        self.pipspp_iters_spinbox.setValue(self.pipspp_settings.iters)
        self.pipspp_iters_spinbox.setToolTip(
            "Number of iterative PIPs++ refinement steps. Higher values may improve stability at higher cost."
        )
        pipspp_layout.addRow("Iterations:", self.pipspp_iters_spinbox)

        self.pipspp_beautify_checkbox = QCheckBox("Use beautify")
        self.pipspp_beautify_checkbox.setChecked(self.pipspp_settings.beautify)
        self.pipspp_beautify_checkbox.setToolTip(
            "Enable the upstream PIPs++ smoothing step during the last refinement iterations."
        )
        pipspp_layout.addRow(self.pipspp_beautify_checkbox)

        self.pipspp_inbounds_visibility_checkbox = QCheckBox("Use in-bounds visibility")
        self.pipspp_inbounds_visibility_checkbox.setChecked(self.pipspp_settings.use_inbounds_visibility)
        self.pipspp_inbounds_visibility_checkbox.setToolTip(
            "PIPs++ does not expose visibility logits in inference, so Trackleed uses an in-bounds heuristic."
        )
        pipspp_layout.addRow(self.pipspp_inbounds_visibility_checkbox)

        self.pipspp_window_size_spinbox = QSpinBox()
        self.pipspp_window_size_spinbox.setRange(2, 4096)
        self.pipspp_window_size_spinbox.setValue(self.pipspp_settings.window_size)
        self.pipspp_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed PIPs++ window. Smaller values reduce VRAM."
        )
        pipspp_layout.addRow("Window size:", self.pipspp_window_size_spinbox)

        self.pipspp_window_overlap_spinbox = QSpinBox()
        self.pipspp_window_overlap_spinbox.setRange(0, 1024)
        self.pipspp_window_overlap_spinbox.setValue(self.pipspp_settings.window_overlap)
        self.pipspp_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive PIPs++ windows. "
            "Trackleed reseeds the next window from this overlap and enforces at least one shared frame when multiple windows are used."
        )
        pipspp_layout.addRow("Window overlap:", self.pipspp_window_overlap_spinbox)

        self.pipspp_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.pipspp_keep_gaps_checkbox.setChecked(self.pipspp_settings.keep_invisible_points_as_gaps)
        self.pipspp_keep_gaps_checkbox.setToolTip(
            "If enabled, rejected or out-of-bounds PIPs++ predictions create gaps instead of forced points."
        )
        pipspp_layout.addRow(self.pipspp_keep_gaps_checkbox)

        self.pipspp_max_step_spinbox = QDoubleSpinBox()
        self.pipspp_max_step_spinbox.setDecimals(1)
        self.pipspp_max_step_spinbox.setRange(0.0, 500.0)
        self.pipspp_max_step_spinbox.setSingleStep(1.0)
        self.pipspp_max_step_spinbox.setValue(self.pipspp_settings.max_step_distance_px)
        self.pipspp_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        pipspp_layout.addRow("Max jump (px):", self.pipspp_max_step_spinbox)

        self.pipspp_max_reacquisition_gap_spinbox = QSpinBox()
        self.pipspp_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.pipspp_max_reacquisition_gap_spinbox.setValue(self.pipspp_settings.max_reacquisition_gap)
        self.pipspp_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        pipspp_layout.addRow("Max reacquisition gap:", self.pipspp_max_reacquisition_gap_spinbox)

        self.pipspp_terminate_after_invisible_spinbox = QSpinBox()
        self.pipspp_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.pipspp_terminate_after_invisible_spinbox.setValue(
            self.pipspp_settings.terminate_after_invisible_frames
        )
        self.pipspp_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        pipspp_layout.addRow("Terminate after gap:", self.pipspp_terminate_after_invisible_spinbox)

        self.pipspp_center_mode_combo = QComboBox()
        self.pipspp_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.pipspp_center_mode_combo.addItem("Manual", "manual")
        pipspp_center_mode_idx = self.pipspp_center_mode_combo.findData(self.pipspp_settings.center_mode)
        self.pipspp_center_mode_combo.setCurrentIndex(
            pipspp_center_mode_idx if pipspp_center_mode_idx >= 0 else 0
        )
        self.pipspp_center_mode_combo.setToolTip(
            "Reference center used by directional gating. Auto estimates the LEED pattern center from detections."
        )
        self.pipspp_center_mode_combo.currentIndexChanged.connect(self._on_pipspp_center_mode_changed)
        pipspp_layout.addRow("Direction center:", self.pipspp_center_mode_combo)

        pipspp_center_row_widget = QWidget()
        pipspp_center_row_layout = QHBoxLayout(pipspp_center_row_widget)
        pipspp_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.pipspp_center_x_spinbox = QDoubleSpinBox()
        self.pipspp_center_x_spinbox.setRange(0.0, 10000.0)
        self.pipspp_center_x_spinbox.setDecimals(1)
        self.pipspp_center_x_spinbox.setValue(self.pipspp_settings.manual_center_x)
        self.pipspp_center_y_spinbox = QDoubleSpinBox()
        self.pipspp_center_y_spinbox.setRange(0.0, 10000.0)
        self.pipspp_center_y_spinbox.setDecimals(1)
        self.pipspp_center_y_spinbox.setValue(self.pipspp_settings.manual_center_y)
        pipspp_center_row_layout.addWidget(self.pipspp_center_x_spinbox)
        pipspp_center_row_layout.addWidget(QLabel(","))
        pipspp_center_row_layout.addWidget(self.pipspp_center_y_spinbox)
        pipspp_layout.addRow("Manual center (x, y):", pipspp_center_row_widget)

        self.pipspp_direction_mode_combo = QComboBox()
        self.pipspp_direction_mode_combo.addItem("None", "none")
        self.pipspp_direction_mode_combo.addItem("Auto", "auto")
        self.pipspp_direction_mode_combo.addItem("Inward", "inward")
        self.pipspp_direction_mode_combo.addItem("Outward", "outward")
        pipspp_direction_idx = self.pipspp_direction_mode_combo.findData(self.pipspp_settings.direction_mode)
        self.pipspp_direction_mode_combo.setCurrentIndex(
            pipspp_direction_idx if pipspp_direction_idx >= 0 else 0
        )
        self.pipspp_direction_mode_combo.setToolTip(
            "Optional directional prior for radial motion relative to the LEED center."
        )
        pipspp_layout.addRow("Direction mode:", self.pipspp_direction_mode_combo)

        self.pipspp_lateral_tol_spinbox = QDoubleSpinBox()
        self.pipspp_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.pipspp_lateral_tol_spinbox.setSingleStep(1.0)
        self.pipspp_lateral_tol_spinbox.setValue(self.pipspp_settings.lateral_angle_tolerance_deg)
        self.pipspp_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        pipspp_layout.addRow("Angular tol (deg):", self.pipspp_lateral_tol_spinbox)

        self.pipspp_max_radial_jump_spinbox = QDoubleSpinBox()
        self.pipspp_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.pipspp_max_radial_jump_spinbox.setSingleStep(1.0)
        self.pipspp_max_radial_jump_spinbox.setValue(self.pipspp_settings.max_radial_jump_px)
        self.pipspp_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        pipspp_layout.addRow("Max radial jump (px):", self.pipspp_max_radial_jump_spinbox)

        self.pipspp_recenter_mode_combo = QComboBox()
        self.pipspp_recenter_mode_combo.addItem("None", "none")
        self.pipspp_recenter_mode_combo.addItem("Peak", "peak")
        self.pipspp_recenter_mode_combo.addItem("Centroid", "centroid")
        self.pipspp_recenter_mode_combo.addItem("Gaussian", "gaussian")
        pipspp_recenter_mode_idx = self.pipspp_recenter_mode_combo.findData(self.pipspp_settings.recenter_mode)
        self.pipspp_recenter_mode_combo.setCurrentIndex(
            pipspp_recenter_mode_idx if pipspp_recenter_mode_idx >= 0 else 2
        )
        self.pipspp_recenter_mode_combo.setToolTip(
            "Local refinement applied after each PIPs++ prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        pipspp_layout.addRow("Recenter mode:", self.pipspp_recenter_mode_combo)

        self.pipspp_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.pipspp_centroid_multiplier_spinbox.setDecimals(2)
        self.pipspp_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.pipspp_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.pipspp_centroid_multiplier_spinbox.setValue(self.pipspp_settings.recenter_patch_multiplier)
        self.pipspp_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        pipspp_layout.addRow("Recenter radius x ROI:", self.pipspp_centroid_multiplier_spinbox)

        self.pipspp_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.pipspp_centroid_rel_thr_spinbox.setDecimals(2)
        self.pipspp_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.pipspp_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.pipspp_centroid_rel_thr_spinbox.setValue(self.pipspp_settings.centroid_rel_threshold)
        self.pipspp_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        pipspp_layout.addRow("Recenter threshold:", self.pipspp_centroid_rel_thr_spinbox)

        self.pipspp_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.pipspp_centroid_max_shift_spinbox.setDecimals(2)
        self.pipspp_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.pipspp_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.pipspp_centroid_max_shift_spinbox.setValue(self.pipspp_settings.centroid_max_shift_ratio)
        self.pipspp_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        pipspp_layout.addRow("Max recenter shift:", self.pipspp_centroid_max_shift_spinbox)

        self.pipspp_warning_label = QLabel()
        self.pipspp_warning_label.setStyleSheet("color: #d9534f;")
        self.pipspp_warning_label.setWordWrap(True)
        pipspp_layout.addRow(self.pipspp_warning_label)
        self._on_pipspp_backend_mode_changed(self.pipspp_backend_mode_combo.currentIndex())
        self._on_pipspp_center_mode_changed(self.pipspp_center_mode_combo.currentIndex())

        form_layout.addRow(self.pipspp_group)

        # --- Offline MFTIQ parameters ---
        self.mftiq_group = QGroupBox("Offline MFTIQ")
        mftiq_layout = QFormLayout(self.mftiq_group)

        self.mftiq_repo_path_edit = QLineEdit(self.mftiq_settings.repo_path or "")
        self.mftiq_repo_path_edit.setPlaceholderText("Path to local mftiq repo root")
        self.mftiq_repo_path_edit.setToolTip(
            "Local clone of the MFTIQ repository. Must contain demo.py, configs/, and src/MFTIQ/."
        )
        self.mftiq_repo_path_edit.editingFinished.connect(self._refresh_mftiq_checkpoints)
        mftiq_layout.addRow("Repo path:", self.mftiq_repo_path_edit)

        self.mftiq_python_executable_edit = QLineEdit(self.mftiq_settings.python_executable or "")
        self.mftiq_python_executable_edit.setPlaceholderText(
            "Optional path to the Python executable inside the dedicated mftiq env"
        )
        self.mftiq_python_executable_edit.setToolTip(
            "Used by the recommended subprocess backend. Leave empty only if Trackleed itself runs "
            "inside the MFTIQ environment."
        )
        self.mftiq_python_executable_edit.editingFinished.connect(self._refresh_mftiq_checkpoints)
        mftiq_layout.addRow("Python executable:", self.mftiq_python_executable_edit)

        self.mftiq_backend_mode_combo = QComboBox()
        self.mftiq_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.mftiq_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        mftiq_backend_mode_idx = self.mftiq_backend_mode_combo.findData(self.mftiq_settings.backend_mode)
        self.mftiq_backend_mode_combo.setCurrentIndex(mftiq_backend_mode_idx if mftiq_backend_mode_idx >= 0 else 0)
        self.mftiq_backend_mode_combo.setToolTip(
            "Subprocess keeps the heavy MFTIQ runtime isolated in its own env. "
            "Direct import requires running the whole app inside the MFTIQ env."
        )
        self.mftiq_backend_mode_combo.currentIndexChanged.connect(self._on_mftiq_backend_mode_changed)
        mftiq_layout.addRow("Backend mode:", self.mftiq_backend_mode_combo)

        mftiq_model_row_widget = QWidget()
        mftiq_model_row_layout = QHBoxLayout(mftiq_model_row_widget)
        mftiq_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.mftiq_checkpoint_combo = QComboBox()
        self.mftiq_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mftiq_checkpoint_combo.setToolTip(
            "Select an MFTIQ checkpoint (.pth) from tracking_models/mftiq or repo/checkpoints."
        )
        self.mftiq_checkpoint_combo.currentIndexChanged.connect(self._on_mftiq_checkpoint_selected)
        self.mftiq_refresh_button = QPushButton("Refresh")
        self.mftiq_refresh_button.setToolTip(
            "Re-scan tracking_models/mftiq and re-run MFTIQ environment diagnostics."
        )
        self.mftiq_refresh_button.clicked.connect(self._refresh_mftiq_checkpoints)
        mftiq_model_row_layout.addWidget(self.mftiq_checkpoint_combo)
        mftiq_model_row_layout.addWidget(self.mftiq_refresh_button)
        mftiq_layout.addRow("Checkpoint:", mftiq_model_row_widget)

        self.mftiq_device_combo = QComboBox()
        self.mftiq_device_combo.addItem("Auto", "auto")
        self.mftiq_device_combo.addItem("GPU", "gpu")
        self.mftiq_device_combo.addItem("CPU", "cpu")
        mftiq_device_idx = self.mftiq_device_combo.findData(self.mftiq_settings.device)
        self.mftiq_device_combo.setCurrentIndex(mftiq_device_idx if mftiq_device_idx >= 0 else 0)
        self.mftiq_device_combo.setToolTip(
            "Device passed to the MFTIQ runtime. Upstream runtime currently expects CUDA."
        )
        mftiq_layout.addRow("Device:", self.mftiq_device_combo)

        self.mftiq_inference_resolution_combo = QComboBox()
        self.mftiq_inference_resolution_combo.addItem("Native", None)
        for label, value in (
            ("256 x 256", (256, 256)),
            ("384 x 512", (384, 512)),
            ("512 x 512", (512, 512)),
        ):
            self.mftiq_inference_resolution_combo.addItem(label, value)
        mftiq_res_idx = self.mftiq_inference_resolution_combo.findData(self.mftiq_settings.inference_resolution)
        self.mftiq_inference_resolution_combo.setCurrentIndex(mftiq_res_idx if mftiq_res_idx >= 0 else 0)
        self.mftiq_inference_resolution_combo.setToolTip(
            "Optional input resolution for MFTIQ. Native keeps the original LEED frame size."
        )
        mftiq_layout.addRow("Inference resolution:", self.mftiq_inference_resolution_combo)

        self.mftiq_occlusion_threshold_spinbox = QDoubleSpinBox()
        self.mftiq_occlusion_threshold_spinbox.setDecimals(2)
        self.mftiq_occlusion_threshold_spinbox.setRange(0.0, 10.0)
        self.mftiq_occlusion_threshold_spinbox.setSingleStep(0.05)
        self.mftiq_occlusion_threshold_spinbox.setValue(self.mftiq_settings.occlusion_threshold)
        self.mftiq_occlusion_threshold_spinbox.setToolTip(
            "Maximum accepted MFTIQ occlusion score. Higher values keep more uncertain points."
        )
        mftiq_layout.addRow("Occlusion threshold:", self.mftiq_occlusion_threshold_spinbox)

        self.mftiq_window_size_spinbox = QSpinBox()
        self.mftiq_window_size_spinbox.setRange(2, 4096)
        self.mftiq_window_size_spinbox.setValue(self.mftiq_settings.window_size)
        self.mftiq_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed MFTIQ window. Smaller values reduce VRAM."
        )
        mftiq_layout.addRow("Window size:", self.mftiq_window_size_spinbox)

        self.mftiq_window_overlap_spinbox = QSpinBox()
        self.mftiq_window_overlap_spinbox.setRange(0, 1024)
        self.mftiq_window_overlap_spinbox.setValue(self.mftiq_settings.window_overlap)
        self.mftiq_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive MFTIQ windows. "
            "Trackleed reseeds the next window from this overlap and enforces at least one shared frame when multiple windows are used."
        )
        mftiq_layout.addRow("Window overlap:", self.mftiq_window_overlap_spinbox)

        self.mftiq_keep_gaps_checkbox = QCheckBox("Keep invisible points as gaps")
        self.mftiq_keep_gaps_checkbox.setChecked(self.mftiq_settings.keep_invisible_points_as_gaps)
        self.mftiq_keep_gaps_checkbox.setToolTip(
            "If enabled, rejected or occluded MFTIQ predictions create gaps instead of forced points."
        )
        mftiq_layout.addRow(self.mftiq_keep_gaps_checkbox)

        self.mftiq_max_step_spinbox = QDoubleSpinBox()
        self.mftiq_max_step_spinbox.setDecimals(1)
        self.mftiq_max_step_spinbox.setRange(0.0, 500.0)
        self.mftiq_max_step_spinbox.setSingleStep(1.0)
        self.mftiq_max_step_spinbox.setValue(self.mftiq_settings.max_step_distance_px)
        self.mftiq_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        mftiq_layout.addRow("Max jump (px):", self.mftiq_max_step_spinbox)

        self.mftiq_max_reacquisition_gap_spinbox = QSpinBox()
        self.mftiq_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.mftiq_max_reacquisition_gap_spinbox.setValue(self.mftiq_settings.max_reacquisition_gap)
        self.mftiq_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible point is still accepted. Set 0 to disable."
        )
        mftiq_layout.addRow("Max reacquisition gap:", self.mftiq_max_reacquisition_gap_spinbox)

        self.mftiq_terminate_after_invisible_spinbox = QSpinBox()
        self.mftiq_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.mftiq_terminate_after_invisible_spinbox.setValue(
            self.mftiq_settings.terminate_after_invisible_frames
        )
        self.mftiq_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        mftiq_layout.addRow("Terminate after gap:", self.mftiq_terminate_after_invisible_spinbox)

        self.mftiq_center_mode_combo = QComboBox()
        self.mftiq_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.mftiq_center_mode_combo.addItem("Manual", "manual")
        mftiq_center_mode_idx = self.mftiq_center_mode_combo.findData(self.mftiq_settings.center_mode)
        self.mftiq_center_mode_combo.setCurrentIndex(mftiq_center_mode_idx if mftiq_center_mode_idx >= 0 else 0)
        self.mftiq_center_mode_combo.setToolTip(
            "Reference center used by directional gating. Auto estimates the LEED pattern center from detections."
        )
        self.mftiq_center_mode_combo.currentIndexChanged.connect(self._on_mftiq_center_mode_changed)
        mftiq_layout.addRow("Direction center:", self.mftiq_center_mode_combo)

        mftiq_center_row_widget = QWidget()
        mftiq_center_row_layout = QHBoxLayout(mftiq_center_row_widget)
        mftiq_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.mftiq_center_x_spinbox = QDoubleSpinBox()
        self.mftiq_center_x_spinbox.setRange(0.0, 10000.0)
        self.mftiq_center_x_spinbox.setDecimals(1)
        self.mftiq_center_x_spinbox.setValue(self.mftiq_settings.manual_center_x)
        self.mftiq_center_y_spinbox = QDoubleSpinBox()
        self.mftiq_center_y_spinbox.setRange(0.0, 10000.0)
        self.mftiq_center_y_spinbox.setDecimals(1)
        self.mftiq_center_y_spinbox.setValue(self.mftiq_settings.manual_center_y)
        mftiq_center_row_layout.addWidget(self.mftiq_center_x_spinbox)
        mftiq_center_row_layout.addWidget(QLabel(","))
        mftiq_center_row_layout.addWidget(self.mftiq_center_y_spinbox)
        mftiq_layout.addRow("Manual center (x, y):", mftiq_center_row_widget)

        self.mftiq_direction_mode_combo = QComboBox()
        self.mftiq_direction_mode_combo.addItem("None", "none")
        self.mftiq_direction_mode_combo.addItem("Auto", "auto")
        self.mftiq_direction_mode_combo.addItem("Inward", "inward")
        self.mftiq_direction_mode_combo.addItem("Outward", "outward")
        mftiq_direction_idx = self.mftiq_direction_mode_combo.findData(self.mftiq_settings.direction_mode)
        self.mftiq_direction_mode_combo.setCurrentIndex(mftiq_direction_idx if mftiq_direction_idx >= 0 else 0)
        self.mftiq_direction_mode_combo.setToolTip(
            "Optional directional prior for radial motion relative to the LEED center."
        )
        mftiq_layout.addRow("Direction mode:", self.mftiq_direction_mode_combo)

        self.mftiq_lateral_tol_spinbox = QDoubleSpinBox()
        self.mftiq_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.mftiq_lateral_tol_spinbox.setSingleStep(1.0)
        self.mftiq_lateral_tol_spinbox.setValue(self.mftiq_settings.lateral_angle_tolerance_deg)
        self.mftiq_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        mftiq_layout.addRow("Angular tol (deg):", self.mftiq_lateral_tol_spinbox)

        self.mftiq_max_radial_jump_spinbox = QDoubleSpinBox()
        self.mftiq_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.mftiq_max_radial_jump_spinbox.setSingleStep(1.0)
        self.mftiq_max_radial_jump_spinbox.setValue(self.mftiq_settings.max_radial_jump_px)
        self.mftiq_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        mftiq_layout.addRow("Max radial jump (px):", self.mftiq_max_radial_jump_spinbox)

        self.mftiq_recenter_mode_combo = QComboBox()
        self.mftiq_recenter_mode_combo.addItem("None", "none")
        self.mftiq_recenter_mode_combo.addItem("Peak", "peak")
        self.mftiq_recenter_mode_combo.addItem("Centroid", "centroid")
        self.mftiq_recenter_mode_combo.addItem("Gaussian", "gaussian")
        mftiq_recenter_mode_idx = self.mftiq_recenter_mode_combo.findData(self.mftiq_settings.recenter_mode)
        self.mftiq_recenter_mode_combo.setCurrentIndex(
            mftiq_recenter_mode_idx if mftiq_recenter_mode_idx >= 0 else 2
        )
        self.mftiq_recenter_mode_combo.setToolTip(
            "Local refinement applied after each MFTIQ prediction: none, brightest peak, centroid, or 2D Gaussian."
        )
        mftiq_layout.addRow("Recenter mode:", self.mftiq_recenter_mode_combo)

        self.mftiq_centroid_multiplier_spinbox = QDoubleSpinBox()
        self.mftiq_centroid_multiplier_spinbox.setDecimals(2)
        self.mftiq_centroid_multiplier_spinbox.setRange(0.50, 3.00)
        self.mftiq_centroid_multiplier_spinbox.setSingleStep(0.05)
        self.mftiq_centroid_multiplier_spinbox.setValue(self.mftiq_settings.recenter_patch_multiplier)
        self.mftiq_centroid_multiplier_spinbox.setToolTip(
            "Multiplier of ROI half-size used as the local refinement search radius."
        )
        mftiq_layout.addRow("Recenter radius x ROI:", self.mftiq_centroid_multiplier_spinbox)

        self.mftiq_centroid_rel_thr_spinbox = QDoubleSpinBox()
        self.mftiq_centroid_rel_thr_spinbox.setDecimals(2)
        self.mftiq_centroid_rel_thr_spinbox.setRange(0.05, 0.95)
        self.mftiq_centroid_rel_thr_spinbox.setSingleStep(0.05)
        self.mftiq_centroid_rel_thr_spinbox.setValue(self.mftiq_settings.centroid_rel_threshold)
        self.mftiq_centroid_rel_thr_spinbox.setToolTip(
            "Relative threshold used to build the centroid mask around the predicted point."
        )
        mftiq_layout.addRow("Recenter threshold:", self.mftiq_centroid_rel_thr_spinbox)

        self.mftiq_centroid_max_shift_spinbox = QDoubleSpinBox()
        self.mftiq_centroid_max_shift_spinbox.setDecimals(2)
        self.mftiq_centroid_max_shift_spinbox.setRange(0.10, 1.00)
        self.mftiq_centroid_max_shift_spinbox.setSingleStep(0.05)
        self.mftiq_centroid_max_shift_spinbox.setValue(self.mftiq_settings.centroid_max_shift_ratio)
        self.mftiq_centroid_max_shift_spinbox.setToolTip(
            "Maximum allowed centroid correction as a fraction of ROI size. Limits jumps to a nearby wrong spot."
        )
        mftiq_layout.addRow("Max recenter shift:", self.mftiq_centroid_max_shift_spinbox)

        self.mftiq_warning_label = QLabel()
        self.mftiq_warning_label.setStyleSheet("color: #d9534f;")
        self.mftiq_warning_label.setWordWrap(True)
        mftiq_layout.addRow(self.mftiq_warning_label)
        self._on_mftiq_backend_mode_changed(self.mftiq_backend_mode_combo.currentIndex())
        self._on_mftiq_center_mode_changed(self.mftiq_center_mode_combo.currentIndex())

        form_layout.addRow(self.mftiq_group)

        # --- Offline ToMP / TaMOs parameters ---
        self.tomp_group = QGroupBox("Offline ToMP / TaMOs")
        tomp_layout = QFormLayout(self.tomp_group)

        self.tomp_repo_path_edit = QLineEdit(self.tomp_settings.repo_path or "")
        self.tomp_repo_path_edit.setPlaceholderText("Path to local pytracking repo root")
        self.tomp_repo_path_edit.setToolTip(
            "Local clone of the PyTracking repository. Must contain pytracking/evaluation/tracker.py "
            "and parameter files for both ToMP and TaMOs."
        )
        self.tomp_repo_path_edit.editingFinished.connect(self._refresh_tomp_checkpoints)
        tomp_layout.addRow("Repo path:", self.tomp_repo_path_edit)

        self.tomp_python_executable_edit = QLineEdit(self.tomp_settings.python_executable or "")
        self.tomp_python_executable_edit.setPlaceholderText(
            "Path to the dedicated tomp env Python executable"
        )
        self.tomp_python_executable_edit.setToolTip(
            "Used only in subprocess mode. Point this to the interpreter of the dedicated "
            "ToMP / TaMOs environment."
        )
        self.tomp_python_executable_edit.editingFinished.connect(self._refresh_tomp_checkpoints)
        tomp_layout.addRow("Python executable:", self.tomp_python_executable_edit)

        self.tomp_backend_mode_combo = QComboBox()
        self.tomp_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.tomp_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        tomp_backend_idx = self.tomp_backend_mode_combo.findData(self.tomp_settings.backend_mode)
        self.tomp_backend_mode_combo.setCurrentIndex(tomp_backend_idx if tomp_backend_idx >= 0 else 0)
        self.tomp_backend_mode_combo.setToolTip(
            "Subprocess mode is the recommended production path when Trackleed runs outside the "
            "dedicated PyTracking env. Direct import requires the current env to be ToMP/TaMOs-ready."
        )
        self.tomp_backend_mode_combo.currentIndexChanged.connect(self._on_tomp_backend_mode_changed)
        tomp_layout.addRow("Backend mode:", self.tomp_backend_mode_combo)

        self.tomp_model_variant_combo = QComboBox()
        self.tomp_model_variant_combo.addItem("ToMP-50", "tomp50")
        self.tomp_model_variant_combo.addItem("ToMP-101", "tomp101")
        self.tomp_model_variant_combo.addItem("TaMOs-resnet50", "tamos_resnet50")
        self.tomp_model_variant_combo.addItem("TaMOs-swin-base", "tamos_swin_base")
        tomp_variant_idx = self.tomp_model_variant_combo.findData(self.tomp_settings.model_variant)
        self.tomp_model_variant_combo.setCurrentIndex(tomp_variant_idx if tomp_variant_idx >= 0 else 0)
        self.tomp_model_variant_combo.setToolTip(
            "Choose which PyTracking bbox tracker Trackleed should run."
        )
        self.tomp_model_variant_combo.currentIndexChanged.connect(self._on_tomp_variant_changed)
        tomp_layout.addRow("Model variant:", self.tomp_model_variant_combo)

        tomp_model_row_widget = QWidget()
        tomp_model_row_layout = QHBoxLayout(tomp_model_row_widget)
        tomp_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.tomp_checkpoint_combo = QComboBox()
        self.tomp_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tomp_checkpoint_combo.setToolTip(
            "Select a checkpoint for the currently selected ToMP / TaMOs variant."
        )
        self.tomp_checkpoint_combo.currentIndexChanged.connect(self._on_tomp_checkpoint_selected)
        self.tomp_refresh_button = QPushButton("Refresh")
        self.tomp_refresh_button.setToolTip(
            "Re-scan tracking_models/tomp, tracking_models/tamos and re-run PyTracking environment diagnostics."
        )
        self.tomp_refresh_button.clicked.connect(self._refresh_tomp_checkpoints)
        tomp_model_row_layout.addWidget(self.tomp_checkpoint_combo)
        tomp_model_row_layout.addWidget(self.tomp_refresh_button)
        tomp_layout.addRow("Checkpoint:", tomp_model_row_widget)

        self.tomp_checkpoint_hint_label = QLabel()
        self.tomp_checkpoint_hint_label.setWordWrap(True)
        self.tomp_checkpoint_hint_label.setStyleSheet("color: #6c757d;")
        tomp_layout.addRow(self.tomp_checkpoint_hint_label)

        self.tomp_device_combo = QComboBox()
        self.tomp_device_combo.addItem("Auto", "auto")
        self.tomp_device_combo.addItem("GPU", "cuda")
        self.tomp_device_combo.addItem("CPU", "cpu")
        tomp_device_idx = self.tomp_device_combo.findData(self.tomp_settings.device)
        self.tomp_device_combo.setCurrentIndex(tomp_device_idx if tomp_device_idx >= 0 else 0)
        self.tomp_device_combo.setToolTip("Device passed to the ToMP / TaMOs runtime.")
        tomp_layout.addRow("Device:", self.tomp_device_combo)

        self.tomp_inference_resolution_combo = QComboBox()
        self.tomp_inference_resolution_combo.addItem("Native", None)
        for label, value in (
            ("256 x 256", 256),
            ("384 x 384", 384),
            ("512 x 512", 512),
        ):
            self.tomp_inference_resolution_combo.addItem(label, value)
        tomp_res_idx = self.tomp_inference_resolution_combo.findData(self.tomp_settings.inference_resolution)
        self.tomp_inference_resolution_combo.setCurrentIndex(tomp_res_idx if tomp_res_idx >= 0 else 0)
        self.tomp_inference_resolution_combo.setToolTip(
            "Optional square inference size for ToMP / TaMOs. Native keeps the original LEED frame size."
        )
        tomp_layout.addRow("Inference resolution:", self.tomp_inference_resolution_combo)

        self.tomp_window_size_spinbox = QSpinBox()
        self.tomp_window_size_spinbox.setRange(2, 4096)
        self.tomp_window_size_spinbox.setValue(self.tomp_settings.window_size)
        self.tomp_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed ToMP/TaMOs window. "
            "Smaller values reduce temporal context and reseed more often."
        )
        tomp_layout.addRow("Window size:", self.tomp_window_size_spinbox)

        self.tomp_window_overlap_spinbox = QSpinBox()
        self.tomp_window_overlap_spinbox.setRange(0, 1024)
        self.tomp_window_overlap_spinbox.setValue(self.tomp_settings.window_overlap)
        self.tomp_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive ToMP/TaMOs windows. "
            "In multi-window mode the backend clamps this to at least 1 shared frame."
        )
        tomp_layout.addRow("Window overlap:", self.tomp_window_overlap_spinbox)

        self.tomp_repr_mode_combo = QComboBox()
        self.tomp_repr_mode_combo.addItem("BBox center", "bbox_center")
        self.tomp_repr_mode_combo.addItem("Peak in bbox", "peak_in_bbox")
        self.tomp_repr_mode_combo.addItem("Gaussian in bbox", "gaussian_in_bbox")
        tomp_repr_idx = self.tomp_repr_mode_combo.findData(self.tomp_settings.representative_point_mode)
        self.tomp_repr_mode_combo.setCurrentIndex(tomp_repr_idx if tomp_repr_idx >= 0 else 0)
        self.tomp_repr_mode_combo.setToolTip(
            "How Trackleed converts the tracked bbox into one representative point."
        )
        tomp_layout.addRow("Representative point:", self.tomp_repr_mode_combo)

        self.tomp_bbox_padding_spinbox = QDoubleSpinBox()
        self.tomp_bbox_padding_spinbox.setDecimals(1)
        self.tomp_bbox_padding_spinbox.setRange(0.0, 256.0)
        self.tomp_bbox_padding_spinbox.setSingleStep(1.0)
        self.tomp_bbox_padding_spinbox.setValue(self.tomp_settings.initial_bbox_padding_px)
        self.tomp_bbox_padding_spinbox.setToolTip(
            "Extra padding in pixels added around roi_rect before the initial bbox is passed to ToMP / TaMOs."
        )
        tomp_layout.addRow("Init bbox padding (px):", self.tomp_bbox_padding_spinbox)

        self.tomp_keep_gaps_checkbox = QCheckBox("Keep invisible bboxes as gaps")
        self.tomp_keep_gaps_checkbox.setChecked(self.tomp_settings.keep_invisible_points_as_gaps)
        self.tomp_keep_gaps_checkbox.setToolTip(
            "If enabled, frames where ToMP / TaMOs return no accepted bbox leave gaps instead of forcing a point."
        )
        tomp_layout.addRow(self.tomp_keep_gaps_checkbox)

        self.tomp_max_step_spinbox = QDoubleSpinBox()
        self.tomp_max_step_spinbox.setDecimals(1)
        self.tomp_max_step_spinbox.setRange(0.0, 500.0)
        self.tomp_max_step_spinbox.setSingleStep(1.0)
        self.tomp_max_step_spinbox.setValue(self.tomp_settings.max_step_distance_px)
        self.tomp_max_step_spinbox.setToolTip(
            "Maximum accepted shift between consecutive accepted positions. Set 0 to disable."
        )
        tomp_layout.addRow("Max jump (px):", self.tomp_max_step_spinbox)

        self.tomp_max_reacquisition_gap_spinbox = QSpinBox()
        self.tomp_max_reacquisition_gap_spinbox.setRange(0, 1000)
        self.tomp_max_reacquisition_gap_spinbox.setValue(self.tomp_settings.max_reacquisition_gap)
        self.tomp_max_reacquisition_gap_spinbox.setToolTip(
            "Maximum invisible gap after which a new visible bbox is still accepted. Set 0 to disable."
        )
        tomp_layout.addRow("Max reacquisition gap:", self.tomp_max_reacquisition_gap_spinbox)

        self.tomp_terminate_after_invisible_spinbox = QSpinBox()
        self.tomp_terminate_after_invisible_spinbox.setRange(0, 1000)
        self.tomp_terminate_after_invisible_spinbox.setValue(
            self.tomp_settings.terminate_after_invisible_frames
        )
        self.tomp_terminate_after_invisible_spinbox.setToolTip(
            "Terminate a track after this many consecutive rejected/invisible frames. Set 0 to disable."
        )
        tomp_layout.addRow("Terminate after gap:", self.tomp_terminate_after_invisible_spinbox)

        self.tomp_center_mode_combo = QComboBox()
        self.tomp_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.tomp_center_mode_combo.addItem("Manual", "manual")
        tomp_center_mode_idx = self.tomp_center_mode_combo.findData(self.tomp_settings.center_mode)
        self.tomp_center_mode_combo.setCurrentIndex(
            tomp_center_mode_idx if tomp_center_mode_idx >= 0 else 0
        )
        self.tomp_center_mode_combo.setToolTip(
            "Reference center used by directional gating. Auto estimates the LEED pattern center from detections."
        )
        self.tomp_center_mode_combo.currentIndexChanged.connect(self._on_tomp_center_mode_changed)
        tomp_layout.addRow("Direction center:", self.tomp_center_mode_combo)

        tomp_center_row_widget = QWidget()
        tomp_center_row_layout = QHBoxLayout(tomp_center_row_widget)
        tomp_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.tomp_center_x_spinbox = QDoubleSpinBox()
        self.tomp_center_x_spinbox.setRange(0.0, 10000.0)
        self.tomp_center_x_spinbox.setDecimals(1)
        self.tomp_center_x_spinbox.setValue(self.tomp_settings.manual_center_x)
        self.tomp_center_y_spinbox = QDoubleSpinBox()
        self.tomp_center_y_spinbox.setRange(0.0, 10000.0)
        self.tomp_center_y_spinbox.setDecimals(1)
        self.tomp_center_y_spinbox.setValue(self.tomp_settings.manual_center_y)
        tomp_center_row_layout.addWidget(self.tomp_center_x_spinbox)
        tomp_center_row_layout.addWidget(QLabel(","))
        tomp_center_row_layout.addWidget(self.tomp_center_y_spinbox)
        tomp_layout.addRow("Manual center (x, y):", tomp_center_row_widget)

        self.tomp_direction_mode_combo = QComboBox()
        self.tomp_direction_mode_combo.addItem("None", "none")
        self.tomp_direction_mode_combo.addItem("Auto", "auto")
        self.tomp_direction_mode_combo.addItem("Inward", "inward")
        self.tomp_direction_mode_combo.addItem("Outward", "outward")
        tomp_direction_idx = self.tomp_direction_mode_combo.findData(self.tomp_settings.direction_mode)
        self.tomp_direction_mode_combo.setCurrentIndex(
            tomp_direction_idx if tomp_direction_idx >= 0 else 0
        )
        self.tomp_direction_mode_combo.setToolTip(
            "Optional directional prior for radial motion relative to the LEED center."
        )
        tomp_layout.addRow("Direction mode:", self.tomp_direction_mode_combo)

        self.tomp_lateral_tol_spinbox = QDoubleSpinBox()
        self.tomp_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.tomp_lateral_tol_spinbox.setSingleStep(1.0)
        self.tomp_lateral_tol_spinbox.setValue(self.tomp_settings.lateral_angle_tolerance_deg)
        self.tomp_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        tomp_layout.addRow("Angular tol (deg):", self.tomp_lateral_tol_spinbox)

        self.tomp_max_radial_jump_spinbox = QDoubleSpinBox()
        self.tomp_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.tomp_max_radial_jump_spinbox.setSingleStep(1.0)
        self.tomp_max_radial_jump_spinbox.setValue(self.tomp_settings.max_radial_jump_px)
        self.tomp_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        tomp_layout.addRow("Max radial jump (px):", self.tomp_max_radial_jump_spinbox)

        self.tomp_warning_label = QLabel()
        self.tomp_warning_label.setStyleSheet("color: #d9534f;")
        self.tomp_warning_label.setWordWrap(True)
        tomp_layout.addRow(self.tomp_warning_label)

        self._on_tomp_backend_mode_changed(self.tomp_backend_mode_combo.currentIndex())
        self._on_tomp_center_mode_changed(self.tomp_center_mode_combo.currentIndex())
        form_layout.addRow(self.tomp_group)

        # --- Offline Cutie parameters ---
        self.cutie_group = QGroupBox("Offline Cutie")
        cutie_layout = QFormLayout(self.cutie_group)

        self.cutie_repo_path_edit = QLineEdit(self.cutie_settings.repo_path or "")
        self.cutie_repo_path_edit.setPlaceholderText("Path to local cutie repo root")
        self.cutie_repo_path_edit.setToolTip(
            "Local clone of the Cutie repository. Must contain scripting_demo.py and cutie/inference/."
        )
        self.cutie_repo_path_edit.editingFinished.connect(self._refresh_cutie_checkpoints)
        cutie_layout.addRow("Repo path:", self.cutie_repo_path_edit)

        self.cutie_python_executable_edit = QLineEdit(self.cutie_settings.python_executable or "")
        self.cutie_python_executable_edit.setPlaceholderText(
            "Used when Backend mode is set to Subprocess"
        )
        self.cutie_python_executable_edit.setToolTip(
            "Used when Backend mode is set to Subprocess. Point it at the interpreter inside the dedicated cutie env."
        )
        self.cutie_python_executable_edit.editingFinished.connect(self._refresh_cutie_checkpoints)
        cutie_layout.addRow("Python executable:", self.cutie_python_executable_edit)

        self.cutie_backend_mode_combo = QComboBox()
        self.cutie_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.cutie_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        cutie_backend_mode_idx = self.cutie_backend_mode_combo.findData(self.cutie_settings.backend_mode)
        self.cutie_backend_mode_combo.setCurrentIndex(cutie_backend_mode_idx if cutie_backend_mode_idx >= 0 else 0)
        self.cutie_backend_mode_combo.setToolTip(
            "Subprocess lets Trackleed stay in the main env while Cutie runs in its own dedicated env."
        )
        self.cutie_backend_mode_combo.currentIndexChanged.connect(self._on_cutie_backend_mode_changed)
        cutie_layout.addRow("Backend mode:", self.cutie_backend_mode_combo)

        cutie_model_row_widget = QWidget()
        cutie_model_row_layout = QHBoxLayout(cutie_model_row_widget)
        cutie_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.cutie_checkpoint_combo = QComboBox()
        self.cutie_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cutie_checkpoint_combo.setToolTip(
            "Select a Cutie checkpoint from tracking_models/cutie or repo/weights."
        )
        self.cutie_checkpoint_combo.currentIndexChanged.connect(self._on_cutie_checkpoint_selected)
        self.cutie_refresh_button = QPushButton("Refresh")
        self.cutie_refresh_button.setToolTip(
            "Re-scan tracking_models/cutie and re-run Cutie environment diagnostics."
        )
        self.cutie_refresh_button.clicked.connect(self._refresh_cutie_checkpoints)
        cutie_model_row_layout.addWidget(self.cutie_checkpoint_combo)
        cutie_model_row_layout.addWidget(self.cutie_refresh_button)
        cutie_layout.addRow("Checkpoint:", cutie_model_row_widget)

        self.cutie_checkpoint_hint_label = QLabel(
            "Hint: use cutie-base-mega.pth for Trackleed. "
            "coco_lvis_h18_itermask.pth is the optional interactive checkpoint from the upstream repo."
        )
        self.cutie_checkpoint_hint_label.setWordWrap(True)
        self.cutie_checkpoint_hint_label.setStyleSheet("color: #6c757d;")
        cutie_layout.addRow(self.cutie_checkpoint_hint_label)

        self.cutie_device_combo = QComboBox()
        self.cutie_device_combo.addItem("Auto", "auto")
        self.cutie_device_combo.addItem("GPU", "cuda")
        self.cutie_device_combo.addItem("CPU", "cpu")
        cutie_device_idx = self.cutie_device_combo.findData(self.cutie_settings.device)
        self.cutie_device_combo.setCurrentIndex(cutie_device_idx if cutie_device_idx >= 0 else 0)
        self.cutie_device_combo.setToolTip("Device passed to the Cutie runtime.")
        cutie_layout.addRow("Device:", self.cutie_device_combo)

        self.cutie_inference_resolution_combo = QComboBox()
        self.cutie_inference_resolution_combo.addItem("Native", None)
        for label, value in (
            ("256 x 256", 256),
            ("384 x 384", 384),
            ("512 x 512", 512),
        ):
            self.cutie_inference_resolution_combo.addItem(label, value)
        cutie_res_idx = self.cutie_inference_resolution_combo.findData(self.cutie_settings.inference_resolution)
        self.cutie_inference_resolution_combo.setCurrentIndex(cutie_res_idx if cutie_res_idx >= 0 else 0)
        self.cutie_inference_resolution_combo.setToolTip(
            "Optional square inference size for Cutie. Native keeps the original LEED frame size."
        )
        cutie_layout.addRow("Inference resolution:", self.cutie_inference_resolution_combo)

        self.cutie_window_size_spinbox = QSpinBox()
        self.cutie_window_size_spinbox.setRange(2, 4096)
        self.cutie_window_size_spinbox.setValue(self.cutie_settings.window_size)
        self.cutie_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed Cutie window. Smaller values reduce VRAM pressure."
        )
        cutie_layout.addRow("Window size:", self.cutie_window_size_spinbox)

        self.cutie_window_overlap_spinbox = QSpinBox()
        self.cutie_window_overlap_spinbox.setRange(0, 1024)
        self.cutie_window_overlap_spinbox.setValue(self.cutie_settings.window_overlap)
        self.cutie_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive Cutie windows. "
            "Trackleed reseeds the next window from this overlap and enforces at least one shared frame when multiple windows are used."
        )
        cutie_layout.addRow("Window overlap:", self.cutie_window_overlap_spinbox)

        self.cutie_repr_mode_combo = QComboBox()
        self.cutie_repr_mode_combo.addItem("Mask centroid", "mask_centroid")
        self.cutie_repr_mode_combo.addItem("Peak in mask", "peak_in_mask")
        self.cutie_repr_mode_combo.addItem("BBox center", "bbox_center")
        self.cutie_repr_mode_combo.addItem("Gaussian in mask", "gaussian_in_mask")
        cutie_repr_idx = self.cutie_repr_mode_combo.findData(self.cutie_settings.representative_point_mode)
        self.cutie_repr_mode_combo.setCurrentIndex(cutie_repr_idx if cutie_repr_idx >= 0 else 0)
        self.cutie_repr_mode_combo.setToolTip(
            "How Trackleed converts the propagated Cutie mask into one representative point."
        )
        cutie_layout.addRow("Representative point:", self.cutie_repr_mode_combo)

        self.cutie_bbox_padding_spinbox = QDoubleSpinBox()
        self.cutie_bbox_padding_spinbox.setDecimals(1)
        self.cutie_bbox_padding_spinbox.setRange(0.0, 256.0)
        self.cutie_bbox_padding_spinbox.setSingleStep(1.0)
        self.cutie_bbox_padding_spinbox.setValue(self.cutie_settings.initial_bbox_padding_px)
        self.cutie_bbox_padding_spinbox.setToolTip(
            "Extra padding in pixels added around roi_rect before building the initial Cutie mask."
        )
        cutie_layout.addRow("Init bbox padding (px):", self.cutie_bbox_padding_spinbox)

        self.cutie_initial_mask_mode_combo = QComboBox()
        self.cutie_initial_mask_mode_combo.addItem("Threshold in bbox", "threshold")
        self.cutie_initial_mask_mode_combo.addItem("Full bbox mask", "full_bbox")
        self.cutie_initial_mask_mode_combo.addItem("Disk in bbox", "disk")
        cutie_mask_mode_idx = self.cutie_initial_mask_mode_combo.findData(self.cutie_settings.initial_mask_mode)
        self.cutie_initial_mask_mode_combo.setCurrentIndex(cutie_mask_mode_idx if cutie_mask_mode_idx >= 0 else 0)
        self.cutie_initial_mask_mode_combo.setToolTip(
            "How Trackleed turns the ROI bbox into the initial seed mask for Cutie."
        )
        cutie_layout.addRow("Initial mask mode:", self.cutie_initial_mask_mode_combo)

        self.cutie_initial_mask_threshold_spinbox = QDoubleSpinBox()
        self.cutie_initial_mask_threshold_spinbox.setDecimals(2)
        self.cutie_initial_mask_threshold_spinbox.setRange(0.00, 1.00)
        self.cutie_initial_mask_threshold_spinbox.setSingleStep(0.05)
        self.cutie_initial_mask_threshold_spinbox.setValue(self.cutie_settings.initial_mask_threshold_rel)
        self.cutie_initial_mask_threshold_spinbox.setToolTip(
            "Relative threshold used by the default bbox->mask builder. Higher values keep only the brightest core."
        )
        cutie_layout.addRow("Threshold rel.:", self.cutie_initial_mask_threshold_spinbox)

        self.cutie_initial_mask_disk_spinbox = QDoubleSpinBox()
        self.cutie_initial_mask_disk_spinbox.setDecimals(2)
        self.cutie_initial_mask_disk_spinbox.setRange(0.05, 1.00)
        self.cutie_initial_mask_disk_spinbox.setSingleStep(0.05)
        self.cutie_initial_mask_disk_spinbox.setValue(self.cutie_settings.initial_mask_disk_radius_ratio)
        self.cutie_initial_mask_disk_spinbox.setToolTip(
            "Disk radius as a fraction of the smaller ROI dimension when the disk seed mode is used."
        )
        cutie_layout.addRow("Disk radius ratio:", self.cutie_initial_mask_disk_spinbox)

        self.cutie_mem_every_spinbox = QSpinBox()
        self.cutie_mem_every_spinbox.setRange(0, 512)
        self.cutie_mem_every_spinbox.setValue(
            0 if self.cutie_settings.mem_every in {None, 0} else int(self.cutie_settings.mem_every)
        )
        self.cutie_mem_every_spinbox.setToolTip(
            "How often Cutie stores memory frames. Set 0 to keep the upstream default."
        )
        cutie_layout.addRow("Memory every N frames:", self.cutie_mem_every_spinbox)

        self.cutie_max_step_spinbox = QDoubleSpinBox()
        self.cutie_max_step_spinbox.setDecimals(1)
        self.cutie_max_step_spinbox.setRange(0.0, 10000.0)
        self.cutie_max_step_spinbox.setSingleStep(1.0)
        self.cutie_max_step_spinbox.setValue(self.cutie_settings.max_step_distance_px)
        self.cutie_max_step_spinbox.setToolTip(
            "Reject representative points that jump farther than this distance from the last accepted Cutie point. Set 0 to disable."
        )
        cutie_layout.addRow("Max jump (px):", self.cutie_max_step_spinbox)

        self.cutie_max_gap_spinbox = QSpinBox()
        self.cutie_max_gap_spinbox.setRange(0, 4096)
        self.cutie_max_gap_spinbox.setValue(self.cutie_settings.max_reacquisition_gap)
        self.cutie_max_gap_spinbox.setToolTip(
            "If a Cutie track stays invalid longer than this gap, later reacquisition attempts are discarded. Set 0 to disable."
        )
        cutie_layout.addRow("Max reacquisition gap:", self.cutie_max_gap_spinbox)

        self.cutie_terminate_gap_spinbox = QSpinBox()
        self.cutie_terminate_gap_spinbox.setRange(0, 4096)
        self.cutie_terminate_gap_spinbox.setValue(self.cutie_settings.terminate_after_invisible_frames)
        self.cutie_terminate_gap_spinbox.setToolTip(
            "Stop writing any more Cutie points after this many consecutive invalid frames. Set 0 to disable."
        )
        cutie_layout.addRow("Terminate after gap:", self.cutie_terminate_gap_spinbox)

        self.cutie_max_area_ratio_spinbox = QDoubleSpinBox()
        self.cutie_max_area_ratio_spinbox.setDecimals(2)
        self.cutie_max_area_ratio_spinbox.setRange(0.0, 100.0)
        self.cutie_max_area_ratio_spinbox.setSingleStep(0.05)
        self.cutie_max_area_ratio_spinbox.setValue(self.cutie_settings.max_mask_area_ratio_change)
        self.cutie_max_area_ratio_spinbox.setToolTip(
            "Maximum allowed multiplicative change of mask area between accepted frames. Set 0 to disable."
        )
        cutie_layout.addRow("Max area drift x:", self.cutie_max_area_ratio_spinbox)

        self.cutie_max_aspect_ratio_spinbox = QDoubleSpinBox()
        self.cutie_max_aspect_ratio_spinbox.setDecimals(2)
        self.cutie_max_aspect_ratio_spinbox.setRange(0.0, 100.0)
        self.cutie_max_aspect_ratio_spinbox.setSingleStep(0.05)
        self.cutie_max_aspect_ratio_spinbox.setValue(self.cutie_settings.max_bbox_aspect_ratio_change)
        self.cutie_max_aspect_ratio_spinbox.setToolTip(
            "Maximum allowed multiplicative change of mask bbox aspect ratio between accepted frames. Set 0 to disable."
        )
        cutie_layout.addRow("Max aspect drift x:", self.cutie_max_aspect_ratio_spinbox)

        self.cutie_bbox_jump_spinbox = QDoubleSpinBox()
        self.cutie_bbox_jump_spinbox.setDecimals(1)
        self.cutie_bbox_jump_spinbox.setRange(0.0, 10000.0)
        self.cutie_bbox_jump_spinbox.setSingleStep(1.0)
        self.cutie_bbox_jump_spinbox.setValue(self.cutie_settings.max_bbox_center_jump_px)
        self.cutie_bbox_jump_spinbox.setToolTip(
            "Maximum accepted shift of the propagated mask bbox center between accepted frames. Set 0 to disable."
        )
        cutie_layout.addRow("Max bbox jump (px):", self.cutie_bbox_jump_spinbox)

        self.cutie_keep_gaps_checkbox = QCheckBox("Keep invisible masks as gaps")
        self.cutie_keep_gaps_checkbox.setChecked(self.cutie_settings.keep_invisible_points_as_gaps)
        self.cutie_keep_gaps_checkbox.setToolTip(
            "If enabled, missing/invalid Cutie masks leave gaps instead of forcing a representative point."
        )
        cutie_layout.addRow(self.cutie_keep_gaps_checkbox)

        self.cutie_center_mode_combo = QComboBox()
        self.cutie_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.cutie_center_mode_combo.addItem("Manual", "manual")
        cutie_center_mode_idx = self.cutie_center_mode_combo.findData(self.cutie_settings.center_mode)
        self.cutie_center_mode_combo.setCurrentIndex(cutie_center_mode_idx if cutie_center_mode_idx >= 0 else 0)
        self.cutie_center_mode_combo.setToolTip(
            "Reference center used by Cutie direction gating. Auto estimates the LEED pattern center from detections."
        )
        self.cutie_center_mode_combo.currentIndexChanged.connect(self._on_cutie_center_mode_changed)
        cutie_layout.addRow("Direction center:", self.cutie_center_mode_combo)

        cutie_center_row_widget = QWidget()
        cutie_center_row_layout = QHBoxLayout(cutie_center_row_widget)
        cutie_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.cutie_center_x_spinbox = QDoubleSpinBox()
        self.cutie_center_x_spinbox.setRange(0.0, 10000.0)
        self.cutie_center_x_spinbox.setDecimals(1)
        self.cutie_center_x_spinbox.setValue(self.cutie_settings.manual_center_x)
        self.cutie_center_y_spinbox = QDoubleSpinBox()
        self.cutie_center_y_spinbox.setRange(0.0, 10000.0)
        self.cutie_center_y_spinbox.setDecimals(1)
        self.cutie_center_y_spinbox.setValue(self.cutie_settings.manual_center_y)
        cutie_center_row_layout.addWidget(self.cutie_center_x_spinbox)
        cutie_center_row_layout.addWidget(QLabel(","))
        cutie_center_row_layout.addWidget(self.cutie_center_y_spinbox)
        cutie_layout.addRow("Manual center (x, y):", cutie_center_row_widget)

        self.cutie_direction_mode_combo = QComboBox()
        self.cutie_direction_mode_combo.addItem("None", "none")
        self.cutie_direction_mode_combo.addItem("Auto", "auto")
        self.cutie_direction_mode_combo.addItem("Inward", "inward")
        self.cutie_direction_mode_combo.addItem("Outward", "outward")
        cutie_direction_idx = self.cutie_direction_mode_combo.findData(self.cutie_settings.direction_mode)
        self.cutie_direction_mode_combo.setCurrentIndex(cutie_direction_idx if cutie_direction_idx >= 0 else 0)
        self.cutie_direction_mode_combo.setToolTip(
            "Optional directional prior applied to the Cutie representative point."
        )
        cutie_layout.addRow("Direction mode:", self.cutie_direction_mode_combo)

        self.cutie_lateral_tol_spinbox = QDoubleSpinBox()
        self.cutie_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.cutie_lateral_tol_spinbox.setSingleStep(1.0)
        self.cutie_lateral_tol_spinbox.setValue(self.cutie_settings.lateral_angle_tolerance_deg)
        self.cutie_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        cutie_layout.addRow("Angular tol (deg):", self.cutie_lateral_tol_spinbox)

        self.cutie_max_radial_jump_spinbox = QDoubleSpinBox()
        self.cutie_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.cutie_max_radial_jump_spinbox.setSingleStep(1.0)
        self.cutie_max_radial_jump_spinbox.setValue(self.cutie_settings.max_radial_jump_px)
        self.cutie_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        cutie_layout.addRow("Max radial jump (px):", self.cutie_max_radial_jump_spinbox)

        self.cutie_warning_label = QLabel()
        self.cutie_warning_label.setStyleSheet("color: #d9534f;")
        self.cutie_warning_label.setWordWrap(True)
        cutie_layout.addRow(self.cutie_warning_label)

        self._on_cutie_backend_mode_changed(self.cutie_backend_mode_combo.currentIndex())
        self._on_cutie_center_mode_changed(self.cutie_center_mode_combo.currentIndex())
        form_layout.addRow(self.cutie_group)

        # --- Offline XMem2 parameters ---
        self.xmem2_group = QGroupBox("Offline XMem2")
        xmem2_layout = QFormLayout(self.xmem2_group)

        self.xmem2_repo_path_edit = QLineEdit(self.xmem2_settings.repo_path or "")
        self.xmem2_repo_path_edit.setPlaceholderText("Path to local xmem2 repo root")
        self.xmem2_repo_path_edit.setToolTip(
            "Local clone of the XMem2 / XMem++ repository. Must contain process_video.py and inference/run_on_video.py."
        )
        self.xmem2_repo_path_edit.editingFinished.connect(self._refresh_xmem2_checkpoints)
        xmem2_layout.addRow("Repo path:", self.xmem2_repo_path_edit)

        self.xmem2_python_executable_edit = QLineEdit(self.xmem2_settings.python_executable or "")
        self.xmem2_python_executable_edit.setPlaceholderText(
            "Path to python.exe inside the dedicated xmem2 env"
        )
        self.xmem2_python_executable_edit.setToolTip(
            "Used when Backend mode is set to Subprocess. Point it at the interpreter inside the dedicated xmem2 env."
        )
        self.xmem2_python_executable_edit.editingFinished.connect(self._refresh_xmem2_checkpoints)
        xmem2_layout.addRow("Python executable:", self.xmem2_python_executable_edit)

        self.xmem2_backend_mode_combo = QComboBox()
        self.xmem2_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.xmem2_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        xmem2_backend_mode_idx = self.xmem2_backend_mode_combo.findData(self.xmem2_settings.backend_mode)
        self.xmem2_backend_mode_combo.setCurrentIndex(
            xmem2_backend_mode_idx if xmem2_backend_mode_idx >= 0 else 0
        )
        self.xmem2_backend_mode_combo.setToolTip(
            "Use subprocess to run XMem2 from a dedicated env while Trackleed stays in the main app env."
        )
        self.xmem2_backend_mode_combo.currentIndexChanged.connect(self._on_xmem2_backend_mode_changed)
        xmem2_layout.addRow("Backend mode:", self.xmem2_backend_mode_combo)

        xmem2_model_row_widget = QWidget()
        xmem2_model_row_layout = QHBoxLayout(xmem2_model_row_widget)
        xmem2_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.xmem2_checkpoint_combo = QComboBox()
        self.xmem2_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.xmem2_checkpoint_combo.setToolTip(
            "Select an XMem checkpoint from tracking_models/xmem2 or repo/saves."
        )
        self.xmem2_checkpoint_combo.currentIndexChanged.connect(self._on_xmem2_checkpoint_selected)
        self.xmem2_refresh_button = QPushButton("Refresh")
        self.xmem2_refresh_button.setToolTip(
            "Re-scan tracking_models/xmem2 and re-run XMem2 environment diagnostics."
        )
        self.xmem2_refresh_button.clicked.connect(self._refresh_xmem2_checkpoints)
        xmem2_model_row_layout.addWidget(self.xmem2_checkpoint_combo)
        xmem2_model_row_layout.addWidget(self.xmem2_refresh_button)
        xmem2_layout.addRow("Checkpoint:", xmem2_model_row_widget)

        self.xmem2_checkpoint_hint_label = QLabel(
            "Hint: use XMem.pth as the main checkpoint. XMem-s012.pth is an alternate upstream variant. "
            "For permanent memory, open the visualizer, right-click a point and mark it as an XMem2 permanent reference before re-tracking."
        )
        self.xmem2_checkpoint_hint_label.setWordWrap(True)
        self.xmem2_checkpoint_hint_label.setStyleSheet("color: #6c757d;")
        xmem2_layout.addRow(self.xmem2_checkpoint_hint_label)

        self.xmem2_device_combo = QComboBox()
        self.xmem2_device_combo.addItem("Auto", "auto")
        self.xmem2_device_combo.addItem("GPU", "cuda")
        xmem2_device_idx = self.xmem2_device_combo.findData(self.xmem2_settings.device)
        self.xmem2_device_combo.setCurrentIndex(xmem2_device_idx if xmem2_device_idx >= 0 else 0)
        self.xmem2_device_combo.setToolTip("Device passed to the XMem2 runtime.")
        xmem2_layout.addRow("Device:", self.xmem2_device_combo)

        self.xmem2_inference_resolution_combo = QComboBox()
        self.xmem2_inference_resolution_combo.addItem("Native", None)
        for label, value in (
            ("256 x 256", 256),
            ("384 x 384", 384),
            ("512 x 512", 512),
        ):
            self.xmem2_inference_resolution_combo.addItem(label, value)
        xmem2_res_idx = self.xmem2_inference_resolution_combo.findData(self.xmem2_settings.inference_resolution)
        self.xmem2_inference_resolution_combo.setCurrentIndex(xmem2_res_idx if xmem2_res_idx >= 0 else 0)
        self.xmem2_inference_resolution_combo.setToolTip(
            "Optional square inference size for XMem2. Native keeps the original LEED frame size."
        )
        xmem2_layout.addRow("Inference resolution:", self.xmem2_inference_resolution_combo)

        self.xmem2_window_size_spinbox = QSpinBox()
        self.xmem2_window_size_spinbox.setRange(2, 4096)
        self.xmem2_window_size_spinbox.setValue(self.xmem2_settings.window_size)
        self.xmem2_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed XMem2 window. Smaller values reduce memory pressure."
        )
        xmem2_layout.addRow("Window size:", self.xmem2_window_size_spinbox)

        self.xmem2_window_overlap_spinbox = QSpinBox()
        self.xmem2_window_overlap_spinbox.setRange(0, 1024)
        self.xmem2_window_overlap_spinbox.setValue(self.xmem2_settings.window_overlap)
        self.xmem2_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive XMem2 windows. "
            "Trackleed reseeds the next window from this overlap and enforces at least one shared frame when multiple windows are used."
        )
        xmem2_layout.addRow("Window overlap:", self.xmem2_window_overlap_spinbox)

        self.xmem2_repr_mode_combo = QComboBox()
        self.xmem2_repr_mode_combo.addItem("Mask centroid", "mask_centroid")
        self.xmem2_repr_mode_combo.addItem("Peak in mask", "peak_in_mask")
        self.xmem2_repr_mode_combo.addItem("BBox center", "bbox_center")
        self.xmem2_repr_mode_combo.addItem("Gaussian in mask", "gaussian_in_mask")
        xmem2_repr_idx = self.xmem2_repr_mode_combo.findData(self.xmem2_settings.representative_point_mode)
        self.xmem2_repr_mode_combo.setCurrentIndex(xmem2_repr_idx if xmem2_repr_idx >= 0 else 0)
        self.xmem2_repr_mode_combo.setToolTip(
            "How Trackleed converts the propagated XMem2 mask into one representative point."
        )
        xmem2_layout.addRow("Representative point:", self.xmem2_repr_mode_combo)

        self.xmem2_bbox_padding_spinbox = QDoubleSpinBox()
        self.xmem2_bbox_padding_spinbox.setDecimals(1)
        self.xmem2_bbox_padding_spinbox.setRange(0.0, 256.0)
        self.xmem2_bbox_padding_spinbox.setSingleStep(1.0)
        self.xmem2_bbox_padding_spinbox.setValue(self.xmem2_settings.initial_bbox_padding_px)
        self.xmem2_bbox_padding_spinbox.setToolTip(
            "Extra padding in pixels added around roi_rect before building the initial XMem2 mask."
        )
        xmem2_layout.addRow("Init bbox padding (px):", self.xmem2_bbox_padding_spinbox)

        self.xmem2_initial_mask_mode_combo = QComboBox()
        self.xmem2_initial_mask_mode_combo.addItem("Threshold in bbox", "threshold")
        self.xmem2_initial_mask_mode_combo.addItem("Full bbox mask", "full_bbox")
        self.xmem2_initial_mask_mode_combo.addItem("Disk in bbox", "disk")
        xmem2_mask_mode_idx = self.xmem2_initial_mask_mode_combo.findData(self.xmem2_settings.initial_mask_mode)
        self.xmem2_initial_mask_mode_combo.setCurrentIndex(xmem2_mask_mode_idx if xmem2_mask_mode_idx >= 0 else 0)
        self.xmem2_initial_mask_mode_combo.setToolTip(
            "How Trackleed turns the ROI bbox into the initial seed mask for XMem2."
        )
        xmem2_layout.addRow("Initial mask mode:", self.xmem2_initial_mask_mode_combo)

        self.xmem2_initial_mask_threshold_spinbox = QDoubleSpinBox()
        self.xmem2_initial_mask_threshold_spinbox.setDecimals(2)
        self.xmem2_initial_mask_threshold_spinbox.setRange(0.00, 1.00)
        self.xmem2_initial_mask_threshold_spinbox.setSingleStep(0.05)
        self.xmem2_initial_mask_threshold_spinbox.setValue(self.xmem2_settings.initial_mask_threshold_rel)
        self.xmem2_initial_mask_threshold_spinbox.setToolTip(
            "Relative threshold used by the default bbox->mask builder. Higher values keep only the brightest core."
        )
        xmem2_layout.addRow("Threshold rel.:", self.xmem2_initial_mask_threshold_spinbox)

        self.xmem2_initial_mask_disk_spinbox = QDoubleSpinBox()
        self.xmem2_initial_mask_disk_spinbox.setDecimals(2)
        self.xmem2_initial_mask_disk_spinbox.setRange(0.05, 1.00)
        self.xmem2_initial_mask_disk_spinbox.setSingleStep(0.05)
        self.xmem2_initial_mask_disk_spinbox.setValue(self.xmem2_settings.initial_mask_disk_radius_ratio)
        self.xmem2_initial_mask_disk_spinbox.setToolTip(
            "Disk radius as a fraction of the smaller ROI dimension when the disk seed mode is used."
        )
        xmem2_layout.addRow("Disk radius ratio:", self.xmem2_initial_mask_disk_spinbox)

        self.xmem2_mem_every_spinbox = QSpinBox()
        self.xmem2_mem_every_spinbox.setRange(0, 512)
        self.xmem2_mem_every_spinbox.setValue(
            0 if self.xmem2_settings.mem_every in {None, 0} else int(self.xmem2_settings.mem_every)
        )
        self.xmem2_mem_every_spinbox.setToolTip(
            "How often XMem2 stores memory frames. Set 0 to keep the upstream default."
        )
        xmem2_layout.addRow("Memory every N frames:", self.xmem2_mem_every_spinbox)

        self.xmem2_max_step_spinbox = QDoubleSpinBox()
        self.xmem2_max_step_spinbox.setDecimals(1)
        self.xmem2_max_step_spinbox.setRange(0.0, 10000.0)
        self.xmem2_max_step_spinbox.setSingleStep(1.0)
        self.xmem2_max_step_spinbox.setValue(self.xmem2_settings.max_step_distance_px)
        self.xmem2_max_step_spinbox.setToolTip(
            "Reject representative points that jump farther than this distance from the last accepted XMem2 point. Set 0 to disable."
        )
        xmem2_layout.addRow("Max jump (px):", self.xmem2_max_step_spinbox)

        self.xmem2_max_gap_spinbox = QSpinBox()
        self.xmem2_max_gap_spinbox.setRange(0, 4096)
        self.xmem2_max_gap_spinbox.setValue(self.xmem2_settings.max_reacquisition_gap)
        self.xmem2_max_gap_spinbox.setToolTip(
            "If an XMem2 track stays invalid longer than this gap, later reacquisition attempts are discarded. Set 0 to disable."
        )
        xmem2_layout.addRow("Max reacquisition gap:", self.xmem2_max_gap_spinbox)

        self.xmem2_terminate_gap_spinbox = QSpinBox()
        self.xmem2_terminate_gap_spinbox.setRange(0, 4096)
        self.xmem2_terminate_gap_spinbox.setValue(self.xmem2_settings.terminate_after_invisible_frames)
        self.xmem2_terminate_gap_spinbox.setToolTip(
            "Stop writing any more XMem2 points after this many consecutive invalid frames. Set 0 to disable."
        )
        xmem2_layout.addRow("Terminate after gap:", self.xmem2_terminate_gap_spinbox)

        self.xmem2_max_area_ratio_spinbox = QDoubleSpinBox()
        self.xmem2_max_area_ratio_spinbox.setDecimals(2)
        self.xmem2_max_area_ratio_spinbox.setRange(0.0, 100.0)
        self.xmem2_max_area_ratio_spinbox.setSingleStep(0.05)
        self.xmem2_max_area_ratio_spinbox.setValue(self.xmem2_settings.max_mask_area_ratio_change)
        self.xmem2_max_area_ratio_spinbox.setToolTip(
            "Maximum allowed multiplicative change of mask area between accepted frames. Set 0 to disable."
        )
        xmem2_layout.addRow("Max area drift x:", self.xmem2_max_area_ratio_spinbox)

        self.xmem2_max_aspect_ratio_spinbox = QDoubleSpinBox()
        self.xmem2_max_aspect_ratio_spinbox.setDecimals(2)
        self.xmem2_max_aspect_ratio_spinbox.setRange(0.0, 100.0)
        self.xmem2_max_aspect_ratio_spinbox.setSingleStep(0.05)
        self.xmem2_max_aspect_ratio_spinbox.setValue(self.xmem2_settings.max_bbox_aspect_ratio_change)
        self.xmem2_max_aspect_ratio_spinbox.setToolTip(
            "Maximum allowed multiplicative change of mask bbox aspect ratio between accepted frames. Set 0 to disable."
        )
        xmem2_layout.addRow("Max aspect drift x:", self.xmem2_max_aspect_ratio_spinbox)

        self.xmem2_bbox_jump_spinbox = QDoubleSpinBox()
        self.xmem2_bbox_jump_spinbox.setDecimals(1)
        self.xmem2_bbox_jump_spinbox.setRange(0.0, 10000.0)
        self.xmem2_bbox_jump_spinbox.setSingleStep(1.0)
        self.xmem2_bbox_jump_spinbox.setValue(self.xmem2_settings.max_bbox_center_jump_px)
        self.xmem2_bbox_jump_spinbox.setToolTip(
            "Maximum accepted shift of the propagated mask bbox center between accepted frames. Set 0 to disable."
        )
        xmem2_layout.addRow("Max bbox jump (px):", self.xmem2_bbox_jump_spinbox)

        self.xmem2_keep_gaps_checkbox = QCheckBox("Keep invisible masks as gaps")
        self.xmem2_keep_gaps_checkbox.setChecked(self.xmem2_settings.keep_invisible_points_as_gaps)
        self.xmem2_keep_gaps_checkbox.setToolTip(
            "If enabled, missing/invalid XMem2 masks leave gaps instead of forcing a representative point."
        )
        xmem2_layout.addRow(self.xmem2_keep_gaps_checkbox)

        self.xmem2_center_mode_combo = QComboBox()
        self.xmem2_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.xmem2_center_mode_combo.addItem("Manual", "manual")
        xmem2_center_mode_idx = self.xmem2_center_mode_combo.findData(self.xmem2_settings.center_mode)
        self.xmem2_center_mode_combo.setCurrentIndex(
            xmem2_center_mode_idx if xmem2_center_mode_idx >= 0 else 0
        )
        self.xmem2_center_mode_combo.setToolTip(
            "Reference center used by XMem2 direction gating. Auto estimates the LEED pattern center from detections."
        )
        self.xmem2_center_mode_combo.currentIndexChanged.connect(self._on_xmem2_center_mode_changed)
        xmem2_layout.addRow("Direction center:", self.xmem2_center_mode_combo)

        xmem2_center_row_widget = QWidget()
        xmem2_center_row_layout = QHBoxLayout(xmem2_center_row_widget)
        xmem2_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.xmem2_center_x_spinbox = QDoubleSpinBox()
        self.xmem2_center_x_spinbox.setRange(0.0, 10000.0)
        self.xmem2_center_x_spinbox.setDecimals(1)
        self.xmem2_center_x_spinbox.setValue(self.xmem2_settings.manual_center_x)
        self.xmem2_center_y_spinbox = QDoubleSpinBox()
        self.xmem2_center_y_spinbox.setRange(0.0, 10000.0)
        self.xmem2_center_y_spinbox.setDecimals(1)
        self.xmem2_center_y_spinbox.setValue(self.xmem2_settings.manual_center_y)
        xmem2_center_row_layout.addWidget(self.xmem2_center_x_spinbox)
        xmem2_center_row_layout.addWidget(QLabel(","))
        xmem2_center_row_layout.addWidget(self.xmem2_center_y_spinbox)
        xmem2_layout.addRow("Manual center (x, y):", xmem2_center_row_widget)

        self.xmem2_direction_mode_combo = QComboBox()
        self.xmem2_direction_mode_combo.addItem("None", "none")
        self.xmem2_direction_mode_combo.addItem("Auto", "auto")
        self.xmem2_direction_mode_combo.addItem("Inward", "inward")
        self.xmem2_direction_mode_combo.addItem("Outward", "outward")
        xmem2_direction_idx = self.xmem2_direction_mode_combo.findData(self.xmem2_settings.direction_mode)
        self.xmem2_direction_mode_combo.setCurrentIndex(
            xmem2_direction_idx if xmem2_direction_idx >= 0 else 0
        )
        self.xmem2_direction_mode_combo.setToolTip(
            "Optional directional prior applied to the XMem2 representative point."
        )
        xmem2_layout.addRow("Direction mode:", self.xmem2_direction_mode_combo)

        self.xmem2_lateral_tol_spinbox = QDoubleSpinBox()
        self.xmem2_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.xmem2_lateral_tol_spinbox.setSingleStep(1.0)
        self.xmem2_lateral_tol_spinbox.setValue(self.xmem2_settings.lateral_angle_tolerance_deg)
        self.xmem2_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        xmem2_layout.addRow("Angular tol (deg):", self.xmem2_lateral_tol_spinbox)

        self.xmem2_max_radial_jump_spinbox = QDoubleSpinBox()
        self.xmem2_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.xmem2_max_radial_jump_spinbox.setSingleStep(1.0)
        self.xmem2_max_radial_jump_spinbox.setValue(self.xmem2_settings.max_radial_jump_px)
        self.xmem2_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        xmem2_layout.addRow("Max radial jump (px):", self.xmem2_max_radial_jump_spinbox)

        self.xmem2_warning_label = QLabel()
        self.xmem2_warning_label.setStyleSheet("color: #d9534f;")
        self.xmem2_warning_label.setWordWrap(True)
        xmem2_layout.addRow(self.xmem2_warning_label)

        self._on_xmem2_backend_mode_changed(self.xmem2_backend_mode_combo.currentIndex())
        self._on_xmem2_center_mode_changed(self.xmem2_center_mode_combo.currentIndex())
        form_layout.addRow(self.xmem2_group)

        # --- Offline SAM2 parameters ---
        self.sam2_group = QGroupBox("Offline SAM2")
        sam2_layout = QFormLayout(self.sam2_group)

        self.sam2_repo_path_edit = QLineEdit(self.sam2_settings.repo_path or "")
        self.sam2_repo_path_edit.setPlaceholderText("Path to local sam2 repo root")
        self.sam2_repo_path_edit.setToolTip(
            "Local clone of the SAM 2 repository. Must contain sam2/build_sam.py and configs/."
        )
        self.sam2_repo_path_edit.editingFinished.connect(self._refresh_sam2_checkpoints)
        sam2_layout.addRow("Repo path:", self.sam2_repo_path_edit)

        self.sam2_python_executable_edit = QLineEdit(self.sam2_settings.python_executable or "")
        self.sam2_python_executable_edit.setPlaceholderText(
            "Optional path to the Python executable inside the dedicated sam2 env"
        )
        self.sam2_python_executable_edit.setToolTip(
            "Used by the recommended subprocess backend. Leave empty only if Trackleed itself runs "
            "inside the SAM 2 environment."
        )
        self.sam2_python_executable_edit.editingFinished.connect(self._refresh_sam2_checkpoints)
        sam2_layout.addRow("Python executable:", self.sam2_python_executable_edit)

        self.sam2_backend_mode_combo = QComboBox()
        self.sam2_backend_mode_combo.addItem("Subprocess (Recommended)", "subprocess")
        self.sam2_backend_mode_combo.addItem("Direct import (Current env)", "direct")
        sam2_backend_mode_idx = self.sam2_backend_mode_combo.findData(self.sam2_settings.backend_mode)
        self.sam2_backend_mode_combo.setCurrentIndex(sam2_backend_mode_idx if sam2_backend_mode_idx >= 0 else 0)
        self.sam2_backend_mode_combo.setToolTip(
            "Subprocess keeps the heavy SAM 2 runtime isolated in its own env. "
            "Direct import requires running the whole app inside the SAM 2 env."
        )
        self.sam2_backend_mode_combo.currentIndexChanged.connect(self._on_sam2_backend_mode_changed)
        sam2_layout.addRow("Backend mode:", self.sam2_backend_mode_combo)

        sam2_model_row_widget = QWidget()
        sam2_model_row_layout = QHBoxLayout(sam2_model_row_widget)
        sam2_model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.sam2_checkpoint_combo = QComboBox()
        self.sam2_checkpoint_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.sam2_checkpoint_combo.setToolTip(
            "Select a SAM 2 checkpoint from tracking_models/sam2 or repo/checkpoints."
        )
        self.sam2_checkpoint_combo.currentIndexChanged.connect(self._on_sam2_checkpoint_selected)
        self.sam2_refresh_button = QPushButton("Refresh")
        self.sam2_refresh_button.setToolTip(
            "Re-scan tracking_models/sam2 and re-run SAM 2 environment diagnostics."
        )
        self.sam2_refresh_button.clicked.connect(self._refresh_sam2_checkpoints)
        sam2_model_row_layout.addWidget(self.sam2_checkpoint_combo)
        sam2_model_row_layout.addWidget(self.sam2_refresh_button)
        sam2_layout.addRow("Checkpoint:", sam2_model_row_widget)

        self.sam2_checkpoint_hint_label = QLabel(
            "Hint: start with sam2.1_hiera_small.pt, then try sam2.1_hiera_base_plus.pt. "
            "sam2.1_hiera_large.pt is heavier and usually much slower."
        )
        self.sam2_checkpoint_hint_label.setWordWrap(True)
        self.sam2_checkpoint_hint_label.setStyleSheet("color: #6c757d;")
        sam2_layout.addRow(self.sam2_checkpoint_hint_label)

        self.sam2_config_path_edit = QLineEdit(self.sam2_settings.config_path or "")
        self.sam2_config_path_edit.setReadOnly(True)
        self.sam2_config_path_edit.setPlaceholderText("Resolved automatically from checkpoint name")
        self.sam2_config_path_edit.setToolTip(
            "Hydra config resolved from the selected checkpoint. Example: sam2.1_hiera_small.pt -> configs/sam2.1/sam2.1_hiera_s.yaml"
        )
        sam2_layout.addRow("Resolved config:", self.sam2_config_path_edit)

        self.sam2_device_combo = QComboBox()
        self.sam2_device_combo.addItem("Auto", "auto")
        self.sam2_device_combo.addItem("GPU", "cuda")
        self.sam2_device_combo.addItem("CPU", "cpu")
        sam2_device_idx = self.sam2_device_combo.findData(self.sam2_settings.device)
        self.sam2_device_combo.setCurrentIndex(sam2_device_idx if sam2_device_idx >= 0 else 0)
        self.sam2_device_combo.setToolTip("Device passed to the SAM 2 runtime.")
        sam2_layout.addRow("Device:", self.sam2_device_combo)

        self.sam2_inference_resolution_combo = QComboBox()
        self.sam2_inference_resolution_combo.addItem("Native", None)
        for label, value in (
            ("512 x 512", 512),
            ("768 x 768", 768),
            ("1024 x 1024", 1024),
        ):
            self.sam2_inference_resolution_combo.addItem(label, value)
        sam2_res_idx = self.sam2_inference_resolution_combo.findData(self.sam2_settings.inference_resolution)
        self.sam2_inference_resolution_combo.setCurrentIndex(sam2_res_idx if sam2_res_idx >= 0 else 0)
        self.sam2_inference_resolution_combo.setToolTip(
            "Optional square input resolution for SAM 2. Native keeps the original LEED frame size."
        )
        sam2_layout.addRow("Inference resolution:", self.sam2_inference_resolution_combo)

        self.sam2_window_size_spinbox = QSpinBox()
        self.sam2_window_size_spinbox.setRange(2, 4096)
        self.sam2_window_size_spinbox.setValue(self.sam2_settings.window_size)
        self.sam2_window_size_spinbox.setToolTip(
            "Number of frames processed by one Trackleed SAM 2 window. Smaller values reduce VRAM pressure."
        )
        sam2_layout.addRow("Window size:", self.sam2_window_size_spinbox)

        self.sam2_window_overlap_spinbox = QSpinBox()
        self.sam2_window_overlap_spinbox.setRange(0, 1024)
        self.sam2_window_overlap_spinbox.setValue(self.sam2_settings.window_overlap)
        self.sam2_window_overlap_spinbox.setToolTip(
            "Number of overlapping frames between consecutive SAM 2 windows. "
            "Trackleed reseeds the next window from this overlap and enforces at least one shared frame when multiple windows are used."
        )
        sam2_layout.addRow("Window overlap:", self.sam2_window_overlap_spinbox)

        self.sam2_prompt_mode_combo = QComboBox()
        self.sam2_prompt_mode_combo.addItem("Point", "point")
        self.sam2_prompt_mode_combo.addItem("BBox", "bbox")
        self.sam2_prompt_mode_combo.addItem("Point + BBox", "point+bbox")
        sam2_prompt_idx = self.sam2_prompt_mode_combo.findData(self.sam2_settings.prompt_mode)
        self.sam2_prompt_mode_combo.setCurrentIndex(sam2_prompt_idx if sam2_prompt_idx >= 0 else 2)
        self.sam2_prompt_mode_combo.setToolTip(
            "Prompt used to initialize each LEED spot/object in SAM 2."
        )
        sam2_layout.addRow("Prompt mode:", self.sam2_prompt_mode_combo)

        self.sam2_repr_mode_combo = QComboBox()
        self.sam2_repr_mode_combo.addItem("Mask centroid", "mask_centroid")
        self.sam2_repr_mode_combo.addItem("Peak in mask", "peak_in_mask")
        self.sam2_repr_mode_combo.addItem("BBox center", "bbox_center")
        self.sam2_repr_mode_combo.addItem("Gaussian in mask", "gaussian_in_mask")
        sam2_repr_idx = self.sam2_repr_mode_combo.findData(self.sam2_settings.representative_point_mode)
        self.sam2_repr_mode_combo.setCurrentIndex(sam2_repr_idx if sam2_repr_idx >= 0 else 0)
        self.sam2_repr_mode_combo.setToolTip(
            "How Trackleed converts the propagated SAM 2 mask into one representative point."
        )
        sam2_layout.addRow("Representative point:", self.sam2_repr_mode_combo)

        self.sam2_bbox_padding_spinbox = QDoubleSpinBox()
        self.sam2_bbox_padding_spinbox.setDecimals(1)
        self.sam2_bbox_padding_spinbox.setRange(0.0, 256.0)
        self.sam2_bbox_padding_spinbox.setSingleStep(1.0)
        self.sam2_bbox_padding_spinbox.setValue(self.sam2_settings.initial_bbox_padding_px)
        self.sam2_bbox_padding_spinbox.setToolTip(
            "Extra padding in pixels added around the seed point when Trackleed builds an init bbox for SAM 2."
        )
        sam2_layout.addRow("Init bbox padding (px):", self.sam2_bbox_padding_spinbox)

        self.sam2_mask_threshold_spinbox = QDoubleSpinBox()
        self.sam2_mask_threshold_spinbox.setDecimals(2)
        self.sam2_mask_threshold_spinbox.setRange(0.00, 1.00)
        self.sam2_mask_threshold_spinbox.setSingleStep(0.05)
        self.sam2_mask_threshold_spinbox.setValue(self.sam2_settings.mask_threshold)
        self.sam2_mask_threshold_spinbox.setToolTip(
            "Threshold applied to SAM 2 mask logits before deriving mask area, bbox, and representative point."
        )
        sam2_layout.addRow("Mask threshold:", self.sam2_mask_threshold_spinbox)

        self.sam2_min_mask_area_spinbox = QDoubleSpinBox()
        self.sam2_min_mask_area_spinbox.setDecimals(1)
        self.sam2_min_mask_area_spinbox.setRange(0.0, 1000000.0)
        self.sam2_min_mask_area_spinbox.setSingleStep(1.0)
        self.sam2_min_mask_area_spinbox.setValue(self.sam2_settings.min_mask_area)
        self.sam2_min_mask_area_spinbox.setToolTip(
            "Reject masks smaller than this pixel area after thresholding."
        )
        sam2_layout.addRow("Min mask area:", self.sam2_min_mask_area_spinbox)

        self.sam2_max_mask_area_spinbox = QDoubleSpinBox()
        self.sam2_max_mask_area_spinbox.setDecimals(1)
        self.sam2_max_mask_area_spinbox.setRange(0.0, 1000000.0)
        self.sam2_max_mask_area_spinbox.setSingleStep(10.0)
        self.sam2_max_mask_area_spinbox.setValue(self.sam2_settings.max_mask_area)
        self.sam2_max_mask_area_spinbox.setToolTip(
            "Reject masks larger than this pixel area. Set 0 to disable the upper bound."
        )
        sam2_layout.addRow("Max mask area:", self.sam2_max_mask_area_spinbox)

        self.sam2_use_vos_checkbox = QCheckBox("Use vos_optimized")
        self.sam2_use_vos_checkbox.setChecked(self.sam2_settings.use_vos_optimized)
        self.sam2_use_vos_checkbox.setToolTip(
            "Enable the optimized video segmentation path when supported by the installed SAM 2 runtime."
        )
        sam2_layout.addRow(self.sam2_use_vos_checkbox)

        self.sam2_postprocessing_checkbox = QCheckBox("Apply postprocessing")
        self.sam2_postprocessing_checkbox.setChecked(self.sam2_settings.apply_postprocessing)
        self.sam2_postprocessing_checkbox.setToolTip(
            "Enable upstream SAM 2 postprocessing when constructing masks."
        )
        sam2_layout.addRow(self.sam2_postprocessing_checkbox)

        self.sam2_offload_video_checkbox = QCheckBox("Offload video to CPU")
        self.sam2_offload_video_checkbox.setChecked(self.sam2_settings.offload_video_to_cpu)
        self.sam2_offload_video_checkbox.setToolTip(
            "Store video frames on CPU inside the SAM 2 state to reduce VRAM usage at the cost of speed."
        )
        sam2_layout.addRow(self.sam2_offload_video_checkbox)

        self.sam2_offload_state_checkbox = QCheckBox("Offload state to CPU")
        self.sam2_offload_state_checkbox.setChecked(self.sam2_settings.offload_state_to_cpu)
        self.sam2_offload_state_checkbox.setToolTip(
            "Move more of the SAM 2 internal state to CPU memory. Useful on smaller GPUs, but slower."
        )
        sam2_layout.addRow(self.sam2_offload_state_checkbox)

        self.sam2_async_loading_checkbox = QCheckBox("Async frame loading")
        self.sam2_async_loading_checkbox.setChecked(self.sam2_settings.async_loading_frames)
        self.sam2_async_loading_checkbox.setToolTip(
            "Let SAM 2 load video frames asynchronously when supported by the runtime."
        )
        sam2_layout.addRow(self.sam2_async_loading_checkbox)

        self.sam2_keep_gaps_checkbox = QCheckBox("Keep invalid masks as gaps")
        self.sam2_keep_gaps_checkbox.setChecked(self.sam2_settings.keep_invisible_points_as_gaps)
        self.sam2_keep_gaps_checkbox.setToolTip(
            "If enabled, invalid/invisible SAM 2 masks produce gaps instead of forced representative points."
        )
        sam2_layout.addRow(self.sam2_keep_gaps_checkbox)

        self.sam2_max_step_spinbox = QDoubleSpinBox()
        self.sam2_max_step_spinbox.setDecimals(1)
        self.sam2_max_step_spinbox.setRange(0.0, 10000.0)
        self.sam2_max_step_spinbox.setSingleStep(1.0)
        self.sam2_max_step_spinbox.setValue(self.sam2_settings.max_step_distance_px)
        self.sam2_max_step_spinbox.setToolTip(
            "Reject representative points that jump farther than this distance from the last accepted SAM 2 point. Set 0 to disable."
        )
        sam2_layout.addRow("Max jump (px):", self.sam2_max_step_spinbox)

        self.sam2_max_gap_spinbox = QSpinBox()
        self.sam2_max_gap_spinbox.setRange(0, 4096)
        self.sam2_max_gap_spinbox.setValue(self.sam2_settings.max_reacquisition_gap)
        self.sam2_max_gap_spinbox.setToolTip(
            "If a SAM 2 track stays invalid longer than this gap, later reacquisition attempts are discarded. Set 0 to disable."
        )
        sam2_layout.addRow("Max reacquisition gap:", self.sam2_max_gap_spinbox)

        self.sam2_terminate_gap_spinbox = QSpinBox()
        self.sam2_terminate_gap_spinbox.setRange(0, 4096)
        self.sam2_terminate_gap_spinbox.setValue(self.sam2_settings.terminate_after_invisible_frames)
        self.sam2_terminate_gap_spinbox.setToolTip(
            "Stop writing any more SAM 2 points after this many consecutive invalid frames. Set 0 to disable."
        )
        sam2_layout.addRow("Terminate after gap:", self.sam2_terminate_gap_spinbox)

        self.sam2_max_area_ratio_spinbox = QDoubleSpinBox()
        self.sam2_max_area_ratio_spinbox.setDecimals(2)
        self.sam2_max_area_ratio_spinbox.setRange(0.0, 100.0)
        self.sam2_max_area_ratio_spinbox.setSingleStep(0.05)
        self.sam2_max_area_ratio_spinbox.setValue(self.sam2_settings.max_mask_area_ratio_change)
        self.sam2_max_area_ratio_spinbox.setToolTip(
            "Maximum allowed multiplicative change of mask area between accepted frames. Set 0 to disable."
        )
        sam2_layout.addRow("Max area drift x:", self.sam2_max_area_ratio_spinbox)

        self.sam2_max_aspect_ratio_spinbox = QDoubleSpinBox()
        self.sam2_max_aspect_ratio_spinbox.setDecimals(2)
        self.sam2_max_aspect_ratio_spinbox.setRange(0.0, 100.0)
        self.sam2_max_aspect_ratio_spinbox.setSingleStep(0.05)
        self.sam2_max_aspect_ratio_spinbox.setValue(self.sam2_settings.max_bbox_aspect_ratio_change)
        self.sam2_max_aspect_ratio_spinbox.setToolTip(
            "Maximum allowed multiplicative change of mask bbox aspect ratio between accepted frames. Set 0 to disable."
        )
        sam2_layout.addRow("Max aspect drift x:", self.sam2_max_aspect_ratio_spinbox)

        self.sam2_max_components_spinbox = QSpinBox()
        self.sam2_max_components_spinbox.setRange(0, 1024)
        self.sam2_max_components_spinbox.setValue(self.sam2_settings.max_mask_components)
        self.sam2_max_components_spinbox.setToolTip(
            "Reject masks split into more connected components than this. Set 0 to disable."
        )
        sam2_layout.addRow("Max components:", self.sam2_max_components_spinbox)

        self.sam2_bbox_jump_spinbox = QDoubleSpinBox()
        self.sam2_bbox_jump_spinbox.setDecimals(1)
        self.sam2_bbox_jump_spinbox.setRange(0.0, 10000.0)
        self.sam2_bbox_jump_spinbox.setSingleStep(1.0)
        self.sam2_bbox_jump_spinbox.setValue(self.sam2_settings.max_bbox_center_jump_px)
        self.sam2_bbox_jump_spinbox.setToolTip(
            "Maximum accepted shift of the propagated mask bbox center between accepted frames. Set 0 to disable."
        )
        sam2_layout.addRow("Max bbox jump (px):", self.sam2_bbox_jump_spinbox)

        self.sam2_center_mode_combo = QComboBox()
        self.sam2_center_mode_combo.addItem("Auto (RANSAC/Image)", "auto")
        self.sam2_center_mode_combo.addItem("Manual", "manual")
        sam2_center_mode_idx = self.sam2_center_mode_combo.findData(self.sam2_settings.center_mode)
        self.sam2_center_mode_combo.setCurrentIndex(sam2_center_mode_idx if sam2_center_mode_idx >= 0 else 0)
        self.sam2_center_mode_combo.setToolTip(
            "Reference center used by SAM 2 direction gating. Auto estimates the LEED pattern center from detections."
        )
        self.sam2_center_mode_combo.currentIndexChanged.connect(self._on_sam2_center_mode_changed)
        sam2_layout.addRow("Direction center:", self.sam2_center_mode_combo)

        sam2_center_row_widget = QWidget()
        sam2_center_row_layout = QHBoxLayout(sam2_center_row_widget)
        sam2_center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.sam2_center_x_spinbox = QDoubleSpinBox()
        self.sam2_center_x_spinbox.setRange(0.0, 10000.0)
        self.sam2_center_x_spinbox.setDecimals(1)
        self.sam2_center_x_spinbox.setValue(self.sam2_settings.manual_center_x)
        self.sam2_center_y_spinbox = QDoubleSpinBox()
        self.sam2_center_y_spinbox.setRange(0.0, 10000.0)
        self.sam2_center_y_spinbox.setDecimals(1)
        self.sam2_center_y_spinbox.setValue(self.sam2_settings.manual_center_y)
        sam2_center_row_layout.addWidget(self.sam2_center_x_spinbox)
        sam2_center_row_layout.addWidget(QLabel(","))
        sam2_center_row_layout.addWidget(self.sam2_center_y_spinbox)
        sam2_layout.addRow("Manual center (x, y):", sam2_center_row_widget)

        self.sam2_direction_mode_combo = QComboBox()
        self.sam2_direction_mode_combo.addItem("None", "none")
        self.sam2_direction_mode_combo.addItem("Auto", "auto")
        self.sam2_direction_mode_combo.addItem("Inward", "inward")
        self.sam2_direction_mode_combo.addItem("Outward", "outward")
        sam2_direction_idx = self.sam2_direction_mode_combo.findData(self.sam2_settings.direction_mode)
        self.sam2_direction_mode_combo.setCurrentIndex(sam2_direction_idx if sam2_direction_idx >= 0 else 0)
        self.sam2_direction_mode_combo.setToolTip(
            "Optional directional prior applied to the SAM 2 representative point."
        )
        sam2_layout.addRow("Direction mode:", self.sam2_direction_mode_combo)

        self.sam2_lateral_tol_spinbox = QDoubleSpinBox()
        self.sam2_lateral_tol_spinbox.setRange(1.0, 180.0)
        self.sam2_lateral_tol_spinbox.setSingleStep(1.0)
        self.sam2_lateral_tol_spinbox.setValue(self.sam2_settings.lateral_angle_tolerance_deg)
        self.sam2_lateral_tol_spinbox.setToolTip(
            "Maximum angular deviation from the expected inward/outward direction."
        )
        sam2_layout.addRow("Angular tol (deg):", self.sam2_lateral_tol_spinbox)

        self.sam2_max_radial_jump_spinbox = QDoubleSpinBox()
        self.sam2_max_radial_jump_spinbox.setRange(0.0, 500.0)
        self.sam2_max_radial_jump_spinbox.setSingleStep(1.0)
        self.sam2_max_radial_jump_spinbox.setValue(self.sam2_settings.max_radial_jump_px)
        self.sam2_max_radial_jump_spinbox.setToolTip(
            "Maximum allowed change in radius relative to the LEED center between accepted frames. Set 0 to disable."
        )
        sam2_layout.addRow("Max radial jump (px):", self.sam2_max_radial_jump_spinbox)

        self.sam2_warning_label = QLabel()
        self.sam2_warning_label.setStyleSheet("color: #d9534f;")
        self.sam2_warning_label.setWordWrap(True)
        sam2_layout.addRow(self.sam2_warning_label)

        self._on_sam2_backend_mode_changed(self.sam2_backend_mode_combo.currentIndex())
        self._on_sam2_center_mode_changed(self.sam2_center_mode_combo.currentIndex())

        form_layout.addRow(self.sam2_group)

        # --- Detection linking (Hungarian) params ---
        self.hungarian_group = QGroupBox("Detection Linking (Hungarian)")
        hung_layout = QFormLayout(self.hungarian_group)
        self.hung_max_dist_spinbox = QDoubleSpinBox()
        self.hung_max_dist_spinbox.setRange(1.0, 200.0)
        self.hung_max_dist_spinbox.setValue(20.0)
        self.hung_max_dist_spinbox.setSingleStep(1.0)
        self.hung_max_dist_spinbox.setToolTip("Maximum distance (px) to associate detections between consecutive frames.")
        hung_layout.addRow("Max distance (px):", self.hung_max_dist_spinbox)
        self.hung_max_gap_spinbox = QSpinBox()
        self.hung_max_gap_spinbox.setRange(0, 9999)
        self.hung_max_gap_spinbox.setValue(2)
        self.hung_max_gap_spinbox.setToolTip("Allowed gap (frames) to keep a track alive when no detection is matched.")
        hung_layout.addRow("Max gap (frames):", self.hung_max_gap_spinbox)
        self.hung_min_len_spinbox = QSpinBox()
        self.hung_min_len_spinbox.setRange(1, 200)
        self.hung_min_len_spinbox.setValue(2)
        self.hung_min_len_spinbox.setToolTip("Discard tracks shorter than this many detections.")
        hung_layout.addRow("Min track length:", self.hung_min_len_spinbox)
        form_layout.addRow(self.hungarian_group)

        # --- Directional detection linking params ---
        self.dir_group = QGroupBox("Directional Detection Linking")
        dir_layout = QFormLayout(self.dir_group)

        self.dir_center_mode_combo = QComboBox()
        self.dir_center_mode_combo.addItems(["Auto (RANSAC)", "Manual"])
        self.dir_center_mode_combo.setCurrentIndex(0)
        self.dir_center_mode_combo.setToolTip("Use RANSAC-estimated pattern center (auto) or manual coordinates.")
        self.dir_center_mode_combo.currentIndexChanged.connect(self._on_dir_center_mode_changed)
        dir_layout.addRow("Center mode:", self.dir_center_mode_combo)

        center_row = QWidget()
        center_row_layout = QHBoxLayout(center_row)
        center_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dir_center_x_spinbox = QDoubleSpinBox()
        self.dir_center_x_spinbox.setRange(0.0, 10000.0)
        self.dir_center_x_spinbox.setDecimals(1)
        self.dir_center_x_spinbox.setValue(0.0)
        self.dir_center_y_spinbox = QDoubleSpinBox()
        self.dir_center_y_spinbox.setRange(0.0, 10000.0)
        self.dir_center_y_spinbox.setDecimals(1)
        self.dir_center_y_spinbox.setValue(0.0)
        center_row_layout.addWidget(QLabel("Cx:"))
        center_row_layout.addWidget(self.dir_center_x_spinbox)
        center_row_layout.addSpacing(8)
        center_row_layout.addWidget(QLabel("Cy:"))
        center_row_layout.addWidget(self.dir_center_y_spinbox)
        dir_layout.addRow("Manual center (px):", center_row)

        self.dir_direction_combo = QComboBox()
        self.dir_direction_combo.addItems(["Auto", "Inward", "Outward"])
        self.dir_direction_combo.setCurrentIndex(0)
        self.dir_direction_combo.setToolTip("Expected dominant radial direction; Auto infers sign from motion.")
        dir_layout.addRow("Direction:", self.dir_direction_combo)

        self.dir_lateral_tol_spinbox = QDoubleSpinBox()
        self.dir_lateral_tol_spinbox.setRange(0.0, 90.0)
        self.dir_lateral_tol_spinbox.setDecimals(1)
        self.dir_lateral_tol_spinbox.setValue(20.0)
        self.dir_lateral_tol_spinbox.setToolTip("Allowed angular deviation from the radial direction before penalty.")
        dir_layout.addRow("Lateral tolerance (deg):", self.dir_lateral_tol_spinbox)

        self.dir_ring_bin_spinbox = QDoubleSpinBox()
        self.dir_ring_bin_spinbox.setRange(1.0, 500.0)
        self.dir_ring_bin_spinbox.setDecimals(1)
        self.dir_ring_bin_spinbox.setValue(25.0)
        self.dir_ring_bin_spinbox.setToolTip("Ring bin size (px) to penalize jumps between distant rings.")
        dir_layout.addRow("Ring bin (px):", self.dir_ring_bin_spinbox)

        self.dir_base_max_dist_spinbox = QDoubleSpinBox()
        self.dir_base_max_dist_spinbox.setRange(1.0, 500.0)
        self.dir_base_max_dist_spinbox.setDecimals(1)
        self.dir_base_max_dist_spinbox.setValue(20.0)
        self.dir_base_max_dist_spinbox.setToolTip("Base max distance for matching (used when dynamic gating is off).")
        dir_layout.addRow("Base max distance (px):", self.dir_base_max_dist_spinbox)

        self.dir_dynamic_max_checkbox = QCheckBox("Dynamic max distance")
        self.dir_dynamic_max_checkbox.setChecked(True)
        self.dir_dynamic_max_checkbox.setToolTip("Adjust match radius from recent step median plus margin.")
        dir_layout.addRow(self.dir_dynamic_max_checkbox)

        self.dir_dynamic_margin_spinbox = QDoubleSpinBox()
        self.dir_dynamic_margin_spinbox.setRange(0.0, 200.0)
        self.dir_dynamic_margin_spinbox.setDecimals(1)
        self.dir_dynamic_margin_spinbox.setValue(5.0)
        self.dir_dynamic_margin_spinbox.setToolTip("Extra slack added to median step when dynamic gating is on.")
        dir_layout.addRow("Dynamic margin (px):", self.dir_dynamic_margin_spinbox)

        self.dir_max_gap_spinbox = QSpinBox()
        self.dir_max_gap_spinbox.setRange(0, 9999)
        self.dir_max_gap_spinbox.setValue(2)
        self.dir_max_gap_spinbox.setToolTip("Allowed gap (frames) to keep a directional track alive when unmatched.")
        dir_layout.addRow("Max gap (frames):", self.dir_max_gap_spinbox)

        self.dir_max_misses_spinbox = QSpinBox()
        self.dir_max_misses_spinbox.setRange(0, 9999)
        self.dir_max_misses_spinbox.setValue(3)
        self.dir_max_misses_spinbox.setToolTip("Consecutive misses tolerated before dropping a track.")
        dir_layout.addRow("Max misses:", self.dir_max_misses_spinbox)

        self.dir_min_len_spinbox = QSpinBox()
        self.dir_min_len_spinbox.setRange(1, 200)
        self.dir_min_len_spinbox.setValue(2)
        self.dir_min_len_spinbox.setToolTip("Discard tracks shorter than this many detections.")
        dir_layout.addRow("Min track length:", self.dir_min_len_spinbox)

        form_layout.addRow(self.dir_group)
        self._on_dir_center_mode_changed(self.dir_center_mode_combo.currentIndex())

        self.tracking_scenario_combo.currentTextChanged.connect(self._on_tracking_scenario_changed)
        self._on_tracking_scenario_changed(self.tracking_scenario_combo.currentText())

        # --- Action Buttons ---
        self.track_button = QPushButton("Track Spots")
        self.track_button.clicked.connect(self.run_tracking)
        form_layout.addRow(self.track_button)

        # layout.addWidget(tracking_group)

        self.visualize_button = QPushButton("Visualize Tracking")
        self.visualize_button.setEnabled(False)
        self.visualize_button.clicked.connect(self.show_tracking_visualizer)
        # layout.addWidget(self.visualize_button)
        form_layout.addRow(self.visualize_button)

        self.analysis_button = QPushButton("Analyze Tracks")
        self.analysis_button.setEnabled(False)
        self.analysis_button.clicked.connect(self.show_analysis_dialog)
        form_layout.addRow(self.analysis_button)

        self.positions_analysis_button = QPushButton("Analyze Positions")
        self.positions_analysis_button.setEnabled(False)
        self.positions_analysis_button.clicked.connect(self.show_positions_analysis_dialog)
        form_layout.addRow(self.positions_analysis_button)

        layout.addWidget(tracking_group)

        # --- Preview Checkbox ---
        self.show_processed_checkbox = QCheckBox("Show Processed Image")
        layout.addWidget(self.show_processed_checkbox)

        layout.addWidget(QLabel("Found Spots:"))
        self.spot_list_widget = QListWidget()
        self.spot_list_widget.currentItemChanged.connect(self.on_spot_selection_changed)
        layout.addWidget(self.spot_list_widget)

        self.polygon_seed_group = QGroupBox("Polygon Seed (Cutie / XMem2)")
        polygon_group_layout = QVBoxLayout(self.polygon_seed_group)
        self.polygon_seed_info_label = QLabel(
            "Draw a polygon for the selected spot. Cutie and XMem2 require this seed and will block tracking without it."
        )
        self.polygon_seed_info_label.setWordWrap(True)
        polygon_group_layout.addWidget(self.polygon_seed_info_label)

        self.polygon_seed_status_label = QLabel("")
        self.polygon_seed_status_label.setWordWrap(True)
        polygon_group_layout.addWidget(self.polygon_seed_status_label)

        polygon_buttons_widget = QWidget()
        polygon_buttons_layout = QHBoxLayout(polygon_buttons_widget)
        polygon_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.spot_polygon_button = QPushButton("Polygon Mask...")
        self.spot_polygon_button.setToolTip(
            "Draw a polygon seed for the selected spot. Cutie and XMem2 will require this polygon."
        )
        self.spot_polygon_button.clicked.connect(self.open_spot_polygon_mask_dialog)
        polygon_buttons_layout.addWidget(self.spot_polygon_button)
        self.clear_spot_polygon_button = QPushButton("Clear Polygon")
        self.clear_spot_polygon_button.setToolTip("Remove the polygon seed from the selected spot.")
        self.clear_spot_polygon_button.clicked.connect(self.clear_selected_spot_polygon_mask)
        polygon_buttons_layout.addWidget(self.clear_spot_polygon_button)
        polygon_group_layout.addWidget(polygon_buttons_widget)
        layout.addWidget(self.polygon_seed_group)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls_widget)
        self.analysis_dock.setWidget(scroll_area)

        self._refresh_segmentation_models()
        self._refresh_tapir_checkpoints()
        self._refresh_locotrack_checkpoints()
        self._refresh_cotracker_checkpoints()
        self._refresh_tapnext_checkpoints()
        self._refresh_trackonr_checkpoints()
        self._refresh_pipspp_checkpoints()
        self._refresh_mftiq_checkpoints()
        self._refresh_tomp_checkpoints()
        self._refresh_cutie_checkpoints()
        self._refresh_xmem2_checkpoints()
        self._refresh_sam2_checkpoints()
        self._on_spot_backend_changed(self.spot_backend_combo.currentText())
        self._update_spot_polygon_buttons()
        self.polygon_seed_group.setVisible(self._polygon_seed_scenario_active())

    def _build_classic_find_spots_panel(self) -> QWidget:
        widget = QWidget()
        form_layout = QFormLayout(widget)

        self.wth_radius_spinbox = QSpinBox()
        self.wth_radius_spinbox.setRange(1, 50)
        self.wth_radius_spinbox.setValue(8)
        form_layout.addRow("WTH Radius:", self.wth_radius_spinbox)

        self.blur_sigma_spinbox = QDoubleSpinBox()
        self.blur_sigma_spinbox.setRange(1.0, 100.0)
        self.blur_sigma_spinbox.setSingleStep(0.5)
        self.blur_sigma_spinbox.setValue(20.0)
        form_layout.addRow("Blur Sigma:", self.blur_sigma_spinbox)

        self.min_distance_spinbox = QSpinBox()
        self.min_distance_spinbox.setRange(1, 100)
        self.min_distance_spinbox.setValue(10)
        form_layout.addRow("Min Peak Distance:", self.min_distance_spinbox)

        self.threshold_rel_spinbox = QDoubleSpinBox()
        self.threshold_rel_spinbox.setRange(0.01, 1.0)
        self.threshold_rel_spinbox.setSingleStep(0.01)
        self.threshold_rel_spinbox.setValue(0.1)
        form_layout.addRow("Relative Threshold:", self.threshold_rel_spinbox)

        self.roi_size_spinbox = QSpinBox()
        self.roi_size_spinbox.setRange(8, 128)
        self.roi_size_spinbox.setSingleStep(2)
        self.roi_size_spinbox.setValue(32)
        form_layout.addRow("ROI Size:", self.roi_size_spinbox)

        return widget

    def _build_dl_find_spots_panel(self) -> QWidget:
        widget = QWidget()
        form_layout = QFormLayout(widget)

        model_row_widget = QWidget()
        model_row_layout = QHBoxLayout(model_row_widget)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dl_model_combo = QComboBox()
        self.dl_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dl_model_combo.setToolTip(
            "Select a trained segmentation model (.pt) placed in tracking_models/ "
            "(legacy fallback: segmentation_models/)."
        )
        self.dl_model_combo.currentIndexChanged.connect(self._on_dl_model_selected)
        self.dl_refresh_models_button = QPushButton("Refresh")
        self.dl_refresh_models_button.setToolTip(
            "Re-scan tracking_models/ and the legacy segmentation_models/ directory."
        )
        self.dl_refresh_models_button.clicked.connect(self._refresh_segmentation_models)
        model_row_layout.addWidget(self.dl_model_combo)
        model_row_layout.addWidget(self.dl_refresh_models_button)
        form_layout.addRow("Model:", model_row_widget)

        self.dl_confidence_spinbox = QDoubleSpinBox()
        self.dl_confidence_spinbox.setRange(0.1, 0.99)
        self.dl_confidence_spinbox.setSingleStep(0.01)
        self.dl_confidence_spinbox.setValue(self.dl_settings.confidence)
        self.dl_confidence_spinbox.setToolTip(
            "Cut-off applied to the U-Net probability map (higher = fewer, cleaner components)."
        )
        self.dl_confidence_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "confidence", float(v))
        )
        form_layout.addRow("Confidence:", self.dl_confidence_spinbox)

        self.dl_min_area_spinbox = QSpinBox()
        self.dl_min_area_spinbox.setRange(1, 10000)
        self.dl_min_area_spinbox.setValue(self.dl_settings.min_area_px)
        self.dl_min_area_spinbox.setToolTip("Discard connected components smaller than this pixel area to avoid noise blobs.")
        self.dl_min_area_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "min_area_px", int(v))
        )
        form_layout.addRow("Min area (px):", self.dl_min_area_spinbox)

        self.dl_max_area_spinbox = QSpinBox()
        self.dl_max_area_spinbox.setRange(10, 100000)
        self.dl_max_area_spinbox.setValue(self.dl_settings.max_area_px)
        self.dl_max_area_spinbox.setToolTip("Discard unusually large components (pixels). Increase if spots cover many pixels.")
        self.dl_max_area_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "max_area_px", int(v))
        )
        form_layout.addRow("Max area (px):", self.dl_max_area_spinbox)

        self.dl_bbox_margin_spinbox = QSpinBox()
        self.dl_bbox_margin_spinbox.setRange(0, 50)
        self.dl_bbox_margin_spinbox.setValue(self.dl_settings.bbox_margin_px)
        self.dl_bbox_margin_spinbox.setToolTip("Extra padding (pixels) around each connected component when creating SpotROI rectangles.")
        self.dl_bbox_margin_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "bbox_margin_px", int(v))
        )
        form_layout.addRow("BBox margin (px):", self.dl_bbox_margin_spinbox)

        self.dl_dilate_spinbox = QSpinBox()
        self.dl_dilate_spinbox.setRange(0, 15)
        self.dl_dilate_spinbox.setValue(self.dl_settings.dilate_radius_px)
        self.dl_dilate_spinbox.setToolTip(
            "Optional dilation radius in pixels applied to the binary mask before measuring components (helps close gaps)."
        )
        self.dl_dilate_spinbox.valueChanged.connect(
            lambda v: setattr(self.dl_settings, "dilate_radius_px", int(v))
        )
        form_layout.addRow("Dilate radius (px):", self.dl_dilate_spinbox)

        info_label = QLabel(
            "Drop your trained checkpoints into <code>tracking_models/</code>.\n"
            "Legacy fallback: <code>segmentation_models/</code>.\n"
            "The model produces a probability mask that we threshold, clean up, and convert to ROIs based on the settings above."
        )
        info_label.setWordWrap(True)
        form_layout.addRow(info_label)

        self.dl_warning_label = QLabel()
        self.dl_warning_label.setStyleSheet("color: #d9534f;")
        self.dl_warning_label.setWordWrap(True)
        form_layout.addRow(self.dl_warning_label)

        return widget

    def _build_yolo_find_spots_panel(self) -> QWidget:
        widget = QWidget()
        form_layout = QFormLayout(widget)

        model_row_widget = QWidget()
        model_row_layout = QHBoxLayout(model_row_widget)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.yolo_model_combo.setToolTip(
            "Select a YOLO checkpoint (.pt) placed in tracking_models/yolo "
            "(legacy fallback: segmentation_models/)."
        )
        self.yolo_model_combo.currentIndexChanged.connect(self._on_yolo_model_selected)
        self.yolo_refresh_models_button = QPushButton("Refresh")
        self.yolo_refresh_models_button.setToolTip(
            "Re-scan tracking_models/yolo and the legacy segmentation_models/ directory."
        )
        self.yolo_refresh_models_button.clicked.connect(self._refresh_segmentation_models)
        model_row_layout.addWidget(self.yolo_model_combo)
        model_row_layout.addWidget(self.yolo_refresh_models_button)
        form_layout.addRow("YOLO model:", model_row_widget)

        self.yolo_conf_spinbox = QDoubleSpinBox()
        self.yolo_conf_spinbox.setRange(0.01, 0.99)
        self.yolo_conf_spinbox.setSingleStep(0.01)
        self.yolo_conf_spinbox.setValue(self.yolo_settings.confidence)
        self.yolo_conf_spinbox.setToolTip("Detection confidence threshold (higher = fewer boxes).")
        self.yolo_conf_spinbox.valueChanged.connect(lambda v: setattr(self.yolo_settings, "confidence", float(v)))
        form_layout.addRow("Confidence:", self.yolo_conf_spinbox)

        self.yolo_iou_spinbox = QDoubleSpinBox()
        self.yolo_iou_spinbox.setRange(0.05, 0.95)
        self.yolo_iou_spinbox.setSingleStep(0.01)
        self.yolo_iou_spinbox.setValue(self.yolo_settings.iou)
        self.yolo_iou_spinbox.setToolTip("NMS IoU threshold used by YOLO.")
        self.yolo_iou_spinbox.valueChanged.connect(lambda v: setattr(self.yolo_settings, "iou", float(v)))
        form_layout.addRow("NMS IoU:", self.yolo_iou_spinbox)

        self.yolo_max_det_spinbox = QSpinBox()
        self.yolo_max_det_spinbox.setRange(1, 9999)
        self.yolo_max_det_spinbox.setValue(self.yolo_settings.max_det)
        self.yolo_max_det_spinbox.setToolTip("Maximum number of detections to keep.")
        self.yolo_max_det_spinbox.valueChanged.connect(lambda v: setattr(self.yolo_settings, "max_det", int(v)))
        form_layout.addRow("Max detections:", self.yolo_max_det_spinbox)

        self.yolo_bbox_margin_spinbox = QSpinBox()
        self.yolo_bbox_margin_spinbox.setRange(-50, 50)
        self.yolo_bbox_margin_spinbox.setValue(self.yolo_settings.bbox_margin_px)
        self.yolo_bbox_margin_spinbox.setToolTip(
            "Padding around YOLO boxes when creating SpotROI. "
            "Positive expands, negative shrinks."
        )
        self.yolo_bbox_margin_spinbox.valueChanged.connect(
            lambda v: setattr(self.yolo_settings, "bbox_margin_px", int(v))
        )
        form_layout.addRow("BBox margin (px):", self.yolo_bbox_margin_spinbox)

        warn = QLabel(
            "YOLO backend requires ultralytics installed. Detected boxes are converted to SpotROI; adjust confidence/IoU to control clutter."
        )
        warn.setWordWrap(True)
        form_layout.addRow(warn)

        self.yolo_warning_label = QLabel()
        self.yolo_warning_label.setStyleSheet("color: #d9534f;")
        self.yolo_warning_label.setWordWrap(True)
        form_layout.addRow(self.yolo_warning_label)

        return widget

    def _refresh_segmentation_models(self):
        dl_model_paths = self._collect_model_paths(
            self.dl_models_dir,
            self.legacy_segmentation_models_dir,
        )
        yolo_model_paths = self._collect_model_paths(
            self.yolo_models_dir,
            self.legacy_segmentation_models_dir,
        )

        # DL models
        self.dl_model_combo.blockSignals(True)
        self.dl_model_combo.clear()
        if not dl_model_paths:
            self.dl_model_combo.addItem("No models found", userData=None)
            self.dl_model_combo.setEnabled(False)
            self._dl_models_available = False
            self.dl_settings.model_path = None
            if hasattr(self, "dl_warning_label"):
                self.dl_warning_label.setText(
                    "No models detected. Copy .pt checkpoints into tracking_models/ "
                    "(legacy fallback: segmentation_models/) and press Refresh."
                )
        else:
            self.dl_model_combo.setEnabled(True)
            for path in dl_model_paths:
                self.dl_model_combo.addItem(path.name, userData=str(path.resolve()))
            self._dl_models_available = True
            if self.dl_settings.model_path:
                idx = self.dl_model_combo.findData(self.dl_settings.model_path)
                if idx >= 0:
                    self.dl_model_combo.setCurrentIndex(idx)
            if not self.dl_settings.model_path and self.dl_model_combo.count() > 0:
                self.dl_model_combo.setCurrentIndex(0)
                self.dl_settings.model_path = self.dl_model_combo.currentData()
            if hasattr(self, "dl_warning_label"):
                self.dl_warning_label.setText("")
        self.dl_model_combo.blockSignals(False)

        # YOLO models
        self.yolo_model_combo.blockSignals(True)
        self.yolo_model_combo.clear()
        if not yolo_model_paths:
            self.yolo_model_combo.addItem("No models found", userData=None)
            self.yolo_model_combo.setEnabled(False)
            self._yolo_models_available = False
            self.yolo_settings.model_path = None
            if hasattr(self, "yolo_warning_label"):
                self.yolo_warning_label.setText(
                    "No YOLO models detected. Copy .pt checkpoints into tracking_models/yolo "
                    "(legacy fallback: segmentation_models/) and press Refresh."
                )
        else:
            self.yolo_model_combo.setEnabled(True)
            for path in yolo_model_paths:
                self.yolo_model_combo.addItem(path.name, userData=str(path.resolve()))
            self._yolo_models_available = True
            if self.yolo_settings.model_path:
                idx = self.yolo_model_combo.findData(self.yolo_settings.model_path)
                if idx >= 0:
                    self.yolo_model_combo.setCurrentIndex(idx)
            if not self.yolo_settings.model_path and self.yolo_model_combo.count() > 0:
                self.yolo_model_combo.setCurrentIndex(0)
                self.yolo_settings.model_path = self.yolo_model_combo.currentData()
            if hasattr(self, "yolo_warning_label"):
                self.yolo_warning_label.setText("")
        self.yolo_model_combo.blockSignals(False)

        self._update_find_spots_button_state()

    def _refresh_tapir_checkpoints(self):
        runtime_mode = self._tapir_runtime_mode()
        self.tapir_settings.runtime_mode = runtime_mode
        status = probe_tapir_environment(
            models_dir=self.tapir_models_dir,
            checkpoint_path=self.tapir_settings.checkpoint_path,
        )
        online_names = {path.name for path in discover_online_tapir_checkpoints(self.tapir_models_dir)}
        filtered_checkpoints = (
            [path for path in status.available_checkpoints if path.name in online_names]
            if runtime_mode == "online"
            else [path for path in status.available_checkpoints if path.name not in online_names]
        )

        self.tapir_checkpoint_combo.blockSignals(True)
        self.tapir_checkpoint_combo.clear()

        if not filtered_checkpoints:
            self.tapir_checkpoint_combo.addItem(
                "No causal checkpoints found" if runtime_mode == "online" else "No offline checkpoints found",
                userData=None,
            )
            self.tapir_checkpoint_combo.setEnabled(False)
            self._tapir_checkpoint_available = False
            self.tapir_settings.checkpoint_path = None
        else:
            self.tapir_checkpoint_combo.setEnabled(True)
            for path in filtered_checkpoints:
                self.tapir_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._tapir_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if (
                    status.selected_checkpoint is not None
                    and status.selected_checkpoint.exists()
                    and status.selected_checkpoint in filtered_checkpoints
                )
                else None
            )
            if selected_checkpoint:
                idx = self.tapir_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.tapir_checkpoint_combo.setCurrentIndex(idx)
                    self.tapir_settings.checkpoint_path = selected_checkpoint
            else:
                self.tapir_settings.checkpoint_path = None
            if not self.tapir_settings.checkpoint_path and self.tapir_checkpoint_combo.count() > 0:
                self.tapir_checkpoint_combo.setCurrentIndex(0)
                self.tapir_settings.checkpoint_path = self.tapir_checkpoint_combo.currentData()

        self.tapir_checkpoint_combo.blockSignals(False)

        effective_status = probe_tapir_environment(
            models_dir=self.tapir_models_dir,
            checkpoint_path=self.tapir_settings.checkpoint_path,
        )

        if effective_status.is_ready and (
            runtime_mode != "online" or effective_status.selected_checkpoint_is_online
        ):
            details = []
            if effective_status.torch_version:
                details.append(f"torch {effective_status.torch_version}")
            details.append("CUDA" if effective_status.cuda_available else "CPU")
            details.append("causal" if runtime_mode == "online" else "offline")
            self.tapir_warning_label.setStyleSheet("color: #5cb85c;")
            self.tapir_warning_label.setText(
                "TAPIR ready. " + ", ".join(details)
            )
            self.tapir_warning_label.setToolTip(format_tapir_environment_report(effective_status))
        else:
            summary = []
            if effective_status.errors:
                summary.append(effective_status.errors[0])
            if effective_status.warnings:
                summary.extend(effective_status.warnings[:1])
            if not summary:
                summary.append(
                    "No causal TAPIR checkpoint selected."
                    if runtime_mode == "online"
                    else "No offline TAPIR checkpoint selected."
                )
            self.tapir_warning_label.setStyleSheet("color: #d9534f;")
            self.tapir_warning_label.setText("\n".join(summary))
            self.tapir_warning_label.setToolTip(format_tapir_environment_report(effective_status))

    def _on_tapir_checkpoint_selected(self, index: int):
        checkpoint_path = self.tapir_checkpoint_combo.itemData(index)
        self.tapir_settings.checkpoint_path = checkpoint_path
        self._refresh_tapir_checkpoints()

    def _refresh_locotrack_checkpoints(self):
        repo_path = self.locotrack_repo_path_edit.text().strip() or self.locotrack_settings.repo_path
        if repo_path:
            self.locotrack_settings.repo_path = repo_path
        self.locotrack_settings.python_executable = self.locotrack_python_executable_edit.text().strip() or None
        self.locotrack_settings.backend_mode = self.locotrack_backend_mode_combo.currentData()
        status = probe_locotrack_environment(
            models_dir=self.locotrack_models_dir,
            checkpoint_path=self.locotrack_settings.checkpoint_path,
            repo_path=self.locotrack_settings.repo_path,
        )

        self.locotrack_checkpoint_combo.blockSignals(True)
        self.locotrack_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.locotrack_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.locotrack_checkpoint_combo.setEnabled(False)
            self._locotrack_checkpoint_available = False
            self.locotrack_settings.checkpoint_path = None
        else:
            self.locotrack_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.locotrack_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._locotrack_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.locotrack_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.locotrack_checkpoint_combo.setCurrentIndex(idx)
                    self.locotrack_settings.checkpoint_path = selected_checkpoint
            if not self.locotrack_settings.checkpoint_path and self.locotrack_checkpoint_combo.count() > 0:
                self.locotrack_checkpoint_combo.setCurrentIndex(0)
                self.locotrack_settings.checkpoint_path = self.locotrack_checkpoint_combo.currentData()

        self.locotrack_checkpoint_combo.blockSignals(False)
        self.locotrack_repo_path_edit.setText(str(status.repo_path) if status.repo_path else (self.locotrack_settings.repo_path or ""))
        self.locotrack_settings.repo_path = str(status.repo_path) if status.repo_path else self.locotrack_settings.repo_path

        python_executable = self.locotrack_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            self.locotrack_warning_label.setStyleSheet("color: #5cb85c;")
            self.locotrack_warning_label.setText(
                "LocoTrack ready. " + ", ".join(details)
            )
            self.locotrack_warning_label.setToolTip(format_locotrack_environment_report(status))
        elif (
            self.locotrack_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.locotrack_warning_label.setStyleSheet("color: #f0ad4e;")
            self.locotrack_warning_label.setText(
                "LocoTrack subprocess configured. Current app env is not LocoTrack-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.locotrack_warning_label.setToolTip(format_locotrack_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.locotrack_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated locotrack env, or run Trackleed from that env."
                )
            self.locotrack_warning_label.setStyleSheet("color: #d9534f;")
            self.locotrack_warning_label.setText("\n".join(summary))
            self.locotrack_warning_label.setToolTip(format_locotrack_environment_report(status))

    def _on_locotrack_checkpoint_selected(self, index: int):
        checkpoint_path = self.locotrack_checkpoint_combo.itemData(index)
        self.locotrack_settings.checkpoint_path = checkpoint_path
        checkpoint_name = str(checkpoint_path or "").lower()
        inferred_size = "base" if "base" in checkpoint_name else "small" if "small" in checkpoint_name else None
        if inferred_size is not None:
            combo_index = self.locotrack_model_size_combo.findData(inferred_size)
            if combo_index >= 0:
                self.locotrack_model_size_combo.setCurrentIndex(combo_index)
                self.locotrack_settings.model_size = inferred_size
        self._refresh_locotrack_checkpoints()

    def _refresh_cotracker_checkpoints(self):
        repo_path = self.cotracker_repo_path_edit.text().strip() or self.cotracker_settings.repo_path
        if repo_path:
            self.cotracker_settings.repo_path = repo_path
        status = probe_cotracker_environment(
            models_dir=self.cotracker_models_dir,
            checkpoint_path=self.cotracker_settings.checkpoint_path,
            repo_path=self.cotracker_settings.repo_path,
        )

        self.cotracker_checkpoint_combo.blockSignals(True)
        self.cotracker_checkpoint_combo.clear()

        offline_checkpoints = [
            path for path in status.available_checkpoints
            if "online" not in path.name.lower()
        ]

        if not offline_checkpoints:
            self.cotracker_checkpoint_combo.addItem("No offline checkpoints found", userData=None)
            self.cotracker_checkpoint_combo.setEnabled(False)
            self._cotracker_checkpoint_available = False
            self.cotracker_settings.checkpoint_path = None
        else:
            self.cotracker_checkpoint_combo.setEnabled(True)
            for path in offline_checkpoints:
                self.cotracker_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._cotracker_checkpoint_available = True
            selected_checkpoint_path = None
            try:
                current_selected = Path(self.cotracker_settings.checkpoint_path).resolve() if self.cotracker_settings.checkpoint_path else None
            except Exception:
                current_selected = None
            if current_selected is not None and current_selected in offline_checkpoints:
                selected_checkpoint_path = str(current_selected)
            else:
                preferred_checkpoint = None
                for path in offline_checkpoints:
                    if path.name.lower() == "scaled_offline.pth":
                        preferred_checkpoint = path
                        break
                if preferred_checkpoint is None:
                    for path in offline_checkpoints:
                        if path.name.lower() == "baseline_offline.pth":
                            preferred_checkpoint = path
                            break
                if preferred_checkpoint is None and offline_checkpoints:
                    preferred_checkpoint = offline_checkpoints[0]
                if preferred_checkpoint is not None:
                    selected_checkpoint_path = str(preferred_checkpoint.resolve())

            selected_checkpoint = (
                selected_checkpoint_path
            )
            if selected_checkpoint:
                idx = self.cotracker_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.cotracker_checkpoint_combo.setCurrentIndex(idx)
                    self.cotracker_settings.checkpoint_path = selected_checkpoint
            if not self.cotracker_settings.checkpoint_path and self.cotracker_checkpoint_combo.count() > 0:
                self.cotracker_checkpoint_combo.setCurrentIndex(0)
                self.cotracker_settings.checkpoint_path = self.cotracker_checkpoint_combo.currentData()

        self.cotracker_checkpoint_combo.blockSignals(False)
        self.cotracker_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.cotracker_settings.repo_path or "")
        )
        self.cotracker_settings.repo_path = str(status.repo_path) if status.repo_path else self.cotracker_settings.repo_path

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            online_count = len(status.available_checkpoints) - len(offline_checkpoints)
            if online_count > 0:
                details.append(f"{online_count} online hidden")
            self.cotracker_warning_label.setStyleSheet("color: #5cb85c;")
            self.cotracker_warning_label.setText(
                "CoTracker3 ready. " + ", ".join(details)
            )
            self.cotracker_warning_label.setToolTip(format_cotracker_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            self.cotracker_warning_label.setStyleSheet("color: #d9534f;")
            self.cotracker_warning_label.setText("\n".join(summary))
            self.cotracker_warning_label.setToolTip(format_cotracker_environment_report(status))

    def _on_cotracker_checkpoint_selected(self, index: int):
        checkpoint_path = self.cotracker_checkpoint_combo.itemData(index)
        self.cotracker_settings.checkpoint_path = checkpoint_path
        self._refresh_cotracker_checkpoints()

    def _refresh_tapnext_checkpoints(self):
        repo_path = self.tapnext_repo_path_edit.text().strip() or self.tapnext_settings.repo_path
        if repo_path:
            self.tapnext_settings.repo_path = repo_path
        status = probe_tapnext_environment(
            models_dir=self.tapnext_models_dir,
            checkpoint_path=self.tapnext_settings.checkpoint_path,
            repo_path=self.tapnext_settings.repo_path,
        )

        self.tapnext_checkpoint_combo.blockSignals(True)
        self.tapnext_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.tapnext_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.tapnext_checkpoint_combo.setEnabled(False)
            self._tapnext_checkpoint_available = False
            self.tapnext_settings.checkpoint_path = None
        else:
            self.tapnext_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.tapnext_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._tapnext_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.tapnext_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.tapnext_checkpoint_combo.setCurrentIndex(idx)
                    self.tapnext_settings.checkpoint_path = selected_checkpoint
            if not self.tapnext_settings.checkpoint_path and self.tapnext_checkpoint_combo.count() > 0:
                self.tapnext_checkpoint_combo.setCurrentIndex(0)
                self.tapnext_settings.checkpoint_path = self.tapnext_checkpoint_combo.currentData()

        self.tapnext_checkpoint_combo.blockSignals(False)
        self.tapnext_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.tapnext_settings.repo_path or "")
        )
        self.tapnext_settings.repo_path = str(status.repo_path) if status.repo_path else self.tapnext_settings.repo_path

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            self.tapnext_warning_label.setStyleSheet("color: #5cb85c;")
            self.tapnext_warning_label.setText(
                "TAPNext ready. " + ", ".join(details)
            )
            self.tapnext_warning_label.setToolTip(format_tapnext_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            self.tapnext_warning_label.setStyleSheet("color: #d9534f;")
            self.tapnext_warning_label.setText("\n".join(summary))
            self.tapnext_warning_label.setToolTip(format_tapnext_environment_report(status))

    def _on_tapnext_checkpoint_selected(self, index: int):
        checkpoint_path = self.tapnext_checkpoint_combo.itemData(index)
        self.tapnext_settings.checkpoint_path = checkpoint_path
        self._refresh_tapnext_checkpoints()

    def _refresh_trackonr_checkpoints(self):
        repo_path = self.trackonr_repo_path_edit.text().strip() or self.trackonr_settings.repo_path
        if repo_path:
            self.trackonr_settings.repo_path = repo_path
        self.trackonr_settings.python_executable = self.trackonr_python_executable_edit.text().strip() or None
        self.trackonr_settings.backend_mode = self.trackonr_backend_mode_combo.currentData()
        status = probe_trackonr_environment(
            models_dir=self.trackonr_models_dir,
            checkpoint_path=self.trackonr_settings.checkpoint_path,
            repo_path=self.trackonr_settings.repo_path,
        )

        self.trackonr_checkpoint_combo.blockSignals(True)
        self.trackonr_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.trackonr_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.trackonr_checkpoint_combo.setEnabled(False)
            self._trackonr_checkpoint_available = False
            self.trackonr_settings.checkpoint_path = None
        else:
            self.trackonr_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.trackonr_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._trackonr_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.trackonr_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.trackonr_checkpoint_combo.setCurrentIndex(idx)
                    self.trackonr_settings.checkpoint_path = selected_checkpoint
            if not self.trackonr_settings.checkpoint_path and self.trackonr_checkpoint_combo.count() > 0:
                self.trackonr_checkpoint_combo.setCurrentIndex(0)
                self.trackonr_settings.checkpoint_path = self.trackonr_checkpoint_combo.currentData()

        self.trackonr_checkpoint_combo.blockSignals(False)
        self.trackonr_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.trackonr_settings.repo_path or "")
        )
        self.trackonr_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.trackonr_settings.repo_path
        )

        python_executable = self.trackonr_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            if status.mmcv_version:
                details.append(f"mmcv {status.mmcv_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            details.append("DINO cached" if status.dinov3_cache_present else "DINO on-demand")
            self.trackonr_warning_label.setStyleSheet("color: #5cb85c;")
            self.trackonr_warning_label.setText(
                "Track-On-R ready. " + ", ".join(details)
            )
            self.trackonr_warning_label.setToolTip(format_trackonr_environment_report(status))
        elif (
            self.trackonr_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.trackonr_warning_label.setStyleSheet("color: #f0ad4e;")
            self.trackonr_warning_label.setText(
                "Track-On-R subprocess configured. Current app env is not Track-On-R-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.trackonr_warning_label.setToolTip(format_trackonr_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.trackonr_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated trackonr env, or run Trackleed from that env."
                )
            self.trackonr_warning_label.setStyleSheet("color: #d9534f;")
            self.trackonr_warning_label.setText("\n".join(summary))
            self.trackonr_warning_label.setToolTip(format_trackonr_environment_report(status))

    def _on_trackonr_checkpoint_selected(self, index: int):
        checkpoint_path = self.trackonr_checkpoint_combo.itemData(index)
        self.trackonr_settings.checkpoint_path = checkpoint_path
        self._refresh_trackonr_checkpoints()

    def _refresh_pipspp_checkpoints(self):
        repo_path = self.pipspp_repo_path_edit.text().strip() or self.pipspp_settings.repo_path
        if repo_path:
            self.pipspp_settings.repo_path = repo_path
        self.pipspp_settings.python_executable = self.pipspp_python_executable_edit.text().strip() or None
        self.pipspp_settings.backend_mode = self.pipspp_backend_mode_combo.currentData()
        status = probe_pipspp_environment(
            models_dir=self.pipspp_models_dir,
            checkpoint_path=self.pipspp_settings.checkpoint_path,
            repo_path=self.pipspp_settings.repo_path,
        )

        self.pipspp_checkpoint_combo.blockSignals(True)
        self.pipspp_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.pipspp_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.pipspp_checkpoint_combo.setEnabled(False)
            self._pipspp_checkpoint_available = False
            self.pipspp_settings.checkpoint_path = None
        else:
            self.pipspp_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.pipspp_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._pipspp_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.pipspp_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.pipspp_checkpoint_combo.setCurrentIndex(idx)
                    self.pipspp_settings.checkpoint_path = selected_checkpoint
            if not self.pipspp_settings.checkpoint_path and self.pipspp_checkpoint_combo.count() > 0:
                self.pipspp_checkpoint_combo.setCurrentIndex(0)
                self.pipspp_settings.checkpoint_path = self.pipspp_checkpoint_combo.currentData()

        self.pipspp_checkpoint_combo.blockSignals(False)
        self.pipspp_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.pipspp_settings.repo_path or "")
        )
        self.pipspp_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.pipspp_settings.repo_path
        )

        python_executable = self.pipspp_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.selected_checkpoint_file is not None
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            self.pipspp_warning_label.setStyleSheet("color: #5cb85c;")
            self.pipspp_warning_label.setText(
                "PIPs++ ready. " + ", ".join(details)
            )
            self.pipspp_warning_label.setToolTip(format_pipspp_environment_report(status))
        elif (
            self.pipspp_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.pipspp_warning_label.setStyleSheet("color: #f0ad4e;")
            self.pipspp_warning_label.setText(
                "PIPs++ subprocess configured. Current app env is not PIPs++-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.pipspp_warning_label.setToolTip(format_pipspp_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.pipspp_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated pipspp env, or run Trackleed from that env."
                )
            self.pipspp_warning_label.setStyleSheet("color: #d9534f;")
            self.pipspp_warning_label.setText("\n".join(summary))
            self.pipspp_warning_label.setToolTip(format_pipspp_environment_report(status))

    def _on_pipspp_checkpoint_selected(self, index: int):
        checkpoint_path = self.pipspp_checkpoint_combo.itemData(index)
        self.pipspp_settings.checkpoint_path = checkpoint_path
        self._refresh_pipspp_checkpoints()

    def _refresh_mftiq_checkpoints(self):
        repo_path = self.mftiq_repo_path_edit.text().strip() or self.mftiq_settings.repo_path
        if repo_path:
            self.mftiq_settings.repo_path = repo_path
        self.mftiq_settings.python_executable = self.mftiq_python_executable_edit.text().strip() or None
        self.mftiq_settings.backend_mode = self.mftiq_backend_mode_combo.currentData()

        status = probe_mftiq_environment(
            models_dir=self.mftiq_models_dir,
            checkpoint_path=self.mftiq_settings.checkpoint_path,
            repo_path=self.mftiq_settings.repo_path,
            config_path=self.mftiq_settings.config_path,
        )

        self.mftiq_checkpoint_combo.blockSignals(True)
        self.mftiq_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.mftiq_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.mftiq_checkpoint_combo.setEnabled(False)
            self._mftiq_checkpoint_available = False
            self.mftiq_settings.checkpoint_path = None
        else:
            self.mftiq_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.mftiq_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._mftiq_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.mftiq_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.mftiq_checkpoint_combo.setCurrentIndex(idx)
                    self.mftiq_settings.checkpoint_path = selected_checkpoint
            if not self.mftiq_settings.checkpoint_path and self.mftiq_checkpoint_combo.count() > 0:
                self.mftiq_checkpoint_combo.setCurrentIndex(0)
                self.mftiq_settings.checkpoint_path = self.mftiq_checkpoint_combo.currentData()

        self.mftiq_checkpoint_combo.blockSignals(False)
        self.mftiq_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.mftiq_settings.repo_path or "")
        )
        self.mftiq_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.mftiq_settings.repo_path
        )

        python_executable = self.mftiq_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.config_exists
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            if status.xformers_version:
                details.append(f"xformers {status.xformers_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            self.mftiq_warning_label.setStyleSheet("color: #5cb85c;")
            self.mftiq_warning_label.setText("MFTIQ ready. " + ", ".join(details))
            self.mftiq_warning_label.setToolTip(format_mftiq_environment_report(status))
        elif (
            self.mftiq_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.mftiq_warning_label.setStyleSheet("color: #f0ad4e;")
            self.mftiq_warning_label.setText(
                "MFTIQ subprocess configured. Current app env is not MFTIQ-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.mftiq_warning_label.setToolTip(format_mftiq_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.mftiq_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated mftiq env, or run Trackleed from that env."
                )
            self.mftiq_warning_label.setStyleSheet("color: #d9534f;")
            self.mftiq_warning_label.setText("\n".join(summary))
            self.mftiq_warning_label.setToolTip(format_mftiq_environment_report(status))

    def _on_mftiq_checkpoint_selected(self, index: int):
        checkpoint_path = self.mftiq_checkpoint_combo.itemData(index)
        self.mftiq_settings.checkpoint_path = checkpoint_path
        self._refresh_mftiq_checkpoints()

    def _refresh_cutie_checkpoints(self):
        repo_path = self.cutie_repo_path_edit.text().strip() or self.cutie_settings.repo_path
        if repo_path:
            self.cutie_settings.repo_path = repo_path
        self.cutie_settings.python_executable = self.cutie_python_executable_edit.text().strip() or None
        self.cutie_settings.backend_mode = self.cutie_backend_mode_combo.currentData()

        status = probe_cutie_environment(
            models_dir=self.cutie_models_dir,
            checkpoint_path=self.cutie_settings.checkpoint_path,
            repo_path=self.cutie_settings.repo_path,
        )

        self.cutie_checkpoint_combo.blockSignals(True)
        self.cutie_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.cutie_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.cutie_checkpoint_combo.setEnabled(False)
            self._cutie_checkpoint_available = False
            self.cutie_settings.checkpoint_path = None
        else:
            self.cutie_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.cutie_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._cutie_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.cutie_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.cutie_checkpoint_combo.setCurrentIndex(idx)
                    self.cutie_settings.checkpoint_path = selected_checkpoint
            if not self.cutie_settings.checkpoint_path and self.cutie_checkpoint_combo.count() > 0:
                self.cutie_checkpoint_combo.setCurrentIndex(0)
                self.cutie_settings.checkpoint_path = self.cutie_checkpoint_combo.currentData()

        self.cutie_checkpoint_combo.blockSignals(False)
        self.cutie_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.cutie_settings.repo_path or "")
        )
        self.cutie_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.cutie_settings.repo_path
        )

        python_executable = self.cutie_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            if status.selected_checkpoint is not None:
                details.append(status.selected_checkpoint.name)
            self.cutie_warning_label.setStyleSheet("color: #5cb85c;")
            self.cutie_warning_label.setText("Cutie ready. " + ", ".join(details))
            self.cutie_warning_label.setToolTip(format_cutie_environment_report(status))
        elif (
            self.cutie_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.cutie_warning_label.setStyleSheet("color: #f0ad4e;")
            self.cutie_warning_label.setText(
                "Cutie subprocess configured. Current app env is not Cutie-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.cutie_warning_label.setToolTip(format_cutie_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.cutie_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated cutie env, or run Trackleed from that env."
                )
            if not summary:
                summary.append("Use the dedicated cutie env or the subprocess backend.")
            self.cutie_warning_label.setStyleSheet("color: #d9534f;")
            self.cutie_warning_label.setText("\n".join(summary))
            self.cutie_warning_label.setToolTip(format_cutie_environment_report(status))

    def _on_cutie_checkpoint_selected(self, index: int):
        checkpoint_path = self.cutie_checkpoint_combo.itemData(index)
        self.cutie_settings.checkpoint_path = checkpoint_path
        self._refresh_cutie_checkpoints()

    def _refresh_xmem2_checkpoints(self):
        repo_path = self.xmem2_repo_path_edit.text().strip() or self.xmem2_settings.repo_path
        if repo_path:
            self.xmem2_settings.repo_path = repo_path
        self.xmem2_settings.python_executable = self.xmem2_python_executable_edit.text().strip() or None
        self.xmem2_settings.backend_mode = self.xmem2_backend_mode_combo.currentData()

        status = probe_xmem2_environment(
            models_dir=self.xmem2_models_dir,
            checkpoint_path=self.xmem2_settings.checkpoint_path,
            repo_path=self.xmem2_settings.repo_path,
        )

        self.xmem2_checkpoint_combo.blockSignals(True)
        self.xmem2_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.xmem2_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.xmem2_checkpoint_combo.setEnabled(False)
            self._xmem2_checkpoint_available = False
            self.xmem2_settings.checkpoint_path = None
        else:
            self.xmem2_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.xmem2_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._xmem2_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.xmem2_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.xmem2_checkpoint_combo.setCurrentIndex(idx)
                    self.xmem2_settings.checkpoint_path = selected_checkpoint
            if not self.xmem2_settings.checkpoint_path and self.xmem2_checkpoint_combo.count() > 0:
                self.xmem2_checkpoint_combo.setCurrentIndex(0)
                self.xmem2_settings.checkpoint_path = self.xmem2_checkpoint_combo.currentData()

        self.xmem2_checkpoint_combo.blockSignals(False)
        self.xmem2_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.xmem2_settings.repo_path or "")
        )
        self.xmem2_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.xmem2_settings.repo_path
        )

        python_executable = self.xmem2_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            if status.selected_checkpoint is not None:
                details.append(status.selected_checkpoint.name)
            self.xmem2_warning_label.setStyleSheet("color: #5cb85c;")
            self.xmem2_warning_label.setText("XMem2 ready. " + ", ".join(details))
            self.xmem2_warning_label.setToolTip(format_xmem2_environment_report(status))
        elif (
            self.xmem2_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.xmem2_warning_label.setStyleSheet("color: #f0ad4e;")
            self.xmem2_warning_label.setText(
                "XMem2 subprocess configured. Current app env is not XMem2-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.xmem2_warning_label.setToolTip(format_xmem2_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.xmem2_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated xmem2 env, or run Trackleed from that env."
                )
            if not summary:
                summary.append("Configure the dedicated xmem2 env or run Trackleed from that env.")
            self.xmem2_warning_label.setStyleSheet("color: #d9534f;")
            self.xmem2_warning_label.setText("\n".join(summary))
            self.xmem2_warning_label.setToolTip(format_xmem2_environment_report(status))

    def _on_xmem2_checkpoint_selected(self, index: int):
        checkpoint_path = self.xmem2_checkpoint_combo.itemData(index)
        self.xmem2_settings.checkpoint_path = checkpoint_path
        self._refresh_xmem2_checkpoints()

    def _refresh_sam2_checkpoints(self):
        repo_path = self.sam2_repo_path_edit.text().strip() or self.sam2_settings.repo_path
        if repo_path:
            self.sam2_settings.repo_path = repo_path
        self.sam2_settings.python_executable = self.sam2_python_executable_edit.text().strip() or None
        self.sam2_settings.backend_mode = self.sam2_backend_mode_combo.currentData()

        status = probe_sam2_environment(
            models_dir=self.sam2_models_dir,
            checkpoint_path=self.sam2_settings.checkpoint_path,
            repo_path=self.sam2_settings.repo_path,
            config_path=None,
        )

        self.sam2_checkpoint_combo.blockSignals(True)
        self.sam2_checkpoint_combo.clear()

        if not status.available_checkpoints:
            self.sam2_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.sam2_checkpoint_combo.setEnabled(False)
            self._sam2_checkpoint_available = False
            self.sam2_settings.checkpoint_path = None
        else:
            self.sam2_checkpoint_combo.setEnabled(True)
            for path in status.available_checkpoints:
                self.sam2_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._sam2_checkpoint_available = True
            selected_checkpoint = (
                str(status.selected_checkpoint.resolve())
                if status.selected_checkpoint is not None and status.selected_checkpoint.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.sam2_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.sam2_checkpoint_combo.setCurrentIndex(idx)
                    self.sam2_settings.checkpoint_path = selected_checkpoint
            if not self.sam2_settings.checkpoint_path and self.sam2_checkpoint_combo.count() > 0:
                self.sam2_checkpoint_combo.setCurrentIndex(0)
                self.sam2_settings.checkpoint_path = self.sam2_checkpoint_combo.currentData()

        self.sam2_checkpoint_combo.blockSignals(False)
        self.sam2_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.sam2_settings.repo_path or "")
        )
        self.sam2_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.sam2_settings.repo_path
        )
        self.sam2_settings.config_path = (
            str(status.config_path.resolve())
            if status.config_path is not None and status.config_path.exists()
            else None
        )
        self.sam2_config_path_edit.setText(self.sam2_settings.config_path or "")

        python_executable = self.sam2_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and status.selected_checkpoint is not None
            and status.checkpoint_exists
            and status.config_exists
            and status.config_identifier is not None
        )

        if status.is_ready:
            details = []
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            details.append("video predictor")
            self.sam2_warning_label.setStyleSheet("color: #5cb85c;")
            self.sam2_warning_label.setText("SAM 2 ready. " + ", ".join(details))
            self.sam2_warning_label.setToolTip(format_sam2_environment_report(status))
        elif (
            self.sam2_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.sam2_warning_label.setStyleSheet("color: #f0ad4e;")
            self.sam2_warning_label.setText(
                "SAM 2 subprocess configured. Current app env is not SAM 2-ready, "
                "so Trackleed will use the external Python executable."
            )
            self.sam2_warning_label.setToolTip(format_sam2_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.sam2_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not status.is_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated sam2 env, or run Trackleed from that env."
                )
            if not summary:
                summary.append("Configure the dedicated sam2 env or run Trackleed from that env.")
            self.sam2_warning_label.setStyleSheet("color: #d9534f;")
            self.sam2_warning_label.setText("\n".join(summary))
            self.sam2_warning_label.setToolTip(format_sam2_environment_report(status))

    def _on_sam2_checkpoint_selected(self, index: int):
        checkpoint_path = self.sam2_checkpoint_combo.itemData(index)
        self.sam2_settings.checkpoint_path = checkpoint_path
        self._refresh_sam2_checkpoints()

    def _on_tapir_center_mode_changed(self, idx: int):
        manual = self.tapir_center_mode_combo.itemData(idx) == "manual"
        self.tapir_center_x_spinbox.setEnabled(manual)
        self.tapir_center_y_spinbox.setEnabled(manual)

    def _collect_model_paths(self, *directories: Path) -> list[Path]:
        resolved_paths: dict[str, Path] = {}
        for directory in directories:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.pt")):
                resolved_paths[str(path.resolve())] = path.resolve()
        return sorted(
            resolved_paths.values(),
            key=lambda path: (path.name.lower(), str(path).lower()),
        )

    def _on_spot_backend_changed(self, backend: str):
        self.current_spot_backend = backend
        index = 0 if backend == "Classic" else 1 if backend == "Deep Learning" else 2
        self.spot_backend_stack.setCurrentIndex(index)
        self._update_find_spots_button_state()

    def _on_dl_model_selected(self, index: int):
        model_path = self.dl_model_combo.itemData(index)
        self.dl_settings.model_path = model_path
        self._update_find_spots_button_state()

    def _on_yolo_model_selected(self, index: int):
        model_path = self.yolo_model_combo.itemData(index)
        self.yolo_settings.model_path = model_path
        self._update_find_spots_button_state()

    def _update_find_spots_button_state(self):
        if not hasattr(self, "find_spots_button"):
            return
        buttons = [self.find_spots_button, getattr(self, "find_all_spots_button", None)]
        if self.current_spot_backend == "Deep Learning":
            enabled = self._dl_models_available and bool(self.dl_settings.model_path)
            for btn in buttons:
                if btn:
                    btn.setEnabled(enabled)
                    if not enabled:
                        btn.setToolTip(
                            "Add a .pt model to tracking_models/ "
                            "(legacy fallback: segmentation_models/) and click Refresh, then pick it from the list."
                        )
                    else:
                        btn.setToolTip("")
        elif self.current_spot_backend == "YOLO":
            enabled = self._yolo_models_available and bool(self.yolo_settings.model_path)
            for btn in buttons:
                if btn:
                    btn.setEnabled(enabled)
                    if not enabled:
                        btn.setToolTip(
                            "Add a YOLO .pt checkpoint to tracking_models/yolo "
                            "(legacy fallback: segmentation_models/) and click Refresh."
                        )
                    else:
                        btn.setToolTip("")
        else:
            for btn in buttons:
                if btn:
                    btn.setEnabled(True)
                    btn.setToolTip("")

    def _tapir_runtime_mode(self, scenario: str | None = None) -> str:
        if scenario is None:
            scenario = self.tracking_scenario_combo.currentText()
        return "online" if scenario == "BootsTAPIR Online / Causal" else "offline"

    def _on_tracking_scenario_changed(self, scenario: str):
        self._update_tracking_params_visibility(scenario)
        if hasattr(self, "tapir_group"):
            self.tapir_group.setTitle(
                "BootsTAPIR Online / Causal"
                if self._tapir_runtime_mode(scenario) == "online"
                else "Offline TAPIR"
            )
        if scenario in {"Offline TAPIR", "BootsTAPIR Online / Causal"}:
            self._refresh_tapir_checkpoints()
        if scenario == "Offline PIPs++":
            self._refresh_pipspp_checkpoints()
        if scenario == "Offline MFTIQ":
            self._refresh_mftiq_checkpoints()
        if scenario == "Offline ToMP":
            self._refresh_tomp_checkpoints()
        if scenario == "Offline Cutie":
            self._refresh_cutie_checkpoints()
        if scenario == "Offline XMem2":
            self._refresh_xmem2_checkpoints()
        if scenario == "Offline SAM2":
            self._refresh_sam2_checkpoints()
        current_item = (
            self.file_list_widget.currentItem()
            if hasattr(self, "file_list_widget")
            else None
        )
        current_spot_item = (
            self.spot_list_widget.currentItem()
            if hasattr(self, "spot_list_widget")
            else None
        )
        selected_spot_id = None
        if current_spot_item is not None:
            selected_spot = current_spot_item.data(Qt.ItemDataRole.UserRole)
            if selected_spot is not None:
                selected_spot_id = int(getattr(selected_spot, "id", -1))
        if current_item is not None:
            current_file = current_item.data(Qt.ItemDataRole.UserRole)
            if current_file is not None:
                self.populate_spot_list(current_file.spots)
                if selected_spot_id is not None:
                    self._reselect_spot_in_list(selected_spot_id)
        self._update_spot_polygon_buttons()

    def _update_tracking_params_visibility(self, scenario: str | None = None):
        """Shows parameter groups only for the currently selected tracking algorithm."""
        if scenario is None:
            scenario = self.tracking_scenario_combo.currentText()

        show_optical_basic = scenario.startswith("Optical Flow")
        show_optical_skip = scenario == "Optical Flow (skip, anchor)"
        show_tapir = scenario in {"Offline TAPIR", "BootsTAPIR Online / Causal"}
        show_locotrack = scenario == "Offline LocoTrack"
        show_cotracker = scenario == "Offline CoTracker3"
        show_tapnext = scenario == "Offline TAPNext"
        show_trackonr = scenario == "Offline Track-On-R"
        show_pipspp = scenario == "Offline PIPs++"
        show_mftiq = scenario == "Offline MFTIQ"
        show_tomp = scenario == "Offline ToMP"
        show_cutie = scenario == "Offline Cutie"
        show_xmem2 = scenario == "Offline XMem2"
        show_sam2 = scenario == "Offline SAM2"
        show_hungarian = scenario == "Detection Linking (Hungarian)"
        show_dir = scenario == "Detection Linking (Directional)"

        if hasattr(self, "optical_basic_group"):
            self.optical_basic_group.setVisible(show_optical_basic)
        if hasattr(self, "optical_skip_group"):
            self.optical_skip_group.setVisible(show_optical_skip)
        if hasattr(self, "tapir_group"):
            self.tapir_group.setVisible(show_tapir)
        if hasattr(self, "locotrack_group"):
            self.locotrack_group.setVisible(show_locotrack)
        if hasattr(self, "cotracker_group"):
            self.cotracker_group.setVisible(show_cotracker)
        if hasattr(self, "tapnext_group"):
            self.tapnext_group.setVisible(show_tapnext)
        if hasattr(self, "trackonr_group"):
            self.trackonr_group.setVisible(show_trackonr)
        if hasattr(self, "pipspp_group"):
            self.pipspp_group.setVisible(show_pipspp)
        if hasattr(self, "mftiq_group"):
            self.mftiq_group.setVisible(show_mftiq)
        if hasattr(self, "tomp_group"):
            self.tomp_group.setVisible(show_tomp)
        if hasattr(self, "cutie_group"):
            self.cutie_group.setVisible(show_cutie)
        if hasattr(self, "xmem2_group"):
            self.xmem2_group.setVisible(show_xmem2)
        if hasattr(self, "sam2_group"):
            self.sam2_group.setVisible(show_sam2)
        if hasattr(self, "hungarian_group"):
            self.hungarian_group.setVisible(show_hungarian)
        if hasattr(self, "dir_group"):
            self.dir_group.setVisible(show_dir)
        if hasattr(self, "polygon_seed_group"):
            self.polygon_seed_group.setVisible(show_cutie or show_xmem2)

    def _on_dir_center_mode_changed(self, idx: int):
        manual = idx == 1
        self.dir_center_x_spinbox.setEnabled(manual)
        self.dir_center_y_spinbox.setEnabled(manual)

    def _on_locotrack_center_mode_changed(self, idx: int):
        manual = self.locotrack_center_mode_combo.itemData(idx) == "manual"
        self.locotrack_center_x_spinbox.setEnabled(manual)
        self.locotrack_center_y_spinbox.setEnabled(manual)

    def _on_cotracker_center_mode_changed(self, idx: int):
        manual = self.cotracker_center_mode_combo.itemData(idx) == "manual"
        self.cotracker_center_x_spinbox.setEnabled(manual)
        self.cotracker_center_y_spinbox.setEnabled(manual)

    def _on_tapnext_center_mode_changed(self, idx: int):
        manual = self.tapnext_center_mode_combo.itemData(idx) == "manual"
        self.tapnext_center_x_spinbox.setEnabled(manual)
        self.tapnext_center_y_spinbox.setEnabled(manual)

    def _on_trackonr_center_mode_changed(self, idx: int):
        manual = self.trackonr_center_mode_combo.itemData(idx) == "manual"
        self.trackonr_center_x_spinbox.setEnabled(manual)
        self.trackonr_center_y_spinbox.setEnabled(manual)

    def _on_pipspp_center_mode_changed(self, idx: int):
        manual = self.pipspp_center_mode_combo.itemData(idx) == "manual"
        self.pipspp_center_x_spinbox.setEnabled(manual)
        self.pipspp_center_y_spinbox.setEnabled(manual)

    def _on_locotrack_backend_mode_changed(self, idx: int):
        backend_mode = self.locotrack_backend_mode_combo.itemData(idx)
        self.locotrack_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "locotrack_warning_label"):
            self._refresh_locotrack_checkpoints()

    def _on_trackonr_backend_mode_changed(self, idx: int):
        backend_mode = self.trackonr_backend_mode_combo.itemData(idx)
        self.trackonr_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "trackonr_warning_label"):
            self._refresh_trackonr_checkpoints()

    def _on_pipspp_backend_mode_changed(self, idx: int):
        backend_mode = self.pipspp_backend_mode_combo.itemData(idx)
        self.pipspp_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "pipspp_warning_label"):
            self._refresh_pipspp_checkpoints()

    def _on_mftiq_backend_mode_changed(self, idx: int):
        backend_mode = self.mftiq_backend_mode_combo.itemData(idx)
        self.mftiq_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "mftiq_warning_label"):
            self._refresh_mftiq_checkpoints()

    def _on_cutie_backend_mode_changed(self, idx: int):
        backend_mode = self.cutie_backend_mode_combo.itemData(idx)
        self.cutie_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "cutie_warning_label"):
            self._refresh_cutie_checkpoints()

    def _on_xmem2_backend_mode_changed(self, idx: int):
        backend_mode = self.xmem2_backend_mode_combo.itemData(idx)
        self.xmem2_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "xmem2_warning_label"):
            self._refresh_xmem2_checkpoints()

    def _on_tomp_backend_mode_changed(self, idx: int):
        backend_mode = self.tomp_backend_mode_combo.itemData(idx)
        self.tomp_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "tomp_warning_label"):
            self._refresh_tomp_checkpoints()

    def _on_tomp_center_mode_changed(self, idx: int):
        manual = self.tomp_center_mode_combo.itemData(idx) == "manual"
        self.tomp_center_x_spinbox.setEnabled(manual)
        self.tomp_center_y_spinbox.setEnabled(manual)

    def _on_xmem2_center_mode_changed(self, idx: int):
        manual = self.xmem2_center_mode_combo.itemData(idx) == "manual"
        self.xmem2_center_x_spinbox.setEnabled(manual)
        self.xmem2_center_y_spinbox.setEnabled(manual)

    def _on_cutie_center_mode_changed(self, idx: int):
        manual = self.cutie_center_mode_combo.itemData(idx) == "manual"
        self.cutie_center_x_spinbox.setEnabled(manual)
        self.cutie_center_y_spinbox.setEnabled(manual)

    def _current_tomp_variant_key(self) -> str:
        return str(self.tomp_model_variant_combo.currentData() or self.tomp_settings.model_variant or "tomp50")

    def _current_tomp_variant_label(self) -> str:
        variant_key = self._current_tomp_variant_key()
        if variant_key == "tomp101":
            return "ToMP-101"
        if variant_key == "tamos_resnet50":
            return "TaMOs-resnet50"
        if variant_key == "tamos_swin_base":
            return "TaMOs-swin-base"
        return "ToMP-50"

    def _refresh_tomp_checkpoints(self):
        repo_path = self.tomp_repo_path_edit.text().strip() or self.tomp_settings.repo_path
        if repo_path:
            self.tomp_settings.repo_path = repo_path
        self.tomp_settings.python_executable = self.tomp_python_executable_edit.text().strip() or None
        self.tomp_settings.backend_mode = self.tomp_backend_mode_combo.currentData()
        self.tomp_settings.model_variant = self._current_tomp_variant_key()

        current_variant = self.tomp_settings.model_variant
        status = probe_tomp_environment(
            tomp_models_dir=self.tomp_models_dir,
            tamos_models_dir=self.tamos_models_dir,
            tomp_checkpoint_path=(
                self.tomp_settings.checkpoint_path
                if current_variant in {"tomp50", "tomp101"}
                else None
            ),
            tamos_checkpoint_path=(
                self.tomp_settings.checkpoint_path
                if current_variant in {"tamos_resnet50", "tamos_swin_base"}
                else None
            ),
            repo_path=self.tomp_settings.repo_path,
        )

        expected_checkpoint_name = expected_tomp_checkpoint_name_for_variant(current_variant)
        if current_variant in {"tamos_resnet50", "tamos_swin_base"}:
            family_checkpoints = status.available_tamos_checkpoints
            selected_checkpoint_obj = select_default_tamos_checkpoint_for_variant(
                family_checkpoints,
                current_variant,
            )
            available_checkpoints = [
                path for path in family_checkpoints if path.name.lower() == expected_checkpoint_name.lower()
            ]
            variant_ready = bool(status.tamos_ready and selected_checkpoint_obj and selected_checkpoint_obj.exists())
            if current_variant == "tamos_swin_base":
                hint_text = (
                    "Hint: use tamos_swin_base.pth.tar. This is the stronger TaMOs-swin-base checkpoint from PyTracking."
                )
            else:
                hint_text = (
                    "Hint: use tamos_resnet50.pth.tar. This is the TaMOs-resnet50 checkpoint from PyTracking."
                )
        else:
            family_checkpoints = status.available_tomp_checkpoints
            selected_checkpoint_obj = select_default_tomp_checkpoint_for_variant(
                family_checkpoints,
                current_variant,
            )
            available_checkpoints = [
                path for path in family_checkpoints if path.name.lower() == expected_checkpoint_name.lower()
            ]
            variant_ready = bool(status.tomp_ready and selected_checkpoint_obj and selected_checkpoint_obj.exists())
            if current_variant == "tomp101":
                hint_text = (
                    "Hint: use tomp101.pth.tar. This is the stronger ToMP-101 checkpoint from PyTracking."
                )
            else:
                hint_text = (
                    "Hint: use tomp50.pth.tar. This is the recommended ToMP-50 checkpoint for the first Trackleed integration."
                )
        self.tomp_checkpoint_hint_label.setText(hint_text)

        self.tomp_checkpoint_combo.blockSignals(True)
        self.tomp_checkpoint_combo.clear()

        if not available_checkpoints:
            self.tomp_checkpoint_combo.addItem("No checkpoints found", userData=None)
            self.tomp_checkpoint_combo.setEnabled(False)
            self._tomp_checkpoint_available = False
            self.tomp_settings.checkpoint_path = None
        else:
            self.tomp_checkpoint_combo.setEnabled(True)
            for path in available_checkpoints:
                self.tomp_checkpoint_combo.addItem(path.name, userData=str(path.resolve()))
            self._tomp_checkpoint_available = True
            selected_checkpoint = (
                str(selected_checkpoint_obj.resolve())
                if selected_checkpoint_obj is not None and selected_checkpoint_obj.exists()
                else None
            )
            if selected_checkpoint:
                idx = self.tomp_checkpoint_combo.findData(selected_checkpoint)
                if idx >= 0:
                    self.tomp_checkpoint_combo.setCurrentIndex(idx)
                    self.tomp_settings.checkpoint_path = selected_checkpoint
            if not self.tomp_settings.checkpoint_path and self.tomp_checkpoint_combo.count() > 0:
                self.tomp_checkpoint_combo.setCurrentIndex(0)
                self.tomp_settings.checkpoint_path = self.tomp_checkpoint_combo.currentData()

        self.tomp_checkpoint_combo.blockSignals(False)
        self.tomp_repo_path_edit.setText(
            str(status.repo_path) if status.repo_path else (self.tomp_settings.repo_path or "")
        )
        self.tomp_settings.repo_path = (
            str(status.repo_path) if status.repo_path else self.tomp_settings.repo_path
        )

        python_executable = self.tomp_settings.python_executable
        python_path_ok = bool(
            python_executable and Path(python_executable).expanduser().exists()
        )
        structural_ready = bool(
            status.repo_path is not None
            and selected_checkpoint_obj is not None
            and selected_checkpoint_obj.exists()
        )

        if variant_ready:
            details = [self._current_tomp_variant_label()]
            if status.torch_version:
                details.append(f"torch {status.torch_version}")
            details.append("CUDA" if status.cuda_available else "CPU")
            if selected_checkpoint_obj is not None:
                details.append(selected_checkpoint_obj.name)
            self.tomp_warning_label.setStyleSheet("color: #5cb85c;")
            self.tomp_warning_label.setText(f"{self._current_tomp_variant_label()} ready. " + ", ".join(details[1:]))
            self.tomp_warning_label.setToolTip(format_tomp_environment_report(status))
        elif (
            self.tomp_settings.backend_mode == "subprocess"
            and structural_ready
            and python_path_ok
        ):
            self.tomp_warning_label.setStyleSheet("color: #f0ad4e;")
            self.tomp_warning_label.setText(
                f"{self._current_tomp_variant_label()} subprocess configured. "
                "Current app env is not PyTracking-ready, so Trackleed will use the external Python executable."
            )
            self.tomp_warning_label.setToolTip(format_tomp_environment_report(status))
        else:
            summary = []
            if status.errors:
                summary.append(status.errors[0])
            if status.warnings:
                summary.extend(status.warnings[:1])
            if (
                self.tomp_settings.backend_mode == "subprocess"
                and structural_ready
                and not python_path_ok
                and not variant_ready
            ):
                summary.append(
                    "Set the Python executable of the dedicated tomp env, or run Trackleed from that env."
                )
            if not summary:
                summary.append("Configure the dedicated tomp env or run Trackleed from that env.")
            self.tomp_warning_label.setStyleSheet("color: #d9534f;")
            self.tomp_warning_label.setText("\n".join(summary))
            self.tomp_warning_label.setToolTip(format_tomp_environment_report(status))

    def _on_tomp_variant_changed(self, index: int):
        self.tomp_settings.model_variant = self.tomp_model_variant_combo.itemData(index) or "tomp50"
        self.tomp_settings.checkpoint_path = None
        self._refresh_tomp_checkpoints()

    def _on_tomp_checkpoint_selected(self, index: int):
        checkpoint_path = self.tomp_checkpoint_combo.itemData(index)
        self.tomp_settings.checkpoint_path = checkpoint_path
        self._refresh_tomp_checkpoints()

    def _on_sam2_backend_mode_changed(self, idx: int):
        backend_mode = self.sam2_backend_mode_combo.itemData(idx)
        self.sam2_python_executable_edit.setEnabled(backend_mode == "subprocess")
        if hasattr(self, "sam2_warning_label"):
            self._refresh_sam2_checkpoints()

    def _on_sam2_center_mode_changed(self, idx: int):
        manual = self.sam2_center_mode_combo.itemData(idx) == "manual"
        self.sam2_center_x_spinbox.setEnabled(manual)
        self.sam2_center_y_spinbox.setEnabled(manual)

    def _on_mftiq_center_mode_changed(self, idx: int):
        manual = self.mftiq_center_mode_combo.itemData(idx) == "manual"
        self.mftiq_center_x_spinbox.setEnabled(manual)
        self.mftiq_center_y_spinbox.setEnabled(manual)

    def on_spot_selection_changed(self, current_item: QListWidgetItem | None, previous_item: QListWidgetItem | None):
        """Handles highlighting the ROI on the image when a spot is selected in the list."""
        if not current_item:
            self.image_viewer.highlight_roi(None) # Clear highlight if nothing is selected
            self._update_spot_polygon_buttons()
            return
            
        # Retrieve the stored SpotROI object
        spot: SpotROI = current_item.data(Qt.ItemDataRole.UserRole)
        if spot:
            # Tell the viewer which spot ID to highlight
            self.image_viewer.highlight_roi(spot.id)
        self._update_spot_polygon_buttons()

    def _get_selected_spot_for_polygon_edit(self) -> tuple[LeedFile | None, SpotROI | None]:
        current_file_item = self.file_list_widget.currentItem()
        current_spot_item = self.spot_list_widget.currentItem()
        if not current_file_item or not current_spot_item:
            return None, None
        leed_file: LeedFile = current_file_item.data(Qt.ItemDataRole.UserRole)
        spot: SpotROI = current_spot_item.data(Qt.ItemDataRole.UserRole)
        if leed_file is None or spot is None:
            return None, None
        return leed_file, spot

    def _polygon_seed_scenario_active(self) -> bool:
        if not hasattr(self, "tracking_scenario_combo"):
            return False
        return self.tracking_scenario_combo.currentText() in {"Offline Cutie", "Offline XMem2"}

    def _update_spot_polygon_buttons(self):
        if (
            not hasattr(self, "spot_polygon_button")
            or not hasattr(self, "clear_spot_polygon_button")
            or not hasattr(self, "polygon_seed_status_label")
        ):
            return
        polygon_workflow_active = self._polygon_seed_scenario_active()
        _, spot = self._get_selected_spot_for_polygon_edit()
        has_spot = spot is not None
        self.spot_polygon_button.setEnabled(bool(polygon_workflow_active and has_spot))
        self.clear_spot_polygon_button.setEnabled(
            bool(polygon_workflow_active and has_spot and spot.has_polygon_seed)
        )

        if not polygon_workflow_active:
            self.polygon_seed_status_label.setStyleSheet("color: #6c757d;")
            self.polygon_seed_status_label.setText(
                "Polygon seed tools are available only for Offline Cutie and Offline XMem2."
            )
            return
        if not has_spot:
            self.polygon_seed_status_label.setStyleSheet("color: #6c757d;")
            self.polygon_seed_status_label.setText(
                "Select a spot to create or edit its polygon seed."
            )
            return
        if spot.has_polygon_seed:
            self.polygon_seed_status_label.setStyleSheet("color: #5cb85c;")
            self.polygon_seed_status_label.setText(
                f"Spot {spot.id}: polygon ready ({len(spot.polygon_seed_points)} vertices)."
            )
            return
        self.polygon_seed_status_label.setStyleSheet("color: #d9534f;")
        self.polygon_seed_status_label.setText(
            f"Spot {spot.id}: polygon missing. Cutie/XMem2 tracking will be blocked until you draw it."
        )

    def open_spot_polygon_mask_dialog(self):
        if not self._polygon_seed_scenario_active():
            QMessageBox.information(
                self,
                "Polygon Seed",
                "Polygon seed editing is available only for Offline Cutie and Offline XMem2.",
            )
            return
        leed_file, spot = self._get_selected_spot_for_polygon_edit()
        if leed_file is None or spot is None:
            QMessageBox.warning(self, "No Spot Selected", "Select a spot before editing its polygon mask.")
            return

        original_image_data = load_tiff(leed_file.filepath)
        if original_image_data is None or original_image_data.size == 0:
            QMessageBox.critical(self, "Error", "Could not load the image to edit the polygon mask.")
            return

        full_res_pixmap = self._convert_numpy_to_pixmap(original_image_data)
        dialog = PolygonMaskDialog(
            full_res_pixmap,
            existing_points=list(getattr(spot, "polygon_seed_points", [])),
            parent=self,
        )
        result = dialog.exec()
        if result != QDialog.DialogCode.Accepted:
            return

        spot.polygon_seed_points = dialog.get_polygon_points()
        self.populate_spot_list(leed_file.spots)
        self._reselect_spot_in_list(spot.id)
        self.refresh_current_view()

    def clear_selected_spot_polygon_mask(self):
        if not self._polygon_seed_scenario_active():
            QMessageBox.information(
                self,
                "Polygon Seed",
                "Polygon seed editing is available only for Offline Cutie and Offline XMem2.",
            )
            return
        leed_file, spot = self._get_selected_spot_for_polygon_edit()
        if leed_file is None or spot is None:
            QMessageBox.warning(self, "No Spot Selected", "Select a spot before clearing its polygon mask.")
            return
        if not spot.has_polygon_seed:
            return
        spot.polygon_seed_points = []
        self.populate_spot_list(leed_file.spots)
        self._reselect_spot_in_list(spot.id)
        self.refresh_current_view()

    def _reselect_spot_in_list(self, spot_id: int):
        for row in range(self.spot_list_widget.count()):
            item = self.spot_list_widget.item(row)
            spot = item.data(Qt.ItemDataRole.UserRole)
            if spot is not None and int(getattr(spot, "id", -1)) == int(spot_id):
                self.spot_list_widget.setCurrentRow(row)
                return

    def run_spot_finding(self):
        """Gathers parameters and runs the spot finding algorithm."""
        # 1. Get the current image file
        current_list_item = self.file_list_widget.currentItem()
        if not current_list_item:
            QMessageBox.warning(self, "No Image", "Please select an image to analyze.")
            return
        leed_file: LeedFile = current_list_item.data(Qt.ItemDataRole.UserRole)

        spot_settings = self._collect_spot_settings()
        try:
            found_spots = self._detect_spots_on_leed_file(leed_file, spot_settings)
            leed_file.spots = found_spots
        except (DLSpotDetectionError, YOLODetectionError) as exc:
            QMessageBox.critical(self, "Detector Error", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Detector Error",
                f"Unexpected error during detection:\n{exc}",
            )
            return
        
        # 5. Update the UI with results
        self.populate_spot_list(found_spots)
        self.refresh_current_view()

    def _detect_spots_on_leed_file(self, leed_file: LeedFile, spot_settings: dict):
        """Core spot detection logic shared by single-frame and batch runs."""
        backend = spot_settings["backend"]
        main_image = load_tiff(leed_file.filepath)
        background_image = load_tiff(self.background_file.filepath) if self.background_file else None
        if main_image is None or main_image.size == 0:
            raise ValueError(f"Could not load image: {leed_file.filepath}")
        main_image = self._apply_analysis_region_mask(main_image)
        if background_image is not None and background_image.size > 0:
            background_image = self._apply_analysis_region_mask(background_image)
        subtract_bg = False
        if backend in ("Deep Learning", "YOLO"):
            subtract_bg = bool(spot_settings.get(backend.lower(), {}).get("subtract_bg")) or bool(
                spot_settings.get("dl", {}).get("subtract_bg") if backend == "Deep Learning" else spot_settings.get("yolo", {}).get("subtract_bg")
            )
        if subtract_bg:
            if background_image is None or background_image.size == 0:
                raise ValueError("Background file not loaded. Load a background image before subtracting.")
            main_image = self._subtract_background_image(main_image, background_image)

        if backend == "Deep Learning":
            dl_cfg = spot_settings["dl"]
            if not dl_cfg.get("model_path"):
                raise ValueError("Select a deep-learning model before running detection.")
            device = dl_cfg.get("device", "Auto")
            return detect_spots_with_model(main_image, self.metadata_mask_list, dl_cfg)
        elif backend == "YOLO":
            y_cfg = spot_settings["yolo"]
            if not y_cfg.get("model_path"):
                raise ValueError("Select a YOLO model before running detection.")
            yolo_settings = YOLOSettings(
                model_path=y_cfg.get("model_path"),
                confidence=float(y_cfg.get("confidence", 0.25)),
                iou=float(y_cfg.get("iou", 0.45)),
                max_det=int(y_cfg.get("max_det", 300)),
                bbox_margin_px=int(y_cfg.get("bbox_margin_px", 2)),
                device=y_cfg.get("device", "auto"),
            )
            return detect_spots_with_yolo(main_image, self.metadata_mask_list, yolo_settings)
        else:
            params = spot_settings["classic"]
            return find_spots(main_image, background_image, self.metadata_mask_list, params)

    def _subtract_background_image(self, main_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """Simple background subtraction with mean scaling and respecting metadata masks."""
        try:
            main_float = img_as_float(main_image)
            bg_float = rgb2gray(background_image) if background_image.ndim == 3 else img_as_float(background_image)
            main_masked_for_mean = main_float.copy()
            bg_masked_for_mean = bg_float.copy()
            for rect in self.metadata_mask_list:
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                main_masked_for_mean[y:y+h, x:x+w] = 0
                bg_masked_for_mean[y:y+h, x:x+w] = 0
            main_mean = main_masked_for_mean[main_masked_for_mean > 0].mean() if (main_masked_for_mean > 0).any() else 0
            bg_mean = bg_masked_for_mean[bg_masked_for_mean > 0].mean() if (bg_masked_for_mean > 0).any() else 0
            scale = (main_mean / bg_mean) if bg_mean > 1e-9 else 1.0
            subtracted = main_float - (bg_float * scale)
            return subtracted
        except Exception as exc:
            print(f"Background subtraction failed, returning original image: {exc}")
            return main_image

    def run_spot_finding_all(self):
        """Run spot detection on all frames in the list using the current backend."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before running batch detection.")
            return
        spot_settings = self._collect_spot_settings()
        files = [f for f in self.data_files if not f.is_background]
        if not files:
            QMessageBox.information(self, "No Frames", "No frames available for detection.")
            return

        progress = QProgressDialog("Detecting spots on all frames...", "Cancel", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        processed = 0
        for idx, lf in enumerate(files, start=1):
            progress.setValue(idx - 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                lf.spots = self._detect_spots_on_leed_file(lf, spot_settings)
            except Exception as exc:
                progress.cancel()
                QMessageBox.critical(self, "Detector Error", f"Failed on {lf.filename}:\n{exc}")
                return
            processed += 1
        progress.setValue(len(files))
        self.populate_file_list()
        current_item = self.file_list_widget.currentItem()
        if current_item is None and self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)
        QMessageBox.information(self, "Find All Spots", f"Processed {processed} frame(s).")

    def copy_spots_to_all_frames(self):
        """Copy spots from the currently selected frame to all other frames."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before copying spots.")
            return
        current_item = self.file_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Frame Selected", "Select a frame to copy spots from.")
            return
        source_file: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
        if not source_file or not source_file.spots:
            QMessageBox.warning(self, "No Spots", "The selected frame has no spots to copy.")
            return

        skip_existing = self.copy_spots_skip_existing_checkbox.isChecked()
        other_files_with_spots = [
            lf for lf in self.data_files
            if lf is not source_file and lf.spots
        ]
        if other_files_with_spots and not skip_existing:
            reply = QMessageBox.question(
                self,
                "Overwrite Spots",
                "Some frames already contain spots. Overwrite them with the current frame's spots?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        copied = 0
        skipped = 0
        for lf in self.data_files:
            if lf is source_file:
                continue
            if skip_existing and lf.spots:
                skipped += 1
                continue
            new_spots = []
            for spot in source_file.spots:
                rect = QRect(spot.roi_rect)
                new_spots.append(
                    SpotROI(
                        id=spot.id,
                        roi_rect=rect,
                        center_x=spot.center_x,
                        center_y=spot.center_y,
                        intensity=spot.intensity,
                        source="manual_copy",
                    )
                )
            lf.spots = new_spots
            copied += 1

        self.populate_spot_list(source_file.spots)
        self.refresh_current_view()
        msg = f"Copied spots to {copied} frame(s)."
        if skipped:
            msg += f" Skipped {skipped} frame(s) with existing spots."
        QMessageBox.information(self, "Copy Spots", msg)

    @staticmethod
    def _square_roi_from_center(
        center_x: float,
        center_y: float,
        target_size: int,
        image_w: int,
        image_h: int,
    ) -> QRect:
        """Create a square ROI centered at (center_x, center_y), clamped to image bounds."""
        if image_w <= 0 or image_h <= 0:
            return QRect(0, 0, 1, 1)

        size = int(max(1, target_size))
        size = min(size, image_w, image_h)
        half = size / 2.0

        x = int(round(float(center_x) - half))
        y = int(round(float(center_y) - half))
        x = max(0, min(max(0, image_w - size), x))
        y = max(0, min(max(0, image_h - size), y))
        return QRect(int(x), int(y), int(size), int(size))

    def normalize_rois_all_frames(self):
        """Normalize all spot ROIs to the current ROI Size while preserving per-spot centers."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load images before normalizing ROIs.")
            return

        target_size = int(self.roi_size_spinbox.value())
        if target_size <= 0:
            QMessageBox.warning(self, "Invalid ROI Size", "ROI Size must be greater than 0.")
            return

        updated_frames = 0
        updated_spots = 0
        skipped_frames = 0

        for lf in self.data_files:
            if lf.is_background or not lf.spots:
                continue

            image_data = load_tiff(lf.filepath)
            if image_data is None or image_data.size == 0:
                skipped_frames += 1
                continue

            image_h, image_w = image_data.shape[:2]
            frame_changed = False

            for spot in lf.spots:
                rect_center = spot.roi_rect.center()
                try:
                    cx_raw = float(spot.center_x)
                except Exception:
                    cx_raw = float("nan")
                try:
                    cy_raw = float(spot.center_y)
                except Exception:
                    cy_raw = float("nan")
                cx = cx_raw if np.isfinite(cx_raw) else float(rect_center.x())
                cy = cy_raw if np.isfinite(cy_raw) else float(rect_center.y())

                new_rect = self._square_roi_from_center(cx, cy, target_size, image_w, image_h)
                if spot.roi_rect != new_rect:
                    frame_changed = True
                spot.roi_rect = new_rect
                updated_spots += 1

            if frame_changed:
                updated_frames += 1

        self.populate_file_list()
        current_item = self.file_list_widget.currentItem()
        if current_item is None and self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

        msg = (
            f"Normalized {updated_spots} spot ROI(s) to size {target_size}px "
            f"across {updated_frames} frame(s)."
        )
        if skipped_frames:
            msg += f" Skipped {skipped_frames} frame(s) due to unreadable images."
        QMessageBox.information(self, "Normalize ROIs", msg)

    def _collect_spot_settings(self) -> dict:
        """Serialize the current spot-finding UI state for persistence or reuse."""
        classic_settings = {
            'wth_radius': self.wth_radius_spinbox.value(),
            'blur_sigma': self.blur_sigma_spinbox.value(),
            'min_distance': self.min_distance_spinbox.value(),
            'threshold_rel': self.threshold_rel_spinbox.value(),
            'roi_size': self.roi_size_spinbox.value(),
        }

        dl_settings = {
            'model_path': self.dl_settings.model_path,
            'confidence': self.dl_confidence_spinbox.value(),
            'min_area_px': self.dl_min_area_spinbox.value(),
            'max_area_px': self.dl_max_area_spinbox.value(),
            'bbox_margin_px': self.dl_bbox_margin_spinbox.value(),
            'dilate_radius_px': self.dl_dilate_spinbox.value(),
            'subtract_bg': self.det_subtract_bg_checkbox.isChecked(),
            'device': self.dl_device_combo.currentText(),
        }

        yolo_settings = {
            'model_path': self.yolo_settings.model_path,
            'confidence': self.yolo_conf_spinbox.value(),
            'iou': self.yolo_iou_spinbox.value(),
            'max_det': self.yolo_max_det_spinbox.value(),
            'bbox_margin_px': self.yolo_bbox_margin_spinbox.value(),
            'subtract_bg': self.det_subtract_bg_checkbox.isChecked(),
            'device': self.dl_device_combo.currentText(),
        }

        return {
            'backend': self.current_spot_backend,
            'classic': classic_settings,
            'dl': dl_settings,
            'yolo': yolo_settings,
        }

    def _apply_spot_settings(self, settings: dict) -> None:
        """Apply serialized spot-finding settings back to the UI widgets."""
        classic = settings.get('classic', {})
        if classic:
            self.wth_radius_spinbox.setValue(int(classic.get('wth_radius', self.wth_radius_spinbox.value())))
            self.blur_sigma_spinbox.setValue(float(classic.get('blur_sigma', self.blur_sigma_spinbox.value())))
            self.min_distance_spinbox.setValue(int(classic.get('min_distance', self.min_distance_spinbox.value())))
            self.threshold_rel_spinbox.setValue(float(classic.get('threshold_rel', self.threshold_rel_spinbox.value())))
            self.roi_size_spinbox.setValue(int(classic.get('roi_size', self.roi_size_spinbox.value())))

        dl = settings.get('dl', {})
        if dl:
            if 'confidence' in dl:
                self.dl_confidence_spinbox.setValue(float(dl['confidence']))
            if 'min_area_px' in dl:
                self.dl_min_area_spinbox.setValue(int(dl['min_area_px']))
            if 'max_area_px' in dl:
                self.dl_max_area_spinbox.setValue(int(dl['max_area_px']))
            if 'bbox_margin_px' in dl:
                self.dl_bbox_margin_spinbox.setValue(int(dl['bbox_margin_px']))
            if 'dilate_radius_px' in dl:
                self.dl_dilate_spinbox.setValue(int(dl['dilate_radius_px']))
            if 'subtract_bg' in dl:
                self.det_subtract_bg_checkbox.setChecked(bool(dl['subtract_bg']))
            if 'device' in dl:
                idx = self.dl_device_combo.findText(str(dl['device']))
                if idx >= 0:
                    self.dl_device_combo.setCurrentIndex(idx)
            if 'model_path' in dl and dl['model_path']:
                idx = self.dl_model_combo.findData(dl['model_path'])
                if idx >= 0:
                    self.dl_model_combo.setCurrentIndex(idx)

        yolo = settings.get('yolo', {})
        if yolo:
            if 'confidence' in yolo:
                self.yolo_conf_spinbox.setValue(float(yolo['confidence']))
            if 'iou' in yolo:
                self.yolo_iou_spinbox.setValue(float(yolo['iou']))
            if 'max_det' in yolo:
                self.yolo_max_det_spinbox.setValue(int(yolo['max_det']))
            if 'bbox_margin_px' in yolo:
                self.yolo_bbox_margin_spinbox.setValue(int(yolo['bbox_margin_px']))
            if 'subtract_bg' in yolo:
                self.det_subtract_bg_checkbox.setChecked(bool(yolo['subtract_bg']))
            if 'device' in yolo:
                idx = self.dl_device_combo.findText(str(yolo['device']))
                if idx >= 0:
                    self.dl_device_combo.setCurrentIndex(idx)
            if 'model_path' in yolo and yolo['model_path']:
                idx = self.yolo_model_combo.findData(yolo['model_path'])
                if idx >= 0:
                    self.yolo_model_combo.setCurrentIndex(idx)

        backend = settings.get('backend')
        if backend:
            idx = self.spot_backend_combo.findText(backend)
            if idx >= 0:
                self.spot_backend_combo.setCurrentIndex(idx)

    def _analysis_circle_to_dict(self):
        if not self.analysis_circle:
            return None
        cx, cy, r = self.analysis_circle
        return {"cx": int(cx), "cy": int(cy), "r": int(r)}

    def _analysis_circle_from_dict(self, data):
        if not data:
            return None
        try:
            return (int(data.get("cx", 0)), int(data.get("cy", 0)), int(data.get("r", 0)))
        except Exception:
            return None

    def _normalize_analysis_ellipse(
        self, data: Any
    ) -> tuple[int, int, int, int, float] | None:
        """Coerce ellipse payload to (cx, cy, rx, ry, angle_deg)."""
        if data is None:
            return None

        def _to_float(value: Any) -> float | None:
            try:
                out = float(value)
            except Exception:
                return None
            if not np.isfinite(out):
                return None
            return out

        def _xy_from_any(value: Any) -> tuple[Any, Any]:
            if value is None:
                return None, None
            if hasattr(value, "x") and hasattr(value, "y"):
                try:
                    return value.x(), value.y()
                except Exception:
                    pass
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                return value[0], value[1]
            return None, None

        cx_raw = cy_raw = rx_raw = ry_raw = angle_raw = None

        if isinstance(data, dict):
            cx_raw, cy_raw = _xy_from_any(data.get("center"))
            if cx_raw is None:
                cx_raw = data.get("cx", data.get("center_x"))
            if cy_raw is None:
                cy_raw = data.get("cy", data.get("center_y"))
            rx_raw = data.get("rx", data.get("radius_x", data.get("a")))
            ry_raw = data.get("ry", data.get("radius_y", data.get("b")))
            angle_raw = data.get(
                "angle_deg",
                data.get("angle", data.get("theta_deg", data.get("theta", 0.0))),
            )
        elif isinstance(data, np.ndarray):
            return self._normalize_analysis_ellipse(data.tolist())
        elif isinstance(data, (tuple, list)):
            if len(data) >= 5:
                cx_raw, cy_raw, rx_raw, ry_raw, angle_raw = data[:5]
            elif len(data) >= 4:
                cx_raw, cy_raw = _xy_from_any(data[0])
                rx_raw, ry_raw, angle_raw = data[1], data[2], data[3]
                if cx_raw is None or cy_raw is None:
                    return None
            else:
                return None
        else:
            return None

        cx = _to_float(cx_raw)
        cy = _to_float(cy_raw)
        rx = _to_float(rx_raw)
        ry = _to_float(ry_raw)
        angle = _to_float(angle_raw)

        if cx is None or cy is None or rx is None or ry is None:
            return None
        if rx <= 0 or ry <= 0:
            return None
        if angle is None:
            angle = 0.0

        return (
            int(round(cx)),
            int(round(cy)),
            int(round(rx)),
            int(round(ry)),
            float(angle),
        )

    def _analysis_ellipse_to_dict(self):
        ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        if not ellipse:
            return None
        cx, cy, rx, ry, angle_deg = ellipse
        return {
            "cx": int(cx),
            "cy": int(cy),
            "rx": int(rx),
            "ry": int(ry),
            "angle_deg": float(angle_deg),
        }

    def _analysis_ellipse_from_dict(self, data):
        return self._normalize_analysis_ellipse(data)

    def set_bbox_transparency(self, percent: float):
        percent = max(0.0, min(100.0, float(percent)))
        self._bbox_transparency_pct = percent
        if hasattr(self, "image_viewer") and self.image_viewer:
            self.image_viewer.set_roi_transparency_percent(percent)

    def prompt_bbox_transparency(self):
        value, ok = QInputDialog.getDouble(
            self,
            "BBox Transparency",
            "Transparency (%): 0 = opaque, 100 = fully transparent",
            float(self._bbox_transparency_pct),
            0.0,
            100.0,
            1,
        )
        if ok:
            self.set_bbox_transparency(value)

    def _cycle_pan_add_delete_mode(self):
        current = getattr(self.image_viewer, "mode", "PAN")
        if current == "PAN":
            self.add_point_mode_action.trigger()
        elif current in {"ADD_ROI", "ADD_POINT"}:
            self.delete_roi_mode_action.trigger()
        else:
            self.pan_mode_action.trigger()

    def _cycle_bbox_transparency(self):
        steps = [0.0, 20.0, 50.0, 80.0]
        current = float(self._bbox_transparency_pct)
        for idx, value in enumerate(steps):
            if abs(current - value) < 0.5:
                next_value = steps[(idx + 1) % len(steps)]
                self.set_bbox_transparency(next_value)
                return
        self.set_bbox_transparency(0.0)

    def show_analysis_dialog(self):
        if not self.tracking_results:
            QMessageBox.information(self, "No Tracking Data", "Please run tracking before analysis.")
            return
        try:
            from trackleed.gui.analysis_dialog import AnalysisResultsDialog
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Analysis Unavailable",
                "Failed to load the analysis dialog.\n"
                f"{exc}\n\n"
                "Install requirements: pip install pyqtgraph PyOpenGL",
            )
            return

        file_map = {f.id: f for f in self.data_files if f.id is not None}
        normalized_ellipse = self._normalize_analysis_ellipse(self.analysis_ellipse)
        self.analysis_ellipse = normalized_ellipse
        dialog = AnalysisResultsDialog(
            self.tracking_results,
            file_map,
            self.background_file,
            analysis_ellipse=normalized_ellipse,
            parent=self,
            save_session_callback=lambda tracks: self.save_session_dialog(tracks),
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            self.tracking_results = dialog.tracks

    def show_positions_analysis_dialog(self):
        if not self.tracking_results:
            QMessageBox.information(self, "No Tracking Data", "Please run tracking before analysis.")
            return
        try:
            from trackleed.gui.positions_analysis_dialog import PositionsAnalysisDialog
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Positions Analysis Unavailable",
                "Failed to load the positions analysis dialog.\n"
                f"{exc}",
            )
            return

        file_map = {f.id: f for f in self.data_files if f.id is not None}
        dialog = PositionsAnalysisDialog(
            self.tracking_results,
            file_map,
            parent=self,
        )
        dialog.exec()

    def _collect_tapir_tracking_params(self) -> dict:
        """Collect the current TAPIR tracking settings from the UI."""
        return {
            "tapir_models_dir": str(self.tapir_models_dir.resolve()),
            "checkpoint_path": self.tapir_settings.checkpoint_path,
            "runtime_mode": self._tapir_runtime_mode(),
            "device": self.tapir_device_combo.currentData(),
            "inference_resolution": int(self.tapir_inference_resolution_combo.currentData()),
            "query_chunk_size": self.tapir_query_chunk_spinbox.value(),
            "window_size": self.tapir_window_size_spinbox.value(),
            "window_overlap": self.tapir_window_overlap_spinbox.value(),
            "use_visibility_gating": self.tapir_visibility_checkbox.isChecked(),
            "visibility_threshold": self.tapir_visibility_threshold_spinbox.value(),
            "use_uncertainty_gating": self.tapir_uncertainty_checkbox.isChecked(),
            "expected_dist_threshold": self.tapir_expected_dist_threshold_spinbox.value(),
            "keep_invisible_points_as_gaps": self.tapir_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.tapir_max_step_spinbox.value(),
            "max_reacquisition_gap": self.tapir_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.tapir_terminate_after_invisible_spinbox.value(),
            "center_mode": self.tapir_center_mode_combo.currentData(),
            "manual_center": (
                self.tapir_center_x_spinbox.value(),
                self.tapir_center_y_spinbox.value(),
            ),
            "direction_mode": self.tapir_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.tapir_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.tapir_max_radial_jump_spinbox.value(),
            "recenter_mode": self.tapir_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.tapir_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.tapir_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.tapir_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_locotrack_tracking_params(self) -> dict:
        """Collect the current LocoTrack tracking settings from the UI."""
        repo_path = self.locotrack_repo_path_edit.text().strip() or self.locotrack_settings.repo_path
        selected_model_size = self.locotrack_model_size_combo.currentData()
        checkpoint_name = str(self.locotrack_settings.checkpoint_path or "").lower()
        if selected_model_size == "auto":
            if "base" in checkpoint_name:
                selected_model_size = "base"
            elif "small" in checkpoint_name:
                selected_model_size = "small"
            else:
                selected_model_size = "small"
        return {
            "locotrack_models_dir": str(self.locotrack_models_dir.resolve()),
            "locotrack_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.locotrack_settings.checkpoint_path,
            "python_executable": self.locotrack_python_executable_edit.text().strip() or None,
            "backend_mode": self.locotrack_backend_mode_combo.currentData(),
            "model_size": selected_model_size,
            "device": self.locotrack_device_combo.currentData(),
            "inference_resolution": int(self.locotrack_inference_resolution_combo.currentData()),
            "query_chunk_size": self.locotrack_query_chunk_spinbox.value(),
            "subprocess_timeout_sec": self.locotrack_settings.subprocess_timeout_sec,
            "window_size": self.locotrack_window_size_spinbox.value(),
            "window_overlap": self.locotrack_window_overlap_spinbox.value(),
            "use_visibility_gating": self.locotrack_visibility_checkbox.isChecked(),
            "visibility_threshold": self.locotrack_visibility_threshold_spinbox.value(),
            "use_uncertainty_gating": self.locotrack_uncertainty_checkbox.isChecked(),
            "expected_dist_threshold": self.locotrack_expected_dist_threshold_spinbox.value(),
            "keep_invisible_points_as_gaps": self.locotrack_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.locotrack_max_step_spinbox.value(),
            "max_reacquisition_gap": self.locotrack_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.locotrack_terminate_after_invisible_spinbox.value(),
            "center_mode": self.locotrack_center_mode_combo.currentData(),
            "manual_center": (
                self.locotrack_center_x_spinbox.value(),
                self.locotrack_center_y_spinbox.value(),
            ),
            "direction_mode": self.locotrack_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.locotrack_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.locotrack_max_radial_jump_spinbox.value(),
            "recenter_mode": self.locotrack_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.locotrack_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.locotrack_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.locotrack_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_cotracker_tracking_params(self) -> dict:
        """Collect the current CoTracker3 tracking settings from the UI."""
        repo_path = self.cotracker_repo_path_edit.text().strip() or self.cotracker_settings.repo_path
        return {
            "cotracker3_models_dir": str(self.cotracker_models_dir.resolve()),
            "cotracker3_repo_path": repo_path,
            "cotracker_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.cotracker_settings.checkpoint_path,
            "device": self.cotracker_device_combo.currentData(),
            "inference_resolution": int(self.cotracker_inference_resolution_combo.currentData()),
            "window_len": self.cotracker_window_len_spinbox.value(),
            "window_size": self.cotracker_window_size_spinbox.value(),
            "window_overlap": self.cotracker_window_overlap_spinbox.value(),
            "use_visibility_gating": self.cotracker_visibility_checkbox.isChecked(),
            "keep_invisible_points_as_gaps": self.cotracker_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.cotracker_max_step_spinbox.value(),
            "max_reacquisition_gap": self.cotracker_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.cotracker_terminate_after_invisible_spinbox.value(),
            "center_mode": self.cotracker_center_mode_combo.currentData(),
            "manual_center": (
                self.cotracker_center_x_spinbox.value(),
                self.cotracker_center_y_spinbox.value(),
            ),
            "direction_mode": self.cotracker_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.cotracker_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.cotracker_max_radial_jump_spinbox.value(),
            "recenter_mode": self.cotracker_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.cotracker_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.cotracker_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.cotracker_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_tapnext_tracking_params(self) -> dict:
        """Collect the current TAPNext tracking settings from the UI."""
        repo_path = self.tapnext_repo_path_edit.text().strip() or self.tapnext_settings.repo_path
        return {
            "tapnext_models_dir": str(self.tapnext_models_dir.resolve()),
            "tapnext_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.tapnext_settings.checkpoint_path,
            "device": self.tapnext_device_combo.currentData(),
            "inference_resolution": int(self.tapnext_inference_resolution_combo.currentData()),
            "autocast": self.tapnext_autocast_checkbox.isChecked(),
            "window_size": self.tapnext_window_size_spinbox.value(),
            "window_overlap": self.tapnext_window_overlap_spinbox.value(),
            "use_visibility_gating": self.tapnext_visibility_checkbox.isChecked(),
            "visibility_threshold": self.tapnext_visibility_threshold_spinbox.value(),
            "use_certainty_gating": self.tapnext_certainty_checkbox.isChecked(),
            "certainty_threshold": self.tapnext_certainty_threshold_spinbox.value(),
            "keep_invisible_points_as_gaps": self.tapnext_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.tapnext_max_step_spinbox.value(),
            "max_reacquisition_gap": self.tapnext_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.tapnext_terminate_after_invisible_spinbox.value(),
            "center_mode": self.tapnext_center_mode_combo.currentData(),
            "manual_center": (
                self.tapnext_center_x_spinbox.value(),
                self.tapnext_center_y_spinbox.value(),
            ),
            "direction_mode": self.tapnext_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.tapnext_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.tapnext_max_radial_jump_spinbox.value(),
            "recenter_mode": self.tapnext_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.tapnext_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.tapnext_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.tapnext_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_trackonr_tracking_params(self) -> dict:
        """Collect the current Track-On-R tracking settings from the UI."""
        repo_path = self.trackonr_repo_path_edit.text().strip() or self.trackonr_settings.repo_path
        inference_resolution = self.trackonr_inference_resolution_combo.currentData()
        if isinstance(inference_resolution, (tuple, list)) and len(inference_resolution) >= 2:
            inference_resolution = (int(inference_resolution[0]), int(inference_resolution[1]))
        else:
            inference_resolution = self.trackonr_settings.inference_resolution
        return {
            "trackonr_models_dir": str(self.trackonr_models_dir.resolve()),
            "trackonr_repo_path": repo_path,
            "track_on_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.trackonr_settings.checkpoint_path,
            "python_executable": self.trackonr_python_executable_edit.text().strip() or None,
            "backend_mode": self.trackonr_backend_mode_combo.currentData(),
            "device": self.trackonr_device_combo.currentData(),
            "inference_resolution": inference_resolution,
            "support_grid_size": self.trackonr_support_grid_spinbox.value(),
            "subprocess_timeout_sec": self.trackonr_settings.subprocess_timeout_sec,
            "window_size": self.trackonr_window_size_spinbox.value(),
            "window_overlap": self.trackonr_window_overlap_spinbox.value(),
            "use_visibility_gating": self.trackonr_visibility_checkbox.isChecked(),
            "visibility_threshold": self.trackonr_visibility_threshold_spinbox.value(),
            "keep_invisible_points_as_gaps": self.trackonr_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.trackonr_max_step_spinbox.value(),
            "max_reacquisition_gap": self.trackonr_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.trackonr_terminate_after_invisible_spinbox.value(),
            "center_mode": self.trackonr_center_mode_combo.currentData(),
            "manual_center": (
                self.trackonr_center_x_spinbox.value(),
                self.trackonr_center_y_spinbox.value(),
            ),
            "direction_mode": self.trackonr_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.trackonr_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.trackonr_max_radial_jump_spinbox.value(),
            "recenter_mode": self.trackonr_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.trackonr_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.trackonr_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.trackonr_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_pipspp_tracking_params(self) -> dict:
        """Collect the current PIPs++ tracking settings from the UI."""
        repo_path = self.pipspp_repo_path_edit.text().strip() or self.pipspp_settings.repo_path
        inference_resolution = self.pipspp_inference_resolution_combo.currentData()
        if isinstance(inference_resolution, (tuple, list)) and len(inference_resolution) >= 2:
            inference_resolution = (int(inference_resolution[0]), int(inference_resolution[1]))
        else:
            inference_resolution = self.pipspp_settings.inference_resolution
        return {
            "pipspp_models_dir": str(self.pipspp_models_dir.resolve()),
            "pipspp_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.pipspp_settings.checkpoint_path,
            "python_executable": self.pipspp_python_executable_edit.text().strip() or None,
            "backend_mode": self.pipspp_backend_mode_combo.currentData(),
            "device": self.pipspp_device_combo.currentData(),
            "inference_resolution": inference_resolution,
            "iters": self.pipspp_iters_spinbox.value(),
            "beautify": self.pipspp_beautify_checkbox.isChecked(),
            "use_inbounds_visibility": self.pipspp_inbounds_visibility_checkbox.isChecked(),
            "subprocess_timeout_sec": self.pipspp_settings.subprocess_timeout_sec,
            "window_size": self.pipspp_window_size_spinbox.value(),
            "window_overlap": self.pipspp_window_overlap_spinbox.value(),
            "keep_invisible_points_as_gaps": self.pipspp_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.pipspp_max_step_spinbox.value(),
            "max_reacquisition_gap": self.pipspp_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.pipspp_terminate_after_invisible_spinbox.value(),
            "center_mode": self.pipspp_center_mode_combo.currentData(),
            "manual_center": (
                self.pipspp_center_x_spinbox.value(),
                self.pipspp_center_y_spinbox.value(),
            ),
            "direction_mode": self.pipspp_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.pipspp_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.pipspp_max_radial_jump_spinbox.value(),
            "recenter_mode": self.pipspp_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.pipspp_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.pipspp_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.pipspp_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_mftiq_tracking_params(self) -> dict:
        """Collect the current MFTIQ tracking settings from the UI."""
        repo_path = self.mftiq_repo_path_edit.text().strip() or self.mftiq_settings.repo_path
        inference_resolution = self.mftiq_inference_resolution_combo.currentData()
        if isinstance(inference_resolution, (tuple, list)) and len(inference_resolution) >= 2:
            inference_resolution = (int(inference_resolution[0]), int(inference_resolution[1]))
        else:
            inference_resolution = None
        return {
            "mftiq_models_dir": str(self.mftiq_models_dir.resolve()),
            "mftiq_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.mftiq_settings.checkpoint_path,
            "python_executable": self.mftiq_python_executable_edit.text().strip() or None,
            "config_path": self.mftiq_settings.config_path,
            "device": self.mftiq_device_combo.currentData(),
            "backend_mode": self.mftiq_backend_mode_combo.currentData(),
            "inference_resolution": inference_resolution,
            "occlusion_threshold": self.mftiq_occlusion_threshold_spinbox.value(),
            "subprocess_timeout_sec": self.mftiq_settings.subprocess_timeout_sec,
            "window_size": self.mftiq_window_size_spinbox.value(),
            "window_overlap": self.mftiq_window_overlap_spinbox.value(),
            "keep_invisible_points_as_gaps": self.mftiq_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.mftiq_max_step_spinbox.value(),
            "max_reacquisition_gap": self.mftiq_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.mftiq_terminate_after_invisible_spinbox.value(),
            "center_mode": self.mftiq_center_mode_combo.currentData(),
            "manual_center": (
                self.mftiq_center_x_spinbox.value(),
                self.mftiq_center_y_spinbox.value(),
            ),
            "direction_mode": self.mftiq_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.mftiq_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.mftiq_max_radial_jump_spinbox.value(),
            "recenter_mode": self.mftiq_recenter_mode_combo.currentData(),
            "recenter_patch_multiplier": self.mftiq_centroid_multiplier_spinbox.value(),
            "centroid_rel_threshold": self.mftiq_centroid_rel_thr_spinbox.value(),
            "centroid_max_shift_ratio": self.mftiq_centroid_max_shift_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_cutie_tracking_params(self) -> dict:
        """Collect the current Cutie tracking settings from the UI."""
        repo_path = self.cutie_repo_path_edit.text().strip() or self.cutie_settings.repo_path
        inference_resolution = self.cutie_inference_resolution_combo.currentData()
        if inference_resolution in {"", 0}:
            inference_resolution = None
        mem_every = self.cutie_mem_every_spinbox.value()
        if mem_every <= 0:
            mem_every = None
        return {
            "cutie_models_dir": str(self.cutie_models_dir.resolve()),
            "cutie_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.cutie_settings.checkpoint_path,
            "python_executable": self.cutie_python_executable_edit.text().strip() or None,
            "device": self.cutie_device_combo.currentData(),
            "backend_mode": self.cutie_backend_mode_combo.currentData(),
            "representative_point_mode": self.cutie_repr_mode_combo.currentData(),
            "inference_resolution": inference_resolution,
            "window_size": self.cutie_window_size_spinbox.value(),
            "window_overlap": self.cutie_window_overlap_spinbox.value(),
            "initial_bbox_padding_px": self.cutie_bbox_padding_spinbox.value(),
            "initial_mask_mode": self.cutie_initial_mask_mode_combo.currentData(),
            "initial_mask_threshold_rel": self.cutie_initial_mask_threshold_spinbox.value(),
            "initial_mask_disk_radius_ratio": self.cutie_initial_mask_disk_spinbox.value(),
            "mem_every": mem_every,
            "subprocess_timeout_sec": self.cutie_settings.subprocess_timeout_sec,
            "keep_invisible_points_as_gaps": self.cutie_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.cutie_max_step_spinbox.value(),
            "max_reacquisition_gap": self.cutie_max_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.cutie_terminate_gap_spinbox.value(),
            "max_mask_area_ratio_change": self.cutie_max_area_ratio_spinbox.value(),
            "max_bbox_aspect_ratio_change": self.cutie_max_aspect_ratio_spinbox.value(),
            "max_bbox_center_jump_px": self.cutie_bbox_jump_spinbox.value(),
            "center_mode": self.cutie_center_mode_combo.currentData(),
            "manual_center": (
                self.cutie_center_x_spinbox.value(),
                self.cutie_center_y_spinbox.value(),
            ),
            "direction_mode": self.cutie_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.cutie_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.cutie_max_radial_jump_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_tomp_tracking_params(self) -> dict:
        """Collect the current ToMP / TaMOs tracking settings from the UI."""
        repo_path = self.tomp_repo_path_edit.text().strip() or self.tomp_settings.repo_path
        inference_resolution = self.tomp_inference_resolution_combo.currentData()
        if inference_resolution in {"", 0}:
            inference_resolution = None
        return {
            "tomp_models_dir": str(self.tomp_models_dir.resolve()),
            "tamos_models_dir": str(self.tamos_models_dir.resolve()),
            "tomp_repo_path": repo_path,
            "pytracking_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.tomp_settings.checkpoint_path,
            "python_executable": self.tomp_python_executable_edit.text().strip() or None,
            "backend_mode": self.tomp_backend_mode_combo.currentData(),
            "model_variant": self.tomp_model_variant_combo.currentData(),
            "device": self.tomp_device_combo.currentData(),
            "representative_point_mode": self.tomp_repr_mode_combo.currentData(),
            "inference_resolution": inference_resolution,
            "window_size": self.tomp_window_size_spinbox.value(),
            "window_overlap": self.tomp_window_overlap_spinbox.value(),
            "initial_bbox_padding_px": self.tomp_bbox_padding_spinbox.value(),
            "subprocess_timeout_sec": self.tomp_settings.subprocess_timeout_sec,
            "keep_invisible_points_as_gaps": self.tomp_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.tomp_max_step_spinbox.value(),
            "max_reacquisition_gap": self.tomp_max_reacquisition_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.tomp_terminate_after_invisible_spinbox.value(),
            "center_mode": self.tomp_center_mode_combo.currentData(),
            "manual_center": (
                self.tomp_center_x_spinbox.value(),
                self.tomp_center_y_spinbox.value(),
            ),
            "direction_mode": self.tomp_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.tomp_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.tomp_max_radial_jump_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_xmem2_tracking_params(self) -> dict:
        """Collect the current XMem2 tracking settings from the UI."""
        repo_path = self.xmem2_repo_path_edit.text().strip() or self.xmem2_settings.repo_path
        inference_resolution = self.xmem2_inference_resolution_combo.currentData()
        if inference_resolution in {"", 0}:
            inference_resolution = None
        mem_every = self.xmem2_mem_every_spinbox.value()
        if mem_every <= 0:
            mem_every = None
        return {
            "xmem2_models_dir": str(self.xmem2_models_dir.resolve()),
            "xmem2_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.xmem2_settings.checkpoint_path,
            "python_executable": self.xmem2_python_executable_edit.text().strip() or None,
            "device": self.xmem2_device_combo.currentData(),
            "backend_mode": self.xmem2_backend_mode_combo.currentData(),
            "representative_point_mode": self.xmem2_repr_mode_combo.currentData(),
            "inference_resolution": inference_resolution,
            "window_size": self.xmem2_window_size_spinbox.value(),
            "window_overlap": self.xmem2_window_overlap_spinbox.value(),
            "initial_bbox_padding_px": self.xmem2_bbox_padding_spinbox.value(),
            "initial_mask_mode": self.xmem2_initial_mask_mode_combo.currentData(),
            "initial_mask_threshold_rel": self.xmem2_initial_mask_threshold_spinbox.value(),
            "initial_mask_disk_radius_ratio": self.xmem2_initial_mask_disk_spinbox.value(),
            "mem_every": mem_every,
            "subprocess_timeout_sec": self.xmem2_settings.subprocess_timeout_sec,
            "keep_invisible_points_as_gaps": self.xmem2_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.xmem2_max_step_spinbox.value(),
            "max_reacquisition_gap": self.xmem2_max_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.xmem2_terminate_gap_spinbox.value(),
            "max_mask_area_ratio_change": self.xmem2_max_area_ratio_spinbox.value(),
            "max_bbox_aspect_ratio_change": self.xmem2_max_aspect_ratio_spinbox.value(),
            "max_bbox_center_jump_px": self.xmem2_bbox_jump_spinbox.value(),
            "center_mode": self.xmem2_center_mode_combo.currentData(),
            "manual_center": (
                self.xmem2_center_x_spinbox.value(),
                self.xmem2_center_y_spinbox.value(),
            ),
            "direction_mode": self.xmem2_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.xmem2_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.xmem2_max_radial_jump_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def _collect_sam2_tracking_params(self) -> dict:
        """Collect the current SAM 2 tracking settings from the UI."""
        repo_path = self.sam2_repo_path_edit.text().strip() or self.sam2_settings.repo_path
        config_path = self.sam2_settings.config_path
        if config_path and not Path(config_path).expanduser().exists():
            config_path = None
        inference_resolution = self.sam2_inference_resolution_combo.currentData()
        if inference_resolution in {"", 0}:
            inference_resolution = None
        max_mask_area = self.sam2_max_mask_area_spinbox.value()
        if max_mask_area <= 0.0:
            max_mask_area = None
        return {
            "sam2_models_dir": str(self.sam2_models_dir.resolve()),
            "sam2_repo_path": repo_path,
            "repo_path": repo_path,
            "checkpoint_path": self.sam2_settings.checkpoint_path,
            "python_executable": self.sam2_python_executable_edit.text().strip() or None,
            "config_path": config_path,
            "device": self.sam2_device_combo.currentData(),
            "backend_mode": self.sam2_backend_mode_combo.currentData(),
            "prompt_mode": self.sam2_prompt_mode_combo.currentData(),
            "representative_point_mode": self.sam2_repr_mode_combo.currentData(),
            "inference_resolution": inference_resolution,
            "window_size": self.sam2_window_size_spinbox.value(),
            "window_overlap": self.sam2_window_overlap_spinbox.value(),
            "initial_bbox_padding_px": self.sam2_bbox_padding_spinbox.value(),
            "mask_threshold": self.sam2_mask_threshold_spinbox.value(),
            "min_mask_area": self.sam2_min_mask_area_spinbox.value(),
            "max_mask_area": max_mask_area,
            "use_vos_optimized": self.sam2_use_vos_checkbox.isChecked(),
            "apply_postprocessing": self.sam2_postprocessing_checkbox.isChecked(),
            "offload_video_to_cpu": self.sam2_offload_video_checkbox.isChecked(),
            "offload_state_to_cpu": self.sam2_offload_state_checkbox.isChecked(),
            "async_loading_frames": self.sam2_async_loading_checkbox.isChecked(),
            "subprocess_timeout_sec": self.sam2_settings.subprocess_timeout_sec,
            "keep_invisible_points_as_gaps": self.sam2_keep_gaps_checkbox.isChecked(),
            "max_step_distance_px": self.sam2_max_step_spinbox.value(),
            "max_reacquisition_gap": self.sam2_max_gap_spinbox.value(),
            "terminate_after_invisible_frames": self.sam2_terminate_gap_spinbox.value(),
            "max_mask_area_ratio_change": self.sam2_max_area_ratio_spinbox.value(),
            "max_bbox_aspect_ratio_change": self.sam2_max_aspect_ratio_spinbox.value(),
            "max_mask_components": self.sam2_max_components_spinbox.value(),
            "max_bbox_center_jump_px": self.sam2_bbox_jump_spinbox.value(),
            "center_mode": self.sam2_center_mode_combo.currentData(),
            "manual_center": (
                self.sam2_center_x_spinbox.value(),
                self.sam2_center_y_spinbox.value(),
            ),
            "direction_mode": self.sam2_direction_mode_combo.currentData(),
            "lateral_angle_tolerance_deg": self.sam2_lateral_tol_spinbox.value(),
            "max_radial_jump_px": self.sam2_max_radial_jump_spinbox.value(),
            "roi_size": self.roi_size_spinbox.value(),
        }

    def run_tracking(self):
        """Dispatches a tracking task to a background worker based on selected scenario."""
        selected_scenario = self.tracking_scenario_combo.currentText()
        worker = None
        cancel_event = Event()
        if "Optical Flow" in selected_scenario:
            optical_params = {"roi_size": self.roi_size_spinbox.value()}
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Optical Flow requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            subtract_background = self.lk_subtract_bg_checkbox.isChecked()
            if subtract_background and not self.background_file:
                QMessageBox.warning(
                    self,
                    "No Background",
                    "Load a background file before enabling background subtraction for Optical Flow.",
                )
                return

            optical_params.update({
                "lk_win_size": self.lk_win_size_spinbox.value(),
                "lk_max_level": self.lk_max_level_spinbox.value(),
                "lk_max_iterations": self.lk_iterations_spinbox.value(),
                "lk_epsilon": self.lk_epsilon_spinbox.value(),
                "lk_min_eigen_threshold": self.lk_min_eig_spinbox.value(),
                "use_centroid_refinement": self.lk_use_centroid_checkbox.isChecked(),
                "centroid_patch_multiplier": self.lk_centroid_multiplier_spinbox.value(),
                "centroid_rel_threshold": self.lk_centroid_rel_thr_spinbox.value(),
                "centroid_max_expand": self.lk_centroid_expand_spinbox.value(),
                "subtract_background": subtract_background,
            })
            if subtract_background and self.background_file:
                optical_params["background_filepath"] = str(self.background_file.filepath)

            if selected_scenario == "Optical Flow (skip, anchor)":
                optical_params.update({
                    "lk_err_threshold": self.lk_error_threshold_spinbox.value(),
                    "lk_fb_threshold": self.lk_fb_threshold_spinbox.value(),
                    "use_forward_backward": self.lk_fb_checkbox.isChecked(),
                    "max_skip_frames": self.lk_max_skip_frames_spinbox.value(),
                    "oob_behavior": self.oob_behavior_combo.currentData(),
                    "use_ncc_check": self.lk_use_ncc_checkbox.isChecked(),
                    "ncc_threshold": self.lk_ncc_threshold_spinbox.value(),
                    "ncc_patch_radius": self.lk_ncc_patch_spinbox.value(),
                })

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())

            if selected_scenario == "Optical Flow (skip, anchor)":
                worker = GenericWorker(
                    track_spots_optical_flow_skip_anchor,
                    file_map,
                    sorted_ids,
                    ref_file.id,
                    optical_params,
                    cancel_event=cancel_event,
                )
            else:
                worker = GenericWorker(
                    track_spots_optical_flow,
                    file_map,
                    sorted_ids,
                    ref_file.id,
                    optical_params,
                    cancel_event=cancel_event,
                )

        elif selected_scenario in {"Offline TAPIR", "BootsTAPIR Online / Causal"}:
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(
                    self,
                    "No Reference",
                    "TAPIR tracking requires selecting a reference frame.",
                )
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            status = probe_tapir_environment(
                models_dir=self.tapir_models_dir,
                checkpoint_path=self.tapir_settings.checkpoint_path,
            )
            requires_online_checkpoint = selected_scenario == "BootsTAPIR Online / Causal"
            if not status.is_ready or (requires_online_checkpoint and not status.selected_checkpoint_is_online):
                QMessageBox.warning(
                    self,
                    "TAPIR Unavailable",
                    format_tapir_environment_report(status),
                )
                return

            tapir_params = self._collect_tapir_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_tapir_online
                if selected_scenario == "BootsTAPIR Online / Causal"
                else track_spots_tapir_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                tapir_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline LocoTrack":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline LocoTrack requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.locotrack_repo_path_edit.text().strip() or self.locotrack_settings.repo_path
            self.locotrack_settings.repo_path = repo_path
            self.locotrack_settings.python_executable = self.locotrack_python_executable_edit.text().strip() or None
            self.locotrack_settings.backend_mode = self.locotrack_backend_mode_combo.currentData()
            status = probe_locotrack_environment(
                models_dir=self.locotrack_models_dir,
                checkpoint_path=self.locotrack_settings.checkpoint_path,
                repo_path=self.locotrack_settings.repo_path,
            )
            python_executable = self.locotrack_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
            )

            if self.locotrack_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "LocoTrack Unavailable",
                        format_locotrack_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "LocoTrack Unavailable",
                        format_locotrack_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "LocoTrack Unavailable",
                        format_locotrack_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `locotrack` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            locotrack_params = self._collect_locotrack_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_locotrack_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                locotrack_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline CoTracker3":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline CoTracker3 requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.cotracker_repo_path_edit.text().strip() or self.cotracker_settings.repo_path
            self.cotracker_settings.repo_path = repo_path
            status = probe_cotracker_environment(
                models_dir=self.cotracker_models_dir,
                checkpoint_path=self.cotracker_settings.checkpoint_path,
                repo_path=self.cotracker_settings.repo_path,
            )
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "CoTracker3 Unavailable",
                    format_cotracker_environment_report(status),
                )
                return

            cotracker_params = self._collect_cotracker_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_cotracker3_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                cotracker_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline TAPNext":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline TAPNext requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.tapnext_repo_path_edit.text().strip() or self.tapnext_settings.repo_path
            self.tapnext_settings.repo_path = repo_path
            status = probe_tapnext_environment(
                models_dir=self.tapnext_models_dir,
                checkpoint_path=self.tapnext_settings.checkpoint_path,
                repo_path=self.tapnext_settings.repo_path,
            )
            if not status.is_ready:
                QMessageBox.warning(
                    self,
                    "TAPNext Unavailable",
                    format_tapnext_environment_report(status),
                )
                return

            tapnext_params = self._collect_tapnext_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_tapnext_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                tapnext_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline Track-On-R":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline Track-On-R requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.trackonr_repo_path_edit.text().strip() or self.trackonr_settings.repo_path
            self.trackonr_settings.repo_path = repo_path
            self.trackonr_settings.python_executable = self.trackonr_python_executable_edit.text().strip() or None
            self.trackonr_settings.backend_mode = self.trackonr_backend_mode_combo.currentData()
            status = probe_trackonr_environment(
                models_dir=self.trackonr_models_dir,
                checkpoint_path=self.trackonr_settings.checkpoint_path,
                repo_path=self.trackonr_settings.repo_path,
            )
            python_executable = self.trackonr_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
            )

            if self.trackonr_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "Track-On-R Unavailable",
                        format_trackonr_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "Track-On-R Unavailable",
                        format_trackonr_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "Track-On-R Unavailable",
                        format_trackonr_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `trackonr` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            trackonr_params = self._collect_trackonr_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_trackonr_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                trackonr_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline PIPs++":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline PIPs++ requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.pipspp_repo_path_edit.text().strip() or self.pipspp_settings.repo_path
            self.pipspp_settings.repo_path = repo_path
            self.pipspp_settings.python_executable = self.pipspp_python_executable_edit.text().strip() or None
            self.pipspp_settings.backend_mode = self.pipspp_backend_mode_combo.currentData()
            status = probe_pipspp_environment(
                models_dir=self.pipspp_models_dir,
                checkpoint_path=self.pipspp_settings.checkpoint_path,
                repo_path=self.pipspp_settings.repo_path,
            )
            python_executable = self.pipspp_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
                and status.selected_checkpoint_file is not None
            )

            if self.pipspp_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "PIPs++ Unavailable",
                        format_pipspp_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "PIPs++ Unavailable",
                        format_pipspp_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "PIPs++ Unavailable",
                        format_pipspp_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `pipspp` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            pipspp_params = self._collect_pipspp_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_pipspp_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                pipspp_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline MFTIQ":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline MFTIQ requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.mftiq_repo_path_edit.text().strip() or self.mftiq_settings.repo_path
            self.mftiq_settings.repo_path = repo_path
            self.mftiq_settings.python_executable = self.mftiq_python_executable_edit.text().strip() or None
            self.mftiq_settings.backend_mode = self.mftiq_backend_mode_combo.currentData()
            status = probe_mftiq_environment(
                models_dir=self.mftiq_models_dir,
                checkpoint_path=self.mftiq_settings.checkpoint_path,
                repo_path=self.mftiq_settings.repo_path,
                config_path=self.mftiq_settings.config_path,
            )

            python_executable = self.mftiq_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
                and status.config_exists
            )

            if self.mftiq_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "MFTIQ Unavailable",
                        format_mftiq_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "MFTIQ Unavailable",
                        format_mftiq_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "MFTIQ Unavailable",
                        format_mftiq_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `mftiq` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            mftiq_params = self._collect_mftiq_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_mftiq_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                mftiq_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline Cutie":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline Cutie requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return
            missing_polygon_seed_ids = [
                int(spot.id)
                for spot in ref_file.spots
                if not getattr(spot, "has_polygon_seed", False)
            ]
            if missing_polygon_seed_ids:
                QMessageBox.warning(
                    self,
                    "Cutie Requires Polygon Mask",
                    "Offline Cutie requires a polygon mask for every reference spot.\n\n"
                    "Missing polygon for spot IDs: "
                    + ", ".join(str(spot_id) for spot_id in missing_polygon_seed_ids),
                )
                return

            repo_path = self.cutie_repo_path_edit.text().strip() or self.cutie_settings.repo_path
            self.cutie_settings.repo_path = repo_path
            self.cutie_settings.python_executable = self.cutie_python_executable_edit.text().strip() or None
            self.cutie_settings.backend_mode = self.cutie_backend_mode_combo.currentData()

            status = probe_cutie_environment(
                models_dir=self.cutie_models_dir,
                checkpoint_path=self.cutie_settings.checkpoint_path,
                repo_path=self.cutie_settings.repo_path,
            )
            python_executable = self.cutie_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
            )
            if self.cutie_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "Cutie Unavailable",
                        format_cutie_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "Cutie Unavailable",
                        format_cutie_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "Cutie Unavailable",
                        format_cutie_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `cutie` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            cutie_params = self._collect_cutie_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_cutie_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                cutie_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline ToMP":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline ToMP requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.tomp_repo_path_edit.text().strip() or self.tomp_settings.repo_path
            self.tomp_settings.repo_path = repo_path
            self.tomp_settings.python_executable = self.tomp_python_executable_edit.text().strip() or None
            self.tomp_settings.backend_mode = self.tomp_backend_mode_combo.currentData()
            self.tomp_settings.model_variant = self._current_tomp_variant_key()

            current_variant = self.tomp_settings.model_variant
            status = probe_tomp_environment(
                tomp_models_dir=self.tomp_models_dir,
                tamos_models_dir=self.tamos_models_dir,
                tomp_checkpoint_path=(
                    self.tomp_settings.checkpoint_path if current_variant in {"tomp50", "tomp101"} else None
                ),
                tamos_checkpoint_path=(
                    self.tomp_settings.checkpoint_path if current_variant in {"tamos_resnet50", "tamos_swin_base"} else None
                ),
                repo_path=self.tomp_settings.repo_path,
            )
            if current_variant in {"tamos_resnet50", "tamos_swin_base"}:
                selected_checkpoint_obj = select_default_tamos_checkpoint_for_variant(
                    status.available_tamos_checkpoints,
                    current_variant,
                )
                variant_ready = bool(status.tamos_ready and selected_checkpoint_obj and selected_checkpoint_obj.exists())
            else:
                selected_checkpoint_obj = select_default_tomp_checkpoint_for_variant(
                    status.available_tomp_checkpoints,
                    current_variant,
                )
                variant_ready = bool(status.tomp_ready and selected_checkpoint_obj and selected_checkpoint_obj.exists())
            python_executable = self.tomp_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and selected_checkpoint_obj is not None
                and selected_checkpoint_obj.exists()
            )
            if self.tomp_settings.backend_mode == "direct":
                if not variant_ready:
                    QMessageBox.warning(
                        self,
                        "ToMP / TaMOs Unavailable",
                        format_tomp_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "ToMP / TaMOs Unavailable",
                        format_tomp_environment_report(status),
                    )
                    return
                if not (variant_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "ToMP / TaMOs Unavailable",
                        format_tomp_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `tomp` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            tomp_params = self._collect_tomp_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_tomp_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                tomp_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline XMem2":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline XMem2 requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return
            missing_polygon_seed_ids = [
                int(spot.id)
                for spot in ref_file.spots
                if not getattr(spot, "has_polygon_seed", False)
            ]
            if missing_polygon_seed_ids:
                QMessageBox.warning(
                    self,
                    "XMem2 Requires Polygon Mask",
                    "Offline XMem2 requires a polygon mask for every reference spot.\n\n"
                    "Missing polygon for spot IDs: "
                    + ", ".join(str(spot_id) for spot_id in missing_polygon_seed_ids),
                )
                return

            repo_path = self.xmem2_repo_path_edit.text().strip() or self.xmem2_settings.repo_path
            self.xmem2_settings.repo_path = repo_path
            self.xmem2_settings.python_executable = self.xmem2_python_executable_edit.text().strip() or None
            self.xmem2_settings.backend_mode = self.xmem2_backend_mode_combo.currentData()

            status = probe_xmem2_environment(
                models_dir=self.xmem2_models_dir,
                checkpoint_path=self.xmem2_settings.checkpoint_path,
                repo_path=self.xmem2_settings.repo_path,
            )
            python_executable = self.xmem2_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
            )
            if self.xmem2_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "XMem2 Unavailable",
                        format_xmem2_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "XMem2 Unavailable",
                        format_xmem2_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "XMem2 Unavailable",
                        format_xmem2_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `xmem2` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            xmem2_params = self._collect_xmem2_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_xmem2_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                xmem2_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Offline SAM2":
            ref_item = self.file_list_widget.currentItem()
            if not ref_item:
                QMessageBox.warning(self, "No Reference", "Offline SAM2 requires selecting a reference frame.")
                return
            ref_file = ref_item.data(Qt.ItemDataRole.UserRole)
            if not ref_file.spots:
                QMessageBox.warning(self, "No Spots", "The reference frame has no spots defined.")
                return

            repo_path = self.sam2_repo_path_edit.text().strip() or self.sam2_settings.repo_path
            self.sam2_settings.repo_path = repo_path
            self.sam2_settings.python_executable = self.sam2_python_executable_edit.text().strip() or None
            self.sam2_settings.backend_mode = self.sam2_backend_mode_combo.currentData()
            status = probe_sam2_environment(
                models_dir=self.sam2_models_dir,
                checkpoint_path=self.sam2_settings.checkpoint_path,
                repo_path=self.sam2_settings.repo_path,
                config_path=self.sam2_settings.config_path,
            )
            if not status.is_ready and self.sam2_settings.config_path:
                status = probe_sam2_environment(
                    models_dir=self.sam2_models_dir,
                    checkpoint_path=self.sam2_settings.checkpoint_path,
                    repo_path=self.sam2_settings.repo_path,
                    config_path=None,
                )
            self.sam2_settings.config_path = (
                str(status.config_path.resolve())
                if status.config_path is not None and status.config_path.exists()
                else None
            )

            python_executable = self.sam2_settings.python_executable
            python_path_ok = bool(
                python_executable and Path(python_executable).expanduser().exists()
            )
            structural_ready = bool(
                status.repo_path is not None
                and status.selected_checkpoint is not None
                and status.checkpoint_exists
                and status.config_exists
                and status.config_identifier is not None
            )

            if self.sam2_settings.backend_mode == "direct":
                if not status.is_ready:
                    QMessageBox.warning(
                        self,
                        "SAM2 Unavailable",
                        format_sam2_environment_report(status),
                    )
                    return
            else:
                if not structural_ready:
                    QMessageBox.warning(
                        self,
                        "SAM2 Unavailable",
                        format_sam2_environment_report(status),
                    )
                    return
                if not (status.is_ready or python_path_ok):
                    QMessageBox.warning(
                        self,
                        "SAM2 Unavailable",
                        format_sam2_environment_report(status)
                        + "\n\nSet the Python executable of the dedicated `sam2` env "
                        "or run Trackleed from that environment.",
                    )
                    return

            sam2_params = self._collect_sam2_tracking_params()

            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            worker = GenericWorker(
                track_spots_sam2_offline,
                file_map,
                sorted_ids,
                ref_file.id,
                sam2_params,
                cancel_event=cancel_event,
            )

        elif selected_scenario == "Detection Linking (Hungarian)":
            params = {
                "max_distance_px": self.hung_max_dist_spinbox.value(),
                "max_gap": self.hung_max_gap_spinbox.value(),
                "min_track_length": self.hung_min_len_spinbox.value(),
            }
            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            if not sorted_ids:
                QMessageBox.warning(self, "No Frames", "Load frames with detections before linking.")
                return
            if all((not lf.spots) for lf in file_map.values()):
                QMessageBox.warning(self, "No Detections", "Run Find Spots / Find All before linking.")
                return
            worker = GenericWorker(
                track_spots_hungarian,
                file_map,
                sorted_ids,
                params,
                cancel_event=cancel_event,
            )
        elif selected_scenario == "Detection Linking (Directional)":
            params = {
                "center_mode": "manual" if self.dir_center_mode_combo.currentIndex() == 1 else "auto",
                "manual_center": (
                    self.dir_center_x_spinbox.value(),
                    self.dir_center_y_spinbox.value(),
                ),
                "direction_mode": ["auto", "inward", "outward"][self.dir_direction_combo.currentIndex()],
                "lateral_angle_tolerance_deg": self.dir_lateral_tol_spinbox.value(),
                "ring_bin_px": self.dir_ring_bin_spinbox.value(),
                "base_max_distance_px": self.dir_base_max_dist_spinbox.value(),
                "dynamic_max_distance": self.dir_dynamic_max_checkbox.isChecked(),
                "dynamic_margin_px": self.dir_dynamic_margin_spinbox.value(),
                "max_gap": self.dir_max_gap_spinbox.value(),
                "max_misses": self.dir_max_misses_spinbox.value(),
                "min_track_length": self.dir_min_len_spinbox.value(),
            }
            file_map = {f.id: f for f in self.data_files if f.id is not None}
            sorted_ids = sorted(file_map.keys())
            if not sorted_ids:
                QMessageBox.warning(self, "No Frames", "Load frames with detections before linking.")
                return
            if all((not lf.spots) for lf in file_map.values()):
                QMessageBox.warning(self, "No Detections", "Run Find Spots / Find All before linking.")
                return
            worker = GenericWorker(
                track_spots_hungarian_directional,
                file_map,
                sorted_ids,
                params,
                cancel_event=cancel_event,
            )

        else:
            QMessageBox.information(self, "Not Implemented", "This scenario is not implemented yet.")
            return

        if worker:
            self._launch_worker_with_progress(worker, "Running tracking...", cancel_event)

    def _launch_worker_with_progress(self, worker: GenericWorker, title: str, cancel_event: Event):
        """Helper method to run any worker with a progress dialog."""
        progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.show()

        self.thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.thread)

        progress_dialog.canceled.connect(lambda: self._handle_worker_cancel(cancel_event, progress_dialog))
        self.worker.progress.connect(progress_dialog.setValue)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.error.connect(self.on_tracking_error)

        # Cleanup
        self.worker.finished.connect(progress_dialog.accept)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(progress_dialog.accept)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: setattr(self, "thread", None))
        self.thread.finished.connect(lambda: setattr(self, "worker", None))

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _handle_worker_cancel(self, cancel_event: Event, dialog: QProgressDialog):
        """Signal handler invoked when the user clicks Cancel on a progress dialog."""
        if cancel_event:
            cancel_event.set()
        dialog.setLabelText("Cancelling, please wait...")
        dialog.setEnabled(False)
        if self.thread:
            try:
                self.thread.requestInterruption()
            except RuntimeError:
                # Thread already deleted, nothing to cancel.
                pass

    def on_tracking_finished(self, tracks: List[Track] | None):
        """Slot to handle the successful completion of the tracking task."""
        if tracks is None:
            print("Tracking task was cancelled by the user.")
            QMessageBox.information(self, "Tracking Cancelled", "Tracking run was cancelled.")
            return
        print("Tracking task finished successfully.")
        self.tracking_results = tracks
        self.visualize_button.setEnabled(True)
        if hasattr(self, "analysis_button"):
            self.analysis_button.setEnabled(True)
        if hasattr(self, "positions_analysis_button"):
            self.positions_analysis_button.setEnabled(True)
        self._apply_tracks_to_files(tracks)
        current_item = self.file_list_widget.currentItem()
        if current_item:
            lf: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
            self.populate_spot_list(lf.spots)
            self.refresh_current_view()
        # Reset progress bar state in case it was cancelled
        QApplication.processEvents() # Process any pending events
        self.show_tracking_visualizer()

    def on_tracking_error(self, error_message: str):
        """Slot to handle errors from the tracking worker."""
        print(f"Tracking task failed: {error_message}")
        QMessageBox.critical(self, "Tracking Error", f"An error occurred during tracking:\n{error_message}")

    def show_tracking_visualizer(self):
        """
        Opens the tracking results visualization dialog and handles the returned data.
        """
        if not self.tracking_results:
            QMessageBox.information(self, "No Tracking Data", "Please run the tracking analysis first.")
            return
        
        current_roi_size = self.roi_size_spinbox.value()
        lk_win_size = self.lk_win_size_spinbox.value()
        lk_max_level = self.lk_max_level_spinbox.value()

        dialog = TrackingDialog(
            self.tracking_results, 
            self.data_files, 
            self.metadata_mask_list,
            roi_size=current_roi_size,
            lk_win_size=lk_win_size,  
            lk_max_level=lk_max_level,
            tapir_params=self._collect_tapir_tracking_params(),
            locotrack_params=self._collect_locotrack_tracking_params(),
            cotracker_params=self._collect_cotracker_tracking_params(),
            tapnext_params=self._collect_tapnext_tracking_params(),
            trackonr_params=self._collect_trackonr_tracking_params(),
            pipspp_params=self._collect_pipspp_tracking_params(),
            mftiq_params=self._collect_mftiq_tracking_params(),
            cutie_params=self._collect_cutie_tracking_params(),
            xmem2_params=self._collect_xmem2_tracking_params(),
            tomp_params=self._collect_tomp_tracking_params(),
            sam2_params=self._collect_sam2_tracking_params(),
            parent=self,
            save_session_callback=lambda tracks: self.save_session_dialog(tracks),
        )

        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.tracking_results = dialog.get_updated_tracks()
            print(f"Tracking data accepted. Final number of tracks: {len(self.tracking_results)}")

            QMessageBox.information(self, "Tracks Confirmed", "Corrected tracks have been saved.")
            self._apply_tracks_to_files(self.tracking_results)
            current_item = self.file_list_widget.currentItem()
            if current_item:
                lf: LeedFile = current_item.data(Qt.ItemDataRole.UserRole)
                self.populate_spot_list(lf.spots)
                self.refresh_current_view()
        else:
            print("Tracking dialog was closed without accepting changes.")

    def _apply_tracks_to_files(self, tracks: List[Track]):
        """Sync SpotROI instances from tracking results back into LeedFile objects for display."""
        spots_by_file: dict[int, list[SpotROI]] = {}
        for tr in tracks:
            for frame_id, spot in tr.spots_by_image_id.items():
                spots_by_file.setdefault(frame_id, []).append(spot)
        file_map = {f.id: f for f in self.data_files if f.id is not None}
        for fid, spots in spots_by_file.items():
            lf = file_map.get(fid)
            if lf:
                lf.spots = spots
    
    def populate_spot_list(self, spots: list[SpotROI]):
        """Populates the spot list widget with analysis results."""
        self.spot_list_widget.clear()
        show_polygon_hint = self._polygon_seed_scenario_active()
        for spot in spots:
            display_text = f"ID: {spot.id}, X: {spot.center_x:.1f}, Y: {spot.center_y:.1f}"
            if show_polygon_hint and getattr(spot, "has_polygon_seed", False):
                display_text += f" | poly:{len(getattr(spot, 'polygon_seed_points', []))}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, spot)
            self.spot_list_widget.addItem(item)
        self._update_spot_polygon_buttons()
    
    def refresh_current_view(self):
        """Refreshes the image display for the currently selected item."""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.display_selected_image(current_item, None)

    def save_session_dialog(self, tracks_override: List[Track] | None = None):
        """Open dialog to save current session (data, masks, spots, tracks)."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory or session before saving.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            "",
            "LEEDTrack Session (*.lts.json);;JSON (*.json);;All Files (*)",
        )
        if not path_str:
            return
        try:
            self._save_session(Path(path_str), tracks_override)
            QMessageBox.information(self, "Session Saved", f"Session saved to:\n{path_str}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session:\n{e}")

    def quick_save_session(self):
        """Quick-save session to save/quick_save.lts.json."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory or session before quick-saving.")
            return
        try:
            self.quick_save_dir.mkdir(parents=True, exist_ok=True)
            self._save_session(self.quick_save_path)
            QMessageBox.information(self, "Quick Save", f"Session saved to:\n{self.quick_save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Quick Save Error", f"Failed to quick-save session:\n{exc}")

    def _save_session(self, path: Path, tracks_override: List[Track] | None = None):
        tracks_list: List[Track] = []
        if isinstance(tracks_override, list):
            tracks_list = tracks_override
            self.tracking_results = tracks_override
            self._apply_tracks_to_files(tracks_override)
        elif self.tracking_results:
            tracks_list = self.tracking_results
        spot_settings = self._collect_spot_settings()
        save_session_to_file(
            path,
            self.data_files,
            self.metadata_mask_list,
            self.background_file,
            tracks_list,
            spot_settings,
            self._analysis_circle_to_dict(),
            self._analysis_ellipse_to_dict(),
        )

    def load_session_dialog(self):
        """Open dialog to load a session from disk."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "LEEDTrack Session (*.lts.json *.json);;All Files (*)",
        )
        if not path_str:
            return
        try:
            self._load_session(Path(path_str))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{e}")

    def quick_load_session(self):
        """Quick-load session from save/quick_save.lts.json."""
        if not self.quick_save_path.exists():
            QMessageBox.information(self, "Quick Load", "No quick-save file found.")
            return
        try:
            self._load_session(self.quick_save_path)
        except Exception as exc:
            QMessageBox.critical(self, "Quick Load Error", f"Failed to quick-load session:\n{exc}")

    def _load_session(self, path: Path):
        files, background, masks, tracks, spot_settings, analysis_circle, analysis_ellipse = load_session_from_file(path)

        missing_files = [str(f.filepath) for f in files if not f.filepath.exists()]
        if missing_files:
            QMessageBox.warning(
                self,
                "Missing Files",
                "Some image files could not be found and were skipped:\n" + "\n".join(missing_files),
            )
        valid_files = [f for f in files if f.filepath.exists()]
        self.data_files = valid_files

        if background and background.filepath.exists():
            self.background_file = background
        else:
            self.background_file = None

        self.metadata_mask_list = masks
        self.analysis_circle = self._analysis_circle_from_dict(analysis_circle)
        self.analysis_ellipse = self._analysis_ellipse_from_dict(analysis_ellipse)

        valid_ids = {f.id for f in self.data_files if f.id is not None}
        filtered_tracks: List[Track] = []
        for tr in tracks:
            filtered_map = {fid: spot for fid, spot in tr.spots_by_image_id.items() if fid in valid_ids}
            if filtered_map:
                filtered_tracks.append(Track(track_id=tr.track_id, spots_by_image_id=filtered_map))
        self.tracking_results = filtered_tracks
        if filtered_tracks:
            self._apply_tracks_to_files(filtered_tracks)
            self.visualize_button.setEnabled(True)
            if hasattr(self, "analysis_button"):
                self.analysis_button.setEnabled(True)
            if hasattr(self, "positions_analysis_button"):
                self.positions_analysis_button.setEnabled(True)

        self.populate_file_list()
        if spot_settings:
            self._apply_spot_settings(spot_settings)

        if self.file_list_widget.count() > 0:
            self.file_list_widget.setCurrentRow(0)
            self.display_selected_image(self.file_list_widget.item(0), None)
        QMessageBox.information(self, "Session Loaded", f"Loaded session from:\n{path}")

    def export_spots_to_yolo(self):
        """Export current spots to YOLO label files for all loaded frames."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory with LEED frames first.")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not target_dir:
            return
        target_path = Path(target_dir)
        labels_dir = target_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        for lf in self.data_files:
            if not lf.spots:
                skipped += 1
                continue
            img = load_tiff(lf.filepath)
            if img is None or img.size == 0:
                skipped += 1
                continue
            height, width = img.shape[:2]
            yolo_lines = []
            for spot in lf.spots:
                rect = spot.roi_rect
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                bw = w / width
                bh = h / height
                cx = float(min(max(cx, 0.0), 1.0))
                cy = float(min(max(cy, 0.0), 1.0))
                bw = float(min(max(bw, 1e-6), 1.0))
                bh = float(min(max(bh, 1e-6), 1.0))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_name = f"{lf.filename}.txt"
            (labels_dir / label_name).write_text("\n".join(yolo_lines))
            exported += 1

        QMessageBox.information(
            self,
            "YOLO Export",
            f"Exported labels for {exported} frames to {labels_dir}\nSkipped {skipped} frames without spots or unreadable images.",
        )

    def export_all_to_yolo(self):
        """Export all frames as YOLO labels + copy images (for easy val set augmentation)."""
        if not self.data_files:
            QMessageBox.information(self, "No Data", "Load a directory with LEED frames first.")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not target_dir:
            return
        target_path = Path(target_dir)
        labels_dir = target_path / "labels"
        images_dir = target_path / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        for lf in self.data_files:
            if not lf.spots:
                skipped += 1
                continue
            img_path = lf.filepath
            if not img_path.exists():
                skipped += 1
                continue
            img = load_tiff(img_path)
            if img is None or img.size == 0:
                skipped += 1
                continue
            height, width = img.shape[:2]
            yolo_lines = []
            for spot in lf.spots:
                rect = spot.roi_rect
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                bw = w / width
                bh = h / height
                cx = float(min(max(cx, 0.0), 1.0))
                cy = float(min(max(cy, 0.0), 1.0))
                bw = float(min(max(bw, 1e-6), 1.0))
                bh = float(min(max(bh, 1e-6), 1.0))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            stem = Path(lf.filename).stem
            label_name = f"{stem}.txt"
            image_name = lf.filename
            (labels_dir / label_name).write_text("\n".join(yolo_lines))
            shutil.copy2(img_path, images_dir / image_name)
            exported += 1

        QMessageBox.information(
            self,
            "YOLO Export",
            f"Exported {exported} frames to {target_path} (images + labels). Skipped {skipped} frames without spots or unreadable images.",
        )
