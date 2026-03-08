from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.measure import EllipseModel

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class EllipseImageLabel(QLabel):
    """Interactive ellipse selector with draw/edit/point-fit modes and wheel zoom."""

    MODE_EDIT = "edit"
    MODE_DRAW = "draw"
    MODE_FIT = "fit"

    def __init__(
        self,
        initial_center: QPoint | None = None,
        initial_radius_x: int = 0,
        initial_radius_y: int = 0,
        initial_angle_deg: float = 0.0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.center: Optional[QPointF] = (
            QPointF(float(initial_center.x()), float(initial_center.y())) if initial_center else None
        )
        self.radius_x = max(0.0, float(initial_radius_x))
        self.radius_y = max(0.0, float(initial_radius_y))
        self.angle_deg = float(initial_angle_deg)

        self._base_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 12.0

        self._mode = self.MODE_EDIT if self.center and self.radius_x > 0 and self.radius_y > 0 else self.MODE_DRAW
        self._drag_kind: Optional[str] = None
        self._drag_start_img: Optional[QPointF] = None
        self._drag_start_center: Optional[QPointF] = None
        self._fit_points: list[QPointF] = []
        self._handle_radius_px = 6.0
        self._min_radius = 2.0

        self.setMouseTracking(True)

    def set_base_pixmap(self, pixmap: QPixmap):
        self._base_pixmap = pixmap
        self._refresh_scaled_pixmap()

    def set_mode(self, mode: str):
        self._mode = mode
        self._drag_kind = None
        self.update()

    def set_angle(self, angle_deg: float):
        self.angle_deg = float(angle_deg)
        self.update()

    def clear_fit_points(self):
        self._fit_points.clear()
        self.update()

    def _refresh_scaled_pixmap(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.clear()
            return
        scaled_w = max(1, int(round(self._base_pixmap.width() * self._zoom)))
        scaled_h = max(1, int(round(self._base_pixmap.height() * self._zoom)))
        scaled = self._base_pixmap.scaled(
            scaled_w,
            scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setFixedSize(scaled.size())
        self.update()

    def _to_image_pos(self, pos: QPoint) -> QPointF:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return QPointF(0.0, 0.0)
        x = float(pos.x()) / self._zoom
        y = float(pos.y()) / self._zoom
        x = max(0.0, min(float(self._base_pixmap.width() - 1), x))
        y = max(0.0, min(float(self._base_pixmap.height() - 1), y))
        return QPointF(x, y)

    def _to_display_pos(self, point: QPointF) -> QPointF:
        return QPointF(point.x() * self._zoom, point.y() * self._zoom)

    def _axis_unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        theta = np.deg2rad(self.angle_deg)
        major_u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        minor_v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
        return major_u, minor_v

    def _handle_positions(self) -> dict[str, QPointF]:
        if not self.center:
            return {}
        c = np.array([self.center.x(), self.center.y()], dtype=np.float64)
        major_u, minor_v = self._axis_unit_vectors()
        return {
            "major_pos": QPointF(*(c + major_u * self.radius_x)),
            "major_neg": QPointF(*(c - major_u * self.radius_x)),
            "minor_pos": QPointF(*(c + minor_v * self.radius_y)),
            "minor_neg": QPointF(*(c - minor_v * self.radius_y)),
        }

    def _hit_handle(self, image_pos: QPointF) -> Optional[str]:
        if not self.center or self.radius_x <= 0 or self.radius_y <= 0:
            return None
        threshold = max(3.0, self._handle_radius_px / max(self._zoom, 1e-6))
        px = image_pos.x()
        py = image_pos.y()
        for name, hp in self._handle_positions().items():
            dx = hp.x() - px
            dy = hp.y() - py
            if np.hypot(dx, dy) <= threshold:
                return name
        return None

    def _point_inside_ellipse(self, image_pos: QPointF) -> bool:
        if not self.center or self.radius_x <= 0 or self.radius_y <= 0:
            return False
        dx = image_pos.x() - self.center.x()
        dy = image_pos.y() - self.center.y()
        theta = np.deg2rad(self.angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        q = (xr * xr) / (self.radius_x * self.radius_x) + (yr * yr) / (self.radius_y * self.radius_y)
        return bool(q <= 1.0)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self._zoom = min(self._zoom_max, self._zoom * 1.15)
        else:
            self._zoom = max(self._zoom_min, self._zoom / 1.15)
        self._refresh_scaled_pixmap()
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        # Draw fit points.
        fit_pen = QPen(QColor(200, 0, 255, 230))
        fit_pen.setWidth(2)
        painter.setPen(fit_pen)
        for p in self._fit_points:
            dp = self._to_display_pos(p)
            painter.drawEllipse(dp, 3, 3)

        # Draw current ellipse.
        if self.center and self.radius_x > 0 and self.radius_y > 0:
            ellipse_pen = QPen(QColor(255, 165, 0, 230))
            ellipse_pen.setWidth(2)
            painter.setPen(ellipse_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            dc = self._to_display_pos(self.center)
            painter.translate(dc)
            painter.rotate(self.angle_deg)
            drx = int(round(self.radius_x * self._zoom))
            dry = int(round(self.radius_y * self._zoom))
            painter.drawEllipse(QPoint(0, 0), drx, dry)
            painter.resetTransform()

            # Draw edit handles.
            handle_pen = QPen(QColor(0, 220, 255, 230))
            handle_pen.setWidth(2)
            painter.setPen(handle_pen)
            handle_size = self._handle_radius_px
            for hp in self._handle_positions().values():
                dp = self._to_display_pos(hp)
                painter.drawRect(
                    int(round(dp.x() - handle_size)),
                    int(round(dp.y() - handle_size)),
                    int(round(2 * handle_size)),
                    int(round(2 * handle_size)),
                )

    def _fit_from_points(self) -> bool:
        if len(self._fit_points) < 4:
            return False

        pts = np.array([[p.x(), p.y()] for p in self._fit_points], dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] < 4:
            return False

        fitted = False
        center_x = center_y = radius_x = radius_y = theta_deg = 0.0

        # Preferred robust fit when enough points are available.
        if pts.shape[0] >= 5:
            model = EllipseModel()
            if model.estimate(pts) and model.params is not None:
                xc, yc, a, b, theta = model.params
                if np.isfinite(xc) and np.isfinite(yc) and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                    center_x = float(xc)
                    center_y = float(yc)
                    radius_x = float(a)
                    radius_y = float(b)
                    theta_deg = float(np.degrees(theta))
                    fitted = True

        # Fallback for exactly 4 points or failed geometric fit.
        if not fitted:
            center = np.mean(pts, axis=0)
            centered = pts - center
            cov = np.cov(centered.T)
            if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                return False
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evecs = evecs[:, order]
            major_u = evecs[:, 0]
            minor_v = evecs[:, 1]
            u_coords = centered @ major_u
            v_coords = centered @ minor_v

            # Solve (u^2)/rx^2 + (v^2)/ry^2 ~= 1 for 1/rx^2 and 1/ry^2.
            A = np.column_stack((u_coords**2, v_coords**2))
            b = np.ones(pts.shape[0], dtype=np.float64)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                inv_rx2, inv_ry2 = float(sol[0]), float(sol[1])
            except np.linalg.LinAlgError:
                inv_rx2, inv_ry2 = 0.0, 0.0

            if inv_rx2 > 0 and inv_ry2 > 0:
                radius_x = float(np.sqrt(1.0 / inv_rx2))
                radius_y = float(np.sqrt(1.0 / inv_ry2))
            else:
                radius_x = float(max(np.max(np.abs(u_coords)), 1.0))
                radius_y = float(max(np.max(np.abs(v_coords)), 1.0))

            center_x = float(center[0])
            center_y = float(center[1])
            theta_deg = float(np.degrees(np.arctan2(major_u[1], major_u[0])))
            fitted = True

        if not fitted:
            return False

        if radius_y > radius_x:
            radius_x, radius_y = radius_y, radius_x
            theta_deg += 90.0

        self.center = QPointF(center_x, center_y)
        self.radius_x = max(self._min_radius, float(radius_x))
        self.radius_y = max(self._min_radius, float(radius_y))
        self.angle_deg = float((theta_deg + 180.0) % 360.0 - 180.0)
        self.update()
        return True

    def mousePressEvent(self, event: QMouseEvent):
        image_pos = self._to_image_pos(event.pos())

        if event.button() == Qt.MouseButton.RightButton:
            if self._mode == self.MODE_FIT:
                self._fit_points.clear()
            else:
                self.center = None
                self.radius_x = 0.0
                self.radius_y = 0.0
            self._drag_kind = None
            self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._mode == self.MODE_FIT:
            self._fit_points.append(image_pos)
            if len(self._fit_points) > 20:
                self._fit_points.pop(0)
            self._fit_from_points()
            self.update()
            return

        if self._mode == self.MODE_DRAW:
            self.center = image_pos
            self.radius_x = 0.0
            self.radius_y = 0.0
            self._drag_kind = "draw"
            self.update()
            return

        # MODE_EDIT
        if self.center and self.radius_x > 0 and self.radius_y > 0:
            handle = self._hit_handle(image_pos)
            if handle is not None:
                self._drag_kind = handle
                self.update()
                return
            if self._point_inside_ellipse(image_pos):
                self._drag_kind = "move"
                self._drag_start_img = image_pos
                self._drag_start_center = QPointF(self.center.x(), self.center.y())
                self.update()
                return

        # If nothing selected in edit mode, start drawing a new ellipse.
        self.center = image_pos
        self.radius_x = 0.0
        self.radius_y = 0.0
        self._drag_kind = "draw"
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self._drag_kind is None:
            return

        image_pos = self._to_image_pos(event.pos())
        if self._drag_kind == "draw" and self.center:
            self.radius_x = max(self._min_radius, abs(image_pos.x() - self.center.x()))
            self.radius_y = max(self._min_radius, abs(image_pos.y() - self.center.y()))
            self.update()
            return

        if self._drag_kind == "move" and self.center and self._drag_start_img and self._drag_start_center:
            dx = image_pos.x() - self._drag_start_img.x()
            dy = image_pos.y() - self._drag_start_img.y()
            self.center = QPointF(self._drag_start_center.x() + dx, self._drag_start_center.y() + dy)
            self.update()
            return

        if self.center and self._drag_kind in {"major_pos", "major_neg", "minor_pos", "minor_neg"}:
            major_u, minor_v = self._axis_unit_vectors()
            vec = np.array([image_pos.x() - self.center.x(), image_pos.y() - self.center.y()], dtype=np.float64)
            if self._drag_kind in {"major_pos", "major_neg"}:
                self.radius_x = max(self._min_radius, abs(float(np.dot(vec, major_u))))
            else:
                self.radius_y = max(self._min_radius, abs(float(np.dot(vec, minor_v))))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_kind = None
            self._drag_start_img = None
            self._drag_start_center = None
            self.update()

    def get_ellipse(self):
        if self.center and self.radius_x > 0 and self.radius_y > 0:
            center = QPoint(int(round(self.center.x())), int(round(self.center.y())))
            return center, int(round(self.radius_x)), int(round(self.radius_y)), float(self.angle_deg)
        return None, 0, 0, 0.0


class EllipseMaskDialog(QDialog):
    """Dialog to pick/edit a single analysis ellipse."""

    def __init__(
        self,
        pixmap: QPixmap,
        existing_center: QPoint | None,
        existing_radius_x: int,
        existing_radius_y: int,
        existing_angle_deg: float,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Set Analysis Ellipse (Wheel: Zoom)")
        self.setMinimumSize(760, 620)

        layout = QVBoxLayout(self)

        self.image_label = EllipseImageLabel(
            initial_center=existing_center,
            initial_radius_x=existing_radius_x,
            initial_radius_y=existing_radius_y,
            initial_angle_deg=existing_angle_deg,
        )
        self.image_label.set_base_pixmap(pixmap)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Edit (move/resize)", EllipseImageLabel.MODE_EDIT)
        self.mode_combo.addItem("Draw New (drag)", EllipseImageLabel.MODE_DRAW)
        self.mode_combo.addItem("Fit From Points (>=4)", EllipseImageLabel.MODE_FIT)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode_combo)

        self.clear_points_button = QPushButton("Clear Fit Points")
        self.clear_points_button.clicked.connect(self.image_label.clear_fit_points)
        controls.addWidget(self.clear_points_button)

        controls.addWidget(QLabel("Rotation (deg):"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSingleStep(1.0)
        self.angle_spin.setValue(float(existing_angle_deg))
        self.angle_spin.valueChanged.connect(self.image_label.set_angle)
        controls.addWidget(self.angle_spin)
        controls.addStretch()
        layout.addLayout(controls)

        hint = QLabel(
            "Edit: przeciągnij wewnątrz elipsy, żeby przesunąć; przeciągnij uchwyty, żeby zmienić rozmiar. "
            "Fit: kliknij >=4 punktów na obwodzie elipsy."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData()
        if isinstance(mode, str):
            self.image_label.set_mode(mode)

    def get_ellipse(self):
        return self.image_label.get_ellipse()
