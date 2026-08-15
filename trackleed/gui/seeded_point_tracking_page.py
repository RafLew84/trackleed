"""Generic Qt configuration page for typed point-tracker snapshots."""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from trackleed.logic.workflows.point_tracking import PointTrackerConfig

_PATH_FIELDS = {
    "models_dir",
    "repo_path",
    "checkpoint_path",
    "python_executable",
    "config_path",
}

_HIDDEN_FIELDS = {"subtract_background"}

class PointTrackerPage(QWidget):
    def __init__(self, config: PointTrackerConfig) -> None:
        super().__init__()
        self._config_type = type(config)
        self._original_values = {
            field.name: getattr(config, field.name) for field in fields(config)
        }
        self.field_widgets: dict[str, QWidget] = {}
        layout = QFormLayout(self)
        for field in fields(config):
            if field.name in _HIDDEN_FIELDS:
                continue
            value = self._original_values[field.name]
            widget = _field_widget(field.name, value)
            self.field_widgets[field.name] = widget
            layout.addRow(_field_label(field.name), widget)

    def config_snapshot(self) -> PointTrackerConfig:
        values = {
            name: _widget_value(name, widget, self._original_values[name])
            for name, widget in self.field_widgets.items()
        }
        return self._config_type(**values)


def _field_widget(name: str, value: Any) -> QWidget:
    if name in _PATH_FIELDS:
        return QLineEdit("" if value is None else str(value))
    if isinstance(value, Enum):
        combo = QComboBox()
        for enum_choice in type(value):
            combo.addItem(str(enum_choice.value), enum_choice)
        combo.setCurrentIndex(max(combo.findData(value), 0))
        return combo
    if isinstance(value, bool):
        checkbox = QCheckBox()
        checkbox.setChecked(value)
        return checkbox
    if isinstance(value, int):
        integer_spinbox = QSpinBox()
        integer_spinbox.setRange(-1_000_000_000, 1_000_000_000)
        integer_spinbox.setValue(value)
        return integer_spinbox
    if isinstance(value, float):
        float_spinbox = QDoubleSpinBox()
        float_spinbox.setDecimals(6)
        float_spinbox.setRange(-1_000_000_000.0, 1_000_000_000.0)
        float_spinbox.setValue(value)
        return float_spinbox
    if isinstance(value, tuple):
        return QLineEdit(" x ".join(str(item) for item in value))
    if value is None:
        return QLineEdit("")
    return QLineEdit(str(value))


def _widget_value(name: str, widget: QWidget, original: Any) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if not isinstance(widget, QLineEdit):
        raise TypeError(f"Unsupported point-config widget for {name}")
    text = widget.text().strip()
    if name in _PATH_FIELDS:
        if name == "models_dir" and not text:
            raise ValueError("Models directory is required")
        return Path(text) if text else None
    if isinstance(original, tuple):
        return _tuple_value(text, integer_items=name == "inference_resolution")
    if original is None and name == "inference_resolution":
        return _optional_resolution(text)
    return text


def _tuple_value(text: str, *, integer_items: bool) -> tuple[Any, ...]:
    parts = tuple(part.strip() for part in text.lower().replace("x", ",").split(","))
    if not parts or any(not part for part in parts):
        raise ValueError("Tuple settings require comma- or x-separated values")
    converter = int if integer_items else float
    return tuple(converter(part) for part in parts)


def _optional_resolution(text: str) -> int | tuple[int, int] | None:
    if not text:
        return None
    if "," in text or "x" in text.lower():
        values = _tuple_value(text, integer_items=True)
        if len(values) != 2:
            raise ValueError("Resolution requires one value or width x height")
        return values[0], values[1]
    return int(text)


def _field_label(name: str) -> str:
    return name.replace("_", " ").capitalize() + ":"
