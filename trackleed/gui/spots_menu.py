"""Approved main-menu surface for spot operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QMenuBar, QWidget


@dataclass(frozen=True)
class SpotsMenuActions:
    detect: Callable[[], None]
    copy_to_all: Callable[[], None]
    normalize_all: Callable[[], None]


def add_spots_menu(
    menu_bar: QMenuBar,
    parent: QWidget,
    actions: SpotsMenuActions,
) -> QMenu:
    """Add the stable Spots menu without exposing detector implementations."""
    menu = menu_bar.addMenu("Spots")
    if menu is None:
        raise RuntimeError("Qt did not create the Spots menu")
    detect_action = QAction("Detect Spots...", parent)
    detect_action.triggered.connect(actions.detect)
    menu.addAction(detect_action)
    menu.addSeparator()
    copy_action = QAction("Copy Spots to All Frames", parent)
    copy_action.triggered.connect(actions.copy_to_all)
    menu.addAction(copy_action)
    normalize_action = QAction("Normalize ROIs (All Frames)", parent)
    normalize_action.triggered.connect(actions.normalize_all)
    menu.addAction(normalize_action)
    return menu
