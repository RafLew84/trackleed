"""Approved main-menu surface for tracking operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QMenuBar, QWidget


@dataclass(frozen=True)
class TrackingMenuActions:
    seeded: Callable[[], None]
    hungarian: Callable[[], None]
    directional: Callable[[], None]
    review: Callable[[], None]


def add_tracking_menu(
    menu_bar: QMenuBar,
    parent: QWidget,
    actions: TrackingMenuActions,
) -> QMenu:
    """Add tracking actions without exposing algorithm configuration."""
    menu = menu_bar.addMenu("Tracking")
    if menu is None:
        raise RuntimeError("Qt did not create the Tracking menu")
    seeded_action = QAction("Seeded Spot Tracking...", parent)
    seeded_action.triggered.connect(actions.seeded)
    menu.addAction(seeded_action)
    hungarian_action = QAction("Detection Linking (Hungarian)...", parent)
    hungarian_action.triggered.connect(actions.hungarian)
    menu.addAction(hungarian_action)
    directional_action = QAction("Detection Linking (Directional)...", parent)
    directional_action.triggered.connect(actions.directional)
    menu.addAction(directional_action)
    menu.addSeparator()
    review_action = QAction("Review / Edit Tracks", parent)
    review_action.triggered.connect(actions.review)
    menu.addAction(review_action)
    return menu
