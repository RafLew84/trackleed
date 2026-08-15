"""Public Analysis menu composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QMenuBar, QWidget


@dataclass(frozen=True)
class AnalysisMenuActions:
    tracks: Callable[[], None]
    positions: Callable[[], None]


def add_analysis_menu(
    menu_bar: QMenuBar,
    parent: QWidget,
    actions: AnalysisMenuActions,
) -> QMenu:
    menu = menu_bar.addMenu("Analysis")

    track_action = QAction("Track Analysis...", parent)
    track_action.triggered.connect(actions.tracks)
    menu.addAction(track_action)

    position_action = QAction("Position Analysis...", parent)
    position_action.triggered.connect(actions.positions)
    menu.addAction(position_action)

    return menu
