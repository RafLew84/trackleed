from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow

from trackleed.gui.analysis_menu import AnalysisMenuActions, add_analysis_menu


class AnalysisMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_menu_has_approved_labels_and_delegates_actions(self) -> None:
        calls: list[str] = []
        window = QMainWindow()
        self.addCleanup(window.deleteLater)

        menu = add_analysis_menu(
            window.menuBar(),
            window,
            AnalysisMenuActions(
                tracks=lambda: calls.append("tracks"),
                positions=lambda: calls.append("positions"),
            ),
        )

        self.assertEqual(menu.title(), "Analysis")
        actions = [action for action in menu.actions() if not action.isSeparator()]
        self.assertEqual(
            [action.text() for action in actions],
            ["Track Analysis...", "Position Analysis..."],
        )
        for action in actions:
            action.trigger()
        self.assertEqual(calls, ["tracks", "positions"])


if __name__ == "__main__":
    unittest.main()
