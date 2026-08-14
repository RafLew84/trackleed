from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow

from trackleed.gui.spots_menu import SpotsMenuActions, add_spots_menu


class SpotsMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_menu_has_approved_labels_and_delegates_actions(self) -> None:
        calls: list[str] = []
        window = QMainWindow()
        self.addCleanup(window.deleteLater)

        menu = add_spots_menu(
            window.menuBar(),
            window,
            SpotsMenuActions(
                detect=lambda: calls.append("detect"),
                copy_to_all=lambda: calls.append("copy"),
                normalize_all=lambda: calls.append("normalize"),
            ),
        )

        self.assertEqual(menu.title(), "Spots")
        menu_actions = menu.actions()
        self.assertTrue(menu_actions[1].isSeparator())
        actionable = [action for action in menu_actions if not action.isSeparator()]
        self.assertEqual(
            [action.text() for action in actionable],
            [
                "Detect Spots...",
                "Copy Spots to All Frames",
                "Normalize ROIs (All Frames)",
            ],
        )
        for action in actionable:
            action.trigger()
        self.assertEqual(calls, ["detect", "copy", "normalize"])


if __name__ == "__main__":
    unittest.main()
