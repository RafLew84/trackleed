from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow

from trackleed.gui.tracking_menu import TrackingMenuActions, add_tracking_menu


class TrackingMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_menu_uses_approved_seeded_tracking_and_review_labels(self) -> None:
        window = QMainWindow()
        calls: list[str] = []
        menu = add_tracking_menu(
            window.menuBar(),
            window,
            TrackingMenuActions(
                seeded=lambda: calls.append("seeded"),
                hungarian=lambda: calls.append("hungarian"),
                directional=lambda: calls.append("directional"),
                review=lambda: calls.append("review"),
            ),
        )
        self.addCleanup(window.deleteLater)

        self.assertEqual(menu.title(), "Tracking")
        labels = [
            action.text() for action in menu.actions() if not action.isSeparator()
        ]
        self.assertEqual(
            labels,
            [
                "Seeded Spot Tracking...",
                "Detection Linking (Hungarian)...",
                "Detection Linking (Directional)...",
                "Review / Edit Tracks",
            ],
        )
        menu.actions()[0].trigger()
        menu.actions()[1].trigger()
        menu.actions()[2].trigger()
        menu.actions()[-1].trigger()

        self.assertEqual(calls, ["seeded", "hungarian", "directional", "review"])


if __name__ == "__main__":
    unittest.main()
