from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_retrack_controller import (
    SeededRetrackLaunchRequest,
    launch_seeded_retrack,
)
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.workflows.bbox_tracking import OstrackBoundingBoxConfig
from trackleed.logic.workflows.seeded_tracking import SeedRepresentation


def _file(frame_id: int) -> LeedFile:
    return LeedFile(
        Path(f"frame-{frame_id}.tif"),
        f"frame-{frame_id}.tif",
        False,
        "Cu",
        111,
        "G",
        300.0,
        80.0,
        frame_id,
        None,
        [],
    )


class SeededBboxRetrackControllerTests(unittest.TestCase):
    def test_review_retrack_builds_public_bbox_workflow_worker(self) -> None:
        files = (_file(1), _file(2))
        track = Track(
            7,
            {1: SpotROI(7, QRect(12, 13, 8, 9), 16.0, 17.0, 1.0, "manual")},
        )
        launched: dict[str, object] = {}
        host = SimpleNamespace(
            launch_retrack_worker=lambda worker, **kwargs: launched.update(
                worker=worker,
                **kwargs,
            ),
        )
        launch_request = SeededRetrackLaunchRequest(
            method_id="ostrack_offline",
            files=files,
            track=track,
            start_frame_id=1,
            config=OstrackBoundingBoxConfig(roi_size=24),
        )

        launch_seeded_retrack(host, launch_request)

        worker = cast(GenericWorker, launched["worker"])
        request = worker.args[0]
        self.assertEqual(request.method_id, "ostrack_offline")
        self.assertIs(
            request.seed_representation,
            SeedRepresentation.BOUNDING_BOX,
        )
        self.assertEqual(request.track.observations[0].roi.width, 8)
        self.assertIn("Offline OSTrack", cast(str, launched["title"]))

    def test_missing_review_configuration_does_not_launch_a_worker(self) -> None:
        files = (_file(1), _file(2))
        track = Track(
            7,
            {1: SpotROI(7, QRect(12, 13, 8, 9), 16.0, 17.0, 1.0, "manual")},
        )
        launched: list[object] = []
        host = SimpleNamespace(
            launch_retrack_worker=lambda worker, **_kwargs: launched.append(worker)
        )
        launch_request = SeededRetrackLaunchRequest(
            method_id="ostrack_offline",
            files=files,
            track=track,
            start_frame_id=1,
            config=None,
        )

        with patch(
            "trackleed.gui.seeded_retrack_controller.QMessageBox.warning"
        ) as warning:
            launch_seeded_retrack(host, launch_request)

        self.assertEqual(launched, [])
        self.assertIn("configuration", warning.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
