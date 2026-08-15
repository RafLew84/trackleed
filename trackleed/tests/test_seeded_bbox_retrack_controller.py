from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_retrack_controller import launch_seeded_retrack
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
        files = {1: _file(1), 2: _file(2)}
        track = Track(
            7,
            {1: SpotROI(7, QRect(12, 13, 8, 9), 16.0, 17.0, 1.0, "manual")},
        )
        launched: dict[str, object] = {}
        dialog = SimpleNamespace(
            _get_retrack_target=lambda: (track, 1),
            seeded_tracking_configs={
                "ostrack_offline": OstrackBoundingBoxConfig(roi_size=24)
            },
            file_map=files,
            sorted_frame_ids=[1, 2],
            launch_retrack_worker=lambda worker, **kwargs: launched.update(
                worker=worker,
                **kwargs,
            ),
        )

        with patch(
            "trackleed.gui.seeded_retrack_controller.GenericWorker",
            side_effect=lambda function, *args: SimpleNamespace(
                function=function,
                args=args,
            ),
        ):
            launch_seeded_retrack(dialog, "ostrack_offline", 7)

        worker = launched["worker"]
        request = worker.args[0]
        self.assertEqual(request.method_id, "ostrack_offline")
        self.assertIs(
            request.seed_representation,
            SeedRepresentation.BOUNDING_BOX,
        )
        self.assertEqual(request.track.observations[0].roi.width, 8)
        self.assertIn("Offline OSTrack", launched["title"])


if __name__ == "__main__":
    unittest.main()
