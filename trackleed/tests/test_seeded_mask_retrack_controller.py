from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_retrack_controller import (
    SeededRetrackLaunchRequest,
    launch_seeded_retrack,
)
from trackleed.logic.generic_worker import GenericWorker
from trackleed.logic.workflows.mask_tracking import CutieMaskConfig
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


class SeededMaskRetrackControllerTests(unittest.TestCase):
    def test_review_retrack_builds_public_mask_workflow_worker(self) -> None:
        files = (_file(1), _file(2))
        seed = SpotROI(
            7,
            QRect(12, 13, 8, 9),
            16.0,
            17.0,
            1.0,
            "cutie_offline",
            [(12.0, 13.0), (20.0, 13.0), (16.0, 22.0)],
        )
        track = Track(7, {1: seed})
        launched: dict[str, object] = {}
        host = SimpleNamespace(
            launch_retrack_worker=lambda worker, **kwargs: launched.update(
                worker=worker,
                **kwargs,
            ),
        )
        launch_request = SeededRetrackLaunchRequest(
            method_id="cutie_offline",
            files=files,
            track=track,
            start_frame_id=1,
            config=CutieMaskConfig(roi_size=24),
        )

        launch_seeded_retrack(host, launch_request)

        worker = cast(GenericWorker, launched["worker"])
        request = worker.args[0]
        self.assertEqual(request.method_id, "cutie_offline")
        self.assertIs(request.seed_representation, SeedRepresentation.MASK)
        self.assertEqual(
            request.track.observations[0].polygon_seed_points,
            ((12.0, 13.0), (20.0, 13.0), (16.0, 22.0)),
        )
        self.assertIn("Offline Cutie", cast(str, launched["title"]))


if __name__ == "__main__":
    unittest.main()
