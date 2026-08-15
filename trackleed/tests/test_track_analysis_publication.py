from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.logic.track_analysis_compatibility import (
    AnalysisPublication,
    publish_analysis,
)
from trackleed.logic.workflows.track_analysis import FitSet
from trackleed.logic.workflows.track_geometry import GeometryResult


def _file(frame_id: int, marker: str) -> LeedFile:
    frame = LeedFile(
        filepath=Path(f"frame_{frame_id:04d}.tif"),
        filename=f"frame_{frame_id:04d}.tif",
        is_background=False,
        element="Au",
        surface=111,
        mode="G",
        temperature=300.0,
        energy=122.0,
        id=frame_id,
        coverage=0.0,
    )
    frame.pattern_center = {"marker": marker}
    frame.post_analysis_results = {"marker": marker}
    return frame


def _track(track_id: int, frame_ids: tuple[int, ...], marker: str) -> Track:
    track = Track(
        track_id=track_id,
        spots_by_image_id={
            frame_id: cast(SpotROI, object()) for frame_id in frame_ids
        },
    )
    track.analysis_results = {
        frame_id: {"marker": marker} for frame_id in frame_ids
    }
    return track


def _publication(
    tracks: list[Track],
    files: list[tuple[int, LeedFile]],
) -> AnalysisPublication:
    return AnalysisPublication(
        tracks=tuple(tracks),
        files=tuple(files),
        fit_set=FitSet(()),
        geometry_result=cast(GeometryResult, object()),
    )


def _legacy_state(
    tracks: list[Track],
    files: dict[int, LeedFile],
) -> object:
    return deepcopy(
        (
            [track.analysis_results for track in tracks],
            {
                frame_id: (
                    frame.pattern_center,
                    frame.post_analysis_results,
                )
                for frame_id, frame in files.items()
            },
        )
    )


class TrackAnalysisPublicationTests(unittest.TestCase):
    def test_result_frame_without_files_rolls_back_every_legacy_target(self) -> None:
        targets = [_track(1, (1,), "old-track")]
        target_files = {1: _file(1, "old-frame")}
        before = _legacy_state(targets, target_files)
        source_track = _track(1, (1,), "new-track")
        source_track.analysis_results[2] = {"marker": "orphan-result"}
        publication = _publication(
            [source_track],
            [(1, _file(1, "new-frame"))],
        )

        with self.assertRaisesRegex(ValueError, "frame"):
            publish_analysis(publication, targets, target_files)

        self.assertEqual(_legacy_state(targets, target_files), before)

    def test_successful_publication_detaches_all_mutable_staging_values(self) -> None:
        targets = [_track(1, (1,), "old-track")]
        target_files = {1: _file(1, "old-frame")}
        source_track = _track(1, (1,), "new-track")
        source_track.analysis_results = {
            1: {"fit": {"volume": 12.5}, "radius": 7.0}
        }
        source_file = _file(1, "new-frame")
        source_file.pattern_center = {"center_x": 10.0, "center_y": 11.0}
        source_file.post_analysis_results = {
            "avg_radius": 7.0,
            "quality": {"accepted": True},
        }
        publication = _publication([source_track], [(1, source_file)])

        publish_analysis(publication, targets, target_files)

        self.assertEqual(targets[0].analysis_results, source_track.analysis_results)
        self.assertEqual(target_files[1].pattern_center, source_file.pattern_center)
        self.assertEqual(
            target_files[1].post_analysis_results,
            source_file.post_analysis_results,
        )
        self.assertIsNot(targets[0].analysis_results, source_track.analysis_results)
        self.assertIsNot(
            targets[0].analysis_results[1]["fit"],
            source_track.analysis_results[1]["fit"],
        )
        self.assertIsNot(target_files[1].pattern_center, source_file.pattern_center)
        self.assertIsNot(
            target_files[1].post_analysis_results["quality"],
            source_file.post_analysis_results["quality"],
        )

        source_track.analysis_results[1]["fit"]["volume"] = -1.0
        source_file.pattern_center["center_x"] = -1.0
        source_file.post_analysis_results["quality"]["accepted"] = False
        self.assertEqual(targets[0].analysis_results[1]["fit"]["volume"], 12.5)
        self.assertEqual(target_files[1].pattern_center["center_x"], 10.0)
        self.assertTrue(
            target_files[1].post_analysis_results["quality"]["accepted"]
        )

    def test_duplicate_source_frame_rolls_back_every_legacy_target(self) -> None:
        targets = [_track(1, (1,), "old-track")]
        target_files = {1: _file(1, "old-frame")}
        before = _legacy_state(targets, target_files)
        publication = _publication(
            [_track(1, (1,), "new-track")],
            [
                (1, _file(1, "new-frame-1")),
                (1, _file(1, "new-frame-1-duplicate")),
            ],
        )

        with self.assertRaisesRegex(ValueError, "frame"):
            publish_analysis(publication, targets, target_files)

        self.assertEqual(_legacy_state(targets, target_files), before)

    def test_invalid_track_id_sets_roll_back_every_legacy_target(self) -> None:
        cases = (
            ("different IDs", (1, 2), (1, 3)),
            ("duplicate source ID", (1, 1), (1, 2)),
            ("duplicate target ID", (1, 2), (1, 1)),
        )
        for label, source_ids, target_ids in cases:
            with self.subTest(label):
                targets = [
                    _track(track_id, (1,), f"old-track-{index}")
                    for index, track_id in enumerate(target_ids)
                ]
                target_files = {1: _file(1, "old-frame")}
                before = _legacy_state(targets, target_files)
                publication = _publication(
                    [
                        _track(track_id, (1,), f"new-track-{index}")
                        for index, track_id in enumerate(source_ids)
                    ],
                    [(1, _file(1, "new-frame"))],
                )

                with self.assertRaisesRegex(ValueError, "track ID"):
                    publish_analysis(publication, targets, target_files)

                self.assertEqual(_legacy_state(targets, target_files), before)

    def test_reordered_tracks_publish_analysis_by_track_id(self) -> None:
        targets = [
            _track(1, (1,), "old-track-1"),
            _track(2, (1,), "old-track-2"),
        ]
        target_files = {1: _file(1, "old-frame")}
        publication = _publication(
            [
                _track(2, (1,), "new-track-2"),
                _track(1, (1,), "new-track-1"),
            ],
            [(1, _file(1, "new-frame"))],
        )

        publish_analysis(publication, targets, target_files)

        self.assertEqual(
            targets[0].analysis_results,
            {1: {"marker": "new-track-1"}},
        )
        self.assertEqual(
            targets[1].analysis_results,
            {1: {"marker": "new-track-2"}},
        )

    def test_track_count_mismatch_rolls_back_every_legacy_target(self) -> None:
        targets = [_track(1, (1,), "old-track")]
        target_files = {1: _file(1, "old-frame")}
        before = _legacy_state(targets, target_files)
        publication = _publication(
            [
                _track(1, (1,), "new-track-1"),
                _track(2, (1,), "new-track-2"),
            ],
            [(1, _file(1, "new-frame"))],
        )

        with self.assertRaises(ValueError):
            publish_analysis(publication, targets, target_files)

        self.assertEqual(_legacy_state(targets, target_files), before)

    def test_missing_target_frame_rolls_back_tracks_and_existing_frames(self) -> None:
        targets = [_track(1, (1, 2), "old-track")]
        target_files = {1: _file(1, "old-frame")}
        before = _legacy_state(targets, target_files)
        publication = _publication(
            [_track(1, (1, 2), "new-track")],
            [
                (1, _file(1, "new-frame-1")),
                (2, _file(2, "new-frame-2")),
            ],
        )

        with self.assertRaisesRegex(ValueError, "frame"):
            publish_analysis(publication, targets, target_files)

        self.assertEqual(_legacy_state(targets, target_files), before)

    def test_missing_source_frame_rolls_back_every_legacy_target(self) -> None:
        targets = [_track(1, (1,), "old-track")]
        target_files = {1: _file(1, "old-frame")}
        before = _legacy_state(targets, target_files)
        publication = _publication(
            [_track(1, (1,), "new-track")],
            [],
        )

        with self.assertRaisesRegex(ValueError, "frame"):
            publish_analysis(publication, targets, target_files)

        self.assertEqual(_legacy_state(targets, target_files), before)


if __name__ == "__main__":
    unittest.main()
