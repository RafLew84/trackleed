from pathlib import Path

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.spot_detection_adapter import (
    build_detection_request,
    publish_detection_result,
)
from trackleed.logic.workflows.spot_detection import (
    DetectedSpot,
    DetectionResult,
    FrameDetections,
    RectSnapshot,
)


def _leed_file(frame_id: int) -> LeedFile:
    return LeedFile(
        filepath=Path(f"frame-{frame_id}.tif"),
        filename=f"frame-{frame_id}.tif",
        is_background=False,
        element="Cu",
        surface=111,
        mode="G",
        temperature=300.0,
        energy=80.0,
        id=frame_id,
        coverage=None,
    )


def _spot(spot_id: int, x: int) -> SpotROI:
    return SpotROI(spot_id, QRect(x, 2, 8, 8), float(x + 4), 6.0, 1.0, "auto")


def test_request_builder_captures_immutable_values_from_gui_models() -> None:
    leed_file = _leed_file(4)
    mask = QRect(1, 2, 3, 4)
    settings = {
        "backend": "Classic",
        "classic": {
            "blur_sigma": 3.0,
            "wth_radius": 4,
            "min_distance": 5,
            "threshold_rel": 0.2,
            "roi_size": 16,
        },
    }

    request = build_detection_request(
        files=(leed_file,),
        spot_settings=settings,
        background_file=None,
        metadata_masks=(mask,),
        analysis_circle=(10, 11, 9),
        analysis_ellipse=None,
    )
    leed_file.filepath = Path("changed.tif")
    settings["classic"]["roi_size"] = 64
    mask.setWidth(99)

    assert request.method_id == "classic"
    assert request.frames[0].source_path == Path("frame-4.tif")
    assert request.config.roi_size == 16
    assert request.metadata_masks[0].width == 3
    assert request.circle_region is not None
    assert request.circle_region.radius == 9


def test_publish_replaces_requested_spots_without_touching_tracks_or_aliasing() -> None:
    first = _leed_file(1)
    second = _leed_file(2)
    tracked_spot = _spot(8, 10)
    first.spots = [tracked_spot]
    second.spots = [_spot(9, 20)]
    track = Track(track_id=3, spots_by_image_id={1: tracked_spot})
    result = DetectionResult(
        frames=(
            FrameDetections(
                frame_id=0,
                spots=(
                    DetectedSpot(
                        id=1,
                        roi=RectSnapshot(3, 4, 8, 8),
                        center_x=7.0,
                        center_y=8.0,
                    ),
                ),
            ),
            FrameDetections(frame_id=1, spots=()),
        )
    )

    publish_detection_result((first, second), result)

    assert len(first.spots) == 1
    assert second.spots == []
    assert track.spots_by_image_id == {1: tracked_spot}
    assert first.spots[0] is not tracked_spot
    assert first.spots[0] is not track.spots_by_image_id[1]


def test_publish_rejects_incomplete_result_before_mutating_any_frame() -> None:
    first = _leed_file(1)
    second = _leed_file(2)
    first.spots = [_spot(1, 10)]
    second.spots = [_spot(2, 20)]
    original = (first.spots, second.spots)
    incomplete = DetectionResult(frames=(FrameDetections(frame_id=0, spots=()),))

    try:
        publish_detection_result((first, second), incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete results must fail before publication")

    assert first.spots is original[0]
    assert second.spots is original[1]
