import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track

SESSION_VERSION = 1


def _rect_to_dict(rect: QRect) -> Dict:
    return {"x": rect.x(), "y": rect.y(), "w": rect.width(), "h": rect.height()}


def _rect_from_dict(data: Dict) -> QRect:
    return QRect(int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))


def _spot_to_dict(spot: SpotROI) -> Dict:
    return {
        "id": spot.id,
        "roi": _rect_to_dict(spot.roi_rect),
        "center_x": float(spot.center_x),
        "center_y": float(spot.center_y),
        "intensity": float(spot.intensity) if spot.intensity is not None else 0.0,
        "source": spot.source,
    }


def _spot_from_dict(data: Dict) -> SpotROI:
    return SpotROI(
        id=int(data["id"]),
        roi_rect=_rect_from_dict(data["roi"]),
        center_x=float(data["center_x"]),
        center_y=float(data["center_y"]),
        intensity=float(data.get("intensity", 0)),
        source=data.get("source", "auto"),
    )


def _track_to_dict(track: Track) -> Dict:
    return {
        "track_id": track.track_id,
        "spots_by_image_id": {str(fid): _spot_to_dict(s) for fid, s in track.spots_by_image_id.items()},
    }


def _track_from_dict(data: Dict) -> Track:
    return Track(
        track_id=int(data["track_id"]),
        spots_by_image_id={int(fid): _spot_from_dict(s) for fid, s in data.get("spots_by_image_id", {}).items()},
    )


def _clean_number(val):
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return val


def _clean_dict_numbers(d: Dict | None) -> Dict | None:
    if d is None:
        return None
    return {k: _clean_number(v) for k, v in d.items()}


def _leedfile_to_dict(lf: LeedFile) -> Dict:
    return {
        "filepath": str(lf.filepath),
        "filename": lf.filename,
        "is_background": lf.is_background,
        "element": lf.element,
        "surface": lf.surface,
        "mode": lf.mode,
        "temperature": _clean_number(lf.temperature),
        "energy": _clean_number(lf.energy),
        "id": lf.id,
        "coverage": _clean_number(lf.coverage),
        "pattern_center": _clean_dict_numbers(lf.pattern_center),
        "spots": [_spot_to_dict(s) for s in lf.spots],
    }


def _leedfile_from_dict(data: Dict) -> LeedFile:
    lf = LeedFile(
        filepath=Path(data["filepath"]),
        filename=data["filename"],
        is_background=bool(data["is_background"]),
        element=data.get("element", ""),
        surface=int(data.get("surface", 0)),
        mode=data.get("mode", ""),
        temperature=float(data.get("temperature", 0)),
        energy=float(data.get("energy", 0)),
        id=data.get("id"),
        coverage=data.get("coverage"),
        pattern_center=data.get("pattern_center"),
    )
    lf.spots = [_spot_from_dict(s) for s in data.get("spots", [])]
    return lf


def serialize_session(
    files: Iterable[LeedFile],
    masks: List[QRect],
    background: LeedFile | None,
    tracks: Iterable[Track],
    spot_settings: Dict | None = None,
) -> Dict:
    return {
        "version": SESSION_VERSION,
        "data_files": [_leedfile_to_dict(f) for f in files],
        "background_file": _leedfile_to_dict(background) if background else None,
        "metadata_masks": [_rect_to_dict(r) for r in masks],
        "tracks": [_track_to_dict(t) for t in tracks],
        "spot_settings": spot_settings or {},
    }


def deserialize_session(data: Dict) -> Tuple[List[LeedFile], LeedFile | None, List[QRect], List[Track], Dict]:
    data_files_raw = data.get("data_files", [])
    if data_files_raw is None:
        data_files_raw = []
    if not isinstance(data_files_raw, list):
        raise ValueError(f"Invalid session: data_files must be a list, got {type(data_files_raw)}")
    files = [_leedfile_from_dict(d) for d in data_files_raw]

    bg_raw = data.get("background_file")
    background = _leedfile_from_dict(bg_raw) if bg_raw else None
    masks_raw = data.get("metadata_masks", [])
    if masks_raw is None:
        masks_raw = []
    if not isinstance(masks_raw, list):
        raise ValueError(f"Invalid session: metadata_masks must be a list, got {type(masks_raw)}")
    masks = [_rect_from_dict(r) for r in masks_raw]

    tracks_raw = data.get("tracks", [])
    if tracks_raw is None:
        tracks_raw = []
    if not isinstance(tracks_raw, list):
        raise ValueError(f"Invalid session: tracks must be a list, got {type(tracks_raw)}")
    tracks = [_track_from_dict(t) for t in tracks_raw]
    spot_settings = data.get("spot_settings", {})
    return files, background, masks, tracks, spot_settings


def save_session_to_file(
    path: Path,
    files: Iterable[LeedFile],
    masks: List[QRect],
    background: LeedFile | None,
    tracks: Iterable[Track],
    spot_settings: Dict | None = None,
) -> None:
    payload = serialize_session(files, masks, background, tracks, spot_settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session_from_file(path: Path) -> Tuple[List[LeedFile], LeedFile | None, List[QRect], List[Track], Dict]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return deserialize_session(data)
