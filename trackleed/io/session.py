from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.analysis_session import (
    decode_analysis_state,
    encode_analysis_state,
)
from trackleed.logic.active_analysis_state import (
    ActiveAnalysisState,
    AnalysisResultStatus,
    AnalysisSourceState,
    AnalysisSourceStatus,
)

SESSION_VERSION = 2


@dataclass(frozen=True)
class SessionContents:
    files: List[LeedFile]
    background: LeedFile | None
    masks: List[QRect]
    tracks: List[Track]
    spot_settings: Dict
    analysis_circle: Dict | None
    analysis_ellipse: Dict | None
    analysis_state: ActiveAnalysisState


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
        "polygon_seed_points": [
            [float(x), float(y)]
            for x, y in getattr(spot, "polygon_seed_points", [])
        ],
    }


def _spot_from_dict(data: Dict) -> SpotROI:
    raw_polygon_points = data.get("polygon_seed_points", [])
    polygon_seed_points = []
    if isinstance(raw_polygon_points, list):
        for point in raw_polygon_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                polygon_seed_points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return SpotROI(
        id=int(data["id"]),
        roi_rect=_rect_from_dict(data["roi"]),
        center_x=float(data["center_x"]),
        center_y=float(data["center_y"]),
        intensity=float(data.get("intensity", 0)),
        source=data.get("source", "auto"),
        polygon_seed_points=polygon_seed_points,
    )


def _track_to_dict(track: Track) -> Dict:
    payload = {
        "track_id": track.track_id,
        "spots_by_image_id": {str(fid): _spot_to_dict(s) for fid, s in track.spots_by_image_id.items()},
    }
    if hasattr(track, "analysis_results"):
        payload["analysis_results"] = {
            str(fid): _analysis_result_to_dict(res)
            for fid, res in track.analysis_results.items()
        }
    return payload


def _track_from_dict(data: Dict) -> Track:
    track = Track(
        track_id=int(data["track_id"]),
        spots_by_image_id={int(fid): _spot_from_dict(s) for fid, s in data.get("spots_by_image_id", {}).items()},
    )
    analysis_results = data.get("analysis_results")
    if isinstance(analysis_results, dict):
        track.analysis_results = {
            int(fid): _analysis_result_from_dict(res)
            for fid, res in analysis_results.items()
        }
    return track


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


def _clean_value(val):
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _clean_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_clean_value(v) for v in val]
    if hasattr(val, "tolist"):
        try:
            return val.tolist()
        except Exception:
            pass
    return _clean_number(val)


def _analysis_result_to_dict(result: Dict | None) -> Dict | None:
    if result is None:
        return None
    cleaned = {}
    for key, value in result.items():
        cleaned[key] = _clean_value(value)
    return cleaned


def _analysis_result_from_dict(result: Dict | None) -> Dict | None:
    if result is None:
        return None
    return _clean_value(result)


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
        "post_analysis_results": _clean_dict_numbers(getattr(lf, "post_analysis_results", None)),
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
    post_results = data.get("post_analysis_results")
    if post_results is not None:
        lf.post_analysis_results = _clean_dict_numbers(post_results)
    return lf


def serialize_session(
    contents: SessionContents,
    session_path: Path | None = None,
) -> Dict:
    state = _analysis_state_for_save(contents.analysis_state, session_path)
    return {
        "version": SESSION_VERSION,
        "data_files": [_leedfile_to_dict(item) for item in contents.files],
        "background_file": (
            _leedfile_to_dict(contents.background) if contents.background else None
        ),
        "metadata_masks": [_rect_to_dict(rect) for rect in contents.masks],
        "tracks": [_track_to_dict(track) for track in contents.tracks],
        "spot_settings": contents.spot_settings,
        "analysis_circle": contents.analysis_circle,
        "analysis_ellipse": contents.analysis_ellipse,
        "analysis_state": encode_analysis_state(state),
    }


def _analysis_state_for_save(
    state: ActiveAnalysisState,
    session_path: Path | None,
) -> ActiveAnalysisState:
    if session_path is None:
        return state
    sources = []
    for source in state.sources:
        try:
            relative_path = source.path.relative_to(session_path.parent)
        except ValueError:
            relative_path = Path(source.path.name)
        sources.append(replace(source, relative_path=relative_path))
    return replace(state, sources=tuple(sources))


def deserialize_session(
    data: Dict,
    session_path: Path | None = None,
) -> SessionContents:
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
    analysis_circle = data.get("analysis_circle")
    analysis_ellipse = data.get("analysis_ellipse")
    version = data.get("version", 1)
    if version == 1:
        analysis_state = _legacy_analysis_state(
            files,
            background,
            tracks,
            session_path,
        )
    elif version == SESSION_VERSION:
        analysis_state = _analysis_state_after_load(
            decode_analysis_state(data.get("analysis_state")),
            files,
            background,
            session_path,
        )
    else:
        raise ValueError(f"Invalid session: unsupported version {version}")
    return SessionContents(
        files,
        background,
        masks,
        tracks,
        spot_settings,
        analysis_circle,
        analysis_ellipse,
        analysis_state,
    )


def _analysis_state_after_load(
    state: ActiveAnalysisState,
    files: List[LeedFile],
    background: LeedFile | None,
    session_path: Path | None,
) -> ActiveAnalysisState:
    sources = tuple(
        _source_after_load(source, files, background, session_path)
        for source in state.sources
    )
    return ActiveAnalysisState(
        state.fit_set,
        state.geometry_result,
        state.fit_set is not None,
        state.result_status,
        sources,
    )


def _source_after_load(
    source: AnalysisSourceState,
    files: List[LeedFile],
    background: LeedFile | None,
    session_path: Path | None,
) -> AnalysisSourceState:
    file = _source_file(source, files, background)
    candidates = [source.path]
    if session_path is not None:
        candidates.extend(
            (session_path.parent / source.relative_path, session_path.parent / source.path.name)
        )
    candidate = next((item for item in candidates if item.is_file()), None)
    if file is not None and candidate is not None:
        file.filepath = candidate
    status = (
        AnalysisSourceStatus.UNVERIFIED
        if candidate is not None and source.pixel_sha256 is not None
        else (
            AnalysisSourceStatus.AVAILABLE
            if candidate is not None
            else AnalysisSourceStatus.SOURCE_UNAVAILABLE
        )
    )
    return replace(source, path=candidate or source.path, status=status)


def _source_file(
    source: AnalysisSourceState,
    files: List[LeedFile],
    background: LeedFile | None,
) -> LeedFile | None:
    if source.is_background:
        return background
    return next((item for item in files if item.id == source.frame_id), None)


def _legacy_analysis_state(
    files: List[LeedFile],
    background: LeedFile | None,
    tracks: List[Track],
    session_path: Path | None,
) -> ActiveAnalysisState:
    has_results = any(
        bool(getattr(track, "analysis_results", {})) for track in tracks
    ) or any(
        file.pattern_center is not None
        or getattr(file, "post_analysis_results", None) is not None
        for file in files
    )
    if not has_results:
        return ActiveAnalysisState()
    sources = tuple(
        _legacy_source_state(file, session_path, is_background=False)
        for file in files
    )
    if background is not None:
        sources += (_legacy_source_state(background, session_path, True),)
    return ActiveAnalysisState(
        result_status=AnalysisResultStatus.LEGACY_UNVERIFIED,
        sources=sources,
    )


def _legacy_source_state(
    file: LeedFile,
    session_path: Path | None,
    is_background: bool,
) -> AnalysisSourceState:
    relative_path = Path(file.filename)
    if session_path is not None:
        try:
            relative_path = file.filepath.relative_to(session_path.parent)
        except ValueError:
            pass
    status = (
        AnalysisSourceStatus.AVAILABLE
        if file.filepath.is_file()
        else AnalysisSourceStatus.SOURCE_UNAVAILABLE
    )
    return AnalysisSourceState(
        frame_id=None if is_background else file.id,
        path=file.filepath,
        relative_path=relative_path,
        pixel_sha256=None,
        status=status,
        is_background=is_background,
    )


def save_session_to_file(
    path: Path,
    contents: SessionContents,
) -> None:
    payload = serialize_session(contents, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session_from_file(path: Path) -> SessionContents:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return deserialize_session(data, path)
