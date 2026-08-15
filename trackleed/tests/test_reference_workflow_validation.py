from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from trackleed.validation.reference_analysis import (
    ReferenceAcceptanceCriteria,
    ReferenceSeriesSpec,
    analyze_reference_series,
)
from trackleed.validation.reference_workflow import (
    ReferenceValidationConfig,
    ReferenceValidationError,
    validate_reference_workflow,
)


class ReferenceWorkflowValidationTests(unittest.TestCase):
    def test_manifest_requires_explicit_manual_golden_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "reference.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reference_root": ".",
                        "benchmark": {},
                        "series": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ReferenceValidationError,
                "manually_approved",
            ):
                validate_reference_workflow(
                    ReferenceValidationConfig(
                        repository_root=root,
                        manifest_path=manifest,
                        report_path=root / ".scratch" / "report.json",
                    )
                )

    def test_real_series_reports_center_methods_refusals_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, analysis_manifest = _write_synthetic_reference_series(root)

            report = analyze_reference_series(
                ReferenceSeriesSpec(
                    name="synthetic-known-center",
                    session_path=session,
                    analysis_manifest_path=analysis_manifest,
                    golden_frame_ids=(1,),
                    criteria=_synthetic_criteria(),
                )
            )

        self.assertEqual(report["self_consistency"]["valid_fits"], 4)
        self.assertEqual(
            set(report["center_methods"]),
            {
                "per_frame",
                "circle_fit",
                "circle_fit_bbox_fallback",
                "ellipse_fit",
                "circle_fit_temporal",
                "global_median",
            },
        )
        self.assertEqual(report["explicit_refusals"]["insufficient_points"], 1)
        self.assertFalse(report["recovery"]["default_enabled"])
        self.assertEqual(
            set(report["recovery"]["counts"]),
            {
                "recovered",
                "failed",
                "ambiguous_recovery",
                "not_needed",
            },
        )

    def test_golden_frame_requires_a_persisted_fit_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, analysis_manifest = _write_synthetic_reference_series(root)
            payload = json.loads(session.read_text(encoding="utf-8"))
            for track in payload["tracks"]:
                track["analysis_results"] = {}
            session.write_text(json.dumps(payload), encoding="utf-8")
            counts = json.loads(analysis_manifest.read_text(encoding="utf-8"))
            counts["counts"]["valid_fits"] = 0
            analysis_manifest.write_text(json.dumps(counts), encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError,
                "approved fit oracle",
            ):
                analyze_reference_series(
                    ReferenceSeriesSpec(
                        name="missing-oracle",
                        session_path=session,
                        analysis_manifest_path=analysis_manifest,
                        golden_frame_ids=(1,),
                        criteria=_synthetic_criteria(),
                    )
                )

    def test_approved_center_stability_threshold_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, analysis_manifest = _write_synthetic_reference_series(root)
            payload = json.loads(session.read_text(encoding="utf-8"))
            payload["data_files"][0]["pattern_center"]["center_x"] = 32.5
            for track in payload["tracks"]:
                result = track["analysis_results"]["1"]
                result["radius"] = float(
                    np.hypot(
                        result["global_center_x"] - 32.5,
                        result["global_center_y"] - 32.0,
                    )
                )
            session.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "center error .* exceeds approved"):
                analyze_reference_series(
                    ReferenceSeriesSpec(
                        name="strict-stability",
                        session_path=session,
                        analysis_manifest_path=analysis_manifest,
                        golden_frame_ids=(1,),
                        criteria=_synthetic_criteria(max_center_error_px=0.1),
                    )
                )

    def test_manifest_requires_both_canonical_benchmark_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "reference.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reference_root": ".",
                        "golden_approval": _approved_goldens(),
                        "benchmark": {"four_track_session": "four.lts.json"},
                        "series": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ReferenceValidationError,
                "twenty_track_session",
            ):
                validate_reference_workflow(
                    ReferenceValidationConfig(
                        repository_root=root,
                        manifest_path=manifest,
                        report_path=root / ".scratch" / "report.json",
                    )
                )

    def test_manifest_rejects_too_few_manually_approved_golden_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("four.lts.json", "twenty.lts.json", "series.lts.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "analysis.json").write_text("{}", encoding="utf-8")
            manifest = root / "reference.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reference_root": ".",
                        "golden_approval": _approved_goldens(),
                        "benchmark": {
                            "four_track_session": "four.lts.json",
                            "twenty_track_session": "twenty.lts.json",
                        },
                        "series": [
                            {
                                "name": "real-series",
                                "session": "series.lts.json",
                                "analysis_manifest": "analysis.json",
                                "golden_frame_ids": list(range(24)),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ReferenceValidationError,
                "25 to 35",
            ):
                validate_reference_workflow(
                    ReferenceValidationConfig(
                        repository_root=root,
                        manifest_path=manifest,
                        report_path=root / ".scratch" / "report.json",
                    )
                )

    def test_manifest_approval_count_must_match_declared_goldens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("four.lts.json", "twenty.lts.json", "series.lts.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "analysis.json").write_text("{}", encoding="utf-8")
            manifest = root / "reference.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reference_root": ".",
                        "golden_approval": _approved_goldens(30),
                        "benchmark": {
                            "four_track_session": "four.lts.json",
                            "twenty_track_session": "twenty.lts.json",
                        },
                        "series": [
                            {
                                "name": "real-series",
                                "session": "series.lts.json",
                                "analysis_manifest": "analysis.json",
                                "golden_frame_ids": list(range(25)),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ReferenceValidationError,
                "approval frame_count",
            ):
                validate_reference_workflow(
                    ReferenceValidationConfig(
                        repository_root=root,
                        manifest_path=manifest,
                        report_path=root / ".scratch" / "report.json",
                    )
                )

    def test_manifest_rejects_duplicate_session_frame_selections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("four.lts.json", "twenty.lts.json", "series.lts.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "analysis.json").write_text("{}", encoding="utf-8")
            repeated_series = {
                "session": "series.lts.json",
                "analysis_manifest": "analysis.json",
                "golden_frame_ids": list(range(15)),
            }
            manifest = root / "reference.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reference_root": ".",
                        "golden_approval": _approved_goldens(30),
                        "benchmark": {
                            "four_track_session": "four.lts.json",
                            "twenty_track_session": "twenty.lts.json",
                        },
                        "series": [
                            {"name": "duplicate-a", **repeated_series},
                            {"name": "duplicate-b", **repeated_series},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ReferenceValidationError, "distinct"):
                validate_reference_workflow(
                    ReferenceValidationConfig(
                        repository_root=root,
                        manifest_path=manifest,
                        report_path=root / ".scratch" / "report.json",
                    )
                )


def _write_synthetic_reference_series(root: Path) -> tuple[Path, Path]:
    centers = ((16.0, 16.0), (48.0, 16.0), (48.0, 48.0), (16.0, 48.0))
    pixels = _synthetic_pixels(centers)
    tracks, spots = _synthetic_tracks(centers)
    image_path = root / "frame.png"
    Image.fromarray(pixels).save(image_path)
    session_payload = _synthetic_session_payload(image_path, tracks, spots)
    session_path = root / "series.lts.json"
    session_path.write_text(json.dumps(session_payload), encoding="utf-8")
    analysis_manifest = root / "analysis.json"
    analysis_manifest.write_text(
        json.dumps(
            {
                "counts": {
                    "frames": 1,
                    "tracks": 4,
                    "tracked_rois": 4,
                    "valid_fits": 4,
                    "invalid_fits": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    return session_path, analysis_manifest


def _synthetic_criteria(
    *,
    max_center_error_px: float = 50.0,
) -> ReferenceAcceptanceCriteria:
    return ReferenceAcceptanceCriteria(
        minimum_method_available_frames=1,
        minimum_temporal_pairs=0,
        max_center_error_px=max_center_error_px,
        max_temporal_jump_px=50.0,
        max_spot_removal_shift_px=50.0,
        max_ransac_seed_spread_px=50.0,
    )


def _approved_goldens(frame_count: int | None = None) -> dict:
    approval = {
        "status": "manually_approved",
        "criteria": {
            "minimum_method_available_frames": 1,
            "minimum_temporal_pairs": 1,
            "max_center_error_px": 50.0,
            "max_temporal_jump_px": 50.0,
            "max_spot_removal_shift_px": 50.0,
            "max_ransac_seed_spread_px": 50.0,
        },
    }
    if frame_count is not None:
        approval["frame_count"] = frame_count
    return approval


def _synthetic_pixels(
    centers: tuple[tuple[float, float], ...],
) -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    pixels = np.full((64, 64), 8.0)
    for center_x, center_y in centers:
        pixels += 220.0 * np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 2.0**2)
        )
    return np.asarray(np.clip(pixels, 0, 255), dtype=np.uint8)


def _synthetic_tracks(
    centers: tuple[tuple[float, float], ...],
) -> tuple[list[dict], list[dict]]:
    radius = float(np.hypot(16.0, 16.0))
    tracks = []
    spots = []
    for track_id, (center_x, center_y) in enumerate(centers, start=1):
        roi = {
            "x": int(center_x - 6),
            "y": int(center_y - 6),
            "w": 13,
            "h": 13,
        }
        spot = {
            "id": track_id,
            "roi": roi,
            "center_x": center_x,
            "center_y": center_y,
            "intensity": 1.0,
            "source": "manual",
        }
        spots.append(spot)
        tracks.append(
            {
                "track_id": track_id,
                "spots_by_image_id": {"1": spot},
                "analysis_results": {
                    "1": {
                        "global_center_x": center_x,
                        "global_center_y": center_y,
                        "radius": radius,
                    }
                },
            }
        )
    return tracks, spots


def _synthetic_session_payload(
    image_path: Path,
    tracks: list[dict],
    spots: list[dict],
) -> dict:
    return {
        "version": 1,
        "data_files": [
            {
                "filepath": str(image_path),
                "filename": image_path.name,
                "is_background": False,
                "element": "X",
                "surface": 1,
                "mode": "G",
                "temperature": 1.0,
                "energy": 1.0,
                "id": 1,
                "coverage": 0.0,
                "pattern_center": {"center_x": 32.0, "center_y": 32.0},
                "post_analysis_results": {},
                "spots": spots,
            }
        ],
        "background_file": None,
        "masks": [],
        "tracks": tracks,
        "spot_settings": {},
        "analysis_circle": None,
        "analysis_ellipse": None,
    }


if __name__ == "__main__":
    unittest.main()
