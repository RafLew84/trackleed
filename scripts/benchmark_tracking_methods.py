#!/usr/bin/env python3
"""Benchmark Optical Flow and TAPIR tracking backends on the same LEED session."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from trackleed.core.data_models import SpotROI, Track


def _lazy_imports():
    global summarize_tracks, summary_to_dict, track_rows_to_dicts
    global DEFAULT_TAPIR_MODELS_DIR, probe_tapir_environment
    global select_default_tapir_checkpoint, discover_tapir_checkpoints
    global track_spots_optical_flow, track_spots_optical_flow_skip_anchor
    global track_spots_tapir_offline, load_session_from_file, save_session_to_file

    from trackleed.analysis.benchmark import (
        summarize_tracks,
        summary_to_dict,
        track_rows_to_dicts,
    )
    from trackleed.analysis.tapir_backend import (
        DEFAULT_TAPIR_MODELS_DIR,
        probe_tapir_environment,
        select_default_tapir_checkpoint,
        discover_tapir_checkpoints,
    )
    from trackleed.analysis.tracking import (
        track_spots_optical_flow,
        track_spots_optical_flow_skip_anchor,
        track_spots_tapir_offline,
    )
    from trackleed.io.session import load_session_from_file, save_session_to_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the same LEED sequence through Optical Flow and TAPIR and export comparison CSVs.",
    )
    parser.add_argument("--session", required=True, help="Input LEEDTrack session (.lts.json).")
    parser.add_argument(
        "--ground-truth-session",
        help="Optional manually corrected session used as ground truth for MOT-style metrics.",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        help="Reference frame ID. If omitted, inferred as the first frame containing spots.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["optical_flow", "optical_flow_skip_anchor", "tapir"],
        default=["optical_flow", "optical_flow_skip_anchor", "tapir"],
        help="Tracking methods to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for CSVs and optional sessions. Defaults to benchmark_output/<session_stem>.",
    )
    parser.add_argument(
        "--save-sessions",
        action="store_true",
        help="Also save predicted sessions per method for manual inspection.",
    )
    parser.add_argument(
        "--gt-match-threshold",
        type=float,
        default=12.0,
        help="Matching threshold in pixels for optional ground-truth metrics.",
    )

    parser.add_argument("--roi-size", type=int, help="ROI size override. Default: from session or 32.")
    parser.add_argument("--subtract-background", action="store_true", help="Enable background subtraction for optical flow.")
    parser.add_argument("--tapir-checkpoint", help="Override TAPIR checkpoint path.")
    parser.add_argument("--tapir-device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--tapir-resolution", type=int, default=256)
    parser.add_argument("--tapir-query-chunk-size", type=int, default=64)
    parser.add_argument("--tapir-window-size", type=int, default=64)
    parser.add_argument("--tapir-window-overlap", type=int, default=8)
    parser.add_argument("--tapir-max-step", type=float, default=0.0)
    parser.add_argument("--tapir-max-reacquisition-gap", type=int, default=0)
    parser.add_argument("--tapir-terminate-after-gap", type=int, default=0)
    parser.add_argument("--tapir-center-mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--tapir-center-x", type=float, default=0.0)
    parser.add_argument("--tapir-center-y", type=float, default=0.0)
    parser.add_argument("--tapir-direction-mode", choices=["none", "auto", "inward", "outward"], default="none")
    parser.add_argument("--tapir-angular-tol", type=float, default=20.0)
    parser.add_argument("--tapir-max-radial-jump", type=float, default=0.0)
    parser.add_argument("--tapir-recenter-mode", choices=["none", "peak", "centroid", "gaussian"], default="centroid")
    parser.add_argument("--tapir-recenter-radius", type=float, default=1.25)
    parser.add_argument("--tapir-centroid-threshold", type=float, default=0.35)
    parser.add_argument("--tapir-max-recenter-shift", type=float, default=0.45)
    parser.add_argument("--tapir-no-visibility-gating", action="store_true")
    parser.add_argument("--tapir-visibility-threshold", type=float, default=0.5)
    parser.add_argument("--tapir-no-uncertainty-gating", action="store_true")
    parser.add_argument("--tapir-expected-dist-threshold", type=float, default=0.5)
    parser.add_argument("--tapir-no-gaps", action="store_true", help="Write invisible TAPIR points instead of leaving gaps.")

    parser.add_argument("--lk-win-size", type=int, default=31)
    parser.add_argument("--lk-max-level", type=int, default=3)
    parser.add_argument("--lk-max-iterations", type=int, default=10)
    parser.add_argument("--lk-epsilon", type=float, default=0.03)
    parser.add_argument("--lk-min-eig", type=float, default=1e-5)
    parser.add_argument("--lk-use-centroid", action="store_true")
    parser.add_argument("--lk-centroid-multiplier", type=float, default=1.5)
    parser.add_argument("--lk-centroid-threshold", type=float, default=0.4)
    parser.add_argument("--lk-centroid-expand", type=int, default=1)

    parser.add_argument("--skip-err-threshold", type=float, default=30.0)
    parser.add_argument("--skip-fb-threshold", type=float, default=8.0)
    parser.add_argument("--skip-no-fb", action="store_true")
    parser.add_argument("--skip-max-skip-frames", type=int, default=0)
    parser.add_argument("--skip-oob-behavior", choices=["skip", "mark_lost"], default="skip")
    parser.add_argument("--skip-use-ncc", action="store_true")
    parser.add_argument("--skip-ncc-threshold", type=float, default=0.65)
    parser.add_argument("--skip-ncc-patch-radius", type=int, default=4)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _lazy_imports()

    session_path = Path(args.session).resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (REPO_ROOT / "benchmark_output" / session_path.stem).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    files, background, masks, session_tracks, spot_settings, analysis_circle, analysis_ellipse = load_session_from_file(session_path)
    files = [f for f in files if f.id is not None]
    if not files:
        raise ValueError("Session contains no data files with frame IDs.")
    all_files_map = {f.id: f for f in files if f.id is not None}
    sorted_frame_ids = sorted(all_files_map.keys())

    reference_frame_id = args.reference_frame or infer_reference_frame_id(files, session_tracks)
    if reference_frame_id is None:
        raise ValueError("Could not infer reference frame. Pass --reference-frame explicitly.")
    if reference_frame_id not in all_files_map:
        raise ValueError(f"Reference frame {reference_frame_id} is missing from session files.")

    ensure_reference_spots(reference_frame_id, all_files_map, session_tracks)
    reference_file = all_files_map[reference_frame_id]
    if not reference_file.spots:
        raise ValueError("Reference frame has no spots. Add spots on the reference frame and save the session first.")

    roi_size = int(args.roi_size or infer_roi_size(reference_file.spots, spot_settings))
    benchmark_frame_ids = sorted_frame_ids[sorted_frame_ids.index(reference_frame_id):]

    gt_tracks = None
    if args.ground_truth_session:
        gt_tracks = load_ground_truth_tracks(Path(args.ground_truth_session).resolve())

    tapir_checkpoint_path = args.tapir_checkpoint
    if "tapir" in args.methods and not tapir_checkpoint_path:
        checkpoints = discover_tapir_checkpoints(DEFAULT_TAPIR_MODELS_DIR)
        default_checkpoint = select_default_tapir_checkpoint(checkpoints)
        tapir_checkpoint_path = str(default_checkpoint) if default_checkpoint else None

    method_specs = build_method_specs(
        args=args,
        roi_size=roi_size,
        background_filepath=str(background.filepath) if background else None,
        tapir_checkpoint_path=tapir_checkpoint_path,
    )

    summary_rows: List[Dict[str, object]] = []
    per_track_rows: List[Dict[str, object]] = []

    print(f"Session: {session_path}")
    print(f"Reference frame: {reference_frame_id}")
    print(f"Reference spots: {len(reference_file.spots)}")
    print(f"Output dir: {output_dir}")
    print("")

    for method_name in args.methods:
        runner, params = method_specs[method_name]

        if method_name == "tapir":
            status = probe_tapir_environment(
                models_dir=params.get("tapir_models_dir"),
                checkpoint_path=params.get("checkpoint_path"),
            )
            if not status.is_ready:
                raise RuntimeError(
                    "TAPIR environment is not ready:\n" + "\n".join(status.errors + status.warnings)
                )

        print(f"[{method_name}] running...")
        started = time.perf_counter()
        tracks = runner(all_files_map, sorted_frame_ids, reference_frame_id, params)
        elapsed = time.perf_counter() - started
        if tracks is None:
            raise RuntimeError(f"{method_name} returned None.")

        summary, track_rows = summarize_tracks(
            tracks=tracks,
            benchmark_frame_ids=benchmark_frame_ids,
            method=method_name,
            runtime_seconds=elapsed,
            ground_truth_tracks=gt_tracks,
            gt_match_threshold_px=args.gt_match_threshold,
        )
        summary_rows.append(summary_to_dict(summary))
        per_track_rows.extend(track_rows_to_dicts(track_rows))

        print(
            f"[{method_name}] tracks={summary.track_count} "
            f"coverage={summary.overall_coverage_ratio:.3f} "
            f"mean_len={summary.mean_track_length:.2f} "
            f"runtime={summary.runtime_seconds:.2f}s"
        )

        if args.save_sessions:
            save_session_to_file(
                output_dir / f"{method_name}.lts.json",
                files,
                masks,
                background,
                tracks,
                spot_settings,
                analysis_circle,
                analysis_ellipse,
            )

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_track.csv", per_track_rows)
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(
            {
                "session": str(session_path),
                "ground_truth_session": str(Path(args.ground_truth_session).resolve()) if args.ground_truth_session else None,
                "reference_frame": int(reference_frame_id),
                "methods": list(args.methods),
                "roi_size": roi_size,
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Wrote summary to {output_dir / 'summary.csv'}")
    print(f"Wrote per-track rows to {output_dir / 'per_track.csv'}")
    return 0


def build_method_specs(
    args: argparse.Namespace,
    roi_size: int,
    background_filepath: str | None,
    tapir_checkpoint_path: str | None,
) -> Dict[str, Tuple[Callable, Dict[str, object]]]:
    optical_flow_params: Dict[str, object] = {
        "roi_size": roi_size,
        "lk_win_size": args.lk_win_size,
        "lk_max_level": args.lk_max_level,
        "lk_max_iterations": args.lk_max_iterations,
        "lk_epsilon": args.lk_epsilon,
        "lk_min_eigen_threshold": args.lk_min_eig,
        "use_centroid_refinement": bool(args.lk_use_centroid),
        "centroid_patch_multiplier": args.lk_centroid_multiplier,
        "centroid_rel_threshold": args.lk_centroid_threshold,
        "centroid_max_expand": args.lk_centroid_expand,
        "subtract_background": bool(args.subtract_background and background_filepath),
    }
    if args.subtract_background and background_filepath:
        optical_flow_params["background_filepath"] = background_filepath

    optical_flow_skip_params = dict(optical_flow_params)
    optical_flow_skip_params.update(
        {
            "lk_err_threshold": args.skip_err_threshold,
            "lk_fb_threshold": args.skip_fb_threshold,
            "use_forward_backward": not bool(args.skip_no_fb),
            "max_skip_frames": args.skip_max_skip_frames,
            "oob_behavior": args.skip_oob_behavior,
            "use_ncc_check": bool(args.skip_use_ncc),
            "ncc_threshold": args.skip_ncc_threshold,
            "ncc_patch_radius": args.skip_ncc_patch_radius,
        }
    )

    tapir_params: Dict[str, object] = {
        "tapir_models_dir": str(DEFAULT_TAPIR_MODELS_DIR.resolve()),
        "checkpoint_path": tapir_checkpoint_path,
        "device": args.tapir_device,
        "inference_resolution": args.tapir_resolution,
        "query_chunk_size": args.tapir_query_chunk_size,
        "window_size": args.tapir_window_size,
        "window_overlap": args.tapir_window_overlap,
        "use_visibility_gating": not bool(args.tapir_no_visibility_gating),
        "visibility_threshold": args.tapir_visibility_threshold,
        "use_uncertainty_gating": not bool(args.tapir_no_uncertainty_gating),
        "expected_dist_threshold": args.tapir_expected_dist_threshold,
        "keep_invisible_points_as_gaps": not bool(args.tapir_no_gaps),
        "max_step_distance_px": args.tapir_max_step,
        "max_reacquisition_gap": args.tapir_max_reacquisition_gap,
        "terminate_after_invisible_frames": args.tapir_terminate_after_gap,
        "center_mode": args.tapir_center_mode,
        "manual_center": (args.tapir_center_x, args.tapir_center_y),
        "direction_mode": args.tapir_direction_mode,
        "lateral_angle_tolerance_deg": args.tapir_angular_tol,
        "max_radial_jump_px": args.tapir_max_radial_jump,
        "recenter_mode": args.tapir_recenter_mode,
        "recenter_patch_multiplier": args.tapir_recenter_radius,
        "centroid_rel_threshold": args.tapir_centroid_threshold,
        "centroid_max_shift_ratio": args.tapir_max_recenter_shift,
        "roi_size": roi_size,
    }

    return {
        "optical_flow": (track_spots_optical_flow, optical_flow_params),
        "optical_flow_skip_anchor": (track_spots_optical_flow_skip_anchor, optical_flow_skip_params),
        "tapir": (track_spots_tapir_offline, tapir_params),
    }


def infer_reference_frame_id(files, tracks: Iterable["Track"]) -> int | None:
    for leed_file in sorted([f for f in files if f.id is not None], key=lambda item: item.id):
        if leed_file.spots:
            return int(leed_file.id)
    candidate_ids = sorted(
        {
            int(frame_id)
            for track in tracks
            for frame_id in track.spots_by_image_id.keys()
        }
    )
    return candidate_ids[0] if candidate_ids else None


def ensure_reference_spots(reference_frame_id: int, all_files_map, tracks: Iterable["Track"]) -> None:
    reference_file = all_files_map[reference_frame_id]
    if reference_file.spots:
        return
    recovered_spots = []
    for track in tracks:
        spot = track.spots_by_image_id.get(reference_frame_id)
        if spot is not None:
            recovered_spots.append(spot)
    recovered_spots.sort(key=lambda spot: int(spot.id))
    reference_file.spots = recovered_spots


def infer_roi_size(reference_spots: Iterable["SpotROI"], spot_settings: Dict) -> int:
    classic = spot_settings.get("classic", {}) if isinstance(spot_settings, dict) else {}
    roi_size = int(classic.get("roi_size", 0) or 0)
    if roi_size > 0:
        return roi_size
    for spot in reference_spots:
        rect = getattr(spot, "roi_rect", None)
        if rect is not None and rect.width() > 0:
            return int(rect.width())
    return 32


def load_ground_truth_tracks(path: Path) -> List["Track"]:
    files, _, _, tracks, _, _, _ = load_session_from_file(path)
    if tracks:
        return tracks

    file_map = {f.id: f for f in files if f.id is not None}
    from trackleed.core.data_models import Track

    gt_tracks: List[Track] = []
    for frame_id, leed_file in file_map.items():
        for spot in leed_file.spots:
            gt_tracks.append(Track(track_id=int(spot.id), spots_by_image_id={int(frame_id): spot}))
    return gt_tracks


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
