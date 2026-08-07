#!/usr/bin/env python3
"""Predeclare deterministic native ViPErLEED execution for both principal scans."""

from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "manuscript/cross_contract/viperleed_principal"
EXECUTION_ROOT = (
    REPO_ROOT
    / ".scratch/trackleed-cross-contract-manuscript/principal-viperleed-execution"
)
PILOT_ROOT = REPO_ROOT / "manuscript/cross_contract/viperleed_pilot"
FRAME_PATTERN = re.compile(r"_(?P<frame>\d{4})_T\d+_E(?P<energy>\d{4})_")
SERIALIZED_TRACKER_PARAMETERS = (
    "6,0,64,1000,1000,3,30,2.5,10,0,0,4,0,0,5,5,100,2,50,2,4,1111,"
    "NaN,NaN,0,NaN,1,0,0,0,20,0,0,0,0,1,512,2,1"
)
EXPECTED_NATIVE_SUFFIXES = (
    "x",
    "y",
    "signif",
    "dx_raw",
    "dy_raw",
    "dx_smooth",
    "dy_smooth",
    "rawInt",
    "backgr",
    "bsigma",
    "rSize",
    "tSize",
    "IVcurves",
)
SCANS = {
    "scan_a": {
        "dataset": "data/26a21_ScanE",
        "frame_count": 35,
        "initialization_frame": 27,
        "initialization_energy_ev": 188.5,
        "terminal_duplicate": [220.5, 220.5],
    },
    "scan_b": {
        "dataset": "data/26a21b_ScanE",
        "frame_count": 46,
        "initialization_frame": 36,
        "initialization_energy_ev": 188.5,
        "terminal_duplicate": [220.4, 220.4],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = StringIO(newline="")
    fields = (
        "scan_id",
        "segment_index",
        "source_frame",
        "energy_ev",
        "source_path",
        "sha256",
        "size_bytes",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def imagej_path(path: Path) -> str:
    return path.resolve().as_posix()


def macro_text(scan_id: str, energies: list[float]) -> str:
    scan = SCANS[scan_id]
    dataset = REPO_ROOT / str(scan["dataset"])
    output = EXECUTION_ROOT / scan_id / "native"
    pattern = PILOT_ROOT / "frozen/pattern.csv"
    screenfit = PILOT_ROOT / "frozen/registration_screenfit.csv"
    prefix = f"viperleed_{scan_id}"
    energy_lines = []
    for start in range(0, len(energies), 8):
        values = ", ".join(format(value, ".10g") for value in energies[start : start + 8])
        energy_lines.append(f"    {values}")
    energy_body = ",\n".join(energy_lines)
    return f'''// Predeclared ticket-03 execution; do not edit after native outcomes exist.
scanId = "{scan_id}";
frameDir = "{imagej_path(dataset)}";
outputDir = "{imagej_path(output)}";
patternPath = "{imagej_path(pattern)}";
screenfitPath = "{imagej_path(screenfit)}";
checkpointPath = outputDir + "/macro_checkpoint.txt";
File.saveString("macro started\\nconfiguration predeclared; no manual outcome annotation\\n", checkpointPath);

run("Image Sequence...", "open=[" + frameDir + "] sort");
rename("principal_{scan_id}");
File.append("{scan['frame_count']}-frame stack opened\\n", checkpointPath);

newImage("principal_mask", "8-bit black", 1000, 1000, 1);
setColor(255);
makeOval(21.000308, 2.011221, 940, 940);
fill();
setColor(0);
makeOval(366.000308, 347.011221, 250, 250);
fill();
run("Select None");
run("Invert LUT");
File.append("frozen annular mask built\\n", checkpointPath);

energies = newArray(
{energy_body}
);
fitText = replace(File.openAsString(screenfitPath), "\\n", "");
fitItems = split(fitText, ",");
indices = newArray(fitItems.length);
for (i = 0; i < fitItems.length; i++)
    indices[i] = parseFloat(fitItems[i]);

call("LEED_Spot_Tracker.startMacro");
File.append("tracker panel started\\n", checkpointPath);
Ext.LEEDsetArray("energy", energies);
Ext.LEEDsetInput("iv", "principal_{scan_id}");
Ext.LEEDsetInput("mask", "principal_mask");
File.append("stack, energy and mask attached\\n", checkpointPath);
Ext.LEEDsetValue("minRange_eV", 0);
Ext.LEEDsetValue("savePlots", 0);
Ext.LEEDsetValue("saveStack", 0);
Ext.LEEDsetValue("neighborBackground", 0);
Ext.LEEDsetValue("minSignificanceSelecting", 4);
Ext.LEEDsetPattern(patternPath);
Ext.LEEDselectSliceForE({scan['initialization_energy_ev']});
File.append("pattern attached; transferring frozen indices\\n", checkpointPath);
Ext.LEEDsetIndices(indices);
File.append("indices set once; no per-frame correction permitted\\n", checkpointPath);
print("Principal {scan_id} pre-track status:\\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDtrackSpots();
File.append("spot tracking completed\\n", checkpointPath);
print("Principal {scan_id} post-track status:\\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDsaveData(outputDir, "{prefix}");
File.append("native whole-pattern data saved before proposal reduction\\n", checkpointPath);
if (!File.exists(outputDir + "/{prefix}_x.csv"))
    exit("Principal {scan_id} failed: native x export is missing");
File.append("macro completed\\n", checkpointPath);
Ext.LEEDclose("tracker stack");
close("*");
run("Quit");
'''


def read_scan(scan_id: str) -> tuple[list[dict[str, object]], list[float]]:
    scan = SCANS[scan_id]
    dataset = REPO_ROOT / str(scan["dataset"])
    paths = sorted(dataset.glob("*.tiff"))
    if len(paths) != scan["frame_count"]:
        raise ValueError(f"{scan_id} frame count changed: {len(paths)}")
    rows: list[dict[str, object]] = []
    energies: list[float] = []
    for segment, path in enumerate(paths):
        match = FRAME_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Cannot parse principal frame: {path.name}")
        frame = int(match.group("frame"))
        energy = int(match.group("energy")) / 10.0
        if frame != segment:
            raise ValueError(f"{scan_id} frame sequence changed at {path.name}")
        energies.append(energy)
        rows.append(
            {
                "scan_id": scan_id,
                "segment_index": segment,
                "source_frame": frame,
                "energy_ev": format(energy, ".10g"),
                "source_path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if energies[-2:] != scan["terminal_duplicate"]:
        raise ValueError(f"{scan_id} terminal duplicate energy changed")
    return rows, energies


def main() -> int:
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    EXECUTION_ROOT.mkdir(parents=True, exist_ok=True)
    scan_contracts: dict[str, object] = {}
    for scan_id, scan in SCANS.items():
        rows, energies = read_scan(scan_id)
        package_scan = PACKAGE_ROOT / "execution" / scan_id
        package_scan.mkdir(parents=True, exist_ok=True)
        manifest_path = package_scan / "input_manifest.csv"
        manifest_path.write_bytes(csv_bytes(rows))
        macro_path = package_scan / "run_viperleed.ijm"
        macro_path.write_text(macro_text(scan_id, energies), encoding="utf-8", newline="\n")
        (EXECUTION_ROOT / scan_id / "native").mkdir(parents=True, exist_ok=True)
        scan_contracts[scan_id] = {
            **scan,
            "energy_range_ev": [energies[0], energies[-1]],
            "input_manifest": {
                "path": manifest_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(manifest_path),
            },
            "macro": {
                "path": macro_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(macro_path),
            },
            "native_output_prefix": f"viperleed_{scan_id}",
        }

    pilot_contract = PILOT_ROOT / "contract.json"
    pattern = PILOT_ROOT / "frozen/pattern.csv"
    screenfit = PILOT_ROOT / "frozen/registration_screenfit.csv"
    scientific_contract = REPO_ROOT / "manuscript/cross_contract/scientific_contract.json"
    fingerprint = REPO_ROOT / "manuscript/scientific_core/fingerprint.json"
    contract = {
        "branch_mapping": {
            "P1_B1": [-1, 0],
            "P1_B2": [1, 0],
            "P2_B1": [1, -1],
            "P2_B2": [-1, 1],
        },
        "configuration": {
            "aperture_and_background": {
                "bright_neighbor_background_subtraction": False,
                "shape": "concentric-circles",
            },
            "corrections": {
                "dark_frame": "none supplied",
                "flat_field": "none supplied",
                "i0": "none supplied",
                "i00": "none supplied",
            },
            "minimum_energy_range_per_beam_ev": 0.0,
            "noise_rejection_significance": 2.5,
            "pattern": {
                "path": pattern.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(pattern),
            },
            "position_smoothing_range_ev": 30.0,
            "screenfit": {
                "path": screenfit.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(screenfit),
                "transfer_policy": "one unchanged pilot initialization per principal scan",
            },
            "screen_mask": {
                "center_xy_px": [491.00030779190064, 472.01122086402023],
                "inner_radius_px": 125.0,
                "outer_radius_px": 470.0,
                "shape": "annulus",
                "size_px": [1000, 1000],
            },
            "search_beam_unseen_for_ev": 10.0,
            "serialized_tracker_parameters_expected": SERIALIZED_TRACKER_PARAMETERS,
        },
        "contract_id": "native-viperleed-principal-scans",
        "evidence_policy": {
            "method_configuration": [
                "one transferred pattern and screenfit initialization per scan",
                "frozen pilot tracker settings",
                "frozen source-frame and energy sequence",
            ],
            "outcome_annotation": "none during native inference",
            "prohibited_after_outcomes_visible": [
                "per-frame correction",
                "manual recovery decision",
                "selective beam deletion",
                "parameter tuning",
                "dropping failed or missing observations",
            ],
        },
        "expected_native_suffixes": list(EXPECTED_NATIVE_SUFFIXES),
        "inference_scope": {
            "acquisition_status": "existing-acquisition-only evidence",
            "inference_status": "new native inference",
            "native_outputs_preserved_before_proposal_reduction": True,
        },
        "pilot_contract": {
            "path": pilot_contract.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(pilot_contract),
        },
        "predecessor_scientific_core": {
            "fingerprint_path": fingerprint.relative_to(REPO_ROOT).as_posix(),
            "fingerprint_sha256": sha256(fingerprint),
            "required_transition": "controlled scientific-core rebaseline after cross-contract analysis and claim-boundary approval",
        },
        "runtime": {
            "imagej_jar_sha256": sha256(
                REPO_ROOT / "third_party/viperleed-runtime/ImageJ/ij.jar"
            ),
            "imagej_version": "1.54u",
            "java_version": "OpenJDK 1.8.0_345",
            "plugin_jar_sha256": sha256(
                REPO_ROOT
                / "third_party/viperleed-runtime/ImageJ/plugins/ViPErLEED_ImageJ_plugins.jar"
            ),
            "plugin_version": "1.03",
        },
        "scans": scan_contracts,
        "schema_version": "1.0",
        "scientific_contract": {
            "path": scientific_contract.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(scientific_contract),
        },
        "status": "predeclared-principal-viperleed-inference-contract",
    }
    contract_path = PACKAGE_ROOT / "contract.json"
    contract_path.write_bytes(json_bytes(contract))
    print(f"Prepared principal ViPErLEED execution contract: {sha256(contract_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
