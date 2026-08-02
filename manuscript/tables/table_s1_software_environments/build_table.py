#!/usr/bin/env python3
"""Build supplementary Table S1 from frozen software and runtime records."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TABLE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = TABLE_ROOT / "source_data"
COMPONENTS_PATH = SOURCE_DATA_DIRECTORY / "component_records.csv"
CONFIGURATION_RECORDS_DIRECTORY = (
    SOURCE_DATA_DIRECTORY / "configuration_records"
)
ENVIRONMENT_ROLES_PATH = SOURCE_DATA_DIRECTORY / "environment_roles.csv"
ENVIRONMENT_SNAPSHOTS_DIRECTORY = (
    SOURCE_DATA_DIRECTORY / "environment_snapshots"
)
SYSTEM_RECORD_PATH = SOURCE_DATA_DIRECTORY / "system_record.json"
SOURCE_PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
UPSTREAM_DIRECTORY = SOURCE_DATA_DIRECTORY / "upstream_contracts"
CAPTURE_SCRIPT_PATH = TABLE_ROOT / "capture_environments.py"
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"

SUPPLEMENT_BEGIN_MARKER = "<!-- BEGIN TABLE S1 -->"
SUPPLEMENT_END_MARKER = "<!-- END TABLE S1 -->"

COMPONENT_COLUMNS = (
    "order",
    "component",
    "study_role",
    "software_version",
    "upstream_revision",
    "configuration_variant",
    "frozen_artifact",
    "artifact_type",
    "artifact_sha256",
    "configuration_record_sha256",
    "configuration_record_status",
    "upstream_configuration_sha256",
    "implementation_record_sha256",
    "environment_label",
)
ENVIRONMENT_ROLE_COLUMNS = (
    "order",
    "environment_label",
    "used_for",
    "execution_mode",
)
EXPECTED_COMPONENTS = (
    "TrackLEED common measurement",
    "LocoTrack",
    "OSTrack",
    "DAM4SAM Large",
    "Lucas–Kanade",
    "EasyLEED",
    "SAM2.1",
    "SAMURAI",
    "DAM4SAM Base+",
    "ViPErLEED",
)
EXPECTED_ENVIRONMENTS = (
    "netcat",
    "locotrack-pytorch",
    "ostrack",
    "dam4sam",
    "sam2",
    "samurai",
    "easyleed",
    "viperleed-imagej",
)
PYTHON_ENVIRONMENT_LABELS = EXPECTED_ENVIRONMENTS[:-1]
SHARED_SAM2_CHECKPOINT = (
    "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)
FAMILY_CONTRACT_SHA256 = (
    "ee579ab76263a3ee20c227f79f83bc1c1d6f26793718209a0b517f5e5131a60c"
)
CONTROLLED_FAMILY_RUNNER_SHA256 = (
    "c95eb72343da954378e083233b943e79beed78328e3695dc71b4a5f51aaa9547"
)
INDEPENDENT_ENERGY_RUNNER_SHA256 = (
    "fc8e2aa3de8e1d7c24ad2d69c6bcb8c05151c9b8a540a1379134d1ba8bbb3eee"
)
CONTROLLED_FAMILY_CONFIGURATION_FILES = {
    "SAM2.1": "sam2-controlled-family.json",
    "SAMURAI": "samurai-controlled-family.json",
    "DAM4SAM Base+": "dam4sam-base-plus-controlled-family.json",
}
INDEPENDENT_ENERGY_CONFIGURATION_FILES = {
    "LocoTrack": "locotrack-independent-energy-scans.json",
    "OSTrack": "ostrack-independent-energy-scans.json",
    "Lucas–Kanade": "lucas-kanade-independent-energy-scans.json",
}
MIXED_MAIN_CONFIGURATION_STATUS = (
    "run-frozen for Cu(111)/translations; post-run normalized from "
    "executed source for independent energy scans"
)
CONTROLLED_FAMILY_IMPLEMENTATION_HASHES = {
    "SAM2.1": {
        "2f6b6442e3aafafa913f78400dab15d4c7c715d93bb7577009dc75beb7b31797",
        CONTROLLED_FAMILY_RUNNER_SHA256,
    },
    "SAMURAI": {
        "bf2933f3f3fc5ef31c0fc0e530d5f270469be83a8858552f04d2989c32f8ae21",
        CONTROLLED_FAMILY_RUNNER_SHA256,
    },
    "DAM4SAM Base+": {
        "ee20f7898d6e74a6f9fc953c806b8e52cab1ba651772049b88849ea8c18c8586",
        CONTROLLED_FAMILY_RUNNER_SHA256,
        INDEPENDENT_ENERGY_RUNNER_SHA256,
    },
}
EASYLEED_CONFIGURATION_FILES = (
    "easyleed-independent-energy-scans.json",
    "easyleed-high-energy-reference.json",
)
EASYLEED_RUN_FROZEN_CONFIGURATION_SHA256 = (
    "54049d3d51c53f33ac8581366a4a979e4a25c2718d69f2b94e60970d6f99aecc"
)
EASYLEED_IMPLEMENTATION_HASHES = {
    "be5701c5a3f86736ae096ad3321980756e532ef457af13059ebf916ebb07e288",
    "d11818a491ed44aa045622d06f6ccc5b4f2d1acb577e02b76b1d758b473ac31c",
    "aea4de426b7c307d9230c3994ee88cc59579a95ae8bf0ce8079690e485f90933",
    "29ceb1e22e89a177a68c2af3f14ef6f331e2656daa83e71611dd4e9edcfbc23b",
}
CORE_PACKAGES = {
    "netcat": ("numpy", "scipy", "opencv-python", "scikit-image"),
    "locotrack-pytorch": (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
    ),
    "ostrack": (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "timm",
    ),
    "dam4sam": (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "hydra-core",
    ),
    "sam2": (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "hydra-core",
    ),
    "samurai": (
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "hydra-core",
    ),
    "easyleed": (
        "numpy",
        "scipy",
        "scikit-image",
        "pyqt5",
    ),
}
PACKAGE_DISPLAY_NAMES = {
    "numpy": "NumPy",
    "scipy": "SciPy",
    "opencv-python": "OpenCV",
    "scikit-image": "scikit-image",
    "torch": "PyTorch",
    "torchvision": "torchvision",
    "timm": "timm",
    "hydra-core": "Hydra",
    "pyqt5": "PyQt5",
}

IDENTITY_COLUMNS = (
    ("component", "Component"),
    ("study_role", "Study role/regime"),
    ("software_version", "Software version"),
    ("upstream_revision", "Upstream revision"),
    ("configuration_variant", "Configuration/variant"),
    ("frozen_artifact", "Frozen artifact"),
    ("artifact_type", "Artifact type"),
    ("artifact_sha256", "Artifact SHA-256"),
    ("environment_label", "Environment"),
)
HASH_COLUMNS = (
    ("component", "Component"),
    ("configuration_record_sha256", "Configuration-record SHA-256"),
    ("configuration_record_status", "Configuration-record status"),
    ("upstream_configuration_sha256", "Upstream configuration SHA-256"),
    ("implementation_record_sha256", "Implementation/adapter SHA-256"),
)
ENVIRONMENT_COLUMNS = (
    ("environment_label", "Environment"),
    ("used_for", "Used for"),
    ("runtime", "Runtime"),
    ("core_packages", "Core package versions"),
    ("execution_runtime", "Execution mode/runtime"),
    ("snapshot_sha256", "Frozen snapshot SHA-256"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return loaded


def _read_csv(
    path: Path,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"Unexpected columns in {path.name}.")
        return list(reader)


def _split_hashes(value: str) -> tuple[str, ...]:
    if value in {"not applicable", "not recorded separately"}:
        return ()
    return tuple(value.split("; "))


def _validate_hash_field(value: str) -> None:
    for digest in _split_hashes(value):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Malformed SHA-256: {digest}")


def _validate_normalized_configuration(
    *,
    component: str,
    filename: str,
    row: dict[str, str],
) -> None:
    path = CONFIGURATION_RECORDS_DIRECTORY / filename
    record = _load_json(path)
    if record.get("component") != component:
        raise ValueError(f"Configuration component changed for {component}.")
    if record.get("record_status") not in {
        "post_run_normalized_from_executed_source",
        "post_run_normalized_from_executed_source_and_output",
    }:
        raise ValueError(f"Configuration provenance changed for {component}.")
    derived = record.get("derived_from")
    if not isinstance(derived, dict):
        raise ValueError(f"Configuration source missing for {component}.")
    source = derived.get("source")
    if source is not None:
        source_path = TABLE_ROOT / str(source)
        source_hash = str(derived.get("source_sha256", ""))
        if (
            not source_path.is_file()
            or source_hash != sha256_file(source_path)
        ):
            raise ValueError(
                f"Configuration source hash changed for {component}."
            )
    if sha256_file(path) not in _split_hashes(
        row["configuration_record_sha256"]
    ):
        raise ValueError(f"Configuration-record hash changed for {component}.")


def _validate_components_against_upstream(
    rows: list[dict[str, str]],
) -> None:
    table02_rows = _read_csv(
        UPSTREAM_DIRECTORY / "table02_methods.csv",
        (
            "order",
            "panel",
            "method",
            "family_representation",
            "configuration",
            "initialization",
            "native_output",
            "proposal_coordinate",
            "reset_policy",
            "study_role",
            "upstream_revision",
            "software_version",
            "checkpoint_identifier",
            "checkpoint_sha256",
            "supporting_artifact_sha256",
            "executed_configuration_sha256",
        ),
    )
    translation = _load_json(
        UPSTREAM_DIRECTORY / "controlled_translation_contract.json"
    )
    family = _load_json(UPSTREAM_DIRECTORY / "sam2_family_contract.json")
    domain_record = (
        UPSTREAM_DIRECTORY / "domain_baselines_installation.md"
    ).read_text(encoding="utf-8")

    table02_by_key = {
        (row["panel"], row["method"]): row for row in table02_rows
    }
    component_links = {
        "LocoTrack": ("Main", "LocoTrack"),
        "OSTrack": ("Main", "OSTrack"),
        "EasyLEED": ("Main", "EasyLEED"),
        "SAM2.1": ("Controlled SAM2 family", "SAM2.1"),
        "SAMURAI": ("Controlled SAM2 family", "SAMURAI"),
        "ViPErLEED": ("External reference", "ViPErLEED"),
    }
    by_component = {row["component"]: row for row in rows}
    for component, key in component_links.items():
        source = table02_by_key[key]
        row = by_component[component]
        if row["upstream_revision"] != source["upstream_revision"]:
            raise ValueError(f"Revision changed for {component}.")

    dam_source = table02_by_key[("Main", "DAM4SAM")]
    for component in ("DAM4SAM Large", "DAM4SAM Base+"):
        if (
            by_component[component]["upstream_revision"]
            != dam_source["upstream_revision"]
        ):
            raise ValueError(f"Revision changed for {component}.")

    configurations = translation.get("method_configurations")
    configuration_hashes = translation.get("method_configuration_sha256")
    resources = translation.get("method_resources")
    if not all(
        isinstance(item, dict)
        for item in (configurations, configuration_hashes, resources)
    ):
        raise ValueError("Main-method upstream records are malformed.")

    main_contract = {
        "LocoTrack": ("locotrack_base_offline", "locotrack"),
        "OSTrack": ("ostrack_vitb_256_mae_ce", "ostrack"),
        "DAM4SAM Large": ("dam4sam_sam21pp_l", "dam4sam"),
        "Lucas–Kanade": ("lucas_kanade_skip_anchor", None),
    }
    for component, (configuration_key, resource_key) in main_contract.items():
        row = by_component[component]
        configuration = configurations[configuration_key]
        run_frozen_hash = str(configuration_hashes[configuration_key])
        if component in INDEPENDENT_ENERGY_CONFIGURATION_FILES:
            _validate_normalized_configuration(
                component=component,
                filename=INDEPENDENT_ENERGY_CONFIGURATION_FILES[component],
                row=row,
            )
            expected_configuration_hashes = {
                run_frozen_hash,
                sha256_file(
                    CONFIGURATION_RECORDS_DIRECTORY
                    / INDEPENDENT_ENERGY_CONFIGURATION_FILES[component]
                ),
            }
            if set(
                _split_hashes(row["configuration_record_sha256"])
            ) != expected_configuration_hashes:
                raise ValueError(
                    f"Configuration hashes changed for {component}."
                )
            if (
                row["configuration_record_status"]
                != MIXED_MAIN_CONFIGURATION_STATUS
            ):
                raise ValueError(
                    f"Configuration-record status changed for {component}."
                )
        else:
            if row["configuration_record_sha256"] != run_frozen_hash:
                raise ValueError(
                    f"Configuration hash changed for {component}."
                )
            if row["configuration_record_status"] != (
                "run-frozen executed configuration"
            ):
                raise ValueError(
                    f"Configuration-record status changed for {component}."
                )
        if resource_key is not None:
            resource = resources[resource_key]
            if row["artifact_sha256"] != str(resource["sha256"]):
                raise ValueError(f"Checkpoint hash changed for {component}.")
        expected_implementation_hashes = {
            str(configuration["adapter_source_sha256"])
        }
        if component in INDEPENDENT_ENERGY_CONFIGURATION_FILES:
            expected_implementation_hashes.add(
                INDEPENDENT_ENERGY_RUNNER_SHA256
            )
        if set(
            _split_hashes(row["implementation_record_sha256"])
        ) != expected_implementation_hashes:
            raise ValueError(f"Adapter hash changed for {component}.")

    common_hashes = {
        str(translation["gaussian_fitter_source_sha256"]),
        str(translation["common_measurement_source_sha256"]),
    }
    if set(
        _split_hashes(by_component["TrackLEED common measurement"][
            "artifact_sha256"
        ])
    ) != common_hashes:
        raise ValueError("Common-measurement source hashes changed.")

    controlled_family = family.get("controlled_family")
    if not isinstance(controlled_family, dict):
        raise ValueError("Controlled-family contract is missing.")
    if controlled_family.get("checkpoint_sha256") != SHARED_SAM2_CHECKPOINT:
        raise ValueError("Controlled-family checkpoint hash changed.")
    family_rows = [by_component[name] for name in EXPECTED_COMPONENTS[6:9]]
    if {row["artifact_sha256"] for row in family_rows} != {
        SHARED_SAM2_CHECKPOINT
    }:
        raise ValueError("Controlled-family component rows differ in weights.")
    for row in family_rows:
        component = row["component"]
        _validate_normalized_configuration(
            component=component,
            filename=CONTROLLED_FAMILY_CONFIGURATION_FILES[component],
            row=row,
        )
        if row["configuration_record_status"] != (
            "post-run normalized from executed source"
        ):
            raise ValueError(
                f"Configuration-record status changed for {component}."
            )
        if set(_split_hashes(row["implementation_record_sha256"])) != (
            CONTROLLED_FAMILY_IMPLEMENTATION_HASHES[component]
        ):
            raise ValueError(
                f"Implementation-source hashes changed for {component}."
            )
    if any(
        FAMILY_CONTRACT_SHA256
        in _split_hashes(row["configuration_record_sha256"])
        for row in rows
    ):
        raise ValueError(
            "Whole-family contract hash mislabeled as method configuration."
        )

    easyleed = by_component["EasyLEED"]
    easyleed_configuration_hashes = {
        EASYLEED_RUN_FROZEN_CONFIGURATION_SHA256,
        *(
            sha256_file(CONFIGURATION_RECORDS_DIRECTORY / filename)
            for filename in EASYLEED_CONFIGURATION_FILES
        ),
    }
    if set(
        _split_hashes(easyleed["configuration_record_sha256"])
    ) != easyleed_configuration_hashes:
        raise ValueError("EasyLEED configuration-record hashes changed.")
    for filename in EASYLEED_CONFIGURATION_FILES:
        path = CONFIGURATION_RECORDS_DIRECTORY / filename
        record = _load_json(path)
        if record.get("component") != "EasyLEED":
            raise ValueError("EasyLEED configuration component changed.")
        if record.get("record_status") != (
            "post_run_normalized_from_executed_source_and_output"
        ):
            raise ValueError("EasyLEED configuration provenance changed.")
        derived = record.get("derived_from")
        if not isinstance(derived, dict):
            raise ValueError("EasyLEED configuration source is missing.")
        for source_key in ("runner", "analysis"):
            source = derived.get(source_key)
            if source is None:
                continue
            source_path = TABLE_ROOT / str(source)
            source_hash = str(derived.get(f"{source_key}_sha256", ""))
            if (
                not source_path.is_file()
                or source_hash != sha256_file(source_path)
            ):
                raise ValueError(
                    "EasyLEED configuration source hash changed."
                )
        executed_outputs = derived.get("executed_outputs")
        if not isinstance(executed_outputs, list) or not executed_outputs:
            raise ValueError("EasyLEED executed-output record is missing.")
        for executed_output in executed_outputs:
            if not isinstance(executed_output, dict):
                raise ValueError("EasyLEED executed-output record is malformed.")
            output_path = TABLE_ROOT / str(executed_output.get("path", ""))
            output_hash = str(executed_output.get("sha256", ""))
            if (
                not output_path.is_file()
                or output_hash != sha256_file(output_path)
            ):
                raise ValueError("EasyLEED executed-output hash changed.")
            output_record = _load_json(output_path)
            if output_record.get("settings") != record.get("native_settings"):
                raise ValueError("EasyLEED executed settings changed.")
    if set(
        _split_hashes(easyleed["implementation_record_sha256"])
    ) != EASYLEED_IMPLEMENTATION_HASHES:
        raise ValueError("EasyLEED adapter-source hashes changed.")
    if easyleed["configuration_record_status"] != (
        "run-frozen for Cu(111)/translations; post-run normalized from "
        "executed sources and outputs for energy scans/external reference"
    ):
        raise ValueError("EasyLEED configuration-record status changed.")

    domain_anchors = (
        by_component["EasyLEED"]["software_version"],
        by_component["EasyLEED"]["upstream_revision"],
        by_component["ViPErLEED"]["software_version"],
        by_component["ViPErLEED"]["upstream_revision"],
        by_component["ViPErLEED"]["artifact_sha256"],
    )
    if any(anchor not in domain_record for anchor in domain_anchors):
        raise ValueError("Domain-baseline software record changed.")
    if (
        by_component["ViPErLEED"]["implementation_record_sha256"]
        != "not applicable"
    ):
        raise ValueError(
            "ViPErLEED runtime JAR was mislabeled as an implementation hash."
        )


def load_and_validate_components(
    source_path: Path = COMPONENTS_PATH,
) -> list[dict[str, str]]:
    """Load the frozen software, model, and configuration records."""

    rows = _read_csv(source_path, COMPONENT_COLUMNS)
    if len(rows) != 10:
        raise ValueError("Table S1 requires ten component records.")
    if [int(row["order"]) for row in rows] != list(range(1, 11)):
        raise ValueError("Component order must remain 1-10.")
    if tuple(row["component"] for row in rows) != EXPECTED_COMPONENTS:
        raise ValueError("Component order changed.")
    if any(
        row["environment_label"] not in EXPECTED_ENVIRONMENTS for row in rows
    ):
        raise ValueError("Unknown component environment.")
    for row in rows:
        for key in (
            "artifact_sha256",
            "configuration_record_sha256",
            "upstream_configuration_sha256",
            "implementation_record_sha256",
        ):
            _validate_hash_field(row[key])

    reader_text = "\n".join(
        value
        for row in rows
        for key, value in row.items()
        if key != "order"
    )
    if any(term in reader_text for term in ("ScanE", "26a21", "26a24")):
        raise ValueError("Table S1 contains internal archive identifiers.")
    _validate_components_against_upstream(rows)
    return rows


def package_version(record: dict[str, object], package_name: str) -> str:
    """Return one installed-distribution version by normalized name."""

    distributions_value = record.get("installed_distributions")
    if not isinstance(distributions_value, list):
        raise ValueError("Environment lacks installed distributions.")
    matches = [
        str(package["version"])
        for package in distributions_value
        if isinstance(package, dict) and package.get("name") == package_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {package_name} distribution in "
            f"{record.get('environment_label')}."
        )
    return matches[0]


def _validate_python_environment(record: dict[str, object]) -> None:
    if record.get("runtime_kind") != "python-conda":
        raise ValueError("Expected a Python Conda environment.")
    if record.get("platform_family") != "Windows 64-bit":
        raise ValueError("Unexpected environment platform.")
    if not str(record.get("python_version", "")).strip():
        raise ValueError("Python version is missing.")
    if not isinstance(record.get("installed_distributions"), list):
        raise ValueError("Installed-distribution inventory is missing.")
    if not isinstance(record.get("conda_packages"), list):
        raise ValueError("Conda package inventory is missing.")
    history_hash = str(record.get("conda_history_sha256", ""))
    _validate_hash_field(history_hash)

    distributions_value = record["installed_distributions"]
    distribution_keys = [
        (str(package["name"]), str(package["version"]))
        for package in distributions_value
        if isinstance(package, dict)
    ]
    if distribution_keys != sorted(distribution_keys):
        raise ValueError("Installed distributions are not sorted.")
    if len({name for name, _ in distribution_keys}) != len(
        distribution_keys
    ):
        raise ValueError("Installed distribution names are not unique.")


def _iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _is_private_or_absolute_path(value: str) -> bool:
    normalized = value.strip()
    lowered = normalized.lower()
    return bool(
        re.match(r"^[a-z]:[\\/]", normalized, flags=re.IGNORECASE)
        or normalized.startswith(("\\\\", "//"))
        or lowered.startswith(("file:", "~/", "~\\"))
        or re.search(r"(?:^|[\\/])users[\\/]", lowered)
        or re.search(r"(?:^|[\\/])home[\\/]", lowered)
        or re.search(r"(?:^|[\\/])mnt[\\/][a-z][\\/]", lowered)
    )


def load_and_validate_environments() -> list[dict[str, object]]:
    """Load eight retained, path-free execution-environment snapshots."""

    records = []
    for label in EXPECTED_ENVIRONMENTS:
        path = ENVIRONMENT_SNAPSHOTS_DIRECTORY / f"{label}.json"
        record = _load_json(path)
        if record.get("environment_label") != label:
            raise ValueError(f"Environment label changed in {path.name}.")
        if record.get("status") != "retained_environment_snapshot":
            raise ValueError(f"Unexpected status in {path.name}.")
        if record.get("capture_date") != "2026-07-29":
            raise ValueError(f"Environment capture date changed in {path.name}.")
        if label in PYTHON_ENVIRONMENT_LABELS:
            _validate_python_environment(record)
        elif record.get("runtime_kind") != "imagej-java":
            raise ValueError("ViPErLEED runtime kind changed.")
        records.append(record)

    if any(
        _is_private_or_absolute_path(value)
        for value in _iter_strings(records)
    ):
        raise ValueError("Environment snapshots contain a personal path.")
    return records


def _load_environment_roles() -> list[dict[str, str]]:
    rows = _read_csv(ENVIRONMENT_ROLES_PATH, ENVIRONMENT_ROLE_COLUMNS)
    if [int(row["order"]) for row in rows] != list(range(1, 9)):
        raise ValueError("Environment role order must remain 1-8.")
    if tuple(row["environment_label"] for row in rows) != EXPECTED_ENVIRONMENTS:
        raise ValueError("Environment role labels changed.")
    return rows


def _runtime_components(record: dict[str, object]) -> str:
    components = record.get("runtime_components")
    if not isinstance(components, list) or len(components) != 3:
        raise ValueError("ViPErLEED runtime components are incomplete.")
    return "; ".join(
        f"{component['name']} {component['version']}"
        for component in components
        if isinstance(component, dict)
    )


def build_environment_rows(
    records: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Reduce full package snapshots to the publication environment block."""

    roles = {
        row["environment_label"]: row for row in _load_environment_roles()
    }
    system_record = _load_json(SYSTEM_RECORD_PATH)
    gpu = system_record.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError("System GPU record is missing.")
    by_label = {
        str(record["environment_label"]): record for record in records
    }
    output = []
    for label in EXPECTED_ENVIRONMENTS:
        record = by_label[label]
        role = roles[label]
        snapshot_path = ENVIRONMENT_SNAPSHOTS_DIRECTORY / f"{label}.json"
        if label in PYTHON_ENVIRONMENT_LABELS:
            runtime = f"Python {record['python_version']}"
            core_packages = "; ".join(
                f"{PACKAGE_DISPLAY_NAMES[name]} "
                f"{package_version(record, name)}"
                for name in CORE_PACKAGES[label]
            )
            if role["execution_mode"] == "CUDA":
                execution_runtime = (
                    f"CUDA; PyTorch build CUDA "
                    f"{record['torch_cuda_build']}"
                )
            else:
                execution_runtime = role["execution_mode"]
        else:
            runtime = _runtime_components(record)
            core_packages = "not applicable"
            execution_runtime = role["execution_mode"]
        output.append(
            {
                "environment_label": label,
                "used_for": role["used_for"],
                "runtime": runtime,
                "core_packages": core_packages,
                "execution_runtime": execution_runtime,
                "snapshot_sha256": sha256_file(snapshot_path),
            }
        )

    cuda_rows = [
        row for row in output if row["execution_runtime"].startswith("CUDA")
    ]
    if len(cuda_rows) != 5:
        raise ValueError("Expected five CUDA execution environments.")
    if not all(str(gpu[key]).strip() for key in ("model", "driver_version")):
        raise ValueError("System GPU identification is incomplete.")
    return output


def _markdown_cell(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ").strip()


def _render_block(
    title: str,
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, str]],
) -> list[str]:
    headers = [label for _, label in columns]
    lines = [
        f"**{title}**",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row[source_name])
                for source_name, _ in columns
            )
            + " |"
        )
    return lines


def render_markdown(
    components: list[dict[str, str]],
    environments: list[dict[str, str]],
) -> str:
    """Render the exact three-block Table S1 supplement text."""

    system = _load_json(SYSTEM_RECORD_PATH)
    gpu = system["gpu"]
    lines = [
        (
            "**Table S1. Software, models, versions, hashes and "
            "environments.** "
            "Software and model identities and the run-frozen records in "
            "blocks A and B come from the completed execution contracts. "
            "Where a method-level configuration was not frozen during a run, "
            "block B instead reports a post-run normalized record derived "
            "from the archived executed source; the status column "
            "distinguishes these cases. "
            "The package inventories in block C were captured on 2026-07-29 "
            "after the completed runs from the retained local environments; "
            "they document the preserved runtimes, but this record does not "
            "establish bitwise reproducibility of GPU inference. The retained "
            "execution "
            f"host was `{system['operating_system']}` with an "
            f"{gpu['model']} ({gpu['memory_mib']} MiB, driver "
            f"{gpu['driver_version']})."
        ),
        "",
    ]
    lines.extend(
        _render_block(
            "A. Software and model identity",
            IDENTITY_COLUMNS,
            components,
        )
    )
    lines.extend([""])
    lines.extend(
        _render_block(
            "B. Configuration and implementation hashes",
            HASH_COLUMNS,
            components,
        )
    )
    lines.extend([""])
    lines.extend(
        _render_block(
            "C. Retained execution environments",
            ENVIRONMENT_COLUMNS,
            environments,
        )
    )
    return "\n".join(lines) + "\n"


def extract_supplement_table(supplement: str) -> str:
    """Return normalized generated text between Table S1 markers."""

    if supplement.count(SUPPLEMENT_BEGIN_MARKER) != 1:
        raise ValueError("Supplement must contain one Table S1 begin marker.")
    if supplement.count(SUPPLEMENT_END_MARKER) != 1:
        raise ValueError("Supplement must contain one Table S1 end marker.")
    start = supplement.index(SUPPLEMENT_BEGIN_MARKER) + len(
        SUPPLEMENT_BEGIN_MARKER
    )
    end = supplement.index(SUPPLEMENT_END_MARKER, start)
    return supplement[start:end].strip("\n") + "\n"


def sync_supplement_table(markdown: str) -> None:
    """Replace the sole Table S1 marker block with generated Markdown."""

    supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
    if supplement.count(SUPPLEMENT_BEGIN_MARKER) != 1:
        raise ValueError("Supplement must contain one Table S1 begin marker.")
    if supplement.count(SUPPLEMENT_END_MARKER) != 1:
        raise ValueError("Supplement must contain one Table S1 end marker.")
    start = supplement.index(SUPPLEMENT_BEGIN_MARKER) + len(
        SUPPLEMENT_BEGIN_MARKER
    )
    end = supplement.index(SUPPLEMENT_END_MARKER, start)
    synchronized = (
        supplement[:start]
        + "\n"
        + markdown.rstrip("\n")
        + "\n"
        + supplement[end:]
    )
    SUPPLEMENT_PATH.write_text(synchronized, encoding="utf-8", newline="\n")


def validate_provenance() -> dict[str, object]:
    """Validate committed snapshots of upstream contracts and run sources."""

    provenance = _load_json(SOURCE_PROVENANCE_PATH)
    sources = provenance.get("sources")
    if not isinstance(sources, list) or len(sources) != 9:
        raise ValueError("Source provenance must list nine upstream records.")
    required_keys = {
        "frozen_path",
        "frozen_sha256",
        "origin_path",
        "origin_sha256",
        "role",
    }
    for source in sources:
        if not isinstance(source, dict) or set(source) != required_keys:
            raise ValueError("Malformed source-provenance entry.")
        for digest_key in ("frozen_sha256", "origin_sha256"):
            _validate_hash_field(str(source[digest_key]))
        frozen_path = (TABLE_ROOT / str(source["frozen_path"])).resolve()
        if TABLE_ROOT not in frozen_path.parents:
            raise ValueError("Frozen provenance path escapes the artifact.")
        if not frozen_path.is_file():
            raise ValueError(f"Missing frozen source: {frozen_path.name}")
        if sha256_file(frozen_path) != source["frozen_sha256"]:
            raise ValueError(f"Frozen source hash changed: {frozen_path.name}")
        origin_path = (
            REPOSITORY_ROOT / str(source["origin_path"])
        ).resolve()
        if (
            origin_path != REPOSITORY_ROOT
            and REPOSITORY_ROOT not in origin_path.parents
        ):
            raise ValueError("Origin provenance path escapes the repository.")
        if not origin_path.is_file():
            raise ValueError(f"Missing origin source: {origin_path.name}")
        if sha256_file(origin_path) != source["origin_sha256"]:
            raise ValueError(f"Origin source hash changed: {origin_path.name}")
    return provenance


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fieldnames} for row in rows
        )


def build_artifact(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    """Write deterministic Table S1 CSV/Markdown outputs and manifest."""

    components = load_and_validate_components()
    environment_records = load_and_validate_environments()
    environments = build_environment_rows(environment_records)
    provenance = validate_provenance()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "table_s1_components.csv": output_directory
        / "table_s1_components.csv",
        "table_s1_environments.csv": output_directory
        / "table_s1_environments.csv",
        "table_s1_versions_hashes_environments.md": output_directory
        / "table_s1_versions_hashes_environments.md",
    }
    component_fields = [
        field for field in COMPONENT_COLUMNS if field != "order"
    ]
    environment_fields = [source for source, _ in ENVIRONMENT_COLUMNS]
    _write_csv(
        output_paths["table_s1_components.csv"],
        components,
        component_fields,
    )
    _write_csv(
        output_paths["table_s1_environments.csv"],
        environments,
        environment_fields,
    )
    output_paths["table_s1_versions_hashes_environments.md"].write_text(
        render_markdown(components, environments),
        encoding="utf-8",
        newline="\n",
    )

    source_paths = sorted(
        path for path in SOURCE_DATA_DIRECTORY.rglob("*") if path.is_file()
    )
    environment_paths = sorted(
        ENVIRONMENT_SNAPSHOTS_DIRECTORY.glob("*.json")
    )
    manifest: dict[str, object] = {
        "schema_version": "1.1",
        "status": "completed",
        "table_contract": {
            "component_count": 10,
            "environment_count": 8,
            "python_environment_count": 7,
            "imagej_environment_count": 1,
            "controlled_family_weights_identical": True,
            "controlled_family_checkpoint_sha256": SHARED_SAM2_CHECKPOINT,
            "configuration_record_provenance_explicit": True,
            "post_run_normalized_configuration_record_count": 8,
            "viperleed_imagej_runtime_artifact_reported_in_environment_snapshot": (
                True
            ),
            "environment_record_timing": (
                "retained environments captured after completed runs"
            ),
            "gpu_inference_reproducibility_claim": (
                "package inventory only; no bitwise-reproducibility claim"
            ),
        },
        "source_data": {
            path.relative_to(TABLE_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        },
        "environment_snapshots": {
            path.name: sha256_file(path) for path in environment_paths
        },
        "upstream_sources": provenance["sources"],
        "capture_script_sha256": sha256_file(CAPTURE_SCRIPT_PATH),
        "builder_sha256": sha256_file(Path(__file__)),
        "outputs": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


if __name__ == "__main__":
    build_artifact()
    sync_supplement_table(
        (
            OUTPUT_DIRECTORY / "table_s1_versions_hashes_environments.md"
        ).read_text(encoding="utf-8")
    )
