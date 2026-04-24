from __future__ import annotations

import importlib
import sys
import types
import unittest


DEPENDENCY_MODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "trackleed.analysis.cotracker_backend": (
        "format_cotracker_environment_report",
        "probe_cotracker_environment",
    ),
    "trackleed.analysis.cutie_backend": (
        "format_cutie_environment_report",
        "probe_cutie_environment",
    ),
    "trackleed.analysis.locotrack_backend": (
        "format_locotrack_environment_report",
        "probe_locotrack_environment",
    ),
    "trackleed.analysis.mftiq_backend": (
        "format_mftiq_environment_report",
        "probe_mftiq_environment",
    ),
    "trackleed.analysis.pipspp_backend": (
        "format_pipspp_environment_report",
        "probe_pipspp_environment",
    ),
    "trackleed.analysis.sam2_backend": (
        "format_sam2_environment_report",
        "probe_sam2_environment",
    ),
    "trackleed.analysis.tapir_backend": (
        "format_tapir_environment_report",
        "probe_tapir_environment",
    ),
    "trackleed.analysis.tapnext_backend": (
        "format_tapnext_environment_report",
        "probe_tapnext_environment",
    ),
    "trackleed.analysis.tomp_backend": (
        "format_tomp_environment_report",
        "probe_tomp_environment",
    ),
    "trackleed.analysis.trackonr_backend": (
        "format_trackonr_environment_report",
        "probe_trackonr_environment",
    ),
    "trackleed.analysis.tracking_bbox_backends": ("track_spots_tomp_offline",),
    "trackleed.analysis.tracking_mask_backends": (
        "track_spots_cutie_offline",
        "track_spots_sam2_offline",
        "track_spots_xmem2_offline",
    ),
    "trackleed.analysis.tracking_point_backends": (
        "track_spots_cotracker3_offline",
        "track_spots_locotrack_offline",
        "track_spots_mftiq_offline",
        "track_spots_pipspp_offline",
        "track_spots_tapir_offline",
        "track_spots_tapir_online",
        "track_spots_tapnext_offline",
        "track_spots_trackonr_offline",
    ),
    "trackleed.analysis.tracking_retrack": (
        "retrack_single_spot_cotracker3_offline",
        "retrack_single_spot_cutie_offline",
        "retrack_single_spot_locotrack_offline",
        "retrack_single_spot_mftiq_offline",
        "retrack_single_spot_pipspp_offline",
        "retrack_single_spot_sam2_offline",
        "retrack_single_spot_tapir_offline",
        "retrack_single_spot_tapir_online",
        "retrack_single_spot_tapnext_offline",
        "retrack_single_spot_tomp_offline",
        "retrack_single_spot_trackonr_offline",
        "retrack_single_spot_xmem2_offline",
    ),
    "trackleed.analysis.xmem2_backend": (
        "format_xmem2_environment_report",
        "probe_xmem2_environment",
    ),
}


def _stub_callable(qualified_name: str):
    def _fn(*args, **kwargs):
        return {"qualified_name": qualified_name, "args": args, "kwargs": kwargs}

    _fn.__name__ = qualified_name.rsplit(".", 1)[-1]
    return _fn


class TrackerRegistryImportMixin:
    registry_module_name = "trackleed.gui.tracker_registry"

    def import_tracker_registry(self):
        originals: dict[str, types.ModuleType | None] = {}
        sys.modules.pop(self.registry_module_name, None)
        for module_name, attr_names in DEPENDENCY_MODULE_ATTRS.items():
            originals[module_name] = sys.modules.get(module_name)
            stub_module = types.ModuleType(module_name)
            for attr_name in attr_names:
                setattr(stub_module, attr_name, _stub_callable(f"{module_name}.{attr_name}"))
            sys.modules[module_name] = stub_module
        self.addCleanup(self._restore_modules, originals)
        self.addCleanup(sys.modules.pop, self.registry_module_name, None)
        return importlib.import_module(self.registry_module_name)

    def _restore_modules(self, originals: dict[str, types.ModuleType | None]) -> None:
        for module_name, original_module in originals.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module


class TrackerRegistryContractTests(TrackerRegistryImportMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.registry = self.import_tracker_registry()

    def test_tracker_specs_are_complete_and_unique(self) -> None:
        specs = self.registry.iter_tracker_specs()
        self.assertEqual(specs, self.registry.TRACKER_SPECS)
        self.assertGreaterEqual(len(specs), 12)
        self.assertEqual(
            self.registry.TRACKING_SCENARIO_NAMES,
            tuple(spec.scenario_name for spec in specs),
        )

        tracker_ids = [spec.tracker_id for spec in specs]
        scenario_names = [spec.scenario_name for spec in specs]
        source_names = [source for spec in specs for source in spec.source_names]

        self.assertEqual(len(tracker_ids), len({value.lower() for value in tracker_ids}))
        self.assertEqual(len(scenario_names), len({value.lower() for value in scenario_names}))
        self.assertEqual(len(source_names), len({value.lower() for value in source_names}))

        required_string_fields = (
            "tracker_id",
            "display_name",
            "scenario_name",
            "settings_type_name",
            "runtime_settings_type_name",
            "collect_params_method_name",
        )

        for spec in specs:
            for field_name in required_string_fields:
                self.assertTrue(
                    getattr(spec, field_name),
                    msg=f"TrackerSpec `{spec.tracker_id}` missing `{field_name}`.",
                )
            self.assertIn(spec.family, ("point", "bbox", "mask"))
            self.assertTrue(spec.source_names)
            self.assertTrue(callable(spec.probe_environment))
            self.assertTrue(callable(spec.format_environment_report))
            self.assertTrue(callable(spec.run_tracking))
            self.assertTrue(callable(spec.run_retrack))
            self.assertIsNotNone(spec.settings_attr_name)
            self.assertIsNotNone(spec.params_argument_name)
            self.assertIsNotNone(spec.panel_group_attr)
            self.assertIsNotNone(spec.warning_label_attr)

    def test_registry_lookups_normalize_case_and_whitespace(self) -> None:
        for spec in self.registry.TRACKER_SPECS:
            self.assertIs(
                self.registry.get_tracker_spec(f"  {spec.tracker_id.upper()}  "),
                spec,
            )
            self.assertIs(
                self.registry.get_tracker_spec_for_scenario(f"  {spec.scenario_name.upper()}  "),
                spec,
            )
            self.assertIs(
                self.registry.require_tracker_spec(spec.tracker_id),
                spec,
            )
            self.assertIs(
                self.registry.require_tracker_spec_for_scenario(spec.scenario_name),
                spec,
            )
            for source_name in spec.source_names:
                self.assertIs(
                    self.registry.get_tracker_spec_for_source(f"  {source_name.upper()}  "),
                    spec,
                )
                self.assertIs(
                    self.registry.require_tracker_spec_for_source(source_name),
                    spec,
                )

        tomp_spec = self.registry.require_tracker_spec("tomp_offline")
        self.assertIs(
            self.registry.get_tracker_spec_for_source("tamos_offline"),
            tomp_spec,
        )

    def test_require_helpers_raise_for_unknown_values(self) -> None:
        self.assertIsNone(self.registry.get_tracker_spec("missing_tracker"))
        self.assertIsNone(self.registry.get_tracker_spec_for_scenario("missing scenario"))
        self.assertIsNone(self.registry.get_tracker_spec_for_source("missing_source"))

        with self.assertRaises(KeyError):
            self.registry.require_tracker_spec("missing_tracker")
        with self.assertRaises(KeyError):
            self.registry.require_tracker_spec_for_scenario("missing scenario")
        with self.assertRaises(KeyError):
            self.registry.require_tracker_spec_for_source("missing_source")

    def test_unique_index_builder_rejects_invalid_or_conflicting_keys(self) -> None:
        spec_a = self.registry.TRACKER_SPECS[0]
        spec_b = self.registry.TRACKER_SPECS[1]

        with self.assertRaises(ValueError):
            self.registry._build_unique_index([(" ", spec_a)], index_name="blank")

        with self.assertRaises(ValueError):
            self.registry._build_unique_index(
                [("dup", spec_a), (" DUP ", spec_b)],
                index_name="collision",
            )

