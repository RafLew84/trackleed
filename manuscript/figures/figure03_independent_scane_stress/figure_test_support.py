"""Shared independent assertions for Figure 3 contract tests."""

from __future__ import annotations

import unittest
from typing import Sequence


EXPECTED_CASE_COUNT_FIELDS = (
    "visible_opportunities",
    "branch_consistent_measurements",
    "evaluated_gaps",
    "recovered_gaps",
    "accepted_measurements_all_frames",
    "geometry_inconsistent_accepted_fits",
    "jointly_consistent_pair_frames",
)


def assert_case_rows_reproduce_aggregates(
    test_case: unittest.TestCase,
    aggregate_rows: Sequence[dict[str, object]],
    case_rows: Sequence[dict[str, object]],
) -> None:
    for aggregate_row in aggregate_rows:
        method_rows = [
            row
            for row in case_rows
            if row["method"] == aggregate_row["method"]
        ]
        test_case.assertEqual(len(method_rows), 4)
        for field in EXPECTED_CASE_COUNT_FIELDS:
            test_case.assertEqual(
                sum(int(row[field]) for row in method_rows),
                aggregate_row[field],
            )
