"""Public contract for the final cross-regime supplementary table."""

from __future__ import annotations

import unittest

import build_table


class TableS7ContractTest(unittest.TestCase):
    def test_table_s7_is_supplementary_and_preserves_denominators(self) -> None:
        rows = build_table.load_and_validate_source()
        blocks = build_table.build_blocks(rows)
        rendered = build_table.render_markdown(blocks)

        self.assertIn("Table S7. Cross-regime numerical summary", rendered)
        self.assertIn("80 displaced-spot", rendered)
        self.assertIn("40 displaced-pair", rendered)
        self.assertIn("post-gap branch-consistent recovery", rendered.lower())
        self.assertNotIn("Table 3", rendered)
        self.assertNotIn("identity recovery", rendered.lower())
        self.assertNotIn("| Composite score |", rendered)
        self.assertNotIn("| Rank |", rendered)


if __name__ == "__main__":
    unittest.main()
