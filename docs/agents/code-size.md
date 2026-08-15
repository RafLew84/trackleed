# Code-size guardrail

Apply this guardrail to every created or modified Python file. Count physical
lines and every function or method definition, including async and nested
definitions.

## Production code

- Design for 300-600 lines per file and fewer than 30 definitions.
- Before a change takes a file above 600 lines or 30 definitions, review where
  a cohesive deep module can be extracted at a real seam.
- New files and files currently below the hard limits finish at no more than
  1000 lines and 50 definitions.
- A function or method finishes at no more than 80 physical lines; target fewer
  than 40.
- A legacy file already above a limit follows a ratchet: the change finishes
  with no increase in either its line count or definition count.

Extraction must concentrate behavior behind a small interface. A mechanical
split into shallow pass-through files does not satisfy the guardrail.

## Tests and exclusions

- Test files finish at no more than 1500 lines and 75 definitions. Existing
  test files above either limit follow the same ratchet.
- Generated code, vendored code, frozen artifacts, and `third_party/` are
  excluded.

## Completion check

Before handing off a code change:

1. Count lines and definitions in every created or modified Python file.
2. Compare oversized legacy files with their fixed-point version.
3. Record any threshold-triggered seam review in the originating issue or PR.
4. Run the repository code-size check when it exists; a later implementation
   ticket must add this fail-closed check to local validation and CI.

The change is complete only when every in-scope file satisfies its hard limit
or legacy ratchet.
