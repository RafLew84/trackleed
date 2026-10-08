# Workflow validation

Install the same dependencies as CI into the project's Python environment:

```sh
python -m pip install -r requirements-validation.txt
```

Then use this single command from the repository root, locally and in CI:

```sh
python -X utf8 scripts/validate_current_workflow.py
```

The startup flag enables UTF-8 before Python reads or imports project code.
The command refuses a process started without UTF-8 mode and prints the correct
invocation. It supplies the same headless defaults as CI: `QT_QPA_PLATFORM=offscreen`
and `MPLBACKEND=Agg`. Existing explicit values are respected.

The gate checks changed text, generator types, the code-size guardrail, and
all configured mandatory tests. Those tests include immutable historical
benchmark packages.
A missing dependency, missing fixture, failed test, skip or expected failure
makes the command fail; acquisition-dependent publication tests retain their
existing separate, explicitly requested local-reference workflow.

## Changed text

The default comparison includes the latest commit relative to its first parent,
plus staged changes, unstaged changes and non-ignored untracked files.
The initial commit is compared with an empty tree. Current working-tree bytes
are checked; independently checking staged bytes prevents a clean working copy
from hiding a broken version prepared for commit. Deleted files are ignored.

For a wider comparison, provide a commit:

```sh
python -X utf8 scripts/validate_current_workflow.py --text-base <commit>
```

`TRACKLEED_VALIDATION_BASE` supplies the same default for CI. Pull requests use
the merge-base with their target branch; pushes use the previous pushed commit,
so an earlier commit in a multi-commit push cannot escape checking. A first push
uses `EMPTY` to check all checked-out tracked files, plus local changes.
Missing comparison history is an error; fetch it instead of skipping the gate.

Only regular checked-out files and staged blobs are checked. Binary formats,
binary data without a known text extension, and explicit Git `-text` attributes
are exempt; frozen upstream bytes and sparse-excluded acquisition artifacts
retain their existing treatment. Known source/document suffixes and explicit
Git `text` attributes are always treated as text, even if corrupted with NULs.

Valid UTF-8 (with or without an initial BOM), consistent LF and consistent CRLF
are accepted. Invalid UTF-8, NUL in text, bare CR, CRCRLF and mixed LF/CRLF are
rejected. The gate reports the path, version being checked and cause; it never
rewrites files. For example:

```text
notes.md (working tree): damaged CRCRLF line endings; use consistent LF or CRLF.
notes.md (staged):1: invalid UTF-8 at byte 0; save as UTF-8.
```

## Generator types and frozen packages

The required mypy version is pinned in `requirements-validation.txt`.
`mypy-generator.ini` isolates the check from user-wide configuration.
The fixed scope is `trackleed/synthetic/`,
`scripts/generate_tracking_benchmark.py`, and
`scripts/export_segmentation_to_png.py`. A missing target or unavailable mypy
fails explicitly. A type error includes mypy's file, line and explanation.

The [frozen package catalog](../trackleed/tests/fixtures/synthetic/README.md)
is the canonical inventory of historical archives, their coverage, producer
commits and SHA-256 digests. Tests verify the pinned bytes, extract temporary
copies, read them with the current public reader, and check independent expected
values. They never call the generator to recreate expected packages.
New versions get new fixtures; old fixtures and their pinned hashes stay fixed.

The mandatory suite includes deliberately invalid temporary repositories and
corrupted copies of historical packages. It asserts nonzero CLI exits with
actionable diagnostics for invalid text and type errors, and rejection of damaged historical
package data. No broken examples are written into the user's working files.

## Results and full logs

Each run prints a short result and an absolute `Full log:` path, on stdout for
success or stderr for failure. The error summary keeps the first cause (including
the first file/type diagnostic); further errors and tracebacks remain in the log.
Logs use UTF-8 and unique names under `.scratch/validation/`, so another run does
not overwrite earlier evidence. They contain the verbose test transcript, Python
stdout/stderr, native diagnostics and output inherited by child processes.
Failure to create a log stops validation with an explicit error.
In CI, the printed path refers to the log in the runner's working directory;
its lifetime follows the runner's workspace retention.

## Implementation acceptance order

Follow this order for each implementation:

1. Small tests for the change and affected regressions; check types regularly.
2. Read all immutable historical benchmark packages with the current reader:
   `python -X utf8 -m unittest trackleed.tests.test_synthetic_frozen -v`.
3. Review Standards and Spec, fix findings, and repeat affected small checks.
4. Run the complete `python -X utf8 scripts/validate_current_workflow.py` gate
   after the review fixes. Record its result and full log path.
5. Generate and verify the stage's large 1000 × 1000 demonstration in a new
   directory. The final generator stage requires the complete two-second example.

A failed step blocks the large demonstration. If the demonstration exposes a bug,
return to small tests, review the fix, and repeat full validation before trying
again. Tools/documentation-only work may record the demonstration as not
applicable with a reason. Preserve existing work and commit only the task's own
changes after acceptance.

The local implementation plan under
`.scratch/trackleed-synthetic-benchmark-implementation/` carries these same
conditions. Links to resolved specification tickets point directly to their
`#answer` sections; links to maps and other documents without an Answer section
continue to point to those documents.
