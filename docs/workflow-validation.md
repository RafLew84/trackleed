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
all configured mandatory tests. Those tests include immutable v1/v2 packages.
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

`trackleed/tests/fixtures/synthetic/` contains two historical ZIP archives,
totalling about 18 KB. The fixture README records their producer commits and
SHA-256 digests. Tests verify the pinned bytes, extract temporary copies,
read them with the current public reader, and check independent expected
values. They never call the generator to recreate expected packages.
New versions get new fixtures; old fixtures and their pinned hashes stay fixed.

The mandatory suite includes deliberately invalid temporary repositories and
corrupted copies of both packages. It asserts nonzero CLI exits with actionable
diagnostics for invalid text and type errors, and rejection of damaged historical
package data. No broken examples are written into the user's working files.
