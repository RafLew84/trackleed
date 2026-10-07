# Validation progress

Start the existing gate once, from the repository root:

```sh
python -X utf8 scripts/validate_current_workflow.py
```

While it runs, read the latest run without starting any checks:

```sh
python -X utf8 scripts/validate_current_workflow.py --status
```

For a specific run, pass its `.status.json` path after `--status`. Every run
has a unique pair under `.scratch/validation/`: `workflow-*.log` and
`workflow-*.status.json`. The JSON reports the PID, UTC start/update times,
current stage and its start time, last completed step, completed/total tests,
last completed test, terminal outcome and full log path. The read command
adds elapsed seconds; it does not change files or launch validation.

Stages are `startup`, `changed_text`, `generator_types`, `code_size`,
`test_discovery`, `tests`, and optional `reference_validation`. A completed
step has finished executing; it may have found an error. The final outcome
is `passed`, `failed`, or `interrupted` (KeyboardInterrupt, exit 130).
The exit code of `--status` describes whether the snapshot was read; inspect
its `status` field for the outcome of validation.

Snapshots are replaced atomically. Fast test completions are coalesced to
at most one update per second after the first test; step transitions and
terminal outcomes are always published. Test output is flushed to the full
log while tests run. The final console output remains short.

The snapshot is evidence of the last reported progress, not a process-health
probe. An unchanged snapshot can mean a long check or a slow test import.
A forcibly killed process or machine shutdown cannot publish a terminal
snapshot. Use the reported PID and log for a targeted check when needed;
an old `running` snapshot alone does not justify starting another run.

Wait on the existing process, inspecting status at 30–60 second intervals.
Only add a diagnostic probe if its result could change the next action.
For a runtime beyond a measured baseline or absent progress, measure a small
reproduction before changing the environment. Compare the same representative
file reads or imports in the candidate environments, record their timings,
and expand a copy or migration only after the measurement supports it.

The gate still checks the same mandatory suite and rejects failures, skips,
and expected failures. Progress reporting does not relax those checks.
