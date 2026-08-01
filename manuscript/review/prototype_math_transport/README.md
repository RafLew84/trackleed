# PROTOTYPE — Markdown-to-DOCX math transport

This throwaway prototype answers one question: which Pandoc authoring/export
contract preserves the manuscript's existing `\(...\)` and `\[...\]` math in
Word as native OMath?

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File manuscript/review/prototype_math_transport/run_prototype.ps1
```

The runner compares:

1. `current-default`: existing single-backslash source with Pandoc's default
   Markdown reader;
2. `dollar-default`: rewritten `$...$` / `$$...$$` source with the default
   reader;
3. `single-backslash-reader`: existing source with
   `markdown+tex_math_single_backslash`.

`results/summary.json` records native OMath counts and critical-symbol checks.
Generated DOCX files and render output are disposable evidence, not production
artifacts.

