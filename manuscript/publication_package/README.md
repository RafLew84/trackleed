# Canonical publication package

This directory is the only canonical document-export path for the TrackLEED
main manuscript and supplementary information. The builder assembles every
numbered main and supplementary figure, converts both Markdown sources with
Pandoc reader `markdown+tex_math_single_backslash`, audits native Word OMath,
then exports each audited DOCX to PDF with Microsoft Word.

Run from the repository root in the `netcat` environment:

```powershell
C:\Users\rlewa\anaconda3\envs\netcat\python.exe manuscript\publication_package\build_publication_package.py
```

`manifest.json` names and hashes exactly four canonical reader-facing files:
main DOCX/PDF and supplementary-information DOCX/PDF. The legacy files under
`manuscript/review/` are explicitly excluded by `document_contract.json` and
must not be distributed as canonical outputs.

The build fails on Pandoc warnings, Math/OMath/fallback count mismatch,
per-expression source/DOCX round-trip mismatch, empty native OMath, non-empty
initial fallback declarations, missing displays, or unallowlisted damaged-math
signatures. Verify an existing package without rebuilding with `--verify`.

Run the real Pandoc/Microsoft Word export twice and compare the hashes of all
four canonical outputs with:

```powershell
C:\Users\rlewa\anaconda3\envs\netcat\python.exe manuscript\publication_package\build_publication_package.py --check-reproducible
```
