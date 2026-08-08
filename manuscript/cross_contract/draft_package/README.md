# Cross-contract scientific draft package

This package is the canonical editable and rendered handoff for the frozen
cross-contract manuscript and Supplementary Information. It transforms the
venue-portable Markdown sources into an A4, two-column scientific draft while
preserving their scientific content, display budgets, denominators, canonical
terminology, and two Authorized deferred publication inputs.

The build is intentionally two-phase. `--prepare` creates a private staging
package and page PNGs. A reviewer must inspect every rendered page before
`--publish-reviewed` can record Page-level render acceptance and publish the
build-once package. Page images are QA intermediates; the accepted page hashes,
checks, renderer provenance, and reviewer decision remain in the frozen
package and are reproduced during validation.

From the repository root, use the documented `netcat` environment:

```powershell
C:\Users\rlewa\anaconda3\envs\netcat\python.exe scripts\build_cross_contract_draft_package.py --prepare --staging <path>
C:\Users\rlewa\anaconda3\envs\netcat\python.exe scripts\build_cross_contract_draft_package.py --publish-reviewed --staging <path> --reproduction-staging <independent-path> --confirm-page-acceptance --reviewer-identity <identity> --review-date <YYYY-MM-DD>
C:\Users\rlewa\anaconda3\envs\netcat\python.exe scripts\build_cross_contract_draft_package.py --validate-only
```

`handoff_report.json` distinguishes scientific-draft completion from
`submission-ready` and names the exact unresolved external inputs. Full
scientific records remain in their upstream manifested packages.
