# Definition of Done: publication content revision

Decision date: 2026-08-02

## Accepted status

The terminal status for this review cycle is `content-revision-complete`. It
means that the approved scientific prose, denominator semantics, sensitivity
narrative, display contracts, and readability changes have been implemented,
scientifically reconstructed, rebaselined, rendered, and accepted.

`Content-revision-complete` can coexist with `scientific-baseline-complete`.
It does not mean `submission-ready`.

## Domain-modeling record and versioning gap

The decision introduces four durable terms:

- **Content-revision-complete** — all approved content, denominator,
  sensitivity, display, reconstruction, rebaseline, and page-level acceptance
  gates in this record have passed; this is not `submission-ready`.
- **Page-level render acceptance** — fail-closed review of every rendered page
  of every canonical reader-facing document, with the reviewed page set and
  acceptance decision recorded.
- **Controlled scientific-core rebaseline** — authorized, versioned,
  predecessor-to-successor fingerprint replacement with preserved history,
  component diff, atomic write, and repeatable clean-checkout validation.
- **Authorized deferred publication input** — an explicit `AUTHOR`,
  `EXPERIMENT`, `RELEASE-AUTHORIZATION`, or `SUBMISSION` placeholder permitted
  at `content-revision-complete` but not treated as completed content.

These entries were added to the workspace `context.md` during domain modeling,
but that file is excluded by the repository's current `.gitignore` and is not a
versioned shared artifact. This is an explicit domain-modeling gap, not silent
omission. Ticket 09 must carry the definitions into its handoff and require the
implementation owner either to version `context.md` deliberately or place them
in another tracked canonical domain document before asserting
`content-revision-complete`.

## Normative acceptance sequence

The gates below are ordered and fail closed. A later gate cannot waive a
failure in an earlier gate.

### 1. Approved scope implemented

- Implement every accepted decision from tickets 02–07.
- Preserve the common-measurement-contract-first contribution hierarchy,
  case-based claim boundary, acquisition-closed scope, and prohibition on a
  composite or universal tracker ranking.
- Preserve all fixed opportunity sets and outcome-specific conditional
  denominators, including the approved `24`-state inventory versus `80/40`
  displaced opportunity distinction.
- Integrate all 15 sensitivity variants without optimization or automatic
  central-contract selection.
- Keep the Abstract self-contained, with no references to tables, figures, or
  Supplementary Information.
- Do not add new acquisition, synthetic ground truth, tracker inference, GUI
  execution, or GPU requirements.

### 2. Targeted source and display validation

- Run the targeted tests and builders for every changed manuscript, SI,
  table, figure, inventory, export, and validation seam.
- Confirm Table S6 contains 75 method aggregates, preserves the 375 canonical
  records, and emits full-precision aggregate CSV and two-sheet XLSX.
- Confirm Figure S2 remains one numbered display item across `a–d` and `e–f
  (continued)` and meets the effective 6.0-pt and 4.5-mm legibility gates.
- Confirm Figure 5 remains full-width, all panels remain present, and its
  smallest effective font is at least 6.0 pt.
- Confirm the DOCX math audit has equal expected Pandoc `Math` and native OMath
  coverage, no silent text fallback, no conversion warnings, and no
  unallowlisted damaged-math signatures.

### 3. Portability closed before rebaseline

- Adopt one explicit LF contract for hash-bound text and enforce it through
  `.gitattributes` and deterministic writers using explicit `newline="\n"` or
  byte writes.
- Demonstrate that a clean Windows checkout with all authenticated local
  inputs rebuilds the same scientific artifacts without line-ending hash
  mismatches.
- Run the old fingerprint against the implemented revision and require it to
  reject the reviewed changed components. An unexpected pass is a coverage
  failure and blocks rebaseline.

### 4. Scientific acceptance checkpoint

- Review the final scientific prose diff, display semantic inventory, claim-
  evidence map, limitations, denominators, and final claim boundary.
- Record explicit user acceptance of the final content and claim boundary.
- Bind that acceptance, the decision records, and the exact implementation
  commit to the rebaseline authorization.

### 5. Controlled scientific-core rebaseline

- Use a versioned replacement command; never delete `fingerprint.json`
  manually.
- Require expected predecessor SHA-256
  `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`,
  the approved decision record, and the exact implementation commit.
- Preserve the predecessor fingerprint in versioned history.
- Record the reviewed diff of all 23 fingerprint components, the reason for
  replacement, authorization, runtime, and predecessor-to-successor link.
- Write the successor atomically and fail on predecessor mismatch, dirty or
  unapproved scientific inputs, incomplete component coverage, or missing
  authorization.

### 6. Post-rebaseline scientific reconstruction

All of the following must pass with one successor digest and no artifact drift:

1. `python -m scripts.freeze_scientific_core --validate-only`;
2. `python -m scripts.reconstruct_publication --clean --validate` in the pinned
   CPU/headless environment;
3. a second full reconstruction from a clean Windows checkout with all
   manifest-declared authenticated local inputs.

The runs exclude tracker inference, GUI, GPU, editorial rendering, upload, DOI
creation, and reviewer-access release construction.

### 7. Canonical document build and page-level render acceptance

- The canonical reader-facing deliverables are the main manuscript and
  Supplementary Information.
- Build each from versioned sources to DOCX and PDF using the approved
  `markdown+tex_math_single_backslash` reader and a versioned manifest.
- Render each DOCX through the pinned document-rendering engine to a
  DOCX-origin PDF and page-image set. Independently render every page of each
  canonical PDF to a PDF-origin page-image set.
- Audit 100% of pages in all four canonical renditions: main DOCX, main PDF,
  SI DOCX, and SI PDF. If a canonical PDF is byte-identical to the PDF produced
  from its DOCX, one page-image set may satisfy both renditions only when the
  manifest records that shared provenance and matching hash.
- The audit checks mathematics, tables, figures, captions, numbering, citations
  and cross-references, clipping, page breaks, effective fonts, image quality,
  missing glyphs, and authorized placeholders.
- Record source-format identity, render engine/version, document hashes, page
  counts, page-image hashes, reviewer identity, per-page pass/fail state, and
  the final acceptance decision for both DOCX- and PDF-origin renditions.
- Any failure requires rebuild and complete re-audit of the affected canonical
  document; sampling is not acceptance.
- An additional document such as `trackleed_scientific_review.docx` is
  noncanonical unless explicitly added to the manifest, rebuilt, and subjected
  to the same complete audit. Stale generated documents cannot represent the
  accepted revision.

## Authorized incomplete inputs

Only explicit placeholders in these categories may remain:

- `AUTHOR`;
- `EXPERIMENT`;
- `RELEASE-AUTHORIZATION`;
- `SUBMISSION`.

The final validation report must list each remaining placeholder by category
and location. Any uncategorized placeholder or missing scientific content
blocks `content-revision-complete`.

## Later stages, outside this status

The following do not block `content-revision-complete` but are required later
for `submission-ready`:

- author names, affiliations, ORCID, CRediT, funding, acknowledgements, and
  declarations;
- the complete author-approved experimental description;
- JAC formatting adapter and cover letter;
- reviewer-access release candidate and external access test;
- release authorization, repository publication, license, DOI, and submission
  metadata.

The JAC adapter may change presentation only. Any later change to scientific
meaning, claim boundaries, denominators, sensitivity, source data, or display
semantics invalidates the successor fingerprint, reopens
`content-revision-complete`, and reruns the affected gates above.

## Final assertion

The status may be asserted only when every normative gate has documentary
evidence and no unresolved failure. A passing scientific reconstruction without
page-level render acceptance is insufficient; a visually accepted document
without a reproducible successor fingerprint is also insufficient.
