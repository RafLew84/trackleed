# Publication status domain contract

Canonical source: this document

Decision date: 2026-08-02

This tracked document is the repository source of truth for the four durable
terms introduced by the publication content-revision workflow. The approved
decision record is
[`08-content-revision-definition-of-done.md`](../.scratch/trackleed-publication-review-convergence/artifacts/08-content-revision-definition-of-done.md).

## Definitions

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

## Status boundaries

These terms extend the publication lifecycle without redefining the existing
statuses:

- `scientific-baseline-complete` continues to mean that the scientific core is
  closed, claim-safe, reconstructable, and validated under its authorized
  deferred-input policy, as documented in
  [`manuscript/scientific_core/README.md`](../manuscript/scientific_core/README.md);
- `content-revision-complete` adds the approved content revision, controlled
  rebaseline, and complete reader-facing page audit;
- `submission-ready` remains a later status requiring completed submission
  inputs and a tested reviewer-access release candidate.

An authorized deferred publication input is an explicit record of incomplete
work. It is not evidence that author, experimental, release, or submission
content has been supplied.

## Change control

Any change to a definition above requires an approved decision record and a
matching update to the contract guard. Local or generated glossaries may copy
these definitions for convenience, but must not override this document.
