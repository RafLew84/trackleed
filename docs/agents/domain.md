# Domain Docs

This is a single-context repository.

## Before exploring, read these

- `context.md` at the repository root
- relevant ADRs under `docs/adr/`, if that directory exists

If an expected domain document does not exist, proceed silently. The domain-modeling workflow creates glossary entries and ADRs lazily, only when terminology or a durable decision is actually resolved.

## Use the glossary's vocabulary

When output names a domain concept, use the term defined in the shared context document. If a needed concept is missing, either reconsider the term or record the gap for domain modeling.

## Publication revision status terms

The versioned canonical definitions of `Content-revision-complete`,
`Page-level render acceptance`, `Controlled scientific-core rebaseline`, and
`Authorized deferred publication input` are in
[`docs/publication-status-contract.md`](../publication-status-contract.md).
For these four terms, that tracked contract takes precedence over a local or
generated copy in `context.md`.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly rather than silently overriding the recorded decision.
