# Proposed cross-contract thesis and claim boundary

Status: proposed until the hash-bound author checkpoint is present and validates

`decision_record.json` is the proposed, result-driven decision record for the
focused cross-contract manuscript. It selects one combined thesis and one
working title, maps every planned abstract, results, and conclusion claim to a
frozen source, and records the language and contribution boundaries that later
figures and prose must obey.

The decision is not final merely because the proposal exists in the repository.
The author must explicitly approve the exact decision-record SHA-256. A valid
confirmation creates a hash-bound `author_checkpoint.json`, an authenticated
manifest, and a fail-closed validation report. Any later change to the title,
thesis, claim wording, limitation, or evidence binding invalidates that
checkpoint.

## Proposed decision

Working title:

> Continuity is not consistency: Cross-contract observability in LEED spot
> measurement chains

Selected thesis: combine continuity-versus-consistency with
measurement-contract-dependent observability. Neither the local Gaussian nor
the native ViPErLEED pathway is treated as position, visibility, or spot-
identity ground truth.

The contribution is the cross-contract, denominator-aware assessment of
failure semantics. It is not invention of LEED spot tracking, radial scaling,
aperture measurement, gap bridging, or a universal method ranking.

## Checkpoint workflow

Preview and validate the proposal from the repository root:

```text
python scripts/freeze_cross_contract_claim_boundary.py --preview
```

After the author explicitly approves the displayed digest, the same command
records the author identity and date with `--confirm`. Figures and prose must
not be finalized before the confirmed package validates.
