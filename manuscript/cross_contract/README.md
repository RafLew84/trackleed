# Cross-contract scientific and manuscript contract

Status: frozen

Schema version: 1.0

Effective date: 2026-08-07

The normative source is
[`scientific_contract.json`](scientific_contract.json). Downstream analysis,
figures, claim maps, prose, and validation packages must record its path, schema
version, and SHA-256 digest. This contract is venue-portable and precedes new
principal-scan analysis or cross-contract drafting.

## Central question

Do LEED spot measurability, gap recovery, and physical consistency remain stable
across a local Gaussian measurement contract and a native whole-pattern
ViPErLEED contract?

The comparison stays inside the acquisition-closed, existing-acquisition-only
evidence boundary. Its unit is a predeclared branch-frame opportunity nested
within a scan, branch, and gap event. Successive frames are not independent
experimental replicates.

## Two measurement contracts

The **local Gaussian contract** is the TrackLEED path from a local measurement
ROI to a two-dimensional Gaussian position. Its endpoint exists only when the
fit is finite and all frozen local acceptance gates pass.

The **native ViPErLEED contract** is the indexed, aperture- and
whole-pattern-based path to native significance and raw beam-position
deviations. Its outputs are not passed through the TrackLEED Gaussian estimator.
A final, smoothed, or extrapolated ViPErLEED coordinate alone is not evidence of
a direct native observation.

Neither path supplies absolute position or spot-identity ground truth.

## Operational vocabulary

| Term | Frozen definition | Must not be interpreted as |
| --- | --- | --- |
| **Gaussian-contract measurability** (`G=1`) | For a predeclared branch-frame opportunity, the local Gaussian measurement returns a finite Gaussian-refined spot position and passes every frozen local acceptance gate. | Physical visibility, native tracker success, or absolute position accuracy. |
| **ViPEr-native direct detectability** (`V=1`) | For the indexed beam mapped to the opportunity, native significance meets the frozen threshold and both raw position deviations are finite. | Physical visibility, spot-identity ground truth, or a Gaussian-refined measurement. |
| **Gaussian-contract gap** (`Gap_G`) | A maximal contiguous run of `G=0` for one branch within one scan, bounded on both sides by `G=1` for that branch. | Physical disappearance or a confirmed I(V) visibility gap. |
| **ViPEr-native visibility gap** (`Gap_V`) | A maximal contiguous run of `V=0` for one indexed beam within one scan, bounded on both sides by `V=1` for the same beam. | Physical disappearance or an estimator-independent visibility gap. |
| **Cross-contract target inconsistency** | An accepted Gaussian-path continuation and a direct native observation of its mapped beam differ by more than the frozen positional tolerance after declared coordinate alignment. | A proven ground-truth identity switch. |
| **Unresolved cross-contract outcome** (`unresolved`) | A retained opportunity or event for which the available contract states cannot support branch-model disagreement or target inconsistency, including absence of a required direct native observation. | Failure, neither-observed, or permission to remove the record from a denominator. |

These states belong to complete analysis pathways, not intrinsically to an
image. Manuscript prose must not use unqualified *identity switch*, *false
persistence*, *physical disappearance*, or *physical visibility* for one of
these contract-defined outcomes.

## Evidence hierarchy

1. Frozen branch-frame outcomes from both principal scans, retaining every
   predefined opportunity.
2. Gap, recovery, agreement, physical-consistency, and target-inconsistency
   outcomes derived from those records with explicit nesting and denominators.
3. Limited, predeclared significance- and position-tolerance sensitivity used
   to test claim stability, not to optimize a method ranking.
4. Restricted pilot evidence used to calibrate the native contract, beam map,
   and positional tolerance; the pilot is not independent confirmation.
5. Selected crops and descriptive diagnostics used only to explain already
   frozen outcomes.

No evidence tier licenses an absolute accuracy or spot-identity claim. The
manuscript contribution is cross-contract, denominator-aware failure
assessment. It is not the invention of LEED spot tracking, radial scaling,
aperture measurement, or gap bridging.

## Manuscript and display budget

| Component | Frozen budget |
| --- | --- |
| Abstract | At most 200 words. |
| Main text | Target 4200-4600 words; hard maximum 5000 words. |
| Main-text exclusions | Abstract, captions, references, declarations, and Supplementary Information. |
| Main displays | Exactly three figures and no tables. |
| Supplementary Information | About 2000 words, at most three figures, and at most two compact single-column tables. |

Per-frame and per-opportunity tables, complete configurations, hashes,
machine-readable results, and validation reports remain manifested repository
artifacts. They do not move into the reader-facing Supplementary Information.

## Deferred inputs

The canonical authorized categories remain `AUTHOR`, `EXPERIMENT`,
`RELEASE-AUTHORIZATION`, and `SUBMISSION`. An authorized placeholder records
unfinished work; it is not completed content.

Two inputs are explicitly outside this ticket set and must appear in the draft
with these unambiguous markers:

- `TODO[EXPERIMENT] Supply and approve the complete experimental description.`
- `TODO[SUBMISSION] Complete and verify the bibliography from primary sources before submission.`

Missing information must not be invented. Any additional author, release, or
venue-specific omission must use its canonical category and state the missing
input explicitly.

## End state and change control

The end state of this effort is a scientifically complete cross-contract draft
with authorized deferred publication inputs. It is not a submission-ready
package. Completion of the ticket set also does not by itself assert
`content-revision-complete` or `scientific-baseline-complete`; those statuses
retain their canonical repository gates, including the controlled
scientific-core rebaseline triggered by new native inference.

Any semantic change to this frozen contract requires an approved follow-up
ticket, a schema or contract revision, and revalidation of all downstream
artifacts bound to the predecessor digest.
