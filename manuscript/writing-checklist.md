# Writing checklist

## Inputs required from the authors

- [ ] Final author order, affiliations, corresponding author, and ORCID IDs.
- [ ] Experimental-team text and approval for the deferred apparatus,
      preparation, acquisition-condition, and parameter description.
- [ ] Funding, facility, and computational-resource acknowledgements.
- [ ] Code and data licenses.
- [ ] Public repository/archive plan and DOI.
- [ ] Decision on whether the manuscript is prepared directly in LaTeX, Word, or
      converted from this Markdown source.

## Literature work

- [x] Cite the physical basis and established practice of quantitative LEED spot
      analysis.
- [x] Cite EasyLEED and ViPErLEED without claiming that automated LEED tracking is
      new.
- [x] Cite the original publications and repositories for LocoTrack, OSTrack,
      SAM2, SAMURAI, and DAM4SAM.
- [x] Cite Lucas-Kanade optical flow.
- [ ] Check the current author instructions for the selected journal immediately
      before formatting the submission.
- [x] Record every citation in one reference database; do not leave method URLs as
      substitutes for scholarly citations where a paper exists.

## Methods completeness

- [ ] Insert the author-approved experimental description without inferring
      conditions or parameters from retained metadata.
- [x] State evidence roles, opportunity counts, and frozen pair selections
      needed to interpret the validated outcomes.
- [x] State that seeds are inputs and are not ground truth.
- [x] Define proposal representations and the `48 x 48` common measurement ROI.
- [x] Define the Gaussian model, acceptance gates, and missing-value policy.
- [x] Define `D`, `Delta D/D`, energy-scan midpoint drift, visibility regimes, and
      post-gap recovery.
- [x] State propagation direction, initialization count, and zero-reset policy for
      every experiment.
- [x] State all common denominators before reporting conditional metrics.
- [x] Record software versions, hashes, checkpoints, and execution environments.

## Statistical and presentation decisions

- [x] Decide on clustered uncertainty intervals using series/pair/trial as the
      replication level.
- [x] Do not treat successive images as independent replicates.
- [x] Decide which exact values stay in the main text and which move to tables or
      supplementary material.
- [x] Keep coverage, recovery, geometry inconsistency, and conditional residual
      separate.
- [x] Do not create a composite score or absolute-accuracy ranking.
- [x] Use one method color palette across every figure.
- [x] Mark rejected fits and visibility gaps explicitly.

## Claim-safety review

- [x] Check every abstract and conclusion statement against
      `claim-evidence-map.md`.
- [x] Do not claim the first automated LEED tracker.
- [x] Do not call controlled equivariance absolute tracking accuracy.
- [x] Do not equate a missing Gaussian fit with a tracker failure.
- [x] Do not call tracker confidence a physical I(V) signal.
- [x] Do not infer structural transitions, anisotropy, or a thermal-expansion
      coefficient from the current evidence.
- [x] Keep the SAM2-family result supporting rather than primary.

## Reproducibility package

- [x] Freeze final source-data tables for every figure.
- [x] Add deterministic figure builders and manifests.
- [ ] Verify the headless reconstruction path from saved native predictions.
- [x] Archive environment records and checkpoint hashes.
- [ ] Validate the release package on a clean machine or environment.
- [ ] Add a data/code availability statement with persistent identifiers.

## Submission-stage checks

- [ ] Re-check journal scope and current formatting requirements.
- [ ] Match abstract length, highlights, graphical abstract, and reference style to
      the current journal instructions.
- [ ] Run spelling, equation, unit, and cross-reference checks.
- [x] Verify that every number in the abstract appears in a validated source table.
- [ ] Ensure all figures remain legible at final publication dimensions.
- [ ] Complete CRediT, competing-interest, funding, and data-availability
      statements.
