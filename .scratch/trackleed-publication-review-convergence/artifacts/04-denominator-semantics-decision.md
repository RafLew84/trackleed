# Decision record: denominator semantics in the main text

Decision date: 2026-08-02

## Scope

This record freezes reader-facing denominator semantics for Table 1,
controlled-translation Results, the Abstract, Discussion, and Conclusions. It
does not change calculations, gates, evaluation records, or source data.

## Controlled-translation inventory and opportunity sets

Table 1 must use these two distinct descriptions:

- **Image count:** `24 digital image states: four source trials × (one seed
  state + five displaced states)`.
- **Opportunity set:** `80 displaced-spot and 40 displaced-pair evaluation
  opportunities per method after exclusion of four seed states`.

The values 96 spot-state and 48 pair-state records describe the complete
inventory before seed exclusion. They may be used only when explicitly labeled
as inventory counts; they are not denominators for coverage, recovery, or
residual outcomes.

Construction of the fixed result sets:

- displaced spots: four trials × five displaced states × four spots = 80;
- displaced pairs: four trials × five displaced states × two pairs = 40.

A method failure remains in the fixed opportunity set. For example, one failed
non-seed spot opportunity is reported as 79/80. A seed state is never added to
the numerator or denominator of a displaced-state outcome.

## Fixed and conditional outcomes

Fixed-opportunity outcomes must be reported as `x/N`, including coverage and
recovery. Conditional residuals are computed only on observations surviving
the gates for that specific method and outcome. They must therefore be labeled
as conditional and use the corresponding method-specific surviving `n` where
needed to interpret the reported statistic.

The following substitutions are prohibited:

- using 80 or 40 as the denominator of a conditional residual merely because
  these are the controlled-translation opportunity sets;
- replacing a fixed coverage or recovery denominator with the smaller
  surviving `n`;
- using 96 or 48 as result denominators when seed states are excluded from the
  evaluated outcome;
- presenting a bare residual whose conditioning population is ambiguous.

## Abstract rule

The Abstract is self-contained. It must not refer to tables, figures, or
Supplementary Information. A conditional quantitative result in the Abstract
must state its relevant surviving `n` directly when that information is needed
to interpret the value. Otherwise, the statement must be rewritten so it is
unambiguous without the `n`; the denominator semantics cannot be delegated to
Table S2, Table S7, a figure, or any other supplementary display.

## Conclusions wording

Use the approved sentence:

> Conditional residuals must therefore be reported with their method-specific
> surviving n, alongside coverage and recovery evaluated on fixed opportunity
> sets and the corresponding geometry-consistency outcomes.

## Implementation checks

The later manuscript implementation must verify that:

1. Table 1 uses the approved 24-state inventory and 80/40 opportunity wording.
2. Every fixed-opportunity result in the main text is expressed as `x/N`.
3. Every conditional residual is explicitly conditional and has an
   interpretable method- and outcome-specific surviving population.
4. No Abstract sentence refers to a table, figure, or Supplementary
   Information.
5. No computation or source-data record changes as a consequence of this
   editorial decision.
