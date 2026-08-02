# Figure 4 - controlled SAM2-family memory trade-off

Status: generated and validated.

## Final artifacts

- [PNG](output/figure04_controlled_sam2_memory_tradeoff.png)
- [PDF](output/pdf/figure04_controlled_sam2_memory_tradeoff.pdf)
- [render manifest](manifest.json)
- [frozen source contract](contract.json)

## Caption

**Figure 4. Controlled SAM2-family continuity–consistency trade-off under
identical checkpoint weights.** **a,** Four states around the first
decreasing-energy B1 \(I(V)\) gap with different recovery outcomes. SAM2,
SAMURAI, and DAM4SAM receive the same bbox seed; circles are proposals,
yellow plus signs accepted Gaussian centers, and red crosses rejected
centers. All crops share the full 128 × 128 field and one linear intensity
scale. The sequence is outcome-selected for mechanistic explanation and is
not a representative-case estimate. **b,** Branch-consistent coverage for two
disjoint pairs, each with 91 baseline-visible spot opportunities per method.
**c,** Total and zero-delay recovery over seven and six evaluable gaps.
**d,** Left, Gaussian fits outside the frozen geometry gate over each method's
accepted-fit denominator; right, median-to-P95 pair-distance residuals
conditional on both branches passing proposal, fit, and geometry gates, with
method-specific \(n\). **e,** Motion- and distractor-aware memory increase
coverage and recovery relative to vanilla propagation but also increase
continuation outside the frozen branch. All methods use byte-identical SAM2.1
base-plus weights, the same 48 × 48-pixel prompt and measurement ROI, one
Gaussian estimator, and no correction or reinitialization. One series and two
pairs support a mechanism result, not a general ranking. Pair-specific
denominators remain separate rather than pooled across the two cases.
Conditional residuals summarize only surviving pairs and must follow
fixed-denominator coverage, recovery, and geometry context.

## Scientific question

What changes when the memory strategy changes under identical SAM2.1
base-plus checkpoint weights?

The controlled family contains:

- SAM2 with vanilla propagation;
- SAMURAI with motion-aware memory;
- DAM4SAM with distractor-aware memory.

All three methods use byte-identical checkpoint weights, the same
single-object 48 x 48 px bbox prompt, mask centroid as the tracking proposal,
the same 48 x 48 px measurement ROI, and the same two-dimensional Gaussian
estimator. Propagation starts from the same 187.9 eV seed and proceeds in both
energy directions without correction or reinitialization.

## Frozen scope

The quantitative comparison contains one controlled energy-scan series, two
disjoint frozen spot pairs, 182 baseline-visible spot opportunities and 13
natural I(V) gaps per method. It is mechanism evidence, not a general ranking
of SAM2-family trackers. The four Gaussian-refined seed centers at 187.9 eV
are frozen in `source_data/pair_seeds.csv`; the pair-2 selection excluded both
spots used by pair 1.

The explanatory image sequence uses the first decreasing-energy I(V) gap for
which the controlled methods have different recovery outcomes. This
outcome-selected sequence is used only to explain the mechanism. Quantitative
panels use both frozen pairs.

## Final panels

- **a.** Four image states around the representative B1 visibility gap:
  pre-gap visibility at 169.2 eV, the 166.5-163.9 eV gap, and the first
  expected recovery at 160.9 eV. Overlay native proposals and accepted or
  rejected Gaussian centers for SAM2, SAMURAI, and DAM4SAM.
- **b.** Branch-consistent Gaussian measurement coverage for both pairs,
  preserving the common denominator of 91 opportunities per pair.
- **c.** Total and zero-delay post-gap recovery for both pairs, with
  denominators 7 and 6.
- **d.** Split view of geometry-inconsistent accepted fits over each method's
  accepted measurements and conditional pair-distance residual `D(E)`
  (median and P95, with the number of jointly consistent pair frames).
- **e.** Mechanistic summary of the continuity benefit and
  false-persistence risk.

Coverage, total and zero-delay recovery, geometry-inconsistent continuation,
and conditional pair-distance residuals must be presented together.
Conditional residuals are supporting diagnostics and cannot be interpreted as
an overall accuracy ranking.

The four raw crops must use one shared linear intensity scale when rendered;
per-crop contrast normalization would visually exaggerate the weak spot inside
the visibility gap.

## Frozen source package

- [source contract](contract.json)
- [family aggregate](source_data/family_aggregate.csv)
- [results by pair](source_data/family_by_pair.csv)
- [frozen pair seeds](source_data/pair_seeds.csv)
- [representative-gap measurements](source_data/representative_gap_measurements.csv)
- [representative-gap selection](source_data/representative_gap.json)
- [source provenance](source_data/source_provenance.json)
- `source_data/crops/` - four raw 128 x 128 pixel TIFF crops

Rebuild and validate:

```bash
python freeze_source_data.py
python freeze_source_data.py --validate-only
python build_figure.py
python build_figure.py --validate-only
python test_freeze_source_data.py
python test_source_contract.py
python test_build_figure.py
```
