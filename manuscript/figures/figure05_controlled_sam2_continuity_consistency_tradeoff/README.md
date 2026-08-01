# Figure 5 — Controlled SAM2-family continuity–consistency trade-off

Status: generated and validated.

## Final artifacts

- [PNG](output/figure05_controlled_sam2_continuity_consistency_tradeoff.png)
- [PDF](output/pdf/figure05_controlled_sam2_continuity_consistency_tradeoff.pdf)
- [render manifest](manifest.json)
- [frozen source contract](contract.json)

## Caption

**Figure 5. Controlled SAM2-family continuity–consistency trade-off under identical
checkpoint weights.** **a,** Four states around a natural B1 I(V) visibility
gap in the controlled scan. SAM2, SAMURAI, and DAM4SAM receive the same
single-object bbox seed; open circles denote native proposals, yellow plus
symbols accepted centers from the common two-dimensional Gaussian fit, and
red crosses rejected Gaussian centers. All twelve views render the complete
frozen 128 x 128 pixel crops with one shared linear intensity scale. The
sequence is the first decreasing-energy gap with different recovery outcomes
and is outcome-selected for mechanistic explanation only. **b,**
Branch-consistent Gaussian measurement coverage for two disjoint frozen pairs,
each with 91 baseline-visible spot opportunities per method. **c,** Total and
zero-delay recovery of the same physics-constrained branch after natural I(V)
gaps, with seven evaluable gaps in pair 1 and six in pair 2. **d,** Left:
accepted Gaussian measurements outside the frozen geometry gate, reported
over each method's accepted measurements in all visibility regimes. Right:
median-to-P95 absolute pair-distance residual \(D(E)\), conditional on both
branches passing proposal, Gaussian-fit, and geometry gates; the
method-specific numbers of jointly consistent pair frames are shown as
\(n\). **e,** Aggregate mechanistic readout. Motion-aware and
distractor-aware memory increase coverage and post-gap branch-consistent
recovery relative to
vanilla propagation, while also increasing geometry-inconsistent accepted
fits. All methods use byte-identical SAM2.1 base-plus checkpoint weights, the
same 48 x 48 pixel prompt and measurement ROI, the same Gaussian estimator,
and no correction or reinitialization. The comparison contains one series and
two pairs and is mechanism evidence, not a general ranking of SAM2-family
methods. Conditional residuals must be interpreted together with coverage,
recovery, and possible false continuity.

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
- **c.** Total and zero-delay post-gap branch-consistent recovery for both
  pairs, with
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
