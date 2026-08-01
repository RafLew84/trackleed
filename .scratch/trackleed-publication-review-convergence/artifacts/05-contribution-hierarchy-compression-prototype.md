# Prototyp hierarchii wkładu i kompresji tekstu

Data prototypu: 2026-08-02

## Pytanie i granice

Prototyp sprawdza minimalny reader-facing rewrite, który przywraca kolejność:

1. common measurement contract;
2. denominator-aware assessment;
3. coverage, recovery i geometry consistency;
4. Gaussian capture-region effect jako mechanizm wspierający.

Nie zmienia wyników, denominatorów, claim boundaries ani zatwierdzonej
15-wariantowej sensitivity. Tekst jest kandydatem do późniejszej implementacji,
nie zmianą właściwego manuskryptu.

## Abstract

### Wariant A — contribution-first (zatwierdzony)

> Quantitative low-energy electron diffraction (LEED) analysis requires
> persistent association between a measurement region and its target
> diffraction spot as patterns move and visibility changes. We compared five
> heterogeneous tracking strategies through a common position-measurement
> contract: each native output supplied a proposal for a 48 × 48-pixel ROI,
> and the same two-dimensional Gaussian estimator produced the reported spot
> position. Robustness was assessed with fixed opportunity sets for coverage
> and post-gap recovery, accepted-fit denominators for geometry inconsistency,
> and method-specific surviving \(n\) for conditional residuals.
>
> Validation combined known translations, a 701-image Cu(111) positive
> control, two independent Ag(111) energy scans with natural \(I(V)\) gaps,
> and a controlled SAM2-family memory experiment. Translation errors were
> 0.0043–0.0265 px with 74/80–80/80 accepted measurements. All methods
> produced 2,804/2,804 Cu(111) measurements and 701/701 endpoints per primary
> pair. In the Ag(111) scans, coverage was 188/221–217/221, recovery
> 17/24–23/24, and conditional median pair-distance residuals 1.737–1.837 px
> (\(n=68–91\)). SAMURAI and DAM4SAM increased coverage over SAM2 but also
> increased geometry-inconsistent continuation; this mechanism result is not
> a general ranking.
>
> Across 15 predeclared perturbations of ROI geometry and acceptance gates,
> absolute outcomes changed, but Lucas–Kanade retained the lowest coverage and
> the denominator-aware interpretation remained stable. The results assess
> measurement-chain robustness, not absolute tracking accuracy. Proposal
> differences were strongly suppressed when the target spot remained inside
> the common estimator's capture region, making this a supporting mechanism
> rather than the primary contribution.

Ten wariant otwiera kontraktem i denominator policy, następnie podaje wyniki, a
capture-region effect pozostawia na końcu. Jest rekomendowany, ponieważ kolejność
zdań odpowiada zamrożonej hierarchii wkładu.

### Wariant B — minimal-change

> Quantitative low-energy electron diffraction (LEED) analysis requires a
> measurement region to remain persistently associated with its diffraction
> spot as patterns move and visibility changes. We evaluated five
> heterogeneous tracking strategies under one position-measurement contract:
> every native output placed the same 48 × 48-pixel ROI, and one
> two-dimensional Gaussian estimator produced the reported position. The
> denominator-aware assessment kept fixed opportunity sets for coverage and
> recovery separate from accepted-fit geometry outcomes and conditional
> residuals with method-specific surviving \(n\).
>
> Evidence comprised known translations, a 701-image Cu(111) positive control,
> two independent Ag(111) energy scans with natural \(I(V)\) gaps, and a
> controlled SAM2-family experiment. Translation errors were 0.0043–0.0265 px
> with 74/80–80/80 accepted measurements. All methods yielded 2,804/2,804
> Cu(111) measurements and 701/701 endpoints per primary pair. Ag(111)
> coverage ranged from 188/221 to 217/221, recovery from 17/24 to 23/24, and
> conditional median pair-distance residuals from 1.737 to 1.837 px
> (\(n=68–91\)). In the controlled memory experiment, improved continuity
> coincided with more geometry-inconsistent continuation and did not establish
> a general ranking.
>
> Across 15 predeclared perturbations of ROI geometry and acceptance gates,
> absolute outcomes changed, while Lucas–Kanade retained the lowest coverage
> and the denominator-aware interpretation remained stable. These experiments
> assess measurement-chain robustness rather than absolute tracking accuracy.
> The common estimator suppressed proposal differences when the intended spot
> remained inside its capture region, supporting—but not defining—the main
> contribution.

Wariant B zachowuje więcej składni obecnego Abstract i jest przez to mniejszym
diffem, ale słabiej oddziela mechanizm fitu od głównej tezy niż wariant A.
Żaden wariant nie odwołuje się do tabel, figur ani Supplementary Information.

Liczebność według jednego jawnego licznika (`Unicode letters/numbers`, złożenia
z łącznikiem jako jedno słowo): obecny Abstract 270 słów, wariant A 244, wariant
B 236. Oba warianty mieszczą się w docelowym zakresie 230–250 słów.

## Objective and contributions — tekst rekomendowany

Zastąpić pierwszy akapit sekcji 1.3 następującym tekstem:

> This study evaluates whether heterogeneous spot-tracking strategies support
> an available and physically self-consistent quantitative LEED position under
> one common proposal-to-ROI-to-Gaussian measurement contract. The primary
> comparison keeps fixed opportunity sets for measurement coverage and
> post-gap branch-consistent recovery separate from accepted-fit geometry
> outcomes and conditional residuals with method-specific surviving \(n\). We
> then test whether native proposal differences are suppressed when the target
> spot remains inside the common estimator's capture region and whether the
> resulting measurements support the pairwise endpoint
> \(\Delta R/R=\Delta D/D\). TrackLEED is the reference implementation and
> reproducibility vehicle; its graphical application and feature set are not
> the subject of the study.

Zastąpić listę contributions listą w tej kolejności:

1. A common proposal-to-ROI-to-Gaussian position-measurement contract for
   heterogeneous point, bounding-box, mask, optical-flow, and LEED-domain
   trackers.
2. A denominator-aware assessment that retains fixed opportunity sets for
   coverage and recovery and outcome-specific surviving sets for conditional
   geometry and residual statistics.
3. Controlled and case-based evidence for measurement-chain equivariance,
   endpoint agreement, motion and natural \(I(V)\)-gap robustness, and the
   continuity–consistency trade-off.
4. A predeclared 15-variant sensitivity analysis that tests whether the claim
   boundary depends on measurement-window geometry or acceptance gates without
   selecting an optimal contract.
5. A mechanistic account of when the common Gaussian estimator suppresses
   proposal differences inside its tested capture region.

## Nagłówek 1.1

Zastąpić:

> Quantitative LEED requires persistent spot identity

przez:

> Quantitative LEED requires persistent spot association

`Association` jest zgodne z treścią sekcji i nie implikuje niedostępnego
image-by-image identity ground truth. Dłuższy wariant z „measurement region and
target spot” powtarza pierwsze zdanie Abstract oraz pierwszy akapit sekcji.

## Discussion — minimalna mapa kompresji

Nie usuwać zatwierdzonego akapitu sensitivity z ticketu 03 ani nowego akapitu
Limitations. Skrócić tylko powtórzenia wyników już podanych w Results i
display items:

- **5.1, akapit 1:** zastąpić dwa zdania z pełnymi wartościami translation i
  Cu(111) jednym zdaniem: `Across the controlled translations and Cu(111)
  positive control, the common estimator reduced much larger native proposal
  differences to closely agreeing refined positions when the intended spot
  remained inside its capture region.` Zachować akapit interpretacyjny o
  equivariance, braku absolute reference i kompletnym measurement chain.
- **5.2, akapit 2:** usunąć ponowne wyliczenie wszystkich zakresów metod i
  zachować: `Lucas–Kanade combined the lowest fixed-denominator coverage and
  recovery with a conditional residual inside the narrow five-method range;
  the residual alone therefore concealed unavailable or inconsistent
  measurements.` Zachować następny akapit definiujący cztery denominator
  classes oraz dodać bez zmian zatwierdzony końcowy akapit sensitivity.
- **5.3, akapit 1:** zastąpić ponowne wyliczenie coverage, recovery i
  inconsistency dla SAM2 family zdaniem: `Under matched weights and prompts,
  the memory-aware variants increased continuity while also increasing
  continuation outside the frozen physical branch.` Zachować ograniczenie do
  jednego scan i dwóch par oraz zakaz general ranking.
- **5.4, akapit 1:** usunąć powtórzenie przebiegu plateau/minimum/cooled state i
  zachować: `Complete Cu(111) measurement coverage yielded mutually consistent
  pairwise endpoints across all five proposal strategies, demonstrating an
  easy positive control rather than a tracker benchmark.`
- **5.5:** zastosować w całości zatwierdzony w tickecie 03 akapit sensitivity i
  width-trigger; nie kompresować go przez powrót do nieprawdziwego zdania, że
  innych ROI nie testowano.

Ta mapa usuwa duplikaty liczbowe, ale zachowuje wszystkie odrębne caveats:
equivariance ≠ accuracy, geometry inconsistency ≠ proven identity switch,
case strata ≠ population replicates i mechanism result ≠ tracker ranking.

## Skrócone podpisy figur

### Figure 1

> **Figure 1. Tracking proposal, common position measurement, and evidence
> ladder.** **a,** Native outputs of five strategies for the same Cu(111) spot:
> LocoTrack point, OSTrack box, DAM4SAM mask, Lucas–Kanade optical-flow point,
> and EasyLEED Kalman position. The view is the final image of the 701-image
> fixed-energy thermal cycle at 56 eV and 92 °C. The metadata-selected case used
> one sequence, spot, and seed without correction or reinitialization and was
> selected without inspecting tracker behavior; the
> empirical OSTrack box and unsmoothed DAM4SAM mask are retained. **b,** Each
> native output is reduced to a proposal coordinate
> centering the common 48 × 48-pixel ROI; one elliptical Gaussian fit to the raw
> pixels supplies the reported position. Proposal centers differ by at most
> 5.60 px and refined centers by 0.0023 px in this explanatory case, which was
> not selected from tracker outcomes and does not rank methods. **c,** Two
> refined positions define pair distance \(D_t\) and \(\Delta D/D\); common
> translation cancels, whereas non-rigid distortion does not. **d,** Controlled
> translations, Cu(111), independent Ag(111) scans, and the controlled SAM2
> family test equivariance, endpoint agreement, motion plus natural \(I(V)\)
> gaps, and a memory mechanism, respectively. They retain separate
> denominators and do not form a composite ranking.

### Figure 4

> **Figure 4. Controlled SAM2-family continuity–consistency trade-off under
> identical checkpoint weights.** **a,** Four states around the first
> decreasing-energy B1 \(I(V)\) gap with different recovery outcomes. SAM2,
> SAMURAI, and DAM4SAM receive the same bbox seed; circles are proposals,
> yellow plus signs accepted Gaussian centers, and red crosses rejected
> centers. All crops share the full 128 × 128 field and one linear intensity
> scale. The sequence is outcome-selected for mechanistic explanation and is
> not a representative-case estimate. **b,** Branch-consistent
> coverage for two disjoint pairs, each with 91 baseline-visible spot
> opportunities per method. **c,** Total and zero-delay recovery over seven and
> six evaluable gaps. **d,** Left, Gaussian fits outside the frozen geometry
> gate over each method's accepted-fit denominator; right, median-to-P95
> pair-distance residuals conditional on both branches passing proposal, fit,
> and geometry gates, with method-specific \(n\). **e,** Motion- and
> distractor-aware
> memory increase coverage and recovery relative to vanilla propagation but
> also increase continuation outside the frozen branch. All methods use
> byte-identical SAM2.1 base-plus weights, the same 48 × 48-pixel prompt and
> measurement ROI, one Gaussian estimator, and no correction or
> reinitialization. One series and two pairs support a mechanism result, not a
> general ranking. Pair-specific denominators remain separate rather than
> pooled across the two cases. Conditional residuals summarize only surviving pairs and
> must follow fixed-denominator coverage, recovery, and geometry context.

### Figure 5

> **Figure 5. Common Gaussian refinement preserves the Cu(111) pairwise
> relative spot-radius endpoint across proposal strategies.** **a,** Temperature
> profile of the 701-image constant-energy cycle with reference, 800 °C
> plateau, post-plateau minimum, and final cooled state; frame identifiers remain
> in source metadata. **b,c,** Pairwise
> \(\Delta R/R=\Delta D/D\) for pairs 1–6 and 3–5 after five proposal strategies
> enter the same estimator. Upper axes show the cross-method median and range;
> lower axes magnify deviations from the median in \(10^{-3}\) percentage points
> on a shared scale. Every method yields 701/701 endpoints per pair. Common
> image translation cancels in pair distance, whereas non-rigid distortion does
> not. **d,** Between-method endpoint range at the three predefined landmarks;
> maxima over the sequence are 0.001710 and 0.002237 percentage points.
> **e,** Separate saved-YOLO diagnostic for pair 2–4: proposals remain present
> throughout the displayed context while 24 consecutive Gaussian fits fail
> only the amplitude gate. This distinguishes
> proposal from measurement availability; it does not show identity loss or
> compare the five methods on that pair. The two primary pairs do not establish
> crystallographic anisotropy, thermal expansion, phase transition, or a
> structural mechanism.

## Rekomendacja HITL

Podpisy policzone tym samym licznikiem zmieniają się następująco:

| Display item | Obecnie | Prototyp | Redukcja |
|---|---:|---:|---:|
| Figure 1 | 249 | 202 | 18.9% |
| Figure 4 | 271 | 216 | 20.3% |
| Figure 5 | 232 | 189 | 18.5% |

Użytkownik zatwierdził wariant A Abstract, rekomendowany Objective i contribution list,
nagłówek `persistent spot association`, mapę selektywnej kompresji Discussion
oraz trzy skrócone podpisy. Właściwy manuskrypt pozostaje bez zmian do
późniejszego handoffu implementacyjnego.
