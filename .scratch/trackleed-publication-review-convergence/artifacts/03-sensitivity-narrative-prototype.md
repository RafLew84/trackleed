# Prototyp integracji sensitivity z narracją i granicą claims

Data prototypu: 2026-08-02

## Pytanie i źródła prawdy

Prototyp rozstrzyga, jak 15-wariantowa **measurement-window and
capture-contract sensitivity** ma wejść do reader-facing manuskryptu bez
przekształcenia jej w optymalizację parametrów albo ranking trackerów.

Każde zdanie liczbowe poniżej pochodzi z:

- `manuscript/source_data/measurement_sensitivity/sensitivity_summary.json`;
- `manuscript/source_data/measurement_sensitivity/sensitivity_validation_report.json`;
- `manuscript/claim_adjudication/output/sensitivity_claim_evidence.json`;
- `manuscript/claim_adjudication/output/final_claim_boundary.json`.

Nie wykonano ponownego tracker inference ani ponownego wyboru konfiguracji.
Reference opportunity sets pozostają zamrożone: 221 baseline-visible
opportunities dla coverage i 24 evaluable natural I(V) gaps dla recovery.
Geometry inconsistency zachowuje denominator zaakceptowanych fitów, a residual
— metodowo i wariantowo zależne pair-frame `n`.

## Fakty, które muszą przetrwać redakcję

- Przeanalizowano dokładnie 15 predeclared variants: dziewięć kombinacji
  `ROI side × proposal-to-fit` (`32/48/64 px × 12/15/18 px`, w tym centralny
  `48 × 48 px / 15 px`) oraz po dwa warianty branch tolerance (`12/24 px`),
  `R²` (`0.80/0.90`) i amplitude gate (`0.75×/1.25×`).
- Pełna koperta wszystkich metod i wariantów wynosiła: coverage `170–221/221`,
  recovery `17–24/24`, geometry-inconsistency fraction `1/222–69/272`
  (`0.45–25.37%`; accepted-fit denominators `187–286`) oraz conditional median
  residual `1.499 px (n=79)–1.939 px (n=99)` (pair-frame `n` across all
  method–variant combinations: `54–104`).
- `64 × 64 px` ROI i `R² = 0.90` obniżały branch-consistent coverage każdej
  metody względem kontraktu centralnego. Zmiana branch tolerance zmieniała
  klasyfikację geometry consistency i zbiór przeżywających residuali zgodnie z
  definicją bramki.
- Oba warianty amplitude gate (`0.75×`, `1.25×`) odtworzyły wszystkie
  raportowane centralne outcomes.
- Lucas–Kanade zachował najniższe coverage we wszystkich 15 wariantach. Nie
  oznacza to globalnego rankingu accuracy; wspiera wyłącznie case-based wniosek
  o rozdzieleniu completeness/recovery od conditional residual.
- Width sensitivity nie została uruchomiona, ponieważ żadna metoda nie miała
  width-only rejection, a liczby fitów blisko width boundary nie osiągnęły
  predeclared thresholds. Nie powstał żaden width contract i centralny
  kontrakt nie został zmieniony.

## Wariant A — osobna sekcja Results (rekomendowany)

Po obecnej sekcji 4.3 dodać `4.4 Predeclared measurement-window and
capture-contract sensitivity`, a obecne sekcje 4.4 i 4.5 przesunąć odpowiednio
do 4.5 i 4.6. Ten wariant daje sensitivity widoczny status w evidence ladder i
jednoznacznie pokazuje jej udział w adjudykacji claims.

### Results — dokładny szkielet akapitów

> To test whether the Ag(111) conclusions depended on the central 48 × 48-pixel
> ROI and acceptance contract, we re-evaluated the stored tracker proposals
> under 15 predeclared measurement-window and capture-contract variants
> (Figure S2; Table S6). The set comprised nine ROI-side × proposal-to-fit
> combinations (32, 48, or 64 px × 12, 15, or 18 px), including the central
> 48 px × 15 px contract, and paired one-factor perturbations of branch
> tolerance (12 or 24 px), the Gaussian \(R^2\) threshold (0.80 or 0.90), and
> the amplitude gate (0.75× or 1.25× its central value). Tracker inference and
> the reference opportunity sets were held fixed: coverage retained 221
> baseline-visible opportunities and recovery retained 24 evaluable natural
> \(I(V)\) gaps. Geometry inconsistency remained conditional on each variant's
> accepted fits, and pair-distance residuals retained their method- and
> variant-specific surviving \(n\).

> The perturbations changed absolute outcomes. Across all methods and
> variants, branch-consistent coverage spanned 170–221/221, post-gap
> branch-consistent recovery 17–24/24, geometry-inconsistency fractions
> 1/222–69/272 (0.45–25.37%; accepted-fit denominators across all
> method–variant combinations, 187–286), and conditional median pair-distance
> residuals 1.499 px (n=79)–1.939 px (n=99; pair-frame n across all
> combinations, 54–104).
> The 64-pixel ROI and \(R^2=0.90\) variants reduced coverage for every method
> relative to the central contract, while changing branch tolerance altered
> geometry classification and the surviving residual set by construction.
> Both amplitude-gate perturbations reproduced the reported central outcomes.
> Despite these numerical changes, Lucas–Kanade retained the lowest coverage
> in every variant, and the central conclusion remained: conditional residuals
> cannot summarize robustness without fixed-denominator coverage and recovery,
> geometry-consistency outcomes, and their own surviving \(n\). These are
> sensitivity results for the tested contracts, not a parameter optimization
> or a comparative accuracy ranking.

> The predeclared width-materiality trigger was not activated. No method
> produced a width-only rejection, and the counts of fits near the width gate
> remained below their method-specific trigger thresholds. Width sensitivity
> was therefore not run, no alternate width contract was generated, and the
> central measurement contract was not changed.

## Wariant B — integracja w istniejącej sekcji 4.3

Wstawić te same trzy akapity na końcu obecnej sekcji 4.3, bez nowego nagłówka i
bez renumeracji dalszych sekcji. Treść i claim boundary pozostają identyczne.
Wariant jest krótszy strukturalnie, ale osłabia widoczność obowiązkowej
sensitivity i łatwiej odczytać ją jako rozwinięcie leave-one-case-out zamiast
osobnej perturbacji measurement contract.

## Tekst wspólny dla obu wariantów

### Abstract — jedno zdanie

Wstawić po zdaniu raportującym centralne wyniki Ag(111):

> Across 15 predeclared perturbations of ROI geometry and acceptance gates,
> absolute outcomes changed, but Lucas–Kanade retained the lowest coverage in
> every variant and the denominator-aware separation of completeness,
> recovery, geometry consistency, and conditional residuals remained.

Zdanie nie podaje nowego rankingu; wskazuje stabilność interpretacji i pozostaje
samowystarczalne, bez odwołań do tabel, rysunków lub Supplementary Information.
Pełne liczby i outcome-specific denominators są raportowane w Results, natomiast
Abstract nie deleguje do nich interpretacji przez reader-facing cross-reference.

### Discussion — nowy końcowy akapit sekcji 5.2

> The measurement-contract sensitivity qualifies rather than universalizes
> these conclusions. The common-estimator interpretation and the separation
> between fixed-denominator completeness and conditional residual quality
> survived all 15 predeclared variants, but the numerical envelopes changed
> with ROI geometry and acceptance gates. The central residual range is
> therefore not generalized to every contract: across the sensitivity set,
> medians spanned 1.499–1.939 px and retained method- and variant-specific
> \(n\). Likewise, Lucas–Kanade's consistently lowest coverage is a result for
> the four frozen case strata and tested contracts, not a population-level
> ranking. Sensitivity thus supports claims about denominator-aware
> measurement-chain robustness within the acquisition-closed case boundary;
> it does not select an optimal configuration or establish absolute tracking
> accuracy.

### Limitations — zastąpić obecny końcowy akapit sekcji 5.5

Usunąć nieprawdziwe zdanie `The present work does not test other ROI sizes...`
i zastąpić cały akapit tekstem:

> Finally, the central Gaussian gates, 48 × 48-pixel ROI, seeds, and branch
> tolerance define the estimator capture contract and affect the observed
> balance between coverage and inconsistency. We tested 15 predeclared
> perturbations of ROI size, proposal-to-fit distance, branch tolerance,
> \(R^2\), and amplitude acceptance using stored proposals from the Ag(111)
> stress test. This local sensitivity is not parameter optimization, a full
> capture-basin map, or evidence for untested detector geometries and data
> regimes; its claims remain limited to the observed numerical envelopes and
> outcome-specific denominators. Width sensitivity was not run because the
> predeclared materiality trigger was not activated, not because width limits
> were shown to be irrelevant. The study also does not test a full structural
> LEED \(I(V)\) workflow. The Cu(111) endpoint is consequently limited to
> relative changes of the measured diffraction-pattern scale; it is not used
> to infer a thermal-expansion coefficient, structural transformation, or
> crystallographic anisotropy.

## Claims po integracji

- **C1:** utrzymać zawężenie do tested data regimes i predeclared measurement
  contracts; nie pisać o estimatorze poza jego tested capture region.
- **C3:** zachować case-based `two recorded Ag(111) sequences / four strata`;
  stabilność outcome separation nie tworzy populacyjnego rankingu.
- **C4:** centralna podobność residuali pozostaje wynikiem centralnego kontraktu;
  stabilny jest denominator-aware argument, nie identyczna wartość residualu w
  każdym wariancie.
- **C2, C5–C9:** bez zmian z sensitivity, ponieważ należą do odrębnych
  kontraktów lub dowodów.

## Decyzja HITL

Użytkownik zatwierdził **wariant A**: osobna sekcja Results, wspólne akapity w
Abstract/Discussion/Limitations oraz powyższa jawna mapa C1/C3/C4. Nie zmieniać
teraz manuskryptu ani scientific-core fingerprint; zatwierdzony szkielet ma
zasilić późniejszy handoff implementacyjny po zamknięciu decyzji 03–07.
