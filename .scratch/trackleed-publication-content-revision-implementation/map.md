# Mapa implementacji rewizji treści publikacji TrackLEED

Label: `wayfinder:map`
Status: `active`

## Destination

Wersjonowana, odtwarzalna i zaakceptowana rewizja main manuscript oraz SI,
która realizuje skonsolidowany handoff i osiąga
`content-revision-complete`, pozostając odrębna od `submission-ready`.

## Normative sources

- [PRD](PRD.md)
- [Skonsolidowany handoff](../trackleed-publication-review-convergence/artifacts/09-consolidated-revision-handoff.md)
- [Definition of Done](../trackleed-publication-review-convergence/artifacts/08-content-revision-definition-of-done.md)

## Execution order

`I01 → {I02, I03, I04} → {I05, I06, I07} → I08 → I09 → I10 → I11 → I12 → I13 → I14`

I02–I04 mogą być wykonywane niezależnie po I01. I05 i I06 zależą od I02, a
I07 od I02 i I04. Pozostałe zależności są fail-closed i nie mogą zostać
pominięte.

## Tickets

- [I01 — wersjonowany kontrakt domenowy](issues/01-wersjonowany-kontrakt-domenowy.md)
- [I02 — przenośny kontrakt LF](issues/02-przenosny-kontrakt-lf.md)
- [I03 — proza naukowa i mianowniki](issues/03-proza-naukowa-i-mianowniki.md)
- [I04 — kanoniczne podpisy Figures 1/4/5](issues/04-kanoniczne-podpisy-figur.md)
- [I05 — Table S6 i eksporty pełnoprecyzyjne](issues/05-table-s6-i-eksporty.md)
- [I06 — Figure S2 continued i legibility](issues/06-figure-s2-continued.md)
- [I07 — Figure 5 full-width i legibility](issues/07-figure5-full-width.md)
- [I08 — kanoniczny builder i audit matematyki](issues/08-kanoniczny-builder-docx-pdf.md)
- [I09 — integracja przed rebaseline](issues/09-integracja-przed-rebaseline.md)
- [I10 — HITL contentu i claim boundary](issues/10-hitl-content-claim-boundary.md)
- [I11 — replacement workflow fingerprintu](issues/11-replacement-fingerprintu.md)
- [I12 — rebaseline i dwie rekonstrukcje](issues/12-rebaseline-i-rekonstrukcje.md)
- [I13 — pełny audit stron](issues/13-pelny-audit-stron.md)
- [I14 — asercja content-revision-complete](issues/14-asercja-content-revision-complete.md)

## Decisions so far

- Podział I01–I14 i granice właścicieli zatwierdzono 2026-08-02 w bramce HITL
  ticketu 09 workflow konwergencji.
- [I01 — wersjonowany kontrakt domenowy](issues/01-wersjonowany-kontrakt-domenowy.md)
  — cztery dokładne definicje utrwalono w śledzonym
  `docs/publication-status-contract.md`; routing domenowy nadaje mu
  pierwszeństwo przed ignorowanym `context.md`, a fail-closed test chroni
  definicje i routing bez zmiany bieżącego scientific-core fingerprintu.
- [I02 — przenośny kontrakt LF](issues/02-przenosny-kontrakt-lf.md)
  — repozytoryjny hash-bound text ma jednoznaczny checkout i zapis LF,
  uwierzytelnione wyjątki zachowują bajty, a clean Windows reconstruction
  przechodzi bez driftu końców linii i zatrzymuje się dopiero na niezmienionym
  historycznym fingerprintcie; [dowody](artifacts/02-publication-line-ending-contract.md).
- [I03 — proza naukowa i mianowniki](issues/03-proza-naukowa-i-mianowniki.md)
  — zatwierdzony Abstract, hierarchy, sensitivity, claim boundaries i
  method-specific surviving `n` są kanoniczne; Table 1 rozdziela inventory 24
  od fixed opportunity sets 80/40 bez zmiany frozen source data, a generowane
  kopie i manifesty pochodzą z builderów; [dowody](artifacts/03-scientific-prose-denominator-semantics.md).
- [I04 — kanoniczne podpisy Figures 1/4/5](issues/04-kanoniczne-podpisy-figur.md)
  — dokładne krótsze captions, 189-word Figure 5, denominator semantics i
  caveats są chronione fail-closed; inventory i Results używają zatwierdzonego
  review numbering, a index i semantic manifests pochodzą z buildera bez
  zmiany danych lub paneli; [dowody](artifacts/04-canonical-figure-captions.md).
- [I05 — Table S6 i eksporty pełnoprecyzyjne](issues/05-table-s6-i-eksporty.md)
  — compact 75-row/seven-column reader view, 15-code human-readable codebook i
  kontrolowane zaokrąglenie pochodzą z jednego buildera; niezmieniony 375-row
  canonical CSV ma pełnoprecyzyjny 75-row companion oraz deterministyczny XLSX
  `method_aggregate`/`case_records`, a manifest i inventory wiążą cały hash
  chain; [dowody](artifacts/05-table-s6-full-precision-exports.md).
- [I06 — Figure S2 continued i legibility](issues/06-figure-s2-continued.md)
  — jeden display item i caption są rozłożone na `a–d` oraz
  `e–f (continued)` przy placement 6.25 in; builder zachowuje dane, panel order,
  codebook i `R48-P15*`, a fail-closed manifest potwierdza efektywny font
  6.076 pt i komórki co najmniej 8.670 × 6.081 mm;
  [dowody](artifacts/06-figure-s2-continued-legibility.md).
- [I07 — Figure 5 full-width i legibility](issues/07-figure5-full-width.md)
  — kompletna pięciopanelowa Figure 5 ma canvas 7.2 × 9.4 in, full-width
  placement 6.25 in, zatwierdzone fonty 7.0–10.0 pt i minimum efektywne
  6.076 pt; builder oraz parser PDF chronią 400-dpi PNG, wektorowy PDF,
  MediaBox, panel order i niezmieniony caption I04;
  [dowody](artifacts/07-figure5-full-width-legibility.md).
- [I08 — kanoniczny builder i audit matematyki](issues/08-kanoniczny-builder-docx-pdf.md)
  — jeden builder tworzy i manifestuje dokładnie cztery kanoniczne main/SI
  DOCX/PDF; fail-closed audit wiąże AST Math z niepustym natywnym OMath i
  per-expression round trip (main 64/64, SI 10/10, fallback 0), a dwa realne
  eksporty Pandoc/Word są bitowo identyczne; [dowody](artifacts/08-canonical-docx-pdf-builder.md).
- [I09 — integracja przed rebaseline](issues/09-integracja-przed-rebaseline.md)
  — jedna niezmieniająca danych bramka raportuje wszystkie 23 komponenty,
  wymaga dokładnie 13 zatwierdzonych zmian i pokrywa prose, displays,
  claim-evidence map, limitations oraz denominators; stary fingerprint odrzuca
  bieżący core oczekiwanym typem błędu, zachowując identyczne bajty, a pełna
  rekonstrukcja 20 bundles przechodzi; [dowody](artifacts/09-pre-rebaseline-integration.md).
- [I10 — HITL finalnego contentu i claim boundary](issues/10-hitl-content-claim-boundary.md)
  — użytkownik jawnie zaakceptował finalny main/SI content, display semantics,
  limitations, denominator semantics oraz boundary 4 preserved / 5 narrowed /
  0 removed; decyzję związano z implementation commit `38a0e6aa668c4a7751fb6661e94af6d58bea92bc`,
  predecessor fingerprintem i reviewed diffem wszystkich 23 komponentów bez
  wykonywania replacement; [decision record](artifacts/10-hitl-content-claim-boundary-decision.md).
- [I11 — wersjonowany replacement fingerprintu](issues/11-replacement-fingerprintu.md)
  — fail-closed CLI wymaga dokładnego predecessora, decision record I10,
  implementation commitu i kanonicznej autoryzacji, blokuje dirty/unapproved
  inputs i niepełne coverage, zachowuje historię oraz pełny 23-component link,
  a fingerprint podmienia atomowo jako ostatni krok; clean-clone `--check-only`
  wyliczył successor bez rebaseline; [dowody](artifacts/11-versioned-fingerprint-replacement-workflow.md).
- [I12 — autoryzowany rebaseline i dwie rekonstrukcje](issues/12-rebaseline-i-rekonstrukcje.md)
  — replacement zachował predecessora i utworzył successor `66b7c599…`; dwa
  niezależne clean Windows checkouty z identycznym, uwierzytelnionym pakietem
  lokalnych wejść przeszły pełny CPU/headless `--clean --validate`, zwróciły
  bajtowo identyczne raporty i pozostały bez driftu;
  [dowody](artifacts/12-authorized-rebaseline-reconstruction.md).
- [I13 — pełny audit stron](issues/13-pelny-audit-stron.md)
  — ponowny audit techniczny czterech kanonicznych rendycji objął 36 stron main
  i 64 strony SI; niezależne eksporty DOCX-origin są bajtowo identyczne z
  kanonicznymi PDF. Figure 5 przywrócono do kontraktowanych 6.25 in i 6.076 pt;
  100 unikalnych obrazów / 200 decyzji nie wykazuje clippingu, brakujących
  glyphów ani nieautoryzowanych placeholderów, a manifest wiąże jawną akceptację
  HITL z dokumentami, provenance i decyzjami per-page;
  [dowody](artifacts/13-page-level-render-audit.md).

## Constraints

- Generated outputs powstają wyłącznie przez buildery.
- Stary fingerprint jest immutable do I12.
- Abstract nie odwołuje się do tabel, rysunków ani Supplementary Information.
- Dozwolone są wyłącznie cztery kategorie deferred placeholders z PRD.
- Każde naruszenie wcześniejszej bramki blokuje wszystkie późniejsze tickety.
