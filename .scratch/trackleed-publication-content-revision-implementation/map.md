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

## Constraints

- Generated outputs powstają wyłącznie przez buildery.
- Stary fingerprint jest immutable do I12.
- Abstract nie odwołuje się do tabel, rysunków ani Supplementary Information.
- Dozwolone są wyłącznie cztery kategorie deferred placeholders z PRD.
- Każde naruszenie wcześniejszej bramki blokuje wszystkie późniejsze tickety.
