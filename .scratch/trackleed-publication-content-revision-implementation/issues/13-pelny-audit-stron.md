# Pełny audit stron main/SI DOCX/PDF

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-human`
Mode: `HITL`
Status: `resolved`
Blocked by: 12

## Question

Czy każda strona czterech kanonicznych renditions przeszła udokumentowane
page-level render acceptance?

## Objective

Zbudować i wyrenderować cztery kanoniczne renditions oraz przeprowadzić
page-level render acceptance 100% ich stron.

## Acceptance

- Audyt obejmuje main DOCX, main PDF, SI DOCX i SI PDF; współdzielony page set
  jest dozwolony wyłącznie przy byte-identical PDF i zapisanym provenance/hash.
- Manifest zapisuje sources, renderer/version, document/page-image hashes,
  page counts, reviewer identity i per-page pass/fail.
- Każda strona przechodzi kontrole matematyki, tabel, figur, podpisów,
  numeracji, cross-references, clippingu, page breaks, fontów, obrazu, glyphów
  i authorized placeholders.
- Failure wymaga rebuild i pełnego re-auditu dotkniętego dokumentu.

## HITL checkpoint

- Finalne artefakty są powtarzalne bajtowo między dwoma niezależnymi checkoutami.
- Main ma 36 stron, a SI 64 strony; 100 unikalnych stron odpowiada 200 zapisom
  per-rendition dla DOCX/PDF.
- DOCX-origin PDF jest byte-identical z kanonicznym PDF dla main i SI.
- Wszystkie strony przeszły techniczny audit 150 DPI; brak pustych stron,
  clippingu, brakujących glyphów i nieautoryzowanych placeholderów. Wszystkie
  fonty są osadzone, a minimalna efektywna rozdzielczość obrazu to 199.7 DPI.
- Po przeglądzie specyfikacji Figure 5 została przywrócona do wymaganego
  full-width placement 6.25 in; jej minimalny efektywny font wynosi 6.076 pt.
- Ponowny pełny audit techniczny i wizualny przeszedł. Użytkownik jawnie
  zaakceptował poprawioną wersję w checkpoint HITL 2026-08-02.

## Answer

Tak. Wszystkie cztery kanoniczne rendycje przeszły udokumentowany i jawnie
zaakceptowany audit każdej strony. Dwa współdzielone zestawy stron są dozwolone,
ponieważ niezależne eksporty DOCX-origin są bajtowo identyczne z odpowiadającymi
im kanonicznymi PDF. Manifest wiąże dokumenty, renderery, 100 unikalnych obrazów
stron i 200 decyzji per-rendition z akceptacją HITL użytkownika.

Dowody: [artifact 13](../artifacts/13-page-level-render-audit.md).
