# Pełny audit stron main/SI DOCX/PDF

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-human`
Mode: `HITL`
Status: `open`
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
