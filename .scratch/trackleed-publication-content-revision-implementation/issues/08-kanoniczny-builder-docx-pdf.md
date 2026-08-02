# Kanoniczny builder DOCX/PDF i audit matematyki

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 03, 04, 05, 06, 07

## Question

Czy jeden wersjonowany builder tworzy kanoniczne main/SI DOCX/PDF z pełnym,
fail-closed transportem matematyki i jednoznacznym manifestem?

## Objective

Utworzyć jeden wersjonowany builder i manifest kanonicznych main manuscript
oraz SI do DOCX/PDF z readerem
`markdown+tex_math_single_backslash` i fail-closed audit matematyki.

## Acceptance

- Pandoc AST `Math` i natywne OMath plus jawne fallbacki mają zgodną liczebność.
- Brak conversion warnings, silent text fallback i nieallowlistowanych
  damaged-math signatures; początkowy manifest fallbacków jest pusty.
- Manifest jednoznacznie wskazuje kanoniczne main/SI DOCX/PDF; dodatkowy review
  DOCX jest jawnie wykluczony albo podlega temu samemu kontraktowi.
- Builder jest odtwarzalny, a targeted OOXML/export tests przechodzą.

## Answer

Tak. Jeden wersjonowany builder składa main manuscript i SI, używa dokładnie
readera `markdown+tex_math_single_backslash`, generuje kanoniczne DOCX przez
Pandoc oraz PDF przez Microsoft Word i zapisuje jednoznaczny manifest czterech
artefaktów. Audit jest fail-closed: porównuje liczbę AST `Math`, natywne OMath,
ich niepustość oraz uporządkowaną zgodność każdej formuły po odczycie zwrotnym
DOCX; oba dokumenty mają zero fallbacków (main 64/64, SI 10/10).

Podwójny rzeczywisty build potwierdził identyczne SHA-256 wszystkich czterech
DOCX/PDF. Targeted tests, manifest verify, Ruff i pełny page-level render review
przeszły; dodatkowe DOCX w `manuscript/review/` są jawnie wykluczone. Abstract
nie zawiera odwołań do tabel, rycin ani Supplementary Information.

Dowody: [artifact 08](../artifacts/08-canonical-docx-pdf-builder.md).

## Comments

- 2026-08-02: początkowe review wykryło brak per-expression audit oraz realnego
  testu podwójnego eksportu; oba findingi zamknięto przed końcowym buildem.
- 2026-08-02: końcowe review Standards i Spec zakończyły się bez findingów.
