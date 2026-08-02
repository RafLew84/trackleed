# Kanoniczny builder DOCX/PDF i audit matematyki

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
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
