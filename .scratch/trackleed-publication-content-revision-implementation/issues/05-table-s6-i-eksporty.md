# Table S6 i eksporty pełnoprecyzyjne

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
Blocked by: 02

## Question

Czy Table S6 jest czytelnym 75-row widokiem, który zachowuje komplet 375
rekordów kanonicznych i udostępnia pełnoprecyzyjne eksporty?

## Objective

Test-first wdrożyć compact Table S6: 75 method aggregates, 15 krótkich kodów,
7 kolumn i reader-facing rounding, przy zachowaniu 375 rekordów kanonicznych.

## Acceptance

- Pełnoprecyzyjny CSV zawiera 75 agregatów.
- XLSX zawiera dokładnie arkusze `method_aggregate` i `case_records` z
  odpowiednio 75 i 300 rekordami.
- Istniejące 375 canonical records pozostaje kompletne i niesprzeczne.
- Builder, manifesty, codebook i testy roundingu/liczebności przechodzą.
