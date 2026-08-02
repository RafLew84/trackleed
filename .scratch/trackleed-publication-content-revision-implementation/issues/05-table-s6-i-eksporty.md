# Table S6 i eksporty pełnoprecyzyjne

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
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

## Answer

Tak. Table S6 ma 75 wierszy, siedem kolumn, 15 czytelnych kodów i zatwierdzone
zaokrąglenie reader-facing. Kanoniczny CSV zachował wszystkie 375 rekordów i
niezmieniony hash; dodano pełnoprecyzyjny CSV 75 agregatów, codebook oraz XLSX z
dokładnie arkuszami `method_aggregate` (75) i `case_records` (300). Manifest,
inventory, testy deterministyczności i read-only Excel QA potwierdzają typowane
liczby, formaty wyświetlania, centralne oznaczenie/cieniowanie i pełny hash
chain.

Dowody: [artifact 05](../artifacts/05-table-s6-full-precision-exports.md).

## Comments

- 2026-08-02: użytkownik jawnie dopuścił XlsxWriter jako generator zastępczy,
  ponieważ runtime artifact-tool nie był dostępny w sesji.
- 2026-08-02: końcowe review Standards i Spec zakończyły się bez findingów.
