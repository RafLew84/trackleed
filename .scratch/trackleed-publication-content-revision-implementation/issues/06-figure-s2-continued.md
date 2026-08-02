# Figure S2 continued i legibility

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 02

## Question

Czy Figure S2 pozostaje jednym display itemem na dwóch stronach i przechodzi
mierzalne bramki czytelności bez zmiany danych lub panel semantics?

## Objective

Wygenerować Figure S2 jako jeden numerowany display item na dwóch kolejnych
stronach: `a–d` oraz `e–f (continued)`.

## Acceptance

- Zachowane są panel order, codebook, centralne oznaczenie, wszystkie dane i
  inventory count jednego display itemu.
- Obie części mają zatwierdzone wymiary źródłowe i placement 6.25 in.
- Najmniejszy efektywny font ma ≥6.0 pt, a komórka heatmapy ≥4.5 mm.
- Testy buildera, manifestu, panel semantics i wymiarów przechodzą.

## Answer

Tak. Figure S2 pozostaje jednym display itemem z jednym captionem, rozłożonym
na kolejne części `a–d` i `e–f (continued)`. Obie części mają zatwierdzone
wymiary, placement 6.25 in, powtórzony codebook i niekolorowe oznaczenie
`R48-P15*`. Manifest potwierdza efektywny font 6.076 pt oraz minimalną komórkę
8.670 × 6.081 mm, bez zmiany danych, kolejności paneli i denominator
semantics.

Dowody: [artifact 06](../artifacts/06-figure-s2-continued-legibility.md).

## Comments

- 2026-08-02: zastosowano alternatywny natywny generator Matplotlib; końcowa
  kontrola objęła oba PNG oraz rozmiar i tekst obu PDF.
- 2026-08-02: finding Spec review o pełnych `x/N` rozwiązano przez renderowanie
  `x/221` i `x/24` bezpośrednio w komórkach paneli a–b.
