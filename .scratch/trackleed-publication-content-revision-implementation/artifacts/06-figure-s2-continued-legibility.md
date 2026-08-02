# I06 — Figure S2 continued i legibility

Status: completed
Date: 2026-08-02

## Zrealizowany zakres

- Natywny builder Matplotlib generuje Figure S2 jako jeden numerowany display
  item w dwóch częściach: `a–d` (7.2 × 10.0 in) oraz `e–f (continued)`
  (7.2 × 6.0 in), każdą w PNG 300 dpi i wektorowym PDF.
- Obie części mają placement 6.25 in. Druga zaczyna się od
  `Figure S2 (continued)` i powtarza ten sam human-readable codebook.
- Zachowano kolejność sześciu paneli, 15 wariantów, pięć metod, wszystkie dane,
  fixed/conditional denominator semantics oraz jeden wpis inventory.
- Centralny kontrakt `R48-P15*` ma jednocześnie asterisk i pełny pomarańczowy
  obrys, więc oznaczenie nie zależy wyłącznie od koloru.
- Caption pozostaje jeden i opisuje panele a–f; README dokumentuje zachowanie
  numeracji i placement obu części.
- Komórki a–b pokazują pełne fixed-denominator ułamki `x/221` i `x/24`, a
  komórki c–d zachowują outcome-specific denominator lub conditional `n`.

## Mierzalna czytelność

Manifest buildera zapisuje i fail-closed sprawdza:

- najmniejszy font źródłowy: 7.0 pt;
- najmniejszy font po umieszczeniu w szerokości 6.25 in: 6.076 pt;
- najmniejsza szerokość komórki heatmapy: 8.670 mm;
- najmniejsza wysokość komórki heatmapy: 6.081 mm.

Wszystkie wartości przekraczają zatwierdzone bramki 6.0 pt i 4.5 mm. Kontrola
wizualna obu PNG w pełnej rozdzielczości usunęła clipping codebooka i zlewanie
ułamków w panelu c. Read-only kontrola PDF potwierdziła po jednej stronie,
wymiary 7.2 × 10.0 in i 7.2 × 6.0 in oraz obecność tytułu i codebooka.

## Weryfikacja

- Initial RED: kontrakt Figure S2 odrzucił brak dwóch części, jednej pozycji
  display oraz metryk fontu i komórek.
- Figure S2 contracts: 4 passed, w tym clean committed rebuild i niezależność
  od ambient Matplotlib style.
- Publication inventory: 8 passed; index, manifest i semantic manifest
  przebudowano kanonicznym builderem.
- Ruff, compileall i `git diff --check`: passed.
- Pełny `trackleed/tests`: 249 passed, 6 failed; sześć zastanych failures nadal
  wynika wyłącznie z `PermissionError` dla niedostępnego
  `manuscript/scientific_panel`. Nie pojawił się nowy failure.
- Frozen predecessor: `freeze_scientific_core --validate-only` odrzucił zmianę
  fingerprintu zgodnie z wymaganiem; rebaseline pozostaje odroczony do I12.

## Granica zmian

Nie zmieniono source data, wartości liczbowych, panel semantics, liczby display
items ani claim boundary. Zmiany użytkownika w `trackleed/io`, innych display
bundles, dokumentach review i `scientific_panel` pozostały poza zakresem.

## Code review

- Standards: 0 hard violations, 0 judgment-call smells.
- Spec: obserwację o brakujących pełnych `x/N` w komórkach a–b rozwiązano;
  końcowy staged diff nie ma pozostałych findingów.
