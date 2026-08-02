# I07 — Figure 5 full-width i legibility

Status: completed
Date: 2026-08-02

## Zrealizowany zakres

- Kanoniczny post-I04 bundle `figure05_cu111_physical_endpoint` pozostaje
  jednym, niepodzielonym full-width main-text display.
- Canvas zmieniono z 7.2 × 9.0 in na zatwierdzone 7.2 × 9.4 in, przy
  placement 6.25 in i PNG 400 dpi (2880 × 3760 px).
- Fonty źródłowe mają dokładnie: base 8.0, axis title 9.0, axis label 8.0,
  tick 7.0, legend 7.0, panel label 10.0, value annotation 7.0 i minor
  annotation 7.0 pt. Minimum efektywne przy placement wynosi 6.076 pt.
- Zachowano wszystkie panele a–e, kolejność, source data, wartości, kodowanie i
  claim semantics. Zatwierdzony 189-słowowy caption I04 pozostał bez zmian.
- Długie tytuły i etykiety paneli d/e złamano, zwiększono odstępy, a adnotacje
  przeniesiono do wolnych pól bez usuwania treści lub danych.

## Format i walidacja

Builder fail-closed sprawdza rozmiar i tryb PNG oraz przez `pypdf` weryfikuje
sygnaturę PDF, jedną stronę, MediaBox 7.2 × 9.4 in i brak rastrowych image
XObjectów także w zagnieżdżonych formach. `pypdf` jest zadeklarowany w
`requirements.txt`. Manifest zapisuje full-width placement, wszystkie fonty,
minimum efektywne oraz wymaganie wektorowego PDF.

Kontrola wizualna PNG w pełnej rozdzielczości potwierdziła brak clippingu i
kolizji po dwóch iteracjach layoutu. Read-only kontrola PDF potwierdziła jedną
stronę, właściwy rozmiar, komplet paneli i wektorowy format.

## Weryfikacja

- Initial RED: renderer zwrócił stare 2880 × 3600 px, a inventory nie
  deklarowało full-width placement.
- Figure 5 bundle: 15 passed, w tym clean committed rebuild.
- Kanoniczny caption contract: 3 passed.
- Publication inventory: 8 passed; index, manifest i semantic manifest
  przebudowano kanonicznym builderem.
- Ruff, compileall i `git diff --check`: passed.
- Pełny `trackleed/tests`: 249 passed, 6 failed; wszystkie sześć zastanych
  failures wynika z `PermissionError` dla niedostępnego
  `manuscript/scientific_panel`. Nie pojawił się nowy failure.
- Frozen predecessor odrzucił zmianę: `Scientific-core fingerprint differs
  from the frozen baseline`; rebaseline pozostaje odroczony do I12.

## Code review

- Standards: początkową obserwację Primitive Obsession dla surowego testu
  bajtów PDF rozwiązano parserem `pypdf`; końcowo 0 findings.
- Spec: 0 actionable findings w obu przebiegach.

## Granica zmian

Nie zmieniono source data, panel inventory, captionu, main manuscript ani
scientific claims. Zmiany użytkownika w `trackleed/io`, innych display bundles,
dokumentach review i `scientific_panel` pozostały poza zakresem.
