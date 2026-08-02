# I04 — kanoniczne podpisy Figures 1, 4 i 5

Status: completed
Date: 2026-08-02

## Zrealizowany zakres

- Zatwierdzone skrócone podpisy z artefaktu 05 zastąpiły caption sections w
  kanonicznych źródłach Figures 1, 4 i 5.
- Figure 5 ma dokładnie 189 słów według wersjonowanego licznika Unicode z
  hyphen compounds liczonymi jako jedno słowo.
- Caption contracts zachowują denominator semantics i caveats: oddzielne
  mianowniki, 91 opportunities na parę w Figure 4, siedem/sześć gaps,
  accepted-fit denominator, method-specific `n`, 701/701 endpoints, 24
  amplitude-only failures oraz zakaz composite/general ranking i
  nieuprawnionych identity/physics claims.
- Kanoniczny display inventory został wyrównany z zatwierdzonym review figure
  set: Figure 3 to Ag(111) stress test, Figure 4 to controlled SAM2 family, a
  Figure 5 to Cu(111) endpoint. Trzy odwołania w Results zostały odpowiednio
  zsynchronizowane.
- `figures-and-tables.md`, publication inventory manifest i semantic manifest
  powstały ponownie przez `build_inventory.py`. Żadnego generated output nie
  edytowano ręcznie.

## TDD i walidacja

RED:

- `pytest manuscript/test_figure_caption_contract.py -q`
- wynik: 3 failures; wykryto stare captions, starą numerację inventory i stare
  cross-reference.

GREEN:

- caption/inventory/manuscript contracts: 16 passed;
- scientific-core inventory tests: 5 passed;
- lokalne bundle tests: Figure 1 — 5 passed; Figure 3 — 13 passed; Figure 4 —
  9 passed; Figure 5 — 14 passed;
- `compileall` dla zmienionych modułów: pass;
- pełny `trackleed/tests`: 246 passed, 6 failed; wszystkie pozostałe zastane
  failures wynikają z `PermissionError` dla `manuscript/scientific_panel`;
- niezmieniony predecessor scientific-core fingerprint odrzuca zmianę zgodnie
  z bramką I12: `Scientific-core fingerprint differs from the frozen baseline`.

Pełny tryb contract-test publication inventory dochodzi do zastanego mismatch
hasha `dam4sam_sam21pp_l adapter_source` w Figure 1, spowodowanego zmianami
użytkownika poza I04. Po niezależnym wykonaniu lokalnych bundle tests użyto
udokumentowanego trybu buildera `--skip-contract-tests` do wygenerowania
inventory artifacts.

## Granica zmian

Nie zmieniono source data, frozen records, obliczeń, paneli, PNG/PDF figur,
claim boundary ani predecessor fingerprintu. Istniejące zmiany użytkownika w
`trackleed/io`, Supplementary Information i `scientific_panel` pozostały poza
zakresem.

## Code review

- Standards: 0 hard violations, 0 judgment-call smells.
- Spec: 0 actionable findings.
- Niezależny review potwierdził, że realignment numeracji jest konieczny, a
  nie scope creep: bez niego dokładne zatwierdzone captions Figure 4 i
  189-word Figure 5 zostałyby przypisane do niewłaściwych paneli.
