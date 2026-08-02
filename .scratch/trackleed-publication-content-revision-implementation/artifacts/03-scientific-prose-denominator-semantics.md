# I03 — proza naukowa i semantyka mianowników

Status: completed
Date: 2026-08-02

## Zrealizowany zakres

- Kanoniczny `manuscript/manuscript.md` zawiera zatwierdzony 244-word
  Abstract A bez odwołań do tabel, figur ani Supplementary Information.
- Wdrożono framing `persistent spot association`, zatwierdzony Objective i
  hierarchię pięciu contributions.
- Results zawierają nową sekcję 4.4 z 15 predeclared sensitivity variants;
  dawne sekcje 4.4 i 4.5 przesunięto do 4.5 i 4.6.
- Conditional residuals w centralnym wyniku mają jawne method-specific
  surviving `n`: 84, 68, 90, 91 i 91.
- Discussion, Limitations i Conclusions utrzymują granice C1/C3/C4 oraz
  rozdzielają fixed opportunity sets od outcome-specific denominators.
- Table 1 rozdziela inventory 24 digital image states od wynikowych fixed
  opportunity sets 80 displaced spots i 40 displaced pairs. Zamrożone dane
  źródłowe 96/48 nie zostały zmienione; reader-facing output i kopia w
  manuskrypcie powstały przez builder.
- Manifest Table 1 oraz publication inventory/semantic inventory zostały
  przebudowane przez ich buildery. Scientific-core fingerprint pozostał
  niezmieniony zgodnie z bramką I12.

## TDD i walidacja

RED przed implementacją:

- `pytest manuscript/test_manuscript_contract.py manuscript/tables/table01_data_evidence_roles/test_build_table.py -q`
- wynik: 2 failures, 10 passed; niezgodne były stary Abstract i stare
  reader-facing 96/48.

GREEN po implementacji:

- testy kontraktu manuskryptu i Table 1: 13 passed;
- `manuscript/manuscript_contract.py`: pass;
- targeted scientific-core inventory validation: 1 passed;
- pełny `trackleed/tests` po odświeżeniu inventory: 246 passed, 6 failed;
  wszystkie pozostałe zastane failure dotyczą niedostępnego
  `manuscript/scientific_panel`;
- `compileall` dla zmienionych modułów: pass;
- walidacja starego scientific-core fingerprintu: oczekiwany fail-closed
  `Scientific-core fingerprint differs from the frozen baseline`.

Pełne `pytest manuscript` nie dochodzi do wykonania testów: kolekcja ma
zastane kolizje modułów o nazwach `test_build_figure`/`test_build_table`, a
`manuscript/scientific_panel` zwraca `PermissionError`. Próba pełnego buildera
publication inventory również potwierdziła zastany mismatch hasha
`dam4sam_sam21pp_l adapter_source`; dlatego po niezależnym wykonaniu testów
zakresowych użyto udokumentowanego trybu buildera `--skip-contract-tests`.

## Code review

- Standards: 0 hard violations; po centralizacji kontraktu 24/80/40 także 0
  pozostałych actionable findings.
- Spec: 0 actionable findings; ponowny przegląd po refaktorze również 0.
- Niezależna kopia zatwierdzonego Abstractu w teście pozostaje celowa: test
  nie pobiera swojego oracle z walidatora, który sprawdza.

## Granica zmian

Nie zmieniono danych naukowych, obliczeń, frozen source data ani starego
fingerprintu. Nie włączono do zakresu istniejących zmian użytkownika w
`trackleed/io`, `trackleed/tests`, Supplementary Information ani w innych
bundle tabel i figur.
