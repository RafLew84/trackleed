# I09 — integracja przed rebaseline

Status: completed
Date: 2026-08-02

## Publiczna bramka

- `scripts/validate_pre_rebaseline.py` jest jednym, niezmieniającym danych
  poleceniem walidacyjnym przed rebaseline.
- Wersjonowany `pre_rebaseline_contract.json` wymaga dokładnie 23 komponentów
  starego fingerprintu oraz dokładnie 13 zatwierdzonych zmian.
- Raport wymienia wszystkie komponenty wraz ze starym i bieżącym SHA-256,
  rolami oraz stanem `changed`/`unchanged`; dodatkowo mapuje zakresy
  `final_prose`, `displays`, `claim_evidence_map`, `limitations` i
  `denominators` na konkretne komponenty.
- Walidator odmawia zapisania raportu pod ścieżką `fingerprint.json`, nie ma
  ścieżki replacement i deklaruje `validation_only: true`.

## Jawny diff 23 komponentów

Raport potwierdza 13 changed i 10 unchanged. Zatwierdzone changed IDs to:

1. `claim-boundary-manifest`
2. `cpu-environment`
3. `display-index`
4. `display-inventory-builder`
5. `display-inventory-contract`
6. `display-semantic-inventory`
7. `fingerprint-validator`
8. `manuscript`
9. `manuscript-validator`
10. `reconstruction-entrypoint`
11. `sensitivity-source-provenance`
12. `supplementary-material`
13. `tracer-authentication`

Brak któregokolwiek z tych ID albo dodatkowy changed ID zatrzymuje walidację.
Integralność poprzednika jest sprawdzana niezależnie przez ponowne wyliczenie
jego logicznego fingerprintu oraz hash kontraktu.

## Integracja i weryfikacja

- Transakcyjna rekonstrukcja reader-facing artifacts uruchomiła deklarowane
  testy wszystkich 20 bundles i zakończyła się powodzeniem: 160 plików
  źródłowych, 49 plików wynikowych.
- Manifest dokumentów kanonicznych: 4 artefakty, passed.
- Kontrakt manuskryptu: 9 claim IDs, passed; obejmuje zakaz odwołań do tabel,
  rycin i Supplementary Information w Abstract.
- Targeted suite po finalnej poprawce: 39 passed.
- Ruff i `compileall`: passed.
- Pełny `trackleed/tests`: 253 passed, 6 failed. Sześć zastanych failures ma
  ten sam wyłączny powód co przed I09: `PermissionError` dla user-owned,
  niedostępnego `manuscript/scientific_panel`; nie ma nowej regresji.

Stary walidator zakończył się dokładnie kodem 2 i komunikatem o różnicy względem
frozen baseline. SHA-256 `fingerprint.json` przed i po walidacji był identyczny:
`3b336ac952ba5c6e87e9bbc53853921ee0b3e1a68c3889c2eeac2fa6f2d01132`.
Logiczny fingerprint poprzednika pozostał
`859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`.

## TDD i code review

Publiczna bramka powstała w kolejnych cyklach red/green dla API, dokładnego
fail-closed diffu, CLI, polityki mutacji i integralności poprzednika. Uwaga
Standards dotycząca porównywania tekstu wyjątku została zamknięta kolejnym
cyklem red/green przez dedykowany `ScientificCoreMismatchError`.

- Standards: finalnie 0 actionable findings.
- Spec: finalnie 0 actionable findings.

## Granica zmian

Nie zastąpiono ani ręcznie nie zmieniono starego fingerprintu. Normalizacja LF
dwóch adapterów była potrzebna do lokalnej rekonstrukcji na Windows; po
odświeżeniu indeksu ich zawartość była identyczna z Git i nie weszła do diffu.
Nie zmieniono user-owned plików `manuscript/scientific_panel`, roboczych tabel
ani zmian w `trackleed/io`.
