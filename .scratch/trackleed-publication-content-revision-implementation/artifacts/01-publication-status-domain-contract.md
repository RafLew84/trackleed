# Wersjonowany kontrakt domenowy statusu rewizji

Data wdrożenia: 2026-08-02

## Wynik

Jedynym wersjonowanym, kanonicznym źródłem czterech definicji jest
[`docs/publication-status-contract.md`](../../../docs/publication-status-contract.md).
Dokument zachowuje dokładne zatwierdzone brzmienie:

- `Content-revision-complete`;
- `Page-level render acceptance`;
- `Controlled scientific-core rebaseline`;
- `Authorized deferred publication input`.

Routing w [`docs/agents/domain.md`](../../../docs/agents/domain.md) wskazuje ten
kontrakt i nadaje mu pierwszeństwo przed lokalną lub generowaną kopią w
ignorowanym `context.md`. Znaczenie `scientific-baseline-complete` oraz
`submission-ready` nie zostało zmienione.

## Fail-closed guard

[`trackleed/tests/test_publication_status_contract.py`](../../../trackleed/tests/test_publication_status_contract.py)
blokuje:

- brak kanonicznego dokumentu;
- brak deklaracji kanonicznego źródła;
- brak, nadmiar lub zmianę brzmienia którejkolwiek z czterech definicji;
- brak routingu dokumentacji domenowej do kontraktu;
- brak któregokolwiek z czterech terminów w routingu.

## TDD i walidacja

1. RED: test definicji zakończył się oczekiwanym `FileNotFoundError` przed
   utworzeniem dokumentu.
2. GREEN: test definicji przeszedł po utworzeniu kontraktu.
3. RED po review: test routingu zakończył się oczekiwanym failure przed zmianą
   `docs/agents/domain.md`.
4. GREEN: oba testy kontraktu przechodzą; `compileall` testu przechodzi.
5. Pełny `pytest` od root nie dotarł do wykonywania testów: 168 błędów
   kolekcji pochodziło z niedostępnych katalogów `data/`, lokalnego
   `tmp/fullsuite-snapshot-*` i kolizji nazw modułów testowych.
6. Pełny `trackleed/tests` wykonał się: 238 passed, 6 failed. Wszystkie sześć
   failure dotyczy istniejącego, usuniętego lub niedostępnego
   `manuscript/scientific_panel`; żaden nie dotyczy plików I01.
7. Review końcowe: Standards 0 findings, Spec 0 findings.

## Wpływ na scientific core

Nowy kontrakt, routing i test nie należą do 23 komponentów bieżącego
scientific-core fingerprintu. `manuscript/scientific_core/README.md` nie został
zmieniony, więc I01 nie wykonuje przedwczesnego rebaseline.
