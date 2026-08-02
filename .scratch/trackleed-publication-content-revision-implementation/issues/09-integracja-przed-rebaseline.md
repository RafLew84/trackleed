# Integracja i walidacja przed rebaseline

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 02, 03, 04, 05, 06, 07, 08

## Question

Czy cała rewizja przechodzi integracyjną walidację i wywołuje oczekiwane
odrzucenie przez stary fingerprint przed rebaseline?

## Objective

Przebudować wszystkie zależne inventory i semantic artifacts, następnie
wykonać integracyjną walidację zatwierdzonej rewizji przed zmianą fingerprintu.

## Acceptance

- Targeted builders/tests i pełny adekwatny suite przechodzą.
- Jawny diff obejmuje prose, displays, claim-evidence map, limitations,
  denominators i wszystkie 23 komponenty fingerprintu.
- Stary fingerprint odrzuca dokładnie oczekiwane zmienione komponenty;
  przypadkowy pass jest traktowany jako brak coverage.
- Nie wykonuje się replacement ani ręcznej zmiany fingerprintu.

## Answer

Tak. Publiczna, niezmieniająca danych bramka raportuje wszystkie 23 komponenty
i wymaga dokładnie 13 zatwierdzonych zmian. Obejmuje final prose, displays,
claim-evidence map, limitations i denominators oraz uruchamia kontrakty
manuskryptu, reader-facing inventory i czterech dokumentów kanonicznych.

Stary fingerprint jest wymagany jako poprawny poprzednik, a następnie musi
odrzucić bieżący scientific core typowanym `ScientificCoreMismatchError`;
nieoczekiwany pass albo inny rodzaj błędu zatrzymuje walidację. Jego bajty i
SHA-256 pozostają niezmienione, a CLI nie pozwala użyć go jako ścieżki raportu.

Dowody: [artifact 09](../artifacts/09-pre-rebaseline-integration.md).

## Comments

- 2026-08-02: transakcyjna rekonstrukcja 20 bundles i wszystkie targeted gates
  przeszły; pełny suite ma wyłącznie sześć zastanych `PermissionError` poza
  granicą własności I09.
- 2026-08-02: końcowe review Standards i Spec zakończyły się bez findingów.
