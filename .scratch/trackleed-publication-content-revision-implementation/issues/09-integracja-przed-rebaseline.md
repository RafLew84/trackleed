# Integracja i walidacja przed rebaseline

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
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
