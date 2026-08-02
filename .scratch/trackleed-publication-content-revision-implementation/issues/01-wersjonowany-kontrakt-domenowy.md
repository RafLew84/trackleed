# Wersjonowany kontrakt domenowy rewizji

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
Blocked by:

## Question

Czy cztery definicje statusu rewizji są dostępne w jednym śledzonym,
kanonicznym kontrakcie domenowym i chronione przed rozbieżnością?

## Objective

Umieścić w śledzonym kanonicznym dokumencie dokładne definicje
`Content-revision-complete`, `Page-level render acceptance`, `Controlled
scientific-core rebaseline` i `Authorized deferred publication input` z
artefaktu 08. Nie force-addować ignorowanego `context.md` bez jawnej decyzji.

## Acceptance

- Jedna wersjonowana lokalizacja jest wskazana przez dokumentację domenową.
- Test lub fail-closed guard wykrywa brak albo rozbieżność czterech definicji.
- Nie zmienia się znaczenie `scientific-baseline-complete` ani
  `submission-ready`.
- Targeted checks oraz review Standards/Spec przechodzą.
