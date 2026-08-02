# Wersjonowany kontrakt domenowy rewizji

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
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

## Answer

Tak. Cztery dokładne definicje są w wersjonowanym
[`docs/publication-status-contract.md`](../../../docs/publication-status-contract.md),
a dokumentacja domenowa kieruje do niego z jawnym pierwszeństwem przed
ignorowanym `context.md`. Test kontraktowy wykrywa brak lub rozbieżność
definicji oraz brak routingu. Szczegóły i wyniki walidacji zapisano w
[`../artifacts/01-publication-status-domain-contract.md`](../artifacts/01-publication-status-domain-contract.md).

## Comments

- 2026-08-02: zastosowano TDD; oba kontrakty najpierw wykazały oczekiwany RED,
  a następnie przeszły w GREEN.
- 2026-08-02: końcowe review zgłosiło Standards 0 / Spec 0 findings.
