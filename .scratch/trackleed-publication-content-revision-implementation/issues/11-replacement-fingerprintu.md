# Wersjonowany replacement workflow fingerprintu

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 10

## Question

Czy istnieje wersjonowany, fail-closed replacement workflow zachowujący
historię i wymagający właściwego poprzednika oraz autoryzacji?

## Objective

Test-first dodać wersjonowany, fail-closed command do kontrolowanego
predecessor-to-successor replacement scientific-core fingerprintu.

## Acceptance

- Wymagany expected predecessor SHA-256 to
  `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`.
- Command wymaga decision record, exact commit i authorization; blokuje dirty
  albo niezatwierdzone scientific inputs i niepełny component coverage.
- Zachowuje historię poprzednika, zapisuje diff wszystkich 23 komponentów i
  predecessor/successor link oraz używa atomic write.
- Testy failure modes przechodzą; ten ticket nie wykonuje rebaseline.

## Answer

Tak. Wersjonowany command wymaga dokładnego predecessora, zaakceptowanego
decision record I10, implementation commitu i kanonicznej autoryzacji. Przed
zapisem blokuje dirty/untracked lub niezatwierdzone scientific inputs,
niepełne coverage oraz każdy drift 23-component diffu. Poprawna ścieżka
zachowuje poprzednika i pełny rekord predecessor→successor, a fingerprint
podmienia atomowo jako ostatni krok.

Tryb `--check-only` przeszedł w czystym klonie i wyliczył successor bez zmiany
fingerprintu. Rzeczywisty rebaseline nie został wykonany i pozostaje zakresem
I12.

Dowody: [artifact 11](../artifacts/11-versioned-fingerprint-replacement-workflow.md).

## Comments

- 2026-08-02: publiczny seam CLI oraz izolowane repozytoria Git zostały jawnie
  zaakceptowane przed pierwszym testem.
- 2026-08-02: produkcyjny check-only zachował identyczne bajty poprzednika;
  replacement nie został uruchomiony.
