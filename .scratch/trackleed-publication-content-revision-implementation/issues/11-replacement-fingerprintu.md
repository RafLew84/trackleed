# Wersjonowany replacement workflow fingerprintu

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
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
