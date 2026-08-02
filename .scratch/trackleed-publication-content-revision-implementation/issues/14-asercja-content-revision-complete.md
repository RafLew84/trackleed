# Asercja content-revision-complete

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
Blocked by: 13

## Question

Czy wszystkie siedem bramek ma komplet dowodów i pozwala bezpiecznie asertować
`content-revision-complete` bez sugerowania `submission-ready`?

## Objective

Zebrać dowody wszystkich siedmiu gates Definition of Done i asertować
`content-revision-complete` tylko przy pełnym sukcesie.

## Acceptance

- Final validation report linkuje każdy gate, test, reconstruction, successor
  fingerprint i per-page audit.
- Raport wymienia każdą pozostałą lokalizację placeholdera wraz z jedną z
  kategorii `AUTHOR`, `EXPERIMENT`, `RELEASE-AUTHORIZATION`, `SUBMISSION`.
- Nie pozostaje uncategorized placeholder ani missing scientific content.
- Status nie jest utożsamiany z `submission-ready`; zakres późniejszych prac
  pozostaje jawny.
- Mapa i ticket zostają zamknięte dopiero po review Standards/Spec.
