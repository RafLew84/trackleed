# Figure 5 full-width i legibility

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 02, 04

## Question

Czy Figure 5 pozostaje kompletną full-width figurą i spełnia kontrakt
efektywnych fontów, formatów oraz placementu?

## Objective

Zmienić renderer Figure 5 do zatwierdzonego kontraktu 7.2 × 9.4 in i osadzenia
full-width 6.25 in, używając podpisu należącego do ticketu 04.

## Acceptance

- Wszystkie panele i dane pozostają bez zmian.
- Fonty źródłowe odpowiadają artefaktowi 06, a minimum efektywne wynosi ≥6.0 pt.
- PDF pozostaje wektorowy, PNG 400 dpi; brak kolizji i clippingu.
- Testy wymiarów, fontów, formatów i semantic inventory przechodzą.

## Answer

Tak. Kanoniczna Figure 5 pozostaje kompletną, niepodzieloną figurą full-width
z pięcioma panelami i niezmienionym 189-słowowym captionem I04. Builder tworzy
canvas 7.2 × 9.4 in do placementu 6.25 in, PNG 400 dpi oraz wektorowy PDF.
Dokładny kontrakt fontów daje minimum efektywne 6.076 pt; render review nie
wykazał kolizji ani clippingu. Manifest, clean rebuild i semantic inventory
wiążą wymiary, fonty, formaty, panel order i output hashes.

Dowody: [artifact 07](../artifacts/07-figure5-full-width-legibility.md).

## Comments

- 2026-08-02: Standards review finding o surowym teście bajtów PDF rozwiązano
  przez strukturalną inspekcję MediaBox i XObjectów z użyciem `pypdf`.
- 2026-08-02: końcowe review Standards i Spec zakończyły się bez findingów.
