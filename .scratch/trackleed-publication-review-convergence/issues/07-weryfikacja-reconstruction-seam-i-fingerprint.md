# Weryfikacja reconstruction seam i konsekwencji dla fingerprint

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `task`
Label: `wayfinder:task`
Mode: `AFK`
Status: `resolved`
Blocked by: 01

## Question

Czy aktualna scientific reconstruction/validation seam rzeczywiście
przechodzi w czystym, headless i CPU-compatible trybie oraz które planowane
zmiany treści lub display inventory będą wymagały ponownego scientific-core
fingerprint?

Zadanie ma wykonać udokumentowaną komendę rekonstrukcji i walidacji, zapisać
wynik bez rozszerzania zakresu o tracker inference, GUI lub GPU oraz wskazać
fail-closed kroki konieczne po późniejszej implementacji. Nie buduje jeszcze
reviewer-access release candidate.

## Answer

Pełny wynik zapisano w [weryfikacji reconstruction seam i
fingerprintu](../artifacts/07-reconstruction-fingerprint-verification.md).

Komenda `python -m scripts.reconstruct_publication --clean --validate`
przeszła w pinned Python 3.12.9, CPU/headless, z kompletem lokalnych upstreamów:
1620 centralnych ocen, 15 wariantów sensitivity, 20 display bundles i 23
komponenty fingerprintu zakończyły się statusem
`scientific-baseline-complete`. Osobne `freeze_scientific_core
--validate-only` potwierdziło digest
`859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`.

Weryfikacja ujawniła jednak lukę portability: zwykły czysty checkout Windows
nie odtwarza historycznej mieszanki LF/CRLF użytej przez hash-bound pakiety i
reader-facing manifests. Scientific reconstruction jest obliczeniowo poprawna
w bieżącym byte-preserving środowisku, lecz clean-checkout gate wymaga
ujednolicenia końców linii i jawnego testu przed rebaseline.

Zmiany zatwierdzone w ticketach 03–06 dotykają komponentów `manuscript`,
`supplementary-material` i/lub display semantic inventory, więc wymagają
kontrolowanego rebaseline. Ticket 02 nie wpływa na fingerprint, o ile pozostaje
wyłącznie zmianą eksportu DOCX. Obecne narzędzie odmawia zastąpienia istniejącego
fingerprintu; wersjonowany replacement workflow musi zostać zamrożony w DoD i
handoffie, bez ręcznego kasowania w ramach tego ticketu.
