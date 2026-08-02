# Konsolidacja specyfikacji i handoff do implementacji

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `task`
Label: `wayfinder:task`
Mode: `HITL`
Status: `resolved`
Blocked by: 08

## Question

Czy wszystkie zatwierdzone decyzje zostały scalone w jeden niesprzeczny plan
rewizji oraz w uporządkowane, możliwe do wykonywania pojedynczo tickety
implementacyjne?

Zadanie ma zachować source-of-truth dla każdego tekstu, wzoru, denominatora,
tabeli i figury; jawnie oznaczyć autorów i warunki eksperymentalne jako
odłożone; zdefiniować walidację DOCX/SI i scientific-core fingerprint; a
następnie przedstawić użytkownikowi finalny handoff do osobnego workflow
implementacyjnego. Nie wykonuje samych poprawek.

## Answer

Tak. Decyzje z ticketów 02–08 zostały scalone bez konfliktów w
[`../artifacts/09-consolidated-revision-handoff.md`](../artifacts/09-consolidated-revision-handoff.md).
Użytkownik zatwierdził rekomendowany podział I01–I14 2026-08-02. Utworzono
osobny workflow
[`../../trackleed-publication-content-revision-implementation/map.md`](../../trackleed-publication-content-revision-implementation/map.md)
z PRD, jednoznacznymi właścicielami komponentów, fail-closed zależnościami,
bramkami HITL, walidacją DOCX/PDF/SI i kontrolowanym replacement fingerprintu.
Autorzy oraz warunki eksperymentalne pozostają jawnie poza zakresem, a
Abstract ma trwały zakaz odwołań do tabel, figur i Supplementary Information.

## Comments

- 2026-08-02: projekt skonsolidowanego handoffu zapisano w artefakcie 09 i
  przedstawiono na bramce HITL.
- 2026-08-02: użytkownik zatwierdził handoff i materializację workflow I01–I14
  odpowiedzią „tak”.
