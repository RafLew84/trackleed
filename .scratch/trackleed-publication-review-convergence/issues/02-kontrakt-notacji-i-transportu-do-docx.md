# Kontrakt notacji matematycznej i transportu do DOCX

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `prototype`
Label: `wayfinder:prototype`
Mode: `HITL`
Status: `open`
Blocked by: 01

## Question

Jaki jeden kontrakt authoring-to-DOCX ma zachować wszystkie wzory, symbole
greckie, indeksy i operatory w manuskrypcie głównym oraz SI bez uszkodzeń typu
`1/`, `()`, `(_` i utraconych parametrów?

Na podstawie [Audytu uwag względem źródeł i wygenerowanych
DOCX](01-audyt-uwag-wzgledem-zrodel-i-docx.md) prototyp ma porównać minimalne
reprezentatywne warianty dla `1/sqrt(E)`, modelu Gaussa, midpoint, LocoTrack
availability i parametrów Lucas–Kanade/EasyLEED. Użytkownik ma zatwierdzić
reprezentację źródłową, sposób eksportu, regułę fallback oraz fail-closed audyt
wyłapujący uszkodzoną notację przed wydaniem dokumentu.
