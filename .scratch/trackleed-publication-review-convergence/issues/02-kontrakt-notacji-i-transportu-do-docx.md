# Kontrakt notacji matematycznej i transportu do DOCX

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `prototype`
Label: `wayfinder:prototype`
Mode: `HITL`
Status: `resolved`
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

## Comments

### 2026-08-02 — prototyp transportu matematyki

Porównanie trzech wariantów zapisano w
[artefakcie prototypu](../artifacts/02-math-transport-prototype.md). Obecny
reader utworzył 0/13 obiektów OMath; zarówno migracja do delimiterów dolarowych,
jak i zachowanie obecnego źródła z readerem
`markdown+tex_math_single_backslash` utworzyły 13/13 i przeszły kontrole pełnej
treści formuł oraz natywnych struktur pierwiastków, ułamków, indeksów dolnych i
górnych. Rekomendacja zachowuje obecne źródło, jawnie zmienia reader,
zabrania cichego fallbacku i dodaje fail-closed audit. Ticket czeka na decyzję
HITL przed zapisaniem odpowiedzi i rozwiązaniem.

## Answer

Użytkownik zatwierdził wariant `single-backslash-reader`. Kanoniczna
reprezentacja źródłowa pozostaje bez zmian: `\(...\)` dla matematyki inline i
`\[...\]` dla display math. Wersjonowany eksport DOCX ma jawnie używać
`--from=markdown+tex_math_single_backslash`.

Cichy fallback do zwykłego tekstu jest zabroniony. Niekonwertowalna formuła
blokuje build; ewentualny wyjątek wymaga jawnego wpisu w manifeście fallbacków,
zachowania źródłowego TeX, obrazu z alt textem i osobnej kontroli wizualnej.
Początkowy manifest pozostaje pusty. Fail-closed audit ma porównywać liczbę
węzłów `Math` w Pandoc AST z natywnymi obiektami `m:oMath`, przerywać na
ostrzeżeniach konwersji, kontrolować krytyczne struktury i symbole OOXML,
skanować podejrzane fragmenty z allowlistą oraz wymagać końcowego renderu i
przeglądu każdej strony.

Dowód i pełne uzasadnienie znajdują się w
[artefakcie prototypu](../artifacts/02-math-transport-prototype.md).
