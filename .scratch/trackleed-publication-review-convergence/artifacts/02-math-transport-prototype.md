# Prototyp kontraktu notacji matematycznej i transportu do DOCX

Data prototypu: 2026-08-02

## Pytanie i seam

Prototyp sprawdza publiczny seam `Markdown -> Pandoc reader -> DOCX OMath` dla
reprezentatywnych zapisów wskazanych w tickecie 02: `1/sqrt(E)`, modelu Gaussa,
midpoint, reguły availability LocoTrack oraz parametrów Lucas--Kanade i
EasyLEED. Kod i wygenerowane pliki w
`manuscript/review/prototype_math_transport/` są oznaczonym materiałem
throwaway, a nie produkcyjnym builderem.

## Porównane warianty

| Wariant | Źródło | Reader Pandoc | OMath | Kontrola struktur i symboli | Wynik |
|---|---|---|---:|---|---|
| `current-default` | obecne `\(...\)` / `\[...\]` | `markdown` | 0/13 | 0/5 | fail |
| `dollar-default` | przepisane `$...$` / `$$...$$` | `markdown` | 13/13 | 5/5 | pass |
| `single-backslash-reader` | obecne `\(...\)` / `\[...\]` | `markdown+tex_math_single_backslash` | 13/13 | 5/5 | pass |

Pięć kontroli wymagało dokładnej, uporządkowanej treści wszystkich 13 formuł
(w tym pełnych operandów, operatorów, progów i parametrów) oraz dokładnych
liczebności natywnych struktur OOXML: `m:rad`, `m:f`, `m:sSub` i `m:sSup`.
Nie wystarcza sama obecność fragmentu tekstu lub liczba obiektów OMath. Pełny wynik maszynowy znajduje się w
`manuscript/review/prototype_math_transport/results/summary.json`.

Wniosek przyczynowy: aktualne źródło jest semantycznie poprawne, ale komenda
eksportu nie włącza rozszerzenia Pandoc rozpoznającego single-backslash math.
Bloki zapisane w innej składni mogą przechodzić, podczas gdy inline math jest
traktowane jak zwykły tekst i traci polecenia TeX. Wyjaśnia to obserwowane
fragmenty `1/`, `()`, `(_` i brakujące symbole.

## Rekomendowany kontrakt

1. **Reprezentacja źródłowa:** zachować jeden kanoniczny zapis
   `\(...\)` dla matematyki inline i `\[...\]` dla display math w manuskrypcie
   głównym oraz SI. Nie wykonywać masowej, semantycznie pustej migracji do
   delimiterów dolarowych.
2. **Eksport:** wersjonowany builder ma zawsze wywoływać Pandoc z jawnym
   `--from=markdown+tex_math_single_backslash`. Zabronione jest poleganie na
   domyślnym readerze.
3. **Fallback:** brak cichego downgrade do zwykłego tekstu. Formuła, której
   Pandoc nie potrafi przekształcić do OMath, blokuje build. Wyjątek wymaga
   jawnego wpisu w wersjonowanym manifeście fallbacków, zachowania źródłowego
   TeX, wygenerowanego obrazu równania z alt textem i osobnej kontroli
   wizualnej. Początkowy manifest ma być pusty; prototyp nie wykazał potrzeby
   fallbacku dla obecnego reprezentatywnego zakresu.
4. **Fail-closed audit:** builder ma (a) policzyć węzły `Math` w Pandoc AST,
   (b) przerwać na ostrzeżeniu konwersji TeX, (c) po eksporcie wymagać zgodnej
   liczby `m:oMath` plus zadeklarowanych fallbacków, (d) sprawdzić krytyczne
   struktury/symbole w OOXML, (e) przeskanować tekst reader-facing pod kątem
   podejrzanych fragmentów `1/`, `()`, `(_`, `_{} ` z jawną allowlistą oraz
   (f) wymagać renderu i przeglądu każdej strony przed wydaniem dokumentu.

## Granice prototypu

Kanoniczny `render_docx.py` nie uruchomił się w dostępnym runtime z powodu
braku `pdf2image`; w WSL nie ma również LibreOffice/`soffice`. Alternatywny
eksport przez niewidoczną instancję Worda przekroczył limit 30 sekund i nie
utworzył PDF. Zgodnie z fallbackiem workflow dokumentowego prototyp przeszedł
kontrolę strukturalną OOXML, ale nie może być przedstawiony jako wizualnie
zweryfikowany. Produkcyjna implementacja nie może ominąć końcowego render gate.

## Decyzja HITL

Rekomendowany jest wariant `single-backslash-reader` z powyższą regułą
fail-closed i pustym początkowym manifestem fallbacków. Alternatywa
`dollar-default` również zachowuje reprezentatywną matematykę, lecz wymaga
masowej zmiany poprawnego source-of-truth i zwiększa zakres oraz ryzyko diffu
bez korzyści transportowej.
