# Reader-facing SI i czytelność display items

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `prototype`
Label: `wayfinder:prototype`
Mode: `HITL`
Status: `resolved`
Blocked by: 01, 03

## Question

Jak uprościć reader-facing Table S6 oraz poprawić czytelność Figure S2 i
Figure 5 bez utraty pełnej precyzji source data, zmiany centralnego kontraktu
ani sztucznego zwiększania liczby głównych wyników?

Prototyp ma rozstrzygnąć krótkie nazwy 15 wariantów, zasady zaokrągleń,
pozostawienie pełnej 75-wierszowej precyzji w CSV/XLSX oraz próg, przy którym
Figure S2 pozostaje jednym display item albo dzieli się na część a–d i e–f.
Dla Figure 5 ma pokazać minimalną zmianę fontów, rozmiaru i długości podpisu
przy zachowaniu full-width role w manuskrypcie.

## Comments

### 2026-08-02 — prototyp reader-facing SI i display items

Warianty Table S6, mierzalny gate czytelności Figure S2 oraz minimalną zmianę
Figure 5 zapisano w
[artefakcie prototypu](../artifacts/06-reader-facing-si-display-prototype.md).
Rekomendacja zachowuje 75 method aggregates w reader table, pełne 375 rekordów
source data, jeden numer Figure S2 rozłożony na części `a–d` i `e–f
(continued)` oraz full-width Figure 5. Ticket czeka na decyzję HITL.

## Answer

Użytkownik zatwierdził pełny wariant rekomendowany w
[artefakcie prototypu](../artifacts/06-reader-facing-si-display-prototype.md).
Table S6 zachowuje wszystkie 75 method aggregates, ale używa siedmiu zwartych
kolumn, 15 jednoznacznych krótkich kodów wariantów i jawnych zasad zaokrągleń.
Fixed outcomes pozostają exact `x/N`, geometry inconsistency zachowuje accepted-
fit denominator, a conditional median/P95 zachowują method- and variant-specific
`n`. Istniejący 375-row canonical CSV zachowuje pełne 75 method aggregates i
300 case records; dodatkowy 75-row aggregate CSV oraz XLSX z arkuszami
`method_aggregate` i `case_records` zachowują pełną precyzję i wchodzą do
manifestu oraz hash chain.

Figure S2 pozostaje jednym numerowanym supplementary display item, ale jest
renderowana na dwóch kolejnych stronach: `a–d` i `e–f (continued)`. Podział nie
tworzy Figure S3, nie zmienia panel order ani inventory count. Każdy finalny
render musi spełnić gate: minimum 6.0 pt dla informative text, minimum 4.5 mm
dla komórek heatmapy, czytelne `x/N` i `n` przy 100% zoom, brak kolizji oraz
zaakceptowany render PDF/PNG. Przy source width 7.2 in i placement 6.25 in
najmniejszy source font wynosi 7.0 pt, czyli efektywnie około 6.08 pt.

Figure 5 pozostaje full-width w main text. Rozmiar źródłowy zmienia się z
`7.2 × 9.0 in` na `7.2 × 9.4 in`, najmniejszy source font na 7.0 pt, co przy
osadzeniu 6.25 in daje około 6.08 pt. Zachowane są wszystkie panele, wektorowy
PDF, PNG 400 dpi oraz zatwierdzony w tickecie 05 podpis 189-word. Decyzja nie
zmienia danych, centralnego kontraktu, wyników ani scientific-core fingerprint.
