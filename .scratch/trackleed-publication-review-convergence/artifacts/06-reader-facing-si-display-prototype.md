# Prototyp reader-facing SI i czytelności display items

Data prototypu: 2026-08-02

## Zakres i stan wejściowy

Prototyp obejmuje wyłącznie prezentację Table S6, Figure S2 i Figure 5. Nie
zmienia centralnego kontraktu, 15 wariantów, żadnego wyniku, denominatora ani
pełnoprecyzyjnego source data.

Stan wejściowy:

- Table S6 ma 75 reader-facing rows (`15 variants × 5 methods`), 12 kolumn i
  pełnoprecyzyjne mediany/P95; towarzyszący CSV zawiera 375 rekordów: 75
  method aggregates oraz 300 case records.
- Figure S2 ma sześć paneli w jednym arkuszu `11.0 × 9.5 in`; najmniejsze
  informative fonts mają `4.2–5.5 pt` już w źródle.
- Figure 5 ma `7.2 × 9.0 in`, najmniejszy font `5.7 pt`, a w review DOCX jest
  osadzana na szerokości `6.25 in` jako full-width main-text figure.

## Table S6

### Krótkie nazwy 15 wariantów

| Pełny identyfikator | Reader-facing code | Znaczenie |
|---|---|---|
| `roi_32_proposal_12` | `R32-P12` | ROI 32 px; proposal-to-fit 12 px |
| `roi_32_proposal_15` | `R32-P15` | ROI 32 px; proposal-to-fit 15 px |
| `roi_32_proposal_18` | `R32-P18` | ROI 32 px; proposal-to-fit 18 px |
| `roi_48_proposal_12` | `R48-P12` | ROI 48 px; proposal-to-fit 12 px |
| `roi_48_proposal_15` | `R48-P15*` | central contract |
| `roi_48_proposal_18` | `R48-P18` | ROI 48 px; proposal-to-fit 18 px |
| `roi_64_proposal_12` | `R64-P12` | ROI 64 px; proposal-to-fit 12 px |
| `roi_64_proposal_15` | `R64-P15` | ROI 64 px; proposal-to-fit 15 px |
| `roi_64_proposal_18` | `R64-P18` | ROI 64 px; proposal-to-fit 18 px |
| `branch_tolerance_12` | `B12` | branch tolerance 12 px |
| `branch_tolerance_24` | `B24` | branch tolerance 24 px |
| `r_squared_0p80` | `R2-80` | \(R^2\) minimum 0.80 |
| `r_squared_0p90` | `R2-90` | \(R^2\) minimum 0.90 |
| `amplitude_multiplier_0p75` | `A0.75` | amplitude gate multiplier 0.75 |
| `amplitude_multiplier_1p25` | `A1.25` | amplitude gate multiplier 1.25 |

Gwiazdka oznacza wyłącznie frozen central contract; nie oznacza wyniku
preferowanego ani wariantu wybranego przez sensitivity.

### Wariant A — compact 75-row table (zatwierdzony)

Zachować wszystkie 75 method-aggregate rows, ale zmniejszyć tabelę do siedmiu
kolumn:

| Variant | Method | Contract change | Coverage | Recovery | Geometry inconsistency | Conditional median / P95 [px] (`n`) |
|---|---|---|---:|---:|---:|---:|
| `R48-P15*` | EasyLEED | central: R48, P15, B18, R2-85, A1.00 | 214/221 | 22/24 | 4/222 | 1.737 / 3.719 (`n=84`) |
| `R48-P15*` | LocoTrack | central: R48, P15, B18, R2-85, A1.00 | 217/221 | 23/24 | 36/272 | 1.837 / 5.973 (`n=91`) |
| `R48-P15*` | OSTrack | central: R48, P15, B18, R2-85, A1.00 | 216/221 | 21/24 | 34/269 | 1.770 / 5.937 (`n=90`) |
| `R48-P15*` | DAM4SAM | central: R48, P15, B18, R2-85, A1.00 | 216/221 | 23/24 | 36/272 | 1.788 / 5.973 (`n=91`) |
| `R48-P15*` | Lucas–Kanade | central: R48, P15, B18, R2-85, A1.00 | 188/221 | 17/24 | 14/206 | 1.764 / 4.442 (`n=68`) |

`Family` i sześć osobnych contract columns znikają z reader table. Ich treść
jest zachowana przez codebook i zwartą kolumnę `Contract change`. Każdy
pięciowierszowy blok zaczyna się krótkim kodem wariantu; centralny blok jest
subtelnie cieniowany, bez użycia koloru jako jedynego nośnika znaczenia.

### Wariant B — 15-row summary plus detailed data

Jedna linia na wariant mogłaby raportować tylko envelope pięciu metod, a 75
wierszy przenieść wyłącznie do pliku danych. Ten wariant jest krótszy, ale
utrudnia sprawdzenie method-specific `n` i może wyglądać jak ranking lub
optymalizacja wariantów. Nie jest rekomendowany.

### Reguły zaokrągleń i pliki pełnoprecyzyjne

- ROI, proposal-to-fit i branch tolerance: całkowite piksele.
- \(R^2\): dwa miejsca po przecinku; amplitude multiplier: dwa miejsca;
  wynikowy amplitude threshold: trzy miejsca po przecinku.
- Coverage, recovery i geometry inconsistency: zawsze dokładne `x/N`.
- Opcjonalny procent po `x/N`: dwa miejsca po przecinku; nigdy zamiast counts.
- Conditional median i P95: trzy miejsca po przecinku; `n` jako exact integer.
- Missing conditional value: em dash plus `n=0`, bez zera i bez imputacji.

Reader-facing Markdown/DOCX jest zaokrąglony, ale liczby nie są zaokrąglane w
źródle. Produkcyjny builder ma zachować istniejący 375-row canonical CSV oraz
dodać pełnoprecyzyjny 75-row method-aggregate CSV odpowiadający jeden-do-jednego
wierszom Table S6. Workbook XLSX ma mieć arkusze
`method_aggregate` (75 rows) i `case_records` (300 rows); komórki przechowują
pełne wartości numeryczne, a format wyświetlania stosuje powyższe zaokrąglenia.
CSV pozostaje source of truth, XLSX jest tylko reader convenience export. Oba
pliki muszą wejść do manifestu i hash chain.

## Figure S2

### Jawny legibility gate

Figure S2 może pozostać na jednym arkuszu tylko wtedy, gdy po osadzeniu w
docelowej szerokości SI:

1. każdy informative label i cell annotation ma co najmniej 6.0 pt;
2. każda heatmap cell ma co najmniej 4.5 mm szerokości i wysokości;
3. pełne `x/N` oraz `n` są czytelne przy 100% zoom bez powiększania strony;
4. żaden tick, legend item, panel label ani colorbar label nie nachodzi na inny;
5. wektorowy PDF i 300-dpi PNG przechodzą page render review.

Obecny jeden arkusz nie przechodzi gate: ma informative fonts `4.2–5.5 pt` już
przy natywnych `11 in`, a przy osadzeniu do `6.25 in` skala wynosi 0.568 i
efektywny zakres spada do około `2.4–3.1 pt`.

### Wariant A — jeden arkusz

Pozostawić układ 2 × 2 + 1 × 2 i zwiększyć fonty. Wymagałby ponadstandardowej
landscape placement i nadal ściska 15 kolumn heatmap w pół szerokości. Jest
dopuszczalny tylko, jeśli finalny venue adapter udowodni przejście powyższego
gate na rzeczywistym page render; nie jest rekomendowany jako default.

### Wariant B — a–d / e–f jako continued figure (zatwierdzony)

Zachować **jeden numer i jeden display item, Figure S2**, ale wygenerować dwie
części na kolejnych stronach SI:

- `Figure S2a–d`, `7.2 × 10.0 in`: cztery pełnoszerokie heatmapy ułożone
  pionowo, ze wspólnym codebookiem wariantów; cell annotations minimum 7.0 pt,
  ticks/legend/colorbar minimum 7.2 pt;
- `Figure S2e–f (continued)`, `7.2 × 6.0 in`: gate margins i width-trigger audit
  ułożone pionowo; annotations minimum 7.0 pt, pozostały tekst minimum 7.2 pt.

Obie części są osadzane na szerokości `6.25 in`. Przy skali `6.25/7.2`
najmniejszy source font 7.0 pt daje efektywne 6.08 pt i przechodzi gate; zmiana
szerokości placement wymaga ponownego obliczenia effective font size.

Caption pozostaje jeden i opisuje wszystkie panele. Druga strona zaczyna się od
`Figure S2 (continued)` i powtarza codebook, ale nie tworzy Figure S3 ani nowego
wyniku. Centralny `R48-P15*` zachowuje obrys i tekstowe oznaczenie. Rozdzielenie
nie zmienia panel order, danych ani inventory count.

## Figure 5

Figure 5 pozostaje full-width main-text figure i nie jest dzielona. Minimalna
zmiana produkcyjna:

- rozmiar źródłowy z `7.2 × 9.0 in` do `7.2 × 9.4 in`;
- osadzenie na pełnej dostępnej szerokości, obecnie `6.25 in`;
- fonty źródłowe: base `8.0`, axis title `9.0`, axis label `8.0`, tick `7.0`,
  legend `7.0`, panel label `10.0`, value annotation `7.0`, minor annotation
  `7.0 pt`;
- PDF pozostaje wektorowy, PNG pozostaje 400 dpi;
- zastosować zatwierdzony w tickecie 05 podpis 189-word zamiast obecnego
  232-word podpisu.

Przy osadzeniu `6.25/7.2` najmniejszy font efektywny wynosi około `6.08 pt`.
Obecny minor font daje około `4.95 pt`. Acceptance gate wymaga braku kolizji i
ucięć, czytelności osi/legend/annotations przy 100% zoom oraz pełnego render
review w DOCX i PDF. Jeśli venue adapter daje mniej niż 6.25 in szerokości,
figura ma otrzymać pełną stronę lub odpowiednio większe fonty; nie należy
usuwać paneli ani przenosić Figure 5 do SI.

## Rekomendacja HITL

Użytkownik zatwierdził compact 75-row Table S6 z pełnoprecyzyjnymi CSV/XLSX, Figure S2 jako
jeden numerowany display rozłożony na strony `a–d` i `e–f (continued)` oraz
full-width Figure 5 z minimalnym zwiększeniem wysokości i fontów i z
189-słowowym podpisem zatwierdzonym w tickecie 05.
