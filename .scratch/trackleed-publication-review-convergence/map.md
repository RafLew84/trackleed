# Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md

Label: `wayfinder:map`
Status: `active`

## Destination

Jednoznacznie rozstrzygnięta i gotowa do przekazania do implementacji
specyfikacja rewizji venue-portable manuskryptu głównego oraz Supplementary
Information, która uwzględnia wszystkie zasadne uwagi z `publikacja.md` i
prowadzi do wersji zaakceptowanej pod względem treści naukowej oraz
czytelności, bez uzupełniania na tym etapie autorów i warunków
eksperymentalnych.

## Notes

- Źródłem uwag jest [`publikacja.md`](../../publikacja.md), ocenione jako
  ukierunkowana recenzja wersji wygenerowanych po zamknięciu scientific core.
- Zamknięte [Mapa repozycjonowania publikacji TrackLEED bez nowych
  akwizycji](../trackleed-publication-repositioning/map.md) oraz [Mapa
  adjudykacji poprawek do planu publikacji
  TrackLEED](../trackleed-publication-amendment-adjudication/map.md) pozostają
  historią wcześniejszych decyzji i nie są ponownie otwierane.
- Zachować zamrożone pozycjonowanie: methodological Research Paper o wspólnym
  spot-position measurement contract, denominator-aware robustness assessment
  i kompletnym measurement chain; TrackLEED pozostaje reference
  implementation, a nie produktem ani przedmiotem globalnego rankingu.
- Nie dodawać nowych akwizycji, synthetic ground truth, nowych serii ani
  Gaussian I(E). Rewizja korzysta wyłącznie z istniejących danych, wyników i
  wygenerowanych artefaktów.
- Zgodnie z decyzją użytkownika oraz definicją
  `scientific-baseline-complete`, dane autorów i opis warunków
  eksperymentalnych mogą nadal pozostać placeholderami. Opinia recenzenta, że
  sam brak tych dwóch wejść blokuje scientific baseline, nie zmienia tej
  lokalnej definicji statusu.
- Adapter JAC, reviewer-access release candidate i finalna bramka
  `submission-ready` są osobnymi późniejszymi etapami i nie mogą projektować
  treści naukowej tej rewizji.
- Każdą uwagę z `publikacja.md` należy sklasyfikować jako dotyczącą źródła
  naukowego, reader-facing display, eksportu DOCX albo wyłącznie późniejszej
  kompletności submission.
- Semantyczna zmiana w tekście, denominatorach, sensitivity lub display
  inventory unieważnia aktualny scientific-core fingerprint i wymaga
  kontrolowanego rebaseline po implementacji.
- W kolejnych sesjach stosować `prototype` do wariantów tekstu, notacji i
  display items, `grilling` wraz z `domain-modeling` do decyzji HITL oraz
  `task` do lokalnych audytów i walidacji.
- Mapa rozstrzyga decyzje i przygotowuje specyfikację; nie wykonuje właściwej
  rewizji manuskryptu ani nie generuje finalnych dokumentów.

## Decisions so far

- [Audyt uwag względem źródeł i wygenerowanych
  DOCX](issues/01-audyt-uwag-wzgledem-zrodel-i-docx.md) — potwierdzone problemy
  rozdzielono między źródło naukowe, eksport DOCX i reader-facing display;
  autorów oraz warunki eksperymentalne odroczono zgodnie z lokalną definicją
  `scientific-baseline-complete`.
- [Kontrakt notacji matematycznej i transportu do
  DOCX](issues/02-kontrakt-notacji-i-transportu-do-docx.md) — zachowano
  źródłowe `\(...\)` / `\[...\]`, zatwierdzono jawny reader
  `markdown+tex_math_single_backslash`, zakaz cichego fallbacku oraz
  fail-closed audit Pandoc AST i natywnego OMath zakończony render gate.
- [Integracja sensitivity z narracją i granicą
  claims](issues/03-integracja-sensitivity-z-narracja-i-claims.md) — wybrano
  osobną sekcję Results 4.4 dla 15 predeclared variants, samowystarczalne zdanie
  Abstract oraz zmiany Discussion i Limitations zachowujące denominator-aware,
  case-based granicę C1/C3/C4 bez optymalizacji i rankingu trackerów.
- [Semantyka denominatorów w main
  text](issues/04-semantyka-denominatorow-w-main-text.md) — rozdzielono
  24-state inventory od stałych opportunity sets 80/40, przypisano conditional
  residuals do method-specific surviving `n` oraz przyjęto samowystarczalny
  Abstract bez odwołań do tabel, rysunków i Supplementary Information.

## Not yet specified

- Brak dodatkowego fog of war. Dokładny podział implementacji zostanie
  określony dopiero po zamrożeniu wszystkich decyzji treściowych i
  walidacyjnych na tej mapie.

## Out of scope

- Dane autorów, affiliations, ORCID, CRediT, funding, acknowledgements i
  declarations; wrócą w osobnym końcowym etapie.
- Reader-facing opis aparatury, przygotowania próbek, UHV, akwizycji, energii i
  temperatur; zostanie dostarczony i zatwierdzony przez zespół eksperymentalny
  na samym końcu.
- Adapter JAC, cover letter, journal-specific formatting i ponowny wybór
  czasopisma.
- Reviewer-access release candidate, publiczny release, DOI i bramka
  `submission-ready`.
- Nowe akwizycje LEED, nowe próbki, nowe pomiary kalibracyjne, synthetic ground
  truth, nowe serie oraz Gaussian I(E).
- Wykonanie poprawek w manuskrypcie, SI, figurach lub generatorze DOCX w ramach
  samej mapy.
