# Skonsolidowany handoff rewizji treści publikacji

Data projektu handoffu: 2026-08-02

## Status i granica tego dokumentu

Dokument scala zatwierdzone decyzje z ticketów 02–08 w jeden plan wykonawczy.
Nie wdraża zmian w manuskrypcie, SI, tabelach, figurach, builderach ani
fingerprincie. Utworzenie osobnego workflow i jego ticketów wymaga zatwierdzenia
przez użytkownika, zgodnie z bramką HITL ticketu 09.

Docelowym statusem osobnego workflow jest `content-revision-complete`, a nie
`submission-ready`.

## Reguły pierwszeństwa i source-of-truth

1. Dane i granice claims mają pierwszeństwo przed reader-facing skrótem.
   Tekst, podpis ani layout nie mogą zmieniać zamrożonych zbiorów wejściowych,
   wyników lub znaczenia claimów.
2. Dokładne zatwierdzone brzmienie ma pierwszeństwo przed starszym tekstem w
   obecnie wygenerowanych DOCX. DOCX są outputem buildera, nie źródłem treści.
3. Dane pełnoprecyzyjne i rekordy kanoniczne mają pierwszeństwo przed
   reader-facing zaokrągleniem. Table S6 jest widokiem 75 agregatów, nie nowym
   zbiorem wyników.
4. Semantyka mianownika zależy od wyniku. `24` opisuje kontrolowany stan
   wejściowy; `80/40` to stałe displaced opportunity sets odpowiednio dla
   spotów/par; geometry inconsistency używa liczby zaakceptowanych fitów, a
   conditional residual — przeżywającego, zależnego od metody i wariantu `n`.
5. Jeden komponent ma jednego właściciela implementacyjnego. Późniejsze tickety
   jedynie konsumują jego output i nie poprawiają go boczną ścieżką.
6. Wygenerowanych artefaktów nie edytuje się ręcznie. Zmiana zaczyna się w
   kanonicznym Markdown, danych, builderze lub wersjonowanym manifeście, a potem
   przebudowuje zależne outputy.
7. Stary fingerprint pozostaje niezmiennym poprzednikiem do autoryzowanego
   rebaseline. Ręczne usunięcie lub ciche odświeżenie `fingerprint.json` jest
   zabronione.

## Macierz skonsolidowanych decyzji

| Przedmiot | Normatywna decyzja | Kanoniczne źródło decyzji | Właściciel wdrożenia |
|---|---|---|---|
| Matematyka Markdown | Zachować `\(...\)` i `\[...\]`; reader Pandoc `markdown+tex_math_single_backslash`; fail-closed AST/OMath i render audit; pusty początkowy manifest fallbacków. | `02-math-transport-prototype.md` | Ticket I08 |
| Abstract | Wariant A, 244 słowa; common measurement contract przed Gaussian capture effect; zawiera sensitivity; bez odwołań do tabel, figur i Supplementary Information. | `05-contribution-hierarchy-compression-prototype.md`, z faktami z `03-sensitivity-narrative-prototype.md` i semantyką z `04-denominator-semantics-decision.md` | Ticket I03 |
| Objective i wkład | Kolejność: common contract, denominator-aware assessment, coverage/recovery/consistency, dopiero potem Gaussian capture effect. | artefakt 05 | Ticket I03 |
| Nagłówek 1.1 | Użyć zatwierdzonego brzmienia z `persistent spot association`, bez nieuprawnionego identity claim. | artefakt 05 | Ticket I03 |
| Table 1 i mianowniki | Pokazać 24 stany jako `4 × (seed + 5 displaced)` oraz właściwe result sets 80 spotów/40 par; 96/48 tylko jako inventory, nie opportunity denominator. | artefakt 04 | Ticket I03 |
| Results sensitivity | Dodać zatwierdzoną sekcję 4.4; dotychczasowe 4.4/4.5 przesunąć do 4.5/4.6; zachować wszystkie liczby i case-boundary z wariantu A. | artefakt 03 | Ticket I03 |
| Discussion i Limitations | Zastosować zatwierdzoną integrację sensitivity i kompresję; nie przedstawiać 15 wariantów jako optymalizacji ani rankingu trackerów; usunąć nieaktualne twierdzenie o braku innych ROI. | artefakty 03 i 05 | Ticket I03 |
| Conclusions | Fixed opportunity denominators przypisać coverage/recovery; conditional residual opisać jako warunkowy na metodowo zależnym przeżywającym `n`. | artefakt 04 | Ticket I03 |
| Podpisy Figures 1/4/5 | Użyć zatwierdzonych skróconych podpisów; podpis Figure 5 ma 189 słów. | artefakt 05 | Ticket I04 |
| Table S6 | Reader-facing 75 wierszy, 15 krótkich kodów, 7 kolumn i kontrolowane zaokrąglenie; zachować 375 rekordów kanonicznych; dodać pełnoprecyzyjny CSV 75 agregatów i XLSX z arkuszami `method_aggregate`/`case_records`. | `06-reader-facing-si-display-prototype.md` | Ticket I05 |
| Figure S2 | Jeden numerowany display item na dwóch kolejnych stronach: `a–d` i `e–f (continued)`; zachować panel order, codebook, centralne oznaczenie i inventory count; efektywny font ≥6.0 pt, komórka heatmapy ≥4.5 mm. | artefakt 06 | Ticket I06 |
| Figure 5 | Nadal full-width i ze wszystkimi panelami; source 7.2 × 9.4 in, placement 6.25 in, wskazane fonty, PDF wektorowy i PNG 400 dpi; efektywny font ≥6.0 pt. | artefakt 06; treść podpisu z artefaktu 05 | Ticket I07 |
| Portability | Jeden jawny kontrakt LF dla hash-bound text, `.gitattributes`, deterministyczne writery i dowód z czystego checkoutu Windows. | `07-reconstruction-fingerprint-verification.md`, gate 3 artefaktu 08 | Ticket I02 |
| Fingerprint | Poprzednik SHA-256 `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`; kontrolowany replacement, diff 23 komponentów, atomic write, jawna autoryzacja i historia. | artefakty 07 i 08 | Tickety I09–I12 |
| Dokumenty kanoniczne | Main manuscript i SI, każdy jako DOCX i PDF; wersjonowany builder/manifest; pełny audit każdej strony wszystkich czterech renditions. | artefakty 02 i 08 | Tickety I08 i I13 |
| Status domenowy | Cztery dokładne definicje z artefaktu 08 muszą trafić do śledzonego, kanonicznego dokumentu domenowego przed asercją statusu. | artefakt 08 | Ticket I01 |
| Odłożone inputy | Dozwolone wyłącznie jawne placeholdery `AUTHOR`, `EXPERIMENT`, `RELEASE-AUTHORIZATION`, `SUBMISSION`. | artefakt 08 | Ticket I14 |

## Niesprzeczne granice naukowe

- Centralny kontrakt pozostaje `48 × 48 px / 15 px`; sensitivity bada odporność
  w 15 predeclared variants i nie wybiera nowego optimum.
- Coverage ma stałe 221 baseline-visible opportunities, recovery 24 evaluable
  natural I(V) gaps. Geometry inconsistency zachowuje zaakceptowane-fit
  denominators (`187–286` w pełnej kopercie), a residual zależne pair-frame `n`
  (`54–104` w pełnej kopercie).
- Pełna koperta sensitivity i dokładne wartości w tekście pochodzą wyłącznie z
  źródeł wymienionych w artefakcie 03. Nie wolno ich przepisywać z renderu
  tabeli ani figury.
- C1/C3/C4 pozostają case-boundary. Nie powstaje composite score, universal
  tracker ranking, nowy width contract ani automatyczna selekcja konfiguracji.
- Nie wykonuje się nowej akwizycji, synthetic ground truth, tracker inference,
  GUI ani GPU.

## Proponowany osobny workflow implementacyjny

Nazwa katalogu po akceptacji:
`.scratch/trackleed-publication-content-revision-implementation/`.

Każdy ticket jest małym, osobno wykonywalnym krokiem z własnymi testami,
artefaktem dowodowym i dokładnie wskazanymi zależnościami.

| ID | Ticket | Zakres i wyjście | Zależności |
|---|---|---|---|
| I01 | Wersjonowany kontrakt domenowy rewizji | Umieścić cztery definicje z artefaktu 08 w śledzonym kanonicznym dokumencie; test/guard wykrywający brak definicji. Nie force-addować ignorowanego `context.md` bez jawnej decyzji. | — |
| I02 | Przenośny kontrakt LF | Najpierw testy Windows/hash; następnie `.gitattributes`, deterministyczne writery i dowód clean-checkout bez LF/CRLF drift. | I01 |
| I03 | Proza naukowa i semantyka mianowników | Wdrożyć dokładne teksty 03–05 w kanonicznym `manuscript.md`, łącznie z Abstract, Results 4.4, Discussion, Limitations, Conclusions, numeracją i Table 1; testy claims/liczb/zakazu odwołań w Abstract. | I01 |
| I04 | Kanoniczne źródła podpisów Figures 1/4/5 | Zmienić źródła podpisów i przebudować bundle bez zmiany danych/paneli; test długości i wymaganych granic claims. | I01 |
| I05 | Table S6 i pełnoprecyzyjne eksporty | Wdrożyć widok 75-row, codebook, rounding, aggregate CSV i dwusheet XLSX; zachować i przetestować 375 rekordów kanonicznych oraz manifesty. | I02 |
| I06 | Figure S2 continued i legibility | Wygenerować części `a–d`/`e–f (continued)` jako jeden display item; testować panel/inventory semantics, fonty i fizyczny rozmiar komórek. | I02 |
| I07 | Figure 5 full-width i legibility | Zmienić renderer i placement zgodnie z kontraktem, zachowując wszystkie panele i dane; użyć podpisu z I04; testować formaty, DPI i efektywne fonty. | I02, I04 |
| I08 | Kanoniczny builder DOCX/PDF i audit matematyki | Jeden wersjonowany builder/manifest main+SI; poprawny Pandoc reader; fail-closed AST↔OMath; jawnie wykluczyć lub uczynić kanonicznym każdy dodatkowy review DOCX. | I03–I07 |
| I09 | Integracja i walidacja przed rebaseline | Przebudować wszystkie zależne inventory/semantic artifacts; uruchomić targeted i pełne testy; przejrzeć component diff; wymagać, by stary fingerprint odrzucił oczekiwane zmiany. | I02–I08 |
| I10 | HITL finalnego contentu i claim boundary | Przedstawić finalny diff prozy, display semantic inventory, claim-evidence map, limitations, denominators i dokumenty-kandydaty; zapisać akceptację użytkownika i dokładny commit. | I09 |
| I11 | Wersjonowany replacement fingerprintu | Test-first dodać fail-closed replacement command: expected predecessor, decision record, exact commit, dirty-input guard, historia, diff 23 komponentów i atomic successor write. | I10 |
| I12 | Autoryzowany rebaseline i dwie rekonstrukcje | Wykonać replacement; `--validate-only`; pełny pinned CPU/headless run; drugi pełny clean-Windows run; jeden digest, zero drift. | I11 |
| I13 | Pełny audit stron main/SI DOCX/PDF | Zbudować cztery kanoniczne renditions, wyrenderować 100% stron i zapisać hashe, provenance oraz per-page pass/fail dla matematyki, display items, layoutu i placeholderów. | I12 |
| I14 | Asercja `content-revision-complete` | Zebrać dowody wszystkich gates, spisać lokalizacje czterech dozwolonych kategorii placeholderów i zamknąć workflow wyłącznie przy braku innych luk. | I13 |

## Definition of Done dla każdego ticketu implementacyjnego

Każdy ticket:

1. wskazuje kanoniczne pliki wejściowe i wygenerowane outputy;
2. rozpoczyna się od testu lub od jawnie zapisanego powodu, dlaczego test-first
   nie jest możliwy dla danego artefaktu;
3. nie zmienia zakresu naukowego ani komponentów należących do innego ticketu;
4. uruchamia targeted tests/builders oraz wszystkie repozytoryjne testy
   adekwatne do zmiany;
5. aktualizuje manifesty i dokumentuje wynik fail-closed gate;
6. kończy się review Standards i Spec oraz osobnym commitem;
7. nie oznacza się jako zakończony, jeśli jego następca musiałby domyślać się
   source-of-truth, akceptacji lub ścieżki odtworzenia.

Globalna sekwencja akceptacji jest dokładnie sekwencją siedmiu bramek z
`08-content-revision-definition-of-done.md`; podział I01–I14 jedynie przypisuje
im właścicieli i nie osłabia żadnej bramki.

## Jawnie odłożony zakres

Poza workflow pozostają: nazwiska autorów, afiliacje, ORCID, CRediT, funding,
acknowledgements i declarations; kompletny author-approved opis warunków
eksperymentalnych; adapter JAC i cover letter; reviewer-access release
candidate i jego external access test; release authorization, publikacja
repozytorium, licencja, DOI i metadata submission.

Odłożenie nie pozwala zastąpić brakującej treści dowolnym tekstem. Na etapie
`content-revision-complete` może pozostać tylko jawny placeholder należący do
jednej z czterech zatwierdzonych kategorii. Abstract nie może odwoływać się do
tabel, rysunków ani materiałów supplementary.

## Decyzja oczekiwana na bramce HITL

Rekomendacja: zatwierdzić powyższy podział I01–I14 i po akceptacji utworzyć
osobny workflow z PRD, mapą oraz ticketami odzwierciedlającymi ten dokument.
Zmiana liczby lub granic ticketów wymaga korekty macierzy właścicieli przed
materializacją workflow.
