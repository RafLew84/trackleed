# Audyt uwag z `publikacja.md` względem źródeł i dokumentów review

Data audytu: 2026-08-02

## Zakres i metoda

Audyt rozstrzyga uwagi z [`publikacja.md`](../../../publikacja.md) na podstawie:

- aktualnych źródeł [`manuscript.md`](../../../manuscript/manuscript.md) i
  [`supplementary-material.md`](../../../manuscript/supplementary-material.md);
- źródeł, outputów i manifestów tabel oraz figur;
- kontraktu w
  [`manuscript/scientific_core/`](../../../manuscript/scientific_core/README.md)
  oraz bezpośrednio
  [`fingerprint.json`](../../../manuscript/scientific_core/fingerprint.json);
- dwóch dokumentów review:
  `manuscript/review/trackleed_scientific_review.docx` i
  `manuscript/review/trackleed_supplementary_information.docx`;
- skryptów budujących artefakty naukowe i obecnej ścieżki składania DOCX.

Rozstrzygnięcia nie zmieniają manuskryptu. `Przyjąć` oznacza potwierdzony
problem albo utrzymany wymóg, `przyjąć częściowo` — trafną obserwację z
zawężonym zakresem, `odrzucić` — wniosek sprzeczny z lokalnym kontraktem, a
`odroczyć` — prawidłową potrzebę należącą do późniejszej kompletności
submission.

## Ustalenia przekrojowe

1. Źródłowy Markdown zawiera poprawną notację matematyczną, lecz oba DOCX
   gubią znaczną część matematyki inline. W głównym DOCX są tylko 6 obiektów
   Word OMath; w SI nie ma żadnego. Uszkodzone są między innymi
   `1/sqrt(E)`, parametry Gaussa, definicja midpoint, zależność radialna oraz
   formuły metod w SI. Jest to defekt transportu do DOCX, nie źródła
   naukowego.
2. Pełna, 15-wariantowa sensitivity istnieje w SI, Table S6, Figure S2 i
   manifestach, a także uczestniczy w claim adjudication. Nie została jednak
   włączona do głównej narracji. Zdanie w Limitations, że nie testowano innych
   rozmiarów ROI, jest już niezgodne z wykonanym audytem sensitivity.
3. Źródła denominatorów poprawnie ustalają 80 displaced-spot i 40
   displaced-pair opportunities. Reader-facing opis w Table 1 nadal podaje
   96/48, chociaż seed states są wyłączone z właściwych mianowników wyników.
   Obliczenia nie wymagają zmiany; wymaga jej kontrakt prezentacji Table 1.
4. Conditional residual ma metodowo zależne `n`, więc zdanie w Conclusions o
   jego „fixed opportunity denominator” jest semantycznie niepoprawne. Stałe
   opportunity denominators dotyczą przede wszystkim coverage i recovery.
5. Figure S2 jest merytorycznie kompletna, ale jej komórki używają etykiet
   rzędu 4.2–4.4 pt. Figure 5 ma bazowy tekst około 7.1 pt, lecz w DOCX została
   umieszczona z szerokością 3.2 cala, co dodatkowo zmniejsza czytelność.
   Problemy są reader-facing i częściowo wynikają z placementu DOCX.
6. Obecna ścieżka składania DOCX nie ma jednego, wersjonowanego
   source-of-truth w repozytorium. Skrypt w `manuscript/review/` jest nieaktualny,
   a bieżące składanie i stylowanie wykonano skryptami roboczymi spoza repo.
7. Lokalna definicja `scientific-baseline-complete` jawnie dopuszcza
   placeholdery autorów i opisu eksperymentu. Ich brak nie unieważnia
   bieżącego statusu, choć pozostaje przeszkodą dla późniejszego
   `submission-ready`.

## Bezpośredni audyt fingerprintu

[`fingerprint.json`](../../../manuscript/scientific_core/fingerprint.json) ma
status `scientific-baseline-complete` i digest
`859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`.
Przyjęte uwagi dotykają bezpośrednio jego komponentów:

- `manuscript` (`final_prose`, `denominators`, `limitations`) — integracja
  sensitivity, semantyka denominatorów, hierarchia i kompresja;
- `supplementary-material` (`final_prose`, `si_inventory`, `denominators`,
  `limitations`) — reader-facing Table S6 i opis SI;
- `display-inventory-contract`, `display-semantic-inventory` i `display-index`
  (`displays`, `si_inventory`, `denominators`) — czytelność i ewentualny
  reader-facing wariant Figure S2/Figure 5;
- `sensitivity-source-provenance`, `sensitivity-summary` i
  `claim-boundary-manifest` (`sensitivity`, `denominators`,
  `claim_boundaries`) — udział sensitivity w adjudykacji claims;
- `reconstruction-entrypoint`, `fingerprint-freeze-entrypoint` i
  `fingerprint-validator` (`builders`, `validators`) — niezależna weryfikacja
  reconstruction seam.

Dlatego semantyczne wdrożenie przyjętych uwag zmieni co najmniej jeden
zahashowany komponent. Bieżący fingerprint może być jedynie walidowany podczas
audytu; kontrolowany rebaseline należy wykonać dopiero po zakończeniu rewizji.

## Macierz adjudykacji

| # | Uwaga lub zalecenie | Decyzja | Warstwa | Dowód / source-of-truth | Dalsza ścieżka |
|---:|---|---|---|---|---|
| 1 | Nie zmieniać kierunku pracy i nie dodawać synthetic ground truth, nowych serii ani Gaussian I(E). | **przyjąć** | źródło naukowe — constraint | Ograniczenie jest zgodne z `context.md`, scientific-core README i zamrożonym claim boundary. Nie jest potrzebna zmiana artefaktu. | Utrzymać jako constraint mapy; bez osobnego ticketu. |
| 2 | Włączyć 15-wariantową sensitivity do Results, Discussion i Limitations oraz odnieść ją do claims. | **przyjąć** | źródło naukowe | `supplementary-material.md`, Table S6, Figure S2 oraz `figure_s2_measurement_sensitivity/manifest.json` potwierdzają 15 wariantów; `manuscript.md` opisuje tylko leave-one-case-out, a Limitations błędnie neguje test innych ROI. | [Integracja sensitivity z narracją i granicą claims](../issues/03-integracja-sensitivity-z-narracja-i-claims.md). |
| 3 | Uzupełnić teraz pustą sekcję warunków eksperymentalnych. | **odroczyć** | odłożona kompletność submission | Placeholder w `manuscript.md` jest zgodny z decyzją użytkownika i lokalną definicją w `context.md` oraz scientific-core README. Dane ma później dostarczyć zespół eksperymentalny. | Poza tą mapą; końcowy etap kompletności submission. |
| 4 | Brak opisu eksperymentu oznacza, że status `scientific-baseline-complete` nie został osiągnięty. | **odrzucić** | odłożona kompletność submission | `context.md`, scientific-core README i bezpośredni status w `fingerprint.json` dopuszczają ten placeholder w obecnym statusie. Opinia używa innej definicji gotowości niż repozytorium. | Bez ticketu; nie zmieniać lokalnego statusu na tej podstawie. |
| 5 | Naprawić uszkodzoną matematykę w DOCX i skontrolować każdą stronę. | **przyjąć** | eksport DOCX | Notacja w `manuscript.md` i `supplementary-material.md` jest poprawna; inspekcja Word potwierdza utratę matematyki w DOCX. Source-of-truth dla transportu nie jest obecnie jednoznacznie wersjonowany. | [Kontrakt notacji matematycznej i transportu do DOCX](../issues/02-kontrakt-notacji-i-transportu-do-docx.md). |
| 6 | Ujednolicić opportunity set Table 1 z denominatorami 80/40. | **przyjąć** | reader-facing display | `table01_data_evidence_roles/output/table01_data_evidence_roles.md` podaje 96/48; `table_s2_gates_denominators/source_data/denominator_records.csv` oraz Tables S5/S7 potwierdzają właściwe 80/40 po wyłączeniu seed states. | [Semantyka denominatorów w main text](../issues/04-semantyka-denominatorow-w-main-text.md). |
| 7 | Poprawić zdanie o fixed denominator dla conditional residual w Conclusions. | **przyjąć** | źródło naukowe | `manuscript.md` przypisuje fixed opportunity denominator jakości warunkowej; Table S2 pokazuje metodowo zależną liczbę branch-consistent pair images. | [Semantyka denominatorów w main text](../issues/04-semantyka-denominatorow-w-main-text.md). |
| 8 | Przesunąć środek ciężkości Abstract i Objective z efektu fitu Gaussa na measurement contract i denominator-aware assessment. | **przyjąć częściowo** | źródło naukowe | Hierarchia Objective rzeczywiście zaczyna się od hipotezy o estimatorze, ale Gaussian fit pozostaje ważnym wynikiem. Potrzebna jest zmiana kolejności i proporcji, nie usunięcie wyniku. Integracja sensitivity musi poprzedzić finalną kompresję. | [Integracja sensitivity z narracją i granicą claims](../issues/03-integracja-sensitivity-z-narracja-i-claims.md), następnie [Hierarchia wkładu i kompresja tekstu](../issues/05-hierarchia-wkladu-i-kompresja-tekstu.md). |
| 9 | Skrócić Abstract do około 230–250 słów i dodać jedno zdanie o sensitivity. | **przyjąć częściowo** | źródło naukowe | Abstract jest za długi względem zaleconego celu, ale docelowy limit nie wynika jeszcze z adaptera konkretnego czasopisma. Kompresja ma zachować claim boundary i pomieścić wynik sensitivity. | Tickety [03](../issues/03-integracja-sensitivity-z-narracja-i-claims.md) i [05](../issues/05-hierarchia-wkladu-i-kompresja-tekstu.md). |
| 10 | Zmienić nagłówek 1.1 z „persistent spot identity” na bezpieczniejsze „association”. | **przyjąć częściowo** | źródło naukowe | Manuskrypt konsekwentnie zaznacza brak identity ground truth, więc nagłówek jest mocniejszy niż reszta claim boundary. Ostateczne brzmienie powinno wynikać z hierarchii narracji, a nie być mechaniczną zamianą jednego słowa. | [Hierarchia wkładu i kompresja tekstu](../issues/05-hierarchia-wkladu-i-kompresja-tekstu.md). |
| 11 | Zaokrąglić reader-facing Table S6, skrócić etykiety, zachowując pełną precyzję w CSV. | **przyjąć** | reader-facing display | `table_s6_measurement_sensitivity/output/table_s6_measurement_sensitivity.md` zawiera 75 wierszy i pełną precyzję; równoległy CSV może pozostać kanonicznym artefaktem dokładnym. | [Reader-facing SI i czytelność display items](../issues/06-reader-facing-si-i-czytelnosc-display-items.md). |
| 12 | Zwiększyć czytelność gęstej Figure S2. | **przyjąć** | reader-facing display | Manifest i builder potwierdzają 15 wariantów oraz bardzo małe etykiety komórek; problem nie podważa danych ani decyzji width-trigger. | [Reader-facing SI i czytelność display items](../issues/06-reader-facing-si-i-czytelnosc-display-items.md). |
| 13 | Zachować Figure 5, lecz poprawić jej czytelność i skrócić podpis. | **przyjąć** | reader-facing display + eksport DOCX | Figura ma ważną rolę dowodową, ale drobne etykiety i placement DOCX o szerokości 3.2 cala stwarzają realne ryzyko nieczytelności. | [Reader-facing SI i czytelność display items](../issues/06-reader-facing-si-i-czytelnosc-display-items.md), z kompresją podpisu w [ticket 05](../issues/05-hierarchia-wkladu-i-kompresja-tekstu.md). |
| 14 | Skrócić główny tekst i długie podpisy, ograniczyć powtórzenia caveatów. | **przyjąć częściowo** | źródło naukowe + reader-facing display | Źródło zawiera długie podpisy (szczególnie Figures 1 i 5) i powtarza część ograniczeń. Nie ma jeszcze journal-specific limitu, więc kryterium musi być semantyczne i venue-portable. | [Hierarchia wkładu i kompresja tekstu](../issues/05-hierarchia-wkladu-i-kompresja-tekstu.md). |
| 15 | Finalnie zweryfikować `python -m scripts.reconstruct_publication --clean --validate`. | **przyjąć częściowo** | źródło naukowe — walidacja | Recenzent potwierdził jedynie, że nie mógł uruchomić ścieżki na podstawie DOCX; nie jest to dowód jej awarii. Repo deklaruje tę komendę, ale potrzebny jest niezależny run i ocena wpływu zmian na fingerprint. | [Weryfikacja reconstruction seam i konsekwencji dla fingerprint](../issues/07-weryfikacja-reconstruction-seam-i-fingerprint.md). |
| 16 | Uzupełnić autorów, availability i licencje. | **odroczyć** | odłożona kompletność submission | Autorzy są świadomie pominięci do końca rewizji treści. Availability, licencje i release należą do submission/release readiness, nie do bieżącej konwergencji naukowej. | Poza tą mapą; późniejszy etap kompletności submission i release candidate. |

Bilans: 8 uwag przyjętych, 5 przyjętych częściowo, 1 odrzucona i 2
odroczone.

## Routing i konsekwencje

- Problemy eksportu DOCX przejmuje ticket 02.
- Sensitivity i jej udział w claim boundary przejmuje ticket 03.
- Reader-facing semantykę denominatorów przejmuje ticket 04.
- Hierarchię wkładu i kompresję tekstu przejmuje ticket 05.
- Czytelność Table S6, Figure S2 i Figure 5 przejmuje ticket 06.
- Wykonalność reconstruction seam i konsekwencje dla fingerprint przejmuje
  ticket 07.
- Autorzy, eksperyment, availability, licencje, adapter JAC i release candidate
  pozostają poza zakresem tej mapy.

Każda przyjęta zmiana semantyczna w prozie, denominatorach, sensitivity albo
display inventory unieważni bieżący scientific-core fingerprint. Rebaseline
może nastąpić dopiero po implementacji i walidacji całego uzgodnionego zakresu,
nie w ramach niniejszego audytu.
