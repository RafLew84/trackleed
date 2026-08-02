# Zamrożenie Definition of Done rewizji treści

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `grilling`
Label: `wayfinder:grilling`
Mode: `HITL`
Status: `resolved`
Blocked by: 02, 03, 04, 05, 06, 07

## Question

Które dokładne warunki muszą być spełnione, aby wersję po tej rewizji uznać
za zaakceptowaną pod względem treści naukowej i czytelności, mimo nadal
odłożonych autorów oraz warunków eksperymentalnych?

Użytkownik ma zatwierdzić wynik wszystkich poprzednich decyzji, wymagany
zakres ponownego renderowania i audytu każdej strony, zasady rebaseline
scientific-core fingerprint oraz granicę między zakończeniem tej rewizji a
późniejszym adapterem JAC, release candidate i statusem `submission-ready`.

## Comments

### 2026-08-02 — decyzja 1: status końcowy rewizji

Użytkownik zatwierdził osobny status `content-revision-complete`. Oznacza on
wdrożenie, rekonstrukcję, kontrolowany rebaseline i page-level acceptance
wszystkich zatwierdzonych zmian treści, denominatorów, sensitivity, display
items i czytelności. Może współistnieć ze `scientific-baseline-complete` mimo
autoryzowanych placeholderów autorów i warunków eksperymentalnych, ale nie
oznacza `submission-ready`.

### 2026-08-02 — decyzja 2: render gate

`Content-revision-complete` wymaga pełnego przebudowania kanonicznych main
manuscript i Supplementary Information do DOCX oraz PDF, wyrenderowania każdej
strony obu formatów do PNG i udokumentowanego audytu 100% stron czterech
renditions: main DOCX, main PDF, SI DOCX i SI PDF. Wspólny page-image set jest
dopuszczalny tylko dla hash-identycznego PDF będącego renderem danego DOCX, z
jawną provenance w manifeście. Kontrola obejmuje matematykę,
tabele, figury, podpisy, numerację, cross-references, clipping, fonty i
autoryzowane placeholdery. Dodatkowy dokument, w tym istniejący
`trackleed_scientific_review.docx`, jest niekanoniczny, dopóki nie zostanie
jawnie wpisany do manifestu, przebudowany i poddany identycznemu audytowi.

### 2026-08-02 — decyzja 3: controlled rebaseline

Fingerprint otrzymuje wersjonowany, fail-closed replacement workflow bez
ręcznego usuwania istniejącego pliku. Operacja wymaga expected old SHA
`859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`,
zatwierdzonego decision record i konkretnego commita; archiwizuje predecessor,
zapisuje diff 23 komponentów oraz relację predecessor → successor i atomowo
zapisuje nowy fingerprint. Rebaseline jest dopuszczalny dopiero po naprawie
LF/CRLF, oczekiwanym odrzuceniu zmienionych komponentów przez stary fingerprint
i udanym clean buildzie. Po replacement muszą przejść `--validate-only`, pełna
clean reconstruction oraz drugi identyczny run z czystego checkoutu Windows,
bez diffu artefaktów.

### 2026-08-02 — decyzja 4: granica późniejszych etapów

`Content-revision-complete` dopuszcza wyłącznie jawne placeholdery kategorii
`AUTHOR`, `EXPERIMENT`, `RELEASE-AUTHORIZATION` i `SUBMISSION`. Nie obejmuje
danych autorów, pełnego opisu eksperymentalnego, adaptera JAC, cover letter,
reviewer-access release candidate, DOI, publikacji repozytorium ani statusu
`submission-ready`. Późniejsze uzupełnienie tych elementów uruchamia dotknięte
walidacje i rebaseline. Adapter JAC może zmieniać format, lecz każda zmiana
znaczenia naukowego ponownie otwiera `content-revision-complete`.

## Answer

Użytkownik zatwierdził kompletny Definition of Done zapisany w
[artefakcie decyzji](../artifacts/08-content-revision-definition-of-done.md).
Status końcowy tej mapy to `content-revision-complete`: wszystkie decyzje
02–07 są wdrożone, finalna treść i claim boundary zaakceptowane, portability
LF/CRLF naprawiona, scientific core odtworzony i kontrolowanie zrebaselined, a
każda strona kanonicznych main manuscript i SI zaakceptowana po renderze.

Status może współistnieć ze `scientific-baseline-complete` przy jawnych
placeholderach `AUTHOR`, `EXPERIMENT`, `RELEASE-AUTHORIZATION` i `SUBMISSION`,
ale nie oznacza `submission-ready`. Adapter JAC, cover letter, author package,
pełny opis eksperymentalny, reviewer-access release candidate, DOI i publikacja
repozytorium pozostają późniejszymi etapami. Zmiana naukowego znaczenia podczas
tych etapów ponownie otwiera tę rewizję i odpowiednie validation/rebaseline
gates.

`Context.md` został zaktualizowany lokalnie o zatwierdzone terminy, ale jest
ignorowany przez obecną konfigurację Git. Wersjonowany artefakt 08 zawiera więc
jawny domain-modeling record i gap: ticket 09 ma przenieść dokładne definicje do
handoffu oraz wymagać świadomego wyboru śledzonego kanonicznego dokumentu domeny
przed nadaniem statusu `content-revision-complete`.
