# PRD: implementacja rewizji treści publikacji TrackLEED

Status: `approved`
Approved: 2026-08-02

## Cel

Wdrożyć zatwierdzoną rewizję venue-portable manuskryptu głównego i
Supplementary Information oraz osiągnąć udokumentowany status
`content-revision-complete`, bez rozszerzania granicy naukowej i bez
uzupełniania odłożonych danych submission.

Normatywną specyfikacją jest
[`../trackleed-publication-review-convergence/artifacts/09-consolidated-revision-handoff.md`](../trackleed-publication-review-convergence/artifacts/09-consolidated-revision-handoff.md),
a Definition of Done —
[`../trackleed-publication-review-convergence/artifacts/08-content-revision-definition-of-done.md`](../trackleed-publication-review-convergence/artifacts/08-content-revision-definition-of-done.md).

## Wymagania funkcjonalne

- Wdrożyć dokładne decyzje tekstowe, matematyczne, denominatorowe i
  reader-facing z artefaktów 02–06 poprzedniego workflow.
- Zachować common-measurement-contract-first hierarchy, case-based claim
  boundary, 15 predeclared sensitivity variants i brak rankingu trackerów.
- Zachować rozdział 24-state inventory, fixed 80/40 opportunity sets oraz
  outcome-specific conditional denominators.
- Utrzymać Abstract jako samowystarczalny: bez odwołań do tabel, rysunków i
  Supplementary Information.
- Zapewnić kanoniczny builder main/SI do DOCX/PDF, poprawny transport
  matematyki i fail-closed audit każdej strony.
- Zamknąć portability LF/CRLF przed kontrolowanym rebaseline scientific core.
- Zachować predecessor fingerprint, jawnie zatwierdzić component diff i
  odtworzyć successor bez driftu w dwóch clean reconstructions.

## Wymagania wykonawcze

- Każdy ticket ma jednego właściciela komponentu i osobny commit.
- Stosować TDD dla kontraktów kodowych; wyjątek wymaga uzasadnienia w artefakcie
  ticketu.
- Nie edytować ręcznie generated outputs.
- Każdy ticket uruchamia targeted checks, a gate integracyjny również pełny
  zestaw adekwatnych testów.
- Każdy ticket przechodzi review Standards i Spec przed rozwiązaniem.

## Poza zakresem

Autorzy, affiliations, ORCID, CRediT, funding, acknowledgements, declarations,
author-approved opis eksperymentu, adapter JAC, cover letter, reviewer-access
release candidate, release authorization, licencja, DOI i submission metadata.
Na końcu mogą pozostać tylko jawne placeholdery `AUTHOR`, `EXPERIMENT`,
`RELEASE-AUTHORIZATION` i `SUBMISSION`.

Nie wolno dodawać nowej akwizycji, synthetic ground truth, tracker inference,
nowego width contract, composite score, GUI ani GPU.

## Kryterium zakończenia

Workflow kończy się wyłącznie po przejściu wszystkich siedmiu bramek z
Definition of Done, pełnym audycie czterech kanonicznych renditions oraz
udokumentowanej asercji `content-revision-complete`.
