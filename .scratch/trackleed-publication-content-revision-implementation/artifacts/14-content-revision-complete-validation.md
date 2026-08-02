# I14 — content-revision-complete validation

## Wynik

Fail-closed command `python scripts/assert_content_revision_complete.py`
zweryfikował wszystkie siedem normatywnych bramek i zapisał status
`content-revision-complete`. Raport jawnie zapisuje `submission_ready: false`;
nie utożsamia obu statusów.

Wersjonowane rekordy:

- kontrakt: `manuscript/content_revision/contract.json`, SHA-256
  `6a26cf473fc72d69381bc60421e37fb03de2d73f21c97333373eff668aa759e9`;
- raport: `manuscript/content_revision/validation_report.json`, SHA-256
  `ca082790d4456aaceeb3732499d4ca512c7d8c4684f9f9032cdaf4b7cecc5391`.

Raport wiąże każdy gate z hashami dowodów i poleceniami walidacyjnymi, dwie
clean Windows CPU/headless reconstructions, successor scientific-core
`66b7c59994fe5369ed02f0c8c04e4f924823a52cc423ab02a66a84302b151a3e`
oraz zaakceptowany page-level audit czterech rendycji i 200 rekordów stron.
Sprawdza również integralność 23 komponentów successora i rekord
predecessor-to-successor.

## Pozostałe placeholdery

Walidator przeszukuje autorytatywne źródła reader-facing, odrzuca wadliwy lub
nieznany `TODO[...]` i zapisuje każdą lokalizację. Pozostało dokładnie siedem
autoryzowanych wpisów w `manuscript/manuscript.md`:

- `AUTHOR`: linie 9, 960, 964 i 968;
- `EXPERIMENT`: linia 312;
- `RELEASE-AUTHORIZATION`: linia 946;
- `SUBMISSION`: linia 954.

`manuscript_contract` potwierdził kompletną zatwierdzoną treść naukową, dziewięć
claim IDs i brak niekategoryzowanego placeholdera lub missing scientific
content. Kopie w czterech dokumentach reader-facing są związane przez
kanoniczny manifest i zaakceptowany audit renderów.

## Granica submission-ready

Poza tym statusem pozostają jawnie:

- authors, affiliations, ORCID, CRediT, funding, acknowledgements i declarations;
- pełny author-approved opis eksperymentu;
- adapter wybranego czasopisma, cover letter i submission metadata;
- reviewer-access release candidate, test zewnętrznego dostępu, release
  authorization, licencja, DOI i publikacja repozytorium.

## TDD, walidacja i review

- I14 assertion tests: `8 passed`;
- targeted manuscript/document/page-audit/fingerprint suite: `39 passed`;
- Ruff: passed;
- pełny `trackleed/tests`: `271 passed, 6 failed`; sześć zastanych failures
  wynika wyłącznie z `PermissionError` dla niedostępnego, user-owned
  `manuscript/scientific_panel`;
- Standards re-review: 0 remaining findings;
- Spec re-review: 0 remaining findings.

Testy fail-closed obejmują drift dowodu, niepełny zestaw siedmiu bramek,
próbę asercji `submission-ready`, niezaakceptowany page audit, nieznany i
wadliwy placeholder oraz deterministyczność raportu.
