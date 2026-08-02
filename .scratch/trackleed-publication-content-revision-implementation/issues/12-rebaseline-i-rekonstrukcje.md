# Autoryzowany rebaseline i dwie rekonstrukcje

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 11

## Question

Czy autoryzowany successor fingerprint jest odtwarzalny bez driftu w pinned i
clean-Windows reconstruction?

## Objective

Wykonać jednorazowy autoryzowany replacement i udowodnić powtarzalność
successor fingerprintu.

## Acceptance

- Replacement używa rekordu akceptacji z ticketu 10 i exact commit.
- `freeze_scientific_core --validate-only` przechodzi.
- Pełny `reconstruct_publication --clean --validate` przechodzi w pinned
  CPU/headless environment.
- Drugi pełny run przechodzi z clean Windows checkout z wszystkimi
  manifest-declared local inputs; digest jest identyczny i nie ma driftu.

## Answer

Tak. Autoryzowany workflow użył exact predecessora, decision recordu I10,
zaakceptowanego implementation commitu i kanonicznej autoryzacji. Successor
`66b7c59994fe5369ed02f0c8c04e4f924823a52cc423ab02a66a84302b151a3e`
przechodzi `freeze_scientific_core --validate-only`.

Dwie pełne rekonstrukcje CPU/headless wykonane w niezależnych clean Windows
checkoutach z tym samym kompletem uwierzytelnionych lokalnych wejść przeszły bez
driftu. Ich raporty JSON są bajtowo identyczne, a oba checkouty pozostały czyste.

Dowody: [artifact 12](../artifacts/12-authorized-rebaseline-reconstruction.md).

## Comments

- 2026-08-02: pierwszy preflight ujawnił, że clean Git checkout wymaga
  dostarczenia ignorowanych, lokalnych wejść danych i `.scratch`; finalne dwa
  przebiegi otrzymały kompletny identyczny pakiet i uwierzytelniły go hashami.
- 2026-08-02: wynik replacementu został przeniesiony do bieżącej gałęzi bez
  ingerencji w user-owned zmiany robocze.
