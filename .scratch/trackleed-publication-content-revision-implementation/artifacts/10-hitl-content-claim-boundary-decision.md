# I10 — decyzja HITL: finalny content i claim boundary

Status: `accepted`
Decision date: 2026-08-02
Decision: `accepted`

## Przedmiot akceptacji

Akceptacja dotyczy dokładnego stanu repozytorium:

- implementation commit:
  `38a0e6aa668c4a7751fb6661e94af6d58bea92bc`;
- commit kanonicznego buildu dokumentów:
  `959d58ff78ca7589f1e617844b788305b8051bc5`;
- predecessor scientific-core fingerprint:
  `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`;
- reviewed component diff report:
  `manuscript/scientific_core/pre_rebaseline_report.json`, SHA-256
  `a04a4c83a4a2651357b21731395ea693443f8a0e81aa499c4c451a20dd23b3b9`.

Test-first nie ma zastosowania do samej decyzji człowieka. Integralność pakietu
decyzyjnego jest zamiast tego chroniona wersjonowanymi hashami, kontraktami I09
i ponowną walidacją manifestu dokumentów przed przedstawieniem decyzji.

## Finalny content

- Abstract jest samowystarczalny i nie odwołuje się do tabel, rycin ani
  Supplementary Information. Hierarchia wkładu zaczyna się od wspólnego
  kontraktu pomiarowego, następnie rozdziela denominator-aware outcomes, a
  Gaussian capture effect pozostawia jako mechanizm wspierający.
- Controlled translations rozdzielają 24 stany inventory od stałych zbiorów
  80 displaced-spot i 40 displaced-pair opportunities na metodę.
- Ag(111) coverage i recovery zachowują stałe mianowniki 221 i 24. Geometry
  inconsistency używa accepted-fit denominators, a conditional residuals są
  nierozłączne z method/variant-specific surviving `n`.
- Sekcja sensitivity obejmuje wszystkie 15 predeclared variants, nie wybiera
  optimum ani nowego centralnego kontraktu i nie tworzy rankingu trackerów.
- Limitations jawnie wykluczają absolute-accuracy ranking, spot-identity ground
  truth, population-level generalization, structural LEED I(V), przypisanie
  przygotowania, kalibrowany współczynnik rozszerzalności, temperaturę
  przemiany i pełną capture-basin map.

Kanoniczne źródła treści:

- `manuscript/manuscript.md` —
  `ebb7b8627c1add431693e9bcc98af6758d3b5e2cf95a9475e74332c40c1cead0`;
- `manuscript/supplementary-material.md` —
  `ed586e984a0efb4e11d68277a0fe1f75ed460cbe017cf5f0e369e57e1c970dda`.

## Display semantics

Reader-facing inventory obejmuje 20 bundles. Figures 1–5 i S1–S6 oraz Tables
1–S7 zachowują rozdzielone role, źródła i kontrakty. Figure S2 pozostaje jednym
numerowanym display item na dwóch częściach `a–d` i `e–f (continued)`, a Figure
5 pozostaje pełna, pięciopanelowa i full-width. Table S6 zachowuje 375 rekordów
kanonicznych oraz udostępnia 75-row reader view, pełnoprecyzyjny aggregate CSV i
dwusheet XLSX.

Powiązane źródła:

- `display_inventory.json` —
  `fe70b93614315b0fe917436bd47a310836400f45455305bcfedb054743081ebb`;
- `semantic_manifest.json` —
  `566d10d7405ce42da7551996e282d45fcb403c45e9d0c84e994a3a75653b0e95`.

## Final scientific claim boundary

Granica obejmuje 9 claims: 4 preserved, 5 narrowed, 0 removed.

- Preserved: C2, C5, C6, C7.
- Narrowed: C1, C3, C4, C8, C9.
- Globalna granica: twierdzenia obowiązują wyłącznie dla testowanych kontraktów
  pomiarowych i reżimów danych. Dowody wspierają odtwarzalność i fizyczną
  samospójność measurement chain, a nie external absolute tracking accuracy,
  spot-identity ground truth, universal method ranking, calibrated structural
  LEED I(V) ani instrument-traceable detector geometry.

Powiązane źródła:

- `manuscript/claim-evidence-map.md` —
  `b51186529b03e692c22a8cf50cf6e66a835101e8e167f045dfea132152196e00`;
- `manuscript/claim_adjudication/output/final_claim_boundary.json` —
  `a0936878a982f2620f8efa9d889f9c35cdf2ab35890d467b95722e0f917cf22c`.

## Reviewed component diff

Raport I09 wiąże wszystkie 23 komponenty: 13 changed i 10 unchanged.

Changed:

1. `claim-boundary-manifest`
2. `cpu-environment`
3. `display-index`
4. `display-inventory-builder`
5. `display-inventory-contract`
6. `display-semantic-inventory`
7. `fingerprint-validator`
8. `manuscript`
9. `manuscript-validator`
10. `reconstruction-entrypoint`
11. `sensitivity-source-provenance`
12. `supplementary-material`
13. `tracer-authentication`

Unchanged:

1. `baseline-package-manifest`
2. `central-panel-authentication`
3. `claim-evidence-map`
4. `experimental-metadata-manifest`
5. `final-claim-boundary`
6. `fingerprint-freeze-entrypoint`
7. `measurement-contract`
8. `reconstruction-orchestrator`
9. `scientific-core-documentation`
10. `sensitivity-summary`

## Dokumenty-kandydaty

Manifest dokumentów przeszedł ponowną walidację bez przebudowy:

- main DOCX — `trackleed_manuscript.docx`, SHA-256
  `c609205522dc42fe1de3640ae6497797d64f005f208fb8ae593b2c2b252ef9c5`;
- main PDF — `trackleed_manuscript.pdf`, SHA-256
  `370e02251c1c1ca09eeabe69904c71aafe24106b3886082972cd22d51d973bd6`;
- SI DOCX — `trackleed_supplementary_information.docx`, SHA-256
  `cc45a3605d917f26a69583fe9bacd4c58a5c3ad8de592a5b38ea1325ae5757a9`;
- SI PDF — `trackleed_supplementary_information.pdf`, SHA-256
  `cf250c17029ef901f8590077e66ee50db46d4fb0d9f015853671650e2b13fada`.

I08 udokumentował pełny przegląd 36 stron main i 65 stron SI oraz poprawny
transport matematyki (main 64/64, SI 10/10, fallback 0). Nie jest to jeszcze
końcowa `Page-level render acceptance`, która zgodnie z sekwencją należy do I13
po kontrolowanym rebaseline.

## Zakres zaakceptowanej decyzji

Rozwiązanie I10 wymagało jawnego potwierdzenia, że użytkownik akceptuje:

1. finalny content main manuscript i SI;
2. display semantics;
3. limitations i denominator semantics;
4. powyższą final scientific claim boundary;
5. powiązanie akceptacji z dokładnym implementation commit oraz reviewed
   component diff.

## Recorded acceptance

On 2026-08-02 the repository user explicitly answered `tak` to the HITL
question asking whether they accept all five items above: final main/SI
content, display semantics, limitations and denominator semantics, final
scientific claim boundary, and the binding to implementation commit
`38a0e6aa668c4a7751fb6661e94af6d58bea92bc` plus the reviewed 23-component
diff. This response authorizes I11 to implement the replacement workflow; it
does not itself authorize manual replacement or modification of
`fingerprint.json`.

## Verification and review

- Kanoniczny document manifest: passed dla czterech kandydatów.
- Deterministyczny pre-rebaseline gate I09: `expected-predecessor-rejection`;
  ponowny zapis raportu nie utworzył diffu.
- Wszystkie hashe źródeł i decision record bindings: passed.
- Pełny `trackleed/tests`: 253 passed, 6 failed. Wszystkie sześć zastanych
  failures wynika wyłącznie z `PermissionError` dla niedostępnego, user-owned
  `manuscript/scientific_panel`; I10 nie zmienia kodu ani tego katalogu.
- Standards review: 0 actionable findings.
- Spec review: 0 actionable findings.
