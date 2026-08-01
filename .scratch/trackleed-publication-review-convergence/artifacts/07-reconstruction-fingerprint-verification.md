# Weryfikacja reconstruction seam i fingerprintu

Data weryfikacji: 2026-08-02

## Werdykt

Scientific reconstruction/validation seam jest obliczeniowo odtwarzalna w
zamrożonym środowisku CPU/headless i przechodzi pełny kontrakt naukowy, ale
obecnie **nie jest przenośna z dowolnego czystego checkoutu Git na Windows**.
Pełny sukces wymagał odtworzenia historycznej mieszanki zakończeń linii LF/CRLF
obecnej w bieżącym worktree oraz dostarczenia wszystkich lokalnych upstreamów
wskazanych przez kontrakty.

To rozróżnienie jest istotne:

- obliczenia, sensitivity, display builders, kontrakt manuskryptu i frozen
  fingerprint przechodzą w pinned runtime;
- zwykły checkout Windows (`core.autocrlf=true`) zatrzymuje się na hashach
  plików, które zamrożono jako LF;
- checkout wymuszający wyłącznie LF przechodzi pakiety wejściowe, lecz zatrzymuje
  się na artefaktach reader-facing zamrożonych albo generowanych jako CRLF;
- brak lokalnych danych nie jest defektem seam: README jawnie wymaga
  authenticated local raw-image paths. Niejawna mieszanka LF/CRLF jest
  natomiast luką portability i Definition of Done.

## Zakres uruchomienia

Punkt odniesienia: commit `8f49456` (późniejsze commity mapy nie zmieniają
scientific core).

Komenda akceptacyjna:

```text
python -m scripts.reconstruct_publication --clean --validate
```

Runtime:

```text
Python 3.12.9
matplotlib 3.10.1
numpy 2.2.5
pandas 2.3.0
pillow 11.2.1
scikit-image 0.25.2
scipy 1.15.2
tifffile 2025.3.30
```

Wykonanie odbyło się w odłączonym worktree. Do snapshotu skopiowano wyłącznie
lokalne wejścia wymagane przez kontrakty:

- 109 unikalnych ścieżek z
  `manuscript/scientific_baseline/input_manifest.json` (81,220,479 B);
- 851 TIFF-ów z
  `manuscript/experimental_metadata/tiff_metadata_inventory.json`
  (851,212,750 B);
- upstream `.scratch/trackleed-physics-method-paper` wymagany przez testy
  kontraktowe display bundles (około 84 MB).

Nie uruchomiono GUI, GPU, model checkpoints, tracker inference ani editorial
renderingu. `CUDA_VISIBLE_DEVICES` było puste, `MPLBACKEND=Agg`, a
`QT_QPA_PLATFORM=offscreen`, zgodnie z kodem orchestratora.

## Wynik pełnego runu

Pełny run zakończył się kodem `0` po 171.6 s. Raport potwierdził:

| Warstwa | Wynik |
|---|---:|
| Central panel | 1620 candidates, 1620 evaluations, 4 cases, 5 methods |
| Fixed reference opportunities | 221 visibility opportunities, 24 gaps |
| Sensitivity | 15 variants, 4860 candidates, 24,300 evaluations |
| Width trigger | `false`; 0 dodatkowych width contracts |
| Table S6 | 375 records: 300 case + 75 method |
| Figure S2 | 9 ROI-matrix + 6 OFAT variants |
| Claim adjudication | 9 claims: 4 preserved, 5 narrowed, 0 removed; checkpoint confirmed |
| Reader-facing inventory | 20 bundles, 158 source files, 44 output files |
| Manuscript contract | 7 main sections, 5 result sections, 9 claim IDs |
| Authorized placeholders | AUTHOR, EXPERIMENT, RELEASE-AUTHORIZATION, SUBMISSION |
| Scientific-core fingerprint | 23 components, wszystkie 13 wymaganych ról |
| Status | `scientific-baseline-complete` |
| Fingerprint SHA-256 | `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff` |

Osobne, read-only:

```text
python -m scripts.freeze_scientific_core --validate-only
```

również zakończyło się kodem `0` i zwróciło ten sam digest oraz 23 komponenty.
Ostrzeżenie SciPy `invalid value encountered in scalar divide` zostało
wyemitowane podczas rekonstrukcji, ale wszystkie fail-closed kontrole i frozen
rekordy przeszły; nie zmieniło statusu ani wyników.

## Ujawnione gate'y portability

| Próba | Fail-closed wynik | Znaczenie |
|---|---|---|
| `git archive --format=zip` | `package hash mismatch: .gitattributes` | Bezpośrednio w ZIP, przed ekstrakcją, `scientific_baseline/.gitattributes` miało 82 B i SHA-256 `b2f960...`, a panel manifest 1341 B i `36946a...`; odpowiada to CRLF, podczas gdy bloby Git mają odpowiednio 78 B i 1323 B. |
| Standardowy Windows worktree | `package hash mismatch: .gitattributes` | Checkout z globalnym `core.autocrlf=true` odtworzył te same CRLF hashe `b2f960...` / `36946a...`, chociaż frozen pakiety oczekują LF `136e86...` / `5fcf89...`. |
| Checkout wymuszający LF | `Publication sensitivity artifact changed: ...table_s6.../manifest.json` | JSON był semantycznie identyczny, lecz generator Windows zapisał manifest CRLF; równocześnie frozen `source_provenance.json` oczekuje CRLF. |
| Niepełny snapshot źródeł | TIFF hash mismatch i brak upstreamów Figure 1 | Kontrakty poprawnie blokują run bez 851 TIFF-ów i źródeł display bundles; zależności wykraczają poza centralny `input_manifest.json`. |
| Byte-preserving snapshot + komplet jawnych upstreamów | sukces | Potwierdza poprawność obliczeń i obecnego fingerprintu, ale nie usuwa luki clean-checkout portability. |

Przed uznaniem seam za przenośną należy w implementacji przyjąć jeden kontrakt
końców linii dla wszystkich hash-bound plików (rekomendowane LF), wymusić go w
`.gitattributes`, a generatory oprzeć na `write_bytes` albo jawnym
`newline="\n"`. Następnie artefakty i fingerprint trzeba odtworzyć w
kontrolowanym rebaseline i sprawdzić zarówno w czystym Windows checkout, jak i
w bieżącym środowisku.

## Wpływ zatwierdzonych ticketów na fingerprint

| Ticket / zmiana | Komponenty fingerprintu | Rebaseline |
|---|---|---|
| 02 — transport matematyki wyłącznie w generatorze DOCX | Brak: editorial DOCX nie należy do scientific core. Zmiana `manuscript.md` lub kanonicznej notacji zmieniłaby komponent `manuscript`. | Nie, jeśli źródłowy Markdown pozostaje bez zmian. |
| 03 — sensitivity w Abstract, Results, Discussion i Limitations | `manuscript` (`final_prose`, `denominators`, `limitations`); ewentualnie `claim-evidence-map` / claim-boundary artifacts, jeśli zmieni się granica claimów. | Tak. |
| 04 — 24 states vs 80/40 opportunities i conditional `n` | `manuscript`; przy zmianie Table 1 także `display-semantic-inventory` i `display-index` przez przebudowany bundle. | Tak. |
| 05 — hierarchia wkładu, kompresja i podpisy Figures 1/4/5 | `manuscript`; zmienione podpisy/bundle wpływają na `display-semantic-inventory` i `display-index`. | Tak. |
| 06 — reader-facing Table S6, Figure S2 i Figure 5 | `supplementary-material`, `display-inventory-contract` (jeśli zmieni się kontrakt inventory), `display-semantic-inventory`, `display-index`; zmiana builderów także jest związana przez semantic manifest. Same pełne wyniki sensitivity mogą pozostać bez zmian. | Tak. |
| Późniejsze dane autorów i eksperymentu | `manuscript` oraz, zależnie od zakresu, `experimental-metadata-manifest`. | Tak, ale poza obecną mapą. |

Sam ticket 07 zmienia wyłącznie tracker i artefakt audytu; nie zmienia żadnego
z 23 komponentów i nie wymaga teraz rebaseline.

## Wymagany fail-closed przebieg po implementacji

1. Zachować bieżący `fingerprint.json` i jego digest jako immutable punkt
   odniesienia; nie odświeżać go podczas zmian 03–06.
2. Zaimplementować cały zatwierdzony zakres treści i display items, odbudować
   właściwe bundle oraz semantic inventory i przejrzeć ich jawny diff.
3. Uruchomić targeted tests/builders. Stary fingerprint powinien odrzucić
   zmienione komponenty; przypadkowe przejście oznacza brak pokrycia kontraktu.
4. Przed rebaseline zamknąć lukę LF/CRLF i potwierdzić, że identyczny build
   powstaje z czystego checkoutu z kompletem zadeklarowanych lokalnych wejść.
5. Uzyskać checkpoint akceptacji treści i claim boundary. Dopiero wtedy wykonać
   jednorazowy, jawnie autoryzowany rebaseline.
6. Obecne `freeze_scientific_core --build` odmawia zastąpienia istniejącego
   fingerprintu. Ticket 08/09 musi więc zdefiniować wersjonowany replacement
   workflow; ręczne usunięcie fingerprintu nie jest zaakceptowaną procedurą.
7. Po utworzeniu nowego fingerprintu uruchomić pełne
   `reconstruct_publication --clean --validate`, osobne `--validate-only` oraz
   drugi run z czystego checkoutu. Wszystkie muszą przejść bez diffu artefaktów.

Reviewer-access release candidate pozostaje poza tym ticketem i nie został
zbudowany.
