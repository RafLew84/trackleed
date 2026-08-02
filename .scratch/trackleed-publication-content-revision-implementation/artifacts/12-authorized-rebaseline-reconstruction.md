# I12 — autoryzowany rebaseline i dwie rekonstrukcje

Status: completed
Date: 2026-08-02

## Autoryzowany replacement

Replacement wykonano w czystym checkoutcie commitu
`790c6da9a3131b44ce36818f4a4f9274ed0a1faa`. Command otrzymał dokładnie:

- predecessor
  `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`;
- decision record I10
  `.scratch/trackleed-publication-content-revision-implementation/artifacts/10-hitl-content-claim-boundary-decision.md`;
- zaakceptowany implementation commit
  `38a0e6aa668c4a7751fb6661e94af6d58bea92bc`;
- kanoniczną autoryzację
  `manuscript/scientific_core/replacement_authorization.json`.

Workflow zwrócił status `replaced`, zachował poprzednika i utworzył rekord
replacementu dla successora
`66b7c59994fe5369ed02f0c8c04e4f924823a52cc423ab02a66a84302b151a3e`.
Wszystkie trzy przeniesione pliki są bajtowo identyczne z wynikiem workflow:

- successor `fingerprint.json`: SHA-256 pliku
  `be946e963239168ecfbb65c0eb73773e982cf0d0123b62515cc45cdf26375bd3`;
- archiwum poprzednika: SHA-256 pliku
  `6a1634b0ff7fa3ef4732e94c939c007e582b2fb0ca2d35a514734c990381f722`;
- rekord replacementu: SHA-256 pliku
  `c1d7e39f75d7ef92a94da5b5449a5d7a3c8abff8f15131e858d4e40842c8e8c1`.

## Pinned CPU/headless runtime

Oba pełne przebiegi używały środowiska `netcat`, które dokładnie odpowiada
`manuscript/scientific_core/cpu-environment.txt`: Python 3.12.9, Matplotlib
3.10.1, NumPy 2.2.5, pandas 2.3.0, Pillow 11.2.1, scikit-image 0.25.2, SciPy
1.15.2, tifffile 2025.3.30 i XlsxWriter 3.2.9. Ustawiono `MPLBACKEND=Agg`, pusty
`CUDA_VISIBLE_DEVICES` oraz `QT_QPA_PLATFORM=offscreen`.

Do każdego clean checkoutu dostarczono ten sam komplet lokalnych wejść
wymaganych przez manifesty i buildery: cztery katalogi danych oraz zamrożone
pakiety `artifacts` i `prototypes` z mapy physics-method-paper. Łącznie było to
3960 plików i 945071402 bajty. Walidatory rekonstrukcji uwierzytelniły wejścia
przez zapisane hashe; żadnego źródła ani frozen copy nie zmieniono.

## Dwie rekonstrukcje

W pierwszym checkoutcie `freeze_scientific_core --validate-only` przeszedł dla
23 komponentów i successora `66b7c599…`. Następnie pełne
`python -m scripts.reconstruct_publication --clean --validate` zakończyło się
powodzeniem. Wynik obejmuje:

- 4 przypadki, 5 metod, 1620 kandydatów i 1620 ewaluacji centralnego panelu;
- 15 wariantów sensitivity, 4860 kandydatów i 24300 ewaluacji;
- zgodność centralnego wariantu oraz central record z frozen records;
- 20 reader-facing bundles, 160 plików źródłowych i 49 wynikowych;
- 9 claim IDs oraz boundary 4 preserved / 5 narrowed / 0 removed;
- scientific-core status `scientific-baseline-complete`.

Trzy pliki rebaseline’u utrwalono w technicznym commicie checkoutu testowego,
po czym utworzono z niego drugi niezależny clean Windows checkout. Przed drugim
przebiegiem `git status --short` był pusty, a `--validate-only` ponownie zwrócił
ten sam successor. Drugi pełny `--clean --validate` zwrócił dokładnie ten sam
raport. SHA-256 obu raportów JSON jest identyczny:
`d231a6a0ba4b79c6505971b83b34fda45fe43ac5ae3c3994ec3fd64a8c8e0a71`.

Po obu rekonstrukcjach `git status --short` był pusty. Nie ma driftu artefaktów,
prozy, danych, tabel ani figur; wersjonowaną zmianą I12 są wyłącznie successor,
append-only archiwum poprzednika i append-only rekord replacementu.

## Testy i review

I12 jest wykonaniem zatwierdzonego workflow, a nie nowym seamem kodu, dlatego
nie dodaje odrębnego cyklu TDD. Publiczny command i wszystkie failure modes
powstały test-first w I11. Pierwszy pełny suite ujawnił trzy testy I09 zależne
od starego live `fingerprint.json`. Istniejący publiczny seam został naprawiony
w cyklu red/green: walidator sprawdza integralność bieżącego successora, po czym
odtwarza pre-rebaseline review przeciwko dokładnemu append-only archiwum
poprzednika. Targeted suite: 4 passed; Ruff: passed.

Końcowy pełny `trackleed/tests`: 263 passed, 6 failed. Wszystkie sześć zastanych
failures wynika wyłącznie z `PermissionError` / braku dostępu do user-owned
`manuscript/scientific_panel`; nie ma nowej regresji.

Niezależny code review:

- Standards: jeden początkowy low finding o zdublowanej walidacji integralności
  zamknięto uproszczeniem gałęzi; finalny re-review: 0 findings.
- Spec: 0 findings; potwierdzono exact authorization chain, 23-component diff,
  append-only historię, dwa clean runs, identyczny digest i zero driftu.
