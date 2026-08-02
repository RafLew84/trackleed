# I11 — wersjonowany replacement workflow fingerprintu

Status: completed
Date: 2026-08-02

## Publiczny command

`scripts/replace_scientific_core.py` udostępnia jeden fail-closed workflow z
wymaganymi argumentami `--expected-predecessor`, `--decision-record`,
`--implementation-commit` i `--authorization`. Opcjonalny `--check-only`
wykonuje pełny preflight i wylicza successor bez tworzenia historii ani zmiany
fingerprintu.

Kanoniczny `replacement_authorization.json` wiąże:

- predecessor
  `859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff`;
- zaakceptowany implementation commit
  `38a0e6aa668c4a7751fb6661e94af6d58bea92bc`;
- decision record I10 wraz z SHA-256;
- reviewed report I09 wraz z SHA-256;
- dokładnie 23 komponenty i dokładnie 13 zatwierdzonych changed IDs;
- uzasadnienie replacementu.

Command akceptuje wyłącznie tę kanoniczną ścieżkę authorization. Sprawdza
pełny 40-znakowy commit i wymaga, by był przodkiem HEAD, a wszystkie rekordy i
komponenty muszą być śledzone oraz czyste.

## Fail-closed validation

Przed jakimkolwiek zapisem workflow sprawdza integralność logicznego
fingerprintu poprzednika, hash kontraktu, decision/authorization/report hashes,
kompletność i tożsamość wszystkich 23 komponentów, current/frozen hashes,
dokładny changed set oraz stan Git scientific inputs. Dirty lub untracked input
w `manuscript/` albo wskazanych ścieżkach naukowych zatrzymuje operację;
niekanoniczne review DOCX i przyszła historia replacementu są jawnie wyłączone.

Po poprawnym preflight command buduje successor do pliku tymczasowego w tym
samym katalogu i ponownie porównuje jego 23 hashe z reviewed diffem. W trybie
replacement:

1. atomowo tworzy niezmienną kopię poprzednika w
   `history/fingerprints/<predecessor>.json`;
2. atomowo tworzy rekord w `history/replacements/<successor>.json` z pełnym
   23-component diffem, predecessor/successor linkiem, autoryzacją, commitem,
   powodem i runtime;
3. dopiero na końcu atomowo podmienia `fingerprint.json` przez `os.replace`.

Konflikt istniejącej historii lub zmiana poprzednika podczas preflight blokuje
replacement przed podmianą fingerprintu.

## TDD i weryfikacja

Uzgodnionym seamem był publiczny CLI uruchamiany na izolowanych repozytoriach
Git. Cykle red/green objęły wymagane argumenty, udany 23-component replacement,
arbitrary dirty manuscript input, kanoniczną autoryzację i bezmutacyjny
`--check-only`. Failure-mode tests obejmują też błędny predecessor i commit,
zmieniony decision record, niepełne coverage oraz konflikt historii.

- I11 CLI tests: 10 passed.
- I11 + fingerprint + I09 integration tests: 19 passed.
- Ruff i `compileall`: passed.
- Pełny `trackleed/tests`: 263 passed, 6 failed. Wszystkie sześć zastanych
  failures wynika wyłącznie z `PermissionError` dla niedostępnego, user-owned
  `manuscript/scientific_panel`.

Clean-clone `--check-only` ze staged snapshotem przeszedł dla wszystkich 23
komponentów, zwrócił status `authorized` i wyliczył przewidywany successor
`66b7c59994fe5369ed02f0c8c04e4f924823a52cc423ab02a66a84302b151a3e`.
SHA-256 pliku poprzednika przed i po był identyczny.

## Code review

- Standards: początkowy finding o zduplikowanej canonicalizacji poprzednika
  zamknięto przez publiczny, współdzielony helper integralności I09 bez zmiany
  hash-bound `scientific_core.py`; finalnie 0 actionable findings.
- Spec: 0 actionable findings; niezależna kontrola wszystkich 23 bieżących
  hashy wykazała 0 mismatches, a fingerprint, contract i reviewed report
  pozostały niezmienione.

## Granica ticketu

I11 nie wykonał rebaseline. Produkcyjny `fingerprint.json` zachowuje SHA-256
pliku `3b336ac952ba5c6e87e9bbc53853921ee0b3e1a68c3889c2eeac2fa6f2d01132`.
Rzeczywiste wykonanie replacementu, commit historii i dwie rekonstrukcje należą
do I12. Nie zmieniono user-owned plików `manuscript/scientific_panel`, tabel,
`trackleed/io` ani roboczych artefaktów review.
