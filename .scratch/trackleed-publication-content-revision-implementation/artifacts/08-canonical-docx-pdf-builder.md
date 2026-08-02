# I08 — kanoniczny builder DOCX/PDF i audit matematyki

Status: completed
Date: 2026-08-02

## Zrealizowany zakres

- `manuscript/publication_package/build_publication_package.py` jest jedynym
  kanonicznym builderem main manuscript i Supplementary Information.
- Builder składa aktualną prozę, bibliografię, Tables 1–S7 oraz Figures 1–5 i
  S1–S6, a szerokie tabele umieszcza w natywnych sekcjach landscape DOCX.
- Pandoc używa dokładnie `markdown+tex_math_single_backslash`, `citeproc` i
  `--fail-if-warnings`; PDF powstaje alternatywnym eksportem Microsoft Word.
- Manifest identyfikuje dokładnie cztery kanoniczne artefakty, ich rozmiary,
  SHA-256, source hashes, wersję buildera i trzy jawnie wykluczone review DOCX.
- DOCX oraz PDF są normalizowane deterministycznie bez rasteryzacji wzorów i
  tabel; początkowy manifest fallbacków pozostaje pusty.

## Fail-closed audit matematyki

Audit porównuje dla każdego dokumentu liczbę Pandoc AST `Math`, liczbę i
niepustość natywnych OMath oraz uporządkowaną treść każdej formuły po odczycie
zwrotnym DOCX przez Pandoc. Dopuszczana normalizacja obejmuje wyłącznie
równoważne warianty notacji wprowadzane przez round trip. Manifest zapisuje
zgodne counts i znormalizowane digests: main 64/64, SI 10/10, fallback 0/0.
Conversion warning, brak/pusta/uszkodzona formuła, silent text fallback,
nieallowlistowana damaged-math signature lub hash drift zatrzymują build.

## Weryfikacja

- `--check-reproducible`: dwa rzeczywiste pełne eksporty Pandoc/Word dały
  identyczne SHA-256 czterech kanonicznych DOCX/PDF.
- Targeted contract/OOXML/export tests: 10 passed.
- Manifest `--verify`: passed; Ruff: passed; compileall: passed.
- Wizualnie sprawdzono każdą z 36 stron main i 65 stron SI: bez clippingu,
  kolizji i zniekształceń matematyki; tabele i ryciny są czytelne.
- Pełny `trackleed/tests`: 249 passed, 6 failed; wszystkie zastane failures
  wynikają z `PermissionError` dla niedostępnego `manuscript/scientific_panel`.
- Abstract nie odwołuje się do tabel, rycin ani Supplementary Information.

## Code review

- Standards: trzy początkowe obserwacje rozwiązano przez `DisplaySpec`, jawny
  continuation label i wspólny kontrakt argumentów Pandoca; końcowo 0 findings.
- Spec: dwie początkowe luki zamknięto per-expression auditem i realnym
  podwójnym buildem; końcowo 0 actionable findings.

## Granica zmian

Nie zmieniono user-owned zmian w `trackleed/io`, tabelach review ani
`manuscript/scientific_panel`. Stary scientific-core fingerprint pozostaje
niezmieniony do kontrolowanego rebaseline w I12.
