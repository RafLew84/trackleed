# I13 — page-level render audit

## Wynik

Ponowny audit zakończył się statusem `page-level-render-accepted` dla czterech
kanonicznych rendycji:

- main DOCX/PDF: 36 stron,
- Supplementary Information DOCX/PDF: 64 strony,
- 100 unikalnych obrazów stron i 200 decyzji per-rendition.

Użytkownik jawnie zaakceptował poprawioną wersję w checkpoint HITL 2026-08-02.
Decyzja, provenance i wynikowy manifest znajdują się w
`manuscript/publication_package/page_audit/`.

## Powtarzalność i provenance

Builder usuwa absolutną ścieżkę checkoutu z właściwości `bibliography` w DOCX.
Dwa niezależne checkouty Windows zwróciły identyczne finalne hashe:

- main DOCX: `6c182d4a4f3edc21745a327ee5f24b7dac53987ad444d96d4cc091ec489c5681`,
- main PDF: `f56adf886d8932dac77a4c091dd71d6efb18bac62427d8f8496d14d94de4e299`,
- SI DOCX: `19e63c36740fc6b51a94097b273f7d32ab58f44b26e1eb2bf9fd4984068a871e`,
- SI PDF: `5051e79e6f754ed53c8832e56a9244965de672bc9eb6d268b74dc541d7c63da7`.

Niezależne eksporty Word `ExportAsFixedFormat` po deterministycznej normalizacji
są bajtowo identyczne z kanonicznymi PDF. To uzasadnia współdzielenie page setu
między DOCX i PDF osobno dla main oraz SI. Pinned pipeline: Pandoc 3.8.2.1,
Microsoft Word 16.0 build 16.0.20228, pypdf 6.2.0, PyMuPDF 1.28.0 przy 150 DPI
i Pillow 11.2.1.

## Kontrola stron

Każda finalna strona została obejrzana na opisanych arkuszach kontaktowych.
Pierwszy przebieg ujawnił dzielone podpisy głównych Figur 1, 3, 4 i 5 oraz dwie
puste strony. Po pełnych rebuildach:

- Figura 1 ma czytelny obraz i pełny podpis na kolejnej stronie;
- Figura 2 zachowuje obraz i podpis na jednej stronie;
- Figury 3–5 mają czytelny obraz i pełny, niepodzielony podpis na kolejnej
  stronie; Figure 5 zachowuje kontraktowane 6.25 in i minimalny efektywny font
  6.076 pt;
- usunięto puste strony main 29 i SI 58;
- powtórny audyt objął wszystkie strony zmienionych dokumentów.

Kontrole automatyczne potwierdziły brak pustych stron, clippingu, obiektów poza
MediaBox i replacement glyphs; wszystkie fonty są osadzone, a minimalna
efektywna rozdzielczość obrazu wynosi 199.7 DPI. Main zawiera dokładnie cztery
autoryzowane kategorie placeholderów, SI nie zawiera żadnego. Numeracja,
podpisy, matematyka, tabele, figury i cross-references przeszły audit.

## Walidacja

- kontrakty buildera i audytu: `11 passed`, Ruff passed;
- `python -m manuscript.publication_package.audit_page_renders --verify`:
  4 rendycje, 200 rekordów stron, accepted;
- adekwatny pełny `trackleed/tests`: `263 passed, 6 failed`; sześć błędów to
  niezmieniony baseline `PermissionError` dla niedostępnego, user-owned
  `manuscript/scientific_panel`.
