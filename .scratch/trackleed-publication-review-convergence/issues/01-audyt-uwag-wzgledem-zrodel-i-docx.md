# Audyt uwag względem źródeł i wygenerowanych DOCX

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `task`
Label: `wayfinder:task`
Mode: `AFK`
Status: `resolved`

## Question

Które uwagi z `publikacja.md` są nadal aktualnymi problemami źródła
naukowego, które wynikają wyłącznie z obecnego eksportu Markdown do DOCX,
które dotyczą reader-facing display, a które należą do jawnie odłożonej
kompletności submission?

Audyt ma porównać każdą uwagę z aktualnym `manuscript/manuscript.md`,
`manuscript/supplementary-material.md`, źródłami tabel i figur, builderami,
manifestami, scientific-core fingerprint oraz dwoma wygenerowanymi dokumentami
review. Wynikiem ma być macierz `przyjąć / przyjąć częściowo / odrzucić /
odroczyć`, z dokładnym source-of-truth i wskazaniem ticketu decyzyjnego, który
ma przejąć każdy aktualny problem. Audyt niczego jeszcze nie poprawia.

## Answer

Audyt został zapisany w [macierzy uwag względem źródeł i dokumentów
review](../artifacts/01-review-source-docx-audit.md). Rozdziela ona 16 uwag na
8 przyjętych, 5 przyjętych częściowo, 1 odrzuconą i 2 odroczone oraz kieruje
aktualne problemy do ticketów 02–07.

Najważniejsze rozstrzygnięcia: sensitivity i semantyka denominatorów są
problemami źródła reader-facing, uszkodzona matematyka jest problemem eksportu
DOCX, gęstość Table S6/Figure S2/Figure 5 jest problemem prezentacji, a autorzy
i opis eksperymentu pozostają świadomie odroczone. Audyt nie zmienia żadnego
artefaktu manuskryptu.
