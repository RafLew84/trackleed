# Kanoniczne podpisy Figures 1, 4 i 5

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 01

## Question

Czy kanoniczne źródła Figures 1, 4 i 5 zawierają zatwierdzone krótsze podpisy
bez utraty denominator semantics i caveats?

## Objective

Wprowadzić zatwierdzone skrócone podpisy z artefaktu 05 w kanonicznych
źródłach Figures 1, 4 i 5 oraz przebudować zależne bundle bez zmiany danych,
paneli lub claim boundary.

## Acceptance

- Podpisy odpowiadają zatwierdzonym wersjom; Figure 5 ma 189 słów.
- Testy chronią wymagane denominator semantics i caveats.
- Semantic inventory i manifesty pochodzą z builderów, nie z ręcznej edycji.

## Answer

Tak. Kanoniczne caption sources zawierają zatwierdzone skrócone wersje,
Figure 5 ma 189 słów, a testy fail-closed chronią dokładne teksty,
denominator semantics, caveats, numerację inventory i cross-reference.
Zależny display index i manifesty zostały przebudowane przez builder bez
zmiany danych lub paneli.

## Comments

- Zatwierdzone podpisy jednoznacznie ustalają review numbering: Figure 3 —
  Ag(111) stress, Figure 4 — SAM2 family, Figure 5 — Cu(111) endpoint.
  Inventory i Results zostały do niego wyrównane, aby caption nie został
  przypisany do niewłaściwych paneli.
- Dowody: [I04 — kanoniczne podpisy figur](../artifacts/04-canonical-figure-captions.md).
