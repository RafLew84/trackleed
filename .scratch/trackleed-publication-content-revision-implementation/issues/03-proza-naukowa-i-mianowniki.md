# Proza naukowa i semantyka mianowników

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 01

## Question

Czy kanoniczny manuskrypt wdraża zatwierdzoną narrację, sensitivity i
semantykę mianowników bez przekroczenia claim boundary?

## Objective

Wdrożyć w kanonicznym `manuscript.md` dokładne zatwierdzone teksty z artefaktów
03–05: 244-word Abstract A, hierarchię Objective/contributions, `persistent
spot association`, Table 1, Results 4.4, przesuniętą numerację, Discussion,
Limitations i Conclusions.

## Acceptance

- Abstract nie odwołuje się do tabel, figur ani Supplementary Information.
- Zachowane są dokładne liczby sensitivity i granice C1/C3/C4.
- `24` opisuje inventory, `80/40` fixed opportunity sets, a conditional
  residual ma jawne method-specific surviving `n`.
- Testy kontraktu manuskryptu sprawdzają tekst, numerację, liczby i zakazane
  claimy; generated output nie jest edytowany ręcznie.

## Answer

Tak. Kanoniczny manuskrypt wdraża zatwierdzoną narrację i sensitivity,
Abstract ma 244 słowa i nie odwołuje się do tabel, figur ani SI, a Table 1
oddziela inventory 24 od fixed opportunity sets 80/40. Conditional residuals
mają jawne method-specific surviving `n`, a testy fail-closed chronią tekst,
numerację, liczby, denominator semantics i zakazane claimy.

## Comments

- Reader-facing Table 1 i jego kopię w manuskrypcie przebudowano przez
  kanoniczny builder bez zmiany frozen source data 96/48.
- Stary scientific-core fingerprint pozostał immutable i zgodnie z planem
  odrzuca zrewidowany content do rebaseline w I12.
- Dowody: [I03 — proza naukowa i semantyka mianowników](../artifacts/03-scientific-prose-denominator-semantics.md).
