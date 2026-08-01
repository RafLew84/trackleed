# Semantyka denominatorów w main text

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `grilling`
Label: `wayfinder:grilling`
Mode: `HITL`
Status: `resolved`
Blocked by: 01

## Question

Jak reader-facing main text ma jednoznacznie rozdzielać całkowite stany
translacji od 80 displaced-spot i 40 displaced-pair evaluation opportunities
oraz fixed opportunity sets od method-specific surviving n dla conditional
residuals?

Na podstawie [Audytu uwag względem źródeł i wygenerowanych
DOCX](01-audyt-uwag-wzgledem-zrodel-i-docx.md) użytkownik ma zatwierdzić
ostateczną semantykę Table 1, zdania w Conclusions i powiązanych odwołań, tak
aby nie zmienić obliczeń ani nie zasugerować, że conditional residual ma stały
mianownik.

## Comments

### 2026-08-02 — decyzja 1: Table 1

Użytkownik zatwierdził rozdzielenie `controlled-translation state inventory`
od `controlled-translation evaluation opportunity set`. Reader-facing Table 1
ma podawać 24 digital image states jako cztery triale × (jeden seed + pięć
displaced states), natomiast kolumna Opportunity set ma podawać 80
displaced-spot i 40 displaced-pair opportunities per method po wyłączeniu
czterech seed states. Wartości 96/48 pozostają wyłącznie liczbami inwentarzowymi
przed seed exclusion i nie są mianownikami wyników.

### 2026-08-02 — decyzja 2: Conclusions

Użytkownik zatwierdził zastąpienie błędnego przypisania conditional residual do
fixed denominator zdaniem: `Conditional residuals must therefore be reported
with their method-specific surviving n, alongside coverage and recovery
evaluated on fixed opportunity sets and the corresponding
geometry-consistency outcomes.` Conditional surviving `n` nie jest fixed
evaluation opportunity set.

### 2026-08-02 — decyzja 3: related references and Abstract

Reader-facing main text uses `x/N` for fixed-opportunity outcomes and labels
conditional statistics explicitly, with their method-specific surviving `n`
where needed for interpretation. The Abstract must remain self-contained: it
must not refer to tables, figures, or Supplementary Information. Consequently,
an Abstract-level conditional result cannot delegate its denominator semantics
to Table S2 or Table S7; the relevant surviving `n` must be stated in the
Abstract or the quantitative statement must be rewritten so it is not
misleading without that `n`.

## Answer

Reader-facing manuscript must distinguish the complete controlled-translation
inventory from the result opportunity sets. Table 1 will report `24 digital
image states: four source trials × (one seed state + five displaced states)`
and, separately, `80 displaced-spot and 40 displaced-pair evaluation
opportunities per method after exclusion of four seed states`. The inventory
counts 96 spot-state and 48 pair-state records include seeds and are not result
denominators.

Fixed-opportunity outcomes must be written as `x/N`. Conditional statistics
must be identified as conditional and accompanied by their outcome- and
method-specific surviving `n` where that `n` is needed to interpret the value;
the fixed opportunity set must never be presented as the denominator of a
conditional residual. Conclusions will state: `Conditional residuals must
therefore be reported with their method-specific surviving n, alongside
coverage and recovery evaluated on fixed opportunity sets and the corresponding
geometry-consistency outcomes.`

The Abstract must be self-contained and must not refer to tables, figures, or
Supplementary Information. An Abstract-level conditional result therefore
states the relevant surviving `n` directly or is rewritten so it remains
unambiguous without that `n`. This decision changes reader-facing semantics
only and does not change any calculation or evaluation set. The complete
implementation contract and edge cases are recorded in the
[decision artifact](../artifacts/04-denominator-semantics-decision.md).
