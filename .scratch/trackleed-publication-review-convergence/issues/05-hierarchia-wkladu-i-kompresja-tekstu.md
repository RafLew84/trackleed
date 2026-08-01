# Hierarchia wkładu i kompresja tekstu

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `prototype`
Label: `wayfinder:prototype`
Mode: `HITL`
Status: `resolved`
Blocked by: 01, 03, 04

## Question

Jaki minimalny zestaw zmian w Abstract, Objective, nagłówku 1.1, Discussion i
podpisach figur przywraca zamrożoną hierarchię wkładu oraz usuwa powtórzenia
bez osłabienia claim boundaries i denominator-aware argumentu?

Prototyp ma utrzymać kolejność: common measurement contract,
denominator-aware assessment, coverage/recovery/consistency, a dopiero potem
Gaussian capture-region effect. Ma porównać wariant abstraktu w docelowym
zakresie 230–250 słów, bezpieczniejsze `persistent spot association` oraz
skrócone podpisy Figures 1, 4 i 5. Decyzja nie może usuwać sensitivity
zatwierdzonej w [Integracji sensitivity z narracją i granicą
claims](03-integracja-sensitivity-z-narracja-i-claims.md).

## Comments

### 2026-08-02 — prototyp hierarchii i kompresji

Porównanie dwóch Abstractów, rekomendowany rewrite Objective/contributions,
bezpieczniejszy nagłówek 1.1, selektywną mapę kompresji Discussion oraz skrócone
podpisy Figures 1, 4 i 5 zapisano w
[artefakcie prototypu](../artifacts/05-contribution-hierarchy-compression-prototype.md).
Prototyp zachowuje zatwierdzone sensitivity, claim boundaries i denominator
semantics. Ticket czeka na decyzję HITL.

## Answer

Użytkownik zatwierdził pełny wariant A z
[artefaktu prototypu](../artifacts/05-contribution-hierarchy-compression-prototype.md).
Reader-facing hierarchia wkładu ma prowadzić od common measurement contract,
przez denominator-aware assessment oraz coverage/recovery/geometry consistency,
do Gaussian capture-region effect jako mechanizmu wspierającego.

Zatwierdzony Abstract contribution-first ma 244 słowa według jawnego licznika,
zachowuje fixed denominators, method-specific surviving `n`, case boundaries i
15-wariantową sensitivity oraz nie odwołuje się do tabel, figur ani
Supplementary Information. Objective i lista contributions otrzymują tę samą
hierarchię, a nagłówek 1.1 zmienia się na `Quantitative LEED requires
persistent spot association`.

Discussion podlega wyłącznie selektywnej kompresji powtórzonych wyników w
5.1–5.4. Nie usuwa się odrębnych caveats, zatwierdzonego akapitu sensitivity ani
nowego akapitu Limitations z ticketu 03. Podpisy Figures 1, 4 i 5 zostają
skrócone odpowiednio z 249 do 202, z 271 do 216 i z 232 do 189 słów, przy
zachowaniu denominatorów, conditioning rules i claim boundaries. Na tym etapie
nie zmienia się właściwego manuskryptu ani scientific-core fingerprint.
