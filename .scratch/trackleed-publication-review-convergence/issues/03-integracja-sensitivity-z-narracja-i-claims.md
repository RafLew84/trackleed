# Integracja sensitivity z narracją i granicą claims

Parent: [Mapa konwergencji manuskryptu TrackLEED po audycie publikacja.md](../map.md)
Type: `prototype`
Label: `wayfinder:prototype`
Mode: `HITL`
Status: `resolved`
Blocked by: 01

## Question

Jak 15-wariantowa measurement-window and capture-contract sensitivity ma wejść
do Results, Discussion, Limitations i Abstract, aby uczestniczyła w
adjudykacji claims bez zamiany analizy sensitivity w optymalizację parametrów
lub ranking trackerów?

Na podstawie [Audytu uwag względem źródeł i wygenerowanych
DOCX](01-audyt-uwag-wzgledem-zrodel-i-docx.md) prototyp ma przedstawić dokładny
szkielet nowych lub zmienionych akapitów: co zmieniało wartości bezwzględne,
które jakościowe wnioski pozostały stabilne, dlaczego width trigger nie został
uruchomiony oraz które obecne zdania Limitations stają się nieprawdziwe.
Decyzja musi zachować case-based claim boundary i pełne denominatory.

## Comments

### 2026-08-02 — prototyp integracji sensitivity

Dokładny szkielet reader-facing akapitów, ich podstawę liczbową i mapę wpływu
na claims zapisano w
[artefakcie prototypu](../artifacts/03-sensitivity-narrative-prototype.md).
Porównano wariant A (osobna sekcja Results, rekomendowany) i wariant B
(włączenie do istniejącej sekcji 4.3). Oba zachowują fixed denominators
`221/24`, conditional denominators dla inconsistency/residual, case-based claim
boundary oraz zakaz optymalizacji i rankingu. Ticket czeka na decyzję HITL.

## Answer

Użytkownik zatwierdził wariant A: sensitivity otrzymuje osobną sekcję Results
`4.4 Predeclared measurement-window and capture-contract sensitivity`, a
dotychczasowe sekcje 4.4 i 4.5 zostają przesunięte do 4.5 i 4.6. Sekcja używa
trzech zatwierdzonych akapitów z artefaktu: definiuje 15 predeclared variants i
zamrożone opportunity sets `221/24`, raportuje zmienność wartości bezwzględnych
z outcome-specific denominators (`1/222–69/272` dla krańców geometry
inconsistency oraz `n=79–99` dla krańców median residual, przy pełnych zakresach
denominatorów odpowiednio `187–286` i `54–104`) oraz wyjaśnia, dlaczego
width-materiality trigger nie został uruchomiony.

Do Abstract, Discussion i Limitations wchodzą teksty wspólne dla obu wariantów
z [artefaktu prototypu](../artifacts/03-sensitivity-narrative-prototype.md).
Abstract zachowuje regułę samowystarczalności z ticketu 04: zatwierdzone zdanie
nie odwołuje się do tabel, rysunków ani Supplementary Information. Limitations
usuwa nieprawdziwe zdanie, że inne rozmiary ROI nie były testowane, i zastępuje
je rzeczywistym zakresem lokalnej sensitivity.

Granica claims pozostaje case-based. C1 jest ograniczone do tested data regimes
i predeclared contracts, C3 do dwóch zarejestrowanych sekwencji Ag(111) i
czterech strata, a C4 do denominator-aware argumentu zamiast niezmienności
wartości residualu. C2 i C5–C9 pozostają bez zmian. Sensitivity nie służy do
wyboru konfiguracji, rankingu trackerów ani generalizacji absolute tracking
accuracy. Na tym etapie nie zmienia się manuskryptu ani scientific-core
fingerprint.
