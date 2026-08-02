# Przenośny kontrakt LF dla hash-bound text

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `resolved`
Blocked by: 01

## Question

Czy wszystkie hash-bound pliki tekstowe są odtwarzane byte-identycznie z
czystego checkoutu Windows dzięki jednemu kontraktowi LF?

## Objective

Test-first ustanowić LF jako jednoznaczny kontrakt hash-bound text przez
`.gitattributes` i deterministyczne writery (`newline="\n"` albo byte writes).

## Acceptance

- Test odtwarza ujawniony drift LF/CRLF i przed zmianą zawodzi.
- Wszystkie hash-bound text paths mają jednoznaczną politykę checkout/write.
- Clean Windows checkout z wymaganymi lokalnymi wejściami nie zgłasza mismatch
  końców linii i generuje byte-identical scientific artifacts.
- Nie wykonuje się rebaseline i nie zmienia starego fingerprintu.

## Answer

Tak. Root `.gitattributes` ustanawia LF dla repozytoryjnego tekstu, jawne
wyjątki `binary` chronią uwierzytelnione pakiety źródłowe, a wszystkie writery
rekonstrukcji deklarują `newline="\n"`. Kontrolowany migrator normalizuje tekst
i propaguje wyłącznie dozwolone zmiany hashy, zachowując byte-preserved dowody,
ich referencje oraz stary fingerprint.

Czysty checkout Windows z `core.autocrlf=true` i wymaganymi lokalnymi wejściami
przeszedł kontrakty 20 reader-facing bundles i pełną rekonstrukcję bez mismatchu
LF/CRLF. Rekonstrukcja zatrzymała się dopiero na oczekiwanym
`Scientific-core fingerprint differs from the frozen baseline`, ponieważ I02
nie wykonuje rebaseline.

Dowody: [artifact 02](../artifacts/02-publication-line-ending-contract.md).

## Comments

- 2026-08-02: RED odtworzył brak `eol=lf`, niejawne `write_text` oraz kolizje
  digestów między normalizowanym tekstem i byte-preserved dowodem.
- 2026-08-02: GREEN potwierdzono w env `netcat`, w czystym snapshotcie Windows
  oraz pełnym fail-closed reconstruction path.
