# Przenośny kontrakt LF dla hash-bound text

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
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
