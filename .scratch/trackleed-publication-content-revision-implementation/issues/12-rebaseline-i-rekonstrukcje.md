# Autoryzowany rebaseline i dwie rekonstrukcje

Parent: [Mapa implementacji rewizji treści publikacji TrackLEED](../map.md)
Type: `task`
Label: `ready-for-agent`
Status: `open`
Blocked by: 11

## Question

Czy autoryzowany successor fingerprint jest odtwarzalny bez driftu w pinned i
clean-Windows reconstruction?

## Objective

Wykonać jednorazowy autoryzowany replacement i udowodnić powtarzalność
successor fingerprintu.

## Acceptance

- Replacement używa rekordu akceptacji z ticketu 10 i exact commit.
- `freeze_scientific_core --validate-only` przechodzi.
- Pełny `reconstruct_publication --clean --validate` przechodzi w pinned
  CPU/headless environment.
- Drugi pełny run przechodzi z clean Windows checkout z wszystkimi
  manifest-declared local inputs; digest jest identyczny i nie ma driftu.
