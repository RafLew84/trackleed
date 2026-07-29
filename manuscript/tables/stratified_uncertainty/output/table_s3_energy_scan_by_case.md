# Table S3. Stratified Ag(111) energy-scan outcomes

Each row is one frozen scan-pair-method case. Conditional residuals are shown only with their surviving pair-frame count.

| Scan | Pair | Method | Coverage | Gap recovery | Geometry inconsistency | Conditional median residual [px] | Conditional n |
|---|---|---|---:|---:|---:|---:|---:|
| Scan A | pair 1 | LocoTrack | 44/44 | 5/5 | 9/57 | 1.983 | 18 |
| Scan A | pair 1 | OSTrack | 44/44 | 5/5 | 9/58 | 1.822 | 19 |
| Scan A | pair 1 | DAM4SAM | 44/44 | 5/5 | 9/58 | 1.825 | 19 |
| Scan A | pair 1 | Lucas-Kanade | 43/44 | 5/5 | 5/49 | 1.983 | 16 |
| Scan A | pair 1 | EasyLEED | 42/44 | 5/5 | 0/42 | 1.526 | 15 |
| Scan A | pair 2 | LocoTrack | 48/50 | 7/7 | 7/57 | 1.561 | 19 |
| Scan A | pair 2 | OSTrack | 49/50 | 6/7 | 7/58 | 1.647 | 20 |
| Scan A | pair 2 | DAM4SAM | 49/50 | 7/7 | 7/58 | 1.650 | 20 |
| Scan A | pair 2 | Lucas-Kanade | 40/50 | 4/7 | 0/40 | 1.560 | 14 |
| Scan A | pair 2 | EasyLEED | 48/50 | 5/7 | 0/48 | 1.650 | 18 |
| Scan B | pair 1 | LocoTrack | 58/60 | 5/6 | 10/78 | 1.336 | 26 |
| Scan B | pair 1 | OSTrack | 57/60 | 5/6 | 10/77 | 1.166 | 25 |
| Scan B | pair 1 | DAM4SAM | 57/60 | 5/6 | 10/77 | 1.164 | 25 |
| Scan B | pair 1 | Lucas-Kanade | 47/60 | 4/6 | 9/59 | 1.131 | 15 |
| Scan B | pair 1 | EasyLEED | 57/60 | 6/6 | 0/58 | 1.131 | 23 |
| Scan B | pair 2 | LocoTrack | 67/67 | 6/6 | 10/80 | 2.068 | 28 |
| Scan B | pair 2 | OSTrack | 66/67 | 5/6 | 8/76 | 1.990 | 26 |
| Scan B | pair 2 | DAM4SAM | 66/67 | 6/6 | 10/79 | 2.035 | 27 |
| Scan B | pair 2 | Lucas-Kanade | 58/67 | 4/6 | 0/58 | 2.034 | 23 |
| Scan B | pair 2 | EasyLEED | 67/67 | 6/6 | 4/74 | 2.065 | 28 |

## Leave-one-case-out sensitivity

One frozen scan-pair case is omitted at a time. The analysis recomputes count-based outcomes only; it is a sensitivity check, not a confidence interval. Case medians are not reaggregated as pooled residuals.

| Method | Coverage range | Gap-recovery range | Geometry-inconsistency range |
|---|---:|---:|---:|
| LocoTrack | 97.4–98.8% | 94.1–100.0% | 12.6–13.5% |
| OSTrack | 97.2–98.8% | 84.2–88.9% | 11.8–13.5% |
| DAM4SAM | 97.2–98.8% | 94.1–100.0% | 12.6–13.6% |
| Lucas-Kanade | 81.9–87.6% | 63.2–76.5% | 3.4–9.5% |
| EasyLEED | 95.5–97.5% | 88.9–100.0% | 0.0–2.4% |
