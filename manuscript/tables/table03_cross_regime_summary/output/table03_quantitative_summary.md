**Table 3. Quantitative summary across evidence regimes.** Each block answers a distinct validation question and retains its original denominators. Translation displacement errors are conditional on accepted Gaussian measurements, with coverage shown first. The energy-scan residual is conditional on jointly branch-consistent pair frames and is shown with its n. Cu endpoint deviations are agreement with the framewise five-method median, not absolute accuracy. DAM4SAM used sam21pp-L for Cu(111) and controlled translations, and sam21pp-B for the energy scans. No composite score or rank is calculated.

**Controlled translation**

| Method | Proposal coverage | Measurement coverage | Median / P95 displacement error [px] | Pair coverage | Maximum absolute Delta D/D [pp] |
|---|---|---|---|---|---|
| LocoTrack | 80/80 (100.00%) | 80/80 (100.00%) | 0.0043 / 0.0130 | 40/40 (100.00%) | 0.001229 |
| OSTrack | 80/80 (100.00%) | 80/80 (100.00%) | 0.0048 / 0.0168 | 40/40 (100.00%) | 0.001952 |
| DAM4SAM | 80/80 (100.00%) | 80/80 (100.00%) | 0.0060 / 0.0154 | 40/40 (100.00%) | 0.001869 |
| Lucas–Kanade | 76/80 (95.00%) | 76/80 (95.00%) | 0.0050 / 0.0127 | 36/40 (90.00%) | 0.001042 |
| EasyLEED | 80/80 (100.00%) | 74/80 (92.50%) | 0.0265 / 0.3818 | 34/40 (85.00%) | 0.062943 |

**Cu(111) positive control**

| Method | Gaussian measurement coverage | Valid endpoints (pair 1-6; pair 3-5) | Median / maximum endpoint deviation [pp] |
|---|---|---|---|
| LocoTrack | 2804/2804 (100.00%) | 701/701; 701/701 | 0.000004 / 0.000299 |
| OSTrack | 2804/2804 (100.00%) | 701/701; 701/701 | 0.000134 / 0.001752 |
| DAM4SAM | 2804/2804 (100.00%) | 701/701; 701/701 | 0.000242 / 0.001564 |
| Lucas–Kanade | 2804/2804 (100.00%) | 701/701; 701/701 | 0.000037 / 0.000462 |
| EasyLEED | 2804/2804 (100.00%) | 701/701; 701/701 | 0.000256 / 0.000949 |

**Independent Ag(111) energy-scan stress test**

| Method | Branch-consistent measurement coverage | Post-gap identity recovery | Geometry-inconsistent continuation | Conditional median pair-distance residual [px] (n) |
|---|---|---|---|---|
| LocoTrack | 217/221 (98.19%) | 23/24 (95.83%) | 36/272 (13.24%) | 1.837 (n=91) |
| OSTrack | 216/221 (97.74%) | 21/24 (87.50%) | 34/269 (12.64%) | 1.770 (n=90) |
| DAM4SAM | 216/221 (97.74%) | 23/24 (95.83%) | 36/272 (13.24%) | 1.788 (n=91) |
| Lucas–Kanade | 188/221 (85.07%) | 17/24 (70.83%) | 14/206 (6.80%) | 1.764 (n=68) |
| EasyLEED | 214/221 (96.83%) | 22/24 (91.67%) | 4/222 (1.80%) | 1.737 (n=84) |
