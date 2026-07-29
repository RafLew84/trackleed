**Table S5. Controlled-translation outcomes by independently initialized source trial.** Each deterministic six-image trial contains a seed and five non-seed steps on the same closed path of known image translation, yielding 20 spot–step and 10 pair–step evaluation opportunities per method. Proposal coverage is available native proposals / 20; for the oracle control it is constructed-ROI availability / 20. Measurement coverage is accepted common Gaussian measurements / 20. Median, 95th-percentile, and maximum displacement errors are conditional on accepted Gaussian measurements. Pair coverage is jointly valid pair measurements / 10, and maximum absolute ΔD/D is conditional on jointly valid pair measurements and reported in percentage points. The four separately initialized source trials from one Cu(111) series are descriptive strata, not four independent acquisitions. The oracle ROI is a measurement control, not a tracker. No inferential interval, composite score, or overall ranking is implied.

| Trial | T [°C] | Method | Proposal / constructed-ROI coverage | Gaussian measurement coverage | Conditional median / P95 / maximum displacement error [px] | Pair coverage | Maximum absolute ΔD/D [pp] |
|---|---:|---|---:|---:|---:|---:|---:|
| initial low-temperature state | 10 | LocoTrack | 20/20 | 20/20 | 0.0031 / 0.0044 / 0.0054 | 10/10 | 0.000181 |
| initial low-temperature state | 10 | OSTrack | 20/20 | 20/20 | 0.0031 / 0.0046 / 0.0061 | 10/10 | 0.000290 |
| initial low-temperature state | 10 | DAM4SAM | 20/20 | 20/20 | 0.0037 / 0.0084 / 0.0094 | 10/10 | 0.000201 |
| initial low-temperature state | 10 | Lucas–Kanade | 20/20 | 20/20 | 0.0036 / 0.0058 / 0.0062 | 10/10 | 0.000188 |
| initial low-temperature state | 10 | EasyLEED | 20/20 | 20/20 | 0.0695 / 0.3250 / 0.4187 | 10/10 | 0.016820 |
| initial low-temperature state | 10 | Oracle ROI control | 20/20 | 20/20 | 0.0036 / 0.0058 / 0.0062 | 10/10 | 0.000188 |
| 800 °C plateau onset | 800 | LocoTrack | 20/20 | 20/20 | 0.0106 / 0.0141 / 0.0159 | 10/10 | 0.001229 |
| 800 °C plateau onset | 800 | OSTrack | 20/20 | 20/20 | 0.0121 / 0.0186 / 0.0244 | 10/10 | 0.001952 |
| 800 °C plateau onset | 800 | DAM4SAM | 20/20 | 20/20 | 0.0110 / 0.0261 / 0.0261 | 10/10 | 0.001869 |
| 800 °C plateau onset | 800 | Lucas–Kanade | 18/20 | 18/20 | 0.0099 / 0.0136 / 0.0140 | 8/10 | 0.001042 |
| 800 °C plateau onset | 800 | EasyLEED | 20/20 | 17/20 | 0.0176 / 0.3880 / 0.4066 | 7/10 | 0.047021 |
| 800 °C plateau onset | 800 | Oracle ROI control | 20/20 | 20/20 | 0.0104 / 0.0137 / 0.0140 | 10/10 | 0.001147 |
| post-plateau contraction minimum | 797 | LocoTrack | 20/20 | 20/20 | 0.0107 / 0.0130 / 0.0136 | 10/10 | 0.000794 |
| post-plateau contraction minimum | 797 | OSTrack | 20/20 | 20/20 | 0.0125 / 0.0170 / 0.0211 | 10/10 | 0.001397 |
| post-plateau contraction minimum | 797 | DAM4SAM | 20/20 | 20/20 | 0.0109 / 0.0154 / 0.0159 | 10/10 | 0.000934 |
| post-plateau contraction minimum | 797 | Lucas–Kanade | 18/20 | 18/20 | 0.0114 / 0.0129 / 0.0134 | 8/10 | 0.000784 |
| post-plateau contraction minimum | 797 | EasyLEED | 20/20 | 17/20 | 0.0139 / 0.3598 / 0.5412 | 7/10 | 0.062943 |
| post-plateau contraction minimum | 797 | Oracle ROI control | 20/20 | 20/20 | 0.0106 / 0.0128 / 0.0134 | 10/10 | 0.000784 |
| final cooled state | 92 | LocoTrack | 20/20 | 20/20 | 0.0033 / 0.0056 / 0.0057 | 10/10 | 0.000130 |
| final cooled state | 92 | OSTrack | 20/20 | 20/20 | 0.0031 / 0.0060 / 0.0063 | 10/10 | 0.000182 |
| final cooled state | 92 | DAM4SAM | 20/20 | 20/20 | 0.0039 / 0.0087 / 0.0109 | 10/10 | 0.000201 |
| final cooled state | 92 | Lucas–Kanade | 20/20 | 20/20 | 0.0039 / 0.0084 / 0.0102 | 10/10 | 0.000161 |
| final cooled state | 92 | EasyLEED | 20/20 | 20/20 | 0.0520 / 0.2251 / 0.3810 | 10/10 | 0.017988 |
| final cooled state | 92 | Oracle ROI control | 20/20 | 20/20 | 0.0039 / 0.0084 / 0.0102 | 10/10 | 0.000161 |
