**Table S5. Controlled translations by source trial.** Each deterministic six-image trial contains one seed and five evaluation steps on the same closed path, providing 20 spot-step and 10 pair-step fixed opportunities per method. Displacement summaries are conditional on accepted Gaussian measurements; pair invariance is conditional on jointly valid pairs, and both conditional n values are shown. The four source trials are descriptive strata drawn from one Cu(111) series, not four independent acquisitions. The oracle ROI is a measurement control, not a tracker. Known translations assess measurement-chain equivariance and common-estimator behavior, not absolute tracking accuracy and not identity ground truth.

| Trial | T [deg C] | Method | Proposal / constructed-ROI coverage | Gaussian measurement coverage | Conditional median / P95 / max displacement error [px] (n) | Pair coverage | Max absolute Delta D/D [pp] (n) |
|---|---:|---|---:|---:|---:|---:|---:|
| initial low-temperature state | 10 | LocoTrack | 20/20 | 20/20 | 0.0031 / 0.0044 / 0.0054 (n=20) | 10/10 | 0.000181 (n=10) |
| initial low-temperature state | 10 | OSTrack | 20/20 | 20/20 | 0.0031 / 0.0046 / 0.0061 (n=20) | 10/10 | 0.000290 (n=10) |
| initial low-temperature state | 10 | DAM4SAM | 20/20 | 20/20 | 0.0037 / 0.0084 / 0.0094 (n=20) | 10/10 | 0.000201 (n=10) |
| initial low-temperature state | 10 | Lucas-Kanade | 20/20 | 20/20 | 0.0036 / 0.0058 / 0.0062 (n=20) | 10/10 | 0.000188 (n=10) |
| initial low-temperature state | 10 | EasyLEED | 20/20 | 20/20 | 0.0695 / 0.3250 / 0.4187 (n=20) | 10/10 | 0.016820 (n=10) |
| initial low-temperature state | 10 | Oracle ROI control | 20/20 | 20/20 | 0.0036 / 0.0058 / 0.0062 (n=20) | 10/10 | 0.000188 (n=10) |
| 800 deg C plateau onset | 800 | LocoTrack | 20/20 | 20/20 | 0.0106 / 0.0141 / 0.0159 (n=20) | 10/10 | 0.001229 (n=10) |
| 800 deg C plateau onset | 800 | OSTrack | 20/20 | 20/20 | 0.0121 / 0.0186 / 0.0244 (n=20) | 10/10 | 0.001952 (n=10) |
| 800 deg C plateau onset | 800 | DAM4SAM | 20/20 | 20/20 | 0.0110 / 0.0261 / 0.0261 (n=20) | 10/10 | 0.001869 (n=10) |
| 800 deg C plateau onset | 800 | Lucas-Kanade | 18/20 | 18/20 | 0.0099 / 0.0136 / 0.0140 (n=18) | 8/10 | 0.001042 (n=8) |
| 800 deg C plateau onset | 800 | EasyLEED | 20/20 | 17/20 | 0.0176 / 0.3880 / 0.4066 (n=17) | 7/10 | 0.047021 (n=7) |
| 800 deg C plateau onset | 800 | Oracle ROI control | 20/20 | 20/20 | 0.0104 / 0.0137 / 0.0140 (n=20) | 10/10 | 0.001147 (n=10) |
| post-plateau contraction minimum | 797 | LocoTrack | 20/20 | 20/20 | 0.0107 / 0.0130 / 0.0136 (n=20) | 10/10 | 0.000794 (n=10) |
| post-plateau contraction minimum | 797 | OSTrack | 20/20 | 20/20 | 0.0125 / 0.0170 / 0.0211 (n=20) | 10/10 | 0.001397 (n=10) |
| post-plateau contraction minimum | 797 | DAM4SAM | 20/20 | 20/20 | 0.0109 / 0.0154 / 0.0159 (n=20) | 10/10 | 0.000934 (n=10) |
| post-plateau contraction minimum | 797 | Lucas-Kanade | 18/20 | 18/20 | 0.0114 / 0.0129 / 0.0134 (n=18) | 8/10 | 0.000784 (n=8) |
| post-plateau contraction minimum | 797 | EasyLEED | 20/20 | 17/20 | 0.0139 / 0.3598 / 0.5412 (n=17) | 7/10 | 0.062943 (n=7) |
| post-plateau contraction minimum | 797 | Oracle ROI control | 20/20 | 20/20 | 0.0106 / 0.0128 / 0.0134 (n=20) | 10/10 | 0.000784 (n=10) |
| final cooled state | 92 | LocoTrack | 20/20 | 20/20 | 0.0033 / 0.0056 / 0.0057 (n=20) | 10/10 | 0.000130 (n=10) |
| final cooled state | 92 | OSTrack | 20/20 | 20/20 | 0.0031 / 0.0060 / 0.0063 (n=20) | 10/10 | 0.000182 (n=10) |
| final cooled state | 92 | DAM4SAM | 20/20 | 20/20 | 0.0039 / 0.0087 / 0.0109 (n=20) | 10/10 | 0.000201 (n=10) |
| final cooled state | 92 | Lucas-Kanade | 20/20 | 20/20 | 0.0039 / 0.0084 / 0.0102 (n=20) | 10/10 | 0.000161 (n=10) |
| final cooled state | 92 | EasyLEED | 20/20 | 20/20 | 0.0520 / 0.2251 / 0.3810 (n=20) | 10/10 | 0.017988 (n=10) |
| final cooled state | 92 | Oracle ROI control | 20/20 | 20/20 | 0.0039 / 0.0084 / 0.0102 (n=20) | 10/10 | 0.000161 (n=10) |
