# Supplementary material

This working supplement contains deterministic display items linked from the
main manuscript. Internal archive identifiers and machine-specific paths are
kept in committed provenance records rather than reader-facing tables.

## Table S1 — software, models, and execution environments

<!-- BEGIN TABLE S1 -->
**Table S1. Software versions, revisions, frozen artifacts, configuration hashes, and retained execution environments.** Software and model identities and the run-frozen records in blocks A and B come from the completed execution contracts. Where a method-level configuration was not frozen during a run, block B instead reports a post-run normalized record derived from the archived executed source; the status column distinguishes these cases. The package inventories in block C were captured on 2026-07-29 after the completed runs from the retained local environments; they document the preserved runtimes, but this record does not establish bitwise reproducibility of GPU inference. The retained execution host was `Microsoft Windows NT 10.0.26200.0, 64-bit` with an NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, driver 610.62).

**A. Software and model identity**

| Component | Study role/regime | Software version | Upstream revision | Configuration/variant | Frozen artifact | Artifact type | Artifact SHA-256 | Environment |
|---|---|---|---|---|---|---|---|---|
| TrackLEED common measurement | All regimes and reported positions | Source files identified by SHA-256 | not recorded separately | 48 × 48 raw-image ROI; two-dimensional elliptical-Gaussian estimator | peak_analyzer.py; common-measurement adapter | implementation source | 5d6af3da893300f4f8bb36196b73e7ad1584566d1f12f5db9a93bf2bb7013416; c4f41ad5debdd5a6e932345553e3127570d83da08045682b8a9b3940efd55364 | netcat |
| LocoTrack | Main five-method matrix | not tagged | 3385a233a270160b253a69050c6a3934527f7775 | Base, offline mode | locotrack_base.ckpt | model checkpoint | 8a5adbaeb610d1f06adfbc7c9076b66f727d674c0fd1d668890201cf3339736c | locotrack-pytorch |
| OSTrack | Main five-method matrix | not tagged | 33b5e12586216b7fd0e95d255bd01ba44cbec759 | ViT-B/256 MAE-CE, epoch 300 | OSTrack_ep0300.pth.tar | model checkpoint | 30c2147dd5799c7d78f5c114bf9ebaf00fa69638568865e6ade3d8cb04ea380f | ostrack |
| DAM4SAM Large | Cu(111) and controlled translations | not tagged | 9c954504b39ebca4c412f207be0787c26bfac85a | sam21pp-L / Hiera Large | sam2.1_hiera_large.pt | model checkpoint | 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318 | dam4sam |
| Lucas–Kanade | Main five-method matrix | TrackLEED implementation | not applicable | Pyramidal skip-anchor; 41 × 41 window; four pyramid levels; forward–backward threshold 8 px | trackleed/analysis/tracking.py | implementation source | 554bc16ec6c2ebc0493b3eac65ccd8180b0f52cc65e56702ba2d60ff12c1275e | netcat |
| EasyLEED | Main five-method matrix and restricted external reference | 2.5.2 | bbe4acd2b5eeb0f5d50a8d018780ace3afd7ff73 | LEED motion model and Kalman filter; 12 px tracking radius in the method input grid | TrackLEED EasyLEED adapters | implementation source | be5701c5a3f86736ae096ad3321980756e532ef457af13059ebf916ebb07e288 | easyleed |
| SAM2.1 | Controlled SAM2 family | 2.1 | 2b90b9f5ceec907a1c18123530e92e794ad901a4 | Hiera Base+ vanilla propagation | sam2.1_hiera_base_plus.pt | model checkpoint | a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5 | sam2 |
| SAMURAI | Controlled SAM2 family | not tagged | 76ba195984892b0d1e3db5d9c9f90bb62175680a | Hiera Base+ with motion-aware memory | sam2.1_hiera_base_plus.pt | model checkpoint | a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5 | samurai |
| DAM4SAM Base+ | Independent energy scans and controlled SAM2 family | not tagged | 9c954504b39ebca4c412f207be0787c26bfac85a | sam21pp-B / Hiera Base+ | sam2.1_hiera_base_plus.pt | model checkpoint | a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5 | dam4sam |
| ViPErLEED | Restricted whole-pattern external reference | 1.03 | 2ed1044b72bb43a936f9456cdb8b9846e0d23e96 | Frozen 18-beam pattern and screen registration at 187.9 eV | ViPErLEED plugin JAR | runtime plugin | dd058af9ea79eaf6bea837efc6188b8f9b67dadf980b540d84826a33062c8a2d | viperleed-imagej |

**B. Configuration and implementation hashes**

| Component | Configuration-record SHA-256 | Configuration-record status | Upstream configuration SHA-256 | Implementation/adapter SHA-256 |
|---|---|---|---|---|
| TrackLEED common measurement | not recorded separately | not recorded separately | not applicable | c4f41ad5debdd5a6e932345553e3127570d83da08045682b8a9b3940efd55364 |
| LocoTrack | 056238541ea2b81555857fd7746b4872e7f2dbfe9fe5418973bcef47b05446df; 111f6c16214e51ac3f0838e587c8a6f564c389594c874966c5bfa554717d7b97 | run-frozen for Cu(111)/translations; post-run normalized from executed source for independent energy scans | b8cd20e5d5bc1bc5988ecde26ea182bc94de7a8d594bbf868e0f80defa9c86a7 | 23cf5087bb90e874e8de20f33f3bd48142a83bcadec8862c162fd11e5e0923e4; fc8e2aa3de8e1d7c24ad2d69c6bcb8c05151c9b8a540a1379134d1ba8bbb3eee |
| OSTrack | dc9bd17199ffc0aaea78fc00d0f68ba0b2b3047ef15fac30ec1c4997ed973736; d834f2be2cfb6f96bafee9fe2fef8b819e85bf69f8b6389d2b188e6d67d8fa4f | run-frozen for Cu(111)/translations; post-run normalized from executed source for independent energy scans | 7eff86b7bef5082378d3daadf54decaa9d00cf95b02687c6b3bf1a79b56da701 | 33755330d3328801a42a807c35442302cb1438fad0167a39ab609dbafc52919c; fc8e2aa3de8e1d7c24ad2d69c6bcb8c05151c9b8a540a1379134d1ba8bbb3eee |
| DAM4SAM Large | d2ffbdc023b7c5be3238a2dca229ebd0bcb980d994b11ca85a2b124d1c85daa1 | run-frozen executed configuration | 742e07598bcab0b09012b3fdf0ed15fec2928c750916c7b226571e9deb3ec563 | ee20f7898d6e74a6f9fc953c806b8e52cab1ba651772049b88849ea8c18c8586 |
| Lucas–Kanade | e755aca300764ee823973948744fcdd1802a8a26d0009b00ef49911b6229e99e; bb4bbb2a166371124e37be32f43fd67dc3785dc5035ecd830ec8df50450a83d3 | run-frozen for Cu(111)/translations; post-run normalized from executed source for independent energy scans | not applicable | 554bc16ec6c2ebc0493b3eac65ccd8180b0f52cc65e56702ba2d60ff12c1275e; fc8e2aa3de8e1d7c24ad2d69c6bcb8c05151c9b8a540a1379134d1ba8bbb3eee |
| EasyLEED | 54049d3d51c53f33ac8581366a4a979e4a25c2718d69f2b94e60970d6f99aecc; cbb16fba737a2c4fadea7643dc3cbfa5908603382adfee116002149b875a5494; 1ffc0c5e429095e78ac8d7d086e733661cc303fa6bdcec290c10b322e694c8ca | run-frozen for Cu(111)/translations; post-run normalized from executed sources and outputs for energy scans/external reference | not applicable | be5701c5a3f86736ae096ad3321980756e532ef457af13059ebf916ebb07e288; d11818a491ed44aa045622d06f6ccc5b4f2d1acb577e02b76b1d758b473ac31c; aea4de426b7c307d9230c3994ee88cc59579a95ae8bf0ce8079690e485f90933; 29ceb1e22e89a177a68c2af3f14ef6f331e2656daa83e71611dd4e9edcfbc23b |
| SAM2.1 | 123c6defcfc619877b0969ce23595998cfd6b95cac2e70e9dd5369560be78c46 | post-run normalized from executed source | not recorded separately | 2f6b6442e3aafafa913f78400dab15d4c7c715d93bb7577009dc75beb7b31797; c95eb72343da954378e083233b943e79beed78328e3695dc71b4a5f51aaa9547 |
| SAMURAI | 79d70dbb54fbc200bdace10d9b7117ded66e88219572e7791d28bf9486274cac | post-run normalized from executed source | not recorded separately | bf2933f3f3fc5ef31c0fc0e530d5f270469be83a8858552f04d2989c32f8ae21; c95eb72343da954378e083233b943e79beed78328e3695dc71b4a5f51aaa9547 |
| DAM4SAM Base+ | 62e9316adac13b4b5fa72ec8b656668590d48bc150e387734373715b04b2b301 | post-run normalized from executed source | not recorded separately | ee20f7898d6e74a6f9fc953c806b8e52cab1ba651772049b88849ea8c18c8586; c95eb72343da954378e083233b943e79beed78328e3695dc71b4a5f51aaa9547; fc8e2aa3de8e1d7c24ad2d69c6bcb8c05151c9b8a540a1379134d1ba8bbb3eee |
| ViPErLEED | not recorded separately | not recorded separately | not applicable | not applicable |

**C. Retained execution environments**

| Environment | Used for | Runtime | Core package versions | Execution mode/runtime | Frozen snapshot SHA-256 |
|---|---|---|---|---|---|
| netcat | TrackLEED common Gaussian measurement, Lucas–Kanade, and result assembly | Python 3.12.9 | NumPy 2.2.5; SciPy 1.15.2; OpenCV 4.11.0.86; scikit-image 0.25.2 | CPU | 14f756ccfe8c2f2781aa1ad8ea405d7c58151529ff476ae58b88efe0629e5668 |
| locotrack-pytorch | LocoTrack inference | Python 3.11.15 | PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128; NumPy 2.4.4; OpenCV 4.13.0.92 | CUDA; PyTorch build CUDA 12.8 | 812c5c9af7cd60058cd6b3bcdde8aac4dd90c4bc46568444f64c9f722d4b5bd5 |
| ostrack | OSTrack inference | Python 3.8.20 | PyTorch 1.13.1+cu117; torchvision 0.14.1+cu117; NumPy 1.24.4; OpenCV 4.13.0.92; timm 0.5.4 | CUDA; PyTorch build CUDA 11.7 | a4b9193c9cf4829f0f697ba35a2112d58011c4056b924a5976022226c3882b11 |
| dam4sam | DAM4SAM Large and Base+ inference | Python 3.10.20 | PyTorch 2.1.0+cu121; torchvision 0.16.0+cu121; NumPy 1.26.4; OpenCV 4.11.0.86; Hydra 1.3.2 | CUDA; PyTorch build CUDA 12.1 | 1b14a78e06eff162d756fbbae9753f84371814d53639abc4275165aeb012147e |
| sam2 | Vanilla SAM2.1 controlled-family inference | Python 3.10.20 | PyTorch 2.5.1+cu121; torchvision 0.20.1+cu121; NumPy 2.2.6; OpenCV 4.13.0.92; Hydra 1.3.2 | CUDA; PyTorch build CUDA 12.1 | 22d4b6254ba42b3391c3df2c468b1fe8142cb86b6ffab67a0ce0026f0afc5bef |
| samurai | SAMURAI controlled-family inference | Python 3.10.20 | PyTorch 2.5.1+cu121; torchvision 0.20.1+cu121; NumPy 1.26.4; OpenCV 4.11.0.86; Hydra 1.3.2 | CUDA; PyTorch build CUDA 12.1 | c2d2c6b2156ba94dd08b7a8cb68e99e6342d655b07431ebfd639010347a5d11a |
| easyleed | EasyLEED Kalman tracking | Python 3.11.15 | NumPy 1.26.4; SciPy 1.17.1; scikit-image 0.26.0; PyQt5 5.15.11 | CPU | e2f4a155c8696dd250f41b9cccb287c76be9fe88da9b3e77aee5dab3577f1539 |
| viperleed-imagej | ViPErLEED whole-pattern tracking | ImageJ 1.54u; OpenJDK 1.8.0_345; ViPErLEED ImageJ plugin 1.03 | not applicable | JVM/CPU | e64540325636048157605336981c63f3680b31b1c5808a1e1d49803f893d6ac3 |
<!-- END TABLE S1 -->

## Table S2 — Gaussian measurement gates and validation denominators

<!-- BEGIN TABLE S2 -->
**Table S2. Gaussian measurement gates and validation denominators.** Block A lists the run-frozen acceptance contract and separates Gaussian acceptance from the subsequent energy-branch consistency classification. Block B retains every predefined opportunity in its fixed denominator, including unavailable proposals and rejected fits. Block C gives the method-dependent n used only for statistics conditional on available native outputs, accepted Gaussian measurements, or jointly branch-consistent pairs. Counts pool repeated observations of propagated trajectories; images, spot opportunities, pair images, and gaps are not statistically independent replicates.

**A. Gaussian measurement and evaluation gates**

| Regime | Stage | Frozen criterion | Boundary and role | Failure handling |
|---|---|---|---|---|
| All experiments | Proposal-to-ROI conversion | Map the proposal to the original 1000 × 1000 image; round to the nearest pixel; place a 48 × 48 raw-intensity ROI. | The ROI must lie completely inside the image; it is rejected rather than clipped or padded. | No proposal or an out-of-bounds ROI produces a missing measurement. |
| Cu(111) and controlled translations | Numerical fit validity | Fit result, R², fitted parameters, and center uncertainties must be finite; center uncertainties must be nonnegative. | Gaussian-acceptance gate applied identically to all compared methods in this regime. | Failure produces a missing measurement. |
| Cu(111) and controlled translations | Fit quality | R² ≥ 0.85 and amplitude A ≥ 0.05. | Both thresholds are inclusive. | Failure produces a missing measurement. |
| Cu(111) and controlled translations | Fitted-center and width geometry | −2 < x_c,y_c < 50 px and 0.5 < σ_x,σ_y < 48 px. | All four bounds are strict; coordinates are local to the 48 × 48 ROI. | Failure produces a missing measurement. |
| Energy scans | Fit initialization and solver bounds | Initialize the center at the smoothed intensity maximum within 16.32 px of the ROI center; bound each fitted center coordinate to ±16.32 px from the ROI center. | Solver constraint, not a post-fit performance metric. | Fit failure produces a missing measurement. |
| Energy scans | Fit quality and widths | R² ≥ 0.85; amplitude A ≥ 0.012; 0.6 ≤ σ_x,σ_y ≤ 13 px. | All threshold bounds are inclusive. | Failure produces a missing measurement. |
| Energy scans | Proposal-to-fit consistency | Euclidean distance between the proposal and Gaussian-refined center ≤ 15 px. | Gaussian-acceptance gate applied identically to all compared methods in this regime. | Failure produces a missing measurement. |
| Energy scans | Branch consistency | Euclidean distance between an accepted Gaussian center and the frozen expected radial branch ≤ 18 px. | Evaluation gate applied after Gaussian acceptance; it does not move the proposal or fitted center. | A fit beyond 18 px remains an accepted Gaussian fit but is classified as geometry-inconsistent. |
| All experiments | Missing-value policy | No interpolation, carry-forward, manual correction, or within-run reinitialization. | Applies after unavailable proposals, out-of-bounds ROIs, fit failures, or gate failures. | The predefined opportunity remains in its fixed denominator. |
| Cu(111) primary pairs | Relative-endpoint reference | D_ref is the median of valid pair distances in the first 10 images, requiring at least 8 valid distances. | Eligibility rule for each method–pair relative endpoint, not a Gaussian-acceptance gate. | If fewer than 8 of 10 distances are valid, the entire relative endpoint is unavailable. |

**B. Fixed opportunity denominators**

| Regime | Metric | Denominator class | Denominator per method | Unit | Construction | Replication/stratification | Scope note |
|---|---|---|---|---|---|---|---|
| Controlled translations | Spot proposal and Gaussian-measurement coverage | Fixed opportunity | 80 | spot–step opportunities | 4 source-state trials × 4 spots × 5 non-seed steps | Four separately initialized trials drawn from one Cu(111) sequence | Failures remain in 80. |
| Controlled translations | Pair coverage and pair-invariance endpoint | Fixed opportunity | 40 | pair–step opportunities | 4 source-state trials × 2 pairs × 5 non-seed steps | Four separately initialized trials drawn from one Cu(111) sequence | Failures remain in 40. |
| Cu(111) positive control | Spot proposal and Gaussian-measurement coverage | Fixed opportunity | 2,804 | spot–image opportunities | 701 images × 4 frozen spots | One sequence; four spots | Failures remain in 2,804. |
| Cu(111) positive control | Primary relative pair endpoint | Fixed opportunity | 701 per pair | pair–image opportunities | 701 images for each of pairs 1–6 and 3–5 | One sequence; two frozen pairs | The two pair strata are not independent series. |
| Cu(111) positive control | D_ref eligibility window | Fixed eligibility window | 10 per method–pair | initial pair–image opportunities | First 10 images; at least 8 valid pair distances required | Applied separately to every method–pair endpoint | This gate determines whether the relative endpoint exists. |
| Cu(111) pair 2–4 diagnostic | Consecutive amplitude-only fit failures | Separate diagnostic | 24 | rejected Gaussian candidates | 24 consecutive proposal-present candidates; all passed R², center, and width gates but failed amplitude | Earlier saved YOLO proposal trajectory; outside the five-method matrix | Diagnostic count, not a denominator for comparing the five methods. |
| Independent energy scans | Branch-consistent measurement coverage | Fixed opportunity | 221 | baseline-visible spot opportunities | Frozen tracker-independent baseline across four scan–pair cases | Two independently acquired series; four frozen scan–pair cases | Every method uses the same 221 opportunities. |
| Independent energy scans | Post-gap identity recovery | Fixed opportunity | 24 | evaluable natural I(V) gaps | Baseline gaps bracketed by visible measurements on the same branch | Two independently acquired series; four frozen scan–pair cases | Every method uses the same 24 gaps. |
| Controlled SAM2 family | Branch-consistent measurement coverage | Fixed opportunity | 182 | baseline-visible spot opportunities | Frozen baseline for two pairs in one memory-analysis scan | One series; two frozen pairs | Mechanism experiment, not a population ranking. |
| Controlled SAM2 family | Post-gap identity recovery | Fixed opportunity | 13 | evaluable natural I(V) gaps | Baseline gaps bracketed by visible measurements on the same branch | One series; two frozen pairs | Mechanism experiment, not a population ranking. |
| Restricted external reference | Branch-image coverage | Fixed opportunity | 60 | branch–image opportunities | 15 images × 4 frozen branches | Subset of the memory-analysis scan; two pairs | 60 opportunities for EasyLEED and 60 for ViPErLEED. |
| Restricted external reference | Predefined branch-gap checks | Fixed opportunity | 4 | branch-gap checks | One check for each frozen branch | Subset of the memory-analysis scan; two pairs | Compatibility check, not external position ground truth. |
| Cu(111) frozen visual audit | Combined overlay review | Audit sample | 24 | combined overlays | 12 temporal quantiles + 12 temperature-cycle anchors selected independently of method output | One Cu(111) sequence | Audit only; not a metric denominator and not used to correct outcomes. |

**C. Method-dependent conditional denominators**

| Regime | Metric | Denominator class | Unit | Method-specific or paired n | Conditioning basis | Conditioning rule |
|---|---|---|---|---|---|---|
| Controlled translations | Native-proposal displacement error | Conditional available proposals | available native spot–step proposals | LocoTrack=80; OSTrack=80; DAM4SAM=80; Lucas–Kanade=76; EasyLEED=80 | Only opportunities with an available native proposal | Reported only alongside fixed proposal-coverage denominator 80. |
| Controlled translations | Gaussian-refined displacement error | Conditional accepted measurements | accepted spot–step measurements | LocoTrack=80; OSTrack=80; DAM4SAM=80; Lucas–Kanade=76; EasyLEED=74; Oracle ROI=80 | Only opportunities with an accepted Gaussian measurement | Reported only alongside fixed measurement-coverage denominator 80. |
| Controlled translations | Pair-invariance error | Conditional jointly valid pairs | jointly valid pair–step measurements | LocoTrack=40; OSTrack=40; DAM4SAM=40; Lucas–Kanade=36; EasyLEED=34; Oracle ROI=40 | Only pair opportunities with both Gaussian measurements accepted | Reported only alongside fixed pair denominator 40. |
| Independent energy scans | Geometry-inconsistent continuation | Conditional accepted fits | accepted Gaussian fits in all visibility regimes | LocoTrack=272; OSTrack=269; DAM4SAM=272; Lucas–Kanade=206; EasyLEED=222 | Accepted Gaussian fits more than 18 px from the expected branch divided by all accepted fits | This denominator is not the 221 baseline-visible coverage denominator. |
| Independent energy scans | Absolute pair-distance residual | Conditional branch-consistent pairs | jointly valid and branch-consistent pair images | LocoTrack=91; OSTrack=90; DAM4SAM=91; Lucas–Kanade=68; EasyLEED=84 | Both branches must pass proposal, Gaussian, and 18 px branch-consistency gates | Residual n is reported with the conditional statistic. |
| Controlled SAM2 family | Geometry-inconsistent continuation | Conditional accepted fits | accepted Gaussian fits in all visibility regimes | SAM2=149; SAMURAI=236; DAM4SAM=233 | Accepted Gaussian fits more than 18 px from the expected branch divided by all accepted fits | This denominator is not the 182 baseline-visible coverage denominator. |
| Controlled SAM2 family | Pair-stratified absolute pair-distance residual | Conditional branch-consistent pairs | jointly valid and branch-consistent pair images | SAM2: pair 1=10, pair 2=9; SAMURAI: pair 1=38, pair 2=34; DAM4SAM: pair 1=36, pair 2=34 | Both branches must pass proposal, Gaussian, and 18 px branch-consistency gates | Each pair-specific residual statistic is reported with its own n. |
| Restricted external reference | Native cross-method coordinate agreement | Conditional paired native outputs | paired branch–image correspondences | EasyLEED–ViPErLEED=60 paired correspondences | Both methods provide the predefined corresponding native branch coordinate | Native-output comparison precedes Gaussian refinement. |
| Restricted external reference | Common-Gaussian center agreement | Conditional jointly accepted fits | jointly accepted common centers | EasyLEED–ViPErLEED=38 jointly accepted centers | Both native outputs yield an accepted common Gaussian center for the same branch image | Reported alongside fixed branch-image denominator 60 per method. |
<!-- END TABLE S2 -->

## Table S3 — stratified Ag(111) energy-scan outcomes

<!-- BEGIN TABLE S3 -->
**Table S3. Stratified outcomes for the independent Ag(111) energy-scan stress test.** Coverage is branch-consistent measurements / baseline-visible opportunities; post-gap identity recovery is recovered / evaluable natural I(V) gaps; geometry inconsistency is accepted Gaussian fits more than 18 px from the expected branch / all accepted Gaussian fits across all visibility regimes. The conditional residual is the median absolute pair-distance residual and is evaluated only when both branches are valid and branch-consistent, with its surviving pair-frame n shown separately. The four scan–pair cases are descriptive strata from two independently acquired scans, not four independent acquisitions.

**A. Outcomes by frozen scan–pair case**

| Scan | Pair | Method | Branch-consistent coverage | Post-gap identity recovery | Geometry-inconsistent accepted fits | Conditional median absolute pair-distance residual [px] | Conditional n |
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

**B. Leave-one-case-out sensitivity**

Each of the four frozen scan–pair cases is omitted once. The remaining three cases are pooled and only the count-based coverage, post-gap identity recovery, and geometry-inconsistency rates are recomputed. The displayed extrema across the four omissions are a sensitivity check, not a confidence interval. Conditional case-median residuals are not reaggregated.

| Method | Coverage range | Post-gap identity recovery range | Geometry-inconsistency range |
|---|---:|---:|---:|
| LocoTrack | 97.4–98.8% | 94.1–100.0% | 12.6–13.5% |
| OSTrack | 97.2–98.8% | 84.2–88.9% | 11.8–13.5% |
| DAM4SAM | 97.2–98.8% | 94.1–100.0% | 12.6–13.6% |
| Lucas-Kanade | 81.9–87.6% | 63.2–76.5% | 3.4–9.5% |
| EasyLEED | 95.5–97.5% | 88.9–100.0% | 0.0–2.4% |
<!-- END TABLE S3 -->

## Table S4 — Cu(111) pairwise physical endpoints

<!-- BEGIN TABLE S4 -->
**Table S4. Cu(111) pairwise relative spot-radius endpoints by tracking method.** Values are ΔR/R = ΔD/D in percent after the native proposal from each method entered the common Gaussian measurement. Every method produced 701/701 valid endpoints for each primary pair. The 800 °C plateau is the median of a predefined five-image interval. The post-plateau minimum was selected as the lowest value after that interval; it occurred in the common image acquired at 797 °C and is therefore outcome-selected. The final cooled state is the predefined final image acquired at 92 °C. D_ref is the median valid pair distance in the first ten images, with at least eight required. Cross-method concordance after common refinement is not external absolute accuracy or an overall tracker ranking. The two pairs come from one series, are not independent replicates, and do not establish crystallographic anisotropy.

| Pair | Method | Valid endpoints | 800 °C plateau median ΔR/R [%] | Post-plateau minimum (797 °C) ΔR/R [%] | Final cooled state (92 °C) ΔR/R [%] |
|---|---|---:|---:|---:|---:|
| 1–6 | LocoTrack | 701/701 | −0.686595 | −0.726452 | 0.121606 |
| 1–6 | OSTrack | 701/701 | −0.686676 | −0.726482 | 0.121625 |
| 1–6 | DAM4SAM | 701/701 | −0.686244 | −0.726195 | 0.121691 |
| 1–6 | Lucas–Kanade | 701/701 | −0.686551 | −0.726440 | 0.121609 |
| 1–6 | EasyLEED | 701/701 | −0.686375 | −0.726389 | 0.121681 |
| 3–5 | LocoTrack | 701/701 | −1.570535 | −1.579948 | −0.152413 |
| 3–5 | OSTrack | 701/701 | −1.570833 | −1.579795 | −0.152417 |
| 3–5 | DAM4SAM | 701/701 | −1.570300 | −1.579962 | −0.152478 |
| 3–5 | Lucas–Kanade | 701/701 | −1.570362 | −1.579880 | −0.152338 |
| 3–5 | EasyLEED | 701/701 | −1.570755 | −1.580123 | −0.152343 |
<!-- END TABLE S4 -->

## Table S5 — controlled translations by source trial

<!-- BEGIN TABLE S5 -->
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
<!-- END TABLE S5 -->

## Table S6 — measurement-window and capture-contract sensitivity

<!-- BEGIN TABLE S6 -->
**Table S6. Measurement-window and capture-contract sensitivity for the independent Ag(111) stress test.** Coverage uses the fixed denominator 221 and post-gap branch-consistent recovery uses the fixed denominator 24. Geometry inconsistency is conditional on all accepted fits, while the pair-distance residual retains its method- and variant-specific conditional n. Values below retain full source precision. No comparative rank or automatic configuration selection is implied.

| Variant | Family | Method | ROI [px] | Proposal-to-fit [px] | Branch tolerance [px] | R² minimum | Amplitude minimum | Coverage | Recovery | Geometry inconsistency | Conditional median / P95 residual [px] (n) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roi_32_proposal_12 | roi_x_proposal_to_fit | EasyLEED | 32 | 12.0 | 18.0 | 0.85 | 0.012 | 209/221 | 21/24 | 2/223 | 1.6267150806055497 / 3.6593837069934345 (n=93) |
| roi_32_proposal_12 | roi_x_proposal_to_fit | LocoTrack | 32 | 12.0 | 18.0 | 0.85 | 0.012 | 220/221 | 24/24 | 38/286 | 1.776224722766358 / 5.7587664749654985 (n=103) |
| roi_32_proposal_12 | roi_x_proposal_to_fit | OSTrack | 32 | 12.0 | 18.0 | 0.85 | 0.012 | 212/221 | 21/24 | 36/275 | 1.6908525469921187 / 5.5908812023816035 (n=95) |
| roi_32_proposal_12 | roi_x_proposal_to_fit | DAM4SAM | 32 | 12.0 | 18.0 | 0.85 | 0.012 | 217/221 | 24/24 | 38/285 | 1.7551570449774374 / 5.783637141506516 (n=102) |
| roi_32_proposal_12 | roi_x_proposal_to_fit | Lucas-Kanade | 32 | 12.0 | 18.0 | 0.85 | 0.012 | 190/221 | 17/24 | 14/215 | 1.6780759632256377 / 3.9896525834825782 (n=77) |
| roi_32_proposal_15 | roi_x_proposal_to_fit | EasyLEED | 32 | 15.0 | 18.0 | 0.85 | 0.012 | 212/221 | 21/24 | 4/234 | 1.6267150806055497 / 3.6593837069934345 (n=93) |
| roi_32_proposal_15 | roi_x_proposal_to_fit | LocoTrack | 32 | 15.0 | 18.0 | 0.85 | 0.012 | 220/221 | 24/24 | 38/286 | 1.776224722766358 / 5.7587664749654985 (n=103) |
| roi_32_proposal_15 | roi_x_proposal_to_fit | OSTrack | 32 | 15.0 | 18.0 | 0.85 | 0.012 | 215/221 | 21/24 | 36/279 | 1.7113164963940335 / 5.567045449790486 (n=99) |
| roi_32_proposal_15 | roi_x_proposal_to_fit | DAM4SAM | 32 | 15.0 | 18.0 | 0.85 | 0.012 | 217/221 | 24/24 | 38/285 | 1.7551570449774374 / 5.783637141506516 (n=102) |
| roi_32_proposal_15 | roi_x_proposal_to_fit | Lucas-Kanade | 32 | 15.0 | 18.0 | 0.85 | 0.012 | 190/221 | 17/24 | 14/215 | 1.6780759632256377 / 3.9896525834825782 (n=77) |
| roi_32_proposal_18 | roi_x_proposal_to_fit | EasyLEED | 32 | 18.0 | 18.0 | 0.85 | 0.012 | 212/221 | 21/24 | 4/239 | 1.6267150806055497 / 3.6593837069934345 (n=93) |
| roi_32_proposal_18 | roi_x_proposal_to_fit | LocoTrack | 32 | 18.0 | 18.0 | 0.85 | 0.012 | 220/221 | 24/24 | 38/286 | 1.776224722766358 / 5.7587664749654985 (n=103) |
| roi_32_proposal_18 | roi_x_proposal_to_fit | OSTrack | 32 | 18.0 | 18.0 | 0.85 | 0.012 | 216/221 | 22/24 | 36/280 | 1.7223217063427114 / 5.561086511642707 (n=100) |
| roi_32_proposal_18 | roi_x_proposal_to_fit | DAM4SAM | 32 | 18.0 | 18.0 | 0.85 | 0.012 | 217/221 | 24/24 | 38/285 | 1.7551570449774374 / 5.783637141506516 (n=102) |
| roi_32_proposal_18 | roi_x_proposal_to_fit | Lucas-Kanade | 32 | 18.0 | 18.0 | 0.85 | 0.012 | 190/221 | 17/24 | 14/215 | 1.6780759632256377 / 3.9896525834825782 (n=77) |
| roi_48_proposal_12 | roi_x_proposal_to_fit | EasyLEED | 48 | 12.0 | 18.0 | 0.85 | 0.012 | 206/221 | 21/24 | 2/209 | 1.6501646559723326 / 3.661697985862128 (n=82) |
| roi_48_proposal_12 | roi_x_proposal_to_fit | LocoTrack | 48 | 12.0 | 18.0 | 0.85 | 0.012 | 217/221 | 23/24 | 36/272 | 1.8366454823238882 / 5.973206417649465 (n=91) |
| roi_48_proposal_12 | roi_x_proposal_to_fit | OSTrack | 48 | 12.0 | 18.0 | 0.85 | 0.012 | 211/221 | 20/24 | 34/262 | 1.7369320320182169 / 5.650432172024203 (n=84) |
| roi_48_proposal_12 | roi_x_proposal_to_fit | DAM4SAM | 48 | 12.0 | 18.0 | 0.85 | 0.012 | 216/221 | 23/24 | 36/272 | 1.788179181553403 / 5.973203407546521 (n=91) |
| roi_48_proposal_12 | roi_x_proposal_to_fit | Lucas-Kanade | 48 | 12.0 | 18.0 | 0.85 | 0.012 | 188/221 | 17/24 | 14/206 | 1.763723634049228 / 4.441505301861808 (n=68) |
| roi_48_proposal_15 | roi_x_proposal_to_fit | EasyLEED | 48 | 15.0 | 18.0 | 0.85 | 0.012 | 214/221 | 22/24 | 4/222 | 1.7367102643215162 / 3.718843442381691 (n=84) |
| roi_48_proposal_15 | roi_x_proposal_to_fit | LocoTrack | 48 | 15.0 | 18.0 | 0.85 | 0.012 | 217/221 | 23/24 | 36/272 | 1.8366454823238882 / 5.973206417649465 (n=91) |
| roi_48_proposal_15 | roi_x_proposal_to_fit | OSTrack | 48 | 15.0 | 18.0 | 0.85 | 0.012 | 216/221 | 21/24 | 34/269 | 1.769749564042172 / 5.937326048535333 (n=90) |
| roi_48_proposal_15 | roi_x_proposal_to_fit | DAM4SAM | 48 | 15.0 | 18.0 | 0.85 | 0.012 | 216/221 | 23/24 | 36/272 | 1.788179181553403 / 5.973203407546521 (n=91) |
| roi_48_proposal_15 | roi_x_proposal_to_fit | Lucas-Kanade | 48 | 15.0 | 18.0 | 0.85 | 0.012 | 188/221 | 17/24 | 14/206 | 1.763723634049228 / 4.441505301861808 (n=68) |
| roi_48_proposal_18 | roi_x_proposal_to_fit | EasyLEED | 48 | 18.0 | 18.0 | 0.85 | 0.012 | 219/221 | 22/24 | 6/237 | 1.8366452839015892 / 5.594865330972504 (n=90) |
| roi_48_proposal_18 | roi_x_proposal_to_fit | LocoTrack | 48 | 18.0 | 18.0 | 0.85 | 0.012 | 217/221 | 23/24 | 36/272 | 1.8366454823238882 / 5.973206417649465 (n=91) |
| roi_48_proposal_18 | roi_x_proposal_to_fit | OSTrack | 48 | 18.0 | 18.0 | 0.85 | 0.012 | 218/221 | 22/24 | 35/273 | 1.8223422561807183 / 5.9374877496368725 (n=93) |
| roi_48_proposal_18 | roi_x_proposal_to_fit | DAM4SAM | 48 | 18.0 | 18.0 | 0.85 | 0.012 | 216/221 | 23/24 | 36/272 | 1.788179181553403 / 5.973203407546521 (n=91) |
| roi_48_proposal_18 | roi_x_proposal_to_fit | Lucas-Kanade | 48 | 18.0 | 18.0 | 0.85 | 0.012 | 192/221 | 18/24 | 14/212 | 1.834648061627263 / 5.569101753620451 (n=73) |
| roi_64_proposal_12 | roi_x_proposal_to_fit | EasyLEED | 64 | 12.0 | 18.0 | 0.85 | 0.012 | 184/221 | 20/24 | 2/187 | 1.5498194989487502 / 3.581807134168525 (n=65) |
| roi_64_proposal_12 | roi_x_proposal_to_fit | LocoTrack | 64 | 12.0 | 18.0 | 0.85 | 0.012 | 199/221 | 23/24 | 36/254 | 1.8831318146568492 / 6.2999301201550475 (n=77) |
| roi_64_proposal_12 | roi_x_proposal_to_fit | OSTrack | 64 | 12.0 | 18.0 | 0.85 | 0.012 | 191/221 | 20/24 | 34/241 | 1.725091507169509 / 6.009009375987748 (n=67) |
| roi_64_proposal_12 | roi_x_proposal_to_fit | DAM4SAM | 64 | 12.0 | 18.0 | 0.85 | 0.012 | 197/221 | 23/24 | 36/252 | 1.7769156445918952 / 6.370625681759837 (n=75) |
| roi_64_proposal_12 | roi_x_proposal_to_fit | Lucas-Kanade | 64 | 12.0 | 18.0 | 0.85 | 0.012 | 170/221 | 17/24 | 14/188 | 1.7736015437714059 / 5.170699494667861 (n=54) |
| roi_64_proposal_15 | roi_x_proposal_to_fit | EasyLEED | 64 | 15.0 | 18.0 | 0.85 | 0.012 | 192/221 | 21/24 | 4/200 | 1.634037525585427 / 3.6851830886002626 (n=67) |
| roi_64_proposal_15 | roi_x_proposal_to_fit | LocoTrack | 64 | 15.0 | 18.0 | 0.85 | 0.012 | 199/221 | 23/24 | 36/254 | 1.8831318146568492 / 6.2999301201550475 (n=77) |
| roi_64_proposal_15 | roi_x_proposal_to_fit | OSTrack | 64 | 15.0 | 18.0 | 0.85 | 0.012 | 196/221 | 21/24 | 34/248 | 1.7545865525480622 / 6.4365867060998365 (n=73) |
| roi_64_proposal_15 | roi_x_proposal_to_fit | DAM4SAM | 64 | 15.0 | 18.0 | 0.85 | 0.012 | 197/221 | 23/24 | 36/252 | 1.7769156445918952 / 6.370625681759837 (n=75) |
| roi_64_proposal_15 | roi_x_proposal_to_fit | Lucas-Kanade | 64 | 15.0 | 18.0 | 0.85 | 0.012 | 170/221 | 17/24 | 14/188 | 1.7736015437714059 / 5.170699494667861 (n=54) |
| roi_64_proposal_18 | roi_x_proposal_to_fit | EasyLEED | 64 | 18.0 | 18.0 | 0.85 | 0.012 | 197/221 | 21/24 | 5/213 | 1.7776908131589835 / 5.726705709996485 (n=73) |
| roi_64_proposal_18 | roi_x_proposal_to_fit | LocoTrack | 64 | 18.0 | 18.0 | 0.85 | 0.012 | 199/221 | 23/24 | 36/254 | 1.8831318146568492 / 6.2999301201550475 (n=77) |
| roi_64_proposal_18 | roi_x_proposal_to_fit | OSTrack | 64 | 18.0 | 18.0 | 0.85 | 0.012 | 198/221 | 22/24 | 35/252 | 1.8138657005661116 / 6.329407723400038 (n=76) |
| roi_64_proposal_18 | roi_x_proposal_to_fit | DAM4SAM | 64 | 18.0 | 18.0 | 0.85 | 0.012 | 197/221 | 23/24 | 36/252 | 1.7769156445918952 / 6.370625681759837 (n=75) |
| roi_64_proposal_18 | roi_x_proposal_to_fit | Lucas-Kanade | 64 | 18.0 | 18.0 | 0.85 | 0.012 | 172/221 | 18/24 | 14/192 | 1.8823051623614901 / 5.9684275263467 (n=57) |
| branch_tolerance_12 | branch_tolerance_ofat | EasyLEED | 48 | 15.0 | 12.0 | 0.85 | 0.012 | 206/221 | 20/24 | 16/222 | 1.6501646559723326 / 3.661697985862128 (n=82) |
| branch_tolerance_12 | branch_tolerance_ofat | LocoTrack | 48 | 15.0 | 12.0 | 0.85 | 0.012 | 203/221 | 20/24 | 69/272 | 1.5504961118749634 / 3.567750839508653 (n=79) |
| branch_tolerance_12 | branch_tolerance_ofat | OSTrack | 48 | 15.0 | 12.0 | 0.85 | 0.012 | 204/221 | 20/24 | 64/269 | 1.5265635388866485 / 3.542023936804128 (n=81) |
| branch_tolerance_12 | branch_tolerance_ofat | DAM4SAM | 48 | 15.0 | 12.0 | 0.85 | 0.012 | 202/221 | 20/24 | 69/272 | 1.4988461207997261 / 3.5479638677671232 (n=79) |
| branch_tolerance_12 | branch_tolerance_ofat | Lucas-Kanade | 48 | 15.0 | 12.0 | 0.85 | 0.012 | 183/221 | 17/24 | 23/206 | 1.7359143955889067 / 3.5547768479332973 (n=66) |
| branch_tolerance_24 | branch_tolerance_ofat | EasyLEED | 48 | 15.0 | 24.0 | 0.85 | 0.012 | 214/221 | 22/24 | 1/222 | 1.7890107473935473 / 5.433272691524464 (n=87) |
| branch_tolerance_24 | branch_tolerance_ofat | LocoTrack | 48 | 15.0 | 24.0 | 0.85 | 0.012 | 217/221 | 23/24 | 19/272 | 1.9386438331731597 / 7.67020708841212 (n=99) |
| branch_tolerance_24 | branch_tolerance_ofat | OSTrack | 48 | 15.0 | 24.0 | 0.85 | 0.012 | 216/221 | 21/24 | 19/269 | 1.828923468649748 / 7.033831472977113 (n=96) |
| branch_tolerance_24 | branch_tolerance_ofat | DAM4SAM | 48 | 15.0 | 24.0 | 0.85 | 0.012 | 216/221 | 23/24 | 19/272 | 1.8725509739514337 / 7.670203690602453 (n=99) |
| branch_tolerance_24 | branch_tolerance_ofat | Lucas-Kanade | 48 | 15.0 | 24.0 | 0.85 | 0.012 | 188/221 | 17/24 | 9/206 | 1.763723634049228 / 4.441505301861808 (n=68) |
| r_squared_0p80 | r_squared_ofat | EasyLEED | 48 | 15.0 | 18.0 | 0.8 | 0.012 | 214/221 | 22/24 | 4/230 | 1.6501646559723326 / 3.6935608502433044 (n=92) |
| r_squared_0p80 | r_squared_ofat | LocoTrack | 48 | 15.0 | 18.0 | 0.8 | 0.012 | 221/221 | 24/24 | 37/286 | 1.834130048881775 / 5.77925816923481 (n=104) |
| r_squared_0p80 | r_squared_ofat | OSTrack | 48 | 15.0 | 18.0 | 0.8 | 0.012 | 217/221 | 22/24 | 34/278 | 1.7432474944041587 / 5.725331719227924 (n=99) |
| r_squared_0p80 | r_squared_ofat | DAM4SAM | 48 | 15.0 | 18.0 | 0.8 | 0.012 | 217/221 | 24/24 | 37/281 | 1.788179181553403 / 5.832321486899693 (n=99) |
| r_squared_0p80 | r_squared_ofat | Lucas-Kanade | 48 | 15.0 | 18.0 | 0.8 | 0.012 | 190/221 | 17/24 | 14/215 | 1.7328652907616515 / 3.900752751308463 (n=77) |
| r_squared_0p90 | r_squared_ofat | EasyLEED | 48 | 15.0 | 18.0 | 0.9 | 0.012 | 194/221 | 21/24 | 4/202 | 1.5694047107091365 / 3.65593743398715 (n=69) |
| r_squared_0p90 | r_squared_ofat | LocoTrack | 48 | 15.0 | 18.0 | 0.9 | 0.012 | 200/221 | 23/24 | 36/255 | 1.8366454823238882 / 6.221266792272805 (n=79) |
| r_squared_0p90 | r_squared_ofat | OSTrack | 48 | 15.0 | 18.0 | 0.9 | 0.012 | 195/221 | 21/24 | 34/247 | 1.7432474944041587 / 6.436595755551291 (n=73) |
| r_squared_0p90 | r_squared_ofat | DAM4SAM | 48 | 15.0 | 18.0 | 0.9 | 0.012 | 196/221 | 23/24 | 36/251 | 1.788179181553403 / 6.364790459861306 (n=75) |
| r_squared_0p90 | r_squared_ofat | Lucas-Kanade | 48 | 15.0 | 18.0 | 0.9 | 0.012 | 170/221 | 17/24 | 14/188 | 1.788483767682294 / 5.106880685325345 (n=55) |
| amplitude_multiplier_0p75 | amplitude_ofat | EasyLEED | 48 | 15.0 | 18.0 | 0.85 | 0.009000000000000001 | 214/221 | 22/24 | 4/222 | 1.7367102643215162 / 3.718843442381691 (n=84) |
| amplitude_multiplier_0p75 | amplitude_ofat | LocoTrack | 48 | 15.0 | 18.0 | 0.85 | 0.009000000000000001 | 217/221 | 23/24 | 36/272 | 1.8366454823238882 / 5.973206417649465 (n=91) |
| amplitude_multiplier_0p75 | amplitude_ofat | OSTrack | 48 | 15.0 | 18.0 | 0.85 | 0.009000000000000001 | 216/221 | 21/24 | 34/269 | 1.769749564042172 / 5.937326048535333 (n=90) |
| amplitude_multiplier_0p75 | amplitude_ofat | DAM4SAM | 48 | 15.0 | 18.0 | 0.85 | 0.009000000000000001 | 216/221 | 23/24 | 36/272 | 1.788179181553403 / 5.973203407546521 (n=91) |
| amplitude_multiplier_0p75 | amplitude_ofat | Lucas-Kanade | 48 | 15.0 | 18.0 | 0.85 | 0.009000000000000001 | 188/221 | 17/24 | 14/206 | 1.763723634049228 / 4.441505301861808 (n=68) |
| amplitude_multiplier_1p25 | amplitude_ofat | EasyLEED | 48 | 15.0 | 18.0 | 0.85 | 0.015 | 214/221 | 22/24 | 4/222 | 1.7367102643215162 / 3.718843442381691 (n=84) |
| amplitude_multiplier_1p25 | amplitude_ofat | LocoTrack | 48 | 15.0 | 18.0 | 0.85 | 0.015 | 217/221 | 23/24 | 36/272 | 1.8366454823238882 / 5.973206417649465 (n=91) |
| amplitude_multiplier_1p25 | amplitude_ofat | OSTrack | 48 | 15.0 | 18.0 | 0.85 | 0.015 | 216/221 | 21/24 | 34/269 | 1.769749564042172 / 5.937326048535333 (n=90) |
| amplitude_multiplier_1p25 | amplitude_ofat | DAM4SAM | 48 | 15.0 | 18.0 | 0.85 | 0.015 | 216/221 | 23/24 | 36/272 | 1.788179181553403 / 5.973203407546521 (n=91) |
| amplitude_multiplier_1p25 | amplitude_ofat | Lucas-Kanade | 48 | 15.0 | 18.0 | 0.85 | 0.015 | 188/221 | 17/24 | 14/206 | 1.763723634049228 / 4.441505301861808 (n=68) |
<!-- END TABLE S6 -->
