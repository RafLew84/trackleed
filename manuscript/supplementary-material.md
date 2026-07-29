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
