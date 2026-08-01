# Figure S1 - complete controlled-translation trials

Figure S1 shows the full diagnostic record for each of the four independently
initialized source trials:

- full commanded and Gaussian-refined displacement paths for every spot and
  method;
- every available native-versus-refined displacement-error observation;
- pair-invariance trajectories, with invalid pair measurements marked.

The deterministic builder consumes the authenticated shared source package at
`manuscript/source_data/controlled_translations/` and writes 300 dpi PNG and
vector PDF outputs plus a manifest binding source, implementation, panel
order, and output SHA-256 hashes.

Run from the repository root:

```bash
python -m manuscript.figures.figure_s1_controlled_translation_trials.build_figure
```

Known translations assess measurement-chain equivariance and
common-estimator behavior. They do not establish absolute tracking accuracy
or identity ground truth.
