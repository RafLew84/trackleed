# Scientific-core reconstruction and fingerprint

The publication scientific core is reconstructed and validated through one
headless CPU command from the repository root:

```text
python -m scripts.reconstruct_publication --clean --validate
```

The command validates the immutable input packages, reconstructs the central
candidate/evaluation layers and all predeclared sensitivity variants, rebuilds
the sensitivity evidence and claim-boundary artifacts, runs every approved
main/SI display builder, regenerates the twenty-item publication inventory,
checks the manuscript contract, and finally validates `fingerprint.json`.

The repository must be accompanied by the authenticated local raw-image paths
declared in `manuscript/scientific_baseline/input_manifest.json`. The command is
a clean scientific rebuild (`--clean`), not a data-download/bootstrap command;
missing or altered local inputs fail before reconstruction.

The runtime is frozen in `cpu-environment.txt`. Reconstruction forces a
headless rendering backend and disables CUDA visibility. It neither opens the
GUI nor performs tracker inference, loads model checkpoints, benchmarks
performance, renders an editorial DOCX/LaTeX/PDF, uploads data, or creates a
DOI.

## Fail-closed baseline

`fingerprint_contract.json` names every semantic component and the roles that
must remain covered: final prose, claim-evidence map, source data, builders,
validators, displays, SI inventory, central outcomes, sensitivity,
denominators, limitations, claim boundaries, and provenance.

The reconstruction command only validates the frozen fingerprint. It never
updates it. A changed component, contract, hash, manifest, opportunity key,
central record, denominator, claim anchor, or provenance record makes the
command fail.

Display builders run transactionally: their regenerated prose blocks, tables,
manifests, PNGs, and PDFs are validated and the committed artifacts are then
restored. Renderer-dependent PNG/PDF bytes are therefore not part of the
semantic fingerprint. Instead,
`manuscript/publication_inventory/semantic_manifest.json` freezes, for all 20
display bundles, the source data, builder and validator code, captions,
renderer-independent manifest fields, denominator policy, and textual
scientific outputs.

Creating a fingerprint is a separate explicit operation:

```text
python -m scripts.freeze_scientific_core --build
```

That operation refuses to replace an existing fingerprint. Read-only
validation is available as:

```text
python -m scripts.freeze_scientific_core --validate-only
```

The frozen status is `scientific-baseline-complete`. The currently authorized
author, experimental-description, submission, and release-authorization
placeholders remain outside the scientific claim boundary until their
deferred-input tickets are completed.
