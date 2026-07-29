# Publication-safe domain-baseline software record

This record preserves only the software identities and runtime artifacts needed
for Table S1. Personal installation paths and internal acquisition identifiers
from the full working installation log are intentionally omitted.

## EasyLEED

- software version: `2.5.2`
- upstream revision: `bbe4acd2b5eeb0f5d50a8d018780ace3afd7ff73`
- license reported by the project: `GPL-2.0`
- Python: `3.11.15`
- NumPy: `1.26.4`
- SciPy: `1.17.1`
- scikit-image: `0.26.0`
- PyQt5: `5.15.11`

The retained environment used NumPy 1.26.4 because the initially resolved
NumPy 2.x environment was incompatible with the EasyLEED Kalman-filter
implementation. Native proposals were reduced through the common 48 x 48 px
raw-image ROI and TrackLEED Gaussian estimator.

## ViPErLEED

- plugin version: `1.03`
- upstream revision: `2ed1044b72bb43a936f9456cdb8b9846e0d23e96`
- license reported by the project: `GPL-3.0-or-later`
- ImageJ: `1.54u`
- OpenJDK: `1.8.0_345`

Runtime artifacts:

| Artifact | SHA-256 |
|---|---|
| ImageJ Windows distribution | `32b05356d3a7906e10ca91491c2be06bfe0397c9e7a3d3b52fdad9e3ef48276c` |
| Active ImageJ `ij.jar` | `6acc016e4630d5c5ecd470b205f8d1abc08ce57568c6ab617a308d2451d9c90d` |
| ViPErLEED plugin JAR | `dd058af9ea79eaf6bea837efc6188b8f9b67dadf980b540d84826a33062c8a2d` |

The active ImageJ JAR is a runtime artifact, not a TrackLEED implementation or
adapter source. The ViPErLEED plugin is retained as a restricted whole-pattern
external reference.
