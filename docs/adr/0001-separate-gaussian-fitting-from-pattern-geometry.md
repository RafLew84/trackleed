# Separate Gaussian fitting from pattern geometry

Status: accepted

Track analysis will persist fingerprinted Gaussian fit results independently
from pattern-center estimation and radius calculation. Geometry consumes an
immutable fit set through a small interface, while a compatibility adapter
continues to provide the existing flattened session, GUI, and export view; this
keeps expensive measurements reusable without coupling their validity to the
selected center-estimation method.

## Consequences

- Changing pattern-center method, configuration, or center-reference tracks
  recomputes geometry without reloading ROIs or repeating Gaussian fits.
- ROI pixels, background inputs, fitting configuration, validation, and fit
  recovery are covered by per-observation fingerprints and invalidate only the
  affected fit entries.
- Older results without sufficient provenance remain inspectable and exportable
  but cannot feed a new geometry calculation.
- Automatic fit recovery is deterministic, bounded, provenance-bearing, and
  opt-in until accepted against synthetic, real-series, and human-reviewed data.
