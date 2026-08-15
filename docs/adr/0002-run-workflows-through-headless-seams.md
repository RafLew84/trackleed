# Run workflows through headless seams and publish atomically

Status: accepted

Spot detection, seeded spot tracking, detection linking, and track analysis
will run through separate typed headless workflow interfaces. Method catalogs
contain stable identities, capabilities, preflight, and runtime adapters but no
widget references; Qt configuration pages bind to those identities separately.
Completed results are published atomically into the existing `LeedFile`,
`Track`, flattened analysis, session, and export views through compatibility
adapters. This creates testable seams and preserves current consumers without
the cost and risk of replacing the entire session model in the first increment.

## Consequences

- `MainWindow` remains the GUI composition shell and holder of current model
  references, but no longer owns algorithm-specific configuration or dispatch.
- Cancellation and failure leave the previous complete result active.
- Runtime workflows and their tests do not depend on Qt or mutable GUI models.
- Gaussian fits may be reused across compatible track changes, while geometry
  is always invalidated when its track set changes.
- Existing dialogs and exporters continue to consume a flattened compatibility
  projection during the migration.
