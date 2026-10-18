# Changelog

## 0.1.0a16 (unreleased)

- Add shared original-precision exact columns with virtual shadow row maps for
  private Boruta integration; preserve alpha 15 float32 training values.
- Reuse the existing binary split workspace for indirect columns and keep the
  materialized exact path available for parity checks.
- Validate bank shapes/values and round maps; add exact tree parity tests for
  binary/multiclass, bootstrap and impurity controls.
- Require explicit `publish=true` manual workflow dispatch for PyPI uploads.
- Keep ordinary classifier storage unchanged while AMD performance acceptance
  remains pending. See docs/virtual-exact-columns.md.
