# Changelog

## 0.1.0a17 (unreleased)

- Count the classes on each side of a random Extra Trees threshold without
  data-dependent branches. Exact binary split search is about 1.9x faster on
  200k x 1000 inputs; trees and predictions are bit-identical to alpha 16.
- Add `ExactColumnTable`: float32 real and shadow columns already in training
  row order, read as contiguous slices with no row or permutation maps. Bankai
  Boruta's default exact session trains on it without materializing an
  augmented matrix. Rust tests pin exact tree parity with materialized input.
- Test that exact Extra Trees `gain` importances equal scikit-learn's mean
  decrease in impurity recomputed from the exported trees.

## 0.1.0a16

- Add shared original-precision exact columns with virtual shadow row maps for
  private Boruta integration; preserve alpha 15 float32 training values.
- Reuse the existing binary split workspace for indirect columns and keep the
  materialized exact path available for parity checks.
- Validate bank shapes/values and round maps; add exact tree parity tests for
  binary/multiclass, bootstrap and impurity controls.
- Require explicit `publish=true` manual workflow dispatch for PyPI uploads.
- Keep ordinary classifier storage unchanged while AMD performance acceptance
  remains pending. See docs/virtual-exact-columns.md.
