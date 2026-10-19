# Changelog

## 0.1.0a19

- Update the README for the current release: drop the outdated status banner,
  describe binned Extra Trees as a regular backend and mention Bankai Boruta.
  Links are absolute so they also work on PyPI.
- Add project URLs (homepage, repository, issues, changelog) to the package
  metadata, so PyPI links to the GitHub repository.

## 0.1.0a18

- Build Random Forest histograms from column-major binned codes, as made by
  `DenseInput::bin_columns`. Each bin still accumulates its rows in mask order,
  so histograms and trees are bit-identical to the row-major layout; Rust tests
  pin this for binary and multiclass targets, with and without bootstrap and
  with serial and parallel histograms. This lets Bankai Boruta's binned shadow
  session train ordinary binned Random Forests as well as Extra Trees.
- Publish to PyPI when a PR is merged into `master` and its version is not on
  PyPI yet; the GitHub release follows from the same run. A manual dispatch
  with `publish=true` remains available to retry.

## 0.1.0a17

- Count the classes on each side of a random Extra Trees threshold without
  data-dependent branches. Exact binary split search is about 1.9x faster on
  200k x 1000 inputs; trees and predictions are bit-identical to alpha 16.
- Add `ExactColumnTable`: float32 real and shadow columns already in training
  row order, read as contiguous slices with no row or permutation maps. Bankai
  Boruta's default exact session trains on it without materializing an
  augmented matrix. Rust tests pin exact tree parity with materialized input.
- Add `DenseInput::bin_columns` and `DenseInput::training_binned_columns`:
  bin real columns once and train on column-major codes whose columns may
  repeat or be permuted (Bankai Boruta's binned shadow session). Rust tests pin
  tree parity with per-fit binning.
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
