# Rust formatting check

`cargo fmt --all -- --check` returns exit code 1 in both the untouched baseline
and the candidate because six existing Rust files have formatting drift under
the installed rustfmt. The baseline check reported 71 diff sections; the
candidate check reported 64. The modified split specialization was formatted
with rustfmt and no longer appears in the candidate formatting diff. A
workspace-wide reformat would change unrelated code, so none was applied.
