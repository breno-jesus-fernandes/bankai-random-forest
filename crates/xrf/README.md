# XRF — Abstract Random Forest

A generic Rust library to build [Random Forest](https://en.wikipedia.org/wiki/Random_forest) implementations for arbitrary data sources, given that they can provide some basic operations.

See the [SoftwareX paper about fru](https://doi.org/10.1016/j.softx.2026.102918) for the details.

Used in:
- the [fru](https://cran.r-project.org/package=fru) R package.
- the [pyfru](https://pypi.org/project/pyfru/) Python package and [fru-arrow](https://crates.io/crates/fru-arrow) Rust crate.
- the [boruta-fru](https://pypi.org/project/boruta-fru/) Python package providing fast & canonical implementation of the [Boruta algorithm](https://doi.org/10.18637/jss.v036.i11); same goes indirectly for an original [R package](https://cran.r-project.org/package=Boruta) which, by default, uses fru.


