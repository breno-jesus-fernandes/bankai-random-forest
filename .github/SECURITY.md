# Dependency security policy

Cargo and uv lockfiles are the reproducible dependency inputs. CI must use locked installs and must not update either lockfile implicitly.

Dependency Review blocks pull requests that introduce high or critical severity vulnerabilities. `pip-audit` audits the exported `uv.lock`; `cargo-deny` audits Rust advisories, licenses, duplicate versions, and dependency sources. Findings that are not blocked by the PR severity threshold remain visible in the audit jobs for review.

Exceptions must be narrowly scoped and recorded beside the corresponding scanner configuration with the advisory or crate, rationale, responsible maintainer, and review date. Rust advisory exceptions belong in `.cargo/deny.toml`. Do not suppress unresolved Python vulnerabilities; update the lockfile or track the exception in a dedicated issue and document its identifier and review date in the security workflow.

GitHub branch rules must require pull requests and each matrix `Test (...)` check, `Rust dependency audit`, `Python dependency audit`, and `Dependency Review` checks before merging into `master`. Configure the PyPI Trusted Publisher for this repository, the `CI and release artifacts` workflow, and the `pypi` environment. Enable GitHub's dependency graph and Dependabot alerts in repository settings.
