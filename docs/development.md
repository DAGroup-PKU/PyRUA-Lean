# Development rules

Install with `pip install -e '.[dev,workflow]'`, then run `make test lint audit`. CI runs the same checks (`.github/workflows/ci.yml`).

- Core tests must run with no simulator, network or credentials.
- Runtime code must not contain experiment IDs or absolute filesystem paths.
- Inputs are `ArtifactRef` values; adapters resolve bytes through their own stores.
- A schema change gets a new version.
- Earlier schema versions are never accepted silently and never gain compatibility aliases.
- Every remote create operation must persist a stable job identity before polling.
- Every verifier must include hard negatives appropriate to its authority.
- Generated observations cannot become geometry, action, or task-state authority.
- A candidate patch cannot modify its own promotion label or root verifier.
- Every requested pipeline invocation (full or partial) is registered as a ledger run and launched through `tools/datafarm_runs.py launch`; see [run ledger](reference/run-ledger.md).
