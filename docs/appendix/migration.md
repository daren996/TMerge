# Migration Notes

The modernization is intentionally breaking at the interface level.

## Old Style

- `python e2e/ingestion_runner.py ...`
- Python config scripts under `e2e/configs/`
- runtime path injection through `.env`, `PYTHONPATH`, or `sys.path.append(...)`
- loosely structured research scripts mixed with tests

## New Style

- `tmerge run pipeline ...`
- YAML configuration files
- installable package under `src/tmerge`
- layered dependencies and optional integrations
- unit and integration tests separated under `tests/`

## Transition Guidance

- Keep `videosys/`, `e2e/`, and `scripts/` as migration references.
- Prefer creating new workflows in `examples/` or project-specific config folders using YAML.
- When you still need a legacy algorithm, call it through the `integration.*` operator adapters instead of wiring it directly into new runtime code.

## Naming Guidance

- Do not rename `videosys` to `src`. These names describe different things.
- `src/` is the source layout root for installable code.
- `tmerge` is the modern package name and public API surface.
- `videosys` is a historical package name and should be treated as a legacy namespace.

## Recommended Mental Model

Use this split when navigating the repository:

- `src/tmerge/`: current productized research-engineering path
- `configs/`, `examples/`: current runnable configuration surface
- `videosys/`, `e2e/`, `scripts/`: historical implementation and migration material

## Suggested Final Layout

The intended long-term shape is:

```text
src/tmerge/           modern runtime, CLI, config, operators, training
configs/              maintained YAML workflows
examples/             minimal runnable examples
tests/                unit and integration tests for the modern path
legacy/               documentation and indexes for historical assets
videosys/             transitional legacy package, eventually minimized or removed
e2e/                  transitional legacy experiment entrypoints
scripts/              transitional legacy utilities
```
