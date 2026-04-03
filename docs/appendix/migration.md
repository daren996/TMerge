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
