# Legacy Assets

This repository now treats the following top-level directories as legacy or
transitional assets:

- `videosys/`
- `e2e/`
- `scripts/`
- parts of `config/`

They are still present because they contain valuable research logic, experiment
history, and migration references. They are no longer the recommended public
interface for the project.

## Rules

- New runtime code goes in `src/tmerge/`.
- New runnable workflows should be expressed as YAML configs under `configs/`
  or `examples/`.
- New CLI-facing behavior should be implemented behind `tmerge ...` commands.
- Legacy directories may be read, adapted, or gradually ported, but should not
  be expanded as the default path for new development.

## Naming Guidance

- `src/` is a source layout, not a business package name.
- `tmerge` is the modern package name and public surface.
- `videosys` is historical and should be understood as a legacy namespace.

## Planned End State

Over time, frequently used capabilities from `videosys/`, `e2e/`, and
`scripts/` should either:

- move into `src/tmerge/` as native implementations, or
- remain explicitly documented as legacy-only.

