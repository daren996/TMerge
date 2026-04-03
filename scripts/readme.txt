This folder contains legacy research scripts and one-off experiment helpers.

New user-facing workflows should move toward:
- `tmerge` CLI commands
- YAML configs under `configs/` or `examples/`
- native implementations under `src/tmerge/`

Keep scripts here only when they are clearly legacy, transitional, or
dataset-specific utilities that have not yet been migrated.
