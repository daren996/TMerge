# Project Layout

The repository now follows a package-first structure.

## Current Recommended Areas

- `src/tmerge/`
  Modern code. Add new runtime logic, CLI behavior, config tooling, adapters,
  and training entrypoints here.
- `configs/`
  Maintained YAML pipeline and workflow configs.
- `configs/training/`
  Maintained training configs used by `tmerge train reid`.
- `examples/`
  Small runnable examples intended to verify install and usage.
- `tests/`
  Automated tests for the package-first workflow.

## Transitional Legacy Areas

- `videosys/`
  Historical implementation namespace. Still used by some migration paths and
  legacy training code.
- `e2e/`
  Historical Python-config pipeline runner and experiment entrypoints.
- `scripts/`
  Historical task scripts and dataset-specific helpers.

## Contributor Rule of Thumb

When adding something new, ask:

1. Is this part of the official workflow?
2. Should a new user discover and rely on it?

If the answer is yes, it belongs in the modern path:

- code in `src/tmerge/`
- config in `configs/`
- docs in `readme.md` or `docs/appendix/`

If the answer is no and it only preserves a historical path, keep it clearly
marked as legacy.
