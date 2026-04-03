## TMerge Code Style

This repository now follows a two-speed migration policy:

- `src/tmerge` and `tests` must follow modern Python engineering conventions.
- `videosys`, `scripts`, `tools`, and `e2e` are legacy research areas. When touched, migrate them incrementally toward the same standard instead of doing broad rewrites in unrelated changes.

### Baseline standards

- Follow PEP 8 for layout, naming, imports, and general readability.
- Use type hints for public functions, operators, factories, and tests that support the typed pipeline.
- Prefer `pathlib.Path` over `os.path` in new code.
- Prefer `logging` in reusable modules and long-running workflows; keep `print` for CLI-facing error reporting or one-off legacy scripts.
- Use dataclasses for transport-style domain models when they primarily hold structured state.
- Write tests with `pytest`, keeping fixtures explicit and assertions behavior-focused.

### Tooling

- `ruff` is the default linter, import sorter, and formatter boundary for mainline code.
- `mypy` checks `src/tmerge` and `tests`; legacy directories are intentionally out of scope until they are migrated.
- `pytest` remains the behavioral safety net for refactors.

Recommended commands:

```bash
python -m ruff check src/tmerge tests
python -m ruff format src/tmerge tests
python -m mypy src/tmerge tests
pytest -q
```

### Review checklist

- Every new or changed public function has parameter and return annotations.
- File and directory handling uses `Path` objects unless an external API requires strings.
- Imports are grouped as standard library, third-party, and first-party.
- Mutating I/O code handles missing paths and encoding explicitly.
- Pipeline and config code prefers small, composable helpers over large script-style functions.

### Priority backlog for legacy code

1. Replace `os.path` and raw string path assembly with `pathlib.Path`.
2. Replace `os.system(...)` with `subprocess.run(..., check=True)`.
3. Add module docstrings and function annotations to reused library code before touching one-off experiment scripts.
4. Move repeated script logic into typed `src/tmerge` modules so behavior can be tested once and reused.
