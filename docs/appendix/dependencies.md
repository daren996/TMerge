# Dependency Layout

The repository now treats dependencies in three layers:

- Core runtime dependencies live in `pyproject.toml` under `project.dependencies`.
- Development tooling lives in the `dev` optional dependency group.
- Heavy research integrations such as OpenMMLab and TorchReID live in optional groups.

Recommended installation commands:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,openmmlab,reid]"
```

This replaces the old environment snapshot style `requirements.txt`.

