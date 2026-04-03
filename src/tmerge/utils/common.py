"""Shared utility helpers used across TMerge modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def natural_frame_key(path: Path) -> tuple[int, str]:
    """Sort key that orders image paths numerically by stem when possible."""
    stem = path.stem
    return (int(stem), path.name) if stem.isdigit() else (10**12, path.name)


def require_optional(module_name: str, *, pip_extra: str | None = None, purpose: str = "") -> Any:
    """Import and return an optional dependency, raising a clear error if missing.

    Parameters
    ----------
    module_name:
        Fully-qualified module name to import (e.g. ``"motmetrics"``).
    pip_extra:
        The pip extras key, e.g. ``"reid"``.  When provided the error
        message will suggest ``pip install -e ".[<extra>]"``.
    purpose:
        A short description of why the module is needed, used in the
        error message.
    """
    try:
        return __import__(module_name, fromlist=["_sentinel"])
    except ModuleNotFoundError as exc:
        if pip_extra:
            install_hint = f'python -m pip install -e ".[{pip_extra}]"'
        else:
            install_hint = f"python -m pip install {module_name}"
        prefix = f"{purpose} requires" if purpose else "This feature requires"
        raise RuntimeError(
            f"{prefix} optional dependency '{module_name}'. "
            f"Install with: {install_hint}"
        ) from exc
