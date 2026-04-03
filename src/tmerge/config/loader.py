"""YAML config loading with support for inheritance and CLI overrides."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    loaded = _load_with_extends(path)
    loaded = _expand_env_vars(loaded)
    if overrides:
        loaded = apply_overrides(loaded, overrides)
    return loaded


def _load_with_extends(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    extends = raw.pop("extends", None)
    if not extends:
        return raw
    base_path = (path.parent / extends).resolve()
    base = _load_with_extends(base_path)
    return deep_merge(base, raw)


def deep_merge(base: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    mutated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in key=value format: {override}")
        key, raw_value = override.split("=", 1)
        target: Any = mutated
        parts = key.split(".")
        for part in parts[:-1]:
            if isinstance(target, list):
                target = target[int(part)]
            else:
                target = target.setdefault(part, {})
        final_key = parts[-1]
        value = yaml.safe_load(raw_value)
        if isinstance(target, list):
            target[int(final_key)] = value
        else:
            target[final_key] = value
    return mutated
