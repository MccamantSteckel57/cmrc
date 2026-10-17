"""YAML configuration with dotted CLI overrides and typed accessors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

Config = dict[str, Any]
_T = TypeVar("_T")

_MISSING = object()


def _deep_merge(base: Config, overlay: Mapping[str, Any]) -> Config:
    merged: Config = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(dict(current), value)
        else:
            merged[key] = value
    return merged


def _parse_override(token: str) -> tuple[list[str], Any]:
    if "=" not in token:
        raise ValueError(f"override {token!r} must be of the form key.subkey=value")
    dotted, raw = token.split("=", 1)
    keys = [part for part in dotted.split(".") if part]
    if not keys:
        raise ValueError(f"override {token!r} has an empty key path")
    return keys, yaml.safe_load(raw)


def _assign(target: Config, keys: Sequence[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def load_config(paths: Sequence[Path], overrides: Sequence[str] = ()) -> Config:
    config: Config = {}
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path.name} must contain a YAML mapping")
        config = _deep_merge(config, payload)
    for token in overrides:
        keys, value = _parse_override(token)
        _assign(config, keys, value)
    return config


def lookup(config: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    cursor: Any = config
    for key in dotted.split("."):
        if not isinstance(cursor, Mapping) or key not in cursor:
            if default is _MISSING:
                raise KeyError(f"missing configuration key {dotted}")
            return default
        cursor = cursor[key]
    return cursor
